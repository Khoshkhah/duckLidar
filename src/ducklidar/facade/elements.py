"""Wall ELEMENTS from SAM 3 instances: photo px -> wall metres through the strip homographies (never the pose).

    PILOT_BUILDING=railspur python tools/pilot/elements.py  -> out/pilot_railspur_model/qa_facade_elements.png (facade02, #2/#4)
    PILOT_BUILDING=netloft  python tools/pilot/elements.py  -> facade00 (#8/#15/#16/#17)

Input per photo: elements/NN.json (instances: label, score, bbox, area) + elements/NN.png (uint16 id map, FIRST prompt
per pixel). Instances are noisy (cars' windows, neighbours' doors, whole-facade 'storefronts', cross-prompt triplicates,
nested double windows); six data-driven gates and a small wall grammar keep what a person would draw.

Conventions: wall metres x from the s0 end, y UP from the wall base (y = H_m at the eave); texture px col = x / TEX_M,
row = h - y / TEX_M (row 0 = top). photo px -> strip px: sm["H"]; strip px -> texture px: (c - px + dx, r - py + dy).
"""
import json, os, sys, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from . import texture as _texture                            # own_building: the chroma-restricted building mask

TEX_M = 0.04
TYPES = ("window", "door", "awning", "storefront")          # kept; 'garage door' -> door. sign/balcony/downspout -> dropped in v1
                                                            # (ponytail: SAM's occluder pass masks every sign/downspout on both walls; material covers them. Add 'sign' decals when a clean one appears.)
MIN_SCORE = 0.4
AWNING_SCORE, AWNING_WIDE = 0.3, 3.0                        # a wide awning may score low (railspur facade03's 5.5 m hipped canopy: 0.33)
AWNING_MAX_H = 1.5                                          # m: SAM's box includes a canopy's roof; the band keeps its underside
CROP_PPM, CROP_OBLIQUE, CROP_SEEN, CROP_SEEN_DOOR = 12.0, 1.5, 0.9, 0.6   # a door's base behind a car is filled with the door's own colour
CROP_OBS = 0.6                                              # a crop needs >= 60 % of the element's height observed (a door box cut at a car roof is not stretched)
SIB_TOL = 0.3                                               # a sibling's crop may differ <= 30 % in width (never a 0.9 m window stretched to a 3.5 m bay)
AWNING_MAX = 1.8                                            # m: taller 'awnings' are roofs (railspur #3's hipped wing roof, 2.4 m)
IMG_W = 640                                                 # the virtual pinhole views
ABUT = 0.15                                                 # m: same-row same-type rects closer than this are one element (sashes, a double door)
ROW_TOL, SNAP_TOL, SNAP_TOL_SF = 0.8, 0.6, 0.7             # m: row clustering / head-sill snap (strips disagree by up to 0.6 m in y)
EDGE_TOL = {"awning": 1.0, "storefront": 99.0}              # m an element may cross a wall end and still be clipped (door 35 crosses s0 by 0.24 m;
                                                            #  a shopfront band runs around the corner: netloft #7, the part on our wall is a bay)
EDGE_DEFAULT = 0.3                                          #  an awning projects toward the camera and reads 0.6 m past the corner on railspur #2)
MIN_SEEN = 0.3                                              # below: the rect sits on texels the strip never saw (neighbour cut, off-frame) — unless an occluder explains it
OCC_EVIDENCE = 0.15                                         # occluder fraction over the box that makes an unseen rect 'behind a tree', position still trusted

GATE_PPM, GATE_YEAR, GATE_STRETCH = 12.0, 2020, 1.5   # source px/m floor (a 0.145 m frame = 1.7 src px at 12); photo-year floor, RELATIVE; max linear magnification


def crop_ok(ppm_src, oblique, date, rect_w, crop_w, best_date=None, obs=1.0):
    """-> None when the crop is acceptable, else a short reason string. Pure.
    ppm_src source px/m, oblique the strip's obliquity, date 'YYYY-MM', rect_w the TARGET width (m),
    crop_w the width (m) the crop covers at its source, best_date the newest date of any photo seeing
    this wall AND able to supply a crop, obs the fraction of the element actually observed (the rest is
    _crop's repeated median row: an invented band, i.e. a stretch in the other axis)."""
    if ppm_src < GATE_PPM: return f"ppm {ppm_src:.0f} < {GATE_PPM:.0f}"
    if oblique > CROP_OBLIQUE: return f"oblique {oblique:.1f} > {CROP_OBLIQUE}"
    stretch = max((rect_w / TEX_M) / max(crop_w * ppm_src, 1e-6), 1.0 / max(obs, 1e-6))   # texels to fill / source px available; an unobserved band counts the same way
    if stretch > GATE_STRETCH: return f"stretch {stretch:.1f}x > {GATE_STRETCH}"
    yr = lambda d: int(str(d)[:4]) if str(d)[:4].isdigit() else 0
    if date and best_date and yr(date) < GATE_YEAR <= yr(best_date): return f"{date} older than {best_date}"
    return None


# ── geometry ────────────────────────────────────────────────────────────────
def photo_to_tex(sm):
    """(3,3) photo px -> texture px (strip H shifted by the stitch offset minus the pad)."""
    px, py = sm["pad"]; dx, dy = sm.get("offset_px", (0, 0))
    T = np.array([[1.0, 0.0, dx - px], [0.0, 1.0, dy - py], [0.0, 0.0, 1.0]])
    return T @ np.asarray(sm["H"], np.float64)


def to_wall(pts_px, sm, H_m):
    """photo px (n,2) -> wall metres (n,2), x from s0, y up. NaN for points at/behind the horizon."""
    A = photo_to_tex(sm); p = np.c_[np.asarray(pts_px, np.float64).reshape(-1, 2), np.ones(len(np.atleast_2d(pts_px)))] @ A.T
    with np.errstate(divide="ignore", invalid="ignore"):
        q = p[:, :2] / p[:, 2:3]
    q[p[:, 2] <= 1e-9] = np.nan
    return np.c_[q[:, 0] * TEX_M, H_m - q[:, 1] * TEX_M]


def _strip_rows_cols(rect, sm):
    """Wall rect -> (r0, r1, c0, c1) in STRIP px, clipped to the strip; None when empty."""
    px, py = sm["pad"]; dx, dy = sm.get("offset_px", (0, 0)); h = sm["h"]; Hs, Ws = sm["valid"].shape
    x0, y0, x1, y1 = rect
    c0 = int(round(x0 / TEX_M)) + px - dx; c1 = int(round(x1 / TEX_M)) + px - dx
    r0 = h - int(round(y1 / TEX_M)) + py - dy; r1 = h - int(round(y0 / TEX_M)) + py - dy
    r0, r1, c0, c1 = max(0, r0), min(Hs, r1), max(0, c0), min(Ws, c1)
    return (r0, r1, c0, c1) if r1 > r0 and c1 > c0 else None


def _seen(rect, sm):
    s = _strip_rows_cols(rect, sm)
    return float(sm["valid"][s[0]:s[1], s[2]:s[3]].mean()) if s else 0.0


def box_to_rect(bbox, sm, H_m):
    """-> (x0, y0, x1, y1, in_quad, seen). Rect from the warped EDGE MIDPOINTS (the bounding rect of the warped quad
    over-reads 10-25 % on oblique views); sorted so x0 < x1, y0 < y1 (the texture is mirrored vs the photo)."""
    bx0, by0, bx1, by1 = bbox
    P = to_wall([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]], sm, H_m)            # photo TL, TR, BR, BL
    if not np.isfinite(P).all(): return None
    left, right, top, bot = (P[0] + P[3]) / 2, (P[1] + P[2]) / 2, (P[0] + P[1]) / 2, (P[2] + P[3]) / 2
    x0, x1 = sorted((left[0], right[0])); y0, y1 = sorted((bot[1], top[1]))
    W_m = sm["w"] * TEX_M
    area = max((x1 - x0) * (y1 - y0), 1e-9)
    clip = max(0.0, min(x1, W_m) - max(x0, 0.0)) * max(0.0, min(y1, H_m) - max(y0, 0.0))
    rect = (x0, y0, x1, y1)
    return (*rect, clip / area, _seen(rect, sm))


def _inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _area(a):
    return max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)


def inside(B, A, frac=0.8):
    return _area(B) > 0 and _inter(B, A) / _area(B) >= frac


def _iou(a, b):
    i = _inter(a, b); return i / max(_area(a) + _area(b) - i, 1e-9)


# ── instances ───────────────────────────────────────────────────────────────
def load_instances(elements_dir, masks_dir, i):
    """json entries + occ (occluder mask mean over bbox), bld (building mask mean), own (id-map pixels / area)."""
    elements_dir, masks_dir = Path(elements_dir), Path(masks_dir)
    jf = elements_dir / f"{i:02d}.json"
    if not jf.exists(): return []
    inst = json.load(open(jf))
    idmap = cv2.imread(str(elements_dir / f"{i:02d}.png"), cv2.IMREAD_UNCHANGED)
    shape = idmap.shape if idmap is not None else (640, 640)
    rd = lambda p, default: (lambda m: m > 127 if m is not None else np.full(shape, default, bool))(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
    occ = rd(masks_dir / f"{i:02d}.png", False); bld = rd(masks_dir / f"{i:02d}_building.png", True)
    out = []
    for e in inst:
        x0, y0, x1, y1 = [int(v) for v in e["bbox"]]
        sl = (slice(max(0, y0), max(y0 + 1, y1)), slice(max(0, x0), max(x0 + 1, x1)))
        own = float((idmap[sl] == e["id"]).sum() / max(e["area"], 1)) if idmap is not None else 1.0
        out.append(dict(e, occ=float(occ[sl].mean()), bld=float(bld[sl].mean()), own=own))
    return out


def _storey(y0, y1, H_m):
    return "upper" if y1 > H_m - 3.0 and y0 > 2.5 else "ground"


def _type_ok(t, x0, y0, x1, y1, H_m, sm, garage=False):
    w, h = x1 - x0, y1 - y0
    if t == "window": return 0.4 <= w <= 4.5 and 0.5 <= h <= 2.5 and y1 <= H_m - 0.3
    if t == "door":                                                          # a door stands on the ground: its head height is its height
        if y0 > 0.6:                                                         # car in front: the band below must be unseen (netloft #17 entrance, box cut at the car roof)
            s = _strip_rows_cols((x0, 0.0, x1, y0), sm)
            if s is None or sm["valid"][s[0]:s[1], s[2]:s[3]].mean() >= 0.5: return False
        return 0.6 <= w <= (5.0 if garage else 3.0) and 1.8 <= y1 <= 3.3
    if t == "storefront": return 1.8 <= h <= 3.5 and y0 <= 1.2 and w >= 1.8 and h <= max(0.5 * H_m, 3.5)   # y0 <= 1.2: netloft #16 sits 0.6 m high before registration; a 1-storey shop wall is mostly glazing
    if t == "awning": return h <= AWNING_MAX and w >= 1.0 and 2.2 <= y0 <= 4.0
    return False


def _candidates(k, sm, insts, H_m, W_m, rej):
    """Stage 1-2: instances of one strip -> candidate dicts in wall metres (gates: score, in_quad, occ, bld, edge, seen, type)."""
    out = []
    for e in sorted(insts, key=lambda e: e["label"] != "awning"):           # awnings first: a storefront box includes the awning above it
        lab = e["label"]; t = "door" if lab == "garage door" else lab
        if t not in TYPES: continue
        if e["score"] < (AWNING_SCORE if t == "awning" else MIN_SCORE): rej.append((e["bbox"], "s")); continue
        r = box_to_rect(e["bbox"], sm, H_m)
        if r is None: rej.append((e["bbox"], "h")); continue
        x0, y0, x1, y1, in_quad, seen = r
        if e["score"] < MIN_SCORE and x1 - x0 < AWNING_WIDE: rej.append((e["bbox"], "s")); continue
        xin = max(0.0, min(x1, W_m) - max(x0, 0.0)) / max(x1 - x0, 1e-6); yin = max(0.0, min(y1, H_m) - max(y0, 0.0)) / max(y1 - y0, 1e-6)
        if xin < 0.6 or yin < 0.5: rej.append((e["bbox"], "q")); continue    # judged per axis: a storefront box includes the roof above (netloft #7, no awning found)
        if e["occ"] >= 0.5: rej.append((e["bbox"], "o")); continue
        if min(e["bld"], e.get("bld_own", 1.0)) <= 0.5: rej.append((e["bbox"], "b")); continue
        tol = EDGE_TOL.get(t, EDGE_DEFAULT)
        if x0 < -tol or x1 > W_m + tol: rej.append((e["bbox"], "x")); continue
        x0, x1, y0, y1 = max(x0, 0.0), min(x1, W_m), max(y0, 0.0), min(y1, H_m)
        if t == "storefront": y1 = min(y1, H_m - 0.5)                          # the fascia is never glazing
        if seen < MIN_SEEN and not _behind_occluder(e, (x0, y0, x1, y1), sm): rej.append((e["bbox"], "e")); continue
        if t == "storefront":
            for a in (a for a in out if a["type"] == "awning" and min(a["rect"][2], x1) - max(a["rect"][0], x0) > 0):
                if y1 > a["rect"][1] and a["rect"][1] - y0 >= 1.8: y1 = a["rect"][1]
        if not _type_ok(t, x0, y0, x1, y1, H_m, sm, garage=lab == "garage door"): rej.append((e["bbox"], "t")); continue
        y_obs = y0
        if t == "door": y0 = 0.0
        out.append(dict(type=t, rect=[x0, y0, x1, y1], y_obs=y_obs, score=float(e["score"]), strip=k, photo=int(sm["photo"]), id=e["id"], bbox=e["bbox"], ppm=float(sm["ppm_src"]),
                        own=e["own"], occ=e["occ"], seen=seen, lights=[], storey=_storey(y0, y1, H_m)))
    return out


def _behind_occluder(e, rect, sm):
    """An unseen rect is still evidence of position when an occluder (leaves, a pole) explains it: the box is inside the frame,
    not in the columns mask_end cut (neighbour), and the occluder mask covers part of it (netloft #17: the tree-hidden window)."""
    bx0, by0, bx1, by1 = e["bbox"]
    if min(bx0, by0) < 2 or max(bx1, by1) > IMG_W - 2 or e["occ"] < OCC_EVIDENCE: return False
    c0, c1 = sm.get("cut_cols", (0, 0)); s = _strip_rows_cols(rect, sm)
    return s is not None and s[2] >= c0 and s[3] <= sm["valid"].shape[1] - c1


def _merge_strip(cands, rej):
    """Stage 3: nesting / cross-prompt merge within one strip. Awnings are a separate layer (never absorbed)."""
    keep = [c for c in cands if c["type"] == "awning"]
    rest = sorted([c for c in cands if c["type"] != "awning"], key=lambda c: -_area(c["rect"]))
    dead = set()
    # (a) same-type parent with >= 2 small children -> children become lights
    for i, p in enumerate(rest):
        if i in dead: continue
        ch = [j for j, c in enumerate(rest) if j != i and j not in dead and c["type"] == p["type"] and inside(c["rect"], p["rect"]) and _area(c["rect"]) < 0.7 * _area(p["rect"])]
        if p["type"] == "storefront":                                    # bays beat the whole-shopfront box when they cover >= 60 % of its width
            if ch and sum(rest[j]["rect"][2] - rest[j]["rect"][0] for j in ch) >= 0.6 * (p["rect"][2] - p["rect"][0]): dead.add(i); rej.append((p["bbox"], "n"))
            continue
        if len(ch) >= 2:
            p["lights"] = [list(rest[j]["rect"]) for j in ch]; p["score"] = max([p["score"]] + [rest[j]["score"] for j in ch])
            for j in ch: dead.add(j); rej.append((rest[j]["bbox"], "l"))
    # (b) cross-type IoU > 0.7 -> one element, type by storey priority
    prio = lambda c: (c["type"] == "window") if c["storey"] == "upper" else {"storefront": 3, "door": 2, "window": 1}[c["type"]]
    for i, a in enumerate(rest):
        if i in dead: continue
        for j in range(i + 1, len(rest)):
            b = rest[j]
            if j in dead or a["type"] == b["type"] or _iou(a["rect"], b["rect"]) <= 0.7: continue
            win = a if prio(a) >= prio(b) else b; lose = b if win is a else a
            win["score"] = max(a["score"], b["score"]); win["lights"] = win["lights"] or lose["lights"]
            dead.add(rest.index(lose)); rej.append((lose["bbox"], "m"))
            if lose is a: break
    # (c) a ground storefront absorbs the windows / doors inside it as lights; everything else inside a kept box is dropped
    for i, p in enumerate(rest):
        if i in dead: continue
        for j, c in enumerate(rest):
            if j == i or j in dead or not inside(c["rect"], p["rect"]): continue
            if p["type"] == "storefront" and p["storey"] == "ground" and c["type"] in ("window", "door"):
                p["lights"].append(list(c["rect"])); rej.append((c["bbox"], "l"))
            else: rej.append((c["bbox"], "n"))
            dead.add(j)
    keep += [c for i, c in enumerate(rest) if i not in dead]
    return keep


# ── registration across strips ──────────────────────────────────────────────
def _features(cands, sm):
    """Unique features of a strip: (type, cx, cy, occ). Lone windows (w >= 0.9, no same-row window centre within 1.5 m) and
    doors. Shop panes (0.88 m alias) never qualify; awning ends neither (see below)."""
    f = []
    W_m = sm["w"] * TEX_M
    cands = [c for c in cands if c["rect"][0] > 0.05 and c["rect"][2] < W_m - 0.05]     # clipped at a wall end: the centre is an artifact (railspur door 35)
    wins = [c for c in cands if c["type"] == "window"]
    for c in wins:
        x0, y0, x1, y1 = c["rect"]; cx = (x0 + x1) / 2
        if x1 - x0 < 0.9: continue
        cy = (y0 + y1) / 2
        if any(abs((o["rect"][0] + o["rect"][2]) / 2 - cx) < 1.5 and abs((o["rect"][1] + o["rect"][3]) / 2 - cy) < 0.6 for o in wins if o is not c): continue
        f.append(("window", cx, cy, c["occ"]))
    f += [("door", (c["rect"][0] + c["rect"][2]) / 2, (c["rect"][1] + c["rect"][3]) / 2, c["occ"]) for c in cands if c["type"] == "door"]
    # ponytail: no awning x-ends — SAM splits an awning into arbitrary segments (netloft #15/#16 registered 2 m off on them); add band ends when a wall needs them
    return f


def _fit(pairs):
    """pairs [(x, y, x_ref, y_ref, clean)] -> (a, b, c, max_residual). The y shift c uses only pairs unoccluded on both
    sides (a leaf-cut box has a biased centre: netloft #8 window 1); with none, y is kept (c = 0) and not judged."""
    P = np.asarray(pairs, float); x, y, xr, yr, clean = P.T
    a = float(np.polyfit(x, xr, 1)[0]) if len(P) >= 2 and np.ptp(x) > 0.5 else 1.0
    a = float(np.clip(a, 0.85, 1.15)); b = float(np.mean(xr - a * x)); cl = clean > 0.5
    c = float(np.median((yr - y)[cl])) if cl.any() else 0.0
    res = float(max(np.abs(a * x + b - xr).max(), np.abs(y + c - yr)[cl].max() if cl.any() else 0.0))
    return a, b, c, res


def register(cands, sm, ref_feats):
    """Fit x' = a x + b, y' = y + c from unique-feature matches -> (a, b, c, n) or None."""
    pairs = []
    for t, cx, cy, occ in _features(cands, sm):
        best = None
        for tr, rx, ry, rocc in ref_feats:
            if tr != t or abs(ry - cy) >= 0.6 or abs(rx - cx) >= 2.5: continue
            if best is None or abs(rx - cx) < abs(best[0] - cx): best = (rx, ry, float(occ < 0.25 and rocc < 0.25))
        if best: pairs.append((cx, cy, *best))
    if not pairs: return None
    if len(pairs) == 1:                                                   # one window / door: translation only (netloft #17 <-> #8 share one upper window)
        cx, cy, rx, ry, clean = pairs[0]
        return (1.0, rx - cx, (ry - cy) if clean else 0.0, 1)
    a, b, c, res = _fit(pairs)
    for _ in range(2):                                                    # up to two wrong matches (a shop pane alias) are dropped
        if res < 0.3 or len(pairs) < 3: break
        worst = int(np.argmax([max(abs(a * p[0] + b - p[2]), abs(p[1] + c - p[3]) if p[4] else 0.0) for p in pairs]))
        pairs.pop(worst); a, b, c, res = _fit(pairs)
    return (a, b, c, len(pairs)) if res < 0.3 else None


def _xform(c, a, b, dy, W_m, H_m):
    """Register a candidate's rect and lights; clipped to the wall. False when < 0.3 m of it remains inside."""
    tr = lambda r: [min(max(a * r[0] + b, 0.0), W_m), min(max(r[1] + dy, 0.0), H_m), min(max(a * r[2] + b, 0.0), W_m), min(max(r[3] + dy, 0.0), H_m)]
    c["rect"] = tr(c["rect"]); c["lights"] = [l for l in (tr(l) for l in c["lights"]) if l[2] - l[0] > 0.1 and l[3] - l[1] > 0.1]
    if "y_obs" in c: c["y_obs"] = min(max(c["y_obs"] + dy, 0.0), H_m)
    return c["rect"][2] - c["rect"][0] >= 0.3 and c["rect"][3] - c["rect"][1] >= 0.3


def _coverage(sms, w):
    """Fraction of texture rows seen per column by the given strips (stitch placement)."""
    cov = np.zeros(w, float)
    if not sms: return cov
    n = 0
    for sm in sms:
        px, py = sm["pad"]; dx, dy = sm.get("offset_px", (0, 0)); v = sm["valid"]
        c0 = px - dx; col = np.zeros(w, float)
        lo, hi = max(0, -c0), min(w, v.shape[1] - c0)
        if hi > lo: col[lo:hi] = v[:, lo + c0:hi + c0].mean(0)
        cov = np.maximum(cov, col)
    return cov


# ── grammar ─────────────────────────────────────────────────────────────────
def _rows(els, tol=ROW_TOL):
    """Cluster same-type elements by y-centre -> sets el['row']."""
    rid = 0
    for t in TYPES:
        grp = sorted([e for e in els if e["type"] == t], key=lambda e: (e["y0"] + e["y1"]) / 2)
        prev = None
        for e in grp:
            cy = (e["y0"] + e["y1"]) / 2
            if prev is None or cy - prev > tol: rid += 1
            e["row"] = rid; prev = cy
    return els


def _snap_rows(els):
    for r in {e["row"] for e in els}:
        m = [e for e in els if e["row"] == r]
        if len(m) < 2: continue
        y0, y1 = np.median([e["y0"] for e in m]), np.median([e["y1"] for e in m])
        tol = SNAP_TOL_SF if m[0]["type"] == "storefront" else SNAP_TOL     # bays share one sill and one head (netloft #16 sits 0.6 m high)
        for e in m:
            if abs(e["y0"] - y0) <= tol and abs(e["y1"] - y1) <= tol:
                d0, d1 = y0 - e["y0"], y1 - e["y1"]; e["y0"], e["y1"] = float(y0), float(y1)
                e["lights"] = [[l[0], l[1] + d0, l[2], l[3] + d1] for l in e["lights"]]


def _pitch_fill(els, sms, w):
    """A missing slot in a row of >= 3 members sharing a pitch, inside the seen x-range and occluded in every strip -> sibling."""
    xr = np.flatnonzero(_coverage(sms, w) > 0)
    if xr.size == 0: return []
    xlo, xhi = xr[0] * TEX_M, xr[-1] * TEX_M
    new = []
    for r in {e["row"] for e in els}:
        m = sorted([e for e in els if e["row"] == r and e["type"] == "window"], key=lambda e: (e["x0"] + e["x1"]) / 2)
        if len(m) < 3: continue
        cx = np.array([(e["x0"] + e["x1"]) / 2 for e in m]); d = np.diff(cx); p = float(np.median(d))
        if p < 1.0 or (np.abs(d - p) < max(0.3, 0.1 * p)).sum() < 2: continue
        for e in m:
            for s in ((e["x0"] + e["x1"]) / 2 - p, (e["x0"] + e["x1"]) / 2 + p):
                if not (xlo <= s <= xhi) or np.abs(cx - s).min() < 0.3 or any(abs((n["x0"] + n["x1"]) / 2 - s) < 0.3 for n in new): continue
                hw = (e["x1"] - e["x0"]) / 2; rect = (s - hw, e["y0"], s + hw, e["y1"])
                if any(min(n["x1"], rect[2]) - max(n["x0"], rect[0]) > 0 for n in m + new): continue   # the slot must be free
                if max([_seen(rect, sm) for sm in sms] or [0]) >= 0.5: continue
                sib = dict(e, x0=rect[0], x1=rect[2], lights=[[l[0] + s - (e["x0"] + e["x1"]) / 2, l[1], l[2] + s - (e["x0"] + e["x1"]) / 2, l[3]] for l in e["lights"]],
                           src_kind="sibling", source=None, crop=None, conf=0.4, photos=[], obs=[], notes=f"pitch {p:.2f} m slot, occluded")
                new.append(sib)
    return new


def _snap_widths(els, tol=0.3):
    """Windows of a row (>= 3) whose width is within tol of the row median take the median width about their centre;
    in a row of >= 4 every single-light window does (an over-scaled strip's 3.9 m and a leaf-cut 1.3 m window in
    netloft #15/#17's 2.15 m row) — a genuine double (lights / parts) keeps its width."""
    for r in {e["row"] for e in els}:
        m = [e for e in els if e["row"] == r and e["type"] == "window"]
        if len(m) < 3: continue
        wm = float(np.median([e["x1"] - e["x0"] for e in m]))
        for e in m:
            w0 = e["x1"] - e["x0"]
            if (abs(w0 - wm) <= tol * wm or (len(m) >= 4 and len(e.get("lights") or []) < 2 and not e.get("parts"))) and abs(w0 - wm) > 1e-6:
                cx = (e["x0"] + e["x1"]) / 2; f = wm / w0
                e["lights"] = [[cx + (l[0] - cx) * f, l[1], cx + (l[2] - cx) * f, l[3]] for l in e["lights"]]
                e["x0"], e["x1"] = cx - wm / 2, cx + wm / 2


def _merge_abutting(els):
    """Same-type same-row rects closer than ABUT in x are one element: the sashes of a 3-light window (netloft facade04),
    a double door (facade08), a narrow window pair (railspur facade03); bays stay separate shopfronts. The union keeps the parts as lights and no crop
    (a row-mate's crop or a synth with the right light count is cleaner than two stitched slivers)."""
    out = []
    for e in sorted(els, key=lambda e: (e["type"], e["row"], e["x0"])):
        p = out[-1] if out else None
        if p and p["type"] == e["type"] and p["row"] == e["row"] and e["type"] in ("window", "door") and e["x0"] - p["x1"] < ABUT and min(p["y1"], e["y1"]) - max(p["y0"], e["y0"]) > 0:
            parts = (p.get("parts") or [[p["x0"], p["y0"], p["x1"], p["y1"]]]) + [[e["x0"], e["y0"], e["x1"], e["y1"]]]
            p.update(x0=min(p["x0"], e["x0"]), x1=max(p["x1"], e["x1"]), y0=min(p["y0"], e["y0"]), y1=max(p["y1"], e["y1"]), parts=parts, lights=parts,
                     score=max(p["score"], e["score"]), photos=sorted(set(p["photos"]) | set(e["photos"])), obs=p["obs"] + e["obs"], notes="merged abutting parts")
            continue
        out.append(e)
    return out


def _absorb_ground_windows(els):
    """On a shopfront wall a ground-floor 'window' is glazing of a bay: inside a bay -> its light; touching a bay's end
    -> the bay grows over it (netloft #17: three slivers of the entrance glazing between two bays from other strips)."""
    sf = [e for e in els if e["type"] == "storefront"]
    if not sf: return els
    keep = []
    for e in els:
        if e["type"] == "window" and e["y0"] < 1.5 and e["y1"] <= 3.6:
            a = _area((e["x0"], e["y0"], e["x1"], e["y1"]))
            hit = max(sf, key=lambda b: _inter((e["x0"], e["y0"], e["x1"], e["y1"]), (b["x0"], b["y0"], b["x1"], b["y1"])))
            if _inter((e["x0"], e["y0"], e["x1"], e["y1"]), (hit["x0"], hit["y0"], hit["x1"], hit["y1"])) >= 0.5 * a:
                hit["lights"].append([e["x0"], e["y0"], e["x1"], e["y1"]]); continue
            near = min(sf, key=lambda b: min(abs(b["x0"] - e["x1"]), abs(b["x1"] - e["x0"])))
            if min(abs(near["x0"] - e["x1"]), abs(near["x1"] - e["x0"])) <= 0.6:
                near["x0"], near["x1"] = min(near["x0"], e["x0"]), max(near["x1"], e["x1"]); near["lights"].append([e["x0"], e["y0"], e["x1"], e["y1"]]); continue
        keep.append(e)
    return keep


def _h(c): return c["rect"][3] - c["rect"][1]


def _bands(cands_by_strip, sms):
    """Awning rects -> horizontal bands (x-gaps < 0.5 m merge); y from the least oblique strip; dedup across strips by overlap."""
    bands = []
    for k, cands in cands_by_strip.items():
        aw = sorted([c for c in cands if c["type"] == "awning"], key=lambda c: c["rect"][0])
        cur = None
        for c in aw:
            ov = min(cur["members"][0]["rect"][3], c["rect"][3]) - max(cur["members"][0]["rect"][1], c["rect"][1]) if cur else 0.0
            if cur and c["rect"][0] - cur["x1"] < 0.5 and ov >= 0.5 * min(_h(cur["members"][0]), _h(c)):   # same band only at the same height (railspur #3: wing roof vs door canopy)
                cur["x1"] = max(cur["x1"], c["rect"][2]); cur["members"].append(c)
            else:
                if cur: bands.append(cur)
                cur = dict(x0=c["rect"][0], x1=c["rect"][2], members=[c], strip=k, oblique=sms[k]["oblique"])
        if cur: bands.append(cur)
    for b in bands:
        best = max(b["members"], key=lambda c: c["score"])
        b["y0"], b["y1"] = best["rect"][1], min(best["rect"][3], best["rect"][1] + AWNING_MAX_H)   # the underside is what SAM sees best; the box may include a canopy roof
    merged = []
    for b in sorted(bands, key=lambda b: b["oblique"]):
        hit = next((m for m in merged if min(m["x1"], b["x1"]) - max(m["x0"], b["x0"]) > 0 and min(m["y1"], b["y1"]) - max(m["y0"], b["y0"]) > 0), None)
        if hit:                                                          # extend only beyond what the better strip saw (parallax stretches an oblique awning end)
            ref = sms[hit["strip"]]
            if b["x0"] < hit["x0"] and _seen((b["x0"], hit["y0"], hit["x0"], hit["y1"]), ref) < 0.5: hit["x0"] = b["x0"]
            if b["x1"] > hit["x1"] and _seen((hit["x1"], hit["y0"], b["x1"], hit["y1"]), ref) < 0.5: hit["x1"] = b["x1"]
            hit["members"] += b["members"]
        else: merged.append(b)
    out = []
    for b in merged:
        obs = b["members"]
        out.append(dict(type="awning", x0=b["x0"], y0=b["y0"], x1=b["x1"], y1=b["y1"], score=max(c["score"] for c in obs), obs=obs,
                        photos=sorted({c["photo"] for c in obs}), lights=[], notes=f"y from strip {b['strip']} (oblique {b['oblique']:.2f})"))
    return out


# ── crops ───────────────────────────────────────────────────────────────────
def _quad4(idmap, e):
    """4-point convex quad of the instance's own pixels, or None."""
    x0, y0, x1, y1 = [int(v) for v in e["bbox"]]
    m = (idmap[y0:y1, x0:x1] == e["id"]).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return None
    hull = cv2.convexHull(max(cs, key=cv2.contourArea))
    per = cv2.arcLength(hull, True)
    for f in (0.02, 0.04, 0.06, 0.08, 0.1, 0.15):
        q = cv2.approxPolyDP(hull, f * per, True).reshape(-1, 2).astype(np.float32)
        if len(q) == 4 and cv2.isContourConvex(q) and cv2.contourArea(q) >= 0.7 * (x1 - x0) * (y1 - y0): return q + (x0, y0)
    return None


def _crop(e, ob, sm, img_rgb, idmap, occ=None):
    """Photo -> element rect through the instance's own quad. Rows mostly behind an occluder (a car at a door's base)
    take the median colour of the clean rows: never a car roof in a door."""
    wc, hc = int(round((e["x1"] - e["x0"]) / TEX_M)), int(round((e["y1"] - e["y0"]) / TEX_M))
    if wc < 2 or hc < 2 or e.get("parts"): return None
    y_obs = max(ob.get("y_obs", e["y0"]), e["y0"]); frac = (e["y1"] - y_obs) / max(e["y1"] - e["y0"], 1e-6)
    if frac < CROP_OBS: return None                                 # a door box cut at a car roof: too little seen to paint
    hc_obs = max(2, min(hc, int(round(hc * frac))))
    bx0, by0, bx1, by1 = ob["bbox"]
    q = _quad4(idmap, ob) if ob["own"] >= 0.5 and idmap is not None else None
    if q is None: q = np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]], np.float32)
    Pw = to_wall(q, sm, e["H_m"])
    corners = np.array([[e["x0"], e["y1"]], [e["x1"], e["y1"]], [e["x1"], y_obs], [e["x0"], y_obs]])   # TL TR BR BL in wall terms (observed part)
    order = [int(np.argmin(np.linalg.norm(corners - p, axis=1))) for p in Pw]
    if sorted(order) != [0, 1, 2, 3]:
        q = np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]], np.float32); Pw = to_wall(q, sm, e["H_m"])
        order = [int(np.argmin(np.linalg.norm(corners - p, axis=1))) for p in Pw]
        if sorted(order) != [0, 1, 2, 3]: return None
    src = np.zeros((4, 2), np.float32)
    for pt, o in zip(q, order): src[o] = pt
    dst = np.array([[0, 0], [wc, 0], [wc, hc_obs], [0, hc_obs]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    ppm_here = np.hypot(*(src[1] - src[0])) / max(e["x1"] - e["x0"], 1e-6)
    if ppm_here < CROP_PPM: return None                             # the element itself is too small in this photo (a far shot)
    out = cv2.warpPerspective(img_rgb, M, (wc, hc_obs), flags=cv2.INTER_AREA if ppm_here > 1 / TEX_M else cv2.INTER_LINEAR)
    if occ is not None:
        bad = cv2.warpPerspective(occ.astype(np.uint8), M, (wc, hc_obs), flags=cv2.INTER_NEAREST).mean(1) > 0.3
        if bad.mean() > 0.4: return None                            # mostly behind something: synth, never a car roof or a passer-by
        if bad.any() and (~bad).sum() >= 3: out[bad] = np.median(out[~bad].reshape(-1, 3), axis=0).astype(np.uint8)
    if hc_obs < hc:                                                 # the unobserved base (behind a car) continues the element's own bottom rows
        out = np.vstack([out, np.repeat(np.median(out[-3:], axis=0, keepdims=True), hc - hc_obs, axis=0)]).astype(np.uint8)
    return out


def _photo_path(man, i):
    f = Path(man[i]["file"]); return f if f.is_absolute() else _texture.CTX.ROOT / f


# ── main entry ──────────────────────────────────────────────────────────────
def parse_wall(F, strips_meta, elements_dir, man, debug=None):
    """-> elements (wall metres) sorted by (type, x0); [] for an unseen wall. Pure: writes nothing.
    debug (dict, optional) receives per-strip kept / rejected boxes for QA: debug['boxes'][k] = (kept [(bbox,label)], rejected [(bbox, reason)])."""
    if not strips_meta: return []
    if not any(sm.get("anchor", {}).get(k) for sm in strips_meta for k in ("s0", "s1", "top")): return []   # pose-prior quads only: positions unknown to metres (netloft facade08 #21) — nothing is placed
    H_m = F["t1"] - F["t0"]; W_m = F["s1"] - F["s0"]; w = strips_meta[0]["w"]
    elements_dir = Path(elements_dir); masks_dir = elements_dir.parent / "masks"
    sms = list(strips_meta); insts_cache = {}
    boxes = {}
    # 1-3: per strip
    per = {}
    for k, sm in enumerate(sms):
        i = int(sm["photo"])
        if i not in insts_cache:
            insts = load_instances(elements_dir, masks_dir, i)
            img = cv2.imread(str(_photo_path(man, i)))
            if img is not None and sm.get("quad") is not None:          # OUR building only: SAM's mask includes the pink neighbour (netloft #15)
                occ = _texture._mask(masks_dir / f"{i:02d}.png", img.shape[:2], False); bld = _texture._mask(masks_dir / f"{i:02d}_building.png", img.shape[:2], True)
                ours = _texture.own_building(img, bld, occ, sm["quad"]); Hh, Ww = bld.shape
                ab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 1:].astype(np.float32) - 128
                ref = np.median(ab[ours], axis=0) if ours.sum() > 100 else np.zeros(2, np.float32)
                other = bld & ~occ & (np.hypot(ab[..., 0], ab[..., 1]) > 8) & (np.hypot(*(ab - ref).transpose(2, 0, 1)) > 15) & ~((ab[..., 0] < -5) & (ab[..., 1] > 5))
                # ^ building of ANOTHER saturated chroma (the pink neighbour): chroma only, our own wall in tree shade is darker but the same teal;
                #   glass, mullions and awnings are grey; foliage (green-yellow) over the wall is not a building
                for e in insts:                                          # the element itself is never wall-coloured: judge the RING around its box
                    x0, y0, x1, y1 = [int(v) for v in e["bbox"]]; ring = np.zeros((Hh, Ww), bool)
                    ring[max(0, y0 - 10):y1 + 10, max(0, x0 - 10):x1 + 10] = True; ring[max(0, y0):y1, max(0, x0):x1] = False
                    nb = bld[ring].sum(); e["bld_own"] = float(1.0 - other[ring].sum() / nb) if nb >= 50 else 1.0
            insts_cache[i] = insts
        rej = []
        c = _merge_strip(_candidates(k, sm, insts_cache[i], H_m, W_m, rej), rej)
        per[k] = c; boxes[k] = ([(x["bbox"], x["type"]) for x in c], rej)
    # 4: reference + registration (chain: each strip registers against everything accepted so far)
    anchored = [k for k, sm in enumerate(sms) if not sm.get("floating") and (sm["anchor"].get("s0") or sm["anchor"].get("s1"))]
    pool = anchored or [k for k, sm in enumerate(sms) if not sm.get("floating")] or list(range(len(sms)))
    ref = max(pool, key=lambda k: sms[k]["ppm_src"])
    order = [ref] + sorted([k for k in range(len(sms)) if k != ref], key=lambda k: (sms[k].get("floating", False), -sms[k]["ppm_src"]))
    accepted, acc_cands, reg_info = [ref], list(per[ref]), {ref: "reference"}
    for c in per[ref]: c["reg"] = True; c["floatun"] = False
    for k in order[1:]:
        fit = register(per[k], sms[k], _ref_feats(acc_cands, sms))
        if fit:
            a, b, c, n = fit
            per[k] = [cnd for cnd in per[k] if _xform(cnd, a, b, c, W_m, H_m)]
            for cnd in per[k]: cnd["reg"] = True; cnd["floatun"] = False
            reg_info[k] = f"registered x'={a:.3f}x{b:+.2f}, y{c:+.2f} ({n} pairs)"
        else:
            for cnd in per[k]: cnd["reg"] = False; cnd["floatun"] = bool(sms[k].get("floating"))
            reg_info[k] = "unregistered" + (" floating: holes only" if sms[k].get("floating") else " (stitch placement, conf x0.7)")
            if sms[k].get("floating"):                                # holes only: rects the aligned strips did not see (a tree-hidden window, an unseen end)
                kept = []
                for cnd in per[k]:
                    if max(_seen(cnd["rect"], sms[j]) for j in accepted) < 0.5: kept.append(cnd)
                    else: boxes[k][1].append((cnd["bbox"], "c"))
                per[k] = kept; boxes[k] = ([(x["bbox"], x["type"]) for x in kept], boxes[k][1])
        accepted.append(k); acc_cands += per[k]
    # 5: dedup across strips (non-awning); awnings -> bands
    els = []
    for c in sorted([c for c in acc_cands if c["type"] != "awning"], key=lambda c: (c["strip"] != ref, -c["score"] * c["seen"])):
        r = c["rect"]; cx, cy, cw, ch = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2, r[2] - r[0], r[3] - r[1]
        hit = None
        for e in els:
            if e["type"] != c["type"]: continue
            ew, eh = e["x1"] - e["x0"], e["y1"] - e["y0"]
            if abs((e["x0"] + e["x1"]) / 2 - cx) < 0.4 and abs((e["y0"] + e["y1"]) / 2 - cy) < 0.4 and abs(ew - cw) <= 0.25 * max(ew, cw) and abs(eh - ch) <= 0.25 * max(eh, ch):
                hit = e; break
        if hit:
            hit["obs"].append(c); hit["photos"] = sorted(set(hit["photos"]) | {c["photo"]}); hit["score"] = max(hit["score"], c["score"])
        else:
            els.append(dict(type=c["type"], x0=r[0], y0=r[1], x1=r[2], y1=r[3], score=c["score"], obs=[c], photos=[c["photo"]], lights=[list(l) for l in c["lights"]],
                            notes=""))
    els += _bands({k: per[k] for k in accepted}, sms)
    # 6: grammar
    els = _resolve_overlaps(els)
    for e in (e for e in els if e["type"] == "storefront"):             # a bay from another strip may still poke through the awning band
        for bnd in (x for x in els if x["type"] == "awning" and min(x["x1"], e["x1"]) - max(x["x0"], e["x0"]) > 0):
            if e["y1"] > bnd["y0"] and bnd["y0"] - e["y0"] >= 1.8: e["y1"] = bnd["y0"]; e["lights"] = [l for l in e["lights"] if l[3] <= bnd["y0"] + 0.1]
    els = _absorb_ground_windows(els)
    _rows(els); els = _merge_abutting(els); _rows(els); _snap_rows(els); _snap_widths(els)
    # 7: confidence
    for e in els:
        o = max(e["obs"], key=lambda c: c["score"] * c["seen"])
        e["conf"] = float(np.clip(e["score"] * o["seen"] * (1.0 if o.get("reg", True) else 0.7) * (0.5 if o.get("floatun") else 1.0), 0, 1))
        e["H_m"] = H_m
    # 8: crops
    imgs, idmaps, occs = {}, {}, {}
    def load(i):
        if i not in imgs:
            im = cv2.imread(str(_photo_path(man, i))); imgs[i] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else None
            idmaps[i] = cv2.imread(str(elements_dir / f"{i:02d}.png"), cv2.IMREAD_UNCHANGED)
            occs[i] = _texture._mask(masks_dir / f"{i:02d}.png", imgs[i].shape[:2], False) if imgs[i] is not None else None
        return imgs[i], idmaps[i], occs[i]
    best_date = max((man[int(sm["photo"])].get("date") or "" for sm in sms                        # newest photo that sees this wall AND could
                     if sm["ppm_src"] >= GATE_PPM and sm["oblique"] <= CROP_OBLIQUE), default="") or None   # itself supply a crop: never blank a wall

    for e in els:
        e["source"], e["crop"], e["src_kind"] = None, None, "synth"
        e["crop_gate"], e["crop_ppm"] = None, None
        if e["type"] == "awning":                                   # its own colour (median of the instance's pixels in the sharpest photo) for the ledge: white canvas, not wall
            cand = [o for o in e["obs"] if sms[o["strip"]]["ppm_src"] >= CROP_PPM]
            if cand:
                o = min(cand, key=lambda o: sms[o["strip"]]["oblique"]); img, idmap, occ = load(o["photo"])
                if img is not None and idmap is not None:
                    m = (idmap == o["id"]) & ~occ
                    if m.sum() >= 50: e["colour"] = [int(v) for v in np.median(img[m], axis=0)]
            continue
        need = CROP_SEEN_DOOR if e["type"] == "door" else CROP_SEEN
        seen_ok = [o for o in e["obs"] if o["seen"] >= need]
        rank = lambda o: (sms[o["strip"]]["oblique"], -sms[o["strip"]]["ppm_src"])
        gate = {id(o): crop_ok(sms[o["strip"]]["ppm_src"], sms[o["strip"]]["oblique"], man[o["photo"]].get("date"),
                               e["x1"] - e["x0"], e["x1"] - e["x0"], best_date,
                               (e["y1"] - max(o.get("y_obs", e["y0"]), e["y0"])) / max(e["y1"] - e["y0"], 1e-6)) for o in seen_ok}
        cand = [o for o in seen_ok if gate[id(o)] is None]
        if not cand:                                                # every candidate refused: record why the best-ranked one was
            if seen_ok: e["crop_gate"] = gate[id(min(seen_ok, key=rank))]
            continue
        o = min(cand, key=rank)
        i = o["photo"]; img, idmap, occ = load(i)
        if img is None: continue
        crop = _crop(e, o, sms[o["strip"]], img, idmap, occ)
        if crop is not None: e["crop"], e["source"], e["src_kind"], e["crop_ppm"] = crop, i, "photo", float(sms[o["strip"]]["ppm_src"])
    for e in els:                                                   # no crop -> a row sibling of nearly the same width (compose resizes it); a merged multi-part element is drawn synth with its parts as lights
        if e["crop"] is None and e["type"] != "awning" and not e.get("parts"):
            sib = [s for s in els if s["row"] == e["row"] and s is not e and s["crop"] is not None and s.get("crop_gate") is None
                   and abs((s["x1"] - s["x0"]) - (e["x1"] - e["x0"])) <= SIB_TOL * (e["x1"] - e["x0"])
                   and crop_ok(s["crop_ppm"] or GATE_PPM, 1.0, None, e["x1"] - e["x0"], s["x1"] - s["x0"]) is None]   # the resize to THIS rect must also pass
            if sib:
                s = min(sib, key=lambda s: abs((s["x1"] - s["x0"]) - (e["x1"] - e["x0"])))
                e["crop"], e["src_kind"], e["source"], e["crop_ppm"] = s["crop"], "sibling", None, s["crop_ppm"]
    els += _pitch_fill(els, [sms[k] for k in accepted], w)
    for e in els:
        e.setdefault("row", 0); e.setdefault("crop_gate", None); e.setdefault("crop_ppm", None)
        e.pop("obs", None); e.pop("H_m", None); e.pop("parts", None)
        e["photos"] = sorted(e["photos"])
    els.sort(key=lambda e: (TYPES.index(e["type"]), e["x0"]))
    if debug is not None: debug.update(boxes=boxes, reg=reg_info, ref=ref)
    return els


def _resolve_overlaps(els):
    """Same-type elements from different strips that still overlap after dedup: storefront bays are trimmed to the
    non-overlapping part (>= 1.8 m left) else dropped; other openings overlapping >= 50 % of the smaller one keep the best."""
    q = lambda e: max(o["score"] * o["seen"] * min(o["ppm"] / 25.0, 1.0) for o in e["obs"]) if e["obs"] else 0.0
    els = sorted(els, key=lambda e: -q(e))
    out = []
    for e in els:
        for o in out:
            if o["type"] != e["type"] or e["type"] == "awning": continue
            ix = min(o["x1"], e["x1"]) - max(o["x0"], e["x0"]); iy = min(o["y1"], e["y1"]) - max(o["y0"], e["y0"])
            if ix <= 0 or iy <= 0: continue
            if e["type"] == "storefront":
                if e["x0"] < o["x0"]: e["x1"] = min(e["x1"], o["x0"])
                else: e["x0"] = max(e["x0"], o["x1"])
                e["lights"] = [l for l in e["lights"] if e["x0"] <= (l[0] + l[2]) / 2 <= e["x1"]]
                if e["x1"] - e["x0"] < 1.8: e = None; break
            elif ix * iy >= 0.5 * min(_area((o["x0"], o["y0"], o["x1"], o["y1"])), _area((e["x0"], e["y0"], e["x1"], e["y1"]))): e = None; break
        if e is not None: out.append(e)
    return out


def _ref_feats(acc_cands, sms):
    feats = []
    for k in {c["strip"] for c in acc_cands}:
        feats += _features([c for c in acc_cands if c["strip"] == k], sms[k])
    return feats


def to_json(elements):
    keep = ("type", "x0", "y0", "x1", "y1", "conf", "source", "src_kind", "photos", "row", "lights", "score", "notes", "colour", "crop_gate", "crop_ppm")
    rnd = lambda v: round(float(v), 2) if isinstance(v, (float, np.floating)) else v
    return [{k: ([[rnd(x) for x in l] for l in e[k]] if k == "lights" else rnd(e[k])) for k in keep if k in e} for e in elements]


# ── QA ──────────────────────────────────────────────────────────────────────
COL = {"window": (60, 200, 60), "door": (230, 120, 30), "awning": (60, 90, 230), "storefront": (200, 60, 200)}


def diagram(W_m, H_m, elements, scale=40):
    """Wall diagram (RGB): wall rectangle, element rects labelled type/source/conf, lights dashed."""
    w, h = int(W_m * scale) + 40, int(H_m * scale) + 78
    im = np.full((h, w, 3), 245, np.uint8)
    X = lambda x: int(20 + x * scale); Y = lambda y: int(20 + (H_m - y) * scale)
    cv2.rectangle(im, (X(0), Y(H_m)), (X(W_m), Y(0)), (90, 90, 90), 2)
    for e in sorted(elements, key=lambda e: bool(e.get("crop_gate"))):   # gated last: a storefront must not paint over the refusal
        c = COL[e["type"]]; p0, p1 = (X(e["x0"]), Y(e["y1"])), (X(e["x1"]), Y(e["y0"]))
        if e.get("crop_gate"): c = (230, 40, 40)                    # gated: refused crop, drawn red with its reason
        fill = tuple(int(255 - (255 - v) * 0.35) for v in c)
        cv2.rectangle(im, p0, p1, fill, -1); cv2.rectangle(im, p0, p1, c, 2 if e.get("src_kind") == "photo" else 1)
        if e.get("src_kind") != "photo": cv2.line(im, p0, p1, c, 1)
        for l in e.get("lights", []):
            q0, q1 = (X(l[0]), Y(l[3])), (X(l[2]), Y(l[1]))
            for x in range(q0[0], q1[0], 6): cv2.line(im, (x, q0[1]), (min(x + 3, q1[0]), q0[1]), c, 1); cv2.line(im, (x, q1[1]), (min(x + 3, q1[0]), q1[1]), c, 1)
            for y in range(q0[1], q1[1], 6): cv2.line(im, (q0[0], y), (q0[0], min(y + 3, q1[1])), c, 1); cv2.line(im, (q1[0], y), (q1[0], min(y + 3, q1[1])), c, 1)
        lab = f"{e['type'][:4]} #{e['source']} {e['conf']:.2f}" if e.get("source") is not None else f"{e['type'][:4]} {e.get('src_kind', '')[:4]} {e['conf']:.2f}"
        if e.get("crop_ppm"): lab += f" {e['crop_ppm']:.0f}px/m"
        cv2.putText(im, lab, (p0[0] + 2, p0[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20, 20, 20), 1)
        if e.get("crop_gate"): cv2.rectangle(im, p0, p1, c, 3); cv2.putText(im, e["crop_gate"], (p0[0] + 2, p1[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 20, 20), 1)
    gated = [e for e in elements if e.get("crop_gate")]
    cv2.putText(im, f"gated: {len(gated)} of {len(elements)} elements" + (f" (reasons: {'; '.join(sorted({e['crop_gate'] for e in gated}))})" if gated else ""),
                (20, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 20, 20) if gated else (90, 90, 90), 1)
    cv2.putText(im, f"{W_m:.1f} x {H_m:.1f} m  ({len(elements)} elements)   texture orientation: s0 end at left (mirrored vs an outside photo); solid = photo crop, slashed = sibling/synth", (20, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    return im


def qa_image(sms, man, els, dbg, W_m, H_m, path):
    tiles = []
    for k, sm in enumerate(sms):
        i = int(sm["photo"]); im = cv2.imread(str(_photo_path(man, i)))
        kept, rej = dbg["boxes"].get(k, ([], []))
        for bb, r in rej:
            x0, y0, x1, y1 = [int(v) for v in bb]; cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 1); cv2.putText(im, r, (x0 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        for bb, t in kept:
            x0, y0, x1, y1 = [int(v) for v in bb]; cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 0), 2); cv2.putText(im, t[:4], (x0 + 2, y0 + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(im, f"strip {k} photo #{i} ppm {sm['ppm_src']:.1f} obl {sm['oblique']:.2f} {'floating' if sm.get('floating') else 'aligned'} | {dbg['reg'].get(k, '')}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
        cv2.putText(im, "red = rejected (s score q quad o occluder b building x edge e unseen t type n nested l light m merged c covered)", (6, im.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        tiles.append(im)
    top = np.hstack(tiles)
    dia = cv2.cvtColor(diagram(W_m, H_m, els), cv2.COLOR_RGB2BGR)
    Wd = max(top.shape[1], dia.shape[1])
    pad = lambda a: np.pad(a, ((0, 0), (0, Wd - a.shape[1]), (0, 0)), constant_values=255)
    crops = [e for e in els if e.get("crop") is not None]
    rows = [pad(top), pad(dia)]
    if crops:
        hc = 60; tiles2 = []
        for e in crops:
            c = e["crop"]; t = cv2.resize(cv2.cvtColor(c, cv2.COLOR_RGB2BGR), (max(4, int(c.shape[1] * hc / c.shape[0])), hc))
            t = np.vstack([t, np.full((14, t.shape[1], 3), 255, np.uint8)])
            cv2.putText(t, f"{e['crop_ppm']:.0f}" if e.get("crop_ppm") else "-", (1, hc + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (20, 20, 20), 1)
            tiles2.append(t)
        strip = np.hstack(tiles2 + [np.full((hc + 14, 4, 3), 255, np.uint8)])
        rows.append(pad(strip[:, :Wd]))
    cv2.imwrite(str(path), np.vstack(rows))


# ── self-check on real data ─────────────────────────────────────────────────

# The self-check lives with the pilot that owns the data: shadowCity2 tools/pilot/facade_elements.py
