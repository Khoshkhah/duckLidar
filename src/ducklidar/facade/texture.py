"""One wall texture from image-space evidence — the glue over facade_quad / rectify / stitch / fill.

    PILOT_BUILDING=railspur python tools/pilot/texture.py   # self-check on facade02 -> qa_facade_texture.png
    PILOT_BUILDING=netloft  python tools/pilot/texture.py   # facade00

Per candidate photo: the pose's prior quad is only a search band; facade_quad snaps the wall's
quad to LSD lines / mask boundaries, facade_rectify warps the photo to a metric strip, the
strips are stitched by translation only (facade_stitch) and the holes (cars, trees, sky rows,
unseen ends) get wall-like fill (facade_fill). The pose is used for ranking, the prior quad,
the solid's self-occlusion map — never for pixel placement. Texture row 0 = top of wall,
column 0 = the s0 end (image RIGHT for an outside camera: textures are mirrored by design).
"""
"""Wall texturing context — the host (build_model, or ducklidar.building3d) fills this in.

The facade modules are pure numpy/OpenCV; these five values are everything they need to know
about the building being processed: where photo paths are rooted, the ground and camera heights,
the solid (V, T) for self-occlusion, and the occlusion_map function itself.
"""


class Context:
    ROOT = None            # Path: photo paths in the manifest are relative to this
    Z_GROUND = 3.7         # m: ground elevation at the building
    CAM_H = 2.3            # m: camera height above ground for a street-level photo
    SOLID = {}             # dict(V=(n,3), T=(m,3)) of the building's solid, or {} to skip self-occlusion
    occlusion_map = None   # f(F, cam, w, h) -> (visible mask (h,w) bool, fraction)
    # ducklidar.building fills these in as well (they were build_model module globals)
    PHOTOS = None          # Path: the photo folder a fetcher wrote (manifest + masks/ + elements/)
    MASKS = None           # Path: occluder / building masks, NN.png and NN_building.png
    MANIFEST = []          # the photo list, for the per-photo `valid` masks
    CLICKS = None          # Path: clicks.json from the dashboard, or None
    W = 640                # px: the photos are square, W x W

    def use(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)
        return self


CTX = Context()


import json, math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import fill, quad, rectify, stitch

TEX_M = rectify.TEX_M

EXACT = 1e6                           # cand score at/above this = clicked pair, the quad is exact
HIDE_PX = 25                          # dilation of the solid's self-occlusion (pose error margin on the texture)
WEAK_SIDE = 0.5                       # side conf below this, with no snapped eave, is not evidence: the prior quad is used instead
MIN_STRIP = 0.03                      # a strip seeing < 3 % of the wall (behind the solid's own parts, off-frame) is skipped
EAVE_GAP = 1.0                        # m: a snapped eave this far below the model top means the solid's wall is taller than the real one


def _mask(path, shape, default):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return m > 127 if m is not None else np.full(shape, default, bool)


def _exact_res(prior):
    q = np.asarray(prior, np.float32).reshape(4, 2)
    return dict(quad=q, conf=dict(top=1.0, bottom=1.0, s0=1.0, s1=1.0), src=dict(top="click", bottom="click", s0="click", s1="click"),
                evidence=np.zeros((0, 4), np.float32), partial=False, ok=True, prior=q.copy())


def mask_end(strip, sl, min_bld=0.5, min_rows=10, max_ab=20.0):
    """Columns to drop from one end of a strip (walking inward, strip[:, sl]): an un-evidenced prior corner that sits
    beyond the real one lands on the neighbour or a sky gap. A column is ours when >= min_bld of its visible rows are
    building per SAM AND its chroma (median Lab a,b of those rows) is within max_ab of the strip's own wall chroma —
    SAM cannot tell the pink neighbour from Net Loft, its colour can (netloft #15). Stops at the first such column."""
    v = strip["valid"][:, sl]; bv = strip["bld"][:, sl] & v; n = v.sum(0); b = bv.sum(0)
    lab = cv2.cvtColor(strip["rgb"][:, sl], cv2.COLOR_RGB2LAB)[..., 1:].astype(np.float32)
    ref = np.median(lab[bv], axis=0) if bv.sum() >= 50 else np.array([128.0, 128.0])
    good = (n >= min_rows) & (b >= min_bld * n)
    for c in np.flatnonzero(good):
        if np.linalg.norm(np.median(lab[bv[:, c], c], axis=0) - ref) <= max_ab: return int(c)
    return v.shape[1]


def own_building(img_bgr, bld, occ, prior, lab_gate=30.0, grow_px=40):
    """The building mask restricted to OUR building: SAM 3 says 'building' for the red neighbour too, so the
    fitter locked onto it (netloft facade10). Reference colour = median Lab of building pixels inside the prior
    quad shrunk by 20 %; keep building pixels within lab_gate of it (chroma weighs double: teal vs red is chroma,
    lit vs shaded wall is L), then only the connected components that touch the prior quad grown by grow_px.
    Falls back to the full mask when the prior interior holds too few building pixels to set a reference."""
    q = np.asarray(prior, np.float32).reshape(4, 2); c = q.mean(0); inner = (c + 0.8 * (q - c)).astype(np.int32)
    core = np.zeros(bld.shape, np.uint8); cv2.fillPoly(core, [inner], 1); core = core.astype(bool) & bld & ~occ
    if core.sum() < 300: return bld
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32); ref = np.median(lab[core], axis=0)
    d = np.sqrt(((lab[..., 0] - ref[0]) ** 2) + 2 * ((lab[..., 1] - ref[1]) ** 2 + (lab[..., 2] - ref[2]) ** 2))
    ours = bld & (d < lab_gate)
    ours = cv2.morphologyEx(ours.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
    n, lab_cc = cv2.connectedComponents(ours.astype(np.uint8))
    near = np.zeros(bld.shape, np.uint8); cv2.fillPoly(near, [q.astype(np.int32)], 1)
    near = cv2.dilate(near, np.ones((2 * grow_px + 1,) * 2, np.uint8)).astype(bool)
    keep = np.zeros(bld.shape, bool)
    for k in range(1, n):
        comp = lab_cc == k
        if (comp & near).any(): keep |= comp
    return keep if keep.sum() >= 0.5 * core.sum() else bld


def texture_wall(F, cands, w, h, man, masks_dir):
    """cands: [(photo index, prior quad float32 (4,2) TL,TR,BR,BL, score)], score >= EXACT = clicked pair.
    -> (tex RGB (h,w,3), valid (h,w) bool, info) or (None, None, dict(reason=...)) when nothing usable was seen."""
    ctx = CTX                                                      # the host supplies ROOT / Z_GROUND / CAM_H / SOLID / occlusion_map
    masks_dir = Path(masks_dir); W_m, H_m = F["s1"] - F["s0"], F["t1"] - F["t0"]
    strips, meta = [], []                                          # meta[k] = (photo i, res, img, exact) for strips[k]
    for i, prior, score in cands:
        p = man[i]; img = cv2.imread(str(ctx.ROOT / p["file"]))
        if img is None: continue
        occ = _mask(masks_dir / f"{i:02d}.png", img.shape[:2], False); bld = _mask(masks_dir / f"{i:02d}_building.png", img.shape[:2], True)
        pitch, fov = p.get("pitch", 8), p.get("fov", 90)             # metadata: the refined pitch sits at the search bounds
        cam_z = p.get("cam_z", ctx.Z_GROUND + ctx.CAM_H); cam_h = cam_z - (F["p0"][2] + F["t0"])   # metadata height: the refined z is 20-50 % off
        exact = score >= EXACT
        bld_fit = bld if exact else own_building(img, bld, occ, prior)      # the fitter must not see the neighbour
        res = _exact_res(prior) if exact else quad.refine_quad(img, bld_fit, occ, prior, (W_m, H_m), pitch=pitch, fov=fov, cam_h=cam_h)
        if not res["ok"]: continue                                  # too far / not in this photo / inconsistent fit
        if not exact and res["src"]["top"] != "lsd" and max(res["conf"]["s0"], res["conf"]["s1"]) < WEAK_SIDE:
            # no eave and only a weak lone side: a wrong candidate (door, window column) drags the whole quad -> prior, unanchored
            res = dict(res, quad=res["prior"].copy(), conf=dict.fromkeys(res["conf"], 0.0), evidence=np.zeros((0, 4), np.float32),
                       src={k: (v if v == "offframe" else "prior") for k, v in res["src"].items()})
        ev = lambda k: res["src"][k] in ("lsd", "mask", "click")
        cut = (ev("s0"), ev("s1"), res["src"]["top"] in ("lsd", "click"), False)
        eave_m = None
        if not exact and res["src"]["top"] == "lsd":                 # the solid's wall may stand taller than the eave the photo shows (netloft facade04:
            hq = quad.quad_height(res["quad"], pitch, fov, cam_h)   # 11.1 m coplanar group, 9 m eave): put the eave at its height, not at the model top
            if hq is not None and hq < H_m - EAVE_GAP: eave_m = hq; res = dict(res, quad=quad.raise_top(res["quad"], pitch, fov, cam_h, H_m))
        strip = rectify.rectify(img, res["quad"], occ, (W_m, H_m), bld=bld, cut=cut)
        if strip is None or strip["oblique"] > 3: continue
        px, py = strip["pad"]
        if eave_m is not None: strip["valid"][:py + int((H_m - eave_m) / TEX_M), :] = False   # above the real eave: never seen, material
        cut_cols = [0, 0]
        if ctx.SOLID:                                                # self-occlusion: the one legitimate use of the pose as a mask
            vis, _ = ctx.occlusion_map(F, np.array([p["x"], p["y"], cam_z]), w, h)
            hidden = cv2.dilate((~vis).astype(np.uint8), np.ones((2 * HIDE_PX + 1,) * 2, np.uint8)).astype(bool)
            strip["valid"][py:py + h, px:px + w] &= ~hidden
        for side, sl in (("s0", slice(None)), ("s1", slice(None, None, -1))):
            if res["src"][side] != "prior": continue                  # evidence-based sides are cut by rectify; off-frame ends need no cut
            cutc = mask_end(strip, sl)
            if cutc: strip["valid"][:, sl][:, :cutc] = False            # netloft #15: the s0 prior sits 3 m inside the pink neighbour
            cut_cols[side == "s1"] = int(cutc)
        if strip["valid"][py:py + h, px:px + w].mean() < MIN_STRIP: continue   # a sliver: nothing to align, only junk to paste
        conf = float(np.mean(list(res["conf"].values())))
        strip.update(prior_dx_m=0.0, prior_dy_m=0.0, anchor=dict(s0=ev("s0"), s1=ev("s1"), top=cut[2]), date=p["date"], cut_cols=tuple(cut_cols), eave_m=eave_m,
                     trim=(res["src"]["s0"] == "prior", res["src"]["s1"] == "prior"),   # pose-only ends: a floating strip may not paint within 3 m of them
                     weight=100.0 if exact else float(score) * min(strip["ppm_src"] / 15, 1.0) * (0.5 + 0.5 * conf))
        strips.append(strip); meta.append((i, res, img, exact))
    mosaic, valid, sinfo = stitch.stitch(strips, w, h)
    if mosaic is None: return None, None, dict(reason="no strip", size=(round(W_m, 1), round(H_m, 1)))
    tex, finfo = fill.fill(mosaic, valid)
    if finfo["src"] == "flat": return None, None, dict(reason="valid < 15%", filled=float(valid.mean()), size=(round(W_m, 1), round(H_m, 1)))
    order = sinfo["order"]; used = [meta[k][0] for k in order]; i0, res0, img0, _ = meta[order[0]]
    fit = any(meta[k][1]["src"][s] in ("lsd", "mask") for k in order for s in ("top", "s0", "s1"))
    method = ("clicked" if any(meta[k][3] for k in order) else "quad-fit" if fit else "prior") + ("+stitch" if len(used) > 1 else "") + ("+fill" if valid.mean() < 0.999 else "")
    photo_of = lambda ks: [meta[k][0] for k in ks]
    info = dict(photos=used, filled=round(float(valid.mean()), 3), size=(round(W_m, 1), round(H_m, 1)), method=method,
                quads={i: dict(conf={a: round(float(b), 3) for a, b in r["conf"].items()}, src=dict(r["src"]), ok=bool(r["ok"]), partial=bool(r["partial"])) for i, r, _im, _e in meta},
                offsets_m={meta[k][0]: v for k, v in sinfo["offsets_m"].items()}, pairs=[(meta[a][0], meta[b][0], *rest) for a, b, *rest in sinfo["pairs"]],
                dropped=photo_of(sinfo["dropped"]), floating=photo_of(sinfo["floating"]), period_m=finfo.get("period_m"),
                fill={k: v for k, v in finfo.items() if k != "src_map"},
                qa=dict(best_i=i0, photo=quad.draw(img0, res0), strip=rectify.overlay(strips[order[0]]), flip=bool(res0["quad"][0, 0] > res0["quad"][1, 0]),
                        owner=sinfo["owner"], strip_photos=[m[0] for m in meta], cut_top=[s["anchor"]["top"] for s in strips]))
    # the strips as elements.parse_wall wants them (photo px -> strip px -> texture px, never the pose); floating strips carry no offset
    kept = [k for k in range(len(strips)) if k not in sinfo["dropped"] and k in sinfo["offsets_m"]]
    info["strips"] = [dict(photo=meta[k][0], H=strips[k]["H"], pad=strips[k]["pad"], w=w, h=h, valid=strips[k]["valid"], bld=strips[k]["bld"],
                           ppm_src=strips[k]["ppm_src"], oblique=strips[k]["oblique"], quad=np.asarray(meta[k][1]["quad"], np.float32).reshape(4, 2),
                           offset_px=tuple(int(round(v / TEX_M)) for v in sinfo["offsets_m"][k]) if k not in sinfo["floating"] else (0, 0),
                           floating=k in sinfo["floating"], anchor=strips[k]["anchor"], src=dict(meta[k][1]["src"]), cut_cols=strips[k]["cut_cols"], eave_m=strips[k]["eave_m"]) for k in kept]
    info["eave_m"] = next((s["eave_m"] for s in strips if s["eave_m"] is not None), None)
    return tex, valid, info


def _font(size):
    try: return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError: return ImageFont.load_default()


LAYER_PAL = np.array([[40, 40, 40], [80, 160, 255], [255, 120, 40], [255, 230, 60], [120, 230, 120], [230, 80, 230], [200, 200, 200]], np.uint8)   # compose.LAYER order


def _diagram(info, flip):
    """elements.diagram of the wall, drawn mirrored when the sheet is flipped (labels stay readable)."""
    W_m, H_m = info["size"]; els = info["elements"]
    if flip: els = [dict(e, x0=W_m - e["x1"], x1=W_m - e["x0"], lights=[[W_m - l[2], l[1], W_m - l[0], l[3]] for l in e.get("lights", [])]) for e in els]
    from . import elements as _elements
    return _elements.diagram(W_m, H_m, els)


def qa_sheet(rows, path, col_w=300):
    """rows: [(name, info, tex RGB)] one per wall -> PNG: photo with prior (magenta) and fitted (green) quad | rectified strip | final texture.
    A composed wall (info has photo_tex) shows 6 columns: photo | strip | photo-stitched | element diagram | composed | layers.
    Strip and texture are mirrored to the photo's orientation when the wall's s0 end sits at the image right (the texture itself is never flipped)."""
    font = _font(13); gap, LH = 6, 20
    def fit(im, cap=None):
        im = Image.fromarray(np.ascontiguousarray(im)); s = col_w / im.width; hh = int(im.height * s)
        if cap and hh > cap: s = cap / im.height; hh = cap
        return im.resize((max(1, int(im.width * s)), max(1, hh)), Image.BILINEAR)
    labels, tiles = ["photo: prior quad magenta, fitted quad green, evidence orange   |   rectified strip (unseen texels darkened)   |   photo-stitched texture   |   element diagram   |   composed texture (= the model's)   |   layers — all in the PHOTO's orientation (flipped when s0 is at the image right)"], [[]]
    for name, info, tex in rows:
        if not info or tex is None:
            labels.append(f"{name}  flat wall colour" + (f"  ({info['reason']})" if info else "")); tiles.append([Image.new("RGB", (col_w, 60), (150, 150, 150))]); continue
        r0 = info["quads"][info["photos"][0]]["src"]
        labels.append(f"{name}  {info['size'][0]} x {info['size'][1]} m   photos {info['photos']}   {info['method']}   filled {info['filled']:.2f}   "
                      f"top {r0['top']} / s0 {r0['s0']} / s1 {r0['s1']} / bot {r0['bottom']}" + (f"   eave {info['eave_m']:.1f} m below a {info['size'][1]} m model wall" if info.get("eave_m") else ""))
        flip = info["qa"]["flip"]; fl = lambda im: im[:, ::-1] if flip else im
        row = [fit(cv2.cvtColor(info["qa"]["photo"], cv2.COLOR_BGR2RGB)), fit(fl(info["qa"]["strip"]), 200)]
        if info.get("photo_tex") is not None and info.get("layers") is not None:
            from collections import Counter
            labels[-1] += f"   elements {dict(Counter(e['type'] for e in info['elements']))} material Lab {info['material']['colour_lab']}{' (shared)' if info['material'].get('shared') else ''}"
            row += [fit(fl(info["photo_tex"]), 200), fit(_diagram(info, flip), 200), fit(fl(tex), 200), fit(fl(LAYER_PAL[info["layers"]]), 200)]
        else: row.append(fit(fl(tex), 200))
        tiles.append(row)
    H = sum(LH + (max(t.height for t in ts) if ts else 0) + gap for ts in tiles); Wd = max(len(ts) for ts in tiles) * (col_w + gap) + gap
    sheet = Image.new("RGB", (Wd, H), "white"); d = ImageDraw.Draw(sheet); y = 0
    for lab, ts in zip(labels, tiles):
        d.text((gap, y + 3), lab, fill="black", font=font); y += LH
        for c, t in enumerate(ts): sheet.paste(t, (gap + c * (col_w + gap), y))
        y += (max(t.height for t in ts) if ts else 0) + gap
    sheet.save(path)
    return path


# ── self-check on real data ─────────────────────────────────────────────────

# The self-check lives with the pilot that owns the data: shadowCity2 tools/pilot/facade_texture.py
