"""Stitch metric strips (facade_rectify) of one wall into one texture: translation-only alignment.

    PILOT_BUILDING=railspur python tools/pilot/stitch.py   # -> out/pilot_railspur_model/qa_facade_stitch.png

Strips are metric (scale right to a few %), so between two strips only a translation is
unknown. It is measured by masked NCC of high-passed grey (FFT, all shifts at once), gated
to +-3 m around the prior; a peak is accepted only when it is clearly better than every peak
>= 25 px away (repeated shop bays otherwise give 10-28 m shifts). Offsets are solved by
weighted least squares with evidence-based corners as anchors; a strip whose pair residual
exceeds 0.6 m is dropped, and a strip whose measured pair with an aligned strip was refused is
demoted to floating (it fills only columns no aligned strip saw). The mosaic is one-owner: a texel comes from one strip, blending
happens only in a 12 px feather at seams -> no doubled windows.

Conventions: texture row 0 = top of wall, column 0 = the s0 end. Strip pixel (r, c) placed
with offset (dx, dy) lands on texture (r - py + dy, c - px + dx). Pair measurement d_ij = dx_j - dx_i.
"""
import math
from pathlib import Path

import cv2
import numpy as np

TEX_M = 0.04
TRIM_PX = 75                          # 3 m, the pose-error bound: a floating strip's un-evidenced (prior) ends are cut back by this much
_ERODE = np.ones((7, 7), np.uint8)


def _prior_px(s):
    return int(round(s.get("prior_dx_m", 0.0) / TEX_M)), int(round(s.get("prior_dy_m", 0.0) / TEX_M))


def highpass(rgb, valid):
    """Grey band-pass (sigma 1.5 minus sigma 10), normalised so invalid texels do not bleed in; 0 where not valid."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    v = valid.astype(np.float32); gv = g * v
    def nblur(sig):
        return cv2.GaussianBlur(gv, (0, 0), sig) / np.maximum(cv2.GaussianBlur(v, (0, 0), sig), 1e-3)
    hp = nblur(1.5) - nblur(10.0)
    hp[~valid] = 0.0
    return hp


def _fft_ncc(A, Ma, B, Mb):
    """Masked NCC for every shift s (B pixel (y,x) sits on A pixel (y+sy, x+sx)), Padfield 2012 via FFT.
    Returns (ncc, n) both indexed [sy mod S0, sx mod S1]."""
    S = (A.shape[0] + B.shape[0] - 1, A.shape[1] + B.shape[1] - 1)
    F = lambda a: np.fft.rfft2(a, S)
    fa, fa2, fma = F(A * Ma), F(A * A * Ma), F(Ma)
    fb, fb2, fmb = np.conj(F(B * Mb)), np.conj(F(B * B * Mb)), np.conj(F(Mb))
    X = lambda p, q: np.fft.irfft2(p * q, S)
    n = X(fma, fmb); sab = X(fa, fb); sa = X(fa, fmb); sb = X(fma, fb); saa = X(fa2, fmb); sbb = X(fma, fb2)
    with np.errstate(divide="ignore", invalid="ignore"):
        nn = np.maximum(n, 1.0)
        va = saa - sa * sa / nn; vb = sbb - sb * sb / nn
        ncc = (sab - sa * sb / nn) / np.sqrt(np.maximum(va, 1e-6) * np.maximum(vb, 1e-6))
    ncc[(va < 1e-3) | (vb < 1e-3)] = 0.0
    return ncc, n


def pair_shift(sa, sb, gate_px=75, dys=None, return_curve=False):
    """-> (dx, dy, score, margin) with dx = dx_b - dx_a in texture px (prior offsets included), or None
    when the overlap is < 10 000 px or the peak sits within 3 px of the gate. Acceptance is the caller's:
    score >= 0.30 and margin >= 0.05."""
    Ma = cv2.erode((sa["valid"] & sa["bld"]).astype(np.uint8), _ERODE).astype(np.float32)
    Mb = cv2.erode((sb["valid"] & sb["bld"]).astype(np.uint8), _ERODE).astype(np.float32)
    if Ma.sum() < 10000 or Mb.sum() < 10000: return None
    A = highpass(sa["rgb"], sa["valid"]); B = highpass(sb["rgb"], sb["valid"])
    ncc, n = _fft_ncc(A, Ma, B, Mb)
    pa, pb = _prior_px(sa), _prior_px(sb); (pxa, pya), (pxb, pyb) = sa["pad"], sb["pad"]
    if dys is None:                                                # both eaves snapped: no vertical freedom; else the prior's height error (up to ~1 m) is in play
        dys = (0,) if sa.get("anchor", {}).get("top") and sb.get("anchor", {}).get("top") else tuple(range(-24, 25, 4))
    d0, e0 = pb[0] - pa[0], pb[1] - pa[1]                     # prior relative offset
    ds = np.arange(-gate_px, gate_px + 1); dyv = np.asarray(dys, int)
    sx = (pxa - pxb) + d0 + ds; sy = (pya - pyb) + e0 + dyv
    # shifts outside the padded FFT range never overlap: clamp their indices to a zero-overlap cell
    ok_x = (sx > -B.shape[1]) & (sx < A.shape[1]); ok_y = (sy > -B.shape[0]) & (sy < A.shape[0])
    C = ncc[np.ix_(sy % ncc.shape[0], sx % ncc.shape[1])]; N = n[np.ix_(sy % n.shape[0], sx % n.shape[1])]
    C = np.where(ok_y[:, None] & ok_x[None, :], C, -np.inf); C[N < 10000] = -np.inf
    if not np.isfinite(C).any(): return None
    k = int(np.argmax(C)); iy, ix = divmod(k, C.shape[1])
    score = float(C[iy, ix]); dx_rel = int(ds[ix]); dy_rel = int(dyv[iy])
    if abs(dx_rel) > gate_px - 3: return None
    far = np.abs(ds - dx_rel) >= 25
    margin = float(score - C[:, far].max()) if far.any() and np.isfinite(C[:, far]).any() else float(score)
    res = (d0 + dx_rel, e0 + dy_rel, score, margin)
    if return_curve: return res, (d0 + ds, np.where(np.isfinite(C), C, np.nan).max(0))
    return res


def _wls(nodes, rows, prior):
    """rows: (kind, i, j, value, w). kind 'p': x_j - x_i = value; 'a': x_i = value."""
    idx = {i: k for k, i in enumerate(nodes)}
    A = np.zeros((len(rows), len(nodes))); b = np.zeros(len(rows))
    for r, (kind, i, j, val, w) in enumerate(rows):
        sw = math.sqrt(w)
        if kind == "p": A[r, idx[j]] = sw; A[r, idx[i]] = -sw
        else: A[r, idx[i]] = sw
        b[r] = sw * val
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    return {i: float(x[idx[i]]) for i in nodes}


def solve_offsets(strips, pairs):
    """-> (offsets {i: (dx, dy)}, dropped [i], residuals {(i, j): (rx, ry)}).
    pairs: (i, j, dx, dy, score, margin[, accepted]); only accepted ones constrain."""
    acc = [p for p in pairs if len(p) < 7 or p[6]]
    nodes = [i for i, s in enumerate(strips) if s is not None]
    prior = {i: _prior_px(strips[i]) for i in nodes}
    dropped, offsets, residuals = [], {}, {}
    while True:
        live = [i for i in nodes if i not in dropped]
        use = [p for p in acc if p[0] in live and p[1] in live]
        sol = []
        for axis, akeys in ((0, ("s0", "s1")), (1, ("top",))):
            rows = []
            for i in live:
                s = strips[i]; an = s.get("anchor", {})
                anchored = any(an.get(k) for k in akeys)
                w = (100.0 if s.get("weight", 1.0) >= 100 else 10.0) if anchored else 0.05
                rows.append(("a", i, i, prior[i][axis], w))
            for (i, j, dx, dy, score, margin, *_r) in use:
                rows.append(("p", i, j, (dx, dy)[axis], min(5.0, 20.0 * margin)))
            sol.append(_wls(live, rows, prior))
        residuals = {(i, j): (sol[0][j] - sol[0][i] - dx, sol[1][j] - sol[1][i] - dy) for (i, j, dx, dy, *_r) in use}
        bad = {}
        for (i, j), (rx, ry) in residuals.items():
            if abs(rx) > 15 or abs(ry) > 15:
                for k in (i, j): bad[k] = bad.get(k, 0.0) + abs(rx) + abs(ry)
        if not bad:
            offsets = {i: (int(round(sol[0][i])), int(round(sol[1][i]))) for i in live}
            break
        # drop the worst offender; a clicked strip (weight >= 100) only when nothing else is involved
        cand = [k for k in bad if strips[k].get("weight", 1.0) < 100] or list(bad)
        dropped.append(max(cand, key=lambda k: (bad[k], -strips[k].get("weight", 1.0))))
    for i in dropped: offsets[i] = prior[i]
    return offsets, dropped, residuals


def _col_range(s):
    """Valid texture columns [c0, c1] of a strip at its prior offset, or None."""
    cols = np.flatnonzero(s["valid"].any(0))
    if cols.size == 0: return None
    dx = _prior_px(s)[0] - s["pad"][0]
    return int(cols[0] + dx), int(cols[-1] + dx)


def stitch(strips, w, h):
    """-> (mosaic (h,w,3) uint8, valid (h,w) bool, info) or (None, None, info) when nothing is valid."""
    info = dict(order=[], offsets_m={}, pairs=[], dropped=[], floating=[], owner=None, filled=0.0)
    live = [i for i, s in enumerate(strips) if s is not None and s["valid"].any()]
    if not live: return None, None, info
    rng = {i: _col_range(strips[i]) for i in live}
    pairs = []
    for a in range(len(live)):
        for b in range(a + 1, len(live)):
            i, j = live[a], live[b]
            if min(rng[i][1], rng[j][1]) - max(rng[i][0], rng[j][0]) < 50: continue
            r = pair_shift(strips[i], strips[j])
            if r is None: pairs.append((i, j, 0, 0, 0.0, 0.0, False)); continue
            dx, dy, score, margin = r
            pairs.append((i, j, dx, dy, score, margin, bool(score >= 0.30 and margin >= 0.05)))
    offsets, dropped, residuals = solve_offsets(strips, [p for p in pairs if p[0] in live and p[1] in live])
    keep = [i for i in live if i not in dropped]
    agree, disagree = {i: set() for i in keep}, {i: set() for i in keep}
    for (i, j, _dx, _dy, score, _m, acc) in pairs:
        if i in dropped or j in dropped: continue
        if acc: agree[i].add(j); agree[j].add(i)
        elif score > 0: disagree[i].add(j); disagree[j].add(i)      # measured and refused: the two strips' content does NOT line up
    corner = lambda i: any(strips[i].get("anchor", {}).get(k) for k in ("s0", "s1"))
    q = lambda i: strips[i].get("weight", 1.0) * strips[i].get("ppm_src", 1.0)
    best_date = strips[max([i for i in keep if corner(i) or agree[i]] or keep, key=q)].get("date")
    key = lambda i: (strips[i].get("weight", 1.0) >= 100, q(i) * (1.5 if strips[i].get("date") == best_date else 1.0))
    # aligned = clicked, or agreeing with an aligned strip, or corner-anchored and not contradicted by one (railspur facade02:
    # #4 hung on its s0 corner but its width-predicted s1 was 15 % short, NCC vs #2 refused -> a fifth ghost window); else floating
    aligned, floating = [], []
    for i in sorted(keep, key=key, reverse=True):
        ok = strips[i].get("weight", 1.0) >= 100 or (agree[i] & set(aligned)) or (corner(i) and not (disagree[i] & set(aligned))) or (not aligned and agree[i])
        (aligned if ok else floating).append(i)
    order = aligned + sorted(floating, key=key, reverse=True)

    mosaic = np.zeros((h, w, 3), np.float32); filled = np.zeros((h, w), bool); owner = np.full((h, w), -1, np.int16)
    for i in order:
        s = strips[i]; dx, dy = offsets[i]; px, py = s["pad"]
        ox, oy = dx - px, dy - py                                  # texture coords of strip pixel (0, 0)
        r0, r1 = max(0, oy), min(h, oy + s["rgb"].shape[0]); c0, c1 = max(0, ox), min(w, ox + s["rgb"].shape[1])
        if r1 <= r0 or c1 <= c0: continue
        new = s["rgb"][r0 - oy:r1 - oy, c0 - ox:c1 - ox].astype(np.float32)
        cand = s["valid"][r0 - oy:r1 - oy, c0 - ox:c1 - ox]
        old = mosaic[r0:r1, c0:c1]; fil = filled[r0:r1, c0:c1]
        take = cand & ~fil
        if i in floating:                                          # unaligned: only columns nothing else has seen (a patch beside aligned texels doubles the roofline)
            take &= (filled.mean(0)[c0:c1] < 0.2)[None, :]
            t0, t1 = s.get("trim", (False, False))                  # ...and never within TRIM_PX of a pose-only end: that is where the neighbour leaks in (netloft #15)
            if t0: take[:, :max(0, px + TRIM_PX - c0)] = False
            if t1: take[:, max(0, px + s["w"] - TRIM_PX - c0):] = False
        if not take.any(): continue
        both = cand & fil
        band = np.zeros_like(both) if i in floating else both      # floating strips fill holes only, no feather
        if both.sum() >= 500:                                      # exposure: match Lab-L to what is already there
            lab = cv2.cvtColor(np.clip(new, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
            lab_old = cv2.cvtColor(np.clip(old, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
            gain = float(np.clip(np.median(lab_old[both, 0] / np.maximum(lab[both, 0], 1.0)), 0.7, 1.4))
            lab[..., 0] = np.clip(lab[..., 0] * gain, 0, 255)
            new = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
        alpha = np.zeros(cand.shape, np.float32); alpha[take] = 1.0
        if band.any():
            dist = cv2.distanceTransform((~take).astype(np.uint8), cv2.DIST_L2, 3)
            alpha[band] = np.clip(1.0 - dist[band] / 12.0, 0.0, 1.0)
        blend = take | band
        old[blend] = alpha[blend, None] * new[blend] + (1 - alpha[blend, None]) * old[blend]
        fil |= take; owner[r0:r1, c0:c1][alpha >= 0.5] = i
        info["order"].append(i)
    info.update(offsets_m={i: (dx * TEX_M, dy * TEX_M) for i, (dx, dy) in offsets.items()}, pairs=pairs, dropped=dropped,
                floating=floating, owner=owner, filled=float(filled.mean()), residuals_px=residuals)
    if not filled.any(): return None, None, info
    return np.clip(mosaic, 0, 255).astype(np.uint8), filled, info


# ── self-check on real data ──────────────────────────────────────────────────
def _qa_image(strips, idxs, pairs_curves, mosaic, info, w, h, path):
    W = 900
    fit = lambda im: cv2.resize(im, (W, max(1, int(im.shape[0] * W / im.shape[1]))), interpolation=cv2.INTER_AREA)
    label = lambda im, t: (cv2.putText(im, t, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1), im)[1]
    rows = []
    for k, i in enumerate(idxs):
        s = strips[k]; v = np.repeat(s["valid"][..., None].astype(np.uint8) * 255, 3, 2)
        rows.append(label(fit(np.hstack([cv2.cvtColor(s["rgb"], cv2.COLOR_RGB2BGR), v])), f"strip {k} = photo #{i}  offset {info['offsets_m'].get(k)} m  anchor {s['anchor']}"))
    for (a, b), (dxs, curve) in pairs_curves.items():
        p = next(q for q in info["pairs"] if q[0] == a and q[1] == b)
        im = np.full((90, W, 3), 30, np.uint8)
        fin = np.isfinite(curve)
        if fin.any():
            xs = ((dxs - dxs[0]) / max(dxs[-1] - dxs[0], 1) * (W - 20) + 10).astype(int)
            ys = (80 - np.clip(np.nan_to_num(curve, nan=-1), -1, 1) * 30 - 30).astype(int)
            pts = np.column_stack([xs, ys])[fin]
            cv2.polylines(im, [pts.astype(np.int32)], False, (0, 200, 255), 1)
            x = int(np.interp(p[2], dxs, xs)); cv2.line(im, (x, 10), (x, 85), (0, 255, 0) if p[6] else (0, 0, 255), 1)
        rows.append(label(im, f"pair {a}-{b}: dx {p[2]} px dy {p[3]} score {p[4]:.2f} margin {p[5]:.2f} {'ACCEPTED' if p[6] else 'rejected'}"))
    if mosaic is not None:
        rows.append(label(fit(cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)), f"mosaic {w}x{h}  filled {info['filled']:.2f}  order {info['order']} dropped {info['dropped']} floating {info['floating']}"))
        pal = np.array([[60, 60, 60], [0, 180, 255], [0, 200, 80], [255, 120, 0], [200, 0, 200], [0, 255, 255], [255, 255, 255]], np.uint8)
        rows.append(label(fit(pal[info["owner"] + 1]), "owner map (grey = hole)"))
    cv2.imwrite(str(path), np.vstack(rows))

# The self-check lives with the pilot that owns the data: shadowCity2 tools/pilot/facade_stitch.py
