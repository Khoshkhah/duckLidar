"""Rectify one photo onto one wall: quad (TL,TR,BR,BL in wall terms) -> metric strip at TEX_M.

    PILOT_BUILDING=railspur python tools/pilot/rectify.py   # -> out/pilot_railspur_model/qa_facade_rectify.png

The strip is the wall texture (w x h texels, row 0 = top of wall, column 0 = the s0 end)
padded by `pad_m` on every side so a later stitch can slide it; `valid` says which texels
were actually seen (in frame, not an occluder, not beyond an evidence-based edge, and with
at least MIN_SRC_PX source pixels per texel along the worst direction — an oblique far end
stretched > 4x is a smear, not a wall). One
warpPerspective, nothing else — the quad's correctness is the quad module's job.
For an outside camera the s0 end sits at the image RIGHT, so strips look mirrored
relative to the photo. By design; never flip.
"""
import math
from pathlib import Path

import cv2
import numpy as np

TEX_M = 0.04
MIN_SRC_PX = 0.25                     # source px per texel along the worst direction (6 px/m); below it the warp stretches > 4x


def src_scale(H, Ws, Hs, step=8):
    """Smallest singular value of the strip->image Jacobian at every strip pixel (source px per texel along the
    most stretched direction), evaluated on a `step` grid and resized."""
    Hi = np.linalg.inv(H); cs = np.arange(0, Ws, step, dtype=np.float64); rs = np.arange(0, Hs, step, dtype=np.float64)
    C, R = np.meshgrid(cs, rs); p = Hi @ np.stack([C.ravel(), R.ravel(), np.ones(C.size)])
    with np.errstate(divide="ignore", invalid="ignore"):
        w = p[2]; u, v = p[0] / w, p[1] / w
        a = (Hi[0, 0] - u * Hi[2, 0]) / w; b = (Hi[0, 1] - u * Hi[2, 1]) / w
        c = (Hi[1, 0] - v * Hi[2, 0]) / w; d = (Hi[1, 1] - v * Hi[2, 1]) / w
        s1 = a * a + b * b + c * c + d * d; det = a * d - b * c
        smin = np.sqrt(np.maximum(0.5 * (s1 - np.sqrt(np.maximum(s1 * s1 - 4 * det * det, 0))), 0))
    smin = np.nan_to_num(smin, nan=0.0, posinf=0.0, neginf=0.0).reshape(R.shape).astype(np.float32)
    return cv2.resize(smin, (Ws, Hs), interpolation=cv2.INTER_LINEAR)


def tex_size(wall_wh):
    w = int(math.ceil(wall_wh[0] / TEX_M)); h = int(math.ceil(wall_wh[1] / TEX_M))
    return max(8, min(w, 2048)), max(8, min(h, 2048))


def _convex(q):
    d = np.roll(q, -1, 0) - q
    cr = d[:, 0] * np.roll(d, -1, 0)[:, 1] - d[:, 1] * np.roll(d, -1, 0)[:, 0]
    return bool((cr > 0).all() or (cr < 0).all())


def rectify(img_bgr, quad, occ, wall_wh, pad_m=(3.0, 0.5), bld=None, cut=(False, False, False, False), extra_valid=None):
    """-> Strip dict(rgb, valid, bld, pad, w, h, H, ppm_src, oblique) or None for a degenerate quad."""
    q = np.asarray(quad, np.float32).reshape(4, 2)
    if not np.isfinite(q).all() or (np.abs(q) > 1e5).any() or not _convex(q): return None
    w, h = tex_size(wall_wh)
    px, py = int(pad_m[0] / TEX_M + 0.5), int(pad_m[1] / TEX_M + 0.5)      # half-up: 0.5 m -> 13 px (round() would give 12)
    Ws, Hs = w + 2 * px, h + 2 * py
    dst = np.array([[px, py], [px + w, py], [px + w, py + h], [px, py + h]], np.float32)
    try:
        H = cv2.getPerspectiveTransform(q, dst)
    except cv2.error:
        return None
    if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12: return None
    warp = lambda a, interp=cv2.INTER_LINEAR: cv2.warpPerspective(a, H, (Ws, Hs), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rgb = cv2.cvtColor(warp(img_bgr), cv2.COLOR_BGR2RGB)
    ih, iw = img_bgr.shape[:2]
    ones = np.zeros((ih, iw), np.float32); ones[2:-2, 2:-2] = 1.0          # 2 px frame-edge bleed is out
    valid = warp(ones) > 0.99
    occ8 = cv2.dilate(np.asarray(occ, np.uint8), np.ones((7, 7), np.uint8))   # 3 px around every occluder
    valid &= ~(warp(occ8.astype(np.float32)) > 0.01)
    if extra_valid is not None: valid &= warp(np.asarray(extra_valid, np.float32)) > 0.5
    valid &= src_scale(H, Ws, Hs) >= MIN_SRC_PX                              # oblique far ends: < 1 source px per 4 texels is a smear, not a wall
    cs0, cs1, ctop, cbot = cut
    if cs0: valid[:, :px] = False
    if cs1: valid[:, px + w:] = False
    if ctop: valid[:py, :] = False
    if cbot: valid[py + h:, :] = False
    bld_s = warp(np.asarray(bld, np.uint8), cv2.INTER_NEAREST) > 0 if bld is not None else np.ones((Hs, Ws), bool)
    e = lambda a, b: float(np.linalg.norm(q[a] - q[b]))
    ppm_src = (e(0, 1) + e(3, 2)) / 2 / max(wall_wh[0], 1e-6)
    vl = sorted((e(0, 3), e(1, 2)))
    return dict(rgb=rgb, valid=valid, bld=bld_s, pad=(px, py), w=w, h=h, H=H.astype(np.float64),
                ppm_src=ppm_src, oblique=vl[1] / max(vl[0], 1e-6))


def overlay(strip):
    """Strip RGB with unseen texels darkened 60% and the texture rectangle outlined."""
    px, py, w, h = *strip["pad"], strip["w"], strip["h"]
    im = strip["rgb"].astype(np.float32); im[~strip["valid"]] *= 0.4
    im = im.astype(np.uint8)
    cv2.rectangle(im, (px, py), (px + w - 1, py + h - 1), (0, 255, 0), 1)
    return im


# ── self-check on real data ──────────────────────────────────────────────────

# The self-check lives with the pilot that owns the data: shadowCity2 tools/pilot/facade_rectify.py
