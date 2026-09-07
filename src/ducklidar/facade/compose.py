"""Composed wall texture: a plain wall MATERIAL everywhere + each detected element drawn at its metric rectangle.

    PILOT_BUILDING=railspur python tools/pilot/compose.py   # facade02 (photos #2/#4) -> qa_facade_compose.png
    PILOT_BUILDING=netloft  python tools/pilot/compose.py   # facade00 (#8/#15/#17)

The photo-stitched texture (facade_texture) carries tree shade, oblique blur, car roofs and neighbour pixels.
Here WE paint the wall: `material` measures one Lab colour + a small periodic grain tile from clean wall texels,
`compose` tiles it over the whole texture (so the base under the cars and the unseen ends are wall) and pastes
every element — a photo crop when one exists, a row-sibling's crop otherwise, a synthesized frame+glass / door
slab when the row has none. `layers` says what each texel is (LAYER) for QA and for relief.

Element dict (from elements.parse_wall): type in LAYER, rect=(x0, y0, x1, y1) wall metres (x from the s0
end, y UP from the wall base), src_kind 'photo'|'sibling'|'synth', crop RGB uint8 (or None), row int (or None),
lights int (or None). Texture px: col = round(x / TEX_M), row = h - round(y / TEX_M) (row 0 = top of wall).
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from . import fill

TEX_M = 0.04
TILE = 64                                   # tile px (2.56 m); periodic by construction
LIT_PCT, GRAIN_MAX = 93, 4.0
SHADE_GAIN = 1.35                           # Kaveh 2026-09-06: composed walls read too dark; the frontal photos are shaded views                # the lit wall (not its tree shade) sets L; grain above 4 L reads as blotches at 25 px/m
SHADE_L    = 130.0                          # Lab L above which the source wall is ALREADY SUNLIT and gets no gain at all: measured,
                                            #   railspur facade03 / netloft facade04 & 28 read L 174 pre-gain and were pushed onto the
                                            #   min(235) clamp = pure white, zero chroma; the genuinely shaded walls read L 82..122.
SHADE_MAX  = 200.0                          # the gain may never carry a wall past a lit-stucco ceiling (was 235 = paper white)
CHROMA_SAME = 22.0                          # Lab ab distance under which two walls are the same material (one building, one colour)
LAYER = dict(wall=0, window=1, door=2, awning=3, storefront=4, sign=5, other=6)
PAINTED = ("storefront", "window", "door")  # draw order; everything else is layer-only


def _rect(e):
    r = e.get("rect")
    return tuple(float(v) for v in (r if r is not None else (e["x0"], e["y0"], e["x1"], e["y1"])))


def _px(rect, w, h):
    """wall-metre rect -> texture (c0, c1, r0, r1) clipped, or None when empty."""
    x0, y0, x1, y1 = rect
    c0, c1 = max(0, round(x0 / TEX_M)), min(w, round(x1 / TEX_M)); r0, r1 = max(0, h - round(y1 / TEX_M)), min(h, h - round(y0 / TEX_M))
    return (c0, c1, r0, r1) if c1 > c0 and r1 > r0 else None


def _lab2rgb(lab):
    lab = np.clip(np.asarray(lab, np.float32), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab.reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(lab.shape)


def _rgb2lab(rgb):
    return cv2.cvtColor(np.ascontiguousarray(rgb, np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(np.shape(rgb)).astype(np.float32)


def material(tex, valid, elements):
    """tex = the PHOTO-stitched texture, valid its seen mask, elements from parse_wall
    -> dict(colour RGB uint8 (3,), colour_lab float (3,), tile (TILE,TILE,3) uint8, grain_std, seams_ok, seam_ratio)."""
    h, w = valid.shape
    em = np.zeros((h, w), np.uint8)
    for e in elements:
        p = _px(_rect(e), w, h)
        if p: em[p[2]:p[3], p[0]:p[1]] = 1
    clean = valid & (cv2.dilate(em, np.ones((7, 7), np.uint8)) == 0)
    if clean.sum() < 200: clean = valid
    colour = fill.wall_colour(tex, clean)                    # already gates to wall-like texels within 35 Lab of the median
    colour_lab = _rgb2lab(colour)
    lab = _rgb2lab(tex.reshape(-1, 3)).reshape(h, w, 3); L = lab[..., 0]
    hp = L - cv2.GaussianBlur(L, (0, 0), 3)
    near = clean & (np.linalg.norm(lab - colour_lab, axis=2) < 25)   # frames and edges do not count as grain
    if near.sum() >= 200:                                            # the photo's wall may be in shade (railspur #2 under the bridge): lit stucco is ~1.35x brighter.
        L0 = float(np.percentile(L[near], LIT_PCT))                  # BUT a wall already at L >= SHADE_L is in the sun and needs no gain — applying it
        gain = SHADE_GAIN if L0 < SHADE_L else 1.0                   # anyway clamped three walls to [235, 128, 128], i.e. white with the grey modelling destroyed
        colour_lab[0] = min(SHADE_MAX, L0 * gain); colour = _lab2rgb(colour_lab)
    g = hp[near]; grain_std = float(np.clip(1.4826 * np.median(np.abs(g - np.median(g))) if g.size >= 50 else 3.0, 1, GRAIN_MAX))   # MAD: shade edges do not inflate it
    # periodic grain: white noise blurred on a 3x3 tiling, centre tile taken -> wraps seamlessly; no photo patch
    # (ponytail: synthetic grain; add a photo patch when a wall shows a real periodic material, e.g. brick)
    n = np.random.default_rng(0).standard_normal((TILE, TILE)).astype(np.float32)
    n = cv2.GaussianBlur(np.tile(n, (3, 3)), (0, 0), 0.7)[TILE:2 * TILE, TILE:2 * TILE]
    n = np.clip(n / max(n.std(), 1e-6), -2.5, 2.5) * grain_std        # every wall texel stays within 2.5 sigma (< 25 Lab) of the colour
    tile_lab = np.stack([colour_lab[0] + n, np.full_like(n, colour_lab[1]), np.full_like(n, colour_lab[2])], -1)
    tile = _lab2rgb(tile_lab)
    L2 = _rgb2lab(np.tile(tile, (2, 2, 1)).reshape(-1, 3)).reshape(2 * TILE, 2 * TILE, 3)[..., 0]
    gy, gx = np.abs(np.diff(L2, axis=0)), np.abs(np.diff(L2, axis=1))      # gy[r] = |L[r+1]-L[r]|: seam between rows TILE-1 and TILE
    seam = (gy[TILE - 1].mean() + gx[:, TILE - 1].mean()) / 2
    interior = (np.delete(gy, TILE - 1, 0).mean() + np.delete(gx, TILE - 1, 1).mean()) / 2
    seam_ratio = float(seam / max(interior, 1e-6))
    return dict(colour=colour, colour_lab=colour_lab, tile=tile, grain_std=grain_std, seams_ok=seam_ratio < 1.2, seam_ratio=seam_ratio)


def same_material(a, b):
    """Two walls' materials are one building's when their chroma agrees (L differs with the light; teal vs pink does not)."""
    return float(np.linalg.norm(a["colour_lab"][1:] - b["colour_lab"][1:])) <= CHROMA_SAME


def synth(kind, wc, hc, material, lights=None):
    """Synthesized element (hc, wc, 3) RGB: window/storefront = light frame + glass with mullions; door = dark slab."""
    L0, a0, b0 = (float(v) for v in material["colour_lab"])
    lab = np.zeros((hc, wc, 3), np.float32)
    frame = (min(L0 + 60, 235), 128, 128)
    if kind == "door":                                              # slab a shade darker than the wall, light frame, a glazed upper panel (the shop doors here are glazed)
        lab[:] = (L0 * 0.6, a0, b0)
        f = max(1, round(0.05 / TEX_M)); lab[:f], lab[:, :f], lab[:, wc - f:] = frame, frame, frame
        g0, g1, gx = max(f + 1, round(0.15 / TEX_M)), max(f + 2, round(hc * 0.5)), max(f + 1, round(0.12 / TEX_M))
        if g1 - g0 > 2 and wc - 2 * gx > 2: lab[g0:g1, gx:wc - gx] = (62, 128, 120)
    else:                                                           # glass: sky-lit blue-grey (not black), light frame, mullions
        grad = np.linspace(10, -10, hc, dtype=np.float32)[:, None]
        lab[..., 0], lab[..., 1], lab[..., 2] = 62 + grad, 128, 118
        f = max(1, round(0.08 / TEX_M))
        lab[:f], lab[hc - f:], lab[:, :f], lab[:, wc - f:] = [frame] * 4
        lights = len(lights) if isinstance(lights, (list, tuple)) else lights          # parse_wall gives the light rects, not a count
        n = int(lights or (2 if hc * TEX_M > 1.6 and wc * TEX_M > 1.2 else 1))
        for k in range(1, n):
            c = round(wc * k / n); lab[f:hc - f, max(0, c - 1):c + 1] = frame
    return _lab2rgb(lab)


def _stretch_ok(m, rect_w):
    """The gate's stretch leg, on a crop we are about to resize to rect_w metres (elements.crop_ok, one predicate)."""
    import facade_elements
    ppm = m.get("crop_ppm") or (m["crop"].shape[1] / max(_rect(m)[2] - _rect(m)[0], 1e-6))     # no recorded ppm: the crop's own texels are its resolution
    # ppm floored and oblique/date passed as pass-values on purpose: parse_wall already judged the SOURCE, this is only the resize
    return elements.crop_ok(max(ppm, elements.GATE_PPM), 1.0, None, rect_w, _rect(m)[2] - _rect(m)[0]) is None


def compose(w, h, material, elements):
    """-> (tex (h,w,3) uint8, layers (h,w) uint8). tex[layers == 0] is the tiled material, exactly."""
    base = np.tile(material["tile"], (math.ceil(h / TILE), math.ceil(w / TILE), 1))[:h, :w]
    tex = base.copy(); layers = np.zeros((h, w), np.uint8)
    order = sorted(elements, key=lambda e: PAINTED.index(e["type"]) if e["type"] in PAINTED else 9)
    for e in order:
        p = _px(_rect(e), w, h)
        if p is None: continue
        c0, c1, r0, r1 = p; kind = e["type"]; lay = LAYER.get(kind, LAYER["other"])
        if kind not in PAINTED:
            layers[r0:r1, c0:c1] = lay; continue                    # awning: its pixels are the underside; the ledge prim gives the depth
        crop = e.get("crop")
        if crop is None and e.get("src_kind") == "sibling" and e.get("row") is not None:   # sibling (pitch-fill slot): nearest row-mate of the same type that has a crop
            xc = (_rect(e)[0] + _rect(e)[2]) / 2
            ew = _rect(e)[2] - _rect(e)[0]
            mates = [m for m in elements if m is not e and m.get("row") == e["row"] and m["type"] == kind and m.get("crop") is not None
                     and m.get("crop_gate") is None                                            # a gated crop must not launder itself through a neighbour
                     and abs((_rect(m)[2] - _rect(m)[0]) - ew) <= 0.3 * ew                     # never stretch a crop > 30 %: a smear, not a window
                     and _stretch_ok(m, ew)]                                                   # nor beyond GATE_STRETCH source pixels
            if mates: crop = min(mates, key=lambda m: abs((_rect(m)[0] + _rect(m)[2]) / 2 - xc))["crop"]
        patch = cv2.resize(np.ascontiguousarray(crop, np.uint8), (c1 - c0, r1 - r0), interpolation=cv2.INTER_AREA) if crop is not None \
            else synth(kind, c1 - c0, r1 - r0, material, e.get("lights"))
        tex[r0:r1, c0:c1] = patch; layers[r0:r1, c0:c1] = lay
    painted = np.isin(layers, [LAYER[k] for k in PAINTED])
    border = painted & (cv2.erode(painted.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)   # 1 px feather inside every pasted rect
    tex[border] = ((tex[border].astype(np.float32) + base[border]) / 2 + 0.5).astype(np.uint8)
    return tex, layers


# ── self-check on real data ─────────────────────────────────────────────────


# The self-check lives with the pilot that owns the data: shadowCity2 tools/pilot/facade_compose.py
