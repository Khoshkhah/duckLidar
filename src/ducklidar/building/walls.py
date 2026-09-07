"""One wall's texture, and the whole building's wall material.

`composite_facade` runs a wall through `ducklidar.facade` (quad → rectify → stitch → fill),
parses its elements and measures its material; `compose_walls` then decides the BUILDING's
material from the best-fitted wall and recomposes every wall that agrees with it, dropping the
ones that were fitted onto a neighbour. Walls no photo sees get `flat_facade` — the returns'
own colour, nothing invented.

`roof_texture` is the roof's counterpart: the survey's own RGB rasterised top-down, flattened
to one base colour with a trace of shading.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

from ..facade import CTX, compose as facade_compose, elements as facade_elements, texture as facade_texture
from .camera import chosen_pose, occlusion_map, project, rank_photos, usable
from .geometry import facade_corners

TEX_M = 0.04                          # facade texture resolution, m/px
ROOF_M = 0.2                          # roof texture resolution, m/px
MAX_VIEWS = 4                         # photos composited per facade, best first


def tex_size(F, tex_m=TEX_M):
    """The wall's texture in pixels at `tex_m` m/px, clamped to 8..2048."""
    w = int(math.ceil((F["s1"] - F["s0"]) / tex_m)); h = int(math.ceil((F["t1"] - F["t0"]) / tex_m))
    return max(8, min(w, 2048)), max(8, min(h, 2048))


def warp_facade(F, P, i, man, tex_m=TEX_M):
    w, h = tex_size(F, tex_m)
    uv, _ = project(facade_corners(F), P["cam"], P["heading"], P["pitch"], P["fov"])
    src = uv.astype(np.float32); dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32); M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.imread(man[i]["file"]); valid = usable(i, img.shape[:2])
    tex = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    vt = cv2.warpPerspective(valid, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
    if CTX.SOLID: vt &= occlusion_map(F, P["cam"], w, h)[0]          # texels the building itself hides from this camera
    return cv2.cvtColor(tex, cv2.COLOR_BGR2RGB), vt


def warp_clicked(F, i, man, pts4, tex_m=TEX_M):
    """Homography from four clicked image points (TL, TR, BR, BL of the facade) — no camera pose involved."""
    w, h = tex_size(F, tex_m)
    src = np.array(pts4, np.float32); dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32); M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.imread(man[i]["file"]); valid = usable(i, img.shape[:2])
    tex = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    vt = cv2.warpPerspective(valid, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
    return cv2.cvtColor(tex, cv2.COLOR_BGR2RGB), vt


def lab_median(t, v):
    if v.sum() < 50: return None
    lab = cv2.cvtColor(t, cv2.COLOR_RGB2LAB).astype(float)
    return np.median(lab[v], axis=0)


def facade_cands(F, poses, man, name=None, max_views=MAX_VIEWS):
    """Photos for facade.texture: clicked pairs first (exact quads, score 1e9), then the ranked photos with a
    PRIOR quad from the chosen pose — per facade falling back to the raw fit / the metadata pose when the
    chosen one cannot project this wall (the prior is only a search band; pixels are placed from image evidence)."""
    CLICKS = Path(CTX.CLICKS) if CTX.CLICKS else None
    clicks = json.load(open(CLICKS)) if CLICKS and CLICKS.exists() else {}
    cands = [(int(i), np.float32(p4), 1e9) for i, p4 in clicks.get(name, {}).items()] if name else []
    for sc, i in rank_photos(F, poses, man)[:max_views + 2]:
        for P in (chosen_pose(poses[i]), poses[i], poses[i].get("meta") or poses[i]):
            uv, dep = project(facade_corners(F), np.asarray(P["cam"]), P["heading"], P["pitch"], P["fov"])
            if (dep > 1.0).all(): cands.append((i, np.float32(uv), float(sc))); break
    return cands


def composite_facade(F, poses, man, pts, rgb, name=None, compose=True, tex_m=TEX_M):
    """Clicked pairs first (exact), then the best photos. Pixel placement is facade.texture's: the wall's
    quad from image evidence, metric strips, translation-only stitch, wall-like hole fill — the pose only
    ranks photos and gives each a prior quad. Nothing usable seen → (None, None) → the wall_base colour."""
    cands = facade_cands(F, poses, man, name)
    if not cands: return flat_facade(F, pts, rgb, tex_m), None
    w, h = tex_size(F, tex_m)
    tex, valid, info = facade_texture.texture_wall(F, cands, w, h, man, Path(CTX.MASKS))
    if tex is None: return None, None
    info.update(photo_tex=tex, valid=valid)                      # the stitched texture + its seen mask: the caller measures the material and composes
    if compose:
        info["elements"] = facade_elements.parse_wall(F, info.get("strips", []), Path(CTX.PHOTOS) / "elements", man)
        info["material"] = facade_compose.material(tex, valid, info["elements"])
    return tex, info


def compose_walls(done, log=print):
    """The building's material is the best-fitted wall's (it also paints the unseen walls). Every seen wall whose own
    chroma agrees is composed with ITS OWN material — the same paint reads L 77 under the bridge and L 200 in the sun
    (railspur facade02 vs 03), a shared L paints a cream wall olive; a wall of another chroma was fitted onto a
    NEIGHBOUR (SAM 3's building mask spans both) and goes flat. Material only when parse found nothing (never the
    smeared stitch). -> the building material dict, or None."""
    scored = [(info["filled"] * (1.0 if info["quads"][info["photos"][0]]["src"]["top"] == "lsd" else 0.5), k) for k, (t, info) in done.items() if info]
    if not scored: return None
    best_k = max(scored)[1]; mat = done[best_k][1]["material"]
    log(f"  building material from facade{best_k:02d}: Lab {mat['colour_lab'].round(0).tolist()} grain {mat['grain_std']:.1f}")
    for k, (t, info) in list(done.items()):
        if not info: continue
        own = info["material"]
        if not facade_compose.same_material(own, mat):
            log(f"  facade{k:02d}: chroma {np.linalg.norm(own['colour_lab'][1:] - mat['colour_lab'][1:]):.0f} from the building's material -> another building was painted, wall goes flat")
            done[k] = (None, None); continue
        h, w = info["valid"].shape
        tex, layers = facade_compose.compose(w, h, own, info["elements"])
        info.update(layers=layers, material=dict(colour_lab=own["colour_lab"].round(1).tolist(), grain_std=round(float(own["grain_std"]), 2), seams_ok=bool(own["seams_ok"])))
        done[k] = (tex, info)
    return mat


def fill_rows(tex, filled):
    """Holes take the median colour of their own texture row — siding repeats horizontally,
    so a row's median is a plausible wall; big inpainted blobs are not. Small holes are then
    inpainted on top so edges stay soft."""
    tex = tex.copy(); h, w = filled.shape
    overall = np.median(tex[filled], axis=0) if filled.any() else np.array([150, 150, 150])
    for r in range(h):
        row_ok = filled[r]
        col = np.median(tex[r, row_ok], axis=0) if row_ok.sum() >= max(4, 0.08 * w) else overall
        tex[r, ~row_ok] = col
    small = cv2.erode(filled.astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
    small &= ~filled
    if small.any() and small.mean() < 0.3:
        tex = cv2.inpaint(tex, small.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    return tex


def flat_facade(F, pts, rgb, tex_m=TEX_M):
    w, h = tex_size(F, tex_m)
    d = np.abs((pts - F["p0"]) @ F["n"]); near = d < 1.0
    col = np.median(rgb[near], axis=0) if near.sum() > 50 else np.array([150, 150, 150])
    return np.tile(col.astype(np.uint8), (h, w, 1))


def roof_texture(pts, rgb, bbox, ring, raised_cells=None, roof_m=ROOF_M):
    """The roof as a clean surface: one base colour from the flat roof returns, the survey's
    shading kept at low amplitude, skylight patches and everything outside the footprint
    painted in the base colour (no nearest-neighbour streaks, no dark holes under the boxes)."""
    import shapely
    from shapely.geometry import Polygon
    ROOF_M = roof_m
    x0, y0, x1, y1 = bbox
    w = int(math.ceil((x1 - x0) / ROOF_M)); h = int(math.ceil((y1 - y0) / ROOF_M))
    cx = np.clip(((pts[:, 0] - x0) / ROOF_M).astype(int), 0, w - 1); cy = np.clip(((y1 - pts[:, 1]) / ROOF_M).astype(int), 0, h - 1)
    idx = cy * w + cx; n = np.bincount(idx, minlength=w * h)
    tex = np.stack([np.bincount(idx, weights=rgb[:, k], minlength=w * h) for k in range(3)], -1)
    has = n > 0; tex[has] /= n[has, None]; tex = tex.reshape(h, w, 3)
    # cells the skylight boxes stand on (their patches in the LiDAR colour are the boxes' shadows/glass)
    sky = np.zeros((h, w), bool)
    if raised_cells is not None:
        for c in raised_cells:
            poly = np.round((c - [x0, y1]) / [ROOF_M, -ROOF_M]).astype(np.int32)
            cv2.fillPoly(sky.view(np.uint8), [poly], 1)
        sky = cv2.dilate(sky.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    gx, gy = np.meshgrid(x0 + (np.arange(w) + 0.5) * ROOF_M, y1 - (np.arange(h) + 0.5) * ROOF_M)
    inside = shapely.contains_xy(Polygon(ring).buffer(0.5), gx.ravel(), gy.ravel()).reshape(h, w)
    flat = has.reshape(h, w) & inside & ~sky
    base = np.median(tex[flat], axis=0) if flat.sum() > 50 else np.array([160, 160, 160])
    detail = np.where(flat[..., None], tex, base)
    detail = cv2.medianBlur(detail.astype(np.uint8), 5).astype(float)
    out = base + 0.12 * (detail - base)                             # essentially one colour; a trace of the survey's shading
    out[~flat] = base
    out = cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (0, 0), 1.0)
    return out, base
