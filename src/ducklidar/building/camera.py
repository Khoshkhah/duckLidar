"""Where each street photo was really taken, and which walls it can see.

The photo's metadata pose is 3-5 m and a few degrees off. `refine_pose` fits it by matching the
solid's projected silhouette to SAM 3's building mask — a grid search on (dx, dy, dz, yaw,
pitch), never on image edges (those chase trees and cars). The fitted pose is used to RANK
photos and to give each one a prior quad; the pixels themselves are placed from image evidence
by `ducklidar.facade`.

Host context (the photo folder, the masks, the manifest, the solid) comes from
`ducklidar.facade.CTX`, the same object the facade modules read.
"""
import math
from pathlib import Path

import cv2
import numpy as np

from ..facade import CTX
from .geometry import facade_corners


def cam_axes(heading, pitch):
    h, p = math.radians(heading), math.radians(pitch)
    fwd = np.array([math.sin(h) * math.cos(p), math.cos(h) * math.cos(p), math.sin(p)])
    right = np.array([math.cos(h), -math.sin(h), 0.0]); up = np.cross(right, fwd)
    return fwd, right, up


def project(P, cam, heading, pitch, fov=90.0, W=None):
    W = CTX.W if W is None else W
    f = (W / 2) / math.tan(math.radians(fov / 2)); fwd, right, up = cam_axes(heading, pitch)
    d = P - cam; depth = d @ fwd
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.column_stack([W / 2 + f * (d @ right) / depth, W / 2 - f * (d @ up) / depth]), depth


def silhouette(V, T, cam, heading, pitch, fov, size):
    """The solid's projected silhouette as a filled mask at `size` px (image is W px)."""
    uv, dep = project(V, cam, heading, pitch, fov)
    sc = size / CTX.W; img = np.zeros((size, size), np.uint8)
    ok = dep > 0.5
    tri_ok = ok[T].all(1)
    if not tri_ok.any(): return img
    polys = (uv[T[tri_ok]] * sc).astype(np.int32)
    cv2.fillPoly(img, list(polys), 1)
    return img


def refine_pose(photo, V, T, size=320):
    """Fit the camera so the solid's silhouette matches SAM 3's building mask.

    Image edges chased trees and cars (the first attempt hit its search bounds on
    most photos). The building mask has no such distraction: the score is the share
    of usable pixels, near the initial projection, where (projected solid) agrees
    with (building mask). Coarse grid on (dx, dy, dz, yaw, pitch), then a fine one.
    """
    MASKS = Path(CTX.MASKS)
    i = photo["_i"]
    bld = cv2.imread(str(MASKS / f"{i:02d}_building.png"), cv2.IMREAD_GRAYSCALE)
    if bld is None: return dict(cam=np.array([photo["x"], photo["y"], photo.get("cam_z", CTX.Z_GROUND + CTX.CAM_H)]), heading=photo["heading"], pitch=photo.get("pitch", 8), fov=photo.get("fov", 90), score0=0, score=0, d=(0, 0, 0, 0, 0))
    occ = cv2.imread(str(MASKS / f"{i:02d}.png"), cv2.IMREAD_GRAYSCALE) > 127
    sc = size / CTX.W
    B = cv2.resize((bld > 127).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
    valid = cv2.resize((~occ).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
    base = np.array([photo["x"], photo["y"], photo.get("cam_z", CTX.Z_GROUND + CTX.CAM_H)]); h0, p0, fov = photo["heading"], photo.get("pitch", 8), photo.get("fov", 90)
    sfm = photo.get("source") == "mapillary"                       # SfM-refined pose: search a small window only
    S0 = silhouette(V, T, base, h0, p0, fov, size).astype(bool)
    roi = cv2.dilate(S0.astype(np.uint8), np.ones((int(60 * sc) * 2 + 1,) * 2, np.uint8)).astype(bool) & valid
    if roi.sum() < 200: return dict(cam=base, heading=h0, pitch=p0, fov=fov, score0=0, score=0, d=(0, 0, 0, 0, 0))

    def score(dx, dy, dz, dyaw, dpitch):
        S = silhouette(V, T, base + [dx, dy, dz], h0 + dyaw, p0 + dpitch, fov, size).astype(bool)
        return float((S[roi] == B[roi]).mean())

    s0 = score(0, 0, 0, 0, 0); best = (s0, 0.0, 0.0, 0.0, 0.0, 0.0)
    R = 1.5 if sfm else 3.0; YAWS = (-2, -1, 0, 1, 2) if sfm else (-4, -2, 0, 2, 4)
    for dx in np.arange(-R, R + 0.1, 1.0 if not sfm else 0.5):
        for dy in np.arange(-R, R + 0.1, 1.0 if not sfm else 0.5):
            for dz in (-1.0, -0.5, 0.0, 0.5):
                for dyaw in YAWS:
                    for dpitch in (-2, 0, 2):
                        v = score(dx, dy, dz, dyaw, dpitch)
                        if v > best[0]: best = (v, dx, dy, dz, dyaw, dpitch)
    _, dx, dy, dz, dyaw, dpitch = best
    for fx in np.arange(-0.75, 0.76, 0.25):
        for fy in np.arange(-0.75, 0.76, 0.25):
            for fz in (-0.25, 0.0, 0.25):
                for fyaw in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5):
                    v = score(dx + fx, dy + fy, dz + fz, dyaw + fyaw, dpitch)
                    if v > best[0]: best = (v, dx + fx, dy + fy, dz + fz, dyaw + fyaw, dpitch)
    s1, dx, dy, dz, dyaw, dpitch = best
    return dict(cam=base + [dx, dy, dz], heading=h0 + dyaw, pitch=p0 + dpitch, fov=fov,
                score0=round(s0, 3), score=round(s1, 3), d=(round(dx, 2), round(dy, 2), round(dz, 2), round(dyaw, 1), dpitch),
                meta=dict(cam=base, heading=h0, pitch=p0, fov=fov, score=round(s0, 3)))


_lum = {}
def luminance(file):
    if file not in _lum: _lum[file] = float(cv2.imread(file, cv2.IMREAD_GRAYSCALE).mean())
    return _lum[file]


_sharp = {}
def sharpness(file):
    """Variance of the Laplacian — blur and haze score low."""
    if file not in _sharp: _sharp[file] = float(cv2.Laplacian(cv2.imread(file, cv2.IMREAD_GRAYSCALE), cv2.CV_64F).var())
    return _sharp[file]


def visible_fraction(F, cam, inset=0.3):
    """Share of five sample points on the facade (4 inset corners + centre) the camera reaches
    without passing through the solid itself — Möller–Trumbore over every triangle."""
    V, T = CTX.SOLID["V"], CTX.SOLID["T"]
    pts = [F["p0"] + F["u"] * s + F["v"] * t for s, t in ((F["s0"] + inset, F["t1"] - inset), (F["s1"] - inset, F["t1"] - inset),
                                                          (F["s1"] - inset, F["t0"] + inset), (F["s0"] + inset, F["t0"] + inset),
                                                          ((F["s0"] + F["s1"]) / 2, (F["t0"] + F["t1"]) / 2))]
    v0, v1, v2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]; e1, e2 = v1 - v0, v2 - v0
    seen = 0
    for q in pts:
        d = q - cam; h = np.cross(d, e2); a = (e1 * h).sum(1)
        ok = np.abs(a) > 1e-9; f = np.where(ok, 1.0 / np.where(ok, a, 1), 0)
        sv = cam - v0; u = f * (sv * h).sum(1); qv = np.cross(sv, e1); w = f * (d * qv).sum(1); t = f * (e2 * qv).sum(1)
        hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > 0.02) & (t < 0.985)
        seen += 0 if hit.any() else 1
    return seen / len(pts)


def occlusion_map(F, cam, w, h, step=0.5):
    """Which texels of the facade the camera actually reaches: a ray grid against the solid (minus the facade's own
    triangles — the real wall is not perfectly planar, and its own face 0.4 m in front of the flat rectangle is not an occluder)."""
    V, T = CTX.SOLID["V"], CTX.SOLID["T"]; T = np.delete(T, F["tris"], axis=0); v0, v1, v2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]; e1, e2 = v1 - v0, v2 - v0
    nx = max(2, min(48, int((F["s1"] - F["s0"]) / step) + 1)); ny = max(2, min(24, int((F["t1"] - F["t0"]) / step) + 1))
    ss = np.linspace(F["s0"] + 0.15, F["s1"] - 0.15, nx); tt = np.linspace(F["t1"] - 0.15, F["t0"] + 0.15, ny)     # row 0 = top
    vis = np.zeros((ny, nx), bool)
    for r, t_ in enumerate(tt):
        for c, s_ in enumerate(ss):
            q = F["p0"] + F["u"] * s_ + F["v"] * t_; d = q - cam; hh = np.cross(d, e2); a = (e1 * hh).sum(1)
            ok = np.abs(a) > 1e-9; f = np.where(ok, 1.0 / np.where(ok, a, 1), 0); sv = cam - v0
            u = f * (sv * hh).sum(1); qv = np.cross(sv, e1); ww = f * (d * qv).sum(1); t = f * (e2 * qv).sum(1)
            hit = ok & (u >= 0) & (ww >= 0) & (u + ww <= 1) & (t > 0.02) & (t < 0.985)
            vis[r, c] = not hit.any()
    return cv2.resize(vis.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool), float(vis.mean())


def chosen_pose(P):
    return P if P.get("score", 1.0) >= 0.45 else P.get("meta", P)   # fit not trusted → the metadata pose, unrefined


def rank_photos(F, poses, man):
    """Every photo that can see the facade, best first: frontal, complete, close, recent, daylight."""
    W = CTX.W
    c = F["p0"] + F["u"] * (F["s0"] + F["s1"]) / 2 + F["v"] * (F["t0"] + F["t1"]) / 2
    out = []
    for i, P in poses.items():
        w = P["cam"] - c; dist = np.linalg.norm(w)
        if w @ F["n"] <= 0: continue                               # camera behind the facade
        cosang = float(w @ F["n"]) / dist
        if cosang < math.cos(math.radians(60)): continue             # grazing views smear
        uv, dep = project(facade_corners(F), P["cam"], P["heading"], P["pitch"], P["fov"])
        if (dep <= 1.0).any(): continue
        inside = ((uv[:, 0] >= -40) & (uv[:, 0] < W + 40) & (uv[:, 1] >= -40) & (uv[:, 1] < W + 40)).mean()
        px_per_m = W / (2 * dist * math.tan(math.radians(P["fov"] / 2)))
        if man[i].get("unused"): continue                             # flagged in the manifest (e.g. bridge-deck Mapillary)
        if luminance(man[i]["file"]) < 80: continue                  # night / dusk (netloft #27: 72, lit windows on a dark wall)
        # distance to the NEAREST point of the wall, not its centre — a 43 m wall seen from one end is close
        rel = P["cam"] - F["p0"]; s_ = float(np.clip(rel @ F["u"], F["s0"], F["s1"])); t_ = float(np.clip(rel @ F["v"], F["t0"], F["t1"]))
        near = F["p0"] + F["u"] * s_ + F["v"] * t_
        if np.linalg.norm(P["cam"] - near) > 36.0: continue          # very far views: other buildings get in between
        if CTX.SOLID and occlusion_map(F, P["cam"], 8, 4)[1] < 0.15: continue   # the solid itself hides (almost) all of it
        P = chosen_pose(P)                                            # untrusted fit → metadata pose, still usable
        year = str(man[i]["date"])[:4]; recency = {"2025": 1.3, "2024": 1.3, "2022": 1.1}.get(year, 0.8)
        s = cosang ** 2 * (0.3 + inside) * min(px_per_m / 15, 1.0) * recency * (0.5 + 0.5 * min(sharpness(man[i]["file"]) / 200.0, 1.0)) * max(P.get("score", 1.0), 0.3)
        out.append((s, i))
    return sorted(out, reverse=True)


def pick_photo(F, poses, man=None):
    r = rank_photos(F, poses, man); return r[0] if r else None


def usable(i, shape):
    """1 where the photo may be painted on a wall: not an occluder (SAM 3 mask), inside the frame."""
    MANIFEST = CTX.MANIFEST
    m = Path(CTX.MASKS) / f"{i:02d}.png"
    v = np.ones(shape, np.float32)
    if m.exists(): v[cv2.imread(str(m), cv2.IMREAD_GRAYSCALE) > 127] = 0.0
    vf = MANIFEST[i].get("valid") if i < len(MANIFEST) else None       # outside the original frame (Mapillary virtual views)
    if vf and Path(vf).exists(): v[cv2.imread(vf, cv2.IMREAD_GRAYSCALE) < 127] = 0.0
    return v
