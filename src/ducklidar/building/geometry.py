"""The building's solid: walls grouped into planar facades, plus the roof furniture.

Pure geometry — no photos, no pose. A "facade" is a group of coplanar wall triangles with an
outward normal, an in-plane basis (u along the wall, v up) and metric bounds (s0..s1, t0..t1),
which is what every texturing step downstream is written against.

    V, T = load_solid(cityjson, ("71699431-0",))     # or bring your own solid
    n = normals(V, T)
    F = facades(V, T, n, V.mean(0))

`skylight_boxes` and `floor_prim` are the two pieces of furniture the LoD2.2 solid does not
carry: roof monitors from the raised returns, and a slab under the footprint.
"""
import math
from collections import defaultdict

import cv2
import numpy as np
from scipy import ndimage


def load_solid(cityjson, solids, lod="2.2"):
    """The named objects of a roofer CityJSON as one (V, T), vertices compacted."""
    from .. import scene as sc
    raw = sc.read_cityjson(str(cityjson), lod=lod)
    want = set(map(str, solids))
    V, T = None, []
    for k, (v, t) in raw.items():
        if str(k) in want:
            V = np.asarray(v, float); T.append(np.asarray(t))
    T = np.vstack(T)
    used = np.unique(T); remap = np.full(len(V), -1); remap[used] = np.arange(len(used))
    return V[used], remap[T]


def normals(V, T):
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1), 1e-9)[:, None]


def facades(V, T, n, centre):
    """Group wall triangles into planar facades; outward normal, in-plane basis, bounds."""
    wall = np.abs(n[:, 2]) < 0.3
    by_ang = defaultdict(list)
    for ti in np.flatnonzero(wall):
        nn = n[ti].copy(); c = V[T[ti]].mean(0)
        if (c[:2] - centre[:2]) @ nn[:2] < 0: nn = -nn         # face outward
        ang = round(math.degrees(math.atan2(nn[1], nn[0])) / 10) * 10 % 360
        a = math.radians(ang); by_ang[ang].append((float(c[:2] @ np.array([math.cos(a), math.sin(a)])), ti))
    out = []
    for ang, items in by_ang.items():
        items.sort(); groups, cur = [], [items[0]]
        for d, ti in items[1:]:
            if d - cur[-1][0] > 0.5: groups.append(cur); cur = [(d, ti)]
            else: cur.append((d, ti))
        groups.append(cur)
        a = math.radians(ang); nn = np.array([math.cos(a), math.sin(a), 0.0])
        u = np.array([nn[1], -nn[0], 0.0]); v = np.array([0.0, 0.0, 1.0])
        for g in groups:
            tis = [ti for _d, ti in g]; pts = V[np.unique(T[tis])]; p0 = pts.mean(0)
            s_ = (pts - p0) @ u; t_ = (pts - p0) @ v
            area = float(sum(np.linalg.norm(np.cross(V[T[i][1]] - V[T[i][0]], V[T[i][2]] - V[T[i][0]])) / 2 for i in tis))
            out.append(dict(tris=np.array(tis), n=nn, u=u, v=v, p0=p0, s0=s_.min(), s1=s_.max(), t0=t_.min(), t1=t_.max(), area=area))
    return out


def facade_corners(F):
    return np.array([F["p0"] + F["u"] * s + F["v"] * t for s, t in ((F["s0"], F["t1"]), (F["s1"], F["t1"]), (F["s1"], F["t0"]), (F["s0"], F["t0"]))])


def outer_edges(F, min_h=5.0, min_w=10.0):
    """The outlines of the major facades — eave, base and corners — what a street photo actually sees."""
    E = []
    for f in F:
        if f["t1"] - f["t0"] < min_h or f["s1"] - f["s0"] < min_w: continue
        c = facade_corners(f)
        E += [(c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])]
    return E


# ── skylights ───────────────────────────────────────────────────────────────
def skylight_boxes(pts, rgb, footprint_ring, cell=0.25, lift=0.6, min_area=1.5, max_area=60.0, max_h=3.5, min_fill=0.5):
    """Roof monitors as boxes: cells standing `lift` above the local roof base, grouped, boxed.

    The roof base under a monitor is the grey opening of the per-cell minimum z with a
    5 m window (a monitor is narrower than that, a roof step is not); anything the base
    does not explain is raised. Components wider than `max_area` are roof steps that
    roofer already models and are skipped; so are components filling < `min_fill` of their
    own rectangle — an L-shaped sliver along a roof step boxed as 17 x 11 m (railspur).
    """
    import shapely
    from shapely.geometry import Polygon
    x0, y0 = pts[:, 0].min(), pts[:, 1].min(); w = int((pts[:, 0].max() - x0) / cell) + 1; h = int((pts[:, 1].max() - y0) / cell) + 1
    cx = ((pts[:, 0] - x0) / cell).astype(int); cy = ((pts[:, 1] - y0) / cell).astype(int); idx = cy * w + cx
    zmin = np.full(w * h, np.inf); np.minimum.at(zmin, idx, pts[:, 2]); zmax = np.full(w * h, -np.inf); np.maximum.at(zmax, idx, pts[:, 2])
    n = np.bincount(idx, minlength=w * h); has = n > 0
    zmin = np.where(has, zmin, np.nan).reshape(h, w); zmax = np.where(has, zmax, np.nan).reshape(h, w)
    filled = np.where(np.isnan(zmin), np.nanmax(zmin), zmin)
    k = int(5.0 / cell) | 1
    base = ndimage.grey_opening(filled, size=(k, k))
    raised = has.reshape(h, w) & ((zmax - base) > lift)
    raised = ndimage.binary_opening(raised, np.ones((2, 2))); raised = ndimage.binary_closing(raised, np.ones((3, 3)))
    lab, nlab = ndimage.label(raised)
    inside = Polygon(footprint_ring).buffer(-1.0)
    boxes = []
    for L in range(1, nlab + 1):
        cells = np.argwhere(lab == L); area = len(cells) * cell * cell
        if not (min_area <= area <= max_area): continue
        rect = cv2.minAreaRect(cells[:, ::-1].astype(np.float32))          # (cx, cy), (w, h), angle in cell coords
        if len(cells) < min_fill * max(rect[1][0] * rect[1][1], 1.0): continue    # not a compact box: a roof-step sliver
        corners = cv2.boxPoints(rect) * cell + [x0, y0]
        if not inside.contains(shapely.geometry.Point(corners.mean(0))): continue
        m = np.isin(idx, np.flatnonzero((lab == L).ravel()))
        if m.sum() < 8: continue
        zb = float(np.nanmedian(base[lab == L])); zt = float(np.percentile(pts[m, 2], 90))
        if not (lift <= zt - zb <= max_h): continue
        col = np.array([222, 226, 230], np.uint8)                      # glass monitors, one white
        boxes.append(dict(corners=corners, z0=zb, z1=zt, col=col, area=round(area, 1)))
    return boxes


def box_prim(b, name):
    c = b["corners"]; z0, z1 = b["z0"], b["z1"]
    P = np.array([[*p, z0] for p in c] + [[*p, z1] for p in c])
    T = [[4, 5, 6], [4, 6, 7]]                                     # top
    for i in range(4):
        j = (i + 1) % 4; T += [[i, j, j + 4], [i, j + 4, i + 4]]   # sides
    tex = np.tile(b["col"], (4, 4, 1))
    return dict(pos=P, uv=np.zeros((8, 2)), idx=np.array(T), tex=tex, name=name)


# ── floor ───────────────────────────────────────────────────────────────────
def floor_prim(ring, z, col=(120, 118, 112)):
    """A floor slab under the building: the footprint, triangulated, at the solid's base."""
    import shapely
    from shapely.geometry import Polygon
    poly = Polygon(ring)
    try:
        import mapbox_earcut
        pts = np.asarray(poly.exterior.coords)[:-1]; tri = mapbox_earcut.triangulate_float64(pts, np.array([len(pts)], np.uint32)).reshape(-1, 3)
    except Exception:
        pts = np.asarray(poly.exterior.coords)[:-1]
        tris = [t for t in shapely.delaunay_triangles(poly).geoms if poly.contains(t.centroid)]
        idx = {tuple(np.round(c, 4)): i for i, c in enumerate(pts)}
        tri = np.array([[idx[tuple(np.round(c, 4))] for c in np.asarray(t.exterior.coords)[:3]] for t in tris])
    P = np.column_stack([pts, np.full(len(pts), z)])
    return dict(pos=P, uv=np.zeros((len(P), 2)), idx=tri, tex=np.tile(np.array(col, np.uint8), (4, 4, 1)), name="floor")
