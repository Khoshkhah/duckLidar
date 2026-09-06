"""Object ids per point -- the baseline dispatch from docs/objects.md.

Per-class, exactly as approved: buildings (with glazing) join to OSM footprints --
buffered 1 m, nearest polygon where buffers overlap -- and unmapped structures fall back
to mesh-graph connected components; veg, small and on_bridge are the five-line baseline
(mask the mesh graph to the class, the 2.5 m kNN cap is the cut, take components);
floating is the baseline plus the marker-watershed raft split (any component wider than
BOAT); ground, water and bridge get one instance each. Components under MIN_PTS get id
-1 and are counted in the report, not hidden.

Needs out/labels.npz (layers.py) and out/mesh_graph.npz (mesh_graph.py); reads the same
study window, so point order matches layers.py by construction (asserted).

    python tools/objects.py [out.npz]
"""
import json
import sys
import time

import numpy as np

import ducklidar as dl
from ._env import OUTDIR, check, out
from .pipeline import B

DECIDE = ["ground", "veg_low", "veg_high", "building", "water", "small", "floating",
          "bridge", "on_bridge", "glazing"]
CC_CLASSES = ["veg_low", "veg_high", "small", "on_bridge"]
ONE_CLASSES = ["ground", "water", "bridge"]
MIN_PTS = 20          # an instance below this is noise: id -1, counted
FOOT_BUFFER = 1.0     # m -- eaves overhang the polygon; OSM itself is offset ~0.5-2 m
BOAT = 25.0           # size prior: no vessel here is longer -- wider floating = a raft
RAFT_PIX = 0.5
PROMINENCE = 1.2      # m a peak must rise above its saddle to seed its own vessel


def raft_split(px, py, pz):
    """Kaveh's choice B: marker watershed on plan-view height, for one welded raft.

    Boats and docks all touch within ~1 m of the pontoon level -- the mesh graph cannot
    separate them there -- but each vessel stands alone above it. So: 0.5 m raster of max
    height above pontoon (p1 of z), boat-scale peaks (>1 m, 3.5 m apart) become markers,
    each large low ribbon (>=100 m2 under 1 m -- a dock) its own marker, watershed on the
    inverted surface. [measured] on the five big rafts: 425 objects, median extent 5-7 m
    (marina slips), docks kept long; ALPINE box-split gave 15 m tiles that ignore height.
    Known cost, accepted: a large vessel with two peaks (bow + wheelhouse) may split.
    """
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed

    zw = np.percentile(pz, 1)
    gx = ((px - px.min()) / RAFT_PIX).astype(int)
    gy = ((py - py.min()) / RAFT_PIX).astype(int)
    nxc = gx.max() + 1
    cell = gy * nxc + gx
    top = np.full((gy.max() + 1) * nxc, -np.inf)
    np.maximum.at(top, cell, pz - zw)
    top = top.reshape(-1, nxc)
    occ = np.isfinite(top)
    ts = ndi.gaussian_filter(np.where(occ, top, 0.0), 1.0)
    # markers by PROMINENCE, not plain local maxima: a long hull's ridge carries several
    # 3.5 m-apart maxima, and each seeded its own object -- Kaveh's "one real object in
    # many parts". h-maxima keeps a peak only if it rises >= PROMINENCE above the saddle
    # to a higher peak: bumps on one ridge merge, separate boats (saddle near pontoon)
    # stay apart. [measured] at the reported mooring row: 29 fragments -> 9 objects, the
    # two long hulls whole at h=1.2 (still split at 0.8); isolated boats unaffected.
    from skimage.morphology import h_maxima
    peaks = h_maxima(np.where(occ, ts, 0.0), PROMINENCE) & occ & (ts > 1.0)
    markers, nm = ndi.label(peaks)
    low = occ & (top <= 1.0)
    llab, _ = ndi.label(low)
    area = np.bincount(llab.ravel()) * RAFT_PIX ** 2
    dock = np.isin(llab, np.flatnonzero(area >= 100)) & (llab > 0)
    dlab, _ = ndi.label(dock)
    markers[dock] = nm + dlab[dock]
    ws = watershed(-ts, markers, mask=occ)
    return ws.ravel()[cell]


def main(argv):
    t0 = time.time()
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    label = np.load(OUTDIR / "labels.npz")["label"]
    mesh = np.load(OUTDIR / "mesh_graph.npz")
    pts = dl.read(check(), B)
    n = len(pts["x"])
    assert n == len(label) == int(mesh["n"]), "labels/mesh/window out of step -- rerun layers.py"
    x, y = np.asarray(pts["x"]), np.asarray(pts["y"])
    xyz = np.column_stack([x, y, np.asarray(pts["z"])])
    idx = {k: i for i, k in enumerate(DECIDE)}

    # the mesh graph, rebuilt exactly as layers.py does (kNN k=10 capped 2.5 m, deduped,
    # plus the stored MST bridges) -- docs/graph.md family 9
    dm, im = cKDTree(xyz).query(xyz, k=11, workers=-1)
    ok = (dm[:, 1:] <= 2.5).ravel()
    a = np.repeat(np.arange(n, dtype=np.int64), 10)[ok]
    b = im[:, 1:].astype(np.int64).ravel()[ok]
    del dm, im, ok
    key = np.unique(np.minimum(a, b) * n + np.maximum(a, b))
    ea, eb = key // n, key % n
    del a, b, key
    ea = np.concatenate([ea, mesh["bridge_src"].astype(np.int64)])
    eb = np.concatenate([eb, mesh["bridge_dst"].astype(np.int64)])
    print(f"{n:,} points, {len(ea):,} mesh edges  ({time.time()-t0:.0f}s)")

    def components(mask):
        """Connected components of the mesh graph restricted to mask -- the baseline."""
        em = mask[ea] & mask[eb]
        g = coo_matrix((np.ones(int(em.sum()), np.int8), (ea[em], eb[em])), shape=(n, n))
        _, comp = connected_components(g, directed=False)
        return comp

    oid = np.full(n, -1, np.int32)
    inst_class, inst_size, inst_osm = [], [], []

    def add_instances(cls, mask, comp, osm_ids=None):
        """Number mask's components; small ones stay -1. Returns (kept, noise_pts)."""
        cid = comp[mask]
        uniq, inv, cnt = np.unique(cid, return_inverse=True, return_counts=True)
        keep = cnt >= MIN_PTS
        new = np.full(len(uniq), -1, np.int32)
        new[keep] = len(inst_class) + np.arange(int(keep.sum()), dtype=np.int32)
        oid[mask] = new[inv]
        for u, c in zip(uniq[keep], cnt[keep]):
            inst_class.append(idx[cls])
            inst_size.append(int(c))
            inst_osm.append(-1 if osm_ids is None else osm_ids.get(int(u), -1))
        return int(keep.sum()), int(cnt[~keep].sum())

    # ---- buildings: footprint join first, CC fallback for the unmapped ----------------------
    from shapely import STRtree, distance, points as sh_points
    from shapely.geometry import Polygon

    osm = json.loads((OUTDIR / "osm_buildings.json").read_text())
    polys = [Polygon(p["coords"]) for p in osm["polys"]]
    osm_way = [p["id"] for p in osm["polys"]]
    tree = STRtree([p.buffer(FOOT_BUFFER) for p in polys])
    bmask = (label == idx["building"]) | (label == idx["glazing"])
    bidx = np.flatnonzero(bmask)
    pgeom = sh_points(np.column_stack([x[bidx], y[bidx]]))
    qi, qp = tree.query(pgeom, predicate="intersects")
    # a point inside two buffers goes to the nearer UNBUFFERED polygon
    poly_of = np.full(len(bidx), -1, np.int64)
    order = np.argsort(qi, kind="stable")
    qi, qp = qi[order], qp[order]
    first = np.r_[True, qi[1:] != qi[:-1]]
    poly_of[qi[first]] = qp[first]
    dup = qi[~first]
    if len(dup):
        for i, p in zip(dup, qp[~first]):
            if distance(pgeom[i], polys[p]) < distance(pgeom[i], polys[poly_of[i]]):
                poly_of[i] = p
    n_dup = len(np.unique(dup))
    mapped = poly_of >= 0
    # mapped points: the polygon IS the instance (identity inherited from the map,
    # the 3DBAG pattern) -- reuse add_instances by handing it poly ids as "components"
    mmask = np.zeros(n, bool)
    mmask[bidx[mapped]] = True
    comp = np.full(n, -1, np.int64)
    comp[bidx[mapped]] = poly_of[mapped]
    k_map, noise_map = add_instances("building", mmask,
                                    comp, {i: osm_way[i] for i in range(len(polys))})
    # unmapped building points: the five-line baseline
    umask = np.zeros(n, bool)
    umask[bidx[~mapped]] = True
    k_cc, noise_cc = add_instances("building", umask, components(umask))
    print(f"  building: {k_map} footprint instances ({int(mapped.sum()):,} pts, "
          f"{n_dup:,} in overlapping buffers), {k_cc} unmapped CC instances "
          f"({int((~mapped).sum()):,} pts), noise {noise_map + noise_cc:,} pts")

    # ---- floating: components, then the raft split for anything wider than a boat ----------
    fmask = label == idx["floating"]
    comp = components(fmask).astype(np.int64)
    fi = np.flatnonzero(fmask)
    nxt = int(comp.max()) + 1
    n_split = 0
    for u in np.unique(comp[fi]):
        ii = fi[comp[fi] == u]
        if max(np.ptp(x[ii]), np.ptp(y[ii])) <= BOAT:
            continue
        sub = raft_split(x[ii], y[ii], np.asarray(pts["z"])[ii])
        comp[ii] = nxt + sub
        nxt += int(sub.max()) + 1
        n_split += 1
    k, noise = add_instances("floating", fmask, comp)
    print(f"  floating: {k} instances ({n_split} rafts split), "
          f"noise {noise:,} of {int(fmask.sum()):,} pts")

    # ---- everything else: per-class components, or one instance ----------------------------
    for cls in CC_CLASSES:
        mask = label == idx[cls]
        k, noise = add_instances(cls, mask, components(mask))
        print(f"  {cls}: {k} instances, noise {noise:,} of {int(mask.sum()):,} pts")
    for cls in ONE_CLASSES:
        mask = label == idx[cls]
        if mask.any():
            oid[mask] = len(inst_class)
            inst_class.append(idx[cls])
            inst_size.append(int(mask.sum()))
            inst_osm.append(-1)
            print(f"  {cls}: 1 instance, {int(mask.sum()):,} pts")

    dest = out("objects.npz", argv)
    np.savez_compressed(dest, oid=oid, inst_class=np.array(inst_class, np.uint8),
                        inst_size=np.array(inst_size, np.int32),
                        inst_osm=np.array(inst_osm, np.int64))
    ic, isz = np.array(inst_class), np.array(inst_size)
    print(f"\n{len(ic):,} instances, {int((oid >= 0).sum()):,}/{n:,} points covered")
    for cls in ("building", "floating", *CC_CLASSES):
        s = np.sort(isz[ic == idx[cls]])
        if len(s):
            print(f"  {cls:<11} n={len(s):<5} median {int(np.median(s)):>6,}  "
                  f"largest {s[-1]:>9,}")
    print(f"wrote {dest}  ({time.time()-t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
