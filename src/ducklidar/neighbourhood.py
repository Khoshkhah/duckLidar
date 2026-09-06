"""The kNN level, computed once — every neighbourhood product derives from it.

A point and its k nearest neighbours is the first thing that cannot be read
off a table row, and the last thing that needs to be computed more than
never: the covariance shape features and the mesh graph's local edges both
derive from one query (the study pipeline used to run it twice — once inside
PDAL for features, once for the graph — and threw both away).

    dist, idx = knn(P, k=16)
    feats = shape_features(P, idx)                 # e1, e2, e3, nz per point
    for src, dst in local_edges(P, nn=(dist, idx)):  # same tree, free
        ...

Partition + halo makes this exact out-of-core: neighbours at survey density
live within ~1 m, so any partition padded by more than the largest query
radius answers identically to a whole-survey run (docs/pipeline.md, level 2).
"""
import numpy as np

__all__ = ["balanced_boxes", "knn", "shape_features"]


def knn(xyz, k=16, *, chunk=1_000_000):
    """`(dist, idx)` of the `k` nearest neighbours per point, excluding the point itself.

    float32 / int32 arrays of shape (n, k) — the reusable product of the one
    KD-tree pass. Chunked queries keep the peak at the arrays themselves.
    """
    from scipy.spatial import cKDTree

    xyz = np.ascontiguousarray(xyz, np.float64)
    n = len(xyz)
    tree = cKDTree(xyz)
    dist = np.empty((n, k), np.float32)
    idx = np.empty((n, k), np.int32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d, i = tree.query(xyz[s:e], k=k + 1, workers=-1)
        dist[s:e], idx[s:e] = d[:, 1:], i[:, 1:]
    return dist, idx


def shape_features(xyz, idx, *, chunk=2_000_000, include_self=False):
    """Eigenvalue shape of each point's neighbourhood: `(e1, e2, e3, nz)` float32 arrays.

    e1 ≥ e2 ≥ e3 are the covariance eigenvalues normalised to sum 1 — the
    complete basis every published shape feature is a formula over
    (linearity (e1−e2)/e1, planarity (e2−e3)/e1, scattering e3/e1, …), so
    store these and derive the rest in a query. `nz` is |z| of the smallest
    eigenvector — the surface normal's verticality: ~1 on a roof, ~0 on a wall.

    ``include_self=True`` adds the point itself to its neighbourhood —
    PDAL's covariancefeatures convention (its knn=16 covariance runs over 17
    points). Measured against PDAL on the Vancouver tile: with self and
    sqrt-eigenvalue ratios the median |Δ| is 0.004, the rest being kNN
    tie-breaking; without self it is 20× worse. PDAL also reports features
    on SQRT eigenvalues by default — apply sqrt to these e's before forming
    PDAL-comparable ratios (normalisation cancels there too).
    """
    xyz = np.ascontiguousarray(xyz, np.float64)
    n, k = idx.shape
    e = np.empty((n, 3), np.float32)
    nz = np.empty(n, np.float32)
    for s in range(0, n, chunk):
        t = min(s + chunk, n)
        nb = xyz[idx[s:t]]                          # (m, k, 3)
        if include_self:
            nb = np.concatenate([xyz[s:t, None, :], nb], axis=1)
        nb -= nb.mean(axis=1, keepdims=True)
        cov = np.einsum("mki,mkj->mij", nb, nb) / nb.shape[1]
        w, v = np.linalg.eigh(cov)                  # ascending
        tot = np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        e[s:t] = (w[:, ::-1] / tot).astype(np.float32)
        nz[s:t] = np.abs(v[:, 2, 0]).astype(np.float32)   # eigenvector of smallest w
    return e[:, 0], e[:, 1], e[:, 2], nz


def balanced_boxes(files, bbox, max_points, *, min_side=50.0):
    """Data-driven partitions: recursive median splits of `bbox` until every
    leaf holds ≤ `max_points` of the store. Returns [(x0, y0, x1, y1, n), …].

    The survey's tile grid is delivery packaging, not a compute plan — a
    fixed grid gives one partition 200 k points and its neighbour 4 M. This
    derives the partition from the data itself: split the longer axis at the
    (approximate) median of the points inside, recurse. Leaves are balanced
    by construction and memory per leaf is bounded before any work starts.
    Add your halo at query time; `min_side` stops pathological slivers.
    """
    import duckdb

    con = duckdb.connect()
    src = "read_parquet([" + ", ".join(f"'{f}'" for f in files) + "])"

    def node(x0, y0, x1, y1):
        n = con.execute(
            f"select count(*) from {src} "
            f"where x >= ? and x < ? and y >= ? and y < ?",
            [x0, x1, y0, y1]).fetchone()[0]
        if n == 0:
            return []
        wide = (x1 - x0) >= (y1 - y0)
        if n <= max_points or (x1 - x0 if wide else y1 - y0) < 2 * min_side:
            return [(x0, y0, x1, y1, int(n))]
        axis = "x" if wide else "y"
        cut = con.execute(
            f"select approx_quantile({axis}, 0.5) from {src} "
            f"where x >= ? and x < ? and y >= ? and y < ?",
            [x0, x1, y0, y1]).fetchone()[0]
        lo, hi = (x0, x1) if wide else (y0, y1)
        cut = min(max(float(cut), lo + min_side), hi - min_side)
        if wide:
            return node(x0, y0, cut, y1) + node(cut, y0, x1, y1)
        return node(x0, y0, x1, cut) + node(x0, cut, x1, y1)

    return node(*bbox)
