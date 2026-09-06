"""The mesh graph: kNN(k=10, cap 2.5 m) UNION the MST bridges -- connected by construction.

Kaveh's design: the kNN graph the swept rules trust is honest but disconnected --
crowns, buildings, boats are islands, and that is itself a signal. The mesh keeps
every local edge exactly as those rules know it and adds the minimum repair: the
MST edges that cross between kNN islands, one fewer than the number of islands
(Boruvka rounds; each island merges along its shortest outgoing edge, which is the
minimum-spanning-tree guarantee of connectivity). Every bridge is typed and carries
its length -- "one component, but only through a 14 m bridge" is the isolation
measurement the marina problem was starving for.

Only the bridges are stored (out/mesh_graph.npz: bridge_src, bridge_dst,
bridge_len, plus k/cap/n); the local half is deterministic from (k=10, cap 2.5)
over the box read order, so any consumer rebuilds it with one KD-tree query.

    python tools/mesh_graph.py      # writes out/mesh_graph.npz
"""
import sys
import time

import numpy as np

import ducklidar as dl
from ._env import OUTDIR, check
from .pipeline import B

K_LOCAL = 10     # the conviction graph's k -- rules swept on it must see the same edges
CAP = 2.5        # m, same reason
K_SEARCH = 16    # cached neighbours for the Boruvka rounds; misses go to exact scans


def _exact_bridge(P, tree_pts_idx, island):
    """Shortest edge from `island` to anywhere outside it, by widening box search.

    Exact: only returns a pair once the best distance is <= the search radius, so
    no closer partner can be hiding outside the box.
    """
    from scipy.spatial import cKDTree
    Pi = P[island]
    lo, hi = Pi.min(0), Pi.max(0)
    inside = np.zeros(len(P), bool)
    inside[island] = True
    r = 10.0
    while True:
        m = (~inside & (P[:, 0] >= lo[0] - r) & (P[:, 0] <= hi[0] + r)
             & (P[:, 1] >= lo[1] - r) & (P[:, 1] <= hi[1] + r))
        cand = np.flatnonzero(m)
        if len(cand):
            d, j = cKDTree(P[cand]).query(Pi, k=1, workers=-1)
            k = int(np.argmin(d))
            if d[k] <= r:
                return int(island[k]), int(cand[j[k]]), float(d[k])
        r *= 2.0


def main(argv):
    t0 = time.time()
    pts = dl.read(check(), B, fields=())     # coordinates only — the graph
    P = np.column_stack([np.asarray(pts["x"], np.float64), np.asarray(pts["y"], np.float64),
                         np.asarray(pts["z"], np.float64)])
    del pts                                  # needs nothing else (tile-scale RAM)
    n = len(P)
    print(f"{n:,} returns")

    from scipy.spatial import cKDTree

    import resource

    def mem(tag):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print(f"  [{tag}] peak rss {rss:.1f} GB", flush=True)

    tree = cKDTree(P)
    qd = np.empty((n, K_SEARCH), np.float32)
    qi = np.empty((n, K_SEARCH), np.int32)
    for s in range(0, n, 1_000_000):
        e = min(s + 1_000_000, n)
        d, i = tree.query(P[s:e], k=K_SEARCH + 1, workers=-1)
        qd[s:e], qi[s:e] = d[:, 1:], i[:, 1:]
    del tree                    # ~2 GB; rebuilt cheaply if a fallback needs it
    print(f"  neighbours cached (k={K_SEARCH})  {time.time() - t0:.0f}s")
    mem("cached")

    # edges built CHUNKED: the one-shot repeat/ravel/filter held three full
    # 683M-element temporaries at once and OOM-killed the tile run three
    # times (2026-08-01) before this
    aa, bb = [], []
    for s in range(0, n, 4_000_000):
        e = min(s + 4_000_000, n)
        okc = qd[s:e, :K_LOCAL] <= CAP
        ac = np.repeat(np.arange(s, e, dtype=np.int32), K_LOCAL)[okc.ravel()]
        bc = np.ascontiguousarray(qi[s:e, :K_LOCAL]).ravel()[okc.ravel()]
        aa.append(ac)
        bb.append(bc)
    a = np.concatenate(aa)
    b = np.concatenate(bb)
    del aa, bb
    n_local = len(a)
    mem("edges")
    # ---- weak components in-house: scipy's csgraph validation recopies a
    # 600M-edge graph to float64, which is the memory the tile does not
    # have (peak 19.8 GB, SIGKILL — measured 2026-08-01). Min-label
    # propagation instead: forward direction via reduceat over the
    # row-sorted edge list, reverse via ufunc.at, pointer jumps between
    # rounds. Exact weak connectivity, no graph object at all.
    counts = np.bincount(a, minlength=n)
    indptr = np.concatenate(([0], np.cumsum(counts)))
    rows = np.flatnonzero(counts)
    starts = indptr[:-1][rows]
    lab = np.arange(n, dtype=np.int32)
    rounds = 0
    while True:
        rounds += 1
        old = lab.copy()
        fw = np.minimum.reduceat(lab[b], starts)
        lab[rows] = np.minimum(lab[rows], fw)
        np.minimum.at(lab, b, lab[a])
        for _ in range(3):
            lab = lab[lab]
        if np.array_equal(lab, old):
            break
    del counts, indptr, rows, starts, old
    uniq, cur = np.unique(lab, return_inverse=True)
    cur = cur.astype(np.int32)
    ncomp0 = len(uniq)
    del uniq, lab
    del a, b
    mem(f"components ({rounds} rounds)")
    print(f"  local edges (k={K_LOCAL}, cap {CAP} m): {n_local:,}   islands: {ncomp0:,}")

    # ---- Boruvka: each round, every island merges along its shortest outgoing edge.
    # The cached k={K_SEARCH} neighbours answer for small/adjacent islands; a dense
    # island whose 16 nearest are all its own (a big blob across open water) falls
    # through to the exact widening-box scan, one per stalled island.
    src, dst, blen = [], [], []
    rows = np.arange(n)
    while True:
        u, lab = np.unique(cur, return_inverse=True)
        m = len(u)
        if m == 1:
            break
        lab = lab.astype(np.int32)
        f = lab[qi] != lab[:, None]
        anyf = f.any(1)
        cand = np.flatnonzero(anyf)
        fc = f[cand].argmax(1)
        cd, cq = qd[cand, fc], qi[cand, fc]
        order = np.lexsort((cd, lab[cand]))
        firsts = np.unique(lab[cand][order], return_index=True)[1]
        picks = order[firsts]                      # per-island shortest outgoing edge

        par = np.arange(m)

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        added = 0
        for o in picks:
            p, q, dpq = int(cand[o]), int(cq[o]), float(cd[o])
            ra, rb = find(lab[p]), find(lab[q])
            if ra != rb:
                par[ra] = rb
                src.append(p); dst.append(q); blen.append(dpq)
                added += 1
        if added == 0:
            # every remaining island is blind at k=16 -- exact-scan the smallest one
            sizes = np.bincount(lab, minlength=m)
            c = int(np.argsort(sizes)[0])
            p, q, dpq = _exact_bridge(P, rows, np.flatnonzero(lab == c))
            par[find(lab[p])] = find(lab[q])
            src.append(p); dst.append(q); blen.append(dpq)
            added = 1
        cur = np.array([find(i) for i in range(m)], np.int32)[lab]
        print(f"  round: {m:,} islands, {added:,} bridges  {time.time() - t0:.0f}s")

    blen = np.array(blen, np.float32)
    assert len(blen) == ncomp0 - 1, (len(blen), ncomp0)
    dest = OUTDIR / "mesh_graph.npz"
    np.savez_compressed(dest, bridge_src=np.array(src, np.int32),
                        bridge_dst=np.array(dst, np.int32), bridge_len=blen,
                        n=n, k=K_LOCAL, cap=CAP)
    hist = {b: int(((blen > lo_) & (blen <= hi_)).sum())
            for b, lo_, hi_ in [("<=5 m", 0, 5), ("5-10 m", 5, 10),
                                ("10-20 m", 10, 20), (">20 m", 20, np.inf)]}
    print(f"connected: {ncomp0:,} islands -> 1 via {len(blen):,} bridges "
          f"(min {blen.min():.2f} / median {np.median(blen):.2f} / max {blen.max():.2f} m)")
    print("  " + "   ".join(f"{k} {v:,}" for k, v in hist.items()))
    print(f"wrote {dest}  ({dest.stat().st_size / 1e6:.1f} MB)  {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
