"""The mesh graph's local half — deterministic, so it can be built anywhere.

lidar-learning's mesh_graph.py builds kNN(k=10, cap 2.5 m) edges over a window
and stores only the MST bridges, because the local edges are a pure function
of the points. This is that function, graduated, so an edge dump (Kaveh,
2026-08-02: the whole graph belongs in Parquet, queryable in DuckDB without
loading it) produces byte-identical connectivity to what the pipeline saw.

Chunked exactly like the tool: the one-shot repeat/ravel/filter held three
full 683M-element temporaries at once and was OOM-killed three times
(2026-08-01) before the chunking.
"""
import json
import time
from pathlib import Path

import numpy as np

__all__ = ["local_edges", "euclidean_mst", "read_mst_parts"]


def local_edges(xyz, k=10, cap=2.5, *, chunk=4_000_000, nn=None):
    """Yield (src, dst) int32 index chunks of the kNN(`k`, `cap`) edge set of `xyz` (n, 3).

    Each point contributes a directed edge to each of its `k` nearest
    neighbours within `cap` metres — the exact edge set mesh_graph.py's
    components run on. Canonicalise (min, max) and drop duplicates downstream
    when an undirected set is wanted; node ids are row indices, so a stable
    cross-window key is the point ids of the rows.

    Pass `nn=(dist, idx)` from :func:`ducklidar.knn` (with ≥ `k` columns) to
    reuse an existing query — the kNN level is computed once, everything
    derives from it.
    """
    if nn is not None:
        dist, idx = nn
        assert idx.shape[1] >= k, f"nn carries {idx.shape[1]} neighbours, need {k}"
        for s in range(0, len(idx), chunk):
            e = min(s + chunk, len(idx))
            ok = (dist[s:e, :k] <= cap).ravel()
            src = np.repeat(np.arange(s, e, dtype=np.int32), k)[ok]
            dst = np.ascontiguousarray(idx[s:e, :k]).ravel()[ok]
            yield src, dst
        return

    from scipy.spatial import cKDTree

    xyz = np.ascontiguousarray(xyz, np.float64)
    n = len(xyz)
    tree = cKDTree(xyz)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d, i = tree.query(xyz[s:e], k=k + 1, workers=-1)
        ok = (d[:, 1:k + 1] <= cap).ravel()
        src = np.repeat(np.arange(s, e, dtype=np.int32), k)[ok]
        dst = i[:, 1:k + 1].astype(np.int32).ravel()[ok]
        yield src, dst


# ---------------------------------------------------------------------------
# The Euclidean MST — Kruskal inside the cap, Borůvka beyond it.
# docs/design/stage2-mst.md is the specification; this is that, and nothing more.
# ---------------------------------------------------------------------------

#: One bucket-file record. len first so a raw byte sort is almost the right sort.
REC = np.dtype([("len", "<f4"), ("src", "<i8"), ("dst", "<i8")])


def _find(parent, ids):
    """Roots for `ids` (vectorised), compressing the paths walked."""
    r = parent[np.asarray(ids)]
    while True:
        rr = parent[r]
        if (rr == r).all():
            parent[ids] = r
            return r
        r = rr


def _union(parent, src, dst):
    """Union every src–dst pair, in one shot, leaving the roots at depth 1.

    The obvious vectorisation — ``parent[max] = min``, repeated until no pair
    is left split — is quadratic in disguise: colliding writes are dropped and
    retried, and the chains it builds make the next :func:`_find` walk them.
    Measured on the old area, it spent >350 s inside a single 17 M-edge bucket.

    Instead: the pairs are a graph over the roots involved, so ask scipy for
    its connected components and point every root straight at its group's
    smallest member. One pass, no retry, and no chain longer than one hop —
    which is what keeps :func:`_find` at two or three gathers forever.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    s, d = _find(parent, src), _find(parent, dst)
    m = s != d
    if not m.any():
        return
    s, d = s[m], d[m]
    u, inv = np.unique(np.concatenate([s, d]), return_inverse=True)
    h, n = len(s), len(u)
    _, lab = connected_components(
        coo_matrix((np.ones(h, np.int8), (inv[:h], inv[h:])), shape=(n, n)),
        directed=False)
    rep = np.empty(lab.max() + 1, np.int64)
    rep[lab[::-1]] = np.arange(n)[::-1]      # u is sorted, so last write = min
    parent[u] = u[rep[lab]]


def _kruskal(parent, src, dst):
    """Indices of the edges Kruskal accepts from `src`/`dst`, which must already
    be in the strict order (len, src, dst). Unions them into `parent`.

    Sequential Kruskal is a per-edge loop, which Python cannot afford at 10^9
    edges — and a numpy Borůvka *within* the bucket is no better, because each
    of its ~15 rounds re-walks the global parent array (measured: 100 s for one
    8.5 M-edge bucket). So the bucket is CONTRACTED first — nodes become the
    components that exist right now, dense-relabelled, and only the edges still
    crossing survive — and scipy's C Kruskal spans that small graph in one
    call.

    The weight handed to scipy is the edge's POSITION in the caller's
    (len, src, dst) order, not its length. Positions are distinct, so the MST
    is unique and is exactly the edge set sequential Kruskal would accept,
    tie-break included — while equal float32 lengths would have left the choice
    to scipy.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    rs, rd = _find(parent, src), _find(parent, dst)
    live = np.flatnonzero(rs != rd)           # both ends already joined: drop
    if not len(live):
        return np.empty(0, np.int64)
    u, inv = np.unique(np.concatenate([rs[live], rd[live]]), return_inverse=True)
    n, k = len(u), len(live)
    i, j = inv[:k], inv[k:]
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    del i, j, inv, rs, rd
    # one edge per node pair — the first, which is the shortest, since the
    # caller sorted; scipy would otherwise SUM the duplicates into a fiction
    first = np.unique(lo.astype(np.int64) * n + hi, return_index=True)[1]
    t = minimum_spanning_tree(coo_matrix(((first + 1).astype(np.float64),
                                          (lo[first], hi[first])),
                                         shape=(n, n))).tocoo()
    take = np.sort(live[t.data.astype(np.int64) - 1])
    _union(parent, src[take], dst[take])
    return take


def _best_per_comp(comp, ln, src, dst):
    """Per component, the single edge minimal under (len, src, dst)."""
    if not len(comp):
        return comp, ln, src, dst
    o = np.lexsort((dst, src, ln, comp))
    cs = comp[o]
    i = o[np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])]
    return comp[i], ln[i], src[i], dst[i]


def _blocks(bbox, side):
    # Blocks are half-open [b, b+side) and deliberately NOT clipped to bbox: a
    # point sitting exactly on the bbox's upper edge must land in the last
    # block, not in none of them (it cost a failing pebble test to notice).
    x0, y0, x1, y1 = bbox
    for bx in np.arange(x0, x1 + side, side)[:max(1, int(np.ceil((x1 - x0) / side)))]:
        for by in np.arange(y0, y1 + side, side)[:max(1, int(np.ceil((y1 - y0) / side)))]:
            yield float(bx), float(by), float(bx + side), float(by + side)


def _write_part(path, src, dst, ln):
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([("src", pa.int64()), ("dst", pa.int64()), ("len", pa.float32())])
    tmp = path.with_suffix(".parquet.part")
    pq.write_table(pa.table({"src": np.asarray(src, np.int64),
                             "dst": np.asarray(dst, np.int64),
                             "len": np.asarray(ln, np.float32)}, schema=schema),
                   tmp, compression="zstd")
    tmp.rename(path)


def read_mst_parts(work):
    """The (src, dst, len) written so far by :func:`euclidean_mst` under `work`."""
    import pyarrow.parquet as pq

    parts = sorted((Path(work) / "parts").glob("*.parquet"))
    if not parts:
        return (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32))
    t = pq.read_table(parts)
    return (t["src"].to_numpy(), t["dst"].to_numpy(), t["len"].to_numpy())


class _Stop(Exception):
    """Time budget spent — state is on disk, call again with resume=True."""

def _ball(parent, read_box, pid, xyz, rc, comps, r0, r_max, say):
    """Each component in `comps` asks, from its own side, for its nearest point
    outside itself: a bounded ball on the sorted store, widened by doubling.

    Certified, not hoped for: the box is the component's bbox grown by *r*, so
    every one of its points has *r* metres of complete knowledge around it, and
    an answer is only accepted at ``d <= r``.
    """
    from scipy.spatial import cKDTree

    out = []
    if not len(comps):
        return out
    o = np.argsort(rc, kind="stable")
    rs = rc[o]
    start = {int(rs[i]): i for i in np.flatnonzero(np.r_[True, rs[1:] != rs[:-1]])}
    end = dict(zip(start, list(start.values())[1:] + [len(rs)]))
    for n, c in enumerate(comps):
        sel = o[start[int(c)]:end[int(c)]]
        P, ids = xyz[sel], pid[sel]
        r = r0
        while r <= r_max:
            box = (P[:, 0].min() - r, P[:, 1].min() - r,
                   P[:, 0].max() + r, P[:, 1].max() + r)
            bxyz, bpid = read_box(box)
            if len(bpid):
                other = _find(parent, bpid) != c
                if other.any():
                    oxyz, opid = bxyz[other], bpid[other]
                    d, i = cKDTree(oxyz).query(P, k=1, workers=-1)
                    d, i = np.atleast_1d(d), np.atleast_1d(i)
                    if d.min() <= r:
                        j = np.lexsort((opid[i], ids, d))[0]
                        out.append((int(c), float(d[j]), int(ids[j]), int(opid[i[j]])))
                        break
            r *= 2
        else:
            raise RuntimeError(f"component {c} found nothing within {r_max} m")
        if say and n and n % 500 == 0:
            say(f"  ball queries: {n:,}/{len(comps):,}")
    return out


def euclidean_mst(edges, n_points, read_box, members, *, sweep_bbox, work,
                  cap=2.5, bucket_m=0.025, block=200.0, halo=25.0, k_probe=32,
                  r0=5.0, r_max=4096.0, read_chunk=16_000_000,
                  resume=False, budget=None, log=print):
    """The exact Euclidean MST of the member points, as Parquet parts under
    ``work/parts`` (read them back with :func:`read_mst_parts`).

    Two regimes, split at `cap`, because one schedule is wrong on both sides of
    it (docs/design/stage2-mst.md):

    (a) **inside the cap — Kruskal, not Borůvka rounds.** Every candidate
        shorter than `cap` is already in `edges`; bucket them by length into
        ``cap / bucket_m`` bins in one pass, then consume the bins in order,
        sorting each in memory. Two passes where round-per-pass Borůvka needs
        ~26 of them.
    (b) **beyond the cap — Borůvka.** What survives (a) is thousands of
        components, one holding almost everything. Each of the others asks for
        its nearest point outside itself; the largest is never asked, because
        *w* is symmetric, so whichever pebble finds the giant supplies the
        giant's edge too. Round 1 is one blocked sweep of the store, which
        answers nearly every component at once and caches the small side's
        coordinates; anything it cannot certify, and every later round, is a
        bounded ball query widened by doubling.

    `edges` yields ``(src, dst, len)`` batches — every pair within `cap`,
    directed or not, duplicates fine. `read_box(bbox)` returns
    ``(xyz (n,3) float64, pid (n,) int64)`` for the MEMBER points in a 2-D
    bbox — membership is the caller's business, so nothing outside *V* is ever
    a candidate. `members` is an iterable of member pid batches, used once to
    size the components. `n_points` sizes the union-find, so it must exceed
    every pid.

    Resumable: with `budget` seconds spent, state is checkpointed under `work`
    and the call raises ``SystemExit(2)``; rerun with ``resume=True``.
    """
    from scipy.spatial import cKDTree

    work = Path(work)
    (work / "parts").mkdir(parents=True, exist_ok=True)
    state_p, parent_p, cache_p = work / "state.json", work / "parent.npy", work / "cache.npz"
    nb = int(np.ceil(cap / bucket_m))
    bpath = [work / f"bucket-{i:03d}.bin" for i in range(nb)]
    t0, peak, last = time.time(), [0.0], [time.time()]

    def rss():
        import resource
        peak[0] = max(peak[0], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
        return peak[0]

    def say(msg):
        if log:
            log(f"[{time.time() - t0:7.1f}s peak {rss():5.2f} GB] {msg}")

    def spend():
        if budget is not None and time.time() - t0 > budget:
            raise _Stop

    state = (json.loads(state_p.read_text()) if resume and state_p.exists()
             else {"phase": "bucket", "batches": 0, "sizes": [0] * nb,
                   "bucket": 0, "round": 1, "candidates": 0})
    parent = (np.load(parent_p) if resume and parent_p.exists()
              else np.arange(n_points, dtype=np.int32))

    def checkpoint(save_parent=True, **kw):
        # `parent` is 4 bytes per store row (1.8 GB for the window), so it is
        # written only when it has actually moved and only at a resumable
        # boundary — saving it per bucket would cost more than the buckets do.
        state.update(kw)
        if save_parent:
            np.save(parent_p, parent)
        state_p.write_text(json.dumps(state))

    try:
        # -- (a1) one pass over the candidates, into length buckets ----------
        if state["phase"] == "bucket":
            for p, s in zip(bpath, state["sizes"]):
                with open(p, "r+b" if p.exists() else "wb") as f:
                    f.truncate(s)                  # undo a half-written batch
            fh = [open(p, "r+b") for p in bpath]
            for f, s in zip(fh, state["sizes"]):
                f.seek(s)
            done, n_in = state["batches"], state["candidates"]
            for j, (src, dst, ln) in enumerate(edges):
                if j < done:
                    continue
                b = np.minimum((ln / bucket_m).astype(np.int32), nb - 1)
                o = np.argsort(b, kind="stable")
                a = np.empty(len(o), REC)
                a["len"], a["src"], a["dst"] = ln[o], src[o], dst[o]
                cut = np.r_[0, np.cumsum(np.bincount(b, minlength=nb))]
                for i in range(nb):
                    if cut[i + 1] > cut[i]:
                        fh[i].write(a[cut[i]:cut[i + 1]].tobytes())
                done, n_in = j + 1, n_in + len(o)
                del a, o, b
                if done % 32 == 0:
                    for f in fh:
                        f.flush()
                    checkpoint(False, batches=done, candidates=n_in,
                               sizes=[f.tell() for f in fh])
                    say(f"bucket pass: {n_in:,} candidates")
                    spend()
            for f in fh:
                f.flush()
            checkpoint(False, phase="kruskal", batches=done, candidates=n_in,
                       sizes=[f.tell() for f in fh])
            for f in fh:
                f.close()
            say(f"bucket pass done: {n_in:,} candidates into {nb} buckets")

        # -- (a2) the buckets in order — Kruskal -----------------------------
        if state["phase"] == "kruskal":
            for i in range(state["bucket"], nb):
                # Drop the already-decided edges FIRST, and do it while
                # STREAMING the bucket in: by the late buckets nearly every
                # candidate has both ends inside one component and can never
                # separate them again, and the bucket that holds the modal
                # length is far from an even 1/nb share of the table (measured
                # on the window: 10.8 %, i.e. 207 M records / 4.1 GB, where a
                # uniform split would have predicted 19 M). Reading it whole to
                # throw 90 % of it away is how this OOMs. The filter is exact —
                # `parent` does not move until the whole bucket is decided, so
                # every chunk sees the same components.
                n_rec = bpath[i].stat().st_size // REC.itemsize
                keep, n_in = [], 0
                for off in range(0, max(n_rec, 1), read_chunk):
                    c = np.fromfile(bpath[i], dtype=REC, count=min(read_chunk, n_rec - off),
                                    offset=off * REC.itemsize)
                    n_in += len(c)
                    keep.append(c[_find(parent, c["src"]) != _find(parent, c["dst"])])
                    del c
                a = np.concatenate(keep) if keep else np.empty(0, REC)
                del keep
                if len(a):
                    # contiguous columns, then drop the struct: fancy indexing a
                    # strided field view costs more than the copy does
                    s_, d_, l_ = a["src"].copy(), a["dst"].copy(), a["len"].copy()
                    del a
                    o = np.lexsort((d_, s_, l_))
                    s_, d_, l_ = s_[o], d_[o], l_[o]
                    del o
                    take = _kruskal(parent, s_, d_)
                    if len(take):
                        _write_part(work / "parts" / f"a{i:03d}.parquet",
                                    s_[take], d_[take], l_[take])
                    say(f"bucket {i:3d} (< {(i + 1) * bucket_m:.3f} m): {n_in:,} "
                        f"candidates, {len(s_):,} still crossing, {len(take):,} accepted")
                    del take, s_, d_, l_
                else:
                    del a
                state["bucket"] = i + 1
                # by elapsed, not by count: the first buckets are minutes each
                # and the last are milliseconds, so a fixed stride either saves
                # 1.8 GB for nothing or loses an hour to one kill
                if time.time() - last[0] > 90:
                    checkpoint()
                    last[0] = time.time()
                spend()
            checkpoint(phase="boruvka")
            say("Kruskal inside the cap done")

        # -- (b) Borůvka over what the cap left behind -----------------------
        sizes = {}
        for pids in members:
            u, c = np.unique(_find(parent, pids), return_counts=True)
            for k_, v_ in zip(u.tolist(), c.tolist()):
                sizes[k_] = sizes.get(k_, 0) + v_
        n_members = sum(sizes.values())
        giant = max(sizes, key=sizes.get)
        say(f"after the cap: {len(sizes):,} components over {n_members:,} points, "
            f"largest {sizes[giant]:,} ({100 * sizes[giant] / n_members:.2f} %)")
        del sizes

        cache = np.load(cache_p) if cache_p.exists() else None
        while True:
            spend()
            rnd = state["round"]
            found, todo = [], None
            if cache is None:
                # Round 1: ONE blocked sweep. It answers nearly every component
                # at once and caches the small side, so no later round has to
                # read the giant's points again.
                # One npz per block, so a sweep too long for one call resumes
                # where it stopped instead of starting over.
                sweep_d = work / "sweep"
                sweep_d.mkdir(exist_ok=True)
                blks = list(_blocks(sweep_bbox, block))
                for nblk, (bx0, by0, bx1, by1) in enumerate(blks):
                    bf = sweep_d / f"{nblk:04d}.npz"
                    if bf.exists():
                        continue
                    xyz, pid = read_box((bx0 - halo, by0 - halo, bx1 + halo, by1 + halo))
                    z = {k: np.empty(0, t) for k, t in
                         (("pid", np.int64), ("c_comp", np.int32), ("c_len", np.float32),
                          ("c_src", np.int64), ("c_dst", np.int64),
                          ("u_comp", np.int32), ("u_bnd", np.float32))}
                    z["xyz"] = np.empty((0, 3))
                    if not len(pid):
                        np.savez(bf, **z)
                        continue
                    r = _find(parent, pid)
                    q = np.flatnonzero((xyz[:, 0] >= bx0) & (xyz[:, 0] < bx1)
                                       & (xyz[:, 1] >= by0) & (xyz[:, 1] < by1)
                                       & (r != giant))
                    z["pid"], z["xyz"] = pid[q], xyz[q]
                    if not len(q):
                        np.savez(bf, **z)
                        continue
                    kk = min(k_probe, len(xyz))
                    dd, ii = cKDTree(xyz).query(xyz[q], k=kk, workers=-1)
                    dd, ii = np.atleast_2d(dd), np.atleast_2d(ii)
                    diff = r[ii] != r[q][:, None]
                    has, j = diff.any(1), diff.argmax(1)
                    row = np.arange(len(q))
                    bd, bi = dd[row, j], ii[row, j]
                    # certified only inside the halo: a 2-D box bounds a 3-D
                    # ball, since 3-D distance >= horizontal distance
                    margin = np.minimum.reduce([
                        xyz[q, 0] - (bx0 - halo), (bx1 + halo) - xyz[q, 0],
                        xyz[q, 1] - (by0 - halo), (by1 + halo) - xyz[q, 1]])
                    ok = has & (bd <= margin)
                    z["c_comp"], z["c_len"] = r[q][ok], bd[ok].astype(np.float32)
                    z["c_src"], z["c_dst"] = pid[q][ok], pid[bi[ok]]
                    # what an uncertified point still proves: nothing outside
                    # its component is nearer than this
                    z["u_comp"] = r[q][~ok]
                    z["u_bnd"] = np.minimum(margin, dd[:, kk - 1])[~ok].astype(np.float32)
                    del xyz, pid, r, dd, ii, diff
                    np.savez(bf, **z)
                    if nblk % 8 == 0:
                        say(f"round 1 sweep: block {nblk + 1}/{len(blks)}")
                    spend()
                got = [np.load(sweep_d / f"{i:04d}.npz") for i in range(len(blks))]
                nblk = len(got)
                cat = lambda k: np.concatenate([g[k] for g in got])   # noqa: E731
                cache = {"pid": cat("pid"), "xyz": cat("xyz")}
                np.savez(cache_p, **cache)
                comp, ln, src, dst = _best_per_comp(cat("c_comp"), cat("c_len"),
                                                    cat("c_src"), cat("c_dst"))
                ub_c, ub_v = cat("u_comp"), cat("u_bnd")
                del got
                bound = {}
                if len(ub_c):
                    bc, bl, _, _ = _best_per_comp(ub_c, ub_v, ub_c, ub_c)
                    bound = dict(zip(bc.tolist(), bl.tolist()))
                good = np.fromiter((ln[i] <= bound.get(int(comp[i]), np.inf)
                                    for i in range(len(comp))), bool, len(comp))
                found = [(int(c), float(l), int(s), int(d)) for c, l, s, d
                         in zip(comp[good], ln[good], src[good], dst[good])]
                todo = sorted(set(bound) - set(comp[good].tolist()))
                say(f"round 1 sweep: {nblk} blocks, {len(cache['pid']):,} points on the "
                    f"small side, {len(found):,} components answered, "
                    f"{len(todo):,} need a widened ball")

            pid, xyz = cache["pid"], cache["xyz"]
            rc = _find(parent, pid)
            giant = int(_find(parent, np.array([giant]))[0])
            live = np.unique(rc[rc != giant])
            todo = live.tolist() if todo is None else [c for c in todo if c in set(live.tolist())]
            found += _ball(parent, read_box, pid, xyz, rc, todo, r0, r_max, say)
            if not found:
                break
            found.sort(key=lambda e: (e[1], e[2], e[3]))
            fs = np.array([e[2] for e in found], np.int64)
            fd = np.array([e[3] for e in found], np.int64)
            fl = np.array([e[1] for e in found], np.float32)
            take = _kruskal(parent, fs, fd)
            _write_part(work / "parts" / f"b{rnd:02d}.parquet", fs[take], fd[take], fl[take])
            left = len(np.unique(_find(parent, pid)))
            say(f"round {rnd}: {len(found):,} minimum outgoing edges, {len(take):,} added, "
                f"{left:,} components left on the small side")
            checkpoint(round=rnd + 1)
            if left <= 1:
                break
    except _Stop:
        checkpoint(state["phase"] != "bucket")
        say("budget spent — checkpointed, rerun with resume=True")
        raise SystemExit(2)

    checkpoint(phase="done")
    return {"peak_gb": round(rss(), 2), "elapsed": round(time.time() - t0, 1),
            "candidates": state["candidates"], "rounds": state["round"]}
