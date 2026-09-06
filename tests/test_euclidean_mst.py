"""euclidean_mst against brute force — the only proof that matters.

scipy's minimum_spanning_tree over the FULL distance matrix is the definition
(docs/design/stage2-mst.md: the MST is defined on the complete graph). Both
tests demand the same EDGE SET, not merely the same total weight.
"""
import tempfile

import numpy as np
import pytest

import ducklidar as dl
from ducklidar.graph import read_mst_parts


def brute_force(P, f32=True):
    """The exact EMST edge set of `P`, as a set of (min, max) index pairs, and its weight.

    `f32` quantises the distance matrix to float32 first, because that is the
    weight the algorithm actually sees: `knn.parquet` stores `len` as float32.
    Without it, real survey points disagree on about one edge in two thousand —
    pairs whose lengths differ by ~1e-10 m are a tie in float32 and are then
    settled by (src, dst), which is the design's tie-break rule doing its job,
    not an error. Measured on a 6 m box of the store: 1 edge of 1,735, total
    weight identical to 6 decimals.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree

    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    if f32:
        d = d.astype(np.float32).astype(np.float64)
    t = minimum_spanning_tree(d).tocoo()
    return {(min(i, j), max(i, j)) for i, j in zip(t.row, t.col)}, float(t.data.sum())


def run(P, cap=2.5, k=10, **kw):
    """euclidean_mst over `P` (indices are the pids), as (edge set, weight)."""
    src, dst = [], []
    for s, d in dl.local_edges(P, k=k, cap=cap):
        src.append(s.astype(np.int64))
        dst.append(d.astype(np.int64))
    src, dst = np.concatenate(src), np.concatenate(dst)
    ln = np.linalg.norm(P[src] - P[dst], axis=1).astype(np.float32)

    def read_box(bbox):
        m = ((P[:, 0] >= bbox[0]) & (P[:, 0] <= bbox[2])
             & (P[:, 1] >= bbox[1]) & (P[:, 1] <= bbox[3]))
        return P[m], np.flatnonzero(m).astype(np.int64)

    bbox = (P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max())
    with tempfile.TemporaryDirectory() as work:
        dl.euclidean_mst(iter([(src, dst, ln)]), len(P), read_box, [np.arange(len(P))],
                         sweep_bbox=bbox, work=work, cap=cap, log=None, **kw)
        s, d, w = read_mst_parts(work)
    return {(min(a, b), max(a, b)) for a, b in zip(s, d)}, float(np.float64(w).sum())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_brute_force_synthetic(seed):
    """Clustered points: dense blobs (Kruskal's regime) far apart (Borůvka's)."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, 300, size=(6, 3)) * [1, 1, 0.05]
    P = np.concatenate([c + rng.normal(0, 1.2, size=(500, 3)) for c in centres])
    P = np.ascontiguousarray(P, np.float64)

    want, want_w = brute_force(P)
    got, got_w = run(P)
    assert len(got) == len(P) - 1
    assert got == want
    assert got_w == pytest.approx(want_w, rel=1e-6)


def test_matches_brute_force_isolated_pebble():
    """One point 400 m from everything — the doubling path, exercised."""
    rng = np.random.default_rng(7)
    P = np.concatenate([rng.normal(0, 3, size=(800, 3)),
                        [[400.0, 400.0, 5.0]],
                        rng.normal(0, 3, size=(400, 3)) + [40, 0, 0]])
    P = np.ascontiguousarray(P, np.float64)
    want, want_w = brute_force(P)
    got, got_w = run(P, r0=5.0)
    assert got == want
    assert got_w == pytest.approx(want_w, rel=1e-6)


def test_block_split_does_not_change_the_answer():
    """The sweep's block grid is an implementation detail, not a parameter:
    a halo smaller than the answer must be caught and widened, not accepted."""
    rng = np.random.default_rng(3)
    P = np.ascontiguousarray(np.concatenate(
        [rng.uniform(0, 200, size=(1500, 2)), rng.uniform(0, 5, size=(1500, 1))], axis=1))
    want, _ = brute_force(P)
    for block, halo in [(200.0, 25.0), (37.0, 2.0), (500.0, 50.0)]:
        got, _ = run(P, block=block, halo=halo)
        assert got == want, f"block={block} halo={halo}"
