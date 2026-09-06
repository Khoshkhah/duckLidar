"""The support forest's one runnable check.

The doubling must count the path EXCLUDING the node and INCLUDING the root,
and the nearer root system must win the point.

    0 --1m-- 1 --1m-- 2 --1m-- 3 --2m-- 4        6 (isolated)
    roots: 0 (ground, cat WALL), 4 (water, cat FOLIAGE)
    node 5 hangs off 2 by 0.5 m.  No distance ties — a tie would make the
    claim non-deterministic and this test flaky.
"""
import numpy as np
from scipy.sparse import csr_matrix

from ducklidar.labeling import (CAT_FOLIAGE, CAT_OTHER, CAT_WALL, SYS_GROUND,
                                SYS_WATER, path_shares, support_forest)

SRC = np.array([0, 1, 1, 2, 2, 3, 3, 4, 2, 5], np.int32)
DST = np.array([1, 0, 2, 1, 3, 2, 4, 3, 5, 2], np.int32)
LEN = np.array([1, 1, 1, 1, 1, 1, 2, 2, .5, .5], np.float64)
CATS = np.array([CAT_WALL, CAT_OTHER, CAT_OTHER, CAT_OTHER,
                 CAT_FOLIAGE, CAT_OTHER, CAT_OTHER], np.uint8)


def test_support_forest():
    G = csr_matrix((LEN, (SRC, DST)), shape=(7, 7))
    f = support_forest(G, {SYS_GROUND: np.array([0]), SYS_WATER: np.array([4])}, CATS)
    assert list(f["system"]) == [1, 1, 1, 2, 2, 1, 0]
    assert list(f["depth"]) == [0, 1, 2, 1, 0, 3, -1]
    assert np.isinf(f["dist"][6]) and abs(f["dist"][5] - 2.5) < 1e-6
    # node 2's path is 1 -> 0: one OTHER (node 1) + one WALL (the root)
    assert list(f["cross"][2]) == [1, 1, 0]
    # node 3's path is 4, the water root: one FOLIAGE, no wall
    assert list(f["cross"][3]) == [0, 0, 1]
    # node 5 hangs off 2, so it inherits 2's path plus 2 itself
    assert list(f["cross"][5]) == [2, 1, 0]
    bsh, vsh = path_shares(f)
    assert abs(bsh[5] - 1 / 3) < 1e-9 and vsh[5] == 0.0
    assert bsh[0] == 0.0 and f["depth"][6] == -1     # roots and orphans stay out


def test_blocker_strands():
    """Deleting a node's edges is how the oracle's blocked experiments work."""
    keep = ~np.array([0, 1, 0, 0, 0, 0, 0], bool)
    k = keep[SRC] & keep[DST]
    f = support_forest(csr_matrix((LEN[k], (SRC[k], DST[k])), shape=(7, 7)),
                       {SYS_GROUND: np.array([0])}, CATS)
    assert list(f["system"]) == [1, 0, 0, 0, 0, 0, 0]
