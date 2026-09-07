"""The two paths that free `building3d` from any one app: points without labels, and no roofer solid.

Both are new in the port — the pilot always had a label table and always had a solid, so
neither path has ever run there. No GPU, no network, no photos: a synthetic gabled shed.
"""
import numpy as np
import pytest

from ducklidar import _building3d as b3d
from ducklidar.building import points as bpts


def shed(n=4000, seed=0):
    """A 12 x 8 m shed on flat ground at z = 2: roof returns 5-7 m, ground returns around it."""
    rng = np.random.default_rng(seed)
    ring = np.array([[0.0, 0.0], [12.0, 0.0], [12.0, 8.0], [0.0, 8.0]])
    rx, ry = rng.uniform(0.3, 11.7, n), rng.uniform(0.3, 7.7, n)
    rz = 5.0 + 2.0 * (1 - np.abs(ry - 4.0) / 4.0)          # a ridge along y = 4
    gx, gy = rng.uniform(-6, 18, n), rng.uniform(-6, 14, n)
    x = np.r_[rx, gx]; y = np.r_[ry, gy]; z = np.r_[rz, np.full(n, 2.0)]
    rgb = np.full((len(x), 3), 160, np.uint8)
    return ring, x, y, z, rgb


def test_points_without_labels_keeps_the_building_and_finds_its_ground():
    """No label table: ground is the low tail inside the footprint, the building is what stands on it."""
    ring, x, y, z, _ = shed()
    inside = (x > 0) & (x < 12) & (y > 0) & (y < 8)
    zg = float(np.percentile(z[inside], bpts.GROUND_PCT))
    assert 1.9 < zg < 2.2, zg                                   # the ground, not the roof, not an outlier
    keep = inside & (z > zg + bpts.MIN_H)
    assert keep.sum() == 4000, keep.sum()                       # every roof return, no ground return
    assert z[keep].min() > 4.9


def test_lookup_labels_are_optional_but_exact_when_given():
    lab = bpts._lookup(np.arange(6), np.array([1, 3, 5]), np.array([9, 3, 0]))
    assert lab.tolist() == [-1, 9, -1, 3, -1, 0]


def test_weld_soup_to_indexed_mesh():
    """A triangle soup with shared corners welds into an indexed mesh, degenerates dropped."""
    tri = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0],
                    [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [2, 0, 0], [2, 0, 0], [2, 1, 0]])          # the third is degenerate
    V, T = b3d.weld(tri)
    assert len(V) == 6 and len(T) == 2, (len(V), len(T))       # 6 distinct corners, the degenerate dropped
    assert np.allclose(V[T].reshape(-1, 3), tri[:6])           # and each triangle keeps its winding


def test_fallback_solid_without_roofer_is_a_closed_building():
    """No solid given: the surveyed outline extruded to the measured top, roof planes on it."""
    shapely = pytest.importorskip("shapely")            # noqa: F841
    pytest.importorskip("mapbox_earcut")
    ring, x, y, z, _ = shed()
    keep = z > 4.0
    P = np.column_stack([x[keep], y[keep], z[keep]])
    V, T = b3d.fallback_solid(ring, P, z_ground=2.0)
    assert len(V) > 4 and len(T) > 4
    assert V[:, 2].min() == pytest.approx(2.0, abs=0.2)        # it stands on the ground it was given
    assert 6.0 < V[:, 2].max() < 8.0, V[:, 2].max()            # and reaches the measured ridge
    # it stays on its own footprint, to within the roof raster's 1 m cell (this branch snaps
    # the measured roof onto a grid, so a cell may straddle the outline — objects.cell_prism)
    assert -1.0 <= V[:, 0].min() and V[:, 0].max() <= 13.0
    assert -1.0 <= V[:, 1].min() and V[:, 1].max() <= 9.0


def test_facades_of_the_fallback_solid_are_usable_walls():
    """The fallback solid feeds the same facade grouping a roofer solid does."""
    pytest.importorskip("mapbox_earcut")
    from ducklidar.building import geometry
    ring, x, y, z, _ = shed()
    keep = z > 4.0
    P = np.column_stack([x[keep], y[keep], z[keep]])
    V, T = b3d.fallback_solid(ring, P, z_ground=2.0)
    n = geometry.normals(V, T)
    F = geometry.facades(V, T, n, V.mean(0))
    assert len(F) >= 4, len(F)                                  # at least the four sides of the shed
    assert all(f["s1"] > f["s0"] and f["t1"] > f["t0"] for f in F)


def test_level_is_validated():
    with pytest.raises(ValueError, match="level"):
        b3d.building3d(np.zeros((4, 2)), points=dict(), level="lots")
