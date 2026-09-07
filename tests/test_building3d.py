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


def _store(tmp_path, classified=True):
    """A shed (roof z=6 over 0..12 x 0..8) on ground z=2, a crown 8..15 m over half of it, a pier column
    and a deck at z=30 over a fifth of it — everything a footprint window catches that is not the shed."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = np.random.default_rng(1)
    def cloud(n, x0, x1, y0, y1, z0, z1, cls, nr):
        return (rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), rng.uniform(z0, z1, n),
                np.full(n, cls, np.uint8), np.full(n, nr, np.uint8))
    parts = [cloud(6000, -8, 20, -8, 16, 1.9, 2.1, 2, 1),        # ground, single returns
             cloud(6000, 0, 12, 0, 8, 5.9, 6.1, 6, 1),           # the roof
             cloud(3000, 6, 12, 0, 8, 8, 15, 5, 3),              # a crown over the east half, multi-return
             cloud(800, 5, 6, 4, 5, 6, 30, 6, 1),                # a pier through it, classed as building
             cloud(1500, 0, 12, 0, 1.0, 29.9, 30.1, 6, 1)]       # the deck: an eighth of the footprint
    x, y, z, c, nr = (np.concatenate(k) for k in zip(*parts))
    if not classified: c = np.ones_like(c)
    tb = pa.table(dict(x=x, y=y, z=z, classification=c, number_of_returns=nr, pid=np.arange(len(x)),
                       red=np.zeros(len(x), np.uint16), green=np.zeros(len(x), np.uint16), blue=np.zeros(len(x), np.uint16)))
    p = tmp_path / "t.parquet"; pq.write_table(tb, p); return p


@pytest.mark.parametrize("classified", [True, False])
def test_building_points_keeps_the_shed_not_the_crown_pier_or_deck(tmp_path, classified):
    """With the survey's classes: the roof's coverage ends the building below the pier and the deck.
    Without: the roof is the layer of single returns; the crown is multi-return and above it."""
    ring = [[0.0, 0.0], [12.0, 0.0], [12.0, 8.0], [0.0, 8.0]]
    d = bpts.building_points(ring, _store(tmp_path, classified))
    assert 1.9 < d["z_ground"] < 2.1
    assert 5.8 < np.percentile(d["z"], 98) < 6.6, np.percentile(d["z"], 98)
    assert d["z"].max() < 8.5, d["z"].max()                      # no crown, no pier, no deck
    assert len(d["z"]) > 4000                                    # but the roof itself is all there


def test_building_model_gives_a_podium_and_a_tower_their_own_walls():
    """A 20 x 10 m podium at 6 m with a 6 x 6 m tower at 20 m over its middle, plus the tower's wall
    returns spilling onto the podium roof: the outline walls stop at 6 m, the tower's at 20 m, and
    the tower's plan is the tower's, not the spill's."""
    from ducklidar import objects
    rng = np.random.default_rng(2)
    ring = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]])
    px, py = rng.uniform(0, 20, 8000), rng.uniform(0, 10, 8000); pz = np.full(8000, 6.0)
    under = (px > 7) & (px < 13) & (py > 2) & (py < 8); px, py, pz = px[~under], py[~under], pz[~under]  # no roof seen under the tower
    tx, ty = rng.uniform(7, 13, 3000), rng.uniform(2, 8, 3000); tz = np.full(3000, 20.0)
    wx, wy = rng.uniform(4, 16, 600), rng.uniform(1, 9, 600); wz = rng.uniform(7, 19, 600)      # the tower's wall returns, spilling over the podium roof
    x, y, z = np.r_[px, tx, wx], np.r_[py, ty, wy], np.r_[pz, tz, wz]
    (soup, cols, kind), = objects.building_model(ring, 0.0, 20.0, x=x, y=y, z=z)
    T = soup.reshape(-1, 3, 3)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); vert = np.abs(n[:, 2]) < 0.3 * np.linalg.norm(n, axis=1)
    W = T[vert]; top = W[:, :, 2].max(1)
    assert 5.5 < np.median(top[top < 15]) < 6.6                      # the outline walls end at the podium roof
    tall = W[top > 15]
    assert len(tall) and 19 < tall[:, :, 2].max() < 21                # the tower has walls to its own roof
    assert 5.5 < tall[:, :, 0].min() < 8.5 and 11.5 < tall[:, :, 0].max() < 14.5, (tall[:, :, 0].min(), tall[:, :, 0].max())


def test_fit_roof_planes_finds_both_slopes_of_a_gable():
    """A 20 x 12 gable, ridge along x at y = 6, 30 % slopes, 5 cm noise -> two planes, every cell snapped."""
    from ducklidar import objects
    rng = np.random.default_rng(3)
    gy, gx = np.mgrid[0:12, 0:20]
    top = 6.0 - 0.3 * np.abs(gy + 0.5 - 6.0) + rng.normal(0, 0.05, gy.shape)
    occ = np.ones_like(top, bool)
    fitted, n = objects.fit_roof_planes(top, occ, 1.0)
    assert n == 2, n
    assert np.abs(fitted - (6.0 - 0.3 * np.abs(gy + 0.5 - 6.0))).max() < 0.25   # within the fitter's own tolerance


def test_a_gable_gets_no_walls_inside_its_footprint():
    """The ridge zone stands 2 m above the eaves but on a slope, not a step: it is not a tier."""
    from ducklidar import objects
    rng = np.random.default_rng(4)
    ring = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 12.0], [0.0, 12.0]])
    x, y = rng.uniform(0, 20, 12000), rng.uniform(0, 12, 12000)
    z = 4.0 + 0.4 * (6.0 - np.abs(y - 6.0)) + rng.normal(0, 0.03, len(x))       # eaves 4 m, ridge 6.4 m
    (soup, cols, kind), = objects.building_model(ring, 0.0, 6.5, x=x, y=y, z=z)
    T = soup.reshape(-1, 3, 3)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); vert = np.abs(n[:, 2]) < 0.3 * np.linalg.norm(n, axis=1)
    W = T[vert].reshape(-1, 3)
    inside = (W[:, 0] > 0.3) & (W[:, 0] < 19.7) & (W[:, 1] > 0.3) & (W[:, 1] < 11.7)
    assert not inside.any(), W[inside][:3]
