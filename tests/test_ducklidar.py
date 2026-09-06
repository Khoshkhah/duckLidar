"""Tests on a synthetic tile, so they run in a second and need no downloaded data.

The fixture is built to contain the situations that actually caused bugs: a cell with a
vertical spread, a building with no ground beneath it, a noise point far below everything,
and a point exactly on a window edge.
"""
import json

import numpy as np
import pytest

import ducklidar as dl

GROUND_Z = 10.0
ROOF_Z = 30.0
CANOPY_Z = 18.0


@pytest.fixture(scope="module")
def tile(tmp_path_factory):
    """A 100 x 100 m tile: flat ground, one 20 x 20 m building, one tree, one noise point."""
    import laspy

    rng = np.random.default_rng(0)
    xs, ys, zs, cs = [], [], [], []

    def add(n, x0, x1, y0, y1, z, cls, jitter=0.05):
        xs.append(rng.uniform(x0, x1, n))
        ys.append(rng.uniform(y0, y1, n))
        zs.append(z + rng.normal(0, jitter, n))
        cs.append(np.full(n, cls, dtype="uint8"))

    add(40_000, 0, 100, 0, 100, GROUND_Z, 2)          # ground everywhere...
    add(6_000, 40, 60, 40, 60, ROOF_Z, 6)             # ...but a roof covers 40-60
    add(2_000, 20, 26, 70, 76, CANOPY_Z, 5)           # a tree
    add(1, 50, 50.1, 50, 50.1, -60.0, 7, jitter=0)    # one noise return, far below

    x = np.concatenate(xs) + 500_000                  # realistic UTM magnitudes
    y = np.concatenate(ys) + 5_400_000
    z = np.concatenate(zs)
    c = np.concatenate(cs)

    # the building's footprint has no ground under it — the whole point of the DTM test
    inside = ((x - 500_000 >= 40) & (x - 500_000 <= 60)
              & (y - 5_400_000 >= 40) & (y - 5_400_000 <= 60) & (c == 2))
    x, y, z, c = x[~inside], y[~inside], z[~inside], c[~inside]

    hdr = laspy.LasHeader(point_format=6, version="1.4")
    hdr.offsets, hdr.scales = [500_000, 5_400_000, 0], [0.001, 0.001, 0.001]
    las = laspy.LasData(hdr)
    las.x, las.y, las.z, las.classification = x, y, z, c
    las.intensity = rng.integers(0, 60_000, len(x)).astype("uint16")
    las.number_of_returns = np.ones(len(x), dtype="uint8")

    p = tmp_path_factory.mktemp("dl") / "tile.las"
    las.write(str(p))
    return p


BOX = (500_020.0, 5_400_020.0, 500_080.0, 5_400_080.0)


def test_box_is_a_bbox():
    assert dl.box(10, 20, 5) == (5, 15, 15, 25)


def test_read_window_drops_noise_and_crops(tile):
    pts = dl.read(tile, BOX)
    assert len(pts["x"]) > 0
    assert 7 not in np.unique(pts["classification"])          # noise gone
    assert pts["x"].min() >= BOX[0] and pts["x"].max() <= BOX[2]
    assert pts["y"].min() >= BOX[1] and pts["y"].max() <= BOX[3]
    assert pts["z"].min() > 0, "the -60 m noise point must not survive"


def test_keeping_noise_is_opt_in(tile):
    everything = dl.read(tile, (0, 0, 1e9, 1e9), drop_noise=False)
    assert 7 in np.unique(everything["classification"])


def test_field_selection_is_honoured(tile):
    pts = dl.read(tile, BOX, fields=())
    assert set(pts) == {"x", "y", "z", "classification"}


def test_empty_window_returns_empty_arrays_not_an_error(tile):
    pts = dl.read(tile, (0.0, 0.0, 1.0, 1.0))
    assert len(pts["x"]) == 0 and "z" in pts


def test_parquet_matches_the_las_exactly(tile):
    direct = dl.read(tile, BOX, parquet=False)
    dl.to_parquet(tile)
    assert dl.parquet_path(tile).exists()
    cached = dl.read(tile, BOX)
    assert len(cached["x"]) == len(direct["x"])
    assert np.allclose(np.sort(cached["z"]), np.sort(direct["z"]))


def test_to_parquet_is_idempotent(tile):
    first = dl.to_parquet(tile)
    stamp = first.stat().st_mtime_ns
    assert dl.to_parquet(tile).stat().st_mtime_ns == stamp


def test_rasterize_is_north_up(tile):
    """Row 0 must be the NORTHMOST row — anything directional depends on it."""
    pts = {"x": np.array([1.5, 1.5]), "y": np.array([9.5, 0.5]),
           "z": np.array([50.0, 10.0]), "classification": np.array([2, 2], dtype="uint8")}
    g = dl.rasterize(pts, (0.0, 0.0, 10.0, 10.0), pix=1.0)
    assert g[0, 1] == 50.0, "the northern point belongs in row 0"
    assert g[9, 1] == 10.0, "the southern point belongs in the last row"


def test_cell_edges_are_half_open(tile):
    """A point exactly on a boundary goes to the cell south/east of it, consistently.

    Pinned because it is invisible until two rasters built the same way disagree by one
    row along an edge.
    """
    bbox = (0.0, 0.0, 10.0, 10.0)
    on_edge = {"x": np.array([2.0]), "y": np.array([9.0]), "z": np.array([1.0]),
               "classification": np.array([2], dtype="uint8")}
    flat, (ny, nx) = dl.cell_index(on_edge, bbox, 1.0)
    assert (int(flat[0]) // nx, int(flat[0]) % nx) == (1, 2)


def test_surfaces_dsm_above_dtm(tile):
    pts = dl.read(tile, BOX)
    g = dl.surfaces(pts, BOX)
    both = np.isfinite(g["dsm"]) & np.isfinite(g["dtm"])
    assert (g["dsm"][both] >= g["dtm"][both] - 1e-9).all()
    assert g["roofs"] is not None and g["canopy"] is not None
    assert np.nanmax(g["roofs"]) == pytest.approx(ROOF_Z, abs=0.5)
    assert np.nanmax(g["canopy"]) == pytest.approx(CANOPY_Z, abs=0.5)


def test_dtm_is_full_of_holes_under_the_building(tile):
    """The measured behaviour this library exists to make visible."""
    pts = dl.read(tile, BOX)
    g = dl.surfaces(pts, BOX)
    ny, nx = dl.shape_for(BOX, 1.0)
    # the building sits at 40-60 m, i.e. 20-40 cells into this window
    core = g["dtm"][ny - 40:ny - 20, 20:40]
    assert np.isnan(core).mean() > 0.9, "no ground returns under a roof"
    assert np.isfinite(g["dtm"]).mean() < 1.0


def test_void_report_finds_the_building_void(tile):
    pytest.importorskip("scipy")
    pts = dl.read(tile, BOX)
    g = dl.surfaces(pts, BOX)
    r = dl.void_report(g["dtm"], pix=1.0)
    assert 0.0 < r["filled"] < 1.0
    assert r["largest_cells"] > 100
    assert r["evidence_max"] > 5.0, "the middle of a 20 m building is far from any ground"


def test_distance_to_evidence_is_zero_where_measured(tile):
    pytest.importorskip("scipy")
    pts = dl.read(tile, BOX)
    g = dl.surfaces(pts, BOX)
    d = dl.distance_to_evidence(g["dtm"], pix=1.0)
    assert (d[np.isfinite(g["dtm"])] == 0).all()
    assert d[~np.isfinite(g["dtm"])].min() > 0


def test_census_reports_separability(tile):
    pts = dl.read(tile, BOX)
    rows, can_separate = dl.census(pts["classification"])
    assert can_separate, "this tile has both vegetation and building classes"
    assert abs(sum(r[3] for r in rows) - 1.0) < 1e-9
    names = {r[1] for r in rows}
    assert {"ground", "building", "high vegetation"} <= names


def test_census_says_no_when_a_class_is_missing():
    rows, can_separate = dl.census(np.array([1, 1, 2, 2, 2], dtype="uint8"))
    assert not can_separate


def test_flight_dates_decode_adjusted_gps_time():
    # 2022-09-08 00:11 UTC, the Vancouver tile's first return
    first, last = dl.flight_dates(np.array([346631078.37, 346793921.69]))
    assert first.year == 2022 and first.month == 9 and first.day == 8
    assert (last - first).total_seconds() == pytest.approx(162843.3, abs=1)


# --- intervals: a result is a value and how much to trust it, never a bare number ---

def test_interval_brackets_the_value(tile):
    pts = dl.read(tile, BOX)
    e = dl.rasterize(pts, BOX, interval=True)
    ok = np.isfinite(e.value)
    assert (e.lo[ok] <= e.value[ok] + 1e-9).all()
    assert (e.hi[ok] >= e.value[ok] - 1e-9).all()
    assert (e.width[ok] >= 0).all()


def test_interval_is_opt_in(tile):
    """Without interval=True the old bare array comes back, so callers do not break."""
    pts = dl.read(tile, BOX)
    assert isinstance(dl.rasterize(pts, BOX), np.ndarray)
    assert isinstance(dl.rasterize(pts, BOX, interval=True), dl.Estimate)


def test_disagreeing_passes_widen_the_interval():
    """Two flight lines that saw the same cell differently must produce a wide band."""
    pts = {"x": np.array([0.5, 0.5, 0.5]), "y": np.array([0.5, 0.5, 0.5]),
           "z": np.array([10.0, 10.1, 30.0]),
           "classification": np.array([2, 2, 2], dtype="uint8"),
           "point_source_id": np.array([1, 1, 2], dtype="uint16")}
    e = dl.rasterize(pts, (0.0, 0.0, 1.0, 1.0), pix=1.0, interval=True)
    assert e.value[0, 0] == 30.0
    assert e.lo[0, 0] == pytest.approx(10.1)      # what the other pass topped out at
    assert e.width[0, 0] == pytest.approx(19.9)
    assert dl.METHOD[int(e.method[0, 0])] == "passes"


def test_edge_strength_is_an_and_not_an_or():
    """One view seeing structure must not mark the cell — that is the whole point of the product.

    Cell A: both views span 0->10 m.  Cell B: only view 2 does, view 1 sees a flat surface.
    """
    x = np.array([0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5])
    pts = {"x": x, "y": np.full(8, 0.5),
           "z": np.array([0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]),
           "point_source_id": np.array([1, 1, 2, 2, 1, 1, 2, 2], dtype="uint16")}
    s = dl.edge_strength(pts, (0.0, 0.0, 2.0, 1.0), 1.0)
    assert s[0, 0] > 5.0                      # both views span it
    assert s[0, 1] == 0.0, "one look seeing flat ground must zero the product, not soften it"
    # and the floor is the opt-in that softens exactly that
    assert dl.edge_strength(pts, (0.0, 0.0, 2.0, 1.0), 1.0, floor=0.01)[0, 1] > 0.0


def test_instances_separates_two_towers(tmp_path):
    """Two 6 m blocks with a gap must come back as two labels, and the ground as none."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)
    xs, ys, zs, src = [], [], [], []
    for view in (1, 2):
        gx, gy = np.meshgrid(np.arange(0.5, 40, .5), np.arange(0.5, 40, .5))
        xs.append(gx.ravel()); ys.append(gy.ravel())
        z = np.zeros(gx.size)
        for x0 in (5, 25):                          # two 10 m blocks, 10 m apart
            on = ((gx.ravel() > x0) & (gx.ravel() < x0 + 10)
                  & (gy.ravel() > 10) & (gy.ravel() < 20))
            z[on] = 6.0
        zs.append(z); src.append(np.full(gx.size, view, dtype="uint16"))
    pts = {"x": np.concatenate(xs), "y": np.concatenate(ys), "z": np.concatenate(zs),
           "point_source_id": np.concatenate(src)}
    lab, edges, ndsm = dl.instances(pts, (0.0, 0.0, 40.0, 40.0), 1.0, min_area=20.0)
    assert lab.max() == 2, f"expected two instances, got {lab.max()}"
    assert lab[np.nan_to_num(ndsm) < 1].max() == 0          # nothing labelled on the ground
    assert edges.any()                                       # the walls were found


def test_at_extremum_reports_the_winner_not_the_maximum():
    """The point of the helper: whose value won, which is not the largest value of the field."""
    pts = {"x": np.array([0.5, 0.5, 0.5]), "y": np.array([0.5, 0.5, 0.5]),
           "z": np.array([10.0, 30.0, 12.0]),
           "point_source_id": np.array([7, 3, 9], dtype="uint16")}
    B = (0.0, 0.0, 1.0, 1.0)
    assert dl.at_extremum(pts, B, 1.0, "point_source_id")[0, 0] == 3          # z=30 won
    assert dl.at_extremum(pts, B, 1.0, "point_source_id", how="min")[0, 0] == 7
    assert np.isnan(dl.at_extremum(pts, B, 1.0, "point_source_id",
                                   mask=np.zeros(3, bool))[0, 0])             # empty cell


def test_single_return_is_zero_width_and_says_so(tile):
    """A tight interval on one return is ignorance, not agreement — method must say 'none'."""
    pts = {"x": np.array([0.5]), "y": np.array([0.5]), "z": np.array([7.0]),
           "classification": np.array([2], dtype="uint8"),
           "point_source_id": np.array([1], dtype="uint16")}
    e = dl.rasterize(pts, (0.0, 0.0, 1.0, 1.0), pix=1.0, interval=True)
    assert e.width[0, 0] == 0.0
    assert e.n[0, 0] == 1
    assert dl.METHOD[int(e.method[0, 0])] == "none"


def test_interval_falls_back_to_spread_without_pass_ids():
    """One pass, several returns: the band is how far max sits above the bulk."""
    pts = {"x": np.full(11, 0.5), "y": np.full(11, 0.5),
           "z": np.array([1.0] * 10 + [9.0]),
           "classification": np.full(11, 2, dtype="uint8")}
    e = dl.rasterize(pts, (0.0, 0.0, 1.0, 1.0), pix=1.0, interval=True)
    assert e.value[0, 0] == 9.0
    assert e.lo[0, 0] == pytest.approx(1.0)        # the single high return is not corroborated
    assert dl.METHOD[int(e.method[0, 0])] == "spread"


def test_blank_cells_have_no_interval(tile):
    pts = dl.read(tile, BOX)
    e = dl.rasterize(pts, BOX, mask=pts["classification"] == 6, interval=True)
    blank = ~np.isfinite(e.value)
    assert blank.any()
    assert np.isnan(e.lo[blank]).all() and np.isnan(e.hi[blank]).all()


def test_trusted_narrows_to_tight_cells(tile):
    pts = dl.read(tile, BOX)
    e = dl.rasterize(pts, BOX, interval=True)
    assert e.trusted(max_width=1e9).sum() == np.isfinite(e.value).sum()
    assert e.trusted(max_width=0.0).sum() <= e.trusted(max_width=0.5).sum()


def test_summary_shares_add_up(tile):
    pts = dl.read(tile, BOX)
    s = dl.rasterize(pts, BOX, interval=True).summary()
    assert s["by_none"] + s["by_spread"] + s["by_passes"] == pytest.approx(1.0)
    assert 0 <= s["filled"] <= 1


def test_surfaces_can_return_estimates(tile):
    pts = dl.read(tile, BOX)
    g = dl.surfaces(pts, BOX, interval=True)
    assert all(isinstance(v, dl.Estimate) for v in g.values() if v is not None)
    assert dl.void_report(g["dtm"])["filled"] < 1.0     # accepts an Estimate too


def test_field_guide_ranks_and_marks(tile, capsys):
    """Fields are not a list of equals — the guide must say which matter and which are dead."""
    dl.field_guide(tile)
    out = capsys.readouterr().out
    for tier in ("CORE", "SITUATIONAL", "DIAGNOSTIC", "USUALLY DEAD"):
        assert tier in out
    assert out.index("CORE") < out.index("USUALLY DEAD"), "ranked, not alphabetical"
    # x/y/z are stored as X/Y/Z but must still appear, and be marked present
    core = out[out.index("CORE"):out.index("SITUATIONAL")]
    for name in ("x", "y", "z", "classification"):
        assert f"[*] {name}" in core, f"{name} should be listed as populated under CORE"
    assert "canopy CAN be separated" in out


def test_every_use_has_a_tier_we_know():
    assert {t for t, _ in dl.USES.values()} <= set(dl.fields.TIER_ORDER)


# --- plotting: static views, so they must work headless ---

def test_plot_views_run_headless(tile):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = dl.read(tile, BOX)
    ax = dl.plan(pts, by="classification")
    assert ax.collections, "plan drew nothing"
    mid = (BOX[1] + BOX[3]) / 2
    ax = dl.section(pts, mid, width=4.0)
    assert ax.collections, "section drew nothing"
    plt.close("all")


def test_section_slices_thin(tile):
    """A section must take a slab, not the whole cloud — else everything overlaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = dl.read(tile, BOX)
    mid = (BOX[1] + BOX[3]) / 2
    ax = dl.section(pts, mid, width=2.0)
    drawn = ax.collections[0].get_offsets().shape[0]
    assert drawn < len(pts["x"]), "the slice kept every point"
    assert drawn > 0
    plt.close("all")


def test_plot_colours_by_an_arbitrary_array(tile):
    """Colouring by a segmentation result is the whole point — it need not be a field."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = dl.read(tile, BOX)
    fake_instances = (pts["z"] > pts["z"].mean()).astype(int)
    ax = dl.plan(pts, by=fake_instances, cmap="tab20")
    assert ax.collections
    plt.close("all")


def test_ground_and_vegetation_get_different_colours():
    assert dl.CLASS_COLOURS[2] != dl.CLASS_COLOURS[5] != dl.CLASS_COLOURS[6]


def test_passes_win_outright_over_spread():
    """A multi-pass cell uses ONLY the flight-line range, not the union with within-cell spread.

    Between-pass disagreement measures whether the measurement repeats; within-cell spread
    measures whether the surface is flat, which is not an uncertainty. A wall sampled
    consistently by two passes is not uncertain merely because it spans 20 m vertically.
    Combining them also forces the expensive per-cell sort to run on every cell.
    """
    pts = {"x": np.full(6, 0.5), "y": np.full(6, 0.5),
           "z": np.array([1.0, 1.0, 20.0, 1.1, 1.1, 20.0]),   # huge spread, both passes agree
           "classification": np.full(6, 2, dtype="uint8"),
           "point_source_id": np.array([1, 1, 1, 2, 2, 2], dtype="uint16")}
    e = dl.rasterize(pts, (0.0, 0.0, 1.0, 1.0), pix=1.0, interval=True)
    assert dl.METHOD[int(e.method[0, 0])] == "passes"
    assert e.width[0, 0] == pytest.approx(0.0, abs=1e-9), \
        "both passes topped out at 20.0, so the interval is zero-width"


# --- surface_kind: what the passes collectively imply is in a cell ---
#
# width alone cannot separate these, which is the point. A flat roof with an aerial above it
# and the edge of a roof both make the per-pass MAXIMA disagree; only the per-pass MEDIANS
# tell them apart. Each case below is a hand-built cell, one metre square.

def _cell(z, src, bbox=(0.0, 0.0, 1.0, 1.0)):
    z = np.asarray(z, dtype=float)
    return {"x": np.full(len(z), 0.5), "y": np.full(len(z), 0.5), "z": z,
            "classification": np.full(len(z), 2, dtype="uint8"),
            "point_source_id": np.asarray(src, dtype="uint16")}


def _kind(z, src):
    e = dl.rasterize(_cell(z, src), (0.0, 0.0, 1.0, 1.0), pix=1.0, interval=True)
    return dl.SURFACE[int(e.surface_kind()[0, 0])], e


def test_flat_when_both_passes_agree_throughout():
    kind, e = _kind([10.0, 10.02, 10.01, 10.0, 9.99, 10.01], [1, 1, 1, 2, 2, 2])
    assert kind == "flat"
    assert e.width[0, 0] < 0.1


def test_clipped_when_tops_differ_but_the_bulk_agrees():
    """A flat roof with an aerial on it: one pass caught the aerial, both saw the roof."""
    kind, e = _kind([10.0, 10.0, 10.0, 25.0,          # pass 1 clipped something tall
                     10.0, 10.0, 10.0, 10.0], [1, 1, 1, 1, 2, 2, 2, 2])
    assert kind == "clipped", "bulk agrees, so this is one surface with clutter above it"
    assert e.bulk_gap[0, 0] < 0.5
    assert e.width[0, 0] > 5, "the maxima still disagree wildly — which is why width alone fails"


def test_stepped_when_the_bulk_itself_disagrees():
    """A real step: one pass sat on the roof, the other on the ground."""
    kind, e = _kind([30.0, 30.1, 29.9, 30.0,
                     5.0, 5.1, 4.9, 5.0], [1, 1, 1, 1, 2, 2, 2, 2])
    assert kind == "stepped"
    assert e.bulk_gap[0, 0] > 20


def test_undersampled_when_one_pass_barely_sampled_the_cell():
    """One stray return from a thin pass is not evidence of a second surface."""
    kind, e = _kind([10.0] * 40 + [30.0], [1] * 40 + [2])
    assert kind == "undersampled"
    assert e.balance[0, 0] < 0.2


def test_unknown_without_repeat_passes():
    kind, _ = _kind([10.0, 10.1, 25.0], [1, 1, 1])
    assert kind == "unknown", "one pass cannot imply anything about a second surface"


def test_clipped_and_stepped_are_not_separable_by_width_alone():
    """The reason the extra evidence exists — width ranks them the wrong way round."""
    _, clipped = _kind([10, 10, 10, 25, 10, 10, 10, 10], [1, 1, 1, 1, 2, 2, 2, 2])
    _, stepped = _kind([30, 30.1, 29.9, 30, 28, 28.1, 27.9, 28], [1, 1, 1, 1, 2, 2, 2, 2])
    assert clipped.width[0, 0] > stepped.width[0, 0], \
        "the clipped cell looks worse by width, though its surface is the solid one"
    assert clipped.bulk_gap[0, 0] < stepped.bulk_gap[0, 0], "bulk_gap gets the order right"


def test_extra_evidence_is_optional_on_the_dataclass():
    """Estimate must still construct without it, so nothing downstream breaks."""
    z = np.zeros((2, 2))
    e = dl.Estimate(z, z, z, z.astype(int), z.astype(int))
    assert e.bulk_gap is None
    assert (e.surface_kind() == 0).all()


def test_summary_reports_the_surface_breakdown(tile):
    pts = dl.read(tile, BOX)
    s = dl.rasterize(pts, BOX, interval=True).summary()
    shares = [s[f"is_{n}"] for n in dl.SURFACE.values()]
    assert sum(shares) == pytest.approx(1.0)


def test_look_azimuth_recovers_the_beam_direction():
    """A line flying due east sweeps north and south; the sign of scan_angle picks which."""
    n = 50
    pts = {"x": np.linspace(0, 100, n), "y": np.full(n, 50.0), "z": np.zeros(n),
           "gps_time": np.linspace(0, 10, n),
           "scan_angle": np.where(np.arange(n) % 2, 2000, -2000).astype("int16"),
           "point_source_id": np.ones(n, dtype="uint16")}
    az = dl.look_azimuth(pts)
    assert az[1] == pytest.approx(180.0, abs=1.0)     # heading 90 + 90 -> looks south
    assert az[0] == pytest.approx(0.0, abs=1.0)       # heading 90 - 90 -> looks north


def test_viewed_from_is_opposite_the_beam():
    """The side the scanner was on, not the way the beam went — they differ by 180°.

    A line flying due east sweeps left (north) and right (south); a beam travelling south was
    fired from the north, so it lights the north-facing side.
    """
    n = 40
    pts = {"x": np.linspace(0, 100, n), "y": np.full(n, 50.0), "z": np.zeros(n),
           "gps_time": np.linspace(0, 10, n),
           "scan_angle": np.where(np.arange(n) % 2, 2000, -2000).astype("int16"),
           "point_source_id": np.ones(n, dtype="uint16")}
    travel, side = dl.look_azimuth(pts), dl.viewed_from(pts)
    assert travel[1] == pytest.approx(180.0, abs=1.0)     # beam heads south
    assert side[1] == pytest.approx(0.0, abs=1.0)         # so the scanner was north
    assert dl.viewed_from(pts, quadrant=True)[1] == 0     # COMPASS[0] == "north"
    assert dl.COMPASS[dl.viewed_from(pts, quadrant=True)[0]] == "south"


def test_viewed_from_parts_merges_nearest_bundle():
    """Looks at 0°, 180° and 315°: parts=2 must put 315° with 0° (45° apart), never with 180°."""
    n = 60
    t = np.linspace(0, 10, n)
    l1 = {"x": t * 10, "y": np.zeros(n), "gps_time": t,          # flies east: looks 0° and 180°
          "scan_angle": np.where(np.arange(n) % 2, 2000, -2000).astype("int16"),
          "point_source_id": np.ones(n, dtype="uint16")}
    l2 = {"x": t * 7, "y": t * 7, "gps_time": t,                 # heading 45°, one side: 315°
          "scan_angle": np.full(n, 2000, dtype="int16"),
          "point_source_id": np.full(n, 2, dtype="uint16")}
    pts = {k: np.concatenate([l1[k], l2[k]]) for k in l1}
    pts["z"] = np.zeros(2 * n)
    az, code = dl.viewed_from(pts), dl.viewed_from(pts, parts=2)
    near = lambda a: np.abs(((az - a + 180) % 360) - 180) < 10
    assert set(code[near(315)]) == set(code[near(0)])
    assert set(code[near(180)]).isdisjoint(set(code[near(315)]))
    assert set(code) == {0, 1}


def test_hysteresis_keeps_connected_weak_and_drops_isolated_weak():
    """Canny's linking assumption: a weak flank attached to a strong corner is boundary;
    the same weak value floating alone is noise."""
    pytest.importorskip("scipy")
    S = np.zeros((9, 9))
    S[4, 2] = 1.0                       # strong corner
    S[4, 3:7] = 0.2                     # weak flank, connected to it
    S[0, 8] = 0.2                       # same value, isolated
    b = dl.hysteresis(S, hi=0.5, lo=0.15)
    assert b[4, 2] and b[4, 5], "connected weak cells must survive"
    assert not b[0, 8], "isolated weak cells must not"
    assert not dl.hysteresis(np.full((5, 5), np.nan), 0.5, 0.15).any()


def test_watershed_lines_close_what_hysteresis_cannot():
    """Two tall blocks joined by a 2-cell gap in a weak ridge: hysteresis leaves them one
    component; the watershed line completes the contour and separates them."""
    pytest.importorskip("skimage")
    S = np.zeros((20, 30))
    ndsm = np.zeros((20, 30))
    ndsm[4:16, 4:26] = 6.0                     # one tall slab...
    S[4:16, [4, 25]] = 1.0                     # strong outer rim, left and right
    S[[4, 15], 4:26] = 1.0                     # strong outer rim, top and bottom
    S[4:16, 14] = 0.3                          # ...with a weak internal ridge splitting it
    S[9:11, 14] = 0.02                         # and a 2-cell hole below every threshold
    hyst_only = dl.boundary(S, ndsm, close=False)
    full = dl.boundary(S, ndsm, close=True)
    from scipy import ndimage as ndi
    tall = ndsm > 2.0
    n_open = ndi.label(tall & ~hyst_only)[1]
    n_closed = ndi.label(tall & ~full)[1]
    assert n_open == 1, "the hole must defeat hysteresis alone for this test to mean anything"
    assert n_closed >= 2, "the watershed line must separate the two cores"


def test_ground_filter_veto_rejects_non_last_returns():
    """A mid-canopy echo at GROUND HEIGHT must still be rejected — it is not a last return,
    so it cannot be ground no matter where it sits."""
    pytest.importorskip("CSF")
    rng = np.random.default_rng(1)
    n = 4000
    pts = {"x": rng.uniform(0, 50, n), "y": rng.uniform(0, 50, n),
           "z": rng.normal(0.0, 0.03, n),
           "return_number": np.ones(n, dtype="uint8"),
           "number_of_returns": np.ones(n, dtype="uint8")}
    # one impostor: at ground height, but echo 1 of 3
    pts["x"][0], pts["y"][0], pts["z"][0] = 25.0, 25.0, 0.0
    pts["number_of_returns"][0] = 3
    g = dl.ground_filter(pts)
    assert not g[0], "a non-last return can never be labelled ground"
    assert g[1:].mean() > 0.9, "the flat field itself must still be found"
    assert dl.ground_filter(pts, last_only=False)[0], "without the veto the impostor passes"


def test_local_boundary_needs_the_floor_and_the_rank_test():
    """Both terms are load-bearing: a faint ridge below the floor is noise (killed by t2),
    a strong ridge is boundary whatever its neighbours do (rescued by t1)."""
    pytest.importorskip("scipy")
    S = np.full((40, 40), 1.0)
    S[:, 20:] = 3.0                   # half the field is busy, so t2 (p75) lands at 3.0
    S[10, 2:18] = 1.4                 # a local bump on the quiet side, but under the floor
    S[30, 2:18] = 9.0                 # unambiguous in absolute terms
    b = dl.local_boundary(S, size=9, t1_pct=99, t2_pct=75)
    assert b[30, 10], "an absolutely strong cell must be a boundary whatever its neighbours"
    assert not b[10, 10], "a sub-floor bump is noise, however it compares locally"


def test_dem_fills_holes_and_says_how_far_the_guess_reached():
    """A synthetic slope with a building-shaped hole punched in it.

    The interpolator has to recover a plane it cannot see, and `distance` has to report honestly
    how far from evidence each recovered cell is — that layer is the whole point of the module,
    so it gets asserted rather than assumed.
    """
    ny, nx = 40, 40
    yy, xx = np.mgrid[0:ny, 0:nx]
    truth = 0.05 * xx + 0.02 * yy          # a gentle plane, which `linear` should nail
    grid = truth.astype(float).copy()
    grid[14:26, 14:26] = np.nan            # a 12 x 12 m roof

    filled, dist = dl.fill(grid, 1.0, method="linear")
    hole = np.isnan(grid)
    assert np.isfinite(filled).all(), "a filled grid has no holes left"
    assert np.abs(filled[hole] - truth[hole]).max() < 1e-6, \
        "a plane between measured cells is exactly what linear interpolation returns"
    assert dist[~hole].max() == 0, "a measured cell is zero metres from evidence"
    # the centre of a 12-wide hole is 6 cells from the rim in the nearest direction
    assert 5.0 <= dist[hole].max() <= 9.0, f"centre distance {dist[hole].max()}"

    # nearest cannot represent a slope: it copies the rim inward and flat-spots the middle
    flat, _ = dl.fill(grid, 1.0, method="nearest")
    assert np.abs(flat[hole] - truth[hole]).max() > 0.1

    # and refusing to guess is available
    capped, _ = dl.fill(grid, 1.0, method="linear", max_gap=3.0)
    assert np.isnan(capped).any(), "max_gap must leave the deep middle unmeasured"
    assert np.isfinite(capped[~hole]).all(), "...without discarding measurements"


def test_auto_fill_switches_method_at_the_crossover():
    """`auto` is the default because neither single method wins everywhere.

    [measured] on real ground returns held back in building-shaped blocks, `nearest` has the
    lower error within about 5 m of evidence and `idw` beyond it — 0.170 against 0.175 at 2-5 m,
    0.481 against 0.509 past 10 m. So the choice belongs to the cell, not to the call, and the
    test that matters is that each half of a hole really is filled by the method that won it.
    """
    ny, nx = 60, 60
    yy, xx = np.mgrid[0:ny, 0:nx]
    # a SLOPE, not a plateau: on flat ground nearest and idw agree exactly and the test proves
    # nothing. The rim runs 11 m on one side to 16 m on the other, so copying the nearest rim
    # value and blending across the hole give visibly different answers in the middle.
    grid = 10.0 + 0.1 * xx
    grid[10:50, 10:50] = np.nan                 # a 40 m hole: middle is ~20 m from any evidence

    filled, dist = dl.fill(grid, 1.0, method="auto")
    hole = np.isnan(grid)
    assert np.isfinite(filled).all()

    near = hole & (dist < dl.dem_mod.CROSSOVER)
    far = hole & (dist >= dl.dem_mod.CROSSOVER)
    assert near.any() and far.any(), "the fixture must exercise both sides of the crossover"

    nearest, _ = dl.fill(grid, 1.0, method="nearest")
    idw, _ = dl.fill(grid, 1.0, method="idw")
    assert np.allclose(filled[near], nearest[near]), "inside the crossover, auto is nearest"
    assert np.allclose(filled[far], idw[far]), "beyond it, auto is idw"
    # and the two really do differ, or the assertions above prove nothing
    assert not np.allclose(nearest[far], idw[far])


def test_fill_defaults_to_auto():
    """The default changed from `linear`, so it is pinned rather than left to drift."""
    grid = np.full((30, 30), 5.0)
    grid[10:20, 10:20] = np.nan
    assert np.allclose(dl.fill(grid, 1.0)[0], dl.fill(grid, 1.0, method="auto")[0])


def test_sensor_track_recovers_a_known_aircraft_position():
    """Put the sensor somewhere, fire pulses through it, and see if it comes back.

    Each pulse gives two returns on one ray from the sensor, so the ray is recoverable and the
    rays meet where the aircraft is. The mirror sweeps +/-20 degrees across-track, which is what
    makes the rays fan out; the second half of the test removes that fan and shows the solve
    degrade, because that degradation is the whole reason `sensor_track` reports conditioning.
    """
    rng = np.random.default_rng(11)
    S = np.array([100.0, 200.0, 1400.0])

    def cloud(angles):
        n = len(angles)
        # ray from the sensor: across-track tilt only, ground near z=0
        dx = np.tan(np.radians(angles)) * S[2]
        gx, gy = S[0] + dx, S[1] + rng.normal(0, 1.0, n)
        first_z = rng.uniform(5.0, 25.0, n)          # a canopy hit on the way down
        f = S + (np.column_stack([gx, gy, np.zeros(n)]) - S) * \
            ((S[2] - first_z) / S[2])[:, None]
        l = np.column_stack([gx, gy, np.zeros(n)])   # the ground hit
        x = np.empty(2 * n); y = np.empty(2 * n); z = np.empty(2 * n)
        x[0::2], y[0::2], z[0::2] = f[:, 0], f[:, 1], f[:, 2]
        x[1::2], y[1::2], z[1::2] = l[:, 0], l[:, 1], l[:, 2]
        t = np.repeat(np.linspace(0.0, 0.4, n), 2)
        return {"x": x, "y": y, "z": z, "gps_time": t,
                "return_number": np.tile([1, 2], n).astype("uint8"),
                "number_of_returns": np.full(2 * n, 2, dtype="uint8")}

    wide = cloud(rng.uniform(-20, 20, 4000))
    t, pos, cond = dl.sensor_track(wide, window=1.0, min_pulses=100)
    assert len(t) == 1
    assert np.linalg.norm(pos[0] - S) < 1.0, f"recovered {pos[0]}, wanted {S}"

    narrow = cloud(rng.uniform(18, 20, 4000))       # only the far swath edge
    _, pos2, cond2 = dl.sensor_track(narrow, window=1.0, min_pulses=100)
    assert cond2[0] < cond[0], "a narrow fan must report worse conditioning"
    assert np.linalg.norm(pos2[0] - S) > np.linalg.norm(pos[0] - S), \
        "and it must actually be less accurate — that is what the number is for"


def test_bridge_deck_needs_both_a_void_and_a_hard_surface():
    """A deck is a hard top with a gap under it. A roof has no gap; a crown is not hard.

    Three synthetic strips, 40 x 40 m each so every one clears `min_cells`:
      deck   — returns at 30 m and again at 0 m in the SAME cells, single-return
      roof   — returns at 30 m only, single-return
      crown  — returns at 30 m and 0 m, but every pulse split

    Only the first is a deck, and it is the pair of tests that says so: the void alone would
    take the crown too, the echo ratio alone would take the roof.
    """
    rng = np.random.default_rng(7)
    n = 20000                 # ~12 returns per 1 m cell per layer; echo_ratio needs >= 4
    xs, ys, zs, nret = [], [], [], []
    for i, (two_layer, split) in enumerate([(True, False), (False, False), (True, True)]):
        x0 = i * 50.0
        x = rng.uniform(x0, x0 + 40, n)
        y = rng.uniform(0, 40, n)
        xs += [x, x] if two_layer else [x]
        ys += [y, y] if two_layer else [y]
        zs += [np.full(n, 30.0), np.zeros(n)] if two_layer else [np.full(n, 30.0)]
        k = 2 if two_layer else 1
        nret += [np.full(n * k, 3 if split else 1, dtype="uint8")]
    pts = {"x": np.concatenate(xs), "y": np.concatenate(ys), "z": np.concatenate(zs),
           "return_number": np.ones(sum(len(a) for a in xs), dtype="uint8"),
           "number_of_returns": np.concatenate(nret)}
    m = dl.bridge_deck(pts, (0.0, 0.0, 150.0, 40.0), 1.0)

    x = pts["x"]
    assert m[(x < 40) & (pts["z"] > 15)].mean() > 0.99, "the deck top is deck"
    assert not m[(x < 40) & (pts["z"] < 15)].any(), "what the deck spans is not deck"
    assert not m[(x > 50) & (x < 90)].any(), "a roof has no void under it"
    assert not m[x > 100].any(), "a crown has a void but is not a hard surface"


def test_echo_ratio_separates_porous_from_solid():
    """A pulse splits in foliage and stops dead on a roof — that is the whole measurement."""
    n = 600
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.uniform(0, 10, n), rng.uniform(20, 30, n)])
    y = np.concatenate([rng.uniform(0, 10, n), rng.uniform(0, 10, n)])
    nret = np.concatenate([np.full(n, 3), np.full(n, 1)]).astype("uint8")
    pts = {"x": x, "y": y, "z": np.zeros(2 * n),
           "return_number": np.ones(2 * n, dtype="uint8"), "number_of_returns": nret}
    r = dl.echo_ratio(pts, (0.0, 0.0, 30.0, 10.0), 10.0)
    assert r[0, 0] == pytest.approx(1.0), "every pulse split — porous"
    assert r[0, 2] == pytest.approx(0.0), "every pulse stopped dead — solid"
    assert np.isnan(r[0, 1]), "an empty cell has no ratio, not a zero"


def test_building_level_is_not_the_minimum():
    """A pad with a hole in it: the datum must be the pad, not the interpolator's lowest dip.

    [measured] over 61 real footprints, `min` sits 0.341 m below the measured perimeter on level
    sites and 1.540 m below on slopes, because a minimum over hundreds of cells samples the
    lowest excursion of a guess. This fixture reproduces that in miniature.
    """
    ny, nx = 40, 40
    grid = np.full((ny, nx), 20.0)
    grid[8:32, 8:32] = np.nan                  # the building footprint, unmeasured
    grid[0, :] = 18.0                          # a ditch along one edge, well outside the pad
    d = {"dtm": grid, "filled": dl.fill(grid, 1.0)[0]}
    foot = np.zeros((ny, nx), bool)
    foot[8:32, 8:32] = True

    r = dl.building_level(d, foot)
    assert r["statistic"] == "p25", "a level ring must use the unbiased statistic"
    assert abs(r["level"] - 20.0) < 0.2, f"datum {r['level']} should sit on the 20 m pad"
    assert r["level"] > d["filled"][foot].min(), "and must not be the minimum, which the ditch drags down"
    assert r["ring_cells"] > 0 and abs(r["ring_level"] - 20.0) < 0.2


def test_building_level_falls_back_to_median_on_a_slope():
    """Once the perimeter genuinely tilts there is no single level, and p25 stops being safe."""
    ny, nx = 40, 40
    yy, xx = np.mgrid[0:ny, 0:nx]
    grid = 20.0 + 0.20 * xx                    # 20 cm per metre: unmistakably a hillside
    grid[8:32, 8:32] = np.nan
    d = {"dtm": grid, "filled": dl.fill(grid, 1.0)[0]}
    foot = np.zeros((ny, nx), bool)
    foot[8:32, 8:32] = True

    r = dl.building_level(d, foot)
    assert r["statistic"] == "median", f"tilt {r['tilt']:.3f} should trip the fallback"
    assert r["tilt"] > 0.02
    assert abs(r["level"] - np.median(d["filled"][foot])) < 1e-9


def test_footprint_comes_from_the_object_s_own_points(tile):
    """A datum needs a footprint, and a footprint comes from the points carrying that object id.

    Anything else assumes somebody already produced a raster of the building, which is the step
    the pipeline is meant to produce rather than consume.
    """
    pts = dl.read(str(tile), BOX)
    roof = pts["classification"] == 6                 # stands in for object_id == n
    foot = dl.footprint(pts, BOX, roof, 1.0)
    assert foot.dtype == bool and foot.shape == dl.shape_for(BOX, 1.0)
    assert foot.any(), "the fixture's building must occupy some cells"
    assert foot.sum() < foot.size, "and must not cover the whole box"

    # the cells it claims are exactly the cells its points fall in
    flat, _ = dl.cell_index({k: np.asarray(pts[k])[roof] for k in "xyz"}, BOX, 1.0)
    assert set(np.flatnonzero(foot.ravel())) == set(np.unique(flat))

    with pytest.raises(ValueError):
        dl.footprint(pts, BOX, np.zeros(len(roof), bool), 1.0)


def test_wall_occupancy_draws_evidence_and_keeps_holes():
    """A wall cell exists only where returns lie near the wall plane; an unseen
    side stays open. This is the pavilion lesson (2026-08-01): synthetic walls
    buried the openness that made the structure what it was."""
    from ducklidar.scene import wall_occupancy

    ring = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    # returns on the south edge only (y=0), a 10 m x 3 m curtain with a
    # 2 m-wide gap (a doorway) between x=4 and x=6
    xs = np.concatenate([np.arange(0.1, 4.0, 0.2), np.arange(6.0, 9.9, 0.2)])
    X = np.repeat(xs, 6)
    Z = np.tile(np.arange(0.3, 3.0, 0.45), len(xs))
    Y = np.zeros_like(X)
    out = wall_occupancy(ring, X, Y, Z, 0.0, 3.0)
    assert out is not None
    soup, cols = out
    assert len(soup) % 6 == 0 and len(soup) == len(cols)
    # every cell hugs the south edge; the other three sides stay open
    assert soup[:, 1].max() < 0.5, "no wall invented on unseen sides"
    # the doorway gap carries no cells
    mid = (soup[::6, 0] > 4.6) & (soup[::6, 0] < 5.2)
    assert not mid.any(), "the hole must stay a hole"
    # measured colour: cells wear their contributing returns' colour
    rgb = np.tile([200, 40, 40], (len(X), 1))
    soup2, cols2 = wall_occupancy(ring, X, Y, Z, 0.0, 3.0, rgb=rgb)
    assert (np.abs(cols2.astype(int) - [200, 40, 40]).max() == 0)


def test_canopy_mesh_builds_a_gable_only_when_the_points_pitch():
    """The pavilion rule (2026-08-01): ridge P98 over outer-band eave P85 by
    more than 1 m means two measured roof planes on the rotated rectangle;
    level returns get a slab; below either roof no wall is invented — a post
    appears only where the laser saw one."""
    shapely = pytest.importorskip("shapely")
    from ducklidar.scene import canopy_mesh

    poly = shapely.box(0.0, 0.0, 12.0, 6.0)
    gx, gy = np.meshgrid(np.arange(0.1, 12.0, 0.2), np.arange(0.1, 6.0, 0.2))
    X, Y = gx.ravel(), gy.ravel()

    # pitched: eaves near 3 m rising to a 5 m ridge along the long axis
    Z = 5.0 - np.abs(Y - 3.0) * (2.0 / 3.0)
    parts = canopy_mesh(poly, X, Y, Z, 0.0)
    roof, _ = parts[0]
    assert len(roof) == 12, "a gable is exactly two roof planes"
    assert roof[:, 2].max() > 4.5, "the ridge stands where it was measured"
    assert roof[:, 2].min() < 4.0, "the eaves stay at eave height"

    # level returns: a 0.3 m slab at the P85 height, nothing above the points
    parts = canopy_mesh(poly, X, Y, np.full_like(X, 4.0), 0.0)
    slab, _ = parts[0]
    assert np.ptp(slab[:, 2]) < 0.35 and abs(slab[:, 2].max() - 4.0) < 0.01

    # too low over the ground is street furniture, not a canopy
    assert canopy_mesh(poly, X, Y, np.full_like(X, 1.0), 0.0) == []

    # a corner post is the only wall reaching the ground
    pz = np.arange(0.2, 3.0, 0.3)
    parts = canopy_mesh(poly, np.r_[X, np.full_like(pz, 0.3)],
                        np.r_[Y, np.full_like(pz, 0.05)],
                        np.r_[np.full_like(X, 4.0), pz], 0.0)
    assert len(parts) == 2
    wall, _ = parts[1]
    low = wall[wall[:, 2] < 1.0]
    assert len(low) and low[:, 0].max() < 1.5, "wall cells only at the post"


def test_footprint_prism_claims_only_the_outline_and_two_heights():
    """The float-home lesson (2026-08-01): when the outline is surveyed and
    the roof height measured, the prism IS those two facts — walls on the
    outline, a flat cap, no bottom (it sits on ground or water)."""
    pytest.importorskip("mapbox_earcut")
    from ducklidar.scene import footprint_prism

    ring = np.array([[0.0, 0.0], [8.0, 0.0], [8.0, 5.0], [0.0, 5.0]])
    soup, cols = footprint_prism(ring, 1.5, 6.0, col=[10, 20, 30])
    assert len(soup) == len(cols) and (cols == [10, 20, 30]).all()
    # roof cap: 2 triangles at ztop, walls: 4 edges x 2 triangles
    assert len(soup) == 6 + 4 * 6
    assert soup[:, 2].min() == 1.5 and soup[:, 2].max() == 6.0
    cap = soup[:6]
    assert (cap[:, 2] == 6.0).all(), "the cap sits at the measured roof"
    # every wall vertex lies on the outline
    on = ((soup[:, 0] % 8.0 == 0) | (soup[:, 1] % 5.0 == 0))
    assert on.all(), "walls are the surveyed outline, nothing else"


def test_crown_thin_keeps_one_return_per_voxel_and_small_crowns_whole():
    """The crown model (2026-08-01): the crown is the tree's own points, but
    nothing in a canopy is finer than a leaf cluster — one return per 28 cm
    voxel keeps the shape without restating it."""
    from ducklidar.scene import crown_thin

    rng = np.random.default_rng(0)
    # 3000 returns crammed into a 1 m cube: at most ~4^3 voxels survive
    X, Y, Z = rng.uniform(0, 1.0, (3, 3000))
    keep = crown_thin(X, Y, Z)
    assert len(keep) < 200, "dense clusters collapse to their voxels"
    assert (np.diff(keep) > 0).all(), "indices come back sorted"
    # thinning is idempotent: survivors are already one-per-voxel
    again = crown_thin(X[keep], Y[keep], Z[keep])
    assert len(again) == len(keep)
    # a small crown passes through untouched
    small = crown_thin(X[:400], Y[:400], Z[:400])
    assert len(small) == 400
    # returns spread wider than the voxel all survive
    G = np.arange(0, 30.0, 1.0)
    assert len(crown_thin(np.repeat(G, 30), np.tile(G, 30),
                          np.zeros(900))) == 900


def test_sat_lift_brightens_shadow_and_adopts_the_witness_in_deep_shadow():
    """The deshadowing rules (2026-08-01): lift only where the satellite is
    clearly brighter, never darken below the survey, and in deep shadow adopt
    the witness because scaling noise just makes bright noise."""
    from ducklidar.scene import sat_lift

    # a 4-pixel satellite strip at 1 m: dark, matched, bright, very bright
    rgb = np.array([[[40, 40, 40], [100, 100, 100],
                     [151, 137, 124], [109, 109, 109]]], np.uint8)
    soup = np.array([[0.5, 0.5, 0.0], [1.5, 0.5, 0.0],
                     [2.5, 0.5, 0.0], [3.5, 0.5, 0.0]], np.float32)
    cols = np.array([[100, 100, 100], [100, 100, 100],
                     [80, 90, 110], [22, 22, 22]], np.uint8)
    out = sat_lift(soup, cols, rgb, 0.0, 1.0, 1.0)
    assert (out[0] == cols[0]).all(), "a darker satellite never darkens"
    assert (out[1] == cols[1]).all(), "matched paint is left alone"
    assert np.abs(out[2].astype(int) - [151, 137, 124]).max() <= 2, \
        "blue-grey shadow lifts per channel toward the warm witness"
    assert (out[3] == [109, 109, 109]).all(), "deep shadow adopts the witness"


def test_trace_footprint_finds_the_building_and_rejects_noise():
    """Where no map polygon exists, the instance's own points trace one:
    occupancy, largest region, simplified contour. Noise blobs under 10 m²
    return None instead of a footprint."""
    pytest.importorskip("skimage")
    pytest.importorskip("shapely")
    from ducklidar.scene import trace_footprint

    gx, gy = np.meshgrid(np.arange(0.0, 10.0, 0.3), np.arange(0.0, 6.0, 0.3))
    poly = trace_footprint(gx.ravel(), gy.ravel())
    assert poly is not None
    assert 45.0 < poly.area < 85.0, "roughly the 10 x 6 blob it was given"
    from shapely.geometry import Point
    assert poly.contains(Point(5.0, 3.0))

    assert trace_footprint(np.array([0.0, 1.5]), np.array([0.0, 1.5])) is None


def test_read_cityjsonseq_detransforms_and_keeps_only_the_asked_lod(tmp_path):
    """The parser owns the CityJSONSeq contract: quantised vertices times the
    dataset transform, faces ear-clipped on their Newell plane, and only the
    requested LoD survives — 2.2 roof planes, not the 1.2 block sharing the
    same object."""
    from ducklidar.scene import read_cityjsonseq

    scale, trans = [0.5, 0.5, 0.5], [10.0, 20.0, 0.0]
    v = [[x, y, z] for z in (0, 2) for y in (0, 2) for x in (0, 2)]
    # cube corners: 0..3 bottom, 4..7 top (x fastest, then y)
    faces = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4],
             [1, 3, 7, 5], [3, 2, 6, 7], [2, 0, 4, 6]]
    geom22 = {"lod": "2.2", "boundaries": [[[f] for f in faces]]}
    geom12 = {"lod": "1.2", "boundaries": [[[faces[0]]]]}
    lines = [
        json.dumps({"transform": {"scale": scale, "translate": trans}}),
        json.dumps({"vertices": v,
                    "CityObjects": {"roofed": {"geometry": [geom22]},
                                    "blocky": {"geometry": [geom12]}}}),
    ]
    p = tmp_path / "cube.city.jsonl"
    p.write_text("\n".join(lines))
    out = read_cityjsonseq(p)
    assert set(out) == {"roofed"}, "only the asked LoD survives"
    V, T = out["roofed"]
    assert T.shape == (12, 3), "six quads ear-clip to twelve triangles"
    assert V[:, 0].min() == 10.0 and V[:, 0].max() == 11.0
    assert V[:, 1].min() == 20.0 and V[:, 2].max() == 1.0
    tri = V[T]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    assert abs(area - 6.0) < 1e-9, "a closed unit cube, no face lost"


def test_mesh_bridges_connects_islands_with_measured_spans():
    """The mesh-graph design: the kNN graph's islands are a signal, and the
    repair is minimal — one MST bridge fewer than the island count, each
    carrying its length. 'One component, but only through a 14 m bridge' is
    the isolation measurement."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from ducklidar.scene import mesh_bridges

    rng = np.random.default_rng(1)
    mk = lambda cx: rng.uniform(-1, 1, (60, 3)) * [1, 1, 0.5] + [cx, 0, 0]
    P = np.vstack([mk(0.0), mk(10.0), mk(30.0)])
    src, dst, blen = mesh_bridges(P[:, 0], P[:, 1], P[:, 2])
    assert len(blen) == 2, "three islands need exactly two bridges"
    assert 6.0 < min(blen) < 10.0 and 16.0 < max(blen) < 20.0, \
        "each bridge carries the real gap it spans"
    # local edges + bridges = one component, by construction
    from scipy.spatial import cKDTree
    d, i = cKDTree(P).query(P, k=11, workers=-1)
    ok = (d[:, 1:] <= 2.5).ravel()
    a = np.repeat(np.arange(len(P)), 10)[ok]
    b = i[:, 1:].ravel()[ok]
    G = coo_matrix((np.ones(len(a) + len(src)),
                    (np.r_[a, src], np.r_[b, dst])), shape=(len(P), len(P)))
    assert connected_components(G, directed=False)[0] == 1

    # blind islands (everyone's nearest are their own) fall through to the
    # exact widening-box scan and still connect
    src, dst, blen = mesh_bridges(P[:, 0], P[:, 1], P[:, 2],
                                  k=3, k_search=3)
    assert len(blen) == 2 and max(blen) < 20.5

    # an already-connected cloud needs no bridges
    src, dst, blen = mesh_bridges(*mk(0.0).T)
    assert len(blen) == 0


def test_euclidean_mst_matches_brute_force(tmp_path):
    """The MST is defined on the COMPLETE graph, so brute force is the only
    proof: scipy over the full distance matrix. Two blobs 30 m apart exercise
    both regimes at once — Kruskal on the kNN candidates inside the 2.5 m cap,
    the bounded ball search for the one link that bridges the gap. Distances
    are quantised to float32 first, because f32 is the weight the algorithm
    has (`knn.parquet` stores `len` as f32); ties then fall to (src, dst)."""
    from scipy.sparse.csgraph import minimum_spanning_tree
    from ducklidar.graph import read_mst_parts

    rng = np.random.default_rng(7)
    P = np.vstack([rng.uniform(0, 5, (150, 3)),
                   rng.uniform(0, 5, (150, 3)) + [30.0, 0, 0]])

    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1).astype(np.float32)
    t = minimum_spanning_tree(d.astype(np.float64)).tocoo()
    want = {(min(i, j), max(i, j)) for i, j in zip(t.row, t.col)}

    src, dst = map(np.concatenate, zip(*dl.local_edges(P, k=10, cap=2.5)))
    ln = np.linalg.norm(P[src] - P[dst], axis=1).astype(np.float32)

    def read_box(bbox):
        m = ((P[:, 0] >= bbox[0]) & (P[:, 0] <= bbox[2])
             & (P[:, 1] >= bbox[1]) & (P[:, 1] <= bbox[3]))
        return P[m], np.flatnonzero(m).astype(np.int64)

    dl.euclidean_mst(iter([(src.astype(np.int64), dst.astype(np.int64), ln)]),
                     len(P), read_box, [np.arange(len(P))],
                     sweep_bbox=(P[:, 0].min(), P[:, 1].min(),
                                 P[:, 0].max(), P[:, 1].max()),
                     work=tmp_path / "mst", log=None)
    a, b, w = read_mst_parts(tmp_path / "mst")
    got = {(min(i, j), max(i, j)) for i, j in zip(a, b)}

    assert len(got) == len(P) - 1, "a spanning tree has |V|-1 edges"
    assert got == want, "the exact EMST, not merely one of the same weight"
    assert w.max() > 20.0, "the blob-to-blob link is beyond the kNN cap"


def test_wall_report_decides_walls_doors_and_windows():
    """Kaveh's spec (2026-08-01): per side, DECIDE — wall or not; door and
    where; windows and where. Absence framed by evidence is an opening;
    absence with no frame is occlusion or an open side, never a hole."""
    from ducklidar.scene import wall_report

    ring = np.array([[0.0, 0.0], [12.0, 0.0], [12.0, 8.0], [0.0, 8.0]])
    # south wall: a 12 x 3 m curtain sampled at 0.15 m, minus a doorway
    # (cells 4-5: x in [1.8, 2.7), ground to 1.8 m) and a window
    # (x in [6.3, 7.65), z in [0.9, 2.25))
    gx, gz = np.meshgrid(np.arange(0.03, 12.0, 0.15),
                         np.arange(0.08, 3.0, 0.15))
    X, Z = gx.ravel(), gz.ravel()
    door = (X >= 1.8) & (X < 2.7) & (Z < 1.8)
    wind = (X >= 6.3) & (X < 7.65) & (Z >= 0.9) & (Z < 2.25)
    X, Z = X[~(door | wind)], Z[~(door | wind)]
    Y = np.zeros_like(X)
    # east side: one post at the corner — open, not a holed wall
    pz = np.arange(0.1, 3.0, 0.15)
    X = np.r_[X, np.full_like(pz, 12.0)]
    Y = np.r_[Y, np.full_like(pz, 0.4)]
    Z = np.r_[Z, pz]

    rep = wall_report(ring, X, Y, Z, 0.0, 3.0)
    assert [r["side"] for r in rep] == [0, 1, 2, 3]
    south, east = rep[0], rep[1]
    assert south["has_wall"], "the curtain is a wall"
    assert not east["has_wall"] and not east["doors"], \
        "a lone post is an open side, not a holed wall"
    assert not rep[2]["has_wall"] and not rep[3]["has_wall"]

    (dr,) = south["doors"]
    assert abs(dr["s"] - 2.25) < 0.5 and abs(dr["height"] - 1.8) < 0.5
    (wn,) = south["windows"]
    assert abs(wn["s"] - 7.0) < 0.5 and abs(wn["z"] - 1.6) < 0.5
    assert 0.9 <= wn["width"] <= 1.9 and 0.9 <= wn["height"] <= 1.9

    # a shadowed lower wall (no returns below 1.5 m, no frame under it)
    # reports NO door and NO window — absence without a frame is occlusion
    keep = gz.ravel() >= 1.5
    rep = wall_report(ring, gx.ravel()[keep], np.zeros(keep.sum()),
                      gz.ravel()[keep], 0.0, 3.0)
    assert rep[0]["has_wall"] and not rep[0]["doors"] \
        and not rep[0]["windows"], "shadow is not a hole"


def test_wall_mesh_builds_full_walls_with_openings_cut():
    """Kaveh's spec (2026-08-01): not a confetti of cells — a walled side is
    one full surface with its detected door and windows cut out as real
    holes; an open side keeps only its evidence cells."""
    from ducklidar.scene import wall_mesh

    ring = np.array([[0.0, 0.0], [12.0, 0.0], [12.0, 8.0], [0.0, 8.0]])
    gx, gz = np.meshgrid(np.arange(0.03, 12.0, 0.15),
                         np.arange(0.08, 3.0, 0.15))
    X, Z = gx.ravel(), gz.ravel()
    door = (X >= 1.8) & (X < 2.7) & (Z < 1.8)
    wind = (X >= 6.3) & (X < 7.65) & (Z >= 0.9) & (Z < 2.25)
    X, Z = X[~(door | wind)], Z[~(door | wind)]
    Y = np.zeros_like(X)
    pz = np.arange(0.1, 3.0, 0.15)
    X = np.r_[X, np.full_like(pz, 12.0)]
    Y = np.r_[Y, np.full_like(pz, 0.4)]
    Z = np.r_[Z, pz]

    parts, rep = wall_mesh(ring, X, Y, Z, 0.0, 3.0)
    assert rep[0]["has_wall"] and len(rep[0]["doors"]) == 1 \
        and len(rep[0]["windows"]) == 1

    # the south wall is one surface whose area is the rectangle minus the
    # door and the window — not a per-cell mosaic
    south = parts[0][0]
    tri = south.reshape(-1, 3, 3)
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    dr, wn = rep[0]["doors"][0], rep[0]["windows"][0]
    want = 12.0 * 3.0 - dr["width"] * dr["height"] - wn["width"] * wn["height"]
    assert abs(area - want) < 1e-6, "full wall minus exactly its openings"
    assert (south[:, 1] == 0).all(), "the wall lies on its side's plane"
    # nothing spans the door: no vertex strictly inside its rectangle
    inside = (south[:, 0] > dr["s"] - dr["width"] / 2 + 1e-6) \
        & (south[:, 0] < dr["s"] + dr["width"] / 2 - 1e-6) \
        & (south[:, 2] < dr["height"] - 1e-6) & (south[:, 2] > 1e-6)
    assert not inside.any(), "the door is a real hole"

    # the post side stays open: evidence cells at the post and at the wall's
    # own corners (its returns are within reach of the adjacent side), but
    # nothing along the middles of the open sides
    cells = parts[-1][0]
    on_east = cells[:, 0] > 11.9
    assert on_east.any() and cells[on_east, 1].max() < 1.5, \
        "the post's cells hug the post"
    mid_open = ((cells[:, 0] < 0.1) | (cells[:, 1] > 7.9)) \
        & (cells[:, 1] > 2.0) & (cells[:, 1] < 7.0) \
        & (cells[:, 0] > 2.0) & (cells[:, 0] < 10.0)
    assert not mid_open.any(), "open-side middles stay open"
    assert len(parts) == 2, "one wall surface + one cells part, nothing else"


def test_wall_mesh_tops_close_the_pediment_under_a_gable():
    """Kaveh's catch (2026-08-01): a flat-topped wall under a gable roof
    misses the pediment. With a top profile the wall follows the rake —
    eave at the corners, ridge at the crossing — and nothing is missed."""
    from ducklidar.scene import wall_mesh

    ring = np.array([[0.0, 0.0], [12.0, 0.0], [12.0, 8.0], [0.0, 8.0]])
    gx, gz = np.meshgrid(np.arange(0.03, 12.0, 0.15),
                         np.arange(0.08, 3.0, 0.15))
    X, Z = gx.ravel(), gz.ravel()
    Y = np.zeros_like(X)
    tops = {0: [(0.0, 3.0), (6.0, 5.0), (12.0, 3.0)]}
    parts, rep = wall_mesh(ring, X, Y, Z, 0.0, 3.0, tops=tops)
    assert rep[0]["has_wall"]
    wall = parts[0][0]
    tri = wall.reshape(-1, 3, 3)
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    assert abs(area - (12.0 * 3.0 + 12.0)) < 1e-6, \
        "the rectangle plus the whole pediment triangle, nothing missed"
    assert abs(wall[:, 2].max() - 5.0) < 1e-6, "the wall reaches the ridge"
    assert (wall[:, 1] == 0).all()


def test_checked_walls_puts_roofers_walls_on_trial():
    """Kaveh's roofer improvement (2026-08-01): roofer extrudes every wall
    without checking. Here each side's faces stand trial — no evidence
    removes the invented wall, evidence with framed openings rebuilds the
    wall with its windows cut, and the roof is never touched."""
    from ducklidar.scene import checked_walls, footprint_prism

    ring = np.array([[0.0, 0.0], [12.0, 0.0], [12.0, 8.0], [0.0, 8.0]])
    soup, cols = footprint_prism(ring, 0.0, 3.0)

    gx, gz = np.meshgrid(np.arange(0.03, 12.0, 0.15),
                         np.arange(0.08, 3.0, 0.15))
    X, Z = gx.ravel(), gz.ravel()
    door = (X >= 1.8) & (X < 2.7) & (Z < 1.8)
    wind = (X >= 6.3) & (X < 7.65) & (Z >= 0.9) & (Z < 2.25)
    X, Z = X[~(door | wind)], Z[~(door | wind)]
    Y = np.zeros_like(X)
    pz = np.arange(0.1, 3.0, 0.15)
    X = np.r_[X, np.full_like(pz, 12.0)]
    Y = np.r_[Y, np.full_like(pz, 0.4)]
    Z = np.r_[Z, pz]

    # CLOSED building (no returns under the roof): shadowed sides keep
    # roofer's faces — absence at the wall plane is shadow, not proof —
    # while the south wall still gets its openings cut
    kept, kcols, parts = checked_walls(soup, cols, ring, X, Y, Z, 0.0, 3.0)
    assert len(kept) == 6 + 3 * 6, \
        "cap + three shadowed walls stand; only the south wall was rebuilt"
    wall = parts[0][0]
    tri = wall.reshape(-1, 3, 3)
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    assert 12.0 * 3.0 - 6.0 < area < 12.0 * 3.0 - 2.0, \
        "full wall minus door and window"
    assert (wall[:, 1] == 0).all()

    # OPEN structure (the laser reaches under the roof): the unevidenced
    # walls are removed and the post survives as evidence cells
    ix, iy = np.meshgrid(np.arange(2.0, 10.0, 0.5), np.arange(2.0, 6.0, 0.5))
    Xo = np.r_[X, ix.ravel()]
    Yo = np.r_[Y, iy.ravel()]
    Zo = np.r_[Z, np.full(ix.size, 0.5)]
    kept, kcols, parts = checked_walls(soup, cols, ring, Xo, Yo, Zo, 0.0, 3.0)
    assert len(kept) == 6 and (kept[:, 2] == 3.0).all(), \
        "only the roof survives on a structure the laser sees under"
    cells = parts[-1][0]
    assert (cells[:, 0] > 11.9).any(), "the post's cells remain"


def test_pier_ways(tmp_path):
    """A synthetic duckOverture extract: the query, the clip, the ring, the coverage guard."""
    duckdb = pytest.importorskip("duckdb")
    pytest.importorskip("pyproj")
    from pyproj import Transformer

    to_ll = Transformer.from_crs(26910, 4326, always_xy=True)
    B = dl.box(490289, 5457557, 250)

    def ls(pts, kind="LINESTRING"):
        body = ", ".join("%.9f %.9f" % to_ll.transform(x, y) for x, y in pts)
        return (kind + "(" + body + ")" if kind == "LINESTRING"
                else kind + "((" + body + "))")

    inside = [(490300.0, 5457500.0), (490350.0, 5457560.0)]
    ring = [(490200.0, 5457400.0), (490240.0, 5457400.0), (490240.0, 5457430.0),
            (490200.0, 5457430.0), (490200.0, 5457400.0)]
    outside = [(490289.0, 5459000.0), (490350.0, 5459000.0)]

    db = str(tmp_path / "toy_overture.duckdb")
    con = duckdb.connect(db)
    con.execute("INSTALL spatial; LOAD spatial")
    blo = to_ll.transform(B[0] - 500, B[1] - 500)
    bhi = to_ll.transform(B[2] + 500, B[3] + 500)
    con.execute("create table boundary as select ST_MakeEnvelope(?, ?, ?, ?) as geom",
                [*blo, *bhi])
    con.execute("create schema base")
    con.execute("""
        create table base.infrastructure as
        select [{'record_id': rid}] as sources, {'primary': nm} as names,
               MAP(ks, vs) as source_tags, 'pier' as class,
               ST_GeomFromText(wkt) as geometry,
               {'xmin': ST_XMin(ST_GeomFromText(wkt)),
                'xmax': ST_XMax(ST_GeomFromText(wkt)),
                'ymin': ST_YMin(ST_GeomFromText(wkt)),
                'ymax': ST_YMax(ST_GeomFromText(wkt))} as bbox
        from (values
          ('w42@3', 'A Dock', ['man_made', 'floating'], ['pier', 'yes'], ?),
          ('w43@1', '',       ['man_made', 'area'],     ['pier', 'yes'], ?),
          ('w44@1', 'Afar',   ['man_made'],             ['pier'],        ?),
          ('msft/7', 'NotOSM', ['man_made'],            ['pier'],        ?)
        ) t(rid, nm, ks, vs, wkt)""",
        [ls(inside), ls(ring, "POLYGON"), ls(outside), ls(inside)])
    con.close()

    ways = dl.pier_ways(db, B)
    by_id = {w["id"]: w for w in ways}
    assert set(by_id) == {42, 43}, "outside-box and non-OSM rows are excluded"
    assert by_id[42]["name"] == "A Dock" and by_id[42]["floating"] == "yes"
    assert all(abs(g[0] - p[0]) < 0.02 and abs(g[1] - p[1]) < 0.02
               for g, p in zip(by_id[42]["coords"], inside)), \
        "round-trip through 4326 stays sub-cm"
    assert by_id[43]["area"] == "yes" and len(by_id[43]["coords"]) == 5, \
        "a polygon pier yields its closed exterior ring"

    with pytest.raises(ValueError):
        dl.pier_ways(db, dl.box(490289, 5457557, 5000))


def test_read_multi_tile(tmp_path):
    """A window straddling two tiles gets both tiles' points; distant tiles cost a header."""
    import laspy

    def write_tile(path, x0):
        hdr = laspy.LasHeader(point_format=3, version="1.2")
        hdr.scales = [0.001, 0.001, 0.001]
        hdr.offsets = [x0, 0.0, 0.0]
        las = laspy.LasData(hdr)
        las.x = np.arange(x0 + 5.0, x0 + 100.0, 10.0)   # 10 points, 5..95 within the tile
        las.y = np.full(10, 50.0)
        las.z = np.full(10, 1.0)
        las.classification = np.full(10, 2, dtype=np.uint8)
        las.write(str(path))
        return path

    a = write_tile(tmp_path / "a.las", 0.0)      # tile x 0..100
    b = write_tile(tmp_path / "b.las", 100.0)    # tile x 100..200
    far = write_tile(tmp_path / "far.las", 9000.0)

    win = (80.0, 0.0, 120.0, 100.0)              # straddles the a|b border at x=100
    pts = dl.read([a, b, far], win)
    assert sorted(pts["x"]) == [85.0, 95.0, 105.0, 115.0], \
        "both sides of the border answer; the far tile contributes nothing"

    # same answer when one side is served by its parquet sidecar at a foreign path
    side = dl.to_parquet(b, tmp_path / "elsewhere.parquet")
    pts2 = dl.read([a, side, far], win)
    assert sorted(pts2["x"]) == sorted(pts["x"])

    # the sidecar carries pid — the point's row in its file, the join key for
    # derived tables — and a window read can ask for it
    withid = dl.read(side, win, fields=("pid",))
    assert set(withid) == {"x", "y", "z", "classification", "pid"}
    full = dl.read(side, (0.0, 0.0, 1000.0, 1000.0), fields=("pid",))
    assert sorted(full["pid"]) == list(range(10)), "pid is a permutation of file rows"

    empty = dl.read([a, b], (300.0, 0.0, 400.0, 100.0))
    assert len(empty["x"]) == 0 and "z" in empty

    assert dl.box(10.0, 20.0, 2.0, 1.0) == (8.0, 19.0, 12.0, 21.0), \
        "rectangular study windows: half_y caps the y reach"


def test_local_edges():
    """The mesh graph's local half: k nearest within the cap, chunking invisible."""
    pytest.importorskip("scipy")

    xyz = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [100.0, 0, 0]])
    got = {(int(a), int(b))
           for src, dst in dl.local_edges(xyz, k=2, cap=2.5)
           for a, b in zip(src, dst)}
    assert got == {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}, \
        "three chained points interconnect; the far point reaches nothing"

    chunked = [dl.local_edges(xyz, k=2, cap=2.5, chunk=1),
               dl.local_edges(xyz, k=2, cap=2.5)]
    sets = [{(int(a), int(b)) for s, d in g for a, b in zip(s, d)} for g in chunked]
    assert sets[0] == sets[1] == got, "chunk size never changes the edge set"

    nn = dl.knn(xyz, k=3)
    via_nn = {(int(a), int(b))
              for s, d in dl.local_edges(xyz, k=2, cap=2.5, nn=nn)
              for a, b in zip(s, d)}
    assert via_nn == got, "edges derived from a cached knn match the direct build"


def test_knn_shape_features():
    """The kNN level: a roof is planar-horizontal, a wall planar-vertical, a line linear."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(0)

    def feats(P):
        dist, idx = dl.knn(P, k=8)
        e1, e2, e3, nz = dl.shape_features(P, idx)
        return e1.mean(), e2.mean(), e3.mean(), nz.mean()

    g = np.stack(np.meshgrid(np.arange(10.0), np.arange(10.0)), -1).reshape(-1, 2)
    roof = np.c_[g, np.full(len(g), 5.0) + rng.normal(0, 0.01, len(g))]
    e1, e2, e3, nz = feats(roof)
    assert e3 < 0.01 and nz > 0.98, "flat horizontal: no thickness, normal straight up"

    wall = np.c_[g[:, :1], np.full((len(g), 1), 2.0), g[:, 1:]]
    wall[:, 1] += rng.normal(0, 0.01, len(g))
    *_, e3w, nzw = feats(wall)
    assert e3w < 0.01 and nzw < 0.02, "flat vertical: normal horizontal"

    line = np.c_[np.arange(0, 50, 0.5), np.zeros(100), np.zeros(100)]
    e1l, e2l, *_ = feats(line)
    assert e1l > 0.98 and e2l < 0.02, "a line is all first eigenvalue"


def test_balanced_boxes(tmp_path):
    """Partitions follow the data: a dense corner gets small boxes, leaves balance."""
    duckdb = pytest.importorskip("duckdb")
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(1)
    dense = rng.uniform(0, 250, (80_000, 2))          # a packed quarter
    sparse = rng.uniform(0, 1000, (20_000, 2))        # thin everywhere else
    xy = np.vstack([dense, sparse])
    f = str(tmp_path / "pts.parquet")
    pq.write_table(pa.table({"x": xy[:, 0], "y": xy[:, 1]}), f)

    boxes = dl.balanced_boxes([f], (0.0, 0.0, 1000.0, 1000.0), 20_000)
    counts = [b[4] for b in boxes]
    assert all(c <= 20_000 for c in counts), "every leaf under the cap"
    assert sum(counts) == 100_000, "leaves partition the points exactly"
    areas = {(b[2] - b[0]) * (b[3] - b[1]) for b in boxes}
    assert max(areas) / min(areas) > 4, "dense regions got smaller boxes"


def test_cell_evidence_and_ground(tmp_path):
    """Stage 5 on tables: GROUP BY evidence, then one global cloth on the cells."""
    duckdb = pytest.importorskip("duckdb")
    pytest.importorskip("CSF")
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(2)
    gx = rng.uniform(0, 100, (40_000, 2))                     # ground, z ~ 10
    rx = rng.uniform(30, 60, (10_000, 2))                     # roof block, z ~ 30
    x = np.r_[gx[:, 0], rx[:, 0]]
    y = np.r_[gx[:, 1], rx[:, 1]]
    z = np.r_[np.full(40_000, 10.0), np.full(10_000, 30.0)]
    z += rng.normal(0, 0.02, len(z))
    f = str(tmp_path / "pts.parquet")
    pq.write_table(pa.table({
        "x": x, "y": y, "z": z,
        "classification": np.r_[np.full(40_000, 2), np.full(10_000, 6)].astype(np.uint8),
        "return_number": np.ones(len(z), np.uint8),
        "number_of_returns": np.ones(len(z), np.uint8)}), f)

    bbox = (0.0, 0.0, 100.0, 100.0)
    cells = dl.cell_evidence([f], bbox, pix=1.0)
    assert cells.num_rows > 9_000, "nearly every 1 m cell is occupied"
    d = {c: cells[c].to_numpy() for c in cells.column_names}
    assert int(d["n"].sum()) == 50_000
    assert float(d["split_share"].max()) == 0.0, "single-return synthetic"

    cell, gz, _grid = dl.ground_cells(cells, bbox, pix=1.0)
    inside = (d["min_z"] > 25)                    # roof-only cells (ground hidden)
    assert inside.any()
    assert np.allclose(gz[~inside], 10.0, atol=0.5), "open ground recovered"
    assert np.allclose(gz[inside], 10.0, atol=1.5), \
        "under the roof the cloth bridges at ground level, not roof level"

    hag = np.full(len(gz), np.nan)                # the per-point join, in miniature
    pc = dl.cell_of(x, y, bbox)
    m = {int(c): g for c, g in zip(cell, gz)}
    hag = z - np.array([m[int(c)] for c in pc])
    assert (hag[-10_000:] > 15).mean() > 0.95, "roof points stand ~20 m above ground"


def test_cell_components(tmp_path):
    """Region questions on the cell grid: two ponds are two bodies, globally."""
    pytest.importorskip("duckdb")
    pytest.importorskip("scipy")
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(3)
    def blob(cx, cy, r, n):
        a = rng.uniform(0, 2 * np.pi, n); d = rng.uniform(0, r, n)
        return np.c_[cx + d * np.cos(a), cy + d * np.sin(a)]
    land = rng.uniform(0, 200, (30_000, 2))
    pond1, pond2 = blob(50, 50, 15, 8_000), blob(150, 150, 10, 5_000)
    xy = np.vstack([land, pond1, pond2])
    cls = np.r_[np.full(30_000, 2), np.full(13_000, 9)].astype(np.uint8)
    f = str(tmp_path / "p.parquet")
    pq.write_table(pa.table({"x": xy[:, 0], "y": xy[:, 1],
                             "z": np.where(cls == 9, 0.0, 5.0),
                             "classification": cls,
                             "return_number": np.ones(len(cls), np.uint8),
                             "number_of_returns": np.ones(len(cls), np.uint8)}), f)

    bbox = (0.0, 0.0, 200.0, 200.0)
    cells = dl.cell_evidence([f], bbox)
    wet = cells["n_water"].to_numpy() > 0
    body = dl.cell_components(cells, wet, bbox)
    ids = set(body[wet]) - {0}
    assert len(ids) == 2, "two ponds, two bodies"
    assert (body[~wet] == 0).all(), "dry cells belong to no body"


def test_stage_report(tmp_path):
    log = tmp_path / "pipeline_report.log"
    line = dl.stage_report("stage4", {"points": 168377550, "components": 9548,
                                      "gate": "99.79%"}, log=log)
    dl.stage_report("stage5", {"leaves": 10}, log=log)
    text = log.read_text().splitlines()
    assert len(text) == 2 and text[0] == line
    assert "points=168,377,550" in text[0] and "gate=99.79%" in text[0]


def test_column_support_separates_a_truck_from_an_eave():
    """The context per-point rules lack: in 2-D a truck and a one-storey eave
    are the same cell (ground below, roof above). The column is not — under the
    eave there is 2 m of walkable air, so it fails the continuity test."""
    bbox = (0.0, 0.0, 3.0, 1.0)                      # 1 m cells; 2 is returnless
    cells = {"cell": np.array([0, 1]), "ground_z": np.array([10.0, 10.0])}
    lvl = lambda z: int(np.floor(z / 0.5))           # noqa: E731 — the voxel table's level
    truck = np.arange(lvl(10.0), lvl(13.0))          # matter 0-3 m, continuous
    eave = np.r_[lvl(10.0),                          # a wall foot / the yard
                 np.arange(lvl(12.5), lvl(13.5))]    # ...air, then the overhang
    voxels = {"cell": np.r_[np.zeros(len(truck), int), np.ones(len(eave), int)],
              "level": np.r_[truck, eave]}

    s = dl.column_support(voxels, cells, bbox, 1.0)
    assert s.supported(0, 3.0) and s.gap_below(0, 3.0) == 0.0
    assert not s.supported(1, 3.0)
    assert s.gap_below(1, 3.0) == pytest.approx(2.0), "0.5-2.5 m is empty"
    assert not s.supported(2, 3.0), "a cell with no evidence is air, not matter"


def test_corridor_offbox_way_paints_no_corner():
    """A width class whose ways all fall outside the box must contribute
    nothing. distance_transform_edt of an all-True grid measures distance to
    the OUTSIDE, so the empty line used to paint a radius-r quarter-disc in
    the corner (83 phantom cells in the reference tile)."""
    bbox = (0.0, 0.0, 50.0, 50.0)
    cells = {"cell": np.arange(2500), "max_z": np.zeros(2500)}
    ways = {"ways": [{"highway": "secondary", "lanes": "5",
                      "coords": [[-500.0, -500.0], [-400.0, -400.0]]}]}
    mask, grid = dl.corridor_cells(ways, cells, np.zeros(2500), bbox)
    assert not grid.any(), f"{int(grid.sum())} phantom corner cells"


def test_path_shares_from_pred_matches_a_hand_built_forest():
    """The stored-forest shares must equal what walking the chain by hand gives.

    Chain 0 <- 1 <- 2 <- 3 with 0 the root, and cats marking node 1 as WALL and
    node 2 as FOLIAGE. Node 3's path crosses 2 (foliage) then 1 (wall) then 0
    (other), so its shares are 1/3 each.
    """
    import numpy as np

    import ducklidar as dl
    from ducklidar.labeling import CAT_FOLIAGE, CAT_OTHER, CAT_WALL

    pred = np.array([-1, 0, 1, 2], np.int64)
    depth = np.array([0, 1, 2, 3], np.int32)
    cats = np.array([CAT_OTHER, CAT_WALL, CAT_FOLIAGE, CAT_OTHER], np.uint8)
    bsh, vsh = dl.path_shares_from_pred(pred, depth, cats)

    assert bsh[0] == 0.0 and vsh[0] == 0.0          # a root is never refined
    assert bsh[1] == 0.0 and vsh[1] == 0.0          # crosses only node 0 (other)
    assert bsh[2] == 0.5 and vsh[2] == 0.0          # crosses 1 (wall) + 0 (other)
    assert abs(bsh[3] - 1 / 3) < 1e-12              # crosses 2, 1, 0
    assert abs(vsh[3] - 1 / 3) < 1e-12


def test_path_shares_from_pred_totals_are_the_depth():
    """One category everywhere ⇒ every reached node's share is exactly 1.

    This is the invariant that validated the operator against the real
    42.8 M-vertex forest: the histogram total must BE the stored depth.
    """
    import numpy as np

    import ducklidar as dl
    from ducklidar.labeling import CAT_WALL

    n = 500
    pred = np.arange(-1, n - 1, dtype=np.int64)      # one long chain
    depth = np.arange(n, dtype=np.int32)
    cats = np.full(n, CAT_WALL, np.uint8)
    bsh, _ = dl.path_shares_from_pred(pred, depth, cats)
    assert (bsh[depth > 0] == 1.0).all()
    assert bsh[0] == 0.0


def test_footprint_ids_agrees_with_footprint_cells():
    """`footprint_ids != fill` must be exactly `footprint_cells`, and carry ids."""
    import numpy as np

    import ducklidar as dl

    bbox = (0.0, 0.0, 10.0, 10.0)
    polys = [{"id": 42, "coords": [[1, 1], [4, 1], [4, 4], [1, 4]]},
             {"id": 7, "coords": [[6, 6], [9, 6], [9, 9], [6, 9]]}]
    idx = dl.footprint_ids(polys, bbox, 1.0)          # default: polygon INDEX
    cells = dl.footprint_cells(polys, bbox, 1.0)
    assert idx.shape == cells.shape
    assert ((idx != -1) == cells).all()
    assert set(np.unique(idx)) == {-1, 0, 1}
    ids = dl.footprint_ids(polys, bbox, 1.0, key="id")  # explicit int ids
    assert set(np.unique(ids)) == {-1, 7, 42}
    # a UUID id must not raise under the default
    uu = [{"id": "f8ae588d-cf44-4b97-8ff2-6f7bd9509c29",
           "coords": polys[0]["coords"]}]
    assert set(np.unique(dl.footprint_ids(uu, bbox, 1.0))) == {-1, 0}


def test_raft_split_separates_two_boats_on_one_pontoon():
    """Two hulls touching at pontoon level must come out as two ids.

    The case the class-induced subgraph cannot cut: everything is connected
    within ~1 m of the water, so only height separates the vessels.
    """
    import numpy as np

    import ducklidar as dl

    rng = np.random.default_rng(0)
    # a 30 x 4 m pontoon at z=0, with two 6 m hulls rising 2.5 m, 12 m apart
    px, py, pz = [], [], []
    gx, gy = np.meshgrid(np.arange(0, 30, 0.25), np.arange(0, 4, 0.25))
    px.append(gx.ravel()); py.append(gy.ravel())
    pz.append(np.zeros(gx.size))
    for cx in (5.0, 21.0):
        hx, hy = np.meshgrid(np.arange(cx, cx + 6, 0.25), np.arange(0, 3, 0.25))
        px.append(hx.ravel()); py.append(hy.ravel())
        pz.append(np.full(hx.size, 2.5))
    px = np.concatenate(px); py = np.concatenate(py); pz = np.concatenate(pz)
    px += rng.normal(0, 0.01, px.size)

    ids = dl.raft_split(px, py, pz)
    assert len(ids) == len(px)
    hull_a = ids[(px > 5) & (px < 11) & (pz > 2)]
    hull_b = ids[(px > 21) & (px < 27) & (pz > 2)]
    assert len(np.unique(hull_a)) == 1 and len(np.unique(hull_b)) == 1, \
        "each hull must be ONE id — h-maxima should merge bumps on one ridge"
    assert hull_a[0] != hull_b[0], "the two hulls must not share an id"


def test_flood_deck_tol_takes_the_covered_slip_and_leaves_the_beach():
    """A marina roofed by its own floats joins the body; dry land does not.

    Three strips at the same 1 m WORK grid, all south-up: open water at
    -1.0, a covered slip whose cloth settled on the pontoon at +0.5 with NO
    class-2 return, and a beach also at +0.5 that IS classed ground. The
    strict tol=0.4 admits neither of the raised strips; deck_tol=1.0 admits
    the slip only — n_ground == 0 is the whole guard.
    """
    pytest.importorskip("scipy")
    import pyarrow as pa

    bbox, nx, ny = (0.0, 0.0, 20.0, 20.0), 20, 20
    cell = np.arange(nx * ny)
    col = cell % nx
    surface = np.where(col < 5, -1.0, 0.5).reshape(ny, nx)   # slip and beach both +0.5
    n_ground = np.where(col < 10, 0, 7)                      # only the beach is ground
    cells = pa.table({"cell": cell.astype(np.int64),
                      "n": np.ones(nx * ny, np.int64),
                      "n_ground": n_ground.astype(np.int64)})
    seeds = col < 5                                          # the open water seeds

    _, off = dl.flood_bodies(cells, seeds, surface, {0: 0.0}, bbox)
    off = off.reshape(ny, nx) > 0
    assert off[:, :5].all() and not off[:, 5:].any(), "default behaviour is unchanged"

    _, grid = dl.flood_bodies(cells, seeds, surface, {0: 0.0}, bbox, deck_tol=1.0)
    grid = grid.reshape(ny, nx)
    on = grid > 0
    assert on[:, 5:10].all(), "the covered slip joins the body"
    assert not on[:, 10:].any(), "the beach is class-2 ground and stays dry"
    assert set(grid[on].tolist()) == {1}, "slip and open water are ONE body"


# ===========================================================================
# ONE MODEL PER OBJECT TYPE, and every model states its KIND (2026-08-05).
# The audit these replace found four types borrowing another type's recipe:
# bridge -> solid_25d from below the waterline (v1's blackout bug), small and
# on_bridge -> car_mesh, glass -> footprint_prism.
# ===========================================================================

def _deck_points(n=4000, seed=3, girder=True, piers=False):
    """A 40 x 12 m deck at z=20, optionally with the girder returns that
    measure its depth, and/or two pier stumps standing down to the water."""
    rng = np.random.default_rng(seed)
    out = [np.column_stack([rng.uniform(0, 40, n), rng.uniform(0, 12, n),
                            np.full(n, 20.0)])]
    if girder:                        # fascia returns 1.0-1.4 m under the deck
        m = 600
        out.append(np.column_stack([rng.uniform(0, 40, m),
                                    rng.choice([0.4, 11.6], m),
                                    rng.uniform(18.6, 19.0, m)]))
    if piers:
        for cx in (10.0, 30.0):
            m = 400
            out.append(np.column_stack([
                cx + rng.uniform(-1.5, 1.5, m), rng.uniform(4, 8, m),
                rng.uniform(0.5, 16.0, m)]))
    return np.vstack(out)


def test_bridge_deck_is_a_slab_with_two_distinct_z_levels():
    """A deck's defining fact is the void it spans: measured, Granville as a
    slab blocks only above 72.6 deg (a sun angle Vancouver never reaches),
    while the same deck extruded from the waterline shades the channel all
    day. So the model must produce a top and an underside — and NOTHING
    between the underside and the water."""
    from ducklidar import objects as O

    P = _deck_points()
    parts = O.bridge_deck_model(P[:, 0], P[:, 1], P[:, 2])
    assert len(parts) == 1
    soup, cols, kind = parts[0]
    assert kind == O.SLAB, "a bridge deck is a slab, never a solid"
    assert len(soup) == len(cols) and len(soup) % 3 == 0
    zs = np.unique(np.round(soup[:, 2], 3))
    assert len(zs) == 2, f"a slab is a (top, bottom) pair, got {zs}"
    bot, top = float(zs[0]), float(zs[1])
    assert top == 20.0, "the deck surface is the measured one"
    assert bot > 15.0, "nothing reaches the waterline: the void stays open"
    assert 0.6 <= top - bot <= 3.0, (top, bot)
    # each level carries real area (a plate, not a wedge)
    assert (soup[:, 2] > top - 1e-3).sum() > 100
    assert (soup[:, 2] < bot + 1e-3).sum() > 100
    # the thickness is MEASURED from the girder returns, not the lab default
    assert abs((top - bot) - 1.35) < 0.15, top - bot
    assert abs((top - bot) - O.DECK_T) > 1e-6


def test_bridge_deck_falls_back_to_the_labs_measured_thickness():
    """With no sub-deck evidence the lab's Granville measurement stands:
    DECK_T = 1.6 m box-girder depth [2026-08-01]."""
    from ducklidar import objects as O

    rng = np.random.default_rng(1)
    n = 3000
    P = np.column_stack([rng.uniform(0, 30, n), rng.uniform(0, 10, n),
                         np.full(n, 12.0)])
    soup, _c, kind = O.bridge_deck_model(P[:, 0], P[:, 1], P[:, 2])[0]
    assert kind == O.SLAB
    assert abs(np.ptp(soup[:, 2]) - O.DECK_T) < 1e-3


def test_bridge_support_is_a_solid_column_and_only_where_measured():
    """Kaveh's ruling (2026-08-05): the support is NOT the bridge. It is an
    object standing on ground or in water, opaque from footing to deck — and
    it is claimed from returns, never spaced by rule."""
    from ducklidar import objects as O

    P = _deck_points(girder=False, piers=True)
    piers = O.bridge_support_model(P[:, 0], P[:, 1], P[:, 2],
                                   ground_z=0.0, deck_bottom=18.4)
    assert len(piers) == 2, "one solid per measured pier"
    for soup, cols, kind in piers:
        assert kind == O.SOLID, "a pier stops light"
        assert len(soup) == 36, "a closed box is 12 triangles"
        assert abs(soup[:, 2].min()) < 1e-6 and abs(soup[:, 2].max() - 18.4) < 1e-6
        assert len(cols) == len(soup)
    # deck-only returns invent no piers
    flat = P[P[:, 2] > 19.0]
    assert O.bridge_support_model(flat[:, 0], flat[:, 1], flat[:, 2],
                                  0.0, 18.4) == []


def test_glass_is_a_zero_thickness_pane_on_its_own_plane():
    """Glazing is a facet, not a volume: a curtain wall is a skin on a wall
    that already exists as its own solid, and it transmits. The pane is built
    in the plane the returns were measured in, so a vertical facade stays
    vertical and has no depth."""
    from ducklidar import objects as O

    rng = np.random.default_rng(5)
    n = 1500
    # a 6 x 4 m window wall at x = 3, scattered +- 2 cm (glass is not perfect)
    P = np.column_stack([3.0 + rng.normal(0, 0.02, n),
                         rng.uniform(0, 6, n), rng.uniform(10, 14, n)])
    parts = O.glass_model(P[:, 0], P[:, 1], P[:, 2])
    assert len(parts) == 1
    soup, cols, kind = parts[0]
    assert kind == O.GLASS, "glass is neither solid nor slab: it transmits"
    assert len(soup) % 6 == 0, "flat quads, two triangles each"
    assert np.ptp(soup[:, 0]) < 0.25, "no thickness across the facet"
    assert np.ptp(soup[:, 1]) > 4.0 and np.ptp(soup[:, 2]) > 3.0
    assert (cols == O.WINCOL).all()
    # a horizontal skylight goes through the SAME method, no special case
    S = np.column_stack([rng.uniform(0, 4, 600), rng.uniform(0, 4, 600),
                         20.0 + rng.normal(0, 0.02, 600)])
    sky = O.glass_model(S[:, 0], S[:, 1], S[:, 2])[0]
    assert sky[2] == O.GLASS and np.ptp(sky[0][:, 2]) < 0.25


def test_small_is_its_own_extrusion_and_never_a_car():
    """947 of 2,155 objects were drawn as cars AFTER looks_like_car said they
    were not cars. A bin sits ON the pavement, has no cabin and no long axis:
    the only honest model is its own measured top extruded to the ground."""
    from ducklidar import objects as O

    rng = np.random.default_rng(7)
    n = 900
    # a 2 m round bollard-ish blob, 1.4 m tall, standing at z = 5
    a = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(rng.uniform(0, 1, n)) * 1.0
    P = np.column_stack([50 + r * np.cos(a), 50 + r * np.sin(a),
                         5.0 + rng.uniform(1.0, 1.4, n)])
    soup, cols, kind = O.small_model(P[:, 0], P[:, 1], P[:, 2], 5.0)[0]
    assert kind == O.SOLID
    assert abs(soup[:, 2].min() - 5.0) < 0.01, "it stands ON the ground"
    assert 6.0 < soup[:, 2].max() < 6.5
    # the car model on the same points would leave a 0.25 m gap under it
    car = O.car_model(P[:, 0], P[:, 1], P[:, 2])
    assert car[0][0][:, 2].min() > soup[:, 2].min() + 0.2
    assert len(cols) == len(soup)
    # too low to be anything: no mesh, not a car
    flat = P.copy()
    flat[:, 2] = 5.0 + rng.uniform(0, 0.3, n)
    assert O.small_model(flat[:, 0], flat[:, 1], flat[:, 2], 5.0) == []


def test_on_bridge_keeps_its_own_datum_the_deck():
    """`on_bridge` exists as a type precisely because these objects stand on a
    deck, not on terrain — 425 instances of it. Its base is its own P2, and no
    terrain grid is consulted."""
    from ducklidar import objects as O

    rng = np.random.default_rng(11)
    n = 700
    P = np.column_stack([rng.uniform(0, 1.2, n), rng.uniform(0, 1.2, n),
                         20.0 + rng.uniform(0, 1.1, n)])
    soup, _c, kind = O.on_bridge_model(P[:, 0], P[:, 1], P[:, 2])[0]
    assert kind == O.SOLID
    assert 19.5 < soup[:, 2].min() < 20.0, "based on the deck it stands on"
    assert soup[:, 2].max() > 20.9


def test_every_type_model_reports_its_own_kind():
    """The contract: one named method per taxonomy type, each returning
    (soup, cols, kind). Nothing borrows, and the kind is not the caller's to
    decide."""
    pytest.importorskip("mapbox_earcut")
    from ducklidar import objects as O

    rng = np.random.default_rng(13)
    ring = np.array([[0.0, 0.0], [8.0, 0.0], [8.0, 6.0], [0.0, 6.0]])
    n = 800
    blob = np.column_stack([rng.uniform(0, 6, n), rng.uniform(0, 3, n),
                            rng.uniform(0.2, 3.0, n)])
    cases = {
        "building": O.building_model(ring, 0.0, 9.0),
        "moorage": O.moorage_model(ring, 0.0, 5.0),
        "water": O.water_model([ring], 1.25),
        "terrain": O.terrain_model([ring], np.arange(4) * 0.1),
        "boat": O.boat_model(blob[:, 0], blob[:, 1], blob[:, 2] + 1.0, 0.0),
        "lowveg": O.lowveg_model(blob[:, 0], blob[:, 1], blob[:, 2]),
        "tree": O.tree_model(blob[:, 0], blob[:, 1], blob[:, 2] * 4, 0.0),
        "lamp": O.lamp_model(np.zeros(50), np.zeros(50),
                             np.linspace(0.2, 6.0, 50), 0.0),
    }
    # a type may ship SEVERAL kinds when its parts differ physically: a tree
    # is an opaque trunk under a porous crown, so `solid` + `slab`
    # [2026-08-05]. The rule is that every part states a kind the sun
    # understands, and one this type is allowed to use.
    want = {"building": {O.SOLID}, "moorage": {O.SOLID}, "water": {O.SURFACE},
            "terrain": {O.SURFACE}, "boat": {O.SOLID}, "lowveg": {O.SLAB},
            "tree": {O.SOLID, O.SLAB}, "lamp": {O.SOLID}}
    for name, parts in cases.items():
        assert parts, f"{name}_model produced nothing"
        for soup, cols, kind in parts:
            assert kind in want[name], (name, kind)
            assert len(soup) == len(cols) and len(soup) % 3 == 0
            assert soup.dtype == np.float32 and cols.dtype == np.uint8
    # water and terrain are the same drape at different heights; both surfaces
    assert {p[2] for p in cases["water"]} == {O.SURFACE}


def test_boat_model_never_returns_dock_geometry():
    """`boat_mesh` silently redirects a low float to `dock_mesh` while keeping
    the boat label, so the export's boat/dock split did not mean what it said.
    A barge is a LOW HULL; which one it is was already decided by the map
    partition before the model was called."""
    from ducklidar import objects as O

    rng = np.random.default_rng(17)
    n = 1200
    # a 10 x 3 m barge standing 0.8 m over the water: under boat_mesh's 1.2 m
    P = np.column_stack([rng.uniform(0, 10, n), rng.uniform(0, 3, n),
                         rng.uniform(0.2, 0.8, n)])
    hull = O.boat_model(P[:, 0], P[:, 1], P[:, 2], 0.0)
    assert hull and hull[0][2] == O.SOLID
    # the hull tapers to a bow: the widest and narrowest stations differ
    soup = hull[0][0]
    assert len(soup) > 60, "a hull, not a traced slab"
    assert O.boat_mesh(P[:, 0], P[:, 1], P[:, 2], 0.0, [1, 2, 3]) is not hull


def test_cell_prism_is_shared_by_the_solid_and_the_slab():
    """The one primitive both a grounded solid and a floating plate are made
    of — the only difference is where the bottom sits. Types share PRIMITIVES,
    never each other's recipes."""
    from ducklidar import objects as O

    top = np.full((4, 4), 10.0)
    occ = np.zeros((4, 4), bool)
    occ[1:3, 1:3] = True
    grounded = O.cell_prism(top, 0.0, occ, np.array([0.0, 0.0]), 1.0)
    plate = O.cell_prism(top, top - 1.6, occ, np.array([0.0, 0.0]), 1.0)
    assert len(grounded) == len(plate), "same cells, same triangles"
    assert grounded[:, 2].min() == 0.0 and plate[:, 2].min() == 8.4
    assert grounded[:, 2].max() == plate[:, 2].max() == 10.0
