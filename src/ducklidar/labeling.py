"""Stage 5 on tables — labeling without windows.

The redesign of the lifted rules (see ``docs/rules-operators.md`` for the
full operator inventory): the point table is what's big; the rasters the
rules actually think on never were. At 1 m cells a whole 2 km window is a
few million rows — hundreds of times smaller than RAM. So:

1. **points → cell evidence** — one streaming ``GROUP BY`` over the Parquet
   store (:func:`cell_evidence`). No window, no partition, any area.
2. **global raster operators** — cloth ground, floods, corridors — run
   once on the small cell table (:func:`ground_cells` is the first;
   others arrive per the worksheet walkthrough). The per-window seams the
   windowed rules had (nine independently-relaxed cloths!) disappear.
3. **join back per point** — row-wise by cell id, in SQL.

The lifted rules in :mod:`ducklidar.rules` stay the measured reference:
every operator moved here must reproduce them (the owned-tile equivalence
harness), the same acceptance that guarded the lift itself.
"""
import numpy as np

__all__ = ["ColumnSupport", "cell_components", "cell_evidence", "cell_of", "column_support", "convict_glass", "corridor_cells", "cube_scatter", "forest_graph", "decide_labels", "flood_bodies", "footprint_cells", "glass_geodesic", "glass_geodesic_blocks", "ground_cells", "knn_csr", "map_vetoes", "marina_superstructure", "path_shares", "support_forest", "water_cells"]

#: label codes — indices into the oracle's DECIDE order (rules/layers.py:614).
(GROUND, VEG_LOW, VEG_HIGH, BUILDING, WATER,
 SMALL, FLOATING, BRIDGE, ON_BRIDGE, GLAZING) = range(10)

#: rules/layers.py:41 LAYERS minus "placed" (l.614) — and the ORDER IS the
#: label code above, the codes the oracle writes into out/labels.npz.
DECIDE = ["ground", "veg_low", "veg_high", "building", "water", "small",
          "floating", "bridge", "on_bridge", "glazing"]
#: rules/layers.py:642-643, verbatim. This order IS the rule (see
#: :func:`decide_labels`); equal specificity is a tie the neighbours settle.
SPEC = {"water": 5, "on_bridge": 5, "floating": 4, "bridge": 4, "glazing": 2,
        "building": 2, "veg_high": 2, "veg_low": 2, "small": 2, "ground": 1}
QUORUM = ("building", "veg_high")   # rules/layers.py:661
QUORUM_MIN = 20                     # l.658-660, swept 2/5/10/20/40/80
K_VOTE = 5                          # l.671
UNSET = 255                         # l.646 — label is uint8, so never -1
#: :func:`decide_labels`'s second return, the how-decided byte.
SETTLED_SINGLE, SETTLED_SPEC, SETTLED_VOTE, SETTLED_RESIDUE = 0, 1, 2, 3

#: root systems of :func:`support_forest`. 0 is "unreached" and never a key.
SYS_NONE, SYS_GROUND, SYS_WATER, SYS_DECK = 0, 1, 2, 3
#: path-crossing categories. The oracle histograms all ten labels
#: (rules/layers.py:774) but reads only two shares (l.791-792), so three
#: columns are the whole rule.
CAT_OTHER, CAT_WALL, CAT_FOLIAGE = 0, 1, 2


def cell_of(x, y, bbox, pix=1.0):
    """Flat cell id for coordinates in `bbox` — the join key between the
    point table and every cell table. Column-major on x: ``cy * nx + cx``."""
    nx = int(np.ceil((bbox[2] - bbox[0]) / pix))
    cx = np.minimum(((np.asarray(x) - bbox[0]) / pix).astype(np.int64), nx - 1)
    cy = ((np.asarray(y) - bbox[1]) / pix).astype(np.int64)
    return cy * nx + cx


def cell_evidence(files, bbox, pix=1.0):
    """The cell table: per-cell aggregates from the whole store, one streaming pass.

    Returns a pyarrow table — ``cell, n, min_z, max_z, split_share,
    n_ground, n_water`` — the shared evidence the regional operators think
    on. Noise classes are excluded, matching :func:`ducklidar.read`.
    """
    import duckdb

    nx = int(np.ceil((bbox[2] - bbox[0]) / pix))
    src = "read_parquet([" + ", ".join(f"'{f}'" for f in files) + "])"
    con = duckdb.connect()
    return con.execute(f"""
        select cast(floor((y - ?) / ?) as bigint) * ? +
               least(cast(floor((x - ?) / ?) as bigint), ? - 1)   as cell,
               count(*)                                           as n,
               min(z)                                             as min_z,
               max(z)                                             as max_z,
               avg(cast(return_number < number_of_returns as double))
                                                                  as split_share,
               sum(cast(classification = 2 as bigint))            as n_ground,
               sum(cast(classification = 9 as bigint))            as n_water
        from {src}
        where x >= ? and x < ? and y >= ? and y < ?
          and classification not in (7, 18)
        group by 1
    """, [bbox[1], pix, nx, bbox[0], pix, nx,
          bbox[0], bbox[2], bbox[1], bbox[3]]).fetch_arrow_table()


def ground_cells(cells, bbox, pix=1.0, pts=None, **csf):
    """Bare-earth elevation per cell — the cloth, run once, globally.

    Pass `pts` (the working points: x, y, z, and return columns if
    available) and the cloth runs at POINT level, like the measured rules:
    dock and hull points classify non-ground and drop out, so the surface
    under a marina interpolates from the surrounding *water* evidence down
    to the plane. The cell-minimum shortcut (cloth on per-cell minima) was
    measured insufficient — a flat raft at +0.5 m reads as ground and the
    water body loses the marina (recall 0.17 vs the oracle). Without
    `pts`, the shortcut is still used (cheap, fine for terrain-only uses).

    Returns ``(cell_ids, ground_z, filled_grid)``; ``**csf`` forwards to
    :func:`ducklidar.ground_filter`.
    """
    from .ground import ground_filter
    from .grid import shape_for
    from .dem import fill

    cell = np.asarray(cells["cell"])
    ny, nx = shape_for(bbox, pix)
    if pts is not None:
        keep = ground_filter(pts, **csf)
        pc = cell_of(pts["x"], pts["y"], bbox, pix)[keep]
        grid = np.full(ny * nx, np.nan)
        grid = np.full(ny * nx, np.inf)
        np.minimum.at(grid, pc, np.asarray(pts["z"])[keep])
        grid[np.isinf(grid)] = np.nan
    else:
        min_z = np.asarray(cells["min_z"])
        cx = (cell % nx).astype(np.float64)
        cy = (cell // nx).astype(np.float64)
        synth = {"x": bbox[0] + (cx + 0.5) * pix,
                 "y": bbox[1] + (cy + 0.5) * pix,
                 "z": min_z.astype(np.float64),
                 "classification": np.zeros(len(cell), np.uint8)}
        keep = ground_filter(synth, **csf)
        grid = np.full(ny * nx, np.nan)
        grid[cell[keep]] = min_z[keep]
    filled, _ = fill(grid.reshape(ny, nx), pix)
    # (cell values, and the FULL filled grid — the flood must walk across
    # cells the laser never hit: open water absorbs the beam, and a third
    # of a creek can be returnless. Measured: occupied-only flooding
    # reached 102k of the oracle's 310k body cells.)
    return cell, filled.ravel()[cell], filled


def cell_components(cells, mask, bbox, pix=1.0, *, connectivity=8):
    """Connected components of a cell mask — floods, bodies, regions, globally.

    `mask` is a boolean per row of `cells` (the caller's predicate: water-ish,
    sparse-dark, corridor…); returns an int array aligned to `cells`: component
    id per masked cell (1..n), 0 elsewhere. Stage 4's idea at raster scale —
    on a few million cells this is one `scipy.ndimage.label` call, so region
    questions (which water body? which flooded area?) are answered once for
    the whole area, never per window.

    The conditioned water level then falls out in SQL: join points to their
    cell's body and `GROUP BY (body, point_source_id)` — one level per
    (water body × flight line), the lesson of the tide measurement.
    """
    from scipy import ndimage as ndi

    from .grid import shape_for

    cell = np.asarray(cells["cell"])
    ny, nx = shape_for(bbox, pix)
    grid = np.zeros(ny * nx, bool)
    grid[cell[np.asarray(mask)]] = True
    structure = np.ones((3, 3)) if connectivity == 8 else None
    lab, n = ndi.label(grid.reshape(ny, nx), structure=structure)
    return lab.ravel()[cell]


class ColumnSupport:
    """What :func:`column_support` measured — the occupancy of every column.

    ``occ[row, k]`` is True where the cell has matter in height band ``k``
    above its own ground, band ``k`` covering ``[k*step, (k+1)*step)`` metres
    (half-open, so 0.5 m with ``step=0.5`` is band 1, not band 0). ``row``
    maps a flat cell id to its row; cells with no evidence at all map to a
    trailing all-empty row, so an unknown column reads as air and never
    raises.
    """

    def __init__(self, cells, occ, row, step, floor_band, min_gap):
        self.cells, self.occ, self.row = cells, occ, row
        self.step, self.floor_band, self.min_gap = step, floor_band, min_gap

    def _span(self, h):
        """Half-open band range ``[k0, k1)`` covering ``[floor_band, h)``."""
        k0 = int(np.floor(self.floor_band / self.step))
        k1 = int(np.ceil(h / self.step))
        return k0, min(max(k1, k0), self.occ.shape[1])

    def gap_below(self, cell, h):
        """Largest empty vertical run between the floor band and `h`, metres.

        `cell` is a flat cell id or an array of them; `h` is a scalar height
        above ground. A column with no evidence returns the whole span.
        """
        k0, k1 = self._span(h)
        air = ~self.occ[self.row[np.asarray(cell)], k0:k1]
        run = np.zeros(air.shape[:-1], np.int32)
        best = np.zeros(air.shape[:-1], np.int32)
        for k in range(air.shape[-1]):        # ≤ max_h/step columns, vectorised
            run = (run + 1) * air[..., k]     # over however many cells are asked
            best = np.maximum(best, run)
        return best * self.step

    def supported(self, cell, h):
        """Is the column continuous — no gap wider than ``min_gap`` — from the
        floor band up to `h`? A truck's column is; a one-storey eave's, with
        walkable air between the yard and the overhang, is not."""
        return self.gap_below(cell, h) <= self.min_gap


def column_support(voxels, cells, bbox, pix=1.0, *, floor_band=0.3,
                   min_gap=1.5, step=0.5, max_h=30.0):
    """The vertical profile per cell — the context the per-point rules lack.

    ``placed`` (rules/layers.py:182-185, 609-612) cannot tell a truck from a
    one-storey eave, because in 2-D both are "ground below, roof above": the
    layer is computed and then thrown away (``DECIDE`` drops it) at a measured
    cost of building IoU 0.830 → 0.766. The difference is the COLUMN. A truck
    is matter all the way up; under an eave there is air you can walk through.

    A table operation over the two cell-level tables — the voxel table (one
    row per occupied 1 m × ``step`` band, ``level = floor(z / step)``) and the
    cell table (``ground_z``) — so it costs a join, not a point cloud.
    Heights are ABOVE GROUND: band ``k`` of a cell is voxel level
    ``k + floor(ground_z / step)``, i.e. k = 0 is the band the ground surface
    itself falls in and bands are half-open upward. Cells with a non-finite
    ``ground_z`` have no datum and are left empty.

    Returns a :class:`ColumnSupport`. `max_h` bounds what can be asked (the
    matrix is ``ncells × max_h/step`` bools — 60 MB for the green window at
    the default); heights above it read as air.
    """
    from .grid import shape_for

    ny, nx = shape_for(bbox, pix)
    cid = np.asarray(cells["cell"]).astype(np.int64)
    gz = np.asarray(cells["ground_z"], float)
    m, nlev = len(cid), int(np.ceil(max_h / step))

    row = np.full(ny * nx, m, np.int64)      # default row m = the empty column
    inbox = (cid >= 0) & (cid < ny * nx)
    row[cid[inbox]] = np.flatnonzero(inbox)

    occ = np.zeros((m + 1, nlev), bool)
    vc = np.asarray(voxels["cell"]).astype(np.int64)
    vl = np.asarray(voxels["level"]).astype(np.int64)
    ok = (vc >= 0) & (vc < ny * nx)
    r = np.where(ok, row[np.where(ok, vc, 0)], m)
    gpad = np.r_[gz, np.nan]                 # row m has no datum, hence no band
    k = vl - np.floor(gpad[r] / step)
    ok &= np.isfinite(k)
    k = np.where(ok, k, 0).astype(np.int64)
    ok &= (k >= 0) & (k < nlev)
    occ[r[ok], k[ok]] = True
    return ColumnSupport(cid, occ, row, step, floor_band, min_gap)


def water_cells(cells, ground_z, bbox, pix=1.0):
    """The oracle's water construction (rules/layers.py), on cell tables.

    Ported verbatim from the measured rules — sparse-and-dark seeds, size
    gate, local-sparsity vote — up to the flood, which is
    :func:`flood_bodies` (per-body levels, the tide lesson). Returns
    ``(seeds, region)``: boolean per row of `cells`. `ground_z` is
    :func:`ground_cells`' filled surface aligned to `cells`.
    """
    from scipy import ndimage as ndi

    from .grid import shape_for
    from .rules.pipeline import (WATER_CLOSE, WATER_DENS, WATER_HAG,
                                 WATER_LOCAL, WATER_MIN_CELLS, WATER_VOTE,
                                 WATER_VOTE_WIN, WATER_WIN)

    cell = np.asarray(cells["cell"])
    ny, nx = shape_for(bbox, pix)
    dens = np.zeros(ny * nx)
    dens[cell] = np.asarray(cells["n"])
    low = np.zeros(ny * nx, bool)                 # a return within WATER_HAG of ground
    low[cell] = (np.asarray(cells["min_z"]) - np.asarray(ground_z)) < WATER_HAG

    sparse = ((dens <= WATER_DENS) & (dens > 0) & low).reshape(ny, nx)
    sparse = ndi.binary_closing(sparse, np.ones((2 * WATER_CLOSE + 1,) * 2))
    wlab, _ = ndi.label(sparse)
    wsz = np.bincount(wlab.ravel())
    region = np.isin(wlab, np.flatnonzero(wsz >= WATER_MIN_CELLS)) & (wlab > 0)
    local = ndi.uniform_filter(dens.reshape(ny, nx), size=WATER_WIN) <= WATER_LOCAL
    both = region | local
    agree = ndi.uniform_filter(both.astype(float), size=WATER_VOTE_WIN) >= WATER_VOTE
    seeds = region | (both & agree)
    return seeds.ravel()[cell], region.ravel()[cell]


def corridor_cells(ways, cells, ground_z, bbox, pix=1.0):
    """The oracle's OSM bridge-corridor construction (rules/layers.py), on cell tables.

    Ported verbatim from the measured rules: each way buffered at ITS OWN
    width (lanes tell the carriageway width; a footway is ~2.5 m of
    structure), carriageway ways authoritative at ANY height (the at-grade
    ramp is the bridge — that is exactly what OSM knows and geometry
    cannot), FOOT ways counted only where genuinely elevated (> 4 m above
    the terrain) — a park footbridge a metre over a pond is a structure ON
    the ground, not a bridge.

    `ways` is the parsed osm_bridges.json dict; `ground_z` is
    :func:`ground_cells`' filled surface aligned to `cells`. Returns
    ``(mask, grid)``: boolean per row of `cells`, and the full boolean
    grid (south-up, like every grid here — the oracle rasters north-up,
    hence the row formula differs from the lab's by orientation only).
    """
    from scipy import ndimage as ndi

    from .grid import shape_for

    cell = np.asarray(cells["cell"])
    ny, nx = shape_for(bbox, pix)

    def radius(w):
        if w.get("highway") in ("footway", "cycleway", "path", "steps"):
            return 2.5
        try:
            return max(3.0, int(w.get("lanes") or 2) * 3.5 / 2 + 1.5)
        except ValueError:
            return 5.0

    groups = {}
    for w in ways["ways"]:
        foot = w.get("highway") in ("footway", "cycleway", "path", "steps")
        r = radius(w)
        line = groups.setdefault((r, foot), np.zeros((ny, nx), bool))
        c = np.asarray(w["coords"], float)
        for (x0, y0), (x1, y1) in zip(c[:-1], c[1:]):
            k = max(int(np.hypot(x1 - x0, y1 - y0) / 0.5), 1) + 1
            xs, ys = np.linspace(x0, x1, k), np.linspace(y0, y1, k)
            rows = np.floor((ys - bbox[1]) / pix).astype(int)   # south-up
            cols = np.floor((xs - bbox[0]) / pix).astype(int)
            ok = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)
            line[rows[ok], cols[ok]] = True
    corr_major = np.zeros((ny, nx), bool)
    corr_foot = np.zeros((ny, nx), bool)
    for (r, foot), line in groups.items():
        if not line.any():
            continue        # a width class whose ways all fall outside the box:
        # distance_transform_edt of an all-True array measures distance to the
        # OUTSIDE, so an empty line paints a radius-r quarter-disc in the corner.
        # The oracle carries that artifact (83 phantom cells in the tile's NW
        # corner, reproduced here pixel for pixel — it is scipy, not a rule).
        m = ndi.distance_transform_edt(~line) <= r / pix
        if foot:
            corr_foot |= m
        else:
            corr_major |= m
    elev = np.zeros(ny * nx, bool)                # cell top > 4 m above ground
    elev[cell] = (np.asarray(cells["max_z"]) - np.asarray(ground_z)) > 4.0
    corr = corr_major | (corr_foot & elev.reshape(ny, nx))
    return corr.ravel()[cell], corr


def flood_bodies(cells, seeds, surface_grid, levels, bbox, pix=1.0, tol=0.4,
                 deck_tol=None):
    """The body flood, per seed-component with ITS level (the tide lesson).

    The oracle flooded with one global zw; here each seed component floods
    with its own `levels[body]` through cells whose filled surface sits at
    or below level+`tol`, then holes are filled — water hidden under docks
    and hulls joins its body. Returns body id per row of `cells` (0 = dry).

    `deck_tol` (default None = off, the oracle's behaviour) opens a SECOND,
    looser admission for cells the cloth settled on a FLOATING deck: filled
    <= level+`deck_tol`, but only where the cell has returns and NOT ONE of
    them is vendor class 2 — `cells["n_ground"] == 0`. That is the guard
    against flooding dry land: a beach, a seawall or a riprap slope inside
    the same height band is classed ground and stays dry, while a pontoon,
    a finger float or a moored hull is not ground and lets the body under
    it. A marina roofed by its own floats never reaches the strict tol
    because the cloth drapes on the deck, not the water.

    Measured on the green window (L = -1.072, deck_tol 1.0) against the
    Granville Island pontoon marina at 490070, 5457818 (609 dry cells the
    stage-5 labels call floating, dilated to 868):

        tol 0.4 alone            0/868 admitted      the marina is lost
        filled <= L+1.0          771/868 (88.8%)     but 16.4% of the added
                                                     cells hold class-2 ground
        + n_ground == 0 guard    742/868 (85.5%)     0.1% class-2 ground

    The deck tolerance is a plateau, not a fit: 0.8/1.0/1.2/1.5 recover
    742/742/744/748 of the same 868 cells. Whole-tile cost of 1.0 is
    +19,431 cells (+5.6%), every added component with DEM p95 <= 0.59 m —
    all of it at sea level, none of it inland.
    """
    from scipy import ndimage as ndi

    from .grid import shape_for

    cell = np.asarray(cells["cell"])
    ny, nx = shape_for(bbox, pix)
    seed_body = cell_components(cells, seeds, bbox, pix)
    surface = np.asarray(surface_grid, float).ravel()   # full grid: returnless
    surface = np.where(np.isnan(surface), np.inf, surface)   # water floods too

    # ONE flood from ALL seeds (credible or not): isolated pockets seed their
    # own chambers — "its own sparsity already testifies it is water" — while
    # the level comes from the credible bodies (median; they are one sea,
    # ±5 cm here). Bodies are components of the flooded region afterwards.
    allseed = np.zeros(ny * nx, bool)
    allseed[cell[np.asarray(seeds)]] = True
    L = None
    if levels:
        L = float(np.median(list(levels.values())))
        mask = allseed | (surface <= L + tol)
        if deck_tol is not None:
            # the floating-deck admission, guarded by "no vendor ground return
            # in this cell" — see the docstring's measurement
            deck = np.zeros(ny * nx, bool)
            deck[cell[np.asarray(cells["n_ground"]) == 0]] = True
            mask |= deck & (surface <= L + deck_tol)
    else:
        mask = allseed
    print(f'    [flood] L={L} seeds={int(allseed.sum()):,} '
          f'mask={int(mask.sum()):,}', flush=True)
    flooded = ndi.binary_propagation(allseed.reshape(ny, nx),
                                     mask=mask.reshape(ny, nx))
    flooded = ndi.binary_fill_holes(flooded).ravel()
    lab, _ = ndi.label(flooded.reshape(ny, nx), structure=np.ones((3, 3)))
    # per-cells-row ids AND the full grid: open water absorbs the laser, so
    # most of a body is returnless — occupied-only accounting undercounts
    # the body 3x (measured: recall 0.35 with a correct flood)
    return lab.ravel()[cell], lab


def knn_csr(edge_files, lo, hi, *, extra=None, block=None, metric=True,
            con=None, batch=4_000_000, mem="2GB", tmp=None):
    """Stage 2's kNN edge table of one area, from Parquet into a CSR graph.

    THE FILTER IS A RANGE, NOT A JOIN. Stage 2's ``pid`` is global — the
    sidecar's own row number plus that file's offset in the registry of
    sorted sidecar paths — and a sidecar is one tile, so a tile's points are
    a *contiguous* pid interval. ``src >= lo and src < hi`` prunes the
    1.92 B-row table by row-group statistics alone: **[measured]** 1.3 s to
    count the owned tile's 426,761,290 edges. Node id is ``pid - lo``, dense.

    Edges *leaving* the area are dropped, which is what keeps the ids dense
    (157,283 of the tile's 426,918,573 — all within the 2.5 m cap of the
    boundary, where the oracle's tile-local KD-tree would have substituted an
    in-tile 10th neighbour instead). Score the interior for that reason.

    Built by STREAMING an ``order by src`` into preallocated CSR arrays, not
    by ``csr_matrix((w, (src, dst)))``. Not premature — **[measured]** at
    10 M nodes / 99.8 M edges the coo route peaks at 3.9 GB with every byte
    scaling (16.9 GB at tile scale, on a box with 19 GB free and a lab
    history of three SIGKILLs), while this route peaks at 4.7 GB of which
    2 GB is a DuckDB sort buffer that does *not* scale. ``float64`` up front
    because scipy's ``validate_graph`` casts to it anyway, and would do that
    as a second full copy of a resident graph.

    ``extra`` = ``(src, dst, len)`` in the same node id space, unioned into
    the sort — the oracle's MST edges (rules/mesh_graph.py), which it
    merges so that every point *has* a path (layers.py:746-754). ``block``
    is a boolean per node: those nodes' edges are dropped, which is how the
    oracle's three blocked experiments delete a class from the graph.
    ``metric=False`` gives unit weights, i.e. a hop graph.
    """
    import duckdb
    import pyarrow as pa
    from scipy.sparse import csr_matrix

    n = hi - lo
    own = con is None      # an unclosed connection whose spill files are being
    con = con or duckdb.connect()      # reaped trips a DuckDB assert at exit
    # set, not config=: a caller's connection must get the cap too, and the
    # cap is the only thing keeping the sort out of the Dijkstra's headroom
    con.execute(f"set memory_limit='{mem}'")
    con.execute(f"set temp_directory='{tmp or '/tmp/ducklidar_sort'}'")
    files = ", ".join(f"'{f}'" for f in np.atleast_1d(edge_files))
    body = (f"select cast(src - {lo} as integer) src, "
            f"cast(dst - {lo} as integer) dst, len "
            f"from read_parquet([{files}]) "
            f"where src >= {lo} and src < {hi} and dst >= {lo} and dst < {hi}")
    if extra is not None:
        con.register("extra_edges", pa.table(
            {"src": np.asarray(extra[0], np.int32),
             "dst": np.asarray(extra[1], np.int32),
             "len": np.asarray(extra[2], np.float32)}))
        body += " union all select src, dst, len from extra_edges"
    m, = con.execute(f"select count(*) from ({body})").fetchone()

    a = np.empty(m, np.int32)
    ind = np.empty(m, np.int32)
    dat = np.empty(m, np.float64)      # float64 now, or scipy copies it later
    keep = None if block is None else ~np.asarray(block)
    at = 0
    for b in con.execute(f"{body} order by src").fetch_record_batch(batch):
        s = b.column(0).to_numpy(zero_copy_only=False)
        d = b.column(1).to_numpy(zero_copy_only=False)
        w = b.column(2).to_numpy(zero_copy_only=False)
        if keep is not None:
            k = keep[s] & keep[d]
            s, d, w = s[k], d[k], w[k]
        j = at + len(s)
        a[at:j], ind[at:j] = s, d
        dat[at:j] = w if metric else 1.0
        at = j
    assert block is not None or at == m, (at, m)
    if own:
        con.close()
    a, ind, dat = a[:at], ind[:at], dat[:at]
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(a, minlength=n), out=indptr[1:])
    del a
    return csr_matrix((dat, ind, indptr), shape=(n, n))


def forest_graph(stage2_dir, lo, hi, **kw):
    """G' = (V, E ∪ B), the COMPLETED graph — the only graph the forest runs on.

    The union of the kNN edges with the MST edges is **part of the
    algorithm, not an option** (Kaveh, 2026-08-02). G alone is disconnected
    by construction (the 2.5 m cap), so a vertex in a component holding no
    seed has no path to any root and the forest is simply undefined there —
    measured: 0.78 % of the reference tile, and they are the boats, the
    detached crowns and the isolated roofs, i.e. exactly the objects whose
    support is the question. Refusing to run without the MST edges is
    therefore a correctness guard, not a convenience.

    `stage2_dir` holds knn.parquet and mst.parquet; both are read as
    one edge set (`knn_csr` takes a list). Extra keywords go to
    :func:`knn_csr` — `block=` to delete a class, `metric=` for the weight.
    """
    import pathlib

    d = pathlib.Path(stage2_dir)
    edges, mst = d / "knn.parquet", d / "mst.parquet"
    if not edges.exists():
        raise FileNotFoundError(f"{edges} — stage 2 has not run")
    if not mst.exists():
        raise FileNotFoundError(
            f"{mst} — the MST completion is missing, and the forest is "
            f"undefined on the un-completed graph (see docs/design/"
            f"stage2-graph.md task 2.6). Run tools/stage2_mst.py after "
            f"stage 4.")
    return knn_csr([str(edges), str(mst)], lo, hi, **kw)


def support_forest(G, roots, cats, *, ncat=3, limit=np.inf, rounds=25):
    """One multi-source expansion over the kNN graph — the attachment table.

    Ported from rules/layers.py: the mesh Dijkstra of the path-entry
    refinement (l.760) with its pointer-doubling path histogram (l.770-794),
    generalised to several root SYSTEMS expanding together, the first to
    arrive claiming the point. The oracle spreads one idea over five blocks —
    a voxel reachability acquittal (l.369-395), a metric conviction with the
    building class deleted (l.396-530), a corridor stranding with
    building∪bridge deleted (l.548-580) and this refinement — and they differ
    only in what the graph is and what is read off it. "Blocker deleted"
    becomes "the path crossed a blocker", which is the form the oracle itself
    measured as the better instrument (93.7% at 62.8k points against the
    deletion form's 96.7% at 5.9k).

    Args:
        G: ``(n, n)`` CSR from :func:`knn_csr`. Each point's ≤10 neighbours
            within 2.5 m, written once by the owner of ``src``, walked
            UNDIRECTED (``directed=False``) exactly as every oracle block
            does (l.454, l.566, l.760).
        roots: ``{system_code: mask or index array}`` — the root sets,
            expanding simultaneously. The oracle seeds ground and nothing
            else (l.760); overlapping sets resolve to the FIRST key, stated
            rather than silently settled by argmin.
        cats: ``uint8[n]``, what a path CROSSING this node counts as. For the
            oracle's rules: :data:`CAT_WALL` for building, :data:`CAT_FOLIAGE`
            for veg_low|veg_high, :data:`CAT_OTHER` for the rest — the exact
            two shares l.791-792 reads out of its ten-column histogram.
        limit: stop expanding beyond this cost (the conviction's
            ``limit=60.0``). ``inf`` for the refinement, whose paths are
            unbounded — which is also why the oracle's 250 m spatial blocking
            (l.417-426) is *not* available to it: that trick is valid only
            with a reach limit.

    Blockers and weights live in :func:`knn_csr` because both are graph
    transforms: ``block=`` deletes a class, and ``metric=True`` makes
    ``dist`` the geodesic the oracle's 60 m bar is stated in and the parent
    tree the one the refinement was swept on. **[measured]** the hop tree is
    a *different* tree and does not inherit the sweep: on the owned tile it
    shares only 31.7% of the metric tree's parents, and while its refinement
    counts look fine (98.9% / 101.4% of the oracle's) the masks agree at only
    IoU 0.713 / 0.866 against the metric tree's 0.991 / 0.996. In a kNN graph
    with edges from 0.05 to 2.5 m, hop-BFS prefers LONG edges and Dijkstra
    short ones, and crown-vs-wall is exactly a question about which matter the
    path hugged. Ship ``metric=True`` unless the rules are re-swept.

    Returns a dict of arrays, one row per node — the attachment table:
    ``system`` uint8 (0 = unreached), ``root`` int32 (-1 = unreached),
    ``dist`` float32 metres, ``depth`` int32 hops (0 at a root, -1 unreached),
    ``cross`` int32 ``(n, ncat)`` — what the path crossed, per category —
    and ``pred`` int32, the predecessor (-1 at a root and where unreached).
    """
    from scipy.sparse.csgraph import dijkstra

    n = G.shape[0]
    ridx, rsys = [], []
    for k, v in roots.items():
        v = np.asarray(v)
        i = np.flatnonzero(v) if v.dtype == bool else v.astype(np.int64)
        ridx.append(i)
        rsys.append(np.full(len(i), k, np.uint8))
    ridx, rsys = np.concatenate(ridx), np.concatenate(rsys)
    order = np.argsort(ridx, kind="stable")
    ridx, rsys = ridx[order], rsys[order]
    first = np.r_[True, ridx[1:] != ridx[:-1]]
    ridx, rsys = ridx[first], rsys[first]

    dist, pred, srcnode = dijkstra(G, directed=False, indices=ridx,
                                   min_only=True, return_predecessors=True,
                                   limit=limit)
    # PEAK IS HERE: directed=False makes scipy hold the TRANSPOSE too, so the
    # graph is resident twice for the length of the call (5.1 GB each at tile
    # scale). Drop G right after this returns, before the doubling below
    # allocates its gather temporary.
    #
    # scipy's NULL_IDX is -9999 for both: a ROOT has pred = -9999 and
    # sources = itself, an UNREACHED node has both -9999.
    reached = srcnode >= 0
    sysmap = np.zeros(n, np.uint8)
    sysmap[ridx] = rsys
    system = np.zeros(n, np.uint8)
    system[reached] = sysmap[srcnode[reached]]
    root = np.where(reached, srcnode, -1).astype(np.int32)
    del sysmap

    # ---- the path histogram, layers.py:770-794 verbatim in structure ------
    # S_i starts as the one-hot of i's PARENT's category and is accumulated up
    # the predecessor forest by pointer doubling, so the final row counts every
    # node on the path EXCLUDING i and INCLUDING the root. The row sum is
    # therefore the hop depth — no separate counter.
    J = pred.astype(np.int32, copy=True)
    isroot = J < 0                       # roots AND unreached, as the oracle has it
    J[isroot] = np.flatnonzero(isroot).astype(np.int32)
    S = np.zeros((n, ncat), np.int32)
    nz = ~isroot
    S[nz, np.asarray(cats)[pred[nz]]] = 1
    del nz
    for _ in range(rounds):
        J2 = J[J]
        if (J2 == J).all():
            break                        # S[root] is all-zero, so the skipped
        S += S[J]                        # final add would be a no-op anyway
        J = J2
    del J, J2

    depth = S.sum(axis=1).astype(np.int32)
    depth[~reached] = -1                 # unreached: no path, not a zero path
    return {"system": system, "root": root, "dist": dist.astype(np.float32),
            "depth": depth, "cross": S,
            "pred": np.where(pred < 0, -1, pred).astype(np.int32)}


def path_shares(f):
    """``(building share, vegetation share)`` of each point's path — l.790-794.

    The shares are zeroed at roots and wherever there is no path, so an
    unreached point is never refined (the oracle's ``_bshm[_rootm] = 0.0``).
    """
    tot = np.maximum(f["cross"].sum(axis=1), 1)
    bsh = f["cross"][:, CAT_WALL] / tot
    vsh = f["cross"][:, CAT_FOLIAGE] / tot
    dead = f["depth"] <= 0
    bsh[dead] = 0.0
    vsh[dead] = 0.0
    return bsh, vsh


def path_shares_from_pred(pred, depth, cats, *, ncat=3, rounds=25):
    """``(building share, vegetation share)`` from a STORED predecessor forest.

    :func:`path_shares` reads the histogram :func:`support_forest` accumulated
    while it walked. Once that forest has been written to ``forest.parquet``
    the walk is over — but ``pred`` **is** the walk's whole result, and the
    histogram is a pointer-doubling partial sum up it (rules/layers.py:770-782).
    So the shares can be re-derived for any category assignment without
    touching the graph at all.

    That matters twice. The Dijkstra and the 12 GB edge table are already paid
    for and on disk, so this turns an >18 GB rebuild into a ~2 GB array pass
    (measured: the rebuild was SIGKILLed by an 18 GB cap on 2026-08-03, from an
    11 GB baseline). And re-deriving against the CURRENT run's labels is *more*
    faithful than reusing the stored ``crossed_wall`` / ``crossed_foliage``
    columns, which were accumulated from whatever labels the forest run held.

    ``pred`` is local node ids (-1 at a root or where unreached); ``cats`` is
    ``uint8[n]`` exactly as :func:`support_forest` takes it.
    """
    n = len(pred)
    ids = np.arange(n, dtype=np.int64)
    J = np.asarray(pred, np.int64).copy()
    root = J < 0
    J[root] = ids[root]
    del ids
    S = np.zeros((n, ncat), np.int32)
    nz = ~root
    S[nz, np.asarray(cats)[J[nz]]] = 1
    del nz, root
    for _ in range(rounds):
        J2 = J[J]
        if (J2 == J).all():
            break
        S += S[J]
        J = J2
    del J
    tot = np.maximum(S.sum(axis=1), 1)
    bsh = S[:, CAT_WALL] / tot
    vsh = S[:, CAT_FOLIAGE] / tot
    dead = np.asarray(depth) <= 0
    bsh[dead] = 0.0
    vsh[dead] = 0.0
    return bsh, vsh


def cube_scatter(flat, hag, scatter, shape, zbins=60):
    """The oracle's "cube": mean scatter in the 3×3×3 m voxel around a point.

    rules/layers.py:464-471 verbatim (l.380-382 and l.484-488 build the same
    voxel id). Every path rule is guarded by it — majority topology alone
    measured 82.2% / 71.7% survey agreement, below the bar — so the forest
    cannot be scored without it.
    """
    from scipy import ndimage as ndi

    ny, nx = shape
    zb = np.clip(hag, 0, zbins - 0.1).astype(np.int32)
    vox = np.asarray(flat, np.int64) * zbins + zb
    nv = ny * nx * zbins
    ss = np.bincount(vox, weights=scatter, minlength=nv).reshape(ny, nx, zbins)
    cn = np.bincount(vox, minlength=nv).astype(float).reshape(ny, nx, zbins)
    k3 = np.ones((3, 3, 3))
    return (ndi.convolve(ss, k3, mode="constant")
            / np.maximum(ndi.convolve(cn, k3, mode="constant"), 1)).ravel()[vox]


def glass_geodesic(edge_files, lo, hi, sub, ground, *, limit=60.0, **kw):
    """Multi-source geodesic to ground through NON-BUILDING matter — l.426-472.

    The conviction's graph feature: metres of shortest path, over the same
    kNN(k=10, cap 2.5 m) edges stage 2 already wrote, from any ground-claimed
    point to each point of ``sub`` (the oracle's ``_zone & ~L["building"]``).
    A crown's shell connects sideways-and-down through its own foliage; glass
    is sealed to the roof and only a detour reaches it.

    ``limit`` is not an optimisation. scipy returns ``inf`` beyond it, which
    is exactly why the oracle's two historically-distinct convictions —
    "unreachable" and "geodesic >= 60 m" — are one test in the shipped code
    (l.473). An unbounded Dijkstra would convict FEWER points.

    The oracle cuts the cloud into 250 m blocks with a 62 m halo and runs a
    per-block cKDTree because 31.4 M points as one graph was two SIGKILLs
    (l.417-426). Here the graph is already built — :func:`knn_csr` streams it
    from Parquet — and ``block=~sub`` deletes the blocker class the same way,
    so one bounded Dijkstra over the whole area gives the same verdicts
    without the decomposition. **The two graphs are not identical**: the
    oracle re-runs kNN *on the subset*, so a point beside a wall gets ten
    non-building neighbours up to 2.5 m away, while the stage-2 table's kNN
    is over ALL points and filtering endpoints only removes edges. Ours is
    therefore strictly sparser -> more points unreachable -> over-conviction.
    Measured on the owned tile, that costs 1.9% of the verdicts (see
    ``out/stage5_dev_conviction.log``); the fallback, if it ever matters, is
    a stage-2 edge table built per blocker class.

    ``sub`` and ``ground`` are boolean per NODE (``pid - lo``), and so is the
    returned distance: ``inf`` outside ``sub`` and wherever no ground lies
    within ``limit`` metres of path.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    sub = np.asarray(sub)
    G = knn_csr(edge_files, lo, hi, block=~sub, **kw)
    # knn_csr trims its preallocation with VIEWS, so the full-length buffers
    # (6.8 GB at tile scale) stay alive behind the CSR — one copy frees them
    # before scipy allocates the transpose that ``directed=False`` needs.
    d, i, p = G.data.copy(), G.indices.copy(), G.indptr
    shape = G.shape
    del G
    return dijkstra(csr_matrix((d, i, p), shape=shape), directed=False,
                    indices=np.flatnonzero(np.asarray(ground) & sub),
                    min_only=True, limit=limit)


def convict_glass(flat, hag, shape, cont, sub, geo, cube, veg_high, building,
                  inside_a, scatter, exg, not_last, *, zbins=60,
                  cube_path=0.25, cube_solo=0.15, cube_roof=0.15, pane=0.5,
                  rescue_fr=0.5, rescue_green=0.02, scatter_min=0.2,
                  tall=2.0, split=6, geo_bar=60.0):
    """The glazing TRIBUNAL — rules/layers.py:396-546, in the oracle's order.

    Glazing is not a class in the lifted rules, it is a trial. The assembly
    stage ends with ``L["glazing"] &= L["veg_high"]`` (l.348), so every
    defendant is a point BOTH layers claim — a crown lapping a roof, or a
    skylight — and ``cont`` here IS ``L["glazing"]``. The only verdict the
    trial can hand down is ``veg_high &= ~convicted``: a convicted point
    loses its vegetation claim, keeps its glazing claim, and falls out of the
    decision as a single-claim glazing seed; an acquitted one stays a tie for
    the neighbour vote to settle.

    Four routes convict, and the ORDER is load-bearing (it is what most of
    the porting bugs turned out to be):

    1. the geodesic (:func:`glass_geodesic`) at ``>= 60`` m, guarded by the
       cube at ``< 0.25`` — the path route alone degraded to 68% survey
       purity after the local-roof widening (l.459-462);
    2. the cube standalone at ``< 0.15``;
    3. the PANE vote — 26-connected components of the glazing-claimed voxels;
       a pane with >= 50% of its points individually convicted convicts
       WHOLE. Majority, not any-touch: panes touch overhanging crowns;
    4. the RESCUE, subtractive and LAST, so it can un-convict what the pane
       vote just convicted: scatter-dense AND green is a roof garden.

    Then two local blocks of the same section, whose inputs are the layers
    the trial just edited — hence they live here and not in the caller:
    the ROOF-TREE FIX (veg inside an outline with solid cube surroundings ->
    building, the oracle's largest single find) and the LAST-RETURN FILTER
    (a splitting building claim in a cell where veg out-claims building 6:1
    is a crown). Counting either before the trial fires it on a different
    population.

    ``geo`` is ``inf`` outside ``sub``, and the oracle's own expression
    (l.473) reads ``inf >= 60`` as a conviction — a contested point that is
    also building-claimed convicts with no graph evidence at all. Ported as
    written, not fixed.

    ``flat`` may be either orientation (:func:`cell_of`'s south-up ids or
    ``dl.cell_index``'s north-up ones): every stencil here is symmetric, so
    the operator only needs the caller to be self-consistent — but the grid
    it returns comes back in whatever frame ``flat`` was in.

    Returns ``(veg_high, building, verdicts)`` with the layers updated and
    ``verdicts`` carrying the four per-point masks plus ``cells``, the
    convicted footprint as a full ``shape`` grid.
    """
    from scipy import ndimage as ndi

    ny, nx = shape
    ncell = ny * nx
    flat = np.asarray(flat, np.int64)
    n = len(flat)
    veg_high = np.asarray(veg_high).copy()
    building = np.asarray(building).copy()

    # -- route 1: the geodesic, the oracle's expression verbatim (`&` binds
    # tighter than `|`, and inf >= 60 is True, so this is "not reached inside
    # 60 m" however the point failed to be reached)
    convicted = (cont & ((~np.isfinite(geo)) & sub | (geo >= geo_bar))
                 & (cube < cube_path))
    # -- route 2: the cube as its own route, tighter bound
    convicted = convicted | (cont & (cube < cube_solo))

    # -- route 3: the pane vote. Dense labelling like the oracle's; the voxel
    # array is bool + int32 over ny*nx*60, the same order of memory
    # :func:`cube_scatter` already spends. (ponytail: a sparse component
    # labelling is written and verified in the recon draft if a window ever
    # makes 60 M voxels the binding constraint.)
    zb = np.clip(hag, 0, zbins - 0.1).astype(np.int32)
    vox = flat * zbins + zb
    gocc = np.zeros(ncell * zbins, bool)
    gocc[vox[cont]] = True
    glab, gn = ndi.label(gocc.reshape(ny, nx, zbins), np.ones((3, 3, 3)))
    del gocc
    gcomp = np.zeros(n, np.int64)
    gcomp[cont] = glab.ravel()[vox[cont]]
    del glab
    ncv = np.bincount(gcomp[cont & convicted], minlength=gn + 1)
    ntot = np.bincount(gcomp[cont], minlength=gn + 1)
    win = np.divide(ncv, np.maximum(ntot, 1)) >= pane
    win[0] = False                       # not a pane: the unclaimed background
    convicted = convicted | (cont & win[gcomp])
    del gcomp, win, ncv, ntot

    # -- route 4: the rescue. Both stencils are over ALL points, not only the
    # defendants — the neighbourhood is the evidence.
    k9 = np.ones((3, 3))
    nt = np.bincount(flat, minlength=ncell).astype(float)
    hs = np.bincount(flat[np.asarray(scatter) >= scatter_min],
                     minlength=ncell).astype(float)
    ge = np.bincount(flat, weights=np.clip(exg, -1, 1), minlength=ncell)
    den = np.maximum(ndi.convolve(nt.reshape(ny, nx), k9, mode="constant"), 1)
    fr = (ndi.convolve(hs.reshape(ny, nx), k9, mode="constant") / den).ravel()
    gm = (ndi.convolve(ge.reshape(ny, nx), k9, mode="constant") / den).ravel()
    rescued = convicted & (fr[flat] >= rescue_fr) & (gm[flat] > rescue_green)
    convicted &= ~rescued

    veg_high &= ~convicted
    # -- the escaped population: veg inside an outline that the glazing
    # suspicion never reached (upper tiers beyond the roof window)
    rooffix = veg_high & inside_a & ~cont & (hag > tall) & (cube < cube_roof)
    veg_high &= ~rooffix
    building |= rooffix
    # -- the last-return filter, counted on the layers as they now stand
    nb = np.bincount(flat[building], minlength=ncell)
    nv = np.bincount(flat[veg_high], minlength=ncell)
    crownsplit = building & not_last & ((nv > split * nb)[flat])
    building &= ~crownsplit

    cells = np.zeros(ncell, bool)
    cells[np.unique(flat[convicted])] = True
    return veg_high, building, {
        "convicted": convicted, "rescued": rescued, "rooffix": rooffix,
        "crownsplit": crownsplit, "cells": cells.reshape(ny, nx)}


def glass_geodesic_blocks(x, y, z, sub, ground, *, block=250.0, halo=62.0,
                          limit=60.0, k=10, cap=2.5):
    """The same geodesic, on the ORACLE'S graph — l.417-472, ported verbatim.

    :func:`glass_geodesic` filters the stage-2 edge table, whose kNN was taken
    over ALL points; deleting the building endpoints then leaves a point beside
    a wall with fewer than ``k`` neighbours, or none. The oracle instead
    rebuilds kNN *on the subset*, so that point gets its ten nearest
    NON-building neighbours however far out they are (up to ``cap``). The two
    graphs are not the same graph, and on the owned tile the difference is
    worth 25% of the verdicts (**[measured]** 218,034 convictions against the
    oracle's 173,910, recall 1.0000 — ours is a strict superset, exactly the
    over-conviction a sparser graph predicts). No stage-2 table can serve
    both: an edge table is kNN under ONE blocker set. This function is what to
    call when the answer must equal the lab's; :func:`glass_geodesic` is what
    to call when the edge table is the only graph in reach.

    Blocked at 250 m with a 62 m halo, which is EXACT and not an
    approximation: every node on a path of geodesic length <= ``limit`` from
    p lies within ``limit`` metres euclidean of p, so a witness path for a
    core point never leaves the core box grown by 60 m. The oracle's reason
    for blocking at all: 31.4 M points as one graph was two SIGKILLs.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    sub = np.asarray(sub)
    idx = np.flatnonzero(sub)
    geo = np.full(len(sub), np.inf)
    if not len(idx):
        return geo
    xs, ys, zs = np.asarray(x)[idx], np.asarray(y)[idx], np.asarray(z)[idx]
    isg = np.asarray(ground)[idx]
    x0, y0 = xs.min(), ys.min()
    for bx in range(int((xs.max() - x0) // block) + 1):
        for by in range(int((ys.max() - y0) // block) + 1):
            cx0, cy0 = x0 + bx * block, y0 + by * block
            inh = ((xs >= cx0 - halo) & (xs < cx0 + block + halo)
                   & (ys >= cy0 - halo) & (ys < cy0 + block + halo))
            li = np.flatnonzero(inh)
            if len(li) < 2:
                continue
            core = ((xs[li] >= cx0) & (xs[li] < cx0 + block)
                    & (ys[li] >= cy0) & (ys[li] < cy0 + block))
            src = np.flatnonzero(isg[li])
            if not core.any() or not len(src):
                continue                    # no ground in reach: stays inf
            p = np.column_stack([xs[li], ys[li], zs[li]])
            p -= p.min(0)                   # float32 after re-origin, or the
            p = p.astype(np.float32)        # UTM easting eats the precision
            kk = min(k + 1, len(li))
            db, qb = cKDTree(p).query(p, k=kk, workers=-1)
            db, qb = db[:, 1:], qb[:, 1:]
            ok = (db <= cap).ravel()
            a = np.repeat(np.arange(len(li), dtype=np.int32), kk - 1)[ok]
            g = coo_matrix((db.ravel()[ok], (a, qb.ravel()[ok].astype(np.int32))),
                           shape=(len(li), len(li))).tocsr()
            d = dijkstra(g, directed=False, indices=src, min_only=True,
                         limit=limit)
            tgt = idx[li[core]]
            geo[tgt] = np.minimum(geo[tgt], d[core])
    return geo


def footprint_cells(polys, bbox, pix=1.0):
    """Map polygons → a boolean cell grid: a cell is in iff its CENTRE is inside a ring.

    rules/layers.py:843-852, same predicate (matplotlib Path, nonzero
    winding, radius 0) and the same ±1 m bbox prefilter, with one mechanical
    change: the prefilter becomes index arithmetic on that rectangle instead
    of a scan over every cell of the grid — identical output, O(Σ polygon
    area) instead of O(npolys × ncells) (**[measured]** 1.43 s → 0.02 s on
    the tile's 305 polygons; the lab's form would be 3,000 × 3.24 M on the
    green window).

    `polys` is a list of ``{"coords": [[x, y], …]}`` in the grid's own CRS
    (the canvases are fetched already projected). Returns the FULL grid
    (ny, nx), south-up like every grid here — the lab rasters north-up, so a
    comparison against a lab grid needs ``[::-1]``. Full, not occupied-only:
    the morphology below reaches through returnless cells and footprints
    cover plenty of them.
    """
    from matplotlib.path import Path as MPath

    from .grid import shape_for

    ny, nx = shape_for(bbox, pix)
    grid = np.zeros((ny, nx), bool)
    for p in polys:
        pc = np.asarray(p["coords"], float)
        # the lab tests centre X = bbox[0] + (cx+0.5)*pix against xmin-1 .. xmax+1
        c0 = int(np.ceil((pc[:, 0].min() - 1.0 - bbox[0]) / pix - 0.5))
        c1 = int(np.floor((pc[:, 0].max() + 1.0 - bbox[0]) / pix - 0.5))
        r0 = int(np.ceil((pc[:, 1].min() - 1.0 - bbox[1]) / pix - 0.5))
        r1 = int(np.floor((pc[:, 1].max() + 1.0 - bbox[1]) / pix - 0.5))
        c0, c1 = max(c0, 0), min(c1, nx - 1)
        r0, r1 = max(r0, 0), min(r1, ny - 1)
        if c1 < c0 or r1 < r0:
            continue
        gx, gy = np.meshgrid(bbox[0] + (np.arange(c0, c1 + 1) + 0.5) * pix,
                             bbox[1] + (np.arange(r0, r1 + 1) + 0.5) * pix)
        hit = MPath(pc).contains_points(np.column_stack([gx.ravel(), gy.ravel()]))
        grid[r0:r1 + 1, c0:c1 + 1] |= hit.reshape(r1 - r0 + 1, c1 - c0 + 1)
    return grid


def car_split(px, py, pz, surf_z, *, pix=0.3, prominence=0.25, peak_h=0.7,
              max_h=3.0, smooth=0.5):
    """Welded street furniture -> one id per OBJECT. The marina watershed, for
    a car park.

    Cars in a park touch the way boats in a raft touch: the class-induced
    subgraph sees one component, so a whole car park arrived as a single
    instance 107.9 m across [2026-08-05]. Splitting by SUPPORT-FOREST root
    fixed that and broke the opposite way — a tree has one trunk and so one
    root, but a car's roof reaches the ground by several paths around its body,
    so one car came apart into three or four fragments. Measured over the
    parking polygons: 1,974 instances of median 1.9 x 1.0 x 0.8 m, a quarter of
    a car each, and only 44 car-shaped (Kaveh, 2026-08-06: "why number of cars
    in parking or bridge is very low", "it's only two cars in the bridge").

    Neither connectivity nor roots. A car is a ROOF standing clear of the
    surface it is parked on, and the gap to the next car is a strip of that
    surface — which is exactly a height watershed: raster the height above
    ``surf_z``, take h-maxima as markers so one roof gives one marker, and
    flood. Same operator as :func:`raft_split`, different scale.

    ``surf_z`` is the surface each point stands on — the DEM on a street, the
    deck on a bridge — one value per point, because a bridge car's datum is
    30 m above a street car's.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    px, py, pz = (np.asarray(a, float) for a in (px, py, pz))
    hag = pz - np.asarray(surf_z, float)
    gx = ((px - px.min()) / pix).astype(int)
    gy = ((py - py.min()) / pix).astype(int)
    nxc = gx.max() + 1
    cell = gy * nxc + gx
    top = np.full((gy.max() + 1) * nxc, -np.inf)
    np.maximum.at(top, cell, hag)
    top = top.reshape(-1, nxc)
    occ = np.isfinite(top) & (top > 0.15) & (top < max_h)
    if not occ.any():
        return np.zeros(len(px), np.int64)
    # 0.3 m cells and a light blur: parked cars leave a gap of about 0.6 m,
    # which is ONE cell at 0.5 m and a sigma-1.0 blur closes it — measured, 3
    # cars 0.6 m apart came out as 2 instances. At 0.3 m with sigma 0.5 the
    # gap is two cells and survives; 4 cars give 4.
    ts = ndi.gaussian_filter(np.where(occ, top, 0.0), smooth)
    peaks = h_maxima(np.where(occ, ts, 0.0), prominence) & occ & (ts > peak_h)
    markers, nm = ndi.label(peaks)
    if nm == 0:
        return np.zeros(len(px), np.int64)
    ws = watershed(-ts, markers, mask=occ)
    left = occ & (ws == 0)
    if left.any():
        lft, _ = ndi.label(left, np.ones((3, 3)))
        ws = np.where(left, lft + int(ws.max()), ws)
    return ws.ravel()[cell]


def raft_split(px, py, pz, *, pix=0.5, prominence=1.2, dock_area=100.0,
               peak_h=1.0):
    """One welded marina raft -> per-vessel ids. objects.py:37-77, verbatim.

    Boats and docks all touch within ~1 m of the pontoon level, so no
    connectivity rule separates them there — the kNN graph, the class-induced
    subgraph and the MST all see one component. But each vessel *stands alone
    above* the pontoon, so the cut is made on height, not on adjacency: a
    ``pix`` raster of max height above the pontoon (p1 of z), boat-scale peaks
    become markers, and a watershed on the inverted surface floods each vessel
    from its own peak.

    Markers come from **h-maxima**, not plain local maxima: a long hull's ridge
    carries several maxima a few metres apart and each would seed its own
    object — Kaveh's "one real object in many parts". h-maxima keeps a peak
    only if it rises ``prominence`` above the saddle joining it to a higher
    peak, so bumps on one ridge merge while separate boats (whose saddle sits
    near the pontoon) stay apart. Each large low ribbon — ``dock_area`` m² under
    ``peak_h`` — is a dock and gets its own marker.

    **[measured, lab]** on the five big rafts: 425 objects, median extent 5-7 m
    (marina slips), docks kept long. At the reported mooring row, 29 fragments
    became 9 objects with the two long hulls whole at prominence 1.2 (still
    split at 0.8); isolated boats unaffected. Known cost, accepted: a large
    vessel with two peaks (bow + wheelhouse) may split.

    Returns one id per input point, 0 where the watershed claimed nothing.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    px, py, pz = np.asarray(px), np.asarray(py), np.asarray(pz)
    zw = np.percentile(pz, 1)                       # the pontoon
    gx = ((px - px.min()) / pix).astype(int)
    gy = ((py - py.min()) / pix).astype(int)
    nxc = gx.max() + 1
    cell = gy * nxc + gx
    top = np.full((gy.max() + 1) * nxc, -np.inf)
    np.maximum.at(top, cell, pz - zw)
    top = top.reshape(-1, nxc)
    occ = np.isfinite(top)
    ts = ndi.gaussian_filter(np.where(occ, top, 0.0), 1.0)

    peaks = h_maxima(np.where(occ, ts, 0.0), prominence) & occ & (ts > peak_h)
    markers, nm = ndi.label(peaks)
    low = occ & (top <= peak_h)
    llab, _ = ndi.label(low)
    area = np.bincount(llab.ravel()) * pix ** 2
    dock = np.isin(llab, np.flatnonzero(area >= dock_area)) & (llab > 0)
    dlab, _ = ndi.label(dock)
    markers[dock] = nm + dlab[dock]
    ws = watershed(-ts, markers, mask=occ)
    # THE LEFTOVERS ARE NOT ONE OBJECT. Every cell the watershed did not claim
    # used to return 0, so the whole raft's residue — walkways, gangways, the
    # gaps between slips — collapsed into a SINGLE instance: measured, 23,404
    # returns spanning 236 x 67 m, filling 7.2% of its own bounding box, and
    # typed a boat. Give the residue its own connected components instead.
    # [2026-08-05]
    left = occ & (ws == 0)
    if left.any():
        lft, nl = ndi.label(left, np.ones((3, 3)))
        ws = np.where(left, lft + int(ws.max()), ws)
    return ws.ravel()[cell]


def footprint_ids(polys, bbox, pix=1.0, key=None, fill=-1):
    """Map polygons → a grid of their **ids**, ``fill`` where no polygon covers.

    :func:`footprint_cells` answers "is this cell inside any footprint"; this
    answers "inside *which*", which is what stage 6 needs to give a building
    instance an identity inherited from the map instead of one invented by the
    geometry (the 3DBAG pattern: the same building keeps its id across reruns
    and across windows, so fusion is a key match rather than a shape match).

    ``key=None`` (the default) stamps the polygon's **index** in ``polys``,
    which always works. Pass ``key="id"`` only when the ids are integers —
    Overture's GERS ids are UUID strings and would raise. The caller maps the
    index back to whatever identity it wants, which is also what lets a string
    id survive.

    Same predicate and the same ±1 m index prefilter as
    :func:`footprint_cells`, so the two agree cell for cell:
    ``footprint_ids(...) != fill`` equals ``footprint_cells(...)``. Overlapping
    footprints resolve last-wins, which is the same convention the boolean
    version's ``|=`` implies.

    Returns the FULL grid (ny, nx), south-up like every grid here.
    """
    from matplotlib.path import Path as MPath

    from .grid import shape_for

    ny, nx = shape_for(bbox, pix)
    grid = np.full((ny, nx), fill, np.int64)
    for i, p in enumerate(polys):
        pc = np.asarray(p["coords"], float)
        c0 = int(np.ceil((pc[:, 0].min() - 1.0 - bbox[0]) / pix - 0.5))
        c1 = int(np.floor((pc[:, 0].max() + 1.0 - bbox[0]) / pix - 0.5))
        r0 = int(np.ceil((pc[:, 1].min() - 1.0 - bbox[1]) / pix - 0.5))
        r1 = int(np.floor((pc[:, 1].max() + 1.0 - bbox[1]) / pix - 0.5))
        c0, c1 = max(c0, 0), min(c1, nx - 1)
        r0, r1 = max(r0, 0), min(r1, ny - 1)
        if c1 < c0 or r1 < r0:
            continue
        gx, gy = np.meshgrid(bbox[0] + (np.arange(c0, c1 + 1) + 0.5) * pix,
                             bbox[1] + (np.arange(r0, r1 + 1) + 0.5) * pix)
        hit = MPath(pc).contains_points(
            np.column_stack([gx.ravel(), gy.ravel()])).reshape(r1 - r0 + 1,
                                                               c1 - c0 + 1)
        block = grid[r0:r1 + 1, c0:c1 + 1]
        block[hit] = i if key is None else int(p.get(key, fill))
        grid[r0:r1 + 1, c0:c1 + 1] = block
    return grid


def marina_superstructure(label, cell, z, body_grid, bbox, pix=1.0):
    """veg/small standing on a floating deck belongs to its vessel — l.810-827.

    Kaveh's marina reports, both the same gap: floating owns the hulls but
    not the superstructure — masts and rigging scatter like foliage, deck
    gear reads as small — and floating's solid-matter formula never claims
    them. The floor is per CELL: the lowest point already labelled
    ``floating`` in that same cell. A cell with no floating point keeps
    ``+inf`` and claims nothing, and that ``inf`` is the entire guard
    against eating the overhanging shore crowns (no floating below them).

    `body_grid` is :func:`flood_bodies`' full grid; truthy means "in a
    water body". `label` is rewritten IN PLACE, as the lab does. Returns
    the boolean per-point mask of what was converted.
    """
    from .grid import shape_for

    ny, nx = shape_for(bbox, pix)
    label, cell, z = np.asarray(label), np.asarray(cell), np.asarray(z)
    fl = label == FLOATING
    if not fl.any():
        return np.zeros(len(label), bool)
    flmin = np.full(ny * nx, np.inf, np.float32)
    np.minimum.at(flmin, cell[fl], z[fl].astype(np.float32))
    inb = np.asarray(body_grid).ravel()[cell] > 0
    rig = (np.isin(label, (VEG_LOW, VEG_HIGH, SMALL)) & inb
           & (z > flmin[cell] - 0.01))
    label[rig] = FLOATING
    return rig


def map_vetoes(label, cell, exg, ms3, doc, bbox, pix=1.0):
    """The three map vetoes — footprint, plant, tank — l.829-908, in the lab's order.

    Where every internal witness is circular (a large self-consistent
    false-veg blob descends through its own mislabelled matter), the map is
    the external fact:

    * **footprint** — veg_high ≥3 cells DEEP inside a mapped building and
      not green → building. Depth is the conjunct that matters: crowns
      genuinely overhang footprints, bare-inside is 50% real veg
      (**[swept]** erode 0 m 72%, 2 m 90.7%, 3 m 95.9% @20.7k, 5 m 99.3%
      @13.8k — shipped 3). Not green, because glass roofs over atrium trees
      exist.
    * **plant** — the concrete plant's towers and conveyors have NO building
      polygon; OSM knows only the tanks and the landuse. Within 20 cells of
      a mapped building AND inside industrial landuse, not green, cube solid
      (**[swept]** 10 m 95.4%, 20 m 94.2% @12.6k, 30 m 94.0%).
    * **tank** — the lattice scatters returns EXACTLY like canopy and the
      murals are painted green, so cube and colour both protect it. Position
      alone convicts: within 20 cells of a mapped storage tank inside
      industrial landuse, no other conjunct (**[swept]** 10 m 98.0%, 20 m
      97.2% @21.5k, 30 m 92.9%, 40 m 78.3% — a real tree line enters).

    STRICTLY SEQUENTIAL, and the nesting is part of the rule: each veto runs
    on the previous one's labels (a point both deep in a footprint and near a
    tank is the footprint veto's, not the tank rule's), plant and tank are
    both inside "there is industrial landuse", tank additionally inside
    "there are storage tanks". `doc` is the parsed osm_buildings.json
    (``polys`` and ``industrial`` are DISJOINT — the fetch routes a way to
    ``industrial`` iff ``landuse=industrial``); `exg` is the per-point excess
    green from RAW (unshifted) 16-bit colour; `ms3` is :func:`cube_scatter`,
    the same array the path refinement is guarded with, and is read by the
    plant rule only.

    ``iterations=N`` on scipy's default structure is a CITY-BLOCK diamond of
    radius N, not a disc of radius N — a buffer in polygon space selects 1.5×
    the area and is a different rule. `label` is rewritten IN PLACE. Returns
    ``(masks, grids)``: per-point masks keyed silo/plant/tank, and the
    south-up cell grids deep/near/near_tank they stand on, so the map half
    can be diffed against the lab with no labels in the way.
    """
    from scipy import ndimage as ndi

    label, cell, exg = np.asarray(label), np.asarray(cell), np.asarray(exg)
    masks, grids = {}, {}

    fp = footprint_cells(doc["polys"], bbox, pix)
    grids["deep"] = deep = ndi.binary_erosion(fp, iterations=3)
    masks["silo"] = silo = ((label == VEG_HIGH) & deep.ravel()[cell]
                            & (exg <= 0.02))
    label[silo] = BUILDING

    ind = doc.get("industrial", [])
    if not ind:
        return masks, grids                     # no landuse: plant AND tank die
    inm = footprint_cells(ind, bbox, pix)

    grids["near"] = near = ndi.binary_dilation(fp, iterations=20) & inm
    masks["plant"] = plant = ((label == VEG_HIGH) & near.ravel()[cell]
                              & (exg <= 0.02) & (np.asarray(ms3) < 0.3))
    label[plant] = BUILDING

    tanks = [p for p in doc["polys"] if p.get("man_made") == "storage_tank"]
    if not tanks:
        return masks, grids
    tkm = footprint_cells(tanks, bbox, pix)
    grids["near_tank"] = nt = ndi.binary_dilation(tkm, iterations=20) & inm
    masks["tank"] = tank = (label == VEG_HIGH) & nt.ravel()[cell]
    label[tank] = BUILDING
    return masks, grids


def decide_labels(claims, flat, shape, xyz, *, quorum=QUORUM,
                  quorum_min=QUORUM_MIN, k=K_VOTE, chunk=2_000_000,
                  verbose=True):
    """THE DECISION — the ten independent claim layers resolved into one
    label per point (rules/layers.py:633-682, in the lab's order).

    The layers know nothing of each other, so a point may be claimed by
    several or by none. One ordered step resolves every point:

    1. exactly one claim → that label, and the point is a **SEED**;
    2. contested → the MORE SPECIFIC layer wins (``SPEC``). Water's overlap
       with ground is 99.8% (the cloth settles on a lake as on a car park),
       so water has almost no single-claim seeds and a pure vote hands the
       whole lake to ground — IoU 0.000, measured. Equal specificity is a
       TIE that specificity refuses to settle;
    2b. **a lone voice is not a witness** — a building or veg_high seed with
       fewer than ``quorum_min`` other same-CLAIM points in its 3×3 cells is
       demoted back to asker BEFORE the seed set is taken (**[swept]**
       2/5/10/20/40/80, peak at 20: veg_high 0.869→0.876, building
       0.848→0.852; by 40 real thin structure demotes and it collapses);
    3. everything still unset — zero-claim residue, ties, demoted seeds —
       takes the majority of its ``k`` nearest SEEDS in 3-D, a contested
       point counting only votes for a label it claimed itself, falling back
       to the free five if no vote matches.

    Rule 3 is pipeline.py's "the residue takes its nearest labelled
    neighbour" promoted from a mop-up pass to the general mechanism, so
    there is ONE residue path here, not two.

    `claims` is a dict ``{layer: bool array}`` covering every name in
    ``DECIDE`` — a dict and not an (n, 10) array on purpose, since the column
    order IS the label code and a caller who stacks it wrong gets silently
    relabelled points. An extra ``placed`` key is ignored, as the oracle
    ignores it (as a competitor it stole one-storey rims from building, IoU
    0.830 → 0.766 — l.610-613). `flat` is the per-point cell id
    (:func:`cell_of`), `shape` its ``(ny, nx)``, `xyz` the (n, 3) point
    coordinates in metres. The vote has NO distance cap — a point 200 m from
    any seed still gets five votes — so the stored kNN edge table (k≤10,
    capped, over all points rather than seeds) cannot stand in for it.

    Returns ``(label, settled)``: the uint8 class code per point and an int8
    saying HOW it was decided — 0 single claim, 1 specificity, 2 neighbour
    vote (tie or demoted seed), 3 residue copy (nothing claimed it).
    """
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree

    missing = [name for name in DECIDE if name not in claims]
    if missing:
        raise ValueError(f"claims missing layers: {missing}")
    stack = np.column_stack([np.asarray(claims[name], bool) for name in DECIDE])
    n, ndec = stack.shape
    ny, nx = shape
    flat = np.asarray(flat)
    spec = np.array([SPEC[name] for name in DECIDE])
    nclaims = stack.sum(axis=1)

    label = np.full(n, UNSET, np.uint8)
    settled = np.zeros(n, np.int8)              # 0 = single claim, the default

    one = nclaims == 1
    label[one] = np.argmax(stack[one], axis=1)

    multi = np.flatnonzero(nclaims >= 2)
    sm = np.where(stack[multi], spec[None, :], -1)
    best = sm.max(axis=1)
    tie = (sm == best[:, None]).sum(axis=1) > 1
    label[multi[~tie]] = sm[~tie].argmax(axis=1)
    settled[multi[~tie]] = SETTLED_SPEC
    del sm, best

    # rule 2b. The count is over the layer's CLAIMS (not the resolved
    # labels — that is a strictly smaller number and demotes more), summed on
    # the 3×3 cell neighbourhood, minus the point itself: the bar is
    # `quorum_min` OTHER points. ``uniform_filter(float) * 9`` is the
    # oracle's own float path and is reproduced call for call — it is a mean,
    # so ×9 lands a ULP off the integer sum, and the *more correct*
    # ``ndi.convolve(ones((3, 3)))`` flips the verdict on 0.47% of cells at
    # this bar. "Fixing" it silently demotes a different population.
    for qname in quorum:
        qi = DECIDE.index(qname)     # the lab writes LAYERS.index(qname),
        # which agrees only because "placed" sits after building and veg_high
        # in LAYERS; it is a live off-by-one for any future quorum layer.
        percell = np.bincount(flat[np.asarray(claims[qname], bool)],
                              minlength=ny * nx).reshape(ny, nx)
        company = ndi.uniform_filter(percell.astype(float), size=3) * 9
        lonely = (label == qi) & (company.ravel()[flat] - 1 < quorum_min)
        label[lonely] = UNSET

    seeds = np.flatnonzero(label != UNSET)
    rest = np.flatnonzero(label == UNSET)
    if verbose:
        print(f"    decision: {int(one.sum()):,} single-claim, "
              f"{int((~tie).sum()):,} settled by specificity, "
              f"{len(rest):,} by neighbours "
              f"({int(one.sum()) + int((~tie).sum()) + len(rest) - n:,} demoted "
              f"by quorum)", flush=True)
    if len(rest):
        xyz = np.asarray(xyz, float)
        kq = min(k, len(seeds))     # the one deviation from verbatim, and it
        # only fires where the oracle would raise IndexError (< k seeds)
        _, nb = cKDTree(xyz[seeds]).query(xyz[rest], k=kq, workers=-1)
        votes = label[seeds[nb.reshape(len(rest), -1)]].astype(np.intp)
        del nb
        # The oracle loops in Python over 3.03 M rows (~50 s of its 51 s).
        # The loop has no sequential dependence — `votes` is materialised from
        # the SEED labels before it starts and it only writes into `rest` — so
        # the chunked bincount below gives the same verdicts, tie-break
        # included: ``np.bincount(v).argmax()`` and ``cnt.argmax(1)`` both take
        # the LOWEST class code (ground beats veg_low beats … beats glazing).
        m = len(rest)
        nc_rest = nclaims[rest]
        out = np.empty(m, np.uint8)
        for s in range(0, m, chunk):
            e = min(s + chunk, m)
            v = votes[s:e]
            w = np.ones(v.shape, np.int64)
            mm = nc_rest[s:e] >= 2
            if mm.any():
                # the claim restriction, with the oracle's `vc if len(vc) else
                # v` fallback: it does NOT force a claimed label, so a point
                # can end up with a label it never claimed
                hit = np.take_along_axis(stack[rest[s:e][mm]], v[mm], axis=1)
                w[mm] = np.where(hit.any(axis=1, keepdims=True), hit, True)
            c = e - s
            cnt = np.bincount((np.arange(c)[:, None] * ndec + v).ravel(),
                              weights=w.ravel(), minlength=c * ndec)
            out[s:e] = cnt.reshape(c, ndec).argmax(axis=1)
        label[rest] = out
        settled[rest] = np.where(nc_rest == 0, SETTLED_RESIDUE, SETTLED_VOTE)
    return label, settled
