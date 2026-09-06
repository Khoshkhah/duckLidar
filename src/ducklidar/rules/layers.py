"""Kaveh's flat-layer redesign, approach 1: independent rule layers, then a local decision.

The tree in ``pipeline.py`` encodes precedence in its structure: water gets first refusal,
ground before vegetation, and every conflict is settled by ORDER. This prototype removes the
order. Each label gets its own layer -- an independent rule over the same features, knowing
nothing of the other layers -- so a point may be claimed by several layers, or by none. The
overlaps are not a bug here; they are the measurement. Then one decision step resolves every
point:

    exactly one layer claims it   -> that label, a SEED
    zero or several layers claim  -> the label of its nearest seeds in 3-D (majority of k=5)

which is the tree's residue rule promoted to the general mechanism: unambiguous physics
decides directly, ambiguous points inherit from their unambiguous neighbours.

v1 scope, stated rather than hidden: the bridge layer is seed+corridor only (no growth), the
vehicle/deck/adjudication passes of the tree are NOT replicated -- this measures whether the
flat architecture can stand, not whether it can reproduce every refinement. Scored exactly
like the tree (judged IoU against the survey) and written as out/layers.npz for the dashboard.

    python tools/layers.py
"""
import sys
import time

def _rss(tag):
    import resource
    print(f"    [rss {tag}] {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f} GB", flush=True)

import numpy as np
from scipy import ndimage as ndi

import ducklidar as dl
from ._env import OUTDIR, check
from .pipeline import (B, CSF_RIGIDNESS, ECHO_MIN, FLOAT_CEILING, FLOAT_DEM_TOL, FLOAT_MOVE,
                      GREEN_MEDIAN, GROUND_CEILING, GROUND_FALLBACK, PLANAR_MIN, SCATTER_MAX,
                      SCATTER_MIN, TALL, WATER_CLOSE, WATER_DENS, WATER_HAG, WATER_LOCAL,
                      WATER_MIN_CELLS, WATER_OWN_DENS, WATER_OWN_INTEN, WATER_PLANE,
                      WATER_VOTE, WATER_VOTE_WIN, WATER_WIN, covariance, line_spread)

LAYERS = ["ground", "veg_low", "veg_high", "building", "water", "small", "floating",
          "bridge", "placed", "on_bridge", "glazing"]


def main(argv):
    t0 = time.time()
    # only the columns the rules read — gps_time/scan_angle/source id are
    # another ~1 GB the tile cannot spare (2026-08-01)
    pts = dl.read(check(), B, fields=("intensity", "return_number",
                                      "number_of_returns", "red", "green",
                                      "blue", "point_source_id"))
    n = len(pts["x"])
    z = np.asarray(pts["z"])
    flat, (ny, nx) = dl.cell_index(pts, B, 1.0)
    print(f"{n:,} returns")

    # ---- shared features, computed once ------------------------------------------------------
    gnd = dl.ground_filter(pts, cloth_resolution=1.0, rigidness=CSF_RIGIDNESS)
    v1 = dl.dem(pts, B, 1.0, ground=gnd)
    hag = z - v1["filled"].ravel()[flat]
    echo = np.nan_to_num(dl.echo_ratio(pts, B, 1.0),
                         nan=0.0).ravel()[flat].astype(np.float32)
    not_last = np.asarray(pts["return_number"]) < np.asarray(pts["number_of_returns"])
    planar, scatter, _ = covariance(n)
    planar = planar.astype(np.float32)
    scatter = scatter.astype(np.float32)
    _r = np.asarray(pts["red"]).astype(float)
    _g = np.asarray(pts["green"]).astype(float)
    _b = np.asarray(pts["blue"]).astype(float)
    exg = ((2 * _g - _r - _b)
           / np.maximum(_r + _g + _b, 1)).astype(np.float32)
    del _r, _g, _b                       # ~1 GB of colour temporaries
    inten = np.asarray(pts["intensity"]).astype(np.float32)
    dens_grid = np.bincount(flat, minlength=ny * nx)
    moved = line_spread(pts, flat, ny * nx)[flat].astype(np.float32)
    top = np.full(ny * nx, -np.inf)
    np.maximum.at(top, flat, z)
    overhead = (top[flat] - z).astype(np.float32)

    # water context (identical construction to the tree's, but used by layers independently)
    low_cell = np.bincount(flat, weights=(hag < WATER_HAG).astype(float),
                           minlength=ny * nx) > 0
    sparse = ((dens_grid <= WATER_DENS) & low_cell).reshape(ny, nx)
    sparse = ndi.binary_closing(sparse, np.ones((2 * WATER_CLOSE + 1,) * 2))
    wlab, _ = ndi.label(sparse)
    wsz = np.bincount(wlab.ravel())
    region = np.isin(wlab, np.flatnonzero(wsz >= WATER_MIN_CELLS)) & (wlab > 0)
    local = ndi.uniform_filter(dens_grid.reshape(ny, nx).astype(float),
                               size=WATER_WIN) <= WATER_LOCAL
    both = region | local
    agree = ndi.uniform_filter(both.astype(float), size=WATER_VOTE_WIN) >= WATER_VOTE
    in_water = (both & agree).ravel()[flat]
    # floating's water context: within 30 m of a water-candidate cell. The acceptance test
    # caught 12.8% of floating claims sitting INLAND -- 'over' (DEM at plane height) is
    # also true of low shoreline land, so parked cars on a low quay read as afloat. The
    # definition says on the water; this conjunct says it in the formula.
    near_water = ndi.binary_dilation(both & agree, iterations=30).ravel()[flat]
    seed_w = region.ravel()[flat] & (hag < WATER_HAG)
    zw = float(np.median(z[seed_w])) if seed_w.any() else 0.0
    over = (np.abs(v1["filled"] - zw) < FLOAT_DEM_TOL).ravel()[flat]
    # the water BODY, by FLOODING (Kaveh's redefinition, and his ordering: fix the body
    # before ground can be fixed above it). "Sparse and dark" finds only water the laser
    # SAW; the body must also include water that docks and hulls cover. We know the level
    # (zw), so the body is everything that level floods, reachable from open water: grow
    # the certain region through every cell whose surface sits at or below zw + 0.4. Under
    # a dock the DEM interpolates to the plane -> flooded; at the beach the terrain climbs
    # above the plane within a cell -> the flood stops at the zw contour, which IS the
    # shoreline. Context only here -- the layers that consume it come next.
    # seeds are the FULL candidate context (region OR locally-sparse, vote-confirmed), not
    # only the big region: a pocket of water beyond the bridge deck has no path for the
    # flood to reach it (the interpolated surface under the deck rises near the shore) but
    # its own sparsity already testifies it is water (Kaveh's missing strip).
    _seeds = region | (both & agree)
    water_body = ndi.binary_propagation(_seeds, mask=_seeds | (v1["filled"] <= zw + 0.4))
    # ...holes closed: where the cloth settled ON a dock or hull, the DEM reads above the
    # plane and the flood walks around it -- but a hole fully surrounded by flooded water
    # IS water (Kaveh spotted the missing pieces). Park ponds enter through their own
    # sparsity seeds; their BOUNDARIES still use the sea's plane, so a pond at a different
    # level has a nominal edge -- the per-body plane remains the recorded refinement.
    water_body = ndi.binary_fill_holes(water_body)
    print(f"    water body: {int(region.sum()):,} sparse-dark cells flooded to "
          f"{int(water_body.sum()):,} cells (holes filled)")

    # band clusters for the low layers (green vote, as in the tree)
    band = (hag >= GROUND_FALLBACK) & (hag <= TALL)
    bcell = np.zeros(ny * nx, bool)
    bcell[np.unique(flat[band])] = True
    clab, _ = ndi.label(bcell.reshape(ny, nx), structure=np.ones((3, 3)))
    cid = clab.ravel()[flat]
    bidx = np.flatnonzero(band)
    bidx = bidx[np.argsort(cid[bidx])]
    med = np.zeros(clab.max() + 1)
    for g in np.split(bidx, np.flatnonzero(np.diff(cid[bidx]) != 0) + 1):
        if len(g):
            med[cid[g[0]]] = np.median(exg[g])
    green = med[cid] > GREEN_MEDIAN

    print(f"    features ready ({time.time()-t0:.0f}s)")

    # ---- the layers: independent, orderless, overlapping -------------------------------------
    penetrable = (echo >= ECHO_MIN) & (not_last | (scatter >= SCATTER_MIN))
    # ground's FORBIDDEN rule (Kaveh's chain: body -> ground -> floating): terrain cannot
    # exist inside the water body -- the WHOLE body. Two narrower zones were tried and
    # measured insufficient: the marina's interpolated DEM slopes up from the shore, so an
    # "at the plane" zone missed most of it and 42k body cells stayed ground. The known
    # cost is the survey's tidal seabed (~2% of ground cells, surveyed at another tide),
    # accepted and bounded in check_layers with its reason.
    _no_terrain = water_body.ravel()[flat]
    L = {
        "ground":   ((gnd & (hag < GROUND_CEILING)) | (hag < GROUND_FALLBACK))
                    & ~_no_terrain,
        "veg_low":  band & green,
        "veg_high": (hag > TALL) & penetrable & (planar < PLANAR_MIN),
        "building": (hag > TALL) & (planar >= PLANAR_MIN) & (scatter < SCATTER_MAX),
        "water":    in_water & (np.abs(z - zw) < WATER_PLANE)
                    & (dens_grid[flat] <= WATER_OWN_DENS) & (inten < WATER_OWN_INTEN),
        "small":    band & ~green,
        # route 1: the movement evidence (a boat floats, so it moves). route 2, Kaveh's,
        # unlocked by the water body: inside the body, solid matter above the plane IS
        # floating -- no movement gate, so the boats' still points, single-look cells and
        # cabins arrive directly. Penetrable points stay out (rigging, and the shore
        # crowns that overhang the body's edge); the footprint take-back still returns
        # wharf sheds to building downstream.
        "floating": (over & near_water & (z > zw) & ~not_last & (hag < 10)
                     & (moved > FLOAT_MOVE) & (overhead < FLOAT_CEILING))
                    | (water_body.ravel()[flat] & (z > zw + 0.15) & (hag < 12)
                       & ~((echo >= ECHO_MIN) & (not_last | (scatter >= SCATTER_MIN)))),
        "bridge":   np.zeros(n, bool),          # filled below: seed + OSM corridor
        # Kaveh's symmetry with floating: a thing STANDING ON the terrain. Just above the
        # ground, the ground itself seen in the same cell (the undercut evidence at cell
        # scale -- a building's walls forbid it), on land, solid (penetrable points belong
        # to vegetation), and not a green band cluster (a shrub is vegetal, not an object).
        # Claims trucks, cars, containers, furniture -- contested cells resolve by
        # specificity: this layer carries support evidence, so it outranks building.
        "placed": np.zeros(n, bool),         # filled below: needs hag_min per cell
        "glazing": np.zeros(n, bool),        # filled by the assembly stage
    }
    zmin_c = np.full(ny * nx, np.inf)
    np.minimum.at(zmin_c, flat, z)
    hagmin_cell = (zmin_c - v1["filled"].ravel())[flat]
    # 1.8-5 m (the band below is `small`'s), and the WHOLE column low (cell top <= 5.5 m
    # above ground) so a two-storey house's eave cell -- roof above, yard's ground in the
    # same cell -- cannot qualify. One-storey eaves remain the known leak; the component
    # interior test that solves them fully is context, not a per-point rule (v2).
    celltop = (top - v1["filled"].ravel())[flat]
    L["placed"] = ((hag >= 1.8) & (hag <= 5.0) & (celltop <= 5.5)
                      & (hagmin_cell < 0.3) & ~over & ~penetrable
                      & ~(band & green))
    deck = dl.bridge_deck(pts, B, 1.0)
    L["bridge"] |= deck
    try:
        import json
        ob = json.loads((OUTDIR / "osm_bridges.json").read_text())
        # buffer each way at ITS OWN width: 8 m suits Granville's carriageways but drew a
        # park footbridge 4x too wide, and the over-wide corridor then claimed park ground
        # and crowns around it (Kaveh spotted the blobs). A footway is ~2.5 m of structure;
        # lanes tell the rest.
        def radius(w):
            if w.get("highway") in ("footway", "cycleway", "path", "steps"):
                return 2.5
            try:
                return max(3.0, int(w.get("lanes") or 2) * 3.5 / 2 + 1.5)
            except ValueError:
                return 5.0
        groups = {}
        for w in ob["ways"]:
            foot = w.get("highway") in ("footway", "cycleway", "path", "steps")
            r = radius(w)
            line = groups.setdefault((r, foot), np.zeros((ny, nx), bool))
            c = np.asarray(w["coords"], float)
            for (x0, y0), (x1, y1) in zip(c[:-1], c[1:]):
                k = max(int(np.hypot(x1 - x0, y1 - y0) / 0.5), 1) + 1
                xs, ys = np.linspace(x0, x1, k), np.linspace(y0, y1, k)
                rows = np.floor((B[3] - ys)).astype(int)
                cols = np.floor((xs - B[0])).astype(int)
                ok = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)
                line[rows[ok], cols[ok]] = True
        # Kaveh's ontology: a park footbridge a metre over a pond is a structure ON the
        # ground, not a bridge. Carriageway ways are authoritative at any height (the
        # at-grade ramp is the bridge); FOOT ways count only where genuinely elevated
        # (> 4 m above the terrain) -- which keeps Granville's 40 m sidewalk and drops
        # the pond footbridges and the shoreline boardwalk to the ground system.
        corr_major = np.zeros((ny, nx), bool)
        corr_foot = np.zeros((ny, nx), bool)
        for (r, foot), line in groups.items():
            m = ndi.distance_transform_edt(~line) <= r
            if foot:
                corr_foot |= m
            else:
                corr_major |= m
        elev = (top - v1["filled"].ravel()).reshape(ny, nx) > 4.0
        corr = corr_major | (corr_foot & elev)
        # Kaveh's contract: where OSM maps the bridge, OSM IS the definition -- the void
        # seed is only the fallback for unmapped structure. So inside the corridor the
        # deck surface is claimed at ANY height (the ramp at grade is still the bridge,
        # that is exactly what OSM knows and geometry cannot), solid points near the cell
        # top; a hag guard here would silently drop the at-grade deck, which is the
        # mistake this line used to make.
        L["bridge"] |= corr.ravel()[flat] & (z > top[flat] - 2.0) & ~penetrable
    except FileNotFoundError:
        corr = np.zeros((ny, nx), bool)
        print("    no osm_bridges.json -- bridge layer is the void seed alone")
    # on_bridge: Kaveh's third support symmetry (floating:water, placed:ground). A solid
    # object standing 0.4-5 m above the DECK SURFACE. The datum is the per-cell minimum of
    # the bridge-claimed returns (a car cannot lower the surface it stands on), eroded 3x3
    # so a covered cell borrows the bare deck beside it, extended across the deck region.
    bcl = np.zeros(ny * nx, bool)
    bcl[np.unique(flat[L["bridge"]])] = True
    dmin = np.full(ny * nx, np.inf)
    np.minimum.at(dmin, flat[L["bridge"]], z[L["bridge"]])
    deckreg = ndi.binary_fill_holes(bcl.reshape(ny, nx)) | corr
    dgrid = ndi.grey_erosion(np.where(bcl, dmin, np.inf).reshape(ny, nx), size=3)
    for _ in range(8):
        need = deckreg & ~np.isfinite(dgrid)
        if not need.any():
            break
        dgrid = np.where(need, ndi.grey_erosion(dgrid, size=3), dgrid)
    hd = z - dgrid.ravel()[flat]
    L["on_bridge"] = (deckreg.ravel()[flat] & np.isfinite(dgrid.ravel()[flat])
                      & (hd >= 0.4) & (hd <= 5.0) & ~penetrable)
    # the graph, applied to the slicing gap (Kaveh's ask): a lamp post was cut at 5 m --
    # shaft to on_bridge, head to bridge. Whole-component voting died measured: at 1 m
    # connectivity the deck furniture is ONE chain (railings join lamps join onward; 93.6k
    # of 99.4k claims sat in ">5 m" mega-components -- the marina lesson, on the deck). The
    # question that works is per COLUMN: what does the SOLID matter above this cell top out
    # at? Solid, because the trolley wires span every lane at ~6 m and are penetrable;
    # a lamp shaft's column tops at its solid head. Columns topping > 5.5 m are structure:
    # their slices leave on_bridge and join bridge whole.
    _sol = ~penetrable & deckreg.ravel()[flat] & np.isfinite(dgrid.ravel()[flat]) & (hd > 0.4)
    _stop = np.zeros(ny * nx)
    np.maximum.at(_stop, flat[_sol], hd[_sol])
    _shaft = L["on_bridge"] & (_stop[flat] > 5.5)
    L["on_bridge"] &= ~_shaft
    L["bridge"] |= _shaft
    print(f"    deck columns: {int(_shaft.sum()):,} shaft-slice points moved to bridge "
          "structure (solid column top > 5.5 m)")

    # ---- extension: each layer's LOCAL rule, and its FORBIDDEN rule (Kaveh's design) ----------
    # Hysteresis as geodesic dilation (docs/layers.md 2d): the layer's formula is the
    # strong core; a weak halo may be REACHED from the core, never started from. Each
    # layer gets its own local rule (halo + reach) -- and a forbidden rule where the
    # measurement said extension poisons it. [swept] reach 2/5/inf: building and veg high
    # gain to 0.865 / 0.882 at reach 2; ground's halo changes nothing (skipped, YAGNI);
    # water's halo is FORBIDDEN outright -- density<=30 at the plane floods through the
    # marina (0.940 -> 0.931 at reach 2, 0.656 at convergence), so water does not extend.
    # FORBIDDEN rules, explicit and per layer (Kaveh's design): zones a layer's EXTENSION
    # may never enter, whatever its support. [measured] without them the building halo took
    # 148,188 points over the water plane (docks are not the walls of any roof) and 303,056
    # in the bridge corridor (the deck is the bridge system's); the veg fringe took 67,491
    # over the plane (rigging is not crown). The core formulas keep their own inline
    # context terms (floating's near_water, placed's ¬over are forbidden rules stated
    # positively); this dict governs where GROWTH may go.
    FORBID = {
        "building": over | corr.ravel()[flat],   # the water system's and the deck's ground
        "veg_high": over,                        # crowns do not grow out of the sea
    }
    EXT = {
        "building": ((hag > TALL) & ~penetrable, 2),   # solidity without flatness: walls
        "veg_high": ((hag > TALL) & (not_last | (scatter >= SCATTER_MIN)), 2),  # fringe
    }
    for _name, (_weak, _reach) in EXT.items():
        _weak = _weak & ~FORBID.get(_name, np.zeros(n, bool))
        _sc = np.zeros(ny * nx, bool)
        _sc[np.unique(flat[L[_name]])] = True
        _wc = np.zeros(ny * nx, bool)
        _wc[np.unique(flat[_weak])] = True
        _wc |= _sc
        _grown = _sc.reshape(ny, nx)
        for _ in range(_reach):
            _grown = ndi.binary_dilation(_grown) & _wc.reshape(ny, nx)
        L[_name] = L[_name] | (_weak & _grown.ravel()[flat])

    # ---- assembly: outlines from all building evidence, then containment claims --------------
    # The lock the glass rule was missing (docs/layers.md 2e). Outlines are built from the
    # building layer's claims -- cores AND extended walls -- closed, hole-filled, with the
    # forbidden zones excluded (the deck and the water are other systems' ground).
    # [measured] the stage is small by the time it runs: extension already assembled the
    # walls, so adoption finds only ~2k unclaimed interior points, and glazing on the
    # assembled hulls buys +0.002 building (0.865 -> 0.867) at W=2.5 with the scatter
    # guard. The remaining miss population -- fully-glass and fully-rough buildings -- has
    # NO core to outline, so no internal assembly can reach it; the recorded next source
    # is external outlines (OSM building footprints).
    forbid_cell = np.zeros(ny * nx, bool)
    forbid_cell[np.unique(flat[over])] = True
    forbid_cell |= corr.ravel()
    bc_a = np.zeros(ny * nx, bool)
    bc_a[np.unique(flat[L["building"]])] = True
    bc_a &= ~forbid_cell
    hull = ndi.binary_fill_holes(ndi.binary_closing(bc_a.reshape(ny, nx), np.ones((3, 3))))
    blab_a, nb_a = ndi.label(hull)
    comp_a = blab_a.ravel()[flat]
    b_a = L["building"]
    _s = np.bincount(comp_a[b_a], weights=z[b_a], minlength=nb_a + 1)
    _c = np.bincount(comp_a[b_a], minlength=nb_a + 1)
    roofz_a = np.divide(_s, _c, out=np.zeros(nb_a + 1), where=_c > 0)
    inside_a = comp_a > 0
    nclaims_a = np.column_stack([L[k] for k in LAYERS]).sum(axis=1)
    adopt_a = (nclaims_a == 0) & inside_a & (hag > TALL)
    L["building"] = L["building"] | adopt_a
    # glazing is its OWN layer (Kaveh's correction: a distinct thing is a distinct layer,
    # not points smuggled into building's claims). Scoring folds it into building, as the
    # tree folded glass with its source byte kept.
    # Glazing is an ADJUDICATION layer, and two attempts to make it a resolved class are
    # recorded dead: at tie specificity it resolves ~1k points (decorative); at spec 3
    # with scatter tightened to 0.15 it resolves 24,801 points that the map shows as
    # SPECKLE, not skylights (glass is almost never the topmost return in its cell), for
    # -0.005 veg and -0.006 building. Its measured value is the contests it creates: with
    # it in the tie role, veg high and building both reached their best (0.888 / 0.870).
    # A real glass class needs an instrument this cloud does not carry.
    # the roof height is LOCAL, not the component mean (Kaveh's correction: on a
    # multi-tier building both tiers' glass sits outside +-2.5 m of the mean and was
    # never suspected -- the 62k roof-tree find was the symptom). Each cell's roof is the
    # highest building-claimed return in its 3x3 neighbourhood; the component mean stays
    # as a fallback window for cells with no local building top.
    _bmax = np.full(ny * nx, -np.inf)
    np.maximum.at(_bmax, flat[L["building"]], z[L["building"]])
    _lroof = ndi.maximum_filter(_bmax.reshape(ny, nx), size=3).ravel()[flat]
    _win = (np.abs(z - _lroof) < 2.5) | (np.abs(z - roofz_a[comp_a]) < 2.5)
    L["glazing"] = penetrable & inside_a & _win & (scatter < SCATTER_MAX)
    # Kaveh's coherence rule: glazing IS the tribunal, so an UNCONTESTED glazing claim
    # must not exist -- without a vegetation dispute there is nothing to adjudicate, and
    # before this rule such claims became unexamined single-claim seeds (a building point
    # with no veg label wearing a glazing cell, his catch). Uncontested penetrable-flat
    # on a roof is simply building glass: it goes to building directly.
    _gl_only = L["glazing"] & ~L["veg_high"]
    L["building"] |= _gl_only
    L["glazing"] &= L["veg_high"]

    # ---- the support graph: Kaveh's path-to-ground, as 3-D connectivity through matter --------
    # Classic family: geodesics on point graphs / morphological reconstruction. The 1 m
    # voxel form: occupancy components of (matter minus building); a point whose component
    # reaches the ground got there WITHOUT passing through building -- and on the contested
    # glazing∩veg set that acquits real crowns at 96.8% survey purity (a crown's shell
    # connects sideways-and-down through its own foliage; glass is sealed to the roof).
    # Recorded negatives of the same instrument: the 1-D column-gap version measures
    # crown-shell-ness INVERTED (big gap below = canopy envelope, 89-93% veg); and
    # floating-by-support finds zero candidates at 1 m -- the whole marina is one
    # shore-connected component through docks and gangways; severing thin links (min-cut)
    # is the recorded next step for that use.
    _zb = np.clip(hag, 0, 59.9).astype(np.int32)
    _vox = flat.astype(np.int64) * 60 + _zb
    _occ = np.zeros(ny * nx * 60, bool)
    _occ[_vox] = True
    _bldv = np.zeros(ny * nx * 60, bool)
    _bldv[_vox[L["building"]]] = True
    _gndv = np.zeros(ny * nx * 60, bool)
    _gndv[_vox[L["ground"]]] = True
    _lab3, _ = ndi.label((_occ & ~_bldv).reshape(ny, nx, 60), structure=np.ones((3, 3, 3)))
    _gc = np.unique(_lab3.ravel()[_gndv & ~_bldv])
    _gc = _gc[_gc > 0]
    ground_conn = np.isin(_lab3.ravel(), _gc)[_vox]
    # the acquittal: a contested glazing claim on a ground-connected point was a crown
    L["glazing"] = L["glazing"] & ~(L["veg_high"] & ground_conn)
    print(f"    support graph: {int((L['veg_high'] & ground_conn & ~L['glazing']).sum()):,} "
          "contested points acquitted as crowns (path to ground avoids building)")
    # ...and the kNN CONVICTION, the other direction (docs/graph.md family 1), now with
    # Kaveh's EDGE LENGTHS: each edge carries its Euclidean metres, so the graph is a
    # metric and the feature is the multi-source Dijkstra GEODESIC to the ground through
    # non-building matter. Two convictions fall out, both swept: unreachable (the binary
    # form, 90% survey-building), and reachable only by an absurd detour -- geodesic >=
    # 60 m measured 100% survey-building, 30-60 m only 77% so the bar stays at 60. The
    # zone is dilated 25 cells because reachability near the zone border is an artifact
    # (clipped paths look unreachable); the detour RATIO (geo/hag) measured too weak to
    # use (medians 1.34 vs 1.23). Convicted points lose their vegetation claim.
    _cont = L["glazing"] & L["veg_high"]
    if _cont.any():
        _zc = np.zeros(ny * nx, bool)
        _zc[np.unique(flat[_cont])] = True
        _zone = ndi.binary_dilation(_zc.reshape(ny, nx), iterations=25).ravel()[flat]
        _sub = _zone & ~L["building"]
        _idx = np.flatnonzero(_sub)
        from scipy.spatial import cKDTree as _KD
        from scipy.sparse import coo_matrix as _coo
        from scipy.sparse.csgraph import dijkstra as _dij
        # spatial blocks with a 62 m halo: the conviction bar is the
        # MEASURED 60 m geodesic, so dijkstra(limit=60) inside a block with
        # a 60 m halo returns every verdict exactly — and the tile's zone is
        # 31.4M points (74% of the cloud), which as one global graph was
        # two SIGKILLs [measured 2026-08-01]. Same verdicts, bounded memory.
        _xs = np.asarray(pts["x"])[_idx]
        _ys = np.asarray(pts["y"])[_idx]
        _zs = z[_idx]
        _isg = L["ground"][_idx]
        print(f"    conviction zone: {len(_idx):,} points", flush=True)
        _gg = np.full(n, np.inf)
        _BS, _HALO = 250.0, 62.0
        _x0b, _y0b = _xs.min(), _ys.min()
        for _bx in range(int((_xs.max() - _x0b) // _BS) + 1):
            for _by in range(int((_ys.max() - _y0b) // _BS) + 1):
                _cx0, _cy0 = _x0b + _bx * _BS, _y0b + _by * _BS
                _inh = (_xs >= _cx0 - _HALO) & (_xs < _cx0 + _BS + _HALO) \
                    & (_ys >= _cy0 - _HALO) & (_ys < _cy0 + _BS + _HALO)
                _li = np.flatnonzero(_inh)
                if len(_li) < 2:
                    continue
                _core = (_xs[_li] >= _cx0) & (_xs[_li] < _cx0 + _BS) \
                    & (_ys[_li] >= _cy0) & (_ys[_li] < _cy0 + _BS)
                if not _core.any():
                    continue
                _src = np.flatnonzero(_isg[_li])
                if not len(_src):
                    continue        # no ground in reach: stays inf ✓
                _Pb = np.column_stack([_xs[_li], _ys[_li], _zs[_li]])
                _Pb -= _Pb.min(0)
                _Pb = _Pb.astype(np.float32)
                _k = min(11, len(_li))
                _db, _qb = _KD(_Pb).query(_Pb, k=_k, workers=-1)
                _db, _qb = _db[:, 1:], _qb[:, 1:]
                _okb = (_db <= 2.5).ravel()
                _ab = np.repeat(np.arange(len(_li), dtype=np.int32), _k - 1)[_okb]
                _Gb = _coo((_db.ravel()[_okb],
                            (_ab, _qb.ravel()[_okb].astype(np.int32))),
                           shape=(len(_li), len(_li))).tocsr()
                _geo = _dij(_Gb, directed=False, indices=_src,
                            min_only=True, limit=60.0)
                _tgt = _idx[_li[_core]]
                _gg[_tgt] = np.minimum(_gg[_tgt], _geo[_core])
        _rss("dijkstra")
        # Kaveh's cube computes FIRST now: after the local-roof widening, the path route
        # alone degraded to 68% (crowns near local rooftops strand for their own reasons,
        # re-swept on his false-detection report); a path conviction must also pass the
        # relaxed material bound. [swept] path & cube<0.2: 99%, <0.25: 96% at 33k,
        # <0.3: 90% -- shipped at 0.25.
        _zb3 = np.clip(hag, 0, 59.9).astype(np.int32)
        _vox3 = flat.astype(np.int64) * 60 + _zb3
        _NV3 = ny * nx * 60
        _ss3 = np.bincount(_vox3, weights=scatter, minlength=_NV3).reshape(ny, nx, 60)
        _cn3 = np.bincount(_vox3, minlength=_NV3).astype(float).reshape(ny, nx, 60)
        _k3 = np.ones((3, 3, 3))
        _ms3 = (ndi.convolve(_ss3, _k3, mode="constant") /
                np.maximum(ndi.convolve(_cn3, _k3, mode="constant"), 1)).ravel()[_vox3]
        _rss("cube")
        _convicted = (_cont & ((~np.isfinite(_gg)) & _sub | (_gg >= 60.0))
                      & (_ms3 < 0.25))
        del _gg, _ss3, _cn3            # past last use; the tile needs the room
        # Kaveh's cube as its own route (standalone, tighter bound): 90-98% pure
        _convicted = _convicted | (_cont & (_ms3 < 0.15))
        # ...and the PANE vote (the graph's object coherence, with the safety the deck
        # taught): a skylight convicts in fragments point-by-point, so components of the
        # glazing-claimed matter vote -- a pane where >= 50% of points are individually
        # convicted convicts WHOLE. Majority, not any-touch: panes can touch overhanging
        # crowns, and a crown's component has ~0% convicted points. [swept] 0.5 -> +952
        # points at 91% survey purity; 0.3 dilutes to 75%, 0.1 is junk.
        _zb2 = np.clip(hag, 0, 59.9).astype(np.int32)
        _gvox = flat[L["glazing"]].astype(np.int64) * 60 + _zb2[L["glazing"]]
        _gocc = np.zeros(ny * nx * 60, bool)
        _gocc[_gvox] = True
        _glab, _gn = ndi.label(_gocc.reshape(ny, nx, 60), np.ones((3, 3, 3)))
        _rss("panes-label")
        _gcomp = np.zeros(n, np.int64)
        _gcomp[L["glazing"]] = _glab.ravel()[_gvox]
        _ncv = np.bincount(_gcomp[L["glazing"] & _convicted], minlength=_gn + 1)
        _ntot = np.bincount(_gcomp[L["glazing"]], minlength=_gn + 1)
        _win = np.divide(_ncv, np.maximum(_ntot, 1)) >= 0.5
        _win[0] = False
        _convicted = _convicted | (L["glazing"] & _win[_gcomp])
        del _gocc, _glab, _gcomp, _win, _ncv, _ntot
        # the RESCUE (Kaveh's scatter-density, composed with colour): the conviction's
        # stated blind spot is vegetation rooted ON the building -- roof gardens, planters,
        # crowns lapping a roof -- stranded exactly like glass. Around a plant there are
        # many high-scatter points AND the matter is green; glass is neither. [swept]
        # scatter-density alone peaks at 85% veg; with the green vote: 98% veg / 2% bld
        # (7,455 points recovered at fr >= 0.5, green > 0.02).
        _hs = np.bincount(flat[scatter >= SCATTER_MIN], minlength=ny * nx).astype(float)
        _nt = np.bincount(flat, minlength=ny * nx).astype(float)
        _k9 = np.ones((3, 3))
        _fr = (ndi.convolve(_hs.reshape(ny, nx), _k9, mode="constant") /
               np.maximum(ndi.convolve(_nt.reshape(ny, nx), _k9, mode="constant"), 1)
               ).ravel()[flat]
        _ge = np.bincount(flat, weights=np.clip(exg, -1, 1), minlength=ny * nx)
        _gm = (ndi.convolve(_ge.reshape(ny, nx), _k9, mode="constant") /
               np.maximum(ndi.convolve(_nt.reshape(ny, nx), _k9, mode="constant"), 1)
               ).ravel()[flat]
        _rescued = _convicted & (_fr >= 0.5) & (_gm > 0.02)
        _convicted &= ~_rescued
        _rss("rescue")
        L["veg_high"] = L["veg_high"] & ~_convicted
        # ...and Kaveh's cube pointed at the ESCAPED population -- veg claims inside
        # outlines that the glazing suspicion never reached (crown-scattered zones, upper
        # tiers beyond the roof window). [measured] cube < 0.15 selects 62,144 points at
        # 97% survey-building, 2% veg -- the roof trees he saw, the largest single find of
        # the day. Solid surroundings are not a crown, wherever the roof window ends.
        _rooffix = (L["veg_high"] & inside_a & ~L["glazing"] & (hag > TALL)
                    & (_ms3 < 0.15))
        L["veg_high"] &= ~_rooffix
        L["building"] |= _rooffix
        print(f"    roof-tree fix: {int(_rooffix.sum()):,} veg claims inside outlines "
              "with solid cube surroundings -> building")
        print(f"    kNN graph: {int(_convicted.sum()):,} convicted as glass, "
              f"{int(_rescued.sum()):,} rescued as rooftop vegetation (scatter-dense + green)")

    # ---- Kaveh's last-return filter, surgical form --------------------------------------------
    # A real roof stops the pulse, so a roof return is a last return. Dropping ALL
    # splitting building claims would cost the 78% that are genuine split roof edges
    # (eaves, vents); the shippable cut is contextual: a splitting claim in a cell where
    # vegetation out-claims building SIX to one is a crown. [swept] veg>1x: 68% veg,
    # veg>3x: 87%, veg>6x: 93% -- 13,333 points dropped at 6% collateral. (Two graph
    # formulations of the same hunt are recorded dead: stranded-without-veg was 64% real
    # building -- ivy and glass walls strand honestly -- and its veg-beneath refinement
    # inverted to 99% building, roofs with crowns lapping below their edges.)
    _nb = np.bincount(flat[L["building"]], minlength=ny * nx)
    _nv = np.bincount(flat[L["veg_high"]], minlength=ny * nx)
    _crownsplit = L["building"] & not_last & ((_nv > 6 * _nb)[flat])
    L["building"] &= ~_crownsplit
    print(f"    last-return filter: {int(_crownsplit.sum()):,} splitting building claims "
          "in crown-dominated cells dropped")

    # ---- the generalized blocker set: not only building -- every structural system ------------
    # (Kaveh's generalization.) A path that exists only THROUGH bridge matter proves bridge
    # structure, exactly as through-building proved glass. In the corridor, veg-claimed
    # points stranded on a kNN graph whose blockers are building ∪ bridge are the TROLLEY
    # WIRES and their kin: [measured] 12,677 stranded points, 99% survey-unclassified (the
    # survey never labels wires), 109 judged-veg collateral. The voxel form of the same
    # test is recorded dead (83% of its stranded were real crowns -- sparse canopy breaks
    # at 1 m; the kNN graph reconnects foliage, as it did for glazing).
    _wz = ndi.binary_dilation(corr, iterations=10).ravel()[flat]
    _wsub = _wz & ~(L["building"] | L["bridge"])
    _widx = np.flatnonzero(_wsub)
    if len(_widx):
        from scipy.spatial import cKDTree as _KD2
        from scipy.sparse import coo_matrix as _coo2
        from scipy.sparse.csgraph import connected_components as _cc2
        _P2 = np.column_stack([np.asarray(pts["x"])[_widx], np.asarray(pts["y"])[_widx],
                               z[_widx]])
        _d2, _q2 = _KD2(_P2).query(_P2, k=11, workers=-1)
        _d2, _q2 = _d2[:, 1:], _q2[:, 1:]
        _ok2 = (_d2 <= 2.5).ravel()
        _a2 = np.repeat(np.arange(len(_P2)), 10)[_ok2]
        _G2 = _coo2((np.ones(_a2.shape, np.int8), (_a2, _q2.ravel()[_ok2])),
                    shape=(len(_P2), len(_P2)))
        _, _lab2 = _cc2(_G2, directed=False)
        _gc3 = np.zeros(_lab2.max() + 1, bool)
        _gc3[_lab2[L["ground"][_widx]]] = True
        _reach2 = np.zeros(n, bool)
        _reach2[_widx] = _gc3[_lab2]
        _wire = L["veg_high"] & corr.ravel()[flat] & (hag > 2) & _wsub & ~_reach2
        L["veg_high"] &= ~_wire
        L["bridge"] |= _wire
        print(f"    corridor stranding: {int(_wire.sum()):,} veg-claimed points with no "
              "path to ground except through structure -> bridge (the wires)")

    # ---- object topology: a small building component ringed by vegetation is a crown ----------
    # (Kaveh's direction: the graphs carry geometry AND topology; per-point geometry is
    # exhausted -- a dense crown shell and a roof are locally identical -- so the signal
    # left is at OBJECT scale.) [swept] area <= 40 cells with a >= 70% vegetation ring:
    # 99% survey-veg, 0% survey-building, at every tried setting -- the cleanest
    # instrument of the day. 655 components, 10,606 points return to vegetation.
    _bc2 = np.zeros(ny * nx, bool)
    _bc2[np.unique(flat[L["building"]])] = True
    _vc2 = np.zeros(ny * nx, bool)
    _vc2[np.unique(flat[L["veg_high"]])] = True
    _bl2, _bn2 = ndi.label(_bc2.reshape(ny, nx), np.ones((3, 3)))
    _vc2 = _vc2.reshape(ny, nx)
    _flip2 = np.zeros((ny, nx), bool)
    for _i in range(1, _bn2 + 1):
        _m2 = _bl2 == _i
        if _m2.sum() > 40:
            continue
        _ring2 = ndi.binary_dilation(_m2) & ~_m2
        if _vc2[_ring2].mean() >= 0.7:
            _flip2 |= _m2
    _crowncomp = L["building"] & _flip2.ravel()[flat]
    L["building"] &= ~_crowncomp
    L["veg_high"] |= _crowncomp
    print(f"    object topology: {int(_crowncomp.sum()):,} points in small veg-ringed "
          "building components returned to vegetation")

    # ---- the measurement the tree never makes: who overlaps whom -----------------------------
    # placed is an EVIDENCE layer in v1, not a competitor: with specificity it stole
    # one-storey rims from building (IoU 0.830 -> 0.766 at spec 3, 0.781 with the
    # low-column guard -- the eave problem, which per-point rules cannot solve; the
    # component-interior test that does is context, i.e. v2). It is measured, exported and
    # drawable, and the decision ignores it.
    DECIDE = [k for k in LAYERS if k != "placed"]
    stack = np.column_stack([L[k] for k in DECIDE])
    nclaims = stack.sum(axis=1)
    print(f"\nclaims per point: 0 -> {float((nclaims==0).mean())*100:.1f}%   "
          f"1 -> {float((nclaims==1).mean())*100:.1f}%   "
          f"2+ -> {float((nclaims>=2).mean())*100:.1f}%")
    print("\noverlap matrix (% of the smaller layer):")
    print(f"{'':>9}" + "".join(f"{k:>9}" for k in LAYERS))
    for i, a in enumerate(LAYERS):
        row = f"{a:>9}"
        for j, b in enumerate(LAYERS):
            if j <= i:
                row += f"{'':>9}"
                continue
            inter = int((L[a] & L[b]).sum())
            denom = max(min(int(L[a].sum()), int(L[b].sum())), 1)
            row += f"{inter/denom*100:>8.1f}%"
        print(row)

    # ---- the decision ------------------------------------------------------------------------
    # Rule 1, learned from the first run of this very file: on a contested point the MORE
    # SPECIFIC evidence wins. Water's overlap with ground is 99.8% (the cloth settles on a
    # lake as on a car park), so water has almost no single-claim seeds and a pure
    # neighbour vote hands the whole lake to ground -- IoU 0.000, measured. Specificity is
    # the tree's own first ordering principle, expressed here as a pairwise resolution rule
    # instead of a global order. Rule 2: what specificity cannot settle (a tie), the
    # neighbours do. Rule 3: what nothing claims inherits from its nearest seeds.
    t = time.time()
    SPEC = {"water": 5, "on_bridge": 5, "floating": 4, "bridge": 4, "glazing": 2,
            "building": 2, "veg_high": 2, "veg_low": 2, "small": 2, "ground": 1}
    spec = np.array([SPEC[k] for k in DECIDE])
    label = np.full(n, 255, np.uint8)
    one = nclaims == 1
    label[one] = np.argmax(stack[one], axis=1)
    multi = np.flatnonzero(nclaims >= 2)
    sm = np.where(stack[multi], spec[None, :], -1)
    best = sm.max(axis=1)
    tie = (sm == best[:, None]).sum(axis=1) > 1
    label[multi[~tie]] = sm[~tie].argmax(axis=1)
    # Rule 2b, Kaveh's isolated-point question made into a rule: A LONE VOICE IS NOT A
    # WITNESS. A single building point inside a crown (or one veg point on a roof) is the
    # covariance flickering, not an object -- and as a seed it votes confidently and
    # wrongly at every neighbourhood size (the vote-size sweep proved votes cannot fix
    # voters). So a building or veg-high seed needs COMPANY: >= 20 same-claim points in
    # its 3x3 cells, or it is demoted to asker and inherits like any unsure point.
    # [swept] quorum 2/5/10/20/40/80 -> peak at 20 (veg high 0.869 -> 0.876, building
    # 0.848 -> 0.852); by 40 real thin structure starts demoting and everything collapses.
    for qname in ("building", "veg_high"):
        qi_ = LAYERS.index(qname)
        percell = np.bincount(flat[L[qname]], minlength=ny * nx).reshape(ny, nx)
        company = ndi.uniform_filter(percell.astype(float), size=3) * 9
        lonely = (label == qi_) & (company.ravel()[flat] - 1 < 20)
        label[lonely] = 255
    from scipy.spatial import cKDTree
    xyz = np.column_stack([np.asarray(pts["x"]), np.asarray(pts["y"]), z])
    seeds = np.flatnonzero(label != 255)
    rest = np.flatnonzero(label == 255)
    _, qi = cKDTree(xyz[seeds]).query(xyz[rest], k=5, workers=-1)
    votes = label[seeds[qi]]
    for row, pi in enumerate(rest):
        v = votes[row]
        if nclaims[pi] >= 2:
            claimed = np.flatnonzero(stack[pi])
            vc = v[np.isin(v, claimed)]
            v = vc if len(vc) else v
        label[pi] = np.bincount(v).argmax()
    print(f"\n    decision: {int(one.sum()):,} single-claim, "
          f"{int((~tie).sum()):,} settled by specificity, "
          f"{len(rest):,} by neighbours ({time.time()-t:.0f}s)")

    # ---- Kaveh's path-entry refinement: refine_veg and refine_building ----------------------
    # The test: walk every point's shortest path to ground on the MESH (kNN k=10/2.5 m
    # + the stored MST bridges, so every point HAS a path) and read the label of the
    # LAST node before ground. A tree enters through its own matter (trunk = a band
    # cluster, undergrowth: 97.5% of veg_high does), a building through its walls. The
    # path alone is NOT a verdict -- unguarded, veg-entering-through-building is 85.7%
    # real wall-hugging trees -- so each layer pairs the path with the cube (material):
    #   refine_veg      false VEGETATION: veg_high whose path is MAJORITY building, cube
    #                   solid and not green -> building.  [measured] 94.0% survey-building
    #                   at 33.6k (entry-only first cut: 96.7% but just 5.9k)
    #   refine_building false BUILDING: building whose path is majority veg_low/veg_high,
    #                   cube scattered and green -> veg.  [measured] 94.0% survey-veg at
    #                   21.2k (entry-only: 96.6% at 13.0k). Majority topology-only is
    #                   82.2% / 71.7% -- still below the bar, so the guards stay
    _gi, _vi, _bi, _li = (DECIDE.index(k) for k in ("ground", "veg_high", "building",
                                                    "veg_low"))
    L["refine_veg"] = np.zeros(n, bool)
    L["refine_building"] = np.zeros(n, bool)
    _meshf = OUTDIR / "mesh_graph.npz"
    if _meshf.exists():
        _brm = np.load(_meshf)
        # the mesh, built lean (2026-08-01, tile scale): chunked f32/int32
        # query, csr assembled directly from the row-sorted kNN lists — no
        # coo, no dedupe sort (per-direction kNN lists are duplicate-free
        # and directed=False walks either direction's entry). The one-shot
        # f64 query alone was 7.6 GB and the dedupe sort another 7 — both
        # were the 21.5 GB SIGKILL.
        from scipy.sparse import csr_matrix as _csrm
        from scipy.sparse.csgraph import dijkstra as _dijk
        import ctypes
        _trim = lambda: ctypes.CDLL("libc.so.6").malloc_trim(0)
        # local-origin f32 point copy for the tree (the conviction-section
        # precedent), xyz freed — it has no reader after this loop
        _xyz32 = (xyz - xyz.min(0)).astype(np.float32)
        del xyz
        _trim()
        _rss("mesh: pts32")
        _treem = cKDTree(_xyz32)
        _rss("mesh: tree")
        # compress per chunk: holding the full 42.7M x 10 neighbour matrices
        # even in f32/int32 was a 3.4 GB plateau at the worst possible
        # moment (21.1 GB SIGKILL after the decision, 2026-08-01)
        _cnt_l, _bmv_l, _wmv_l = [], [], []
        for _s in range(0, n, 2_000_000):
            _e = min(_s + 2_000_000, n)
            _dd, _qq = _treem.query(_xyz32[_s:_e], k=11, workers=-1)
            _dc = _dd[:, 1:].astype(np.float32)
            _qc = _qq[:, 1:].astype(np.int32)
            _okc = _dc <= 2.5
            _cnt_l.append(_okc.sum(1).astype(np.int64))
            _bmv_l.append(_qc[_okc])
            _wmv_l.append(_dc[_okc])
        del _treem, _xyz32
        _trim()
        _rss("mesh: edges")
        _cntm = np.concatenate(_cnt_l)
        _bmv = np.concatenate(_bmv_l)
        _wmv = np.concatenate(_wmv_l)
        del _cnt_l, _bmv_l, _wmv_l
        # merge the MST bridges into the row structure by insertion — the
        # 6k bridge entries ride into the 400M-edge lists without a sort
        _bs = _brm["bridge_src"].astype(np.int64)
        _bd = _brm["bridge_dst"].astype(np.int64)
        _bw = _brm["bridge_len"].astype(np.float32)
        _ba = np.concatenate([_bs, _bd])
        _bb = np.concatenate([_bd, _bs]).astype(np.int32)
        _bww = np.concatenate([_bw, _bw])
        _ip0 = np.concatenate(([0], np.cumsum(_cntm)))
        _bmv = np.insert(_bmv, _ip0[_ba], _bb)
        _wmv = np.insert(_wmv, _ip0[_ba], _bww)
        np.add.at(_cntm, _ba, 1)
        _ip = np.concatenate(([0], np.cumsum(_cntm)))
        _Gm = _csrm((_wmv, _bmv, _ip), shape=(n, n))
        del _bmv, _wmv, _ip0, _ba, _bb, _bww, _cntm
        _trim()
        _rss("mesh: csr")
        _, _predm, _ = _dijk(_Gm, directed=False, indices=np.flatnonzero(label == _gi),
                             min_only=True, return_predecessors=True)
        del _Gm
        _trim()
        _rss("mesh: dijkstra")
        # per-path label MAJORITY (Kaveh's second cut, replacing last-node-before-ground:
        # entry-only reads the base clutter both classes share -- unguarded it was 85.7%
        # innocent trees -- while the majority reads the whole route: a leaning tree
        # passes BRIEFLY through the wall but its path is mostly its own crown).
        # Histogram by pointer-doubling partial sums up the predecessor forest.
        _ida = np.arange(n, dtype=np.int64)
        _Jm = _predm.astype(np.int64)
        _rootm = _Jm < 0
        _Jm[_rootm] = _ida[_rootm]
        _Sm = np.zeros((n, len(DECIDE)), np.int32)
        _nzm = ~_rootm
        _Sm[_nzm, label[_predm[_nzm]]] = 1
        for _ in range(25):
            _J2m = _Jm[_Jm]
            if (_J2m == _Jm).all():
                break
            _Sm += _Sm[_Jm]
            _Jm = _J2m
        # Kaveh's 2/3-agreement form, plus a floor on the rival: a label is suspect when
        # its OWN class holds under 2/3 of the path AND the rival holds over 1/3. The
        # evolution, each step measured: argmax majority (94.0% at 33.6k) -> true
        # majority share>0.5 (95.5% at 41.2k, argmax dominated) -> self<2/3 & rival>1/3
        # (93.7% at 62.8k -- the bigger catch wins the scoreboard). Self-agreement alone
        # (his bare 2/3 rule) is 86.9%/88.3% -- below the bar; paths legitimately wander
        # through trunk-small and base clutter, so the rival floor is what convicts.
        _totm = np.maximum(_Sm.sum(axis=1), 1)
        _bshm = _Sm[:, _bi] / _totm
        _vshm = (_Sm[:, _li] + _Sm[:, _vi]) / _totm
        _bshm[_rootm] = 0.0
        _vshm[_rootm] = 0.0
        del _Sm
        L["refine_veg"] = ((label == _vi) & (_vshm < 2 / 3) & (_bshm > 1 / 3)
                           & (_ms3 < 0.3) & (exg <= 0.02))
        L["refine_building"] = ((label == _bi) & (_bshm < 2 / 3) & (_vshm > 1 / 3)
                                & (_ms3 > 0.3) & (exg > 0.02))
        label[L["refine_veg"]] = _bi
        label[L["refine_building"]] = _vi
        print(f"    path-agreement refinement: {int(L['refine_veg'].sum()):,} veg -> building "
              f"(path backs veg < 2/3, building > 1/3, cube solid), "
              f"{int(L['refine_building'].sum()):,} building -> veg "
              f"(path backs building < 2/3, veg > 1/3, cube scattered + green)")
    else:
        print("    path-entry refinement SKIPPED -- run tools/mesh_graph.py first "
              "(out/mesh_graph.npz missing)")

    # ---- marina superstructure: the rig and gear belong to their vessel ---------------------
    # Kaveh's marina reports, both one gap: floating owns the hulls but not the
    # superstructure -- masts/rigging (veg-labelled, scattered like foliage) and deck
    # gear (small-labelled, band-height) stand on floating decks that floating's
    # solid-matter formula never claims. [measured] points labelled veg/small inside the
    # water body above the column's lowest resolved-floating point: 31,134, only 2.0%
    # survey-veg (the overhanging shore crowns stay safe: no floating below them).
    _fli, _smi = DECIDE.index("floating"), DECIDE.index("small")
    _flpt2 = label == _fli
    if _flpt2.any():
        _flmin2 = np.full(ny * nx, np.inf, np.float32)
        np.minimum.at(_flmin2, flat[_flpt2], z[_flpt2].astype(np.float32))
        _inb2 = water_body.ravel()[flat]
        _rig = (np.isin(label, (_li, _vi, _smi)) & _inb2
                & (z > _flmin2[flat] - 0.01))
        label[_rig] = _fli
        print(f"    marina superstructure: {int(_rig.sum()):,} veg/small points above "
              "floating matter in the water body -> floating (masts, rigging, deck gear)")

    # ---- footprint veto: not-green vegetation deep inside a mapped building -----------------
    # The silo lesson: a large self-consistent false-veg blob defeats every internal
    # witness -- its paths descend through its own mislabelled matter (circular), the
    # component colour vote dilutes across fused shore trees (~90% ceiling), material
    # alone stalls at 71-78%. The external fact breaks the circle: OSM maps the silos.
    # Crowns genuinely overhang footprints (bare inside is 50% real veg), so the rule
    # requires DEPTH: [swept] erode 0 m: 72% -- 2 m: 90.7% -- 3 m: 95.9% at 20.7k --
    # 5 m: 99.3% at 13.8k. Shipped at 3 m; not green because glass roofs over atrium
    # trees exist and green stays protected.
    _osmf = OUTDIR / "osm_buildings.json"
    if _osmf.exists():
        import json as _json
        from matplotlib.path import Path as _MPath
        _polys = _json.load(open(_osmf))["polys"]
        _cxg, _cyg = np.meshgrid(B[0] + np.arange(nx) + 0.5, B[3] - np.arange(ny) - 0.5)
        _pg = np.column_stack([_cxg.ravel(), _cyg.ravel()])
        _fpm = np.zeros(ny * nx, bool)
        for _pl in _polys:
            _pc = np.array(_pl["coords"])
            _mb = ((_pg[:, 0] >= _pc[:, 0].min() - 1) & (_pg[:, 0] <= _pc[:, 0].max() + 1)
                   & (_pg[:, 1] >= _pc[:, 1].min() - 1) & (_pg[:, 1] <= _pc[:, 1].max() + 1))
            if _mb.any():
                _ib = np.flatnonzero(_mb)
                _fpm[_ib] |= _MPath(_pc).contains_points(_pg[_ib])
        _deep = ndi.binary_erosion(_fpm.reshape(ny, nx), iterations=3).ravel()[flat]
        _silo = (label == _vi) & _deep & (exg <= 0.02)
        label[_silo] = _bi
        print(f"    footprint veto: {int(_silo.sum()):,} not-green veg points >= 3 m "
              "inside a mapped building -> building (the silos)")
        # the plant rule (Kaveh's follow-up: veg still on the silo): the concrete
        # plant's towers and conveyors have NO building polygon -- OSM knows only the
        # tanks and the landuse. Near a mapped building, inside industrial landuse,
        # not green, cube solid: [swept] 10 m: 95.4% -- 20 m: 94.2% at 12.6k
        # (shipped) -- 30 m: 94.0%. Bare industrial is 76% real trees; the guards
        # carry it.
        _indl = _json.load(open(_osmf)).get("industrial", [])
        if _indl:
            _inm = np.zeros(ny * nx, bool)
            for _pl in _indl:
                _pc = np.array(_pl["coords"])
                _mb = ((_pg[:, 0] >= _pc[:, 0].min() - 1) & (_pg[:, 0] <= _pc[:, 0].max() + 1)
                       & (_pg[:, 1] >= _pc[:, 1].min() - 1) & (_pg[:, 1] <= _pc[:, 1].max() + 1))
                if _mb.any():
                    _ib = np.flatnonzero(_mb)
                    _inm[_ib] |= _MPath(_pc).contains_points(_pg[_ib])
            _near = (ndi.binary_dilation(_fpm.reshape(ny, nx), iterations=20)
                     & _inm.reshape(ny, nx)).ravel()[flat]
            _plant = (label == _vi) & _near & (exg <= 0.02) & (_ms3 < 0.3)
            label[_plant] = _bi
            print(f"    plant rule: {int(_plant.sum()):,} not-green cube-solid veg within "
                  "20 m of a mapped building in industrial landuse -> building")
            # the tank rule (Kaveh: "nothing fixed about silo" -- and he was right):
            # the plant's lattice (conveyor trusses, pipework at 21 m) scatters returns
            # EXACTLY like canopy, so the cube guard that protects real crowns protected
            # it too, and the murals are literally painted green. No internal witness is
            # left; position alone convicts. [swept] within N m of a mapped storage tank
            # inside industrial landuse, no other condition: 10 m 98.0% -- 20 m 97.2%
            # at 21.5k (shipped) -- 30 m 92.9% -- 40 m 78.3% (a real tree line enters).
            _tkl = [_pl for _pl in _polys if _pl.get("man_made") == "storage_tank"]
            if _tkl:
                _tkm = np.zeros(ny * nx, bool)
                for _pl in _tkl:
                    _pc = np.array(_pl["coords"])
                    _mb = ((_pg[:, 0] >= _pc[:, 0].min() - 1)
                           & (_pg[:, 0] <= _pc[:, 0].max() + 1)
                           & (_pg[:, 1] >= _pc[:, 1].min() - 1)
                           & (_pg[:, 1] <= _pc[:, 1].max() + 1))
                    if _mb.any():
                        _ib = np.flatnonzero(_mb)
                        _tkm[_ib] |= _MPath(_pc).contains_points(_pg[_ib])
                _neart = (ndi.binary_dilation(_tkm.reshape(ny, nx), iterations=20)
                          & _inm.reshape(ny, nx)).ravel()[flat]
                _tank = (label == _vi) & _neart
                label[_tank] = _bi
                print(f"    tank rule: {int(_tank.sum()):,} veg points within 20 m of a "
                      "mapped storage tank in industrial landuse -> building "
                      "(the lattice and the murals)")
    else:
        print("    footprint veto SKIPPED -- run tools/osm_buildings.py first "
              "(out/osm_buildings.json missing)")

    # ---- score exactly as the tree is scored -------------------------------------------------
    cls = np.asarray(pts["classification"])
    judged = ~np.isin(cls, (0, 1))
    IDX = {k: i for i, k in enumerate(DECIDE)}
    print("\n--- layered method, agreement where the survey committed ---")
    bld_all = (label == IDX["building"]) | (label == IDX["glazing"])
    for name, mine, theirs in (("ground", label == IDX["ground"], cls == 2),
                               ("vegetation high", label == IDX["veg_high"], cls == 5),
                               ("vegetation low", label == IDX["veg_low"], cls == 3),
                               ("building", bld_all, cls == 6),
                               ("water", label == IDX["water"], cls == 9)):
        m, th = mine & judged, theirs & judged
        tp = float(np.logical_and(m, th).sum())
        print(f"  {name:<16} IoU {tp/max(float(np.logical_or(m,th).sum()),1):.3f}  "
              f"precision {tp/max(float(m.sum()),1):.2f}  recall {tp/max(float(th.sum()),1):.2f}")

    # per-cell masks for the dashboard, topmost point wins, as the tree exports
    order = np.argsort(z)
    lab_cell = np.full(ny * nx, 255, np.uint8)   # 255 = no returns; 0 is ground's index
    lab_cell[flat[order]] = label[order]
    def topcell(k):
        return (lab_cell == IDX[k]).reshape(ny, nx)
    # the RAW claim mask per layer -- any claimed point in the cell -- so the layer viewer
    # can show the overlaps themselves, which are this method's whole point
    claims = {}
    for k in LAYERS:
        cc = np.zeros(ny * nx, bool)
        cc[np.unique(flat[L[k]])] = True
        claims["claim_" + k] = cc.reshape(ny, nx)
    # glazing presence: glass is almost never the TOPMOST return of its cell, so the
    # resolved (topmost) view erases it; this mask marks any cell containing resolved
    # glass anywhere in its column
    _gpres = np.zeros(ny * nx, bool)
    _gpres[np.unique(flat[label == IDX["glazing"]])] = True
    # refinement presence: like glazing, refined points are rarely the topmost return,
    # so the viewer gets any-in-column cell masks
    _rvp = np.zeros(ny * nx, bool)
    _rvp[np.unique(flat[L["refine_veg"]])] = True
    _rbp = np.zeros(ny * nx, bool)
    _rbp[np.unique(flat[L["refine_building"]])] = True
    # FINAL per-layer presence masks (Kaveh's "all layers get the best"): the layer
    # AFTER the decision and every refinement, as any-in-column cell masks -- the
    # resolved (topmost) grid erases understory layers, presence does not
    finals = {}
    for k in DECIDE:
        fc = np.zeros(ny * nx, bool)
        fc[np.unique(flat[label == IDX[k]])] = True
        finals["final_" + k] = fc.reshape(ny, nx)
    np.savez_compressed(OUTDIR / "layers.npz",
                        building=topcell("building") | topcell("glazing"),
                        vegetation=topcell("veg_high"),
                        water=topcell("water"), ground=topcell("ground"),
                        water_body=water_body, glazing_cells=_gpres.reshape(ny, nx),
                        refine_veg_cells=_rvp.reshape(ny, nx),
                        refine_building_cells=_rbp.reshape(ny, nx),
                        label=lab_cell.reshape(ny, nx), deck_corridor=corr,
                        **claims, **finals)
    # full-resolution resolved label, the interface objects.py instances on. Point order is
    # dl.read's file order for the same box, so consumers re-read the window and assert n.
    np.savez_compressed(OUTDIR / "labels.npz", label=label)
    # a 1-in-8 point sample with its resolved layer, for the 3-D view (same seed policy as
    # viz_points3d: random, not a stride, so the scan pattern is not sampled periodically)
    keep = np.random.default_rng(0).random(n) < 1 / 8
    np.savez_compressed(OUTDIR / "layers_pts.npz",
                        x=np.asarray(pts["x"])[keep].astype(np.float32),
                        y=np.asarray(pts["y"])[keep].astype(np.float32),
                        z=z[keep].astype(np.float32),
                        red=(np.asarray(pts["red"])[keep] >> 8).astype(np.uint8),
                        green=(np.asarray(pts["green"])[keep] >> 8).astype(np.uint8),
                        blue=(np.asarray(pts["blue"])[keep] >> 8).astype(np.uint8),
                        label=label[keep])
    print(f"\nwrote {OUTDIR/'layers.npz'}  ({time.time()-t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
