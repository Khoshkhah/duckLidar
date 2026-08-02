"""Points to a raster, and how much of the result is actually measured.

Gridding is where a point cloud stops being measurements and starts being a model, and the
losses are structural rather than incidental. Two of them are worth stating in code because
they are easy to forget once the array looks like a surface:

**A cell's value is a choice.** 120 returns in one square metre give max 11.57, P95 9.52,
P75 8.39, median 3.67. `max` is the DSM convention and also the statistic a single lucky
return moves furthest — built from two flight lines separately, a DSM agrees to 0.02 m on
flat ground and disagrees by over 2 m on 15% of edge cells.

**A gridded DTM is not a DTM.** Under a roof or a dense crown no pulse reaches the ground,
so there is nothing to take a minimum of: 78.8% of building cells have no ground return at
all, and the ground grid comes out 49% filled against 84% for the surface. No choice of
reducer fixes that — the returns are not there. The fill has to be interpolated, **globally**
(the voids merge across footprints, roads and water into a few large regions; the largest
here is 8.3 ha), and it should carry :func:`distance_to_evidence` alongside it so a filled
value 90 m from any measurement is not mistaken for a measured one.
"""
from __future__ import annotations

import numpy as np

from .estimate import Estimate


def shape_for(bbox, pix):
    """`(rows, cols)` for this window at this cell size."""
    return (int(round((bbox[3] - bbox[1]) / pix)), int(round((bbox[2] - bbox[0]) / pix)))


def cell_index(pts, bbox, pix):
    """Flat cell index per point, north-up — row 0 is the northmost row.

    North-up is the near-universal raster convention (GeoTIFF, GDAL, rasterio all assume it)
    and getting it upside down is silent: the array still looks like a surface, it is just
    mirrored, so anything directional computed from it — slope, aspect, a shadow — comes out
    reflected. `tests/test_ducklidar.py` pins it.
    """
    ny, nx = shape_for(bbox, pix)
    col = ((pts["x"] - bbox[0]) / pix).astype(int).clip(0, nx - 1)
    row = ((bbox[3] - pts["y"]) / pix).astype(int).clip(0, ny - 1)
    return row * nx + col, (ny, nx)


def at_extremum(pts, bbox, pix=1.0, field="scan_angle", *, how="max", values="z", mask=None):
    """Grid the value of `field` belonging to the return that *won* each cell.

    Not the same as gridding `field` — that answers "what is the steepest angle here?", this
    answers "what produced the number the DSM kept?". With ``field="point_source_id"`` it maps
    which flight line each DSM cell actually came from, which is the provenance layer for
    every other grid: a cell whose value came from one pass has no second opinion behind it.

    **[measured]** Grouping the returns by *scan-angle interval* instead is a real alternative
    partition, and on a 250 m Granville window it loses: quartile angle bands split only 62.5%
    of cells two ways against the lines' 79.0%, agree with the line split to under a centimetre
    where both apply, and find the larger disagreement on 5.9% of shared cells. The reason is
    that one line's returns in a 1 m cell all arrive at one angle, so the bands re-derive the
    lines — while merging 21.3% of multi-line cells into a single band and throwing their second
    opinion away. Split by line; use the angle as a covariate, which is what this returns.
    """
    flat, (ny, nx) = cell_index(pts, bbox, pix)
    v = np.asarray(pts[values] if isinstance(values, str) else values, dtype="float64")
    f = np.asarray(pts[field] if isinstance(field, str) else field, dtype="float64")
    if mask is not None:
        flat, v, f = flat[mask], v[mask], f[mask]

    # sort by (cell, value) so each cell's winner lands last; scatter indices, last write wins
    order = np.lexsort((v if how == "max" else -v, flat))
    won = np.full(ny * nx, -1, dtype="int64")
    won[flat[order]] = np.arange(len(order))
    out = np.full(ny * nx, np.nan)
    got = won >= 0
    out[got] = f[order][won[got]]
    return out.reshape(ny, nx)


def rasterize(pts, bbox, pix=1.0, *, mask=None, how="max", values="z", interval=False,
              quantile=0.90):
    """Reduce the selected returns onto a grid. NaN where no point landed.

    `how` is `"max"` for a surface, `"min"` for terrain. Nothing here fills the gaps — see
    the module docstring for why that is a separate, global step.

    With ``interval=True`` returns an :class:`~ducklidar.estimate.Estimate` instead of a bare
    array: the same values, plus a per-cell interval taken from repeat flight lines where two
    saw the cell, and otherwise from how far the extremum sits above `quantile` of the cell's
    own returns. A bare number cannot distinguish a 2 cm cell from a 19 m one, and both occur
    in the same raster.
    """
    flat, (ny, nx) = cell_index(pts, bbox, pix)
    v = np.asarray(pts[values] if isinstance(values, str) else values, dtype="float64")
    src = pts.get("point_source_id") if interval else None
    if mask is not None:
        flat, v = flat[mask], v[mask]
        if src is not None:
            src = np.asarray(src)[mask]

    take = np.maximum if how == "max" else np.minimum
    fill = -np.inf if how == "max" else np.inf
    out = np.full(ny * nx, fill, dtype="float64")
    take.at(out, flat, v)
    value = np.where(np.isfinite(out), out, np.nan)
    if not interval:
        return value.reshape(ny, nx)

    n = np.bincount(flat, minlength=ny * nx)

    # --- evidence 2 first: what independent flight lines said. It is both the better
    # evidence and the cheaper to compute, and on real data it covers 87-97% of cells — so
    # working out which cells it *cannot* answer lets the expensive fallback below run on a
    # fraction of the points instead of all of them.
    multi = np.zeros(ny * nx, bool)
    pmin = np.full(ny * nx, np.inf)
    pmax = np.full(ny * nx, -np.inf)
    bulk_gap = np.full(ny * nx, np.nan)
    balance = np.full(ny * nx, np.nan)
    if src is not None:
        span = int(np.max(src)) + 1
        key = flat.astype("int64") * span + np.asarray(src, dtype="int64")
        uk, inv = np.unique(key, return_inverse=True)
        per = np.full(len(uk), fill, dtype="float64")
        take.at(per, inv, v)
        cells = (uk // span).astype("int64")
        np.maximum.at(pmax, cells, per)
        np.minimum.at(pmin, cells, per)
        npass = np.bincount(cells, minlength=ny * nx)
        multi = npass >= 2

        # The extremum is the *least* robust thing a pass knows about a cell. Comparing what
        # each pass says about the BULK instead separates "one surface, something clipped
        # above it" from "genuinely two surfaces": on wide cells the passes disagree by a
        # median 3.75 m on the max and 0.26 m on the median. And how evenly each pass sampled
        # the cell says whether both had a fair look, or one merely clipped its corner.
        order = np.argsort(inv, kind="stable")
        gs = inv[order]
        gv = v[order]
        start = np.searchsorted(gs, np.arange(len(uk)), "left")
        stop = np.searchsorted(gs, np.arange(len(uk)), "right")
        cnt = stop - start
        med = gv[start + cnt // 2]                       # per-pass median, no full sort
        gmax = np.full(ny * nx, -np.inf)
        gmin = np.full(ny * nx, np.inf)
        np.maximum.at(gmax, cells, med)
        np.minimum.at(gmin, cells, med)
        bulk_gap = np.where(multi, gmax - gmin, np.nan)

        cmax = np.zeros(ny * nx, dtype="int64")
        cmin = np.full(ny * nx, np.iinfo("int64").max)
        np.maximum.at(cmax, cells, cnt)
        np.minimum.at(cmin, cells, cnt)
        balance = np.where(multi, cmin / np.maximum(cmax, 1), np.nan)

    # --- evidence 1: how far the extremum sits above the bulk of its own cell. Only needed
    # where repeat passes could not answer, which is the minority of cells.
    need = ~multi & (n > 1)
    bulk = np.full(ny * nx, np.nan)
    if need.any():
        sel = need[flat]
        fsub, vsub = flat[sel], v[sel]
        order = np.lexsort((vsub, fsub))
        fs, vs = fsub[order], vsub[order]
        idx = np.arange(ny * nx)
        start = np.searchsorted(fs, idx, "left")
        stop = np.searchsorted(fs, idx, "right")
        cnt = stop - start
        q = np.clip(quantile if how == "max" else 1.0 - quantile, 0.0, 1.0)
        pick = np.where(cnt > 0, start + np.floor(q * np.maximum(cnt - 1, 0)).astype(int), 0)
        bulk = np.where(cnt > 0, vs[np.clip(pick, 0, max(len(vs) - 1, 0))], np.nan)

    lo = np.where(how == "max", np.fmin(bulk, value), value)
    hi = np.where(how == "max", value, np.fmax(bulk, value))
    method = np.where(np.isfinite(bulk) & (n > 1), 1, 0)

    lo = np.where(multi, np.fmin(pmin, lo), lo)
    hi = np.where(multi, np.fmax(pmax, hi), hi)
    method = np.where(multi, 2, method)

    blank = ~np.isfinite(value)
    lo[blank] = np.nan
    hi[blank] = np.nan
    method[blank] = 0
    blank2 = blank
    bulk_gap[blank2] = np.nan
    balance[blank2] = np.nan
    return Estimate(value.reshape(ny, nx), lo.reshape(ny, nx), hi.reshape(ny, nx),
                    n.reshape(ny, nx), method.reshape(ny, nx),
                    bulk_gap.reshape(ny, nx), balance.reshape(ny, nx))


def surfaces(pts, bbox, pix=1.0, *, interval=False, ground=None):
    """The usual four grids at once: ``dsm``, ``dtm``, ``canopy``, ``roofs``.

    **Read this before trusting `dtm`.** By default the ground is taken from the survey's own
    ``classification`` field — someone else's answer, produced by their software after the
    flight, sometimes hand-edited, and missing entirely from many surveys. Gridding it is not
    *computing* a DTM, it is displaying a decision that was already made.

    Pass ``ground=`` a boolean mask to use your own instead —
    :func:`ducklidar.ground.ground_filter` derives one from x, y, z alone:

        gnd = dl.ground_filter(pts)
        g = dl.surfaces(pts, bbox, ground=gnd)

    `canopy` and `roofs` are `None` when the survey did not classify them — the normal case,
    and the thing to check before planning any work that needs them. With ``interval=True``
    every grid comes back as an :class:`~ducklidar.estimate.Estimate`.
    """
    from .fields import BUILDING, GROUND, VEGETATION

    c = pts["classification"]
    present = set(np.unique(c).tolist())
    is_ground = (c == GROUND) if ground is None else np.asarray(ground, dtype=bool)

    def grid(mask, how):
        return rasterize(pts, bbox, pix, mask=mask, how=how, interval=interval)

    return {
        "dsm": grid(None, "max"),
        "dtm": grid(is_ground, "min"),
        "canopy": (grid(np.isin(c, VEGETATION), "max")
                   if set(VEGETATION) & present else None),
        "roofs": grid(c == BUILDING, "max") if BUILDING in present else None,
    }


def distance_to_evidence(grid, pix=1.0):
    """Metres from every cell to the nearest cell that holds a real measurement.

    How far a filled value is from anything that was actually measured — the layer a filled
    DTM should never travel without. It is not a confidence in the statistical sense; it is
    the distance over which an interpolator had to invent. Interpolation next to a
    measurement is well constrained; interpolation 90 m out is invented terrain, and no
    algorithm changes that — an interpolator will happily produce a smooth, plausible surface
    over open water. Measured on one urban window: half the voids sit within 10 m of a real
    ground return, but the tail reaches 98 m.
    """
    from scipy import ndimage

    a = grid.value if isinstance(grid, Estimate) else grid
    return ndimage.distance_transform_edt(~np.isfinite(a)) * pix


def void_report(grid, pix=1.0):
    """How empty a grid is, how the holes clump, and how far they are from evidence.

    Returns a dict — `filled`, `n_voids`, `largest_cells`, `largest_ha`, and the distance
    percentiles. Worth printing next to any grid that will be interpolated, so nobody
    mistakes a half-empty array for a finished surface.
    """
    from scipy import ndimage

    a = grid.value if isinstance(grid, Estimate) else grid
    hole = ~np.isfinite(a)
    lbl, n_voids = ndimage.label(hole)
    sizes = np.bincount(lbl.ravel())[1:] if n_voids else np.array([0])
    d = distance_to_evidence(a, pix)[hole] if hole.any() else np.array([0.0])
    return {
        "filled": float(np.isfinite(a).mean()),
        "n_voids": int(n_voids),
        "largest_cells": int(sizes.max()),
        "largest_ha": float(sizes.max() * pix**2 / 1e4),
        "evidence_p50": float(np.percentile(d, 50)),
        "evidence_p90": float(np.percentile(d, 90)),
        "evidence_max": float(d.max()),
    }
