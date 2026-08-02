"""A DEM, and an honest account of how much of it was measured.

Three words that get used interchangeably and are not the same thing:

======  ==========================================  ===========================
term    what it is                                  how it is made here
======  ==========================================  ===========================
DSM     the top of everything — roofs, crowns       ``rasterize(how="max")``
DTM     bare earth, as if the city were removed     ground returns only, gridded
DEM     the umbrella term for either                say which one you mean
======  ==========================================  ===========================

**A DTM is mostly not measured.** A pulse cannot reach the ground under a roof, so the grid of
ground returns comes back full of holes wherever a building stands — and every DTM you have ever
opened has had those holes filled by interpolation and then handed to you as if it were data.
The filling is a *model*. This module does it, and refuses to hide it: :func:`dem` returns the
raw grid, the filled grid, and the distance from every cell to the nearest real measurement, so
"is this number a measurement?" stays answerable afterwards.

The interpolator is chosen by measurement, not by convention — see :func:`fill` and
``docs/dem.md`` for the hold-out test.
"""
from __future__ import annotations

import numpy as np

from .grid import distance_to_evidence, rasterize


CROSSOVER = 5.0   # m from evidence where `nearest` stops winning and `idw` starts


def _idw(pts, vals, targets, k=12):
    """Inverse distance weighting over the k nearest measurements."""
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    d, idx = tree.query(targets, k=min(k, len(pts)))
    d, idx = np.atleast_2d(d), np.atleast_2d(idx)
    w = 1.0 / np.maximum(d, 1e-6) ** 2
    return (w * vals[idx]).sum(1) / w.sum(1)


def fill(grid, pix=1.0, *, method="auto", max_gap=None):
    """Interpolate the holes. Returns ``(filled, distance_to_evidence)``.

    ``method``:

    ``linear``
        Delaunay triangulation of the measured cells, then planar interpolation inside each
        triangle. This is the classic TIN DTM and it is the default because it is what the
        surface actually is between two ground hits: a slope. [measured] see ``docs/dem.md``.
    ``nearest``
        Copy the closest measurement. Fast, exactly right at the edges of the convex hull, and
        it produces flat plateaus that look like terraces on a hillside.
    ``idw``
        Inverse distance weighting over the k nearest measurements. Smooth, and it pulls every
        hole toward the local mean — which flattens valleys and bulges ridges.
    ``auto``
        **The default.** ``nearest`` within :data:`CROSSOVER` metres of evidence, ``idw`` beyond
        it, decided per cell from ``distance_to_evidence``.

    Why ``auto`` and not the ``linear`` this used to default to: [measured] by holding real
    ground returns back in BUILDING-SHAPED blocks, which is what a void actually looks like —
    scattering the held-out cells instead puts every one of them 1–2 m from surviving evidence
    and tests nothing. Mean absolute error, in metres, by distance to the nearest remaining
    ground return:

    ==========  =========  ========  =======
    distance    nearest    linear    idw
    ==========  =========  ========  =======
    1–2 m       **0.082**  0.105     0.096
    2–5 m       **0.170**  0.183     0.175
    5–10 m      0.324      0.324     **0.313**
    >10 m       0.506      0.556     **0.479**
    ==========  =========  ========  =======

    They cross at about 5 m, so no single method is right everywhere and ``linear`` — the old
    default — wins nowhere on that test. Delaunay across a wide hole makes long thin triangles
    that overshoot: its p95 is the worst of the three (1.597 m) while its RMSE is the best, which
    is a method that is right on average and occasionally badly wrong. See ``docs/dem.md``.

    ``max_gap`` in metres leaves anything further than that from a real measurement as NaN.
    A 60 m warehouse has a middle 30 m from the nearest ground return; interpolating it invents
    a floor nobody saw. Refusing to guess is a legitimate answer and the default is *not* to,
    only because most consumers of a DEM cannot handle NaN — set it when yours can.
    """
    from scipy.interpolate import griddata

    g = np.asarray(grid, float)
    known = np.isfinite(g)
    if not known.any():
        raise ValueError("nothing measured — cannot interpolate from an empty grid")
    dist = distance_to_evidence(g, pix)
    if known.all():
        return g.copy(), dist

    ny, nx = g.shape
    yy, xx = np.nonzero(known)
    pts = np.column_stack([xx, yy]).astype(float)
    vals = g[known]
    ty, tx = np.nonzero(~known)
    targets = np.column_stack([tx, ty]).astype(float)

    if method == "nearest":
        out_vals = griddata(pts, vals, targets, method="nearest")
    elif method == "idw":
        out_vals = _idw(pts, vals, targets)
    elif method == "auto":
        # per cell, not per grid: the two methods cross at CROSSOVER and neither wins twice
        near = dist[ty, tx] < CROSSOVER
        out_vals = np.empty(len(targets))
        if near.any():
            out_vals[near] = griddata(pts, vals, targets[near], method="nearest")
        if (~near).any():
            out_vals[~near] = _idw(pts, vals, targets[~near])
    elif method == "linear":
        out_vals = griddata(pts, vals, targets, method="linear")
        # Delaunay covers only the convex hull; outside it, fall back rather than leave NaN
        gap = ~np.isfinite(out_vals)
        if gap.any():
            out_vals[gap] = griddata(pts, vals, targets[gap], method="nearest")
    else:
        raise ValueError(f"unknown method {method!r}")

    filled = g.copy()
    filled[ty, tx] = out_vals
    if max_gap is not None:
        filled[dist > max_gap] = np.nan
    return filled, dist


def dem(pts, bbox, pix=1.0, *, ground=None, method="auto", max_gap=None):
    """A DTM with its own provenance attached.

    Returns a dict:

    ``dtm``
        measured only — NaN wherever no ground return landed. This is the honest grid.
    ``filled``
        ``dtm`` with the holes interpolated. Use it, but do not call it data.
    ``distance``
        metres from each cell to the nearest real ground return. **This is the layer that makes
        the other two usable**: at 0 m the value is measured, at 15 m it is a guess about the
        middle of a warehouse.
    ``dsm``
        the top of everything, for context and for nDSM.
    ``stats``
        void fraction, the widest hole, and how much of the fill is far from evidence.

    ``ground`` is a boolean mask over `pts`. Pass one from :func:`ducklidar.ground_filter`;
    leaving it None uses the survey's own classification, which is someone else's answer and
    absent from many surveys.
    """
    from .fields import GROUND

    c = np.asarray(pts["classification"])
    is_ground = (c == GROUND) if ground is None else np.asarray(ground, bool)
    if not is_ground.any():
        raise ValueError("no ground returns — classify first, or pass ground=")

    raw = rasterize(pts, bbox, pix, mask=is_ground, how="min")
    dsm = rasterize(pts, bbox, pix, how="max")
    filled, dist = fill(raw, pix, method=method, max_gap=max_gap)

    void = float(np.isnan(raw).mean())
    far = float((dist > 5.0).mean())
    return {"dtm": raw, "filled": filled, "distance": dist, "dsm": dsm,
            "stats": {"void": void, "far_from_evidence": far,
                      "max_distance": float(np.nanmax(dist)),
                      "ground_returns": int(is_ground.sum()), "pix": pix,
                      "method": method}}


def footprint(pts, bbox, member, pix=1.0):
    """The grid cells occupied by one labelled object's points.

    The link between the two halves of a labelling pipeline. Once a point carries an
    ``object_id`` the footprint is not something to be drawn or fetched — it is wherever that
    object's own returns landed:

        foot = dl.footprint(pts, bbox, object_id == 7)
        datum = dl.building_level(dem_result, foot)["level"]

    ``member`` is a boolean mask over `pts`. Points outside `bbox` are dropped rather than
    clamped to the edge, which would smear a building onto the rim of the grid.
    """
    from .grid import cell_index

    member = np.asarray(member, bool)
    if not member.any():
        raise ValueError("no points in this object")
    flat, shape = cell_index({k: np.asarray(pts[k])[member] for k in ("x", "y", "z")}, bbox, pix)
    ny, nx = shape
    flat = flat[(flat >= 0) & (flat < ny * nx)]
    out = np.zeros(ny * nx, bool)
    out[flat] = True
    return out.reshape(ny, nx)


def building_level(dem_result, footprint, *, ring=3, tilt_max=0.02):
    """The single ground datum under one building, and how much to trust it.

    A building sits on a pad, so its height should be measured against **one** level rather than
    against a bowl the interpolator invented. Nothing is measured under a footprint, so the level
    has to come from a robust statistic over the filled cells — and *which* statistic is not a
    matter of taste.

    [measured] on 61 real footprints, each candidate statistic minus the level of the **measured**
    ring of ground returns around the building:

    ==========  =================  ==================
    statistic   level sites (46)   sloped sites (15)
    ==========  =================  ==================
    min         −0.341 m           −1.540 m
    p10         −0.045 m           −0.999 m
    **p25**     **+0.025 m**       −0.487 m
    median      +0.101 m           **+0.147 m**
    mean        +0.119 m           −0.013 m
    ==========  =================  ==================

    So ``min`` — the obvious choice — is biased low by 34 cm on level ground and 1.5 m on a
    slope, worst case 4.4 m, because a minimum over hundreds of cells samples the lowest
    excursion of an *interpolated* surface. Every building would come out that much too tall.
    ``p25`` is unbiased to 2.5 cm where the site is level, which is 75% of them here, and
    ``median`` is the safer of the two once the ground genuinely tilts.

    ``ring`` is how many cells outside the footprint count as its perimeter; the tilt of a plane
    fitted through those measured cells decides which statistic is used, above ``tilt_max`` in
    metres of fall per metre.

    Returns a dict: ``level``, the ``statistic`` chosen, the ring ``tilt``, the ring's own
    ``ring_level`` for comparison, and ``ring_cells``. Read `level` against `tilt`: on a steep
    site a building has no single level and this says so rather than pretending.
    """
    from scipy import ndimage as ndi

    foot = np.asarray(footprint, bool)
    filled = np.asarray(dem_result["filled"], float)
    raw = np.asarray(dem_result["dtm"], float)
    if foot.shape != filled.shape:
        raise ValueError(f"footprint {foot.shape} does not match the grid {filled.shape}")
    if not foot.any():
        raise ValueError("empty footprint")

    band = ndi.binary_dilation(foot, iterations=ring) & ~foot & np.isfinite(raw)
    tilt, ring_level = float("nan"), float("nan")
    if band.sum() >= 8:
        yy, xx = np.nonzero(band)
        z = raw[band]
        A = np.column_stack([xx, yy, np.ones(len(xx))])
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        tilt = float(np.hypot(coef[0], coef[1]))
        ring_level = float(np.median(z))

    inner = filled[foot]
    inner = inner[np.isfinite(inner)]
    if not len(inner):
        raise ValueError("no filled ground under this footprint")
    # a ring too small to fit a plane through tells us nothing about the slope, so take the
    # statistic that is merely good rather than the one that is best only when level
    stat = "median" if (not np.isfinite(tilt) or tilt >= tilt_max) else "p25"
    level = float(np.median(inner) if stat == "median" else np.percentile(inner, 25))
    return {"level": level, "statistic": stat, "tilt": tilt,
            "ring_level": ring_level, "ring_cells": int(band.sum()),
            "cells": int(foot.sum())}


def ndsm(dem_result):
    """Height above ground: DSM minus the filled DTM.

    Uses `filled`, necessarily — a roof sits exactly where the DTM has no measurement, so an
    nDSM computed from the raw grid is NaN on every building, which is the one place it is
    wanted. Read it alongside `distance`: a roof height is DSM minus a guess.
    """
    return dem_result["dsm"] - dem_result["filled"]
