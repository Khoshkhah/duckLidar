"""Where the aircraft was — recovered from the returns, with no trajectory file.

A survey ships points, not the flight path. But the path is implicit in the points, because a
laser pulse travels in a straight line: **every return of one pulse is collinear with the
sensor.** A pulse that splits on a tree and again on the ground gives two points, and the line
through them passes through the aircraft. Collect many such lines fired at nearly the same
instant and they meet at one place — where the aircraft was.

That is Roussel et al. (2020), what lidR's ``track_sensor`` implements, and it uses no scan
angle at all. It is pure geometry from the point positions, so it is independent of the mirror
angle the file records, which makes the two a genuine cross-check rather than two views of one
assumption.

**The conditioning is the whole game.** The lines are near-vertical, so they are near-parallel,
and near-parallel lines intersect badly. What saves it is angular spread: a window that catches
the mirror at both ends of its sweep has lines fanned across 40°, and one that catches only a
swath edge has them all within 5°. [measured] on the pilot tile, per flight line, the recovered
height's own standard deviation tracks exactly that:

============  ==================  ===========  =========
flight line   scan angles seen    height s.d.  m/s
============  ==================  ===========  =========
223           −15.9 … 19.9°       **8.3 m**    63.9
224           −9.1 … 19.9°        10.9 m       73.2
225           −19.9 … −9.5°       30.4 m       64.7
222           −19.9 … −1.0°       31.4 m       73.2
221           16.0 … 19.9°        71.2 m       77.7
172           14.1 … 20.0°        72.1 m       82.7
============  ==================  ===========  =========

So a line that only clips your extent with the far edge of its swath gives an answer nine times
noisier than one that flies over it. :func:`sensor_track` returns the conditioning per window so
that can be filtered on rather than discovered later.

Recovered for the pilot tile: **~1400 m above the ground, 64–83 m/s**, six lines, five parallel
east–west ~400 m apart flown alternately in opposite directions, plus one north–south tie line
flown 45 hours later. None of which is written down anywhere in the file.
"""
from __future__ import annotations

import numpy as np


def pulses(pts):
    """First and last return of every multi-return pulse, as (first_xyz, unit_up, time).

    Returns of one pulse are identified by a shared ``gps_time`` inside one flight line — the
    file records the firing time, not the echo time, so they are bit-identical. Checked on the
    pilot tile: every multi-return group has exactly one distinct time.

    The direction points from the LAST return to the FIRST, i.e. back up the beam toward the
    aircraft, which is the half of the line the sensor is actually on.
    """
    t = np.asarray(pts["gps_time"])
    nr = np.asarray(pts["number_of_returns"])
    rn = np.asarray(pts["return_number"])
    xyz = np.column_stack([np.asarray(pts[k], float) for k in "xyz"])

    order = np.lexsort((rn, t))
    t, nr, rn, xyz = t[order], nr[order], rn[order], xyz[order]

    first = np.flatnonzero((rn == 1) & (nr > 1))
    last = first + nr[first] - 1
    # A pulse split across a chunk boundary, or a malformed count, would index into the next
    # pulse; require the two ends to still share a firing time.
    ok = (last < len(t)) & (t[np.minimum(last, len(t) - 1)] == t[first])
    first, last = first[ok], last[ok]

    p = xyz[first]
    u = p - xyz[last]
    n = np.linalg.norm(u, axis=1, keepdims=True)
    good = n[:, 0] > 1e-6            # a zero-length pair carries no direction
    return p[good], (u / np.maximum(n, 1e-12))[good], t[first][good]


def _intersect(p, u):
    """The point minimising squared distance to every line ``p + s·u``.

    Each line contributes the projector ``I − uuᵀ``, which measures displacement perpendicular
    to it. Summing gives normal equations solvable in closed form. The eigenvalue ratio of that
    sum is returned as the conditioning: 1 would mean the lines fan in every direction, 0 that
    they are parallel and the intersection is undefined along the shared axis.
    """
    m = len(u) * np.eye(3) - u.T @ u
    b = (p - np.einsum("ij,ij->i", u, p)[:, None] * u).sum(0)
    ev = np.linalg.eigvalsh(m)
    if ev.max() <= 0:
        return np.full(3, np.nan), 0.0
    return np.linalg.solve(m, b), float(ev.min() / ev.max())


def sensor_track(pts, *, window=0.5, min_pulses=500):
    """Sensor position per time window: ``(t, xyz, conditioning)``, one row per window.

    `window` in seconds trades two errors against each other: shorter windows assume less about
    the aircraft holding still, longer ones fan the beam wider and condition the solve better.
    0.5 s is ~35 m of travel at survey speed and spans a good part of one mirror sweep.

    Filter on the returned conditioning — see the table in this module's docstring for what
    happens when you do not.
    """
    p, u, t = pulses(pts)
    if len(t) < min_pulses:
        return np.empty(0), np.empty((0, 3)), np.empty(0)

    edges = np.arange(t.min(), t.max(), window)
    rows = []
    for e in edges:
        idx = np.flatnonzero((t >= e) & (t < e + window))
        if len(idx) < min_pulses:
            continue
        s, cond = _intersect(p[idx], u[idx])
        rows.append((e + window / 2, s, cond))
    if not rows:
        return np.empty(0), np.empty((0, 3)), np.empty(0)
    return (np.array([r[0] for r in rows]),
            np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows]))


def tracks(pts, *, window=0.5, min_returns=50_000, min_pulses=500):
    """:func:`sensor_track` per flight line — ``{point_source_id: (t, xyz, cond)}``.

    Split by line first, always: two lines overflying the same ground at different times give
    pulses whose beams cross, and intersecting *those* returns a point somewhere between two
    aircraft positions that the aircraft never occupied.
    """
    line = np.asarray(pts["point_source_id"])
    out = {}
    for code in np.unique(line):
        m = line == code
        if m.sum() < min_returns:
            continue
        sub = {k: np.asarray(v)[m] for k, v in pts.items()}
        t, pos, cond = sensor_track(sub, window=window, min_pulses=min_pulses)
        if len(t):
            out[int(code)] = (t, pos, cond)
    return out
