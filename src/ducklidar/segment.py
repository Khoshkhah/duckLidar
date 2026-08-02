"""Separating a surface into instances, using disagreement between views.

The idea this is built on: a survey looks at the same ground several times, from several flight
lines, at several angles. :func:`ducklidar.grid.rasterize` with ``interval=True`` gives a
per-cell interval width for *one* pool of returns. Compute that **per view** instead and
combine the views, and the two ways of combining answer different questions:

* **max** across views — *any* view saw structure here. Noisy: one grazing pass catching a
  facade marks the cell, and the map lights up almost everywhere.
* **product** across views — *every* view saw structure here. Conservative, and that is the
  point: a cell survives only if the structure is visible from every direction the survey
  looked, which is what a real vertical discontinuity does.

**[measured]** On a 250 m Granville window the product map splits into two distinct signatures,
and the split is the useful part:

* **buildings show only their outline** — a flat roof reads ~0 from every view in its interior,
  and metres at its edge.
* **trees show a filled blob** — a crown is porous, so every view disagrees with itself across
  the whole canopy, not just at its rim.

So thresholding the product and keeping what is *below* it removes vegetation as a side effect,
with no classification field and no species assumption — the geometry does it.
"""
from __future__ import annotations

import numpy as np

from .grid import rasterize

#: No floor, by design. A look that saw a flat surface contributes exactly 0, and that zero
#: propagates — which is the whole point of multiplying rather than averaging: **one look
#: seeing flat ground is enough to say there is no edge here.** Raise it only to soften the AND
#: into something closer to a mean, and know that is what you are doing.
WIDTH_FLOOR = 0.0


def _default_by(pts):
    """The 2-part direction channel when the fields for it exist, else the flight line."""
    if all(k in pts for k in ("gps_time", "scan_angle", "point_source_id")):
        return viewed_from(pts, parts=2)
    return "point_source_id"


def edge_strength(pts, bbox, pix=1.0, *, by=None, floor=WIDTH_FLOOR):
    """Per-cell interval width computed per view, then multiplied across views.

    Returns the **geometric mean**, not the raw product: cells differ in how many views saw
    them, and a raw product would rank a cell three views agreed on below one that two did.

    `by` defines a view: a field name, or an array of group codes per return. The default
    (``by=None``) is :func:`viewed_from(parts=2) <viewed_from>` — two direction bundles that
    each cover the whole map with no returns dropped, so 93.5% of cells get the *identical*
    2-look test. Flight lines score slightly higher on vegetation removal (74% against 70%),
    but only by applying stronger multi-look tests on the minority of cells that support them —
    an inhomogeneous test. Falls back to ``"point_source_id"`` when `gps_time` or `scan_angle`
    is absent. ``docs/instances.md`` has the measured comparison.

    **A view with no evidence is dropped, not counted as flat.** A view holding one return in a
    cell reports width exactly 0 and ``method="none"`` — that is ignorance, and multiplying it in
    would erase a real edge the other views agree on. **[measured]** 2% of per-view cells are
    like that, and letting them through zeroes 5.7% of the map instead of 1.7%.

    **Zero propagates, and that is deliberate.** A look with evidence that reports width exactly
    0 saw a flat surface, and one such look is enough to conclude there is no edge here — the
    result is exactly 0, not a small number. That is the difference between multiplying and
    averaging, and it is why `floor` defaults to 0. Note this is the *opposite* case to the
    paragraph above: no evidence means the look is **dropped**; evidence of flatness means the
    whole product is **zeroed**.

    Measured flatness rarely lands on exactly 0 — returns differ by centimetres, so a flat roof
    reads ~0.04 m — but where it does, the zero is the answer and is kept.

    NaN where nothing landed.
    """
    if by is None:
        by = _default_by(pts)
    key = np.asarray(pts[by] if isinstance(by, str) else by)
    views = np.unique(key)
    e = [rasterize({k: v[key == g] for k, v in pts.items()}, bbox, pix, interval=True)
         for g in views]
    w = np.stack([x.width for x in e])
    if floor:
        w = np.clip(w, floor, None)
    ok = np.isfinite(w) & (np.stack([x.method for x in e]) > 0)
    k = ok.sum(0)

    flat = np.any(ok & (w <= 0), axis=0)              # one flat look zeroes the product
    pos = ok & (w > 0)
    n_pos = pos.sum(0)
    logw = np.where(pos, np.log(np.where(pos, w, 1.0)), 0.0)
    gm = np.where(n_pos > 0, np.exp(logw.sum(0) / np.maximum(n_pos, 1)), 0.0)
    return np.where(k > 0, np.where(flat, 0.0, gm), np.nan)


def angle_rank(pts, bbox, pix=1.0, *, max_rank=4):
    """Per-return channel code: **how oblique this cell's looks were, ranked within the cell.**

    Binning \\|scan angle\\| globally does not answer "at what angle was *this cell* seen". Scan
    angle is across-track distance from the aircraft — ``h·tan(θ)`` — so a global band is a
    *place*, a pair of strips parallel to the flight path, not a property of a cell.
    **[measured]** each of four quartile bands reaches only 36–68% of the window, and 25% of
    cells are reached by one band alone. Which band a cell lands in says where it is, not how it
    was looked at.

    Ranking fixes the frame of reference. Each `(cell, flight line)` pair is one look at that
    cell; sort a cell's looks by their mean \\|scan angle\\| and number them. Rank 0 is *the most
    vertical look this cell got*, rank 1 the next, and so on — a question every cell answers for
    itself, in its own terms.

    Ranks above `max_rank` are folded into the last one, so the channel count stays bounded.
    Returns an int array over `pts`, ready to pass as ``by=``.
    """
    from .grid import cell_index

    flat, _ = cell_index(pts, bbox, pix)
    src = np.asarray(pts["point_source_id"], dtype="int64")
    sa = np.abs(np.asarray(pts["scan_angle"], dtype="float64") * 0.006)

    span = int(src.max()) + 1
    uk, inv = np.unique(flat.astype("int64") * span + src, return_inverse=True)
    mean_ang = np.bincount(inv, weights=sa) / np.bincount(inv)      # angle of each look
    cells = uk // span

    order = np.lexsort((mean_ang, cells))                # steepest last, within each cell
    cs = cells[order]
    rank_sorted = np.arange(len(cs)) - np.searchsorted(cs, cs, "left")
    rank = np.empty(len(uk), dtype="int64")
    rank[order] = np.minimum(rank_sorted, max_rank - 1)
    return rank[inv]


#: Quadrant names in the order :func:`viewed_from` codes them: 0 N, 1 E, 2 S, 3 W.
COMPASS = ("north", "east", "south", "west")


def look_azimuth(pts):
    """Compass direction the beam **travelled**, per return. Degrees from north.

    Recoverable although it is not stored: a line's heading is how `x, y` advance with
    `gps_time`, and the sign of `scan_angle` says which side of the aircraft the beam went
    (negative is left of the flight direction, positive is right). So ``travel = heading ± 90°``.

    Most questions want :func:`viewed_from` instead — the *opposite* direction, which is the one
    that says which side of a thing was lit. Beam travelling north means the scanner was to the
    south.

    Needs `gps_time` and `scan_angle`. Returns float degrees in [0, 360).
    """
    x, y = np.asarray(pts["x"]), np.asarray(pts["y"])
    t, line = np.asarray(pts["gps_time"]), np.asarray(pts["point_source_id"])
    out = np.empty(len(x))
    for ln in np.unique(line):
        m = line == ln
        o = np.argsort(t[m])
        dx = np.polyfit(t[m][o], x[m][o], 1)[0]
        dy = np.polyfit(t[m][o], y[m][o], 1)[0]
        heading = np.degrees(np.arctan2(dx, dy))
        out[m] = heading + np.where(np.asarray(pts["scan_angle"])[m] >= 0, 90.0, -90.0)
    return out % 360.0


def viewed_from(pts, *, quadrant=False, parts=None):
    """**Which side of the cell the scanner was on**, per return. Degrees from north.

    This is the direction that matters for structure, and it is the *opposite* of
    :func:`look_azimuth`: a beam travelling north came from the south, so it lit the
    **south-facing** side of whatever it hit. Splitting a cell's returns this way asks "was this
    cell ever looked at from the east?" — which neither the flight line nor the scan angle can
    answer, and which decides whether a given wall exists in the data at all.

    With ``quadrant=True`` returns codes 0–3 indexing :data:`COMPASS`, ready to pass as ``by=``.

    **[measured]** Vancouver 2022, Granville, of the cells holding any return:

    ==============  =========  ========
    scanner was     returns    reaches
    ==============  =========  ========
    north           5,000,445  97.2%
    south           4,479,060  97.1%
    west            1,240,932  27.2%
    **east**        **0**      **0%**
    ==============  =========  ========

    Two nearly complete looks from opposite sides, one partial third, and a fourth that does not
    exist — three of the four lines fly east–west and only one flies north–south. **No cell in
    this window was ever viewed from the east**, so no east-facing wall is in the data anywhere,
    at any density. That is a property of the flight plan, not of the processing, and no amount
    of cleverness recovers it.

    It is also the best-covering of the three channel definitions: 67.3% of cells were seen from
    exactly two sides and 27.1% from three, against flight lines that individually reach 23–80%.

    **``parts=k`` merges the looks into k direction bundles, dropping nothing.** A *look* is one
    flight line sweeping one side (line × scan-angle sign); each has a single azimuth. Looks are
    merged agglomeratively by circular distance between bundle centres until `k` remain, and the
    codes come back per return, ready for ``by=``. This exists because compass quadrants leave a
    part that cannot cover the map: **[measured]** this survey's five looks sit at 359.6°, 5.6°,
    185.6°, 190.5° and 271.0°; ``parts=2`` gives {356–6°} and {186–190° ∪ 271°} — the lone 271°
    look (line 172, 27% coverage) joins the southern bundle because 80.5° beats 88.6°. Both
    parts then cover ~95% of the window and **93.5% of cells get the identical 2-look test**,
    against 27% for the 3-quadrant split and 12% for flight lines.
    """
    az = (look_azimuth(pts) + 180.0) % 360.0
    if parts is not None:
        line = np.asarray(pts["point_source_id"], dtype="int64")
        sign = (np.asarray(pts["scan_angle"]) >= 0).astype("int64")
        uk, first, inv = np.unique(line * 2 + sign, return_index=True, return_inverse=True)
        look_az, w = az[first], np.bincount(inv).astype("float64")

        def centre(c):
            a = np.radians(look_az[c])
            return np.degrees(np.arctan2((w[c] * np.sin(a)).sum(),
                                         (w[c] * np.cos(a)).sum())) % 360.0

        # ponytail: O(n^3) agglomerative merge — n is the number of looks, single digits
        clusters = [np.array([i]) for i in range(len(uk))]
        while len(clusters) > parts:
            best = None
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    d = abs(centre(clusters[i]) - centre(clusters[j]))
                    d = min(d, 360.0 - d)
                    if best is None or d < best[0]:
                        best = (d, i, j)
            clusters[best[1]] = np.concatenate([clusters[best[1]], clusters.pop(best[2])])
        code = np.empty(len(uk), dtype="int64")
        for rank, ci in enumerate(np.argsort([centre(c) for c in clusters])):
            code[clusters[ci]] = rank
        return code[inv]
    return (((az + 45.0) % 360.0) // 90.0).astype("int64") if quadrant else az


def hysteresis(S, hi=0.5, lo=0.15):
    """Boundary mask by hysteresis: seeds above `hi`, extended through connected cells above `lo`.

    The single-threshold boundary treats cells independently and throws away the one thing rim
    cells have — that they form *lines*, strong at corners and weak along the flanks. Hysteresis
    keeps a weak cell only when it is connected to a strong one, which is Canny's linking
    assumption and, [measured], exactly our geometry: on the default pipeline it moves the
    result from 68% building / 79% vegetation dropped to **74% / 95%** (hi 0.5, lo 0.15) — the
    largest single improvement any change has produced. `lo` carries the effect; `hi` barely
    matters (0.5 vs 1.0 within a point). Below lo≈0.05 the boundary floods a third of the map
    and starts eating buildings; the default keeps a margin above that cliff.

    NaN in `S` counts as 0 — no evidence is not a boundary. See ``docs/boundaries.md`` for the
    survey this was chosen from.
    """
    from scipy import ndimage as ndi

    Sv = np.nan_to_num(S, nan=0.0)
    strong, weak = Sv > hi, Sv > lo
    lab, _ = ndi.label(weak)
    keep = np.unique(lab[strong & (lab > 0)])
    return np.isin(lab, keep[keep > 0])


def bridge_deck(pts, bbox, pix=1.0, *, void=6.0, echo=0.5, min_cells=300, top=1.5):
    """Per-point mask of returns on a bridge deck, found by geometry alone.

    This survey classifies no bridges: there is no ASPRS class 17 anywhere in the tile, and
    100% of the 9,735 returns standing above 5 m over open water — the Granville deck at a
    median 39.6 m over False Creek — are filed as class 1, unclassified. So the deck is in the
    cloud but unnamed, and a viewer colours it the same grey as everything else the vendor
    left unlabelled.

    A deck has one signature nothing else in a city shares: **a hard surface with a void under
    it in the same cell**. Two tests, in that order, because neither is sufficient alone.

    *Void* — the cell holds returns at least `void` metres below its own top. A building roof
    fails this: the pulse stops at the roof and nothing beneath is ever measured. Only walls
    fail it at the perimeter, one cell wide, which the 3×3 opening removes.

    *Hard* — echo ratio below `echo`. This is the test that matters: on its own the void
    condition marks 21.8% of the grid and 23 of its 25 large components are tree crowns, which
    also have a top surface with ground beneath. Foliage splits pulses and a deck does not.

    [measured] on the pilot tile the two together leave **exactly two components**: 12.0 × 383 m
    and 11.6 × 396 m, tops at 34.1 and 33.7 m, top layers 162,852 and 163,068 returns of which
    19 and 23 are classified building and none are vegetation. Those are Granville Bridge's two
    carriageways. No elongation test is applied — both survivors have length/width above 30, so
    it would not change this answer, and a shape filter tuned on one bridge is a filter tuned on
    one bridge. Loosening `echo` to 0.2 shatters the deck into three pieces where the surface is
    wet or grated; 0.5 holds it together.

    Returns a boolean array over `pts`: True for returns within `top` metres of their cell's
    top surface, in a surviving component. The layer below — water, road, whatever the deck
    spans — is deliberately not included, since it is not deck.
    """
    from scipy import ndimage as ndi

    from .grid import cell_index

    flat, (ny, nx) = cell_index(pts, bbox, pix)
    z = np.asarray(pts["z"])
    zmax = np.full(ny * nx, -np.inf)
    np.maximum.at(zmax, flat, z)
    below = np.zeros(ny * nx, bool)
    np.logical_or.at(below, flat, z < zmax[flat] - void)

    er = np.nan_to_num(echo_ratio(pts, bbox, pix), nan=1.0)   # NaN cells are too sparse to trust
    cand = ndi.binary_opening(below.reshape(ny, nx) & (er < echo), np.ones((3, 3)))
    lab, _ = ndi.label(cand, np.ones((3, 3)))
    cnt = np.bincount(lab.ravel())
    cnt[0] = 0
    keep = np.isin(lab, np.flatnonzero(cnt >= min_cells)).ravel()
    return keep[flat] & (z > zmax[flat] - top)


def echo_ratio(pts, bbox, pix=1.0, *, min_n=4):
    """Per-cell share of returns whose pulse split — the literature's **echo ratio**.

    A pulse that meets foliage keeps going and records several echoes; one that meets a roof
    stops dead. So the fraction of a cell's returns belonging to multi-return pulses is a
    *material* measurement, and it is the strongest single discriminator in this project:

    ==============  ==========  ==========  =================
    surface         echo ratio  vs ground   what it is
    ==============  ==========  ==========  =================
    open ground     3.0%        —           pulses stop dead
    roof            6.5%        ×2.2        stop dead, plus edge hits
    **crown**       **74.1%**   **×25**     porous
    ==============  ==========  ==========  =================

    ×25 against the interval width's ×8.9, from physics the width cannot see. Reported at ~98%
    correctness for urban vegetation in the full-waveform literature (Höfle & Pfeifer and
    after), which is consistent.

    **It also fires on glass.** Kaveh raised this, and it is real: a glazed roof or skylight
    splits a pulse much as foliage does. [measured] the cut removes 6.5% of roof interiors —
    but 89% of that is genuinely tree-like (local height range 3.53 m, against crowns' 3.86 m):
    overhanging branches and rooftop clutter. The true glass candidates are the **flat** ones,
    **0.7% of roof interior**.

    Two rescue rules were tried and neither is worth adopting. Keeping locally flat cells
    recovers 0.8 points of roof for 26 points of crown removal. Keeping locally *planar* ones
    (rms residual from a fitted plane, the physically right discriminator since glass is planar
    and canopy is not) recovers 0.4–0.9 points for 0–4 points of crown — better, but still not
    a trade worth taking, because a dense crown's interior is planar too at 5×5. **[measured]**
    kept roof rms 0.54 m, removed-roof rms 1.33 m, crown rms 1.43 m: the removed roof cells sit
    with the crowns, not with the roofs, which is the evidence that they mostly *are* vegetation.

    So the 0.7% is a known, quantified, unfixed cost. Separating glass from foliage properly
    needs intensity or full-waveform, which this survey does not calibrate.

    NaN where a cell holds fewer than `min_n` returns — a ratio over two returns is noise.
    Needs `return_number` and `number_of_returns`.
    """
    from .grid import cell_index

    flat, (ny, nx) = cell_index(pts, bbox, pix)
    multi = np.asarray(pts["number_of_returns"]) > 1
    tot = np.bincount(flat, minlength=ny * nx).astype("float64")
    hit = np.bincount(flat[multi], minlength=ny * nx)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(tot >= min_n, hit / np.maximum(tot, 1), np.nan)
    return r.reshape(ny, nx)


def local_boundary(S, *, size=15, nbr_pct=50, t1_pct=95, t2_pct=75):
    """Kaveh's boundary rule: an absolute floor, a local rank test, and an absolute override.

    ::

        mask = S > percentile_nbr_pct(neighbourhood)     larger than a good many neighbours
        edge = (S > t1)  OR  (S > t2  AND  mask)

    Three terms, and each is load-bearing:

    * **`t2`, the floor.** Without it a *relative* rule is unusable on a field whose flat
      baseline is near zero: on the widths field a flat surface reads ~0.04 m, so "30% above
      the neighbourhood" is 0.012 m and noise clears it everywhere — **[measured]** 61% of
      cells marked, visually pure confetti. `t2` is what makes a local rule survive contact
      with a low-baseline field.
    * **the mask, a rank test.** A rim's width is 0.06–0.14 m, far below any usable global cut,
      because a rim is a small bump above *its own* flat roof. Comparing against the
      neighbourhood is what finds it. A percentile reference beats mean/std and morphological
      ones — **[measured]** F1 71% against Sauvola 43%, Niblack 52%, white top-hat 61%, h-dome
      48%: this field's heavy tail drags any mean or opening upward, and a percentile ignores it.
    * **`t1`, the override.** A cell strong enough in absolute terms is a boundary whatever its
      neighbours do — it rescues long uniform runs where the neighbourhood is high too and the
      local test goes quiet.

    **[measured]** boundary quality (outlines of buildings and crowns, 2-cell tolerance — this
    is a boundary and is judged as one, never by the instance step):

    ===================================  =========  ======  =====
    rule                                 precision  recall  F1
    ===================================  =========  ======  =====
    global single cut                    44%        58%     50%
    global hysteresis                    49%        70%     58%
    **this**                             50%        92%     **65%**
    ===================================  =========  ======  =====

    On the counts field a two-sided variant of the mask with connected linking reaches 71%, but
    it does not transfer to widths; this formulation works on both, which is why it is the one
    exposed. Percentiles are taken over the finite values of `S`, so the same call fits any
    field regardless of units. NaN counts as 0.
    """
    from scipy import ndimage as ndi

    Sv = np.nan_to_num(S, nan=0.0)
    t1, t2 = np.nanpercentile(S, t1_pct), np.nanpercentile(S, t2_pct)
    # >= not >: a cell equal to its neighbourhood reference is not BELOW it, and on a plateau
    # -- a flat-topped ridge, or any field with repeated values -- strict inequality silently
    # excludes every cell of it at once. [measured] on a continuous field the difference is a
    # rounding error (34.42% of cells marked against 34.50%), because ties concentrate on flat
    # ground where t2 rejects them anyway; on a discrete field like returns-per-cell it is not.
    mask = Sv >= ndi.percentile_filter(Sv, nbr_pct, size=size)
    return (Sv > t1) | ((Sv > t2) & mask)


def boundary(S, ndsm=None, *, hi=0.5, lo=0.15, min_height=2.0, close=True,
             seed_max=0.05):
    """The full boundary map: hysteresis linking, then watershed lines to close the contours.

    Stage 2 of ``docs/boundaries.md``. Hysteresis keeps weak rim cells that are *connected* to
    strong ones, but cannot close a ring whose flank dips below `lo`. The watershed can: flood
    `S` from the cores (tall regions the hysteresis boundary already isolates) plus one
    background marker for the ground, and **the lines where floods meet are closed contours by
    construction** — two adjacent cores get separated along the weak ridge between them no
    matter how weak it is, because the flood uses the ridge's *relative* height, not a cut.

    The crown-flood caution from the earlier failed experiment does not apply: the basins are
    discarded and only the lines kept, so nothing grows into anything. A crown ends up inside
    whichever basin reaches it, and then the instance step removes it exactly as before.

    `close=False` reverts to hysteresis alone; `ndsm=None` forces that too (no markers without
    a height). NaN in `S` counts as 0 — no evidence is not a ridge.
    """
    from scipy import ndimage as ndi

    edges = hysteresis(S, hi, lo)
    if not close or ndsm is None:
        return edges
    from skimage.segmentation import watershed

    h = np.nan_to_num(ndsm, nan=0.0)
    tall = h > min_height
    core = ndi.binary_opening(tall & ~edges, np.ones((3, 3)))
    Sv = np.nan_to_num(S, nan=0.0)
    # Markers must be FINER than the cores, or a merged pair of buildings is one marker and can
    # never receive an internal line. Seeds are confident interior plateaus — S below seed_max,
    # a third of `lo` — so two attached buildings with a weak party-wall ridge get two seeds and
    # the line falls on the ridge. Specks under 10 cells are dropped or every vent splits a roof.
    seed = ndi.binary_opening(core & (Sv < seed_max), np.ones((3, 3)))
    markers, n = ndi.label(seed)
    sizes = np.bincount(markers.ravel())
    markers[np.isin(markers, np.flatnonzero(sizes < 10))] = 0
    markers[~tall] = markers.max() + 1        # the ground floods as one basin
    ws = watershed(Sv, markers, watershed_line=True)
    return edges | ((ws == 0) & tall)


def instances(pts, bbox, pix=1.0, *, ground=None, edge=0.5, edge_low=0.15, min_height=2.0,
              min_area=30.0, by=None, close=True, seed_max=0.05, veg_ratio=0.8):
    """Label separate above-ground structures. Returns ``(labels, edges, ndsm)``.

    Cells whose views all agree there is structure are treated as **boundaries**, and what is
    left above `min_height` is labelled by connected component. Boundaries are not assigned to
    an instance — this finds cores, not complete footprints.

    Args:
        ground: boolean mask over `pts` marking bare earth, from
            :func:`ducklidar.ground.ground_filter`. Without it the ground is a 51-pixel rolling
            minimum of the DSM, which is wrong wherever a structure is wider than that window.
        edge: boundary seed threshold — cells above it start a boundary.
        edge_low: hysteresis extension threshold — connected cells above it join the boundary
            (:func:`hysteresis`). ``None`` reverts to the plain single cut.
        close: also add watershed lines (:func:`boundary`), which close every core's contour
            and separate adjacent cores along their shared weak ridge.
        veg_ratio: cells whose :func:`echo_ratio` exceeds this are dropped as vegetation
            **before** the boundary step, so crowns are removed as *material* rather than being
            cut into pieces by a boundary rule. ``None`` disables it. Needs the return fields.

            **[measured]** the cut is necessary — without it 31% of instances are vegetation and
            building share falls to 57% — but its threshold trades vegetation against roof:

            ==========  ========  ===========  ===========
            veg_ratio   building  veg dropped  roof lost
            ==========  ========  ===========  ===========
            none        57%       39%          0.0%
            0.9         70%       78%          0.4%
            **0.8**     **75%**   **91%**      **1.3%**
            0.7         76%       94%          2.7%
            0.5         77%       97%          6.5%
            ==========  ========  ===========  ===========

            0.8 is the default because 0.5 bought its last two points of building share with
            five times the roof damage — and that damage is concentrated on glazing, which is a
            real part of a building rather than clutter.
        min_area: square metres; smaller components are dropped as clutter.
        by: what counts as one view — a field name, or an array of group codes per return.
            See :func:`edge_strength`.

    **[measured]** 250 m Granville window, four flight lines, `ground=` from
    :func:`~ducklidar.ground.ground_filter`: **126–128** instances at ≥30 m² (CSF is
    multithreaded, so the count moves by one or two between runs), 68% of their area carrying
    the survey's building class and **81% of all vegetation cells dropped** — with no
    classification field used anywhere in producing them. Without `ground=` the same call gives
    136 instances at only 54% building, because the rolling minimum reads the whole island as
    15–20 m above "ground" and merges half the window into one component.

    What it still gets wrong: a city block of attached buildings is *one* instance, which is a
    definition question rather than an error — but so are two buildings that merely touch, which
    is not. Splitting those needs the roof planes, not just the boundary.
    """
    from scipy import ndimage as ndi

    strength = edge_strength(pts, bbox, pix, by=by)
    dsm = rasterize(pts, bbox, pix, how="max")

    if ground is None:
        # ponytail: rolling minimum stands in for a ground filter. Pass ground= for the real
        # thing — this one rides up onto anything wider than its window.
        span = int(round(51 / pix)) | 1
        floor_z = ndi.minimum_filter(np.nan_to_num(dsm, nan=np.nanmin(dsm)), size=span)
    else:
        floor_z = rasterize(pts, bbox, pix, mask=np.asarray(ground, dtype=bool), how="min")
        floor_z = _fill_nearest(floor_z)
    ndsm = dsm - floor_z

    if edge_low is not None:
        edges = boundary(strength, ndsm, hi=edge, lo=edge_low, min_height=min_height,
                         close=close, seed_max=seed_max)
    else:
        edges = np.nan_to_num(strength, nan=0.0) > edge

    tall = np.nan_to_num(ndsm, nan=0.0) > min_height
    if veg_ratio is not None and {"return_number", "number_of_returns"} <= set(pts):
        # material before geometry: a crown removed here never reaches the boundary rule, so it
        # cannot be chopped into spurious instances by one
        tall &= ~(np.nan_to_num(echo_ratio(pts, bbox, pix), nan=0.0) > veg_ratio)
    core = ndi.binary_opening(tall & ~edges, np.ones((3, 3)))
    lab, _ = ndi.label(core)
    big = np.flatnonzero(np.bincount(lab.ravel()) * pix * pix >= min_area)
    lab = np.where(np.isin(lab, big[big > 0]), lab, 0)
    # renumber so labels are 1..n with no gaps
    ids = np.unique(lab)
    return np.searchsorted(ids, lab), edges, ndsm


def _fill_nearest(grid):
    """Nearest-neighbour fill of NaN holes — a DTM is half empty and subtraction needs a value."""
    from scipy import ndimage as ndi

    hole = ~np.isfinite(grid)
    if not hole.any():
        return grid
    idx = ndi.distance_transform_edt(hole, return_distances=False, return_indices=True)
    return grid[tuple(idx)]
