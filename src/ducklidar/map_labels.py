"""One labeling method per type the map carries a footprint for.

Kaveh's rule (2026-08-05): *"every object with footprint in overture must have
a special method for labeling, like bridge, building, dock"*. The map is an
external fact, not decoration — where Overture draws an outline, that outline
is evidence about what the points inside it ARE. Each such type gets its own
named method to say so, exactly as each type already has its own instance
detector (stage 6.3b/c/d) and its own 3-D model (:mod:`ducklidar.objects`).

Before this, the map reached the labels in two hard-coded places — the bridge
corridor and the building footprint veto — and every other mapped type was
guessed from geometry alone. That is how a DOCK came to have no labeling rule
at all: its points are claimed by ``floating`` ("solid matter standing above
the water plane"), which says nothing about pontoons, and dock-versus-boat was
then decided downstream by a height threshold. Overture maps 99 pier ways over
this tile; the survey never had to guess.

Every method has one contract::

    <type>_claim(feats, x, y, z, *, ground_z, wlvl, ...) -> bool mask

It **claims**; it does not decide. :func:`map_claims` applies the methods in a
fixed order and each may only overwrite the labels named in ``TAKE_FROM`` —
stated per type, never global — so a map error cannot erase a confident
geometric label. Each also carries a PHYSICAL guard, because a footprint says
where a thing is in plan and nothing about height: the map alone would claim
the water under a bridge, the crown over a footprint and the hull moored to a
pier.
"""
import numpy as np

# Every method may take only these labels. A type that could take anything
# would make the map outrank the survey everywhere, which is not the deal:
# the map is authoritative on WHERE, the returns on WHAT and HOW HIGH.
TAKE_FROM = {
    "water":    ("ground",),
    "dock":     ("water", "ground", "small"),
    "bridge":   ("ground", "small", "floating", "veg_low", "veg_high"),
    "building": ("small", "floating"),
    "ground":   ("small", "veg_low"),
}
# Order matters and is part of the rule: the water surface is settled first so
# the dock can be cut out of it, the deck before the things standing on it, and
# ground last so it can only take what nothing else claimed.
ORDER = ("water", "dock", "bridge", "building", "ground")

PIER_HALF = 1.25     # m — a pier is a WALKWAY. The same half-width stage 6.3b
                     # already gives a footway, and the same thing Kaveh said:
                     # "dock is not too wide, similar to walking path". 2.5 m
                     # (a 5 m ribbon) was measurably too generous. [2026-08-05]
DOCK_BAND = (-0.3, 1.5)   # m about the water plane: a pontoon rides on it
WATER_BAND = 0.5     # m about the water plane
DECK_BAND = 2.0      # m below the cell top: the deck surface, at any height
WALL_MIN = 2.0       # m above the DEM before a point inside a footprint is wall
GROUND_BAND = 0.4    # m about the DEM: a surface, not a thing standing on it


def _polys(feats, want_class=None, want_subtype=None, geom="polygon",
           want_theme=None):
    """Overture features -> shapely geometries, lines buffered to walkways."""
    import shapely

    out = []
    for f in feats:
        if want_theme is not None and f.get("theme") not in want_theme:
            continue
        if want_class is not None and f.get("class") not in want_class:
            continue
        if want_subtype is not None and f.get("subtype") not in want_subtype:
            continue
        xy = np.asarray(f["coords"], float)
        if f["geom"] == "polygon" and len(xy) >= 4:
            g = shapely.Polygon(xy)
            g = g if g.is_valid else g.buffer(0)
        elif f["geom"] == "line" and len(xy) >= 2 and geom in ("line", "any"):
            g = shapely.LineString(xy).buffer(PIER_HALF, cap_style="flat")
        else:
            continue
        if g.is_empty:
            continue
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda q: q.area)
        out.append(g)
    return out


def _inside(geoms, x, y, grow=0.0):
    """Points inside any of the geometries (one prepared union, one pass)."""
    import shapely

    if not geoms:
        return np.zeros(len(x), bool)
    u = shapely.union_all(geoms)
    if grow:
        u = u.buffer(grow)
    return shapely.contains_xy(u, x, y)


# ---- one method per mapped type -----------------------------------------

def water_claim(feats, x, y, z, *, ground_z=None, wlvl=0.0):
    """WATER: inside a mapped water polygon and AT the water plane.

    The band is what stops the map from swallowing the marina: a hull, a
    pontoon and a bridge all stand over water in plan, and only the returns
    that came back at the plane itself are the surface.
    """
    return (_inside(_polys(feats, want_subtype={"water", "ocean", "lake",
                                                "pond", "river", "stream"}),
                    x, y)
            & (np.abs(z - wlvl) < WATER_BAND))


def dock_claim(feats, x, y, z, *, ground_z=None, wlvl=0.0):
    """DOCK / PONTOON: inside a mapped pier way, riding on the water plane.

    The type that had no labeling rule at all. `floating` claims any solid
    matter above the water and cannot tell a walkway from a hull, so 99 mapped
    pier ways went unused and dock-versus-boat fell to a height threshold
    downstream. A pier way is narrow — Kaveh: "dock is not too wide, similar to
    a walking path" — so a centreline buffers to a walkway, and the height band
    is a pontoon's: it rides ON the water, it does not stand over it.
    """
    return (_inside(_polys(feats, want_class={"pier"}, geom="line"), x, y)
            & (z > wlvl + DOCK_BAND[0]) & (z < wlvl + DOCK_BAND[1]))


def bridge_claim(feats, x, y, z, *, ground_z=None, wlvl=0.0, cell_top=None):
    """BRIDGE DECK: inside a mapped bridge polygon, at the top of its column.

    Where the map draws a bridge, the map IS the definition — the at-grade ramp
    is as much the bridge as the 40 m span, and no height rule can know that.
    What the map cannot say is which returns are the DECK: everything under the
    span is inside the polygon too. ``cell_top`` (the per-point height of the
    tallest return in its cell) settles it — the deck is what the sky sees.
    """
    ins = _inside(_polys(feats, want_subtype={"bridge"},
                         want_theme={"infrastructure"}), x, y)
    if cell_top is None:
        return ins & (ground_z is not None) & (z > np.asarray(ground_z) + 2.0)
    return ins & (z > np.asarray(cell_top, float) - DECK_BAND)


def building_claim(feats, x, y, z, *, ground_z=None, wlvl=0.0):
    """BUILDING: inside a mapped footprint and standing clear of the ground.

    Deliberately narrow, because the footprint veto in
    :func:`~ducklidar.labeling.map_vetoes` already owns the veg_high case with
    its measured 3-cell erosion. This one only recovers the structure that the
    object rules took: a wharf shed read as `small`, a boathouse read as
    `floating`.
    """
    if ground_z is None:
        return np.zeros(len(x), bool)
    return (_inside(_polys(feats, want_theme={"building"}), x, y)
            & (z > np.asarray(ground_z, float) + WALL_MIN))


def ground_claim(feats, x, y, z, *, ground_z=None, wlvl=0.0):
    """GROUND: inside a mapped land surface and AT the DEM.

    Roads, footways, pitches, parks, parking. Runs last and takes least: only
    returns lying on the DEM itself, so the car parked on the road and the
    shrub beside it keep their own labels.
    """
    if ground_z is None:
        return np.zeros(len(x), bool)
    geoms = _polys(feats, want_class={"pitch", "playground", "park", "grass",
                                      "parking", "marina", "sand", "wood",
                                      "forest", "recreation"})
    geoms += _polys(feats, want_class={"footway", "path", "steps", "cycleway",
                                       "pedestrian", "service", "residential",
                                       "living_street", "unclassified",
                                       "tertiary", "secondary", "primary",
                                       "trunk", "motorway"}, geom="line")
    return (_inside(geoms, x, y)
            & (np.abs(z - np.asarray(ground_z, float)) < GROUND_BAND))


def building_veto(feats, x, y, z, *, ground_z=None, wlvl=0.0, eave=None):
    """A BUILDING MUST HAVE A FOOTPRINT — the mask of the ones that do not.

    Kaveh, 2026-08-05: *"every building must have footprint. we don't assign
    building label to a point without footprint"*, because *"at the moment you
    assign building label to many points and they are not really points of a
    building"*. He is right, and the returns say so: 817,628 building-labelled
    points lie outside every Overture footprint, and **71.6% of them stand
    under 6 m above the DEM** (36.1% under 3 m) while only 1.6% reach above
    25 m — canopies, wharf sheds, walls and yard clutter, a median 5.7 m and a
    p90 of 37 m beyond the nearest outline. They are solid, but they are not
    buildings.

    ``eave`` is the same 3 m allowance stage 6.3c already uses: a footprint is
    the outline at GROUND level and roofs overhang it, so a strict test throws
    away real roof (77.8% inside at 0 m against 90.9% at 3 m). Vetoed points
    are NOT deleted — they still stand and still shade — they are handed to the
    label that means "a solid thing that is not terrain and not a mapped
    structure": `floating` over the water, `small` on land.
    """
    eave = EAVE if eave is None else eave
    return ~_inside(_polys(feats, want_theme={"building"}), x, y, grow=eave)


EAVE = 3.0           # m — a footprint is the GROUND outline; roofs overhang it
BANK = 8.0           # m — vegetation this far out from any bank is not vegetation

def vegetation_veto(feats, x, y, z, *, ground_z=None, wlvl=0.0, bank=None):
    """VEGETATION CANNOT STAND ON OPEN WATER — the mask of the ones that do.

    Kaveh, 2026-08-05: *"some points got vegetation label on water! most of
    them are sails!"*. Sailcloth and rigging return the laser exactly the way
    foliage does — multiple, scattered, penetrable — so 47,786 vegetation
    returns fell inside an Overture water polygon, standing 3.9 to 16.3 m above
    the plane. A tree is rooted in the ground; nothing vegetal grows out of a
    marina.

    The trap is that shore crowns really do overhang the water, so the water
    polygon is ERODED by ``bank`` before the test. Measured on this tile:
    47,786 returns inside at 0 m, 10,091 at 4 m, 6,490 at 8 m, 5,324 at 15 m —
    the bank-overhang population is gone by 8 m and the curve flattens, so
    that is where the open-water population begins. Vetoed points become
    `floating`, which is what a sail on a boat is, and the floating machinery
    then sorts them into vessel or unknown.
    """
    import shapely

    bank = BANK if bank is None else bank
    g = _polys(feats, want_subtype={"water", "ocean", "lake", "pond",
                                    "river", "stream"})
    if not g:
        return np.zeros(len(x), bool)
    u = shapely.union_all(g).buffer(-bank)
    if u.is_empty:
        return np.zeros(len(x), bool)
    return shapely.contains_xy(u, x, y)


CLAIM = {"water": water_claim, "dock": dock_claim, "bridge": bridge_claim,
         "building": building_claim, "ground": ground_claim}

# What each claim writes into the ten-label decision. `dock` has no label of
# its own — a pontoon IS floating matter — so it writes `floating` and reports
# itself separately, which is what gives stage 6 a dock it did not have to
# guess by height.
AS_LABEL = {"water": "water", "dock": "floating", "bridge": "bridge",
            "building": "building", "ground": "ground"}


BRIDGE_CORR = 22.0   # m — half-width of a mapped bridge centreline's corridor


def ground_without_bridges(grid, bbox, feats, pix=1.0, wlvl=None,
                           gx=None, gy=None, gz=None):
    """The ground grid with the BRIDGES TAKEN OUT OF IT.

    Stage 5's classifier calls a deck bare earth wherever the deck is the only
    thing under the sky, so the grid IS the bridge there — measured, a median
    of 29.69 m inside a bridge footprint. Anything that asks "how high is this
    above the ground?" near a bridge gets a wrong answer from it: a tree crown
    standing on Granville's deck at 40 m reads as a 10 m tree on 30 m ground
    and sails through every height guard (Kaveh, 2026-08-06).

    Repaired in three steps, best evidence first:

    1. the GROUND RETURNS themselves where the laser saw under the deck —
       97,744 of them on this tile, and against them the raw grid is a median
       +27.34 m out while this is +0.00
    2. the WATERLINE where the bridge crosses water, so the fill does not draw
       a ramp between the two banks
    3. a relaxed fill from the edges for whatever is left

    Both polygons AND buffered centrelines are masked: Granville is mapped
    mostly as lines, and masking only its four polygons left most of its length
    untouched.
    """
    from scipy import ndimage as ndi
    import shapely

    polys = [g for f, g in [(f, _poly_of(f)) for f in feats]
             if g is not None and (f.get("class") in ("bridge", "cantilever")
                                   or f.get("subtype") == "bridge")]
    if not polys:
        return np.asarray(grid, float)
    u = shapely.union_all([g if g.geom_type != "LineString"
                           else g.buffer(BRIDGE_CORR) for g in polys])
    my, mx = np.mgrid[0:grid.shape[0], 0:grid.shape[1]]
    bad = shapely.contains_xy(u, bbox[0] + (mx + 0.5) * pix,
                              bbox[3] - (my + 0.5) * pix)
    out = np.asarray(grid, float).copy()
    if not bad.any():
        return out
    if gx is not None and len(gx):
        gxi = np.clip(((np.asarray(gx) - bbox[0]) / pix).astype(int),
                      0, grid.shape[1] - 1)
        gyi = np.clip(((bbox[3] - np.asarray(gy)) / pix).astype(int),
                      0, grid.shape[0] - 1)
        flat = gyi.astype(np.int64) * grid.shape[1] + gxi
        o = np.argsort(flat, kind="stable")
        fs, zs = flat[o], np.asarray(gz, float)[o]
        e = np.flatnonzero(np.r_[True, fs[1:] != fs[:-1]])
        ends = np.r_[e[1:], len(fs)]
        med = np.array([np.median(zs[a_:b_]) for a_, b_ in zip(e, ends)])
        seen = np.zeros(grid.size, bool)
        vals = np.zeros(grid.size)
        seen[fs[e]] = True
        vals[fs[e]] = med
        seen = seen.reshape(grid.shape)
        vals = vals.reshape(grid.shape)
        take = bad & seen
        out[take] = vals[take]
        bad = bad & ~seen
        if not bad.any():
            return out
    if wlvl is not None:
        wet = [_poly_of(f) for f in feats
               if f.get("theme") == "water" and f["geom"] == "polygon"]
        wet = [w for w in wet if w is not None]
        if wet:
            iswet = shapely.contains_xy(shapely.union_all(wet),
                                        bbox[0] + (mx + 0.5) * pix,
                                        bbox[3] - (my + 0.5) * pix)
            out[bad & iswet] = float(wlvl)
            bad = bad & ~iswet
            if not bad.any():
                return out
    ii = ndi.distance_transform_edt(bad, return_distances=False,
                                    return_indices=True)
    out[bad] = np.asarray(grid, float)[tuple(ii)][bad]
    for _ in range(200):
        sm = ndi.uniform_filter(out, size=5)
        out[bad] = sm[bad]
    return out


def _poly_of(f):
    import shapely
    xy = np.asarray(f["coords"], float)
    if f["geom"] == "polygon" and len(xy) >= 4:
        g = shapely.Polygon(xy)
        return g if g.is_valid else g.buffer(0)
    if f["geom"] == "line" and len(xy) >= 2:
        return shapely.LineString(xy)
    return None


def map_claims(label, x, y, z, feats, decide, *, ground_z=None, wlvl=0.0,
               cell_top=None, only=None):
    """Run every per-type claim in :data:`ORDER`. Returns ``(label, mtype)``.

    ``label`` is rewritten IN PLACE, as :func:`~ducklidar.labeling.map_vetoes`
    does. ``mtype`` is a per-point object array naming the type the MAP claimed
    — empty string where it claimed nothing — which is the sub-label stage 6
    otherwise has to infer.
    """
    label = np.asarray(label)
    mtype = np.zeros(len(label), dtype="<U8")
    idx = {k: i for i, k in enumerate(decide)}
    report = {}
    for name in ORDER:
        if only is not None and name not in only:
            continue
        kw = {"ground_z": ground_z, "wlvl": wlvl}
        if name == "bridge":
            kw["cell_top"] = cell_top
        want = CLAIM[name](feats, x, y, z, **kw)
        allowed = np.isin(label, [idx[k] for k in TAKE_FROM[name] if k in idx])
        take = want & allowed
        label[take] = idx[AS_LABEL[name]]
        mtype[want] = name              # the map's opinion, even where it lost
        report[name] = (int(want.sum()), int(take.sum()))
    # ...and the rules that run the other way, as vetoes.
    if only is None or "vegetation_veto" in only:
        v = np.isin(label, [idx[k] for k in ("veg_high", "veg_low") if k in idx])
        sail = v & vegetation_veto(feats, x, y, z)
        label[sail] = idx["floating"]
        mtype[sail] = "sail"
        report["vegetation_veto"] = (int(sail.sum()), int(sail.sum()))
    # ...and the one rule that runs the other way: no footprint, no building.
    if only is None or "building_veto" in only:
        b = label == idx["building"]
        loose = b & building_veto(feats, x, y, z)
        wet = loose & (z < wlvl + 1.5)
        label[wet] = idx["floating"]
        label[loose & ~wet] = idx["small"]
        mtype[loose] = ""
        report["building_veto"] = (int(loose.sum()), int(loose.sum()))
    return label, mtype, report
