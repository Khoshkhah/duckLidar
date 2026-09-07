"""One building's own returns: a footprint + a point store -> points, colour, ring, ground.

    from ducklidar.building import points
    d = points.building_points(ring, "data/lidar/*.parquet")
    d["x"], d["y"], d["z"], d["rgb"], d["ring"], d["z_ground"]

The footprint is grown by the eave (3 m) because a roof overhangs its walls, and the
returns inside that are the building's — everything from the roof down to the ground it
stands on. Which of those are *wall* and which are *ground* is the only real question, and
there are two ways to answer it:

  * **with labels** — a classification per point, when the survey or a labelling pass has
    one. Pass `labels=(pid, label)` (two aligned arrays) and the classes to keep. This is
    exact, and it is what shadowCity2's stage 5 gives.
  * **without** — the local ground is the 2nd percentile of z inside the footprint, and the
    building is everything above it by `min_h`. Costs nothing, needs no label table, and is
    what frees this function from any one app.

The ground height is returned either way: it is what the facade code measures wall heights
from, and a wrong one tilts every wall.
"""
import numpy as np

EAVE = 3.0        # m: a roof overhangs its walls, so the footprint is grown by this to catch the eave returns
MIN_H = 1.0       # m: without labels, a return this far above local ground is the building, not its forecourt
GROUND_PCT = 2.0  # the ground under a building is the low tail of z inside the footprint, not its minimum (outliers)
MIN_CLASSIFIED = 50  # class-6 returns inside the footprint before the survey's own labels are trusted
COVER = 0.25      # of the footprint's 1 m cells: a layer covering less is not this building's roof
COVER_TAIL = 0.02 # ...but what stands on the covering layer stays while it covers this much...
TAIL_M = 6.0      # ...and ENDS within this: a plant room does, a bridge pier through the roof does not
BAND_BIN = 0.5    # m: the z histogram bin the roof layer is found in
BAND_FRAC = 0.05  # a bin holding less than this share of the roof's own peak is no longer the building


def footprint_ring(source, bid=None, osm_id=None):
    """The building's outline as an (n, 2) ring, from a GeoJSON-ish feature file.

    `source` is a path to a JSON file with a "features" list of dicts carrying "coords"
    (and optionally "theme"/"bid"), which is what a map extract writes; `bid` matches a
    feature's id by prefix, `osm_id` matches it exactly. Pass a ring or a shapely polygon
    straight through instead when you already have one — this helper is a convenience, not
    a required path.
    """
    import json
    if not isinstance(source, (str, bytes)) and not hasattr(source, "open"):
        return as_ring(source)
    feats = json.load(open(source))
    feats = feats["features"] if isinstance(feats, dict) else feats
    def match(f):
        if bid is not None: return str(f.get("bid", "")).startswith(str(bid))
        if osm_id is not None: return str(f.get("id", f.get("osm_id", ""))) == str(osm_id)
        return f.get("theme") == "building"
    f = next(x for x in feats if match(x))
    c = f["coords"]
    return np.asarray(c[0] if isinstance(c[0][0], (list, tuple)) else c, float)


def as_ring(fp):
    """A footprint as an (n, 2) ring, from a ring, a shapely Polygon, or a coordinate list."""
    if hasattr(fp, "exterior"):
        return np.asarray(fp.exterior.coords)[:-1]
    a = np.asarray(fp, float)
    if len(a) > 1 and np.allclose(a[0], a[-1]): a = a[:-1]
    return a


def roof_band(z, z_ground, min_h=MIN_H, bin_m=BAND_BIN, frac=BAND_FRAC):
    """(lo, hi) in z: the building's own layer of SINGLE returns above the ground.

    `z` is the single (number_of_returns == 1) returns inside the footprint. A roof stops a
    pulse dead, so it is a dense band in the histogram of these; a crown or a bridge deck
    above it is porous (multi-return) or separated by nearly empty bins. The band is the
    lowest bin at `frac` of the strongest and every contiguous bin above it still at `frac`.

    [measured] Granville Island, 2026-09-06, 175 footprints with an Overture height, model
    top = P98 of the kept returns: the raw P98 of everything inside the collar was a median
    1.56x the Overture height and 90 of 175 were over 1.5x (the Granville Bridge deck gave one
    building 115 m; a crown gave a 5 m shop 33 m). Splitting on single returns alone: the
    tree-covered shop's singles are 2719 / 2856 per 2 m in the roof and under 150 above it,
    against 200-500 per bin of multi-returns all the way up.
    """
    lo = z_ground + min_h
    zz = z[z >= lo]
    if len(zz) < 20: return lo, np.inf
    edges = np.arange(lo, zz.max() + bin_m, bin_m); cnt, _ = np.histogram(zz, edges)
    strong = cnt >= frac * cnt.max()
    top = first = int(np.argmax(strong))
    while top + 1 < len(cnt) and strong[top + 1]: top += 1
    return lo, float(edges[top + 1])


def coverage(x, y, z, ncell, z_ground, edges):
    """Per height bin, the share of the footprint's `ncell` 1 m cells holding one of these returns."""
    zz = z - z_ground
    cx, cy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    cov = np.zeros(len(edges) - 1)
    for i in range(len(cov)):
        s = (zz >= edges[i]) & (zz < edges[i + 1])
        cov[i] = len(np.unique(cx[s] * 1_000_003 + cy[s])) / ncell
    return cov


def roof_top(x, y, z, ncell, z_ground, x1=None, y1=None, z1=None, cover=COVER, tail=COVER_TAIL, tail_m=TAIL_M, bin_m=2.0):
    """The z above which the survey's building returns no longer cover this footprint.

    `x, y, z` are the returns the survey classed as BUILDING; `x1, y1, z1` those it left
    UNCLASSIFIED (optional). Per `bin_m` of height, the share of the footprint's 1 m cells that
    hold a return. A roof covers the footprint; a bridge pier passing through it, or a tower's
    wall column, covers a few percent. The top is the highest bin of building returns at
    `cover` (or the highest holding any, when none covers that much), plus what stands ON it
    — the bins above, building or unclassified, still at `tail` — provided that ends within
    `tail_m`: a plant room or a parapet does, a pier does not.

    [measured] Granville Island 2022, coverage per 2 m: a shop's class-6 roof 64 %, the
    Granville Bridge pier through it 5-8 % for 100 m, the deck 22 %; a podium roof 33 %, the
    tower shaft above 4-7 %, the tower's top 19 %; a tower filling its footprint 83 %. So a
    quarter separates the deck from every roof. gers_003362c0's plant room is class 1, 9 % of
    the cells and 4 m above its class-6 tower top; the bridge deck over gers_0b9fca8a is class
    1 too, 60 % of the cells, 20 m of empty air above a 12 m class-6 roof — which is why the
    unclassified returns may only extend a roof, never be one.
    """
    zz = z - z_ground
    if len(zz) < 2 or ncell == 0: return np.inf
    zmax = max(zz.max(), (z1 - z_ground).max() if z1 is not None and len(z1) else 0.0)
    edges = np.arange(0.0, zmax + bin_m, bin_m)
    cov = coverage(x, y, z, ncell, z_ground, edges)
    hit = np.nonzero(cov >= cover)[0]
    top = int(hit[-1] if len(hit) else np.nonzero(cov > 0)[0][-1])
    if x1 is not None and len(z1):
        cov = np.maximum(cov, coverage(x1, y1, z1, ncell, z_ground, edges))
    end = top
    while end + 1 < len(cov) and cov[end + 1] >= tail: end += 1
    if (end - top) * bin_m <= tail_m: top = end                # a tail that ends is the roof's own
    return z_ground + float(edges[top + 1])


def building_points(footprint, store, *, eave=EAVE, labels=None, keep_classes=(3, 9),
                    min_h=MIN_H, fields=("red", "green", "blue", "number_of_returns")):
    """The returns that are this building, inside `footprint` grown by `eave`.

    `store` is anything `dl.read` takes — a parquet path, a LAS/LAZ file, or a list of tiles.
    `labels` is an optional `(pid, label)` pair of aligned arrays (pid sorted); when given,
    only `keep_classes` survive and the ground is the median z of class 0. Without it the
    ground is the `GROUND_PCT`th percentile inside the footprint and the building is
    everything `min_h` above it.

    Returns dict(x, y, z, rgb (n,3) uint8, label, ring (m,2), z_ground) — the same keys the
    pilot's npz carries, so a cached npz and a fresh read are interchangeable.
    """
    import shapely
    from shapely.geometry import Polygon

    from ..read import box as _box, read as _read_fn

    ring = as_ring(footprint)
    poly = Polygon(ring)
    grown = poly.buffer(eave)
    x0, y0, x1, y1 = grown.bounds
    want = ("pid", *fields) if "pid" not in fields else tuple(fields)
    pts = _read_fn(store if isinstance(store, (list, tuple)) else str(store),
                     (x0 - 2, y0 - 2, x1 + 2, y1 + 2), fields=want)
    x, y, z = (np.asarray(pts[k]) for k in ("x", "y", "z"))
    rgb = (np.column_stack([np.asarray(pts[k]) >> 8 for k in ("red", "green", "blue")]).astype(np.uint8)
           if all(k in pts for k in ("red", "green", "blue")) else np.full((len(x), 3), 150, np.uint8))
    inside = shapely.contains_xy(grown, x, y)

    if labels is not None:
        label = _lookup(np.asarray(pts["pid"]), *labels)
        keep = inside & np.isin(label, list(keep_classes))
        g = inside & (label == 0)
        z_ground = float(np.median(z[g])) if g.sum() > 50 else float(np.percentile(z[inside], GROUND_PCT))
    else:
        from ..fields import BUILDING, GROUND
        UNCLASSIFIED = 1
        label = np.full(len(x), -1, np.int16)
        core = shapely.contains_xy(poly, x, y)
        cls = np.asarray(pts["classification"]) if "classification" in pts else np.zeros(len(x), np.uint8)
        # UNCLASSIFIED counts as building inside the footprint: the survey left gers_003362c0's rooftop
        # plant room as class 1 and its roof came out with a hole. What is not building (a crane cable)
        # is cut where the coverage stops.
        bld, g = np.isin(cls, (BUILDING, UNCLASSIFIED)), (cls == GROUND) & inside
        z_ground = (float(np.median(z[g])) if g.sum() >= MIN_CLASSIFIED
                    else float(np.percentile(z[inside], GROUND_PCT)) if inside.any() else 0.0)
        if (cls == GROUND).sum() >= MIN_CLASSIFIED and (core & (cls == BUILDING)).sum() < MIN_CLASSIFIED:
            # a classified survey with no building return inside the footprint: a parking lot, a
            # cleared site, a shed under a crown — NOT a building. 14 of Granville Island's 248
            # (gers_06c1f1c6, an OSM parking way, got a 4.4 m box of cars and trees, 2026-09-07).
            keep = np.zeros(len(x), bool)
        elif (core & (cls == BUILDING)).sum() >= MIN_CLASSIFIED:  # the survey says which returns are building
            ncell = len(np.unique(np.floor(x[core]).astype(np.int64) * 1_000_003 + np.floor(y[core]).astype(np.int64)))
            m6, m1 = core & (cls == BUILDING), core & (cls == UNCLASSIFIED)
            hi = roof_top(x[m6], y[m6], z[m6], ncell, z_ground, x[m1], y[m1], z[m1])
            keep = inside & bld & (z > z_ground + min_h) & (z <= hi)   # class 6 at ground level (a wall's foot, a fence) is not the roof
        else:                                                    # it does not: the roof is a layer of single returns
            single = np.asarray(pts["number_of_returns"]) == 1 if "number_of_returns" in pts else np.ones(len(x), bool)
            lo, hi = roof_band(z[core & single], z_ground, min_h)
            keep = inside & (z > lo) & (z < hi)

    return dict(x=x[keep], y=y[keep], z=z[keep], rgb=rgb[keep], label=label[keep],
                ring=ring, z_ground=z_ground)


def _lookup(pid, lab_pid, lab_val, offset=0):
    """label per point by sorted-lookup of `pid + offset` in `lab_pid`; -1 where absent."""
    lab_pid = np.asarray(lab_pid); lab_val = np.asarray(lab_val)
    i = np.minimum(np.searchsorted(lab_pid, pid + offset), len(lab_pid) - 1)
    return np.where(lab_pid[i] == pid + offset, lab_val[i], -1)


def load_npz(path):
    """A cached `building_points` result written with `np.savez` — the pilot's points file."""
    d = np.load(path)
    out = {k: d[k] for k in d.files}
    out.setdefault("z_ground", None)
    if out.get("z_ground") is not None: out["z_ground"] = float(out["z_ground"])
    return out


def demo():
    """Self-check: a square building on flat ground, with and without labels."""
    rng = np.random.default_rng(0)
    ring = np.array([[0.0, 0], [10, 0], [10, 10], [0, 10]])
    # 400 wall/roof returns 4-8 m up inside, 400 ground returns at 1 m spread over a wider area
    bx, by = rng.uniform(0.5, 9.5, 400), rng.uniform(0.5, 9.5, 400)
    gx, gy = rng.uniform(-5, 15, 400), rng.uniform(-5, 15, 400)
    x = np.r_[bx, gx]; y = np.r_[by, gy]
    z = np.r_[rng.uniform(4, 8, 400), np.full(400, 1.0)]
    inside = (x > 0) & (x < 10) & (y > 0) & (y < 10)
    z_ground = float(np.percentile(z[inside], GROUND_PCT))
    assert abs(z_ground - 1.0) < 0.5, z_ground
    keep = inside & (z > z_ground + MIN_H)
    assert keep.sum() == 400, keep.sum()          # every building return, no ground
    lab = _lookup(np.arange(8), np.array([1, 3, 5]), np.array([9, 3, 0]))
    assert lab.tolist() == [-1, 9, -1, 3, -1, 0, -1, -1], lab.tolist()
    assert len(as_ring(np.r_[ring, ring[:1]])) == 4
    print("points.demo ok")


if __name__ == "__main__":
    demo()
