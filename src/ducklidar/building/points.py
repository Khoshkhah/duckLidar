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


def building_points(footprint, store, *, eave=EAVE, labels=None, keep_classes=(3, 9),
                    min_h=MIN_H, fields=("red", "green", "blue")):
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
        z_ground = float(np.percentile(z[inside], GROUND_PCT)) if inside.any() else 0.0
        label = np.full(len(x), -1, np.int16)
        keep = inside & (z > z_ground + min_h)

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
