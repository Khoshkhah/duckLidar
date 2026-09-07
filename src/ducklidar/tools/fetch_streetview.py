"""TOOL — Street View around one footprint: probe (free), fetch (billable), sheet + map.

**Not part of a build.** `ducklidar.building3d` never calls this; it reads the folder this
writes. Run it once per building, yourself, with your own key:

    python -m ducklidar.tools.fetch_streetview --ring footprint.json --out out/photos
    python -m ducklidar.tools.fetch_streetview --points out/pilot_netloft_points.npz --out out/photos

Metadata probes on rings 12/30/50 m outside the footprint find every panorama within 20 m
of a probe; panoramas *inside* the footprint (indoor photospheres) are skipped; one
640x640, fov 90, pitch 8 view per panorama is fetched, aimed at the nearest footprint edge.
With `--footprints` (a duckOverture extract) a panorama whose line of sight to that edge
crosses ANOTHER footprint is not bought either: on Granville Island that was a third of
all photos (2149 of 6525, 2026-09-06), each showing the neighbour instead of the building.
With `--tile` (the point store) THE SURVEY decides what a camera can see: a camera with no
ground return under it stands on a bridge deck or a roof (ten of gers_0112d1b3's 28 photos
were shot from the Granville Bridge, 25 m above the shop, 2026-09-07), and a line of sight
with returns in it — a crown, a truck, a neighbour the map does not know — is blocked.
`survey_check` applies the same test to a manifest already bought.

The key comes from `--key`, else `$GOOGLE_STREETVIEW_KEY`, else a `GOOGLE_STREETVIEW_KEY=`
line in the `--env` file. Check Google's terms for your use — the pilot's use was
non-commercial.

Writes `<out>/streetview/NNN.jpg` + `manifest.json` (the list `building3d` reads), plus a
contact sheet and a coverage map to look at.
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

RINGS = (12.0, 30.0, 50.0)   # m outside the footprint: near enough for a facade, far enough for the whole wall
RADIUS = 20                  # m: a probe claims any panorama this close
LIMIT, MAX_DIST = 28, 70     # how many images to buy, and how far away is still useful


def api_key(key=None, env=None):
    """The Street View key, from the argument, the environment, or an env file."""
    if key: return key
    if os.environ.get("GOOGLE_STREETVIEW_KEY"): return os.environ["GOOGLE_STREETVIEW_KEY"]
    if env and Path(env).exists():
        for line in open(env):
            if line.startswith("GOOGLE_STREETVIEW_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("no Street View key: pass --key, set GOOGLE_STREETVIEW_KEY, or point --env at a file holding it")


def meta(lat, lng, key, radius=RADIUS):
    q = urllib.parse.urlencode(dict(location=f"{lat:.6f},{lng:.6f}", radius=radius, source="outdoor", key=key))
    return json.load(urllib.request.urlopen(
        f"https://maps.googleapis.com/maps/api/streetview/metadata?{q}", timeout=20))


def discover(poly, key, rings=RINGS, crs=26910):
    """Every outdoor panorama around the footprint. Metadata calls are free."""
    from pyproj import Transformer
    from shapely.geometry import Point
    to_ll = Transformer.from_crs(crs, 4326, always_xy=True)
    to_utm = Transformer.from_crs(4326, crs, always_xy=True)
    panos = {}
    for dist in rings:
        b = poly.buffer(dist).exterior; n = max(12, int(b.length / 12))
        for i in range(n):
            p = b.interpolate(i / n, normalized=True)
            lng, lat = to_ll.transform(p.x, p.y)
            d = meta(lat, lng, key)
            if d.get("status") != "OK" or d["pano_id"] in panos: continue
            loc = d["location"]; ex, ey = to_utm.transform(loc["lng"], loc["lat"])
            panos[d["pano_id"]] = dict(pano_id=d["pano_id"], date=d.get("date"), lat=loc["lat"],
                                       lng=loc["lng"], x=ex, y=ey,
                                       inside=poly.buffer(2).contains(Point(ex, ey)),
                                       dist_m=round(Point(ex, ey).distance(poly), 1))
    return panos


def neighbours(db, poly, margin=MAX_DIST, crs=26910):
    """Every other footprint within `margin` of `poly`, from a duckOverture extract."""
    import duckdb
    from shapely.geometry import Polygon
    x0, y0, x1, y1 = poly.buffer(margin).bounds
    con = duckdb.connect(str(db), read_only=True)
    try:
        con.execute("LOAD spatial")
        rows = con.execute(
            "select ST_AsGeoJSON(g) from (select ST_Transform(geometry, 'EPSG:4326', ?, always_xy := true) g"
            " from buildings.building) where ST_Intersects(g, ST_MakeEnvelope(?, ?, ?, ?))",
            [f"EPSG:{crs}", x0, y0, x1, y1]).fetchall()
    finally:
        con.close()
    out = [Polygon(json.loads(gj)["coordinates"][0]) for gj, in rows]
    return [q for q in out if not q.equals(poly) and q.intersection(poly).area < 0.5 * q.area]


def blocked(poly, pano, others, min_len=1.0):
    """Does another footprint stand between this panorama and the edge it would be aimed at?"""
    from shapely.geometry import LineString, Point
    cam = Point(pano["x"], pano["y"])
    near = poly.exterior.interpolate(poly.exterior.project(cam))
    los = LineString([cam, near]).difference(poly.buffer(0.5))
    return any(q.intersection(los).length > min_len for q in others)


def survey_check(poly, panos, store, z_ground, cam_h=2.5, near_m=0.6, min_hits=15):
    """The survey's verdict on each panorama: sets `p["survey"]` to "ok" or the reason it cannot see the wall.

    Ground: the lowest return within 1.5 m of the camera, of any class, is what it stands on.
    Line of sight: from `cam_h` above that to three points across the nearest wall, at 2 m and
    5 m up, sampled every 0.5 m from 2 m past the camera to 1.5 m short of the wall; `min_hits`
    building returns (or ten times that of crown) within `near_m` of a ray block it, and a photo
    is refused when all three rays are blocked.
    """
    import ducklidar as dl
    from scipy.spatial import cKDTree
    from shapely.geometry import Point
    for p in panos:
        if p.get("survey"): continue
        cam = np.array([p["x"], p["y"]])
        near = poly.exterior.interpolate(poly.exterior.project(Point(*cam))); wall = np.array([near.x, near.y])
        x0, y0 = np.minimum(cam, wall) - 2; x1, y1 = np.maximum(cam, wall) + 2
        pts = dl.read(str(store), (x0, y0, x1, y1), fields=())
        x, y, z, c = (np.asarray(pts[k]) for k in ("x", "y", "z", "classification"))
        at_cam = np.hypot(x - cam[0], y - cam[1]) < 1.5
        # the camera stands on the LOWEST return around it, whatever the survey called it: a
        # boardwalk (class 1), a path under a hedge (class 5), a street 4 m above a waterfront
        # building's ground, the water below a quay — none of them class 2. A surface 6 m or more
        # above the building's ground is the bridge deck (27-33 m here); a position covered 90 %
        # by a structure 8 m up is on it or under it (under it the photo has no sky).
        local = float(z[at_cam].min()) if at_cam.any() else z_ground
        if local - z_ground >= 6.0:
            p["survey"] = f"the camera stands {local - z_ground:.0f} m above the building's ground (a bridge deck)"; continue
        cell = lambda m: len(set(zip(np.floor(x[m] * 2).astype(int), np.floor(y[m] * 2).astype(int))))
        if cell(at_cam) >= 5 and cell(at_cam & (z - local >= 8.0)) >= 0.9 * cell(at_cam):
            p["survey"] = "a structure over the camera (on a bridge deck, or under it)"; continue
        cam_z = local + cam_h
        # THREE rays across the wall, at 2 m and 5 m up. A ray is blocked by another BUILDING's
        # returns (class 6, 15 within 0.6 m) or a dense crown (class 5, 150): what the photo can
        # never show. A car or a person is not a block — the masks paint those out. The photo is
        # refused only when every ray is blocked: it sees none of the wall.
        ring_l = poly.exterior; s0 = ring_l.project(Point(*cam))
        targets = [np.array(ring_l.interpolate(min(max(s0 + ds, 0.0), ring_l.length)).coords[0]) for ds in (-3.0, 0.0, 3.0)]
        obst6 = (c == 6) & (z > z_ground + 0.5); obst5 = (c == 5) & (z > z_ground + 0.5)
        t6 = cKDTree(np.column_stack([x[obst6], y[obst6], z[obst6]])) if obst6.any() else None
        t5 = cKDTree(np.column_stack([x[obst5], y[obst5], z[obst5]])) if obst5.any() else None
        blocked, worst = 0, 0
        for wall in targets:
            L = np.linalg.norm(wall - cam)
            if L < 4: continue
            u = (wall - cam) / L; t = np.arange(2.0, L - 1.5, 0.5)
            if not len(t): continue
            samples = np.vstack([np.column_stack([cam[0] + t * u[0], cam[1] + t * u[1], cam_z + (z_ground + h - cam_z) * t / L]) for h in (2.0, 5.0)])
            n6 = len(set(j for h in t6.query_ball_point(samples, near_m) for j in h)) if t6 else 0
            n5 = len(set(j for h in t5.query_ball_point(samples, near_m) for j in h)) if t5 else 0
            if n6 >= min_hits or n5 >= 10 * min_hits: blocked += 1; worst = max(worst, n6 + n5)
        if blocked == 3:
            p["survey"] = f"{worst} building or crown returns in every line of sight"; continue
        p["survey"] = "ok"
    return panos


def fetch(poly, panos, out, key, limit=LIMIT, max_dist=MAX_DIST, others=()):
    """One view per outside panorama, aimed at the nearest footprint edge. THIS is the billable part."""
    from shapely.geometry import Point
    sv = Path(out) / "streetview"; sv.mkdir(parents=True, exist_ok=True)
    outside = sorted((p for p in panos.values() if not p["inside"] and p["dist_m"] <= max_dist
                      and not blocked(poly, p, others)),
                     key=lambda p: p["dist_m"])
    for p in outside[:limit]:
        near = poly.exterior.interpolate(poly.exterior.project(Point(p["x"], p["y"])))
        heading = round(math.degrees(math.atan2(near.x - p["x"], near.y - p["y"])) % 360)
        q = urllib.parse.urlencode(dict(size="640x640", pano=p["pano_id"], heading=heading,
                                        pitch=8, fov=90, key=key))
        fn = sv / f"{p['pano_id'][:16]}_{heading:03d}.jpg"
        fn.write_bytes(urllib.request.urlopen(
            f"https://maps.googleapis.com/maps/api/streetview?{q}", timeout=30).read())
        p.update(heading=heading, pitch=8, fov=90, file=str(fn))
        time.sleep(0.1)
    json.dump(outside, open(sv / "manifest.json", "w"), indent=1)
    return [p for p in outside if "file" in p]


def sheet_and_map(poly, panos, imgs, out):
    """A contact sheet and a plan of where the cameras stand — for looking at, not for the build."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    out = Path(out)
    cols = 6; rows = math.ceil(len(imgs) / cols)
    sheet = Image.new("RGB", (cols * 324, max(rows, 1) * 274), "white")
    for i, p in enumerate(imgs):
        im = Image.open(p["file"]).convert("RGB"); im.thumbnail((320, 240))
        tile = Image.new("RGB", (320, 270), "white"); tile.paste(im, (0, 0))
        ImageDraw.Draw(tile).text((4, 246), f"#{i} {p['date']} · {p['dist_m']} m · heading {p['heading']}°", fill="black")
        sheet.paste(tile, ((i % cols) * 324, (i // cols) * 274))
    sheet.save(out / "streetview_contact_sheet.png")
    fig, ax = plt.subplots(figsize=(7, 7)); ax.plot(*poly.exterior.xy, color="#d81b60", lw=2)
    for i, p in enumerate(imgs):
        ax.plot(p["x"], p["y"], "o", color="#1e88e5", ms=6); h = math.radians(p["heading"])
        ax.arrow(p["x"], p["y"], 8 * math.sin(h), 8 * math.cos(h), head_width=2, color="#1e88e5", lw=0.8)
        ax.text(p["x"] + 1.5, p["y"] + 1.5, str(i), fontsize=8)
    for p in panos.values():
        if p["inside"]: ax.plot(p["x"], p["y"], "x", color="#999", ms=6)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Street View panoramas around the footprint (× = inside, skipped)",
                 loc="left", fontsize=10, weight="bold")
    fig.savefig(out / "streetview_coverage.png", dpi=130, bbox_inches="tight")


def ring_from(points=None, ring=None):
    """The footprint ring: from a points npz written by `building.points`, or a JSON coordinate list."""
    if points: return np.load(points)["ring"]
    c = json.load(open(ring))
    if isinstance(c, dict): c = c.get("coords", c.get("ring"))
    return np.asarray(c[0] if isinstance(c[0][0], (list, tuple)) else c, float)


def main(argv=None):
    from shapely.geometry import Polygon
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--points", help="npz from building.points, carrying `ring`")
    ap.add_argument("--ring", help="JSON coordinate list of the footprint instead")
    ap.add_argument("--out", required=True, help="photo folder to write (building3d reads this)")
    ap.add_argument("--key", help="Street View API key (else $GOOGLE_STREETVIEW_KEY, else --env)")
    ap.add_argument("--env", help="file holding a GOOGLE_STREETVIEW_KEY= line")
    ap.add_argument("--crs", type=int, default=26910, help="EPSG of the ring (default 26910, UTM 10N)")
    ap.add_argument("--limit", type=int, default=LIMIT, help=f"images to fetch (default {LIMIT})")
    ap.add_argument("--footprints", help="duckOverture extract: skip panoramas another footprint blocks")
    ap.add_argument("--tile", help="the point store: skip cameras with no ground under them or returns in the line of sight")
    ap.add_argument("--probe-only", action="store_true", help="metadata only — free, fetches nothing")
    a = ap.parse_args(argv)
    if not (a.points or a.ring): ap.error("need --points or --ring")
    key = api_key(a.key, a.env)
    poly = Polygon(ring_from(a.points, a.ring))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    panos = discover(poly, key, crs=a.crs)
    print(f"{len(panos)} panoramas; dates {sorted(Counter(p['date'] for p in panos.values()).items())}")
    if a.probe_only:
        json.dump(list(panos.values()), open(Path(a.out) / "panos.json", "w"), indent=1); return 0
    others = neighbours(a.footprints, poly, crs=a.crs) if a.footprints else ()
    if a.tile:
        from ducklidar.building.points import building_points
        zg = building_points(np.asarray(poly.exterior.coords)[:-1], a.tile)["z_ground"]
        survey_check(poly, list(panos.values()), a.tile, zg)
        panos = {k: v for k, v in panos.items() if v.get("survey") == "ok"}
        print(f"{len(panos)} panoramas the survey says can see the building")
    imgs = fetch(poly, panos, a.out, key, limit=a.limit, others=others)
    print(f"fetched {len(imgs)} -> {Path(a.out)/'streetview'}")
    sheet_and_map(poly, panos, imgs, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
