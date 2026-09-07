"""TOOL — Street View around one footprint: probe (free), fetch (billable), sheet + map.

**Not part of a build.** `ducklidar.building3d` never calls this; it reads the folder this
writes. Run it once per building, yourself, with your own key:

    python -m ducklidar.tools.fetch_streetview --ring footprint.json --out out/photos
    python -m ducklidar.tools.fetch_streetview --points out/pilot_netloft_points.npz --out out/photos

Metadata probes on rings 12/30/50 m outside the footprint find every panorama within 20 m
of a probe; panoramas *inside* the footprint (indoor photospheres) are skipped; one
640x640, fov 90, pitch 8 view per panorama is fetched, aimed at the nearest footprint edge.

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


def fetch(poly, panos, out, key, limit=LIMIT, max_dist=MAX_DIST):
    """One view per outside panorama, aimed at the nearest footprint edge. THIS is the billable part."""
    from shapely.geometry import Point
    sv = Path(out) / "streetview"; sv.mkdir(parents=True, exist_ok=True)
    outside = sorted((p for p in panos.values() if not p["inside"] and p["dist_m"] <= max_dist),
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
    imgs = fetch(poly, panos, a.out, key, limit=a.limit)
    print(f"fetched {len(imgs)} -> {Path(a.out)/'streetview'}")
    sheet_and_map(poly, panos, imgs, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
