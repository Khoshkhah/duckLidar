"""Fetch OSM bridge centrelines for the study box -- the ramps the void test cannot see.

Kaveh's observation, and it is a fact about bridges rather than about this tile: a bridge is
not all "on fly". It starts at grade, climbs, and comes down again, and the void test can only
see the part with air underneath. The growth rule recovers some ramp by continuity but is
stopped by its own 4 m floor -- which exists to keep it from running down onto ordinary roads.
OSM already knows exactly where the bridge is: every carriageway is a way tagged
``bridge=yes``, ending where the structure meets the ground. That boundary is the external
fact the geometry cannot supply, so inside the OSM corridor the pipeline may follow the deck
all the way to grade, and outside it the 4 m floor still protects the roads.

Writes ``out/osm_bridges.json``: one record per way, coordinates already in the tile's UTM
(EPSG:26910). The pipeline reads it if present and falls back to pure geometry, loudly, if
not -- same contract as RGB.

    python tools/osm_bridges.py [out.json]
"""
import json
import sys
import urllib.parse
import urllib.request

import ducklidar as dl
from ._env import out, STUDY_BOX

B = dl.box(*STUDY_BOX)
MARGIN = 100.0          # metres beyond the box: a ramp may re-enter
OVERPASS = "https://overpass-api.de/api/interpreter"


def main(argv):
    from pyproj import Transformer

    to_ll = Transformer.from_crs(26910, 4326, always_xy=True)
    to_utm = Transformer.from_crs(4326, 26910, always_xy=True)
    lon0, lat0 = to_ll.transform(B[0] - MARGIN, B[1] - MARGIN)
    lon1, lat1 = to_ll.transform(B[2] + MARGIN, B[3] + MARGIN)
    q = (f'[out:json][timeout:60];way["bridge"]["highway"]'
         f'({lat0:.6f},{lon0:.6f},{lat1:.6f},{lon1:.6f});out geom;')
    req = urllib.request.Request(OVERPASS,
                                 urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": "lidar-learning/ducklidar"})
    with urllib.request.urlopen(req, timeout=90) as r:
        data = json.load(r)

    ways = []
    for w in data.get("elements", []):
        t = w.get("tags", {})
        coords = [to_utm.transform(g["lon"], g["lat"]) for g in w.get("geometry", [])]
        if len(coords) < 2:
            continue
        ways.append({"id": w["id"], "name": t.get("name", ""),
                     "highway": t.get("highway", ""), "lanes": t.get("lanes", ""),
                     "layer": t.get("layer", ""),
                     "coords": [[round(x, 2), round(y, 2)] for x, y in coords]})
    dest = out("osm_bridges.json", argv)
    dest.write_text(json.dumps({"crs": "EPSG:26910", "box_margin": MARGIN, "ways": ways},
                               indent=1))
    names = sorted({w["name"] for w in ways if w["name"]})
    print(f"wrote {dest}: {len(ways)} bridge ways ({', '.join(names)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
