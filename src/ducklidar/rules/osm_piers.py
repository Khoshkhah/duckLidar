"""Fetch OSM pier/marina ways for the study box -- the walkway skeleton of the docks.

The marina raft is one giant floating instance; OSM maps each walkway as a
``man_made=pier`` way (mostly centrelines, a few closed areas). Those ways are the
external fact that lets the raft be split into named walkways -- the same role
osm_bridges.json plays for the bridge deck.

Writes ``out/osm_piers.json``: one record per way, coordinates in the tile's UTM
(EPSG:26910). _recon uses it if present and leaves docks unsplit, loudly, if not.

    python tools/osm_piers.py [out.json]
"""
import json
import sys
import urllib.parse
import urllib.request

import ducklidar as dl
from ._env import out, STUDY_BOX

B = dl.box(*STUDY_BOX)
MARGIN = 50.0
OVERPASS = "https://overpass-api.de/api/interpreter"


def main(argv):
    from pyproj import Transformer

    to_ll = Transformer.from_crs(26910, 4326, always_xy=True)
    to_utm = Transformer.from_crs(4326, 26910, always_xy=True)
    lon0, lat0 = to_ll.transform(B[0] - MARGIN, B[1] - MARGIN)
    lon1, lat1 = to_ll.transform(B[2] + MARGIN, B[3] + MARGIN)
    q = (f'[out:json][timeout:60];way["man_made"~"pier|breakwater|quay"]'
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
                     "man_made": t.get("man_made", ""),
                     "area": t.get("area", ""), "floating": t.get("floating", ""),
                     "coords": [[round(x, 2), round(y, 2)] for x, y in coords]})
    dest = out("osm_piers.json", argv)
    dest.write_text(json.dumps({"crs": "EPSG:26910", "box_margin": MARGIN, "ways": ways},
                               indent=1))
    print(f"wrote {dest}: {len(ways)} pier ways "
          f"({sum(1 for w in ways if w['name'])} named)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
