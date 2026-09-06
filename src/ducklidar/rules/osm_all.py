"""Fetch ALL OSM ways for the study box -- the complete map, one file.

The piecemeal fetchers (buildings, bridges, piers, water, structures) each carry
a curated slice; classification keeps discovering it needs one more feature type
(footway boardwalks over the water were 'barges' until Kaveh spotted them,
2026-07-31). This grabs every way with tags + geometry so _recon can consult
anything without another fetch round-trip.

Writes ``out/osm_all.json``: [{id, tags, coords(UTM)}].

    python tools/osm_all.py [out.json]
"""
import json
import sys
import urllib.parse
import urllib.request

import ducklidar as dl
from ._env import out, STUDY_BOX

B = dl.box(*STUDY_BOX)
MARGIN = 40.0
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]


def main(argv):
    from pyproj import Transformer

    to_ll = Transformer.from_crs(26910, 4326, always_xy=True)
    to_utm = Transformer.from_crs(4326, 26910, always_xy=True)
    lon0, lat0 = to_ll.transform(B[0] - MARGIN, B[1] - MARGIN)
    lon1, lat1 = to_ll.transform(B[2] + MARGIN, B[3] + MARGIN)
    q = (f'[out:json][timeout:90];('
         f'way({lat0:.6f},{lon0:.6f},{lat1:.6f},{lon1:.6f});'
         f'node({lat0:.6f},{lon0:.6f},{lat1:.6f},{lon1:.6f})[~"."~"."];'
         ');out tags geom;')
    data = None
    for host in OVERPASS:
        try:
            req = urllib.request.Request(host,
                                         urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": "lidar-learning/ducklidar"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
            break
        except Exception as e:
            print(f"{host}: {e}")
    if data is None:
        sys.exit("all overpass mirrors failed")

    ways, nodes = [], []
    for w in data.get("elements", []):
        if w["type"] == "node":
            x, y = to_utm.transform(w["lon"], w["lat"])
            nodes.append({"id": w["id"], "tags": w.get("tags", {}),
                          "xy": [round(x, 2), round(y, 2)]})
            continue
        coords = [to_utm.transform(g["lon"], g["lat"]) for g in w.get("geometry", [])]
        if len(coords) < 2:
            continue
        ways.append({"id": w["id"], "tags": w.get("tags", {}),
                     "coords": [[round(x, 2), round(y, 2)] for x, y in coords]})
    dest = out("osm_all.json", argv)
    dest.write_text(json.dumps({"crs": "EPSG:26910", "box_margin": MARGIN,
                                "ways": ways, "nodes": nodes}))
    from collections import Counter
    keys = Counter(k for w in ways for k in w["tags"]
                   if k in ("highway", "building", "man_made", "natural", "leisure"))
    print(f"wrote {dest}: {len(ways)} ways ({dict(keys)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
