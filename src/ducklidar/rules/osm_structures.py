"""Fetch OSM special-structure footprints for the study box: storage tanks and
roof-only canopies.

Tanks (man_made=storage_tank) are cylinders -- the Granville Island silos -- and
canopies (building=roof) are slabs on posts; both render wrongly as walled
extrusions without their footprints. _recon matches instances to these polygons
and swaps in the right primitive; missing file = no swap, loudly.

Writes ``out/osm_structures.json`` (UTM coords, like the bridges/piers files).

    python tools/osm_structures.py [out.json]
"""
import json
import sys
import urllib.parse
import urllib.request

import ducklidar as dl
from ._env import out, STUDY_BOX

B = dl.box(*STUDY_BOX)
OVERPASS = "https://overpass-api.de/api/interpreter"


def main(argv):
    from pyproj import Transformer

    to_ll = Transformer.from_crs(26910, 4326, always_xy=True)
    to_utm = Transformer.from_crs(4326, 26910, always_xy=True)
    lon0, lat0 = to_ll.transform(B[0], B[1])
    lon1, lat1 = to_ll.transform(B[2], B[3])
    bbox = f"({lat0:.6f},{lon0:.6f},{lat1:.6f},{lon1:.6f})"
    q = (f'[out:json][timeout:60];(way["man_made"="storage_tank"]{bbox};'
         f'way["building"="roof"]{bbox};);out geom;')
    req = urllib.request.Request(OVERPASS,
                                 urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": "lidar-learning/ducklidar"})
    with urllib.request.urlopen(req, timeout=90) as r:
        data = json.load(r)

    recs = []
    for w in data.get("elements", []):
        t = w.get("tags", {})
        coords = [to_utm.transform(g["lon"], g["lat"]) for g in w.get("geometry", [])]
        if len(coords) < 4:
            continue
        recs.append({"id": w["id"],
                     "kind": "tank" if t.get("man_made") == "storage_tank" else "canopy",
                     "height": t.get("height", ""),
                     "coords": [[round(x, 2), round(y, 2)] for x, y in coords]})
    dest = out("osm_structures.json", argv)
    dest.write_text(json.dumps({"crs": "EPSG:26910", "structures": recs}, indent=1))
    from collections import Counter
    print(f"wrote {dest}: {dict(Counter(r['kind'] for r in recs))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
