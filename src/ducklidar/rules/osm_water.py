"""Fetch OSM water geometry for the study box: coastline ways + closed water polys.

The sea layer needs an authority for WHERE water can be: the no-ground-evidence
heuristic alone also fires in scan shadows on land (under the bridge viaduct it
painted real roads blue -- Kaveh, 2026-07-31). OSM's convention: water lies on
the RIGHT of a coastline way's direction; viz_scene3d self-calibrates the sign
against measured water cells rather than trusting the convention blindly.

Writes ``out/osm_water.json`` (UTM coords, like the other osm_* files).

    python tools/osm_water.py [out.json]
"""
import json
import sys
import urllib.parse
import urllib.request

import ducklidar as dl
from ._env import out, STUDY_BOX

B = dl.box(*STUDY_BOX)
MARGIN = 60.0
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]


def main(argv):
    from pyproj import Transformer

    to_ll = Transformer.from_crs(26910, 4326, always_xy=True)
    to_utm = Transformer.from_crs(4326, 26910, always_xy=True)
    lon0, lat0 = to_ll.transform(B[0] - MARGIN, B[1] - MARGIN)
    lon1, lat1 = to_ll.transform(B[2] + MARGIN, B[3] + MARGIN)
    bbox = f"({lat0:.6f},{lon0:.6f},{lat1:.6f},{lon1:.6f})"
    q = (f'[out:json][timeout:60];(way["natural"="coastline"]{bbox};'
         f'way["natural"="water"]{bbox};);out geom;')
    data = None
    for host in OVERPASS:
        try:
            req = urllib.request.Request(host,
                                         urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": "lidar-learning/ducklidar"})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.load(r)
            break
        except Exception as e:
            print(f"{host}: {e}")
    if data is None:
        sys.exit("all overpass mirrors failed")

    coast, polys = [], []
    for w in data.get("elements", []):
        t = w.get("tags", {})
        coords = [to_utm.transform(g["lon"], g["lat"]) for g in w.get("geometry", [])]
        if len(coords) < 2:
            continue
        rec = {"id": w["id"],
               "coords": [[round(x, 2), round(y, 2)] for x, y in coords]}
        if t.get("natural") == "coastline":
            coast.append(rec)                      # direction matters: keep as-is
        elif len(coords) > 3 and coords[0] == coords[-1]:
            polys.append(rec)
    dest = out("osm_water.json", argv)
    dest.write_text(json.dumps({"crs": "EPSG:26910", "box_margin": MARGIN,
                                "coastlines": coast, "polys": polys}, indent=1))
    print(f"wrote {dest}: {len(coast)} coastline ways, {len(polys)} water polygons")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
