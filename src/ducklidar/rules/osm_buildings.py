"""Fetch OSM building footprints for the study box -- the external fact for false veg.

The silo lesson: a large self-consistent false-veg blob defeats every internal
instrument -- its paths descend through its own mislabelled matter (circular), its
component dilutes the colour vote with fused shore trees, and material alone stalls at
~90%. What geometry cannot supply, the map can: the silos are mapped structures. A
"not vegetal AND inside a mapped building" conjunction has the independence the
internal witnesses lack.

Writes ``out/osm_buildings.json``: one record per closed way (polygon) tagged
``building=*`` or ``man_made`` in (silo, storage_tank, works, tower), coordinates in
the tile's UTM (EPSG:26910). Same contract as osm_bridges.json: consumers read it if
present and fall back, loudly, if not.

    python tools/osm_buildings.py [out.json]
"""
import json
import sys
import urllib.parse
import urllib.request

import ducklidar as dl
from ._env import out, STUDY_BOX

B = dl.box(*STUDY_BOX)
MARGIN = 20.0
OVERPASS = "https://overpass-api.de/api/interpreter"


def main(argv):
    from pyproj import Transformer

    to_ll = Transformer.from_crs(26910, 4326, always_xy=True)
    to_utm = Transformer.from_crs(4326, 26910, always_xy=True)
    lon0, lat0 = to_ll.transform(B[0] - MARGIN, B[1] - MARGIN)
    lon1, lat1 = to_ll.transform(B[2] + MARGIN, B[3] + MARGIN)
    bb = f"({lat0:.6f},{lon0:.6f},{lat1:.6f},{lon1:.6f})"
    # relations too: industrial complexes are often multipolygons, and skipping them
    # left the hotspot silos unmapped on the first fetch (Kaveh's catch)
    q = (f'[out:json][timeout:90];('
         f'way["building"]{bb};'
         f'way["man_made"~"^(silo|storage_tank|works|tower)$"]{bb};'
         f'relation["building"]{bb};'
         f'relation["man_made"~"^(silo|storage_tank|works|tower)$"]{bb};'
         f'way["landuse"="industrial"]{bb};'
         f'relation["landuse"="industrial"]{bb};'
         f');out geom;')
    req = urllib.request.Request(OVERPASS,
                                 urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": "lidar-learning/ducklidar"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)

    polys, industrial = [], []
    for w in data.get("elements", []):
        t = w.get("tags", {})
        rings = []
        if w["type"] == "way":
            rings = [w.get("geometry", [])]
        elif w["type"] == "relation":
            rings = [m.get("geometry", []) for m in w.get("members", [])
                     if m.get("role") == "outer"]
        for ring in rings:
            coords = [to_utm.transform(g["lon"], g["lat"]) for g in ring]
            if len(coords) < 4 or coords[0] != coords[-1]:
                continue
            rec = {"id": w["id"], "building": t.get("building", ""),
                   "man_made": t.get("man_made", ""), "name": t.get("name", ""),
                   "coords": [[round(x, 2), round(y, 2)] for x, y in coords]}
            (industrial if t.get("landuse") == "industrial" else polys).append(rec)
    dest = out("osm_buildings.json", argv)
    dest.write_text(json.dumps({"crs": "EPSG:26910", "box_margin": MARGIN,
                                "polys": polys, "industrial": industrial}, indent=1))
    print(f"wrote {dest}: {len(polys)} building + {len(industrial)} industrial polygons")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
