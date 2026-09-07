"""TOOL — a map of the tile's buildings: every footprint, its id, and a link to the spot in Google Earth.

    python -m ducklidar.tools.map --root . --footprints footprints/granville_island.duckdb \
        --tile tiles/490000_5457000.parquet --out out/map.html

Every `buildings.building` footprint wholly inside the tile is drawn on a Leaflet map (OpenStreetMap
and Esri World Imagery tiles — the page needs the network), coloured by what `out/<id>/` holds:
built, skipped (no building returns), or not built yet. Type an id (or its first characters) in the
search box to fly to that building. Click one: its id (click to copy), the
Overture height and class, the model's level and build time, and links that open the same spot in
Google Earth and in Google Maps satellite view — so a model can be checked against the world.
The ids and the WGS84 centre of every footprint are also listed under the map.
"""
import argparse
import json
import sys
from pathlib import Path

LEAFLET_CSS = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"
LEAFLET_JS = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>ducklidar — %TITLE%</title>
<link rel="stylesheet" href="%CSS%">
<style>
body{margin:0;font:14px system-ui}#map{height:78vh}
header{padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.sw{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin-right:4px}
code{font:12px ui-monospace,monospace;background:#f1f3f5;padding:2px 5px;border-radius:4px;cursor:copy}
table{border-collapse:collapse;margin:10px 16px;font-size:12px}td,th{padding:3px 10px;border-bottom:1px solid #eee;text-align:left}
.leaflet-tooltip.id{font:11px ui-monospace,monospace;background:rgba(255,255,255,.85);border:0;box-shadow:none;padding:1px 4px}
</style></head><body>
<header><b>ducklidar — %TITLE%</b>
 <span><span class="sw" style="background:#1e88e5"></span>built (%NB%)</span>
 <span><span class="sw" style="background:#9e9e9e"></span>skipped, no building returns (%NS%)</span>
 <span><span class="sw" style="background:#fb8c00"></span>not built yet (%NN%)</span>
 <input id="q" list="ids" placeholder="search an id, e.g. gers_0112d1b3" style="font:13px ui-monospace,monospace;padding:4px 8px;width:260px"><datalist id="ids"></datalist>
 <span style="color:#666">click a footprint for its id and Google Earth / Maps links · <label><input type="checkbox" id="sat"> satellite</label> · <label><input type="checkbox" id="labels"> ids on the map</label></span>
</header>
<div id="map"></div>
<table><thead><tr><th>id</th><th>state</th><th>class</th><th>Overture height</th><th>model built</th><th>lat, lon</th><th></th></tr></thead><tbody id="rows"></tbody></table>
<script src="%JS%"></script>
<script>
const B = %DATA%;
const map = L.map("map");
const osm = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {maxZoom: 20, attribution: "© OpenStreetMap"}).addTo(map);
const sat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {maxZoom: 20, attribution: "Esri World Imagery"});
document.getElementById("sat").onchange = e => { if (e.target.checked) { map.removeLayer(osm); sat.addTo(map); } else { map.removeLayer(sat); osm.addTo(map); } };
const col = s => s === "built" ? "#1e88e5" : s === "skipped" ? "#9e9e9e" : "#fb8c00";
const earth = b => `https://earth.google.com/web/@${b.lat},${b.lon},0a,120d,35y,0h,45t,0r`;
const gmaps = b => `https://www.google.com/maps/@${b.lat},${b.lon},120m/data=!3m1!1e3`;
const group = L.featureGroup().addTo(map);
const rows = document.getElementById("rows"), polys = {};
const dl = document.getElementById("ids");
B.forEach(b => {
  const poly = L.polygon(b.ring, {color: col(b.state), weight: 2, fillOpacity: 0.25}).addTo(group);
  polys[b.id] = poly;
  poly.bindTooltip(b.id, {permanent: false, direction: "center", className: "id"});   // on hover; the checkbox pins them all
  const o = document.createElement("option"); o.value = b.id; dl.appendChild(o);
  poly.bindPopup(`<b><code onclick="navigator.clipboard.writeText('${b.id}')">${b.id}</code></b><br>${b.state}${b.built ? " · built " + b.built : ""}<br>Overture: ${b.cls || "?"}${b.height ? ", " + b.height + " m" : ""}<br>${b.lat.toFixed(6)}, ${b.lon.toFixed(6)}<br><a target="_blank" href="${earth(b)}">Google Earth</a> · <a target="_blank" href="${gmaps(b)}">Google Maps satellite</a>`);
  const tr = document.createElement("tr");
  tr.innerHTML = `<td><code onclick="navigator.clipboard.writeText('${b.id}')">${b.id}</code></td><td>${b.state}</td><td>${b.cls || ""}</td><td>${b.height ? b.height + " m" : ""}</td><td>${b.built || ""}</td><td>${b.lat.toFixed(6)}, ${b.lon.toFixed(6)}</td><td><a target="_blank" href="${earth(b)}">Earth</a> · <a target="_blank" href="${gmaps(b)}">Maps</a></td>`;
  tr.onclick = () => { map.fitBounds(poly.getBounds(), {maxZoom: 19}); poly.openPopup(); };
  rows.appendChild(tr);
});
map.fitBounds(group.getBounds());
let hi = null;
function go(id){
  id = id.trim(); const hit = Object.keys(polys).find(k => k === id) || Object.keys(polys).find(k => k.startsWith(id)) || Object.keys(polys).find(k => k.includes(id));
  if (!hit) return;
  if (hi) hi.setStyle({weight: 2, color: col(B.find(b => b.id === hi._id).state)});
  const poly = polys[hit]; poly._id = hit; hi = poly; poly.setStyle({weight: 5, color: "#d81b60"});
  map.fitBounds(poly.getBounds(), {maxZoom: 19}); poly.openPopup();
}
const q = document.getElementById("q");
q.addEventListener("change", () => go(q.value)); q.addEventListener("keydown", e => { if (e.key === "Enter") go(q.value); });
document.getElementById("labels").onchange = e => Object.values(polys).forEach(p => { p.unbindTooltip(); p.bindTooltip(p === hi ? hi._id : B.find(b => polys[b.id] === p).id, {permanent: e.target.checked, direction: "center", className: "id"}); });
if (location.hash.length > 1) go(decodeURIComponent(location.hash.slice(1)));
</script></body></html>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="the data root (holds out/)")
    ap.add_argument("--footprints", required=True, help="a duckOverture extract with buildings.building")
    ap.add_argument("--tile", required=True, help="the point store: only footprints wholly inside it are mapped")
    ap.add_argument("--out", default="out/map.html", help="page to write")
    a = ap.parse_args(argv)

    from pyproj import Transformer
    from .build_tile import footprints, tile_extent, CRS
    import duckdb

    root = Path(a.root)
    fps = footprints(a.footprints, tile_extent(a.tile))
    con = duckdb.connect(a.footprints, read_only=True)
    try:
        con.execute("LOAD spatial")
        attrs = {i: (c, h) for i, c, h in con.execute("select id, class, height from buildings.building").fetchall()}
    finally:
        con.close()
    to_ll = Transformer.from_crs(CRS, 4326, always_xy=True)
    rows = []
    for oid, ring in fps:
        bid = "gers_" + oid[:8]
        odir = root / "out" / bid
        meta = json.loads((odir / "meta.json").read_text()) if (odir / "meta.json").is_file() else {}
        state = "built" if (odir / "building.glb").is_file() else ("skipped" if (root / "buildings" / bid / "footprint.json").is_file() and odir.is_dir() else "not built")
        xs, ys = zip(*ring)
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        lon, lat = to_ll.transform(cx, cy)
        latlon = [list(reversed(to_ll.transform(x, y))) for x, y in ring]
        cls, h = attrs.get(oid, (None, None))
        rows.append(dict(id=bid, ring=latlon, lat=lat, lon=lon, state=state, built=meta.get("built", ""),
                         cls=cls, height=round(h, 1) if h else None))
    n = lambda s: sum(1 for r in rows if r["state"] == s)
    html = (PAGE.replace("%DATA%", json.dumps(rows)).replace("%CSS%", LEAFLET_CSS).replace("%JS%", LEAFLET_JS)
            .replace("%TITLE%", f"buildings of {Path(a.tile).stem}")
            .replace("%NB%", str(n("built"))).replace("%NS%", str(n("skipped"))).replace("%NN%", str(n("not built"))))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(html)
    print(f"wrote {a.out}: {len(rows)} footprints, {n('built')} built, {n('skipped')} skipped, {n('not built')} not built")
    return 0


if __name__ == "__main__":
    sys.exit(main())
