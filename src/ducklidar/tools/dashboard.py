"""TOOL — one page for a whole folder of buildings: pick one, see its points, model and photos.

    python -m ducklidar.tools.dashboard --root . --out out/dashboard.html
    python -m ducklidar.tools.dashboard --root . --only gers_08133905 w714927893

Walks the layout of `docs/data-layout.md`: every `out/<id>/` that holds a `building.glb` is a
building, its photos come from `buildings/<id>/photos/manifest.json`, and its returns from
`buildings/<id>/points.npz` when one was cached (else the points panel says so). One HTML file, fully
self-contained — three.js, the models, the thumbnails and the points are all embedded, so it
opens from disk with no network and can be mailed.

Left: the LiDAR returns in the survey's own colour. Right: the model, orbit and wireframe.
Below: the photos with the camera that took each one marked in the model view. A building
picker at the top switches everything at once. Nothing here is required to build a model;
this is for looking at what was built.
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

MAX_POINTS = 120_000                   # per building, subsampled — the page must stay openable
THUMB = 420                            # px, the long edge of an embedded photo


def b64(data, mime):
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def thumb(path, w=THUMB):
    from PIL import Image
    im = Image.open(path).convert("RGB"); im.thumbnail((w, w))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=78)
    return b64(buf.getvalue(), "image/jpeg")


def collect(root, only=(), max_points=MAX_POINTS, log=print):
    """-> [building dict] for every out/<id>/building.glb under `root`."""
    import numpy as np
    root = Path(root); out = []
    for d in sorted((root / "out").iterdir() if (root / "out").is_dir() else []):
        glb = d / "building.glb"
        if not glb.is_file() or (only and d.name not in only): continue
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").is_file() else {}
        ox, oy, oz = meta.get("origin_utm", (0, 0, 0))
        b = dict(id=d.name, name=meta.get("name", d.name), level=meta.get("level", "?"), built=meta.get("built", ""),
                 glb=base64.b64encode(glb.read_bytes()).decode(), photos=[], points=None, npts=0,
                 walls=[dict(id=f["id"], w=f["w"], h=f["h"], facing=f["facing"], method=f.get("method", ""),
                             filled=f.get("filled", 0), elements=len(f.get("elements", [])))
                        for f in meta.get("facades", []) if f.get("w", 0) >= 3 and f.get("h", 0) >= 2.5],
                 qa=b64((d / "wall_qa.png").read_bytes(), "image/png") if (d / "wall_qa.png").is_file() else None)
        # points: a cached npz next to the footprint, else nothing (the panel says so)
        npz = next((p for p in (root / "buildings" / d.name / "points.npz", d / "points.npz") if p.is_file()), None)
        if npz is not None:
            z = np.load(npz); n = len(z["x"]); step = max(1, n // max_points)
            xyz = np.column_stack([z["x"][::step] - ox, z["z"][::step] - oz, -(z["y"][::step] - oy)]).astype(np.float32)
            rgb = z["rgb"][::step].astype(np.uint8) if "rgb" in z.files else np.full((len(xyz), 3), 160, np.uint8)
            b.update(points=base64.b64encode(xyz.tobytes()).decode(),
                     colors=base64.b64encode(rgb.tobytes()).decode(), npts=len(xyz), npts_all=n)
        # photos: the manifest a fetcher wrote
        ph = root / "buildings" / d.name / "photos"
        mf = next((m for m in (ph / "manifest.json", ph / "streetview" / "manifest.json",
                               ph / "manifest_all.json") if m.is_file()), None)
        if mf is not None:
            man = json.loads(mf.read_text())
            for i, p in enumerate(man):
                f = Path(p.get("file", ""))
                # a fetcher may write paths relative to its own cwd or to the manifest — try both
                if not f.is_absolute():
                    f = next((c for c in (mf.parent / f, root / f, Path.cwd() / f) if c.is_file()), mf.parent / f)
                if not f.is_file() or p.get("unused"): continue      # the build skipped it; so does the page
                b["photos"].append(dict(i=i, src=thumb(f), date=p.get("date", ""), dist=p.get("dist_m", ""),
                                        heading=p.get("heading", 0), unused=p.get("unused", ""),
                                        x=round(p.get("x", ox) - ox, 2), y=round(-(p.get("y", oy) - oy), 2)))
        log(f"  {b['id']:<24} {b['level']:<8} {len(b['walls']):3d} walls  {len(b['photos']):3d} photos  "
            f"{b['npts']:>7,} pts  {len(b['glb']) * 3 // 4 // 1024:>6} KB model")
        out.append(b)
    return out


VENDOR = Path(__file__).parent / "vendor"          # three.js r128 + GLTFLoader + OrbitControls, MIT


def three_js():
    """The three.js bundle, inlined — the page must open with no network at all (a browser on
    another machine, an air-gapped review, an emailed file)."""
    return "\n".join((VENDOR / n).read_text() for n in ("three.min.js", "GLTFLoader.js", "OrbitControls.js"))


def page(buildings):
    return (TEMPLATE.replace("%THREE%", three_js())
            .replace("%DATA%", json.dumps(buildings)).replace("%N%", str(len(buildings)))
            .replace("%GEN%", __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")))


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>ducklidar — buildings</title>
<style>
:root{--ink:#1f2328;--muted:#667085;--line:#e3e6ea;--bg:#f6f7f9}
html,body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
header{padding:14px 20px 10px;display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}
header h1{margin:0;font-size:17px}header span{color:var(--muted)}
select{font:14px system-ui;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:#fff}
#bid{font:13px ui-monospace,monospace;margin-left:10px;padding:3px 8px;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:copy;user-select:all}
#prev,#next{font:14px system-ui;margin-left:4px;padding:2px 8px;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:pointer}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:0 20px}
.card{background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden}
.card h2{margin:0;padding:9px 14px;font-size:13px;border-bottom:1px solid var(--line);font-weight:600}
.card h2 span{color:var(--muted);font-weight:400;margin-left:8px}
canvas{display:block;width:100%;height:460px}
.strip{padding:12px 20px 20px}.strip h2{font-size:13px;margin:0 0 8px}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:8px}
.ph{background:#fff;border:1px solid var(--line);border-radius:8px;overflow:hidden;cursor:pointer}
.ph img{width:100%;display:block;aspect-ratio:1;object-fit:cover}
.ph div{padding:5px 8px;font-size:11px;color:var(--muted)}
.ph.off{opacity:.4}.ph.sel{outline:3px solid #d81b60}
table{border-collapse:collapse;font-size:12px;width:100%}
td,th{padding:3px 10px 3px 0;text-align:left}th{color:var(--muted);font-weight:500}
#big{position:fixed;inset:0;background:rgba(0,0,0,.86);display:none;align-items:center;justify-content:center;z-index:9}
#big img{max-width:92vw;max-height:92vh}
label{font-size:12px;color:var(--muted);margin-left:12px}
.note{color:var(--muted);padding:20px;font-size:13px}
</style></head><body>
<header><h1>ducklidar — buildings <small id="gen"></small></h1>
<select id="pick"></select><code id="bid" title="click to copy the id"></code><button id="prev" title="previous">&lsaquo;</button><button id="next" title="next">&rsaquo;</button><span id="sub"></span></header>
<div class="grid">
 <div class="card"><h2>LiDAR returns<span id="ptsinfo"></span></h2><canvas id="cpts"></canvas></div>
 <div class="card"><h2>3-D model<span id="mdlinfo"></span>
   <label><input type="checkbox" id="wire"> wireframe</label>
   <label><input type="checkbox" id="cams" checked> cameras</label></h2><canvas id="cmodel"></canvas></div>
</div>
<div class="strip"><h2>Walls</h2><div id="walls"></div></div>
<div class="strip"><h2>Photos <span style="color:var(--muted);font-weight:400">— click one to see it large and mark its camera in the model; faded ones were not used</span></h2>
 <div class="gallery" id="gallery"></div></div>
<div id="big" onclick="this.style.display='none'"><img id="bigimg"></div>
<script>%THREE%</script>
<script>
const B = %DATA%;
const GEN = "%GEN%";
const bin = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
function view(canvas){
  const r = new THREE.WebGLRenderer({canvas, antialias:true}); r.setPixelRatio(devicePixelRatio); r.outputEncoding = THREE.sRGBEncoding;
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0xe9eef2);
  const cam = new THREE.PerspectiveCamera(45, 1, 0.5, 3000); cam.position.set(38, 24, 42);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 0.85));
  const sun = new THREE.DirectionalLight(0xffffff, 0.6); sun.position.set(-40, 80, 30); scene.add(sun);
  const g = new THREE.Mesh(new THREE.PlaneGeometry(600, 600), new THREE.MeshLambertMaterial({color:0xd7d2c4}));
  g.rotation.x = -Math.PI/2; g.position.y = -0.05; scene.add(g);
  const ctl = new THREE.OrbitControls(cam, canvas); ctl.target.set(0, 5, 0);
  const frame = obj => {                       // sit the camera where the whole building fits
    const bb = new THREE.Box3().setFromObject(obj); if (bb.isEmpty()) return;
    const c = bb.getCenter(new THREE.Vector3()), s = bb.getSize(new THREE.Vector3());
    const d = Math.max(s.x, s.z, s.y) * 1.7 + 8;
    cam.position.set(c.x + d * 0.75, c.y + d * 0.55, c.z + d * 0.75);
    ctl.target.copy(c); cam.near = d / 200; cam.far = d * 20; cam.updateProjectionMatrix(); ctl.update();
  };
  const fit = () => { const w = canvas.clientWidth, h = canvas.clientHeight; r.setSize(w, h, false); cam.aspect = w/h; cam.updateProjectionMatrix(); };
  addEventListener("resize", fit); fit();
  (function loop(){ requestAnimationFrame(loop); ctl.update(); r.render(scene, cam); })();
  return {scene, cam, ctl, frame, clear(keep){ [...scene.children].forEach(o => { if (o !== keep && !o.isLight && o !== g) scene.remove(o); }); }};
}
const P = view(document.getElementById("cpts")), M = view(document.getElementById("cmodel"));
let model = null, camGroup = null, marks = {};
function show(b){
  const bid = document.getElementById("bid"); bid.textContent = b.id;
  bid.onclick = () => { navigator.clipboard?.writeText(b.id); bid.style.background = "#dff5e1"; setTimeout(() => bid.style.background = "#fff", 400); };
  document.getElementById("sub").textContent = `${b.level} · ${b.walls.length} walls · ${b.photos.length} photos · built ${b.built || "?"}`;
  // points
  P.clear();
  document.getElementById("ptsinfo").textContent = b.points
    ? `${b.npts.toLocaleString()} of ${b.npts_all.toLocaleString()} shown, the survey's own colour`
    : "no cached points.npz for this building";
  if (b.points){
    const pos = new Float32Array(bin(b.points).buffer), col = bin(b.colors);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3, true));
    const pc = new THREE.Points(geo, new THREE.PointsMaterial({size:0.16, vertexColors:true}));
    P.scene.add(pc); P.frame(pc);
  }
  // model
  M.clear(); model = null; marks = {};
  document.getElementById("mdlinfo").textContent = `${(b.glb.length * 3 / 4 / 1024) | 0} KB`;
  new THREE.GLTFLoader().parse(bin(b.glb).buffer, "", g => {
    model = g.scene; model.traverse(o => { if (o.material && o.material.map) o.material.map.encoding = THREE.sRGBEncoding; });
    M.scene.add(model); M.frame(model); wire();
  });
  camGroup = new THREE.Group(); M.scene.add(camGroup);
  b.photos.forEach(p => { if (p.unused) return;
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.4, 10, 10), new THREE.MeshLambertMaterial({color:0x1e88e5}));
    m.position.set(p.x, 2.3, p.y);
    const d = new THREE.Vector3(Math.sin(p.heading*Math.PI/180), 0, -Math.cos(p.heading*Math.PI/180));
    camGroup.add(m, new THREE.ArrowHelper(d, m.position, 3.5, 0x1e88e5, 1.1, 0.55)); marks[p.i] = m; });
  camGroup.visible = document.getElementById("cams").checked;
  // walls
  document.getElementById("walls").innerHTML = b.walls.length ? `<table><tr><th>wall<th>size (m)<th>facing<th>method<th>filled<th>elements</tr>` +
    b.walls.slice().sort((a, z) => (z.filled - a.filled) || (z.w * z.h - a.w * a.h)).map(w => `<tr><td>${w.id}<td>${w.w} × ${w.h}<td>${w.facing}°<td>${w.method}<td>${w.filled ? (w.filled*100).toFixed(0)+" %" : "—"}<td>${w.elements || "—"}</tr>`).join("") + "</table>"
    : `<div class="note">no walls of 3 m or more</div>`;
  // photos
  const gal = document.getElementById("gallery"); gal.innerHTML = "";
  b.photos.forEach(p => { const d = document.createElement("div");
    d.className = "ph" + (p.unused ? " off" : "");
    d.innerHTML = `<img src="${p.src}"><div>#${p.i} · ${p.date} · ${p.dist} m · ${p.heading}°${p.unused ? " · " + p.unused : ""}</div>`;
    d.onclick = () => { document.querySelectorAll(".ph").forEach(x => x.classList.remove("sel")); d.classList.add("sel");
      Object.values(marks).forEach(m => m.material.color.set(0x1e88e5));
      if (marks[p.i]) marks[p.i].material.color.set(0xd81b60);
      document.getElementById("bigimg").src = p.src; document.getElementById("big").style.display = "flex"; };
    gal.appendChild(d); });
  if (!b.photos.length) gal.innerHTML = `<div class="note">no photos for this building — geometry and material only</div>`;
}
function wire(){ const on = document.getElementById("wire").checked; model && model.traverse(o => { if (o.material) o.material.wireframe = on; }); }
document.getElementById("wire").onchange = wire;
document.getElementById("cams").onchange = e => { if (camGroup) camGroup.visible = e.target.checked; };
const pick = document.getElementById("pick");
B.forEach((b, i) => { const o = document.createElement("option"); o.value = i; o.textContent = `${b.name}  (${b.level})`; pick.appendChild(o); });
pick.onchange = () => show(B[+pick.value]);
document.getElementById("gen").textContent = "page written " + GEN;
document.getElementById("prev").onclick = () => { pick.value = Math.max(0, +pick.value - 1); show(B[+pick.value]); };
document.getElementById("next").onclick = () => { pick.value = Math.min(B.length - 1, +pick.value + 1); show(B[+pick.value]); };
if (B.length) show(B[0]); else document.body.insertAdjacentHTML("beforeend", '<div class="note">no buildings found under out/</div>');
</script></body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="the data root (holds out/ and buildings/)")
    ap.add_argument("--out", default="out/dashboard.html", help="page to write")
    ap.add_argument("--only", nargs="*", default=(), help="building ids to include (default: all)")
    ap.add_argument("--max-points", type=int, default=MAX_POINTS, help=f"points per building (default {MAX_POINTS})")
    a = ap.parse_args(argv)
    root = Path(a.root)
    print(f"scanning {root / 'out'}")
    bs = collect(root, only=set(a.only), max_points=a.max_points)
    if not bs:
        print("no buildings found — expected out/<id>/building.glb"); return 1
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(bs))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(bs)} buildings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
