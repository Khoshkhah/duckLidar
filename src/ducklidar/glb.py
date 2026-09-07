"""A textured mesh as one GLB, and a self-contained page to look at it.

    prim = dict(pos=(n,3) UTM, uv=(n,2), idx=(m,3), tex=HxWx3 RGB uint8, name=str)
    n = write_glb("out/building.glb", prims, origin=(ox, oy, oz), name="Net Loft")
    Path("viewer.html").write_text(viewer_html(Path("out/building.glb"), "Net Loft"))

Hand-written glTF 2.0: one primitive per texture, JPEG-encoded, everything in one buffer.
Positions are re-based on `origin` and turned Y-up (glTF's convention) — the file carries no
CRS, so keep `origin` and the EPSG code in your own metadata.
"""
import base64
import json
import struct
from pathlib import Path

import numpy as np


def write_glb(path, prims, origin, name="model", generator="ducklidar"):
    """prims: list of dict(pos (n,3) UTM, uv (n,2), idx (m,3), tex RGB uint8 HxWx3, name)."""
    import cv2
    ox, oy, oz = origin
    bin_parts, views, accessors, images, textures, materials, mesh_prims = [], [], [], [], [], [], []
    off = 0

    def add_view(data):
        nonlocal off
        data = bytes(data); pad = (-len(data)) % 4
        bin_parts.append(data + b"\0" * pad); views.append(dict(buffer=0, byteOffset=off, byteLength=len(data)))
        off += len(data) + pad; return len(views) - 1

    for i, p in enumerate(prims):
        pos = np.column_stack([p["pos"][:, 0] - ox, p["pos"][:, 2] - oz, -(p["pos"][:, 1] - oy)]).astype(np.float32)   # Y-up
        uv = p["uv"].astype(np.float32); idx = p["idx"].astype(np.uint32).ravel()
        vpos = add_view(pos.tobytes()); vuv = add_view(uv.tobytes()); vidx = add_view(idx.tobytes())
        accessors += [dict(bufferView=vpos, componentType=5126, count=len(pos), type="VEC3", min=pos.min(0).tolist(), max=pos.max(0).tolist()),
                      dict(bufferView=vuv, componentType=5126, count=len(uv), type="VEC2"),
                      dict(bufferView=vidx, componentType=5125, count=len(idx), type="SCALAR")]
        ok, png = cv2.imencode(".jpg", cv2.cvtColor(p["tex"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        vimg = add_view(png.tobytes()); images.append(dict(bufferView=vimg, mimeType="image/jpeg", name=p["name"]))
        textures.append(dict(source=len(images) - 1, sampler=0))
        materials.append(dict(name=p["name"], doubleSided=True, pbrMetallicRoughness=dict(baseColorTexture=dict(index=len(textures) - 1), metallicFactor=0.0, roughnessFactor=0.9)))
        a = len(accessors) - 3
        mesh_prims.append(dict(attributes=dict(POSITION=a, TEXCOORD_0=a + 1), indices=a + 2, material=len(materials) - 1))
    gltf = dict(asset=dict(version="2.0", generator=generator), scene=0, scenes=[dict(nodes=[0])], nodes=[dict(mesh=0, name=name)],
                meshes=[dict(primitives=mesh_prims)], materials=materials, textures=textures, images=images,
                samplers=[dict(magFilter=9729, minFilter=9987, wrapS=33071, wrapT=33071)],
                bufferViews=views, accessors=accessors, buffers=[dict(byteLength=off)])
    js = json.dumps(gltf).encode(); js += b" " * ((-len(js)) % 4); bb = b"".join(bin_parts)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bb)))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js); fh.write(struct.pack("<II", len(bb), 0x004E4942) + bb)
    return off


VIEWER = """<!doctype html><html><head><meta charset="utf-8"><title>%NAME% — model</title>
<style>html,body{margin:0;height:100%;background:#e9eef2;font:13px system-ui}#hud{position:absolute;left:12px;top:10px;color:#333}</style></head><body>
<div id="hud"><b>%NAME%</b> · roofer LoD2.2 solid · roof from LiDAR colour · facades from Street View · composed walls: material + measured elements + relief · drag to orbit, wheel to zoom</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const glb = Uint8Array.from(atob("%GLB%"), c => c.charCodeAt(0)).buffer;
const scene = new THREE.Scene(); scene.background = new THREE.Color(0xe9eef2);
const cam = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.5, 2000); cam.position.set(60, 40, 70);
const r = new THREE.WebGLRenderer({antialias:true, preserveDrawingBuffer:true}); r.setSize(innerWidth, innerHeight); r.outputEncoding = THREE.sRGBEncoding; document.body.appendChild(r.domElement);
// exposure: hemi 0.85 + sun 0.6 on a MeshStandard (glTF PBR) material sums to ~1.45x albedo with no tone mapping, so a
// wall whose composed texture is RGB 103,106,100 rendered 192,195,199 — near-white, and blue-shifted by the 0x8899aa
// ground bounce. Sum ~= 1.0 and a NEUTRAL ground keeps the render on the texture's own colour; ACESFilmic rolls the
// highlights off instead of clipping them, so the sunlit side no longer goes to paper white.
r.toneMapping = THREE.ACESFilmicToneMapping; r.toneMappingExposure = 1.0;
scene.add(new THREE.HemisphereLight(0xffffff, 0xb9b3a6, 0.55)); const sun = new THREE.DirectionalLight(0xffffff, 0.75); sun.position.set(-40, 80, 30); scene.add(sun);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.MeshLambertMaterial({color:0xd7d2c4})); ground.rotation.x = -Math.PI/2; ground.position.y = -0.05; scene.add(ground);
const ctl = new THREE.OrbitControls(cam, r.domElement); ctl.target.set(0, 6, 0);
new THREE.GLTFLoader().parse(glb, "", g => { g.scene.traverse(o => { if (o.material && o.material.map) o.material.map.encoding = THREE.sRGBEncoding; }); scene.add(g.scene); });
addEventListener("resize", () => { cam.aspect = innerWidth/innerHeight; cam.updateProjectionMatrix(); r.setSize(innerWidth, innerHeight); });
(function loop(){ requestAnimationFrame(loop); ctl.update(); r.render(scene, cam); })();
</script></body></html>"""


def viewer_html(glb_path, name):
    """The GLB embedded in a one-file three.js page — no server, no side-car."""
    return VIEWER.replace("%GLB%", base64.b64encode(Path(glb_path).read_bytes()).decode()).replace("%NAME%", name)
