"""One call, one textured 3-D building: `dl.building3d(footprint, store=..., photos=...)`.

    import ducklidar as dl
    b = dl.building3d(ring, store="data/lidar/*.parquet", photos="out/photos", out="out/netloft")
    b.glb         # Path — the textured model
    b.walls       # [{id, w, h, facing, photos, filled, elements, …}, …]
    b.elements    # {wall id: [{type, x0, y0, x1, y1, conf, …}, …]}   metres on the wall
    b.qa          # Path — the per-wall QA sheet, or None
    b.meta        # dict — origin, CRS, prims, walls, level, counts

The stages, in the order a building is built (`ducklidar.building.*` and `ducklidar.facade`):

    points     the building's own returns from the store, inside the footprint + eave
    geometry   a solid -> planar facades; skylight boxes; the floor slab
    camera     each photo's real pose, fitted by silhouette against its building mask
    walls      per-wall texture, the building's material, the roof from the survey's colour
    glb        the primitives as one file, plus a viewer page

`level` buys skin, not geometry:

    "none"      the solid, the roof texture, skylights, the floor — no photos read
    "material"  + one measured wall colour, no photo API, no GPU, no masks needed
    "photo"     + the composed texture per wall a photo sees (needs photos + masks)
    "joinery"   + frames, sills, mullions, glass, jambs as GEOMETRY.  Default.

**This function never fetches.** `photos`, `masks` and `elements` are FOLDERS a tool wrote
(`ducklidar.tools.fetch_streetview`, `ducklidar.tools.segment`); a build reads them and
stops if they are absent, dropping to the level the inputs support. Likewise `solid=` takes
a roofer CityJSON if you have one, and without it the geometry falls back to
`objects.building_model` — the surveyed outline extruded to the measured top with the
returns' own roof planes on it. Running roofer is `ducklidar.tools.roofer`, a subprocess,
never linked and never required.
"""
import json
import math
from pathlib import Path

import numpy as np

LEVELS = ("none", "material", "photo", "joinery")
Z_GROUND = 3.7      # m: fallback ground elevation, replaced by the points' own measurement
CAM_H = 2.3         # m: a street-level camera's height above ground
W = 640             # px: the photos are square


class Building:
    """The result of `building3d`: paths to what was written, and what was measured."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        n = sum(1 for w in self.walls if w.get("photos"))
        return (f"<Building {self.meta['name']!r} {len(self.walls)} walls "
                f"({n} composed), {len(self.meta['prims'])} prims, level={self.meta['level']}>")


def building3d(footprint, *, store=None, points=None, photos=None, masks=None, elements=None,
               solid=None, solids=(), out=None, level="joinery", name="building",
               labels=None, indoor=(), clicks=None, z_ground=None, compose=True,
               qa=True, write=True, root=None, log=print):
    """A textured 3-D building. `footprint` is a ring, a shapely polygon, or a points npz.

    store       point store for `dl.read` (a parquet path, a LAS/LAZ, or a list of tiles)
    points      a prepared dict / npz path from `building.points.building_points`, instead of `store`
    photos      folder a fetcher wrote: manifest.json + the images.  None -> level caps at "material"
    masks       occluder / building masks (default `photos/masks`)
    elements    SAM 3 element folder (default `photos/elements`); absent -> caps at "photo"
    solid       roofer CityJSON path; `solids` names the objects in it.  None -> the fallback solid
    out         folder to write into; None writes nothing and only returns the object
    write       False writes only the side-cars (poses, QA) and leaves the GLB bundle to the
                caller — for a host that wants its own file names
    level       one of "none", "material", "photo", "joinery"
    labels      (pid, label) arrays for `building_points`, optional — the library never needs them
    indoor      photo indices to skip (indoor photospheres); the manifest's `unused` flag also skips
    root        base for RELATIVE photo paths in the manifest (default the working directory);
                absolute paths, which is what the fetchers write, ignore it
    """
    from . import facade, glb
    from .building import camera, geometry, points as pts_mod, walls as wl
    from .facade import compose as fcompose, elements as felements, relief as frelief, texture as ftexture

    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, not {level!r}")
    out = Path(out) if out else None
    if out: out.mkdir(parents=True, exist_ok=True)
    photos = Path(photos) if photos else None
    masks = Path(masks) if masks else (photos / "masks" if photos else None)
    elements = Path(elements) if elements else (photos / "elements" if photos else None)

    # ---- points: the building's own returns, and the ground it stands on -------------------
    d = _points(footprint, points, store, labels, pts_mod)
    P = np.column_stack([d["x"], d["y"], d["z"]]); rgb = np.asarray(d["rgb"], float)
    ring = np.asarray(d["ring"])
    measured = z_ground if z_ground is not None else d.get("z_ground")
    zg = float(measured if measured is not None else Z_GROUND)
    if measured is not None:            # only a MEASURED ground is worth a line; the default is not news
        log(f"  ground at the building: z = {zg:.2f} m")

    # ---- geometry: the solid, its facades, its outer edges ---------------------------------
    V, T = (geometry.load_solid(solid, solids) if solid is not None
            else fallback_solid(ring, P, zg))
    n = geometry.normals(V, T); centre = V.mean(0)
    F, slivers = geometry.drop_slivers(geometry.facades(V, T, n, centre))
    F, inner = geometry.outer_walls(F, ring, base_z=float(V[:, 2].min()))
    F = geometry.close_corners(F); E = geometry.outer_edges(F)
    if slivers or inner:
        log(f"  {len(slivers)} sliver + {len(inner)} interior facades dropped; {len(F)} outer walls")
    SOLID = dict(V=V, T=T)
    log(f"solid: {len(V)} vertices, {len(T)} triangles → {len(F)} facades, "
        f"{int((n[:, 2] > 0.3).sum())} roof triangles, {len(E)} outer edges")

    # ---- photos: the manifest a fetcher wrote, and each one's real pose ---------------------
    man, poses = [], {}
    want_photos = level in ("photo", "joinery", "material") and photos is not None
    if want_photos:
        man = read_manifest(photos, log=log)
    facade.CTX.use(ROOT=Path(root) if root else Path.cwd(), Z_GROUND=zg, CAM_H=CAM_H, W=W,
                   SOLID=SOLID, occlusion_map=camera.occlusion_map, PHOTOS=photos,
                   MASKS=masks, MANIFEST=man, CLICKS=clicks)
    if man and level in ("photo", "joinery"):
        indoor = set(indoor)
        for i, p in enumerate(man):
            if i in indoor or p.get("unused"): continue
            p["_i"] = i; poses[i] = camera.refine_pose(p, V, T)
            log(f"  photo #{i:2d} {p['date']} {p['dist_m']:5.1f} m: silhouette agreement "
                f"{poses[i]['score0']:.3f} → {poses[i]['score']:.3f}  (dx, dy, dz, yaw, pitch) {poses[i]['d']}")
        if out: _write_poses(poses, out / "poses.json")

    # ---- walls -----------------------------------------------------------------------------
    ordered = sorted(F, key=lambda f: -f["area"])
    street = [k for k, Fk in enumerate(ordered)
              if Fk["area"] >= 1.0 and Fk["t0"] + Fk["p0"][2] < V[:, 2].min() + 3.0]
    done = {}
    street_set = set(street)
    if poses:
        _probe(ordered, street_set, poses, man, camera, wl, log)
        for k in street:
            done[k] = wl.composite_facade(ordered[k], poses, man, P, rgb,
                                          name=f"facade{k:02d}", compose=compose)
    base_mat = wl.compose_walls(done, log=log) if (done and compose) else None
    wall_base = base_mat["colour"] if base_mat else np.array([150, 150, 150], np.uint8)
    log(f"  wall base colour (the building's own wall colour): {wall_base.tolist()}")

    qa_path = None
    if qa and out and done:
        qa_path = ftexture.qa_sheet([(f"facade{k:02d}", info, tex)
                                     for k, (tex, info) in sorted(done.items())
                                     if ordered[k]["area"] >= 15], out / "wall_qa.png")
        log(f"  wall QA sheet: {qa_path}")

    prims, sheet_tiles, wall_list = [], [], []
    for k, Fk in enumerate(ordered):
        tex, info = done.get(k, (None, None))
        if info is None or tex is None:
            w, h = wl.tex_size(Fk)
            tex = fcompose.compose(w, h, base_mat, [])[0] if base_mat else np.tile(wall_base, (h, w, 1))
        # every wall is recorded, composed or not — the caller wants the building's plan, not
        # only the walls a photo happened to reach (Kaveh, 2026-09-06: a run with no photos
        # reported 0 facades while building 170 of them).
        if True:
            wall_list.append(dict(
                id=f"facade{k:02d}", w=round(Fk["s1"] - Fk["s0"], 1), h=round(Fk["t1"] - Fk["t0"], 1),
                facing=round(math.degrees(math.atan2(Fk["n"][1], Fk["n"][0]))),
                corners=geometry.facade_corners(Fk).tolist(),
                photos=(info or {}).get("photos", []), quads=(info or {}).get("quads", {}),
                method=(info or {}).get("method", "flat"), filled=(info or {}).get("filled", 0.0),
                **({"elements": felements.to_json(info["elements"]), "material": info["material"]}
                   if info and info.get("layers") is not None else {})))
        # THE WALL IS ITS OWN RECTANGLE, NOT THE SOLID'S TRIANGLES. A roofer facade is a set of
        # coplanar triangles that do NOT tile the plane: railspur facade00 has 11 vertices but only
        # 7 triangles, in two disconnected patches, so reusing them left a diagonal row of
        # triangular HOLES through the wall (Kaveh, 2026-09-06 — he sent the picture). The facade's
        # (s0, s1, t0, t1) box is the wall; two triangles cover it exactly and the texture, whose uv
        # is defined on that same box, lands right.
        c = geometry.facade_corners(Fk)                      # TL, TR, BR, BL in UTM
        vpos = np.asarray(c, float)
        uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        idx = np.array([[0, 1, 2], [0, 2, 3]])
        nm = f"facade{k:02d}" + ("_photos" + "-".join(map(str, info["photos"])) if info else "_flat")
        rp = []
        if level == "joinery" and info and info.get("elements"):
            origin_wall = Fk["p0"] + Fk["u"] * Fk["s0"] + Fk["v"] * Fk["t0"]
            rp = frelief.relief_prims(Fk, info["elements"], origin_wall, tex, name=f"facade{k:02d}")
        if rp: prims += rp
        else: prims.append(dict(pos=vpos, uv=uv, idx=idx, tex=tex, name=nm))
        sheet_tiles.append((nm, Fk, tex))
        log(f"  {nm:<26} {Fk['s1']-Fk['s0']:5.1f} × {Fk['t1']-Fk['t0']:4.1f} m  "
            f"facing {math.degrees(math.atan2(Fk['n'][1], Fk['n'][0])):6.1f}°  "
            + (f"photos {info['photos']} filled {info['filled']:.0%}" if info else "flat colour")
            + (f"  composed: {len(info['elements'])} elements, "
               f"relief {[p['name'].split('_', 1)[1] for p in rp]}"
               if info and info.get("layers") is not None else ""))

    # ---- roof, skylights, floor -------------------------------------------------------------
    boxes = geometry.skylight_boxes(P, rgb, ring)
    roof = np.flatnonzero(n[:, 2] > 0.3); vid = np.unique(T[roof])
    remap = np.full(len(V), -1); remap[vid] = np.arange(len(vid))
    bbox = (V[:, 0].min() - 1, V[:, 1].min() - 1, V[:, 0].max() + 1, V[:, 1].max() + 1)
    rtex, roof_base = wl.roof_texture(P, rgb, bbox, ring, [b["corners"] for b in boxes])
    log(f"  roof base colour (median of flat roof returns): {roof_base.round(0).astype(int).tolist()}")
    uv = np.column_stack([(V[vid, 0] - bbox[0]) / (bbox[2] - bbox[0]),
                          (bbox[3] - V[vid, 1]) / (bbox[3] - bbox[1])])
    rvid = np.unique(T[roof]); rremap = np.full(len(V), -1); rremap[rvid] = np.arange(len(rvid))
    ruv = np.column_stack([(V[rvid, 0] - bbox[0]) / (bbox[2] - bbox[0]), (bbox[3] - V[rvid, 1]) / (bbox[3] - bbox[1])])
    prims.append(dict(pos=V[rvid], uv=ruv, idx=rremap[T[roof]], tex=rtex, name="roof_lidar"))
    for j, b in enumerate(boxes): prims.append(geometry.box_prim(b, f"skylight{j:02d}"))
    prims.append(geometry.floor_prim(ring, float(V[:, 2].min()) - 0.02))
    log(f"  floor: footprint slab at z = {V[:, 2].min():.2f} m")
    log(f"  skylights: {len(boxes)} monitors as boxes, "
        f"heights {sorted(round(b['z1']-b['z0'], 1) for b in boxes)}")

    # ---- write -------------------------------------------------------------------------------
    origin = (centre[0], centre[1], float(V[:, 2].min()))
    meta = dict(name=name, origin_utm=list(origin), crs="EPSG:26910", up="Y", level=level,
                prims=[p["name"] for p in prims], facades=wall_list,
                z_ground=zg, photos=len(man), poses=len(poses))
    b = Building(glb=None, walls=wall_list, elements={w["id"]: w.get("elements", []) for w in wall_list},
                 qa=qa_path, meta=meta, prims=prims, skylights=boxes, origin=origin,
                 textures=sheet_tiles, roof_tex=rtex, solid=SOLID, facades=F, points=d,
                 composed=done, ordered=ordered)
    if out and write:
        b.glb = out / "building.glb"
        meta["bytes"] = glb.write_glb(b.glb, prims, origin, name=name)
        (out / "viewer.html").write_text(glb.viewer_html(b.glb, name))
        json.dump(b.elements, open(out / "elements.json", "w"), indent=1)
        json.dump([dict(corners=x["corners"].tolist(), z0=x["z0"], z1=x["z1"], area=x["area"])
                   for x in boxes], open(out / "skylights.json", "w"), indent=1)
        json.dump(meta, open(out / "meta.json", "w"), indent=1)
        log(f"wrote {b.glb} ({meta['bytes']/1e6:.1f} MB), viewer.html, elements.json, meta.json")
    return b


def fallback_solid(ring, P, z_ground):
    """No roofer solid: the surveyed outline extruded to the measured top, its own roof planes on it.

    `objects.building_model` already is this — prism when the returns cannot say more, the
    plane-fitted roof when they can — so this only unpacks its triangle soup into the
    indexed (V, T) the facade code wants.
    """
    from . import objects
    ztop = float(np.percentile(P[:, 2], 98)) if len(P) else z_ground + 6.0
    parts = objects.building_model(np.asarray(ring), z_ground, ztop,
                                   x=P[:, 0], y=P[:, 1], z=P[:, 2])
    if not parts:
        from .scene import footprint_prism
        soup, _ = footprint_prism(np.asarray(ring), z_ground, max(ztop, z_ground + 3.0))
    else:
        soup = np.asarray(parts[0][0], float)
    return weld(soup.reshape(-1, 3))


def weld(soup, tol=4):
    """A triangle soup (3n, 3) -> indexed (V, T), vertices merged at `tol` decimals.

    Winding is preserved: a triangle keeps its own corner ORDER, which is what makes its
    normal point out of the building rather than into it.
    """
    soup = np.asarray(soup, float)
    _, first, inv = np.unique(np.round(soup, tol), axis=0, return_index=True, return_inverse=True)
    keep_v = np.sort(first)                              # keep the soup's own vertex order
    V = soup[keep_v]
    remap = np.empty(len(first), int)                    # unique()'s id -> row in V
    remap[inv[keep_v]] = np.arange(len(keep_v))
    T = remap[inv.ravel()].reshape(-1, 3)
    ok = (T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])
    return V, T[ok]


def read_manifest(photos, log=print):
    """The photo list a fetcher wrote: manifest_all.json if the user merged sources, else the sweep's."""
    photos = Path(photos)
    mf = photos / "manifest_all.json"
    if not mf.exists(): mf = photos / "streetview/manifest.json"
    if not mf.exists(): mf = photos / "manifest.json"
    if not mf.exists():
        log(f"  no manifest under {photos} — walls get the building's material only")
        return []
    man = [p for p in json.load(open(mf)) if "file" in p]
    log(f"  {len(man)} photos from {mf.name}")
    return man


def _points(footprint, points, store, labels, pts_mod):
    """Points from a prepared dict, a cached npz, or a fresh read of the store."""
    if points is not None:
        return points if isinstance(points, dict) else pts_mod.load_npz(points)
    if isinstance(footprint, (str, Path)) and str(footprint).endswith(".npz"):
        return pts_mod.load_npz(footprint)
    if store is None:
        raise ValueError("building3d needs either points=… or store=… to find the building's returns")
    return pts_mod.building_points(footprint, store, labels=labels)


def _probe(ordered, street, poses, man, camera, wl, log):
    """The best-covered wall's upper half is the building's wall colour — the reference the rest match."""
    probe = []
    for k, Fk in enumerate(ordered[:12]):
        if k not in street: continue
        r = camera.rank_photos(Fk, poses, man)
        if not r: continue
        t, v = wl.warp_facade(Fk, camera.chosen_pose(poses[r[0][1]]), r[0][1], man)
        top = np.zeros_like(v); top[: int(v.shape[0] * 0.4)] = True
        m = wl.lab_median(t, v & top)
        if m is not None: probe.append((float((v & top).mean()), m))
    if probe:
        log(f"  wall reference colour (Lab) from the best-covered wall: {max(probe)[1].round(0).tolist()}")


def _write_poses(poses, path):
    json.dump({i: dict(cam=P["cam"].tolist(), heading=P["heading"], pitch=P["pitch"], fov=P["fov"],
                       score=P["score"],
                       meta=dict(P["meta"], cam=P["meta"]["cam"].tolist()) if "meta" in P else None)
               for i, P in poses.items()}, open(path, "w"), indent=1)
