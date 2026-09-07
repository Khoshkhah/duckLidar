"""TOOL — every footprint inside a tile, built as a 3-D building: photos, masks, then the build.

    python -m ducklidar.tools.build_tile --tile tiles/490000_5457000.parquet \
        --footprints footprints/granville_island.duckdb --root . --env ../shadowCity2/.env --segment

Reads `buildings.building` from a duckOverture extract, keeps the footprints that lie WHOLLY
inside the tile (a footprint cut by the tile edge has half its returns missing — it is
skipped and counted, not built wrong), and runs the three stages of docs/data-layout.md
over `buildings/<id>/` and `out/<id>/`, each stage skipping what is already there:

    fetch     `--env KEY_FILE`: Street View around each footprint (billable, `fetch_streetview`),
              `--jobs` buildings at once
    segment   `--segment`: SAM 3 over every photo folder, ONE model load (`segment`, needs a GPU).
              Each building is built the moment its folder is segmented, and `out/dashboard.html`
              is refreshed after every finished building — so there is something to look at
              hours before the last photo is done.
    build     `dl.building3d` per footprint, `--jobs` at once; with photos -> `--level`, else material.
              A building already built at the level its inputs allow is kept, not rebuilt.

The id is `gers_` + the first 8 characters of the Overture id, as the pilot named them.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CRS = 26910


def footprints(db, extent, crs=CRS):
    """-> [(overture id, ring [[x, y], …] in `crs`)] for buildings wholly inside `extent`."""
    import duckdb
    x0, y0, x1, y1 = extent
    con = duckdb.connect(str(db), read_only=True)
    try:
        con.execute("LOAD spatial")
        rows = con.execute(
            "select id, ST_AsGeoJSON(g) from (select id,"
            "  ST_Transform(geometry, 'EPSG:4326', ?, always_xy := true) g from buildings.building)"
            " where ST_Within(g, ST_MakeEnvelope(?, ?, ?, ?)) order by id",
            [f"EPSG:{crs}", x0, y0, x1, y1]).fetchall()
    finally:
        con.close()
    return [(i, json.loads(gj)["coordinates"][0]) for i, gj in rows]


def neighbours_of(fps, reach=6.0):
    """-> {overture id: [rings of the other footprints within `reach` m]} — the plan must stop at them."""
    import shapely
    polys = [shapely.Polygon(r) for _, r in fps]
    tree = shapely.STRtree(polys)
    out = {}
    for k, (oid, ring) in enumerate(fps):
        near = tree.query(polys[k].buffer(reach), predicate="intersects")
        out[oid] = [fps[j][1] for j in near if j != k]
    return out


def tile_extent(tile):
    """(x0, y0, x1, y1) from the Parquet row-group statistics — no points are read."""
    import pyarrow.parquet as pq
    md = pq.read_metadata(tile)
    names = [md.row_group(0).column(i).path_in_schema for i in range(md.row_group(0).num_columns)]
    lo, hi = {}, {}
    for r in range(md.num_row_groups):
        for k in ("x", "y"):
            s = md.row_group(r).column(names.index(k)).statistics
            lo[k] = min(lo.get(k, s.min), s.min); hi[k] = max(hi.get(k, s.max), s.max)
    return lo["x"], lo["y"], hi["x"], hi["y"]


def _fetch(args):
    """One `fetch_streetview` run, as a subprocess so a network failure is one building's."""
    bdir, env, db = args
    r = subprocess.run([sys.executable, "-m", "ducklidar.tools.fetch_streetview", "--ring",
                        str(bdir / "footprint.json"), "--out", str(bdir / "photos"), "--env", env,
                        "--footprints", db],
                       capture_output=True, text=True)
    return bdir.name, r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1:]


def _check(args):
    """The survey's verdict on a building's bought photos: `unused` where a camera cannot see the wall."""
    bdir, tile = args
    import numpy as np
    from shapely.geometry import Polygon
    from ducklidar.building.points import building_points
    from ducklidar.tools.fetch_streetview import survey_check
    mf = bdir / "photos" / "streetview" / "manifest.json"
    man = json.loads(mf.read_text()); ring = json.loads((bdir / "footprint.json").read_text())["ring"]
    todo = [p for p in man if "file" in p and not p.get("survey")]
    if not todo: return bdir.name, 0, 0
    zg = building_points(ring, tile)["z_ground"]
    survey_check(Polygon(ring), todo, tile, zg)
    n_bad = 0
    for p in todo:
        if p["survey"] != "ok" and not p.get("unused"): p["unused"] = p["survey"]; n_bad += 1
    mf.write_text(json.dumps(man, indent=1))
    return bdir.name, len(todo), n_bad


def _build(args):
    """One `building3d`, in a worker; returns (id, level, error or None)."""
    bid, ring, tile, bdir, odir, level, nbs = args
    import numpy as np
    import ducklidar as dl
    from ducklidar.building.points import building_points
    photos = bdir / "photos"
    lvl = level if (photos / "streetview" / "manifest.json").exists() else "material"
    try:
        d = building_points(ring, tile)
        if len(d["z"]) < 20:                          # the survey shows no building here: no model, and no stale one either
            for f in odir.glob("*"): f.unlink()
            return bid, "none", None
        b = dl.building3d(ring, points=d, photos=photos if lvl != "material" else None,
                          solid=next(bdir.glob("*.city.json"), None), out=odir, level=lvl, name=bid,
                          log=lambda *_: None, neighbours=nbs)
        # the dashboard shows the returns the blue outline is traced from: within 1.5 m of the footprint
        # and outside every neighbour's — not the 3 m collar, whose returns beside a bigger neighbour
        # are the neighbour's, and not the footprint alone, which left the blue line in empty space
        import shapely
        keep = shapely.contains_xy(shapely.Polygon(np.asarray(ring, float)).buffer(1.5), d["x"], d["y"])
        for nb in nbs: keep &= ~shapely.contains_xy(shapely.Polygon(np.asarray(nb, float)), d["x"], d["y"])
        d = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v) for k, v in d.items()}
        np.savez(bdir / "points.npz", **d)
        # both outlines, for the dashboard to set side by side: the map's, and the one the returns draw
        from ducklidar.objects import survey_outline, survey_footprint
        from shapely.geometry import Polygon
        pts0 = building_points(ring, tile)
        meta = json.loads((odir / "meta.json").read_text())
        meta["footprint"] = [list(map(float, q)) for q in ring]
        outl = survey_outline(ring, pts0["x"], pts0["y"], pts0["z"], neighbours=nbs)
        meta["survey_outline"] = [[list(map(float, q)) for q in r] for r in outl]
        prop = survey_footprint(ring, max(outl, key=lambda r: Polygon(r).area)) if outl else None
        meta["survey_footprint"] = [list(map(float, q)) for q in prop] if prop is not None else []
        A, B = Polygon(ring), (Polygon(prop) if prop is not None else None)
        meta["footprint_iou"] = round(A.intersection(B).area / A.union(B).area, 3) if B is not None and B.is_valid else None
        meta["footprint_area"] = round(A.area, 1); meta["survey_footprint_area"] = round(B.area, 1) if B is not None else None
        (odir / "meta.json").write_text(json.dumps(meta, indent=1))
        return bid, lvl, None
    except Exception as e:                       # one bad footprint must not stop the other 247
        return bid, lvl, f"{type(e).__name__}: {e}"


def dashboard(root):
    """Refresh `out/dashboard.html` over everything built so far."""
    if getattr(dashboard, "off", False): return
    subprocess.run([sys.executable, "-m", "ducklidar.tools.dashboard", "--root", str(root),
                    "--out", str(root / "out" / "dashboard.html")], capture_output=True)


def built_level(odir):
    """The level `out/<id>/meta.json` says it was built at, or None."""
    try:
        return json.loads((odir / "meta.json").read_text()).get("level")
    except (OSError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", required=True, help="the point store (a Parquet sidecar)")
    ap.add_argument("--footprints", required=True, help="a duckOverture extract with buildings.building")
    ap.add_argument("--root", default=".", help="the data root (holds buildings/ and out/)")
    ap.add_argument("--env", help="file with a GOOGLE_STREETVIEW_KEY= line: fetch photos for buildings without any")
    ap.add_argument("--segment", action="store_true", help="run SAM 3 over every photo folder (GPU)")
    ap.add_argument("--level", default="joinery", help="cap for buildings that have photos (default joinery)")
    ap.add_argument("--jobs", type=int, default=8, help="parallel fetches / builds (default 8)")
    ap.add_argument("--only", nargs="*", default=(), help="building ids to build (default: all)")
    ap.add_argument("--force", action="store_true", help="rebuild even where out/<id> is already at that level")
    ap.add_argument("--no-dashboard", action="store_true", help="do not refresh out/dashboard.html (a page of 248 buildings is 400 MB)")
    a = ap.parse_args(argv)

    root = Path(a.root)
    dashboard.off = a.no_dashboard
    ext = tile_extent(a.tile)
    fps = footprints(a.footprints, ext)
    nbs = neighbours_of(fps)                       # from ALL footprints in the tile, before --only narrows the list
    if a.only: fps = [(i, r) for i, r in fps if "gers_" + i[:8] in set(a.only)]
    print(f"{len(fps)} footprints wholly inside {a.tile} ({ext[0]:.0f},{ext[1]:.0f})..({ext[2]:.0f},{ext[3]:.0f})")
    todo = []
    for oid, ring in fps:
        bid = "gers_" + oid[:8]
        bdir = root / "buildings" / bid
        bdir.mkdir(parents=True, exist_ok=True)
        fp = bdir / "footprint.json"
        if not fp.exists():
            fp.write_text(json.dumps({"id": oid, "source": "overture", "crs": f"EPSG:{CRS}", "ring": ring}, indent=1))
        todo.append((bid, ring, bdir, root / "out" / bid, nbs.get(oid, [])))

    if a.env:
        t0 = time.time()
        need = [(b, a.env, a.footprints) for _, _, b, _, _ in todo if not (b / "photos" / "streetview" / "manifest.json").exists()]
        print(f"fetch: {len(need)} buildings without photos", flush=True)
        with ProcessPoolExecutor(a.jobs) as ex:
            for k, (bid, rc, tail) in enumerate(ex.map(_fetch, need), 1):
                print(f"  {bid}  {'ok' if rc == 0 else 'FAILED'}  {' '.join(tail)}  ({k}/{len(need)}, {time.time() - t0:.0f} s)", flush=True)

    def job(bid, ring, bdir, odir, nb):
        """The build this building's inputs allow now, or None when out/ already has it."""
        want = a.level if (bdir / "photos" / "streetview" / "manifest.json").exists() else "material"
        if not a.force and (odir / "building.glb").exists() and built_level(odir) == want:
            return None
        return (bid, ring, a.tile, bdir, odir, want, nb)

    # THE SURVEY CHECKS EVERY PHOTO — the bought ones too: a camera on the bridge deck, or one
    # with a crown between it and the wall, is flagged unused, and segment and build skip it
    have = [(b, a.tile) for _, _, b, _, _ in todo if (b / "photos" / "streetview" / "manifest.json").exists()]
    if have:
        t0 = time.time(); checked = flagged = 0
        with ProcessPoolExecutor(a.jobs) as ex:
            for bid, n, bad in ex.map(_check, have):
                checked += n; flagged += bad
        print(f"survey check: {checked} photos checked, {flagged} flagged unused ({time.time() - t0:.0f} s)", flush=True)

    t0 = time.time()
    done = {"built": 0, "failed": []}

    def report(bid, lvl, err):
        if err: done["failed"].append((bid, err)); print(f"  {bid}  FAILED {err}", flush=True)
        elif lvl == "none": done["skipped"] = done.get("skipped", 0) + 1; print(f"  {bid}  no building returns — skipped", flush=True)
        else: done["built"] += 1; print(f"  {bid}  {lvl:8s} built  ({time.time() - t0:.0f} s)", flush=True)

    with ProcessPoolExecutor(a.jobs) as ex:
        if a.segment:
            by_folder = {str(b / "photos"): (bid, r, b, o, nb) for bid, r, b, o, nb in todo
                         if (b / "photos" / "streetview" / "manifest.json").exists()}
            print(f"segment: {len(by_folder)} photo folders (already-segmented photos are skipped)", flush=True)
            seg = subprocess.Popen([sys.executable, "-m", "ducklidar.tools.segment", "--photos", *by_folder],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   env={**os.environ, "PYTHONUNBUFFERED": "1"})
            pending = {}
            for line in seg.stdout:
                print(line, end="", flush=True)
                if line.startswith("done "):
                    j = job(*by_folder[line[5:].strip()])
                    if j: pending[ex.submit(_build, j)] = j[0]
                ready = [f for f in pending if f.done()]
                for f in ready:                                  # harvest, then refresh the page once
                    pending.pop(f); report(*f.result())
                if ready: dashboard(root)
            for f in pending:
                report(*f.result())
            if pending: dashboard(root)
            seg.wait()

        jobs = [j for j in (job(*t) for t in todo) if j]
        kept = len(todo) - len(jobs) - done["built"] - len(done["failed"])
        print(f"build: {len(jobs)} buildings still to build, {kept} kept", flush=True)
        for bid, lvl, err in ex.map(_build, jobs):
            report(bid, lvl, err)
        if jobs: dashboard(root)
    print(f"built {done['built']}, kept {kept}, skipped {done.get('skipped', 0)} (no building returns), failed {len(done['failed'])} in {time.time() - t0:.0f} s", flush=True)
    for bid, err in done["failed"]:
        print(f"  {bid}: {err}")
    return 1 if done["failed"] else 0

if __name__ == "__main__":
    sys.exit(main())
