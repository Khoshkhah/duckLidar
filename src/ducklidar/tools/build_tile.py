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


def _build(args):
    """One `building3d`, in a worker; returns (id, level, error or None)."""
    bid, ring, tile, bdir, odir, level = args
    import numpy as np
    import ducklidar as dl
    from ducklidar.building.points import building_points
    photos = bdir / "photos"
    lvl = level if (photos / "streetview" / "manifest.json").exists() else "material"
    try:
        d = building_points(ring, tile)
        np.savez(bdir / "points.npz", **d)          # the dashboard shows these next to the model
        dl.building3d(ring, points=d, photos=photos if lvl != "material" else None,
                      solid=next(bdir.glob("*.city.json"), None), out=odir, level=lvl, name=bid,
                      log=lambda *_: None)
        return bid, lvl, None
    except Exception as e:                       # one bad footprint must not stop the other 247
        return bid, lvl, f"{type(e).__name__}: {e}"


def dashboard(root):
    """Refresh `out/dashboard.html` over everything built so far."""
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
    a = ap.parse_args(argv)

    root = Path(a.root)
    ext = tile_extent(a.tile)
    fps = footprints(a.footprints, ext)
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
        todo.append((bid, ring, bdir, root / "out" / bid))

    if a.env:
        t0 = time.time()
        need = [(b, a.env, a.footprints) for _, _, b, _ in todo if not (b / "photos" / "streetview" / "manifest.json").exists()]
        print(f"fetch: {len(need)} buildings without photos", flush=True)
        with ProcessPoolExecutor(a.jobs) as ex:
            for k, (bid, rc, tail) in enumerate(ex.map(_fetch, need), 1):
                print(f"  {bid}  {'ok' if rc == 0 else 'FAILED'}  {' '.join(tail)}  ({k}/{len(need)}, {time.time() - t0:.0f} s)", flush=True)

    def job(bid, ring, bdir, odir):
        """The build this building's inputs allow now, or None when out/ already has it."""
        want = a.level if (bdir / "photos" / "streetview" / "manifest.json").exists() else "material"
        if not a.force and (odir / "building.glb").exists() and built_level(odir) == want:
            return None
        return (bid, ring, a.tile, bdir, odir, want)

    t0 = time.time()
    done = {"built": 0, "failed": []}

    def report(bid, lvl, err):
        if err: done["failed"].append((bid, err)); print(f"  {bid}  FAILED {err}", flush=True)
        else: done["built"] += 1; print(f"  {bid}  {lvl:8s} built  ({time.time() - t0:.0f} s)", flush=True)

    with ProcessPoolExecutor(a.jobs) as ex:
        if a.segment:
            by_folder = {str(b / "photos"): (bid, r, b, o) for bid, r, b, o in todo
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
    print(f"built {done['built']}, kept {kept}, failed {len(done['failed'])} in {time.time() - t0:.0f} s", flush=True)
    for bid, err in done["failed"]:
        print(f"  {bid}: {err}")
    return 1 if done["failed"] else 0

if __name__ == "__main__":
    sys.exit(main())
