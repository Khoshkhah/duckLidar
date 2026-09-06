"""Where the test tile and the output directory are, independent of your shell's cwd.

The tile is a hand-downloaded 800 MB file that belongs to another project, so it cannot live in
this repo — but that is no reason for the tools to only work from one directory. Set
``LIDAR_TILE`` to point them at a different file.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# LIDAR_OUT redirects every tool's artifacts — one folder per tile
# when processing neighbours (2026-08-02), default the classic out/
OUTDIR = Path(os.environ.get("LIDAR_OUT", REPO / "out"))

TILE = Path(os.environ.get(
    "LIDAR_TILE", Path.home() / "projects/shadowCity/data/lidar/490000_5457000.zip"))

# the study window: (cx, cy, half) in EPSG:26910. Every tool reads this so
# results stay comparable; LIDAR_BOX="cx cy half" rescopes the whole
# toolchain at once (2026-08-01, the shadowCity2 extension — the box was
# hardcoded in 41 tools before this). A fourth value makes the box a
# rectangle: "cx cy half_x half_y" (2026-08-02, so a border tile's window
# is tile∩area+60 m in each axis instead of a square that overreads).
_env_box = os.environ.get("LIDAR_BOX")
STUDY_BOX = (tuple(float(v) for v in _env_box.split()) if _env_box
             else (490289.0, 5457557.0, 250.0))

# neighbour tiles (LIDAR_NEIGHBOURS, colon-separated zips/parquets) fill the
# borders: check() hands dl.read the whole list, so a box leaning over this
# tile's edge reads the neighbours' points instead of a one-sided cut
# (2026-08-02 — border neighbourhoods were half-empty before this).
NEIGHBOURS = [p for p in os.environ.get("LIDAR_NEIGHBOURS", "").split(":") if p]


def out(name, argv=None):
    """Resolve the output path: an explicit argument if given, else `out/<name>` in the repo."""
    if argv and len(argv) > 1:
        return Path(argv[1]).expanduser().resolve()
    OUTDIR.mkdir(exist_ok=True)
    return OUTDIR / name


def check():
    """Fail early and usefully rather than deep inside a zipfile open."""
    if not TILE.exists():
        raise SystemExit(f"tile not found: {TILE}\n"
                         f"set LIDAR_TILE=/path/to/tile.zip (or .las/.laz/.copc.laz)")
    if NEIGHBOURS:
        missing = [p for p in NEIGHBOURS if not Path(p).exists()]
        if missing:
            raise SystemExit(f"LIDAR_NEIGHBOURS not found: {missing}")
        return [str(TILE)] + [p for p in NEIGHBOURS if Path(p) != TILE]
    return str(TILE)


def window_pids(bbox):
    """Global pids aligned to ``dl.read(check(), bbox)`` order.

    Global pid = the source's offset in the **registry** + the point's pid
    within its file. The registry is the sorted sidecar list of the store —
    *not* the list this window happens to read. The two must be independent:
    a window that reads one tile and a window that reads all nine have to give
    the same global pid to the same return, or a join against a stage table
    silently matches the wrong rows. (Reading the old area over the full
    9-sidecar list also pulls in 101 extra points from neighbour-tile overlap,
    so the read list cannot be widened just to fix the offsets.)

    Falls back to cumulative offsets over the read list when the sources are
    not parquet sidecars of one store (zips, a COPC URL), which is the only
    case the old behaviour was correct for anyway.
    """
    import numpy as np
    import pyarrow.parquet as pq

    import ducklidar as dl

    srcs = check()
    srcs = srcs if isinstance(srcs, list) else [srcs]
    registry = sorted(Path(srcs[0]).parent.glob("*.parquet"))
    base, seen = {}, 0
    for p in registry:
        base[str(p.resolve())] = seen
        seen += pq.ParquetFile(str(p)).metadata.num_rows
    if not all(str(Path(s).resolve()) in base for s in srcs):
        base = None                       # not one store's sidecars — see docstring

    out, off = [], 0
    for s in srcs:
        p = dl.read(s, bbox, fields=("pid",))
        if "pid" not in p:
            raise SystemExit(f"{s} carries no pid — rebuild the sidecar with dl.to_parquet")
        out.append(p["pid"] + (base[str(Path(s).resolve())] if base else off))
        off += pq.ParquetFile(s).metadata.num_rows
    return np.concatenate(out)
