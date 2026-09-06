# Stage 0 — the store

**Purpose:** turn delivered survey files (LAS/LAZ inside zips, no index,
no random access) into a queryable columnar store.

## Task 0.1 — convert & spatially sort (`dl.to_parquet`)

**The Problem:** the delivered files allow no random access, and
unsorted, every row group spans the whole tile and nothing prunes — a
window query has to read everything.

**The Solution:** read a survey file once and sort its points into 50 m
cells (nearby in the city → nearby in the file): stream the LAS in
chunks → one in-RAM sort key (`cell_y << 24 | cell_x`) → argsort → write
row group by row group, so a sorted copy of the tile never exists whole
in memory. Output is Parquet with zstd, ~500 k rows per row group.
Parquet keeps min/max statistics per row group, so sorted, a window
query skips every row group that can't contain the window.

**The Result:** a queryable store — measured on a Vancouver tile: 0.9 s
instead of 9.7 s and 3.1 GB peak.

**Cost:** ~2 min and ~3 GB RAM per 42 M-point tile, once ever.
**Knobs:** `SORT_CELL_M` (50 m), `ROW_GROUP` (500 k), compression.

**Optimize?** Rarely worth it — it runs once. Candidates if it ever
matters: smaller sort cells (finer pruning, costlier key), per-column
codecs (delta for gps_time). COPC is the alternative format (octree in
the file) — rejected because Parquet gives column pruning and SQL for
free.

## Task 0.2 — assign `pid`

**The Problem:** all later results are tables keyed by point, and the
points are never copied again — so every point needs one permanent name.

**The Solution:** every point gets its row number in its file, assigned
after the sort — permanent. Global pid = (file's offset in the registry:
sorted sidecar paths) + row.

**The Result:** a stable key for every later table. The one fragility:
the registry order must never change once derived tables exist.

**Optimize?** Nothing to optimize; one `argsort` inversion at build.

## Task 0.3 — window reads (`dl.read`)

**The Problem:** consumers ask for a bbox, but a window near a file
boundary spans files.

**The Solution:** bbox → numpy columns, from one file or many at once (a
boundary window reads both sides); non-touching files cost one
header/statistics check. Noise classes dropped by default.

**The Result:** numpy columns for any window, whatever the file layout.

**Optimize?** Column pruning is free — ask for fewer `fields`. The
multi-source loop is serial per file; parallelizing per-file reads would
help only for windows spanning many files (rare — not worth it yet).

## Task 0.4 — census (`dl.describe`, `dl.census`)

**The Problem:** planning against what a survey is assumed to carry,
instead of what it actually carries.

**The Solution:** inventory the survey — classes, colour, returns
histogram — before anything is planned. Costs seconds.

**The Result:** the facts first. Not a compute task; a discipline.
