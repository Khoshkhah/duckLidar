# Stage 5 — the tables, and the process that writes each one

Every stage-5 operator has been built and measured against the oracle
(`docs/design/stage5-labeling.md`, `stage5-cost.md`). None had written its
table: the assembled run was one process, and a monolith's peak is the **sum**
of everything still alive in it. Measured on the old area, one process:

    [rss dijkstra]  16.7 GB     the conviction, still resident
    [rss rescue]    17.5 GB     still resident
    [rss mesh: csr] 22.4 GB     +5 GB on top -> the 23 GB box is exhausted

**Task per process, table per task** makes the peak the *largest single task*
instead. Each writer reads its inputs from Parquet, writes one table, prints
its peak RSS, asserts a declared budget, and exits; memory returns to baseline
between tasks. The runner is `shadowCity2/tools/stage5_tables.py`, one
subcommand per task.

Two conventions hold across every table here:

- **`cell` is `dl.cell_of`** — `cy * nx + cx`, **south-up**, in the WORK frame
  of `data/stage5/cells.parquet` (the area box + a 60 m margin, 1 m pixels).
  The box, the pixel size and the area name travel in each file's Parquet
  **schema metadata**, because a cell id means nothing without the box it was
  cut from. The oracle's grids are north-up: `[::-1]` before comparing.
- **`pid` is the GLOBAL point id** (`int64`), the same key
  `data/stage5/labels.parquet` and `forest.parquet` already use — never a
  row number.

## The tables

### `water.parquet` — task 5.4  (body grid IoU **0.9546**, precision 0.9961)

| column | type | meaning |
|---|---|---|
| `cell` | int64 | WORK cell, south-up |
| `body` | int32 | flood-component id, always ≥ 1 |

One row per **wet** cell. Dry is absence, not `body = 0`: the flooded region is
a small part of the grid, and a `0` row per dry cell would be 1.25 M rows of
nothing.

Rows are **not** restricted to `cells.parquet`'s occupied cells. Open water
absorbs the beam, so most of a body is returnless — occupied-only accounting
undercounts a body 3× (measured: recall 0.35 with a correct flood). The row set
is therefore every cell of `flood_bodies`' full grid with a label, which is also
the object the 0.9546 was measured on.

### `water_levels.parquet` — task 5.4, the level side

| column | type | meaning |
|---|---|---|
| `seed_body` | int32 | **seed**-component id (≥ 1) |
| `level` | float64 | median z of that body's low returns in its sparse-dark cells; null if it has none |
| `n_water` | int64 | vendor class-9 returns over the body's cells — the credibility gate (≥ 500) |

Schema metadata carries `flood_level`, the single value actually flooded with:
`median(level where n_water >= 500)`.

**Changed from the proposal, twice.**

1. `point_source_id` is **dropped**. `flood_bodies` takes `{body: level}` and
   collapses it to one median before it propagates — there is no per-flight-line
   level anywhere in the measured construction, so a `(body × flight line)`
   table would be data no operator reads. The column exists in the store, so
   tide/flight-line conditioning is a one-column upgrade later — but it is a new
   capability and goes to Kaveh as a question first, not in as a schema.
2. `body` is renamed **`seed_body`**, because it is a different id space from
   `water.parquet`'s `body`. `flood_bodies` labels the *seed* components, keys
   `levels` by those ids, then floods and **re-labels** the flooded region; the
   ids that come out are not the ids that went in. Two columns called `body`
   that don't join is a bug waiting in a query.

### `corridors.parquet` — task 5.5  (IoU **0.9976**, precision 1.0000)

| column | type | meaning |
|---|---|---|
| `cell` | int64 | WORK cell, south-up, inside an OSM bridge corridor |

One row per corridor cell; absence is "not a corridor".

**Changed from the proposal:** `corridor bool` is gone (every row would be
`true`), and so is `width f32`. `corridor_cells` buffers each way at *its own*
radius and ORs the results into one mask — a per-cell width is not a thing it
computes, and inventing one (max radius of a covering way?) is a new capability,
not a column. Like `water.parquet`, rows are grid cells, not occupied cells: the
distance transform paints corridor over returnless deck shadow, and that is the
mask that measured 0.9976.

**Written, and what it scores in its own frame.** 0.9976 / precision 1.0000 is
the operator run in the ORACLE's frame (the tile, no margin). The table is
written in the WORK frame, and its tile slice scores **IoU 0.9972, precision
0.9997, recall 0.9975** — 42,236 rows, 34,671 cells inside the tile against the
oracle's 34,746. The whole of that gap is one artifact and cloth noise:

* **83 of the 86 missing cells are the oracle's own NW-corner phantom**, the
  quarter-disc `distance_transform_edt` paints when a width class has no ways in
  the box (`corridor_cells`' comment). In the WORK frame that disc lands in the
  WORK corner, 60 m outside the tile, so it is not there to agree with. Discount
  it and the table reads IoU **0.9992**, 3 missing cells and 11 extra out of
  34,746.
* CSF is not bit-reproducible: two identical dev runs gave 42,235 and 42,236
  cells (IoU 0.9972 and 0.9974). Re-running the tile-frame comparison against
  this table's cloth gives 0.9975 / 0.9999 — the recorded 0.9976 / 1.0000 to
  within one cell.

### `claims.parquet` — task 5.6

| column | type | meaning |
|---|---|---|
| `pid` | int64 | global point id |
| `claims` | uint16 | one bit per layer, bit *i* = `dl.DECIDE[i]` |

One row per point of the area, including points with **no** claim
(`claims = 0`) — they are the decision's rule-3 residue, and a
complete point-indexed column joins to `labels.parquet` row for row.

A bitmask, not ten columns: a point may hold several claims, and resolving them
is the decision's job, not the schema's.

    bit  0 ground   1 veg_low   2 veg_high   3 building   4 water
         5 small    6 floating  7 bridge     8 on_bridge  9 glazing
        10 placed

**Changed from the proposal:** bit 10, `placed`. It is the 11th entry of
`rules/layers.py:41 LAYERS`, deliberately *excluded* from `DECIDE`
(`layers.py:614` — with specificity it steals one-storey rims from building,
IoU 0.830 → 0.766; the eave problem) but computed, measured and drawable. `uint16`
has the room, dropping it would discard a measured layer, and bits 0–9 remain
exactly `dl.DECIDE`, so a consumer building the decision's input slices the low
ten bits and never sees it.

**Open question for Kaveh — the containment cycle.** Bit 9 (`glazing`) and
building's *adopted* points are formed in the assembly block
(`layers.py:313-360`), i.e. **after** the outlines, which are hulled from
building's claims, which are 5.6's output. The cycle is real in the lab's own
code: 5.6 → outlines → two amended point-level bits. So `claims.parquet` as
written by this task holds the claims **as of the end of the per-class rules**:
bit 9 is 0 and building carries no adoption. The amendment is two predicates over
`outlines.parquet` plus point features —

    adopt   = (no claim) & inside an outline & hag > TALL          -> building
    glazing = penetrable & inside an outline & |z - roof| < 2.5 & scatter < MAX

— and it has no home among these five subcommands. It belongs to the `decide`
task (not in this runner), or `claims` grows a second pass. **The table is
written at the snapshot above** — the ruling is still open, and changes only
whether a later pass amends two bits; it does not change what is here.

**Written, and what it scores.** 42,707,336 rows (one per return of the tile,
216,226 of them with `claims = 0`), 60.6 MB, 48 s, peak RSS 11.3 / 11.5 GB on
two runs.

The task runs the ORACLE'S OWN SOURCE: `rules/layers.py` is read and patched by
two anchored substitutions — `pid` added to the run's own `dl.read` (the rules
never look at it; recovering the join key with a second read of 42.7 M points
costs 1.4 GB), and a hook after the last line of the LOCAL-rule loop that packs
the eleven layers and raises. Nothing past that line runs: no assembly, no
conviction, no decision, no 22.4 GB mesh. `B` is the **tile**, the frame every
`claim_*` grid in `layers.npz` was measured in — the corridors task paid for
that lesson.

Per-cell claim grids against `layers.npz`'s `claim_<class>` (both north-up, no
flip — that flip belongs to the south-up cell tables):

| layer | IoU | mine | oracle | |
|---|---|---|---|---|
| `ground` | **1.0000** | 526,823 | 526,823 | exact |
| `veg_low` | **1.0000** | 52,965 | 52,965 | exact |
| `water` | **1.0000** | 60,668 | 60,668 | exact |
| `small` | **1.0000** | 171,480 | 171,480 | exact |
| `floating` | **1.0000** | 58,289 | 58,289 | exact |
| `on_bridge` | **1.0000** | 13,890 | 13,890 | exact |
| `placed` | **1.0000** | 46,913 | 46,913 | exact |
| `bridge` | 0.9947 | 40,698 | 40,914 | precision **1.0000** — a strict subset; the 216 missing cells are `_wire` (`layers.py:578`, corridor stranding), added later |
| `veg_high` | 0.9655 | 294,649 | 284,527 | recall 0.9999 — the oracle's set inside mine; the 10,124 extra are `_convicted` / `_rooffix`, removed later |
| `building` | 0.9225 | 337,509 | 318,750 | both directions, as the amendment list requires: `adopt_a`, `_gl_only`, `_rooffix` add, `_crownsplit`, `_crowncomp` remove |
| `glazing` | 0 | 0 | 85,104 | 0 by construction — the assembly block forms it |

Seven of the eleven layers reproduce the oracle **cell for cell**, which pins
the whole shared upstream (read, cloth, DEM, covariance, echo ratio, water body,
the band/green vote, the OSM corridor, the extension loop) to the oracle at
zero error. The other four are exactly the layers `layers.py` amends *after*
the snapshot, and each differs only in that direction. `layers.npz`'s
`claim_*` grids are the FINAL `L`, so those four have no contemporaneous
oracle to match; the snapshot is the design's, not a miss.

CSF jitter is visible and harmless: `placed` claims 746,959 points here against
746,972 in the eave sweep's variant A — 13 points, and the same 46,913 cells.

### `outlines.parquet` — task 5.7

| column | type | meaning |
|---|---|---|
| `cell` | int64 | WORK cell, south-up |
| `outline` | int32 | outline component id, always ≥ 1 |
| `roof_z` | float32 | that outline's mean roof height (mean z of its building claims) |

One row per cell inside an outline; absence is outside. The outline is the
closed, hole-filled hull of the building claims with the forbidden zones (deck
corridor, overhead) cut out — so it covers cells with no returns at all, which is
the entire point of hole-filling.

**Changed from the proposal:** `in_outline bool` is dropped — it is exactly
`outline > 0`, i.e. row presence. `roof_z` is added: it is a per-component
product of the same block (`roofz_a`, `layers.py:333`) that the glazing window
reads as its fallback, it costs 4 B per cell and compresses to nearly nothing,
and the alternative is a second file holding one float per component.

**Written on the old area**: 1,395 outlines over 375,365 cells (37.5 % of the
tile), 703 KB, 47 s, peak 11.4 GB. `layers.npz` saves no hull grid, so the
check is what the assembly alone produces: `claim_glazing` can only be born
inside an outline (`layers.py:356`) and is never widened afterwards, so
**100.00 % of the oracle's 85,104 glazing cells fall inside these outlines,
0 outside** — a hull one cell too small would show here. Containment of the
oracle's final `claim_building` is 87.11 %, and the deficit is the hull's own
definition (`bc_a &= ~forbid_cell`): 96.81 % of the 41,093 cells outside stand
in this run's `forbid_cell`, taking containment in `outlines ∪ forbid_cell` to
**99.59 %**; the residual 1,311 cells are claims added *after* assembly (the
kNN conviction), which no hull built before it can contain. Post-assembly, mine
vs the oracle's final grids: building recall 0.9992 (342,133 vs 318,750 — the
later `_crownsplit`/`_crowncomp` remove), glazing recall **1.0000** (146,174 vs
85,104 — the support-graph acquittal removes the rest). Both frames north-up,
no flip; the written table's WORK cells were mapped back independently and
reproduce the same 375,365 cells and the same two numbers.

### `forest.parquet` — task 5.8  (refine_veg **0.9919**, refine_building **0.9969**)

Already written by `tools/stage5_run.py`; the schema is unchanged, and this is
its statement of record.

| column | type | meaning |
|---|---|---|
| `pid` | int64 | global point id |
| `pred` | int64 | predecessor pid — **−1 at a root and where unreached** |
| `root` | int64 | the root reached, global pid; −1 unreached |
| `system` | uint8 | root system: 1 ground, 2 water, 3 deck; **255 unreached** |
| `dist` | float32 | geodesic metres to the root |
| `depth` | int32 | hops (0 at a root, −1 unreached) |
| `crossed_wall` | int32 | building-class nodes on the path |
| `crossed_foliage` | int32 | veg-class nodes on the path |

One row per **vertex slot** of the completed graph *G′* = kNN ∪ MST over the
area's pid range — not per point of the tile. `pred` is the arc set: the forest
is `(V, pred)` and every consumer reconstructs paths from it. `system = 255`
rather than 0 for unreached so that "no system" can never be mistaken for
`SYS_NONE = 0` in a join. `CAT_OTHER` is not stored: it is
`depth - crossed_wall - crossed_foliage`.

## Budgets

Each writer prints `resource.getrusage` peak RSS and **fails** above its budget.
A budget that is never checked is decoration. These are ceilings with headroom
over the measured/estimated cost in `stage5-cost.md`, on the 23 GB box:

| task | budget | basis |
|---|---|---|
| `water` | 6 GB | ~1 GB of 1.25 M-cell grid ops, plus one pass over 42.7 M points (x, y, z ≈ 1.0 GB) for the per-body level |
| `corridors` | 6 GB | **measured 5.2 GB** — re-declared from 3 GB, whose basis ("no point pass, the cloth is cached") was wrong: the cached cloth is the WORK-extent one, and the 60 m ring alone drapes it over the Granville deck, so every elevated footway there reads at-grade (IoU 0.8419 vs 0.9974). The task runs its own AREA-extent cloth: 42.7 M points through `dl.read` ≈ 3.5 GB transient plus CSF |
| `claims` | 12 GB | **measured 11.3 GB** — re-declared from 10 GB, whose estimate was low. Staged inside the run: `dl.read` of the tile's 8 columns **5.5 GB**, cloth + DEM **8.5**, `covariance`'s 2.9 GB PDAL FEAT las **9.7**, the rules themselves **10.1**, then +1.2 for the `pid` copy and the 42.7 M-row Parquet write |
| `outlines` | 12 GB | **measured 11.4 GB** — re-declared from 6 GB, whose basis assumed the assembly would be *ported* and read `claims.parquet`. It is not ported: `rules/layers.py:313-367` **is** the assembly and runs here under the same anchored substitution as `claims`, hooked one line after the block closes, so this task carries `claims`' cost (read 5.5, cloth/DEM 8.5, FEAT las 9.7, rules 10.1) and stops one block later. Cheaper than porting a block whose `forbid_cell` needs `over`, a per-point term `claims.parquet` does not carry — and stage-5 peak is unchanged, the forest still dominates |
| `forest` | 14 GB | **measured 11 GB** (`out/stage5_dev_forest.log`) — `directed=False` makes scipy hold the CSR's transpose, 5.1 GB twice — plus the chunked writer |

Peak for the whole of stage 5 becomes **14 GB, the forest**, instead of the
monolith's 22.4 GB. That is the memory fix, and it is the same act as writing
the tables.
