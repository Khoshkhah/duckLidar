# Stage 5 at window scale — partition + halo, the forest's tail, the area as a parameter

**Status: PROPOSAL, 2026-09-05. Nothing below is built.** ▶ marks Kaveh's
decisions. Every number is measured on this machine (20 cores, 23 GB) from
the runs and tables named next to it, or marked *estimated*.

## The Problem

Stages 2 and 4 cover the whole green window (191,950,769 points). Stage 5
covers the owned tile only (42,707,336 points), and cannot be pointed at
the window:

- **Memory.** The full-tile tables run peaks at **17.4 GB** for 42.7 M
  points (`out/stage5_tile.log`, 2026-09-05: 268 s). Resident state is
  ≈ 280 B per point plus ~5 GB fixed, so the window in one pass is
  **55–78 GB** — 2.5–3.5× the box.
- **The driver is pinned.** `stage5_run.py --area` accepts only `old`;
  `LO, HI = 221_897_438, 264_667_331`, `OWNED` and `TILE` are hard-coded
  in 12 tools (`stage5_run/tables/cells/dev`, `stage7_objects/scene`,
  `stage8_shadows`, `viz_*`, `fetch_map/footprints`); the forest read and
  the features join assert a *contiguous* pid range — a spatial window
  spanning tiles has none.
- **The design's own promise was unmeasured.** [stage5-cost.md](stage5-cost.md)
  says partition + halo is available "because the measured path depth is
  short … it needs the tail measured". The tail is measured below, and it
  is not short.

## The Data

### Where the tile run's memory goes (`out/stage5_tile.log`, 2026-09-05)

| step | peak RSS | what is resident |
|---|---|---|
| pids | 5.2 GB | the owned sidecar's pid column + the read |
| after the rules' load (before `water`) | 11.9 GB | 42.7 M × (x y z f64, 7 survey fields, hag/echo/planar/scatter/exg/inten/moved/overhead f32, cell index) + stage-2 features |
| geodesic | 14.9 GB | + the 250 m blocks' subset kNN graphs |
| decide | 16.1 GB | + the 10 claim masks and the settled byte |
| map.claims / labels.pq | 17.4 GB | + point-in-polygon masks, the pyarrow table |

### Reach per operator — how far evidence travels into an answer

Anything with bounded reach is exact under partition + halo ≥ reach with
owner-writes ([pipeline.md](../pipeline.md), level 2). Measured bounds:

| operator | reach | source |
|---|---|---|
| row-wise priors, kNN features, `decide` (K_VOTE 5), residue | ≤ 2.5 m | stage-2 tables, cap |
| grid filters in the rules (`uniform_filter` 5/9 cells, dilations 10/20/25/30 cells, closing 5×5) | **≤ 30 m** | `rules/layers.py` l.84–97, 409, 556, 874, 899 |
| cloth ground (CSF, rigidness 1, 1 m) | tens of m | A1; unverified — see the tile-as-4 test |
| water body flood, water levels | body size — **but already a global cell operator** (`water_cells`+`flood_bodies` on `cells`) | 5.4 |
| corridors | map ways — global cell operator | 5.5 |
| conviction geodesic | **60 m, exact by construction** (250 m blocks, 62 m halo) | `glass_geodesic_blocks` |
| map vetoes / claims (20 m buffers, 3 m eave) | ≤ 20 m | D1–D3, `map_labels` |
| building outlines + containment (the lab's in-memory assembly; `outlines.parquet` exists but the run does not read it) | extent p50 3 m, p90 15 m, p99 146 m, **max 794 m**; 39 > 60 m, 24 > 100 m, **14 > 150 m** of 1,395 | `outlines.parquet`, measured 2026-09-05 |
| **support forest** (Groundwalk, feeds only `path_shares`) | see below | `forest.parquet` ⋈ `labels.parquet`, 2026-09-05 |

### The forest's tail — measured, and it is long

Geodesic distance to the ground root, all 42,707,336 attached points:
p50 6.4 m, p90 30 m, **p99 243 m, p99.9 381 m, max 400 m**; hops p50 11,
p99 838, max 1,835. 6.1 % of points have a path longer than 60 m. Who
owns that tail: bridge 42 %, building 26 %, floating 19 %, on_bridge 7 % —
decks reach ground only at abutments, boats reach shore over MST edges.

What matters is the **refinement's subjects** — veg_low, veg_high,
building (22,340,400 points), the only labels `path_shares` can flip:

| halo H | subjects with path > H | worst-case share of ALL tile points that could differ |
|---|---|---|
| 30 m | 1,948,294 (8.72 %) | 4.56 % |
| 60 m | 714,447 (3.20 %) | 1.67 % |
| 100 m | 197,339 (0.88 %) | 0.46 % |
| **150 m** | **32,831 (0.147 %)** | **0.077 %** |
| 200 m | 23,838 (0.107 %) | 0.056 % |

("Worst case" = every truncated path changes the verdict; most will not.)
The gate's budget is 0.5 % total. So a 60 m halo — enough for every other
operator — is **not** enough for the forest, and 100 m spends the whole
budget on it. Hop-count streaming is out: thousands of passes.

### The store, the window, the buffer

Nine sidecars, **458,741,189** points; the stage-1 window table is
191,950,769 (window 1800 × 1800 m + 60 m margin; ~59 pts/m²). The design
asks a 1 km shadow buffer (`area.json`); the nine tiles provide **W 389 m,
E 811 m, S 657 m, N 543 m**. The full buffer needs a 4 × 4 tile block —
seven more downloads (`fetch_tiles.py`).

## The Solution

### 1. The area is data — one place decides it

`data/area.json` already records bbox, margins and the sidecars. One
module, `tools/_area.py`, reads it and hands every tool: the area bbox,
the working box (+60 m), the shadow box (+buffer), the sidecar list, the
registry pid offsets. **Every `LO/HI/OWNED/TILE` constant is deleted.**
Where a tool needs "the reference tile" (the gate), it asks the area for
the tile's box and derives the pid ranges. `LIDAR_RUNBOX=tile|box` becomes
`--area <name>` with `tile` and `lab_box` as named areas in the same file,
so today's fast loop survives unchanged.

Pid stays the join key. A spatial partition's pid set is a union of
ranges (one run of 50 m sort cells per touched file); the two contiguity
asserts (`h_forest`, the features join) become **sorted pid-set joins**
with row-group pruning. `forest_graph(stage2_dir, lo, hi)` becomes
`forest_graph(stage2_dir, pids)` — edges with `src` in the set (the
owner-writes rule means every edge appears exactly once, under its src).

### 2. Three passes, stage-major, each finishing its table

**5A — global cell tables, streamed from the store.** `cell_evidence`
already takes `(files, bbox)` and streams — window-ready. Voxels likewise.
**Ground** is the exception: the cloth must run at point level (the
cell-minimum shortcut was measured insufficient — marina recall 0.17) and
192 M points do not fit one CSF. ▶ *Run CSF per partition with a 60 m
halo, owner cells only, assembled into one `ground_grid.npy` for the
window.* Then water bodies, water levels and corridors run once, globally,
on the 3.5 M-cell window table — unchanged code, exact by construction.

**5B — partitioned labeling.** `dl.balanced_boxes(files, area, max_pts)`
as in stage 2, each partition read with halo **H** from every sidecar it
touches; the exec'd rules run **unchanged** (the same `SUBS`) on
partition + halo; the owner writes `pid, label_rules, settled, label,
map_type` for its own points to `part-NNNN`; consolidation sorts by pid
into `labels.parquet` and deletes the parts (invariant 2). Inside 5B the
forest is per partition: Groundwalk over the partition's completed graph
(kNN ∪ MST edges with src in the halo'd set), `path_shares` read off it —
exact for every point whose path is ≤ H, which is the table above.

**5C — the gate, twice.**
1. *Before the window, the tile as four partitions.* Run the tile with
   the chosen H as 4 partitions and diff against today's frozen single-run
   `label_rules` (99.9655 % against the lab). This is the direct test of
   "halo ≥ reach" — for the cloth, the outlines and the forest at once —
   at 4 × ~100 s. **It fixes H before anything larger runs.**
2. *The window's tile slice.* Rows with pid in the tile's ranges vs the
   same reference, bar ≥ 99.5 %, **and the disagreements binned by distance
   to the nearest partition border** — if they concentrate within H of a
   border the halo is short; if they are spread, they are something else.

### 3. Sizing

At 59 pts/m², a 16 M-point partition is ~520 m on a side.

| halo H | box with halo | points per run | est. peak RSS | forest exact for subjects | worst-case gate cost |
|---|---|---|---|---|---|
| 60 m | 640 m | 24 M | ~10 GB | 96.8 % | 1.67 % |
| 100 m | 720 m | 31 M | ~13 GB | 99.12 % | 0.46 % |
| **150 m** | **820 m** | **40 M** | **~17 GB** (the tile run's footprint, proven) | **99.85 %** | **0.077 %** |

RSS is scaled from the tile run's 280 B/pt; one worker at a time at
150 m (two would fit at 60 m, which the forest forbids). Window: 12
partitions × ~4.5 min ≈ **1 h** for 5B (*estimated* from 268 s / 42.7 M).

## What is exact, what is bounded, what is decided

| | verdict |
|---|---|
| row-wise, kNN features, decide, residue, grid filters, map step, geodesic | **exact** under any H ≥ 62 m |
| water body, water levels, corridors | **exact** — global on cells, as now |
| cloth ground | exact up to the cloth's relaxation reach — *verified by the tile-as-4 test*, not assumed |
| outlines + containment | **bounded**: outlines wider than H are cut at partition borders; at 150 m that is 14 of 1,395 — fused low-roof complexes (794 m at 15 m roof, 405 m at 15 m) and two tower blocks (287 m at 53 m, 275 m at 43 m). Containment claims *near the cut* may differ. ▶ |
| support forest → path_shares | **bounded**: ≤ 0.147 % of subjects at 150 m, measured on the tile |
| everything downstream of `labels.parquet` (stage 6 streams the edges once, whole-window by construction) | unchanged |

## The Result

`data/stage5/labels.parquet` for the area (192 M rows, ~2.5 GB, sorted by
pid, `label_rules` + `label`), `ground_grid.npy` / `cells` / `water` /
`corridors` for the window, the two gate reports. Stage 6 runs as is.
Stages 7 and 8 need only the area module (they are pinned to `OWNED`,
not to any tile-sized assumption). Stage 8.2 becomes possible to the
extent the area includes the buffer — see ▶ 1.

## Costs

| | time | RAM | basis |
|---|---|---|---|
| 5A cells + voxels (window) | ~2 min | ~2 GB | *estimated*, DuckDB streaming |
| 5A ground, partitioned | ~25 min | ≤ 6 GB per partition | *estimated* from the tile's ~4 min CSF × 4.5 × 1.3 halo overhead |
| 5A water, levels, corridors | < 1 min | < 1 GB | measured on the tile, cell-sized |
| 5B labeling, 12 partitions | **~1 h** | **~17 GB**, one worker | *estimated* from 268 s / 17.4 GB per 40 M |
| 5C gates | 15 s each | 1 GB | measured |
| stage 6 on the window | ~5 min | 1.5 GB + labels | 194 s on the box, dominated by the edge stream that already reads everything |
| stage 7 objects on the window | ~30 min | unmeasured | *estimated*, 146 s on the box × ~11 in points |
| stage 8, 3 × 3 km at 1 m | ~1 min | small | v1: 7.1 M cells × 25 suns in 40 s |

**If the area is the nine tiles (459 M points):** stage 2 ≈ 31 min
(768 s / 192 M), stage 4 ≈ 70 min, stage 5 ≈ 3.5 h — and the **MST's
Kruskal buckets peaked at 10.33 GB on 192 M** ([stage2-mst.md](stage2-mst.md)),
so ~25 GB at 459 M: the MST needs its own memory work before nine tiles.

## Rejected

- **One pass on a bigger machine.** 55–78 GB; there is no such machine
  here, and the stage-major design exists so there need not be.
- **A global streamed forest by hops** (one DuckDB self-join per hop):
  hops p99 838, max 1,835 — thousands of passes over 1.9 B edges.
- **A global in-RAM forest with recomputed weights.** Symmetric CSR
  indices alone for 2.1 B undirected arcs are ~17 GB before xyz, dist,
  pred and the heap.
- **Dieting `layers.py` to fit the window in one pass.** Needs 4×; the lab
  already took two passes at it (255d527, cf97a75) for ~5 GB.
- **Tile-major.** Retired by decision 2026-08-02; borders leak.
- **Dropping the MST from G′.** Forbidden 2026-08-02 — 0.78 % unreachable
  and they are the boats and detached roofs.
- **Cell-minimum cloth.** Measured insufficient (marina recall 0.17).

## Decision points ▶

1. **Area of the first run.** (a) the window, 192 M — everything stages 2/4
   already hold, stage 8.2 stays an upper bound; (b) the nine tiles,
   459 M — buffer 389–811 m, needs stages 2/4 re-run and the MST memory
   work; (c) fetch seven tiles for the full 1 km. *Recommend (a) now,
   gate on the tile slice, then (b).*
2. **Halo.** *Recommend 150 m*, one worker, ~1 h. Confirmed or moved by
   the tile-as-4 test, which is the first thing to run.
3. **Outlines.** In-partition assembly (14 wide outlines cut; measured
   effect from the tile-as-4 test) vs. substituting the global
   `outlines.parquet` into the rules (exact, but a new `SUBS` span that
   must re-pass the tile gate). *Recommend in-partition first; substitute
   only if the test shows border error there.*
4. **Ground.** In-run cloth per partition (the path that passes today)
   vs. one global 5.2 grid substituted into the rules (one truth, but
   another `SUBS` to gate). *Recommend in-run first.*
5. **The far 0.15 %.** Accept as bounded and measured, or add a second
   pass for far points only (delta-stepping over the streamed edges from
   the partitions' frontiers). *Recommend accept; revisit if gate 2 shows
   them.*

## Impact

shadowCity2 `tools/`: new `_area.py`; `stage5_run.py` (partition loop,
pid-set joins, parts + consolidate, `--area`); `stage5_cells.py`
(partitioned ground, area frame); `stage5_tables.py` (area frame);
`stage5_gate.py` (slice mode, border binning); `fetch_map.py` /
`fetch_footprints.py` (area bbox — `fetch_map` already takes one);
`stage7_*`, `stage8_shadows.py`, `viz_*` (area instead of `OWNED/TILE`,
can follow later). ducklidar: `forest_graph` / `knn_csr` take a pid set;
one test on the synthetic tile. *Estimated* build: ~2 days, plus the
measured runs above. Docs: `stages.md` row 5, `tables.md` (`label_rules`),
this file.

## Optimize?

- The 280 B/pt baseline is the lever: halve it and two workers fit at
  150 m, or partitions double. Candidates: drop `moved`/`overhead` after
  their consumers, f32 for x/y in local origin (the mesh build already
  does this), free the survey colour fields after `exg`.
- The hybrid forest (▶ 5) would make the forest exact at any H and let H
  fall back to 62 m, the geodesic's — then the halo overhead drops from
  2.5× to 1.5× and the run halves.
- One cloth (▶ 4) removes ~4 min per 42.7 M from every partition.
