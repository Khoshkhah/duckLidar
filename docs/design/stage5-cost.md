# Stage 5 — what each task costs on the old area

The old area is the reference tile: **42,707,336 points**, **1,023,444
occupied 1 m cells** (1120 × 1120 grid including the 60 m margin), and
**426.8 M directed edges** in its slice of the stage-2 graph. Every number
below is for that area on this machine (20 cores, 23 GB RAM).

Each row says where its number comes from — this table is worth nothing if
it mixes measurements with wishes:

- **measured** — printed by a run, with the log it came from;
- **derived** — measured for the same work in a different wrapper (the
  oracle's own log, or the code's documented peak);
- **estimated** — reasoned from the data shape, and explicitly not yet run.

## The reference: what the oracle costs

The lifted lab code (`ducklidar.rules`, windowed) does **all** of stage 5
on this area in **332–385 s wall-clock** (three measured runs), of which
its shared-evidence block is 36–38 s and its final decision 23–29 s. Peak
RSS in the tile-era runs reached **13.4–19.1 GB**. That is the bar the
table version has to beat while reproducing its labels.

## Per task

| # | task | time | peak RAM | source |
|---|------|------|----------|--------|
| 5.1 | cell evidence (`cell_evidence`) | ~15 s | ~2 GB | estimated — one DuckDB `GROUP BY` streaming 42.7 M rows; the engine spills rather than holds |
| 5.2 | ground surface (`ground_cells`) | ~4 min | ~5 GB | measured wall-clock for 5.1+5.2 together (`pipeline_report.log`, two runs 4.5 min apart); RAM estimated — CSF over 42.7 M points is 1.03 GB of xyz plus its internals, then `dl.dem` rasterises |
| 5.3 | first labels — the `priors` view | ~0 s | ~0 | measured by construction — it is a view; cost is paid inside whatever scans it |
| 5.4 | water bodies (`water_cells` + `flood_bodies`) | ~20 s | ~1 GB | estimated from the cell-grid shape (1.25 M-cell boolean ops, `ndimage.label`, one propagation); the 5-minute figures in earlier logs were dominated by rebuilding evidence, not by this |
| 5.5 | bridge corridors (`corridor_cells`) | ~10 s | ~1 GB | estimated — vector buffering of ~90 ways then rasterisation |
| 5.6 | layer construction (per-class claims) | ~20 s | ~3 GB | estimated — ten boolean masks over 42.7 M points |
| 5.7 | outlines + containment | ~30 s | ~2 GB | estimated — polygonisation over 1.25 M cells |
| 5.8 | **support forest** (chosen variant: metric + MST edges + ground roots) | **144 s** (85 s graph + 59 s traverse) | **~11 GB** | **measured** (`out/stage5_dev_forest.log`); the four-variant sweep spans 91–205 s. RAM derived from the code's own note — `directed=False` makes scipy hold the transpose, 5.1 GB twice |
| 5.9 | conviction trial | ~60 s | ~4 GB | estimated — 31.5 M contested points judged on k-hop neighbourhoods |
| 5.10 | map vetoes | ~15 s | ~2 GB | estimated — point-in-polygon and buffer joins against ~500 footprints |
| 5.11 | decision + residue | ~25 s | ~3 GB | derived — the oracle prints 23–29 s for exactly this resolution over the same points |

**Full table-side stage 5 on the old area: ≈ 7–10 minutes, peak ~11 GB**,
dominated by two tasks — the ground cloth (5.2) and the support forest
(5.8) — which together are ~80 % of the wall-clock and set the memory
ceiling on their own.

## Where the time actually goes, and what to do

1. **The graph rebuild is the single largest repeated cost.** 53–110 s of
   every forest run is spent turning 426.8 M Parquet edge rows into a CSR
   that is identical every time. Caching `indptr`/`indices`/`data` as
   `.npy` removes it from every iteration after the first. Not yet done.
2. **The cloth is 4 minutes and runs once per area** — already cached as
   `data/stage5/cells.parquet` + `ground_grid.npy`, so it is paid once and
   read thereafter. That cache is why operator iterations now take minutes
   instead of the 5+ they cost earlier.
3. **The forest sets the memory ceiling** (~11 GB, and the transpose is
   most of it). This is also what makes the whole green window
   (4.5× this area) impossible in one pass today — see the scaling note in
   [stage5-labeling.md](stage5-labeling.md) task 5.8. The fix is
   partition + halo, which is available because the measured path depth is
   short (p50 8–11 hops, geodesic p50 4–10 m); it needs the tail measured
   before the halo can be set, exactly as stage 2's was.
4. **The variant sweep is settled — re-deriving it costs 10 minutes, so
   don't.** Four walks were measured against the oracle's refinement
   (`out/stage5_dev_forest.log`). The forest answers "what is holding this
   point up?" by walking outward from a starting system, step by step
   (each step ≤ 2.5 m), recording what the path went THROUGH — leaves mean
   tree, wall means building. The walks differ in three things: distance
   in metres or in steps, whether stranded islands get one shortest jump
   (the lab's MST edges), and where the walk starts.

   | name | the walk | refine_veg | refine_building |
   |---|---|---|---|
   | **Groundwalk** ✅ | metres · gaps bridged · from the ground | **0.9938** | **0.9992** |
   | Island-bound | metres · no jumps · from the ground | 0.9875 | 0.9735 |
   | Stepcount | **steps** · gaps bridged · from the ground | 0.7144 | 0.8684 |
   | Three-source | metres · gaps bridged · **ground + water + bridges** | 0.8799 | 0.9880 |

   Island-bound proves the jumps are needed: without them 0.78 % of points
   can never reach ground and buildings fall to 0.9735. Stepcount proves
   distance must be metric: inside foliage a step is 5 cm, across a gap
   2.5 m, and counting them alike collapses vegetation to 0.7144.
   Three-source is Kaveh's design, and the comparison above is UNFAIR to
   it: the oracle's refine_* grids were themselves produced by a
   ground-only analysis, so a ground-only walk necessarily wins that
   test. The honest reading is that Groundwalk is what *reproduces the
   oracle* (which the gate requires), while Three-source answers a
   question the lab never computed — which system holds each point up
   (floating on water, resting on a deck, standing on land) — and so has
   no oracle artifact to be measured against at all. Both run: same
   function, two root sets, two questions.

5. **Do not "optimise" the metric away.** Measured: dropping the geodesic
   for a hop tree collapses `refine_veg` agreement from 0.9875 to 0.7144 —
   and it is not even faster, because the priority queue is not the
   bottleneck, the graph build is.

## How to refresh these numbers

The estimated rows become measured the first time
`tools/stage5_run.py` runs the whole chain end to end — it prints per-step
timings and appends a `dl.stage_report` line. Replace any row this table
still marks *estimated* with that run's numbers, and note the date.
