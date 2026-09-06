# The tables: every stage's inputs, outputs, columns, and file names

The data contract of the pipeline. Each stage reads tables, writes tables;
this page fixes the schemas and names. Paths shown are the canonical layout
under an application's `data/` directory (the green-window build in
shadowCity2 is the worked example); the schemas are the contract wherever
the files live. `pid` is always the point's global id: the file-offset of
its sidecar in the registry (sorted sidecar paths) plus its row in that
file.

---

## Stage 0 — store

**In:** the survey's delivered files — LAS / LAZ / zip / COPC, one per tile.

**Out:** `data/lidar/<tile>.parquet` — one sidecar per survey file,
spatially sorted (50 m cells), zstd. **This is the only table that stores
points; every other table refers to them by `pid`.**

| column | type | meaning |
|---|---|---|
| x, y, z | f64 | position, survey CRS (metres) |
| classification | u8 | vendor class (ASPRS: 2 ground, 6 building, 9 water, 7/18 noise) |
| intensity | u16 | return strength |
| return_number, number_of_returns | u8 | echo position in its pulse |
| scan_angle | i16 | look obliqueness |
| gps_time | f64 | acquisition time (flight-line resolution) |
| point_source_id | u16 | flight line |
| red, green, blue | u16 | colourisation, where the survey has it |
| pid | i64 | the point's row in this file, assigned after the sort — permanent |

## Stage 1 — the working table

*The store holds whatever files the survey delivered; stage 1 defines the
project's points: **clip to the target area, drop noise, attach the
row-wise attributes.** One place owns the area filter; every later stage
reads this view instead of re-deriving the clip.*

**In:** the store (`data/lidar/*.parquet`) + the area definition
(`data/area.json`).

**Out:** the **view `points`** inside `data/<area>_store.duckdb` — a
definition, not a file (the filter and the derived columns are pushed
into every scan; row-group pruning makes the clip nearly free, and
materialising would copy ~200 M rows for nothing). On the green window:
191.9 M of the store's 458.7 M points (42 %). The view is exactly:

```sql
create view points as
select *,                                            -- every store column
       cast(return_number as double) / number_of_returns as echo_ratio
from read_parquet('data/lidar/*.parquet')
where x >= X0-60 and x < X1+60                       -- area bbox + 60 m
  and y >= Y0-60 and y < Y1+60                       --   working margin
  and classification not in (7, 18);                 -- noise dropped
-- view points_all = the unclipped store, for the rare consumer that needs it
```

So `points` has **all 14 store columns plus**:

| column | type | meaning |
|---|---|---|
| echo_ratio | double | this return's position in its pulse: < 1 means the pulse split (foliage character); = 1 on single/last returns |

Greenness (the colour index) joins this view when its exact formula is
lifted from the rules during the 5a walkthrough — same pattern, one more
expression.

**The rule for view vs file:** a stage-1 column stays virtual while it is
a cheap expression every consumer can scan; it graduates to a Parquet
file (`data/stage1/<column>.parquet`, columns `pid` + the value) only if it becomes expensive to
compute or is needed by consumers that cannot query the view. None has
crossed that line yet.

(The `constants` table living in the same .duckdb belongs to **stage 3**,
not stage 1 — see below.)

## Stage 2 — the graph and the shape features

**In:** the store (x, y, z, pid; balanced partitions + halo).
**Out:** two tables, nothing else. **After stage 1 nothing is named by
tile — not even staging**: the sweep's transient parts are `part-NNNN`
files with no geographic meaning, and the run itself concatenates them
into the two tables and deletes them before reporting done. The tables
are unsorted (decided 2026-08-02): every consumer streams them whole or
hash-joins on pid — nothing range-queries them, so no order is paid for.

`data/stage2/features.parquet` — the kNN-16 shape of every point's neighbourhood:

| column | type | meaning |
|---|---|---|
| pid | i64 | the point |
| e1, e2, e3 | f32 | covariance eigenvalues, sorted e1 ≥ e2 ≥ e3, normalised to sum 1 (self-inclusive neighbourhood, PDAL convention). Planarity = (√e2−√e3)/√e1, scattering = √e3/√e1, etc. — derive in the query |
| nz | f32 | \|z\| of the normal (smallest eigenvector): ~1 roof, ~0 wall |

`data/stage2/knn.parquet` — **the whole graph, one file**:

| column | type | meaning |
|---|---|---|
| src, dst | i64 | pids; directed src → dst, one row per discovered direction. `least/greatest + distinct` for the undirected set |
| len | f32 | Euclidean distance (≤ 2.5 m by construction; k ≤ 10 per src) |

Each edge is written once, by the partition that owns its `src`.

`data/stage2/mst.parquet` — **the Euclidean MST of the points**: the
minimum spanning tree of the complete graph on *V* with *w* = ‖u − v‖ (3-D).
Not a completion of *G* and not derived from `knn.parquet` — the kNN edges
are only the candidate stream inside the 2.5 m cap; every edge longer than
that is found by search (design: [design/stage2-mst.md](design/stage2-mst.md)).
Exactly |V| − 1 rows.

| column | type | meaning |
|---|---|---|
| src, dst | i64 | the pid pair; undirected, one row per tree edge |
| len | f32 | ‖src − dst‖, 3-D Euclidean. Unbounded — most rows are ≪ 2.5 m, a few thousand are the long links no cap could supply |

**Measured, 2026-08-02** (window, `tools/stage2_mst.py --area window`):
**191,950,768 rows = |V| − 1**, **1,294,693,984 bytes** (1.29 GB), total
weight 28,831,927.9 m. Lengths p50 **0.1310**, p90 0.2637, p99 0.4793,
p99.9 0.9633, max **22.9261 m**; **4,896 edges exceed 2.5 m** (0.0026 % of
the tree, 15,575.922 m). Union-find over the file ends with one component, so
it is a spanning tree by measurement, not by claim.
(`mst_old_area.parquet` is the same thing over the old tile: 42,707,335 rows,
305,681,612 bytes, weight 6,843,702.973 m.)

**It has no parameters, so it holds every scale at once.** Deleting every
MST edge longer than *t* leaves exactly the single-linkage clusters at
scale *t*: `where len <= t`, then connected components of what remains. The
cap stops being a commitment baked into a computation and becomes a query.

**How it differs from stage 4 at t = 2.5 m — it does not agree, and the
disagreement is a real finding.** Measured on the old tile: the MST cut at
2.5 m gives **1,892** components, `stage4/components.parquet` gives
**3,058**. Ten of the MST's components each swallow several stage-4 ones
(1,197 in all, 309,147 points) because **1,429 of the completion edges are
shorter than 2.5 m** — point pairs inside the cap that k = 10 never
recorded, so no amount of kNN connectivity could have joined them (a
measured example: two points 0.99 m apart whose 10th neighbours sit at 0.55
and 0.78 m). The opposite direction, one stage-4 component landing in 22 of
ours, is the old tile's own boundary — stage 4 routes through points outside
it — and every one of those 77,643 points lies within 65 m of the tile edge.
At window scale the same truncation shows as 5,720 sub-cap completion edges.

**Two graphs, both first-class.** `knn.parquet` alone is *G* — it
answers *"what is one physical piece"*, and **stage 4's components are
computed on it ALONE, never on the union** (on the union there is exactly
one component by construction, and the 10,499 islands vanish).
`knn.parquet ∪ mst.parquet` is *G′* — connected — and answers
*"what path exists"*: the support forest, path-to-ground, isolation.

**Ordering:** the MST consumes the kNN edges as its candidate stream, so
production order is edges (stage 2) → `mst.parquet`. It does **not** need
stage 4 — only verification 4 reads the components, to compare partitions.
The artifact belongs to the graph and lives in `data/stage2/`; the tool that
writes it is `tools/stage2_mst.py`.
See [design/stage2-graph.md](design/stage2-graph.md) task 2.6.

`data/stage2/voxels.parquet` — **vertical evidence**: one row per
occupied (1 m column × 0.5 m height band). The 3-D partner of
`cells.parquet`, joined on `cell`.

| column | type | meaning |
|---|---|---|
| cell | i64 | the 2-D cell (joins to `cells.parquet`) |
| level | i32 | `floor(z / 0.5)` — the height band |
| n | i64 | returns in this voxel |
| min_z, max_z | f32 | extremes inside the voxel |
| split_share | f32 | share of split pulses — penetrable vs solid |
| n_ground, n_water, n_building, n_veg | i64 | vendor class counts |

0.5 m is pinned by the rules' own constants (`WATER_HAG` 0.5,
`GROUND_CEILING` 1.0, `TALL` 2.0 all divide by it), not chosen by taste.
Measured on the reference tile: 4,245,142 occupied voxels — only 4.2× the
2-D table, because the laser hits surfaces and a column has just 3.76
occupied levels on average. Evidence only: voxel adjacency is NOT 2.5 m
connectivity, and a wire fills a whole voxel — never a source of geometry.
Design: [design/stage2-voxels.md](design/stage2-voxels.md).

## Stage 3 — global constants

**In:** the whole store. **Out:** rows in `constants` (see stage 1) — only
values with one scene-wide answer. Water level is *not* one (tide per
flight line; see stages.md) — it is computed conditioned, at use.

## Stage 4 — connected components

**In:** `data/stage2/knn.parquet` (streamed), the registry row counts.
**Out:** `data/stage4/components.parquet`:

| column | type | meaning |
|---|---|---|
| pid | i64 | every point that has stage-2 features |
| comp | i32 | its component of G; ids compacted, 0 = largest (terrain-connected mass) |

## Stage 5 — labeling

Outputs: `cells.parquet` (evidence), `forest.parquet` (the support forest), `labels.parquet` (the contract). Two levels: **5a first labels** (row-wise proposals and seeds over
`points ⋈ features`) then **5b revision** (regional evidence and the
support forest overturn proposals; see stages.md). Both levels end in the
one output table below.

**In:** the `points` view (stage 1); `features.parquet` (stage 2 —
planarity/scattering are formulas over e1..e3); `knn.parquet` (the
graph, for the support forest and the revisions); `components.parquet`
(stage 4); the map facts (the Overture extract and/or OSM canvases —
both queryable, `read_json_auto` / spatial join).

**Execution design: DECIDED (Kaveh, 2026-08-02).** 5a = the `priors`
VIEW in the store db (expressions over `points ⋈ features`, the rules'
named constants, computed in the scan). 5b = table operators per
docs/rules-operators.md's roadmap: cell evidence → global cloth → water
bodies (lab predicate verbatim, levels per body × flight line) →
corridors → the support forest (multi-source BFS) → vetoes → decision →
residue. The lifted windowed rules are a validation oracle only; the
window stays unlabeled until the table implementation passes the
reference gate (≥ 99.5 %). Staging is transient, never contract.

**Out:** `data/stage5/labels.parquet` — one table, every working point
exactly once:

| column | type | meaning |
|---|---|---|
| pid | i64 | the point |
| label | i8 | 0 ground · 1 veg_low · 2 veg_high · 3 building · 4 water · 5 small · 6 floating · 7 bridge · 8 on_bridge · 9 glazing |
| settled | i8 | label provenance as confidence, filled by cause: 0 single claim (uncontested) · 1 settled by specificity · 2 settled by neighbour vote · 3 residue copy of nearest labelled neighbour. Records HOW the label was decided, not a pseudo-probability; the gate compares labels only, provenance is reported |

`data/stage5/cells.parquet` — the cell evidence table
(`dl.cell_evidence`): `cell i64 · n i64 · min_z f64 · max_z f64 ·
split_share f64 · n_ground i64 · n_water i64` — cell id from
`dl.cell_of` (row-major on y, `cy·nx + cx`); `max_z` is the measured
DSM per cell — the top surface (highest return, noise excluded);
occupied cells only. A filled DSM over returnless cells would be a
model, not a measurement — added only if a consumer needs it — plus the 5.2 ground-trust
columns: `ground_z f32` (the bare-earth value) · `ground_measured bool`
(the cloth kept actual evidence in this cell — value is measurement,
not interpolation) · `ground_dist f32` (metres to the nearest measured
cell, from fill's distance_to_evidence; the honest uncertainty proxy —
grows under wide roofs) · `ground_fp f32` (cells inside a MAPPED
building footprint only: the building's single ground datum from
`dl.building_level` — the ring of measured ground around the footprint,
one number per building stamped into all its cells; null outside mapped
footprints, trust fields per building_level's conventions, fixed at
implementation). Mean/variance per cell were rejected (variance
measures roughness of the cell's contents, not ground uncertainty; mean
biases up with grass) — Kaveh 2026-08-02. Persisted evidence table
(decided by Kaveh 2026-08-02), auxiliary — later stages consume only
`labels.parquet`. Explained column by column in [cells.md](cells.md).

### `data/stage5/forest.parquet` — the support forest

The shortest-path forest F = (V, A) of the completed graph G′ = (V, E ∪ B)
rooted at the seed set R (ground ⊍ water ⊍ bridge decks): each vertex
stores its parent arc, the seed that reached it, that seed's system — the
tree's label — and what its path crossed. Persisted (Kaveh, 2026-08-02):
144 s and 11 GB to rebuild, and it answers "what holds this point up"
directly.

| column | type | meaning |
|---|---|---|
| pid | i64 | the vertex |
| pred | i64 | parent in F (the arc set); −1 at a root or unreached |
| root | i64 | the seed vertex that reached it — the tree identity |
| system | u8 | the tree's label: 0 ground · 1 water · 2 bridge · 255 unreached |
| dist | f32 | geodesic metres to the root |
| depth | i32 | hops to the root; −1 unreached |
| crossed_wall | i32 | wall-like vertices on the path |
| crossed_foliage | i32 | foliage-like vertices on the path |

Written once per root set — Groundwalk (R = ground, reproduces the
oracle's refinement) and the three-system walk (attachment questions).
Explained column by column, with the queries it enables, in
[forest.md](forest.md).

## Stage 6 — instances

**In:** `labels.parquet`; `knn.parquet` (components of the
class-induced subgraphs, and same-class fusion); footprint way ids from
the map tables. (Any per-partition instance staging is transient.)

**Out:** `data/stage6/instances.parquet`:

| column | type | meaning |
|---|---|---|
| pid | i64 | every point belonging to an instance (noise/unassigned absent) |
| inst | i32 | global instance id, 0 = largest |

and `data/stage6/instance_table.parquet` — one row per thing:

| column | type | meaning |
|---|---|---|
| inst | i32 | the instance |
| class | u8 | stage-5 label code of its class |
| size | i64 | owned points |
| osm | i64 | map way id for mapped instances (buildings, piers, bridges), −1 otherwise |

## Stage 7 — scene build

**In:** `instances.parquet` + `instance_table.parquet` (which thing each
point belongs to and what class it is), the store (each instance's points
by pid), the map footprints, and stage 5's `ground_grid.npy` for the datum
under a building. **Out:** two files, because the stage answers two
questions — what blocks light, and what the thing looks like.

`data/stage7/scene.npz` — the **shadow** scene: one array per part, keyed
`{bid}_{i}_{kind}` with a matching `{bid}_{i}_col`.

| key | type | meaning |
|---|---|---|
| `{bid}_{i}_{kind}` | f32 (n, 3) | triangle soup, n divisible by 3; `kind` ∈ {`solid`, `slab`} |
| `{bid}_{i}_col` | u8 (n, 3) | per-vertex colour |

`kind` is the whole point: **solid** blocks light, **slab** has open air
beneath it (a canopy the openness gate proved, a bridge deck), so the low
sun passes under. Stage 8 consumes exactly this distinction.

`data/stage7/scene_export.npz` — the **object** scene, per class, in the
same layer format the lab's export uses so one viewer renders both:

| key | type | meaning |
|---|---|---|
| `{layer}_pos` | f32 (n, 3) | all of that layer's triangles, concatenated |
| `{layer}_col` | u8 (n, 3) | per-vertex colour |
| `{layer}_runs` | i64 (2k,) | flat `[inst id, vertex count]` pairs — which object each block belongs to |

`layer` ∈ {`building`, `boat`, `dock`, `floating`, `bridge`, `object`,
`glass`}. Sidecar `objects_meta.json` records the box, the water level and
the per-layer object counts.

## Stage 8 — shadows

**In:** `scene.npz` (solid + slab parts), stage 5's ground grid, and the
store's returns for the vegetation slab. **Out:** GeoTIFFs on the run's
grid, EPSG:26910:

| file | meaning |
|---|---|
| `sun_hours.tif` | hours of direct sun per m² over the day's protocol |
| `surface.tif` | the solid blocker surface, NaN where nothing stands |
| `slab_top.tif`, `slab_bot.tif` | the slab pair — light passes between them |

The protocol is v1's (fall equinox, 10:00–16:00, 15-minute steps → 25
casts, so 6.25 h is a fully lit cell). **Without the 1 km buffer these are
an upper bound**, and the stage says so in its report line.

---

## Frozen results — `MANIFEST.json`

Any stage directory may carry a `MANIFEST.json` written by
`tools/stage_freeze.py`. It is not an input to anything; it pins what a
finished stage produced so it can be identified later without trusting
anyone's memory of it.

| key | meaning |
|---|---|
| `files` | per contract file: `rows`, `bytes`, `sha256`, and the parquet key-value metadata |
| `run`, `gate` | the stage's own report lines from `out/pipeline_report.log` |
| `ducklidar_commit`, `shadowcity2_commit` | what the code was |
| `not_done`, `caveats` | what the stage deliberately does NOT do, and what differs from the reference |

`stage_freeze.py --verify` re-hashes and reports drift, which answers "is
what is on disk still the thing that passed the gate?"

---

## One picture

```
survey files ─0→ lidar/<tile>.parquet (pid!)
                 │
                 1→ view: points = area clip + noise drop + row-wise attrs
                 2→ stage2/features.parquet  stage2/knn.parquet
                 3→ constants (decide-once shelf; empty by decision)
                 4→ stage4/components.parquet
                 5→ stage5/labels.parquet     (+ cell tables)
                 6→ stage6/instances.parquet  instance_table.parquet
                 7→ scene parts + terrain
                 8→ sun_hours.tif
```

Every arrow is "read tables, write tables"; every table joins to the
others on `pid` (points), `cell` (rasters), `inst` (things), or
`comp` (pieces).
