# Stage 2 — the voxel table

**Purpose:** make the *vertical* structure of the survey queryable, so the
rules that reason about what is above what stop being per-point numpy and
become joins.

## Task — voxel evidence

**The Problem.** The cell table (`cells.parquet`) is a 2-D projection: one
row per 1 m column, holding `min_z`, `max_z`, counts. That collapses the
column, and several measured rules are *about* the column's interior:

- **the openness gate** — "the laser reached the FLOOR under this ROOF" is
  the difference between a carport and a shed, and it is a statement about
  two separate occupied heights in one column;
- **the eave leak**, documented in the lab's own comments — "a two-storey
  house's eave cell: roof above, the yard's ground in the same cell" —
  which `min_z`/`max_z` cannot express, because both readings are correct
  and the structure between them is what matters;
- **deck columns** ("solid column top > 5.5 m under the deck"),
  **marina superstructure** ("veg/small *above* floating matter"),
  **overhead** (what hangs above me) — all vertical-profile questions
  answered today by per-point work.

**The Solution.** Aggregate the points into **0.5 m voxels**: one row per
occupied (column, height level). Not a geometry source and not a
connectivity structure — *evidence*, exactly like the cell table, one
dimension richer.

**Why 0.5 m, and not a taste.** A voxel height is a parameter, and this
project has spent the day removing arbitrary parameters. This one is
pinned by the rules' own measured constants: `WATER_HAG = 0.5`,
`GROUND_CEILING = 1.0`, `TALL = 2.0` — **0.5 m divides all three**, so
every existing threshold falls on a voxel boundary instead of inside one.
Any other height would put a rule's decision line in the middle of a cell.

**The Result — measured, old area (42.7 M points):**

| voxel height | occupied voxels | vs the 2-D table |
|---|---|---|
| 2.0 m | 2,181,326 | 2.1× |
| **0.5 m (chosen)** | **4,245,142** | **4.2×** |
| 0.25 m | 6,120,868 | 6.0× |

Going 3-D costs 3–6× the 2-D table, not 100×, because **the laser only
hits surfaces**: the average column has just **3.76 occupied 1 m levels**.
The window at 0.5 m is ~19 M rows — a few hundred MB beside a 12 GB edge
table.

## Why it belongs to stage 2

Stage 2 already reads every point, in balanced partitions, and holds them
in memory to compute the kNN. The voxel aggregation is a `GROUP BY` over
data that is *already there* — computing it anywhere else means reading
192 M points again. This is the same argument that made stage 2 compute
the kNN once instead of twice, and that moves the MST length buckets into
stage 2: **do not pay twice for data you already hold.**

**Partition correctness.** Points are owner-masked as everywhere else, so
no point is counted twice. A voxel straddling a partition boundary
receives a partial row from each side; consolidation sums them
(`group by cell, level`), which is exact because every contribution is a
count or an extremum.

## Output

`data/stage2/voxels.parquet`

| column | type | meaning |
|---|---|---|
| cell | i64 | the 2-D cell — joins straight to `cells.parquet` |
| level | i32 | `floor(z / 0.5)` — the height band |
| n | i64 | returns in this voxel |
| min_z, max_z | f32 | extremes inside the voxel |
| split_share | f32 | share of returns whose pulse split — penetrable vs solid |
| n_ground, n_water, n_building, n_veg | i64 | vendor class counts |

## What it does and does not do

**Does:** answer "what is above what" as SQL. The openness gate becomes
*"has this column an occupied voxel near ground AND one above?"*; deck
columns and superstructure become profile queries; `overhead` becomes a
lookup.

**Does not:** replace the graph. Voxel adjacency is *not* 2.5 m
connectivity — a wall voxel and a floor voxel touch as boxes while their
points may be metres apart. Nor is it geometry: a wire or a railing
occupies a whole voxel, so it is evidence *for* rules, never a source of
shape.

## Verification

1. `sum(n)` over the voxel table = the working point count exactly (no
   point counted twice or dropped at a partition seam).
2. Per cell, `min(min_z)` and `max(max_z)` over its voxels equal the 2-D
   table's `min_z` / `max_z` — the projection must agree with the
   projection it came from.
3. The openness gate, expressed as a voxel query on the reference tile,
   must reproduce the rule's own per-point answer within the tolerance the
   label gate uses.

## What it closed (the eave leak, partially)

**The problem, in the lab's own words.** `rules/layers.py` l.182-185, on the
low-column guard inside `L["placed"]`:

> the WHOLE column low (cell top <= 5.5 m above ground) so a two-storey
> house's eave cell -- roof above, yard's ground in the same cell -- cannot
> qualify. ONE-STOREY EAVES REMAIN THE KNOWN LEAK; the component interior
> test that solves them fully is context, not a per-point rule (v2).

and l.609-612, on the cost of the leak:

> placed is an EVIDENCE layer in v1, not a competitor: with specificity it
> stole one-storey rims from building (IoU 0.830 -> 0.766 at spec 3, 0.781
> with the low-column guard -- THE EAVE PROBLEM, WHICH PER-POINT RULES
> CANNOT SOLVE; the component-interior test that does is context, i.e. v2).

So `DECIDE = [k for k in LAYERS if k != "placed"]`: the layer is computed,
measured, and thrown away, because 2-D `min_z`/`max_z` read a truck and a
one-storey eave identically — ground below, roof above.

**What the voxels add.** The column's interior. A truck is matter from the
floor band to its roof; under an eave there is walkable air between the
yard's ground return and the overhang. `dl.column_support` (stage 5 →
`labeling.py`) turns the voxel table plus `cells.ground_z` into a
`ncells × max_h/step` occupancy matrix and answers `supported(cell, h)` —
"no empty vertical run wider than `min_gap` between the floor band and `h`".

**Measured**, old area WORK box (489940, 5456940, 491060, 5458060), 1 m pix,
42.7 M returns, building IoU against the survey's own classification
("layered method, agreement where the survey committed"). Harness:
`shadowCity2/tools/stage5_dev.py eave`, which patches `rules/layers.py` by
anchored text substitution and execs it — `rules/*` is untouched. Variant A
reproduces the lab's shipped run exactly (building 0.832 / ground 0.947 /
veg_high 0.917 / water 0.898 vs `tile_layers14.log`'s 0.832 / 0.946 / 0.917
/ 0.898), so the patched path is the oracle.

| variant | building IoU | placed claims surviving |
|---|---|---|
| **A** — `placed` dropped from `DECIDE` (today's shipped behaviour) | **0.832** | 746,972 computed, all discarded |
| **B** — `placed` competing at spec 3, no column test | 0.816 | 747,305, all kept |
| **C** — B + the column test, `min_gap = 0.5` m | **0.829** | 259,093 |
| C, `min_gap = 1.0` m | 0.827 | 377,561 |
| C, `min_gap = 1.5` m | 0.826 | 481,014 |
| C, `min_gap = 2.0` m | 0.823 | 585,589 |

The `min_gap` curve is strictly monotone: tighter gap → fewer surviving
claims → closer to A. It extrapolates to A only at "remove everything", so
**no `min_gap` setting makes `placed` a net-positive competitor**.

**The test refuses the right columns.** At `min_gap = 1.5` m it removes
265,945 claims (35.6 %) in 17,656 cells. Every sampled profile is the eave
signature and never a truck — e.g. cell 327387, ground 6.71 m, 49 claims at
hag 2.8-3.8 m, gap 2.5 m, `|#.....###.......|` (each `#` a 0.5 m band with
matter, band 0 = the ground band). Population check against the survey, not
a sample: in the cells the test stripped, the judged returns in the 1.8-5.0 m
band are 48.2 % survey-BUILDING / 51.2 % veg_high; in the 29,519 cells where
every claim survived, 32.1 % building / 57.3 % veg_high. Removed columns are
1.5× enriched in survey building — the test preferentially strips roof and
eave columns, as designed. (The veg_high half is overhanging crown: also
correctly not a "placed object", but it does not move building IoU.)

**How partial the fix is.** The voxel test recovers **13 of B's 16
IoU-thousandths** — 0.816 → 0.829 — but A is 0.832, so letting `placed`
compete still costs **-0.003**. `placed` therefore stays out of `DECIDE`.
Where the residual lives: building precision is 0.94 in both A and C, while
recall falls 0.88 → 0.87, i.e. the surviving supported claims take points
the decision would otherwise have resolved as building rather than
mislabelling new ones. What is left is eaves with sub-0.5 m gaps (a porch
roof over a step — 0.5 m is one voxel band, already the strictest possible
setting) and genuinely continuous wall-adjacent structure. **Per-point
specificity is doing the remaining damage, not the void test.**

The lab's 0.766 / 0.781 are from an older pipeline (spec 3 without and with
the low-column guard). Today's `rules/layers.py` already carries
`celltop <= 5.5`, and object topology, the map vetoes and the path
refinement have all landed since; 0.816 is the honest current cost of
letting `placed` compete unguarded, and it is the number C beats.

**Not attempted** (each a new capability, so Kaveh's call): dropping
`placed`'s specificity from 3 to 2 so it ties `building` and the neighbour
vote settles it, with the column test still on; and support AND a bounded
component footprint (a truck is ~10-30 cells, a roof rim is a long thin
ribbon) — which is the component-interior test the lab actually deferred.
