# Stage 5 — labeling

**Purpose:** name every point — ground, water, roof, foliage, glass,
bridge, floating. Two levels: **5a proposes, 5b revises.** Nothing is
final until revision has run.

**Design status: DECIDED (Kaveh, 2026-08-02) — all four questions closed.**
Contract fixed (in: the stage tables; out: one `labels.parquet`).
5a = a view. Support forest = multi-source BFS with path composition.
Water seeds = the lab predicate verbatim. And the FREEZE: the green
window stays unlabeled until the table implementation passes the gate —
the table version is the final version; the lifted lab rules
(`ducklidar.rules`) are demoted to a **validation oracle**, run only to
produce reference labels for the equivalence diff, never to ship output.

Task order below is dependency order: the evidence tasks (5.1–5.2)
precede the first labels (5.3) because 5a's ground seeds are "vendor
class agreeing with the cloth" — the ground surface must exist first.

Outputs: three files — 5.1 writes data/stage5/cells.parquet (inspectable
evidence; 5.2 adds its ground-trust columns into it), 5.8 writes data/stage5/forest.parquet, and 5.11 writes
data/stage5/labels.parquet (pid i64, label i8 0..9, settled i8 0..3 —
the contract consumed by later stages; see
[../tables.md](../tables.md)); 5.3–5.10 are in-run intermediates.

## Evidence (feeds both levels)

## Task 5.1 — cell evidence → `cells` (`data/stage5/cells.parquet`)

*`dl.cell_evidence`.*

**The Problem:** the rules think on rasters, not on millions of
individual points.

**The Solution:** one `GROUP BY cell` over the working table →
per-1 m-cell aggregates (n, min/max z, split share, class counts).

**The Result:** the cell tables the rules think on — a 2 km window is
~4 M cells, megabytes.

**Optimize?** It's one streaming pass; extend columns as operators need
them (per-flight-line stats for movement evidence) rather than adding
passes.

**Output:** `data/stage5/cells.parquet` — `cell i64, n i64, min_z f64,
max_z f64, split_share f64, n_ground i64, n_water i64` (one row per
occupied 1 m cell; persisted so the evidence is queryable in DuckDB
between runs; rebuilt when the store changes). `max_z` is the measured
DSM per cell — the top surface (highest return, noise excluded);
occupied cells only. A filled DSM over returnless cells would be a
model, not a measurement — added only if a consumer needs it.

## Task 5.2 — ground surface → `cells` (adds columns to `data/stage5/cells.parquet`)

*`dl.ground_cells`.*

**The Problem:** the windowed rules ran the cloth per window, leaving
per-window cloth seams.

**The Solution:** the cloth simulation dropped once on the whole area's
cell-minimum surface; fill; per-point height-above-ground is then a join
by cell id. Under a roof the cloth sees nothing; for mapped buildings
the honest datum is the ring of ground around the footprint.

**The Result:** one seam-free ground surface for the whole area —
ground_z per cell (into cells.parquet) + the filled surface grid
(numpy, in memory).

**Optimize?** The cloth runs on ~4 M cells — seconds to minutes; nothing
to chase.

**Output:** three ground-trust columns written into
`data/stage5/cells.parquet` — `ground_z f32` (the bare-earth value),
`ground_measured bool` (the cloth kept actual evidence in this cell —
value is measurement, not interpolation), `ground_dist f32` (metres to
the nearest measured cell, from fill's distance_to_evidence; the honest
uncertainty proxy — grows under wide roofs), and `ground_fp f32` — for
cells inside a MAPPED building footprint, the building's single ground
datum computed by `dl.building_level` from the ring of measured ground
around the footprint (the laser never sees ground under a roof; the
ring is the honest evidence — one number per building, stamped into all
its cells); null outside mapped footprints, trust fields per
building_level's conventions (ring evidence + tilt check), fixed at
implementation — plus the full filled surface grid (numpy, in memory).

## Level 1 — FIRST LABELS (was 5a)

## Task 5.3 — first labels → `priors` (a VIEW in the store db, no file)

*The 5a level; decided 2026-08-02.*

**The Problem:** every point needs a first proposed name before
revision can argue about it — and the thresholds must live in exactly
one place.

**The Solution:** threshold expressions over `points ⋈ features`: foliage
proposal (split pulse + scattering), roof/wall-like (planarity + normal),
ground seeds (vendor class agreeing with the cloth), water seeds (vendor
class 9). Computed in the scan, never materialized; thresholds are the
rules' named constants, living in exactly one place — the view text.

**The Result:** proposals every 5b task reads, at no storage cost —
the `priors` VIEW in the store db (a definition; no rows stored
anywhere).

**Optimize?** Free by construction. The open item is the threshold *set*
(walkthrough with Kaveh) — not performance.

**Output:** the `priors` VIEW in the store db (a definition — no rows stored), columns:
`pid i64` · `foliage_prop bool` (pulse split: echo_ratio < 1, AND scattering above SCATTER_MIN) · `roof_like bool` (planarity above PLANAR_MIN AND normal near-vertical: high nz) · `wall_like bool` (planarity above PLANAR_MIN AND normal near-horizontal: low nz) · `ground_seed bool` (vendor class = 2 AND z within the cloth-agreement tolerance of the cell's ground_z) · `water_seed bool` (vendor class = 9).
What the columns mean, in plain words: **foliage** = the leafy matter of
plants — leaves, twigs, branches of trees and bushes. `foliage_prop` is
the *foliage proposal*: "this point looks like plant matter, pending
revision." It fires only when two independent signals agree — the laser
pulse **split** into several echoes (it clipped a leaf and kept going;
solid surfaces echo once), and the point's neighbourhood is a fuzzy 3-D
**scatter** rather than a flat plane. Either signal alone lies sometimes
(wires split pulses; roof edges scatter) — demanding both is what makes
it a usable proposal. `roof_like` / `wall_like` = flat neighbourhood
with the surface facing up / sideways; `ground_seed` / `water_seed` =
the vendor said ground (and the cloth agrees) / said water. All five are
Level-1 proposals: cheap opening bids that Level 2 exists to overturn —
in the reference run ~31 M proposed-foliage points entered the contested
zone, and the silos, masts and tank lattices all lost their foliage
claims there.

The rules, one per column (over `points ⋈ features` on pid, plus the
cell's ground for the seed; planarity = (√e2−√e3)/√e1, scattering =
√e3/√e1 — the PDAL-convention formulas over the stage-2 basis):

| column | rule | constants (value — source) |
|---|---|---|
| foliage_prop | `return_number < number_of_returns AND scattering > SCATTER_MIN` | SCATTER_MIN = 0.2 — `rules.pipeline` |
| roof_like | `planarity > PLANAR_MIN AND nz > NZ_ROOF` | PLANAR_MIN = 0.7 — `rules.pipeline`; NZ_ROOF ≈ 0.9 — **new constant, ▶ to be measured at the gate** (the lab had no verticality feature) |
| wall_like | `planarity > PLANAR_MIN AND nz < NZ_WALL` | PLANAR_MIN = 0.7; NZ_WALL ≈ 0.3 — **new, ▶ same** |
| ground_seed | `classification = 2 AND the cloth kept the point` (the CSF keep mask itself — the lab's own construction, not a tolerance re-derivation) | cloth constants: cloth_resolution = 1.0, CSF_RIGIDNESS — `rules.pipeline` |
| water_seed | `classification = 9` | — |

Two honesty notes: NZ_ROOF/NZ_WALL are the only new constants in 5a —
everything else is the rules' verbatim; and `scattering ≤ SCATTER_MAX`
(0.3) pairs with PLANAR_MIN in the lab's building test — whether the
roof/wall proposals also carry that conjunct is settled when the gate
measures both variants.

## Level 2 — REVISION (was 5b)

The first level proposes; this level overturns proposals wherever
regional or structural evidence contradicts them — nothing is final
until revision has run.

## Task 5.4 — water bodies (`dl.water_cells`, `dl.flood_bodies`)

Finding bodies of water so the computer knows where the water surface is.

**1. Find the seeds (the starting points).**
*The Problem:* water is dark and flat — laser pulses mostly don't come
back from it. *The Solution:* look for cells where returns are very
sparse and all near the ground; group them; groups of at least 400 m²
are definite water, and a local-sparsity vote catches tricky pockets
(water beyond a bridge deck) the big groups miss. *The Result:* seed
cells, in components.

**2. Flood (fill in what the laser missed).**
*The Problem:* water hidden under docks, boats and piers was never hit —
but it is still part of the water body, and the floating rules depend on
it being included. *The Solution:* each credible body expands outward
through every cell whose ground surface sits within 0.4 m of the body's
level. Under a dock the surface interpolates flat → floods; at a beach
the ground climbs → the flood stops exactly at the shoreline. Holes left
inside (a hull the cloth settled on) are filled — once, on the union of
all bodies, never per body (a ring-shaped component would fill the land
it encircles; measured: 81 % of the old area flooded before this fix).
*The Result:* complete water bodies, shoreline to shoreline.

**3. Figure out the water height (levels).**
*The Problem:* laser shadows beside tall buildings look like dark water;
give such a fake body its own (land-height) level and it floods the
surrounding land. *The Solution:* a body's level is the median height of
its low seed points — but a body is only trusted with a level if vendor
data confirms water there (class-9 evidence above a threshold). And
because tides move between airplane passes, consumers read heights per
**(body × flight line)** — in the real data the sea stood 60 cm higher
during one pass than another. *The Result:* one flood level per credible
body, and a conditioned levels table for every rule that asks "how high
is the water here, as measured by this pass?" In-run: body id per cell
(int, 0 = dry) + the conditioned levels table (body × flight line),
held for consumers within the run.

**Optimize?** `ndimage.label` on 4 M cells is instant; the flood loop
runs once per leveled body (a handful), level-less specks pass through
untouched.

**Output:** in-run — `body i32` per cell (0 = dry), plus the levels
table `body i32, point_source_id u16, level f64` (one row per water
body × flight line).

## Task 5.5 — bridge corridors

**The Problem:** where a bridge runs, and how wide, is a map fact the
points alone don't carry.

**The Solution:** map ways buffered each at its own width (lanes tell
it); deck claimed at any height inside the corridor; shaft columns under
the deck join the bridge. Map facts are queryable (Overture db /
`read_json_auto`) → `ST_Buffer` → cells.

**The Result:** bridge corridor cells — a corridor mask per cell, in
memory.

**Optimize?** Vector ops on hundreds of ways — negligible.

**Output:** in-run — `corridor bool` per cell (with the way's width
behind it).

## Task 5.6 — layer construction: each class states its claim

**The Problem:** a point can be claimed by several systems at once (a dock
point is floating AND small AND near-ground), and some claims must be
*impossible* rather than merely unlikely — terrain cannot exist inside a
water body, no matter how the geometry reads.

**The Solution:** every class gets TWO rules, exactly as the lab wrote
them: a **LOCAL rule** — what evidence claims this class here — and a
**FORBIDDEN rule** — what disqualifies it regardless of the local
evidence. Ground's forbidden rule is the WHOLE water body (Kaveh's chain:
body → ground → floating; two narrower zones were tried and measured
insufficient, and the cost, the survey's tidal seabed at ~2 % of ground
cells, is accepted and bounded). Floating's local rule is solid matter
above the body's plane. The layers are independent, orderless and
overlapping *by design* — a point may end up with three claims, and
nothing here resolves them.

**The Result:** one boolean claim per class per point — the stack the
revisions argue over and the decision (5.11) resolves. This is where the
ground CLAIM is born; it is not yet the ground LABEL, and the seeds in
5.3 were not either.

**Optimize?** Mask algebra over columns already computed — negligible.

**Output:** in-run — `claim_<class> bool` per point for the ten classes
(ground, veg_low, veg_high, building, water, small, floating, bridge,
on_bridge, glazing).

**`placed` — a vertical support gate, and still out of the decision.** The
eleventh layer, `placed` (objects standing on the ground — trucks, cars,
containers), is computed and then dropped from `DECIDE`, because in 2-D a
truck and a one-storey eave are the same reading: ground below, roof above
(the lab's own comments, `rules/layers.py` l.182-185 and l.609-612 — "ONE
STOREY EAVES REMAIN THE KNOWN LEAK... the component interior test that
solves them fully is context, not a per-point rule (v2)").

The stage 2 voxel table supplies that context: `dl.column_support` asks
whether a column is CONTINUOUS from the floor band up to the claim's height,
and a claim is kept only where it is. Measured on the old area's WORK box —
A (`placed` dropped, shipped) 0.832 building IoU; B (`placed` at spec 3,
ungated) 0.816; C (B + the gate) 0.829 at `min_gap = 0.5` m, monotonically
worse as the gap loosens. The gate refuses 65 % of B's claims and the
refused columns are 1.5× enriched in survey-BUILDING returns, i.e. it strips
exactly the eave rims it was built for — but **-0.003 remains, so `placed`
still does NOT enter the decision.** Full numbers and the `min_gap` curve:
[stage 2 — the voxel table](stage2-voxels.md#what-it-closed-the-eave-leak-partially).

## Task 5.7 — building outlines and containment

**The Problem:** building evidence is scattered points, but a building is
an AREA: a courtyard, an atrium, a shadowed inner wall belong to the
building even where the point evidence is ambiguous or absent.

**The Solution:** polygonize outlines from ALL building evidence
together, then claim containment cell by cell inside them — the lab's
"assembly: outlines from all building evidence, then containment claims"
block. The map footprint is the authority where one exists; the traced
outline covers what the map never mapped.

**The Result:** building claims extended from scattered evidence to whole
areas — and the outlines that the roof-tree fix (5.9) and the footprint
veto (5.10) both test against.

**Optimize?** Raster polygonization over ~4 M cells; the reach is one
outline, so it stays regional and cheap.

**Output:** in-run — outline polygons + `in_outline bool` per cell.

## Task 5.8 — the support forest → `forest` (`data/stage5/forest.parquet`)

*Multi-source Dijkstra — metric, not hop-counting; decided 2026-08-02.*

**The graph is G′ = (V, E ∪ B) — always.** The union of stage 2's kNN
edges with its MST edges is *part of this algorithm, not an option*
(Kaveh, 2026-08-02): G alone is disconnected by the 2.5 m cap, so a vertex
whose component holds no seed has no path to any root and the forest is
undefined there — measured 0.78 % of the reference tile, and they are the
boats, detached crowns and isolated roofs whose support is precisely the
question. `dl.forest_graph()` reads both tables and refuses to run without
the bridges. (The earlier "Island-bound" run, without the bridges, is not
an alternative — it is the measurement that proves the union is required:
buildings fell from 0.9992 to 0.9735 and 0.78 % of vertices became
unreachable.)

**Measured 2026-08-02 — the walk on that graph is "Groundwalk"** (geodesic
metres, roots = ground only): IoU 0.9938 on the oracle's refine_veg and
0.9992 on refine_building. Rejected: *Stepcount* (hops instead of metres —
vegetation 0.7144, because a step inside foliage is 5 cm and a step across
a gap 2.5 m); *Three-source* (ground+water+bridge roots) scores 0.9880 on
buildings but 0.8799 on vegetation against this reference — though that
comparison is unfair, since the reference was itself produced ground-only.
Three-source remains the right walk for the attachment questions
(floating, on-bridge, superstructure): the same function, run a second
time with the other root set. Costs: [stage5-cost.md](stage5-cost.md).

**The Problem:** many rules ask the same question — what is this point
attached to, and through what? — and the windowed pipeline answered it
seven separate times.

**The Solution:** the spine. Root sets — ground, water bodies, bridge
decks — expand simultaneously over `knn.parquet`, hop by hop, through
solid matter; first system to reach a point claims it. Each point
records (root, hop depth, what the path crossed — wall-like /
foliage-like from 5a).

**The Result:** one attachment table that subsumes seven windowed
operators (path-to-ground acquittal, corridor stranding, object
topology, path-entry refinement, floating roots, deck columns,
superstructure). In-run: pid → (root system, hop depth, path
composition); may be staged to a transient parquet if RAM demands,
never a contract file.

**Cost model:** one hash-join of the frontier against `knn` ∪ `mst` per
hop; paths to ground are tens of hops (structure height ÷ 2.5 m), so tens
of streamed passes — or an in-RAM frontier walk at ~2 GB. **Optimize?**
Decide streamed-vs-RAM by measuring the first implementation; both are
correct.

**Who reads it (2026-08-03):** exactly one rule — `dl.path_shares` feeds
the path-entry refinement with `crossed_wall` / `crossed_foliage`
(measured 0.9919 / 0.9969). `pred`, `root`, `system`, `dist` and `depth`
are written but consumed by nothing yet; the six other operators this walk
was meant to subsume are still the lab's separate blocks. See
[../forest.md](../forest.md) for the full accounting — the generality is
not yet earned.

**Output:** `data/stage5/forest.parquet`, bound in the store db as the
view **`forest`** — **the forest itself**, one row per vertex (persisted, decided 2026-08-02: it costs 144 s and 11 GB to
rebuild, and "what holds this point up" is a question worth asking without
recomputing).

**Definition of what is stored.** Let G′ = (V, E ∪ B) be the completed
graph (stage 2 task 2.6) and R ⊆ V the seed set, partitioned into root
systems R = R_ground ⊍ R_water ⊍ R_bridge. Multi-source Dijkstra from R
gives, for every v reachable from R:
d(v) = min_{r∈R} d_{G′}(r,v), root(v) = the argmin seed, and pred(v) =
v's predecessor on that shortest path (⊥ for v ∈ R). The arc set
A = {(pred(v), v)} is a **spanning forest F = (V, A) of G′ rooted at R** —
the shortest-path forest. Its trees are T_r = {v : root(v) = r}, one per
seed vertex, and **each tree carries the label of its root's system**:
label(v) = sys(root(v)).

| column | type | meaning |
|---|---|---|
| pid | i64 | the vertex |
| pred | i64 | its parent in F — the arc set; −1 at a root and where unreached |
| root | i64 | pid of the seed that reached it — **the tree's identity** |
| system | u8 | **the tree's label**: 0 ground · 1 water · 2 bridge deck · 255 unreached |
| dist | f32 | d(v), geodesic metres along the tree to its root |
| depth | i32 | hops to the root (0 at a root, −1 unreached) |
| crossed_wall | i32 | wall-like vertices on the path — the building evidence |
| crossed_foliage | i32 | foliage-like vertices on the path — the crown evidence |

`pred` is what makes this a stored forest rather than a stored labelling:
the arcs are recoverable, so any consumer can walk a vertex back to its
root without re-running Dijkstra. Written once per root set — Groundwalk
(R = R_ground, which reproduces the oracle's refinement) and the
three-system walk (R = R_ground ⊍ R_water ⊍ R_bridge, which answers
attachment) — distinguished by a `roots` column value or by file suffix,
fixed at implementation.

## Task 5.9 — the conviction trial (REDESIGNED 2026-08-03)

**What it decides.** Glass looks exactly like foliage to the laser — a
window scatters the beam the way leaves do, so the local shape features
say "vegetation" for a glazed facade. No per-point feature separates them.
The test that does: **can this point reach the ground without passing
through building matter?** A tree can (down the trunk); a window cannot
(the only way down is through the building).

Bars, both swept by the lab: **unreachable** → 90 % survey-building;
**reachable only via ≥ 60 m of detour** → 100 % (30–60 m was 77 %, so the
bar sits at 60).

### Why the old design was wrong

It ran a full multi-source Dijkstra over **all 31.4 M contested points**
(74 % of the tile) on a rebuilt graph — **16.7 GB, the largest memory
consumer in stage 5** — while task 5.8 had just computed a forest that
already answers the question for most of them, and then ignored it.

### The redesign: acquit first, try only the remainder

**Step 1 — free acquittal from the forest (no trial).**

> **Theorem.** If a point's shortest path to ground in the kNN graph
> crosses **no building vertex**, that path survives the deletion of the
> buildings intact, so the point is reachable in the sub-cloud graph with
> geodesic ≤ its forest `dist`.
>
> *Why the path transfers:* if edge (u,v) is a kNN edge and both endpoints
> survive, v is still among u's ten nearest **survivors** — removing
> points can only promote a neighbour, never demote it. So every
> full-graph edge between survivors exists in the rebuilt graph, and the
> whole path with it.

Therefore `crossed_building == 0 and dist < 60` ⇒ **acquitted**, at zero
cost, from a table 5.8 already produced.

Two requirements this puts on 5.8's walk, both cheap:
- `cats` must count the **building CLASS** (what 5.9 deletes), not
  "wall-like" shape (what the refinement uses) — a different `cats` array,
  same walk;
- it must run on **`knn` alone, without the MST** — an MST edge is not a
  kNN edge and would not survive into the rebuilt graph, so a path using
  one proves nothing. This sits exactly right with 5.9's standing rule
  that the MST is forbidden here.

**Step 2 — the trial, for the defendants the forest could not clear.**
Points with `crossed_building > 0` genuinely need the test, because a
shortest path through a building does *not* prove no alternative exists (a
crown resting on a roof takes the roof as its shortest way down while
still being able to descend its own trunk). For those only: rebuild kNN on
the surviving non-building points — **rebuilt, never filtered**; measured,
filtering `knn.parquet` leaves 553,587 points wrongly unreachable and
flips 103,180 verdicts — then run the same `support_forest` machinery with
`limit=60`, and convict on unreachable or `dist ≥ 60`.

**Step 3 — the verdict table.** Because step 2 is the same operator as
5.8 on a different graph, its output has the same shape, and 5.9 can emit
a forest rather than a bare verdict — giving `pred`/`root`/`dist` a second
consumer instead of leaving them unread.

### The measurement — step 1 clears 78.1 % [measured]

Old area, owned tile, 42.7 M returns; forest over `knn` alone, roots =
the oracle's ground layer ∩ `sub`, `cats` = the deleted building class.

| | points |
|---|---|
| defendants (`sub`) | 31,466,173 |
| **acquitted free by the forest** | **24,564,096 (78.1 %)** |
| remaining for the trial | 6,902,077 |

### …and it buys nothing. NOT BUILT — measured 2026-08-03

The clear rate is real and the theorem holds. The *inference* drawn from it
here was wrong, and the correction matters more than the original claim:

**The trial's cost is graph-bound, not defendant-bound.** Read
`glass_geodesic_blocks`: per 250 m block it builds `cKDTree(p).query(p, k=11)`
over every SURVIVOR (`sub`) in block+halo, then one Dijkstra. The prefilter
acquits *defendants*; it removes nothing from `sub`, because deleting survivors
would change the graph and the answers. So the tree and the walk are identical
whether 31.4 M points or 6.9 M points still need a verdict.

The only saving available is skipping a block whose core holds no un-cleared
defendant. **Measured on the old tile: 16 of 16 occupied blocks contain
building points** (min 1,514, median 655,136), so every block has un-cleared
defendants and none is skippable. The saving is exactly zero.

Two further reasons it would not have paid even so:

* the prefilter needs a forest over **kNN alone** with **building-class**
  `cats`. `forest.parquet` is over kNN ∪ MST with shape-based `cats`, so
  "step 1 is free, 5.8 already computed it" is false of the forest on disk — it
  would need its own ~16.7 GB walk, *more* than the 14.1 GB trial it replaces;
* the run's peak is the **decide** step at 15.2 GB, above the geodesic's 14.1,
  so shrinking the geodesic cannot lower the peak at all.

**Status: WONTFIX.** The conviction keeps trying all 31.4 M defendants. Revisit
only if `sub` itself can be shrunk (which changes verdicts, so it needs its own
gate) or if a future area is sparse enough that whole blocks come out
building-free.

**Soundness, checked against the real trial's verdicts, not argued:** of
the 24.56 M points cleared, **0 were convicted by the trial and 0 were
unreachable in it**. The theorem holds on every point.

**Use a 5 m margin — `dist < 55`, not `dist < 60`.** The forest's `dist`
and the trial's `geo` should be *equal* for a cleared point (its path
survives deletion intact), and they are not: `geo` runs up to **1.04 m
longer**, cause not yet identified. That is the unsafe direction — a point
with `dist` just under the bar could have `geo` just over it. No point
actually did, but the margin costs only **12,095 points (0.05 % of the
cleared set)** and buys 5× the observed disagreement, so there is no
reason to run without it. Points in [55, 60) simply fall through to the
trial, which decides them properly.

**One trap, paid for once.** Seed the forest from the **ground layer**,
not from decided ground (`label == 0`). The two differ by 5,364 points,
and seeding from the decision produced 1,437 points cleared by the forest
yet unreachable in the trial — which read as a broken theorem and was
purely a root-set mismatch. The prefilter's roots must be exactly the
trial's roots (`glass_geodesic`'s `ground & sub`) or the comparison is
meaningless.

**Output:** `pid i64, verdict i8, geodesic f32` — verdict 0 acquitted by
the forest, 1 acquitted by trial, 2 convicted unreachable, 3 convicted by
detour. The provenance is kept because "which step decided this" is the
same kind of fact as `settled` in the labels table.

## Task 5.10 — map vetoes

**The Problem:** some calls (the silos, plant, tank rules) turn on what
the map says stands there.

**The Solution:** point-in-polygon and buffer tests against
footprints/landuse — spatial joins, DuckDB-native.

**The Result:** map-backed vetoes — per-point veto overrides, in
memory. Negligible cost.

**Output:** in-run — `pid i64, veto_label i8` for overridden points
only.

## Task 5.11 — decision & residue → `labels` (`data/stage5/labels.parquet`)

**The Problem:** a point can sit in several layers at once, and some
points end up in none.

**The Solution:** ordered resolution of the layer stack per point
(~25 s measured); then unlabeled residue takes its nearest labelled
neighbour through the mesh. Row-wise + one kNN join.

**The Result:** one final label per point, written to the stage's
contract file `data/stage5/labels.parquet`, with `settled i8` — label
provenance as confidence: 0 = single claim (uncontested) · 1 = settled
by specificity · 2 = settled by neighbour vote · 3 = residue copy of
nearest labelled neighbour — filled by cause: it records HOW each label
was decided, not a pseudo-probability; the gate compares labels only,
provenance is reported. Nothing to optimize.

**Output:** THE contract file `data/stage5/labels.parquet` —
`pid i64, label i8 (0..9), settled i8 (0..3)`.

## The gate

Whatever implements 5a+5b must reproduce the lab reference labels
(42.7 M points) ≥ 99.5 % by pid/coordinate join. Measured precedent: the
lifted rules on store reads scored 99.79 %; the residual was quantified
(kNN tie-breaking), not waved through.
