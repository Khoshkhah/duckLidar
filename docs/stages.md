# Every stage, every task

The survey-to-scene pipeline as ducklidar implements it: what each stage
does, and each task inside it, explained as simply as possible. Graph
vocabulary is used where it *is* the simple explanation. The theory (why
stages are ordered by data dependency, why partitions never leak into
results) is in [pipeline.md](pipeline.md); the labeling-rule inventory is
in [rules-operators.md](rules-operators.md); every table's columns and
file names are in [tables.md](tables.md).

Everything below is a table in, a table out. A point is a row; `pid` is
its permanent name. One stage's output is the next stage's input, and any
stage can be rerun alone.

---

## Stage 0 — the store

*Turn delivered survey files into one queryable table.*

- **Convert** (`to_parquet`): each LAS/LAZ/zip becomes a Parquet file,
  points sorted into 50 m cells so that nearby-in-the-city means
  nearby-in-the-file. The sort is the spatial index: a window query skips
  whole row groups by their min/max statistics instead of scanning.
- **Assign `pid`**: every point gets its row number in its file, forever.
  All later results (features, labels, instances) are tables keyed by
  `pid` — they never need to be stored with the points.
- **Read windows** (`read`): give a box, get numpy columns — from one
  file, several files at once (a window near a file boundary reads both
  sides), or a COPC URL. Files that don't touch the box cost a header.
- **Know the survey** (`describe`, `census`): what classes, fields,
  colours, returns the survey actually carries — checked before anything
  is planned, because a survey without a field lacks every analysis
  built on it.

## Stage 1 — the working table

*From "whatever the survey delivered" to "the project's points": clip the
store to the target area (plus the 60 m working margin), drop the noise
classes, attach everything a point knows about itself.*

- **Clip**: the area filter lives here and nowhere else — later stages
  read this table instead of each re-deriving the boundary. On the
  reference window this keeps 42 % of the delivered points.
- **Row-wise attributes**: same expression on every row —

- **Echo ratio**: did this laser pulse split into several returns?
  Foliage splits pulses; solid surfaces don't. Column ÷ column.
- **Greenness**: colour index from RGB, where the survey is colourised.
- **Vendor class**: the survey's own per-point label, carried as
  evidence, never as truth.
- **Discipline**: any threshold used here must be a constant or computed
  from the whole table — a percentile computed inside a window silently
  turns a row-wise label into a window-dependent one.

## Stage 2 — the graph and the shape features

*Build G = (V, E, w) and the local shape of every neighbourhood — the one
expensive spatial query, run once.*

- **kNN** (`knn`): for every node, its 16 nearest nodes and distances.
  Everything else in this stage derives from this single query.
- **Shape features** (`shape_features`): the covariance eigenvalues of
  each node's neighbourhood, normalised, plus the normal's verticality.
  Flat-horizontal (roof), flat-vertical (wall) and scattered (foliage)
  neighbourhoods separate here. PDAL-compatible mode: self-inclusive
  neighbourhood, sqrt-eigenvalue ratios (measured equivalent).
- **Edges** (`local_edges`): keep each node's k ≤ 10 neighbours within
  2.5 m as directed weighted edges `(src, dst, len)`, stored as
  **`knn.parquet`** — that table IS *G*. Stage 2 also writes
  **`mst.parquet`**, the Euclidean MST; *G* alone answers "what is one
  physical piece", and *G′* = `knn` ∪ `mst` answers "what path exists".
  Node ids are `pid`s, so both join across runs and areas with no shared
  numbering.
- **Partitioning** (`balanced_boxes`): the query needs points co-resident
  in memory, so the area is swept in data-driven partitions (recursive
  median splits — every leaf ≤ N points) with halos wider than any query
  radius. Halo ≥ reach ⇒ results identical to an impossible
  whole-area-in-RAM run; each partition keeps only its own points'
  answers ("the owner writes").

## Stage 3 — global constants

*The shelf for values with one scene-wide answer — currently empty.*

- Its first candidate, a pooled water level, was measured and deleted:
  the survey's flight lines stand 2.7 m apart in sea level across two
  days (tide), so "the" water level doesn't exist. A statistic is only
  informative about the population it is conditioned on; water level is
  per (water body × flight line), computed where used (stage 5).
- The stage keeps its discipline: no rule may privately measure a
  scene-wide constant from its own partition.

## Stage 4 — connected components

*Which nodes form one physical piece: components of G, once, globally.*

- **Union-find over streamed `knn.parquet`** — on `knn` ALONE, never the
  union with `mst`, because on *G′* there is one component by construction
  and the 10,499 islands vanish. One parent entry per node
  (4 bytes × |V|), edges flow past and are never resident. Batched unions
  retry colliding writes until no pair is split.
- **Output**: `pid → component`. A boat is a component; a crown is a
  component; terrain plus everything path-connected to it is one giant
  component (99.2 % of nodes on the reference window).
- Because every edge is seen at once, a bridge whose ends lie in
  different survey files is one component by construction — no border
  repair exists because no border was cut.
- Components are geometry, not meaning: "touching" ≠ "same object".
  Cutting them by class is stage 6.

## Stage 5 — labeling

*Name every node: ground, water, roof, foliage, glass, bridge, floating.*

Two levels, run in order: **first labels**, then **revision**. The first
level proposes; the second level overturns proposals wherever regional or
structural evidence contradicts them. Nothing is final until revision has
run.

**Then the map speaks** (`map_labels.map_claims`). Where Overture draws an
outline, that outline is evidence about what the points inside it ARE — one
named method per mapped type, each of which may overwrite only the labels it is
allowed to, and each carrying a physical guard because a footprint says nothing
about height. Two rules run the other way as vetoes: **no footprint, no
building** (812,837 points, 71.6% of them under 6 m above ground), and nothing
vegetal grows on open water (sailcloth returns like foliage; the polygon is
eroded 8 m first so shore crowns survive).

### 5a — first labels (a VIEW in the store db — computed in the scan, never materialized; Kaveh, 2026-08-02)

Threshold tests over columns that already exist (`points ⋈ features` on
pid). Each point gets a *proposal* and each system gets its *seeds*:

- scattered neighbourhood + split pulse → foliage proposal;
- planar + vertical normal → wall-like; planar + horizontal → roof-like;
- vendor ground class (agreeing with the cloth) → **ground seed**;
- vendor water class → **water seed**; low intensity + low + flat joins it;
- greenness separates plant matter from grey structure wearing a
  foliage-like shape.

### 5b — revision (regional and structural; the measured judgment)

The tasks, in dependency order:

1. **Shared evidence — cell tables** (`cell_evidence`): one
   `GROUP BY cell` → per-1 m-cell aggregates (min/max z, split share,
   class counts, movement between flight lines, overhead). The rasters
   the rules think on are cell tables — a 2 km window is ~4 M cells,
   megabytes.
2. **Ground surface** (`ground_cells`): one cloth simulation over the
   whole area's cell-minimum surface; height-above-ground joins back to
   every point by cell id.
3. **Water bodies**: connected components *of cells* grown from the
   water seeds (stage 4's union-find at raster scale); per-body level =
   `GROUP BY (body, flight line)` — the conditioned water level.
4. **Bridge systems**: map ways buffered each at its own width, deck
   claimed at any height inside the corridor; solid column tops under
   the deck join the bridge structure.
5. **The support forest — every point attached to what holds it up.**
   The task you won't find under one name in the code, but it is the
   spine of revision: take the mesh (*G′* = `knn` ∪ `mst`) and grow, from the
   three *root systems* — **ground, water body, bridge deck** — a
   spanning forest through solid matter. Every elevated point's tree
   tells who carries it: attached through a wall → building system;
   through a trunk with a path that avoids building matter → crown
   (the acquittal — 9.7 M points on the reference tile); through the
   deck → on-bridge; floating matter roots in the water body, and its
   superstructure (masts, rigging) follows its vessel. In the lifted
   code this forest appears in pieces — "support graph: path-to-ground",
   "generalized blocker set: every structural system", "object
   topology", the body→ground→floating chain — but they are one idea:
   **labels flow down the forest from the roots; a point's label must be
   consistent with its attachment.**
6. **The conviction trial**: contested foliage/building/glass points
   (31 M on the reference tile) judged on the mesh by neighbourhood
   verdicts — planarity, greenness, return position — the kNN court.
7. **Map vetoes**: the footprint veto (not-green "foliage" deep inside a
   mapped building → the silos), the plant and tank rules (industrial
   landuse), each a spatial join against the map tables.
8. **The decision**: resolve each point's surviving claims by
   specificity order — one label per point (measured: ~25 s).
9. **Residue attachment**: whatever is still unlabeled takes its nearest
   labelled neighbour through the mesh — the forest's last leaves.

**Reference implementation**: the lab's rules, lifted verbatim into
`ducklidar.rules`, remain the measured authority — 5a and 5b run
interleaved inside them. Every task moved onto tables must reproduce them
on the reference tile (the equivalence harness: bit-identical for the
lift; 99.79 % for the store-fed adaptation, residual quantified as kNN
tie-breaking). Per-operator inventory and the table plan:
[rules-operators.md](rules-operators.md).

## Stage 6 — instances

*Names become things: this house, that tree, that boat.*

- **Class-induced subgraph components**: restrict G to edges whose both
  endpoints share a class, take components. A foliage node touching a
  roof node no longer connects them — the tree/house weld of stage 4
  separates by construction.
- **Map identity for mapped classes**: a building instance is its
  footprint's way id (identity inherited from the map — the same id on
  every side of every partition, so cross-area fusion is a key match,
  not geometry matching).
- **A different cut per type.** Connectivity welds, and no single operator
  unwelds everything. Each type is separated by the evidence that actually
  distinguishes it — see [type-guards](design/type-guards.md):

  | type | the cut | why connectivity fails |
  |---|---|---|
  | floating | `raft_split` — h-maxima watershed over the pontoon | boats touch within 1 m of the deck |
  | ground | Overture land use, buffered road centrelines | a tennis court is not a component |
  | building | Overture footprints, grown 3 m for eaves | a terrace shares walls |
  | floating | mapped pier ways, then free components | the map cuts what height cannot |
  | veg_high | the **support-forest root** | one tree, one trunk, one root |
  | small / on_bridge | `car_split` — watershed over the parked surface | parked cars touch |

  The last two are deliberately different operators. The forest root is exact
  for a tree and *shreds* a car, because a roof reaches the ground by several
  paths around the body: applied to cars it gave 1,974 instances of median
  1.9 x 1.0 x 0.8 m, a quarter of a car each.
- **The type, decided once.** Every finer name — car, boat, dock, tree, lamp,
  moorage, the twelve ground surfaces — is decided HERE and written to
  `instance_table.type`. Stage 7 reads it and never re-derives one.
- **Fusion**: instances found by different runs/partitions are unioned
  through same-class edges and shared map ids — union-find again, over
  instances instead of nodes.

## Stage 7 — scene build

*Instances become 3D models that only claim what was measured.*

- **Footprints**: from the map where mapped (`pier_ways`, Overture
  extracts), traced from the instance's own points where not
  (`scene.trace_footprint`).
- **One named model per type**, in `objects`. No model calls another; where two
  types share geometry they share a PRIMITIVE (`box_local`, `cell_prism`,
  `hull_soup`, `drape`, `scene.footprint_prism`). An empty return is a real
  answer: "this instance does not earn geometry".
- **Buildings**: the footprint extruded from the ground to the P95 of the
  building's own returns, with roofer's LoD2.2 roof on top where a solid
  exists. Roofer's WALLS are not used — they have holes (one silo's had a
  64 deg gap; 6 of 82 buildings get under 85% coverage) — and Overture's
  `height` attribute is not used either, being wrong in both directions.
  `scene.checked_walls` is a real wall trial but is not on this path: the wall
  comes from the surveyed outline, so there is nothing to put on trial.
- **Bridges**: one connected bridge, one deck, SWEPT along its chained
  centreline (`objects.bridge_ribbon`) rather than rasterised from a polygon —
  a footprint is larger than the evidence wherever the laser was thin, and
  dropping the unsupported cells turns a deck into a field of 2 m cubes.
  Supports every 60 m, guard rails at 1.26 m, street lights by
  `objects.bridge_lights` (whose datum is the deck, not the ground).
- **Canopies and crowns**: a trunk (`solid`) under a crown (`slab`), with the
  crown returns exported per tree so a canopy is selectable rather than one
  anonymous cloud.
- **Terrain**: the stage-5 ground — but read through
  `map_labels.ground_without_bridges`, because stage 5's grid IS the deck under
  a bridge (median 29.69 m inside a footprint).
- **Kinds**: every part states what the sun does with it — `solid` stops light,
  `slab` lets it under, `glass` transmits, `surface` only receives. A type may
  ship several: a tree is a `solid` trunk under a `slab` crown.
- **Merge**: all instances' models into one scene for the area.

## Stage 8 — shadows

*Run the sun over the merged scene.*

- Rasterize the scene into ground (receives shadows), solid blockers,
  and slabs (things with open air beneath — bridge decks, evidenced-open
  carports the low sun shines under).
- Cast per sun position; integrate to sun-hours per square metre.
- Compute on the area plus a shadow buffer — a tower a kilometre outside
  the area owns someone's morning light.

---

## After stage 1, there are no tiles

Tiles are how the survey was *delivered* (stage 0) and how sweeps *stage*
their partitions internally — never how results live. From stage 2 on,
every stage's output is **one table**: `features.parquet`,
`knn.parquet`, `components.parquet`, `labels.parquet`,
`instances.parquet`. A consumer of any stage sees one file per fact,
whatever partitioning produced it.

## The execution rule that holds it together

Stages advance **globally**: a stage runs over the whole area and
materializes its table before the next stage starts. Partitions exist
only *inside* a stage, as a memory tool, padded by halos wider than any
rule's reach — so the output of every stage is identical to a
single-machine-with-infinite-RAM run, and no partition boundary can be
found in any result.
