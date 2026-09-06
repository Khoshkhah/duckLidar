# Stage 5b — every operator that is not row-wise

The worksheet for moving the labeling rules onto the table. 5a (row-wise
priors: thresholds over `points ⋈ features` columns) already runs as SQL by
construction. Everything below is what remains — each operator with what it
decides, the evidence it reads, its reach (the halo bound today), and its
current computational form. The **Table:** slot is yours: per item, how it
should be implemented against the DuckDB/Parquet tables — or "stays
windowed", which is also an answer.

Sources: `src/ducklidar/rules/layers.py` and `pipeline.py` (lifted verbatim
from the lab). Reach = how far evidence
travels into the answer; anything with bounded reach is exact under
partition+halo, so "table vs window" is an engineering choice, not
correctness.

## A — shared regional evidence (built once per window, consumed by everything)

| id | operator | decides | reads | reach | current form |
|----|----------|---------|-------|-------|--------------|
| A1 | **cloth ground filter** (`dl.ground_filter`, CSF) | bare-earth mask from x,y,z alone | coordinates | cloth stiffness scale (~10s of m) | physics simulation over a 1 m raster |
| A2 | **DEM v1 + HAG** | height-above-ground per point | A1 + z | interpolation across voids (10s of m) | rasterize → fill → per-point lookup |
| A3 | **echo-ratio grid** | per-cell share of split pulses | return columns | 1 m cell | raster reduce |
| A4 | **density grid** | returns per cell | x,y | 1 m cell | `bincount` |
| A5 | **movement evidence** (`line_spread`) | per-cell disagreement between flight lines (things that moved between passes) | per-line rasters | 1 m cell | per-`point_source_id` raster compare |
| A6 | **overhead** | top-of-cell minus point z (what hangs above me) | z per cell | 1 m cell | per-cell max → per-point diff |
| A7 | **water body** | the flooded region: sparse-dark cells grown from seeds, holes filled; then the per-body plane | A2, A3, A4, intensity, class 9 seeds | flood fill — reach = body size (unbounded within a body) | seeded raster flood + fill |

## B — layer construction (per class: a LOCAL rule and a FORBIDDEN rule)

Mostly cell/row-wise *given A* — the regional part is inherited from the
evidence, not the rule.

| id | operator | decides | reads | reach | current form |
|----|----------|---------|-------|-------|--------------|
| B1 | **ground layer** | ground claims; FORBIDDEN inside the whole water body (tidal-seabed cost accepted & bounded) | A1, A2, A7 | body size | mask logic over rasters |
| B2 | **water layer** | water claims from the body | A7 | body size | mask |
| B3 | **floating** | solid matter above the body's plane IS floating (no movement gate inside the body); movement route outside | A5, A7, planarity/solidity | body size | mask logic |
| B4 | **building outlines + containment** | outlines polygonized from all building evidence, then containment claims cell-by-cell | building priors, A2, A6 | outline extent | raster → polygon assembly |
| B5 | **bridge corridor** | OSM ways buffered *each at its own width* (lanes tell the width; footways ~2.5 m); inside the corridor the deck is claimed at ANY height (the at-grade ramp is the bridge — that is what OSM knows and geometry cannot) | osm_bridges.json, A2 | corridor width | vector buffer → raster |
| B6 | **deck shaft columns** | solid column tops > 5.5 m under the deck → bridge structure | B5, per-cell columns | cell | column logic |
| B7 | **small / glazing / veg layers** | per-class local+forbidden tests | A2, A3, features, exg (greenness) | cell | mask logic |

## C — graph & topology revisions (the conviction/acquittal machinery)

| id | operator | decides | reads | reach | current form |
|----|----------|---------|-------|-------|--------------|
| C1 | **support graph: path-to-ground** | contested veg/building points acquitted as crowns when their 3-D path to ground avoids building matter (9.7 M points in the owned tile) | the kNN mesh, labels-so-far | path length (bounded by structure height) | graph traversal |
| C2 | **conviction zone (kNN glass/veg trial)** | 31 M contested points tried on the mesh: convicted as glass / vegetation by neighbourhood verdicts | mesh, features, exg | k-hop (small) | graph sweep |
| C3 | **roof-tree fix** | veg claims inside building outlines re-examined | B4, C1 | outline | mask + graph |
| C4 | **last-return filter (surgical)** | crown-splitting building claims removed by return-position evidence | return columns, local context | cell | mask |
| C5 | **corridor stranding** | veg-claimed points with no path out of a corridor → wires/structure | B5, mesh | corridor | graph |
| C6 | **object topology** | a small building component *ringed* by vegetation is a crown | component adjacency | component | component test |
| C7 | **path-entry refinement** (`refine_veg` / `refine_building`) | label flips where mesh-bridge entry paths disagree with the claim | mesh bridges (MST) | bridge length | graph |
| C8 | **overlap measurement** | "who overlaps whom" between systems — the measurement the tree never makes | layer stack | cell | raster compare |

## D — map-context vetoes (footprints as the outer authority)

| id | operator | decides | reads | reach | current form |
|----|----------|---------|-------|-------|--------------|
| D1 | **footprint veto** | not-green "veg" ≥ 3 m inside a mapped building → building (the silos) | osm_buildings ⋈ points, exg | footprint | point-in-polygon + depth |
| D2 | **plant rule** | not-green cube-solid veg within 20 m of a mapped building in industrial landuse → building | footprints + landuse, features | 20 m | buffer test |
| D3 | **tank rule** | veg within 20 m of a mapped storage tank in industrial landuse → building (lattice, murals) | footprints | 20 m | buffer test |
| D4 | **marina superstructure** | veg/small above floating matter inside the water body → floating (masts, rigging, gear) | A7, B3 | column | column mask |

## E — assembly & residue

| id | operator | decides | reads | reach | current form |
|----|----------|---------|-------|-------|--------------|
| E1 | **the decision** | final label per point from the layer stack, by specificity/priority order | all layers | row-wise *given the layers* | ordered mask resolution |
| E2 | **residue partition** | unlabeled residue takes its nearest labelled neighbour | mesh | nearest-neighbour | kNN assignment |
| E3 | **large vehicles out of building** | trucks etc. reclassified | movement + geometry | cell | mask |
| E4 | **DEM v2, filled by cause** | the final terrain, holes filled with their reason attached | final labels | interpolation | raster |

## The support forest (Kaveh's framing, 2026-08-02)

Several C/B/D operators are pieces of one idea: a **spanning forest of the
mesh rooted at the structural systems** — ground, water body, bridge deck —
grown through solid matter, so every elevated point is attached to what
holds it up, and labels must be consistent with attachment. C1
(path-to-ground), C5 (corridor stranding), C6 (object topology), C7
(path-entry refinement), B3 (floating roots in the body), B6 (deck
columns), D4 (superstructure follows its vessel) are all views of this
forest. **Decided (Kaveh, 2026-08-02): multi-source BFS** — every root
set expands simultaneously over `knn.parquet`, hop by hop; the first
system to reach a point claims it; each point records its root system,
hop depth, and path composition (wall-like / foliage-like crossings from
the 5a priors) — the acquittal evidence. Output: one attachment table,
`pid → (root, depth, crossed)`, subsuming the seven operators above.

## Observations for the redesign discussion

- The genuinely **unbounded-reach** operators are the two floods: A7 (water
  body — reach = the body) and B4's outline assembly. Everything else is
  bounded by a corridor, a footprint, a column, k hops, or a structure
  height — comfortably inside partition+halo, or expressible as joins.
- C1/C2/C5/C7 are **graph queries over the mesh** — `knn.parquet` already
  hold that graph globally; these are candidates for the DuckDB
  bounded-hop-join pattern (one self-join per hop) or for a masked
  union-find pass like stage 6's.
- A7's flood is a **connected-components problem on cells**: rasterize the
  seed/sparse-dark predicate to a cell table, run the same union-find used
  in stage 4, per-body planes become `GROUP BY component` medians — with
  the (body × flight line) conditioning from the water_z decision.
- D1–D3 are **spatial joins** (point-in-polygon, buffer) — DuckDB spatial
  does both natively against the Overture tables.
- E1 is row-wise given the layer columns; E2 is a kNN join.

Fill in per id: **Table:** …

## Redesign roadmap (status, order, gates)

Every step lands in `ducklidar.labeling`, and merges only when the full
stage-5 output still matches the reference tile (final-label equivalence —
the same harness that gated the lift: 99.79 % measured, residual
quantified). ▶ = needs Kaveh's call before code.

| step | covers | primitive | status |
|---|---|---|---|
| 1 | A3, A4, A6 + class counts | `cell_evidence` — one GROUP BY | ✅ built, tested |
| 2 | A1, A2 | `ground_cells` — one global cloth + fill | ✅ built, tested |
| 3 | A7 bodies (and every later flood/region) | `cell_components` — components of a cell mask | ✅ primitive built; predicate decided (Kaveh 2026-08-02): the lab's verbatim — class-9 seeds + sparse-dark growth per WATER_* constants, as SQL over cell_evidence. ⬜ port + gate |
| 4 | A5 movement | per-(cell × flight line) aggregates — `GROUP BY cell, point_source_id` | ⬜ small |
| 5 | conditioned water levels | SQL over step 3: `GROUP BY (body, point_source_id)` | ⬜ small, after 3's predicate |
| 6 | 5a priors | a **view** over `points ⋈ features` with the rules' constants (form decided: view, not table — Kaveh 2026-08-02) | ▶ walkthrough — the threshold set is the measured judgment |
| 7 | B5, B6 corridors | map ways (queryable via `read_json_auto` / Overture db) → `ST_Buffer` → cells | ⬜ |
| 8 | **the support forest** (C1, C5, C6, C7, B3, B6, D4) | multi-source BFS over `knn.parquet`, path composition recorded (decided 2026-08-02) | ⬜ build, gate on reference labels |
| 9 | C2 conviction trial | k-hop verdicts on `knn.parquet`, or stays windowed (~3 min — may not be worth moving) | ▶ |
| 10 | D1–D3 vetoes | spatial joins against the Overture tables | ⬜ |
| 11 | E1 decision | ordered mask resolution — row-wise over the layer columns | ⬜ |
| 12 | E2 residue | nearest labelled neighbour — kNN join | ⬜ |
| 13 | B4 outlines | last: polygonization; possibly stays raster-side | ▶ |

Until the last step passes its gate, the lifted windowed rules remain the
production path — the redesign replaces them only by equivalence, never by
enthusiasm.
