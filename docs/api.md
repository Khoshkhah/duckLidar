# API reference

Every public function, grouped by module. Signatures abbreviated where
defaults are long; each function's docstring carries the full story and
the measurements behind its defaults — `help(dl.read)` is the authority.

Most functions take `pts` (the column dict `dl.read` returns) and `bbox`
(from `dl.box`). Grids default to 1 m cells (`pix=1.0`).

## read — the store

| | |
|---|---|
| `box(cx, cy, half, half_y=None)` | `(minx, miny, maxx, maxy)` window; `half_y` makes it a rectangle |
| `to_parquet(src, dest=None, …)` | build the spatially sorted sidecar; idempotent. Adds `pid` — the point's row in its file, the join key for every derived table |
| `read(src, bbox, *, fields, drop_noise, parquet)` | window → dict of numpy columns; `src` may be a list of tiles, a zip/LAS/LAZ, a COPC URL, or a `.parquet`. Ask for `pid` via `fields` |
| `parquet_path(src)` | where the sidecar for `src` lives |
| `EXTRAS`, `NOISE` | default extra columns; ASPRS classes dropped as noise |

## fields — know your survey

| | |
|---|---|
| `describe(src, sample_bbox=None)` | full inventory: header, CRS, fields, classes, returns histogram |
| `census(classification)` | counts/shares per class; can canopy be told from roof? |
| `field_guide(src=None, …)` | what every LAS field is for; which this file has |
| `flight_dates(gps_time, …)` | first/last UTC datetime the survey was flown |
| `ASPRS`, `GROUND`, `BUILDING`, `VEGETATION`, `MEANING`, `USES` | class codes and their meanings |

## grid — rasters that admit their gaps

| | |
|---|---|
| `rasterize(pts, bbox, pix, *, mask, how, values, interval, quantile)` | reduce returns onto a grid; NaN where nothing landed |
| `surfaces(pts, bbox, …)` | dsm / dtm / canopy / roofs in one call |
| `at_extremum(pts, bbox, pix, field, *, how, values, mask)` | value of the return that won each cell |
| `void_report(grid, pix)` | how empty, how the holes clump, distance to evidence |
| `distance_to_evidence(grid, pix)` | metres from each cell to the nearest measured cell |
| `cell_index(pts, bbox, pix)`, `shape_for(bbox, pix)` | grid plumbing |

## dem / estimate — terrain with provenance

| | |
|---|---|
| `dem(pts, bbox, …)` | DTM as an `Estimate`: value + per-cell interval, method, evidence |
| `fill(grid, …)` | interpolate holes; returns `(filled, distance_to_evidence)` |
| `ndsm(dem_result)` | height above ground |
| `building_level(dem_result, footprint, …)` | the ground datum under one building, with trust |
| `footprint(pts, bbox, member, pix)` | cells occupied by one instance's points |
| `Estimate` | value, lo/hi, n, method — a result is a number *and* its trust |

## ground — bare earth from geometry alone

| | |
|---|---|
| `ground_filter(pts, …)` | boolean bare-earth mask via cloth simulation (x, y, z only) |
| `score_against(mask, classification)` | grade a mask against the survey's own labels |

## segment — instances by disagreement between views

| | |
|---|---|
| `instances(pts, bbox, …)` | label above-ground structures → `(labels, edges, ndsm)` |
| `edge_strength(pts, bbox, *, by, floor)` | per-view interval width, multiplied across views |
| `boundary(S, ndsm, …)`, `hysteresis(S, hi, lo)`, `local_boundary(S, …)` | edge maps → closed contours |
| `echo_ratio(pts, bbox, …)` | share of split pulses per cell — vegetation |
| `bridge_deck(pts, bbox, pix, …)` | returns on a bridge deck, by geometry alone |
| `look_azimuth(pts)`, `viewed_from(pts, …)`, `angle_rank(pts, bbox, …)` | which direction each return was seen from |

## neighbourhood / graph — the kNN level

| | |
|---|---|
| `balanced_boxes(files, bbox, max_points, *, min_side)` | data-driven partitions: recursive median splits of the store until every leaf holds ≤ max_points — the compute plan derived from the data, not the survey's tile grid |
| `knn(xyz, k=16, *, chunk)` | `(dist, idx)` of each point's k nearest — the one query everything below derives from |
| `shape_features(xyz, idx, *, chunk)` | `(e1, e2, e3, nz)`: normalised covariance eigenvalues + normal verticality; every published shape feature is a formula over these |
| `local_edges(xyz, k=10, cap=2.5, *, chunk, nn)` | the deterministic local edge set, in index chunks; pass `nn=knn(…)` to reuse the query |
| `scene.mesh_bridges(x, y, z, k, cap, k_search)` | MST links between the kNN islands |

## trajectory — where the aircraft was

| | |
|---|---|
| `pulses(pts)` | first/last return of each multi-return pulse + unit vector up the beam |
| `sensor_track(pts, *, window, min_pulses)` | sensor position per time window |
| `tracks(pts, …)` | `sensor_track` per flight line |

## scene — reconstruction that only claims what was measured

| | |
|---|---|
| `trace_footprint(x, y, cell, min_area)` | footprint polygon from an instance's own points |
| `footprint_prism(ring, z0, ztop, …)` | outline extruded, roof by ear-cutting |
| `wall_report(ring, x, y, z, z0, ztop, …)` | per side: wall or no wall, and its openings |
| `wall_mesh(…)`, `wall_occupancy(…)` | walls rendered as those decisions |
| `checked_walls(soup, cols, ring, …)` | audit a solid model's walls against the returns |
| `canopy_mesh(poly, x, y, z, z0, …)` | measured roof with open sides |
| `crown_thin(x, y, z, cell)` | one display return per voxel |
| `read_cityjsonseq(path, lod)` | CityJSONSeq (e.g. roofer output) → meshes |
| `sat_lift(soup, cols, rgb, …)` | deshadow vertex colours against satellite imagery |

## labeling — stage 5 on tables

| | |
|---|---|
| `cell_evidence(files, bbox, pix=1.0)` | one `GROUP BY cell` over the store → the per-cell evidence table the regional rules think on |
| `ground_cells(cells, bbox, pix=1.0, **csf)` | bare-earth per cell: ONE cloth simulation over the whole area's cell-minimum surface — no per-window seams |
| `cell_components(cells, mask, bbox, pix=1.0)` | connected components of any cell mask — water bodies, floods, regions — once, globally; conditioned levels then = `GROUP BY (body, flight line)` |
| `cell_of(x, y, bbox, pix=1.0)` | the point↔cell join key |
| `column_support(voxels, cells, bbox, pix=1.0, *, floor_band=0.3, min_gap=1.5, step=0.5, max_h=30.0)` | the vertical profile per cell from the stage 2 voxel table → `ColumnSupport.supported(cell, h)` / `.gap_below(cell, h)`: is the column continuous from the floor up, or is there walkable air (a truck vs a one-storey eave) |
| `raft_split(px, py, pz, *, pix=0.5, prominence=1.2, dock_area=100)` | one welded marina raft → per-vessel ids. Boats touch within ~1 m of the pontoon so NO connectivity rule separates them; this cuts on height — h-maxima peaks as markers, docks as large low ribbons, watershed on the inverted surface |
| `path_shares_from_pred(pred, depth, cats)` | `(building share, veg share)` from a STORED predecessor forest. `pred` is the walk's whole result, so the histogram is re-derived by pointer doubling for any categories — no graph, ~2 GB instead of >18 |
| `car_split(px, py, pz, surf_z, *, pix=0.3, prominence=0.25)` | welded street furniture → one id per ROOF. The marina watershed at car scale: a car is a roof clear of what it is parked on, and the gap to the next is a strip of that surface. Cells are 0.3 m because a 0.6 m gap is ONE cell at 0.5 m and a blur closes it. `surf_z` is per point — a bridge car's datum is 30 m above a street car's |
| `footprint_ids(polys, bbox, pix=1.0, key="id")` | map polygons → a grid of their **ids** (`footprint_cells` answers only *whether* a cell is inside one). Gives an instance an identity inherited from the map |
| `rules.*` | the lab's labeling rules, lifted verbatim — the measured reference every table operator must reproduce |

## objects — what an instance looks like in 3-D

*`scene` builds the pieces a BUILDING is made of; this is the rest of the
cascade. No context object: each recipe takes the arrays it needs, so a
table pipeline can drive them. Every one returns `[(soup, cols), ...]` —
an (n, 3) float32 triangle soup and matching uint8 vertex colours — and an
empty list is a real answer ("this instance does not earn geometry").*

| | |
|---|---|
| `objects.solid_25d(x, y, z, z0, pix=1.0)` | the roof **as measured**: top surface on the instance's own grid, boundary walls, flat cap. What replaces an external LoD2 fitter — dormers and parapets survive because they were measured |
| `objects.boat_mesh(x, y, z, wlvl, col)` | hull with a pointed bow, cabin sized from HULL points only (a mast must not become a 15 m cabin), mast where the highest returns bunch tightly. A flat float falls through to `dock_mesh` |
| `objects.dock_mesh(x, y, z, wlvl, col=None)` | dock/pontoon: outline traced from the returns, water between marina fingers kept as HOLES, extruded waterline→deck |
| `objects.car_mesh(x, y, z, col)` | body + cabin boxes in the object's own frame; the base is its OWN 2nd percentile, never a DEM lookup — a car on a bridge stands on the deck |
| `objects.lamp_mesh(x, y, z, ground_z, toward=None)` | street light: pole, arm, lantern, the arm overhanging the roadway |
| `objects.crown_mesh(x, y, z, base_z, …)` | a canopy SHELL: the crown's measured top closed down to where the crown begins, not to the ground. `tree_model` ships it as `slab` — modelled solid, every tree blacks out the ground beneath it |
| `objects.unknown_model(x, y, z, …)` | matter we cannot name: the 2.5-D surface the laser saw and nothing more. It shades correctly because the matter is really there, and claims nothing about what it IS |
| `objects.pca_frame(px, py)`, `objects.box_local(...)`, `objects.ring_prism(...)` | the oriented-object primitives everything above is built from |

### Is it one? — the type tests

*A bounding box cannot tell a car from a clump of canopy. Each test asks a
physical question and takes the CONTEXT that answers it; each threshold comes
from a measured population where the two groups separate. See
[type-guards](design/type-guards.md).*

| | |
|---|---|
| `objects.looks_like_car(x, y, z, veg_frac=None, exg=None)` | car-shaped, **and** not standing inside a tree (`veg_frac`, 0.5) nor strongly green. The offender that forced this: 3.9 × 2.0 × 2.4 m — every dimension a van's — with 83% `veg_high` within 6 m |
| `objects.looks_like_boat(px, py, pz, wlvl, …)` | compact (fills ≥22% of its own plan box), ≤25 m, floats (base within 3 m of the waterline) and is not a tower (≤15 m). The 236 × 67 m "boat" filled 7.2% |
| `objects.looks_like_tree(x, y, z, ground_z)` | rooted: ≤35 m tall, crown ≤20 m wide, base ≤10 m above ground. The base bound is 10 and not 3 because a crown legitimately starts several metres up — p95 of real trees is 8.1 m |
| `objects.looks_like_deck(x, y, z, ratio=3.0)` | broad and flat. Real decks score plan/vertical 5.8–22.3; a lamp mast, a gantry and a truss portal score 1.2, 0.9, 0.5 |
| `objects.looks_like_pole(x, y, z)` | >2.5 m tall, <1.2 m across — a street lamp on the GROUND. On a bridge use `bridge_lights` |

### Bridges

| | |
|---|---|
| `objects.chain_ways(ways, tol=1.0)` | join map polylines sharing endpoints into one chain per bridge — swept separately they leave seams |
| `objects.bridge_ribbon(chain, x, y, z, …, bents=False, rail=False)` | a deck SWEPT along its centreline: P82 height per 3 m station, edges from the lateral P0.5/P99.5 kept per SIDE (Granville's east sidewalk cantilevers 4.6 m further), one median width per corridor, gaps interpolated. Optionally its bents (one per 60 m) and guard rails (1.26 m, west side glass) |
| `objects.bridge_lights(x, y, z, …)` | street lights ON a bridge. Their datum is the DECK, not the ground, so a 10 m standard on a 35 m deck reads as 45 m and fails every ground test. Clusters ≤3.5 m across whose head stands 4–15 m over deck returns BESIDE them, with a consensus filter: one bridge, one lamp model |
| `objects.bridge_support_model(x, y, z, ground_z, deck_bottom, …)` | piers claimed from evidence. `deck_bottom` may be the DECK SURFACE itself, because a footprint that ramps from grade to 40 m has no single underside |
| `objects.fit_roof_planes(top, occ, pix)` | snap a measured roof grid onto its own planes — ridges and dormers survive, the 1 m staircase does not |
| `objects.ground_from_returns(ring, px, py, pz, k=25)` | height per ring vertex from the surface's OWN returns. Never sample a DEM near a bridge: stage 5's grid IS the deck there |

## map_labels — one labeling method per mapped type

*The map is an external fact, not decoration: where Overture draws an outline,
that outline is evidence about what the points inside it ARE. Each method
CLAIMS; `map_claims` applies them in a fixed order and each may overwrite only
the labels it is allowed to, so a map error cannot erase a confident geometric
label. Each also carries a physical guard, because a footprint says where a
thing is in plan and nothing about height.*

| | |
|---|---|
| `map_claims(label, x, y, z, feats, decide, …)` | run every claim and veto in order; returns `(label, map_type, report)`. `map_type` is the sub-label the MAP asserts per point |
| `water_claim` / `dock_claim` / `bridge_claim` / `building_claim` / `ground_claim` | one per mapped type. A dock is a mapped pier way at pontoon height — the type that previously had no labeling rule at all |
| `building_veto(feats, x, y, z, …)` | **no footprint, no building.** 812,837 points outside every outline, 71.6% of them under 6 m above ground — canopies, wharf sheds, yard clutter. They keep standing and shading; they stop being buildings |
| `vegetation_veto(feats, x, y, z, …)` | nothing vegetal grows on open water. Sailcloth returns like foliage; the polygon is eroded 8 m first so overhanging shore crowns survive (37,695 of 47,786 sit within 4 m of a bank) |
| `ground_without_bridges(grid, bbox, feats, …)` | the DEM with the bridges taken out of it, rebuilt from the ground returns seen UNDER the deck (median error +0.00 m; the raw grid is +27.34). Polygons **and** buffered centrelines are masked — Granville is mapped mostly as lines |

## overture — map facts, locally

| | |
|---|---|
| `pier_ways(db, box, margin, crs)` | pier/breakwater/quay ways from a [duckOverture](../../duckOverture) extract |

## plot / viz — looking at results

| | |
|---|---|
| `plot.plan(pts, by, …)` | from above, coloured by any field |
| `plot.section(pts, at, width, axis, …)` | thin slice from the side |
| `plot.compare(pts, at, by=(a, b), …)` | same slice, two colourings — survey vs algorithm |
| `compare_channels(pts, bbox, …)` | self-contained HTML comparison page |
| `notebook_view(html, …)` | run the page inside a notebook |
| `default_channels(pts)` | a sensible starting channel set |
