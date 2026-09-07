# ducklidar

**Query airborne LiDAR instead of loading it.**

A LiDAR survey is delivered as huge files of laser returns — hundreds of
millions of points, no index, no random access. ducklidar treats a survey
the way a database treats a table: convert once to spatially sorted
Parquet, then ask for exactly the window and columns you need, in under a
second, from files that never fit in memory.

On top of that store it provides the analysis layers a survey actually
gets used for: honest rasters (DSM/DTM that admit where they had no
evidence), ground filtering, instance segmentation, bridge-deck detection,
flight-trajectory recovery, and scene reconstruction — walls with their
real openings, tree crowns, footprint prisms.

```python
import ducklidar as dl

dl.describe("tile.zip")                          # what is in this survey?
dl.to_parquet("tile.zip")                        # once: build the queryable sidecar
pts = dl.read("tile.zip", dl.box(490289, 5457557, 250))   # 0.9 s window
g = dl.surfaces(pts, dl.box(490289, 5457557, 250))         # dsm/dtm/canopy/roofs
print(dl.void_report(g["dtm"]))                  # how much of that is measured?
```

## Why it exists

Three ideas run through every function:

1. **Query, don't load.** A `.las` inside a `.zip` supports no random
   access at all — a 500 m window costs a full parse of the whole tile
   (measured: 9.7 s, 3.1 GB peak). The Parquet sidecar answers the same
   window in 0.9 s via row-group pruning, with column pruning free on top.
   Multiple tiles act as one store: `dl.read([a, b, c], bbox)` answers a
   window across survey tile borders.
2. **Census before you plan.** Surveys differ completely in what they
   carry — classes, colour, returns per pulse. What a survey contains
   decides what is possible with it; `dl.describe` says so in one call,
   before you write any code.
3. **A grid is a model, not a measurement.** A DTM comes out roughly half
   empty, because the laser cannot see the ground under a roof. Functions
   here return the holes, the distance to real evidence, and per-cell
   intervals — instead of letting a smoothly interpolated array pass for
   a measured one.

## Install

```bash
pip install -e .            # core deps: numpy, laspy[lazrs], pyarrow
pytest -q                   # tests run on a synthetic tile — no data needed
```

Heavier features are opt-in extras (see `pyproject.toml`):
`grid` (scipy), `ground` (cloth simulation), `boundary` (scikit-image),
`viz` (matplotlib), `scene` (shapely, earcut), `overture` (duckdb, pyproj).

## What's in the box

| Area | Functions | One line |
|---|---|---|
| Store | `to_parquet`, `read`, `box`, `describe`, `census`, `field_guide`, `flight_dates` | files → queryable windows; every point carries a `pid` join key |
| Rasters | `rasterize`, `surfaces`, `at_extremum`, `void_report`, `distance_to_evidence` | points → grids that admit their gaps |
| Terrain | `dem`, `ndsm`, `fill`, `building_level`, `footprint`, `ground_filter`, `score_against` | ground with provenance attached |
| Segmentation | `instances`, `boundary`, `edge_strength`, `echo_ratio`, `bridge_deck`, `look_azimuth`, `viewed_from` | objects found by disagreement between views |
| Neighbourhood | `knn`, `shape_features`, `local_edges`, `scene.mesh_bridges` | one kNN query; features and the mesh graph both derive from it |
| Trajectory | `pulses`, `sensor_track`, `tracks` | where the aircraft was, from returns alone |
| Scene | `scene.wall_report`, `scene.checked_walls`, `scene.trace_footprint`, `scene.footprint_prism`, `scene.crown_thin`, … | reconstruction that only builds what was measured |
| Facades | `facade.texture_wall`, `facade.elements.parse_wall`, `facade.compose.compose`, `facade.relief.relief_prims` | a wall from street photos: its outline found IN the image, windows/doors/awnings measured, then drawn and built as joinery |
| Objects | `objects.*_model` (one per type), `bridge_ribbon`, `bridge_lights`, `fit_roof_planes` | an instance → triangles, and the KIND the sun reads: solid / slab / glass / surface |
| Type tests | `looks_like_car`, `looks_like_boat`, `looks_like_tree`, `looks_like_deck`, `looks_like_pole` | is it one? a physical question per type, thresholds from measured populations |
| Instance cuts | `raft_split`, `car_split`, `crown_split`, `footprint_ids` | what connectivity welds, cut by the evidence that actually separates it |
| Map labeling | `map_labels.map_claims`, `building_veto`, `vegetation_veto`, `ground_without_bridges` | the map as evidence about what points ARE — one method per mapped type |
| Map facts | `pier_ways` | OSM/Overture ways from a local [duckOverture](../duckOverture) extract |
| Looking | `plot.plan`, `plot.section`, `plot.compare`, `compare_channels`, `notebook_view` | plots and a self-contained HTML viewer |

## Facades from street photos

`ducklidar.facade` textures a building's walls from ordinary street-level
photos. The camera pose is never trusted for pixel placement — it only says
roughly where to look. Each wall's outline is found **in the image** (the
eave against the sky, the corners on vertical lines), rectified to a metric
strip, stitched with the other photos of that wall, and the holes left by
cars and trees are filled with the building's own wall colour. The windows,
doors and awnings detected in every photo become one list in metres on the
wall, and that list is both painted and **built**: frame bars, projecting
sills, mullions, glass set back, door jambs, awning fascias.

```python
from ducklidar import facade
facade.CTX.use(ROOT=repo, Z_GROUND=3.65, SOLID=dict(V=V, T=T), occlusion_map=occ)

tex, valid, info = facade.texture_wall(F, cands, w, h, photos, masks_dir)   # photo-stitched
els  = facade.elements.parse_wall(F, info["strips"], elements_dir, photos)  # metres, deduped, rows
mat  = facade.compose.material(tex, valid, els)
wall = facade.compose.compose(w, h, mat, els)                               # the composed wall
prims = facade.relief.relief_prims(F, els, origin, wall)                    # glTF joinery
```

Measured on Granville Island ([shadowCity2](../shadowCity2)'s pilot, three
buildings): element positions land within about 2 % of the wall where a
photo sees it frontally, and the joinery sizes came from the crops
themselves — frame 0.145 m, mullion 0.155 m, sill 0.065 m proud 0.075 m,
glass 0.035 m behind. A wall no photo sees properly gets the building's
material and **nothing invented** — which, on a real block, is most walls.

## One textured building, one call

`dl.building3d` composes all of the above: the building's own returns, a
solid, every photo's real pose, the walls, the roof, and the GLB.

```python
import ducklidar as dl

b = dl.building3d(ring, store="data/lidar/*.parquet", photos="out/photos",
                  out="out/netloft", level="joinery")
b.glb        # Path — out/netloft/building.glb, next to viewer.html
b.walls      # [{id, w, h, facing, photos, filled, elements, …}, …]
b.elements   # {wall id: [{type, x0, y0, x1, y1, conf, …}, …]}  metres on the wall
b.qa         # Path — the per-wall QA sheet
b.meta       # origin, CRS, level, prims, coverage
```

`level` buys skin, not geometry — `"none"` is the solid alone, `"material"`
one measured wall colour (no GPU, no photo API, and often enough for a
shadow study), `"photo"` the composed texture, `"joinery"` that plus frames,
sills, mullions and glass as real geometry.

**A build never calls an API.** `photos`, `masks` and `elements` are folders
a tool wrote; without them a wall gets the building's material and nothing
invented. `solid=` takes a roofer LoD2.2 CityJSON if you have one — without
one the geometry falls back to the surveyed outline, each edge raised to the
roof height measured along it, every roof tier standing above that (a tower
on a podium, a plant room on the tower) traced from the returns and given its
own walls, and the returns' own roof planes on top — so roofer is never required.
Labels are optional too: with none, the survey's own building class decides
which returns are the building, cut where they stop covering the footprint
(a bridge deck over a shop is not its roof); a survey without classes falls
back to the layer of single returns above the ground, which a crown is not.

## Tools

Things you *run*, never things a build imports — each spends something a
library must not spend by itself: an API quota, a GPU, or another program's
licence.

```bash
python -m ducklidar.tools.fetch_streetview --points b.npz --out out/photos
python -m ducklidar.tools.segment          --photos out/photos     # SAM 3, both passes, one image load
python -m ducklidar.tools.roofer --points box.laz --footprints b.gpkg --out out/roofer --bin ./roofer
python -m ducklidar.tools.build_tile --tile tiles/490000_5457000.parquet --footprints footprints/granville_island.duckdb --root .
```

`build_tile` is the whole tile in one go: every footprint of a duckOverture
extract that lies wholly inside the tile becomes `out/gers_<id>/building.glb`
(geometry and measured material; with photos where `buildings/<id>/photos`
exists). Measured on Granville Island: 246 buildings in 17 s. Footprints the
tile edge cuts are skipped and counted, not built with half their returns.

`roofer` is GPL-3 and is run as a **subprocess, never linked** and never
installed as a dependency; its module docstring carries the glibc shim the
2.35 hosts need. `segment` needs `transformers` + a GPU (`pip install
ducklidar[sam]`), `fetch_streetview` a Street View key of your own.

Extras: `facade` (scikit-image, shapely). The masks come from any segmenter
that labels occluders, buildings and facade elements; the tool uses SAM 3.
The design behind the one call is
[docs/design/building3d.md](docs/design/building3d.md).

## Documentation

- [Getting started](docs/getting-started.md) — from a survey file to
  terrain, objects and a viewer page, step by step.
- [Every stage, every task](docs/stages.md) — the whole survey-to-scene
  pipeline, stage by stage, each task explained simply.
- [Design](docs/design/README.md) — one file per stage: how each task
  works, what it costs, its knobs, and an Optimize? block per task for
  questioning the design part by part.
- [The tables](docs/tables.md) — each stage's inputs, outputs, file
  names, and every column of every table: the pipeline's data contract.
- [The forest table](docs/forest.md) — `data/stage5/forest.parquet`:
  what holds each point up, and the path that proves it.
- [The cell table](docs/cells.md) — `data/stage5/cells.parquet` on its
  own page: every column, plainly, with how it's computed and what it
  says about trust.
- [LiDAR concepts](docs/concepts.md) — returns, classes, DSM/DTM, voids:
  the background the API assumes, in ten minutes.
- [The pipeline by data dependency](docs/pipeline.md) — store / row-wise /
  neighbourhood / components / global: what each processing level may
  know, and how partitioning stays exact.
- [The labeling rules](docs/rules-operators.md) — the measured rules
  (lifted verbatim into `ducklidar.rules`) and the worksheet for their
  table-shaped redesign.
- [The photo manifest](docs/photo-manifest.md) — bring your own photos:
  what a folder of images must say about itself so `building3d` can use it.
- [Where the data lives](docs/data-layout.md) — one folder per building:
  the conventional layout and naming for tiles, footprints, photos and outputs.
  `python -m ducklidar.tools.dashboard --root .` then puts every built building on
  one page: pick one, orbit its points and its model, click its photos.
- [API reference](docs/api.md) — every public function, grouped, with
  signatures.

## Provenance

The library was born in a study of a real city survey (Vancouver 2022,
42.7 M returns per tile); every non-obvious rule and threshold in the code
carries the measurement that justified it in its docstring. Functions
graduate here only with synthetic-tile tests that run in seconds on a
machine with no data.
