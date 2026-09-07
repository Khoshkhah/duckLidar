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
| Buildings | `building3d` (**proposed**, see [design](docs/design/building3d.md)) | an OSM way id → a textured 3-D building: footprint, its returns, solid, facades from street photos |
| Objects | `objects.*_model` (one per type), `bridge_ribbon`, `bridge_lights`, `fit_roof_planes` | an instance → triangles, and the KIND the sun reads: solid / slab / glass / surface |
| Type tests | `looks_like_car`, `looks_like_boat`, `looks_like_tree`, `looks_like_deck`, `looks_like_pole` | is it one? a physical question per type, thresholds from measured populations |
| Instance cuts | `raft_split`, `car_split`, `crown_split`, `footprint_ids` | what connectivity welds, cut by the evidence that actually separates it |
| Map labeling | `map_labels.map_claims`, `building_veto`, `vegetation_veto`, `ground_without_bridges` | the map as evidence about what points ARE — one method per mapped type |
| Map facts | `pier_ways` | OSM/Overture ways from a local [duckOverture](../duckOverture) extract |
| Looking | `plot.plan`, `plot.section`, `plot.compare`, `compare_channels`, `notebook_view` | plots and a self-contained HTML viewer |

## One building, in one call (proposed)

The pieces above compose into a building: give an OSM way id, get a
textured model back — footprint from the map, returns from the store,
a roof solid, and facades reconstructed from street photos with windows,
doors and awnings as measured 3-D joinery.

```python
b = dl.building3d(714927893, store="data/lidar/*.parquet", out="out/")
b.glb            # the textured model
b.walls          # per wall: size, facing, photos used, coverage
b.elements       # per wall: windows / doors / awnings, in metres
```

Not built yet: the method is proven on three buildings in
[shadowCity2](../shadowCity2)'s pilot (`tools/pilot/`), and the design for
moving it here — the API, the four private stages, the photo-source and
segmenter interfaces, and the five decisions it needs — is
[docs/design/building3d.md](docs/design/building3d.md). What it can and
cannot do is stated there: geometry is exact from the LiDAR, element
positions are measured to about 2 % of the wall, and a wall no photo sees
properly gets the building's material and nothing invented.

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
- [API reference](docs/api.md) — every public function, grouped, with
  signatures.

## Provenance

The library was born in a study of a real city survey (Vancouver 2022,
42.7 M returns per tile); every non-obvious rule and threshold in the code
carries the measurement that justified it in its docstring. Functions
graduate here only with synthetic-tile tests that run in seconds on a
machine with no data.
