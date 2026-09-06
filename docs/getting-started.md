# Getting started

From a survey file you've never opened to terrain, objects and a viewer
page. Every step is runnable as-is; replace `tile.zip` with any
LAS/LAZ/zip/COPC file. New to LiDAR? Read [concepts.md](concepts.md)
first — ten minutes.

## 0. Install

```bash
pip install -e .                  # numpy, laspy[lazrs], pyarrow
pip install -e ".[grid,scene]"    # + scipy, shapely… as needed
```

## 1. Census before you plan

```python
import ducklidar as dl

dl.describe("tile.zip")
```

This prints the header, CRS, point count, every field with its meaning,
the class inventory and the returns histogram. It answers, before you
write code: is there vegetation classification? colour? multiple returns?
What a survey carries decides what is possible with it.

```python
pts = dl.read("tile.zip", some_box)
dl.census(pts["classification"])   # counts/shares per class
dl.field_guide("tile.zip")         # what each field is for, which exist here
dl.flight_dates(pts["gps_time"])   # when it was flown
```

## 2. Build the sidecar, then query windows

```python
dl.to_parquet("tile.zip")          # once per tile; idempotent
B = dl.box(490289, 5457557, 250)   # centre x, centre y, half-width (metres)
pts = dl.read("tile.zip", B)       # now ~1 s instead of a full-file parse
```

`pts` is a plain dict of numpy columns: `pts["x"]`, `pts["z"]`,
`pts["classification"]`, … Noise classes are already dropped
(`drop_noise=False` to keep them). Ask for fewer columns with
`fields=()` — column pruning is free.

Several tiles act as one store — a window near a tile border simply
reads both sides:

```python
tiles = ["489000.zip", "490000.zip", "491000.zip"]
pts = dl.read(tiles, B)            # non-overlapping tiles cost ~nothing
```

Rectangles: `dl.box(cx, cy, half_x, half_y)`.

## 3. Look at it

```python
from ducklidar import plot
plot.plan(pts)                     # from above, coloured by class
plot.section(pts, at=5457557)      # a 2 m slice seen from the side
```

The section view is the one that shows vertical structure — canopy over
ground, the profile of a roof.

## 4. Surfaces, honestly

```python
g = dl.surfaces(pts, B)            # dsm / dtm / canopy / roofs (1 m grids)
print(dl.void_report(g["dtm"]))    # how empty, how clumped, distance to evidence
```

The DTM will be full of NaN — the laser never saw the ground under roofs.
That emptiness is information; don't interpolate it away without keeping
the receipt:

```python
d = dl.dem(pts, B)                 # DTM with provenance attached
d.value                            # the filled grid
d.method                           # per cell: measured / interpolated / …
dl.distance_to_evidence(g["dtm"])  # metres to the nearest real measurement
height = dl.ndsm(d)                # DSM − filled DTM: height above ground
```

No ground classification in your survey? `dl.ground_filter(pts)` decides
bare earth from x, y, z alone (cloth simulation), and
`dl.score_against(mask, pts["classification"])` grades it against the
vendor's labels where they exist.

## 5. Objects

```python
labels, edges, ndsm = dl.instances(pts, B)
```

`instances` finds separate above-ground structures by *disagreement
between views*: cells where looks from different directions saw different
heights are edges; hysteresis links them; watershed closes the contours.
`labels` is a per-cell instance id grid.

Special cases with their own detectors:

```python
deck = dl.bridge_deck(pts, B, 1.0)     # bridge decks by geometry alone
er   = dl.echo_ratio(pts, B)           # vegetation likelihood per cell
```

## 6. The kNN level: features and the mesh graph from one query

Neighbour lookup is the expensive step, so run it once and derive
everything from it:

```python
import numpy as np
P = np.column_stack([pts["x"], pts["y"], pts["z"]])

dist, idx = dl.knn(P, k=16)               # THE query — cache these
e1, e2, e3, nz = dl.shape_features(P, idx)  # planarity etc. are formulas over these
for src, dst in dl.local_edges(P, nn=(dist, idx)):   # k=10 within 2.5 m — free now
    ...                                   # connectivity edges, in chunks

from ducklidar import scene
bridges = scene.mesh_bridges(P[:, 0], P[:, 1], P[:, 2])   # MST links between islands
```

Persist the derived tables keyed by `pid` (ask `dl.read` for the `pid`
field) and connectivity questions stop needing the points at all — see
[pipeline.md](pipeline.md) for the full table-centric architecture.

## 7. A page you can send someone

```python
html = dl.compare_channels(pts, B)     # self-contained HTML, no server
open("view.html", "w").write(html)
dl.notebook_view(html)                 # or inline in a notebook
```

## 8. Scene reconstruction (the `scene` module)

For turning labelled instances into 3D geometry that only claims what was
measured: `trace_footprint` (outline from the instance's own points),
`footprint_prism`, `wall_report` / `wall_mesh` / `checked_walls` (walls
with their real doors and windows cut out — or removed entirely when the
laser proves the side open), `canopy_mesh`, `crown_thin`. Each docstring
carries the measurements that set its defaults.

## Where to next

- [pipeline.md](pipeline.md) — how these pieces compose into a
  partitioned, whole-city pipeline without ever holding the city in RAM.
- [api.md](api.md) — the full function list.
