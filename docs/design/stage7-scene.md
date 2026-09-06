# Stage 7 — scene build

> The **per-class object cascade** — boats, docks, cars, the bridge — is its
> own task, in [stage7-objects.md](stage7-objects.md). This page covers the
> building pieces, the terrain, and the merge.

**Purpose:** instances become 3-D models that only claim what was measured.

**Status: BUILT** (2026-08-03), old area.
`shadowCity2/tools/stage7_scene.py` writes the shadow scene;
`tools/stage7_objects.py` writes the object scene. Schemas are fixed in
[../tables.md](../tables.md).

## Task 7.1 — footprints

**The Problem:** not every instance has a map polygon.

**The Solution:** where a map footprint claims returns, **that polygon is
the building** — Kaveh's standing rule, and it is what keeps two
neighbours sharing a wall from fusing into one blob. Returns no footprint
claims fall through to a polygon traced from their own points
(`scene.trace_footprint`: occupancy grid → morphology → contour).

**The Result — measured, old area:** 305 map footprints; 1,002 unmapped
jobs traced; **818 buildings** in the finished scene.

**Grouping the unmapped ones is plan-view, and that is measured, not
assumed.** Feeding stage 6's graph instances in instead cost **158
structures** (660 buildings against 818): the class-induced subgraph
shatters sparse unmapped building returns into **35,358 components of
median size 2**, and 33,713 fall under the 60-return floor and are
dropped. Graph components make good *objects* and bad *footprints* — a
shed's scattered returns are one blob in plan view and many components in
3-D. Do not retry this without changing the floor.

## Task 7.2 — roof solids — **roofer is NOT used**

**The Problem:** buildings need roof shape, and the tool the lab used —
`roofer`, an external GPL LoD2 fitter — is not available here. It was
built under `/tmp` and did not survive a reboot; it is not a conda
package, so recovering it is a C++ source build.

**The Solution:** take the roof **from the returns**
(`dl.objects.solid_25d`): a top surface on the instance's own grid,
boundary walls, a flat cap. Where a footprint is available and the
structure is closed, `scene.footprint_prism` extrudes the outline between
two measured heights as the honest minimum.

**Why this is the better answer, not the fallback.** An LoD2 fitter
replaces a roof with idealised primitives; the measured surface keeps
dormers, parapets, stair heads and plant rooms **because they were
measured**. That is the project's own rule applied to shape rather than to
labels. The cost is honest too: a measured roof is noisier than a fitted
one, and a sparse roof yields a lumpier surface.

**The Result — measured on the lab's own study box:** **396 buildings**
against the lab's roofer-built **268**.

## Task 7.3 — the wall trial and the openness gate

**The Problem:** a wall face with no returns might be open — or merely
hidden. The honesty rule: absence at the wall plane is occlusion, not
proof of openness; **ground seen under the roof is proof.**

**The Solution:** every wall face stands trial against the returns
(`scene.wall_report` / `scene.checked_walls`) — kept where evidenced,
openings cut where framed, removed where the laser saw floor under the
roof. The openness gate itself is the overlap of two 1 m rasters:
floor-under-roof cells over roof cells **> 0.3**.

**Why the overlap and not "ground returns inside the polygon".** That was
the first rule and it misfired at tile scale: Overture footprints include
yards, grass fired the gate, and ordinary houses grew footprint-sized
gables (Kaveh, 2026-08-01: *"why put on every building a very very large
gable roof!"*). A yard has no roof above it; a carport floor does.

**The Result — measured, old area:** **114 of 818** buildings opened, and
their roofs are tagged `slab` so the low sun passes underneath. The lab's
roofer path adds walls without checking — its own comment says so — which
is the gap this closes.

## Task 7.4 — canopies, crowns, terrain

**Crowns** ship as points, never meshes: for a shadow raster the returns
already *are* the canopy (`scene.crown_thin`, one display return per
voxel) — 8,958,416 crown points on the tile, 1,841,515 on the lab box.

**Terrain** comes from stage 5.2's `ground_grid.npy` — one cloth
simulation over the whole area's cell minima, so there are no per-window
seams. Re-running a DEM here would be a second, differently-seamed answer
to a settled question.

**⚠ The grid must be flipped, and this is a contract, not a tweak.**
Stage 5's operators raster **SOUTH-up** (`dl.cell_of`, row 0 = min y); the
scene engine and the viewer read grids **NORTH-up** (`_ground_level` and
`terrain_soup` both index `bbox[3] - y`). **[measured]** ground returns sit
**0.038 m** from the south-up reading and **4.770 m (p90 18.8 m)** from the
north-up reading of the same array. Passed through unflipped, every
building takes its `z0` from the mirrored row. `dl.rasterize` is already
north-up (`dl.cell_index`), so only stage 5's grids need `[::-1]`.

## Task 7.5 — merge: two outputs, two consumers

**The Problem:** stage 8's rasterizer needs to know how each part blocks
light, while a viewer needs to know what class each part *is*. Those are
different groupings of the same geometry.

**The Solution:** write both.

| file | grouped by | consumer |
|---|---|---|
| `scene.npz` | part **kind** — `solid` / `slab` | stage 8's rasterizer |
| `scene_export.npz` | object **class** — building/boat/bridge/object | the 3-D page |

`solid` blocks light outright; `slab` carries `(top, bottom)` so the low
sun passes beneath — bridge decks and evidenced-open canopies. That single
distinction is why the wall trial matters downstream, and it is what v1's
nDSM got wrong by extruding open structures solid.

**The Result — measured, old area:** 1,294 parts from 818 buildings,
**114 slab / 1,180 solid**, in 184 s.

## What stage 7 does not do

- **No LoD2 fitting** — see 7.2. Roofs are measured surfaces.
- **`glazing` is folded into `building`.** A window is part of the
  building (`bld_all = building | glazing` in the rules), so it is not a
  separate layer yet; rendering it translucent needs the roof-glass and
  facade-opening parts split out of the building recipe.
- **No dock or float-home class.** Stage 5 decides only the ten classes in
  `dl.DECIDE`; v1's cascade has `DOCK` and `MOORAGE` of its own, so both
  arrive here as `floating` and get a vessel recipe. The natural source is
  the map's pier ways (`dl.pier_ways`) — a dock is a mapped thing.
