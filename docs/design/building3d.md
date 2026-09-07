# `dl.building3d(osm_id)` — one call, one textured 3-D building

**Status: PROPOSAL, 2026-09-06. Nothing below is built.** ▶ marks Kaveh's
decisions. Every number is measured in the shadowCity2 pilot
(`tools/pilot/`, 3 buildings: Net Loft, Railspur Alley, Granville Island
Stage) or in this library.

## The Problem

The pilot proved a building can be reconstructed and textured end to end,
but the recipe lives in nine scripts under another repo's `tools/`, wired
together by a registry entry per building, an environment variable and a
run of five commands in order. Nothing about it is reusable: another
project cannot ask for a building, and the pilot's own scripts cannot run
two buildings without editing a file. Meanwhile every *piece* it uses is
either already in ducklidar (`read`, `dem`, `scene.*`, `map_labels`,
`overture.pier_ways`) or is generic enough to belong here.

What is missing is the composition: **an OSM way id in, a textured 3-D
building out.**

## The Data

What the pilot needs per building, and where each part comes from today:

| input | source today | in ducklidar? |
|---|---|---|
| footprint polygon | Overture extract (`duckOverture/data/*.duckdb`) by `bid`, or OSM way id | `overture.pier_ways` reads that db; no building query |
| its LiDAR returns | `dl.read` over the store + the stage-5 label table | yes (`read`), the labels are shadowCity2's |
| roof solid | roofer LoD2.2, run once per tile, read by `scene.read_cityjson` | reader yes, runner no |
| street photos | Google Street View metadata + image API, ~30 calls | no |
| occluder / building masks | SAM 3 text prompts, GPU | no |
| facade elements | SAM 3 element prompts, GPU | no |
| wall outline fit, rectify, stitch, fill | `facade_quad/rectify/stitch/fill.py` | no — the four modules |
| element parse, compose, joinery relief | `facade_elements/compose/relief.py` | no — the three modules |
| GLB writer | `build_model.write_glb` | no |

Measured cost per building (this machine, RTX 3060 Ti):

| stage | Railspur (24×17 m) | Net Loft (43×51 m) | Stage (56×43 m) |
|---|---|---|---|
| points from the store | 4 s | 25 s | 12 s |
| Street View sweep (~50 probes, 28 images) | 90 s | 120 s | 150 s |
| SAM 3 occluders | 60 s | 150 s | 130 s |
| SAM 3 elements | 90 s | 210 s | 180 s |
| fit → compose → relief → GLB | 40 s | 240 s | 90 s |

Coverage is the real variable, not size: Railspur composed 2 of 16 walls,
the Net Loft 5 of 170, the Stage 2 of 10 — because Street View only sees
what fronts a street.

## The Solution

One public function, four private stages, no new heavy dependency in the
core install.

```python
import ducklidar as dl

b = dl.building3d(714927893, store="data/lidar/*.parquet", out="out/")
b.glb          # Path — the textured model
b.walls        # [{id, w, h, facing, photos, elements, coverage}, …]
b.elements     # {wall id: [{type, x0, y0, x1, y1, conf, source}, …]}   metres
b.qa           # Path — the per-wall QA sheet
```

`building3d(osm_id_or_footprint, *, store, out, photos="streetview",
solid=None, level="joinery", cache=True)`:

1. **footprint** — `osm_id` → polygon. From the local Overture/duckOSM
   extract when one covers the point (no network), else Overpass. ▶ 1
2. **points** — `dl.read(store, box(footprint))`, filtered to the
   building's own returns: inside the footprint grown by the 3 m eave,
   above local ground. Labels are used when the caller passes them,
   never required — this is what frees the function from shadowCity2's
   stage 5.
3. **geometry** — a roofer LoD2.2 solid when `solid=` is given or a
   cached one covers the footprint, else `scene.footprint_prism` +
   `objects.fit_roof_planes`. Plus skylight boxes from the raised
   returns, and the floor slab.
4. **skin** — `level=`:
   - `"none"` — geometry only,
   - `"material"` — one measured wall colour per building,
   - `"photo"` — the composed texture (material + element crops),
   - `"joinery"` — as `photo`, plus frames, sills, mullions, glass,
     jambs, awning fascias as geometry. **Default.**

The seven pilot modules move in as **`ducklidar.facade`** (a subpackage:
`quad`, `rectify`, `stitch`, `fill`, `elements`, `compose`, `relief`) and
`ducklidar.glb` (the writer). They are already pure numpy/OpenCV with no
shadowCity2 imports except `build_model` for constants — that coupling is
the port's only real work.

**Photo sources** are a small interface, not a hard-coded Google client:

```python
class PhotoSource:                      # ducklidar.photos
    def around(self, footprint, radius) -> [Photo]   # Photo: file, x, y, heading, pitch, fov, date
```

with `StreetView(key=...)`, `Mapillary(token=...)`, and `Folder(path)` for
the user's own photos. **`Folder` is the one that matters long term**: the
pilot's whole quality ceiling is Street View's coverage and ±3 m poses.

**Masks** are a `Segmenter` interface with one implementation
(`Sam3(prompts)`), so the GPU dependency stays optional and a caller can
substitute anything that returns the same masks.

## What is exact, what is bounded, what is not solved

| | |
|---|---|
| geometry (solid, skylights, floor, joinery placement) | **exact** — LiDAR and the element list, no photo pixels move a vertex |
| wall outline in a photo | **evidence-based**, verified: eave/corner snapping lands within 3–5 px where the eave is visible; the pose is only a search band |
| element positions | **measured**, ±2 % of the wall on the verified walls (railspur facade02) |
| joinery sizes | **measured** from the crops: frame 0.145 m, mullion 0.155 m, sill 0.065 m proud 0.075 m, glass 0.035 m back |
| walls no photo sees properly | **material only** — nothing invented (2 of 16, 5 of 170, 2 of 10 walls composed on the three pilots) |
| a wall whose outline locks onto a set-back storey or a neighbour | **known failure**, guarded by a colour check, not solved (railspur facade03) |
| low-confidence detections | gated by `crop_ok`; a `conf` floor is still ▶ 3 |

## The Result

`out/<osm_id>/`: `building.glb`, `viewer.html`, `wall_qa.png`,
`elements.json`, `meta.json` (footprint, origin UTM, CRS, coverage per
wall, photo manifest, versions). The `Building` object above is the API.

## Costs

Estimated from the pilot table: 5–15 min per building alone, dominated by
the two SAM 3 passes and the photo fetch. Batched over a tile with the
model resident and one roofer run, **6–10 h for the 352 footprints of the
Vancouver tile** — the estimate the pilot supports, not a measurement.

## Rejected

- **A CLI instead of a function.** The pilot already is a CLI; the thing
  that is missing is programmatic use.
- **Keeping the modules in shadowCity2 and importing them from ducklidar.**
  Inverts the dependency: shadowCity2 is the app, ducklidar the library.
- **Bundling SAM 3 / a Google client as required deps.** Optional extras
  (`facade`, `sam`, `streetview`) keep `import ducklidar` cheap, which is
  this library's rule.
- **Fetching photos inside `building3d` by default.** ▶ 2 — see below.
- **Waiting for the open pilot bugs to be fixed first.** They are quality
  bugs in the skin; the API and the port are orthogonal, and the pilot
  scripts become the API's first caller, which is how they get fixed.

## Decision points ▶

1. **Footprint lookup.** (a) local extract only, fail if not covered
   (offline, deterministic); (b) local, falling back to Overpass.
   *Recommend (a) for the library, with an `osm.py` helper for (b).*
2. **Photo fetching.** (a) `building3d` never fetches: the caller passes
   a `PhotoSource`; (b) it fetches when given a key. *Recommend (a) —
   a library function that spends the user's API quota by default is
   surprising.*
3. **A confidence floor on elements** (the pilot shipped a 0.01 "door").
   *Recommend 0.35, with the value in `meta.json`.*
4. **Where the roofer solid comes from.** (a) caller supplies it;
   (b) ducklidar runs the GPL binary as a subprocess when configured.
   *Recommend (a) now, (b) behind `roofer_bin=` later — the licence rule
   says subprocess only, never linked.*
5. **Scope of this change.** (a) API + port only, pilot keeps its bugs;
   (b) API + port + fix the known skin bugs in the same pass.
   *Recommend (a): a working port with today's quality, then fixes with
   the pilot as a regression test.*

## Impact

New in ducklidar: `building3d.py`, `facade/` (7 modules ported),
`glb.py`, `photos.py`, `segment.py`, extras `facade`/`sam`/`streetview`,
tests on the synthetic tile (geometry, element placement, GLB validity —
no network, no GPU), README row + a usage section. shadowCity2's
`tools/pilot/` shrinks to the registry, the fetchers and the QA sheet,
calling `dl.building3d`. *Estimated* 2–3 days.

## Optimize?

- The two SAM 3 passes are 60–70 % of the wall-clock and share an image
  load; one pass with both prompt sets would nearly halve it.
- Per-tile batching (roofer once, model resident, photos fetched in one
  sweep) is what makes 352 buildings a night rather than a week.
- `level="material"` costs no GPU and no photo API at all — for a shadow
  study that is often enough, and it is the mode a city-scale run should
  default to.
