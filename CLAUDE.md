# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ducklidar` — query airborne LiDAR instead of loading it. LAS/LAZ → spatially sorted
Parquet sidecar, windowed reads, then the analysis layers on top: honest rasters, ground
filtering, labeling, instance segmentation, scene/object meshes, and `dl.building3d`, which
turns one footprint (+ optional street photos) into a textured GLB.

It graduated out of the `../lidar-learning` lab (2026-08-02). The lab holds the measurements
and notebooks; this repo holds the library. `../shadowCity2` is the pilot app that consumes
it; `../duckOverture` produces the footprint extracts. Read the README first — it is current
and explains the three design ideas every function follows.

## Environment and commands

No `.venv`. Use the conda env `lidar`:

```bash
PY=~/miniconda3/envs/lidar/bin/python
$PY -m pytest -q                              # 118 tests, ~20 s, synthetic tiles only — no data needed
$PY -m pytest -q tests/test_building3d.py     # one file
$PY -m pytest -q tests/test_ducklidar.py -k parquet
pip install -e .[dev]                         # if the env ever needs rebuilding; extras in pyproject.toml
```

Tools (run, never imported by a build — each spends an API quota, a GPU, or a GPL licence):

```bash
$PY -m ducklidar.tools.build_tile --tile tiles/490000_5457000.parquet --footprints footprints/granville_island.duckdb --root .
$PY -m ducklidar.tools.dashboard  --root . --out out/dashboard.html    # one self-contained page of every out/<id>/building.glb
$PY -m ducklidar.tools.roofer / fetch_streetview / segment              # see each module's docstring
```

Local data lives in `tiles/`, `footprints/`, `buildings/<id>/`, `out/<id>/` — all gitignored,
layout defined in `docs/data-layout.md`. The Granville Island tile is `tiles/490000_5457000.parquet`
(42.7 M points, EPSG:26910); the footprint extract is a duckOverture `.duckdb` whose buildings
are in schema `buildings.building`, geometry in EPSG:4326.

There is no mkdocs config here; `docs/` is plain Markdown read from the repo.

## Architecture

**Package root (`src/ducklidar/__init__.py`)** re-exports the store, raster, terrain,
segmentation, graph and labeling API. Two things are deliberately *not* imported there:

- `dl.building3d` is loaded lazily via module `__getattr__` (it pulls OpenCV/scikit-image).
- `ducklidar.rules` is never imported by the root: it is the lab's stage-5 labeling pipeline
  lifted **verbatim** (`pipeline.py`, `layers.py`, `lidr_trees.py`), configured through env vars
  in `rules/_env.py` (`LIDAR_OUT`, `LIDAR_BOX`, `LIDAR_STAGE2`, `LIDAR_NEIGHBOURS`). It is
  gated by label equivalence against the lab's reference tile — do not "improve" a rule there
  without re-running that gate (docs/design/stage5-labeling.md).

**Heavy deps are imported inside functions**, never at module top. `import ducklidar` must
stay cheap (numpy, pyarrow, laspy only).

**The building pipeline** (`_building3d.py` orchestrates; `building/` and `facade/` do the work):

1. `building/points.building_points` — reads the footprint grown by an eave collar from the
   store, finds ground as a low percentile inside the footprint (or from labels if given).
   Points are always read *with a collar*; a footprint-sized window breaks the library's own
   percentile thresholds.
2. Solid — a roofer LoD2.2 CityJSON if passed, else `fallback_solid` → `objects.building_model`
   (outline extruded to the returns' 98th-percentile top, roof planes fitted from returns).
3. `building/geometry`, `building/walls` — facades as planar face groups with outer edges.
4. `facade/` textures each wall from photos: `quad` (find the wall IN the image, never trust the
   pose) → `rectify` → `stitch` → `fill` → `elements` (SAM 3 detections → metres on the wall)
   → `compose` (texture) and `relief` (windows/doors as geometry). `facade.CTX` is the one
   host-to-module context object.
5. `glb.py` writes the bundle: `building.glb`, `viewer.html`, `elements.json`, `meta.json`.

`level` ("none" / "material" / "photo" / "joinery") caps by what inputs exist and never
over-promises: no `photos` → material; no `elements/` → photo. A build never calls a network
API; photos/masks/elements are folders a tool wrote earlier. Mask and element files are named
by the photo's **index in the manifest**, zero-padded to two digits (`docs/photo-manifest.md`).

**Scene and objects** — `scene.py` builds what a *building* is made of (footprint prisms,
checked walls, crowns); `objects.py` is the rest of the cascade, one `*_model` per object
type returning `[(triangle soup, colours, kind)]`. An empty list is a real answer ("no
geometry earned"); there are no extruded fallbacks. The `kind` (solid / slab / glass /
surface) is what the shadow stage reads. No model calls another model; shared geometry is a
shared primitive (`*_mesh`).

**The staged table pipeline** (`docs/design/`, `docs/tables.md`, `docs/pipeline.md`) — store →
working table → kNN graph/MST → global constants → components → labeling → instances →
scene → shadows. Its invariants are in `docs/design/README.md`; the ones that bite:
stage-major execution, halo ≥ reach for any partition, one decision in one place (stage 7
reads stage 6's `type`, never re-derives it), and **a threshold is measured, not chosen** —
every constant in a type test carries the population that justified it in its docstring.

## Conventions that are enforced by review, not by tooling

- **Docstrings carry the measurement.** A non-obvious rule or threshold is documented with the
  numbers that justified it, often with the date and the quote from Kaveh that motivated it.
  Keep that when editing; add it when introducing a constant.
- **Tests run on synthetic data in seconds.** Fixtures build a small tile or a gabled shed in
  numpy (`tests/test_ducklidar.py::tile`, `tests/test_building3d.py::shed`). A feature only
  graduates here with such a test; never make a test depend on `tiles/` or a network.
- **Docs are part of the change.** `docs/api.md` lists every public function; the per-stage
  design docs use the fixed pattern The Problem → The Solution → The Result → Optimize?.
  Proposals are marked **PROPOSAL** in `docs/design/` and in commit subjects.
- **Commit subjects read as a sentence about what changed and why**, e.g.
  `building3d: a wall reaches the ground, and the roof is the top`.
- `roofer` is GPL-3: always a subprocess via `tools/roofer.py`, never linked or installed as a
  dependency. `segment` needs a GPU; `fetch_streetview` needs the user's own key.
- Every point carries a `pid` join key from the store; building folder ids are
  `gers_<first 8 chars of the Overture id>` (or `w<osm way id>`), see `docs/data-layout.md`.
