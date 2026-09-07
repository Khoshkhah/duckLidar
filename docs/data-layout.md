# Where the data lives — one folder per building

`building3d` takes explicit paths, so any layout works. This page defines the
**conventional** one, so that a folder of buildings can be handed to a script,
to a colleague, or to a future you, and just run.

```
<root>/
  tiles/                              the point store — LAS/LAZ or the Parquet sidecars
    490000_5457000.parquet
  footprints/                         where the outlines come from (any one of these)
    granville_island.duckdb              a duckOverture / duckOSM extract
    footprints.geojson                   or plain GeoJSON
  buildings/
    <id>/                             ONE FOLDER PER BUILDING — the id is yours (see below)
      footprint.json                  the outline, if not taken from the extract
      photos/
        manifest.json                 the contract — docs/photo-manifest.md
        0001.jpg  0002.jpg  …         the images
        masks/                        00.png, 00_building.png, 01.png, …
        elements/                     00.json, 00.png, 01.json, …
      solid.city.json                 a roofer LoD2.2 solid, if you have one
  out/
    <id>/                             what a build writes
      building.glb  viewer.html  elements.json  meta.json  wall_qa.png
```

Nothing here is enforced by the library. It is enforced by *you* passing
`photos=<root>/buildings/<id>/photos`, and by the defaults: `masks` and
`elements` default to `photos/masks` and `photos/elements`, so if you follow
this layout you pass one path per building and the rest follows.

## Naming

**The building id (`<id>`)** is the folder name and the only name you must
choose. Use a stable, unique identifier from the source you actually query, in
this order of preference:

1. **The OSM way id** — `w714927893`. Stable, public, easy to look up.
2. **The Overture GERS id**, or its first 8 characters when the full one is
   unwieldy — `c4962fc2`. This is what the pilot used.
3. **A slug you invent** — `net-loft`. Fine for a handful of buildings; it
   stops being fine when two people mean different things by `warehouse`.

Prefix the source when you mix them: `w714927893`, `gers_c4962fc2`, `my_shed`.
Keep it to letters, digits, hyphen and underscore, so it is a safe folder name
on every platform.

**Images** may be called anything — the manifest's `file` field is what the
library reads, and it may be absolute or relative to the manifest. Two
conventions that save pain:

- **Sequential** — `0001.jpg`, `0002.jpg`, matching the manifest order. What
  the doc's `NN` mask names refer to.
- **Source-tagged** — `sv_fhahKbQn_348.jpg` (Street View pano and heading),
  `IMG_0431.jpg` (straight off your phone). The fetcher writes this style.

Either is fine. The one rule that is **not** optional: **mask and element files
are named by the photo's INDEX in the manifest**, zero-padded to two digits —
photo `manifest[7]` has `masks/07.png`, `masks/07_building.png`,
`elements/07.json`, `elements/07.png`. Reorder the manifest and you must rename
those, so append new photos at the end.

## The smallest thing that works

Geometry and material need no photos at all — a tile and an outline:

```python
import ducklidar as dl
b = dl.building3d(ring, store="tiles/490000_5457000.parquet",
                  out="out/w714927893", level="material")
```

Add photos and you get textured walls; add masks and the cars and trees stop
being painted on them; add the element folder and the windows and doors become
geometry:

```python
b = dl.building3d(ring, store="tiles/490000_5457000.parquet",
                  photos="buildings/w714927893/photos",     # masks/, elements/ found inside
                  solid="buildings/w714927893/solid.city.json",
                  out="out/w714927893", level="joinery")
```

`level` never silently over-promises: without `photos` it caps at `"material"`,
without `elements/` it caps at `"photo"`.

## Many buildings

Every footprint of an extract that lies wholly inside a tile, in one command —
it writes `buildings/<id>/footprint.json` and `out/<id>/` for each, keeps any
`out/<id>/building.glb` that already exists, and uses `buildings/<id>/photos`
when present:

```bash
python -m ducklidar.tools.build_tile --tile tiles/490000_5457000.parquet \
    --footprints footprints/granville_island.duckdb --root .
```

Or walk the folder yourself. The id is the folder name, and everything else follows from
it:

```python
from pathlib import Path
root = Path("…")
for d in sorted((root / "buildings").iterdir()):
    if not d.is_dir(): continue
    ph = d / "photos"
    dl.building3d(footprint_for(d.name),
                  store=root / "tiles" / "490000_5457000.parquet",
                  photos=ph if ph.exists() else None,
                  solid=next(d.glob("*.city.json"), None),
                  out=root / "out" / d.name,
                  level="joinery", name=d.name)
```

A building with no `photos/` folder gets geometry and material, and says so in
its `meta.json`. That is the normal case on a real block: most walls face no
street, and the honest output is a correctly shaped, correctly coloured
building with nothing invented on it.

## Two rules that are not about naming

**Read with a collar.** A footprint-sized window is too small for some of the
library's own machinery: a cloth ground filter drapes over the roof if the
building fills its window, and any threshold computed as a percentile of a few
hundred thousand points is a different number from the same threshold over a
tile. So the points for one building are read with a margin around the
footprint, and the ground comes from that margin — or from the area, or from
the caller — never from the footprint alone.

**The manifest is input, never output.** A build writes its fitted poses,
its QA and its models into `out/<id>/`, and never edits the folder you gave it.
