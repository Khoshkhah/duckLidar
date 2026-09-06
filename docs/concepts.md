# LiDAR concepts — the ten-minute background

Everything the API assumes you know, and nothing more. If you have worked
with LiDAR before, skip to [getting started](getting-started.md).

## What the data is

An aircraft flies over a city firing laser pulses downward — hundreds of
thousands per second — and times the reflections. Each reflection becomes
a **return**: a point with x, y, z plus a record of how it was measured.
A survey of one square kilometre is typically tens of millions of returns.

A return is not just coordinates. The fields that matter:

- **`return_number` / `number_of_returns`** — one pulse can reflect
  several times: off a branch, a lower branch, then the ground. A pulse
  that split is almost always vegetation — solid surfaces (roofs, roads)
  return once. This ratio is the cheapest vegetation detector there is
  (`echo_ratio`).
- **`classification`** — most surveys ship with a vendor-assigned class
  per point (ground / building / vegetation / water / noise — the ASPRS
  code list, `dl.ASPRS`). Quality varies; treat it as evidence, not truth
  (`census` tells you whether canopy can even be told from roof in a
  given file).
- **`intensity`** — how strongly the surface reflected. Water absorbs,
  metal glares.
- **`scan_angle`, `gps_time`, `point_source_id`** — how oblique the look
  was, when it happened, and which flight line it belongs to. Together
  they let you reconstruct where the aircraft was (`sensor_track`) and
  which *side* a wall was seen from (`viewed_from`).
- **`red, green, blue`** — some surveys are colourised from imagery.

`dl.describe(file)` prints all of this for a real file — run it before
planning anything, because a survey that lacks a field lacks the analyses
built on it.

## The shadow problem

The laser only measures what it can see from above. Under every roof
there is a hole in the data; behind every tower there is a wedge the
scanner never reached. This is the single most important fact about
airborne LiDAR:

**absence of points is not absence of things.**

It is why this library returns rasters with NaN holes, `void_report`
(how empty, how clumped, how far from evidence), and per-cell intervals —
and why `checked_walls` treats a wall the laser couldn't see differently
from a wall it saw through.

## The map owns the plan, the survey owns the height

A survey and a map know different things, and neither may do the other's job.
Overture says WHERE a thing is and what shape it is in plan; the returns say
how high it is.

Getting this backwards is measurable. A dock's width taken from the returns
comes out as 0.87 x whatever window you probe — a marina has no edge, because
the boats moored either side float at pontoon height too. A building's height
taken from the map comes out as a blanket 33.0 m above the maximum return, or
6.97 m on a 4-storey block. Each is the other's job done badly.

The corollary is that a map claim still needs a physical guard: a footprint
says nothing about height, so the map alone would claim the water under a
bridge and the crown over a roof. See
[type-guards](design/type-guards.md).

## A type must be earned

The ten labels say what MATTER is — ground, vegetation, building, water. They
do not say what a THING is. Car, boat, dock, tree, lamp are stage-6 **types**,
and each has a named test that asks a physical question: is it compact, is it
floating, is it rooted in the ground, is it reachable by road. A thing that
answers wrong becomes `unknown`, which is a real answer and not a failure —
there is more unnamed matter in a city than there are named categories.

## Surfaces: DSM, DTM, nDSM

- **DSM** (digital surface model): the *top* surface — highest return per
  cell. Roofs, canopy, wires.
- **DTM** (digital terrain model): the *bare earth* — ground returns
  only. Half its cells are holes (the shadow problem); filling them is
  interpolation, i.e. modelling, and `dem()` keeps the distinction: every
  cell knows whether it was measured or filled, and how far the nearest
  real evidence is.
- **nDSM**: DSM minus filled DTM — height above ground. The input to
  "what is a building, what is a tree" questions.

`surfaces()` builds the usual four (dsm / dtm / canopy / roofs) in one
call.

## From points to objects

Points don't come labelled "this building". Getting objects out is a
chain, each level needing more context than the last:

1. **per point** — fields of the return itself (echo ratio, colour);
2. **per neighbourhood** — a point plus its k nearest: local shape
   (flat? scattered? vertical?), the kNN mesh (`local_edges`);
3. **per component** — connected pieces of that mesh are instance
   candidates (`instances`, boundaries by `edge_strength`);
4. **per scene** — decisions no window can make alone: the water plane,
   object identity across partitions, shadows.

[pipeline.md](pipeline.md) treats this ladder in depth — it is the
architecture of any larger system built on this library.

## Coordinate systems

Everything works in the survey's projected CRS (metres — e.g. UTM). A
window is always an axis-aligned `(minx, miny, maxx, maxy)` box built
with `dl.box(cx, cy, half)`. Nothing here reprojects; if your map data is
in lon/lat, transform it first (see `pier_ways` for the pattern).
