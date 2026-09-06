# Stage 7 — the object cascade

**Purpose:** every stage-6 instance becomes the geometry its class earns.
Not just buildings.

## Task 7.0 — per-class dispatch (`data/stage7/scene_export.npz`)

**The Problem.** Stage 7's first implementation modelled **buildings and
nothing else**: `build_scene` opens with `bld = flatnonzero(cls ==
BUILDING)` and never looks at another class. So stage 6 does the work of
separating 2,117 floating instances — 675 of them vessels cut out of
welded rafts by 6.3 — and stage 7 discards every one. The same for the
bridge, for street furniture, for the docks. The rendered page showed
terrain, buildings, sea and crown points, while the lab's `scene3d.html`
showed seven object layers.

**The Solution.** One dispatch over the instance table, each class routed
to the recipe that suits it, all in `ducklidar.objects`:

| stage-5 class | instance source | geometry |
|---|---|---|
| building, glazing | 6.2 map id, 6.4 fused | measured roof solid + walls on trial |
| floating | 6.3 raft-split per vessel | hull + cabin + mast, or a dock slab |
| small, on_bridge | 6.1 components | car boxes in the object's own frame |
| bridge | 6.1 components | 2.5-D solid from the deck returns |
| veg_high | 6.1 components | crown points — no meshes, as v1 |

**The design change from the lab.** Its recipes take a `ctx` object
carrying an entire run — points, labels, DEM, water level, OSM, roofer
output — so they cannot be called without assembling a lab run first.
Here each recipe takes **the arrays it needs**; a datum it cannot derive
(the water level, the ground under a lamp) is an argument. That is what
makes them reachable from a table pipeline.

## Why this is better than the lab's, not a weaker copy

**Roofs are measured, not fitted.** The lab's buildings come from
`roofer`, an external LoD2 fitter which replaces a roof with idealised
primitives — and which is a GPL binary that lived in `/tmp` and did not
survive a reboot. `dl.objects.solid_25d` takes the roof surface **from the
returns**: a top TIN on the instance's own grid, boundary walls, a flat
cap. Dormers, parapets, stair heads and plant rooms survive because they
were measured. This is the project's standing rule — claim what was
measured — applied to shape instead of to labels.

**Walls stand trial.** The lab's own comment on its roofer path reads
"roofer adds walls without checking". `dl.scene.checked_walls` keeps a
wall where the returns evidence one, cuts openings where they frame one,
and removes it where the laser saw floor under the roof.

**Better instances underneath.** Boats come from the 3-D graph plus the
height watershed of 6.3, not a 2-D cascade.

**The Result — measured on the lab's own study box**
(490039, 5457307, 490539, 5457807), 10.7 M of the tile's 42.7 M returns,
7 s:

| layer | this pipeline | the lab | |
|---|---|---|---|
| building | **396** | 268 | measured roofs, no roofer |
| boat | **265** | 79 | 6.3's raft split reaching the scene |
| object | **989** | 354 | cars + street furniture |
| bridge | 35 | 2 | **fragmented — see below** |
| dock | 0 | 35 | **missing** |
| floating (float homes) | 0 | 24 | **missing** |
| glass | 0 | 175 | **missing** |

## What is still missing, and why

Three of the lab's layers have no counterpart yet, and the reason is the
same for two of them: **the label set does not carry the class.** v1's
cascade has `DOCK` and `MOORAGE` as classes of their own; stage 5 decides
only the ten in `dl.DECIDE`, so a dock and a float home both arrive as
`floating` and get a vessel recipe. Splitting them needs either a
stage-5 class or a stage-6 rule keyed on the map's pier ways
(`dl.pier_ways`), which is the natural source — a dock is a mapped thing.

`glass` is different: `glazing` **is** a stage-5 class (158,464 points on
the tile) and 5.9's tribunal decides it. Today stage 7 folds it into
`building` because a window is part of the building. To render it as its
own translucent layer the roof-glass and facade-opening parts have to be
separated out of the building recipe, which is geometry work, not a
labelling gap.

`bridge` at 35 objects against the lab's 2 is a **weld/fragment
mismatch**, not a missing class: stage 6 leaves the deck in components
that the lab's cascade fuses along the map's bridge ways.
