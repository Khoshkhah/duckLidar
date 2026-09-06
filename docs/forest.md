# The forest table

`data/stage5/forest.parquet` — one row per point, answering **"what is
holding this point up, and how do I get there?"**

This page is the table on its own terms: what it is, every column, and the
queries it makes possible. The stage that produces it is
[design/stage5-labeling.md](design/stage5-labeling.md) task 5.8; its cost
is in [design/stage5-cost.md](design/stage5-cost.md); the contract summary
is in [tables.md](tables.md).

## What it is

Stage 2 builds a graph *G* = (*V*, *E*, *w*): the points are vertices, and
each point is joined to its k ≤ 10 nearest neighbours within 2.5 m,
weighted by real distance. That graph is deliberately **disconnected** —
a boat, a crown, an isolated roof are their own islands. Stage 2 task 2.6
adds the minimum-weight completion *B* (the MST over the quotient graph),
giving the **connected** graph *G′* = (*V*, *E* ∪ *B*).

**The forest always runs on *G′*.** The union is part of the algorithm,
not a knob: on *G* alone a vertex in a seedless component has no path to
any root, so its answer does not exist — 0.78 % of the reference tile,
and precisely the boats, detached crowns and isolated roofs the question
is about. `dl.forest_graph()` reads `knn.parquet` and
`mst.parquet` together and refuses to run without both.

Now pick the **seed set** *R* — the things that hold other things up:

*R* = *R*_ground ⊍ *R*_water ⊍ *R*_bridge

and run **one multi-source Dijkstra** from all seeds at once. Every vertex
*v* gets three facts:

- **d(v)** — the geodesic distance to the nearest seed, in metres along
  the graph (not straight-line);
- **root(v)** — *which* seed reached it first;
- **pred(v)** — the vertex just before *v* on that shortest path.

The arcs *A* = {(pred(v), v)} form a **spanning forest F = (V, A) of *G′*
rooted at *R*** — the shortest-path forest. Its trees are
*T_r* = {v : root(v) = r}, one per seed, and **every tree carries the
label of its root's system**: label(v) = sys(root(v)). A point in the
water's tree is floating; a point in a bridge deck's tree is on the
bridge; a point in the ground's tree stands on land.

The forest also records **what each path went through** — a path that
climbed a wall is evidence of a building, a path that climbed through
leaves is evidence of a crown. That is the single most useful thing in the
table, and it is why the *path*, not just the label, has to be stored.

## Columns

| column | type | meaning |
|---|---|---|
| `pid` | i64 | the vertex — the point's permanent id (stage 0) |
| `pred` | i64 | its **parent** in the forest: the previous vertex on the shortest path to its root. −1 at a root and where unreached. This column IS the arc set — with it, any consumer walks a point back to its root without re-running Dijkstra |
| `root` | i64 | the pid of the seed that reached this vertex — **the tree's identity** |
| `system` | u8 | **the tree's label**, from its root: `0` ground · `1` water · `2` bridge deck · `255` unreached |
| `dist` | f32 | d(v): geodesic metres from the root, along the graph |
| `depth` | i32 | hops from the root (0 at a root, −1 unreached) |
| `crossed_wall` | i32 | how many wall-like vertices the path crossed — building evidence |
| `crossed_foliage` | i32 | how many foliage-like vertices the path crossed — crown evidence |

Unreached vertices (`system = 255`, `pred = -1`, `depth = -1`) are points
in a component containing no seed. With the MST edges in place this is
0.03 % of the reference tile; without them it was 0.78 %, which is why the
completion exists.

## Two root sets, two files

The same function is run twice, because two different questions need
different seeds:

- **Groundwalk** — *R* = *R*_ground only. This is what reproduces the
  oracle's refinement (measured IoU 0.9938 / 0.9992) and is what the label
  gate is judged on. Use it to ask *"tree or building?"*, because forcing
  every path to start at the ground is what makes a crown's path climb the
  trunk and record the foliage it crossed.
- **Three-system** — *R* = ground ⊍ water ⊍ bridge. Use it to ask *"what
  system holds this up?"* — floating, on a deck, on land. A branch over
  water is reached *from the water* in two steps here, which is the right
  answer to this question and the wrong answer to the previous one.

Distinguished by a `roots` column value or a file suffix (fixed at
implementation).

## What actually uses it today — ONE thing

Grep of every consumer, 2026-08-03:

| column | consumer |
|---|---|
| `crossed_wall`, `crossed_foliage` | **`dl.path_shares` → the path-entry refinement** (`refine_veg` / `refine_building`). Measured IoU **0.9919 / 0.9969** against the oracle. |
| `system` | one log line ("attached N of M") |
| `pred`, `root`, `dist`, `depth` | **nothing yet** — written to the table, read by no rule |

That is the whole list. The refinement asks one question of the forest —
*what share of my path to ground was building, and what share was
foliage?* — and flips labels on the answer.

## What it was DESIGNED for, and is not yet used for

These are the operators the forest was meant to subsume. **None of them
reads it today**; each is still the lab's own block, and two
(`marina_superstructure`, `map_vetoes`) are ported as independent
functions that never touch the forest. Listed as work, not as capability:

- **the crown acquittal** — `where system = 0 and crossed_wall = 0 and
  crossed_foliage > 0` — ⬜ not wired
- **floating** — a point in a water tree above the body's plane — ⬜
- **on-bridge** — a point in a bridge-deck tree — ⬜
- **superstructure** — masts sharing their vessel's tree — ⬜
- **corridor stranding** — ⬜
- **isolation** — walk `pred`, read the `dist` jumps — ⬜
- **"everything this dock carries"** — every vertex whose `root` is one of
  the dock's seeds — ⬜

Until those land, `pred` / `root` / `system` are a structure without a
reader, and the three-system walk has no consumer at all. Two honest ways
forward: port them (making the "spine" real), or shrink the table to
`crossed_*` and drop the rest. Not yet decided.

**One use with a proof behind it, worth doing first** (2026-08-03): the
forest can acquit a large share of task 5.9's defendants for free. If a
point's shortest path in the kNN graph crosses **no building vertex**,
that path survives the deletion of the buildings intact — because an edge
(u,v) whose endpoints both survive is still a kNN edge among the
survivors (removing points can only promote a neighbour, never demote
it). So `crossed_building == 0 and dist < 60` ⇒ reachable ⇒ acquitted,
with no trial. 5.9's trial currently runs on 31.4 M points at 16.7 GB —
the largest memory consumer in stage 5. This needs the forest run with
`cats` = the building CLASS (not "wall-like" shape) and on **knn alone,
without the MST**, since an MST edge is not a kNN edge and would not
survive into the rebuilt graph.

## Notes

- `dist` is **geodesic**, not straight-line: 4 m of walking through a
  crown, not 4 m through the air. Measured consequence — replacing the
  metric with a hop count collapses vegetation agreement from 0.9938 to
  0.7144, because a step inside foliage is 5 cm and a step across a gap is
  2.5 m.
- The forest is a *shortest-path* forest, so its trees are not the
  connected components of *G*: components say what is one physical piece
  (`data/stage4/components.parquet`), trees say what reached it first and
  by what route. Both are needed, and neither substitutes for the other.
