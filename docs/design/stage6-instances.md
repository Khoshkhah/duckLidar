# Stage 6 — instances

**Purpose:** names become things: *this* house, *that* tree, *that* boat.
One row per thing, one instance id per point.

Outputs: only 6.4 writes files; 6.1–6.3 are in-run intermediates — the
contract is data/stage6/instances.parquet + instance_table.parquet
(see [../tables.md](../tables.md)).

## Task 6.1 — components of the class-induced subgraphs

**The Problem:** stage 4 welds touching objects — the tree and the house
it leans on are one component.

**The Solution:** restrict G to edges whose endpoints share a class
(stage 5), take connected components — stage 4's union-find with a label
filter on the edge stream.

**The Result:** a foliage node touching a roof node no longer connects
them, so the tree/house weld of stage 4 separates by construction.
Produces a provisional instance id per point (pid → provisional_inst,
int32, -1 for noise/unassigned) — an in-run intermediate (memory or
transient staging, never a contract file).

**Cost:** same shape as stage 4 (one streamed pass, parent array in RAM).
**Optimize?** Could reuse stage 4's roots as a starting partition —
components only ever split under a mask, never merge — skipping most
union work. Nice trick, only if this stage's runtime ever matters.

## Task 6.2 — map identity for mapped classes

**The Problem:** the same building found in different partitions or
reruns must be recognised as the same thing without matching geometry.

**The Solution:** a building instance is its footprint's way id —
identity inherited from the map (the 3DBAG pattern).

**The Result:** the same id everywhere, so cross-partition fusion is a
key match, not geometry matching. Produces the map key per mapped
instance (provisional_inst → osm way id) — an in-run intermediate,
never a contract file.

**Optimize?** It's a join. Nothing to do.

## Task 6.3 — watershed splits — **BUILT**

**The Problem:** what the subgraph can't separate — the marina raft: one
component, many walkways and boats. Boats and docks all touch within ~1 m
of the pontoon, so the kNN graph, the class-induced subgraph and the MST
all see ONE component. No connectivity rule can cut it.

**The Solution:** cut on **height**, not on adjacency (`dl.raft_split`).
A 0.5 m raster of height above the pontoon (p1 of z); markers from
**h-maxima** at prominence 1.2 m, not plain local maxima — a long hull's
ridge carries several maxima and each would seed its own boat; each large
low ribbon (≥100 m² under 1 m) is a dock and gets its own marker;
watershed on the inverted surface. The trigger is a size prior, not a
threshold on the data: no vessel here is longer than 25 m, so a wider
floating component is by definition a raft.

**The Result — measured, old area:** 19 rafts split into **675 vessels and
docks**. Floating went from 1,461 instances (largest 259,554 points) to
**2,117 (largest 54,473)**.

| | before 6.3 | after |
|---|---|---|
| floating instances | 1,461 | 2,117 |
| largest floating | 259,554 pts | 54,473 |

**Optimize?** Per-instance rasters are tiny; negligible.

## Task 6.4 — fusion

**The Problem:** instances found separately (partitions, reruns) are
pieces of one thing.

**The Solution:** union through shared map ids and same-class edges —
union-find over *instances* (thousands), trivially cheap.

**The Result:** the stage's two contract files, both under
`data/stage6/`: `instances.parquet` (pid i64, inst i32 — one row per
assigned point) and `instance_table.parquet` (inst i32, class u8,
size i64, osm i64 — one row per thing).

**Design status:** the shape is agreed in principle (it re-reads
`objects.py`'s dispatch as graph ops over the tables), but the
implementation is unwritten pending stage 5 — its input is the labels
table. Cost projection: stage-4-like, well under an hour.
