# The pipeline, sorted by what each step is allowed to know

A survey arrives as tiles of points. A scene leaves as labeled objects. In
between, every computation belongs to one of four levels, defined not by
execution order but by **how far each point's answer can reach**. Get the
level right and partitioning becomes an implementation detail; get it wrong
and tile borders leak into the results.

The lab ([lidar-learning](../../lidar-learning/docs/)) carries the measured
reasoning for every rule; this page is the library-side map: what each level
needs, which `ducklidar` functions serve it, and what form its results take
on disk.

## Level 0 — the store

One spatially sorted Parquet sidecar per survey tile (`dl.to_parquet`,
50 m sort cells). The sort is the index: a row group is a compact patch of
ground, so a window select prunes on row-group statistics instead of
scanning (measured on the Vancouver 2022 tile: 0.9 s instead of 9.7 s, and
column pruning free on top).

After conversion the tile files stop mattering. `dl.read(sources, bbox)`
takes *every* sidecar you have and answers the window across them —
non-overlapping sources cost a header or a row-group statistic. The store
is one logical table; "tile" is no longer a concept queries know about.

There is no KD-tree on disk and none is missing: the sort order does the
tree's job for storage, and per-partition trees rebuild in milliseconds
(a 50 m cell of ~100 k points builds in ~50 ms). Persist query *answers*
(level 2's tables), never the tree.

## Level 1 — row-wise: functions of one point

Echo ratio from `return_number / number_of_returns`, greenness where the
survey is colourised, the vendor classification itself (`dl.describe` /
`dl.census` tell you what a survey actually carries — census before you
plan). No box, no neighbours: the same expression on every row of the
table, streamable over the whole store out-of-core.

Trap: an expression is only row-wise if its *thresholds* are constants.
A percentile computed inside a window turns a row-wise label into a
window-dependent one — compute such thresholds once, globally (the
per-window water level was this bug at level 3).

## Level 2 — neighbourhood-wise: a point and its k nearest

The first level that needs points *co-resident in memory*: k-nearest-
neighbour lookup wants a KD-tree. Everything here derives from one product,
the **kNN-16 table** (`dl.knn` — per point, 16 references and distances,
computed once):

- covariance shape features (`dl.shape_features` — normalised eigenvalues
  e1 ≥ e2 ≥ e3 plus normal verticality, the basis every published shape
  feature is a formula over) read the 16 neighbours' positions;
- the mesh graph (`dl.local_edges(…, nn=…)`, k=10 within a 2.5 m cap —
  the exact edge set the components run on) keeps the capped subset.

Partition + halo makes this level exact, not approximate:

1. **Halo ≥ the largest query radius.** ~1 m covers 16-NN at survey
   density, 2.5 m covers the capped edges, 60 m covers the context rules.
   With the halo filled from neighbouring sidecars (a list to `dl.read`),
   every point sees the same neighbourhood it would in one giant
   unpartitioned run — identical results, bounded memory.
2. **The owner keeps the answer.** Halo points are computed on both sides
   of a border; each partition writes results only for its own points.

Partition size is then a free memory/throughput knob — 50 m cells to
1 km tiles give byte-identical answers.

Durable form: edge tables in Parquet with **coordinates as the node key**
(millimetre integers). Row indices are only meaningful inside one window;
coordinates join across partitions with no shared numbering, so the whole
survey's graph is a folder of edge files DuckDB queries in place.

**One rule about reusing the graph, measured 2026-08-02.** A
vertex-deleted subgraph of a kNN graph is *not* the kNN graph of the
surviving vertices — delete a class and the survivors keep only the edges
they happened to share, never the replacements a fresh kNN would give
them. So:

- **connectivity** questions on a filtered subgraph are safe on the stored
  `knn.parquet` (components only ask whether same-class vertices are joined);
- **distance and path** questions through a filtered subgraph are not —
  the path must route around the deleted vertices, and those detour edges
  exist only in a kNN recomputed on the sub-cloud.

Measured cost of ignoring it: the stage-5 conviction trial run on the
filtered stored table left 553,587 points unreachable and changed 103,180
verdicts against the reference; rebuilt per 250 m block on the sub-cloud
it reproduces the reference exactly.

## Level 3 — component-wise: connectivity

Instances are connected components of the level-2 graph, and connectivity
has unbounded reach — this is the first level a partition cannot answer
alone. Two exact escapes, so the whole graph still never sits in RAM:

- **Components-of-components.** The 2.5 m edge cap means every cross-border
  edge lives within 2.5 m of the border. Reconstruct just the border-strip
  edges, contract each partition's components to single nodes, connect them
  where strip edges cross: components of that small graph *are* the global
  components.
- **The map as registry.** Objects with a footprint or way id (buildings,
  piers, bridges) take identity from the map, not the geometry — the same
  id on both sides of any border, merge by key.

Bounded-hop graph queries (point-to-ground paths, isolation within N
metres) stay partition-local or run in DuckDB over the edge files —
one self-join per hop, out-of-core.

## Level 4 — global: one answer for the whole scene

Decisions where any partition-local estimate is a bug: the water plane
(one sea — pool the evidence, decide once, hand the constant down),
cross-partition object identity (level 3's outputs reconciled), and
shadows (a tower a kilometre outside the window owns your morning light —
compute on the merged scene with a shadow buffer, never per partition).

Global here means *the decision*, not the data: these run on reductions —
instance tables, pooled seeds, merged solids — kilobytes to megabytes,
while the points stay in the store.

## The shape of it

```
store      Parquet sidecars, spatially sorted     dl.to_parquet, dl.read (pid = join key)
level 1    row-wise labels                        columns on the store
level 2    kNN-16 → features, edges               dl.knn, dl.shape_features, dl.local_edges
level 3    components → instances                 one global union-find over knn.parquet
level 4    water plane, identity, shadows         decisions on reductions
```

Each level's output is a table keyed by `pid`, read by the next level. The
only things that ever occupy memory are one partition plus its halo, and
the reductions (a union-find parent array is 4 bytes per point — a whole
nine-tile survey's connectivity fits in 1.5 GB regardless of edge count).

## Execution order is stage-major

Run each level over the *whole* area — materializing its table — before
the next level starts, rather than running the whole ladder on one
partition at a time. Same results either way when halos are right, but
stage-major means: every level's output exists as one complete, queryable
dataset; global constants (water plane, thresholds) are computed between
levels from the whole table instead of patched per partition; components
need no cross-partition repair at all, because level 3 sees every edge at
once; and any level can be rerun or improved without touching the others.
The partition is a memory tool *inside* level 2 — never the unit of
pipeline progress.
