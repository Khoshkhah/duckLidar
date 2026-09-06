# Stage 4 — connected components

**Purpose:** which points form one physical piece — connected components
of G, computed once, globally, so nothing is ever cut at a partition
border (a bridge is born whole).

## Task 4.1 — union-find over streamed `knn.parquet`

**The Problem:** the components must be global, but ~450 M edges can't
sit in RAM — and batched, vectorised union-find has a trap: colliding
writes (last-wins) can drop a union. Single-pass batched union-find
silently loses merges — a real bug caught by a unit test.

**The Solution:** stream **`knn.parquet` alone** — never `knn` ∪ `mst`,
because on the completed graph *G′* there is exactly one component by
construction and the 10,499 meaningful islands vanish (the ordering
constraint in design/stage2-graph.md task 2.6). One parent entry per point
in the store (int32,
4 B × |V| ≈ 1.6 GB — the only thing in RAM); `knn.parquet` streams past
in 8 M-row batches and is never resident. Per batch, find both
endpoints' roots (vectorised, with path compression on the walked
chains) and write `parent[max_root] = min_root`; the batch **iterates
until no pair is left split** — every pass merges at least one root, so
it terminates, and dropped writes get retried.

**The Result (measured):** 168.4 M window points, ~450 M edges → 9,548
components in ~30 min; giant component 99.2 % (terrain absorbs
everything path-connected to it).

**Knobs:** batch size. **Optimize? — two live candidates:**
1. The parent array spans the whole store (407 M entries) though only the
   working set (192 M) appears in edges — a pid→dense-index remap would
   halve memory to ~0.8 GB. Only matters on smaller machines.
2. Most of the 30 min is decompressing edge batches, not union-find.
   Reading only during off-peak of other stages, or lighter compression
   for edges, would shave it. Not urgent — it runs once per area.

## Task 4.2 — component ids out

**The Problem:** raw union-find roots are arbitrary pids, not usable
ids.

**The Solution:** roots are looked up for every pid in the features
table, compacted, and ranked by size (comp 0 = largest).

**The Result:** `components.parquet`. Nothing subtle; cost is one more
scan.

**What stage 4 is NOT:** semantics. Touching ≠ same object (the tree
welds to the house it leans on). Class-aware cutting is stage 6, using
stage 5's labels — by design, so this stage never needs re-running when
label rules change.
