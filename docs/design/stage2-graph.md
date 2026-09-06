# Stage 2 — the graph and the shape features

**Purpose:** build G = (V, E, w) — every point's capped kNN edges — and
each point's neighbourhood shape. The one expensive spatial query in the
pipeline; everything downstream reads its two tables.

## Task 2.1 — partition planning (`dl.balanced_boxes`)

**The Problem:** a fixed grid gives one partition 200 k points and its
neighbour 4 M — memory is unpredictable and parallel workers straggle.

**The Solution:** cut the working bounds into leaves of ≤ MAX_PTS points
each (currently 8 M → 32 leaves of ~6 M), by recursive median splits:
count the box (SQL), split the longer axis at the median (SQL
`approx_quantile`), recurse.

**The Result:** every unit of work the same size, so memory is bounded
*before* work starts and workers never straggle.

**Cost:** ~1 min of counting queries. **Knobs:** `MAX_PTS`, `min_side`.
**Optimize?** Counts re-scan the store per tree node (~2× leaves
queries); could be one pass building a coarse histogram instead —
worth it only if planning ever dominates (it doesn't: 1 min vs ~30 min
sweep).

## Task 2.2 — the read window (ownership + halo)

**The Problem:** queries near a leaf's border look outside it, yet the
results must be identical to a whole-area run.

**The Solution:** each worker reads its leaf + halo; owns (writes
results for) only the leaf, half-open edges. The rule: halo ≥ the
farthest any query in this stage reaches.

**The Result:** reads overlap; writes never do.

**Settled by measurement, 2026-08-02:** halo = **30 m**, and the
invariant is now *verified every run* rather than assumed — each
partition asserts that no owned point's 16th neighbour lies beyond the
halo, and prints the margin (`reach 18.9/30 m`).

The measurement that decided it (29.6 M sampled points across four
boxes): reach is p50 0.37 m, p99.99 5.5 m, **max 18.9 m** — and the long
tail is not dense city (ground never exceeds 6.5 m, city fabric 8–9 m)
but **sparse water**, where a lone return's 16th neighbour sits at
p99.9 9.9 m. So the 10 m halo this note used to propose would have
truncated real neighbourhoods; 60 m was safe but read ~1.7× the points
each partition owns. 30 m keeps a 1.6× margin over the observed maximum
at ~1.3× amplification, and the assertion catches any survey where that
margin is wrong instead of silently corrupting features.

## Task 2.3 — kNN (`dl.knn`)

**The Problem:** the neighbour query is the expensive one — and the old
pipeline computed it twice (PDAL, then the graph) and threw both away.

**The Solution:** one KD-tree per partition; every point's 16 nearest +
distances (k=17 query, self excluded in the arrays).

**The Result:** a single query both downstream products derive from.

**Cost:** the tree build + query ≈ tens of seconds per 6 M-point leaf,
multi-threaded. **Optimize?** cKDTree is already parallel; the win came
from *not duplicating* the query. GPU or approximate NN would trade
exactness for speed — refused: the edge set defines the graph everything
else trusts.

## Task 2.4 — shape features (`dl.shape_features`)

**The Problem:** the rules judge each point by its neighbourhood's shape
(planarity, scattering, linearity), and the values must be comparable to
what PDAL produced.

**The Solution:** per point, covariance eigenvalues of its neighbourhood
(self-inclusive, PDAL's convention), normalised, + normal verticality.
Stored as the basis (e1, e2, e3, nz); planarity/scattering/linearity are
formulas over them, derived in queries (sqrt-eigenvalue ratios for
PDAL-comparable values).

**The Result:** measured equivalent to PDAL — median |Δ| 0.004, residual
= kNN tie-breaking.

**Cost:** batched `eigh` over (n, 3, 3) — roughly a minute per leaf.
**Optimize?** A closed-form 3×3 eigen solver (Cardano) is ~3× faster than
LAPACK here; worth trying if stage 2 ever needs to be faster. Storing
f16 instead of f32 would halve the table — not worth the precision risk
near rule thresholds.

## Task 2.5 — edges (`dl.local_edges`)

**The Problem:** the graph needs its edges, but recomputing neighbours
would duplicate task 2.3's query.

**The Solution:** keep the first k ≤ 10 neighbours within 2.5 m as
directed weighted edges (src, dst, len), reusing the task-2.3 arrays
(`nn=`). Each edge written once, by its src's owner.

**The Result:** the edge table of G.

**Optimize?** Nothing measurable — it's a mask and a gather over arrays
already in RAM.

## Task 2.6 — completing the graph: the MST over the quotient graph

**The Problem.** Let *G* = (*V*, *E*, *w*) be the graph stage 2 builds:
*V* the working points, *E* = {(u,v) : v is among u's k ≤ 10 nearest and
‖u−v‖ ≤ 2.5 m}, *w*(u,v) = ‖u−v‖. The cap makes *G* **disconnected** —
its connected components are the physical islands (stage 4 finds
*k* = 10,499 of them on the green window, one holding 99.2 % of *V*).
Every query that walks paths — the support forest, any path-to-ground
question — is undefined on a vertex whose component contains no root:
0.78 % of the reference tile's vertices are unreachable for this reason,
and they are precisely the objects whose support is most in question
(boats, isolated roofs, detached crowns).

**The Solution.** Complete *G* to a connected graph by adding a
**minimum-weight completion**, defined on the quotient graph:

- Let *C* = {C₁ … C_k} be the connected components of *G*.
- Let **H = (C, E_H, w_H)** be the **quotient (contracted) graph**: one
  vertex per component, and for i ≠ j
  &nbsp;&nbsp;*w_H*(i,j) = min { ‖u − v‖ : u ∈ C_i, v ∈ C_j }
  — the **closest-pair distance** between the two components. Each such
  edge is **realized** by its argmin vertex pair (u\*, v\*) ∈ C_i × C_j.
- Let **T = MST(H)** and let **B** be the set of realizing vertex pairs of
  T's edges. |B| = k − 1 when H is connected.

Then **G′ = (V, E ∪ B)** is connected, and by MST optimality on H no other
set of k − 1 inter-component edges has smaller total weight. B is exactly
the lab's "mesh bridges" (`rules/mesh_graph.py`, `scene.mesh_bridges`),
stated as what it is.

**Computation.** **Borůvka's algorithm on H**: each round, every current
super-vertex selects its minimum-weight outgoing edge; the selected edges
are added and their endpoints merged; O(log k) rounds. The realizing
edges cannot be read off *E* — an inter-component pair is by definition
farther apart than the 2.5 m cap — so each is a **closest-pair query
between point sets**, answered by spatial search.

**Ordering.** B depends on *C*, and *C* is computed from *E* alone
(stage 4). So the pipeline order is edges → components → bridges, and
**stage 4 must keep computing components on *G*, never on *G′*** — on
*G′* there is exactly one component by construction, and the 10,499
meaningful islands would vanish. The artifact belongs to the graph
(stage 2); the computation runs once components exist.

**The Result:** `data/stage2/mst.parquet` — `src i64, dst i64,
len f32` in the same pid space as `knn.parquet`, one row per MST edge of
H (k − 1 rows). A consumer that wants path queries unions the two tables;
a consumer that wants physical islands reads `knn.parquet` alone. Both
are first-class: *G* answers "what is one physical piece", *G′* answers
"what path exists".

**Scale.** k is small (10,499), so MST(H) itself is trivial; the cost is
the closest-pair queries over 192 M vertices. The plan is the two-level
pattern stage 4 uses: candidate realizing edges per balanced partition
(with halo), unioned into a candidate H, then Borůvka; any component left
isolated in H falls back to an exact widening search, which is what the
lab's `_exact_bridge` does.

## Task 2.7 — staging & consolidation

**The Problem:** a long sweep can die mid-run, and after stage 1 every
output must end up a single table (one file per fact).

**The Solution:** each partition writes transient `part-NNNN` files
(resume capability: a rerun skips finished parts); the run ends by
concatenating parts into `features.parquet` + `knn.parquet` and
deleting staging. **No sorting** (decided 2026-08-02): every consumer
streams the whole table or hash-joins on pid — nothing range-queries
these tables, so ordering would pay for pruning nothing uses.

**The Result:** two tables, and a resumable run.

**Optimize?** Concatenation re-encodes ~11 GB once (~minutes). Could be
raw row-group copies (no re-encode) via pyarrow — small win, low risk,
not yet needed.

## Task 2.8 — parallel execution

**The Problem:** 32 leaves of work on a 23 GB machine.

**The Solution:** partitions are independent by construction
(invariant 3), so a process pool runs them; 3 workers × ~4 GB fits the
23 GB machine.

**The Result:** the sweep runs in parallel within memory bounds.

**Knob:** `--workers`. **Optimize?** More workers contend for cores
(each partition's kNN is itself threaded); 3–4 is the sweet spot here.
On a bigger machine, workers scale linearly until disk read saturates.
