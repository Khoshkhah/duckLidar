# Stage 2 — the Euclidean MST (Kruskal inside the cap, Borůvka beyond it)

**Purpose:** one canonical, parameter-free structure over the survey's
points — the minimum spanning tree of the point cloud itself, computed
independently of the kNN graph.

## Why a real MST and not a patch

The kNN graph carries two arbitrary numbers: k = 10 and cap = 2.5 m. Both
were choices. The Euclidean MST has **no parameters** — for a given point
set it is unique (when distances are distinct) and it is the same tree
whatever anyone later decides k or the cap should be.

It also subsumes a stage:

> **Delete every MST edge longer than *t* and what remains are exactly the
> single-linkage clusters at scale *t*.**

So the MST holds stage 4's answer *at every scale at once*. Components at
2.5 m become `where len <= 2.5`; at 1 m or 5 m, a different filter over
the same table. The cap stops being a commitment baked into a computation
and becomes a query parameter.

And connectivity comes free: a spanning tree is connected by definition,
so the completion stage 5.8 needs is a by-product rather than a repair.

## The definition

Let *V* be the working points and *w*(u,v) = ‖u − v‖ (3-D Euclidean).
The **Euclidean minimum spanning tree** *T* is the spanning tree of the
complete graph (*V*, *V*×*V*, *w*) of minimum total weight; |T| = |V| − 1
edges. Note *T* is defined on the COMPLETE graph — every pair is a
candidate — which is exactly what makes it independent of the kNN graph.

## The three steps, in plain words

**Step 1 — the bucket pass.** *The problem:* Kruskal must see the edges
shortest-first, and there are 426.8 M of them on one tile (1.9 B on the
window); sorting that many at once is expensive. *The solution:* every
length lies between 0 and 2.5 m, so no comparison sort is needed — set out
**100 boxes** (0–2.5 cm, 2.5–5 cm, … 2.475–2.5 m), read each edge once and
drop it in the box its length belongs to. *The result:* "shortest first"
becomes "box 0, then box 1, …", and the only real sorting is a few million
edges inside one box at a time. **Measured: 441 s** — and it is pure
re-reading of edges stage 2 already had in hand, which is why writing the
boxes during stage 2 deletes this step.

**Step 2 — the Kruskal sweep.** *The problem:* out of all those candidates,
choose exactly the edges that form the tree. *The solution:* walk them
shortest-first and ask one question per edge — **are these two points
already connected, directly or through a chain?** If yes, skip it: it would
close a cycle, and whatever already connects them is shorter. If no, keep
it and record that the two sides are now one. The bookkeeping is union-find,
which answers "already connected?" in almost constant time. *The result:*
the MST of everything the kNN edges could reach. **Measured: ~190 s**, and
heavily front-loaded — 28.3 M of the 42.7 M edges are accepted in the first
7 boxes of 100, because short edges greedily consume the merges.

**Step 3 — the Borůvka completion.** *The problem:* the sweep joins
everything that was within 2.5 m of something. What was never within 2.5 m
of anything — boats, detached crowns, isolated roofs — is still separate:
3,320 pieces on the reference tile. *The solution:* every remaining piece
asks one question, **"what is the nearest point that is not mine?"** — all
of them at once — takes the answer as an edge, and merges. Then the merged
groups ask again. *The result:* one connected spanning tree.
**Measured: 4 rounds, 13 s** — 3,320 → 341 → 32 → 4 → 1.

**Why two different algorithms.** The two halves of the problem have
opposite shapes. Inside the cap there are *billions of candidate edges but
they are already computed*, so streaming wins and searching is pointless.
Beyond the cap there are *only thousands of pieces but no candidates at
all*, so searching wins and streaming has nothing to stream.

## The algorithm: Kruskal inside the cap, Borůvka beyond it

Borůvka is the right *framework* — but executed literally it is the wrong
*schedule* for this data, and the arithmetic says so before any code is
written.

**Borůvka's shape.** Every component picks its minimum outgoing edge, all
are added at once, components merge; ≤ ⌈log₂|V|⌉ rounds. Correct by the
**cut property**: the minimum-weight edge crossing any cut belongs to some
MST, and each selected edge is exactly that for the cut (C, V∖C). Ties
broken by (len, src, dst), so runs are reproducible and no cycle forms.

**Why round-per-pass fails here.** The cost unit is one pass over the edge
table: **measured, 22 minutes** for stage 4's union-find to stream 1.9 B
rows out of 12 GB of zstd. Round 1 (group by src, take the shortest) is
one such pass and is cheap. But it leaves roughly |V|/2.5 ≈ **77 million**
components, not a handful — and every later round needs another full pass
to find each component's minimum outgoing edge. ~26 rounds × 22 min ≈
**9–10 hours** for the window, before any exactness work. The error is
reasoning about the end state (10,499 fat components, where per-component
queries are cheap) while ignoring the middle (tens of millions of tiny
ones, where they are not).

**The schedule that works — two regimes, split at the cap:**

**(a) Inside 2.5 m — Kruskal in sorted order, not Borůvka.** Every
candidate edge shorter than the cap is already in `knn.parquet`. Sorting
those edges by length and streaming them through a union-find (**Kruskal**)
achieves in *one* pass what 26 Borůvka rounds achieve, because processing
in globally increasing order means every edge accepted is the minimum
crossing its own cut at that moment — the same cut-property guarantee, one
sweep instead of one sweep per round.

Better still, the lengths are **bounded** (0 … 2.5 m), so the sort is a
**bucket sort**, not a comparison sort: write each edge into one of ~100
length buckets of 2.5 cm in a single pass, then consume the buckets in
order. Edges within a bucket may be processed in any order — an error
bounded by the bucket width — so use exact ordering *within* a bucket by
sorting each bucket in memory (a bucket is ~19 M rows, trivially sortable).
Two passes total, and the union-find is the same O(α) structure stage 4
already uses.

**(b) Beyond 2.5 m — Borůvka over what remains.** After (a), the surviving
components are exactly the kNN graph's components (measured: 10,499, one
holding 99.19 %). Here Borůvka is ideal, because a per-component query is
now cheap and there are only thousands of them: each component asks for its
closest point outside itself, the answers are added, components merge,
repeat — 2–3 rounds in practice. The query is answered by a **bounded ball
search** on the spatially sorted store (a row-group prune, not a scan),
widened by doubling for the genuinely isolated pieces, and always asked
**from the smaller side** — *w* is symmetric, so the 190 M-point component
is never searched.

**Exactness, stated precisely.** (a) yields the exact MST of the graph
whose edges are the kNN pairs; (b) yields the exact minimum-weight
completion over the components. Their union is the exact Euclidean MST
**iff** every EMST edge shorter than 2.5 m is a kNN pair. That can fail
only where a point has ≥ 10 neighbours closer than its EMST partner, so it
must be **checked, not assumed**: for a sample of accepted edges, confirm
by bounded ball query that no shorter cross-component pair existed at the
moment it was accepted (verification 3 below), and report the rate.

## Parallelism — where it exists and where it cannot

**Kruskal is the sequential-friendly algorithm; Borůvka is the
parallel-friendly one.** Phase (a) chose Kruskal for pass-efficiency, so
its core sweep is sequential *by nature*, not by implementation:

**Parallel:**

1. **The bucketing pass** — scattering 1.9 B edges into ~100 length
   buckets is I/O-bound; DuckDB threads it natively.
2. **Sorting the buckets** — 100 independent sorts.
3. **Phase (b), the Borůvka completion** — 10,499 components each asking
   for their nearest outside point are independent queries. This is where
   worker parallelism belongs, and it is the phase that scales with cores.
4. **The verification sampling** — independent by construction.

**Not parallel: the union-find sweep.** An edge is accepted based on the
state every earlier edge produced. Parallelising it means changing the
algorithm, not the code.

**And it does not need to be — it needs to be COMPILED.** 1.9 B union-find
operations at ~10–20 ns each is **30–60 s in compiled code**. The failure
mode to guard against is writing that loop in interpreted Python, where
the same work takes hours and looks like an algorithmic problem when it is
a language problem. Use a jitted kernel (numba if available) or a
vectorised batched union with a convergence loop — but if a batched form
is used, it must be **proved equal to the sequential sweep** on a sample,
because batching within a bucket can accept two edges that close a cycle,
and "it looked fine" is not proof.

**Measure the sweep separately** from the bucketing and the completion, so
a slow run points at the phase responsible instead of at the algorithm.

## Cost, predicted before the run

| | old area (42.7 M pts, 427 M edges) | window (192 M pts, 1.9 B edges) |
|---|---|---|
| bucket + Kruskal inside the cap | ~5–8 min | ~45–60 min |
| Borůvka completion beyond the cap | ~1–2 min | ~5–10 min |
| exactness checks | ~5 min | ~15–20 min |
| **total** | **~15 min** | **~1.5 h** |

versus ~10 h for round-per-pass Borůvka. Memory is not the binding
constraint in either schedule: a union-find parent array is 4 bytes per
point (0.77 GB for the window) and one bucket at a time is ~19 M rows.
**The run must report actuals against this table** — a prediction that is
never checked is decoration.

## Why the point cloud makes this cheap

Measured on the green window (`data/stage4/components.parquet`, which is
the kNN graph's own component structure at cap 2.5 m):

- after the short edges are consumed, **one component holds 190,402,957
  points (99.19 %)** and 10,498 hold 1,547,812 between them, median size
  **15 points**;
- so from round ~2 onward the work is dominated by a few thousand tiny
  components, each needing one local query over a handful of points;
- and the query is always asked **from the small side**: *w* is symmetric,
  so the giant's minimum outgoing edge is whichever pebble found it. The
  190 M-point component is never searched.

## Memory plan

| resident | size |
|---|---|
| point coordinates, float32 (x, y, z) | 2.3 GB |
| component label per point, int32 | 0.77 GB |
| the tree so far (src, dst, len), growing to \|V\|−1 | ≤ 3.1 GB, spilled per round |
| one local neighbourhood at a time | MB |

Peak target **≤ 6 GB**, with the edge table written out per round rather
than held. Nothing here needs a KD-tree over all 192 M points — the
global tree (≈ 10 GB with scipy's float64) is what this design exists to
avoid.

## Output

`data/stage2/mst.parquet` — `src i64, dst i64, len f32`, |V| − 1 rows
(≈ 3.8 GB). Independent of `knn.parquet`; neither derives from the other.

Consumers:

- **connectivity** for the support forest: *G′* = kNN ∪ MST is connected
  by construction. The kNN edges still carry the local structure (many
  paths, honest geodesics); the MST only supplies the long links the cap
  cut. A tree alone would distort distance — every path forced through
  one route — so the union matters, not the MST alone.
- **components at any scale**: `where len <= t` and take connected
  components of the result; at t = 2.5 m this reproduces stage 4.
- **isolation**: the length of the MST edge attaching a piece is exactly
  how weakly it is attached.

## Measured — the old area, 2026-08-02 (first prototype)

Superseded by "Measured — the run of record" below, which re-ran the same
area with the shipped implementation: 414 s / 4.11 GB, not 744 s / 6.5 GB.

42,707,336 points, 426,761,290 candidate edges, on 20 cores / 23 GB.

| step | time | note |
|---|---|---|
| bucket pass | **441 s** | 59 % of the run; pure Parquet read at ~1 M edges/s |
| Kruskal sweep (100 boxes) | ~190 s | 28.3 M of 42.7 M edges accepted in the first 7 boxes |
| Borůvka completion | **13 s** | 4 rounds: 3,320 → 341 → 32 → 4 → 1 |
| **total** | **744 s ≈ 12.4 min** | predicted ~15 min |
| **peak RSS** | **6.5 GB** | predicted ≤ 6 GB — slightly over, in the completion phase |

Result: **42,707,335 edges = |V| − 1 exactly** (a spanning tree, no gaps,
no cycles); total weight 6,843,702.5 m; lengths p50 0.142 m, p99 0.499 m,
max 22.93 m; **1,891 edges exceed the 2.5 m cap** — 0.004 % of the tree,
and precisely the links the kNN graph could never supply.

**The correction this measurement forces.** The bucket pass is 59 % of the
run and re-reads edges stage 2 held in memory moments earlier. Stage 2 must
write the length boxes as it computes the kNN (decided 2026-08-02), turning
441 s into a ~20 s raw read: the old area drops to ~4 min and the window to
well under an hour. Same trick as computing the kNN once instead of twice —
do not pay twice for data you already had.

## Measured — the run of record, 2026-08-02 (both areas)

The implementation is `ducklidar.graph.euclidean_mst` (+ `read_mst_parts`),
driven by `shadowCity2/tools/stage2_mst.py`. Every number below comes from a
run; logs in `shadowCity2/out/stage2_mst_{old,window}.log`, one line per run
in `out/pipeline_report.log`.

One departure from the section above, and it is a schedule change inside the
design, not a redesign: **phase-(b) round 1 is one blocked sweep of the sorted
store** (200 m core + 25 m halo, k = 32 probe, an answer certified only when
the found distance is inside the halo margin — a 2-D box bounds a 3-D ball
because 3-D distance ≥ horizontal distance) instead of one bounded ball query
per component. It also caches the small side's coordinates, so no later round
reads the giant again. Anything the sweep cannot certify, and every later
round, falls back to the bounded ball query with doubling exactly as designed.
It costs more seconds than the earlier per-component prototype (44 s vs 13 s
on the old area) and buys all 3,320 components in one pass.

### Cost and memory, actuals against the prediction

| | old area (42.7 M pts, 426,761,290 candidates) | window (192.0 M pts, 1,918,835,200 candidates) |
|---|---|---|
| bucket pass | 204.0 s | ~250 s |
| Kruskal, 100 buckets | 148.5 s | ~945 s |
| component sizing | 7.7 s | 15.9 s |
| Borůvka completion | 44.2 s (4 rounds) | 189 s |
| consolidate to Parquet | 9.6 s | ~22 s |
| **total** | **414 s = 6.9 min** (predicted ~15 min) | **1,736 s = 28 min 56 s** (predicted ~1.5 h) |
| **peak RSS** | **4.11 GB** | **10.33 GB** |
| verification pass (separate) | 94 s + 99 s, peak 7.33 / 4.37 GB | 165 s, peak 7.56 GB |

The old area ran in one foreground call (budget 520 s, never spent). The
window ran in 4 calls under the resume checkpoint; ~120 s of the 1,736 s is
work redone after call 1 overran its budget, so ~1,632 s is process time.

**The 6 GB memory target is wrong, and by 72 %.** The window peaked at
**10.33 GB**, in Kruskal buckets 4–7 (0.100–0.200 m), where the accept filter
still leaves 40–68 M candidate edges crossing components in one bucket
(bucket 5: 114.7 M candidates, 67.6 M still crossing). The memory plan above
counted the resident arrays and the "one bucket at a time" — it did not count
that a *bucket* at the mode of the length distribution is a tenth of a
billion 20-byte records, which is 2.3 GB before the sort's copy. Later calls
peaked at 6.02 GB (buckets 8–20) and 4.42 GB (buckets 21–99 + Borůvka); the
4.42 GB in the tool's report line is that last call's `ru_maxrss`, not the
run's. The real bound is **~11 GB for the window**, still well inside 23 GB;
the fix, if it ever needs one, is to split the mode buckets, not to change
the schedule. (Call 1's own peak was lost: its stdout was block-buffered and
died with the SIGTERM at 600 s. Every later call ran `python -u`.)

The other prediction that missed: the **bucket pass is no longer 59 % of the
run** (204 s of 414 s = 49 % on the old area, ~250 s of 1,736 s = 14 % on the
window). Kruskal, not the read, dominates at window scale.

### The four verification results

**1 — edge count, acyclic, connected. PASS, both areas.** Old: 42,707,335
edges = |V| − 1 exactly; window: 191,950,768 = |V| − 1. Union-find over the
whole output edge set ends with **one** component in both. |E| = |V| − 1 plus
one component *is* a spanning tree. Total weight 6,843,702.973 m (old),
28,831,927.9 m (window).

**2 — total weight ≤ the cheap alternative. PASS.** The wording needed one
decision to become measurable: the shared term is the MST of the kNN graph,
which is literally the phase-(a) output (old area: 6,835,928.744 m over
42,704,015 edges); what differs is the completion over the resulting
3,321-node quotient. Ours is the Borůvka MST of that quotient with true
inter-component distances; the alternative it replaces is the **star** —
attach each of the 3,320 leftover pieces straight to the 99.01 % main body.
The star was measured, not assumed (1,059 pieces needed their own bounded
ball query against the giant, doubling from r₀ = 5 m).

- ours 6,835,928.744 + **7,774.230** = 6,843,702.973 m
- star 6,835,928.744 + **17,060.792** = 6,852,989.536 m

Ours ≤ alt: **true**; the completion alone is 54.4 % cheaper, 9,286.562 m
saved on the tile.

**3 — cut-property spot-check. PASS, 50/50, 0 differ.** Fifty sampled
components, each brute-forced on a local neighbourhood, agree with the
selected minimum outgoing edge.

**4 — threshold agreement with stage 4. FAILS, and both directions are
real.** Cutting the old area's MST at 2.5 m gives **1,892** components;
`data/stage4/components.parquet` restricted to the same points gives
**3,058**; the (our-root, stage-4-comp) relation has 3,079 pairs, so it is
not a bijection.

*Merges (the finding about k = 10).* 10 of our components each swallow more
than one stage-4 component — 1,197 of them in all, 309,147 points moving.
The cause is measured: of the 3,320 completion edges, **1,429 are shorter
than 2.5 m** — point pairs inside the cap that k = 10 never recorded. Median
example, length 0.9933 m: pid 259,733,407 at (490082.476, 5457946.373, 4.504)
in stage-4 comp 3617 and pid 259,733,421 at (490082.447, 5457947.280, 4.100)
in comp 0. Why k = 10 could not see it, measured on the store: the first
point's 10th neighbour is 0.5450 m away and the second's is 0.7764 m — both
nearer than the 0.9933 m link, so the pair falls off the end of both lists.
Exactly the truncation failure mode this design named in advance. The window
shows the same thing at scale: 10,616 completion edges of which only 4,896
exceed the cap, so **5,720 are pairs within 2.5 m that k = 10 missed**.

*Splits (a boundary artifact, not a finding).* One stage-4 component — comp 0,
the window's giant — lands in 22 of ours, 77,643 points outside its largest
piece. This can only come from stage-4 connectivity routing through points
*outside* the old tile: phase (a) is the MST of the kNN graph restricted to
V = tile members, so its components are the kNN-within-tile components, and
completion edges only coarsen that. Confirmed by measurement: all 77,643
points lie within 65.26 m of the tile edge (p50 28.6 m, p90 54.5 m), against
p50 127.7 m / p90 284.8 m for comp 0 as a whole.

### Length distribution and the shape of the tree

Window (191,950,768 edges): p50 **0.1310 m**, p90 0.2637 m, p99 0.4793 m,
p99.9 0.9633 m, max **22.9261 m**. **4,896 edges exceed the 2.5 m cap** —
0.0026 % of the tree, 15,575.922 m of the 28,831,927.9 m total: precisely the
links the kNN graph could never supply. Old area: 1,891 over the cap
(6,147.942 m of 6,843,702.973 m).

Split by phase — window: 191,940,152 edges from Kruskal inside the cap,
10,616 from the Borůvka completion. Old area: 42,704,015 / 3,320, and the
post-cap structure of phase (a) is 3,321 components, largest 42,285,538
(99.01 %). Kruskal is front-loaded as predicted: bucket 0 alone accepts
525,313 of its 1,058,206 candidates.

### Correctness against brute force

`tests/test_euclidean_mst.py` — five tests, exact **edge-set** agreement with
scipy's `minimum_spanning_tree` over the full distance matrix: synthetic
blobs (3 seeds), one point 400 m away (the doubling path), 1,500 points at
three different (block, halo) settings — 200/25, 37/2, 500/50 — identical in
all three, i.e. the sweep's block grid is not a parameter of the answer; and
real store points (758-, 1,736-, 5,000- and 8,000-point boxes).

One caveat, chased down rather than smoothed over: brute force must quantise
the distance matrix to **float32** first, because that is the weight the
algorithm has (`knn.parquet` stores `len` as f32). Against a float64 brute
force the 6 m box disagreed on exactly 1 edge of 1,735 — ours 805–1578 at
0.15565988567833519, brute's 913–1578 at 0.15565988558985475, a difference of
9e-11 m, which is a tie in float32 settled by (src, dst): this design's own
tie-break rule doing its job. Total weight identical to 6 decimals.

### Output

`data/stage2/mst.parquet` — 191,950,768 rows, **1,294,693,984 bytes**
(1.29 GB, not the ≈ 3.8 GB predicted; zstd on sorted-ish i64 pids).
`data/stage2/mst_old_area.parquet` — 42,707,335 rows, 305,681,612 bytes.

## Verification (what the implementation must prove, not assume)

1. **Edge count** = |V| − 1 exactly, and the result is acyclic and
   connected (union-find over the output ends with one component).
2. **Total weight** ≤ the total weight of any spanning structure we can
   build otherwise — in particular ≤ (MST of the kNN graph + the quotient
   completion), which is the cheap approximation this replaces.
3. **Cut property spot-check**: sample components mid-run and confirm by
   brute force on a local neighbourhood that the selected edge really is
   the minimum outgoing one.
4. **Threshold agreement**: cutting the MST at 2.5 m must reproduce
   stage 4's components — any disagreement is a real finding about the
   kNN truncation (a pair within 2.5 m that k = 10 never recorded), and
   must be reported with counts rather than smoothed over.
