# Every type answers a physical question

*How a return becomes a named object, and the four rules that keep it honest.*

A point starts as a laser return and ends as a triangle with a colour and a
kind. Three stages decide different things about it, and almost every bug this
library has had came from two of them deciding the same thing.

| stage | decides | unit |
|---|---|---|
| **5 labeling** | which of the ten labels a POINT carries | a point |
| **6 instances** | which points are one THING, and what type of thing | a set of points |
| **7 objects** | what 3-D geometry that thing earns, and what the sun does with it | triangles + a kind |

The ten labels are fixed: `ground veg_low veg_high building water small floating
bridge on_bridge glazing`. Everything finer — car, boat, dock, tree, lamp,
moorage, the twelve ground surfaces — is a stage-6 **type**, not a label. There
is no `car` class in the survey and never was.

---

## Rule 1 — one decision, one place

**Whoever decides a thing is the only one who decides it.**

This is not a style preference. Three separate bugs, each found by a user
looking at the scene, were the same defect:

| what shipped | why |
|---|---|
| a 129.9 m "boat" | stage 7 ended its floating branch with `return "boat"`, overriding stage 6 |
| a car with no bridge under it | the dashboard re-derived the type with its own copy of an old height rule |
| 3,585 trees on bridge decks, 5,275 inside buildings | stage 7's tree loop keyed on the stage-5 LABEL, so every stage-6 guard was invisible |

In each case the upstream fix was correct and had no effect, because a
downstream stage re-decided it. The cure was always deletion, never a better
threshold.

**Consequently:** stage 7 reads `instance_table.type`. It does not re-derive a
type from geometry. The one exception is documented in place — a building
footprint standing on a water footprint is a float home, which stage 6 cannot
see because it is a fact about two map layers meeting.

---

## Rule 2 — the map owns the plan, the survey owns the height

Overture says WHERE a thing is and what shape it is in plan. The returns say
how high it is. Neither may do the other's job.

Measured consequences of getting this backwards:

* **Docks.** Their width was taken from the spread of returns. A marina's
  returns have no edge — probe at 3, 6, 12 and 24 m and the apparent width is
  always 0.87 × the window, because moored boats float at pontoon height too.
  A dock's width is now *stated* (1.25 m half-width, a walkway) from the mapped
  centreline.
* **Bridge decks.** Buffering 59 centrelines by the measured spread gave
  98,045 m² of overlapping corridor that still missed 42% of what the bridge
  polygons cover.
* **Building heights.** Overture's `height` attribute is *not* used at all: six
  footprints where it claimed taller all carry an identical `height = 33.0`
  above even their maximum return, and a 4-storey university whose geometry
  reaches 22 m carries `6.97`. The survey measures the roof.
* **Ground under a bridge.** Stage 5's DEM *is* the deck there (median 29.69 m
  inside a bridge footprint). Everything that needs "the ground" reads
  `map_labels.ground_without_bridges`, which rebuilds it from the 97,744 ground
  returns the laser got *under* the deck — median error +0.00 m against them,
  where the raw grid is +27.34.

---

## Rule 3 — a type must be earned, not fallen into

A type is a claim about what something IS. Every one now has a named test that
asks a **physical** question, and a thing that answers wrong becomes `unknown`
rather than being filed under whatever is nearest.

| type | the question | where |
|---|---|---|
| `building` | is there a footprint, and does the laser see a roof under the sky? | stage 5 veto + stage 7 |
| `bridge` | is it broad and flat, and clear of the ground it crosses? | `objects.looks_like_deck` |
| `on_bridge` | is there a mapped bridge, and do I stand ON its surface? | stage 6 |
| `dock` | does a mapped pier way run under me? | stage 6.3d |
| `boat` | am I compact, floating, and not a tower? | `objects.looks_like_boat` |
| `car` | am I car-shaped, out of the canopy, and reachable by road? | `objects.looks_like_car` |
| `tree` | am I rooted in the ground, one crown, and not indoors? | `objects.looks_like_tree` |
| `lamp` | am I a pole? on a bridge, 4–15 m over the DECK | `looks_like_pole`, `bridge_lights` |
| `unknown` | *(none — this is the honest answer)* | — |

`small` was removed as a type. "A small thing on the ground" is not a statement
about what something is; matter we cannot name is `unknown`.

### The guards are context, not just shape

A bounding box cannot tell a car from a clump of canopy — both are
3.9 × 2.0 × 2.4 m. What separates them is **what is around and underneath**:

* 83% of the returns within 6 m of that "car" were `veg_high`
* the nearest road was 59.4 m away, on a building roof 3.8 m up
* real cars sit a median **0.0 m** from a mapped way, 90% within 9.1 m

So `looks_like_car` takes a canopy fraction and a colour, and stage 6 additionally
requires a drivable way within 15 m. Every type's guard follows that shape:
dimensions first, then the surface it stands on.

---

## Rule 4 — a threshold must be measured

Every number in these tests came from a measured population where the two
groups separate. Numbers chosen without that have each caused a regression:

| chosen | consequence |
|---|---|
| `BR_MIN = 2000` returns for a bridge | deleted four real bridges, one directly under an object labelled `on_bridge` |
| `PIER_MAX_L = 20 m` for a support | deleted every real Granville bent — they are 44–55 m long and 4.7 m wide |
| a 6 m reach from deck to support | kept 3 of 19 components; the deck occludes its own piers, so real ones stop 8–10 m short |

Whereas the measured ones separate cleanly:

* deck vs ground-under-bridge: **5.7–31.4 m** above ground against **0.0 m**
* deck vs mast: plan/vertical **5.8–22.3** against **1.2, 0.9, 0.5**
* pier vs debris: rise/clearance **0.41–0.80** against **0.00–0.35**
* boat vs raft: box fill **≥22%** against **7.2%**

When a test rejects far more than the measurement predicted, the test is
wrong — that is how the over-split car instances were found.

---

## Instance detection: the right cut per type

Connectivity alone welds; one operator does not fit every type. Stage 6 applies
a different cut to each, in this order:

| step | type | the cut | why connectivity fails |
|---|---|---|---|
| 6.1 | all | components of the class-induced kNN subgraph | — |
| 6.3 | floating | `raft_split` — h-maxima watershed on height over the pontoon | boats touch within 1 m of the deck |
| 6.3b | ground | Overture land-use polygons and buffered road centrelines | a tennis court is not a connected component |
| 6.3c | building | Overture footprints, grown 3 m for eaves | a terrace shares walls |
| 6.3d | floating | mapped pier ways, then free components | the map cuts what height cannot |
| 6.3e | veg_high | **support-forest root** | one tree, one trunk, one root |
| 6.3f | small / on_bridge | `car_split` — h-maxima watershed on height over the surface | parked cars touch |

**6.3e and 6.3f are deliberately different operators**, and that difference is
the lesson. The support forest is perfect for a tree — every crown return
reaches the ground through its own trunk. Applied to cars it *shredded* them,
because a roof reaches the ground by several paths around the body: measured
over the parking polygons, 1,974 instances of median 1.9 × 1.0 × 0.8 m, a
quarter of a car each, of which 44 were car-shaped.

A car is a **roof standing clear of what it is parked on**, and the gap to the
next car is a strip of that surface — a height watershed. At 0.5 m cells a
0.6 m gap is one cell and a σ=1.0 blur closes it (four test cars → two
instances); at 0.3 m with σ=0.5 the gap survives and four cars give four.

---

## Kinds: what the sun does with a thing

Geometry is not enough; the shadow engine needs to know how light meets it.
Every model states its own kind, because that is a property of what the thing
IS.

| kind | meaning | examples |
|---|---|---|
| `solid` | light stops | buildings, hulls, trunks, bents, cars |
| `slab` | a (top, bottom) pair with air beneath — light passes UNDER | bridge decks, tree crowns, low vegetation |
| `glass` | a zero-thickness pane that transmits | windows, the west bridge rail |
| `surface` | receives light, casts nothing | water, terrain, land use |

The bridge is the reason this exists: modelled `solid` it blacks out the
waterway all day; as a `slab` it blocks only above 72.6°, an angle the
Vancouver sun never reaches.

A type may ship **several kinds** when its parts differ physically — a tree is
an opaque `solid` trunk under a porous `slab` crown.

---

## What is deliberately NOT done

* **No post-processing pass that bolts detail onto finished objects.**
  Clustering the returns above a building's P95 and extruding each as its own
  solid was tried and rejected: it put 45 blocks on 83 buildings and read as
  debris. A plant room modelled apart from its roof is not the same thing as a
  roof that has a plant room on it. If roof detail matters, the roof must carry
  it — one measured surface over the whole footprint.
* **No invented objects where the survey saw nothing.** Five mapped bridges
  carry no returns; they get no geometry and the run says so. The one
  exception is bridge supports, which are stated by rule (one per 60 m) because
  an airborne survey cannot see under its own deck — and the docstring declares
  that it is an invention.
* **No silent caps.** Where a rule drops things, the count is printed. The
  `on_bridge` invariant — nothing stands on a bridge that was not built —
  reports every run.
