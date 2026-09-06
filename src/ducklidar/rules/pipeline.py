"""The labelling pipeline, end to end: a class and an object id for every point.

Kaveh's ordering, and the DEM is built twice on purpose:

    ground label -> DEM v1 -> tree + building labels -> object ids -> DEM v2

Because a void under a roof and a void under a canopy are different problems, and you cannot
tell which one you are filling until the labels exist. Under a building the ground is a level
pad, deliberately discontinuous with the terrain around it; under a canopy the terrain simply
continues and the laser was blocked. v1 is filled generically and its only job is a height good
enough to label with. v2 is filled **by cause**, and every height that matters comes off v2.

Two passes, not a loop: the second pass only moves the DEM under buildings, which is exactly
where no label depends on it — a roof is a roof whether the floor beneath it is 20.0 or 20.3 m.

See ``docs/labelling-pipeline.md`` for the reasoning and ``docs/dem.md`` for the fill and datum
measurements. Every threshold here was swept somewhere, not guessed.

    python tools/pipeline.py [--pix 1.0] [--no-trees]
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi_mod

import ducklidar as dl
from ._env import OUTDIR, check, STUDY_BOX

B = dl.box(*STUDY_BOX)
PIX = 1.0
FEAT = Path("/tmp/granville_feat_16.las")     # written by tools/pdal_native.py
RSCRIPT = Path("/home/kaveh/miniconda3/envs/rlidr/bin/Rscript")

# Nine FINAL classes -- a full partition, every point in exactly one. UNLABELLED and GLAZING
# are transient: the residue pass empties the first, the footprint adjudication the second, and
# both are asserted empty before anything is written.
(UNLABELLED, GROUND, VEG_LOW, VEG_HIGH, BUILDING, WATER, SMALL,
 FLOATING, GLAZING, BRIDGE) = range(10)
SMALL_BRIDGE = 10
NAMES = {GROUND: "ground", VEG_LOW: "vegetation low", VEG_HIGH: "vegetation high",
         BUILDING: "building", WATER: "water", SMALL: "small object",
         FLOATING: "boat or dock", BRIDGE: "bridge", SMALL_BRIDGE: "small object on bridge"}

# Two levels, Kaveh's design. Level 1 is five FAMILIES -- ground, vegetation, building, water,
# bridge -- and is a total partition on its own; every level-2 class nests inside exactly one
# family.
#   ground      <- ground, small object     a car or a kerb stands ON the ground system
#   vegetation  <- vegetation low + high    short vegetation is still vegetation (ASPRS 3 agrees)
#   building    <- building                 a built structure standing on the ground
#   water       <- water, boat or dock      a floating thing belongs to the water system
#   bridge      <- bridge, small object     Kaveh's call (2026-07-30), and it agrees with ASPRS
#                  on bridge                giving bridges a top-level class: the deck is its own
#                                           support system, and what stands on it belongs to it
FAM_GROUND, FAM_VEG, FAM_BUILDING, FAM_WATER, FAM_BRIDGE = 1, 2, 3, 4, 5
FAMILY = {GROUND: FAM_GROUND, SMALL: FAM_GROUND,
          VEG_LOW: FAM_VEG, VEG_HIGH: FAM_VEG,
          BUILDING: FAM_BUILDING,
          WATER: FAM_WATER, FLOATING: FAM_WATER,
          BRIDGE: FAM_BRIDGE, SMALL_BRIDGE: FAM_BRIDGE}
FAM_NAMES = {FAM_GROUND: "ground", FAM_VEG: "vegetation",
             FAM_BUILDING: "building", FAM_WATER: "water", FAM_BRIDGE: "bridge"}
# source byte: which stage decided, so a disagreement stays visible instead of being resolved
SRC = {"none": 0, "csf": 1, "echo": 2, "covariance": 3, "footprint": 4, "sparse": 5,
       "low": 6, "flat_splitter": 7, "on_water": 8, "surface": 9, "shape": 10, "knn": 11,
       "deck": 12, "deck_band": 13, "vehicle": 14, "notgreen": 15, "suspended": 16,
       "unrooted": 17}

# CSF rigidness: 1 steep, 2 relief, 3 flat. The library default is 3 and it was swept on a
# 120 m FLAT window, where the choice moved F1 by under a point and the docs said outright
# that on real relief "this is the parameter that decides whether the cloth bridges a
# valley". This 500 m box has the park slope in it. [measured] on the full box, rigidness
# 1 gives IoU 0.932 against 3's 0.926, recall 0.95 against 0.94. Set here rather than in
# DEFAULTS, because the right value follows the terrain and the flat window still prefers 3.
CSF_RIGIDNESS = 1
# A point the cloth rejects that is still AT the ground surface is ground. Without this the
# tree had no fallback: CSF says no, so the point drops to the `hag <= 2` branch and a
# return five centimetres above the ground gets filed as street furniture. [measured] that
# was 107,092 points -- 84% of all missed ground -- sitting at a median hag of 0.05 m.
#     fallback   none    0.15    0.25    0.40    0.60
#     IoU       0.932   0.957   0.959   0.961   0.961
#     precision  0.98    0.98    0.98    0.98    0.97
# 0.4 m is where recall stops paying for itself, and it is close to CSF's own
# class_threshold of 0.5 -- the same question asked of the settled cloth.
GROUND_FALLBACK = 0.4
# ...and the veto in the other direction, Kaveh's: after the cloth, take the cars back out.
# The CSF branch had NO height test, so whatever the cloth called ground was ground even a
# metre up. [measured] the effect here is small -- only 11,650 of 3.5 M ground points stand
# above 1 m, because a 1 m cloth at rigidness 1 is stiff enough not to drape over a car, and
# because dl.dem takes the MINIMUM return per cell so a car never lifts the surface it would
# be measured against. IoU moves 0.932 -> 0.931, inside the noise.
# Kept anyway, for the same reason as the boats: 54% of what it drops is unclassified, a
# point a metre above the ground surface is not ground, and those points belong in `low
# object` where a car belongs. The metric cannot see cars, so it does not get a vote.
GROUND_CEILING = 1.0
# The 0.4-2 m band splits on penetrability alone: through it -> vegetation low; solid ->
# SMALL OBJECT, one class holding cars, kerbs, bollards and street furniture together. The
# car-vs-kerb footprint split (4-25 m2, elongation < 3) was tried and dropped: its thresholds
# had nothing to be swept against, and on the map it fired on roof edges and yard clutter as
# much as on the car parks. A class that cannot be checked should at least not pretend to a
# finer distinction than the evidence supports; the shape pass can come back per OBJECT later.
GREEN_MEDIAN = 0.02  # a band cluster is vegetation only if its median excess green says so
ECHO_MIN, TALL = 0.45, 2.0            # vegetation low/high boundary: swept 1.5-3.0, 2.0 wins on every column
# ...confirmed per POINT before it is believed. echo_ratio is computed on a 1 m grid, so a
# single overhanging branch labelled the whole square metre, roof included -- 10.1% of the
# survey's building points leaked into vegetation. Requiring the point itself to show the
# same physics (the beam continued past it, or its neighbourhood is scattered rather than
# planar) takes IoU 0.820 -> 0.837, precision 0.89 -> 0.93, and the roof leak to 6.5%.
# Per-point rules ALONE are all worse -- best 0.745 -- so the cell keeps its say.
SCATTER_MIN = 0.2
PLANAR_MIN, SCATTER_MAX = 0.7, 0.3    # building:   IoU 0.609 [tools/pdal_native.py]
# water: swept, IoU 0.820 at precision 0.91. Water reflects the beam AWAY from the sensor,
# so a water cell gets about 1 return where ground gets 49 -- p95 of 9 against ground's p05
# of 24. Intensity is the obvious feature and adds NOTHING once sparsity is in: the sweep
# peaks with the intensity bound removed entirely.
WATER_DENS, WATER_HAG = 8, 0.5
# ...and water is a REGION, not a cell. A lake is thousands of contiguous sparse cells; a
# dark sparse patch of road is a handful. [measured] the ceiling for ANY per-cell rule here
# is IoU 0.964 -- a majority vote that already knows the answer -- and the per-cell version
# reached only 0.820, so the gap was never resolution. 99% of its false positives were
# ground in cells with a median of 6 returns, and 58% of its misses were within 2 m of the
# shore. Closing the gaps then demanding size fixes both: 0.945 at precision 0.99.
# ORDER MATTERS: demanding size WITHOUT closing first collapses recall to 0.42, because the
# raw sparse cells are fragmented by every dock and moored boat.
WATER_CLOSE, WATER_MIN_CELLS = 2, 400   # 400 m2: a smaller pond will not be found
# ...OR a LOCAL rule, which catches what the region misses: judge a cell by its
# NEIGHBOURHOOD rather than by itself. A water cell with 20 returns sits among cells with 1
# -- its neighbourhood is water even where it is not sparse itself, which is most of the
# shoreline. Alone it scores 0.926; unioned with the region, 0.956, because two rules at
# precision 0.99 miss different things.
WATER_WIN, WATER_LOCAL = 5, 6
# ...and then clipped to the water PLANE, which is Kaveh's boat warning made into a rule.
# A boat is not the water it floats on, and the survey never named one, so no IoU here can
# see the mistake. hag cannot catch it either: CSF drapes its cloth over docks and hulls,
# so the DEM is RAISED in a marina and a deck 2 m up still reads hag < 0.5. Against the
# fitted water surface it cannot hide. [measured] this removed 896 above-plane points the
# region rule was already swallowing, and 1,001 from the wider rule, for 0.003 of IoU.
WATER_PLANE = 0.5
# ...and a neighbour vote, Kaveh's. The local rule has no connectivity test at all, so an
# isolated sparse patch of road passes it on its own merits. Requiring most of a cell's
# neighbourhood to agree costs nothing and nearly halves the false positives -- 256 -> 136,
# IoU 0.953 -> 0.957. Note this beats requiring proximity to the water body (0.950): a
# shoreline cell 30 m along a narrow inlet is genuinely water and distance would reject it,
# while an isolated road cell has no water neighbours at any distance.
# The vote is a FRACTION of the window, not a count of neighbours, and that distinction is
# the whole rule. [measured] `>= 2 water neighbours in 3x3` removes 6 false positives out of
# 256, because 81% of them are single cells sitting ON THE SHORELINE, touching real water --
# their neighbours genuinely ARE water, so no count can reject them. A fraction can: a cell
# inside the water has water all round, a cell on the edge has land on one side and fails
# half of a wide window. [swept] 50% is optimal at every window size, 3x3 through 9x9, so
# only the window is fitted here; 9x9 takes precision to 1.00 with 79 false against 136.
WATER_VOTE, WATER_VOTE_WIN = 0.5, 9
# ...and finally the point's OWN cell must look like water, which is Kaveh's dock picture
# made into a rule. A floating dock floats: it sits at the water plane, inside the water
# region, surrounded by water, so every test above says water. What it is not is sparse or
# dark. [measured] 39% of everything we called water -- 12,078 points -- was UNCLASSIFIED by
# the survey, with cell density median 30 and intensity median 262, against true water's 2
# and 7. This filter drops 9,968 of them and loses 362 real water points: 27 to 1.
#
# It costs judged IoU, and that is the point. The survey does not classify docks, so they
# are invisible to every score in this file -- the same blind spot as the boats. A metric
# that cannot see an error must not be allowed to veto fixing it.
WATER_OWN_DENS, WATER_OWN_INTEN = 20, 150
FLOAT_DEM_TOL = 0.5   # m: how close the DEM must sit to the water plane to count as over it
# A boat FLOATS, so it moves; a quay does not. This tile was flown over 45 hours with four
# overlapping lines, so most cells are seen more than once, at different times and tides.
# [measured] height disagreement between flight lines, per cell:
#     ground   median 0.04 m      building median 0.06 m
#     water    median 0.20 m      vessels  median 0.19 m
# Things on water move four times as far between passes as things on land. It is the only
# signal here specific to FLOATING rather than to position, and it cuts the ground
# confusion from 7.4% to 2.3% for 16 points of coverage.
FLOAT_MOVE = 0.12     # m of between-line disagreement before something counts as afloat
# A boat has open sky above it. A bridge PIER does not -- it carries a deck 27 m up, and the
# pier itself rises out of the water, is solid, and sits over a DEM at water level because
# nothing under a bridge returns ground either. Every other test here passes it. What gives
# it away is what is OVERHEAD: [measured] candidates have a median of 0.64 m of structure
# above them and p95 of 8.9 m, but p99 of 26.8 m -- the tail is the bridge. 10 m is generous
# for a mast and far below a deck.
FLOAT_CEILING = 10.0  # m of structure allowed above a floating point
# A large vehicle is a building to every per-point test: a truck roof is planar, solid and
# 2-5 m up, exactly like a shed. Width and shape were measured dead (kiosks are as narrow as
# trucks). What separates them is physical: a building's walls reach the ground, so no beam
# can return from inside its footprint, while a vehicle stands on wheels and the oblique
# beams find road under the body. [measured] undercut -- the share of interior cells whose
# lowest return is at ground level -- is ~1.00 for vehicles that provably moved between
# flight lines, 0.07 for large buildings. Movement is the second, independent guard: nothing
# static moves 0.6 m between lines 45 h apart. Collateral of the undercut test is roofs over
# open space (bus shelters, carports) -- mostly street furniture, which is where the small
# object class puts them anyway.
# The area cap is 400, not truck-size, because parked vehicles MERGE: the component closing
# joins a row of trucks in a car park into one blob ([measured] up to 188 m2 on this tile,
# still with undercut 0.91), and an 80 m2 cap left every such row in building. 400 is safe
# because the undercut guard does the real work -- a real building of any size keeps its
# walls to the ground. Stats are taken over the component's BUILT cells only: the closing's
# gap cells between cars hold bare ground, and their zero heights dragged a row's median top
# below the gate.
VEH_AREA = (6, 400)     # m2 footprint: below is band-size, above even rows do not reach
VEH_TOP = (1.8, 5.0)    # m above ground at the top: vans to trucks
VEH_MOVE = 0.6          # m of between-line disagreement: certain movement
VEH_UNDER = 0.5         # share of interior cells with a ground return beneath


def classify(f):
    """One decision tree. Every point takes exactly one path, and every split cites its number.

    Written as a tree rather than as a list of rules with precedence because precedence was
    emergent and invisible: each rule tested ``label == UNLABELLED`` and the ORDER of the
    statements decided the answer. Here the order is the structure, and it can be read.

      in a sparse region OR sparse neighbourhood, 41/81 neighbours agree,
      at the water plane, own cell sparse and dark ?
        yes -> WATER                                     IoU 0.940  prec 1.00  rec 0.94
         no -> the cloth says ground, AND hag < 1 m ?    the ceiling keeps cars out
                 yes -> GROUND                           IoU 0.964  prec 0.98  rec 0.98
                  no -> hag < 0.4 m ?                    rejected by the cloth, still at
                          yes -> GROUND                  the surface: 84% of missed ground
                           no -> hag <= 2 m ?
                                  yes -> the beam went through it ?
                                           yes -> VEGETATION LOW    grass, hedges (ASPRS 3)
                                            no -> SMALL OBJECT      cars, kerbs, furniture
                                   no -> penetrable ?    echo >= 0.45 AND
                                                         (not-last OR scatter >= 0.2)
                                          yes -> planar >= 0.7 ?
                                                   yes -> glazing, parked for adjudication
                                                    no -> VEGETATION HIGH  IoU 0.838
                                           no -> planar >= 0.7 AND scatter < 0.3 ?
                                                   yes -> BUILDING         IoU 0.877
                                                    no -> residue, resolved below

    Then the passes the tree cannot make, each needing something it does not have -- and after
    them the partition is TOTAL: every point carries exactly one of the nine labels, asserted.

      * FLOATING -- needs the water plane and the flight lines. Over a DEM at water level,
        above the plane, solid, under 10 m of structure overhead, and MOVING more than 0.12 m
        between flight lines: a boat floats, so it moves (ground disagrees between lines by
        0.04 m, vessels by 0.19 m). Takes from ground, small object and the residue only.
      * ADOPT THE FOOTPRINT -- needs the building components. Any residue point standing inside
        one becomes BUILDING: planarity fires on flat roof interiors and nothing else, so
        walls and parapets came out unlabelled until this pass. 0.310 -> 0.658 when added.
      * ADJUDICATE GLAZING -- same footprints. A flat pulse-splitter ON a building is glass and
        folds into BUILDING (the source byte still says which); off one it was a flat-topped
        tree and returns to VEGETATION HIGH. Flat + splitting alone puts only 17% of its
        patches on a building.
      * RESIDUE -- what remains is the boundary fuzz between classes, mostly single cells where
        a crown meets a roof. It joins its nearest labelled neighbour in 3-D, marked "knn" in
        the source byte so it stays auditable.

    ``penetrable`` is the cell's echo ratio AND the point's own agreement — the beam continued
    past it, or its neighbourhood is scattered rather than planar. The cell alone leaked 10.1% of
    roof points into vegetation; the point alone scores worse than the cell (0.745 against 0.820);
    together, 0.837 and a 6.5% leak.

    Water is tested first because a cloth settles onto a flat lake exactly as onto a car park, so
    99.8% of water would otherwise be called ground. The more specific rule gets first refusal.

    **Ground before vegetation, and it was tested rather than assumed.** Grass sits at ground
    level and is penetrable, so 44% of ground's false positives are ASPRS class 3 and it looks as
    though vegetation should go first. [measured] it should not:

    ==================================  ==========  ===========  ==============
    ordering                            ground IoU  low veg IoU  low veg prec
    ==================================  ==========  ===========  ==============
    ground first                        **0.959**   **0.366**    **0.87**
    penetrable first, hag <= 2          0.951       0.357        0.55
    penetrable first, hag <= 0.5        0.950       0.340        0.54
    ==================================  ==========  ===========  ==============

    It loses on both. The cause is a resolution mismatch: CSF decides per POINT, the echo ratio
    is per 1 m CELL, so a cell with any overhanging foliage marks the road beneath it penetrable
    too. Ground first acts as a guard, the precise test claiming real ground before the coarse
    one can take it -- the same principle that put water first.
    """
    lab = np.full(len(f["hag"]), UNLABELLED, np.uint8)
    src = np.zeros(len(f["hag"]), np.uint8)

    # Inside a water body, three things exist and only one of them is water. All three tests
    # were built to defend the water label; naming their outcomes costs nothing and stops a
    # pontoon being filed as ground.
    #
    #   at the plane, own cell sparse and dark  ->  WATER
    #   anything else standing in the water     ->  FLOATING
    #
    # ONE class, not two, and Kaveh is right that it has to be. A dock and a boat both FLOAT, so
    # both sit at the water plane at their lowest -- a boat's hull is at the waterline and its
    # cabin above it, which means a height threshold cuts THROUGH a boat rather than between a
    # boat and a pontoon. The evidence was ambiguous either way (83% of floating clusters came
    # out pure, but the largest was 147 dock cells to 30 vessel, and a walkway with boats moored
    # to it SHOULD look mixed), and with no ASPRS class for either there is nothing to arbitrate.
    #
    # So the claim is reduced to what the measurements actually support: these points are inside
    # a water body, are not water, and are not ground. Splitting them needs per-OBJECT clustering
    # and a height profile, which is stage 6 work, not another threshold here.
    inw = f["in_water"]
    water = inw & f["at_plane"] & f["water_like"]
    floating = np.zeros(len(water), bool)     # decided after the tree; see float_pass()
    claimed = water
    ground = ~claimed & f["csf"] & (f["hag"] < GROUND_CEILING)
    rest = ~claimed & ~ground
    at_surface = rest & (f["hag"] < GROUND_FALLBACK)     # rejected by the cloth, still ground
    low = rest & ~at_surface & (f["hag"] <= TALL)
    # ...the CELL must agree, not just the point. Car glass splits a pulse exactly as foliage
    # does, so windscreens were landing in vegetation low; a grass cell has high echo and a car
    # cell does not. Kaveh spotted the vehicles on the level-1 map.
    # The band's penetrability test is PHYSICALLY INVALID at car height, which is why every
    # geometric gate failed and why the cars sat green on Kaveh's map. Discrete-return LiDAR
    # cannot separate echoes closer than about a metre, so 0.5 m grass returns a SINGLE echo and
    # reads "solid", while a car's edges split beams to the road and read "penetrable" -- the
    # test is near-inverted below 2 m. Six geometric variants were swept into that wall (cell
    # echo, neighbourhood echo, hollow-skin: 0.268-0.484 against the 0.496 baseline).
    #
    # The instrument that works was never geometric: THIS TILE IS COLOURISED, and vegetation is
    # green where cars and asphalt are not (band excess green: living vegetation p50 0.04, the
    # unclassified band where cars live p90 0.037). Per point the colour is shadow-noisy, so the
    # decision is per OBJECT: connected band cells vote by their MEDIAN excess green -- a row of
    # cars has median ~0, a hedgerow does not, and a median shrugs off shadow pixels.
    #
    # [measured] cluster median > 0.02: unclassified band points in vegetation 326k -> 26k
    # (-92%), precision 0.75 -> 0.80, veg-low IoU 0.498 -> 0.382. The IoU pays because September
    # shrubs are brown and shaded; the same map-beats-blind-metric call as the boats and the
    # docks -- a reference with no car class does not get to keep the cars green.
    pen = f["not_last"] | (f["scatter"] >= SCATTER_MIN)
    cell, ncell = f["cell"], f["ncell"]
    lcell = np.zeros(ncell, bool)
    lcell[np.unique(cell[low])] = True
    clab, _ = ndi_mod.label(lcell.reshape(f["shape"]), structure=np.ones((3, 3)))
    cid = clab.ravel()[cell]
    bidx = np.flatnonzero(low)
    bidx = bidx[np.argsort(cid[bidx])]
    med = np.zeros(clab.max() + 1)
    for g in np.split(bidx, np.flatnonzero(np.diff(cid[bidx]) != 0) + 1):
        if len(g):
            med[cid[g[0]]] = np.median(f["exg"][g])
    if f["has_rgb"]:
        low_veg = low & (med[cid] > GREEN_MEDIAN)
    else:
        # A colourless cloud (NRCan 2016 ships no RGB) falls back to the echo test, with its
        # measured failure stated instead of hidden: vehicles will read as low vegetation,
        # because at this density they are geometrically indistinguishable from shrubs --
        # cluster medians scatter 0.281 vs 0.332, roughness 0.122 vs 0.128 m. Only colour
        # separates them (0.003 vs 0.069, twenty-fold), and this cloud has none.
        print("    WARNING: no RGB in this cloud -- band vegetation falls back to the echo "
              "test; expect vehicles to read as low vegetation")
        low_veg = low & pen
    small = low & ~low_veg
    # ...and OUT of tall as well. Leaving them in let the building branch reclaim a point that
    # had already been called ground, because the later assignment wins: building went 0.878 ->
    # 0.863 on one missing term.
    tall = rest & ~low & ~at_surface
    penetrable = (f["echo"] >= ECHO_MIN) & (f["not_last"] | (f["scatter"] >= SCATTER_MIN))
    # Foliage and glass both split a pulse -- that is why the echo ratio confuses them -- but
    # only one of them is FLAT. [measured, docs/instances.md] flat + splitting alone puts just
    # 17% of its patches on a building, the rest being crowns that happen to be locally flat; the
    # third term takes it to 100%. That term is the building footprint, which does not exist yet
    # here, so a flat splitter is parked in its own class and the footprint pass adjudicates:
    # inside a building it is GLAZING, outside it was a flat-topped tree all along.
    flat_splitter = tall & penetrable & (f["planar"] >= PLANAR_MIN)
    veg = tall & penetrable & ~flat_splitter
    bld = tall & ~penetrable & (f["planar"] >= PLANAR_MIN) & (f["scatter"] < SCATTER_MAX)

    for mask, cl, why in ((water, WATER, "sparse"),
                          (ground, GROUND, "csf"), (at_surface, GROUND, "surface"),
                          (low_veg, VEG_LOW, "echo"), (small, SMALL, "low"),
                          (veg, VEG_HIGH, "echo"),
                          (flat_splitter, GLAZING, "flat_splitter"),
                          (bld, BUILDING, "covariance")):
        lab[mask] = cl
        src[mask] = SRC[why]
    return lab, src


def line_spread(pts, flat, ncell):
    """Per cell, how far the flight lines disagree about the height of what is in it.

    Median z per (cell, line), then max minus min across lines. The median rather than the mean
    because one stray return through a mast would otherwise read as movement. NaN where only one
    line saw the cell, and a NaN never passes a `>` test, so single-look cells are excluded from
    the floating class rather than guessed at.
    """
    z = np.asarray(pts["z"])
    key = flat.astype(np.int64) * 1000 + np.asarray(pts["point_source_id"]).astype(np.int64)
    order = np.lexsort((z, key))
    ks, ze = key[order], z[order]
    idx = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    med = np.array([ze[a:b][(b - a) // 2] for a, b in zip(idx[:-1], idx[1:])])
    cell = (ks[idx[:-1]] // 1000).astype(np.int64)
    n = np.bincount(cell, minlength=ncell)
    lo = np.full(ncell, np.inf); hi = np.full(ncell, -np.inf)
    np.minimum.at(lo, cell, med); np.maximum.at(hi, cell, med)
    return np.where(n >= 2, hi - lo, np.nan)


def float_pass(label, f, dem_over_water):
    """Anything standing on the water that the tree has not already explained.

    The definition that works is NOT "a non-water thing inside sparse water" -- that was the
    first attempt and it found 2% of the vessels, because a marina packed with boats is not
    sparse and never entered the water candidate set at all. Nor is it the water body's
    morphological envelope: the marina block is attached to the shore, so closing cannot reach
    it (5%).

    What does work is that **the DEM knows where the water is**. There are no ground returns
    under a marina, so the surface interpolates to water level, while land sits metres higher.
    [measured] DEM median: -1.08 m inside the water mask, -0.72 m over the vessels, +3.68 m
    everywhere else. Standing above that surface, solid rather than penetrable, over a DEM at
    water level -- 71% of the vessel-like cells, against 2% for the first attempt.

    Applied AFTER the tree and only to points it left as ground, low or unlabelled. A quayside
    shed on a low wharf satisfies every test here, and letting BUILDING claim it first removes
    that leak by construction rather than by another threshold.
    """
    # Widened after Kaveh found boats hiding in building (a planar cabin) and in vegetation
    # (rigging splits pulses like foliage). Safe because the guards discriminate where the label
    # could not: a quayside shed sits over a water-level DEM too, but it does not MOVE between
    # flight lines (0.06 m against a vessel's 0.19), and land vegetation never stands over a DEM
    # at the water plane.
    take = np.isin(label, (GROUND, SMALL, UNLABELLED, BUILDING, VEG_LOW, VEG_HIGH))
    return (take & dem_over_water & f["above_plane"] & ~f["not_last"] & (f["hag"] < 10)
            & (f["moved"] > FLOAT_MOVE) & (f["overhead"] < FLOAT_CEILING))


def stage(msg, t0=None):
    print(f"{'':>4}{msg}" if t0 is None else f"{'':>4}{msg} ({time.time()-t0:.0f}s)")


def covariance(n):
    """Planarity and scattering per point, aligned to `dl.read`'s order.

    Two sources, same numbers:

    * ``LIDAR_STAGE2`` set → the stage-2 feature tables, joined by pid.
      PDAL's definitions are eigenvalue ratios (planarity (λ2−λ3)/λ1,
      scattering λ3/λ1), so the stored normalised eigenvalues give the
      identical values — the normalisation cancels.
    * else → the PDAL FEAT las, exactly as before. [checked] PDAL preserves
      point order through this pipeline — the coordinates agree with
      `dl.read` to the LAS scale of 5 mm — so the features can be indexed
      directly. If that ever stops being true the labels will silently
      belong to the wrong returns, so it is asserted.
    """
    import os

    s2 = os.environ.get("LIDAR_STAGE2")
    if s2:
        import duckdb
        import pyarrow as pa

        from ._env import window_pids

        pid = window_pids(B)
        if len(pid) != n:
            raise SystemExit(f"stage2 window carries {len(pid):,} pids, the cloud has {n:,}")
        feat = os.path.join(s2, "features.parquet")          # one table per stage —
        if not os.path.exists(feat):                          # parts only mid-sweep
            feat = os.path.join(s2, "features/*.parquet")
        import tempfile

        con = duckdb.connect()
        # bounded, and spilling rather than dying: the unbounded join hash-built
        # over ALL 191.9 M feature rows and died with MemoryError inside a 19 GB
        # cap, at a step whose own working set is ~1 GB
        con.execute("set memory_limit='4GB'")
        con.execute(f"set temp_directory='{tempfile.gettempdir()}/ducklidar_cov'")
        con.register("w", pa.table({"pid": pid, "i": np.arange(n, dtype=np.int64)}))
        # the window is a contiguous pid range (one tile's sidecar), so this
        # predicate prunes the scan to its row groups — the join then runs over
        # the window's own slice instead of the whole store
        lo, hi = int(pid.min()), int(pid.max())
        i, e1, e2, e3 = con.execute(
            f"select w.i, f.e1, f.e2, f.e3 from w "
            f"join read_parquet('{feat}') f using (pid) "
            f"where f.pid between {lo} and {hi}").fetchnumpy().values()
        if len(i) != n:
            raise SystemExit(f"stage2 features matched {len(i):,} of {n:,} pids — "
                             f"window and stage2 registry disagree")
        planar = np.empty(n, np.float32)
        scatter = np.empty(n, np.float32)
        # PDAL default mode: features are ratios of SQRT eigenvalues
        r1, r2, r3 = np.sqrt(e1), np.sqrt(e2), np.sqrt(e3)
        planar[i] = (r2 - r3) / r1
        scatter[i] = r3 / r1
        return planar, scatter, None

    import laspy

    if not FEAT.exists():
        raise SystemExit(f"{FEAT} missing — run tools/pdal_native.py first")
    las = laspy.read(str(FEAT))
    if len(las.x) != n:
        raise SystemExit(f"{FEAT} has {len(las.x):,} points, the cloud has {n:,}")
    return np.asarray(las.Planarity), np.asarray(las.Scattering), np.asarray(las.x)


def lidr_crowns(pts, veg, dest):
    """Per-crown ids for the vegetation points, via lidR. Returns an id per vegetation point."""
    import laspy

    from .lidr_trees import R_SCRIPT, CHM_RES, WINDOW, MIN_HEIGHT

    src = Path(tempfile.gettempdir()) / "pipeline_veg.laz"
    hdr = laspy.LasHeader(point_format=6, version="1.4")
    hdr.scales = [0.001] * 3
    hdr.offsets = [np.floor(np.asarray(pts[k])[veg].min()) for k in "xyz"]
    las = laspy.LasData(hdr)
    for k in "xyz":
        setattr(las, k, np.asarray(pts[k])[veg])
    las.classification = np.full(int(veg.sum()), 5, np.uint8)   # lidR wants a ground class...
    gz = np.asarray(pts["z"])[veg]
    las.classification[gz <= np.percentile(gz, 1)] = 2          # ...so give it the lowest 1%
    las.write(str(src))

    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as f:
        f.write(R_SCRIPT % {"res": CHM_RES, "ws": WINDOW, "hmin": MIN_HEIGHT})
        script = f.name
    r = subprocess.run([str(RSCRIPT), script, str(src), str(dest)],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"Rscript failed:\n{r.stderr[-2000:]}")
    out = laspy.read(str(dest))
    return np.asarray(out.treeID).astype(np.int32)


def main(argv):
    pix = float(argv[argv.index("--pix") + 1]) if "--pix" in argv else PIX
    pts = dl.read(check(), B)
    n = len(pts["x"])
    z = np.asarray(pts["z"])
    flat, (ny, nx) = dl.cell_index(pts, B, pix)
    print(f"{n:,} returns, {nx}x{ny} grid at {pix:g} m\n")

    # ---- 1. ground label, from x, y, z alone ------------------------------------------------
    print("1. ground label")
    t = time.time()
    gnd = dl.ground_filter(pts, cloth_resolution=1.0, rigidness=CSF_RIGIDNESS)
    stage(f"{int(gnd.sum()):,} ground returns ({gnd.mean()*100:.1f}%)", t)

    # ---- 2. DEM v1: generic fill, only has to be good enough to label with -------------------
    print("\n2. DEM v1 — generic fill")
    t = time.time()
    v1 = dl.dem(pts, B, pix, ground=gnd)
    hag1 = z - v1["filled"].ravel()[flat]
    stage(f"void {v1['stats']['void']*100:.1f}%, "
          f"{v1['stats']['far_from_evidence']*100:.1f}% further than 5 m from evidence", t)

    # ---- 3. tree and building labels ---------------------------------------------------------
    print("\n3. labels")
    t = time.time()
    feats = {
        "hag": hag1,
        "echo": np.nan_to_num(dl.echo_ratio(pts, B, pix), nan=0.0).ravel()[flat],
        "density": np.bincount(flat, minlength=ny * nx)[flat],
        "not_last": np.asarray(pts["return_number"]) < np.asarray(pts["number_of_returns"]),
        "csf": gnd,
    }
    feats["cell"], feats["ncell"] = flat, ny * nx
    feats["shape"] = (ny, nx)
    _r = np.asarray(pts["red"]).astype(float)
    _g = np.asarray(pts["green"]).astype(float)
    _b = np.asarray(pts["blue"]).astype(float)
    feats["exg"] = (2 * _g - _r - _b) / np.maximum(_r + _g + _b, 1)
    feats["has_rgb"] = bool((_r + _g + _b).max() > 0)
    # water's region test needs the grid, so it is computed here and handed to the tree as a
    # per-point feature -- classify() stays a pure per-point function
    # sparse AND low, before the region test. Sparsity alone lets high sparse cells -- masts,
    # crane booms, the odd thin roof edge -- into the closing, where they bridge regions that
    # should not connect: precision 0.99 -> 0.89 without this line.
    low_cell = np.bincount(flat, weights=(hag1 < WATER_HAG).astype(float),
                           minlength=ny * nx) > 0
    dens_grid = np.bincount(flat, minlength=ny * nx)
    sparse = ((dens_grid <= WATER_DENS) & low_cell).reshape(ny, nx)
    sparse = ndi_mod.binary_closing(sparse, np.ones((2 * WATER_CLOSE + 1,) * 2))
    wlab, _ = ndi_mod.label(sparse)
    wsz = np.bincount(wlab.ravel())
    region = np.isin(wlab, np.flatnonzero(wsz >= WATER_MIN_CELLS)) & (wlab > 0)
    local = ndi_mod.uniform_filter(dens_grid.reshape(ny, nx).astype(float),
                                   size=WATER_WIN) <= WATER_LOCAL
    both = region | local
    agree = (ndi_mod.uniform_filter(both.astype(float), size=WATER_VOTE_WIN)
             >= WATER_VOTE)
    cand = (both & agree).ravel()[flat]
    # the water surface, from the region rule's own points -- nothing external, no class field
    seed = region.ravel()[flat] & (hag1 < WATER_HAG)
    zw = float(np.median(z[seed])) if seed.any() else 0.0
    feats["in_water"] = cand
    feats["at_plane"] = np.abs(z - zw) < WATER_PLANE
    feats["above_plane"] = z > zw
    feats["water_like"] = ((dens_grid[flat] <= WATER_OWN_DENS)
                           & (np.asarray(pts["intensity"]).astype(float) < WATER_OWN_INTEN))
    print(f"    water plane at {zw:.2f} m")
    feats["planar"], feats["scatter"], _ = covariance(n)
    label, source = classify(feats)
    scatter = feats["scatter"]
    moved_cell = line_spread(pts, flat, ny * nx)
    feats["moved"] = moved_cell[flat]
    top = np.full(ny * nx, -np.inf)
    np.maximum.at(top, flat, z)
    feats["overhead"] = top[flat] - z          # structure standing above this point
    over = (np.abs(v1["filled"] - zw) < FLOAT_DEM_TOL).ravel()[flat]
    fl = float_pass(label, feats, over)
    label[fl] = FLOATING
    source[fl] = SRC["on_water"]
    # The bridge, by the library's own measured detector: a hard surface with a void under it in
    # the same cell -- a roof has no void, a crown is not hard. [measured, segment.py] on this
    # tile it finds exactly two components, Granville's two carriageways, and 100% of the deck is
    # unclassified in the survey, so until here it was polluting BUILDING's red on the map.
    # Labelled BEFORE the building components are built, so the deck never becomes a "building",
    # never gets adopted into one, and never stamps a building datum into DEM v2.
    deck = dl.bridge_deck(pts, B, pix)
    label[deck] = BRIDGE
    source[deck] = SRC["deck"]
    # The detector SEEDS the bridge, it does not bound it. Its void test needs 6 m of clear air
    # below the top in the same cell -- true mid-span, false where the deck crosses close over
    # roofs and false down the approach ramps, so the map showed one bridge in two colours,
    # orange mid-span and building-red at both ends. Same cure as water: seed with the strict
    # test, grow with the continuous one. A deck is a surface that CONTINUES: extend into a
    # neighbouring cell while its top carries on at the same height (a ramp falls ~0.05 m per
    # cell; 1 m allows joints and railings), stays hard (echo < 0.5 -- foliage cannot join), and
    # remains well above the ground DEM (4 m -- where the embankment rises to meet the ramp, the
    # DEM rises with it and the growth stops by itself).
    er_cell = np.nan_to_num(dl.echo_ratio(pts, B, pix), nan=1.0)
    ztop = np.full(ny * nx, -np.inf)
    np.maximum.at(ztop, flat, z)
    ztop = ztop.reshape(ny, nx)
    dcell = np.zeros(ny * nx, bool)
    dcell[np.unique(flat[deck])] = True
    dcell = dcell.reshape(ny, nx)
    dem1 = v1["filled"]
    # Kaveh's fact about bridges: they are not all "on fly" -- a bridge starts at grade,
    # climbs, and comes down. The 4 m DEM floor exists to keep the growth off ordinary roads,
    # but it also stops the growth partway down every ramp, exactly where the embankment
    # rises to meet the deck. The geometry cannot know where the structure truly ends; OSM
    # can -- every carriageway is a way tagged bridge=yes that stops at the abutment. So
    # inside the OSM corridor (centrelines buffered 8 m, tools/osm_bridges.py) the floor is
    # waived and the deck may be followed to grade; outside it the floor still protects the
    # roads. Without the file the pipeline still runs, with the old partial-ramp behaviour
    # named instead of hidden -- the same contract as a cloud that ships no RGB.
    corridor = np.zeros((ny, nx), bool)
    try:
        ob = json.loads((OUTDIR / "osm_bridges.json").read_text())
        line = np.zeros((ny, nx), bool)
        for w in ob["ways"]:
            c = np.asarray(w["coords"], float)
            for (x0_, y0_), (x1_, y1_) in zip(c[:-1], c[1:]):
                n_ = max(int(np.hypot(x1_ - x0_, y1_ - y0_) / (pix * 0.5)), 1) + 1
                xs = np.linspace(x0_, x1_, n_)
                ys = np.linspace(y0_, y1_, n_)
                rows = np.floor((B[3] - ys) / pix).astype(int)
                cols = np.floor((xs - B[0]) / pix).astype(int)
                ok = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)
                line[rows[ok], cols[ok]] = True
        corridor = ndi_mod.distance_transform_edt(~line, sampling=pix) <= 8.0
        print(f"    OSM corridor: {len(ob['ways'])} bridge ways, "
              f"{int(corridor.sum()):,} cells")
    except FileNotFoundError:
        print("    WARNING: out/osm_bridges.json missing -- run tools/osm_bridges.py; "
              "growth keeps its 4 m floor, so the ramps stay partial")
    grown = dcell.copy()
    for _ in range(400):
        ref = ndi_mod.grey_dilation(np.where(grown, ztop, -np.inf), size=3)
        cand = (ndi_mod.binary_dilation(grown) & ~grown
                & (np.abs(ztop - ref) < 1.0) & (((ztop - dem1) > 4.0) | corridor)
                & (er_cell < 0.5))
        if not cand.any():
            break
        grown |= cand
    ext = (grown & ~dcell).ravel()[flat] & (z > ztop.ravel()[flat] - 1.5) \
        & np.isin(label, (BUILDING, UNLABELLED))
    label[ext] = BRIDGE
    source[ext] = SRC["deck"]
    # A car on the deck is not deck, and hag cannot see it -- the ground DEM is 28 m down, so
    # the whole vehicle sits inside the deck's own top band. Same cure as everywhere else a
    # datum was wrong: measure against the right surface. The deck datum is the per-cell
    # MINIMUM of the bridge-labelled returns (a car cannot lower the surface it stands on --
    # the same reason dl.dem takes the minimum), eroded 3x3 so a cell a vehicle covers
    # entirely takes its datum from the bare deck beside it (the deck continues within 1 m
    # per cell, so the erosion cannot fall off it). The ground band's own bounds are reused
    # unchanged: 0.4 m under it is deck surface and railing kerbs, above 2 m would be masts
    # and cables. No colour test up here -- nothing grows on a carriageway, so the band goes
    # to SMALL_BRIDGE directly, and it stays a separate class so level 1 can file it under
    # the bridge family: a car on the deck belongs to the bridge system, not the ground one.
    bmask = label == BRIDGE
    deckcell = np.zeros(ny * nx, bool)
    deckcell[np.unique(flat[bmask])] = True
    dmin = np.full(ny * nx, np.inf)
    np.minimum.at(dmin, flat[bmask], z[bmask])
    # The claim region is the deck WITH its interior holes filled. A trolley wire or a lamp
    # six metres up makes its cell's top the wire, not the deck: the seed's top band misses
    # the surface, the cell never becomes deck, and the building label underneath survived as
    # red speckles ON the deck (Kaveh spotted them on the map). Those cells are enclosed by
    # deck on all sides, so fill_holes names exactly them -- and their datum is borrowed from
    # the neighbours by erosion, which cannot fall off a surface that continues within 1 m.
    # ...and the OSM corridor, which is what recovers the separated sidewalk. Granville's
    # walkway (sidewalk:right=separate in OSM) is cantilevered off the carriageway behind a
    # railing gap, so the top-continuity growth never reaches it and it is not an interior
    # hole either -- planar, hard, at deck level, it sat labelled building over open water
    # (15,492 points at median z 40.2 against the deck's 40.2). OSM maps the walkway as its
    # own bridge=yes way, so the corridor knows it is deck when the geometry cannot.
    reach = ndi_mod.binary_fill_holes(deckcell.reshape(ny, nx)) | corridor
    # one erosion everywhere -- a cell a vehicle covers entirely borrows the bare deck
    # beside it -- then everything reachable fills from its neighbours without pulling the
    # already-finite datum any lower
    dgrid = ndi_mod.grey_erosion(np.where(deckcell, dmin, np.inf).reshape(ny, nx), size=3)
    for _ in range(40):
        need = reach & ~np.isfinite(dgrid)
        if not need.any():
            break
        dgrid = np.where(need, ndi_mod.grey_erosion(dgrid, size=3), dgrid)
    datum = dgrid.ravel()
    claim = reach.ravel()[flat] & np.isfinite(datum[flat])
    hag_deck = z - datum[flat]
    # deck surface hiding under the wires: at the datum, still labelled building/unlabelled
    resurf = (claim & np.isin(label, (BUILDING, UNLABELLED))
              & (hag_deck >= -0.5) & (hag_deck < GROUND_FALLBACK))
    label[resurf] = BRIDGE
    source[resurf] = SRC["deck"]
    onb = (claim & (hag_deck >= GROUND_FALLBACK) & (hag_deck <= TALL)
           & np.isin(label, (BRIDGE, BUILDING, UNLABELLED)))
    label[onb] = SMALL_BRIDGE
    source[onb] = SRC["deck_band"]
    # ...and what stands taller than the band is bridge STRUCTURE. A lamp post's head is a
    # planar cluster 4-9 m over the deck, so covariance called it building, and the residue
    # pass then snowballed the shaft onto it (489 knn points on 92 covariance seeds in one
    # sidewalk blob) -- and because the tallest point wins the cell on the map, one lamp
    # painted its whole cell building-red. Poles, gantries and rails on the deck belong to
    # the bridge system; 12 m is generous for street furniture and below any real roof that
    # could share a cell where the ramp runs at grade. Vegetation is deliberately NOT taken:
    # crowns genuinely overhang the ramps, and a tree stays a tree wherever it leans.
    mast = (claim & (hag_deck > TALL) & (hag_deck < 12.0)
            & np.isin(label, (BUILDING, UNLABELLED)))
    label[mast] = BRIDGE
    source[mast] = SRC["deck"]
    # ...and the deck system is closed to the building passes below. Where the deck overlaps
    # a building footprint the CELL holds both (a roof below, the deck above), so footprint
    # adoption was re-claiming deck furniture -- poles and gantries above the 2 m band -- as
    # BUILDING, and the glazing fold was doing the same to windscreens. Kaveh spotted both as
    # building-red on the deck. At or above deck level the point belongs to the bridge
    # system; below it, a roof under the deck is still a roof and stays adoptable.
    ondeck = claim & (z > datum[flat] - 0.5)
    print(f"    bridge: {int(deck.sum()):,} seeded by the void test, "
          f"{int(ext.sum()):,} grown along the deck, "
          f"{int(resurf.sum()):,} deck-surface points reclaimed from building/unlabelled, "
          f"{int(onb.sum()):,} in the 0.4-2 m band above the deck datum -> small object")
    stage("  ".join(f"{NAMES[k]} {int((label==k).sum()):,}" for k in NAMES), t)

    # ---- 4. object ids ------------------------------------------------------------------------
    print("\n4. object ids")
    from scipy import ndimage as ndi

    obj = np.zeros(n, np.uint32)
    t = time.time()
    # Kaveh's ordering: vegetation decides before building, so the vegetation-vs-building
    # conflict is fixed on the VEGETATION side first, before the components are built. The
    # false trees are penetrable but not vegetal: a construction site's scaffolding, solar
    # racks on a roof, tent canopies -- all split pulses exactly like foliage, and 77% of
    # them sit within one cell of a building. Two instruments, both swept on the cluster's
    # own veg-high points:
    #   * colour -- median excess green <= 0.005 and touching a building: not green, not a
    #     tree. 15.8:1 at 0.005; 0.01 collapses to 1.2:1, because shadowed and September-
    #     brown crowns live between 0.005 and 0.01.
    #   * containment -- solar racks at exg 0.007-0.009 sit exactly in that gap, but a rack
    #     row lies ENCLOSED inside its roof where a real crown overhangs the edge: median
    #     exg <= 0.02 AND >= 70% of the cluster inside the filled building mask.
    #   * hugging -- the Ocean Concrete conveyor (exg 0.0063, one real shadowed tree
    #     cluster sits at 0.0076, so colour alone has no margin) runs pressed against its
    #     structure for its whole length: median exg <= 0.01 AND >= 50% of the cluster's
    #     cells within one cell of building. Of the 12 clusters this flips, eleven are 0%
    #     survey-tree; the only "loss" is the conveyor itself, which the survey also
    #     mislabelled as vegetation -- Kaveh's oblique photo is the arbiter there.
    # Together: 1,028 survey-building cells fixed for 51 real tree cells (20:1).
    # Flipped BEFORE bcell, so these points join their building's component and object id.
    if feats["has_rgb"]:
        vcell = np.zeros(ny * nx, bool)
        vcell[np.unique(flat[label == VEG_HIGH])] = True
        vlab_g, nvg = ndi.label(vcell.reshape(ny, nx), np.ones((3, 3)))
        bcells0 = np.zeros(ny * nx, bool)
        bcells0[np.unique(flat[label == BUILDING])] = True
        bcells0 = bcells0.reshape(ny, nx)
        nearb = ndi.binary_dilation(bcells0, iterations=2).ravel()
        bfill = ndi.binary_fill_holes(ndi.binary_closing(bcells0, np.ones((3, 3)))).ravel()
        vid = vlab_g.ravel()[flat]
        gidx = np.flatnonzero(label == VEG_HIGH)
        gidx = gidx[np.argsort(vid[gidx])]
        gmed = np.full(nvg + 1, np.nan)
        for g in np.split(gidx, np.flatnonzero(np.diff(vid[gidx]) != 0) + 1):
            if len(g):
                gmed[vid[g[0]]] = np.median(feats["exg"][g])
        gtouch = np.zeros(nvg + 1, bool)
        ginside = np.zeros(nvg + 1)
        ghug = np.zeros(nvg + 1)
        near1b = ndi.binary_dilation(bcells0).ravel()
        vl = vlab_g.ravel()
        for i in np.unique(vl[vl > 0]):
            m = vl == i
            gtouch[i] = nearb[m].any()
            ginside[i] = float(bfill[m].mean())
            ghug[i] = float(near1b[m].mean())
        grey = ((label == VEG_HIGH)
                & (((gmed[vid] <= 0.005) & gtouch[vid])
                   | ((gmed[vid] <= 0.02) & (ginside[vid] >= 0.7))))
        # A fourth term -- exg <= 0.01 AND hugging the building for >= 50% of the cluster,
        # aimed at the Ocean Concrete conveyor -- was tried and REVERTED: in pipeline
        # context (pre-adoption building cells, pre-flip clusters) it took 30k points and
        # cost veg high 0.907 -> 0.899. Tuned tighter it becomes a filter fitted to one
        # structure. The conveyor is the recorded known limit instead, unless OSM maps it.
        label[grey] = BUILDING
        source[grey] = SRC["notgreen"]
        stage(f"{int(grey.sum()):,} points in not-green penetrable clusters on buildings "
              "-> building (scaffolds, solar racks, canopies)")
    else:
        print("    WARNING: no RGB -- not-green veg-on-building pass skipped; scaffolds "
              "and solar racks will read as vegetation")
    # Suspended structures out of vegetation -- and colour deliberately plays no part,
    # because it cannot: the Ocean Concrete conveyor gantries are painted (Kaveh's find).
    # What no tree can fake is HANGING IN THE AIR: a crown's returns start 2-4 m off the
    # ground (foliage, trunk), a suspended truss's lowest return is 15 m up with clear air
    # beneath. [swept] cluster median of each cell's lowest >2 m return: >= 8 m flips 11
    # clusters at 22:1 (134 non-tree cells for 6); >= 5 m collapses to 0.1:1, because
    # crowns overhang lower ground all the time. Runs with or without RGB.
    # Also recorded dead here: splitting veg clusters into grey/green cells before the
    # colour vote (for the construction-site remnant) -- 0.4:1, the grey partition at 0.01
    # cuts through shadowed real crowns (2,382 tree cells lost). The remnant stays until a
    # better instrument exists.
    vcell2 = np.zeros(ny * nx, bool)
    vcell2[np.unique(flat[label == VEG_HIGH])] = True
    slab, ns = ndi.label(vcell2.reshape(ny, nx), np.ones((3, 3)))
    low_cell = np.full(ny * nx, np.inf)
    vh = (label == VEG_HIGH) & (hag1 > 2)
    np.minimum.at(low_cell, flat[vh], hag1[vh])
    sl = slab.ravel()
    susp = np.zeros(ns + 1, bool)
    for i in np.unique(sl[sl > 0]):
        vals = low_cell[sl == i]
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 4 and float(np.median(vals)) >= 8.0:
            susp[i] = True
    hang = (label == VEG_HIGH) & susp[sl[flat]]
    label[hang] = BUILDING
    source[hang] = SRC["suspended"]
    stage(f"{int(hang.sum()):,} points in suspended clusters -> building "
          "(gantries, cables -- nothing vegetal hangs in the air)")
    # ...and nothing vegetal is ROOTED IN THE SEA. The wharf face reads penetrable --
    # fenders, pipes and rails split pulses like foliage -- and drew a green strip along
    # the dock edge on Kaveh's map. But its cells sit over a DEM at the water plane, where
    # no tree can stand. [swept] a veg-high cluster with >= 50% of its cells over the
    # plane: 40 clusters, 350 non-tree cells, ZERO survey-tree cells lost at every
    # threshold -- a shore crown leaning over the water never dominates its own cluster.
    # Touching a building it is wharf structure; free-standing over water it is a vessel's
    # rigging and belongs to the water system.
    overc = (np.abs(dem1 - zw) < FLOAT_DEM_TOL).ravel()
    ofrac = np.zeros(ns + 1)
    for i in np.unique(sl[sl > 0]):
        ofrac[i] = float(overc[sl == i].mean())
    bcells1 = np.zeros(ny * nx, bool)
    bcells1[np.unique(flat[label == BUILDING])] = True
    nearb1 = ndi.binary_dilation(bcells1.reshape(ny, nx), iterations=2).ravel()
    sea = (label == VEG_HIGH) & (ofrac[sl[flat]] >= 0.5)
    tob = sea & nearb1[flat]
    label[tob] = BUILDING
    source[tob] = SRC["unrooted"]
    tof = sea & ~nearb1[flat]
    label[tof] = FLOATING
    source[tof] = SRC["unrooted"]
    stage(f"{int(tob.sum()):,} + {int(tof.sum()):,} points in clusters rooted over the "
          "water plane -> building / boat-or-dock (nothing grows out of the sea)")
    bcell = np.zeros(ny * nx, bool)
    bcell[np.unique(flat[label == BUILDING])] = True
    bcell = ndi.binary_closing(bcell.reshape(ny, nx), np.ones((3, 3)))   # bridge roof gaps
    blab, nb = ndi.label(bcell)
    obj[label == BUILDING] = blab.ravel()[flat[label == BUILDING]]
    # The float pass runs before the components exist and was nibbling wharf buildings
    # point by point: a pile-supported roof sits over a water-level DEM (no ground returns
    # under a wharf) and its sloped edges fake enough between-line disagreement to pass the
    # movement bar. The footprint gets the last word: floating points inside the FILLED
    # building mask return to BUILDING. A boat cannot be reclaimed -- its cells hold no
    # building points, so no footprint ever encloses it. [measured] 517 cells, 32%
    # survey-building, the rest unclassified wharf structure.
    refloat = (label == FLOATING) & ndi.binary_fill_holes(bcell).ravel()[flat]
    label[refloat] = BUILDING
    source[refloat] = SRC["footprint"]
    print(f"    {int(refloat.sum()):,} floating points inside building footprints "
          "returned to building (wharf roofs the float pass had nibbled)")

    # The object stage feeds BACK into the label stage, and this is the point of ordering them
    # this way. `planarity >= 0.7` fires on the flat interior of a roof and on nothing else: a
    # wall is vertical, a parapet is an edge, an antenna is a line, so 53.2% of the survey's
    # building points came out unlabelled. But a point standing 10 m up INSIDE a building
    # footprint is part of that building whatever its local neighbourhood looks like. The
    # footprint knows what the neighbourhood cannot.
    inside = blab.ravel()[flat] > 0
    adopt = inside & (hag1 > TALL) & (label == UNLABELLED) & ~ondeck
    label[adopt] = BUILDING
    source[adopt] = SRC["footprint"]
    obj[adopt] = blab.ravel()[flat[adopt]]
    # adjudicate the flat splitters: on the deck it is a windscreen (small object on bridge),
    # on a building it is glass, off both it was a flat crown
    deckglass = (label == GLAZING) & ondeck
    label[deckglass] = SMALL_BRIDGE
    source[deckglass] = SRC["deck_band"]
    off = (label == GLAZING) & ~inside
    label[off] = VEG_HIGH
    source[off] = SRC["echo"]
    on = label == GLAZING
    label[on] = BUILDING           # glass on a roof IS building; the source byte still says how
    obj[on] = blab.ravel()[flat[on]]
    # ...and the windows the flat gate missed: a skylight splits pulses like a crown but its
    # frames break local planarity, so it fell to vegetation high. Inside a building footprint,
    # penetrable-but-not-crown-like (scatter below the building bound) is glass, not canopy; a
    # real crown overhanging the roof keeps its high scatter and stays vegetation.
    # A skylight lives AT ROOF HEIGHT of its own building; an overhanging crown mostly does
    # not. The scatter gate alone was tried first and ate 250,000 real canopy points (veg high
    # 0.891 -> 0.867); anchoring to each component's roof plane is what makes it safe.
    bsum = np.bincount(blab.ravel()[flat[label == BUILDING]],
                       weights=z[label == BUILDING], minlength=nb + 1)
    bcnt = np.bincount(blab.ravel()[flat[label == BUILDING]], minlength=nb + 1)
    roofz = np.divide(bsum, bcnt, out=np.zeros(nb + 1), where=bcnt > 0)
    comp = blab.ravel()[flat]
    win = ((label == VEG_HIGH) & inside & ~ondeck & (scatter < SCATTER_MAX)
           & (np.abs(z - roofz[comp]) < 1.5))
    label[win] = BUILDING
    source[win] = SRC["footprint"]
    obj[win] = blab.ravel()[flat[win]]
    print(f"    glazing: {int(on.sum()):,} points folded into building, "
          f"{int(deckglass.sum()):,} windscreens on the deck, "
          f"{int(off.sum()):,} flat crowns returned to vegetation, "
          f"{int(win.sum()):,} window points reclaimed from vegetation")
    stage(f"{nb:,} building components; footprint adopted {int(adopt.sum()):,} more points", t)


    # ---- full partition: the residue takes its nearest labelled neighbour --------------------
    # What is left is the boundary fuzz -- tall, neither penetrable nor planar, mostly single
    # cells where a crown meets a roof or a wall meets the air. No rule claims it because it is
    # not a kind of thing; it is the edge between two kinds. So it joins whatever it abuts:
    # nearest labelled neighbour in 3-D, which for a wall edge is the wall and for a crown edge
    # the crown. The source byte says "knn", so these stay auditable rather than laundered.
    resid = label == UNLABELLED
    if resid.any():
        from scipy.spatial import cKDTree
        t = time.time()
        xyz = np.column_stack([np.asarray(pts["x"]), np.asarray(pts["y"]), z])
        donors = np.flatnonzero(~resid)
        _, qi = cKDTree(xyz[donors]).query(xyz[resid], k=1, workers=-1)
        label[resid] = label[donors[qi]]
        source[resid] = SRC["knn"]
        stage(f"residue: {int(resid.sum()):,} boundary points joined their nearest "
              f"labelled neighbour", t)
    assert not np.isin(label, (UNLABELLED, GLAZING)).any(), "the partition must be total"

    # ---- large vehicles out of building ------------------------------------------------------
    # Kaveh's call: a truck is not a building. Runs AFTER the residue on purpose -- an
    # earlier placement missed cars whose sparse covariance points went to the residue and
    # came back as building via knn, materialising vehicle components this pass never saw.
    # Constants and their measurements are at the top of the file. Route one is per
    # COMPONENT: vehicle-sized footprint, vehicle height, and then either the laser saw the
    # road UNDER it (a building's walls forbid that) or it moved between flight lines
    # (nothing static does). A car is only 2-3 cells wide, so erosion often leaves no
    # interior to measure; those small components fall back to the undercut over their FULL
    # footprint with a stricter bar (edge cells see the neighbouring ground even on a real
    # kiosk), plus the hard/water guards of route two.
    t = time.time()
    zmin_c = np.full(ny * nx, np.inf)
    np.minimum.at(zmin_c, flat, z)
    hag_min2d = zmin_c.reshape(ny, nx) - dem1
    hag_top2d = ztop - dem1
    moved2d = moved_cell.reshape(ny, nx)
    landward = np.abs(dem1 - zw) > FLOAT_DEM_TOL
    bnow = np.zeros(ny * nx, bool)
    bnow[np.unique(flat[label == BUILDING])] = True
    braw2 = bnow.reshape(ny, nx)
    vlab, nv = ndi.label(ndi.binary_closing(braw2, np.ones((3, 3))), np.ones((3, 3)))
    vmask = np.zeros((ny, nx), bool)
    ncomp = 0
    for i in range(1, nv + 1):
        m = vlab == i
        sm = m & braw2                   # stats over built cells, not the closing's gaps
        a = int(sm.sum())
        if not (VEH_AREA[0] <= a <= VEH_AREA[1]):
            continue
        if not (VEH_TOP[0] <= float(np.median(hag_top2d[sm])) <= VEH_TOP[1]):
            continue
        mv = float(np.nanmedian(moved2d[sm])) if np.isfinite(moved2d[sm]).any() else 0.0
        inner = ndi.binary_erosion(m)
        if inner.any():
            ok = float((hag_min2d[inner] < 0.3).mean()) > VEH_UNDER
        else:
            ok = float((hag_min2d[sm] < 0.3).mean()) > 0.85
        # every route demands a HARD top (a glass atrium's skylights return the interior
        # floor and fake undercut 1.0; a truck splits nothing) and dry land (a boat over the
        # water-plane DEM passes every other test)
        ok = (ok and float(np.median(er_cell[sm])) < 0.5
              and bool(np.median(landward[sm])))
        if mv > VEH_MOVE or ok:
            vmask |= sm
            ncomp += 1
    take = (label == BUILDING) & vmask.ravel()[flat]
    label[take] = SMALL
    source[take] = SRC["vehicle"]
    obj[take] = 0
    n1 = int(take.sum())
    # ...and the trucks the components cannot see: a row parked against a warehouse merges
    # into the WAREHOUSE's component, which fails every size gate. Same physics, per cell:
    # a building cell at vehicle height with road visible beneath it. The 2x2 opening is the
    # guard that replaces the interior erosion -- a real low house shows ground-under only in
    # its one-cell perimeter ring (the wall shares the cell with the yard), and a ring one
    # cell wide cannot survive the opening, while a vehicle is 2-3 cells wide and does.
    # The guard against a false flip is the ROOFLINE STEP. A low house's eave also shows
    # ground beneath (the yard under the overhang), and a first cut without this guard took
    # 151,083 points and cost building recall 0.92 -> 0.88 -- the survey said those were
    # buildings. An eave's top runs CONTINUOUSLY into its own roof; a truck's top steps
    # against the wall it is parked at. So a candidate region is a vehicle only if the
    # building cells it touches are >= 1 m taller or lower than it -- or it touches none.
    # Two more guards, both found on the map: a SKYLIGHT lets the beam through to the
    # interior floor, faking ground-under on a real market roof -- but glass splits the
    # pulse, so those cells are penetrable where a truck is hard (echo < 0.5). And a boat at
    # a dock passes every vehicle test -- it is low, solid, and floats over "ground" -- but
    # it does it over a DEM at the water plane, which nothing parked on land can.
    bnow = np.zeros(ny * nx, bool)
    bnow[np.unique(flat[label == BUILDING])] = True
    bnow = bnow.reshape(ny, nx)
    cand = (bnow & (hag_top2d >= VEH_TOP[0]) & (hag_top2d <= VEH_TOP[1])
            & (hag_min2d < 0.3) & (er_cell < 0.5) & landward)
    reg, _ = ndi.label(cand, structure=np.ones((3, 3)))
    opened = np.unique(reg[ndi.binary_opening(cand, np.ones((2, 2)))])
    sz = np.bincount(reg.ravel())
    vmask2 = np.zeros((ny, nx), bool)
    for g in opened:
        if not g or not (VEH_AREA[0] <= sz[g] <= VEH_AREA[1]):
            continue
        r = reg == g
        ring = ndi.binary_dilation(r) & bnow & ~cand
        # the ring must EXIST: with no non-candidate building around it there is nothing to
        # measure the roofline step against -- that is either a free-standing vehicle (the
        # component route's job) or an entire glass roof whose skylights faked ground-under
        # in every cell, which is how a market roof got eaten whole. Isolated regions are
        # not this route's to claim.
        if not ring.any():
            continue
        step = np.abs(np.median(hag_top2d[ring]) - np.median(hag_top2d[r]))
        if step < 1.0:
            continue                            # continuous roofline: an eave, not a truck
        vmask2 |= r
    take2 = (label == BUILDING) & vmask2.ravel()[flat]
    label[take2] = SMALL
    source[take2] = SRC["vehicle"]
    obj[take2] = 0
    stage(f"{ncomp} building components were vehicles ({n1:,} points), plus "
          f"{int(take2.sum()):,} points of vehicles parked against buildings -> small object", t)

    # A blanket "building over the water plane -> boat or dock" was tried here (2026-07-30)
    # and reverted the same day: it took 82,330 points and Kaveh's own review overruled it --
    # some structures over the water genuinely ARE buildings (pile-supported wharf
    # structures), and the water-family agreement collapsed 0.300 -> 0.178. Splitting a
    # floating home from a wharf shed needs a floats-vs-piles instrument (tide movement at
    # the structure's own cells), not a blanket rule. Recorded so the dead end stays dead.

    # ---- level 1: the four families, a total partition of their own --------------------------
    level1 = np.zeros(n, np.uint8)
    for k, fam in FAMILY.items():
        level1[label == k] = fam
    assert (level1 > 0).all(), "every point must belong to a family"

    ntree = 0
    if "--no-trees" not in argv:
        t = time.time()
        tid = lidr_crowns(pts, label == VEG_HIGH,
                          Path(tempfile.gettempdir()) / "pipeline_crowns.las")
        ntree = int(tid.max())
        obj[label == VEG_HIGH] = tid + nb              # ids stay unique across classes
        obj[(label == VEG_HIGH) & (obj == nb)] = 0     # ...but 0 still means "no object"

    # each carriageway is its own object, downstream of every building and crown id.
    # Connectivity comes from deck OR on-deck cells: a row of cars re-labelled SMALL_BRIDGE
    # would otherwise punch holes in the mask and split one carriageway into fragments
    # (measured: 9 components instead of 2 + ramps without this union).
    bcell2 = np.zeros(ny * nx, bool)
    bcell2[np.unique(flat[np.isin(label, (BRIDGE, SMALL_BRIDGE))])] = True
    brlab, nbr = ndi.label(bcell2.reshape(ny, nx), structure=np.ones((3, 3)))
    m = label == BRIDGE
    obj[m] = brlab.ravel()[flat[m]] + nb + ntree
    if nbr:
        stage(f"{nbr} bridge carriageways")
        stage(f"{ntree:,} tree crowns", t)

    # ---- 5. DEM v2: filled by CAUSE ----------------------------------------------------------
    print("\n5. DEM v2 — filled by cause")
    t = time.time()
    raw2 = v1["dtm"].copy()
    stamped = 0
    # a truck must not stamp a building pad into the DEM: skip any footprint component the
    # vehicle pass emptied of building points
    alive = np.zeros(nb + 1, bool)
    alive[np.unique(blab.ravel()[flat[label == BUILDING]])] = True
    for i in range(1, nb + 1):
        if not alive[i]:
            continue
        foot = blab == i
        if foot.sum() < 20:
            continue
        try:
            lv = dl.building_level(v1, foot)
        except ValueError:
            continue
        hole = foot & ~np.isfinite(raw2)        # never overwrite a real return in a courtyard
        raw2[hole] = lv["level"]
        stamped += int(hole.sum())
    filled2, dist2 = dl.fill(raw2, pix)
    hag2 = z - filled2.ravel()[flat]
    moved = np.abs(filled2 - v1["filled"])
    stage(f"{stamped:,} cells stamped with a building datum; "
          f"DEM moved by {np.nanmean(moved):.3f} m mean, {np.nanmax(moved):.2f} m max", t)

    # ---- 6. out -------------------------------------------------------------------------------
    print("\n6. writing")
    lab_cell = np.zeros(ny * nx, np.uint8)
    order = np.argsort(z)                      # tallest wins the cell, as seen from above
    lab_cell[flat[order]] = label[order]
    obj_cell = np.zeros(ny * nx, np.uint32)
    obj_cell[flat[order]] = obj[order]

    # Per-class cell masks use ANY point, not the topmost one. The topmost rule is right for a
    # picture -- it is what you would see from above -- and wrong for scoring, because the truth
    # rasters mark a cell if it holds ANY point of the class. Comparing "the tallest thing here
    # is ground" against "there is ground here" scored our ground at 0.355 where it is 0.932, and
    # was quietly understating building and vegetation the same way.
    # TOPMOST, not "any". A wooded cell contains ground AND canopy, so "any point of class k"
    # answers "is there any k here", which is true of ground almost everywhere -- 90% of
    # vegetation cells hold a ground return, because the beam gets through. That is not a
    # decision about the cell, and a map drawn from it shows trees inside the ground layer.
    # The topmost point's label is what you would see from above, and it is what the imagery
    # underneath the dashboard shows. Truth is built the same way, so the two are comparable.
    def topcell(k):
        return (lab_cell == k).reshape(ny, nx)
    l1_cell = np.zeros(ny * nx, np.uint8)
    l1_cell[flat[order]] = level1[order]
    np.savez_compressed(OUTDIR / "pipeline.npz",
                        label=lab_cell.reshape(ny, nx), object_id=obj_cell.reshape(ny, nx),
                        level1=l1_cell.reshape(ny, nx),
                        vegetation=topcell(VEG_HIGH), building=topcell(BUILDING),
                        water=topcell(WATER), ground=topcell(GROUND),
                        floating=topcell(FLOATING), lowveg=topcell(VEG_LOW),
                        small=topcell(SMALL), bridge=topcell(BRIDGE),
                        smallbridge=topcell(SMALL_BRIDGE),
                        dem=filled2, distance=dist2, n_building=nb, n_tree=ntree)
    print(f"    wrote {OUTDIR/'pipeline.npz'}")

    print("\n--- summary " + "-" * 52)
    for k, name in NAMES.items():
        m = label == k
        print(f"  {name:<12} {int(m.sum()):>10,}  {m.mean()*100:>5.1f}%   "
              f"objects: {len(np.unique(obj[m])) - (1 if (obj[m] == 0).any() else 0):>6,}")
    print(f"  {'hag change':<12} {np.abs(hag2-hag1).mean():>10.3f} m mean, "
          f"{np.abs(hag2-hag1).max():.2f} m max between v1 and v2")

    cls = np.asarray(pts["classification"])
    # Where the points actually went. An IoU column says a class is wrong; this says what it was
    # called instead, which is the only version you can act on.
    print("\n--- confusion: the survey's class (rows) vs this pipeline's label (columns) ---")
    groups = [("ground", cls == 2), ("vegetation", np.isin(cls, (3, 4, 5))),
              ("building", cls == 6), ("water", cls == 9),
              ("unclassified", np.isin(cls, (0, 1)))]
    print(f"  {'survey':<14}{'n':>11}" + "".join(f"{NAMES[k]:>12}" for k in NAMES))
    for name, m in groups:
        if not m.any():
            continue
        print(f"  {name:<14}{int(m.sum()):>11,}"
              + "".join(f"{float((label[m]==k).mean())*100:>11.1f}%" for k in NAMES))

    # Scored ONLY where the survey committed. 24% of this cloud is unclassified, and counting
    # those against us measures how closely we imitate a survey's silence, which is not the goal.
    judged = ~np.isin(cls, (0, 1))
    print(f"\n--- agreement where the survey committed ({judged.mean()*100:.0f}% of points) "
          "— a sanity check, NOT a verdict ---")
    for name, mine, theirs in (("ground", label == GROUND, cls == 2),
                               ("vegetation high", label == VEG_HIGH, cls == 5),
                               ("vegetation low", label == VEG_LOW, cls == 3),
                               ("vegetation both", np.isin(label, (VEG_LOW, VEG_HIGH)),
                                np.isin(cls, (3, 4, 5))),
                               ("building", label == BUILDING, cls == 6),
                               ("water", label == WATER, cls == 9)):
        m, th = mine & judged, theirs & judged
        tp = float(np.logical_and(m, th).sum())
        print(f"  {name:<12} IoU {tp/max(float(np.logical_or(m,th).sum()),1):.3f}  "
              f"precision {tp/max(float(m.sum()),1):.2f}  "
              f"recall {tp/max(float(th.sum()),1):.2f}")

    print("\n--- level 1, the five families (bridge has no ASPRS family to score against) ---")
    print(f"  {'bridge':<12} {float((level1 == FAM_BRIDGE).mean())*100:.1f}% of points, unscored")
    for fam, mine, theirs in ((FAM_GROUND, level1 == FAM_GROUND, cls == 2),
                              (FAM_VEG, level1 == FAM_VEG, np.isin(cls, (3, 4, 5))),
                              (FAM_BUILDING, level1 == FAM_BUILDING, cls == 6),
                              (FAM_WATER, level1 == FAM_WATER, cls == 9)):
        m, th = mine & judged, theirs & judged
        tp = float(np.logical_and(m, th).sum())
        print(f"  {FAM_NAMES[fam]:<12} IoU {tp/max(float(np.logical_or(m,th).sum()),1):.3f}  "
              f"precision {tp/max(float(m.sum()),1):.2f}  "
              f"recall {tp/max(float(th.sum()),1):.2f}")
    print("  NOTE the water family is judged against a reference with a DOCUMENTED error here:"
          "\n  the survey files 50,877 floating-structure points as building, so a correct"
          "\n  boat-or-dock label counts against the family. The map is the arbiter, not this row.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
