"""The map vetoes' one runnable check — the traps that change the counts.

A 20x20 m grid, one 9x9 m "building" (its 3-cell erosion leaves a 3x3 core),
one 1x1 m "storage tank" in the corner, industrial landuse over everything.

    ORDER: the rules are sequential, each on the previous one's labels, so a
    point deep inside the footprint AND within 20 cells of the tank belongs
    to the FOOTPRINT veto, not the tank rule. OR-ing three predicates against
    one frozen array gives the same final labels but different counts — and
    the counts are the scoreboard.
    DEPTH: a point merely inside a footprint is not convicted (crowns
    overhang; bare-inside is 50% real veg).
    COLOUR: green protects a point from the footprint and plant rules — and
    NOT from the tank rule, which has no conjunct but position.
    DIAMOND: `iterations=N` is a city-block diamond of radius N, not a disc.
"""
import numpy as np

from ducklidar.labeling import (BUILDING, FLOATING, SMALL, VEG_HIGH,
                                footprint_cells, map_vetoes,
                                marina_superstructure)

BBOX = (0.0, 0.0, 20.0, 20.0)
RING = [[2, 2], [11, 2], [11, 11], [2, 11], [2, 2]]        # 9x9, erodes to 3x3
TANK = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]            # one cell, the corner
DOC = {"polys": [{"coords": RING}, {"coords": TANK, "man_made": "storage_tank"}],
       "industrial": [{"coords": [[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]]}]}


def test_footprint_cells_is_centre_in_ring():
    fp = footprint_cells([{"coords": RING}], BBOX)
    assert fp.sum() == 81 and fp[2:11, 2:11].all()         # centres 2.5 .. 10.5


def test_veto_order_depth_and_colour():
    cell = np.array([6 * 20 + 6,      # deep inside, not green      -> footprint veto
                     3 * 20 + 3,      # inside but shallow          -> plant rule
                     7 * 20 + 7,      # deep but GREEN, near tank   -> tank rule
                     19 * 20 + 19])   # green, 38 cells from tank    -> survives
    label = np.array([VEG_HIGH] * 4, np.uint8)
    exg = np.array([0.0, 0.0, 0.5, 0.5])
    masks, grids = map_vetoes(label, cell, exg, np.zeros(4), DOC, BBOX)

    assert grids["deep"].sum() == 9 and grids["deep"][5:8, 5:8].all()
    assert list(label) == [BUILDING, BUILDING, BUILDING, VEG_HIGH]
    assert [int(masks[k].sum()) for k in ("silo", "plant", "tank")] == [1, 1, 1]
    # the first point is within the tank's reach as well; the footprint veto
    # took it first, so the tank rule must not count it a second time
    assert grids["near_tank"][6, 6] and not masks["tank"][0]
    # city-block diamond, not a disc: (19,1) is 20 cells from the tank, (19,2) is 21
    assert grids["near_tank"][19, 1] and not grids["near_tank"][19, 2]


def test_marina_floor_is_per_cell():
    body = np.ones((20, 20), bool)
    #    cell 0: a floating deck at z=1 and a mast at z=3 -> the mast joins it
    #    cell 1: no floating point -> +inf floor, the overhanging crown is safe
    label = np.array([FLOATING, SMALL, VEG_HIGH], np.uint8)
    rig = marina_superstructure(label, np.array([0, 0, 1]),
                                np.array([1.0, 3.0, 3.0]), body, BBOX)
    assert list(rig) == [False, True, False]
    assert list(label) == [FLOATING, FLOATING, VEG_HIGH]
