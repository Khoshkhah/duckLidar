"""The trial's ORDER, on a toy small enough to reason about by hand.

The port's real acceptance is the owned-tile equivalence run (shadowCity2
tools/stage5_dev.py conviction knn — bit-identical to rules/layers.py on all
four verdicts). This guards the thing that run cannot isolate and that every
re-authoring of the block gets wrong: the four steps read layers the previous
step just edited.
"""
import numpy as np

import ducklidar as dl

NY = NX = 6


def _base(n, cell):
    """A defendant population with no evidence either way."""
    return dict(
        flat=np.asarray(cell), hag=np.full(n, 5.0), shape=(NY, NX),
        cont=np.ones(n, bool), sub=np.ones(n, bool),
        geo=np.zeros(n),                       # reached: route 1 acquits
        cube=np.full(n, 0.9),                  # scattered: route 2 acquits
        veg_high=np.ones(n, bool), building=np.zeros(n, bool),
        inside_a=np.zeros(n, bool), scatter=np.zeros(n),
        exg=np.full(n, -1.0), not_last=np.zeros(n, bool))


def test_pane_votes_whole_then_rescue_undoes_it():
    # ten points in one cell => one 26-connected pane. Six convict on the cube
    # alone, so the pane's 60% majority convicts all ten.
    n = 10
    a = _base(n, np.full(n, 2 * NX + 2))
    a["cube"][:6] = 0.1
    _, _, v = dl.convict_glass(**a)
    assert int(v["convicted"].sum()) == n, "the pane vote did not carry the pane"

    # ...and the rescue runs LAST, so it can free what the vote just convicted:
    # make the cell scatter-dense and green.
    a["scatter"] = np.full(n, 1.0)
    a["exg"] = np.full(n, 1.0)
    _, _, v = dl.convict_glass(**a)
    assert int(v["convicted"].sum()) == 0 and int(v["rescued"].sum()) == n


def test_last_return_filter_counts_after_the_roof_tree_fix():
    # one cell, 8 veg points + 1 splitting building point. Before the roof-tree
    # fix veg out-claims building 8:1 and the splitter is dropped; the fix moves
    # the veg into building first, so 0 > 6 * 9 is false and it survives.
    n = 9
    a = _base(n, np.full(n, 2 * NX + 2))
    a["cont"] = np.zeros(n, bool)              # no trial, only the two fixes
    a["veg_high"][8] = False
    a["building"][8] = True
    a["not_last"][8] = True
    _, bld, v = dl.convict_glass(**a)
    assert int(v["rooffix"].sum()) == 0 and int(v["crownsplit"].sum()) == 1

    a["inside_a"] = np.ones(n, bool)           # now the fix fires
    a["cube"] = np.full(n, 0.1)
    _, bld, v = dl.convict_glass(**a)
    assert int(v["rooffix"].sum()) == 8, "the escaped veg did not move"
    assert int(v["crownsplit"].sum()) == 0, "the 6:1 ratio fired on the old counts"
    assert bld.all()


def test_infinite_geodesic_convicts_and_the_cube_still_guards():
    n = 4
    # cells two apart in both axes, so no two points share a 26-neighbourhood
    # and the pane vote cannot carry an acquittal into a conviction
    a = _base(n, np.array([0, 2, 4, 2 * NX]))
    a["geo"] = np.full(n, np.inf)              # nothing reached the ground
    a["cube"] = np.array([0.2, 0.3, 0.2, 0.3])
    _, _, v = dl.convict_glass(**a)
    # < 0.25 convicts through route 1, >= 0.25 does not — inf is a conviction
    # only when the cube agrees (layers.py:473)
    assert v["convicted"].tolist() == [True, False, True, False]
