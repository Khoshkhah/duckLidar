"""Deciding which returns are ground, from x, y, z alone.

**This is the step that makes a DTM a DTM.** Everything else in this package will happily grid
whatever you hand it; the hard question is which returns are bare earth, and a LAS file's
``classification`` field is *someone else's answer* to it — assigned by the survey's software
after the flight, sometimes hand-edited, and absent altogether from many surveys. Using it and
calling the result "our DTM" is circular.

:func:`ground_filter` answers it from the geometry: **cloth simulation** (Zhang et al. 2016).
Turn the cloud upside down, drape a stiff sheet over it under gravity, and keep the returns the
cloth settles onto. It needs no classification, no intensity and no returns metadata — three
coordinates and nothing else.

**[measured]** On a 200 m window of 2.0 M returns with the survey's labels wiped, scored
against those labels afterwards:

===========  =========  ==========  ========  ======
filter       found      precision   recall    F1
===========  =========  ==========  ========  ======
``smrf``     733,115    68.6%       99.9%     81.4%
``csf``      613,121    80.3%       97.9%     88.2%
===========  =========  ==========  ========  ======

Both find nearly all the real ground. The difference is what else they sweep in, and it is
decisive: **42% of SMRF's false positives are buildings**, which drags a DTM upward under every
flat roof. None of CSF's are — a roof pushes the cloth up, and the cloth cannot reach into a
large flat roof's interior. Its false positives are 99% *unclassified*, some of which is
probably ground the survey never labelled.

A caveat that applies to every number above: **the survey's classification is not truth**, it is
another algorithm's output. These score *agreement*, not correctness.
"""
from __future__ import annotations

import numpy as np

#: Matched to PDAL's ``filters.csf`` defaults, which **[measured]** beat ours by 5 F1 points on
#: a 500 m urban window. The two that matter interact, so they must move together:
#: ``cloth_resolution`` 1.0 with ``slope_smooth`` **on** is the best pair (F1 81.6%, 1% of its
#: errors are buildings); 0.5 with smoothing on is the worst (77.5%, **23%** buildings), because
#: a fine cloth already sags toward roofs and the smoothing pass then pulls it the rest of the
#: way. `rigidness` 3 is the "flat terrain" setting; drop to 2 for rolling relief and 1 for
#: steep slopes, or the cloth bridges over valleys.
DEFAULTS = dict(cloth_resolution=1.0, class_threshold=0.5, rigidness=3,
                iterations=500, slope_smooth=True)


def ground_filter(pts, *, cloth_resolution=None, class_threshold=None, rigidness=None,
                  iterations=None, slope_smooth=None, last_only=True):
    """Boolean mask over `pts`: which returns are bare earth. Uses x, y, z only.

    Runs in two steps. **Step 1, the veto:** a return that is not its pulse's LAST echo cannot
    be ground — the ground is where the pulse stops. Physically necessary and **[measured]** at
    exactly 100.0% of this survey's ground returns; the veto discards ~20% of all returns
    (mid-canopy hits) before the cloth ever sees them. It only ever *rejects*: a roof's single
    echo is also a last return and passes to step 2, which is what rejects it — by geometry,
    metres above the settled cloth. This is also the industry default (`lasground` classifies
    on last returns unless told otherwise); here it is derived from the physics instead.
    **Step 2, the cloth:** on the surviving candidates, cloth simulation finds the lowest
    smooth surface, and candidates within `class_threshold` of it are ground.

    Args:
        last_only: apply the veto. Needs `return_number` and `number_of_returns` in `pts`;
            silently skipped when either is absent, since the veto can only be computed from
            the pulse structure.
        cloth_resolution: cloth grid spacing in metres. Smaller follows detail and costs
            time; larger smooths over it. 0.5 suits 40 pts/m² city data.
        class_threshold: how far a return may sit from the settled cloth and still count as
            ground, in metres.
        rigidness: 1 steep, 2 relief, 3 flat. Too rigid and the cloth bridges a valley; too
            slack and it wraps up the side of a building.
        slope_smooth: CSF's post-processing pass over the settled cloth. We had this **off**
            on the assumption that it was only for steep terrain — **[measured]** wrong, and it
            cost 5 F1 points: with `cloth_resolution` 1.0 it lifts recall 80.7% → 94.4% at
            unchanged precision. It is what PDAL defaults to, and turning it on reproduces
            PDAL's score exactly. Only turn it off with a fine cloth, where it drags the sheet
            onto roofs.

    Returns a bool array, not a modified `pts` — so the caller can keep the survey's own
    classification alongside it and compare, which is the only way to know it worked.
    """
    import CSF

    p = dict(DEFAULTS)
    for k, v in dict(cloth_resolution=cloth_resolution, class_threshold=class_threshold,
                     rigidness=rigidness, iterations=iterations,
                     slope_smooth=slope_smooth).items():
        if v is not None:
            p[k] = v

    n = len(np.asarray(pts["x"]))
    cand = np.ones(n, dtype=bool)
    if last_only and "return_number" in pts and "number_of_returns" in pts:
        cand = np.asarray(pts["return_number"]) == np.asarray(pts["number_of_returns"])

    xyz = np.column_stack([np.asarray(pts["x"], dtype="float64")[cand],
                           np.asarray(pts["y"], dtype="float64")[cand],
                           np.asarray(pts["z"], dtype="float64")[cand]])
    csf = CSF.CSF()
    csf.params.cloth_resolution = float(p["cloth_resolution"])
    csf.params.class_threshold = float(p["class_threshold"])
    csf.params.rigidness = int(p["rigidness"])
    csf.params.interations = int(p["iterations"])          # the library spells it this way
    csf.params.bSloopSmooth = bool(p["slope_smooth"])
    csf.setPointCloud(xyz)

    ground, non_ground = CSF.VecInt(), CSF.VecInt()
    csf.do_filtering(ground, non_ground, exportCloth=False)   # else CSF dumps cloth_nodes.txt in cwd

    picked = np.zeros(len(xyz), dtype=bool)
    picked[np.asarray(ground, dtype="int64")] = True
    mask = np.zeros(n, dtype=bool)
    mask[np.flatnonzero(cand)[picked]] = True          # back to full-length indexing
    return mask


def score_against(mask, classification, ground_class=2):
    """Compare a computed ground mask with a survey's own labels.

    Returns precision, recall and F1 — but read them as **agreement**, not accuracy. The
    survey's classification is another algorithm's output, not ground truth, so a
    disagreement does not say which of the two is wrong.
    """
    truth = np.asarray(classification) == ground_class
    mask = np.asarray(mask, dtype=bool)
    tp = int((mask & truth).sum())
    fp = int((mask & ~truth).sum())
    fn = int((~mask & truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"found": int(mask.sum()), "agreed": tp,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "extra": fp, "missed": fn}
