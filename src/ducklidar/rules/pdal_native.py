"""Buildings and trees from the points themselves — no render, no network, no training.

Everything else in this comparison flattened the cloud to an image first, which throws away the
one thing LiDAR has that a camera does not: the local *shape* of the neighbourhood around each
return. PDAL's `filters.covariancefeatures` puts it back. Take the k nearest neighbours of a
point, form their covariance matrix, and its three eigenvalues say what shape they make:

======================  ===================  ==============================
eigenvalue pattern      feature              what it is
======================  ===================  ==============================
one large, two small    **linearity**        a wire, an edge, a branch
two large, one small    **planarity**        a roof, a wall, the road
all three similar       **scattering**       foliage — structure in every direction
======================  ===================  ==============================

That is the same physical distinction the echo ratio makes — solid versus porous — measured a
completely different way, from geometry rather than from pulse splitting. Which makes it a real
second opinion rather than a restatement.

The classifier is two thresholds on features PDAL computes, plus height from our own DEM. No
learning of any kind. [measured] the features separate the classes exactly as the physics says
they should — median values over 10.7M points:

=================  =========  ==========
class              planarity  scattering
=================  =========  ==========
ground             0.758      0.048
building           0.655      0.120
high vegetation    0.324      **0.405**
=================  =========  ==========

Thresholds swept rather than guessed, since every other method in this comparison was given its
best operating point. Buildings peak at ``planarity >= 0.7, scattering < 0.3`` (IoU 0.609) and
vegetation at ``scattering >= 0.4`` (IoU 0.572). Both are recall-heavy — 0.91 and 0.87 — and
precision-poor, 0.65 and 0.63: eigenvalue shape says *what kind of surface* a neighbourhood is,
and says nothing about where one object ends and the next begins.

    python tools/pdal_native.py [--knn 16]
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import ducklidar as dl
from ._env import OUTDIR, check, STUDY_BOX

B = dl.box(*STUDY_BOX)
LAZ = Path("/tmp/lidar-eval/granville.laz")
PDAL = Path("/home/kaveh/miniconda3/envs/lidar/bin/pdal")


def features(knn):
    """Per-point Linearity / Planarity / Scattering / Verticality, via PDAL."""
    import laspy

    out = Path(tempfile.gettempdir()) / f"granville_feat_{knn}.las"
    if not out.exists():
        pipeline = [
            str(LAZ),
            {"type": "filters.covariancefeatures", "knn": knn, "threads": 8,
             "feature_set": "Dimensionality"},
            {"type": "writers.las", "filename": str(out), "extra_dims": "all",
             "minor_version": 4, "dataformat_id": 6},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"pipeline": pipeline}, f)
            spec = f.name
        t0 = time.time()
        r = subprocess.run([str(PDAL), "pipeline", spec], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"pdal failed:\n{r.stderr[-2000:]}")
        print(f"  covariancefeatures knn={knn}: {time.time()-t0:.0f}s")
    las = laspy.read(str(out))
    names = [d.name for d in las.point_format.extra_dimensions]
    return las, names


def main(argv):
    knn = int(argv[argv.index("--knn") + 1]) if "--knn" in argv else 16
    if not LAZ.exists():
        raise SystemExit(f"{LAZ} missing — run tools/export_laz.py first")

    las, extra = features(knn)
    print(f"  extra dims: {extra}")
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    plan = np.asarray(getattr(las, "Planarity"))
    scat = np.asarray(getattr(las, "Scattering"))
    cls = np.asarray(las.classification)
    print(f"  {len(x):,} points")

    pts = {"x": x, "y": y, "z": z, "classification": cls}
    gnd = dl.ground_filter(pts, cloth_resolution=1.0)
    r = dl.dem(pts, B, 1.0, ground=gnd)
    flat, (ny, nx) = dl.cell_index(pts, B, 1.0)
    h = z - r["filled"].ravel()[flat]          # height above ground, per point

    print(f"\n{'class':>18} {'n':>10} {'planarity':>10} {'scattering':>11}")
    for code, name in ((2, "ground"), (6, "building"), (5, "high vegetation")):
        m = cls == code
        if m.sum():
            print(f"{name:>18} {m.sum():>10,} {np.median(plan[m]):>10.3f} "
                  f"{np.median(scat[m]):>11.3f}")

    TALL, PLANAR, SCATTER = 2.0, 0.70, 0.30      # swept; see the module docstring
    VEG_SCATTER = 0.40
    tall = h > TALL
    is_bld = tall & (plan >= PLANAR) & (scat < SCATTER)
    is_veg = tall & (scat >= VEG_SCATTER)

    gt_b = np.zeros(ny * nx, bool); gt_b[np.unique(flat[cls == 6])] = True
    gt_v = np.zeros(ny * nx, bool); gt_v[np.unique(flat[np.isin(cls, (3, 5))])] = True
    out = {}
    print(f"\n{'':>28} {'IoU':>6} {'prec':>6} {'rec':>6}")
    for name, sel, gt in (("building", is_bld, gt_b), ("vegetation", is_veg, gt_v)):
        p = np.zeros(ny * nx, bool); p[np.unique(flat[sel])] = True
        tp = np.logical_and(p, gt).sum()
        iou = tp / max(np.logical_or(p, gt).sum(), 1)
        print(f"  PDAL covariance {name:<11} {iou:>6.3f} {tp/max(p.sum(),1):>6.2f} "
              f"{tp/max(gt.sum(),1):>6.2f}")
        out[name] = p.reshape(ny, nx)

    np.savez_compressed(OUTDIR / "pdal_native.npz", **out)
    print(f"\nwrote {OUTDIR/'pdal_native.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
