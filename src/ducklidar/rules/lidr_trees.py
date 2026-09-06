"""Per-tree crowns from lidR — the point-native answer to "give me all points of that tree".

Kaveh's complaint, measured: SAM reaches 46% of the vegetated cells and labels 63% of the
vegetation points inside the ones it reaches. A mask names a surface and a crown is a volume, so
no amount of prompting closes that. lidR segments crowns in three dimensions, where the boundary
between two touching trees actually exists.

Why lidR and not PDAL's ``filters.litree``: they are the same Li 2012 algorithm, and PDAL's ran
**45 minutes at 100% CPU on this tile without emitting anything**. lidR is what people actually
run at this size, and it also offers ``dalponte2016``, which works off a canopy height model
instead of walking every point — a different complexity class, not a faster loop.

**Licence.** lidR is GPL-3.0. Per this project's rule it may be *run as an external tool* and
must never be vendored into or linked from shadowCity, which is MIT. This file shells out to
Rscript and reads a LAS back; nothing links.

    python tools/lidr_trees.py                -> out/lidr.npz, then tools/compare_dashboard.py
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import ducklidar as dl
from ._env import OUTDIR, STUDY_BOX

B = dl.box(*STUDY_BOX)
PIX = 1.0
LAZ = Path("/tmp/lidar-eval/granville.laz")
RSCRIPT = Path("/home/kaveh/miniconda3/envs/rlidr/bin/Rscript")
CHM_RES = 0.5         # m; the canopy model dalponte2016 grows crowns on
WINDOW = 5.0          # m; local-maximum window for finding treetops
MIN_HEIGHT = 3.0      # m above ground before anything counts as a tree

R_SCRIPT = """
suppressPackageStartupMessages(library(lidR))
args <- commandArgs(trailingOnly = TRUE)
las <- readLAS(args[1], select = "xyzc")
if (is.empty(las)) stop("empty LAS")
cat("points:", nrow(las@data), "\\n")

# normalise against the survey's own ground class. This project normally refuses to trust the
# file's classification, but here it only decides where the ground is, not what is a tree.
las <- normalize_height(las, tin())
cat("normalised\\n")

chm <- rasterize_canopy(las, res = %(res)s, algorithm = p2r(0.2, na.fill = tin()))
ttops <- locate_trees(chm, lmf(ws = %(ws)s, hmin = %(hmin)s))
cat("treetops:", nrow(ttops), "\\n")

las <- segment_trees(las, dalponte2016(chm, ttops, th_tree = %(hmin)s))
id <- las@data$treeID
id[is.na(id)] <- 0L
las <- add_lasattribute(las, as.integer(id), "treeID", "individual tree id")
writeLAS(las, args[2])
cat("wrote", args[2], "\\n")
"""


def candidates():
    """Write the vegetation candidates lidR should see, and only those.

    dalponte2016 grows a crown around every treetop in the canopy model, and an urban canopy
    model is full of roofs: unfiltered it claimed 56.2% of this cloud and scored precision 0.35.
    It is a forest algorithm and this is not a forest, so something has to say what is foliage
    before it says where one crown stops.

    That something is ``echo_ratio``, which is the best vegetation method on the comparison page
    at IoU 0.688 — a return that lets light through is foliage, a return off a roof does not.
    Semantic stage from geometry, instance stage from lidR; neither one reads a class field.
    """
    import laspy
    from ._env import check

    dest = Path(tempfile.gettempdir()) / "lidr_veg_candidates.laz"
    if dest.exists():
        return dest
    pts = dl.read(check(), B)
    gnd = dl.ground_filter(pts, cloth_resolution=1.0)
    _, _, ndsm = dl.instances(pts, B, 1.0, ground=gnd)
    echo = np.nan_to_num(dl.echo_ratio(pts, B, 1.0), nan=0.0)
    h = np.nan_to_num(ndsm, nan=0.0)
    veg_cell = ((h > 2.0) & (echo >= 0.45)).ravel()
    flat, _ = dl.cell_index(pts, B, 1.0)
    keep = veg_cell[flat]
    print(f"  echo-ratio candidates: {int(keep.sum()):,} of {len(keep):,} points "
          f"({keep.mean()*100:.1f}%)")

    hdr = laspy.LasHeader(point_format=6, version="1.4")
    hdr.scales = [0.001] * 3
    hdr.offsets = [np.floor(np.asarray(pts[k])[keep].min()) for k in "xyz"]
    las = laspy.LasData(hdr)
    for k in "xyz":
        setattr(las, k, np.asarray(pts[k])[keep])
    las.classification = np.asarray(pts["classification"])[keep].astype(np.uint8)
    las.write(str(dest))
    return dest


def segment():
    """Run lidR and hand back x, y, z, treeID, classification per point."""
    import laspy

    src = candidates()
    dest = Path(tempfile.gettempdir()) / f"lidr_{CHM_RES:g}_{WINDOW:g}_{MIN_HEIGHT:g}_veg.las"
    if not dest.exists():
        if not RSCRIPT.exists():
            raise SystemExit(f"{RSCRIPT} missing — lidR lives in the `rlidr` conda env")
        with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as f:
            f.write(R_SCRIPT % {"res": CHM_RES, "ws": WINDOW, "hmin": MIN_HEIGHT})
            script = f.name
        t0 = time.time()
        r = subprocess.run([str(RSCRIPT), script, str(src), str(dest)],
                           capture_output=True, text=True)
        print(r.stdout.strip())
        if r.returncode:
            raise SystemExit(f"Rscript failed:\n{r.stderr[-3000:]}")
        print(f"  lidR dalponte2016: {time.time()-t0:.0f}s")
    las = laspy.read(str(dest))
    names = [d.name for d in las.point_format.extra_dimensions]
    if "treeID" not in names:
        raise SystemExit(f"no treeID in {dest}; extra dims are {names}")
    return (np.asarray(las.x), np.asarray(las.y), np.asarray(las.z),
            np.asarray(las.treeID).astype(np.int32), np.asarray(las.classification))


def main(argv):
    if not LAZ.exists():
        raise SystemExit(f"{LAZ} missing — run tools/export_laz.py first")

    x, y, z, tid, cls = segment()
    ntree = int(tid.max())
    print(f"  {len(x):,} points, {ntree:,} crowns, "
          f"{int((tid > 0).sum()):,} points assigned ({(tid > 0).mean()*100:.1f}%)")

    pts = {"x": x, "y": y, "z": z}
    flat, (ny, nx) = dl.cell_index(pts, B, PIX)
    veg = np.zeros(ny * nx, bool)
    veg[np.unique(flat[tid > 0])] = True

    # which crown owns the cell: the highest point's, since that is the one visible from above
    order = np.argsort(z)
    cell_id = np.zeros(ny * nx, np.int32)
    cell_id[flat[order]] = tid[order]

    # Ground truth from the FULL cloud, never from the segmented file. The segmented LAS holds
    # only the echo-ratio candidates, so taking truth from it scores against a truth restricted
    # to where we had already decided to look -- which reported 0.726 where the honest number is
    # 0.589. A ground truth that has been filtered by the method under test is not one.
    from ._env import check
    full = dl.read(check(), B)
    fflat, _ = dl.cell_index(full, B, PIX)
    fcls = np.asarray(full["classification"])
    gt_cell = np.zeros(ny * nx, bool)
    gt_cell[np.unique(fflat[np.isin(fcls, (3, 4, 5))])] = True
    gt = np.isin(cls, (3, 4, 5))
    tp = float(np.logical_and(veg, gt_cell).sum())
    print(f"\n  vs ASPRS 3/4/5 on the 1 m grid:  IoU "
          f"{tp/max(float(np.logical_or(veg,gt_cell).sum()),1):.3f}  "
          f"precision {tp/max(veg.sum(),1):.2f}  recall {tp/max(gt_cell.sum(),1):.2f}")

    # per-point too, because "all the points of that tree" is a per-point question
    # per point, also against the full cloud: recall must be over EVERY vegetation return,
    # including the ones the candidate filter dropped before lidR ever saw them
    own = tid > 0
    tpp = float(np.logical_and(own, gt).sum())
    nveg_full = float(np.isin(fcls, (3, 4, 5)).sum())
    print(f"  per point:                       IoU "
          f"{tpp/max(float(own.sum()) + nveg_full - tpp, 1):.3f}  "
          f"precision {tpp/max(own.sum(),1):.2f}  recall {tpp/max(nveg_full,1):.2f}")

    sizes = np.bincount(tid[tid > 0])
    if len(sizes) > 1:
        s = np.sort(sizes[1:])
        print(f"  points per crown: median {int(np.median(s)):,}  "
              f"p10 {int(s[len(s)//10]):,}  p90 {int(s[len(s)*9//10]):,}  max {int(s[-1]):,}")

    dest = OUTDIR / "lidr.npz"
    np.savez_compressed(dest, vegetation=veg.reshape(ny, nx),
                        treeid=cell_id.reshape(ny, nx), ntree=ntree)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
