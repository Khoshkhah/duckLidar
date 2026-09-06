"""Write the study box out as a LAZ, so tools that only speak files can read it.

Everything else in this repo reads the tile through ``dl.read`` and keeps the cloud in numpy.
PotreeConverter, CloudCompare and QGIS all want a path instead, and the tile itself is an
800 MB ``.zip`` with a ``.las`` inside that none of them will open.

The bridge deck is relabelled to ASPRS 17 on the way out, since this survey classifies no
bridges and ``dl.bridge_deck`` is the only thing that knows otherwise. That makes the class
usable in any viewer, not just ours.

    python tools/export_laz.py [output path]        # the 500 m box, ~10.7M points
    python tools/export_laz.py --tile [output]      # the whole tile, 42.7M points

Set LIDAR_TILE to use a different file.
"""
import sys

import numpy as np

import ducklidar as dl
from ._env import check, out, STUDY_BOX

argv = [a for a in sys.argv if a != "--tile"]
WHOLE = "--tile" in sys.argv
TILE = check()
OUT = out("granville_tile.laz" if WHOLE else "granville.laz", argv)
B = None if WHOLE else dl.box(*STUDY_BOX)


def main():
    import laspy

    pts = dl.read(TILE, B) if B is not None else dl.read(TILE)
    n = len(pts["x"])
    cls = np.asarray(pts["classification"])
    if B is not None:
        deck = dl.bridge_deck(pts, B, 1.0)
        cls = np.where(deck, 17, cls)
        print(f"{n:,} points, {int(deck.sum()):,} relabelled bridge deck")
    else:
        print(f"{n:,} points (whole tile — no deck relabel, that needs a bbox)")

    hdr = laspy.LasHeader(point_format=7, version="1.4")
    # 1 mm scale like the source; offsets at the data's own origin so the int32 range is not
    # wasted on the 5,457,000 m northing.
    hdr.scales = [0.001, 0.001, 0.001]
    hdr.offsets = [np.floor(np.asarray(pts[k]).min()) for k in "xyz"]
    las = laspy.LasData(hdr)
    for k in "xyz":
        setattr(las, k, np.asarray(pts[k]))
    las.classification = cls.astype(np.uint8)
    las.intensity = np.asarray(pts["intensity"])
    las.return_number = np.asarray(pts["return_number"])
    las.number_of_returns = np.asarray(pts["number_of_returns"])
    las.point_source_id = np.asarray(pts["point_source_id"])
    las.gps_time = np.asarray(pts["gps_time"])
    for b in ("red", "green", "blue"):
        setattr(las, b, np.asarray(pts[b]))

    las.write(str(OUT))
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
