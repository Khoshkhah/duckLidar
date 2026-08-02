"""Getting points out of a file, without reading the file.

The central claim of this package: **a point cloud is a thing you query, not a thing you
load**. A `.las` inside a `.zip` is the opposite of that — deflate has to be inflated from
byte zero and LAS carries no spatial index, so a 500 m window costs a full parse of every
point in the tile (measured: 9.7 s and a 3.1 GB peak, to produce 481 MB).

So :func:`read` prefers, in order:

1. a **Parquet** sidecar — bbox pushdown against row-group statistics, plus column pruning
   (0.9 s and 0.1 s respectively for the same window)
2. a **COPC** file or URL — its octree answers a bounding box directly, over HTTP if need be
3. the raw LAS/LAZ/zip — streamed in chunks and filtered per chunk, so the file is never
   resident whole even in the fallback

:func:`to_parquet` builds the sidecar once. The spatial sort it does is not optional: LAS
points are in acquisition order, so without it every row group spans the whole tile, every
row group's x/y statistics overlap any query, and pushdown prunes nothing at all.
"""
from __future__ import annotations

import pathlib
import zipfile

import numpy as np

#: Fields worth carrying beyond x, y, z and classification. `number_of_returns` is the best
#: canopy proxy when a survey has no vegetation class; `intensity` and `scan_angle` explain
#: each other; red/green/blue exist only in point format 7+.
EXTRAS = ("intensity", "return_number", "number_of_returns", "scan_angle",
          "gps_time", "point_source_id", "red", "green", "blue")

#: ASPRS classes that mean "do not use this point". Dropping them is not cosmetic: on the
#: Vancouver 2022 tile it moves the floor from -76.4 m to -3.8 m.
NOISE = (7, 18)

#: Cell size points are sorted into before writing Parquet. Small enough that a row group
#: covers a compact patch, large enough that the sort key is cheap.
SORT_CELL_M = 50.0
ROW_GROUP = 500_000


def box(cx, cy, half):
    """`(minx, miny, maxx, maxy)` for a square window — the shape every function here takes."""
    return (cx - half, cy - half, cx + half, cy + half)


def _opener(src):
    """A callable returning an open laspy reader for a path, a zip, or a URL.

    The zip member is *streamed*: `ZipFile.open` inflates lazily, where `ZipFile.read` would
    materialise the whole 1.5 GB LAS in memory before laspy sees a byte of it.
    """
    import laspy

    src = str(src)
    if src.lower().endswith(".zip"):
        zf = zipfile.ZipFile(src)
        inner = [m for m in zf.namelist() if m.lower().endswith((".las", ".laz"))]
        if not inner:
            raise ValueError(f"no .las/.laz inside {src}: {zf.namelist()[:5]}")
        return lambda: laspy.open(zf.open(inner[0]))
    return lambda: laspy.open(src)


def parquet_path(src):
    """Where :func:`to_parquet` puts the sidecar for `src`."""
    return pathlib.Path(str(pathlib.Path(src).with_suffix("")) + ".parquet")


def _columns(chunk, fields):
    have = set(chunk.point_format.dimension_names)
    d = {"x": np.asarray(chunk.x), "y": np.asarray(chunk.y), "z": np.asarray(chunk.z),
         "classification": np.asarray(chunk.classification)}
    d.update({k: np.asarray(chunk[k]) for k in fields if k in have})
    return d


def _inside(d, bbox, drop_noise):
    keep = ((d["x"] >= bbox[0]) & (d["x"] <= bbox[2])
            & (d["y"] >= bbox[1]) & (d["y"] <= bbox[3]))
    if drop_noise:
        keep &= ~np.isin(d["classification"], NOISE)
    return keep


def to_parquet(src, dest=None, *, fields=EXTRAS, chunk=2_000_000, overwrite=False):
    """Convert a LAS/LAZ/zip to a spatially sorted Parquet sidecar. Returns its path.

    Written row group by row group, so a sorted copy of the whole tile never exists at once.
    Idempotent: returns immediately if the sidecar is already there.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    dest = pathlib.Path(dest) if dest else parquet_path(src)
    if dest.exists() and not overwrite:
        return dest

    with _opener(src)() as fh:
        n = fh.header.point_count
        cols, at = {}, 0
        for ch in fh.chunk_iterator(chunk):
            part = _columns(ch, fields)
            if not cols:
                cols = {k: np.empty(n, dtype=v.dtype) for k, v in part.items()}
            m = len(part["x"])
            for k, v in part.items():
                cols[k][at:at + m] = v
            at += m
    cols = {k: v[:at] for k, v in cols.items()}

    # The sort is what makes row-group pruning possible at all — see the module docstring.
    key = (((cols["y"] // SORT_CELL_M).astype("int64") << 24)
           + (cols["x"] // SORT_CELL_M).astype("int64"))
    order = np.argsort(key, kind="stable")
    del key

    schema = pa.schema([(k, pa.from_numpy_dtype(v.dtype)) for k, v in cols.items()])
    tmp = dest.with_suffix(dest.suffix + ".part")
    with pq.ParquetWriter(tmp, schema, compression="zstd") as w:
        for i in range(0, at, ROW_GROUP):
            idx = order[i:i + ROW_GROUP]
            w.write_table(pa.table({k: v[idx] for k, v in cols.items()}, schema=schema))
    tmp.rename(dest)                       # never leave a half-written sidecar behind
    return dest


def read(src, bbox, *, fields=EXTRAS, drop_noise=True, parquet=True):
    """Every return inside `bbox`, as a dict of numpy arrays.

    Columnar because that is what the questions are: "all z where class is 5", never
    "everything about point 12,345". A column is contiguous, so numpy walks it in one C loop
    and every fetched cache line is useful. Measured against the alternatives on 1 M points,
    same query: columnar 1.5 ms / 26 MB, numpy structured array 3.3 ms / 26 MB — identical
    memory, and the gap is cache locality alone — Python rows 32 ms / 343 MB.
    """
    import laspy

    src = str(src)
    want = ["x", "y", "z", "classification", *fields]

    cache = parquet_path(src)
    if parquet and not src.startswith("http") and cache.exists():
        import pyarrow.parquet as pq

        have = set(pq.ParquetFile(cache).schema.names)
        tb = pq.read_table(cache, columns=[c for c in want if c in have],
                           filters=[("x", ">=", bbox[0]), ("x", "<=", bbox[2]),
                                    ("y", ">=", bbox[1]), ("y", "<=", bbox[3])])
        d = {c: tb[c].to_numpy(zero_copy_only=False) for c in tb.column_names}
        keep = _inside(d, bbox, drop_noise)          # row groups overlap the edges
        return {k: v[keep] for k, v in d.items()}

    if src.startswith("http"):
        from laspy.copc import Bounds

        with laspy.CopcReader.open(src) as r:
            p = r.query(Bounds(mins=np.array(bbox[:2]), maxs=np.array(bbox[2:])))
        d = _columns(p, fields)
        keep = _inside(d, bbox, drop_noise)
        return {k: v[keep] for k, v in d.items()}

    parts = []
    with _opener(src)() as fh:
        for ch in fh.chunk_iterator(2_000_000):
            d = _columns(ch, fields)
            keep = _inside(d, bbox, drop_noise)
            if keep.any():
                parts.append({k: v[keep] for k, v in d.items()})
    if not parts:
        with _opener(src)() as fh:
            empty = _columns(next(fh.chunk_iterator(1)), fields)
        return {k: v[:0] for k, v in empty.items()}
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
