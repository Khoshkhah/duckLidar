"""build_tile: the tile's extent from Parquet statistics, and only footprints WHOLLY inside it."""
import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("pyproj")

from ducklidar.tools.build_tile import footprints, tile_extent


def test_only_footprints_wholly_inside_the_tile_are_built(tmp_path):
    from pyproj import Transformer
    to_ll = Transformer.from_crs(26910, 4326, always_xy=True)

    def wkt(x0, y0, x1, y1):
        pts = [to_ll.transform(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]
        return "POLYGON((" + ", ".join(f"{lon} {lat}" for lon, lat in pts) + "))"

    db = tmp_path / "fp.duckdb"
    con = duckdb.connect(str(db))
    con.execute("INSTALL spatial; LOAD spatial; CREATE SCHEMA buildings")
    con.execute("CREATE TABLE buildings.building AS SELECT * FROM (VALUES "
                f"('inside', ST_GeomFromText('{wkt(490100, 5457100, 490120, 5457120)}')),"
                f"('straddles', ST_GeomFromText('{wkt(490990, 5457100, 491010, 5457120)}'))"
                ") t(id, geometry)")
    con.close()

    got = footprints(db, (490000, 5457000, 491000, 5458000))
    assert [i for i, _ in got] == ["inside"]
    ring = got[0][1]
    assert len(ring) == 5 and abs(ring[0][0] - 490100) < 0.01 and abs(ring[0][1] - 5457100) < 0.01


def test_tile_extent_comes_from_the_statistics(tmp_path):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    x = np.array([490000.0, 490999.9]); y = np.array([5457000.0, 5457999.9])
    pq.write_table(pa.table({"x": x, "y": y, "z": x * 0}), tmp_path / "t.parquet")
    assert tile_extent(tmp_path / "t.parquet") == (490000.0, 5457000.0, 490999.9, 5457999.9)
