"""Map facts from a duckOverture extract instead of a rate-limited API.

Overture republishes OSM ways with their source tags, and duckOverture
(../duckOverture) delivers them as one local ``.duckdb`` per area. Asking
that file is the difference between eight tile runs that each court an
Overpass 429 and zero network calls — the facts are already on disk.

    ways = dl.pier_ways("green_window.duckdb", dl.box(490289, 5457557, 250))

Records match what the study tooling expects of ``osm_piers.json``: one dict
per way — id, name, man_made, area, floating, coords in the study CRS.
"""
import json

__all__ = ["pier_ways"]


def pier_ways(db, box, margin=50.0, crs=26910):
    """Pier/breakwater/quay ways from duckOverture extract ``db``, clipped to ``box``.

    ``box`` is (x0, y0, x1, y1) in ``crs``; ``margin`` is added all round,
    mirroring the Overpass fetch this replaces. The way ids are the OSM way
    ids (Overture keeps them in ``sources.record_id``), so downstream joins
    against OSM-derived files keep working.

    Raises ValueError when the extract does not cover the padded box — a
    silently narrower answer would surface later as unsplit docks, far from
    the cause.
    """
    import duckdb
    from pyproj import Transformer

    to_ll = Transformer.from_crs(crs, 4326, always_xy=True)
    to_xy = Transformer.from_crs(4326, crs, always_xy=True)
    lon0, lat0 = to_ll.transform(box[0] - margin, box[1] - margin)
    lon1, lat1 = to_ll.transform(box[2] + margin, box[3] + margin)

    con = duckdb.connect(db, read_only=True)
    try:
        con.execute("LOAD spatial")
        ext = con.execute("select ST_Extent(geom) from main.boundary").fetchone()[0]
        if not (ext["min_x"] <= lon0 and ext["min_y"] <= lat0
                and ext["max_x"] >= lon1 and ext["max_y"] >= lat1):
            raise ValueError(
                f"extract {db} covers lon {ext['min_x']:.4f}..{ext['max_x']:.4f}, "
                f"lat {ext['min_y']:.4f}..{ext['max_y']:.4f} — not the padded box "
                f"({lon0:.4f},{lat0:.4f})..({lon1:.4f},{lat1:.4f})")
        rows = con.execute(
            "select sources[1].record_id, coalesce(names['primary'], ''),"
            "  coalesce(source_tags['man_made'], class),"
            "  coalesce(source_tags['area'], ''),"
            "  coalesce(source_tags['floating'], ''),"
            "  ST_AsGeoJSON(geometry)"
            " from base.infrastructure"
            " where class in ('pier', 'breakwater', 'quay')"
            "  and bbox.xmax >= ? and bbox.xmin <= ?"
            "  and bbox.ymax >= ? and bbox.ymin <= ?"
            " order by 1",
            [lon0, lon1, lat0, lat1]).fetchall()
    finally:
        con.close()

    ways = []
    for rid, name, man_made, area, floating, gj in rows:
        if not rid or not rid.startswith("w"):     # only OSM ways carry a way id
            continue
        g = json.loads(gj)
        ll = (g["coordinates"] if g["type"] == "LineString"
              else g["coordinates"][0] if g["type"] == "Polygon" else [])
        coords = [to_xy.transform(lon, lat) for lon, lat in ll]
        if len(coords) < 2:
            continue
        ways.append({"id": int(rid[1:].split("@")[0]), "name": name,
                     "man_made": man_made, "area": area, "floating": floating,
                     "coords": [[round(x, 2), round(y, 2)] for x, y in coords]})
    return ways
