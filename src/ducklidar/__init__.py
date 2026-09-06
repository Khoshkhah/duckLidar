"""ducklidar — query airborne LiDAR instead of loading it.

    import ducklidar as dl

    dl.describe("tile.zip")                     # what is actually in this survey
    dl.to_parquet("tile.zip")                   # once: a queryable sidecar
    pts = dl.read("tile.zip", dl.box(490289, 5457557, 250))
    g = dl.surfaces(pts, bbox)                  # dsm / dtm / canopy / roofs
    dl.void_report(g["dtm"])                    # how much of that is real

Three ideas, in order of how much they matter:

1. **Query, do not load.** A `.las` in a `.zip` supports no random access at all. Convert
   once to a spatially sorted Parquet and a window costs 0.9 s instead of 9.7 s, with column
   pruning free on top.
2. **Census before you plan.** Which classes a survey carries decides what is possible with
   it, and surveys differ completely. :func:`describe` says so in one call.
3. **A grid is a model, not a measurement.** Especially a DTM, which comes out roughly half
   empty because nothing reaches the ground under a roof. :func:`void_report` and
   :func:`distance_to_evidence` keep that visible instead of letting a filled array pass for
   a measured one.

Built while learning the subject — see `docs/fundamentals.md` for the reasoning and the
measurements behind each of these, all taken on real tiles.
"""
from .estimate import METHOD, SURFACE, Estimate
from .fields import (ASPRS, BUILDING, GROUND, MEANING, USES, VEGETATION, census, describe,
                     field_guide, flight_dates)
from .plot import CLASS_COLOURS, compare, plan, section
from .ground import DEFAULTS as GROUND_DEFAULTS, ground_filter, score_against
from .grid import (at_extremum, cell_index, distance_to_evidence, rasterize, shape_for,
                   surfaces, void_report)
from .segment import (COMPASS, WIDTH_FLOOR, angle_rank, boundary, bridge_deck,
                      edge_strength,
                      echo_ratio, hysteresis, instances, local_boundary,
                      look_azimuth,
                      viewed_from)
from .viz import compare_channels, default_channels, notebook_view
from .read import EXTRAS, NOISE, box, parquet_path, read, to_parquet
from .overture import pier_ways
from .graph import euclidean_mst, local_edges, read_mst_parts
from .neighbourhood import balanced_boxes, knn, shape_features
from . import objects  # noqa: F401
from .labeling import (CAT_FOLIAGE, CAT_OTHER, CAT_WALL, DECIDE, SPEC, SYS_DECK,
                       SYS_GROUND, SYS_NONE, SYS_WATER, ColumnSupport, cell_components,
                       cell_evidence, column_support,
                       cell_of, convict_glass, corridor_cells, cube_scatter, decide_labels,
                       flood_bodies, footprint_cells, footprint_ids, forest_graph, glass_geodesic,
                       glass_geodesic_blocks,
                       ground_cells, knn_csr, map_vetoes, marina_superstructure, path_shares,
                       car_split,
                       raft_split,
                       path_shares_from_pred,
                       support_forest, water_cells)
from .report import stage_report
from . import dem as dem_mod          # for CROSSOVER, which tests and tools cite
from .dem import building_level, dem, fill, footprint, ndsm
from .trajectory import pulses, sensor_track, tracks

__version__ = "0.1.0"

__all__ = [
    "ASPRS", "BUILDING", "Estimate", "METHOD", "SURFACE", "EXTRAS", "GROUND", "GROUND_DEFAULTS", "MEANING", "NOISE", "USES", "VEGETATION",
    "CLASS_COLOURS", "WIDTH_FLOOR", "at_extremum", "box", "cell_index", "compare", "census", "describe", "distance_to_evidence", "flight_dates", "ground_filter",
    "field_guide", "parquet_path", "plan", "rasterize", "read", "section", "score_against", "shape_for", "surfaces", "to_parquet",
    "building_level", "footprint", "compare_channels", "default_channels", "notebook_view",
    "edge_strength", "instances",
    "balanced_boxes", "cell_components", "cell_evidence", "cell_of", "forest_graph", "knn_csr", "support_forest", "euclidean_mst", "ground_cells", "knn", "local_edges", "look_azimuth", "pier_ways", "read_mst_parts", "shape_features", "stage_report", "void_report",
    "CAT_FOLIAGE", "CAT_OTHER", "CAT_WALL", "SYS_DECK", "SYS_GROUND", "SYS_NONE", "SYS_WATER",
    "ColumnSupport", "column_support",
    "cube_scatter", "knn_csr", "path_shares", "path_shares_from_pred", "support_forest",
    "car_split", "footprint_cells", "footprint_ids", "map_vetoes",
    "marina_superstructure", "raft_split",
]
