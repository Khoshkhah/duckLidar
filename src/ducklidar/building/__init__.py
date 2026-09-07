"""A 3-D building from a solid, its LiDAR returns and street photos.

Four modules, each usable alone, in the order a building is built:

    points     the building's own returns from a footprint + a point store, and its ground
    geometry   the solid -> planar facades with a metric in-plane basis; skylight boxes from
               the raised returns; the floor slab
    camera     each photo's real pose (silhouette vs SAM 3's building mask), which walls it
               sees, and the solid's self-occlusion of a wall
    walls      one wall's texture through `ducklidar.facade`, the building's wall material,
               and the roof texture from the survey's own colour

They take their host context from `ducklidar.facade.CTX` — the photo folder, the masks, the
manifest, the solid, `Z_GROUND`/`CAM_H`, and `camera.occlusion_map`:

    from ducklidar import facade
    from ducklidar.building import camera, geometry, walls
    V, T = geometry.load_solid(cityjson, solids); n = geometry.normals(V, T)
    SOLID = dict(V=V, T=T)
    facade.CTX.use(ROOT=root, Z_GROUND=z, CAM_H=2.3, SOLID=SOLID, occlusion_map=camera.occlusion_map,
                   PHOTOS=photos, MASKS=photos / "masks", MANIFEST=man, CLICKS=out / "clicks.json")

`ducklidar.glb.write_glb` turns the resulting primitives into a file. `dl.building3d` will
compose all of this into one call.
"""
from . import camera, geometry, points, walls
from .camera import (cam_axes, chosen_pose, luminance, occlusion_map, pick_photo, project,
                     rank_photos, refine_pose, sharpness, silhouette, usable, visible_fraction)
from .points import building_points, footprint_ring, load_npz
from .geometry import (box_prim, facade_corners, facades, floor_prim, load_solid, normals,
                       outer_edges, skylight_boxes)
from .walls import (composite_facade, compose_walls, facade_cands, fill_rows, flat_facade,
                    lab_median, roof_texture, tex_size, warp_clicked, warp_facade)

__all__ = ["camera", "geometry", "points", "walls",
           "box_prim", "building_points", "footprint_ring", "load_npz", "cam_axes", "chosen_pose", "composite_facade", "compose_walls",
           "facade_cands", "facade_corners", "facades", "fill_rows", "flat_facade",
           "floor_prim", "lab_median", "load_solid", "luminance", "normals", "occlusion_map",
           "outer_edges", "pick_photo", "project", "rank_photos", "refine_pose",
           "roof_texture", "sharpness", "silhouette", "skylight_boxes", "tex_size", "usable",
           "visible_fraction", "warp_clicked", "warp_facade"]
