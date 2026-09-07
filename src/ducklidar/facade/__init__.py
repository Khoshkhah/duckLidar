"""Facades from street photos — the wall's own image evidence, never the camera pose.

A wall (a planar face group of the building's solid) is textured in five steps, each a module
here, each usable alone:

    quad      where the wall really is IN a photo: the prior quad from the pose is only a search
              band; the eave snaps to the building mask's roofline, the sides to LSD verticals
    rectify   that quad -> a metric strip at TEX_M m/px, with a validity mask
    stitch    several strips of one wall -> one mosaic, aligned by their snapped corners
    fill      occluders (cars, trees) and unseen wall -> the building's own wall colour
    elements  the SAM 3 instances of every photo -> ONE list of windows / doors / awnings in
              metres on the wall, deduplicated and regularised into rows
    compose   material + those elements drawn at their measured places = the wall texture
    relief    the same elements as GEOMETRY: frame bars, projecting sills, mullions from the
              detected panes, glass set back, door jambs, awning fascias and brackets

`texture.texture_wall` runs quad -> rectify -> stitch -> fill for one wall;
`compose.compose` draws it; `relief.relief_prims` gives the glTF primitives.

Measured on Granville Island (shadowCity2's pilot, 3 buildings): element positions land within
about 2 % of the wall where a photo sees it frontally; joinery sizes came from the crops
themselves (frame 0.145 m, mullion 0.155 m, sill 0.065 m proud 0.075 m, glass 0.035 m back).
A wall no photo sees properly gets the building's material and nothing invented.

The host tells these modules about the building through one object:

    from ducklidar.facade import CTX
    CTX.use(ROOT=repo_root, Z_GROUND=3.65, SOLID=dict(V=V, T=T), occlusion_map=my_occlusion_map)
"""
from . import compose, elements, fill, quad, rectify, relief, stitch, texture
from .texture import CTX, Context, texture_wall, qa_sheet

__all__ = ["CTX", "Context", "compose", "elements", "fill", "qa_sheet", "quad",
           "rectify", "relief", "stitch", "texture", "texture_wall"]
