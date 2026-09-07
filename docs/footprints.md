# Three outlines for one building

The dashboard's **Plan** panel draws three outlines over the survey's returns seen from above.

| colour | name | where it comes from | what it is for |
|---|---|---|---|
| **red** | Overture footprint | the `buildings.building` geometry of the duckOverture extract, itself from OSM, Microsoft or the City of Vancouver | **the plan the 3-D model is built on** — always, exactly. Walls stand on its edges, the roof is clipped to it, tiers are cut inside it. |
| **blue** | survey outline | `objects.survey_outline`: the building's returns within 1.5 m of the red footprint and outside every neighbour's footprint, traced at 0.5 m by `scene.trace_footprint` | what the 2022 LiDAR says the building's extent is. Not used for geometry. |
| **green** | corrected footprint | `objects.survey_footprint`: the red footprint with each of its edges moved sideways onto the returns' boundary (method below) | a second opinion on the footprint: the map's edges and corners, at the positions the survey measures. Not used for geometry. |

## How the green footprint is made

The idea: the map knows the building's *shape* (its edges, their bearings, its corners); the survey knows *where* the building is. So every edge of the Overture footprint is kept and slid sideways to where the returns end beside it.

1. **The returns' boundary.** The blue outline — `scene.trace_footprint` at 0.5 m over the building's returns within 1.5 m of the red footprint and outside every neighbour's footprint — is densified to a point every 0.25 m.
2. **One offset per edge.** For each edge of the red footprint, take the boundary points that project onto the edge (up to 1 m past either end) and lie within 2.5 m of its line, on either side. The median of their signed distances from the line is the edge's offset. An edge with fewer than 8 such points keeps the map's position.
3. **Move the edge.** The edge's line is shifted by its offset, parallel to itself — its bearing is the map's, its position the survey's.
4. **Re-form the corners.** Each new corner is where two consecutive shifted lines meet. Where two consecutive edges are within 10° of parallel (a zigzag of short edges that all moved onto one straight wall), the corner is the moved edge's own end point instead, so the intersection cannot run away.
5. **Accept or keep.** The result stands only if it is a valid polygon between half and twice the red area; otherwise the red footprint is kept unchanged.

What this can and cannot do. It corrects a shifted footprint (gers_0362696a, gers_00ba2bb9: two metres), an undersized one (gers_06098c52) and a mis-drawn edge (gers_0112d1b3's north side). It cannot add a part the map does not have (gers_058f8092 is a 16 m² footprint inside a building three times that size — the missing part belongs to the neighbouring footprints), and it cannot remove a part the survey does not see (gers_0b9fca8a's empty east half stays, because its edges have no returns to move to). Both of those need the map fixed upstream.

The panel's header gives the two areas and their overlap (intersection over union). An IoU near 100 % means map and survey agree; the buildings where it is low are the ones to check against Google Earth — `python -m ducklidar.tools.map` writes a map page with every footprint, its id, and a link that opens the spot in Google Earth.

## Measured on Granville Island (tile 490000_5457000, 2026-09-07)

Of the 14 sample buildings, on 7 the Overture footprint disagrees with the survey by more than a metre somewhere:

- gers_058f8092 and gers_060510c2 are small rectangles inside buildings about three times their size — the rest of each building lies in neighbouring footprints, so the model is that footprint's share.
- gers_00ba2bb9 and gers_0362696a sit about two metres off the returns.
- gers_06098c52 and gers_08133905 are undersized on one side.
- gers_0dccf0d5 zigzags along an inner edge where the returns show a straight wall.

## Why the model still follows red

A model whose footprint does not fit the map's is of no use to a map (Kaveh, 2026-09-07). So the survey decides only what stands on the footprint — the roof planes, the tiers, the heights. The green outline is there so a wrong red one can be seen, and fixed upstream, not silently replaced.

The three outlines and the numbers are written into each building's `out/<id>/meta.json` as `footprint` (red), `survey_outline` (blue), `survey_footprint` (green), `footprint_area`, `survey_footprint_area`, `footprint_iou` (overlap of red and green).
