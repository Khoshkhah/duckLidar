# The cell table — `data/stage5/cells.parquet`

One row per occupied 1 m cell of the working area: the evidence rasters
the labeling rules think on, persisted so it's queryable in DuckDB
between runs; rebuilt when the store changes.

The cell id is the point↔cell join key from `dl.cell_of`:
`cy·nx + cx`, rows counted from the SOUTH. Note: ducklidar's raster
functions are north-up — always convert at the boundary; two measured
flip bugs came from this.

## The columns

| column | type | plain words | how it's computed | what it tells you about trust |
|---|---|---|---|---|
| cell | i64 | which square metre this row is | `dl.cell_of`: `cy·nx + cx`, rows counted from the south | the join key every per-cell fact meets on |
| n | i64 | returns in the cell | counted in the one `GROUP BY cell` pass (task 5.1) | base sampling density — how well the laser saw this square metre |
| min_z | f64 | lowest return | min over the cell's returns (task 5.1) | the cell-minimum surface the cloth drops onto |
| max_z | f64 | the measured DSM per cell — the top surface | highest return, noise excluded (task 5.1) | occupied cells only; a filled DSM over returnless cells would be a model, not a measurement — added only if a consumer needs it |
| split_share | f64 | share of returns whose pulse split | aggregated in the same pass (task 5.1) | foliage character of the cell |
| n_ground | i64 | vendor class-2 (ground) returns | class count in the same pass (task 5.1) | seed evidence for ground |
| n_water | i64 | vendor class-9 (water) returns | class count in the same pass (task 5.1) | seed evidence for water |
| ground_z | f32 | the bare-earth value | the cloth dropped once on the whole area's cell-minimum surface, then fill (task 5.2) | see `ground_measured` and `ground_dist` for whether it's measurement or interpolation |
| ground_measured | bool | the cloth kept actual evidence in this cell | flag from the cloth (task 5.2) | value is measurement, not interpolation |
| ground_dist | f32 | metres to the nearest measured ground cell | fill's distance_to_evidence (task 5.2) | the honest uncertainty proxy — uncertainty grows with it, and it grows under wide roofs |
| ground_fp | f32 | the building's single ground datum, for cells inside a MAPPED building footprint | `dl.building_level`, from the ring of measured ground around the footprint — one number per building, stamped into all its cells (task 5.2) | the laser never sees ground under a roof; the ring is the honest evidence. Null outside mapped footprints; trust fields per building_level's conventions (ring evidence + tilt check), fixed at implementation |

## Proposed, not yet approved

**decided 2026-08-02: adopt** (implementation lands with the next cells-writer revision):

- `dsm_support i16` — returns within 0.5 m of the top; a lone spike owns
  `max_z` when this is 1.
- `dsm_lines u8` — distinct flight lines seeing the top cluster —
  independent confirmation.

---

Every column is either a measurement or carries the story of how it was
made — no pseudo-confidences.
