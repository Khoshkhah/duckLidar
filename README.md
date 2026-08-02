# ducklidar

Query airborne LiDAR instead of loading it — LAS/LAZ/COPC to Parquet,
windows, honest rasters, and scene-reconstruction primitives.

```python
import ducklidar as dl
pts = dl.read("tile.zip", dl.box(490289, 5457557, 250))
g = dl.surfaces(pts, bbox)          # dsm / dtm / canopy — with void_report
from ducklidar import scene         # walls with measured openings, canopies,
                                    # prisms, crown thinning, mesh bridges,
                                    # roofer CityJSONSeq parsing, checked_walls
```

Born in [lidar-learning](../lidar-learning) — the lab where every function
here was prototyped, measured, and graduated (its `docs/fundamentals.md`
carries the reasoning and the measurements). The lab remains the proving
ground; only what survives its verdicts lands here, with synthetic-tile
tests that run in seconds on a machine with no data.

```bash
pip install -e .            # deps: numpy, laspy[lazrs], pyarrow
pytest -q                   # extras: grid/ground/boundary/viz/scene in pyproject
```
