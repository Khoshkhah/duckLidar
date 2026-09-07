# The photo manifest — bringing your own photos

`ducklidar.building3d` never fetches an image. It reads a **folder** that holds
the photos and one `manifest.json` describing them. The Street View tool
(`python -m ducklidar.tools.fetch_streetview`) writes such a folder; so can you.
Any photo set that follows this page works exactly like the fetched ones.

```
my_photos/
  manifest.json
  IMG_0431.jpg
  IMG_0432.jpg
  …
```

## What a photo must tell the library

Only three things matter, and **all three are hints, not measurements**: the
wall's real position in the image is found from the image itself (its eave
against the sky, its corners on vertical lines). A camera off by two metres or
three degrees still works. A camera pointing at the wrong building does not.

1. **Where the camera stood** — `x`, `y` in a projected CRS (metres), and the
   height above ground if it was not a person's eye level.
2. **Where it looked** — `heading` (degrees, 0 = grid north, clockwise) and
   `pitch` (degrees, + = up).
3. **How wide** — `fov`, the horizontal field of view in degrees.

Everything else in the record is optional and only makes the result better:
a date (recent photos are preferred), a name, a note.

## `manifest.json`

A JSON **list**, one object per image, in any order.

| field | type | required | meaning |
|---|---|---|---|
| `file` | string | **yes** | path to the image; absolute, or relative to the manifest |
| `x`, `y` | number | **yes** | camera position in the CRS of the footprint (metres) |
| `heading` | number | **yes** | where the camera looked, degrees, 0 = grid north, clockwise |
| `fov` | number | **yes** | horizontal field of view, degrees |
| `pitch` | number | no | degrees, + = up. Default 0 (level) |
| `cam_z` | number | no | camera height in the CRS's vertical datum. Default: ground + 1.6 m |
| `date` | string | no | `"YYYY-MM"`. Recency breaks ties between photos of one wall |
| `dist_m` | number | no | camera to footprint, metres. Computed if absent |
| `unused` | string | no | a reason. Any photo with this set is skipped |
| `id`, `note` | any | no | carried through untouched, for your own bookkeeping |

Minimal, and enough:

```json
[
  {"file": "IMG_0431.jpg", "x": 490222.6, "y": 5457609.0, "heading": 348, "fov": 68},
  {"file": "IMG_0432.jpg", "x": 490231.2, "y": 5457612.4, "heading": 22,  "fov": 68,
   "pitch": 4, "cam_z": 5.2, "date": "2026-09"}
]
```

## Getting the numbers from your own photos

**A phone already records all three.** Its EXIF carries GPS latitude and
longitude, a compass heading (`GPSImgDirection`), and the focal length plus the
sensor size, which give the field of view:

```
fov = 2 * atan(sensor_width_mm / (2 * focal_length_mm))
```

For a phone shooting at a 26 mm equivalent, `fov` is about 68 degrees; the EXIF
tag `FocalLengthIn35mmFilm` gives it directly as
`2 * atan(36 / (2 * f35))`. Convert the GPS position to the footprint's CRS
(`pyproj`), and you have a manifest.

Two habits make the result much better:

- **Stand where you can see a whole wall**, corner to corner if possible,
  including the eave against the sky. That is what the wall fitting locks onto.
- **Shoot level-ish and note if you did not.** A phone's pitch is in EXIF too
  (`GPSPitch` on some devices, or the accelerometer-derived orientation); if you
  leave `pitch` out it is assumed level, which is right for most street photos.

**If the metadata is missing or wrong**, the pose is still only a hint, and
there are three ways round it:

1. Photogrammetry over your photo set recovers exact camera positions from the
   images alone (COLMAP). Needs overlap between shots.
2. Click the wall's four corners in the dashboard's align panel. That bypasses
   the pose entirely for that wall.
3. Put in a rough position and heading by hand — where you stood on the map,
   which way you faced. Ten metres and ten degrees is close enough to start.

## What the library adds later

The build writes its own fields back into its output (never into your file):
the fitted camera pose and its agreement score, whether a photo turned out to
be an indoor panorama, and which walls each photo painted. Your manifest is
read-only input.

## Masks and elements

Textured walls also need per-photo masks, which
`python -m ducklidar.tools.segment` produces from the same folder:

```
my_photos/
  manifest.json
  masks/NN.png            occluders — cars, people, trees, sky (255 = do not use)
  masks/NN_building.png   building pixels
  elements/NN.json        windows, doors, awnings detected in that photo
  elements/NN.png         their instance-id map
```

`NN` is the photo's **index in the manifest**, zero-padded to two digits. Supply
them yourself if you have a better segmenter; the library only reads the files.
Without masks the wall fitting still runs, but cars and trees end up painted on
the wall — which is what they are for.
