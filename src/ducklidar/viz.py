"""One self-contained HTML page for looking at the result.

A static figure cannot answer "what happened *there*" — and every question about instance
segmentation is a question about somewhere specific. :func:`compare_channels` builds a page
showing **one channel definition at a time**, with six switchable layers, a legend that says
what every colour means, and pixel-accurate zoom. Every image is inlined as a data URI: no CDN,
no server, nothing to fetch. It drops into a notebook with ``display(HTML(...))`` or onto disk
as one file.

The layers are the pipeline in order — which view won each cell, the product across views, the
boundaries it implies, the instances that survive — so the page is also the explanation.

Missing data is drawn as a **hatch**, never as a colour. On an inferno ramp a black "no data"
cell is indistinguishable from a measured zero, which is exactly the confusion this page exists
to prevent.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np

from .grid import at_extremum, rasterize
from .segment import (COMPASS, echo_ratio, edge_strength, instances,
                      local_boundary, viewed_from)

TEMPLATE = Path(__file__).with_name("_viewer.html")

LAYERS = [
    ("channels", "channels", "Which view won the cell — this is what a channel is."),
    ("product", "product (AND)", "Per-view interval width, multiplied. Bright = every view "
                                 "saw structure."),
    ("boundaries", "boundaries", "The product over the edge threshold."),
    ("local", "boundaries · local rule", "Each cell judged against its own 15 m "
                                         "neighbourhood instead of one global cut."),
    ("vegetation", "vegetation (echo ratio)", "Cells whose pulses split — removed as material "
                                              "before the boundary step. Same for every "
                                              "channel."),
    ("instances", "instances", "Above 2 m, minus the boundaries, connected components ≥30 m²."),
    ("height", "height above ground", "From the ground filter. Same for every channel."),
    ("surface", "surface (DSM)", "Highest return per cell. Same for every channel."),
]
SHARED = {"vegetation", "height", "surface"}


def _png(rgba, colours):
    """Paletted PNG with a reserved fully-transparent index, as a data URI.

    The reserved index has to be *allocated in the palette*, not merely written into the
    ``tRNS`` chunk. Two traps, both silent — PIL emits an all-opaque alpha table rather than
    failing, and missing data then renders as black, which on an inferno ramp cannot be told
    apart from a measured zero:

    * ``quantize(colors=n)`` returns a palette holding only the colours it actually *used*,
      which is often far fewer than `n`. Reserving index `n-1` therefore lands outside it.
      The free slot is at ``len(getpalette()) // 3``, wherever that happens to be.
    * ``optimize=True`` renumbers the palette, and the reserved index moves with it.
    """
    from PIL import Image

    im = Image.fromarray(rgba, "RGBA")
    hole = np.asarray(im.getchannel("A")) < 128
    q = im.convert("RGB").quantize(colors=colours - 1, method=Image.MEDIANCUT)
    pal = q.getpalette()
    blank = min(len(pal) // 3, 255)                  # first index past the colours in use
    arr = np.asarray(q).copy()
    arr[hole] = blank
    out = Image.fromarray(arr, "P")
    out.putpalette(pal[: 3 * blank] + [0, 0, 0])
    buf = io.BytesIO()
    out.save(buf, "PNG", transparency=blank)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _ramp_css(cmap, n=16):
    """The colormap as a CSS gradient, so the legend bar cannot drift from the image."""
    from matplotlib import colormaps

    stops = [colormaps[cmap](i / (n - 1)) for i in range(n)]
    return "linear-gradient(90deg," + ",".join(
        "#%02x%02x%02x" % tuple(int(c * 255) for c in s[:3]) for s in stops) + ")"


def _cat_hex(cmap, i, vmin=-0.5, vmax=9.5):
    from matplotlib import colormaps

    c = colormaps[cmap]((i - vmin) / (vmax - vmin))
    return "#%02x%02x%02x" % tuple(int(v * 255) for v in c[:3])


def _shade(grid, cmap, vmin, vmax, colours=128):
    from matplotlib import colormaps, colors as mcolors

    norm = mcolors.Normalize(vmin, vmax)
    rgba = (colormaps[cmap](norm(np.ma.masked_invalid(grid))) * 255).astype("uint8")
    rgba[..., 3] = np.where(np.isfinite(grid), 255, 0)
    return _png(rgba, colours)


def _labels(lab, seed=0):
    from matplotlib import colormaps

    rng = np.random.default_rng(seed)
    n = int(lab.max())
    out = np.zeros(lab.shape + (4,), dtype="uint8")
    if n:
        hue = rng.permutation(np.linspace(0.03, 0.97, n))
        rgb = (colormaps["nipy_spectral"](hue)[:, :3] * 255).astype("uint8")
        m = lab > 0
        out[m, :3] = rgb[lab[m] - 1]
        out[m, 3] = 255
    return _png(out, 128)


def fetch_basemap(bbox, *, epsg=26910, px=1000):
    """Real imagery of the window as a data URI, or None when offline.

    Fetched once at BUILD time and inlined — the published page has a CSP that blocks live
    tiles, so the map has to travel inside the file like every other layer. Esri's export
    endpoint accepts our projected bbox directly, which sidesteps reprojection entirely.
    """
    import urllib.request

    from PIL import Image

    url = ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/"
           f"export?bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}&bboxSR={epsg}"
           f"&imageSR={epsg}&size={px},{px}&format=jpg&f=image")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
        im = Image.open(io.BytesIO(data))
        im.load()
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


MAP_LEGEND = {"kind": "cat", "title": "Esri World Imagery",
              "items": [{"label": "source: Esri, Maxar, Earthstar Geographics",
                         "hex": "#3a5f4a"}],
              "why": "The real ground, fetched once at build time for the same coordinates as "
                     "every other layer. A different date than the LiDAR — trees and boats "
                     "move; buildings mostly do not."}


def _binary(mask):
    out = np.zeros(mask.shape + (4,), dtype="uint8")
    out[mask] = 255
    return _png(out, 4)


def notebook_view(html, height=1180, path=None):
    """Wrap a page so it actually runs inside a notebook. Returns an IPython display object.

    Two layers of notebook behaviour get in the way, and both fail *silently* — nothing in the
    output, nothing in the console:

    1. **JupyterLab does not execute ``<script>`` in HTML output.** ``display(HTML(page))``
       renders the shell — headings, rail, empty frame — and no image. An ``<iframe srcdoc>``
       is a real document, so its scripts run. It also isolates the page, which matters because
       two viewers would otherwise both declare ``const LAYERS`` in one global scope.
    2. **An untrusted notebook has its iframes stripped entirely**, and a notebook executed by
       ``nbconvert`` is untrusted when opened. So the iframe that fixes (1) can itself vanish.

    Nothing can be rendered inline against (2) — the fix is to run the cell yourself, or to Trust
    the notebook. What this *can* do is fail visibly, so `path` is worth passing: the link to the
    standalone file is plain markup that survives sanitising, and it says what happened.
    """
    from html import escape

    from IPython.display import HTML

    # Absolute, and as *text* rather than a link: a relative href resolves against JupyterLab's
    # own URL rather than the filesystem, and browsers block file:// links from an http page —
    # so a link here would fail in exactly the situation it exists to rescue.
    where = (f' It is written to <code>{escape(str(Path(path).resolve()))}</code> —'
             f' open that in a browser.') if path else ''
    hint = ('<p style="font:13px/1.5 system-ui;margin:0 0 8px;opacity:.75">'
            'Interactive viewer below. Blank? A notebook opened untrusted has its iframes '
            f'stripped — re-run this cell to trust the output.{where}</p>')
    return HTML(hint + f'<iframe srcdoc="{escape(html, quote=True)}" '
                f'style="width:100%;height:{height}px;border:1px solid rgba(127,145,165,.3);'
                f'border-radius:4px"></iframe>')


def default_channels(pts):
    """One channel: the 2-part direction split, labelled with the looks' real azimuths.

    Earlier versions offered flight line and scan angle alongside it for comparison; the
    measured verdicts live in ``docs/instances.md`` and the viewer now focuses on the one
    channel actually used. `parts=2` is the largest division where **every part covers the
    whole map with no returns dropped** — the looks' azimuths decide the merge, not a label.
    """
    az = viewed_from(pts)
    two = viewed_from(pts, parts=2)
    quad = viewed_from(pts, quadrant=True)
    share = {COMPASS[q]: float(np.mean(quad == q)) for q in np.unique(quad)}

    def deg(part):
        vals = sorted({int(round(a)) % 360 for a in np.unique(az[two == part])})
        return ", ".join(f"{v}°" for v in vals)

    return {
        "look": dict(name="direction", by=two,
                     what="each look merged to the nearest of two direction bundles",
                     labels=[f"from {deg(p)}" for p in (0, 1)],
                     note=f"part 1 seen from {deg(0)}; part 2 seen from {deg(1)} — "
                          "every part covers the map, no returns dropped"),
    }, share


def compare_channels(pts, bbox, pix=1.0, *, ground=None, channels=None, score_with=None,
                     edge=0.5):
    """Build the comparison page. Returns HTML as a string.

    Args:
        ground: bare-earth mask, from :func:`~.ground.ground_filter`. Strongly recommended —
            without it the height comes from a rolling minimum, which on coastal ground reads
            the whole land mass as elevated.
        channels: ``{key: {"name", "by", "what", "note"}}``. Defaults to
            :func:`default_channels`.
        score_with: ``{"building": mask, "vegetation": mask}`` over `pts`, used **only** to
            score the result afterwards. Nothing in the method reads a classification field, and
            passing this does not change a single pixel — it fills in the comparison table.
    """
    chans, share = default_channels(pts) if channels is None else (channels, {})
    if channels is not None:
        share = {}

    ref = {}
    for name, m in (score_with or {}).items():
        ref[name] = np.isfinite(rasterize(pts, bbox, pix, mask=np.asarray(m, dtype=bool)))

    layers, stats, strengths, legend = {}, {}, {}, {}
    for key, spec in chans.items():
        by = spec["by"]
        codes = np.asarray(pts[by] if isinstance(by, str) else by)
        strength = edge_strength(pts, bbox, pix, by=by)
        lab, edges, ndsm = instances(pts, bbox, pix, ground=ground, by=by, edge=edge)
        inst = lab > 0
        strengths[key] = strength

        won = at_extremum(pts, bbox, pix,
                          by if isinstance(by, str) else codes.astype("float64"))
        seen = np.unique(won[np.isfinite(won)])
        idx = np.full(won.shape, np.nan)
        for i, c in enumerate(seen):
            idx[won == c] = i

        layers[key] = {
            "local": _binary(local_boundary(strength)),
            "channels": _shade(idx, "tab10", -0.5, 9.5),
            "product": _shade(strength, "inferno", 0, 3),
            "boundaries": _binary(edges),
            "instances": _labels(lab),
        }
        groups = np.unique(codes)
        names = spec.get("labels") or [str(g) for g in groups]

        # step 1 on its own: what each single look contributes, before anything is combined.
        # Without these the page only ever shows the product, and "one look at a time" -- the
        # thing the method is built out of -- is not visible anywhere.
        looks = []
        for i, g in enumerate(groups):
            sub = {k: v[codes == g] for k, v in pts.items()}
            e = rasterize(sub, bbox, pix, interval=True)
            w = np.where(e.method > 0, e.width, np.nan)
            layers[key][f"look{i}"] = _shade(w, "inferno", 0, 3)
            looks.append({"id": f"look{i}", "name": names[i] if i < len(names) else str(g),
                          "cover": float(np.isfinite(w).mean())})
            legend.setdefault(key, {})[f"look{i}"] = {
                "kind": "ramp", "title": f"interval width — {looks[-1]['name']} alone, metres",
                "css": _ramp_css("inferno"), "lo": "0 m — flat",
                "hi": "3 m+ — a step this look saw",
                "why": "One look on its own. It has evidence on "
                       f"{looks[-1]['cover']:.0%} of the window; everywhere else it contributes "
                       "nothing and is excluded from the product, rather than counted as flat."}
        legend.setdefault(key, {}).update({
            "channels": {"kind": "cat", "title": "which look won the cell",
                         "items": [{"label": n, "hex": _cat_hex("tab10", i)}
                                   for i, n in enumerate(names[:len(seen)])],
                         "why": "The highest return in a cell decides its value; this is the "
                                "look that return came from."},
            "product": {"kind": "ramp", "title": "product across looks — metres",
                        "css": _ramp_css("inferno"), "lo": "0 m — flat",
                        "hi": "3 m+ — every look saw a step",
                        "why": "Buildings show as an outline, tree crowns as a filled blob. "
                               "That difference is what separates them."},
            "local": {"kind": "cat",
                      "title": "Kaveh's rule — (S>t1) or (S>t2 and above 50% of neighbours)",
                      "items": [{"label": "boundary", "hex": "#ffffff"},
                                {"label": "not a boundary", "hex": "#101820"}],
                      "why": "Judge every cell against its neighbours and mark it where the "
                             "value deviates either way by 20%, then keep only the components "
                             "holding a globally strong cell. [measured] as a boundary (object "
                             "outlines, 2-cell tolerance) on counts: F1 70% against the global "
                             "rule's 58%, precision 61% vs 49%. The local test alone reaches "
                             "95% recall at 47% precision — the seed removes flat-ground "
                             "noise."},
            "boundaries": {"kind": "cat",
                           "title": f"hysteresis: seeds > {edge} m, grown while > 0.15 m",
                           "items": [{"label": "boundary", "hex": "#ffffff"},
                                     {"label": "not a boundary", "hex": "#101820"}],
                           "why": "A weak cell is boundary only if connected to a strong one "
                                  "(rims are lines, strong at corners, weak along flanks), "
                                  "then watershed lines close every contour: interior plateaus "
                                  "seed a flood of S, and the lines where floods meet separate "
                                  "attached buildings along their shared weak ridge."},
            "instances": {"kind": "cat", "title": "one colour per instance",
                          "items": [{"label": "colours are arbitrary — they only distinguish "
                                              "neighbours", "hex": "#8f4fd0"},
                                    {"label": "not an instance: ground, low, or a boundary",
                                     "hex": "#101820"}],
                          "why": f"Connected components above 2 m, at least 30 m². "
                                 f"{int(lab.max())} of them here."},
            "height": {"kind": "ramp", "title": "height above ground — metres",
                       "css": _ramp_css("viridis"), "lo": "0 m", "hi": "30 m+",
                       "why": "DSM minus the DTM from the cloth-simulation ground filter."},
            "surface": {"kind": "ramp", "title": "surface elevation — metres",
                        "css": _ramp_css("cividis"), "lo": "low", "hi": "high",
                        "why": "The highest return in each cell. No ground filtering."},
        })
        stats[key] = {
            "name": spec["name"], "what": spec["what"], "note": spec["note"],
            "looks": looks,
            "instances": int(lab.max()),
            "channels": int(len(groups)),
            "min_cover": float(min(
                np.isfinite(rasterize({k: v[codes == g] for k, v in pts.items()},
                                      bbox, pix)).mean() for g in groups)),
            "crown": float(np.nanmedian(strength[ref["vegetation"]])) if "vegetation" in ref
                     else None,
            "roof": float(np.nanmedian(strength[ref["building"] & ~edges])) if "building" in ref
                    else None,
            "building": float(np.mean(ref["building"][inst])) if "building" in ref else None,
            "veg_dropped": float(1 - np.sum(ref["vegetation"] & inst)
                                 / max(ref["vegetation"].sum(), 1))
                           if "vegetation" in ref else None,
        }

    dsm = rasterize(pts, bbox, pix)
    _, _, ndsm = instances(pts, bbox, pix, ground=ground)
    lo, hi = float(np.nanmin(dsm)), float(np.nanmax(dsm))
    shared = {"height": _shade(ndsm, "viridis", 0, 30),
              "surface": _shade(dsm, "cividis", lo, hi)}
    if {"return_number", "number_of_returns"} <= set(pts):
        er = echo_ratio(pts, bbox, pix)
        shared["vegetation"] = _binary(np.nan_to_num(er, nan=0.0) > 0.5)
        VEG_LEGEND = {"kind": "cat", "title": "echo ratio > 0.5 — removed as vegetation",
                      "items": [{"label": "vegetation, dropped before the boundary step",
                                 "hex": "#ffffff"},
                                {"label": "kept", "hex": "#101820"}],
                      "why": "The share of a cell's returns whose pulse SPLIT. Foliage is "
                             "porous so a pulse passes through it; a roof stops it dead. "
                             "[measured] crown 74%, roof 6.5%, ground 3% — a ×25 separation, "
                             "against the interval width's ×8.9. This is why the instances "
                             "layer has almost no trees while this one and the boundaries "
                             "layer are full of them."}
        for v in legend.values():
            v["vegetation"] = dict(VEG_LEGEND)
    basemap = fetch_basemap(bbox)
    for v in legend.values():
        v["surface"]["lo"], v["surface"]["hi"] = f"{lo:.0f} m", f"{hi:.0f} m"
    for v in layers.values():
        v.update(shared)

    corr = {}
    keys = list(chans)
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            ok = np.isfinite(strengths[ka]) & np.isfinite(strengths[kb])
            corr[f"{ka}|{kb}"] = float(np.corrcoef(strengths[ka][ok], strengths[kb][ok])[0, 1])

    meta = {"corr": corr, "rows": int(dsm.shape[0]), "cols": int(dsm.shape[1]),
            "returns": int(len(pts["z"])), "pix": pix, "order": keys, "compass": share,
            "map": basemap, "statistic": "interval width"}
    order = [{"id": i, "name": n, "note": d, "shared": i in SHARED} for i, n, d in LAYERS]

    html = TEMPLATE.read_text()
    for token, value in [("/*LAYERS*/", layers), ("/*STATS*/", stats),
                         ("/*META*/", meta), ("/*ORDER*/", order),
                         ("/*LEGEND*/", legend)]:
        html = html.replace(token, json.dumps(value))
    return html
