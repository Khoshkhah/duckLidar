"""Looking at a point cloud, and at what an algorithm did to it.

Static matplotlib views, deliberately — they work in a notebook, in a saved figure and over
SSH, and matplotlib is already here. For turning a cloud around with the mouse, use
CloudCompare (open the file, no code) or PyVista; see `docs/tools.md`.

Three views cover almost every question:

``plan``     from above. Where things are.
``section``  a thin slice from the side. **The one that shows what a raster throws away** —
             ground under a canopy, a bridge deck over a void, the vertical spread inside a
             cell that a DSM reduces to one number.
``compare``  the same section twice, coloured two different ways: classification against a
             segmentation result, say, so what an algorithm decided is visible next to what
             the survey said.
"""
from __future__ import annotations

import numpy as np

#: Colours for the ASPRS classes worth telling apart at a glance. Ground brown, vegetation
#: green, buildings blue — the convention every LiDAR tool uses, so a figure reads without
#: a legend if it has to.
CLASS_COLOURS = {
    0: "#b9bfcc", 1: "#b9bfcc", 2: "#8a6d3b", 3: "#7fbf7f", 4: "#5f9a5f",
    5: "#3f7d3f", 6: "#4c72b0", 7: "#d2601a", 9: "#3fa7d6", 10: "#8b7355",
    11: "#9aa0ad", 17: "#7a5cad", 18: "#d2601a",
}


def _colours(pts, by):
    """`(values, kind)` — how to colour the points. `by` is a field name or an array."""
    from .fields import ASPRS

    v = pts[by] if isinstance(by, str) else np.asarray(by)
    if isinstance(by, str) and by == "classification":
        cols = np.array([CLASS_COLOURS.get(int(c), "#b9bfcc") for c in v])
        labels = {int(c): ASPRS.get(int(c), str(c)) for c in np.unique(v)}
        return cols, ("class", labels)
    if isinstance(by, str) and by in ("red", "green", "blue"):
        rgb = np.stack([pts[k] for k in ("red", "green", "blue")], 1).astype("float64")
        return np.clip(rgb / max(rgb.max(), 1), 0, 1), ("rgb", None)
    return v, ("scalar", None)


def _legend(ax, labels, fontsize=7.5):
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=CLASS_COLOURS.get(c, "#b9bfcc"), label=n)
                       for c, n in sorted(labels.items())],
              loc="upper right", fontsize=fontsize, framealpha=.92)


def plan(pts, by="classification", ax=None, s=0.6, cmap="viridis", title=None, every=1):
    """Points from above, coloured by a field. `every=n` thins for speed."""
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(7, 7), constrained_layout=True)[1]
    x, y = pts["x"][::every], pts["y"][::every]
    c, (kind, labels) = _colours({k: v[::every] for k, v in pts.items()}, by)
    ax.scatter(x, y, s=s, c=c, cmap=cmap if kind == "scalar" else None, linewidths=0)
    if kind == "class":
        _legend(ax, labels)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or f"plan — coloured by {by if isinstance(by, str) else 'array'}",
                 loc="left", fontsize=10, weight="bold")
    return ax


def section(pts, at, width=2.0, axis="y", by="classification", ax=None, s=3, cmap="viridis",
            title=None):
    """A thin slice seen from the side — the view that shows vertical structure.

    `axis="y"` takes a strip of constant y (looking north), `axis="x"` a strip of constant x.
    `width` is how deep the slice is, in metres; keep it thin or everything overlaps.
    """
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(12, 4.5), constrained_layout=True)[1]
    across, along = ("y", "x") if axis == "y" else ("x", "y")
    m = np.abs(pts[across] - at) <= width / 2
    sub = {k: v[m] for k, v in pts.items()}
    c, (kind, labels) = _colours(sub, by)
    ax.scatter(sub[along], sub["z"], s=s, c=c, cmap=cmap if kind == "scalar" else None,
               linewidths=0)
    if kind == "class":
        _legend(ax, labels)
    ax.set_xlabel(f"{along} (m)"); ax.set_ylabel("z (m)")
    ax.grid(alpha=.15)
    ax.set_title(title or f"section at {across}={at:,.0f}, {width:g} m deep "
                          f"— {m.sum():,} returns", loc="left", fontsize=10, weight="bold")
    return ax


def compare(pts, at, by=("classification", None), width=2.0, axis="y", labels=("", ""),
            figsize=(13, 8), **kw):
    """The same slice twice, coloured two ways — what the survey said vs what a run decided.

    The point of it: a segmentation is only believable next to the data it came from.
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True,
                            constrained_layout=True)
    for a, b, lab in zip(axs, by, labels):
        section(pts, at, width=width, axis=axis, by=b, ax=a,
                title=lab or None, **kw)
    return fig, axs
