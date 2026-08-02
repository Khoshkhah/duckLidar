"""A result is a value **and** how much to trust it — never a bare number.

Every cell of a gridded surface is a summary of returns that disagree, and the disagreement
is not small where it matters. Measured on one urban window at 1 m:

* the DSM is `max` of the returns in a cell, and how far that extremum sits above the bulk
  (`max` − P90) has a median of **7.4 cm** but a p99 of **6.05 m**
* where two flight lines both saw a cell, they disagree by a median of **6.8 cm** and a p99
  of **8.48 m** — and on edge cells, by more than 2 m fully **15%** of the time

A single number hides all of that. :class:`Estimate` carries it instead, so a 0.02 m cell and
a 19 m cell cannot be mistaken for each other downstream.

**Two sources of evidence, and the better one is used per cell:**

``passes``
    The spread of the statistic across flight lines that independently saw the cell. This is
    a real repeatability measurement rather than a modelling assumption, and it is the one to
    prefer. It needs two passes over the cell — 79% of the grid, in the tile measured.

``spread``
    The gap between the statistic and a high quantile of the same cell's returns: how much
    the answer rests on a single point. Always available, and it flags 83% of the cells the
    between-pass test flags at 0.25 m, but it is a proxy, not a measurement.

``none``
    One return, or one pass. The interval is zero-width and that is a statement of ignorance,
    not of agreement — check ``n`` before believing a tight band.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: How the interval for a cell was arrived at. Kept per cell because it varies across a grid.
METHOD = {0: "none", 1: "spread", 2: "passes"}

#: What the passes collectively imply is in a cell — see :meth:`Estimate.surface_kind`.
SURFACE = {0: "unknown", 1: "flat", 2: "clipped", 3: "stepped", 4: "undersampled"}


@dataclass
class Estimate:
    """A grid of values with a per-cell interval and the evidence behind it.

    ``value`` is the statistic. ``lo``/``hi`` bracket it — never a symmetric error bar,
    because these distributions are not symmetric: a `max` can be dragged up by one spurious
    return but can only ever be too *low* by having missed something taller.
    """

    value: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    n: np.ndarray                     # returns behind each cell
    method: np.ndarray                # 0 none, 1 spread, 2 passes
    bulk_gap: np.ndarray = None       # how far apart the passes' MEDIANS are
    balance: np.ndarray = None        # thinnest pass / thickest, by return count

    @property
    def width(self):
        """`hi - lo` per cell — the number to threshold on when deciding what to trust."""
        return self.hi - self.lo

    @property
    def shape(self):
        return self.value.shape

    def trusted(self, max_width=0.5, min_returns=1):
        """Mask of cells whose interval is tight enough and backed by enough returns."""
        return (np.isfinite(self.value) & (self.width <= max_width)
                & (self.n >= min_returns))

    def surface_kind(self, bulk_m=0.5, thin=0.2):
        """Classify each cell into what the passes collectively imply is there.

        ``width`` alone cannot tell a flat roof with an aerial on it from the edge of a roof,
        because both make the per-pass *maxima* disagree. Comparing the per-pass **medians**
        can: on one surface they agree to ~0.10 m however far apart the tops are.

        Returns an int array — 0 unknown (one pass or none), 1 one surface, 2 one surface
        with something clipped above it, 3 two surfaces (a real step), 4 undersampled (one pass
        barely sampled the cell, so its disagreement is not evidence of structure).
        """
        out = np.zeros(self.value.shape, dtype="uint8")
        if self.bulk_gap is None:
            return out
        seen = np.isfinite(self.bulk_gap)
        under = seen & np.isfinite(self.balance) & (self.balance < thin)
        stepped = seen & ~under & (self.bulk_gap > bulk_m)
        clipped = seen & ~under & ~stepped & (self.width > bulk_m)
        flat = seen & ~under & ~stepped & ~clipped
        out[flat], out[clipped], out[stepped], out[under] = 1, 2, 3, 4
        return out

    def summary(self):
        """One-line-per-fact dict, for printing next to a grid."""
        ok = np.isfinite(self.value)
        w = self.width[ok]
        out = {"cells": int(ok.sum()), "filled": float(ok.mean())}
        if w.size:
            out |= {"width_p50": float(np.percentile(w, 50)),
                    "width_p90": float(np.percentile(w, 90)),
                    "width_p99": float(np.percentile(w, 99)),
                    "within_0.5m": float(np.mean(w <= 0.5))}
        for code, name in METHOD.items():
            out[f"by_{name}"] = float(np.mean(self.method[ok] == code)) if ok.any() else 0.0
        if self.bulk_gap is not None and ok.any():
            k = self.surface_kind()[ok]
            for code, name in SURFACE.items():
                out[f"is_{name}"] = float(np.mean(k == code))
        return out

    def __repr__(self):
        s = self.summary()
        if "width_p50" not in s:
            return f"<Estimate {self.shape} empty>"
        return (f"<Estimate {self.shape} {s['filled']:.0%} filled, width p50 "
                f"{s['width_p50']:.2f} m / p90 {s['width_p90']:.2f} m, "
                f"{s['by_passes']:.0%} from repeat passes>")
