"""mesh_graph.npz in minutes: the graduated Borůvka instead of the serial rounds.

Writes exactly what rules/mesh_graph.py writes (bridge_src, bridge_dst,
bridge_len, n, k, cap — window-order indices), but through
:func:`ducklidar.scene.mesh_bridges`: every island merges along its
shortest outgoing edge each round (O(log islands) rounds), where the lab
tool merged one island per round and cost 1–3 hours per large window
(measured 2026-08-02: 819 rounds at ~5–10 s each). Same (k, cap, k_search),
same bridge semantics; the MST may pick different equal-weight ties, which
is why the labels gate runs after any switch to this path.

    python -m ducklidar.rules.mesh_fast
"""
import numpy as np

import ducklidar as dl

from ._env import OUTDIR, check
from .pipeline import B

K_LOCAL, CAP, K_SEARCH = 10, 2.5, 16


def main():
    from ducklidar.scene import mesh_bridges

    pts = dl.read(check(), B, fields=())
    n = len(pts["x"])
    src, dst, blen = mesh_bridges(pts["x"], pts["y"], pts["z"],
                                  k=K_LOCAL, cap=CAP, k_search=K_SEARCH)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTDIR / "mesh_graph.npz",
                        bridge_src=src, bridge_dst=dst, bridge_len=blen,
                        n=n, k=K_LOCAL, cap=CAP)
    print(f"mesh_fast: {n:,} returns, {len(src):,} bridges", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
