"""Scene reconstruction primitives that survived the learning.

Promoted from the study-box tooling 2026-08-01 (see lidar-learning
tools/_recon.py history and docs/tools.md verdicts). Everything here is
measured-first: footprints and map tags are canvases and labels, never
geometry sources on their own.
"""
import numpy as np

__all__ = ["canopy_mesh", "checked_walls", "crown_thin", "footprint_prism",
           "mesh_bridges", "read_cityjsonseq", "sat_lift", "trace_footprint",
           "wall_mesh", "wall_occupancy", "wall_report"]


def checked_walls(soup, cols, ring, x, y, z, z0, ztop, rgb=None, cell=0.45,
                  reach=0.9, min_coverage=0.25, open_structure=None):
    """The judgment roofer skips: it extrudes every wall; this checks first.

    Kaveh, 2026-08-01: "in roofer add wall without any checking! in this
    method first check for wall and decide should we add wall or not, and
    then decide about windows or holes and which location." Given a solid's
    triangle soup and its footprint, each side's vertical faces stand trial
    against the returns:

    * evidence of a wall, no framed openings → the solid's own faces stay
      (roofer's crisp gable ends are better than any rebuild);
    * evidence of a wall WITH framed openings → the side's faces are
      replaced by full wall surfaces with the door/window rectangles cut
      out at their detected positions, top following the old faces' profile
      per column;
    * no evidence, on a building measured OPEN → the invented wall is
      removed; only the evidence cells remain, so carports and pavilion
      sides fall open.

    Openness is measured inside the footprint, not at the wall plane —
    because real closed walls carry only 5–45 % wall-plane coverage in this
    survey's grazing geometry, absence there is shadow, not proof. But the
    laser reaching the GROUND under the roof is proof: returns within 2 m
    of ground level inside the inset footprint run 1.1–11 /m² on the box's
    open structures (carports) and median 0.00 /m², P90 0.11, on its 135
    closed roofer buildings [measured 2026-08-01]. Only a building past
    1 return/m² may lose walls; a closed building's shadowed sides keep
    roofer's faces and only framed windows get cut. Pass ``open_structure``
    explicitly when the caller can measure it better — an instance's own
    points usually EXCLUDE the ground returns under its roof (they carry
    the ground class), so measure the gate on the full cloud.

    Roof and floor faces (|normal_z| ≥ 0.3) and vertical faces matching no
    footprint side (interior roof steps) are never touched.

    Returns ``(kept_soup, kept_cols, parts)`` — the surviving original
    triangles with their colours, plus new ``(soup, cols)`` parts for
    rebuilt walls and open-side evidence cells.
    """
    soup = np.asarray(soup)
    tris = soup.reshape(-1, 3, 3)
    tcols = np.asarray(cols).reshape(len(tris), 3, 3)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    vertical = np.abs(nrm[:, 2]) < 0.3
    cen = tris.mean(1)
    P = np.column_stack([np.asarray(x, float), np.asarray(y, float),
                         np.asarray(z, float)])
    if rgb is not None:
        rgb = np.asarray(rgb)
    ring = np.asarray(ring, float)
    rep = {r["side"]: r for r in
           wall_report(ring, x, y, z, z0, ztop, cell=cell, reach=reach,
                       min_coverage=min_coverage)}
    # the openness gate: only a building the laser demonstrably sees UNDER
    # may lose walls; wall-plane absence alone is shadow
    if open_structure is None:
        import shapely
        inner = shapely.Polygon(ring).buffer(-reach)
        open_structure = False
        if not inner.is_empty and inner.area > 1.0:
            im = shapely.contains_xy(inner, P[:, 0], P[:, 1]) \
                & (P[:, 2] > z0 - 0.5) & (P[:, 2] < z0 + 2.0)
            open_structure = im.sum() / inner.area > 1.0
    drop = np.zeros(len(tris), bool)
    parts = []
    open_soup, open_cols = [], []
    for k in range(len(ring)):
        if k not in rep:
            continue
        r = rep[k]
        a, b = np.asarray(r["a"]), np.asarray(r["b"])
        L = r["length"]
        u = (b - a) / L
        nn = np.array([b[1] - a[1], a[0] - b[0]]) / L
        s = (cen[:, :2] - a) @ u
        d = (cen[:, :2] - a) @ nn
        cand = vertical & ~drop & (np.abs(d) < reach) \
            & (np.abs(nrm[:, :2] @ nn) > 0.8) & (s > -0.2) & (s < L + 0.2)
        if not r["has_wall"] and not open_structure:
            continue                       # shadowed wall on a closed
        if not cand.any():                 # building: roofer's faces stand
            if not r["has_wall"]:
                _edge_cells(a, b, P, z0, ztop, rgb, None, cell, reach,
                            open_soup, open_cols)
            continue
        if r["has_wall"] and not (r["doors"] or r["windows"]):
            continue                       # the solid's own faces stand
        drop |= cand
        if not r["has_wall"]:
            _edge_cells(a, b, P, z0, ztop, rgb, None, cell, reach,
                        open_soup, open_cols)
            continue
        # rebuild: per-column top follows the dropped faces' profile
        V = tris[cand]
        vs = (V[..., :2].reshape(-1, 2) - a) @ u
        vz = V[..., 2].reshape(-1)
        zbot = float(vz.min())
        ns = max(int(np.ceil(L / cell)), 1)
        top = np.full(ns, -np.inf)
        for t in V:
            ts = (t[:, :2] - a) @ u
            for e0, e1 in ((0, 1), (1, 2), (2, 0)):
                s0, s1 = ts[e0], ts[e1]
                za, zb = t[e0, 2], t[e1, 2]
                j0 = int(np.clip(min(s0, s1) / cell, 0, ns - 1))
                j1 = int(np.clip(max(s0, s1) / cell, 0, ns - 1))
                for j in range(j0, j1 + 1):
                    sc = (j + 0.5) * cell
                    if abs(s1 - s0) > 1e-9:
                        f = np.clip((sc - s0) / (s1 - s0), 0, 1)
                        top[j] = max(top[j], za + f * (zb - za))
                    else:
                        top[j] = max(top[j], za, zb)
        holes = [(dd["s"] - dd["width"] / 2, dd["s"] + dd["width"] / 2,
                  z0, z0 + dd["height"]) for dd in r["doors"]]
        holes += [(w["s"] - w["width"] / 2, w["s"] + w["width"] / 2,
                   w["z"] - w["height"] / 2, w["z"] + w["height"] / 2)
                  for w in r["windows"]]
        if rgb is not None:
            rel = P[:, :2] - a
            ps = rel @ u
            pd = rel @ nn
            m = (ps > -0.2) & (ps < L + 0.2) & (np.abs(pd) < reach) \
                & (P[:, 2] > z0 + 0.05) & (P[:, 2] < ztop + 0.3)
            side_col = np.median(rgb[m], axis=0) if m.any() \
                else np.array([150.0, 150.0, 150.0])
        else:
            side_col = np.array([150.0, 150.0, 150.0])
        wsoup = []
        j = 0
        while j < ns:                      # runs of equal top → one rect
            if not np.isfinite(top[j]):
                j += 1
                continue
            j1 = j
            while j1 + 1 < ns and np.isfinite(top[j1 + 1]) \
                    and abs(top[j1 + 1] - top[j]) < 0.05:
                j1 += 1
            sa, sb = j * cell, min((j1 + 1) * cell, L)
            hs = [(h[0] - sa, h[1] - sa, h[2], h[3]) for h in holes
                  if h[0] < sb and h[1] > sa]
            for xa, xb, za, zb in _rects_minus(sb - sa, zbot, top[j], hs):
                p0, p1 = a + u * (sa + xa), a + u * (sa + xb)
                wsoup.append(np.array(
                    [[p0[0], p0[1], za], [p1[0], p1[1], za],
                     [p1[0], p1[1], zb], [p0[0], p0[1], za],
                     [p1[0], p1[1], zb], [p0[0], p0[1], zb]]))
            j = j1 + 1
        if wsoup:
            wsoup = np.vstack(wsoup).astype(np.float32)
            parts.append((wsoup, np.tile(np.clip(side_col, 0, 255)
                                         .astype(np.uint8),
                                         (len(wsoup), 1))))
    if open_soup:
        parts.append((np.vstack(open_soup).astype(np.float32),
                      np.clip(np.vstack(open_cols), 0, 255)
                      .astype(np.uint8)))
    keep = ~drop
    return (tris[keep].reshape(-1, 3).astype(np.float32),
            tcols[keep].reshape(-1, 3), parts)


def _rects_minus(L, zlo, zhi, holes):
    """Decompose the rectangle [0,L] x [zlo,zhi] minus hole rects into
    sub-rectangles: cut at every hole edge, keep cells whose centre lies in
    no hole. Holes are (s0, s1, zb, zt)."""
    holes = [(max(h[0], 0.0), min(h[1], L), max(h[2], zlo), min(h[3], zhi))
             for h in holes]
    xs = sorted({0.0, L, *(h[0] for h in holes), *(h[1] for h in holes)})
    out = []
    for xa, xb in zip(xs, xs[1:]):
        if xb - xa < 1e-6:
            continue
        zs = {zlo, zhi}
        for h in holes:
            if h[0] < xb and h[1] > xa:
                zs |= {h[2], h[3]}
        zsl = sorted(z for z in zs if zlo <= z <= zhi)
        cx = (xa + xb) / 2
        for za, zb in zip(zsl, zsl[1:]):
            if zb - za < 1e-6:
                continue
            cz = (za + zb) / 2
            if any(h[0] <= cx <= h[1] and h[2] <= cz <= h[3] for h in holes):
                continue
            out.append((xa, xb, za, zb))
    return out


def wall_mesh(ring, x, y, z, z0, ztop, rgb=None, col=None, cell=0.45,
              reach=0.9, min_coverage=0.25, tops=None):
    """Walls as decisions rendered: full surfaces with their openings cut.

    Kaveh, 2026-08-01, on the occupancy confetti: "for each wall you put
    some random cell there — detect window on a wall and create full wall
    with window." So: ``wall_report`` decides per side; a side that IS a
    wall becomes one solid surface from the ground to ``ztop``, in the
    median colour of its own in-band returns, with every framed opening
    (door, window) subtracted as a real rectangular hole at its detected
    position. A side that is NOT a wall keeps only its evidence cells —
    posts stay posts, open sides stay open, and no surface is invented
    where the report refused to call one.

    ``tops`` closes the wall against a pitched roof (Kaveh, 2026-08-01:
    the flat-topped wall "missed some part" under #54's gable): a dict
    ``{side index: [(s, z), ...]}`` of top-profile breakpoints along the
    side. Where a side's profile rises above ``ztop``, the piece between —
    the pediment under a gable rake — is filled as part of the same wall.

    Returns ``(parts, report)``: parts are ``(soup, cols)`` tuples (solid
    walls first, then the open sides' evidence cells), report is
    ``wall_report``'s verdict so callers can glaze the openings.
    """
    rep = wall_report(ring, x, y, z, z0, ztop, cell=cell, reach=reach,
                      min_coverage=min_coverage)
    P = np.column_stack([np.asarray(x, float), np.asarray(y, float),
                         np.asarray(z, float)])
    if rgb is not None:
        rgb = np.asarray(rgb)
    ring = np.asarray(ring, float)
    parts = []
    open_ring = []
    for r in rep:
        a, b = np.asarray(r["a"]), np.asarray(r["b"])
        if not r["has_wall"]:
            open_ring.append((a, b))
            continue
        L = r["length"]
        u = (b - a) / L
        holes = [(d["s"] - d["width"] / 2, d["s"] + d["width"] / 2,
                  z0, z0 + d["height"]) for d in r["doors"]]
        holes += [(w["s"] - w["width"] / 2, w["s"] + w["width"] / 2,
                   w["z"] - w["height"] / 2, w["z"] + w["height"] / 2)
                  for w in r["windows"]]
        if col is None and rgb is not None:
            e = b - a
            rel = P[:, :2] - a
            s = rel @ u
            d = rel @ np.array([e[1], -e[0]]) / L
            m = (s > -0.2) & (s < L + 0.2) & (np.abs(d) < reach) \
                & (P[:, 2] > z0 + 0.05) & (P[:, 2] < ztop + 0.3)
            side_col = np.median(rgb[m], axis=0) if m.any() \
                else np.array([150, 150, 150])
        else:
            side_col = np.asarray([150, 150, 150] if col is None else col)
        soup = []
        for xa, xb, za, zb in _rects_minus(L, z0, ztop, holes):
            p0, p1 = a + u * xa, a + u * xb
            soup.append(np.array(
                [[p0[0], p0[1], za], [p1[0], p1[1], za], [p1[0], p1[1], zb],
                 [p0[0], p0[1], za], [p1[0], p1[1], zb], [p0[0], p0[1], zb]]))
        if tops and r["side"] in tops:
            # the pediment: fill between the flat ztop and the roof profile
            pts = [(0.0, float(ztop)), (L, float(ztop))]
            pts += [(float(sv), float(hv))
                    for sv, hv in sorted(tops[r["side"]], reverse=True)
                    if hv > ztop + 1e-3]
            if len(pts) > 2:
                import mapbox_earcut
                arr = np.asarray(pts)
                tri = mapbox_earcut.triangulate_float64(
                    arr, np.array([len(arr)], np.uint32)).reshape(-1, 3)
                if len(tri):
                    g = arr[tri.ravel()]
                    p3 = np.column_stack([a[0] + u[0] * g[:, 0],
                                          a[1] + u[1] * g[:, 0], g[:, 1]])
                    soup.append(p3)
        if soup:
            soup = np.vstack(soup).astype(np.float32)
            parts.append((soup, np.tile(np.clip(side_col, 0, 255)
                                        .astype(np.uint8), (len(soup), 1))))
    if open_ring:
        soup, cols = [], []
        for a, b in open_ring:
            _edge_cells(a, b, P, z0, ztop, rgb, col, cell, reach, soup, cols)
        if soup:
            parts.append((np.vstack(soup).astype(np.float32),
                          np.clip(np.vstack(cols), 0, 255).astype(np.uint8)))
    return parts, rep


def wall_report(ring, x, y, z, z0, ztop, cell=0.45, reach=0.9,
                min_coverage=0.25):
    """Decide, per footprint side: wall or no wall — and the openings in it.

    ``wall_occupancy`` renders raw evidence, and raw absence lies twice: an
    open side and a laser-shadowed wall both read as missing cells (Kaveh,
    2026-08-01: the occupancy view "doesn't work truly" as hole detection).
    This report makes the calls instead, on the same evidence grid:

    * **has_wall** — a side is a wall when at least ``min_coverage`` of its
      (along, height) cells carry returns; a post or an open pavilion side
      stays ``False``.
    * **doors** — on a walled side, an empty region that reaches the ground,
      has wall evidence on both sides (jambs) and above (lintel), and is
      door-sized: 0.5–3.0 m wide, 1.5–3.5 m tall. Reported at its centre
      ``s`` metres along the side from vertex ``a``.
    * **windows** — an empty region framed on all four sides (jambs, lintel,
      sill), off the ground, 0.3–4.0 m in each direction. Reported at its
      centre ``(s, z)``.

    Absence with no frame around it is *not* an opening — it is occlusion or
    an open side, and claiming holes there was exactly the false-hole failure
    on shadowed lower walls. Openings smaller than about two cells (~1 m at
    the default) are sealed by the same closing that bridges scan-line gaps,
    so basement windows below that size go unreported rather than guessed.

    What the study box measured about these decisions [2026-08-01]: airborne
    grazing geometry sees walls poorly — most real building sides carry 5–45 %
    coverage, the pavilion's OPEN sides carry 24–44 % (fascia band plus
    furniture within reach), so wall-plane evidence alone cannot separate the
    two; treat ``has_wall`` at the default threshold as "evidence of a
    surface", not proof of enclosure. And zero door-shaped framed openings
    exist in the whole box even with the ground rule relaxed — doors live in
    recesses this survey never saw. The report tells you what was measured;
    it cannot invent what wasn't.

    Returns one dict per side of at least 0.6 m: ``side``, ``a``, ``b``,
    ``length``, ``coverage``, ``has_wall``, ``doors``, ``windows``.
    """
    from scipy import ndimage as ndi

    P = np.column_stack([np.asarray(x, float), np.asarray(y, float),
                         np.asarray(z, float)])
    ring = np.asarray(ring, float)
    report = []
    for k in range(len(ring)):
        a, b = ring[k], ring[(k + 1) % len(ring)]
        e = b - a
        L = float(np.hypot(*e))
        if L < 0.6:
            continue
        u = e / L
        rel = P[:, :2] - a
        s = rel @ u
        d = rel @ np.array([e[1], -e[0]]) / L
        m = (s > -0.2) & (s < L + 0.2) & (np.abs(d) < reach) \
            & (P[:, 2] > z0 + 0.05) & (P[:, 2] < ztop + 0.3)
        ns = int(L / cell) + 1
        nz = max(int((ztop - z0) / cell) + 1, 1)
        side = {"side": k, "a": tuple(a), "b": tuple(b), "length": L,
                "coverage": 0.0, "has_wall": False, "doors": [], "windows": []}
        if m.any():
            si = (s[m] / cell).astype(int).clip(0, ns - 1)
            zi = ((P[m, 2] - z0) / cell).astype(int).clip(0, nz - 1)
            occ = np.zeros((nz, ns), bool)
            occ[zi, si] = True
            occ = ndi.binary_closing(np.pad(occ, 1, mode="edge"),
                                     np.ones((2, 2)))[1:-1, 1:-1]
            side["coverage"] = float(occ.mean())
            if side["coverage"] >= min_coverage:
                side["has_wall"] = True
                emp, ne = ndi.label(~occ)
                for lb in range(1, ne + 1):
                    zz, ss = np.nonzero(emp == lb)
                    z0c, z1c = int(zz.min()), int(zz.max())
                    s0c, s1c = int(ss.min()), int(ss.max())
                    w = (s1c - s0c + 1) * cell
                    h = (z1c - z0c + 1) * cell
                    jambs = occ[z0c:z1c + 1, :s0c].any() \
                        and occ[z0c:z1c + 1, s1c + 1:].any()
                    lintel = occ[z1c + 1:, s0c:s1c + 1].any()
                    sill = occ[:z0c, s0c:s1c + 1].any()
                    sm = (s0c + s1c + 1) / 2 * cell
                    zm = z0 + (z0c + z1c + 1) / 2 * cell
                    if z0c == 0 and jambs and lintel \
                            and 0.5 <= w <= 3.0 and 1.5 <= h <= 3.5:
                        side["doors"].append(
                            {"s": sm, "width": w, "height": h})
                    elif z0c > 0 and jambs and lintel and sill \
                            and 0.3 <= w <= 4.0 and 0.3 <= h <= 4.0:
                        side["windows"].append(
                            {"s": sm, "z": zm, "width": w, "height": h})
        report.append(side)
    return report


def _exact_bridge(P, island):
    """Shortest edge from ``island`` to anywhere outside it, by widening box
    search. Exact: only returns a pair once the best distance is within the
    search radius, so no closer partner can be hiding outside the box."""
    from scipy.spatial import cKDTree
    Pi = P[island]
    lo, hi = Pi.min(0), Pi.max(0)
    inside = np.zeros(len(P), bool)
    inside[island] = True
    r = 10.0
    while True:
        m = (~inside & (P[:, 0] >= lo[0] - r) & (P[:, 0] <= hi[0] + r)
             & (P[:, 1] >= lo[1] - r) & (P[:, 1] <= hi[1] + r))
        cand = np.flatnonzero(m)
        if len(cand):
            d, j = cKDTree(P[cand]).query(Pi, k=1, workers=-1)
            k = int(np.argmin(d))
            if d[k] <= r:
                return int(island[k]), int(cand[j[k]]), float(d[k])
        r *= 2.0


def mesh_bridges(x, y, z, k=10, cap=2.5, k_search=16):
    """The mesh graph's stored half: MST bridges between the kNN islands.

    The kNN(``k``, ``cap``) graph that per-return rules trust is honest but
    disconnected — crowns, buildings, boats are islands, and that is itself a
    signal. The mesh keeps every local edge exactly as those rules know it
    and adds the minimum repair: the MST edges that cross between islands,
    one fewer than the number of islands (Borůvka rounds; each island merges
    along its shortest outgoing edge, which is the minimum-spanning-tree
    guarantee of connectivity). Every bridge is typed by its endpoints and
    carries its length — "one component, but only through a 14 m bridge" is
    the isolation measurement the marina problem was starving for [measured
    2026-08-01, k=10, cap 2.5 m on the study box].

    Only the bridges need storing: the local half is deterministic from
    (``k``, ``cap``) over the same point order, so any consumer rebuilds it
    with one KD-tree query. The cached ``k_search`` neighbours answer for
    small and adjacent islands; a dense island whose nearest are all its own
    (a big blob across open water) falls through to an exact widening-box
    scan, one per stalled island.

    Returns ``(src, dst, length)`` — int32 point indices and float32 metres,
    all empty when the local graph is already one component.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    P = np.column_stack([np.asarray(x, np.float64), np.asarray(y, np.float64),
                         np.asarray(z, np.float64)])
    n = len(P)
    k_search = max(k_search, k)
    tree = cKDTree(P)
    qd = np.empty((n, k_search), np.float32)
    qi = np.empty((n, k_search), np.int32)
    for s in range(0, n, 1_000_000):
        e = min(s + 1_000_000, n)
        d, i = tree.query(P[s:e], k=k_search + 1, workers=-1)
        qd[s:e], qi[s:e] = d[:, 1:], i[:, 1:]

    ok = (qd[:, :k] <= cap).ravel()
    a = np.repeat(np.arange(n, dtype=np.int32), k)[ok]
    b = qi[:, :k].ravel()[ok]
    G = coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(n, n))
    # weak connectivity on the directed adjacency: identical components
    # without the G + G.T symmetrization that OOM-killed tile-scale runs
    ncomp0, cur = connected_components(G, directed=True, connection="weak")
    del G, a, b, ok
    cur = cur.astype(np.int32)

    src, dst, blen = [], [], []
    while True:
        u, lab = np.unique(cur, return_inverse=True)
        m = len(u)
        if m == 1:
            break
        lab = lab.astype(np.int32)
        f = lab[qi] != lab[:, None]
        cand = np.flatnonzero(f.any(1))
        fc = f[cand].argmax(1)
        cd, cq = qd[cand, fc], qi[cand, fc]
        order = np.lexsort((cd, lab[cand]))
        firsts = np.unique(lab[cand][order], return_index=True)[1]
        picks = order[firsts]              # per-island shortest outgoing edge

        par = np.arange(m)

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        added = 0
        for o in picks:
            p, q, dpq = int(cand[o]), int(cq[o]), float(cd[o])
            ra, rb = find(lab[p]), find(lab[q])
            if ra != rb:
                par[ra] = rb
                src.append(p); dst.append(q); blen.append(dpq)
                added += 1
        if added == 0:
            # every remaining island is blind at k_search: exact-scan the
            # smallest one
            sizes = np.bincount(lab, minlength=m)
            c = int(np.argsort(sizes)[0])
            p, q, dpq = _exact_bridge(P, np.flatnonzero(lab == c))
            par[find(lab[p])] = find(lab[q])
            src.append(p); dst.append(q); blen.append(dpq)
        cur = np.array([find(i) for i in range(m)], np.int32)[lab]

    assert len(blen) == ncomp0 - 1, (len(blen), ncomp0)
    return (np.array(src, np.int32), np.array(dst, np.int32),
            np.array(blen, np.float32))


def trace_footprint(x, y, cell=0.5, min_area=10.0):
    """Footprint polygon traced from an instance's own points.

    The mapped footprint is the authority where one exists — on the study box
    OSM mapped 81 of 629 building instances, and the other 548 never got past
    a blocky extrusion until this trace gave each a polygon of its own
    [measured 2026-08-01]: 0.5 m occupancy grid → morphological close, fill,
    open → largest connected region → its contour, simplified at the cell
    size. Returns a shapely Polygon, or ``None`` when nothing coherent of at
    least ``min_area`` m² emerges (10 m² keeps sheds and drops noise blobs).
    """
    from scipy import ndimage as ndi
    from shapely.geometry import Polygon
    from skimage import measure

    x, y = np.asarray(x, float), np.asarray(y, float)
    x0, y0 = x.min() - cell, y.min() - cell
    gx = ((x - x0) / cell).astype(int)
    gy = ((y - y0) / cell).astype(int)
    m = np.zeros((gy.max() + 2, gx.max() + 2), bool)
    m[gy, gx] = True
    m = ndi.binary_closing(m, np.ones((3, 3)))
    m = ndi.binary_fill_holes(m)
    m = ndi.binary_opening(m, np.ones((2, 2)))
    lb, nl = ndi.label(m)
    if nl == 0:
        return None
    m = lb == np.argmax(np.bincount(lb.ravel())[1:]) + 1
    cont = measure.find_contours(np.pad(m, 1).astype(float), 0.5)
    if not cont:
        return None
    c = max(cont, key=len)
    poly = Polygon(np.column_stack([x0 + (c[:, 1] - 1) * cell,
                                    y0 + (c[:, 0] - 1) * cell])).simplify(cell)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly if poly.area >= min_area else None


def _newell(pts):
    n = np.zeros(3)
    for i in range(len(pts)):
        n += np.cross(pts[i], pts[(i + 1) % len(pts)])
    ln = np.linalg.norm(n)
    return n / ln if ln > 0 else np.array([0.0, 0.0, 1.0])


def _earclip(poly2):
    idx = list(range(len(poly2)))
    if len(idx) < 3:
        return []
    area = sum(poly2[i][0] * poly2[(i + 1) % len(poly2)][1]
               - poly2[(i + 1) % len(poly2)][0] * poly2[i][1]
               for i in range(len(poly2)))
    if area < 0:
        idx = idx[::-1]
    tris, guard = [], 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        n = len(idx)
        for k in range(n):
            a, b, c = idx[(k - 1) % n], idx[k], idx[(k + 1) % n]
            pa, pb, pc = poly2[a], poly2[b], poly2[c]
            cr = (pb[0] - pa[0]) * (pc[1] - pa[1]) \
                - (pb[1] - pa[1]) * (pc[0] - pa[0])
            if cr <= 1e-12:
                continue
            ok = True
            for j in idx:
                if j in (a, b, c):
                    continue
                p = poly2[j]
                d1 = (pb[0] - pa[0]) * (p[1] - pa[1]) \
                    - (pb[1] - pa[1]) * (p[0] - pa[0])
                d2 = (pc[0] - pb[0]) * (p[1] - pb[1]) \
                    - (pc[1] - pb[1]) * (p[0] - pb[0])
                d3 = (pa[0] - pc[0]) * (p[1] - pc[1]) \
                    - (pa[1] - pc[1]) * (p[0] - pc[0])
                if d1 >= 0 and d2 >= 0 and d3 >= 0:
                    ok = False
                    break
            if ok:
                tris.append((a, b, c))
                idx.pop(k)
                break
        else:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def _face_tris(verts, ring):
    P = verts[ring]
    nrm = _newell(P)
    ax = np.argmax(np.abs(nrm))
    cols = [c for c in range(3) if c != ax]
    p2 = P[:, cols]
    if nrm[ax] < 0:
        p2 = p2[:, ::-1]
    return [(ring[a], ring[b], ring[c]) for a, b, c in _earclip(p2)]


def read_cityjsonseq(path, lod="2.2"):
    """Parse a CityJSONSeq file into ``{object id: (vertices, triangles)}``.

    The first line carries the dataset transform; each following line is a
    feature whose quantised vertices detransform to world coordinates. Every
    face's outer ring is ear-clipped in the plane of its Newell normal, so
    non-convex roofer faces triangulate correctly; inner rings are ignored
    (roofer's LoD2.2 solids carry none). Only geometries of the requested
    ``lod`` are kept — this is how the 2.2 roof planes are told apart from
    the 1.2 block extrusion sharing the same object. Written against the
    output of roofer (see docs/tools.md), which delivered real measured roof
    planes for the study box's buildings [measured 2026-08-01].

    Returns vertices as an (n, 3) float array per object and triangles as
    (m, 3) int indices into it.
    """
    import json
    from pathlib import Path

    lines = Path(path).read_text().splitlines()
    meta = json.loads(lines[0])
    scale = np.array(meta["transform"]["scale"])
    trans = np.array(meta["transform"]["translate"])
    out = {}
    for ln in lines[1:]:
        if not ln.strip():
            continue
        f = json.loads(ln)
        verts = np.array(f["vertices"], float) * scale + trans
        for oid, obj in f.get("CityObjects", {}).items():
            for geom in obj.get("geometry", []):
                if geom.get("lod") != lod:
                    continue
                T = []
                for shell in geom["boundaries"]:
                    for face in shell:
                        T += _face_tris(verts, face[0])
                if T:
                    out[oid] = (verts.copy(), np.array(T, np.int64))
    return out


def read_cityjson(path, lod="2.2"):
    """Parse a PACKED CityJSON file into ``{object id: (vertices, triangles)}``.

    Same content as :func:`read_cityjsonseq`, different container: the packed
    form carries one ``vertices`` array for the whole dataset and every object
    indexes into it, instead of one feature per line with its own vertices.
    roofer emits the *stream*; the lab's ``cityjson_pack.py`` packs it — and
    when the stream lives in a temp directory, the packed file is the only copy
    that survives a reboot. Which is exactly what happened here, so this reader
    is what recovers real LoD2.2 roof planes without the binary.

    Faces are ear-clipped in the plane of their Newell normal, as in the
    streamed reader, so non-convex roof faces triangulate correctly.
    """
    import json
    from pathlib import Path

    doc = json.loads(Path(path).read_text())
    tr = doc.get("transform", {})
    scale = np.array(tr.get("scale", [1.0, 1.0, 1.0]))
    trans = np.array(tr.get("translate", [0.0, 0.0, 0.0]))
    verts = np.array(doc["vertices"], float) * scale + trans
    out = {}
    for oid, obj in doc.get("CityObjects", {}).items():
        T = []
        for geom in obj.get("geometry", []):
            if str(geom.get("lod")) != str(lod):
                continue
            b = geom["boundaries"]
            # Solid: [shell][face][ring]; MultiSurface: [face][ring]
            shells = b if geom.get("type") == "Solid" else [b]
            for shell in shells:
                for face in shell:
                    if face and isinstance(face[0], list):
                        T += _face_tris(verts, face[0])
        if T:
            out[oid] = (verts, np.array(T, np.int64))
    return out


def sat_lift(soup, cols, rgb, x0, y1, res):
    """Deshadow per-vertex paint, witnessed by an orthorectified satellite.

    The survey's colour drape keeps the shadows of the flight moment: one
    building's lower roof band was blue-grey (121,131,143) where the
    satellite showed warm tan (151,137,124), while its top roof was bright in
    both — so a single gain per building misses the shadow [measured
    2026-08-01]. Each painted vertex is compared with the satellite pixel at
    its own (x, y): only where the satellite is clearly brighter (luminance
    ratio > 1.15) does the survey colour lift toward it, per channel (gain
    clipped to 0.9–2.2), so a blue shadow cast comes out too. The satellite
    has its own shadows, so nothing is darkened below ~the survey value. In
    deep shadow (ratio > 2.2; one wall read luminance 22 against 109) the
    survey pixel is noise and scaling noise just makes bright noise — the
    witness's colour is adopted outright.

    ``rgb`` is an (H, W, 3) north-up image whose top-left corner sits at
    ``(x0, y1)`` in the soup's coordinates, ``res`` metres per pixel.
    Returns new uint8 colours; the inputs are not modified.
    """
    soup = np.asarray(soup, float)
    cols = np.asarray(cols)
    if not len(soup):
        return cols
    c = ((soup[:, 0] - x0) / res).astype(int).clip(0, rgb.shape[1] - 1)
    r = ((y1 - soup[:, 1]) / res).astype(int).clip(0, rgb.shape[0] - 1)
    sat = np.asarray(rgb, float)[r, c]
    svy = np.maximum(cols.astype(float), 1.0)
    w = np.array([0.299, 0.587, 0.114])
    ratio = (sat @ w) / np.maximum(svy @ w, 1.0)
    g = np.clip(sat / svy, 0.9, 2.2)
    g[ratio <= 1.15] = 1.0
    out = svy * g
    deep = ratio > 2.2
    out[deep] = sat[deep]
    return np.clip(out, 0, 255).astype(np.uint8)


def crown_thin(x, y, z, cell=0.28):
    """Thin a tree crown to one display return per ``cell``-metre voxel.

    Kaveh's crown model (2026-08-01): the crown IS the tree's own points —
    no envelope, no skin the laser never measured — but a canopy has no
    structure finer than roughly a leaf cluster, so rendering every return
    only spends triangles restating the same cluster. One return per 28 cm
    voxel keeps the shape and the depth sparkle at a fraction of the points.
    Crowns of 400 returns or fewer pass through untouched — thinning them
    saves nothing worth the pass.

    Returns sorted indices into the inputs (first return in each voxel wins).
    """
    if len(x) <= 400:
        return np.arange(len(x))
    vox = np.round(np.column_stack([np.asarray(x, float), np.asarray(y, float),
                                    np.asarray(z, float)]) / cell
                   ).astype(np.int64)
    key = vox[:, 0] + vox[:, 1] * 2_000_003 + vox[:, 2] * 4_000_000_007
    _, keep = np.unique(key, return_index=True)
    return np.sort(keep)


def footprint_prism(ring, z0, ztop, col=None):
    """Footprint outline extruded ``z0`` → ``ztop``, roof cap by ear-cutting.

    The walls are the surveyed outline, not voxel stair-steps — this replaced
    the 1 m point-blob solid for the Sea Village float homes, whose Overture
    footprints and measured P95 roof heights were both better than anything
    a gridded solid could rebuild [measured 2026-08-01]. It is also the
    honest fallback when a plane-fitted LoD2 roof is rejected: a prism claims
    only the outline and two heights, nothing about roof shape.

    No bottom cap — the prism sits on ground or water, where no return ever
    lands. ``ring`` is an (n, 2) outline without a closing duplicate vertex.
    Returns ``(soup, cols)`` like the other builders here.
    """
    import mapbox_earcut

    ring = np.asarray(ring, float)
    col = np.asarray([150, 150, 150] if col is None else col, np.uint8)
    tri = mapbox_earcut.triangulate_float64(
        ring, np.array([len(ring)], np.uint32)).reshape(-1, 3)
    top = np.column_stack([ring, np.full(len(ring), float(ztop))])
    bot = np.column_stack([ring, np.full(len(ring), float(z0))])
    soup = [top[tri.ravel()]]
    for k in range(len(ring)):
        a, b = k, (k + 1) % len(ring)
        soup.append(np.array([bot[a], bot[b], top[b],
                              bot[a], top[b], top[a]]))
    soup = np.vstack(soup).astype(np.float32)
    return soup, np.tile(col, (len(soup), 1))


def wall_occupancy(ring, x, y, z, z0, ztop, rgb=None, col=None,
                   cell=0.45, reach=0.9):
    """Measured wall panels with holes, on a footprint outline used as a canvas.

    Each (along-edge, height) cell of ``cell`` metres renders as wall exactly
    where returns lie within ``reach`` of that wall plane — open sides stay
    open, posts appear where they stand, passages become holes. Cells wear
    the median colour of their own contributing returns when ``rgb`` (an
    (n, 3) uint8 array) is given, else ``col``, else mid-grey.

    Why not extrude the footprint: an airborne survey sees more than roofs —
    the Granville Island picnic pavilion (OSM building=roof) carries 1,826
    sub-eave returns (17% of its points) forming one connected structure
    [measured 2026-08-01]; extruded walls would bury exactly the openness
    that makes it a pavilion, and synthetic corner posts stood where nothing
    was measured. Occupancy at 0.45 m yielded 548 evidence-backed cells there.

    Caution for CLOSED buildings: their lower walls are often laser-shadowed
    by roof overhangs and neighbours, so raw occupancy punches false holes in
    real walls. Use on open structures (canopies, pavilions); closed walls
    need an occlusion-aware rule first.

    Returns ``(soup, cols)`` — an (m*6, 3) float32 triangle soup and matching
    (m*6, 3) uint8 colours — or ``None`` if nothing was seen.
    """
    P = np.column_stack([np.asarray(x, float), np.asarray(y, float),
                         np.asarray(z, float)])
    ring = np.asarray(ring, float)
    soup, cols = [], []
    for k in range(len(ring)):
        _edge_cells(ring[k], ring[(k + 1) % len(ring)], P, z0, ztop,
                    rgb, col, cell, reach, soup, cols)
    if not soup:
        return None
    return (np.vstack(soup).astype(np.float32),
            np.clip(np.vstack(cols), 0, 255).astype(np.uint8))


def _edge_cells(a, b, P, z0, ztop, rgb, col, cell, reach, soup, cols):
    """Append one edge's evidence cells (quads + colours) to soup/cols."""
    from scipy import ndimage as ndi

    e = b - a
    L = float(np.hypot(*e))
    if L < 0.6:
        return
    u = e / L
    rel = P[:, :2] - a
    s = rel @ u
    d = rel @ np.array([e[1], -e[0]]) / L
    m = (s > -0.2) & (s < L + 0.2) & (np.abs(d) < reach) \
        & (P[:, 2] > z0 + 0.05) & (P[:, 2] < ztop + 0.3)
    if not m.any():
        return
    ns = int(L / cell) + 1
    nz = max(int((ztop - z0) / cell) + 1, 1)
    si = (s[m] / cell).astype(int).clip(0, ns - 1)
    zi = ((P[m, 2] - z0) / cell).astype(int).clip(0, nz - 1)
    occ = np.zeros((nz, ns), bool)
    occ[zi, si] = True
    # close on an edge-padded grid: scipy erodes borders against an
    # all-empty outside, which silently deleted the ground row
    occ = ndi.binary_closing(np.pad(occ, 1, mode="edge"),
                             np.ones((2, 2)))[1:-1, 1:-1]
    if col is None and rgb is not None:
        R = np.asarray(rgb, float)[m]
        cs = np.zeros((nz, ns, 3))
        cn = np.zeros((nz, ns))
        np.add.at(cs, (zi, si), R)
        np.add.at(cn, (zi, si), 1)
    for zz, ss in zip(*np.nonzero(occ)):
        p0 = a + u * (ss * cell)
        p1 = a + u * min((ss + 1) * cell, L)
        zl, zh = z0 + zz * cell, z0 + (zz + 1) * cell
        soup.append(np.array([[p0[0], p0[1], zl], [p1[0], p1[1], zl],
                              [p1[0], p1[1], zh], [p0[0], p0[1], zl],
                              [p1[0], p1[1], zh], [p0[0], p0[1], zh]]))
        if col is None and rgb is not None and cn[zz, ss] > 0:
            cols.append(np.tile(cs[zz, ss] / cn[zz, ss], (6, 1)))
        else:
            cols.append(np.tile(col if col is not None
                                else [150, 150, 150], (6, 1)))


def canopy_mesh(poly, x, y, z, z0, col=None, rgb=None):
    """Open-sided canopy (map tag building=roof): a measured roof, no solid.

    An extruded solid is wrong for a market awning — the open sides are the
    structure's meaning. The roof is a flat 0.3 m slab at the P85 return
    height by default; a PITCHED canopy builds a gable instead: when the
    ridge (P98 of all returns) stands more than 1.0 m over the eave (P85 of
    the returns in the outer band, beyond 30 % of the half-width across the
    short PCA axis), two measured roof planes rise on the minimum rotated
    rectangle. That rule was settled on the Granville Island picnic pavilion
    (OSM building=roof, gabled), where a flat slab flattened a roof the
    returns clearly pitched [measured 2026-08-01]. Below either roof nothing
    is invented: ``wall_occupancy`` paints only the cells the laser saw —
    posts and partial walls appear, open sides and passages stay open.

    ``poly`` is a shapely Polygon used as the canvas (the footprint outline,
    never a geometry source); ``z0`` is the ground elevation there. Returns a
    list of ``(soup, cols)`` parts — empty if the P85 roof height clears the
    ground by less than 2 m, which is street furniture, not a canopy.
    """
    import mapbox_earcut
    import shapely

    x, y, z = (np.asarray(a, float) for a in (x, y, z))
    col = np.asarray([150, 150, 150] if col is None else col, np.uint8)
    ring = np.asarray(poly.simplify(0.4).exterior.coords)[:-1]
    ztop = float(np.percentile(z, 85))
    if ztop - z0 < 2.0:
        return []
    d = np.column_stack([x, y])
    d = d - d.mean(0)
    _, V = np.linalg.eigh(d.T @ d)
    v = d @ V[:, 0]                                # across the short axis
    outer = np.abs(v - (v.min() + v.max()) / 2) > 0.30 * np.ptp(v)
    if outer.sum() >= 20:
        eave = float(np.percentile(z[outer], 85))
        ridge = float(np.percentile(z, 98))
        if ridge - eave > 1.0:
            rect = np.asarray(shapely.minimum_rotated_rectangle(poly)
                              .exterior.coords)[:-1]
            e = np.roll(rect, -1, 0) - rect
            s = int(np.argmax(np.hypot(e[:, 0], e[:, 1])))
            A, B, C, D = (rect[(s + k) % 4] for k in range(4))
            r0, r1 = (B + C) / 2, (D + A) / 2
            Ae, Be, Ce, De = (np.array([p[0], p[1], eave])
                              for p in (A, B, C, D))
            R0 = np.array([r0[0], r0[1], ridge])
            R1 = np.array([r1[0], r1[1], ridge])
            roof = np.vstack([np.array([Ae, Be, R0, Ae, R0, R1]),
                              np.array([Ce, De, R1, Ce, R1, R0])]
                             ).astype(np.float32)
            parts = [(roof, np.tile(col, (len(roof), 1)))]
            # walls close against the roof: each side's top profile is the
            # gable line — eave at the corners, ridge where the side
            # crosses the ridge axis (the flat-topped wall missed the
            # pediment, Kaveh, 2026-08-01)
            rd = r1 - r0
            rd = rd / max(float(np.hypot(*rd)), 1e-9)
            halfW = max(float(np.hypot(*(C - B))) / 2, 1e-6)

            def _htop(p):
                dz = abs((p[0] - r0[0]) * rd[1] - (p[1] - r0[1]) * rd[0])
                return float(np.clip(ridge - (ridge - eave) * dz / halfW,
                                     eave, ridge))

            tops = {}
            for k in range(len(ring)):
                a2, b2 = ring[k], ring[(k + 1) % len(ring)]
                L2 = float(np.hypot(*(b2 - a2)))
                if L2 < 0.6:
                    continue
                bp = [(0.0, _htop(a2)), (L2, _htop(b2))]
                ca = (a2[0] - r0[0]) * rd[1] - (a2[1] - r0[1]) * rd[0]
                cb = (b2[0] - r0[0]) * rd[1] - (b2[1] - r0[1]) * rd[0]
                if ca * cb < 0:
                    f = ca / (ca - cb)
                    pm = a2 + (b2 - a2) * f
                    bp.insert(1, (float(f * L2), _htop(pm)))
                tops[k] = bp
            w, _ = wall_mesh(ring, x, y, z, z0, eave, rgb=rgb, tops=tops)
            return parts + w
    tri = mapbox_earcut.triangulate_float64(
        ring, np.array([len(ring)], np.uint32)).reshape(-1, 3)
    top = np.column_stack([ring, np.full(len(ring), ztop)])
    bot = np.column_stack([ring, np.full(len(ring), ztop - 0.3)])
    soup = [top[tri.ravel()], bot[tri[:, ::-1].ravel()]]
    for k in range(len(ring)):                     # slab rim
        a, b = k, (k + 1) % len(ring)
        soup.append(np.array([bot[a], bot[b], top[b], bot[a], top[b], top[a]]))
    soup = np.vstack(soup).astype(np.float32)
    parts = [(soup, np.tile(col, (len(soup), 1)))]
    w, _ = wall_mesh(ring, x, y, z, z0, ztop - 0.3, rgb=rgb)
    return parts + w
