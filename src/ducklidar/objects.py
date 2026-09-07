"""Per-class object meshes: what a labelled instance looks like in 3-D.

``ducklidar.scene`` builds the pieces a BUILDING is made of — footprints, roof
solids, walls that stood trial. This module is the rest of the cascade: the
recipes that turn a boat, a dock, a car, a street light or a tree into geometry.
Ported from the lab's ``_recon.py``, which is where they were measured, with one
design change that matters for reuse:

**No context object.** The lab's recipes take a ``ctx`` carrying the whole run —
points, labels, DEM, water level, OSM, roofer output. Here each recipe takes the
arrays it actually needs, so a caller with a table pipeline (stage 5 labels,
stage 6 instances) can drive them without assembling a lab run first. Where a
recipe needs a datum it cannot derive — the water level, the ground under a
lamp — it is an argument, not a lookup.

**One named model per object TYPE** (``*_model``, at the bottom of this file).
Each returns ``[(soup, cols, kind), ...]``: an (n, 3) float32 triangle soup with
n divisible by 3, matching uint8 vertex colours, and the part's KIND. An empty
list means "this instance does not earn geometry", which is a real answer — the
lab's rule is no extruded fallbacks on the map. No model calls another model;
where two types share geometry they share a PRIMITIVE. The older ``*_mesh``
recipes are those primitives, and they carry no kind.

**The kind is the most important property a part has**, because it is what the
sun sees:

``solid``    light stops at it — a closed volume standing on its base.
``slab``     a (top, bottom) pair with open air between it and the ground:
             light passes UNDERNEATH. Measured: Granville Bridge as a slab
             blocks only above 72.6°, an angle the Vancouver sun never reaches;
             modelled solid it blacks out the waterway all day (v1's nDSM bug).
``glass``    a zero-thickness pane on a surface that already exists. It has no
             volume and it TRANSMITS, so it is neither of the above.
``surface``  ground, water, land use: it receives light and casts nothing.

Primitives (:func:`box_local`, :func:`ring_prism`, :func:`top_surface`,
:func:`cell_prism`, :func:`solid_25d`, :func:`hull_soup`, :func:`drape` and
``scene.footprint_prism``) return raw geometry with no kind — they are the
building blocks models are made of, not models.
"""
import numpy as np

DOCK_COL = np.array([150, 140, 125], np.uint8)      # structure gets its material
CROWN_COL = np.array([86, 122, 74], np.uint8)        # foliage, not the survey RGB
DECK_T = 1.6        # lab, measured on Granville: box-girder depth [2026-08-01]
FLAT_WIN = 15.0     # m — the neighbourhood a deck cell is judged against
FLAT_TOL = 2.5      # m — beyond this it is a mast or a pier, not road surface
PIER_MAX_W = 8.0    # m — a support is THIN. Granville's are bents, not
                    # columns: measured 26-55 m long and only 4-5 m wide,
                    # rising 46-96% of their clearance. Capping LENGTH threw
                    # away every real one; width is what separates a bent from
                    # a building. [2026-08-05]
PIER_FOOT = 1.5     # m — a support starts THIS close to the ground it stands on
PIER_RISE = 0.40    # of the clearance it spans; below this it is not a column
RAIL_H = 1.26       # lab: P90 rail height over deck, both sides [2026-08-01]
WINCOL = np.array([160, 200, 235], np.uint8)        # lab's window blue
DECK_COL = np.array([168, 170, 172], np.uint8)      # lab: deck top grey
PIER_COL = np.array([132, 130, 126], np.uint8)      # concrete
RAIL_COL = np.array([148, 152, 158], np.uint8)      # lab: guard-rail grey

SOLID, SLAB, GLASS, SURFACE = "solid", "slab", "glass", "surface"


def pca_frame(px, py):
    """``(mu, e_long, e_cross, u, v)`` — the instance's own axes in plan.

    A boat, a car and a dock are all *oriented* objects, and their long axis is
    the first principal component of their plan-view returns. Everything built
    on ``box_local`` works in this frame, so a hull is modelled along the hull
    rather than along easting.
    """
    c = np.stack([np.asarray(px), np.asarray(py)], 1)
    mu = c.mean(0)
    dd = c - mu
    _w, V = np.linalg.eigh(dd.T @ dd)
    return mu, V[:, 1], V[:, 0], dd @ V[:, 1], dd @ V[:, 0]


def box_local(mu, eu, ev, u0, u1, v0, v1, z0, z1):
    """A closed box in a local (eu, ev) frame — 12 triangles, as a vertex list."""
    def w3(uu, vv, zz):
        return (mu[0] + uu * eu[0] + vv * ev[0],
                mu[1] + uu * eu[1] + vv * ev[1], zz)

    c = [w3(u0, v0, z0), w3(u1, v0, z0), w3(u1, v1, z0), w3(u0, v1, z0),
         w3(u0, v0, z1), w3(u1, v0, z1), w3(u1, v1, z1), w3(u0, v1, z1)]
    idx = [(0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6),
           (3, 0, 4), (3, 4, 7), (4, 5, 6), (4, 6, 7), (0, 2, 1), (0, 3, 2)]
    return [c[a] for t in idx for a in t]


def ring_prism(ring_top, ring_bot, cen_top):
    """Sides between two rings, plus a fan roof to ``cen_top``."""
    soup = []
    K = len(ring_top)
    for k in range(K):
        a, b = ring_top[k], ring_top[(k + 1) % K]
        a0, b0 = ring_bot[k], ring_bot[(k + 1) % K]
        soup += [a0, b0, b, a0, b, a]
        soup += [a, b, cen_top]
    return soup


def car_mesh(x, y, z, col):
    """A car: two stacked boxes in its own frame — body, then cabin.

    The base is the instance's OWN 2nd percentile, never a DEM lookup: a car on
    a bridge deck stands on the deck, and a grid would put it on the water.
    """
    mu, eu, ev, u, v = pca_frame(x, y)
    L, W = np.ptp(u), np.ptp(v)
    z0 = float(np.percentile(z, 2))
    hh = float(np.percentile(z, 95)) - z0
    if L < 1.0 or hh <= 0.2:
        return []
    soup = box_local(mu, eu, ev, -L / 2, L / 2, -W / 2, W / 2,
                     z0 + 0.25, z0 + 0.62 * hh)
    soup += box_local(mu, eu, ev, -L / 2 + 0.22 * L, L / 2 - 0.18 * L,
                      -W / 2 + 0.08 * W, W / 2 - 0.08 * W,
                      z0 + 0.62 * hh, z0 + hh)
    soup = np.asarray(soup, np.float32)
    return [(soup, np.tile(np.asarray(col, np.uint8), (len(soup), 1)))]


def _pole(cx, cy, z0, z1, r=0.11, n=6):
    soup = []
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rim = np.column_stack([cx + np.cos(ang) * r, cy + np.sin(ang) * r])
    for q in range(n):
        a, b = rim[q], rim[(q + 1) % n]
        soup += [[a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z1],
                 [a[0], a[1], z0], [b[0], b[1], z1], [a[0], a[1], z1]]
    return soup


def lamp_mesh(x, y, z, ground_z, toward=None):
    """A street light: pole, arm, lantern — the arm overhanging the roadway.

    ``toward`` is a unit vector the arm should point along (the roadway's
    centreline). Without it the arm points +x, which is honest but arbitrary.
    """
    GREY = np.array([70, 72, 76], np.uint8)
    WARM = np.array([255, 238, 190], np.uint8)
    cx, cy = float(np.median(x)), float(np.median(y))
    ztop = float(np.percentile(z, 98))
    zarm = ztop - 0.25
    if zarm - ground_z < 1.0:
        return []
    de = np.asarray(toward, float) if toward is not None else np.array([1.0, 0.0])
    de = de / max(np.hypot(*de), 1e-9)
    ev = np.array([-de[1], de[0]])
    pole = np.asarray(_pole(cx, cy, ground_z, zarm), np.float32)
    arm = np.asarray(box_local(np.array([cx + de[0] * 0.55, cy + de[1] * 0.55]),
                               de, ev, -0.65, 0.65, -0.05, 0.05,
                               zarm - 0.08, zarm), np.float32)
    head = np.asarray(box_local(np.array([cx + de[0] * 1.15, cy + de[1] * 1.15]),
                                de, ev, -0.35, 0.35, -0.16, 0.16,
                                zarm - 0.35, zarm - 0.02), np.float32)
    grey = np.vstack([pole, arm])
    return [(grey, np.tile(GREY, (len(grey), 1))),
            (head, np.tile(WARM, (len(head), 1)))]


def dock_mesh(x, y, z, wlvl, col=None, cell=0.5):
    """A dock or pontoon: its traced outline, extruded waterline -> deck.

    The outline is traced from the returns and simplified, with the water
    between marina fingers kept as HOLES — the cell-wise solid this replaced
    staircased along every diagonal walkway.
    """
    from scipy import ndimage as ndi
    import mapbox_earcut
    import shapely.geometry as sg
    from skimage import measure

    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    col = DOCK_COL if col is None else np.asarray(col, np.uint8)
    x0, y0 = x.min() - cell, y.min() - cell
    gx = ((x - x0) / cell).astype(int)
    gy = ((y - y0) / cell).astype(int)
    occ = np.zeros((gy.max() + 2, gx.max() + 2), bool)
    occ[gy, gx] = True
    occ = ndi.binary_closing(occ, np.ones((2, 2)))
    deck = wlvl + np.clip(np.median(z) - wlvl, 0.3, 1.2)
    z0 = wlvl - 0.25

    lab, ncomp = ndi.label(occ)
    soup = []
    for k in range(1, ncomp + 1):
        m = lab == k
        if m.sum() < 40:
            continue
        filled = ndi.binary_fill_holes(m)
        rings = [max(measure.find_contours(np.pad(filled, 1).astype(float), 0.5),
                     key=len)]
        holes, nh = ndi.label(filled & ~m)
        for h in range(1, nh + 1):
            hm = holes == h
            if hm.sum() < 8:
                continue
            rings.append(max(measure.find_contours(
                np.pad(hm, 1).astype(float), 0.5), key=len))
        polys = []
        for r in rings:
            xy = np.column_stack([x0 + (r[:, 1] - 1) * cell,
                                  y0 + (r[:, 0] - 1) * cell])
            polys.append(np.asarray(sg.LinearRing(xy).simplify(0.4).coords)[:-1])
        if any(len(p) < 3 for p in polys):
            continue
        verts = np.vstack(polys)
        ends = np.cumsum([len(p) for p in polys])
        tri = mapbox_earcut.triangulate_float64(verts, ends).reshape(-1, 3)
        top = np.column_stack([verts, np.full(len(verts), deck)])
        bot = np.column_stack([verts, np.full(len(verts), z0)])
        soup.append(top[tri.ravel()])
        soup.append(bot[tri[:, ::-1].ravel()])
        for p in polys:
            for a in range(len(p)):
                b = (a + 1) % len(p)
                soup.append(np.array([[p[a][0], p[a][1], z0],
                                      [p[b][0], p[b][1], z0],
                                      [p[b][0], p[b][1], deck],
                                      [p[a][0], p[a][1], z0],
                                      [p[b][0], p[b][1], deck],
                                      [p[a][0], p[a][1], deck]]))
    if not soup:
        return []
    soup = np.vstack(soup).astype(np.float32)
    return [(soup, np.tile(col, (len(soup), 1)))]


def boat_mesh(x, y, z, wlvl, col):
    """A vessel: hull with a pointed bow, a cabin, and a mast if it has one.

    A flat float — anything under 1.2 m over the waterline — is a barge or
    pontoon and goes to :func:`dock_mesh` instead: it has an outline, not a hull.
    The cabin is sized from HULL points only, because a mast would otherwise
    become a 15 m cabin; the masthead is found where the highest returns bunch
    tightly, since stays and rigging defeat a plain 2-D cluster test.
    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    col = np.asarray(col, np.uint8)
    if float(np.percentile(z, 95)) - wlvl < 1.2:
        return dock_mesh(x, y, z, wlvl, col=col)
    soup = hull_soup(x, y, z, wlvl)
    if soup is None:
        return []
    return [(soup, np.tile(col, (len(soup), 1)))]


def hull_soup(x, y, z, wlvl):
    """Hull + cabin + mast in the vessel's own PCA frame, or ``None``.

    Split out of :func:`boat_mesh` so a recipe can build a hull without
    inheriting that function's "a low float is a dock" redirect — a barge is a
    low hull, not a pontoon, and only its OWNER knows which it is.
    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    mu, eu, ev, u, v = pca_frame(x, y)
    L, W = np.ptp(u), np.ptp(v)
    if L < 1.5:
        return None

    deck = wlvl + np.clip(np.percentile(z, 30) - wlvl, 0.4, 1.6)
    ss = np.linspace(0, 1, 8)
    hw = np.maximum(np.where(ss < 0.7, 0.92, 0.92 * (1 - ss) / 0.3) * W / 2,
                    0.04 * W)
    uu = u.min() + ss * L
    ring = ([(uu[k], +hw[k]) for k in range(8)]
            + [(uu[k], -hw[k]) for k in range(8)][::-1])

    def w3(uv, zz):
        return np.array([mu[0] + uv[0] * eu[0] + uv[1] * ev[0],
                         mu[1] + uv[0] * eu[1] + uv[1] * ev[1], zz])

    soup = ring_prism([w3(p, deck) for p in ring],
                      [w3((p[0], p[1] * 0.55), wlvl - 0.25) for p in ring],
                      w3((u.min() + 0.45 * L, 0), deck))

    tall = z > deck + 4.0
    body = z[~tall] if tall.any() and (~tall).sum() >= 20 else z
    top95 = float(np.percentile(body, 95))
    if top95 - deck > 0.7:
        cu = float(np.median(u[z > deck + 0.4])) if (z > deck + 0.4).any() else 0.0
        cl, cw = 0.45 * L, 0.62 * W
        ctop = min(top95, deck + 3.2)
        cor = [(cu - cl / 2, -cw / 2), (cu + cl / 2, -cw / 2),
               (cu + cl / 2, cw / 2), (cu - cl / 2, cw / 2)]
        soup += ring_prism([w3(p, ctop) for p in cor],
                           [w3(p, deck) for p in cor], w3((cu, 0), ctop))

    if tall.sum() >= 15:
        zmax = float(np.percentile(z[tall], 99))
        sel = tall & (z > zmax - 3)
        if sel.sum() >= 3 and max(np.ptp(x[sel]), np.ptp(y[sel])) < 2.5:
            soup += [np.asarray(p) for p in
                     _pole(float(np.median(x[sel])), float(np.median(y[sel])),
                           deck, zmax, r=0.14)]

    return np.asarray(soup, np.float32)


def top_surface(x, y, z, pix=1.0):
    """The measured top of a thing, on a ``pix`` grid: ``(top, occ, lo)``.

    The shared half of every 2.5-D recipe — what the laser saw looking down.
    Cells with no return are filled from their nearest measured neighbour and
    the result median-filtered at 3 cells, so one dropout does not punch a
    hole. ``occ`` is the closed/opened occupancy (the thing's plan extent),
    ``lo`` the grid origin. Returns ``None`` when nothing survives opening.
    """
    from scipy import ndimage as ndi

    P = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])
    lo = P[:, :2].min(0) - pix
    gx = ((P[:, 0] - lo[0]) / pix).astype(int)
    gy = ((P[:, 1] - lo[1]) / pix).astype(int)
    nx, ny = gx.max() + 2, gy.max() + 2
    top = np.full((ny, nx), -np.inf)
    np.maximum.at(top, (gy, gx), P[:, 2])
    occ = np.isfinite(top)
    occ = ndi.binary_closing(occ, np.ones((3, 3)))
    occ = ndi.binary_opening(occ, np.ones((2, 2)))
    if not occ.any():
        return None
    ii = ndi.distance_transform_edt(~np.isfinite(top), return_distances=False,
                                    return_indices=True)
    return ndi.median_filter(top[tuple(ii)], size=3), occ, lo


def cell_prism(top, bot, occ, lo, pix):
    """Cells between two height fields: top caps, bottom caps, boundary walls.

    The primitive both a grounded solid and a floating plate are made of —
    the only difference is where ``bot`` sits. A scalar ``bot`` gives a volume
    standing on a datum (:func:`solid_25d`); ``bot = top - t`` gives a plate of
    thickness ``t`` with open air beneath it (:func:`bridge_deck_model`).
    Walls are emitted only at the occupancy boundary, so the interior stays
    hollow.
    """
    bot = np.broadcast_to(np.asarray(bot, float), top.shape)
    ny, nx = top.shape
    soup = []
    ys, xs = np.nonzero(occ)
    for r, c in zip(ys, xs):
        X0, Y0 = lo[0] + c * pix, lo[1] + r * pix
        X1, Y1 = X0 + pix, Y0 + pix
        t, b = float(top[r, c]), float(bot[r, c])
        soup += [[X0, Y0, t], [X1, Y0, t], [X1, Y1, t],
                 [X0, Y0, t], [X1, Y1, t], [X0, Y1, t]]
        soup += [[X0, Y1, b], [X1, Y1, b], [X1, Y0, b],
                 [X0, Y1, b], [X1, Y0, b], [X0, Y0, b]]
        for dr, dc, (ax, ay, bx, by) in ((-1, 0, (X0, Y0, X1, Y0)),
                                         (1, 0, (X1, Y1, X0, Y1)),
                                         (0, -1, (X0, Y1, X0, Y0)),
                                         (0, 1, (X1, Y0, X1, Y1))):
            rr, cc = r + dr, c + dc
            if 0 <= rr < ny and 0 <= cc < nx and occ[rr, cc]:
                continue
            soup += [[ax, ay, b], [bx, by, b], [bx, by, t],
                     [ax, ay, b], [bx, by, t], [ax, ay, t]]
    return np.asarray(soup, np.float32)


def solid_25d(x, y, z, z0, pix=1.0):
    """A 2.5-D solid: top surface on a grid, boundary walls, flat bottom cap.

    The honest fallback for a thing with a measured top and a known base — a
    marine shed extruded from the waterline, not from terrain.
    """
    got = top_surface(x, y, z, pix)
    if got is None:
        return []
    top, occ, lo = got
    return cell_prism(np.maximum(top, z0 + 0.05), z0, occ, lo, pix)


ROOF_SPAN = 0.30     # a real roof is a thin band at the top of a building;
                     # measured over roofer's 152 solids the span/height median
                     # is 0.33 and 54 exceed 0.50 — those are not roofs
ROOF_TOL = 0.25      # m — a cell this close to a plane belongs to it
ROOF_MIN = 12        # cells — smaller than this is not a roof facet


def fit_roof_planes(top, occ, pix=1.0, tol=None, min_cells=None, rounds=8):
    """Snap a measured roof grid onto PLANES — the facets a roof is made of.

    ``solid_25d`` takes the per-cell maximum and extrudes it, so a roof comes
    out as a staircase of 1 m steps: honest, but it is not what a roof looks
    like, and it is the reason the lab's buildings read better than ours
    wherever we lack one of its `roofer` LoD2 solids (we recovered 152 of them;
    87 are used, the rest fall back to the prism).

    This is the honest middle: no idealised primitive is fitted to the
    building, but a roof IS piecewise planar, so the measured surface is
    segmented into planes and each cell is snapped onto its own. Ridges, hips
    and dormers survive because they are separate planes; the staircase does
    not, because it was never in the roof.

    Greedy and deterministic: take the largest remaining region, fit a plane by
    least squares, claim every unassigned cell within ``tol`` of it, repeat.
    Cells no plane claims keep their measured height, which is the right
    answer for a plant room or a parapet the planes cannot describe.
    """
    tol = ROOF_TOL if tol is None else tol
    min_cells = ROOF_MIN if min_cells is None else min_cells
    out = np.array(top, float, copy=True)
    free = np.array(occ, bool, copy=True)
    ny, nx = out.shape
    gy, gx = np.mgrid[0:ny, 0:nx]
    X = gx * pix
    Y = gy * pix
    n_planes = 0
    for _ in range(int(rounds)):
        if free.sum() < min_cells:
            break
        # seed on the flattest neighbourhood left, so a big facet goes first
        idx = np.flatnonzero(free.ravel())
        A = np.column_stack([X.ravel()[idx], Y.ravel()[idx],
                             np.ones(len(idx))])
        b = out.ravel()[idx]
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        pred = coef[0] * X + coef[1] * Y + coef[2]
        hit = free & (np.abs(out - pred) < tol)
        if hit.sum() < min_cells:
            break
        # refit on the claimed cells only, then claim again
        jdx = np.flatnonzero(hit.ravel())
        A2 = np.column_stack([X.ravel()[jdx], Y.ravel()[jdx],
                              np.ones(len(jdx))])
        coef, *_ = np.linalg.lstsq(A2, out.ravel()[jdx], rcond=None)
        pred = coef[0] * X + coef[1] * Y + coef[2]
        hit = free & (np.abs(out - pred) < tol)
        if hit.sum() < min_cells:
            break
        out[hit] = pred[hit]
        free &= ~hit
        n_planes += 1
    return out, n_planes


def solid_25d_planar(x, y, z, z0, pix=1.0):
    """A 2.5-D solid whose roof is snapped onto its own planes."""
    got = top_surface(x, y, z, pix)
    if got is None:
        return []
    top, occ, lo = got
    fitted, _n = fit_roof_planes(top, occ, pix)
    return cell_prism(np.maximum(fitted, z0 + 0.05), z0, occ, lo, pix)


def crown_split(x, y, z, ground_z, stems=None, *, pix=1.0, prominence=2.0,
                min_h=3.0, snap=6.0):
    """Merged canopy → one id per crown. Prominence watershed on the CHM,
    seeded by MAPPED STEMS where the map has them.

    The lab's version detects treetops from the canopy model alone, which is
    the only option when there is no map. Overture carries individual trees
    (1,976 over this tile) and a mapped stem is better evidence of "there is a
    tree here" than a bump in the canopy: a leaning crown's highest point sits
    off its trunk, two trees whose crowns merge often show one peak, and a
    roof edge can produce a peak that is not a tree at all.

    So both are used: every mapped stem within ``snap`` metres of canopy
    becomes a marker, and CHM peaks seed only where no stem claims the area.
    ``stems`` is an (n, 2) array of stem positions in the same CRS; None falls
    back to the lab's CHM-only behaviour exactly.

    Returns one int32 id per input point.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    x0, y0 = x.min(), y.min()
    gx = ((x - x0) / pix).astype(int)
    gy = ((y - y0) / pix).astype(int)
    nx = gx.max() + 1
    cell = gy * nx + gx
    top = np.full((gy.max() + 1) * nx, -np.inf)
    np.maximum.at(top, cell, z - ground_z)
    top = top.reshape(-1, nx)
    occ = np.isfinite(top)
    ts = ndi.gaussian_filter(np.where(occ, top, 0.0), 1.2)

    seeded = np.zeros_like(occ)
    n_map = 0
    if stems is not None and len(stems):
        s = np.asarray(stems, float)
        sx = ((s[:, 0] - x0) / pix).astype(int)
        sy = ((s[:, 1] - y0) / pix).astype(int)
        ok = (sx >= 0) & (sy >= 0) & (sx < nx) & (sy < occ.shape[0])
        sx, sy = sx[ok], sy[ok]
        if len(sx):
            # a stem only counts where the laser actually saw canopy near it
            near = ndi.binary_dilation(occ & (ts > min_h),
                                       np.ones((2 * int(snap / pix) + 1,) * 2))
            keep = near[sy, sx]
            sy, sx = sy[keep], sx[keep]
            seeded[sy, sx] = True
            n_map = int(keep.sum())

    peaks = h_maxima(np.where(occ, ts, 0.0), prominence) & occ & (ts > min_h)
    if n_map:                      # CHM peaks only where no stem claims it
        claimed = ndi.binary_dilation(seeded,
                                      np.ones((2 * int(snap / pix) + 1,) * 2))
        peaks &= ~claimed
    markers, nm = ndi.label(seeded | peaks)
    if nm < 2:
        return (gx // 10 + 1000 * (gy // 10) + 1).astype(np.int32)
    return watershed(-ts, markers, mask=occ).ravel()[cell].astype(np.int32)


def trunk_mesh(x, y, z, ground_z, stem=None):
    """A tree's TRUNK only — the crown is its own returns.

    Kaveh's model (2026-08-01): the viewer renders the thinned returns and no
    skin the laser never measured is drawn over them. ``stem`` is the mapped
    position: an overhanging crown's centroid can land on a roof, and a trunk
    must not grow out of a building.
    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    ztop = float(np.percentile(z, 98))
    zbase = max(float(np.percentile(z, 5)), ground_z + 1.0)
    if ztop - ground_z < 2.5 or ztop <= zbase:
        return []
    cx, cy = stem if stem is not None else (float(np.median(x)),
                                            float(np.median(y)))
    h = ztop - ground_z
    rb, rt = 0.15 + 0.025 * h, 0.06 + 0.012 * h
    zt = zbase + (ztop - zbase) * 0.35
    K = 8
    ang = np.linspace(0, 2 * np.pi, K, endpoint=False)
    quads = []
    for k in range(K):
        a0, a1 = ang[k], ang[(k + 1) % K]
        pa, pb = (np.cos(a0), np.sin(a0)), (np.cos(a1), np.sin(a1))
        quads += [(cx + pa[0] * rb, cy + pa[1] * rb, ground_z),
                  (cx + pb[0] * rb, cy + pb[1] * rb, ground_z),
                  (cx + pb[0] * rt, cy + pb[1] * rt, zt),
                  (cx + pa[0] * rb, cy + pa[1] * rb, ground_z),
                  (cx + pb[0] * rt, cy + pb[1] * rt, zt),
                  (cx + pa[0] * rt, cy + pa[1] * rt, zt)]
    soup = np.asarray(quads, np.float32)
    return [(soup, np.tile(np.array([98, 76, 54], np.uint8), (len(soup), 1)))]


def lowveg_graph(x, y, z, rgb=None, *, cell=0.15, k=6, max_edge=1.2,
                 width=0.03):
    """Low vegetation as its own kNN GRAPH, drawn as thin ribbons.

    Kaveh's model (2026-08-01): a shrub has no surface worth inventing, so it
    is rendered as the graph of its own thinned returns — 15 cm cells because
    small vegetation is too sparse otherwise, k=6, and no edge ever bridges a
    gap wider than ``max_edge``. Edges become narrow ribbons so they render and
    pick through the same triangle path as everything else.
    """
    from scipy.spatial import cKDTree

    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    key = (np.floor(x / cell).astype(np.int64) * 73856093
           ^ np.floor(y / cell).astype(np.int64) * 19349663
           ^ np.floor(z / cell).astype(np.int64) * 83492791)
    _u, keep = np.unique(key, return_index=True)
    if len(keep) < 4:
        return []
    P = np.column_stack([x[keep], y[keep], z[keep]])
    C = (np.asarray(rgb)[keep] if rgb is not None
         else np.tile(np.array([110, 140, 80], np.uint8), (len(P), 1)))
    d, nb = cKDTree(P).query(P, k=min(k + 1, len(P)))
    a = np.repeat(np.arange(len(P)), nb.shape[1] - 1)
    b, dd = nb[:, 1:].ravel(), d[:, 1:].ravel()
    a, b = a[dd <= max_edge], b[dd <= max_edge]
    if not len(a):
        return []
    uniq = np.unique(np.minimum(a, b) * len(P) + np.maximum(a, b))
    a, b = uniq // len(P), uniq % len(P)
    A, B = P[a], P[b]
    u = np.cross(B - A, [0.0, 0.0, 1.0])
    un = np.linalg.norm(u, axis=1, keepdims=True)
    u = np.where(un > 1e-6, u / np.maximum(un, 1e-9), [1.0, 0.0, 0.0]) * width
    soup = np.stack([A - u, A + u, B + u, A - u, B + u, B - u], 1).reshape(-1, 3)
    cols = np.stack([C[a], C[a], C[b], C[a], C[b], C[b]], 1).reshape(-1, 3)
    return [(soup.astype(np.float32), cols.astype(np.uint8))]


TREE_MAX_H = 35.0    # m — no urban tree here is taller
TREE_MAX_W = 20.0    # m — wider than this is merged canopy, not one tree
TREE_MAX_BASE = 10.0  # m — a crown starts a few metres up; 16 m is a roof


def looks_like_tree(x, y, z, ground_z):
    """Is this crown a TREE? — the type's own test, measured on 2,981 of them.

    `veg_high` says the matter is foliage-like; it does not say the thing is a
    tree. A tree is ROOTED IN THE GROUND, is one crown rather than a merged
    canopy, and is not taller than trees grow here. Measured [2026-08-05]:

        height above ground  p05  3.6  p50  8.9  p95 21.7  max 70.9
        crown width          p05  2.7  p50  6.0  p95 13.8  max 41.1
        base above ground    p05  1.9  p50  3.0  p95  8.1  max 68.9

    The bulk is a street tree; the tails are not. Object 2730 was 1.9 x 2.0 m
    in plan starting 16.3 m above the ground with 8,868 building returns within
    8 m — a sliver on a roof, published as a 21.4 m tree.

    The base bound is 10 m and not 3 m on purpose: a crown legitimately begins
    several metres up because the trunk returns few points, and p95 of the real
    population is 8.1 m. At 3 m this would reject half the trees on the tile.
    """
    z = np.asarray(z, float)
    top = float(np.percentile(z, 98)) - ground_z
    base = float(np.percentile(z, 2)) - ground_z
    wide = max(float(np.ptp(np.asarray(x, float))),
               float(np.ptp(np.asarray(y, float))))
    return (top <= TREE_MAX_H and wide <= TREE_MAX_W
            and base <= TREE_MAX_BASE and top >= 2.0)


CAR_VEG = 0.5        # of what surrounds it is canopy -> it IS canopy
CAR_GREEN = 30.0     # excess green above which a "car" is foliage


def looks_like_car(x, y, z, veg_frac=None, exg=None):
    """Is this a CAR? Dimensions, and then the two things a box test misses.

    v2's label set has no CAR class — stage 5 decides `small` and `on_bridge`,
    and the split between a car, a street light and a bin is made here. The
    test used to be three numbers and nothing else::

        2.5 <= L <= 6.5 and 1.4 <= W <= 2.6 and 1.2 <= H <= 2.2

    which makes ANY blob of that size a car. Measured [2026-08-05]: instance
    8125 was 3.9 x 2.0 x 2.4 m standing 14.9 m above the ground, and it was a
    clump of tree canopy — 83% of the returns within 6 m of it were `veg_high`.
    Kaveh: *"you didn't good enough guard for car labeling"*.

    So two context guards, both measured on this tile:

    * ``veg_frac`` — how much of what surrounds it is canopy. A car parked in a
      street is not inside a tree; the offender scored 0.83.
    * ``exg`` — its own excess green. Weak alone (veg_high p50 15 against
      `small` p50 0, and the offender read 24, inside the overlap), so it only
      rejects the strongly green.

    Both are optional: with neither, this is the old geometric test, which is
    what a caller that has no labels or colour can honestly do.
    """
    mu, eu, ev, u, v = pca_frame(x, y)
    L, W = float(np.ptp(u)), float(np.ptp(v))
    H = float(np.percentile(z, 95) - np.percentile(z, 2))
    if not (2.5 <= L <= 6.5 and 1.4 <= W <= 2.6 and 1.2 <= H <= 2.2):
        return False
    if veg_frac is not None and veg_frac > CAR_VEG:
        return False
    if exg is not None and exg > CAR_GREEN:
        return False
    return True


def bridge_lights(x, y, z, band=2.0, max_plan=3.5, head_lo=4.0,
                  head_hi=15.0, link=1.0, consensus=2.0, min_pts=12):
    """STREET LIGHTS ON A BRIDGE — the lab's recipe (``_recon.py:2240``).

    A lamp on a bridge is not found the way a lamp on a street is. Its datum is
    the DECK, 30 m up, so every height test against the ground is meaningless:
    measured, a 10 m standard on Granville reads as 45 m and fails
    :func:`looks_like_pole` outright, which is why this project built ONE light
    where the lab builds ~22 (Kaveh, 2026-08-06).

    The lab's tests, ported with its measurements intact:

    * cluster what stands more than ``band`` above the deck, in plan
    * a lamp is at most ``max_plan`` = 3.5 m across — wider is a sign gantry or
      a cable. (This project allowed 1.2 m, which no lamp head passes.)
    * its head stands ``head_lo``..``head_hi`` = 4..15 m above the deck, and the
      datum comes from the DECK RETURNS BESIDE IT, never a grid: the grid lies
      at the deck edges, and reading 31 m under a 37 m deck turned two 3 m
      rail-top blobs into "7 m lights" [Kaveh, 2026-08-01]
    * **consensus** — with 5 or more candidates, heads agree on one bridge, so
      anything more than ``consensus`` m from the median head is a wire scrap
      or an air blip, not a lamp

    Returns ``[(mask, base_z, head), ...]`` over the input arrays.
    """
    from scipy import ndimage as ndi

    x, y, z = (np.asarray(a, float) for a in (x, y, z))
    if len(z) < min_pts:
        return []
    # the deck under each point: the low band of its own neighbourhood
    gx = ((x - x.min()) / 2.0).astype(int)
    gy = ((y - y.min()) / 2.0).astype(int)
    nx = gx.max() + 1
    cell = gy * nx + gx
    dmin = np.full((gy.max() + 1) * nx, np.inf)
    np.minimum.at(dmin, cell, z)
    above = z > dmin[cell] + band
    if above.sum() < min_pts:
        return []
    ax, ay, az = x[above], y[above], z[above]
    ix = ((ax - ax.min()) / link).astype(int)
    iy = ((ay - ay.min()) / link).astype(int)
    occ = np.zeros((iy.max() + 1, ix.max() + 1), bool)
    occ[iy, ix] = True
    lab_, n = ndi.label(occ, np.ones((3, 3)))
    cid = lab_[iy, ix]
    cands = []
    for k in range(1, n + 1):
        cm = cid == k
        if cm.sum() < min_pts:
            continue
        if max(float(np.ptp(ax[cm])), float(np.ptp(ay[cm]))) > max_plan:
            continue                      # a sign gantry or a cable, not a lamp
        cxm, cym = float(np.median(ax[cm])), float(np.median(ay[cm]))
        near = (np.abs(x - cxm) < 8) & (np.abs(y - cym) < 8) & ~above
        if near.sum() < 30:
            continue
        zd = float(np.median(z[near]))    # the deck BESIDE it, not a grid
        head = float(np.percentile(az[cm], 98)) - zd
        if not (head_lo <= head <= head_hi):
            continue
        base = (np.abs(x - cxm) < 4) & (np.abs(y - cym) < 4) & ~above
        cands.append((cm, float(np.median(z[base])) if base.sum() >= 10 else zd,
                      head))
    if len(cands) >= 5:                   # one bridge, one lamp model
        med = float(np.median([h for _, _, h in cands]))
        cands = [c for c in cands if abs(c[2] - med) <= consensus]
    out = []
    for cm, zb, head in cands:
        full = np.zeros(len(z), bool)
        full[np.flatnonzero(above)[cm]] = True
        out.append((full, zb, head))
    return out


def looks_like_pole(x, y, z):
    """Pole-shaped? taller than 2.5 m and under 1.2 m across in plan."""
    H = float(np.percentile(z, 98) - np.percentile(z, 2))
    return H > 2.5 and max(float(np.ptp(x)), float(np.ptp(y))) < 1.2


# ===========================================================================
# ONE NAMED MODEL PER OBJECT TYPE (Kaveh's rule, 2026-08-05)
#
# Every sub-label in the stage-6 taxonomy has its own entry point below, and
# **no model calls another model**. Where two types genuinely share geometry
# they share a PRIMITIVE — `box_local`, `ring_prism`, `cell_prism`,
# `hull_soup`, `drape`, `scene.footprint_prism` — never each other's recipe.
# Each returns `[(soup, cols, kind), ...]` and decides its own KIND, because
# what the sun does with a thing is a property of what the thing IS.
#
# The audit this replaces (2026-08-05) found four borrowings: `bridge` used
# `solid_25d` from BELOW THE WATERLINE (v1's nDSM blackout, reintroduced),
# `small` and `on_bridge` both used `car_mesh` (947 objects drawn as cars
# after `looks_like_car` said they were not cars), and `glass` used
# `footprint_prism` (a solid duplicate of its own host building).
# ===========================================================================


def ground_from_returns(ring, px, py, pz, k=25, fallback=None):
    """Height for each ring vertex from the surface's OWN ground returns.

    A named ground surface — a road, a footway, a pitch — is drawn from the
    returns stage 6 gave it, so those returns are what it stands on. Sampling
    a DEM instead lets the DEM's mistakes become terrain: stage 5's ground
    grid has the bridges baked into it (41,886 cells inside a bridge footprint
    read a median of 29.69 m), and draping on it drew a 13 m2 road whose own
    returns lie at -0.40..0.97 m as a surface at 38.78 m, plus a brown wall
    climbing to the deck. Evidence cannot make that mistake: the median of the
    ``k`` nearest returns is a height the survey actually saw. [2026-08-05]
    """
    from scipy.spatial import cKDTree

    ring = np.asarray(ring, float)
    P = np.column_stack([np.asarray(px, float), np.asarray(py, float)])
    if len(P) < k:
        if fallback is None:
            return None
        return np.full(len(ring), float(fallback))
    _d, ii = cKDTree(P).query(ring, k=k)
    return np.median(np.asarray(pz, float)[ii], axis=1)


def drape(rings, zs, col=None):
    """Ring(s) triangulated flat at given heights — a SURFACE, no volume.

    The shared primitive of every "ground with a material" recipe: water,
    roads, parks, land use. ``rings`` is a list of (n, 2) outlines (the first
    is the shell, the rest are holes, so a park does not paint over the
    building standing in it); ``zs`` is one height, or one height per vertex
    when the surface follows the terrain. Returns a bare soup, no kind — the
    type that called it says what it is.
    """
    import mapbox_earcut

    rings = [np.asarray(r, float) for r in rings]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return None
    verts = np.vstack(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    try:
        tri = mapbox_earcut.triangulate_float64(verts, ends).reshape(-1, 3)
    except Exception:
        return None
    if not len(tri):
        return None
    zz = np.broadcast_to(np.asarray(zs, float), (len(verts),))
    return np.column_stack([verts, zz])[tri.ravel()].astype(np.float32)


def _tint(soup, col, default):
    col = np.asarray(default if col is None else col, np.uint8)
    return np.tile(col, (len(soup), 1))


# ---- structure -----------------------------------------------------------

def building_model(ring, z0, ztop, col=None, mesh=None,
                   x=None, y=None, z=None, pix=1.0):
    """A BUILDING: a closed envelope from the ground to its measured roof.

    Light stops at a house — every face of it — so the kind is ``solid``, and
    the geometry has to be closed. Three sources, in order of what they know:

    1. **A roofer LoD2.2 solid** where one stands in this footprint. That solid
       IS the building — ridges, hips, dormers, fitted from the same returns.
    2. **The returns themselves**, snapped onto their own roof planes
       (:func:`fit_roof_planes`) and clipped to the surveyed outline. This is
       new [2026-08-05]: only 152 roofer solids survived the run that produced
       them and 87 are used, so most buildings fell to (3) — and (3) is a BOX.
       Kaveh: *"some part of that building is not [there]... the lab version
       better handled some buildings"*. It did, because the lab has a roofer
       solid for every building and we do not; a plane-fitted roof is the
       honest way to close that gap without inventing a primitive.
    3. **The outline and two heights** — `scene.footprint_prism`, a flat top.
       Only where the returns cannot say more.

    **Rooftop detail is NOT bolted on afterwards.** Clustering the returns that
    stand above the P95 and extruding each as its own little solid was tried
    and rejected [Kaveh, 2026-08-06: "post processing is not good idea at least
    in this way"]. It put 45 blocks on 83 buildings and read as debris, because
    a plant room modelled apart from its roof is not the same thing as a roof
    that has a plant room on it. If the detail matters, the ROOF has to carry
    it — one surface measured over the whole footprint — not the body plus
    afterthoughts.

    No bottom cap: the prism sits on ground, where no return ever lands.
    """
    from .scene import footprint_prism

    if mesh is not None and len(mesh):
        # THE MAP OWNS THE PLAN, THE SURVEY OWNS THE HEIGHT, AND ROOFER OWNS
        # THE ROOF **ONLY WHEN IT HAS ONE** (Kaveh, 2026-08-06).
        #
        # Passing roofer's solid through whole shipped roofer's walls as ours,
        # and they have holes: the Granville silo 8dc75f68 had three angular
        # gaps over 30 deg, one of 64 deg — a 3.7 m chord of absent wall — over
        # a footprint roofer covered to 85%. Six of 82 buildings fall under
        # 85%, the worst 888 m2 of building against 104 m2 of roofer.
        #
        # Sampling the wall height from roofer's ROOF failed too, and worse:
        # measured over all 152 solids, the roof surface spans a median 33% of
        # the building's height and **54 of them span more than half**, the
        # worst 99%. That is not a roof, and the silo's wall came out at 18 m
        # on one side and 26.8 m on the other because of it.
        #
        # So the body is the surveyed outline extruded to the MEASURED top, and
        # roofer's roof rides on it only where it is thin enough to be a roof.
        M = np.asarray(mesh, float).reshape(-1, 3, 3)
        nrm = np.cross(M[:, 1] - M[:, 0], M[:, 2] - M[:, 0])
        cz = nrm[:, 2] / np.maximum(np.linalg.norm(nrm, axis=1), 1e-9)
        zc = M[:, :, 2].mean(1)
        roof = M[cz > 0.5]
        # KEEP THE ROOF, ALWAYS. The span test existed because the wall height
        # was sampled FROM the roof, so a ragged roofer surface made a ragged
        # wall. The wall now goes to the measured top instead, which makes
        # roofer's roof purely additive: triangles below the prism's top are
        # inside a closed solid and invisible, and those above are the ridges,
        # hips and dormers that are the whole reason to use roofer. Dropping
        # them cost a university block its roof detail (Kaveh, 2026-08-06).
        keep_roof = len(roof) >= 2
        from .scene import footprint_prism
        body, bcols = footprint_prism(ring, z0, ztop, col=col)
        if not len(body):
            return []
        soup = np.asarray(body, np.float32)
        if keep_roof:
            soup = np.vstack([soup, roof.reshape(-1, 3).astype(np.float32)])
        return [(soup, _tint(soup, col, (170, 165, 158)), SOLID)]
    if ztop - z0 < 2.0:
        return []
    if x is not None and len(np.asarray(x)) >= 60:
        got = top_surface(x, y, z, pix)
        if got is not None:
            import shapely
            top, occ, lo = got
            fitted, nplanes = fit_roof_planes(top, occ, pix)
            gy, gx = np.mgrid[0:top.shape[0], 0:top.shape[1]]
            occ = occ & shapely.contains_xy(
                shapely.Polygon(ring), lo[0] + (gx + 0.5) * pix,
                lo[1] + (gy + 0.5) * pix)
            if occ.sum() >= 4:
                # THE WALLS COME FROM THE OUTLINE, THE ROOF FROM THE RETURNS.
                # cell_prism extrudes every 1 m cell from the ground, so a building whose
                # footprint runs diagonally (bearings 39 deg / 129 deg on railspur) came out as an
                # axis-aligned voxel staircase: 160 wall triangles whose normals were all
                # exactly 0 or 90 deg, grouped into 38 "facades" of which 13 were 1 m slivers.
                # The prism gives the true wall planes; the fitted cells ride on top as the
                # roof only, which is the same division the roofer branch above makes.
                roof = cell_prism(np.maximum(fitted, z0 + 0.05), z0, occ, lo, pix)
                body, bcols = footprint_prism(ring, z0, ztop, col=col)
                if len(body):
                    soup = np.asarray(body, np.float32)
                    if len(roof):
                        R = np.asarray(roof, np.float32).reshape(-1, 3, 3)
                        nz = np.cross(R[:, 1] - R[:, 0], R[:, 2] - R[:, 0])[:, 2]
                        up = nz / np.maximum(np.linalg.norm(np.cross(R[:, 1] - R[:, 0], R[:, 2] - R[:, 0]), axis=1), 1e-9)
                        soup = np.vstack([soup, R[up > 0.5].reshape(-1, 3)])   # the roof faces only
                    return [(soup, _tint(soup, col, (170, 165, 158)), SOLID)]
    soup, cols = footprint_prism(ring, z0, ztop, col=col)
    return [(soup, cols, SOLID)] if len(soup) else []


def moorage_model(ring, wlvl, ztop, col=None):
    """A FLOAT HOME: a house whose ground floor is the sea.

    Same closed envelope as a building — it is opaque, kind ``solid`` — but
    its base is the WATERLINE, never the DEM. The DEM under a float home reads
    the float home's own deck (that is why stage 6 typed 0 moorages while 21
    footprints sat entirely on water), and a terrain base would sink the walls
    to the seabed. Lab constant: base at ``wlvl - 0.25`` (`float_home_mesh`).
    """
    from .scene import footprint_prism

    z0 = wlvl - 0.25
    if ztop - z0 < 1.0:
        return []
    soup, cols = footprint_prism(ring, z0, ztop, col=col)
    return [(soup, cols, SOLID)] if len(soup) else []


def glass_model(x, y, z, col=None, cell=0.75, min_n=3):
    """GLAZING: zero-thickness panes lying on the facet the returns lie on.

    A curtain wall is a sub-metre skin on one face of a building that already
    exists as its own solid, and a skylight is a patch of the roof it sits in
    — neither is a volume, so a prism was wrong three times over: it duplicated
    the host, it claimed the host's whole outline (the map polygon owns >50 %
    of the glazing returns), and it made a transmissive thing a full blocker.

    So the pane is built where the glass WAS MEASURED, in the plane it was
    measured in: PCA gives the facet's own frame, the returns are gridded at
    ``cell`` metres INSIDE that plane, and every cell holding at least
    ``min_n`` returns becomes one flat quad at that cell's mean offset. Works
    the same for a vertical facade and a horizontal skylight, and follows a
    stepped or curved wall because each cell keeps its own offset. Lab
    constants from `_roof_glass`: 0.75 m cells, >= 3 glazing points.

    Kind ``glass``: no volume, and it transmits.
    """
    P = np.column_stack([np.asarray(x, float), np.asarray(y, float),
                         np.asarray(z, float)])
    if len(P) < 2 * min_n:
        return []
    mu = P.mean(0)
    d = P - mu
    _w, V = np.linalg.eigh(d.T @ d)
    nrm, e2, e1 = V[:, 0], V[:, 1], V[:, 2]      # smallest eigenvector = normal
    u, v, o = d @ e1, d @ e2, d @ nrm
    iu = np.floor(u / cell).astype(np.int64)
    iv = np.floor(v / cell).astype(np.int64)
    u0, v0 = iu.min(), iv.min()
    nu = int(iu.max() - u0) + 1
    key = (iu - u0) + (iv - v0) * nu
    cnt = np.bincount(key)
    off = np.bincount(key, weights=o)
    keep = np.flatnonzero(cnt >= min_n)
    if not len(keep):
        return []
    cu = ((keep % nu) + u0 + 0.5) * cell
    cv = ((keep // nu) + v0 + 0.5) * cell
    co = off[keep] / cnt[keep]
    h = cell / 2
    quad = np.array([(-h, -h), (h, -h), (h, h),
                     (-h, -h), (h, h), (-h, h)])
    verts = (mu
             + (cu[:, None, None] + quad[None, :, 0, None]) * e1
             + (cv[:, None, None] + quad[None, :, 1, None]) * e2
             + co[:, None, None] * nrm)
    soup = verts.reshape(-1, 3).astype(np.float32)
    return [(soup, _tint(soup, col, WINCOL), GLASS)]


def looks_like_deck(x, y, z, ratio=3.0, min_plan=5.0):
    """Is this bridge instance a DECK, or something standing on one?

    `bridge` is a label about MATTER — it is the bridge's stuff — so a light
    standard, a gantry and a truss portal all carry it, and stage 7 swept each
    one into a "deck". Measured on Granville [2026-08-05]: object 1478 was
    9.3 m across and 10.4 m TALL, sitting 7.4 m above the carriageway beside
    it, and shipped as a deck 37.4..48.1 m thick.

    A carriageway is broad and flat. Across every bridge instance on this tile
    the two populations do not touch: real decks score plan/vertical of
    5.8 to 22.3, the three impostors score 1.2, 0.9 and 0.5. Anything that
    fails this is not a deck — it STANDS on one, which is what `on_bridge`
    means.
    """
    plan = max(float(np.ptp(np.asarray(x, float))),
               float(np.ptp(np.asarray(y, float))))
    vert = float(np.ptp(np.asarray(z, float)))
    return plan >= min_plan and plan >= ratio * max(vert, 1e-9)


def chain_ways(ways, tol=1.0):
    """Join polylines that share an endpoint (within ``tol`` m) into chains.

    A bridge arrives from the map as several ways; swept separately they make
    several decks with seams between them. The lab chains them first
    (``_recon.py:_chain_ways``) so one bridge is one ribbon, and that is what
    makes its Granville a single continuous deck instead of our 7 fragments.
    """
    ch = [list(map(tuple, np.asarray(w, float))) for w in ways if len(w) >= 2]
    d = lambda p, q: float(np.hypot(p[0] - q[0], p[1] - q[1]))
    changed = True
    while changed:
        changed = False
        for a in range(len(ch)):
            if not ch[a]:
                continue
            for b in range(len(ch)):
                if a == b or not ch[b]:
                    continue
                A, B = ch[a], ch[b]
                if d(A[-1], B[0]) <= tol:
                    ch[a] = A + B[1:]
                elif d(A[-1], B[-1]) <= tol:
                    ch[a] = A + B[::-1][1:]
                elif d(A[0], B[0]) <= tol:
                    ch[a] = A[::-1] + B[1:]
                elif d(A[0], B[-1]) <= tol:
                    ch[a] = B + A[1:]
                else:
                    continue
                ch[b] = []
                changed = True
    return [np.asarray(c, float) for c in ch if len(c) >= 2]


def bridge_ribbon(chain, x, y, z, thickness=None, step=3.0, fixed_half=None,
                  smooth=15, lat_max=45.0, min_pts=40, ground_z=None,
                  bents=False, bent_every=20, bent_taper=0.8, bent_along=2.4,
                  bent_across=0.62, bent_min_clear=3.0, rail=False):
    """A deck SWEPT along its centreline — the lab's recipe, and the right one.

    Building a deck by rasterising its footprint cannot work: a polygon is
    larger than the evidence wherever the laser was thin, so the unsupported
    cells are either dropped (the deck shatters) or kept (it invents surface).
    Dropping them turned Granville into a field of 2 m cubes with a ribbon on
    top [Kaveh, 2026-08-05]. A road has no such failure mode when it is swept:
    the centreline is continuous by construction, and a station with too few
    returns is INTERPOLATED between its neighbours rather than deleted.

    Per ``step`` metres along the chain, from the returns within ``lat_max``:

    * height  — P82 of z, then median- AND mean-filtered over ``smooth``
      stations. 15 stations is 45 m at the 3 m step: a bridge changes grade
      over tens of metres, and the lab's 9 left the profile changing slope
      several times along one span (Kaveh, 2026-08-05)
    * edges   — P0.5 and P99.5 of the LATERAL offset, left and right kept
      separately, because Granville's east sidewalk cantilevers 4.6 m further
      than its west and a symmetric width leaves that row of street lights
      hanging in the air (the lab measured this: deck -12.0..+16.6 m)
    * width   — ONE median edge per corridor, not per station: Kaveh's ruling,
      2026-07-31, "a deck's width does not wobble bin to bin"

    Emits the slab pair (carriageway, underside ``thickness`` below) with both
    fascias closed. ``fixed_half`` overrides the measurement for ramps, which
    are too narrow to measure against a wide window without borrowing the main
    deck's returns.
    """
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree

    t = DECK_T if thickness is None else float(thickness)
    C = np.asarray(chain, float)
    seg = np.hypot(*np.diff(C, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 20.0:
        return None
    ss = np.arange(0.0, s[-1] + step / 2, step)
    cx, cy = np.interp(ss, s, C[:, 0]), np.interp(ss, s, C[:, 1])
    tx, ty = np.gradient(cx), np.gradient(cy)
    tl = np.maximum(np.hypot(tx, ty), 1e-9)
    tx, ty = tx / tl, ty / tl
    nx, ny = -ty, tx
    P = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    z = np.asarray(z, float)
    tree = cKDTree(P)
    half = (fixed_half + 2.0) if fixed_half else lat_max
    zs = np.full(len(ss), np.nan)
    lo = np.full(len(ss), np.nan)
    hi = np.full(len(ss), np.nan)
    for k in range(len(ss)):
        q = np.asarray(tree.query_ball_point([cx[k], cy[k]], half + 2.0), int)
        if len(q) < min_pts:
            continue
        lat = (P[q, 0] - cx[k]) * nx[k] + (P[q, 1] - cy[k]) * ny[k]
        lon = (P[q, 0] - cx[k]) * tx[k] + (P[q, 1] - cy[k]) * ty[k]
        m = (np.abs(lon) < step) & (np.abs(lat) < half)
        if m.sum() < min_pts:
            continue
        # A MAST IS NOT THE CARRIAGEWAY. Light standards, gantries and the
        # truss reach 8-16 m over the deck and sit inside the same lateral
        # window, so P82 jumped at whichever station happened to hold one and
        # the grade came out sawtoothed — measured, the slope changed SIGN 7
        # times and swung by up to 0.43 between adjacent 5 m bins, on a road
        # whose real grade is about 0.05. Drop what stands clear above the
        # station's own median before taking the deck height. [2026-08-05]
        zv = z[q[m]]
        zv = zv[zv < float(np.median(zv)) + 3.0]
        if len(zv) < min_pts // 2:
            continue
        zs[k] = np.percentile(zv, 82)
        lo[k] = np.percentile(lat[m], 0.5)
        hi[k] = np.percentile(lat[m], 99.5)
    ok = ~np.isnan(zs)
    if ok.sum() < 8:
        return None
    a, b = int(np.argmax(ok)), len(ok) - int(np.argmax(ok[::-1])) - 1
    sl = slice(a, b + 1)
    zz = zs[sl]
    good = ~np.isnan(zz)
    zz = np.interp(np.arange(len(zz)), np.flatnonzero(good), zz[good])
    # median first, mean second: a median removes a station the outlier guard
    # above still let through, a mean alone only smears it over its neighbours.
    zz = ndi.median_filter(zz, size=max(3, (int(smooth) // 2) * 2 + 1),
                           mode="nearest")
    zz = ndi.uniform_filter1d(zz, max(3, int(smooth)), mode="nearest")
    if fixed_half:
        wlo, whi = -float(fixed_half), float(fixed_half)
    else:
        # +-32, not the lab's +-19. Its OSM line is the roadway CENTRE, so a
        # symmetric-ish clip fits; ours is the longest of several parallel ways
        # merged into one bridge, so the deck sits far off to one side.
        # Measured on Granville [2026-08-06]: the deck runs -2.8 to +30.0 m
        # about this chain, and a 20 m window with a 19 m clip cut 10 m off the
        # east side — which is precisely where the cantilevered sidewalk is, so
        # its whole row of street lights stood in the air with no deck beneath.
        # Kaveh said this in the lab on 2026-08-01 and again here: "one side
        # all streetlight are on fly".
        wlo = float(np.clip(np.nanmedian(lo[sl]) - 0.4, -32.0, -3.0))
        whi = float(np.clip(np.nanmedian(hi[sl]) + 0.4, 3.0, 32.0))
    X, Y = cx[sl], cy[sl]
    NX, NY = nx[sl], ny[sl]

    def at(k, off, zc):
        return np.array([X[k] + NX[k] * off, Y[k] + NY[k] * off, zc])

    soup = []

    def quad(p, q, r, w):
        soup.extend([p, q, r, p, r, w])

    for k in range(len(zz) - 1):
        L0, R0 = at(k, wlo, zz[k]), at(k, whi, zz[k])
        L1, R1 = at(k + 1, wlo, zz[k + 1]), at(k + 1, whi, zz[k + 1])
        D = np.array([0.0, 0.0, t])
        quad(L0, R0, R1, L1)                      # the carriageway
        quad(L1 - D, R1 - D, R0 - D, L0 - D)      # the underside
        quad(L0, L1, L1 - D, L0 - D)              # west fascia
        quad(R1, R0, R0 - D, R1 - D)              # east fascia
    deck = np.asarray(soup, np.float32) if soup else None
    if deck is None:
        return None
    # GUARD RAILS, measured not mapped. Neither OSM nor Overture carries them;
    # the returns do — the lab measured 1.26 m over the deck on both sides,
    # present at 56/61% of stations, and renders the west band as glass because
    # it holds 1,785 glass returns (the sidewalk windscreen) against zero east.
    # A rail is a real object to the sun: it is what stands between a low
    # winter sun and the carriageway. [2026-08-05]
    rails = []
    if rail and not fixed_half:
        for k in range(len(zz) - 1):
            for off, side in ((wlo, "west"), (whi, "east")):
                a0, a1 = at(k, off, zz[k]), at(k + 1, off, zz[k + 1])
                h = np.array([0.0, 0.0, RAIL_H])
                rails.append((np.asarray(
                    [a0, a1, a1 + h, a0, a1 + h, a0 + h], np.float32), side))
    rail_parts = []
    if rails:
        for side, kind in (("west", GLASS), ("east", SOLID)):
            R = [r for r, sd in rails if sd == side]
            if R:
                RS = np.vstack(R).astype(np.float32)
                rail_parts.append((RS, _tint(RS, None, RAIL_COL), kind))
    if not bents or ground_z is None:
        return (deck, [], rail_parts) if rail else deck
    # THE LAB'S BENTS, in the frame that defines them (``_recon.py:1587``).
    # They have to be built HERE, where wlo/whi/z2 and the station frame are in
    # scope: a bent is `hw * 0.62` across the DECK and 2.4 m along it, and
    # re-deriving the half-width from the finished soup gave 2.2 m instead of
    # 18 m. One every ``bent_every`` stations = 60 m at the lab's 3 m step,
    # never within 4 stations of an end, tapered 0.8 to the footing, under the
    # deck CENTRE (mid of the asymmetric corridor, not the map line).
    hw = (whi - wlo) / 2
    mid = (wlo + whi) / 2
    TXs, TYs = tx[sl], ty[sl]
    bent = []
    for k in range(4, len(zz) - 4, int(bent_every)):
        ztop = zz[k] - t
        if ztop - ground_z < bent_min_clear:
            continue
        pw, pd = hw * bent_across, bent_along
        cor = [(-pd, mid - pw), (pd, mid - pw), (pd, mid + pw), (-pd, mid + pw)]
        ct = [np.array([X[k] + TXs[k] * a_ + NX[k] * b_,
                        Y[k] + TYs[k] * a_ + NY[k] * b_, ztop])
              for a_, b_ in cor]
        cb = [np.array([X[k] + TXs[k] * a_ * bent_taper
                        + NX[k] * (mid + (b_ - mid) * bent_taper),
                        Y[k] + TYs[k] * a_ * bent_taper
                        + NY[k] * (mid + (b_ - mid) * bent_taper),
                        ground_z]) for a_, b_ in cor]
        for q in range(4):
            a2, b2 = ct[q], ct[(q + 1) % 4]
            a0, b0 = cb[q], cb[(q + 1) % 4]
            bent += [a2, b2, b0, a2, b0, a0]
    bent_parts = ([] if not bent else
                  [(np.asarray(bent, np.float32),
                    _tint(np.asarray(bent, np.float32), None, PIER_COL),
                    SOLID)])
    return (deck, bent_parts, rail_parts) if rail else (deck, bent_parts)


def bridge_deck_model(x, y, z, col=None, pix=1.0, thickness=None):
    """A BRIDGE DECK: a SLAB — the road surface and its underside, air below.

    A deck's defining physical fact is the void it spans. Extruded from the
    waterline it is an opaque wall across the navigation channel: measured,
    Granville Bridge as a slab blocks only above 72.6 deg, an angle the
    Vancouver sun never reaches, while the same deck modelled solid shades the
    waterway all day — v1's original nDSM bug, and the reason the slab pair
    exists at all.

    The deck surface comes from the returns (`top_surface`, so a crowned or
    ramped deck keeps its own profile) and the underside is that surface
    lowered by a MEASURED thickness: the girder and fascia returns that fell
    below the deck plane, at their P90, clipped to 0.6-3.0 m. With no such
    evidence it falls back to the lab's Granville measurement,
    ``DECK_T = 1.6`` m [2026-08-01]. Nothing is built between the underside
    and the ground — the piers are their own objects
    (:func:`bridge_support_model`).
    """
    from scipy import ndimage as ndi

    got = top_surface(x, y, z, pix)
    if got is None:
        return []
    top, occ, lo = got
    # A deck is SMOOTH. `top_surface` takes the per-cell max, so a lamp mast
    # puts the "deck" 12 m above the road (measured: bridge-labelled returns
    # reach 49.5 m over a 32 m deck) and a pier flank puts it at 2 m wherever
    # no deck return landed in that cell. Both are one cell wide against a
    # neighbourhood that is not, so compare each cell with the median of the
    # ~15 m around it: a ramp carries its neighbourhood with it and survives,
    # a spike does not. [2026-08-05]
    med = ndi.median_filter(top, size=int(round(FLAT_WIN / pix)) | 1)
    occ = occ & (np.abs(top - med) < FLAT_TOL)
    if not occ.any():
        return []
    z = np.asarray(z, float)
    gx = np.clip(((np.asarray(x, float) - lo[0]) / pix).astype(int),
                 0, top.shape[1] - 1)
    gy = np.clip(((np.asarray(y, float) - lo[1]) / pix).astype(int),
                 0, top.shape[0] - 1)
    dz = top[gy, gx] - z
    under = dz[(dz > 0.3) & (dz < 8.0)]
    if thickness is None:
        thickness = (float(np.percentile(under, 90)) if len(under) >= 20
                     else DECK_T)
    t = float(np.clip(thickness, 0.6, 3.0))
    # The polygon path that used to live here rasterised the footprint and
    # emitted a prism per cell, which shattered into 2 m cubes wherever the
    # outline was larger than the evidence. A deck is SWEPT, not rasterised —
    # see :func:`bridge_ribbon`, which is what stage 7 calls. This remains the
    # recipe for a deck with no map line at all. [2026-08-05]
    soup = cell_prism(top, top - t, occ, lo, pix)
    if not len(soup):
        return []
    return [(soup, _tint(soup, col, DECK_COL), SLAB)]


def bridge_bents(chain, deck, ground_z, spacing=60.0, thickness=None,
                 min_clear=3.0, taper=0.8, along=2.4, across=0.62, col=None):
    """Supports placed by RULE along the deck's axis — the lab's recipe.

    Kaveh, 2026-08-05: *"what you add as support for the bridge doesn't have
    any evidence, so for this case we should use lab approach"*. He is right
    about the evidence. What the return-driven detector found under Granville
    was long, thin, vertical bands 44-55 m long and 4.7 m wide — that is the
    deck's own FASCIA caught obliquely, not a column, and no amount of
    filtering makes a fascia into a pier. An airborne survey simply does not
    see under its own deck.

    So the supports are stated, not claimed, exactly as the lab states them
    (``_recon.py:1587``): one bent every ``spacing`` metres along the chain,
    skipped where the clearance is under ``min_clear``, sized from the deck's
    own half-width (``across``) and tapered ``taper`` toward the footing. This
    is an honest invention and the docstring is where it is declared: the
    geometry says "a bridge this long has piers about here", which is true of
    every bridge, rather than "the laser saw concrete here", which was not.
    """
    from scipy.spatial import cKDTree

    t = DECK_T if thickness is None else float(thickness)
    C = np.asarray(chain, float)
    D = np.asarray(deck, float)
    if len(C) < 2 or len(D) < 3:
        return []
    seg = np.hypot(*np.diff(C, axis=0).T)
    s_ = np.concatenate([[0.0], np.cumsum(seg)])
    if s_[-1] < spacing:
        return []
    tree = cKDTree(D[:, :2])
    parts = []
    for dist in np.arange(spacing, s_[-1] - spacing / 2, spacing):
        cx = float(np.interp(dist, s_, C[:, 0]))
        cy = float(np.interp(dist, s_, C[:, 1]))
        ax = float(np.interp(min(dist + 1.0, s_[-1]), s_, C[:, 0])) - cx
        ay = float(np.interp(min(dist + 1.0, s_[-1]), s_, C[:, 1])) - cy
        n = np.hypot(ax, ay)
        if n < 1e-9:
            continue
        eu = np.array([ax / n, ay / n])
        ev = np.array([-eu[1], eu[0]])
        near = tree.query_ball_point([cx, cy], spacing / 2)
        if len(near) < 12:
            continue
        Q = D[near]
        lat = (Q[:, 0] - cx) * ev[0] + (Q[:, 1] - cy) * ev[1]
        mid = float(np.median(lat))
        half = float(np.percentile(lat, 97) - np.percentile(lat, 3)) / 2
        ztop = float(np.percentile(Q[:, 2], 20)) - t     # the soffit
        if ztop - ground_z < min_clear or half < 1.0:
            continue
        pw = max(half * across, 1.5)
        mu = np.array([cx + ev[0] * mid, cy + ev[1] * mid])
        top = box_local(mu, eu, ev, -along, along, -pw, pw, ztop - 0.01, ztop)
        bot = box_local(mu, eu, ev, -along * taper, along * taper,
                        -pw * taper, pw * taper, ground_z, ground_z + 0.01)
        T = np.asarray(top, float).reshape(-1, 3)[:4]
        B = np.asarray(bot, float).reshape(-1, 3)[:4]
        soup = []
        for q in range(4):
            a_, b_ = T[q], T[(q + 1) % 4]
            a0, b0 = B[q], B[(q + 1) % 4]
            soup += [a0, b0, b_, a0, b_, a_]
        soup = np.asarray(soup, np.float32)
        parts.append((soup, _tint(soup, col, PIER_COL), SOLID))
    return parts


def bridge_support_model(x, y, z, ground_z, deck_bottom, col=None,
                         cell=1.0, min_cells=4):
    """A PIER or COLUMN: a solid standing on ground or in water.

    Kaveh's ruling (2026-08-05): the support is NOT the bridge. The bridge is
    the deck the traffic runs on; the pier is an object that happens to hold it
    up, and it is opaque from its footing all the way to the deck underside —
    kind ``solid``, where the deck is a ``slab``. Splitting them is what lets
    the sun pass BETWEEN the piers, which is what actually happens under a
    bridge.

    Piers are claimed from evidence, never spaced by rule: the returns lying
    more than 0.5 m below ``deck_bottom`` are gridded in plan, each connected
    component of at least ``min_cells`` cells is one support, and each is
    extruded from ``ground_z`` (or the waterline the caller passes) to the deck
    underside as a box on its own PCA axes. No sub-deck returns means no
    supports, and the caller says so out loud — the lab invents a pier every
    20th station (~60 m), and inventing one here would put concrete where the
    survey saw open water.
    """
    from scipy import ndimage as ndi

    x, y, z = (np.asarray(a, float) for a in (x, y, z))
    # `deck_bottom` may be the DECK ITSELF (an (M, 3) underside), and for a
    # real bridge it has to be: Granville's footprint ramps from grade to 40 m,
    # so one scalar underside is either below the ramps (selecting nothing —
    # measured, 1 pier for the whole bridge) or above the span (selecting the
    # whole city). Judge every return against the deck DIRECTLY ABOVE IT.
    if np.ndim(deck_bottom) == 2:
        from scipy.spatial import cKDTree
        DB = np.asarray(deck_bottom, float)
        k_ = min(8, len(DB))                # the soup carries the carriageway
        _d, ii = cKDTree(DB[:, :2]).query(  # and the underside at the same xy,
            np.column_stack([x, y]), k=k_)  # so take the LOWER — a support
        db = DB[ii, 2].min(axis=1) if k_ > 1 else DB[ii, 2]   # stops at the soffit
    else:
        db = np.full(len(x), float(deck_bottom))
    sel = (z < db - 0.5) & (db - ground_z >= 1.5)
    if sel.sum() < 20:
        return []
    sx, sy, sz, sdb = x[sel], y[sel], z[sel], db[sel]
    gx = ((sx - sx.min()) / cell).astype(int)
    gy = ((sy - sy.min()) / cell).astype(int)
    occ = np.zeros((gy.max() + 1, gx.max() + 1), bool)
    occ[gy, gx] = True
    lab, n = ndi.label(occ, np.ones((3, 3)))
    cid = lab[gy, gx]
    parts = []
    for k in range(1, n + 1):
        m = cid == k
        if int(m.sum()) < min_cells or np.count_nonzero(lab == k) < min_cells:
            continue
        # A support REACHES THE DECK. Without this, any blob under the span —
        # a shed on the quay, a moored hull — is extruded from the ground to
        # the deck underside: measured, a 6.4 m tall cloud became a 34 m
        # column of concrete the laser never saw. [2026-08-05]
        top = float(np.median(sdb[m]))          # the deck THIS one holds up
        clear = top - ground_z
        if clear < 1.5:
            continue
        # A SUPPORT RISES FROM THE GROUND THROUGH THE GAP. The old test asked
        # how close the component got to the deck, which is the wrong axis: an
        # airborne laser cannot see a pier's top, because the deck is over it.
        # Measured under Granville, every real pier's returns stop 8-10 m short
        # of the deck, so that test kept 3 of 19 components and the bridge had
        # two supports. Measuring the COLUMN instead separates cleanly — real
        # piers span 0.41-0.80 of their clearance and start on the ground,
        # everything else spans 0.00-0.35 and is the ground itself.
        # [2026-08-05]
        zmin, zmax = float(sz[m].min()), float(sz[m].max())
        if zmin - ground_z > PIER_FOOT or (zmax - zmin) < PIER_RISE * clear:
            continue
        mu, eu, ev, u, v = pca_frame(sx[m], sy[m])
        L, W = float(np.ptp(u)), float(np.ptp(v))
        # BOUNDED BOTH WAYS. Without an upper bound a component that happens to
        # connect under the span is extruded whole: measured, supports came out
        # 78 x 89, 125 x 119 and 44 x 46 m — city blocks standing as concrete.
        # A pier is a column. [2026-08-05]
        if L < 0.8 or W < 0.4 or W > PIER_MAX_W:
            continue
        soup = np.asarray(box_local(mu, eu, ev, -L / 2, L / 2, -W / 2, W / 2,
                                    ground_z, top), np.float32)
        parts.append((soup, _tint(soup, col, PIER_COL), SOLID))
    return parts


# ---- objects -------------------------------------------------------------

def car_model(x, y, z, col=None):
    """A CAR: body and cabin, two boxes on its own PCA axes — kind ``solid``.

    The base is the vehicle's OWN 2nd percentile, never a DEM lookup: a car on
    a bridge deck stands on the deck, and a grid would drop it in the water.
    Lab proportions: 0.25 m wheel gap, roof break at 0.62 h, cabin inset 22 %
    front / 18 % rear / 8 % each side.
    """
    parts = car_mesh(x, y, z, WINCOL if col is None else col)
    return [(s, c, SOLID) for s, c in parts]


def lamp_model(x, y, z, ground_z, toward=None, col=None):
    """A STREET LIGHT: pole, arm, lantern — kind ``solid``.

    A pole is thin by definition, so it is claimed by SHAPE, not by return
    count: the tallest lamp on this tile carries 50 returns and a 60-return
    floor excluded every one of them, which is how a fully measured recipe
    became dead code. Lab geometry: hexagonal pole of radius 0.11 m, a 1.3 m
    arm overhanging the roadway, a 0.70 x 0.32 m lantern hung just under the
    arm.
    """
    return [(s, c, SOLID) for s, c in lamp_mesh(x, y, z, ground_z, toward)]


def small_model(x, y, z, ground_z, pix=0.5, col=None):
    """A SMALL OBJECT that is not a car and not a lamp — a bin, a bollard, a
    planter, a transformer box, a stack of crates.

    Its shape is unknown, so the only honest model is its OWN measured top
    extruded to the ground: `solid_25d`, which claims exactly the plan extent
    the laser saw and no more. Drawing it as a car — which is what this
    pipeline did to 947 of 2,155 objects, every one of them AFTER
    `looks_like_car` returned False — invents a 0.25 m ground clearance under
    a bin that sits on the pavement, a cabin on a thing with no cabin, and a
    long axis for a rotationally symmetric object.

    The lab's rule (`small_mesh`): 120 points and h > 0.5 m, else no mesh at
    all. The pixel is halved to 0.5 m here because a 1 m bin in a 1 m cell is
    one cell, so the floor comes down with it. Kind ``solid``.
    """
    x, y, z = (np.asarray(a, float) for a in (x, y, z))
    if len(z) < 60 or float(np.percentile(z, 95)) - ground_z <= 0.5:
        return []
    if max(float(np.ptp(x)), float(np.ptp(y))) > 20.0:
        return []          # 20 m across is not street furniture, it is debris
    soup = solid_25d(x, y, z, ground_z, pix=pix)
    if not len(soup):
        return []
    return [(soup, _tint(soup, col, (150, 148, 145)), SOLID)]


def on_bridge_model(x, y, z, pix=0.5, col=None):
    """DECK FURNITURE: an object standing ON a bridge, not on the ground.

    Physically the same unknown blob as :func:`small_model`, and deliberately a
    separate method, because the property that makes it its own type is its
    DATUM: it stands on a deck 20 m up. Stage 6 records 425 of these and stage
    7 used to fold them into the `car`/`small` layers, after which nothing
    downstream could tell a bollard on the bridge from a bollard on the quay.
    Its base is its own 2nd percentile — the deck it rests on — because no
    terrain grid knows the deck is there. Kind ``solid``.
    """
    x, y, z = (np.asarray(a, float) for a in (x, y, z))
    if len(z) < 40 or max(float(np.ptp(x)), float(np.ptp(y))) > 20.0:
        return []
    z0 = float(np.percentile(z, 2)) - 0.10
    if float(np.percentile(z, 95)) - z0 <= 0.5:
        return []
    soup = solid_25d(x, y, z, z0, pix=pix)
    if not len(soup):
        return []
    return [(soup, _tint(soup, col, (150, 148, 145)), SOLID)]


def boat_model(x, y, z, wlvl, col=None):
    """A VESSEL: hull with a pointed bow, cabin, mast — kind ``solid``.

    Floats at the waterline and is opaque above it, so the hull is closed from
    ``wlvl - 0.25`` up. This calls :func:`hull_soup` and NOT `boat_mesh`,
    whose "under 1.2 m over the water is a dock" redirect silently gave 
    low vessels dock geometry while keeping the boat label — a barge is a low
    hull, not a pontoon, and which one this is was already decided by the map
    partition before the model was called.
    """
    soup = hull_soup(x, y, z, wlvl)
    if soup is None or not len(soup):
        return []
    return [(soup, _tint(soup, col, (180, 180, 185)), SOLID)]


BOAT_FLOAT = 3.0     # m — a hull's lowest return sits this close to the water
BOAT_MAX_H = 15.0    # m — hull plus mast; p99 of the real population is 12.0


def looks_like_boat(px, py, pz, wlvl, max_len=25.0, min_fill=0.22,
                    pix=1.0):
    """Is this floating instance a VESSEL? — the boat type's own test.

    `floating` says only "solid matter standing above the water plane", which
    a hull, a pontoon and a gangway all satisfy, so "boat" had been defined by
    what it is NOT: whatever the pier ways did not claim. That let a 236 x 67 m
    pontoon network be called a boat [Kaveh, 2026-08-05]. A vessel has
    properties of its own, and these are the two that separate it:

    * **Bounded.** The lab's measured prior for this marina is that no vessel
      is longer than ``max_len`` = 25 m; anything longer is a raft.
    * **Compact.** A hull FILLS its own footprint. The offending instance
      filled 7.2% of its bounding box; a pontoon network is mostly the water
      between its fingers. Measured against the plan occupancy, not the
      convex hull, so an L-shaped marina finger fails it too.

    Freeboard is deliberately NOT a test: a barge rides as low as a pontoon,
    and height is exactly the rule that used to get this wrong.
    """
    px, py = np.asarray(px, float), np.asarray(py, float)
    L = max(float(np.ptp(px)), float(np.ptp(py)))
    if L > max_len or L < 1.0:
        return False
    # A BOAT FLOATS, and it is not a tower. Measured over 447 vessels
    # [2026-08-05]: the base sits a median 0.23 m above the waterline and p95
    # is 2.76 m, but 18 had bases up to 40.65 m — those are not on the water at
    # all. Height p99 is 12.0 m (hull plus mast); 21 m is a crane or a
    # building. Kaveh: "everything in water is not boat and some must be noise".
    if pz is not None:
        pz = np.asarray(pz, float)
        base = float(np.percentile(pz, 2))
        if base - wlvl > BOAT_FLOAT:
            return False
        if float(np.percentile(pz, 95)) - base > BOAT_MAX_H:
            return False
    gx = ((px - px.min()) / pix).astype(int)
    gy = ((py - py.min()) / pix).astype(int)
    occ = np.zeros((gy.max() + 1, gx.max() + 1), bool)
    occ[gy, gx] = True
    return bool(occ.sum() >= min_fill * occ.size)


def unknown_model(x, y, z, base_z=None, col=None, pix=0.5):
    """MATTER WE CANNOT NAME — its own returns, claimed and nothing more.

    Kaveh, 2026-08-05: *"if want to collapse to some label it must be
    unknown"*. When a type's own test fails — floating that is neither a mapped
    dock nor vessel-shaped, and there were 1,684 of those — the honest answer
    is not to file it under a type that means something else (it had been going
    to `small`, which means "a thing standing ON the terrain"). It is to say we
    do not know.

    So this model invents no shape at all: a 2.5-D solid of exactly the surface
    the laser saw, standing on the instance's own 2nd percentile. It shades
    correctly because the matter is really there, and it claims nothing about
    what the matter IS.
    """
    x, y, z = (np.asarray(a, float) for a in (x, y, z))
    if len(z) < 30:
        return []
    z0 = float(np.percentile(z, 2)) - 0.10 if base_z is None else float(base_z)
    if float(np.percentile(z, 95)) - z0 <= 0.3:
        return []
    soup = solid_25d(x, y, z, z0, pix=pix)
    if not len(soup):
        return []
    return [(soup, _tint(soup, col, (140, 140, 145)), SOLID)]


def dock_model(x, y, z, wlvl, ring=None, deck=None, col=None):
    """A DOCK or PONTOON: a walkway floating at the waterline — kind ``solid``.

    Solid, and that costs nothing: the volume it fills between ``wlvl - 0.25``
    and its deck is WATER, so no light is wrongly blocked (this is why the
    bridge is a slab and the dock is not — the bridge's void is air the sun
    crosses, the dock's is sea).

    Two shape sources, one geometry: where the map carries the pier's own
    outline (``ring``) that outline is the authority — an OSM/Overture way is
    surveyed, while a traced blob at a walkway's edge includes boat nibbles and
    laser gaps; otherwise the outline is traced from the returns, holding the
    water between marina fingers as holes. Deck height stays measured either
    way (lab: ``wlvl + clip(median - wlvl, 0.3, 1.2)``).
    """
    from .scene import footprint_prism

    if ring is not None:
        if deck is None:
            deck = float(np.median(z))
        # A PONTOON RIDES ON THE WATER — 0.3 to 1.2 m of freeboard, whatever
        # the returns say. The caller's `deck` is the median of every return
        # near the way, which in a marina means the moored boats and the shore
        # too: measured, 11 of 36 docks came out above the band and one stood
        # 4.67 m over the water. The clip used to apply only when `deck` was
        # None, so passing one bypassed the very rule that defines a dock.
        # [2026-08-05]
        deck = wlvl + float(np.clip(deck - wlvl, 0.3, 1.2))
        soup, cols = footprint_prism(
            ring, wlvl - 0.25, deck, col=DOCK_COL if col is None else col)
        return [(soup, cols, SOLID)] if len(soup) else []
    return [(s, c, SOLID) for s, c in dock_mesh(x, y, z, wlvl, col=col)]


# ---- vegetation ----------------------------------------------------------

def crown_mesh(x, y, z, base_z, pix=1.0, smooth=1.0, col=None):
    """A CANOPY SHELL: the crown's measured top, closed down to ``base_z``.

    The lower half of a tree's 3-D model. Kaveh asked for a model built from
    the tree INSTANCE (2026-08-05) rather than a bare stick, so the crown gets
    geometry — but only the surface the laser actually described. The canopy
    top comes from :func:`top_surface` (per-cell maximum, filled and median
    filtered), lightly smoothed because foliage is not a staircase, and the
    shell is closed at ``base_z`` — the height where the crown begins, not the
    ground, so nothing is claimed in the trunk space beneath it.

    Kind is decided by the caller: a canopy is POROUS, so :func:`tree_model`
    ships it as ``slab`` — light reaches under it, which is the whole reason
    the shadow scene never modelled a crown as a solid.
    """
    from scipy import ndimage as ndi

    got = top_surface(x, y, z, pix)
    if got is None:
        return None
    top, occ, lo = got
    if not occ.any():
        return None
    top = ndi.gaussian_filter(top, smooth)
    base = np.full_like(top, float(base_z))
    soup = cell_prism(top, np.minimum(base, top - 0.05), occ, lo, pix)
    return np.asarray(soup, np.float32) if len(soup) else None


def tree_model(x, y, z, ground_z, stem=None, crown=True):
    """A TREE: a trunk that stops light, under a canopy that filters it.

    Kaveh's original model (2026-08-01) meshed the TRUNK only and left the
    crown as raw returns — "no skin is drawn over a canopy the laser already
    described point by point". That is why a tree could not be selected in the
    scene: its only geometry was a stick. He asked for a model built from the
    tree instance (2026-08-05), so the crown is now meshed too — from its own
    measured top surface, nothing invented.

    The two parts carry DIFFERENT kinds, which is the point:

    * trunk — ``solid``. A tapered 8-sided column from the ground to 35 % of
      the crown height. Light stops at it.
    * crown — ``slab``. Its measured top, closed down to where the crown
      begins. A slab because light DOES reach under a canopy; modelled solid,
      every tree would black out the ground beneath it.
    """
    parts = [(s, c, SOLID) for s, c in trunk_mesh(x, y, z, ground_z, stem)]
    if not crown or len(z) < 60:
        return parts
    z = np.asarray(z, float)
    base = float(np.percentile(z, 25))
    if float(np.percentile(z, 98)) - base < 1.0:
        return parts
    cm = crown_mesh(x, y, z, base)
    if cm is not None and len(cm):
        parts.append((cm, _tint(cm, None, CROWN_COL), SLAB))
    return parts


def lowveg_model(x, y, z, rgb=None):
    """LOW VEGETATION: the shrub's own kNN graph, drawn as thin ribbons.

    A hedge has no surface worth inventing, so it is rendered as the graph of
    its own thinned returns. Kind ``slab``: it is a porous body sitting ON the
    ground with its underside at the ground, which is exactly how the shadow
    scene already treats it (`scene.veg_slab`, bottom at the minimum return) —
    it shades what it covers and nothing passes beneath it, but it is not a
    closed volume.
    """
    return [(s, c, SLAB) for s, c in lowveg_graph(x, y, z, rgb)]


# ---- water and terrain ---------------------------------------------------

def water_model(rings, level, col=None):
    """WATER: one flat sheet at the measured level — kind ``surface``.

    The map is the authority on where water is, the survey on what level it
    sits at. A surface receives light and casts nothing: a water body has no
    height above its neighbours to shade them with, and giving it a volume
    would make the sea a wall.
    """
    soup = drape(rings, float(level), col)
    if soup is None:
        return []
    return [(soup, _tint(soup, col, (46, 104, 150)), SURFACE)]


def terrain_model(rings, zs, col=None):
    """GROUND with a material — road, footway, steps, park, pitch, marina.

    Every terrain sub-label is one physical thing: ground, wearing a surface.
    So it is DRAPED, never extruded — each vertex takes the DEM's own height
    (the caller looks them up) plus a few centimetres to clear the terrain
    triangles. Kind ``surface``: it receives light and casts nothing, because
    it IS the receiving ground.

    Unnamed `ground` is deliberately absent from any scene: the base terrain is
    the DEM itself, and drawing it twice would z-fight with itself.
    """
    soup = drape(rings, zs, col)
    if soup is None:
        return []
    return [(soup, _tint(soup, col, (138, 128, 112)), SURFACE)]
