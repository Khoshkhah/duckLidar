"""Relief for a composed facade: window insets, door recesses, awning ledges as extra GLB prims.

    PILOT_BUILDING=railspur python tools/pilot/relief.py   # facade02 -> qa_facade_relief.png (+ qa_relief_test.glb, qa_relief_viewer.html)
    PILOT_BUILDING=netloft  python tools/pilot/relief.py   # facade00

The facade's own solid triangles become a polygon in wall metres (x from the s0 end, y UP from the base). Every
window / storefront / door rectangle that lies inside it is cut out of the wall as a hole (earcut with holes);
the hole gets no back face at all: instead the composed texture's own rectangle is glazing set behind REAL JOINERY,
and REAL JOINERY: a frame of 4 bars around the aperture standing FRAME_PROUD of the wall, a sill projecting
SILL_PROJ out of it, mullion/transom bars taken from the DETECTED PANES, and the glass GLASS_BACK behind the frame
face. Doors get the same frame plus a slab in the DOOR_RECESS. Awnings become LEDGE_DEPTH-deep wedges (the detected
band height y0..y1 against the wall, tapering to LEDGE_THICK at the front edge) plus a FASCIA_H front edge and 2-3
brackets. Every constant is MEASURED off the photo crops — see tools/pilot/docs/design/joinery.md; painting them into
the texture cannot work because each is 1-4 texels at the composed texture's 25 px/m, while as geometry each is a
surface with its own normal that catches the sun at any zoom.
Elements outside the polygon stay texture-only (relief never invents mesh); a wall without elements returns []
and keeps its flat prim. The pose is never used here — everything is metric wall coordinates from parse_wall.
"""
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box


INSET = dict(window=0.08, storefront=0.08, door=0.15)       # m into the wall. The WINDOW path no longer uses this
                                                            # (superseded by FRAME_PROUD + GLASS_BACK = 0.055 m); it still
                                                            # selects which element types become holes, and INSET["door"]
                                                            # IS DOOR_RECESS — the measured 0.25 m jamb = 0.15 + FRAME_W.
DOOR_RECESS = INSET["door"]

# ── joinery, every value measured off the crops (docs/design/joinery.md; n = the sample it came from) ──
FRAME_W     = 0.145   # m frame bar width, all 4 bars. FWHM 0.145 (n=14) / threshold 0.147 (n=14) / pane inset 0.127
FRAME_PROUD = 0.02    # m frame front face IN FRONT of the wall — the wall-side shadow notch on every L profile
FRAME_D     = 0.06    # m frame bar depth = FRAME_PROUD + GLASS_BACK, so the bar spans its face to the glass. Derived.
GLASS_BACK  = 0.035   # m glass behind the frame FRONT face. Frame->glass fall-off, median 0.033, n=14
MULLION_W   = 0.155   # m mullion/transom width. Gaps between DETECTED panes: railspur el0 0.155, el2 0.161, netloft 0.150
MULLION_D   = 0.045   # m mullion depth. Derived: must reach the glass but stop short of the frame face (< FRAME_D)
SILL_H      = 0.065   # m sill slab thickness. Bright band at the base, median of 7 railspur windows
SILL_DROP   = 0.080   # m sill TOP below the opening base. Band top, same 7
SILL_PROJ   = 0.075   # m sill projection. From the sill's OWN cast shadow, median of the 5 that show one (~45 deg sun)
SILL_OVER   = 0.030   # m sill overhang past each jamb. Derived (half FRAME_W); the real overhang is under 1 texel
PANE_TARGET = 0.90    # m fallback division pitch. Measured panes: railspur 0.95-1.00, netloft storefront 0.63-0.81
DOOR_SLAB_D = 0.04    # m door slab thickness in the recess. Derived
FASCIA_H    = 0.15    # m awning front edge. The dark band y 3.35..3.20 on railspur el8 (= LEDGE_THICK, which is why
                      #   that constant already looked right)
FASCIA_D    = 0.06    # m fascia thickness. Derived
BRACKET_DROP = 0.45  # m the knee brace's rise down the wall. Derived: ~half the min awning depth, so the
                      #   strut sits at a plausible ~45 deg; the brace is hidden by the fascia in every crop
BRACKET_W   = 0.06    # m bracket bar width. Derived — the fascia hides the brackets in every crop
BRACKET_MIN = 0.15    # m: below this much FREE wall under the awning a brace is not drawn at all (rather than
                      #   piercing the opening below it, which is what a fixed BRACKET_DROP did)
AWNING_JOIN = 1.0     # m: two awnings of the same band closer than this are ONE awning — clipping them to the
                      #   poly separately left a wall gap and a wedge stabbing through its neighbour's top
MIN_JOINERY = 0.45    # m below this in either dimension: frame + glass only (bars would meet in the middle)
GLASS_SHADE = 0.55    # synth glass = material median x this, blue-shifted
LEDGE_DEPTH, LEDGE_THICK, LEDGE_MAX = 0.8, 0.15, 2.5        # m out of the wall (min / max) / thickness at the front edge (the awning is a wedge: full detected height at the wall)
LEDGE_SLOPE = 35.0                                          # deg: depth = rise / tan(slope) — a 1.4 m fabric awning reaches 2 m out, a 0.5 m canopy 0.8 m
MIN_CONF = 0.15                                             # below this an element is a non-detection: it gets NO mesh (netloft facade14's
                                                            #   door is conf 0.01, facade00's squashed window 0.14)
ROW_SNAP = 0.30                                             # a hole whose height deviates > 30 % from its row's median is snapped to it
MIN_HOLE, MIN_LEDGE = 0.3, 1.0                              # m: smaller openings / shorter awnings get no mesh
LEDGE_SHADE = 0.5                                           # flat material = median texture colour x shade


_last_holes = []                                            # (part, element, rect) of the LAST relief_prims call — QA only


def _rect(e):
    r = e.get("rect")
    return tuple(float(v) for v in (r if r is not None else (e["x0"], e["y0"], e["x1"], e["y1"])))


def facade_polygon(F, origin):
    """The facade's solid triangles in wall metres, unioned and closed over sub-10 cm gaps (two roofer faces on railspur
    facade02 leave an 8 cm slit). May be a MultiPolygon; use `.geoms` via _parts."""
    from .texture import CTX
    V = CTX.SOLID["V"]; T = CTX.SOLID["T"]
    P = V[T[F["tris"]]] - origin                                # (k, 3, 3)
    xy = np.stack([P @ F["u"], P @ F["v"]], -1)                 # (k, 3, 2)
    return shapely.unary_union([Polygon(t) for t in xy]).buffer(0.05).buffer(-0.05)


def _clip_rect(parts, x0, y0, x1, y1):
    """An opening that pokes out of the facade polygon -> (part index, the largest axis-aligned rect of it that the
    part does contain), or (None, None) when nothing usable is left. Shapely gives the intersection's bounds; those
    bounds can still leave the polygon on a non-rectangular part, so the result is re-checked with contains().
    Without this a real window (railspur facade02 x 0.57..2.91, conf 0.92) is silently demoted to texture-only and
    reads flat beside four framed row-mates."""
    best = (None, None, 0.0)
    for k, part in enumerate(parts):
        out = box(x0, y0, x1, y1).difference(part)
        if out.is_empty: continue
        ox0, oy0, ox1, oy1 = out.bounds                         # trim ONE side back past everything that sticks out; the corner
        for r in ((ox1, y0, x1, y1), (x0, y0, ox0, y1), (x0, oy1, x1, y1), (x0, y0, x1, oy0)):   # notch on railspur's roofline
            if r[2] - r[0] < MIN_HOLE or r[3] - r[1] < MIN_HOLE: continue                        # is cut by exactly one of these
            a = (r[2] - r[0]) * (r[3] - r[1])
            if a > best[2] and part.contains(box(*r).buffer(-0.02)): best = (k, r, a)
    if best[2] < 0.5 * (x1 - x0) * (y1 - y0): return None, None      # less than half survives: it is not that opening any more
    return best[0], best[1]


def _parts(g):
    return [p for p in getattr(g, "geoms", [g]) if not p.is_empty and p.area > 0]


def _flat(colour):
    return np.tile(np.clip(colour, 0, 255).astype(np.uint8), (4, 4, 1))


def ccw_out(t2d, tris):
    """(u, v) is left-handed about n: flip triangles so their signed area in (x, y) is negative = CCW seen from outside"""
    a = t2d[tris]; s = (a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1]) - (a[:, 2, 0] - a[:, 0, 0]) * (a[:, 1, 1] - a[:, 0, 1])
    tris = tris.copy(); tris[s > 0] = tris[s > 0][:, ::-1]; return tris


# ── the joinery kit: everything is a box, so winding is reasoned about ONCE ──────────────────────
_BOX_F = np.array([[0, 1, 2], [0, 2, 3],      # d1, the OUTER face (+n)  vertices 0..3 = (x0,y0) (x1,y0) (x1,y1) (x0,y1)
                   [7, 6, 5], [7, 5, 4],      # d0, the inner face       vertices 4..7 = the same at d0
                   [3, 2, 6], [3, 6, 7],      # top    y1
                   [4, 5, 1], [4, 1, 0],      # bottom y0
                   [0, 3, 7], [0, 7, 4],      # x0 end
                   [5, 6, 2], [5, 2, 1]])     # x1 end


def _box(P, x0, y0, x1, y1, d0, d1):
    """One axis-aligned box in wall coordinates -> (pos (8,3), idx (12,3)).
    x, y in wall metres; d0 < d1 are depths along n (d > 0 outward, so d1 is the outer face). P is the caller's
    P(x, y, d). Winding follows ccw_out's rule: (u, v) is LEFT-handed about n, so a face seen from outside is
    CW in (x, y) — _BOX_F is written that way once and every joinery part reuses it.
    Degenerate boxes (any extent <= 1e-6) return empty arrays, so callers need no guard."""
    if x1 - x0 <= 1e-6 or y1 - y0 <= 1e-6 or d1 - d0 <= 1e-6:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    q = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return np.array([P(x, y, d) for d in (d1, d0) for x, y in q]), _BOX_F.copy()


def _cat(parts):
    """[(pos, idx)] -> one (pos, idx) with the indices rebased."""
    pos, idx, base = [], [], 0
    for p_, i_ in parts:
        if not len(p_): continue
        pos.append(p_); idx.append(np.asarray(i_) + base); base += len(p_)
    if not pos: return np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.vstack(pos), np.vstack(idx)


def _frame_bars(P, x0, y0, x1, y1):
    """4 bars FRAME_W wide INSIDE the opening rect (the measured pane inset is from the rect inward, so the frame
    occupies the rect's border and the glass is the remaining aperture), spanning FRAME_PROUD - FRAME_D .. FRAME_PROUD.
    Jambs stop short of head and sill bar so nothing overlaps (a mitre is not resolvable, and overlaps z-fight).
    4 boxes = 48 triangles."""
    d0, d1 = FRAME_PROUD - FRAME_D, FRAME_PROUD
    w = min(FRAME_W, (x1 - x0) / 2.5, (y1 - y0) / 2.5)                  # a tiny opening cannot hold a full-width frame
    return _cat([_box(P, x0, y1 - w, x1, y1, d0, d1),                   # head
                 _box(P, x0, y0, x1, y0 + w, d0, d1),                   # sill bar
                 _box(P, x0, y0 + w, x0 + w, y1 - w, d0, d1),           # left jamb
                 _box(P, x1 - w, y0 + w, x1, y1 - w, d0, d1)])          # right jamb


def _sill(P, x0, y0, x1, y1):
    """One box under the opening: x0-SILL_OVER .. x1+SILL_OVER, y0-SILL_DROP-SILL_H .. y0-SILL_DROP, depth
    -0.02 .. SILL_PROJ (it starts just inside the wall so there is no gap at the joint). 12 triangles.
    Empty for an opening below MIN_JOINERY, or one that reaches the ground (a door-height opening gets no sill)."""
    if min(x1 - x0, y1 - y0) < MIN_JOINERY or y0 - SILL_DROP - SILL_H < 0: return np.zeros((0, 3)), np.zeros((0, 3), int)
    return _box(P, x0 - SILL_OVER, y0 - SILL_DROP - SILL_H, x1 + SILL_OVER, y0 - SILL_DROP, -0.02, SILL_PROJ)


def _division(x0, y0, x1, y1, panes):
    """The mullion/transom LINES for one opening -> (x centres, y centres), in wall metres. Pure, no geometry.
    With >= 2 detected panes the bar centres are the midpoints of the GAPS between adjacent panes — which is exactly
    how MULLION_W was measured. A gap > 0.5 m is a bay boundary, not a mullion (netloft's 0.95 m), and is skipped;
    overlapping panes (the SAM boxes overlap on netloft's storefront) give no gap. Without panes: a plausible
    division at PANE_TARGET pitch. Capped nv <= 5 / nh <= 2 so a 10 m storefront cannot blow the budget."""
    xs, ys = [], []
    panes = [p for p in (panes or []) if p[2] > p[0] and p[3] > p[1]]
    if len(panes) >= 2:
        P = sorted(panes, key=lambda p: p[0]); right = P[0][2]
        for q in P[1:]:                                             # forward sweep: only a real gap past everything so far
            gap = q[0] - right
            if 0.02 < gap < 0.5: xs.append(right + gap / 2)
            right = max(right, q[2])
        rows = []                                                   # cluster pane bottoms, 0.15 m tolerance
        for q in sorted(P, key=lambda p: p[1]):
            if rows and q[1] - np.mean(rows[-1]) < 0.15: rows[-1].append(q[1])
            else: rows.append([q[1]])
        if len(rows) >= 2:
            lo = max(q[3] for q in P if abs(q[1] - np.mean(rows[0])) < 0.15)     # top of the lower band
            hi = min(q[1] for q in P if abs(q[1] - np.mean(rows[1])) < 0.15)     # bottom of the next
            if 0.02 < hi - lo < 0.5: ys.append((lo + hi) / 2)
    lim0 = FRAME_W + MULLION_W / 2
    if not [x for x in xs if x0 + lim0 < x < x1 - lim0]:                # panes gave nothing usable (2 overlapping SAM boxes,
        xs = []                                                         # or one bay boundary): fall back to the pitch division
        nv = min(5, max(0, round((x1 - x0) / PANE_TARGET) - 1))
        xs = [x0 + (x1 - x0) * (k + 1) / (nv + 1) for k in range(nv)]
        if not ys and y1 - y0 > 1.6: ys = [(y0 + y1) / 2]
    lim = FRAME_W + MULLION_W / 2                                   # a bar this close to the edge would merge into the frame
    return ([x for x in xs if x0 + lim < x < x1 - lim][:5], [y for y in ys if y0 + lim < y < y1 - lim][:2])


def _mullions(P, x0, y0, x1, y1, panes):
    """_division -> bars MULLION_W wide, depth FRAME_PROUD - MULLION_D .. FRAME_PROUD - 0.01 (just short of the
    frame face, so the frame stays the outer plane). 12 triangles per bar."""
    if min(x1 - x0, y1 - y0) < MIN_JOINERY: return np.zeros((0, 3)), np.zeros((0, 3), int)
    xs, ys = _division(x0, y0, x1, y1, panes); h = MULLION_W / 2; d0, d1 = FRAME_PROUD - MULLION_D, FRAME_PROUD - 0.01
    return _cat([_box(P, x - h, y0 + FRAME_W, x + h, y1 - FRAME_W, d0, d1) for x in xs] +
                [_box(P, x0 + FRAME_W, y - h, x1 - FRAME_W, y + h, d0, d1) for y in ys])


def _glass(P, UV, x0, y0, x1, y1):
    """One quad at depth FRAME_PROUD - GLASS_BACK, inset FRAME_W all round so it sits behind the frame APERTURE.
    UV is the caller's UV(x, y), so the glass samples the same composed-texture rect the old back face did — the
    crop's own glass pixels, reflections and all. 2 triangles."""
    w = min(FRAME_W, (x1 - x0) / 2.5, (y1 - y0) / 2.5)
    a, b, c, e = x0 + w, y0 + w, x1 - w, y1 - w
    if c - a <= 1e-6 or e - b <= 1e-6: return np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 3), int)
    d = FRAME_PROUD - GLASS_BACK
    return (np.array([P(a, b, d), P(c, b, d), P(c, e, d), P(a, e, d)]), np.array([UV(a, b), UV(c, b), UV(c, e), UV(a, e)]),
            np.array([[0, 2, 1], [0, 3, 2]]))                       # CW in (x, y) = CCW from outside


def _door_parts(P, UV, x0, y0, x1, y1, panes):
    """Door: the jamb is _frame_bars (the measured 0.25 m jamb = DOOR_RECESS 0.15 + FRAME_W 0.145), plus a SLAB box
    inset FRAME_W at the sides and top, y0 .. y1-FRAME_W, at depth -DOOR_RECESS .. -DOOR_RECESS + DOOR_SLAB_D.
    The slab is TEXTURED, not flat — the railspur crop shows a glazed upper panel (upper/mid/lower L 61/88/56) the
    composed texture already carries — so it goes in the glass prim. No sill on a door. 48 + 12 triangles."""
    fr = _frame_bars(P, x0, y0, x1, y1)
    w = min(FRAME_W, (x1 - x0) / 2.5, (y1 - y0) / 2.5)
    a, b, c, e = x0 + w, y0, x1 - w, y1 - w
    sp, si = _box(P, a, b, c, e, -DOOR_RECESS, -DOOR_RECESS + DOOR_SLAB_D)
    q = [(a, b), (c, b), (c, e), (a, e)]                            # _box's vertex order, outer ring then inner
    suv = np.array([UV(x, y) for _d in (0, 1) for x, y in q]) if len(sp) else np.zeros((0, 2))
    return fr, (sp, suv, si)


def _awning_extras(P, xa, xb, yb, yt, D, obstacles=()):
    """Fascia + brackets for one awning wedge (xa..xb, yb..yt, front lip at depth D).
    fascia: one box across the span at the front edge, FASCIA_H down from the lip, depth D-FASCIA_D .. D — the
      measured 0.15 m dark band on railspur el8.
    brackets: n = clip(round(span / 3), 2, 3) knee braces BRACKET_W thick, each a triangular prism whose legs lie
      ON the wall and ON the awning's underside (see _bracket) — so a bracket touches both surfaces it braces.
      The COUNT comes from the awning's own width because the bracket pitch is NOT measurable (the fascia hides
      them in every crop) — stated, not disguised as a measurement.
      `obstacles` = the accepted opening rects on this wall: a brace may only occupy FREE wall, so its drop is
      clipped to stop at the highest opening its own x-span overlaps, and it is SKIPPED when that leaves under
      BRACKET_MIN. On netloft facade00 the storefronts top out at exactly the awning base, so every brace is
      suppressed — which is also what the photo shows (one continuous band, no visible brackets).
    -> (pos, idx), 12 + 0..36 triangles."""
    yf = yb + LEDGE_THICK
    parts = [_box(P, xa, yf - FASCIA_H, xb, yf, D - FASCIA_D, D)]          # the front edge, hanging at the wedge's lip
    n = int(np.clip(round((xb - xa) / 3.0), 2, 3))
    for k in range(n):
        c = xa + (xb - xa) * (k + 0.5) / n
        x0, x1 = c - BRACKET_W / 2, c + BRACKET_W / 2
        top = max([r[3] for r in obstacles if r[0] < x1 and r[2] > x0 and r[3] <= yb + 1e-6] or [0.0])   # free wall runs yb down to `top`
        drop = min(BRACKET_DROP, D, yb - top)
        if drop >= BRACKET_MIN: parts.append(_bracket(P, x0, x1, yb, drop, D - FASCIA_D))
    return _cat(parts)


def _bracket(P, x0, x1, yb, drop, D):
    """One knee brace under an awning: a triangular prism BRACKET_W thick in x, in the (depth, y) plane.
    Its two right-angle legs are the wall (d = 0, from yb down to yb - drop) and the awning underside
    (y = yb, from d = 0 out to D); the hypotenuse is the strut face. It therefore TOUCHES both surfaces it
    braces instead of floating below the wedge, which is what the flat slab did."""
    if x1 - x0 <= 1e-6 or drop <= 1e-6 or D <= 1e-6:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    #        0 wall-top      1 wall-bottom          2 front tip        then the same at x1
    tri = lambda x: [P(x, yb, 0.0), P(x, yb - drop, 0.0), P(x, yb, D)]
    pos = np.array(tri(x0) + tri(x1))
    idx = np.array([[0, 1, 2], [5, 4, 3],                      # the two triangular ends
                    [0, 3, 4], [0, 4, 1],                      # wall leg   (d = 0)
                    [1, 4, 5], [1, 5, 2],                      # hypotenuse (the visible strut face)
                    [2, 5, 3], [2, 3, 0]])                     # top leg    (y = yb, against the awning)
    return pos, idx


def _frame_colour(tex, layers, rect, material_median):
    """The FLAT material for the joinery prim, from the CROP'S OWN FRAME PIXELS — not a constant, not the wall colour.
    Take the composed-texture rect; take its border ring, round(FRAME_W / TEX_M) = 4 texels wide (where the frame
    was measured to be); in Lab keep the ring texels above the ring's median L (the LIT half of the bar — the shaded
    half is contaminated by the reveal shadow) and return their median RGB.
    Falls back to material_median x 1.4 when the ring is too small or is not distinguishable from the wall (median L
    within 8 of it): a light frame is the safe default for both pilot buildings.
    Reference (railspur facade02): per-window medians ~[136,133,121]..[92,93,91], overall ~[118,121,114], wall ~[62,63,52]."""
    from . import compose as fc
    h, w = tex.shape[:2]
    fb = np.clip(np.asarray(material_median, float) * 1.4, 0, 255)
    p = fc._px(rect, w, h)
    if p is None: return fb
    c0, c1, r0, r1 = p; k = max(1, round(FRAME_W / fc.TEX_M))
    if c1 - c0 < 2 * k + 2 or r1 - r0 < 2 * k + 2: return fb
    sub = tex[r0:r1, c0:c1]; ring = np.ones(sub.shape[:2], bool); ring[k:-k, k:-k] = False
    px = sub[ring]
    if len(px) < 20: return fb
    L = fc._rgb2lab(px)[:, 0]
    lit = px[L > np.median(L)]
    if len(lit) < 10 or abs(float(np.median(L)) - float(fc._rgb2lab(material_median.reshape(1, 3))[0, 0])) < 8: return fb
    return np.median(lit, axis=0)


def _glass_tex(material_median):
    """Synthesised glazing for an element whose composed-texture region is not trustworthy (src_kind 'synth', or the
    crop gate refused it): a 64-row vertical ramp, sky-lit blue-grey at the top to darker at the bottom, built the
    same way compose.synth builds its glass band (Lab L 62 +/- 10, a 128, b 118). One prim per wall, never one
    per opening."""
    from . import compose as fc
    grad = np.linspace(10, -10, 64, dtype=np.float32)[:, None]
    L0 = float(fc._rgb2lab(np.asarray(material_median, np.uint8).reshape(1, 3))[0, 0]) * GLASS_SHADE
    lab = np.zeros((64, 4, 3), np.float32); lab[..., 0] = max(20.0, L0) + grad; lab[..., 1] = 128; lab[..., 2] = 118
    return fc._lab2rgb(lab)


def relief_prims(F, elements, origin, tex, name="facade", layers=None):
    """F facade dict; elements from parse_wall (wall metres, y up); origin (3,) UTM = the wall's s0-end base point;
    tex = the COMPOSED texture (h,w,3); layers = compose's second return, optional (only _frame_colour would use it,
    and it reads the ring off `tex` alone, so the existing build_model call keeps working unchanged).
    -> [] when nothing qualifies, else up to 5 write_glb prims that replace the flat facade prim:
    `<name>_composed` (the wall with holes, no back faces), `<name>_joinery` (frames, sills, mullions, door jambs,
    awning fascias and brackets — one flat measured frame colour), `<name>_glass` (glass quads + door slabs, the
    composed texture), `<name>_glass_synth` (openings whose crop is not trustworthy), `<name>_ledges` (awning wedges)."""
    W_m, H_m = F["s1"] - F["s0"], F["t1"] - F["t0"]
    u, v, n = F["u"], F["v"], F["n"]
    P = lambda x, y, d=0.0: origin + u * x + v * y + n * d      # d > 0 outward
    UV = lambda x, y: (x / W_m, 1.0 - y / H_m)
    poly = facade_polygon(F, origin); parts = _parts(poly)
    if not parts: return []

    holes = []                                                  # (part index, element, rect); largest first, a rect overlapping an accepted hole stays texture-only (a door inside a bay)
    cands = [e for e in elements if e["type"] in INSET and e.get("conf", 1.0) >= MIN_CONF]      # a conf-0.01 detection is noise: no mesh
    hmed = {}                                                   # (type, row) -> median height, so a squashed outlier snaps to its row
    for e in cands:
        hmed.setdefault((e["type"], e.get("row")), []).append(_rect(e)[3] - _rect(e)[1])
    hmed = {k: float(np.median(v)) for k, v in hmed.items() if len(v) >= 3}
    for e in sorted(cands, key=lambda e: -(_rect(e)[2] - _rect(e)[0]) * (_rect(e)[3] - _rect(e)[1])):
        x0, y0, x1, y1 = _rect(e)
        m = hmed.get((e["type"], e.get("row")))
        if m and abs((y1 - y0) - m) > ROW_SNAP * m: y1 = y0 + m                 # keep the base (the sill line is the reliable edge)
        if x1 - x0 < MIN_HOLE or y1 - y0 < MIN_HOLE: continue
        if any(box(*r).intersects(box(x0, y0, x1, y1)) for _k, _e, r in holes): continue
        b = box(x0, y0, x1, y1).buffer(-0.02)
        for k, part in enumerate(parts):
            if part.contains(b): holes.append((k, e, (x0, y0, x1, y1))); break
        else:                                                   # not wholly inside: CLIP to the part that holds most of it rather than
            k, r = _clip_rect(parts, x0, y0, x1, y1)            # dropping a real, high-confidence opening to flat paint beside its framed row-mates
            if k is not None and not any(box(*rr).intersects(box(*r)) for _k, _e, rr in holes):   # a clip must not land on an accepted
                holes.append((k, e, r))                                                          # hole (a door inside its own bay)
    ledges = []
    aw = []                                                     # awnings of the same band, merged over gaps < AWNING_JOIN (netloft
    for e in sorted((e for e in elements if e["type"] == "awning"), key=lambda e: _rect(e)[0]):   # facade00's two segments are one canopy:
        x0, y0, x1, y1 = _rect(e)                                                                # clipped apart they left a gap AND overlapped
        if aw and x0 - aw[-1][1] < AWNING_JOIN and min(aw[-1][3], y1) - max(aw[-1][2], y0) > 0:
            p = aw[-1]; aw[-1] = (p[0], max(p[1], x1), min(p[2], y0), max(p[3], y1), p[4]); continue
        aw.append((x0, x1, y0, y1, e))
    for x0, x1, y0, y1, e in aw:
        cut = poly.intersection(LineString([(poly.bounds[0] - 1, y1), (poly.bounds[2] + 1, y1)]))
        bx0, bx1 = (cut.bounds[0], cut.bounds[2]) if not cut.is_empty else (poly.bounds[0], poly.bounds[2])
        xa, xb = max(x0, bx0), min(x1, bx1)
        if xb - xa >= MIN_LEDGE: ledges.append((e, xa, xb, y0, y1))
    if not holes and not ledges: return []

    # ── wall with holes (one prim, the composed texture); the glass quad replaced the old back face ──
    pos, uv, idx = [], [], []
    import mapbox_earcut
    for k, part in enumerate(parts):
        rings = [np.asarray(part.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in part.interiors]
        rings += [np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]) for kk, _e, (x0, y0, x1, y1) in holes if kk == k]
        verts = np.vstack(rings); ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
        t = mapbox_earcut.triangulate_float64(verts, ends).reshape(-1, 3)
        t = ccw_out(verts, t)
        base = len(pos)
        pos += [P(x, y) for x, y in verts]; uv += [UV(x, y) for x, y in verts]; idx += (t + base).tolist()
    _last_holes[:] = holes                                      # for the self-check only: `holes` in the prim is JSON, this keeps the element identity
    prims = [dict(pos=np.array(pos), uv=np.array(uv), idx=np.array(idx), tex=tex, name=f"{name}_composed",
                  holes=[dict(type=e["type"], rect=r) for _k, e, r in holes])]   # NO back faces: the glass quad replaces them

    med = np.median(tex.reshape(-1, 3), axis=0)                 # >= 60 % of texels are material: its median is the wall colour
    # ── joinery: frame bars, sill, mullions/transoms, door jamb + slab; glass behind the frame ──
    flat, glass, synth = [], [], []                             # (pos, idx) / (pos, uv, idx) / (pos, uv, idx)
    for _k, e, (x0, y0, x1, y1) in holes:
        panes = [p for p in (e.get("lights") or []) if isinstance(p, (list, tuple)) and len(p) == 4]
        good = e.get("src_kind") in ("photo", "sibling") and not e.get("crop_gate")   # spec_gate: a refused crop is not glazing
        if e["type"] == "door":
            fr, slab = _door_parts(P, UV, x0, y0, x1, y1, panes)
            flat.append(fr); (glass if good else synth).append(slab)
        else:
            flat += [_frame_bars(P, x0, y0, x1, y1), _sill(P, x0, y0, x1, y1), _mullions(P, x0, y0, x1, y1, panes)]
            g = _glass(P, UV, x0, y0, x1, y1)
            (glass if good else synth).append(g)
    for e, xa, xb, yb, yt in ledges:                            # the wedge itself stays in _ledges; its fascia + brackets are joinery
        yb2 = min(yb, yt - LEDGE_THICK)
        flat.append(_awning_extras(P, xa, xb, yb2, yt, float(np.clip((yt - yb2 - LEDGE_THICK) / np.tan(np.radians(LEDGE_SLOPE)), LEDGE_DEPTH, LEDGE_MAX)),
                                   obstacles=[r for _k, _e, r in holes]))
    fp, fi = _cat(flat)
    if len(fi):
        col = np.median([_frame_colour(tex, layers, r, med) for _k, _e, r in holes], axis=0) if holes else med * 1.4
        prims.append(dict(pos=fp, uv=np.zeros((len(fp), 2)), idx=fi, tex=_flat(col), name=f"{name}_joinery"))
    for group, tx, nm in ((glass, tex, f"{name}_glass"), (synth, _glass_tex(med), f"{name}_glass_synth")):
        gp, gi = _cat([(a, c) for a, _b, c in group])
        if not len(gi): continue
        parts_uv = [b for a, b, _c in group if len(a)]
        if nm.endswith("_synth"):                                # the 4x64 ramp: rescale each PART's v over its own height
            out = []                                             # (a quad has 4 vertices, a door slab is a box with 8)
            for b in parts_uv:
                b = np.asarray(b, float).copy(); vv = b[:, 1]
                b[:, 1] = (vv - vv.min()) / max(float(vv.max() - vv.min()), 1e-9); b[:, 0] = 0.5
                out.append(b)
            parts_uv = out
        guv = np.vstack(parts_uv)
        prims.append(dict(pos=gp, uv=guv, idx=gi, tex=tx, name=nm))
    # ── ledges: a slab hanging at the awning's top edge, 5 faces (nothing against the wall) ──
    if ledges:
        lp, li = [], []
        for e, xa, xb, yb, yt in ledges:
            yb = min(yb, yt - LEDGE_THICK); yf = yb + LEDGE_THICK; base = len(lp)      # wall face yb..yt, front edge yb..yf
            D = float(np.clip((yt - yf) / np.tan(np.radians(LEDGE_SLOPE)), LEDGE_DEPTH, LEDGE_MAX))
            #      0         1         2          3          4          5          6          7
            lp += [P(xa, yb), P(xb, yb), P(xb, yb, D), P(xa, yb, D), P(xa, yt), P(xb, yt), P(xb, yf, D), P(xa, yf, D)]
            faces = [[0, 2, 1], [0, 3, 2],      # bottom (seen from below)
                     [4, 5, 6], [4, 6, 7],      # top
                     [3, 7, 6], [3, 6, 2],      # front (d = LEDGE_DEPTH)
                     [0, 4, 7], [0, 7, 3],      # xa end
                     [1, 2, 6], [1, 6, 5]]      # xb end
            li += (np.array(faces) + base).tolist()
        col = next((np.array(e["colour"], float) for e, *_r in ledges if e.get("colour")), med * LEDGE_SHADE)   # the awning's own colour when a photo gave one
        prims.append(dict(pos=np.array(lp), uv=np.zeros((len(lp), 2)), idx=np.array(li), tex=_flat(col), name=f"{name}_ledges",
                          ledges=[dict(x0=xa, x1=xb, y0=yb, y1=yt) for _e, xa, xb, yb, yt in ledges]))
    return prims


# ── self-check on real data ─────────────────────────────────────────────────

def _screenshots(html_path, views, out_png, size=(900, 600)):
    """views: [(eye Y-up xyz, target Y-up xyz)] -> one PNG stacked vertically; None when playwright cannot run."""
    import cv2
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    shots = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
        pg = b.new_page(viewport=dict(width=size[0], height=size[1])); pg.goto(Path(html_path).as_uri()); pg.wait_for_timeout(3500)
        for eye, tgt in views:
            pg.evaluate("([e, t]) => { cam.position.set(e[0], e[1], e[2]); ctl.target.set(t[0], t[1], t[2]); ctl.update(); r.render(scene, cam); }", [list(map(float, eye)), list(map(float, tgt))])
            pg.wait_for_timeout(300); tmp = str(out_png) + ".part.png"; pg.screenshot(path=tmp); shots.append(cv2.imread(tmp))
        b.close()
    Path(tmp).unlink(missing_ok=True)
    cv2.imwrite(str(out_png), np.vstack(shots))
    return out_png

# The self-check lives with the pilot that owns the data: shadowCity2 tools/pilot/facade_relief.py
