"""Where a wall really is in a photo — the quad from image evidence, the pose only as a search band.

    PILOT_BUILDING=railspur python tools/pilot/quad.py   # self-check on photo #2 / facade02
    PILOT_BUILDING=netloft  python tools/pilot/quad.py   # photo #16 / facade00 (eave under trees)

Levelled virtual camera (roll 0, pitch p from the METADATA): the horizon is the row
v_h = 320 + f tan p and every 3-D vertical projects to a line through the vertical vanishing
point VP = (320, 320 + f / tan p). So a wall's quad has two free parameters per side (its u at
the horizon), a top line (LSD segments with building below / not-building above), and a bottom
line through the top's horizontal vanishing point (mask/ground boundary, or derived from the
camera height when cars hide the base). Quad order is TL, TR, BR, BL in WALL terms: TL is the
s0 end, which for an outside camera is usually the image RIGHT — never reorder by x.

Where this departs from the design note, and why (all measured on the pilot photos):
  - the roofline gate checks the occluder mask 6 px BELOW a segment, not at it: sky is an occluder
    and touches every eave (railspur #2 lost its whole eave otherwise);
  - the eave's u-extent chains fascia/soffit segments (building above and below, both endpoints
    under the roof edge) — LSD fragments the roof edge where its contrast fades, the fascia runs
    to the corner (railspur #2: run ends at 502, corner at 498);
  - corners are fragmented at 640 px (35 + 16 px of LSD at railspur #2's corner), so a dense
    appearance profile along VP-tilted columns adds candidates: rows where a 10 px strip left of
    the column differs from one on the right;
  - the mid-wall gate accepts a candidate either near the eave run's end / a mask free end, OR
    when the two sides also differ with 12-30 px strips (a corner or a plane step of the same
    building; a post, downspout or mullion is thin and fails) — Net Loft's eave is continuous
    across its plane steps, so the run end alone missed netloft #2 (u 298) and #8 (u 265);
  - a building-coverage gate (< 25 % of the prior's interior) returns the prior with ok False
    for photos that do not see the wall at all (bridge-deck views of railspur facade00);
  - the roofline gate wants SKY-like pixels (pale / blue, `sky_like`) above a segment, not just
    "not building": the occluder mask lumps sky with trees, so a canopy boundary 50-80 px under
    the eave passed as a roofline (netloft #15/#17 -> the upper storey vanished);
  - the top search band is max(60, 0.2 h_px) (netloft #6: the prior top sat 70 px above the eave);
  - with no eave found, the wall's horizontal VP is re-fitted from its own horizontals inside the
    prior (mode of their horizon crossings) so the top/bottom slope no longer carries the pose
    heading's error (netloft #6 / facade04 sheared the awning by 5 % of the height).
Extra result keys for QA: run (eave u-extent), modes (every side mode with its anchors), peaks
(the accepted ones), quad_raw (the corners before the sanity check).
"""
import math, os, sys
import cv2
import numpy as np

W = 640


# ── camera / line helpers ───────────────────────────────────────────────────
def cam_consts(pitch, fov):
    f = 320 / math.tan(math.radians(fov / 2)); p = math.radians(pitch)
    return f, p, 320 + f * math.tan(p), np.array([320.0, 320 + f / math.tan(p)])


def hline(a, b): return np.cross([a[0], a[1], 1.0], [b[0], b[1], 1.0])
def meet(l, m):
    x = np.cross(l, m); return x[:2] / x[2] if abs(x[2]) > 1e-12 else np.array([np.inf, np.inf])
def v_at(l, u): return -(l[0] * u + l[2]) / l[1]
def u_at(l, v): return -(l[1] * v + l[2]) / l[0]
def cross2(a, b): return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
def pdist(pts, l): return np.abs(np.asarray(pts) @ l[:2] + l[2]) / math.hypot(l[0], l[1])
def angdiff(a, b): d = np.abs(a - b) % 180; return np.minimum(d, 180 - d)


def sky_like(img_bgr):
    """Pixels that look like sky (pale / blue, median-filtered): the occluder mask lumps sky with trees, but a roofline
    has SKY above it, a tree-canopy boundary has leaves (green, dark, textured) above it."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV); Hh, Sh, Vh = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sky = ((Sh < 70) & (Vh > 140)) | ((Hh >= 95) & (Hh <= 130) & (Sh > 30) & (Vh > 90))
    return cv2.medianBlur(sky.astype(np.uint8) * 255, 5) > 127


def segs(gray_u8, min_len=15):
    """LSD segments as float32 (N,5) [x1,y1,x2,y2,length], shorter than min_len dropped."""
    L = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray_u8)[0]
    if L is None: return np.zeros((0, 5), np.float32)
    L = L.reshape(-1, 4).astype(np.float32); n = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
    return np.column_stack([L, n])[n >= min_len]


def eave_point(u, v, pitch, fov, dz_t):
    """Camera coords (right, up, fwd) of the point on ray (u,v) at height dz_t above the camera; None if dep <= 1."""
    f, p, vh, _ = cam_consts(pitch, fov)
    if abs(v - vh) < 1e-6: return None
    dep = -f * dz_t / (math.cos(p) * (v - vh))
    if dep <= 1: return None
    return np.array([(u - 320) * dep / f, -(v - 320) * dep / f, dep])


def _ray(u, v, pitch, fov):
    """World (up, forward) per unit of the levelled camera's ray through (u, v)."""
    f, p, _vh, _ = cam_consts(pitch, fov); y, z = -(v - 320) / f, 1.0
    return y * math.cos(p) + z * math.sin(p), z * math.cos(p) - y * math.sin(p)


def quad_height(quad, pitch, fov, cam_h):
    """Metric height of a wall quad from the camera geometry alone: each side's bottom corner sits cam_h below the camera,
    which fixes its forward distance; the top corner at that distance gives the eave height. Mean of both sides, or None.
    The solid's wall may be taller than the real eave (a coplanar group with a higher part: netloft facade04 11.1 vs ~8 m)."""
    q = np.asarray(quad, np.float64).reshape(4, 2); hs = []
    for t, b in ((0, 3), (1, 2)):
        ub, fb = _ray(*q[b], pitch, fov); ut, ft = _ray(*q[t], pitch, fov)
        if ub >= -1e-6 or ft <= 1e-6: continue
        fwd = fb * (-cam_h / ub); hs.append(ut * fwd / ft + cam_h)
    return float(np.mean(hs)) if hs else None


def raise_top(quad, pitch, fov, cam_h, H_m):
    """Move TL / TR up their side lines to the model's top height H_m (bisection on v). The rows between the real eave
    and the model top are then wall the photo never saw (texture_wall marks them invalid)."""
    q = np.asarray(quad, np.float64).reshape(4, 2).copy()
    for t, b in ((0, 3), (1, 2)):
        ub, fb = _ray(*q[b], pitch, fov)
        if ub >= -1e-6: continue
        fwd = fb * (-cam_h / ub); d = q[t] - q[b]
        height = lambda s: (lambda ut, ft: ut * fwd / ft + cam_h if ft > 1e-6 else 1e9)(*_ray(*(q[b] + s * d), pitch, fov))
        lo, hi = 1.0, 1.0
        while height(hi) < H_m and hi < 8: hi *= 1.5
        for _ in range(30):
            mid = (lo + hi) / 2; lo, hi = (mid, hi) if height(mid) < H_m else (lo, mid)
        q[t] = q[b] + (lo + hi) / 2 * d
    return q.astype(np.float32)


def mode1d(vals, weights, lo, hi, kernel=3, nms=6, n=3, min_mass=40):
    """Weighted 1-D mode(s) on a 1 px grid with a ±kernel box: [(value, mass, index_mask)] best first."""
    grid = np.arange(math.floor(lo), math.ceil(hi) + 1); acc = np.zeros(len(grid))
    idx = np.round(np.asarray(vals) - grid[0]).astype(int)
    for i, w in zip(idx, weights):
        if 0 <= i < len(grid): acc[max(0, i - kernel):i + kernel + 1] += w
    out = []
    for _ in range(n):
        i = int(acc.argmax())
        if acc[i] < min_mass: break
        inl = np.abs(idx - i) <= kernel
        out.append((float(np.average(np.asarray(vals)[inl], weights=np.asarray(weights)[inl])), float(acc[i]), inl))
        acc[max(0, i - nms):i + nms + 1] = 0
    return out


def side_profile(lab, occ, r0, r1, uhs, vh, VP, gap=3, width=10, thr=18):
    """Rows (of r0..r1) where the wall looks different left vs right of the VP-tilted column at horizon u = uhs[k]
    (Lab distance of the two strips' means > thr — teal vs the pink neighbour at netloft #15 has the same grey level).
    A corner separates two faces (or wall from sky/ground) over its full height; a downspout or a mullion does not."""
    H, Wi = lab.shape[:2]; rows = np.arange(r0, r1 + 1)
    U = uhs[None, :] + (rows[:, None] - vh) * (VP[0] - uhs[None, :]) / (VP[1] - vh)
    cs = np.zeros((len(rows), Wi + 1, 3), np.float32); np.cumsum(lab[rows], axis=1, out=cs[:, 1:])   # row prefix sums: a strip mean is two lookups
    def strip(o0, o1):
        a = np.clip(np.round(U + o0).astype(int), 0, Wi - 1); b = np.clip(np.round(U + o1).astype(int) + 1, 1, Wi)
        return (cs[rows[:, None] - r0, b] - cs[rows[:, None] - r0, a]) / np.maximum(b - a, 1)[..., None]
    diff = np.linalg.norm(strip(-gap - width, -gap) - strip(gap, gap + width), axis=2)
    Uc = np.clip(np.round(U).astype(int), 0, Wi - 1)
    ok = ~occ[rows[:, None], Uc] & ~occ[rows[:, None], np.clip(Uc - gap - width, 0, Wi - 1)] & ~occ[rows[:, None], np.clip(Uc + gap + width, 0, Wi - 1)]
    ok &= (U - gap - width >= 0) & (U + gap + width <= Wi - 1)          # both strips inside the frame (a clipped strip is a fake edge)
    return ((diff > thr) & ok).sum(0).astype(float)


# ── the fit ─────────────────────────────────────────────────────────────────
def refine_quad(img_bgr, bld, occ, prior, wall_wh, pitch=8.0, fov=90.0, cam_h=None):
    prior = np.asarray(prior, np.float32).reshape(4, 2); Wm, Hm = wall_wh
    f, p, vh, VP = cam_consts(pitch, fov); H, Wi = bld.shape
    TL, TR, BR, BL = prior
    res = dict(quad=prior.copy(), conf=dict(top=0.0, bottom=0.0, s0=0.0, s1=0.0),
               src=dict(top="prior", bottom="prior", s0="prior", s1="prior"),
               evidence=np.zeros((0, 4), np.float32), partial=False, ok=False, prior=prior.copy(), vh=vh, vp=VP)
    h_px = (np.linalg.norm(BL - TL) + np.linalg.norm(BR - TR)) / 2
    if h_px < 100: return res                                       # 0. too far: nothing to snap to
    inside = np.zeros((H, Wi), np.uint8); cv2.fillPoly(inside, [np.clip(prior, -1e4, 1e4).round().astype(np.int32)], 1)
    if inside.sum() < 500 or bld[inside.astype(bool)].mean() < 0.25: return res   # the wall is not in this photo (bridge deck, big tree)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY); S = segs(gray); sky = sky_like(img_bgr)
    lab = cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB), (0, 0), 1.0).astype(np.float32)
    mid = (S[:, :2] + S[:, 2:4]) / 2; ang = np.degrees(np.arctan2(S[:, 3] - S[:, 1], S[:, 2] - S[:, 0])) % 180
    mx = np.clip(mid[:, 0].round().astype(int), 0, Wi - 1); my = mid[:, 1].round().astype(int)
    yc, ya, yb = (np.clip(my + d, 0, H - 1) for d in (0, -6, 6))
    ev = []

    # 2. TOP: RANSAC line over roofline-like segments (building below, open above, not an occluder)
    top = hline(TL, TR); d0 = TR - TL; a0 = math.degrees(math.atan2(d0[1], d0[0])); band = max(60, 0.2 * h_px)
    open_above = ~bld[ya, mx] & (sky[ya, mx] | (my - 6 < 0))       # sky (or off-frame) above: a canopy boundary 50-80 px under the eave has leaves above it
    cand = (pdist(S[:, :2], top) <= band) & (pdist(S[:, 2:4], top) <= band) & (angdiff(ang, a0) <= 12) & bld[yb, mx] & open_above & ~occ[yb, mx]   # occ checked on the building side: sky (an occluder) touches every eave
    C = S[cand]; top_conf, run = 0.0, None
    if len(C):
        best = (0.0, None)
        for c in C:                                                  # ponytail: exhaustive O(N^2) over N ~ 10-60 roofline candidates; fine at 640 px
            l = hline(c[:2], c[2:4]); inl = (pdist(C[:, :2], l) <= 3) & (pdist(C[:, 2:4], l) <= 3)
            if C[inl, 4].sum() > best[0]: best = (float(C[inl, 4].sum()), inl)
        top_conf = min(1.0, best[0] / float(np.linalg.norm(d0)))
        if top_conf >= 0.15:
            P = np.vstack([C[best[1], :2], C[best[1], 2:4]]).astype(np.float32)
            vx, vy, x0, y0 = cv2.fitLine(P, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            top = hline((x0, y0), (x0 + vx, y0 + vy)); res["src"]["top"] = "lsd"; run = [float(P[:, 0].min()), float(P[:, 0].max())]
            ev.append(C[best[1], :4])
            # the eave's u-extent: LSD fragments the roof edge where its contrast fades, but the fascia/soffit lines
            # (building above AND below, a few px under the roof edge) run to the same corner — chain them, gap <= 12 px
            b1 = v_at(top, S[:, 0]) - S[:, 1]; b2 = v_at(top, S[:, 2]) - S[:, 3]   # both endpoints: negative = under the top line
            on = (np.abs(b1) <= 3) & (np.abs(b2) <= 3); under = (b1 <= -3) & (b2 <= -3) & (b1 >= -0.15 * h_px) & (b2 >= -0.15 * h_px) & bld[ya, mx]
            band = (angdiff(ang, math.degrees(math.atan2(-top[0], top[1]))) <= 3) & bld[yb, mx] & ~occ[yb, mx] & (on | under)
            ext = sorted((min(a, b), max(a, b)) for a, b in S[band][:, [0, 2]])
            for _ in range(4):
                for a, b in ext:
                    if a <= run[1] + 12 and b >= run[0] - 12: run = [min(run[0], a), max(run[1], b)]
    res["conf"]["top"] = float(top_conf)
    Lb0 = hline(BL, BR)
    if res["src"]["top"] != "lsd":
        # no eave: the wall's horizontal VP from its own horizontals (sills, awnings, siding seams) inside the prior — the pose
        # heading is off by a few degrees, which shears the strip; the prior's height stays, only the slope is re-fitted
        inq = inside[np.clip(my, 0, H - 1), mx].astype(bool) & bld[yc, mx] & ~occ[yc, mx] & (angdiff(ang, a0) <= 15) & (S[:, 4] >= 20)
        if inq.sum() >= 3:
            uh_ = np.array([u_at(hline(c[:2], c[2:4]), vh) for c in S[inq]]); fin = np.isfinite(uh_)
            phi = np.degrees(np.arctan((uh_[fin] - 320) / f)); phi0 = math.degrees(math.atan((u_at(top, vh) - 320) / f)) if abs(top[0]) > 1e-9 else 0.0
            m = mode1d(phi, S[inq, 4][fin], phi0 - 12, phi0 + 12, kernel=1, nms=3, n=1, min_mass=60)
            if m:
                vph = (320 + f * math.tan(math.radians(m[0][0])), vh)
                top = hline((TL + TR) / 2, vph); Lb0 = hline((BL + BR) / 2, vph); res["src"]["top"] = "prior"; res["vp_h"] = float(vph[0])
    VPh = np.cross(top, [0.0, 1.0, -vh])                             # horizontal VP of the wall, kept homogeneous (may be ideal)

    # 3. SIDES: LSD verticals through VP + building-mask free ends, gated by the top run's end / a free end
    dz_t = None if cam_h is None else Hm - cam_h
    to_vp = VP - mid; a_vp = np.degrees(np.arctan2(to_vp[:, 1], to_vp[:, 0]))
    uh_seg = mid[:, 0] + (vh - mid[:, 1]) * (VP[0] - mid[:, 0]) / (VP[1] - mid[:, 1])
    def sline(uh): return hline((uh, vh), VP)
    def uh_of(u, v): return u + (vh - v) * (VP[0] - u) / (VP[1] - v)
    lines, peaks, prior_uh = {}, {}, {}
    sb = float(np.clip(0.15 * np.linalg.norm(d0), 70, 120))         # side search band: a 3 m pose error is 100 px on a 700 px wide wall (netloft #15)
    for name, A, B, out_dir in (("s0", TL, BL, int(np.sign(TL[0] - TR[0]) or 1)), ("s1", TR, BR, int(np.sign(TR[0] - TL[0]) or 1))):
        lines[name] = hline(A, B); uhp = u_at(lines[name], vh); prior_uh[name] = uhp; peaks[name] = []
        if not (-5 <= A[0] <= Wi + 5): res["src"][name] = "offframe"; res["partial"] = True; continue
        sel = (S[:, 4] >= max(15, 0.1 * h_px)) & (angdiff(ang, a_vp) <= 4) & ~occ[yc, mx] & (np.abs(uh_seg - uhp) <= sb)
        vals = list(uh_seg[sel]); wts = list(S[sel, 4]); srcs = ["lsd"] * int(sel.sum()); segs_used = list(S[sel, :4])
        hs = B[1] - A[1]; r0, r1 = int(max(0, A[1] + 0.15 * hs)), int(min(H - 1, B[1] - 0.15 * hs))
        free_u = []; far = None
        if r1 - r0 >= 10:
            uhs = np.arange(math.floor(uhp - sb), math.ceil(uhp + sb) + 1, dtype=float); prof = side_profile(lab, occ, r0, r1, uhs, vh, VP)
            far = side_profile(lab, occ, r0, r1, uhs, vh, VP, gap=12, width=18)   # do the two sides differ BROADLY? posts and mullions do not
            cov = bld[r0:r1 + 1].mean(0); oc = (bld[r0:r1 + 1] | occ[r0:r1 + 1]).mean(0); inw = -out_dir
            for c in range(13, Wi - 13):
                if not (cov[c] < 0.4 <= cov[c + inw] and cov[c + 5 * inw] > 0.6 and cov[c + 4 * out_dir] < 0.2): continue
                uh = uh_of(c, (r0 + r1) / 2)
                if abs(uh - uhp) > sb: continue
                if oc[c + out_dir * 1:c + out_dir * 13:out_dir].mean() > 0.2:
                    # an occluder continues past the mask's end: a trunk cutting the mask, or a corner with a tree beyond it (netloft #15,
                    # the pink neighbour behind its canopy). A corner keeps the mask away for >= 40 px AND shows a colour edge on the free rows.
                    beyond = cov[c + out_dir:c + out_dir * 41:out_dir]
                    if beyond.size < 20 or beyond.max() > 0.2 or prof[np.abs(uhs - uh) <= 3].max(initial=0) < 0.15 * (r1 - r0): continue
                vals.append(uh); wts.append(float(r1 - r0)); srcs.append("mask"); segs_used.append(np.array([c, r0, c, r1], np.float32))
                free_u.append(meet(sline(uh), top)[0])
            pk = mode1d(uhs, prof, uhs[0], uhs[-1], kernel=0, nms=6, n=4, min_mass=0.3 * (r1 - r0))
            for uh, mass, inl in pk:                                    # one observation per profile peak, weight = rows
                vals.append(uh); wts.append(mass); srcs.append("lsd"); segs_used.append(np.array([uh_of(uh, vh) + (r0 - vh) * (VP[0] - uh) / (VP[1] - vh), r0, uh + (r1 - vh) * (VP[0] - uh) / (VP[1] - vh), r1], np.float32))
        anchors = list(free_u)
        if run is not None:
            u_ext = run[1] if out_dir > 0 else run[0]
            if 20 < u_ext < Wi - 20: anchors.append(u_ext)
        for uh, mass, inl in mode1d(vals, wts, uhp - sb, uhp + sb):
            u_top = meet(sline(uh), top)[0]
            res.setdefault("modes", []).append((name, round(uh, 1), round(mass), round(float(u_top), 1), [round(float(a), 1) for a in anchors]))
            near = np.abs(uhs - uh) <= 3 if far is not None else None                      # of the rows with an edge here, do the two broad sides differ too?
            broad = far is not None and near.any() and far[near].max() >= 0.5 * max(prof[near].max(), 0.2 * (r1 - r0))
            if not broad and (not anchors or min(abs(u_top - a) for a in anchors) > 10): continue   # a post / downspout / mullion
            src = "mask" if sum(w for w, s, k in zip(wts, srcs, inl) if k and s == "mask") > sum(w for w, s, k in zip(wts, srcs, inl) if k and s == "lsd") else "lsd"
            peaks[name].append(dict(uh=uh, mass=mass, conf=min(1.0, mass / 150), src=src, segs=[s for s, k in zip(segs_used, inl) if k]))

    res["run"] = run; res["peaks"] = {k: [(round(c["uh"], 1), round(c["mass"]), c["src"]) for c in v] for k, v in peaks.items()}
    # 4. WIDTH consistency: a pair of corners must be W_m apart along the eave; a lone corner predicts the other
    def eave_of(uh):
        q = meet(sline(uh), top); return None if dz_t is None else eave_point(q[0], q[1], pitch, fov, dz_t)
    other = {"s0": "s1", "s1": "s0"}; chosen = {}
    if dz_t is not None and peaks["s0"] and peaks["s1"]:
        best = None
        for c0 in peaks["s0"]:
            for c1 in peaks["s1"]:
                P0, P1 = eave_of(c0["uh"]), eave_of(c1["uh"])
                if P0 is None or P1 is None or abs(np.linalg.norm(P0 - P1) - Wm) > 0.08 * Wm + 0.3: continue
                if best is None or c0["mass"] + c1["mass"] > best[0]: best = (c0["mass"] + c1["mass"], c0, c1)
        if best: chosen = {"s0": best[1], "s1": best[2]}
    elif dz_t is None and peaks["s0"] and peaks["s1"]:
        chosen = {"s0": peaks["s0"][0], "s1": peaks["s1"][0]}
    if not chosen and (peaks["s0"] or peaks["s1"]):
        a = max((pk[0] for pk in peaks.values() if pk), key=lambda c: c["mass"]); name = "s0" if a in peaks["s0"] else "s1"
        chosen = {name: a}; o = other[name]
        pred = None
        if dz_t is not None:                                           # walk W_m along the eave from the found corner
            P0 = eave_of(a["uh"]); fc = (TR if name == "s0" else TL); Pf = eave_point(fc[0], fc[1], pitch, fov, dz_t)
            if P0 is not None and Pf is not None and np.linalg.norm(Pf - P0) > 0.5:
                P1 = P0 + Wm * (Pf - P0) / np.linalg.norm(Pf - P0)
                if P1[2] > 1:
                    u1, v1 = 320 + f * P1[0] / P1[2], 320 - f * P1[1] / P1[2]
                    pred = (sline(uh_of(u1, v1)), "width", 0.5 * a["conf"])
        if pred is None: pred = (sline(prior_uh[o] + a["uh"] - prior_uh[name]), res["src"][o] if res["src"][o] == "offframe" else "prior", 0.0)
        lines[o], res["src"][o], res["conf"][o] = pred
    for name, c in chosen.items():
        lines[name] = sline(c["uh"]); res["src"][name] = c["src"]; res["conf"][name] = c["conf"]; ev += [np.asarray(c["segs"]).reshape(-1, 4)]

    # 5. BOTTOM: mask/ground boundary columns with nothing (no car) below, on a line through VPh; else derived
    ua, ub = sorted((meet(lines["s0"], Lb0)[0], meet(lines["s1"], Lb0)[0]))
    cols = np.arange(int(max(0, math.ceil(ua))), int(min(Wi - 1, math.floor(ub))) + 1); pts = []
    for u in cols:
        vb = v_at(Lb0, u)
        if not np.isfinite(vb): continue
        r0, r1 = int(max(0, vb - 50)), int(min(H - 7, vb + 50))
        rows = np.flatnonzero(bld[r0:r1 + 1, u])
        if not len(rows): continue
        y = r0 + rows[-1]
        if not (bld[y + 1, u] or occ[y + 1, u] or bld[y + 6, u] or occ[y + 6, u]): pts.append((u, y))
    bottom = Lb0
    if len(pts):
        pts = np.array(pts, float); v320 = np.array([v_at(np.cross([u, y, 1.0], VPh), 320) for u, y in pts])
        ok = np.isfinite(v320); pts, v320 = pts[ok], v320[ok]
        m = mode1d(v320, np.ones(len(v320)), v320.min(), v320.max(), n=1, min_mass=1) if len(v320) else []
        if m:
            inl = m[0][2]; cover = inl.sum() / max(1, len(cols))
            if cover >= 0.20:
                bottom = np.cross([320.0, float(np.median(v320[inl])), 1.0], VPh); res["src"]["bottom"] = "mask"; res["conf"]["bottom"] = float(min(1.0, cover))
                ev.append(np.column_stack([pts[inl], pts[inl] + [0, 1]]).astype(np.float32))
    if res["src"]["bottom"] == "prior" and dz_t is not None:
        two = []
        for l in (lines["s0"], lines["s1"]):
            u, vt = meet(l, top)
            if not np.isfinite(vt) or vt >= vh - 1: continue
            dep = -f * dz_t / (math.cos(p) * (vt - vh)); dep_b = dep - Hm * math.sin(p)
            if dep_b <= 1: continue
            v_b = vh + f * cam_h / (math.cos(p) * dep_b); two.append((u_at(hline((u, vt), VP), v_b), v_b))
        if len(two) == 2 and abs(two[0][0] - two[1][0]) > 1:
            bottom = hline(*two); res["src"]["bottom"] = "derived"; res["conf"]["bottom"] = 0.5

    # 6. corners + sanity
    quad = np.array([meet(top, lines["s0"]), meet(top, lines["s1"]), meet(bottom, lines["s1"]), meet(bottom, lines["s0"])], np.float32)
    res["evidence"] = np.vstack(ev).astype(np.float32) if ev else np.zeros((0, 4), np.float32); res["quad_raw"] = quad
    def area(q): return 0.5 * abs(cross2(q[1] - q[0], q[2] - q[0]) + cross2(q[2] - q[0], q[3] - q[0]))
    e = np.roll(quad, -1, 0) - quad; cr = cross2(e, np.roll(e, -1, 0))
    if (np.isfinite(quad).all() and (np.all(cr > 0) or np.all(cr < 0)) and quad[3, 1] - quad[0, 1] >= 30 and quad[2, 1] - quad[1, 1] >= 30
            and 0.5 * area(prior) <= area(quad) <= 2 * area(prior)):
        res["quad"] = quad; res["ok"] = True
    else:
        res["conf"] = dict(top=0.0, bottom=0.0, s0=0.0, s1=0.0); res["src"] = dict(top="prior", bottom="prior", s0="prior", s1="prior")
    return res


def draw(img_bgr, res):
    out = img_bgr.copy()
    def poly(q): return [np.clip(np.nan_to_num(q, nan=0, posinf=1e5, neginf=-1e5), -1e5, 1e5).round().astype(np.int32)]
    cv2.polylines(out, poly(res["prior"]), True, (255, 0, 255), 1, cv2.LINE_AA)
    cv2.polylines(out, poly(res["quad"]), True, (0, 255, 0), 2, cv2.LINE_AA)
    for x1, y1, x2, y2 in res["evidence"]: cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 140, 255), 1)
    c, s = res["conf"], res["src"]; fmt = lambda x: f"{x:.2f}"[1:] if x < 1 else "1.0"
    txt = f"top {fmt(c['top'])} {s['top']} | s0 {fmt(c['s0'])} {s['s0']} | s1 {fmt(c['s1'])} {s['s1']} | bot {fmt(c['bottom'])} {s['bottom']}" + ("" if res["ok"] else " | NOT OK")
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(out, (2, 2), (tw + 10, th + 10), (0, 0, 0), -1)         # cv2 5 changes glyph advance with thickness: box, not outline
    cv2.putText(out, txt, (6, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ── self-check on real data ─────────────────────────────────────────────────

# The self-check lives with the pilot that owns the data: shadowCity2 tools/pilot/facade_quad.py
