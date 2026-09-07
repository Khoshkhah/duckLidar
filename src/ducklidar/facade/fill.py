"""Fill the holes of a facade texture with plausible wall — never a smear.

A wall texture (row 0 = top of wall, column 0 = the s0 end, TEX_M m/px) comes with a
`valid` mask: texels painted from a photo. The rest are holes — cars, people, trees,
sky rows above a low eave, unseen wall ends. Facades repeat horizontally (siding,
window bays), not vertically, so the fill is horizontal too:

  1. period copy   — if the column profile has a strong horizontal period L, a hole
                     texel takes the texel L, 2L, 3L columns away (rare on the pilot data).
  2. wall colour   — every remaining hole texel takes ONE colour: the median Lab of WALL-LIKE
                     painted texels (within 35 Lab of the painted median, upper 60 % of rows)
                     — unseen wall is unknown wall, painted flat. (A per-row field, tried first,
                     interpolated a different median per row and read as horizontal streaks on
                     walls seen only 20-50 %: the critics' main complaint.)
  3. thin inpaint  — only holes thinner than 9 px are inpainted (TELEA); blobs never are,
                     that is where smears come from.
  4. soft borders  — a 3 px band inside each hole blends towards a sigma-1 blur so the
                     seam is not a hard line; no colour is invented.

Hard rules: no resize, no nearest-neighbour extension, colours only from valid texels
of the same texture, out.shape == tex.shape and out[valid] == tex[valid] exactly.

    PILOT_BUILDING=railspur python tools/pilot/fill.py   # self-check on photo #2 / facade02
"""
import cv2
import numpy as np

TEX_M = 0.04                          # m/px, same as build_model.TEX_M (not imported: build_model pulls ducklidar)
SRC_VALID, SRC_PERIOD, SRC_ROW, SRC_INPAINT = 1, 2, 3, 4


def period_px(tex, valid, min_m=1.0, max_m=10.0, min_strength=0.5):
    """Horizontal period of the texture in px, or None when the autocorrelation peak is weak.

    Column profile p[c] = mean |Sobel_x(grey)| over the valid texels of column c (columns with
    < 30% valid rows get the profile's median), mean-removed; normalised autocorrelation
    r[L] over the columns where both c and c+L are >= 30% valid; L = the strongest local
    maximum in [min_m, max_m]."""
    h, w = valid.shape
    grey = cv2.cvtColor(tex, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sx = np.abs(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3))
    cnt = valid.sum(0); okc = cnt >= 0.3 * h
    if okc.sum() < 2 * int(min_m / TEX_M): return None
    p = np.zeros(w); p[okc] = (sx * valid).sum(0)[okc] / cnt[okc]; p[~okc] = np.median(p[okc])
    p -= p.mean()
    L0, L1 = int(min_m / TEX_M), min(int(max_m / TEX_M), w - 2)
    if L1 <= L0: return None
    r = np.full(L1 + 2, -np.inf)
    for L in range(L0 - 1, L1 + 2):
        both = okc[:w - L] & okc[L:]
        a, b = p[:w - L][both], p[L:][both]
        den = np.sqrt((a ** 2).sum() * (b ** 2).sum())                  # symmetric normalisation: |r| <= 1, a multiple cannot outscore the fundamental
        r[L] = (a * b).sum() / den if den > 0 and both.sum() >= L0 else -np.inf
    Ls = np.arange(L0, L1 + 1)
    peak = Ls[(r[Ls] > r[Ls - 1]) & (r[Ls] >= r[Ls + 1]) & np.isfinite(r[Ls])]
    if len(peak) == 0: return None
    best = r[peak].max()
    L = int(peak[r[peak] >= best - 0.05][0])                          # the fundamental, not a multiple that ties with it
    return L if r[L] >= min_strength else None


def wall_rows(tex, filled, win=8, lab_gate=35.0, min_n=200, sigma=4.0):
    """-> uint8 (h, 3) RGB: the wall colour of every texture row (median Lab of wall-like painted texels in a
    +-win row window; rows with < min_n such texels are linearly interpolated from the nearest rows that have them;
    the profile is then Gaussian-smoothed over sigma rows). Wall-like = within lab_gate of the texture's median Lab,
    so shop glass, sky rims and leaves painted by a loose mask do not set the fill colour."""
    h, w = filled.shape
    lab = cv2.cvtColor(tex, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref = np.median(lab[filled], axis=0)
    wall = filled & (np.linalg.norm(lab - ref, axis=2) < lab_gate)
    if wall.sum() < 50: wall = filled
    rows = np.full((h, 3), np.nan, np.float32)
    for r in range(h):
        sel = wall[max(0, r - win):r + win + 1]
        if sel.sum() >= min_n: rows[r] = np.median(lab[max(0, r - win):r + win + 1][sel], axis=0)
    good = np.flatnonzero(np.isfinite(rows[:, 0]))
    if len(good) == 0: rows[:] = ref
    else:
        for k in range(3): rows[:, k] = np.interp(np.arange(h), good, rows[good, k])      # linear between, held constant beyond the ends
    if sigma > 0: rows = cv2.GaussianBlur(rows.reshape(h, 1, 3), (0, 0), sigmaY=sigma, sigmaX=0.01).reshape(h, 3)
    return cv2.cvtColor(np.clip(rows, 0, 255).astype(np.uint8).reshape(h, 1, 3), cv2.COLOR_LAB2RGB).reshape(h, 3)


def wall_colour(tex, filled, lab_gate=35.0):
    """uint8 RGB(3): the wall colour of a texture - median Lab of wall-like painted texels (within lab_gate of the
    painted median, so glass, sky rims and leaves do not vote), taken from the UPPER 60 % of the painted rows where
    shopfronts do not dominate."""
    h = filled.shape[0]; lab = cv2.cvtColor(tex, cv2.COLOR_RGB2LAB).astype(np.float32)
    top = np.zeros_like(filled); top[: max(1, int(h * 0.6))] = True
    sel = filled & top
    if sel.sum() < 200: sel = filled
    ref = np.median(lab[sel], axis=0); wall = sel & (np.linalg.norm(lab - ref, axis=2) < lab_gate)
    if wall.sum() < 50: wall = sel
    col = np.median(lab[wall], axis=0)
    return cv2.cvtColor(np.clip(col, 0, 255).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_LAB2RGB).reshape(3)


def fill(tex, valid, min_valid=0.15, flat_rgb=None):
    """-> (out uint8 (h,w,3), info dict(src 'flat'|'photo', valid_frac, period_m, n_period, n_row, n_inpaint, src_map))."""
    h, w = valid.shape
    valid = valid.astype(bool); valid_frac = float(valid.mean())
    if valid_frac < min_valid:
        col = flat_rgb if flat_rgb is not None else (np.median(tex[valid], axis=0) if valid.sum() >= 50 else (150, 150, 150))
        out = np.tile(np.asarray(col, np.uint8).reshape(1, 1, 3), (h, w, 1))
        return out, dict(src="flat", valid_frac=round(valid_frac, 3), period_m=None, n_period=0, n_row=0, n_inpaint=0, src_map=None)

    out = tex.copy(); filled = valid.copy(); src = np.where(valid, SRC_VALID, 0).astype(np.uint8)
    # 1. period copy
    L = period_px(tex, valid); n_period = 0
    if L:
        cols = np.arange(w)
        for k in (1, -1, 2, -2, 3, -3):
            s = k * L; inrange = (cols + s >= 0) & (cols + s < w)
            take = ~filled & np.roll(valid, -s, axis=1) & inrange[None, :]
            if take.any():
                out[take] = np.roll(tex, -s, axis=1)[take]; filled[take] = True; src[take] = SRC_PERIOD; n_period += int(take.sum())
    hole = ~filled                                                    # what steps 3-4 consider a hole
    # 2. wall field: per row the median Lab of wall-like painted texels in a +-8 row window, interpolated, smoothed
    n_row = int(hole.sum())
    if n_row:
        # ONE wall colour for every unseen texel. The per-row field (wall_rows) interpolated a different
        # median per row and, on walls seen only 20-50 %, that read as horizontal streaks - the critics'
        # main complaint. Unseen wall is unknown wall: paint it the building's wall colour, flat.
        out[hole] = wall_colour(out, filled); src[hole] = SRC_ROW
    # 3. thin-hole inpaint only: holes that do not survive a 9x9 opening (thinner than 9 px)
    k9 = np.ones((9, 9), np.uint8); hu = hole.astype(np.uint8)
    small = hole & (cv2.morphologyEx(hu, cv2.MORPH_OPEN, k9) == 0)
    n_inpaint = 0
    if small.any() and small.mean() < 0.3:
        out = cv2.inpaint(out, small.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA); src[small] = SRC_INPAINT; n_inpaint = int(small.sum())
    # 4. soften the hole borders: a 3 px band inside the holes blends towards a sigma-1 blur
    if hole.any():
        dist = cv2.distanceTransform(hu, cv2.DIST_L2, 3)
        band = hole & (dist <= 3)
        if band.any():
            a = np.clip(dist / 3.0, 0, 1)[..., None]
            blur = cv2.GaussianBlur(out, (0, 0), 1.0)
            mix = a * out.astype(np.float32) + (1 - a) * blur.astype(np.float32)
            out[band] = np.clip(mix[band] + 0.5, 0, 255).astype(np.uint8)
    out[valid] = tex[valid]                                            # the hard rule, whatever OpenCV did at the borders
    return out, dict(src="photo", valid_frac=round(valid_frac, 3), period_m=round(L * TEX_M, 2) if L else None,
                     n_period=n_period, n_row=n_row, n_inpaint=n_inpaint, src_map=src)


def qa(tex, valid, out, info=None):
    """tex with holes in magenta | out | source map (valid grey, period green, row blue, inpaint red)."""
    a = tex.copy(); a[~valid] = (255, 0, 255)
    src = info.get("src_map") if info else None
    pal = np.array([[0, 0, 0], [128, 128, 128], [0, 200, 0], [40, 80, 255], [230, 30, 30]], np.uint8)
    m = pal[src] if src is not None else np.where(valid[..., None], pal[1], pal[0])
    sep = np.full((tex.shape[0], 4, 3), 255, np.uint8)
    return np.hstack([a, sep, out, sep, m])

# The self-check lives with the pilot that owns the data: shadowCity2 tools/pilot/facade_fill.py
