"""What is in a LAS file, and what it means.

The census in :func:`describe` is the first thing to run on any new survey. **Which fields a
file carries, and which are present but empty, is a fact about that survey** — not about
LiDAR — and it decides what you can do with the data. Two files over the same ground
disagreed completely: NRCan's 2016 cloud is 88.8% *unclassified* with no vegetation or
building class at all, while Vancouver's 2022 tile has building 26.8% and high vegetation
23.0%. One can separate canopy from roof; the other cannot.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

#: ASPRS standard classification codes.
ASPRS = {
    0: "never classified", 1: "unclassified", 2: "ground", 3: "low vegetation",
    4: "medium vegetation", 5: "high vegetation", 6: "building", 7: "noise",
    8: "model key-point", 9: "water", 10: "rail", 11: "road surface",
    12: "overlap (legacy)", 13: "wire guard", 14: "wire conductor", 15: "transmission tower",
    16: "wire connector", 17: "bridge deck", 18: "high noise",
}

VEGETATION = (3, 4, 5)
GROUND = 2
BUILDING = 6

#: What each stored field is for. The notes are what they turn out to mean in practice, not
#: what the specification says.
MEANING = {
    "X": "raw integer easting — x = X * scale + offset",
    "Y": "raw integer northing (negative when the offset is large — not corruption)",
    "Z": "raw integer elevation above the vertical datum",
    "intensity": "echo strength. NOT calibrated; falls off with range and angle",
    "return_number": "which echo of this pulse, 1 = first thing hit",
    "number_of_returns": "echoes this pulse produced. >1 means it went through something",
    "synthetic": "flag: modelled, not measured",
    "key_point": "flag: keep when thinning",
    "withheld": "flag: the producer says do not use this point",
    "overlap": "flag: in the overlap between two flight lines",
    "scanner_channel": "which scanner head, multi-channel sensors",
    "scan_direction_flag": "mirror sweeping left-to-right or right-to-left",
    "edge_of_flight_line": "flag: last point of a scan line — the swath edge",
    "classification": "ASPRS class — the field that decides what a cloud is worth",
    "user_data": "free byte, producer-defined",
    "scan_angle": "degrees off vertical in 0.006 deg units. Provenance, not input",
    "point_source_id": "which flight line produced this point",
    "gps_time": "when the pulse was fired — the only field that dates the survey",
    "red": "colour, usually sampled from a concurrent orthophoto",
    "green": "colour",
    "blue": "colour",
    "nir": "near infrared, where the survey captured it",
}


#: What each field is *for* — the question it answers, and how much it is worth. Ordered by
#: how often it earns its place, because a flat list of 22 fields implies they matter equally
#: and they do not: four are load-bearing, most surveys leave eight of them empty.
#:
#: ``tier`` is one of:
#:   "core"        you will use it every time
#:   "situational" reach for it when a specific question comes up
#:   "diagnostic"  provenance — use it to distrust data, not to compute with
#:   "usually dead" present in the format, empty in most files
USES = {
    "x": ("core", "position. The data."),
    "y": ("core", "position. The data."),
    "z": ("core", "elevation above the vertical datum — NOT height above ground"),
    "classification": ("core",
        "decides what a cloud is worth. Census it first: with vegetation AND building "
        "classes you can grid canopy and roof separately; without them you cannot"),
    "number_of_returns": ("situational",
        "the best canopy proxy when a survey has no vegetation class. Multi-return points "
        "sit a median 6.34 m above local ground against 0.50 m for single returns"),
    "return_number": ("situational",
        "with number_of_returns: 'last of many' is the classic cheap ground proxy"),
    "point_source_id": ("situational",
        "which flight line. Two lines over one cell disagreeing IS the interval "
        "— see rasterize(interval=True)"),
    "gps_time": ("situational",
        "the only field that dates the survey. The header's creation_date can be months "
        "later — five, on the Vancouver tile"),
    "red": ("situational", "colourised 3-D output, and a check on classification: a "
                           "'vegetation' point that is grey is not vegetation"),
    "green": ("situational", "see red"),
    "blue": ("situational", "see red"),
    "nir": ("situational", "vegetation index work, where the survey captured it"),
    "intensity": ("diagnostic",
        "NOT calibrated and never corrected for range or angle, so geometry dominates: the "
        "darkest returns are oblique partial hits, not dark materials. A hint, not a "
        "measurement"),
    "scan_angle": ("diagnostic",
        "provenance — it is already inside x, y, z. Worth filtering on only in wide-swath "
        "surveys: useless on Vancouver 2022 (accuracy flat to its 20 deg limit), meaningful "
        "on NRCan 2016 at +-48 deg"),
    "withheld": ("diagnostic", "the producer says do not use this point. Honour it if set"),
    "overlap": ("diagnostic", "thin doubled-up passes. Empty in the Vancouver tile"),
    "edge_of_flight_line": ("diagnostic",
        "drop unreliable swath-edge points. Empty in the Vancouver tile"),
    "synthetic": ("usually dead", "modelled rather than measured"),
    "key_point": ("usually dead", "keep when thinning"),
    "scanner_channel": ("usually dead", "multi-channel sensors only"),
    "scan_direction_flag": ("usually dead", "mirror direction. No known use here"),
    "user_data": ("usually dead", "free byte, producer-defined"),
    "X": ("core", "raw integer form of x — x = X * scale + offset"),
    "Y": ("core", "raw integer form of y (negative when the offset is large)"),
    "Z": ("core", "raw integer form of z"),
}

TIER_ORDER = ("core", "situational", "diagnostic", "usually dead")


def field_guide(src=None, classification=None, file=None):
    """What every field is for, and — given a file — which of them that file actually has.

    The list of fields is not a list of equals. Four are load-bearing, a handful are worth
    reaching for when a specific question comes up, two are provenance you use to *distrust*
    data rather than compute with, and most surveys leave eight of them entirely empty.
    Passing a file marks each one populated, empty or absent, so the answer is about the data
    in front of you rather than the format in the abstract.
    """
    out = print if file is None else (lambda *a: print(*a, file=file))
    status = {}
    if src is not None:
        import laspy

        from .read import _opener

        with _opener(src)() as fh:
            pts = fh.read()
        for d in pts.point_format.dimensions:
            a = np.asarray(pts[d.name])
            uniq = np.unique(a)
            status[d.name] = "empty" if len(uniq) == 1 and uniq[0] == 0 else "populated"
        classification = pts.classification if classification is None else classification

    # laspy exposes x/y/z as scaled accessors while the file stores raw X/Y/Z, so look the
    # coordinates up under their stored names or they vanish from the guide entirely.
    stored = {"x": "X", "y": "Y", "z": "Z"}
    for tier in TIER_ORDER:
        names = [k for k, (t, _) in USES.items() if t == tier and not k.isupper()]
        if src is not None:
            names = [n for n in names if stored.get(n, n) in status] or names
        out(f"\n{tier.upper()}")
        for name in names:
            key = stored.get(name, name)
            mark = {"populated": "*", "empty": "-", None: " "}.get(status.get(key), "?")
            if src is not None and key not in status:
                mark = "x"
            out(f"  [{mark}] {name:22s} {USES[name][1]}")
    if src is not None:
        out("\n  [*] populated   [-] present but all zero   [x] not in this point format")
    if classification is not None:
        rows, sep = census(classification)
        out(f"\n  classification here: "
            + ", ".join(f"{n} {s:.0%}" for _, n, _, s in sorted(rows, key=lambda r: -r[2])[:5]))
        out(f"  -> canopy {'CAN' if sep else 'CANNOT'} be separated from roof, "
            f"which is the single fact that decides what this cloud is good for")

SCAN_ANGLE_DEG = 0.006          # LAS 1.4 stores scan angle in these units
GPS_LEAP_SECONDS = 18           # GPS runs ahead of UTC; 18 since 2017


def census(classification):
    """Counts and shares per class, plus whether canopy can be told from roof.

    Returns ``(rows, can_separate)`` where rows are ``(code, name, count, share)``.
    """
    c = np.asarray(classification)
    # bincount, not unique: a class code is a small non-negative integer, so counting is O(n)
    # where np.unique sorts. 26 ms against 80 ms on 10.7 M returns, same answer.
    if c.dtype.kind in "ui" and c.size:
        n = np.bincount(c.ravel(), minlength=256)
        codes = np.nonzero(n)[0]
        counts = n[codes]
    else:
        codes, counts = np.unique(c, return_counts=True)
    total = counts.sum() or 1
    rows = [(int(c), ASPRS.get(int(c), "?"), int(n), float(n / total))
            for c, n in zip(codes, counts)]
    present = set(codes.tolist())
    can_separate = bool(set(VEGETATION) & present) and BUILDING in present
    return rows, can_separate


def flight_dates(gps_time, gps_time_type="STANDARD", leap_seconds=GPS_LEAP_SECONDS):
    """`(first, last)` UTC datetimes a survey was flown, from `gps_time`.

    The header's `creation_date` is when the *file* was written, which can be months later —
    on the Vancouver tile it is five months after the plane landed. This is the only field
    that dates the data. "Standard" GPS time is stored with 1e9 subtracted.
    """
    t = np.asarray(gps_time)
    if not t.size:
        return None, None
    adj = 1e9 if str(gps_time_type).upper().endswith("STANDARD") else 0.0
    epoch = _dt.datetime(1980, 1, 6, tzinfo=_dt.timezone.utc)
    return tuple(epoch + _dt.timedelta(seconds=float(v) + adj - leap_seconds)
                 for v in (t.min(), t.max()))


def describe(src, sample_bbox=None, file=None):
    """Print the complete inventory of a file: header, CRS, every field, returns histogram.

    Fields that are present but entirely zero are called out, because that is usually a
    format conversion having silently dropped them — `overlap` and `edge_of_flight_line`
    especially, which are how you thin doubled-up passes and drop swath-edge points.
    """
    import laspy

    from .read import _opener

    out = print if file is None else (lambda *a: print(*a, file=file))
    src = str(src)
    if src.startswith("http"):
        from laspy.copc import Bounds

        with laspy.CopcReader.open(src) as r:
            hdr = r.header
            b = sample_bbox or (hdr.mins[0], hdr.mins[1],
                                hdr.mins[0] + 100, hdr.mins[1] + 100)
            pts = r.query(Bounds(mins=np.array(b[:2]), maxs=np.array(b[2:])))
        note = "  (ranges from a sample — a COPC is not read whole)"
    else:
        with _opener(src)() as fh:
            hdr, pts = fh.header, fh.read()
        note = ""

    out("HEADER")
    out(f"  LAS {hdr.version}, point format {hdr.point_format.id} "
        f"({hdr.point_format.size} bytes/point), {hdr.point_count:,} points")
    out(f"  written {hdr.creation_date} by {hdr.generating_software.strip()!r} "
        f"on {hdr.system_identifier.strip()!r}")
    out(f"  scales {hdr.scales} -> {hdr.scales[0]*1000:g} mm    offsets {hdr.offsets}")
    out(f"  bounds x {hdr.mins[0]:,.1f}–{hdr.maxs[0]:,.1f}  "
        f"y {hdr.mins[1]:,.1f}–{hdr.maxs[1]:,.1f}  z {hdr.mins[2]:,.1f}–{hdr.maxs[2]:,.1f}")

    if "gps_time" in pts.point_format.dimension_names:
        lo, hi = flight_dates(pts.gps_time,
                              getattr(hdr.global_encoding, "gps_time_type", "STANDARD"))
        if lo:
            out(f"  FLOWN {lo:%Y-%m-%d %H:%M} .. {hi:%Y-%m-%d %H:%M} UTC "
                f"(not the creation date above)")

    out("\nMETADATA RECORDS")
    for v in hdr.vlrs:
        out(f"  {v.user_id!r} record {v.record_id}: {v.description!r}")
        if hasattr(v, "parse_crs"):
            try:
                out(f"      -> {v.parse_crs().to_string()}")
            except Exception:
                pass

    out(f"\nEVERY FIELD{note}")
    for d in pts.point_format.dimensions:
        a = np.asarray(pts[d.name])
        uniq = np.unique(a)
        empty = "   <- present but ALL ZERO" if len(uniq) == 1 and uniq[0] == 0 else ""
        rng = f"{a.min()} .. {a.max()}" if a.size else "—"
        out(f"  {d.name:22s} {str(a.dtype):9s} {rng:28s} {MEANING.get(d.name, '')}{empty}")

    rows, sep = census(pts.classification)
    out("\nCLASSIFICATION")
    for code, name, n, share in sorted(rows, key=lambda r: -r[2]):
        out(f"  {code:3d} {name:20s} {n:>12,}  {share:6.2%}")
    out(f"  -> canopy {'CAN' if sep else 'CANNOT'} be separated from roof")

    by_return = hdr.number_of_points_by_return
    if any(by_return):
        out("\nRETURNS PER PULSE (whole file, from the header)")
        top = max(by_return)
        for i, k in enumerate(by_return, 1):
            if k:
                out(f"  {i:2d}  {k:>12,}  {k/sum(by_return):6.2%}  {'#' * int(40*k/top)}")
    return hdr
