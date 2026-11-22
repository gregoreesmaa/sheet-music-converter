"""Marker-track extraction for hybrid staff-line geometry (Phase 2, Task T2.4).

Pipeline context: clean system strips carry exact RowNet staff lines;
their degraded twins are warped by ``photo-degrade`` (perspective,
curl, rotation, lens — recorded per file in ``*_params.json`` with a
reproducible ``file_seed``). To transfer the clean lines onto the
degraded geometry EXACTLY, ``build_hybrid_cache.py`` draws small black
marker dots on the clean staff lines and re-runs the ``photo-degrade``
binary with the recorded ``file_seed`` (byte-identical warp —
verified 5/5 byte-equal on markerless re-runs, 2026-11-08; markers do
not perturb the warp: all geometric params fields identical with and
without markers). This module turns the marker-only difference image
(``|marked - plain|`` PNG re-degrades) into per-line tracks:

1. ``find_blobs``: connected components of the ``> core`` difference
   mask filtered by area (marker cores, not stains/tears).
2. ``hough_lines``: global Hough voting over slopes ``±0.12`` finds
   the staff-line combs (the diff holds almost purely marker blobs,
   so lyrics/extenders cannot outvote lines), then a quadratic refit
   through ``±3 px`` inliers captures keystone/tilt/slow ripple.
3. ``cluster_groups`` + ``_fill_staff``: Hough lines are clustered
   into ``S`` staves at the largest y-gaps; each staff is anchored on
   its best pattern-fitting subset (artifact lines fit no uniform
   grid) and missing slots are sibling-interpolated.
4. ``track_curves`` + ``local_contrast``: interpolated dense curves and
   a glare-robust validator (line darkness vs neighboring inter-line
   paper, cancelling illumination gradients).

All deterministic (sorted processing, no RNG). Reference:
``todo/phase-2-hybrid-dewarp.md``, Task T2.4.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = [
    'find_blobs',
    'hough_lines',
    'track_curves',
    'local_contrast',
    'usage_frac',
    'extract_curves',
]

CORE_MIN_AREA = 2
CORE_MAX_AREA = 400
MIN_TRACK_PTS = 5
MIN_TRACK_SPAN_FRAC = 0.35


def find_blobs(diff: np.ndarray, core: float = 30.0) -> list[tuple[float,
                                                              float]]:
    """Marker-bloc centroids ``[(x, y)]`` from an abs-difference image.

    Connected components of ``diff > core`` with area in
    ``[CORE_MIN_AREA, CORE_MAX_AREA]`` (marker cores; tears/stains are
    far larger, JPEG speckle far smaller). Sorted by ``(x, y)``.
    """
    d = np.asarray(diff)
    lab, n = ndimage.label(d > core)
    if n == 0:
        return []
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    cents = ndimage.center_of_mass(d > core, lab, range(1, n + 1))
    pts = [(float(c[1]), float(c[0])) for i, c in enumerate(cents)
           if CORE_MIN_AREA <= sizes[i] <= CORE_MAX_AREA]
    pts.sort()
    return pts


def hough_lines(blobs: list[tuple[float, float]], sp: float,
                w_in: int, h_in: int) -> list[list[tuple[float, float]]]:
    """Staff-line blob groups by global Hough voting + quadratic refit.

    The difference image holds (almost) purely marker blobs — lyrics,
    extenders and slurs carry no markers — so staff lines are simply
    the strongest ``y = m * x + b`` combs. Accumulates blob votes over
    slopes ``±0.12`` (rotation is ``<= 5 deg``) at 0.002 steps,
    greedily takes peaks (suppressing ``±max(3, 0.35 * sp)`` in ``b``
    so adjacent lines resolve), refits each line as a quadratic
    through ``±3 px`` inliers (capturing keystone/tilt/slow ripple),
    re-collects inliers once, and returns the inlier point sets
    (``>= 6`` points each) sorted by intercept. Deterministic.
    """
    if not blobs:
        return []
    ms = np.arange(-0.12, 0.12001, 0.002)
    b_lo, b_hi = -200.0, float(h_in) + 200.0
    nb = int(b_hi - b_lo) + 1
    acc = np.zeros((len(ms), nb), dtype=np.int32)
    xs = np.array([p[0] for p in blobs])
    ys = np.array([p[1] for p in blobs])
    for j, m in enumerate(ms):
        bb = np.clip((ys - m * xs - b_lo).astype(int), 0, nb - 1)
        np.add.at(acc[j], bb, 1)
    taken: list[tuple[float, float, list[int]]] = []
    suppress_b = max(3.0, 0.35 * sp)
    for _ in range(25):
        j, i = np.unravel_index(int(np.argmax(acc)), acc.shape)
        peak = int(acc[j, i])
        if peak < 6:
            break
        m = float(ms[j])
        b = b_lo + float(i)
        dist = np.abs(ys - (m * xs + b))
        idx = [k for k in range(len(blobs)) if dist[k] <= 3.0]
        # Quadratic refit through inliers, then one re-collection.
        if len(idx) >= 6:
            qx = xs[idx]
            qy = ys[idx]
            A = np.stack([qx * qx, qx, np.ones_like(qx)], axis=1)
            coef, *_ = np.linalg.lstsq(A, qy, rcond=None)
            pred = coef[0] * xs * xs + coef[1] * xs + coef[2]
            dist2 = np.abs(ys - pred)
            idx = [k for k in range(len(blobs)) if dist2[k] <= 3.0]
        if len(idx) >= 6:
            taken.append((m, b, idx))
        jj0, jj1 = max(0, j - 5), min(len(ms), j + 6)
        ii0, ii1 = max(0, int(i - suppress_b)), min(nb, int(i + suppress_b) + 1)
        acc[jj0:jj1, ii0:ii1] = 0
    groups = []
    for _, _, idx in sorted(taken, key=lambda t: t[1]):
        pts = sorted((float(xs[k]), float(ys[k])) for k in idx)
        groups.append(pts)
    return groups


def _med_y(t):
    return float(np.median([p[1] for p in t]))


def cluster_groups(long, n_staves):
    """Split long tracks into ``S`` staff groups at the largest y-gaps.

    Inter-staff gaps dwarf lyric gaps, so the ``S - 1`` largest gaps
    between consecutive (y-sorted) tracks are the staff boundaries.
    Raises ``ValueError`` on fewer than ``S`` tracks.
    """
    if len(long) < n_staves:
        raise ValueError(f'only {len(long)} long tracks, need {n_staves} '
                         f'staff groups')
    ordered = sorted(long, key=_med_y)
    gaps = [(_med_y(ordered[i + 1]) - _med_y(ordered[i]), i)
            for i in range(len(ordered) - 1)]
    cuts = sorted(i for _, i in sorted(gaps, reverse=True)[:n_staves - 1])
    groups, start = [], 0
    for c in cuts + [len(ordered) - 1]:
        groups.append(ordered[start:c + 1])
        start = c + 1
    return groups


#: Pattern-fit residual gate (px): anchor subsets must sit on the
#: uniform ``off + k * sp`` grid this tightly, else the group is
#: refit with fewer anchors (zigzag/crosslinked tracks excluded) and
#: the rest interpolated. Contrast validation downstream is the final
#: guard; this gate only decides how many anchors to trust.
FILL_RESID_GATE = 2.0


def _fill_staff(group, sp, w, shorts=()):
    """Five slot-ordered curves; short fragments anchor, longs shape.

    ``group`` holds the staff's LONG Hough lines (curve sources);
    ``shorts`` holds short fragments near the staff (position anchors
    only — too gappy to shape curves). Anchors are assigned to line
    slots by best pattern fit (median residual to ``off + k * sp``
    over subsets of anchors and subsets of slots); the largest anchor
    set fitting within ``FILL_RESID_GATE`` wins. Curves come from
    LONG-anchored slots; short-anchored and empty slots are
    interpolated between validated neighbor curves (extrapolated at
    ends, flat-offset from a lone survivor). Crosslinked/zigzag tracks
    fit no grid tightly and are excluded. Returns ``(curves5, resid,
    n_anchored)`` with ``n_anchored`` counting LONG-anchored slots;
    raises ``ValueError`` when even a pair fits nothing.
    """
    from itertools import combinations
    longs = sorted(group, key=_med_y)
    items = longs + [t for t in sorted(shorts, key=_med_y)]
    is_long = [True] * len(longs) + [False] * (len(items) - len(longs))
    meds = [_med_y(t) for t in items]
    n = len(items)
    best = None
    for m in range(min(5, n), 1, -1):
        for tcombo in combinations(range(n), m):
            for scombo in combinations(range(5), m):
                cand = [meds[i] for i in tcombo]
                slots = list(scombo)
                off = float(np.median(
                    [y - k * sp for y, k in zip(cand, slots)]))
                resid = sum(abs(y - (off + k * sp))
                            for y, k in zip(cand, slots))
                if resid > FILL_RESID_GATE * m:
                    continue
                totlen = sum(len(items[i]) for i in tcombo)
                key = (-m, resid, -totlen)
                if best is None or key < best[0]:
                    best = (key, list(tcombo), slots)
        if best is not None and best[0][0] == -m:
            break
    if best is None:
        raise ValueError('no anchor subset fits the line grid')
    (_, resid, _), tcombo, slots = best
    slot_curve = {}
    for ii, k in zip(tcombo, slots):
        if is_long[ii]:
            slot_curve[k] = track_curves([items[ii]], w)[0]
    n_anchored = len(slot_curve)
    if not slot_curve:
        raise ValueError('anchors are all short fragments; no curve source')
    present = sorted(slot_curve)
    curves = []
    for k in range(5):
        if k in slot_curve:
            curves.append(slot_curve[k])
            continue
        lo = [p for p in present if p < k]
        hi = [p for p in present if p > k]
        if lo and hi:
            a, b = lo[-1], hi[0]
            wa = (b - k) / (b - a)
            curves.append(wa * slot_curve[a] + (1 - wa) * slot_curve[b])
        elif lo:
            a = lo[-1]
            if len(lo) >= 2:
                a2 = lo[-2]
                step = (slot_curve[a] - slot_curve[a2]) / (a - a2)
                curves.append(slot_curve[a] + step * (k - a))
            else:
                curves.append(slot_curve[a] + (k - a) * sp)
        else:
            b = hi[0]
            if len(hi) >= 2:
                b2 = hi[1]
                step = (slot_curve[b2] - slot_curve[b]) / (b2 - b)
                curves.append(slot_curve[b] + step * (k - b))
            else:
                curves.append(slot_curve[b] + (k - b) * sp)
    return curves, resid, n_anchored


def MIN_TRACK_SPX(w_in):
    return MIN_TRACK_SPAN_FRAC * w_in


def track_curves(tracks: list[list[tuple[float, float]]],
                 w_in: int) -> list[np.ndarray]:
    """Dense ``(W_in,)`` curves by linear interpolation (ends clamped)."""
    grid = np.arange(w_in, dtype=np.float64)
    out = []
    for t in tracks:
        xs = np.array([p[0] for p in t], dtype=np.float64)
        ys = np.array([p[1] for p in t], dtype=np.float64)
        o = np.argsort(xs)
        xs, ys = xs[o], ys[o]
        out.append(np.interp(grid, xs, ys,
                             left=float(ys[0]),
                             right=float(ys[-1])).astype(np.float64))
    return out


def usage_frac(blobs: list[tuple[float, float]],
               curves: list[np.ndarray], tol: float = 3.0) -> float:
    """Fraction of blobs within ``tol`` px of any curve.

    Markers are transferred truth: a correct assignment leaves almost
    every marker blob near its line (artifact speckle is the only
    residue), while a misassigned curve abandons its line's blobs.
    Unlike per-column support (which merely measures marker density),
    usage is density-invariant.
    """
    if not blobs:
        return 0.0
    w_in = max(int(max(c.shape[0] for c in curves)), 1)
    used = 0
    for x, y in blobs:
        xi = min(w_in - 1, max(0, int(round(x))))
        if any(abs(float(c[xi]) - y) <= tol for c in curves):
            used += 1
    return used / len(blobs)


def local_contrast(gray: np.ndarray, curve: np.ndarray,
                   sp: float) -> tuple[float, float]:
    """Line darkness vs neighboring inter-line paper + surround level.

    Compares the ``±1``-row band on the curve against paper bands
    ``±1.5 * sp`` rows away (same columns): positive contrast means the
    curve sits on darker ink than its local surround. Returns
    ``(contrast, surround)`` where surround is the median paper level:
    a negative contrast under a BRIGHT surround is a glare washout
    (correct curve, bleached optics), while negative contrast under
    NORMAL surround is an anti-line (misplaced onto paper).
    """
    g = np.asarray(gray, dtype=np.float32)
    h, w = g.shape
    grid = np.arange(w)
    rows = np.clip(curve.astype(int), 0, h - 1)
    line = (g[np.maximum(0, rows - 1), grid] + g[rows, grid]
            + g[np.minimum(h - 1, rows + 1), grid]) / 3.0
    off = max(3, int(round(1.5 * sp)))
    up = g[np.clip(rows - off, 0, h - 1), grid]
    dn = g[np.clip(rows + off, 0, h - 1), grid]
    paper = (up + dn) / 2.0
    return float(np.median(paper - line)), float(np.median(paper))


#: Blob-core thresholds tried highest-first (deterministic): faint
#: marker cores need a low floor, but the lowest floor admits artifact
#: speckle, so the first (highest) core passing all validations wins.
CORE_TRIES = (25.0, 18.0, 12.0)


def extract_curves(diff: np.ndarray, gray_real: np.ndarray, n_staves: int,
                   sp: float, core: float = 30.0) -> tuple[
                       list[np.ndarray], dict]:
    """Blobs -> Hough lines -> staff curves + validation.

    Raises ``ValueError`` when grouping, pattern fit, or per-line
    validation (blob support + smoothness + hard contrast floor) fail.
    Returns ``(curves, info)`` with curves in top-to-bottom staff-line
    order and ``info`` carrying blob/Hough counts, per-line contrasts,
    supports, and worst smoothness.
    """
    g = np.asarray(gray_real)
    h, w = g.shape[-2], g.shape[-1]
    cores = [core] if core != 30.0 else list(CORE_TRIES)
    last_err: ValueError | None = None
    for co in cores:
        try:
            return _extract_at_core(diff, g, n_staves, sp, co)
        except ValueError as e:
            last_err = e
            continue
    raise last_err if last_err is not None else ValueError('no cores')


def _long_tracks(tracks, w):
    return [t for t in tracks
            if len(t) >= MIN_TRACK_PTS
            and t[-1][0] - t[0][0] >= MIN_TRACK_SPX(w)]


#: Validation gates: a correct assignment uses almost every marker
#: blob (artifact speckle excepted) and draws smooth warp-following
#: curves. Optical contrast is advisory except for anti-lines: below
#: -10 under NORMAL surround paper the curve sits on blank paper
#: (misplaced); the same reading under a glare-bleached surround
#: (paper 25+ over the strip median) is a washout of a correct curve.
MIN_USAGE = 0.60
MAX_SMOOTH = 2.0
MIN_CONTRAST_HARD = -10.0
GLARE_PAPER_LIFT = 25.0


def _finalize(g, curves, sp, blobs, info):
    """Usage + smoothness + glare-aware contrast validation, y-order."""
    paper_med = float(np.median(np.asarray(g, dtype=np.float32)))
    cl = [local_contrast(g, c, sp) for c in curves]
    contrasts = [c for c, _ in cl]
    smooth = []
    for c in curves:
        d2 = np.abs(c[2:] - 2 * c[1:-1] + c[:-2])
        smooth.append(float(d2.max()) if len(d2) else 0.0)
    usage = usage_frac(blobs, curves)
    info.update({'contrasts': [round(x, 1) for x in contrasts],
                 'usage': round(usage, 3),
                 'smooth_max': round(max(smooth), 2) if smooth else 0.0,
                 'min_contrast': round(min(contrasts), 1)
                 if contrasts else 0.0})
    bad = [i for i, ((co, su), sm) in enumerate(zip(cl, smooth))
           if sm > MAX_SMOOTH
           or (co < MIN_CONTRAST_HARD
               and su < paper_med + GLARE_PAPER_LIFT)]
    if usage < MIN_USAGE:
        raise ValueError('blob usage %.3f below floor %.2f (curves %d)' %
                         (usage, MIN_USAGE, len(curves)))
    if bad:
        raise ValueError('line validation failed at %s (contrast %s)' %
                         (bad, info['contrasts']))
    order = np.argsort([c.mean() for c in curves])
    return [curves[i] for i in order], info


def _extract_at_core(diff, g, n_staves, sp, core):
    h, w = g.shape[-2], g.shape[-1]
    blobs = find_blobs(diff, core=core)
    hough = hough_lines(blobs, sp, w, h)
    base = {'n_blobs': len(blobs), 'n_hough': len(hough), 'core': core}
    # Cluster Hough line groups into S staff groups, anchor each group
    # on its best pattern-fitting subset (artifact lines fit no uniform
    # grid and are excluded), fill the remaining slots by sibling
    # interpolation, validate everything by local contrast.
    long = _long_tracks(hough, w)
    shorts = [t for t in hough
              if len(t) >= 3 and t not in long]
    groups = cluster_groups(long, n_staves)
    curves, resids, anchored = [], [], 0
    for grp in groups:
        gymin = min(_med_y(t) for t in grp)
        gymax = max(_med_y(t) for t in grp)
        near = [t for t in shorts
                if gymin - 2 * sp <= _med_y(t) <= gymax + 2 * sp]
        c5, r, na = _fill_staff(grp, sp, w, shorts=near)
        curves.extend(c5)
        resids.append(r)
        anchored += na
    # Hallucination cap: at most 1 in 5 slots may be interpolated.
    # (A wrong staff count, e.g. S=3 on two staves, would interpolate
    # a third of all slots from paper: structurally rejected here.)
    need = n_staves * 5
    if anchored < need - need // 5:
        raise ValueError('only %d of %d slots anchored' % (anchored, need))
    info = dict(base, anchored=anchored,
                fill_resid=round(max(resids), 2) if resids else 0.0)
    try:
        return _finalize(g, curves, sp, blobs, info)
    except ValueError:
        pass
    # Sibling rescue: lines failing validation inside an otherwise
    # strong staff (>=3 optically-good siblings) are re-interpolated
    # from those siblings (discarding the junk-anchored curve) and
    # re-validated. Recovers single-line misassignments; anything
    # needing more is structurally skipped.
    return _sibling_rescue(g, curves, sp, blobs, n_staves, info)


def _line_ok(g, curve, sp, paper_med):
    (co, su), d2 = local_contrast(g, curve, sp), np.abs(
        curve[2:] - 2 * curve[1:-1] + curve[:-2])
    sm = float(d2.max()) if len(d2) else 0.0
    return (sm <= MAX_SMOOTH
            and not (co < MIN_CONTRAST_HARD
                     and su < paper_med + GLARE_PAPER_LIFT))


def _sibling_rescue(g, curves, sp, blobs, n_staves, info):
    gg = np.asarray(g, dtype=np.float32)
    paper_med = float(np.median(gg))
    curves = list(curves)
    rescued = 0
    for s in range(n_staves):
        seg = curves[s * 5:(s + 1) * 5]
        good = [i for i, c in enumerate(seg) if _line_ok(g, c, sp, paper_med)]
        bad = [i for i in range(5) if i not in good]
        if not bad or len(good) < 3:
            continue
        for k in bad:
            lo = [i for i in good if i < k]
            hi = [i for i in good if i > k]
            if lo and hi:
                a, b = lo[-1], hi[0]
                wa = (b - k) / (b - a)
                curves[s * 5 + k] = wa * seg[a] + (1 - wa) * seg[b]
            elif lo and len(lo) >= 2:
                a, a2 = lo[-1], lo[-2]
                step = (seg[a] - seg[a2]) / (a - a2)
                curves[s * 5 + k] = seg[a] + step * (k - a)
            elif hi and len(hi) >= 2:
                b, b2 = hi[0], hi[1]
                step = (seg[b2] - seg[b]) / (b2 - b)
                curves[s * 5 + k] = seg[b] + step * (k - b)
            elif lo:
                curves[s * 5 + k] = seg[lo[-1]] + (k - lo[-1]) * sp
            elif hi:
                curves[s * 5 + k] = seg[hi[0]] + (k - hi[0]) * sp
            rescued += 1
    info = dict(info, rescued=rescued)
    if not rescued:
        raise ValueError('no rescuable lines')
    return _finalize(g, curves, sp, blobs, info)
