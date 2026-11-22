"""Blind windowed staff-line curve detector for system strips (Phase 2, Task T2.4).

Complements the RowNet neural detector (``build_sys_data.neural_detect_staves``,
exact on clean strips but blind on phone-photo degraded strips: RowNet v2
outputs ``max prob ~0.003`` on degraded strips, measured 2026-11-08) and the
``layout-staff`` classical global-profile detector (rotation smear up to
``5 deg`` across ``~1850 px`` strips fragments its grouping: only ``14/100``
degraded strips yield exactly ``S`` complete staves, measured 2026-11-08).

Method (pure stdlib + numpy + Pillow, deterministic — no RNG, no weights):
flat-field illumination normalization -> binary ink mask (same ``140``
threshold as ``layout-staff``) -> overlapping column windows (``192 px``,
step ``96 px``; within a window even a ``5 deg`` tilt drifts only ``17 px``
and perspective/curl waves are near-linear) -> per-window row ink profiles
-> local-max peak rows -> per-window template fit of ``S`` staves x ``5``
lines at staff space ``sp`` (origins anchored on observed peaks, staves
separated by ``>= 4*sp + 3`` so lyric lines fall in the gaps unmatched)
-> cross-window outlier rejection + interpolation to dense per-column
curves consumable by ``rectify.rectify_system_strip``.

Reference: ``todo/phase-2-hybrid-dewarp.md``, Task T2.4.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

__all__ = [
    'flat_ink_mask',
    'window_peaks',
    'estimate_sp',
    'sp_candidates',
    'fit_window_origins',
    'detect_strip_curves',
    'curves_to_table',
    'table_to_curves',
]

INK_THRESHOLD = 140
FLAT_RADIUS = 30
FLAT_TARGET_MEAN = 235.0
WIN = 96
STEP = 48
CENTROID_HALF = 6
PEAK_FLOOR = 0.05
PEAK_MERGE_PX = 3
MATCH_TOL_PX = 2.0
MIN_STAFF_GAP_PX = 3.0  # added to 4*sp for the inter-staff minimum gap
SP_MIN, SP_MAX = 4.0, 40.0
COVERAGE_FLOOR = 0.6


def flat_ink_mask(gray: Image.Image) -> np.ndarray:
    """Binary ink mask ``(H, W)`` float32 after flat-field normalization.

    Divides by a heavily blurred background estimate (``GaussianBlur``
    radius 30) rescaled to paper-white 235, then thresholds at 140 —
    the same ink threshold ``layout-staff`` and ``build_sys_data`` use.
    Removes lighting gradients/vignette while keeping staff strokes.
    """
    if not isinstance(gray, Image.Image):
        raise TypeError(f'gray must be a PIL Image, got {type(gray).__name__}')
    g = gray.convert('L')
    a = np.asarray(g, dtype=np.float32)
    bg = np.asarray(g.filter(ImageFilter.GaussianBlur(radius=FLAT_RADIUS)),
                    dtype=np.float32)
    flat = np.clip(a / np.maximum(bg, 1.0) * FLAT_TARGET_MEAN, 0, 255)
    return (flat < INK_THRESHOLD).astype(np.float32)


def window_peaks(ink: np.ndarray, win: int = WIN, step: int = STEP,
                 floor: float = PEAK_FLOOR,
                 merge_px: int = PEAK_MERGE_PX) -> list[tuple[int, list[float]]]:
    """``[(x_center, [peak rows])]`` per overlapping column window.

    Peaks are strict local maxima (``±2`` neighbourhood) of the window
    row ink-fill profile above ``floor``, merged within ``merge_px``
    keeping the stronger row, then refined to the profile centroid over
    ``±CENTROID_HALF`` rows (unbiased under within-window tilt: a
    sloped line's row histogram peaks off-center but its centroid sits
    at the window-center value). Deterministic.
    """
    h, w = ink.shape
    out = []
    x0 = 0
    while x0 < w:
        x1 = min(x0 + win, w)
        prof = ink[:, x0:x1].mean(axis=1)
        raw = [r for r in range(2, h - 2)
               if prof[r] > floor and prof[r] >= prof[r - 1]
               and prof[r] > prof[r + 1]
               and prof[r] >= prof[r - 2] and prof[r] >= prof[r + 2]]
        merged: list[int] = []
        for p in raw:
            if merged and p - merged[-1] <= merge_px:
                if prof[p] > prof[merged[-1]]:
                    merged[-1] = p
            else:
                merged.append(p)
        refined = []
        for p in merged:
            lo, hi = max(0, p - CENTROID_HALF), min(h, p + CENTROID_HALF + 1)
            band = prof[lo:hi]
            tot = float(band.sum())
            if tot <= 0:
                refined.append(float(p))
            else:
                idx = np.arange(lo, hi, dtype=np.float64)
                refined.append(round(float((idx * band).sum() / tot), 2))
        out.append(((x0 + x1) // 2, refined))
        if x1 >= w:
            break
        x0 += step
    return out


def sp_candidates(windows: list[tuple[int, list[float]]],
                  k: int = 3) -> list[tuple[float, int]]:
    """Top-``k`` staff-space candidates from pooled adjacent-peak gaps.

    Histograms rounded integer gaps in ``[4, 40]`` px (smoothed ``±1``);
    returns ``[(sp, count)]`` sorted by count desc (ties: smaller sp
    first — a doubled ``2*sp`` harmonic must not shadow the true ``sp``).
    Blurred/ringed strips often peak at a line-split harmonic, so
    callers try several candidates and keep the best-coverage fit.
    Raises ``ValueError`` when no usable gap exists.
    """
    hist: dict[int, int] = {}
    for _, peaks in windows:
        for i in range(len(peaks) - 1):
            g = int(round(peaks[i + 1] - peaks[i]))
            if 4 <= g <= 40:
                hist[g] = hist.get(g, 0) + 1
    if not hist:
        raise ValueError('no peak gaps for staff-space estimate')
    smooth = {g: hist.get(g - 1, 0) + hist.get(g, 0) + hist.get(g + 1, 0)
              for g in hist}
    ranked = sorted(smooth, key=lambda g: (-smooth[g], g))
    out: list[tuple[float, int]] = []
    for g in ranked:
        if all(abs(g - s) >= 2 for s, _ in out):
            out.append((float(g), smooth[g]))
        if len(out) >= k:
            break
    out = [(s, c) for s, c in out if SP_MIN <= s <= SP_MAX]
    if not out:
        raise ValueError('staff-space candidates out of range')
    return out


def estimate_sp(windows: list[tuple[int, list[float]]]) -> float:
    """Top-1 staff space (see :func:`sp_candidates`)."""
    return sp_candidates(windows, k=1)[0][0]


def _slot_rows(peaks: list[float], origins: list[float | None],
               sp: float) -> tuple[list[list[float | None]], set[int]]:
    """Match peaks to template slots (each peak used at most once).

    Returns per-staff lists of 5 rows-or-``None`` (unmatched slots and
    unplaced (``None``-origin) staves stay ``None``) plus the set of
    used peak indices. Greedy top-to-bottom, slots in order;
    deterministic.
    """
    used: set[int] = set()
    out: list[list[float | None]] = []
    for y0 in origins:
        if y0 is None:
            out.append([None] * 5)
            continue
        staff: list[float | None] = []
        for k in range(5):
            want = y0 + k * sp
            bi, bd = -1, MATCH_TOL_PX + 1e-9
            for i, p in enumerate(peaks):
                if i in used:
                    continue
                d = abs(p - want)
                if d < bd:
                    bd, bi = d, i
            if bi >= 0:
                used.add(bi)
                staff.append(float(peaks[bi]))
            else:
                staff.append(None)
        out.append(staff)
    return out, used


def _match_count(peaks: list[int], origins: list[float],
                 sp: float) -> tuple[int, list[int]]:
    """Count peaks matched to the ``S x 5`` template (each peak used once)."""
    rows, used = _slot_rows(peaks, origins, sp)
    assign = [-1] * len(peaks)
    for s, staff in enumerate(rows):
        for k, r in enumerate(staff):
            if r is not None:
                for i, p in enumerate(peaks):
                    if float(p) == r and assign[i] < 0:
                        assign[i] = s * 5 + k
                        break
    return len(used), assign


def fit_window_origins(peaks: list[float], n_staves: int,
                       sp: float) -> tuple[list[float | None], int]:
    """Best ``S`` staff-top origins for one window's peak rows.

    Candidate origins are peak-anchored (``p - k*sp`` for ``k`` in 0..4,
    snapped to ``0.1 px``); staves are placed top-to-bottom with a minimum
    gap of ``4*sp + 3`` px so lyric/text lines between staves stay
    unmatched. Maximizes matched peaks; deterministic tie-break (first
    best in sorted-candidate order). Returns ``(origins, matched)`` where
    an unplaceable staff (zero matches past the gap) yields a ``None``
    origin instead of failing the whole window, so notehead-pileup
    windows still contribute their placeable staves. Raises
    ``ValueError`` only when NO staff places (empty or barren window).
    """
    if n_staves < 1 or n_staves > 4:
        raise ValueError(f'n_staves must be 1..4, got {n_staves}')
    if not peaks:
        raise ValueError('no peaks to fit')
    cands = sorted({round(float(p) - k * sp, 1)
                    for p in peaks for k in range(5)})
    min_gap = 4.0 * sp + MIN_STAFF_GAP_PX
    # Sequential greedy top-to-bottom: place each staff at the origin
    # with the most matches on still-unused peaks (staves are >= min_gap
    # apart, so they never compete for the same peak in practice; the
    # joint optimum coincides with the greedy one). Deterministic:
    # candidates ascending, strict > keeps the first (topmost) best.
    remaining = list(peaks)
    origins: list[float | None] = []
    total_hit = 0
    lo = -4.0 * sp
    for _ in range(n_staves):
        best_y0, best_rows, best_hit = None, None, 0
        for y0 in cands:
            if y0 < lo:
                continue
            rows, used_now = _slot_rows(remaining, [y0], sp)
            hit = len(used_now)
            if hit > best_hit:
                best_y0, best_rows, best_hit = y0, rows[0], hit
        if best_y0 is None:
            origins.append(None)
            continue
        origins.append(best_y0)
        total_hit += best_hit
        drop = {float(r) for r in best_rows if r is not None}
        remaining = [p for p in remaining if float(p) not in drop]
        lo = best_y0 + min_gap
    if total_hit == 0:
        raise ValueError(f'no staff places in {len(peaks)} peaks (sp={sp})')
    return origins, total_hit


def _link_curves(fits: list[tuple[int, list[list[float | None]]]],
                 n_staves: int, w_in: int,
                 rej_px: float = 4.0) -> list[list[np.ndarray]]:
    """Link per-window slot rows into dense per-column curves.

    For each ``(staff, line)`` slot, collects ``(x_center, row)`` points
    across windows, rejects points farther than ``rej_px`` from a
    rolling median (width 5, edges fall back to the global median), and
    linearly interpolates to every input column (ends clamped). Returns
    ``S`` staves of 5 ``(W_in,)`` float64 arrays. Raises ``ValueError``
    when a slot keeps fewer than 2 points.
    """
    curves: list[list[np.ndarray]] = []
    for s in range(n_staves):
        staff_curves = []
        for k in range(5):
            pts = [(float(xc), float(rows[s][k]))
                   for xc, rows in fits
                   if rows[s][k] is not None]
            if len(pts) < 2:
                raise ValueError(f'staff {s} line {k}: only '
                                 f'{len(pts)} window points')
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            order = np.argsort(xs)
            xs, ys = xs[order], ys[order]
            # Trend-aware Hampel: tilt makes adjacent windows differ by
            # several px, so a flat median lags real slopes. Detrend by
            # the robust adjacent-slope median, take a rolling median of
            # the detrended values, re-trend the reference.
            if len(ys) >= 2:
                sl = np.median([(ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j])
                                for j in range(len(ys) - 1)
                                if xs[j + 1] > xs[j]])
                sl = float(sl) if np.isfinite(sl) else 0.0
            else:
                sl = 0.0
            det = ys - sl * xs
            med = float(np.median(det))
            keep = np.ones(len(ys), dtype=bool)
            for i in range(len(ys)):
                lo_i, hi_i = max(0, i - 2), min(len(ys), i + 3)
                ref = ((float(np.median(det[lo_i:hi_i])) + sl * xs[i])
                       if hi_i - lo_i >= 3 else med + sl * xs[i])
                if abs(ys[i] - ref) > rej_px:
                    keep[i] = False
            if keep.sum() < 2:
                raise ValueError(f'staff {s} line {k}: rejection left '
                                 f'{int(keep.sum())} points')
            xs, ys = xs[keep], ys[keep]
            grid = np.arange(w_in, dtype=np.float64)
            # Linear extension past the end kept windows (tilted edge
            # staves keep drifting; flat clamping would freeze them):
            # least-squares slope over the last/first up-to-4 kept
            # points (robust to one noisy end window).
            def _slope(xe, ye):
                xe = np.asarray(xe, dtype=np.float64)
                ye = np.asarray(ye, dtype=np.float64)
                xm, ym = xe.mean(), ye.mean()
                den = ((xe - xm) ** 2).sum()
                return float(((xe - xm) * (ye - ym)).sum() / den) \
                    if den > 0 else 0.0
            sl_l = _slope(xs[:4], ys[:4])
            sl_r = _slope(xs[-4:], ys[-4:])
            curve = np.interp(grid, xs, ys)
            left = grid < xs[0]
            right = grid > xs[-1]
            curve[left] = ys[0] + sl_l * (grid[left] - xs[0])
            curve[right] = ys[-1] + sl_r * (grid[right] - xs[-1])
            staff_curves.append(curve)
        curves.append(staff_curves)
    return curves


def _fit_with_sp(wins: list[tuple[int, list[float]]], n_staves: int,
                 sp_est: float, w_in: int) -> tuple[
                     list[list[np.ndarray]], dict]:
    """Anchor + propagate + link at a fixed staff space (one candidate).

    Returns ``(curves, stat)`` with ``stat`` carrying ``matched``,
    ``total``, ``n_fits``. Raises ``ValueError`` when placement,
    linking, or the window-count floor fails (coverage is judged by
    the caller across candidates).
    """
    # Independent per-window fits (label shifts and phantom locks in
    # single windows are removed downstream by the trend-aware Hampel
    # in _link_curves, which keeps only rows consistent with their
    # neighbors; windows placing nothing are skipped and bridged).
    placed_rows: dict[int, list[list[float | None]]] = {}
    for xc, peaks in wins:
        try:
            origins, _ = fit_window_origins(peaks, n_staves, sp_est)
        except ValueError:
            continue
        rows, _ = _slot_rows(peaks, origins, sp_est)
        placed_rows[xc] = rows
    fits: list[tuple[int, list[list[float | None]]]] = []
    matched, total = 0, 0
    for xc in sorted(placed_rows):
        rows = placed_rows[xc]
        # Count matched slots directly from the stored rows: a slot is
        # matched when its row is not None. (Re-matching is unnecessary;
        # peaks are window-local.)
        matched += sum(1 for staff in rows for r in staff
                       if r is not None)
        total += n_staves * 5
        fits.append((xc, rows))
    if len(fits) < 3:
        raise ValueError(f'only {len(fits)} of {len(wins)} windows placed '
                         f'{n_staves} staves')
    curves = _link_curves(fits, n_staves, w_in)
    stat = {'matched': matched, 'total': total, 'n_fits': len(fits)}
    return curves, stat


#: Coverage at which a prior-sp fit is accepted without trying the
#: blind histogram candidates (deterministic early accept).
PRIOR_ACCEPT = 0.8


def detect_strip_curves(gray: Image.Image, n_staves: int,
                        sp: float | None = None,
                        sp_prior: float | None = None) -> tuple[
                            list[list[np.ndarray]], dict]:
    """Detect ``S`` staves x 5 staff-line curves on a system strip.

    Args:
        gray: PIL image (any mode; converted to ``L``).
        n_staves: staff count ``S`` (1..4). For training/sys-dev strips
            this comes from the RowNet detection on the clean sibling
            (same system by construction); it is geometry metadata,
            never label content.
        sp: staff space override (single attempt, no search).
        sp_prior: staff-space hint tried FIRST (clean sibling ``sp``
            scaled by strip-width ratio for degraded strips); accepted
            outright at coverage ``>= PRIOR_ACCEPT``, else the blind
            histogram candidates compete and the best coverage wins.

    Returns ``(curves, info)`` with ``curves[s][k]`` a ``(W_in,)``
    float64 per-column row array and ``info`` carrying ``sp``,
    ``sp_source`` (``given``/``prior``/``hist``), ``coverage``
    (matched slots / total window slots), ``n_windows`` and
    ``matched`` counts. Raises ``ValueError`` when estimation,
    placement, linking, or the coverage floor fails.
    """
    if n_staves < 1 or n_staves > 4:
        raise ValueError(f'n_staves must be 1..4, got {n_staves}')
    g = gray.convert('L')
    w_in, h_in = g.size
    ink = flat_ink_mask(g)
    wins = window_peaks(ink)
    cands: list[tuple[float, int, bool]] = []
    if sp is not None:
        cands = [(float(sp), 0, False)]
    else:
        if sp_prior is not None and SP_MIN <= sp_prior <= SP_MAX:
            cands.append((float(sp_prior), 0, True))
        for s, c in sp_candidates(wins):
            if all(abs(s - q) >= 1.0 for q, _, _ in cands):
                cands.append((s, c, False))
    if not cands:
        raise ValueError('no staff-space candidates')
    best = None
    for sp_est, count, is_prior in cands:
        try:
            curves, stat = _fit_with_sp(wins, n_staves, sp_est, w_in)
        except ValueError:
            continue
        cov = stat['matched'] / max(1, stat['total'])
        key = (cov, is_prior, count)
        if best is None or key > best[0]:
            best = (key, curves, stat, sp_est, is_prior)
        if is_prior and cov >= PRIOR_ACCEPT:
            break
    if best is None:
        raise ValueError('all staff-space candidates failed')
    _, curves, stat, sp_est, is_prior = best
    coverage = stat['matched'] / max(1, stat['total'])
    info = {'sp': sp_est,
            'sp_source': ('prior' if (sp is None and is_prior)
                          else ('given' if sp is not None else 'hist')),
            'coverage': coverage, 'n_windows': len(wins),
            'matched': stat['matched'], 'total': stat['total'],
            'n_fits': stat['n_fits'], 'w_in': w_in, 'h_in': h_in}
    if coverage < COVERAGE_FLOOR:
        raise ValueError(f'coverage {coverage:.2f} below floor '
                         f'{COVERAGE_FLOOR}')
    return curves, info


def curves_to_table(curves: list[list[np.ndarray]],
                    stride: int = 8) -> dict:
    """Compress curves to a JSON-safe control-point table.

    Samples every ``stride``-th column plus the final column per line;
    ``table_to_curves`` restores dense curves by linear interpolation
    (``rectify._as_curve`` resamples identically). A 4-staff strip at
    ``W=1851`` shrinks from ``20 x 1851`` floats to ``20 x 232``.
    """
    table = {'stride': int(stride), 'w_in': int(curves[0][0].shape[0]),
             'staves': []}
    for staff in curves:
        rows = []
        for line in staff:
            line = np.asarray(line, dtype=np.float64)
            idx = list(range(0, line.shape[0], stride))
            if idx[-1] != line.shape[0] - 1:
                idx.append(line.shape[0] - 1)
            rows.append([round(float(line[i]), 2) for i in idx])
        table['staves'].append(rows)
    return table


def table_to_curves(table: dict) -> list[list[np.ndarray]]:
    """Restore dense ``(W_in,)`` curves from :func:`curves_to_table`."""
    w_in = int(table['w_in'])
    grid = np.arange(w_in, dtype=np.float64)
    stride = int(table['stride'])
    src = np.array([i for i in range(0, w_in, stride)
                    ] + ([w_in - 1] if (w_in - 1) % stride else []),
                   dtype=np.float64)
    out = []
    for staff in table['staves']:
        lines = []
        for rows in staff:
            a = np.asarray(rows, dtype=np.float64)
            if a.shape[0] != src.shape[0]:
                raise ValueError('table/w_in shape mismatch: '
                                 f'{a.shape[0]} vs {src.shape[0]}')
            lines.append(np.interp(grid, src, a).astype(np.float64))
        out.append(lines)
    return out
