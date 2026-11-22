"""Hardened multi-scale / adaptive-threshold RowNet photo fallback.

Root causes of ``cannot determine S`` on 28/31 real photos in
``build_hybrid_cache.clean_lines`` / ``build_sys_data.neural_detect_staves``:
  1. Hardcoded global binarization (``gray < 140``) collapses shadowed paper
     to black blobs and washes out faint staff lines in bright regions.
  2. Hardcoded global probability threshold (``probs > 0.3``) drops weak
     peaks on tilted/shadowed/low-contrast photos (where RowNet outputs
     0.08-0.28) and fails the entire page whenever a single line dips below
     threshold (4/5 peaks visible) or a lyric/beam adds a 6th peak.
  3. ``_guess_S`` caps ``1 <= n <= 4``, rejecting 5-7 staff systems.

This module provides a pure-NumPy, deterministic, multi-scale fallback:
  * :func:`adaptive_binarize` — O(H*W) integral-image local mean threshold
    robust to strong illumination gradients and shadows.
  * :func:`smooth_and_detrend_profile` — multi-scale Gaussian smoothing +
    local baseline removal to suppress broad shadow humps.
  * :func:`estimate_staff_space` — harmonic comb autocorrelation over
    candidate staff spaces ``sp in [min_sp, max_sp]``.
  * :func:`detect_staves_from_probs` / :func:`infer_staff_count` — 5-line
    periodic comb template matching + adaptive threshold + greedy NMS +
    per-line local peak refinement + missing-line interpolation, supporting
    ``S = 1..max_staves`` (default 8).
"""

from __future__ import annotations

import numpy as np


def adaptive_binarize(
    gray: np.ndarray,
    block_size: int = 31,
    offset: float = 12.0,
) -> np.ndarray:
    """Local mean-threshold binarization (1.0 = ink, 0.0 = paper) in O(H*W).

    Uses a summed-area table (integral image) so shadowed paper with
    strong illumination gradients still separates dark staff lines from
    local background without turning shadow regions solid black.
    """
    a = np.asarray(gray, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f'expected 2-D grayscale image, got shape {a.shape}')
    h, w = a.shape
    r = max(1, int(block_size) // 2)
    pad = np.pad(a, ((r, r), (r, r)), mode='reflect')
    ii = np.zeros((pad.shape[0] + 1, pad.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(pad.astype(np.float64), axis=0), axis=1)
    k = 2 * r + 1
    win_sum = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    local_mean = (win_sum / float(k * k)).astype(np.float32)
    return (a < (local_mean - float(offset))).astype(np.float32)


def _gauss_1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return arr.astype(np.float64, copy=True)
    rad = max(1, int(round(3.0 * sigma)))
    x = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(arr.astype(np.float64), (rad, rad), mode='edge')
    return np.convolve(padded, k, mode='valid')


def smooth_and_detrend_profile(
    probs: np.ndarray,
    sigma: float = 1.0,
    bg_window: int = 41,
) -> np.ndarray:
    """Smooth split peaks and subtract slow shadow baseline from 1-D probs."""
    p = np.asarray(probs, dtype=np.float64).ravel()
    if p.size == 0:
        return p.copy()
    sm = _gauss_1d(p, sigma) if sigma > 0 else p.copy()
    win = max(5, int(bg_window) | 1)
    rad = win // 2
    padded = np.pad(sm, (rad, rad), mode='edge')
    cs = np.concatenate(([0.0], np.cumsum(padded)))
    bg = (cs[win:] - cs[:-win]) / float(win)
    return np.maximum(sm - 0.85 * bg, 0.0)


def estimate_staff_space(
    probs: np.ndarray,
    min_sp: float = 4.0,
    max_sp: float = 40.0,
) -> tuple[float, float]:
    """Estimate dominant staff-line spacing ``sp`` via harmonic comb autocorrelation.

    Returns ``(best_sp, snr)``. Evaluates lag-1..lag-4 autocorrelation minus
    half-step (lag-0.5, 1.5) anti-correlation so 5-line staves stand out
    sharply even under heavy noise or missing individual lines.
    """
    p = smooth_and_detrend_profile(probs, sigma=0.8, bg_window=41)
    n = p.size
    if n < int(4 * min_sp) + 4:
        raise ValueError(f'profile too short ({n} rows) for min_sp={min_sp}')
    hi = min(float(max_sp), (n - 4) / 4.2)
    if hi < min_sp:
        raise ValueError(f'profile too short ({n} rows) for min_sp={min_sp}')

    # Fast FFT autocorrelation of zero-mean positive signal.
    sig = p - p.mean()
    energy = float(np.dot(sig, sig))
    if energy <= 1e-12:
        return float(min_sp), 0.0
    m = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(sig, n=m)
    ac = np.fft.irfft(f * np.conj(f), n=m)[:n] / energy

    best_sp = float(min_sp)
    best_score = -1e9
    scores = []
    # 0.5-px grid for sub-pixel spacing accuracy.
    candidates = np.arange(min_sp, hi + 1e-6, 0.5)
    for sp in candidates:
        pos = 0.0
        w_pos = 0.0
        for mult, wt in ((1.0, 1.0), (2.0, 0.85), (3.0, 0.65), (4.0, 0.45)):
            lag = mult * sp
            if lag < n - 1:
                i0 = int(lag)
                frac = lag - i0
                val = ac[i0] * (1.0 - frac) + ac[i0 + 1] * frac
                pos += wt * val
                w_pos += wt
        neg = 0.0
        w_neg = 0.0
        for mult, wt in ((0.5, 0.6), (1.5, 0.4), (2.5, 0.25)):
            lag = mult * sp
            if lag < n - 1:
                i0 = int(lag)
                frac = lag - i0
                val = ac[i0] * (1.0 - frac) + ac[i0 + 1] * frac
                neg += wt * val
                w_neg += wt
        score = (pos / max(w_pos, 1e-6)) - 0.5 * (neg / max(w_neg, 1e-6))
        scores.append(score)
        if score > best_score:
            best_score = score
            best_sp = float(sp)

    scores_arr = np.asarray(scores, dtype=np.float64)
    med = float(np.median(scores_arr))
    mad = float(np.median(np.abs(scores_arr - med))) + 1e-6
    snr = (best_score - med) / (1.4826 * mad)
    return best_sp, float(snr)


def _comb_response(profile: np.ndarray, sp: float) -> np.ndarray:
    """5-line periodic comb score at every candidate top-line row ``y0``."""
    n = profile.size
    span = int(np.ceil(4.0 * sp))
    if n <= span + 1:
        return np.zeros(0, dtype=np.float64)
    out_len = n - span
    y0 = np.arange(out_len, dtype=np.float64)
    idx = np.arange(n, dtype=np.float64)
    pos = np.zeros(out_len, dtype=np.float64)
    for k in range(5):
        pos += np.interp(y0 + k * sp, idx, profile)
    pos /= 5.0
    neg = np.zeros(out_len, dtype=np.float64)
    for k in range(4):
        neg += np.interp(y0 + (k + 0.5) * sp, idx, profile)
    neg /= 4.0
    return pos - 0.75 * neg


def _refine_five_lines(
    raw_probs: np.ndarray,
    detrended: np.ndarray,
    y0: float,
    sp: float,
) -> tuple[list[float], int]:
    """Refine each of the 5 lines locally around ``y0 + k*sp``; interpolate weak ones."""
    n = raw_probs.size
    search_r = max(1, int(round(0.35 * sp)))
    lines = []
    found_mask = []
    # Local floor relative to this staff's own peak so low-contrast
    # shadowed staves still refine accurately.
    band_lo = max(0, int(round(y0 - sp)))
    band_hi = min(n, int(round(y0 + 5.0 * sp)))
    band_peak = float(detrended[band_lo:band_hi].max()) if band_hi > band_lo else 0.0
    line_floor = max(0.02, 0.25 * band_peak)

    for k in range(5):
        pred = y0 + k * sp
        c = int(round(pred))
        lo = max(0, c - search_r)
        hi = min(n, c + search_r + 1)
        if hi <= lo:
            lines.append(float(np.clip(pred, 0.0, float(n - 1))))
            found_mask.append(False)
            continue
        seg = detrended[lo:hi]
        rel = int(np.argmax(seg))
        if float(seg[rel]) >= line_floor:
            best = lo + rel
            # Parabolic sub-pixel peak refinement when interior.
            if 0 < best < n - 1:
                y_m, y_c, y_p = detrended[best - 1], detrended[best], detrended[best + 1]
                denom = y_m - 2.0 * y_c + y_p
                if abs(denom) > 1e-9:
                    delta = np.clip(0.5 * (y_m - y_p) / denom, -0.5, 0.5)
                    lines.append(float(best + delta))
                else:
                    lines.append(float(best))
            else:
                lines.append(float(best))
            found_mask.append(True)
        else:
            lines.append(float(pred))
            found_mask.append(False)

    n_found = sum(found_mask)
    if 2 <= n_found < 5:
        # Linear fit over observed lines to interpolate missing/occluded line(s).
        ks = np.array([k for k, ok in enumerate(found_mask) if ok], dtype=np.float64)
        ys = np.array([lines[k] for k, ok in enumerate(found_mask) if ok], dtype=np.float64)
        slope, intercept = np.polyfit(ks, ys, 1)
        if 0.7 * sp <= slope <= 1.3 * sp:
            for k, ok in enumerate(found_mask):
                if not ok:
                    lines[k] = float(np.clip(intercept + slope * k, 0.0, float(n - 1)))
    return lines, n_found


def detect_staves_from_probs(
    probs: np.ndarray,
    max_staves: int = 8,
    min_sp: float = 4.0,
    max_sp: float = 40.0,
    min_comb_score: float = 0.03,
    min_lines_found: int = 3,
) -> tuple[list[list[float]], float, dict]:
    """Detect 5-line staves from a 1-D RowNet probability curve.

    Adaptive across lighting/contrast/tilt/missing-line degradations:
      1. Multi-scale smoothing + slow-baseline detrending.
      2. Harmonic comb autocorrelation to lock onto ``sp`` (re-checked at
         coarse smoothing if fine scale is weak).
      3. 5-line periodic comb score + adaptive floor
         (``max(min_comb_score, 0.30 * best_comb, med + 3.0 * mad)``).
      4. Greedy non-maximum suppression (separation ``>= 4.3 * sp``) up to
         ``max_staves``, with per-line local refinement and linear
         interpolation when 1-2 lines are occluded.
    """
    raw = np.asarray(probs, dtype=np.float64).ravel()
    if raw.size < int(4 * min_sp) + 4:
        raise ValueError(f'probability curve too short: {raw.size}')

    # Multi-scale: try fine sigma=0.8 first; fall back to broader sigma=1.6
    # for tilted/motion-blurred photos where peaks are smeared.
    best_sp, snr = estimate_staff_space(raw, min_sp=min_sp, max_sp=max_sp)
    det_fine = smooth_and_detrend_profile(raw, sigma=0.8, bg_window=max(31, int(round(5 * best_sp)) | 1))
    comb = _comb_response(det_fine, best_sp)
    used_sigma = 0.8
    if comb.size == 0 or float(comb.max()) < min_comb_score:
        det_coarse = smooth_and_detrend_profile(
            raw, sigma=1.6, bg_window=max(41, int(round(6 * best_sp)) | 1))
        comb_c = _comb_response(det_coarse, best_sp)
        if comb_c.size > 0 and (comb.size == 0 or float(comb_c.max()) > float(comb.max())):
            det_fine = det_coarse
            comb = comb_c
            used_sigma = 1.6

    if comb.size == 0:
        raise ValueError('cannot determine S: empty comb response')

    peak_comb = float(comb.max())
    med = float(np.median(comb))
    mad = float(np.median(np.abs(comb - med))) + 1e-6
    adaptive_floor = max(min_comb_score, 0.30 * peak_comb, med + 3.0 * mad)
    if peak_comb < adaptive_floor or (snr < 1.2 and peak_comb < 0.08):
        raise ValueError(
            f'cannot determine S: weak periodic signal '
            f'(peak_comb={peak_comb:.4f}, snr={snr:.2f})'
        )

    # Greedy NMS over comb peaks separated by at least 4.3 * sp rows.
    min_sep = max(int(round(4.3 * best_sp)), 4)
    order = np.argsort(comb)[::-1]
    selected_y0 = []
    selected_scores = []
    for idx in order:
        sc = float(comb[idx])
        if sc < adaptive_floor:
            break
        if any(abs(int(idx) - y) < min_sep for y in selected_y0):
            continue
        selected_y0.append(int(idx))
        selected_scores.append(sc)
        if len(selected_y0) >= max_staves:
            break

    if not selected_y0:
        raise ValueError('cannot determine S: no staff candidates survived NMS')

    # Sort top-to-bottom and refine 5 lines per staff.
    pairs = sorted(zip(selected_y0, selected_scores), key=lambda p: p[0])
    lines5 = []
    kept_scores = []
    lines_observed = []
    for y0, sc in pairs:
        five, n_found = _refine_five_lines(raw, det_fine, float(y0), best_sp)
        if n_found >= min_lines_found:
            lines5.append(five)
            kept_scores.append(sc)
            lines_observed.append(n_found)

    if not lines5:
        raise ValueError('cannot determine S: no candidate had >= %d lines' % min_lines_found)

    # Re-estimate median spacing from refined lines.
    diffs = [st[k + 1] - st[k] for st in lines5 for k in range(4)]
    refined_sp = float(np.median(diffs)) if diffs else float(best_sp)
    info = {
        'sp_est': float(best_sp),
        'sp_refined': refined_sp,
        'snr': float(snr),
        'sigma': used_sigma,
        'adaptive_floor': float(adaptive_floor),
        'scores': kept_scores,
        'lines_observed': lines_observed,
    }
    return lines5, refined_sp, info


def infer_staff_count(
    probs: np.ndarray,
    max_staves: int = 8,
    min_sp: float = 4.0,
    max_sp: float = 40.0,
) -> tuple[int, list[list[float]], float, dict]:
    """Return ``(S, lines5, sp, info)`` from a 1-D staff-probability curve."""
    lines5, sp, info = detect_staves_from_probs(
        probs, max_staves=max_staves, min_sp=min_sp, max_sp=max_sp
    )
    return len(lines5), lines5, sp, info
