#!/usr/bin/env python3
"""Staffline-aware vertical dewarping & unit_size scale normalization for UniOMR.

Promoted from Iter 13 cloud ablation ``i13-10`` (``oemer-staff-dewarp-unitsize``):
  - ``oemer_staffline_vertical_dewarp``: OEMER/muskew-inspired column-band
    curved staff-line vertical dewarping. Computes the 24-stripe shift profile
    at a ``<= 1280px`` proxy scale for fast ``<80ms`` CPU execution and
    auto-skips flat pages when ``peak_to_peak < auto_skip_px`` (default 1.5 px),
    guaranteeing bit-identical zero-drift output on clean flat pages.
  - ``estimate_unitsize_max_dim``: Homr-inspired inter-staffline spacing
    (``unit_size``) estimator using vertical autocorrelation of horizontal
    staffline ridges with **first-local-peak** detection
    (``ac[k] > ac[k-1] and ac[k] >= ac[k+1]``) so noisy photo profiles never
    falsely pin to the left lag boundary ``4.0``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_DEWARP_PROXY_MAX_DIM = 1280
DEFAULT_DEWARP_AUTO_SKIP_PX = 1.5

DEFAULT_UNITSIZE_BASE_DIM = 1408
DEFAULT_UNITSIZE_MIN_DIM = 1152
DEFAULT_UNITSIZE_MAX_DIM = 1600
# Reference inter-staffline spacing at 1280px is 7.5 px (scaled linearly with base_max_dim).
_REF_UNIT_PX_AT_1280 = 7.5


def oemer_staffline_vertical_dewarp(
    img_pil: Any,
    auto_skip_px: float = DEFAULT_DEWARP_AUTO_SKIP_PX,
    proxy_max_dim: int = DEFAULT_DEWARP_PROXY_MAX_DIM,
) -> Tuple[Any, Dict[str, Any]]:
    """Staffline-aware vertical ``y(x)`` dewarping with fast proxy-scale estimation.

    Extracts horizontal staff-line ridge responses on a ``<= proxy_max_dim``
    downscaled grayscale proxy, tracks sequential vertical phase shifts across
    24 column stripes outward from the page center via 1D cross-correlation,
    fits a Gaussian-smoothed displacement curve ``y(x)`` clamped to ``|dy| <= 3% H``,
    and remaps the full-resolution image with replicate paper-border fill.

    Clean flat pages whose peak-to-peak vertical shift is ``< auto_skip_px``
    return the original ``img_pil`` object untouched (bit-identical, zero drift).
    """
    import cv2
    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter1d

    t0 = time.perf_counter()
    w, h = img_pil.size
    if h < 128 or w < 128:
        return img_pil, {"dewarp_ms": 0.0, "max_dy_px": 0.0, "skipped": True}

    # Estimate the 24-stripe shift curve at <= proxy_max_dim for <80ms CPU speed.
    max_side = max(w, h)
    if proxy_max_dim > 0 and max_side > proxy_max_dim:
        scale = float(proxy_max_dim) / float(max_side)
        pw = max(64, int(round(w * scale)))
        ph = max(64, int(round(h * scale)))
        gray = np.asarray(
            img_pil.convert("L").resize((pw, ph), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        x_scale_back = float(w) / float(pw)
        y_scale_back = float(h) / float(ph)
    else:
        pw, ph = w, h
        gray = np.asarray(img_pil.convert("L"), dtype=np.float32)
        x_scale_back = 1.0
        y_scale_back = 1.0

    # Invert so dark ink is positive; isolate horizontal staffline ridges.
    inv = 255.0 - gray
    k_len = max(15, int(round(pw * 0.025)))
    horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
    staff_ridge = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_k)

    n_stripes = 24
    xs = np.linspace(0, pw, n_stripes + 1, dtype=np.int32)
    centers_p = 0.5 * (xs[:-1] + xs[1:])
    profiles = np.zeros((n_stripes, ph), dtype=np.float32)
    for i in range(n_stripes):
        strip = staff_ridge[:, xs[i] : xs[i + 1]]
        if strip.size == 0:
            continue
        prof = strip.mean(axis=1)
        prof = prof - prof.mean()
        norm = float(np.linalg.norm(prof))
        if norm > 1e-4:
            profiles[i] = prof / norm

    # Track sequential vertical shifts outward from the center stripe.
    max_step = max(4, int(round(ph * 0.008)))
    shifts = np.zeros(n_stripes, dtype=np.float32)
    mid = n_stripes // 2

    def _best_shift(ref: np.ndarray, cur: np.ndarray) -> float:
        if np.linalg.norm(ref) < 1e-4 or np.linalg.norm(cur) < 1e-4:
            return 0.0
        best_s = 0
        best_v = -1.0
        for s in range(-max_step, max_step + 1):
            if s >= 0:
                val = float(np.dot(ref[: ph - s], cur[s:]))
            else:
                val = float(np.dot(ref[-s:], cur[: ph + s]))
            if val > best_v:
                best_v = val
                best_s = s
        return float(best_s) if best_v > 0.15 else 0.0

    for i in range(mid + 1, n_stripes):
        shifts[i] = shifts[i - 1] + _best_shift(profiles[i - 1], profiles[i])
    for i in range(mid - 1, -1, -1):
        shifts[i] = shifts[i + 1] + _best_shift(profiles[i + 1], profiles[i])

    # Remove median offset, smooth, scale to original pixels, and clamp to <= 3% H.
    shifts = shifts - float(np.median(shifts))
    shifts = gaussian_filter1d(shifts, sigma=1.8, mode="nearest")
    shifts_orig = shifts * y_scale_back
    max_dy = float(h * 0.03)
    shifts_orig = np.clip(shifts_orig, -max_dy, max_dy)

    centers_orig = centers_p * x_scale_back
    col_x = np.arange(w, dtype=np.float32)
    col_dy = np.interp(col_x, centers_orig, shifts_orig).astype(np.float32)

    peak_to_peak = float(col_dy.max() - col_dy.min())
    if peak_to_peak < float(auto_skip_px):
        ms = (time.perf_counter() - t0) * 1000.0
        return img_pil, {
            "dewarp_ms": round(ms, 2),
            "max_dy_px": round(peak_to_peak, 2),
            "skipped": True,
        }

    map_x = np.broadcast_to(col_x[None, :], (h, w)).astype(np.float32)
    row_y = np.arange(h, dtype=np.float32)[:, None]
    map_y = (row_y + col_dy[None, :]).astype(np.float32)

    rgb = np.asarray(img_pil.convert("RGB"))
    remapped = cv2.remap(
        rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    ms = (time.perf_counter() - t0) * 1000.0
    return Image.fromarray(remapped), {
        "dewarp_ms": round(ms, 2),
        "max_dy_px": round(peak_to_peak, 2),
        "skipped": False,
    }


def estimate_unitsize_max_dim(
    img_pil: Any,
    base_max_dim: int = DEFAULT_UNITSIZE_BASE_DIM,
    min_dim: int = DEFAULT_UNITSIZE_MIN_DIM,
    max_dim: int = DEFAULT_UNITSIZE_MAX_DIM,
    target_unit_px: Optional[float] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Estimate adaptive ``max_dim`` from inter-staffline spacing (``unit_size``).

    Measures the dominant vertical staff-space period at ``base_max_dim`` via
    1D autocorrelation of horizontal ink ridges, using **first-local-peak**
    detection (``ac[k] > ac[k-1] and ac[k] >= ac[k+1]``) instead of raw
    ``argmax`` so monotonically decaying noisy profiles cannot pin to lag 4.0.
    """
    import cv2
    import numpy as np
    from PIL import Image

    t0 = time.perf_counter()
    base_dim = int(base_max_dim)
    target_u = (
        float(target_unit_px)
        if target_unit_px is not None
        else _REF_UNIT_PX_AT_1280 * (float(base_dim) / 1280.0)
    )

    w0, h0 = img_pil.size
    if w0 <= 0 or h0 <= 0:
        ms = (time.perf_counter() - t0) * 1000.0
        return base_dim, {
            "unitsize_ms": round(ms, 2),
            "est_unit_px": round(target_u, 2),
            "ac_peak": 0.0,
            "target_max_dim": base_dim,
        }

    ref_scale = float(base_dim) / float(max(w0, h0))
    rw = max(64, int(round(w0 * ref_scale)))
    rh = max(64, int(round(h0 * ref_scale)))
    gray = np.asarray(
        img_pil.convert("L").resize((rw, rh), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    inv = 255.0 - gray

    k_len = max(15, int(round(rw * 0.025)))
    horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
    ridge = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_k)
    prof = ridge.mean(axis=1)
    prof = prof - prof.mean()
    if float(np.linalg.norm(prof)) < 1e-4:
        ms = (time.perf_counter() - t0) * 1000.0
        return base_dim, {
            "unitsize_ms": round(ms, 2),
            "est_unit_px": round(target_u, 2),
            "ac_peak": 0.0,
            "target_max_dim": base_dim,
        }

    ac = np.correlate(prof, prof, mode="full")[len(prof) - 1 :]
    if ac[0] > 1e-6:
        ac = ac / ac[0]

    lo = 4
    hi = min(32, len(ac) - 2)
    if hi <= lo:
        ms = (time.perf_counter() - t0) * 1000.0
        return base_dim, {
            "unitsize_ms": round(ms, 2),
            "est_unit_px": round(target_u, 2),
            "ac_peak": 0.0,
            "target_max_dim": base_dim,
        }

    # First-local-peak detection: strictly reject monotonically decaying left edges.
    min_corr = 0.08
    peaks: List[Tuple[int, float]] = []
    for k in range(lo, hi + 1):
        val = float(ac[k])
        if val > float(ac[k - 1]) and val >= float(ac[k + 1]) and val >= min_corr:
            peaks.append((k, val))

    if not peaks:
        est_u = target_u
        best_corr = 0.0
    else:
        max_peak = max(v for _, v in peaks)
        sig_thr = max(min_corr, 0.35 * max_peak)
        best_lag, best_corr = next(
            ((k, v) for k, v in peaks if v >= sig_thr),
            peaks[0],
        )
        est_u = float(best_lag)

    ratio = target_u / max(3.5, est_u)
    target_dim = int(round(base_dim * ratio))
    target_dim = max(int(min_dim), min(int(max_dim), target_dim))
    # Snap to multiple of 32 for clean FPN/axial stride alignment.
    target_dim = int(round(target_dim / 32.0) * 32)
    target_dim = max(int(min_dim), min(int(max_dim), target_dim))

    ms = (time.perf_counter() - t0) * 1000.0
    return target_dim, {
        "unitsize_ms": round(ms, 2),
        "est_unit_px": round(est_u, 2),
        "ac_peak": round(best_corr, 4),
        "target_max_dim": target_dim,
    }
