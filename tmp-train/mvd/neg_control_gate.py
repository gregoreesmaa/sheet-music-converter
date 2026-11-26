#!/usr/bin/env python3
"""Fast CPU negative-control pre-gate for OMR full-page inference.

Rejects non-music pages (blank staves, text-only program notes, title pages,
completely blank pages) before running neural decoding so the model never
hallucinates notes on negative controls.

Empirically calibrated on the 125-page OMR-NorthStar local eval pack:
  - `v_frac`: fraction of pixels belonging to vertical dark-ink runs of length
    >= max(3, int(0.004 * H)). Blank staves have ~0.00000; real music pages
    have stems/barlines >= 0.00305 (~6x above default v_thr=0.0005).
  - `h_frac`: fraction of pixels belonging to horizontal dark-ink runs of length
    >= max(16, int(0.04 * W)). Text-only program notes have ~0.00071; real
    music pages have staff lines >= 0.00248 (~3.5x above default h_thr=0.0015).
    If the 0-degree pass falls below `h_thr`, [-3 deg, +3 deg] rotations are
    checked and the max is taken so skewed archival scans never false-trigger.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from PIL import Image


def _to_grayscale_resized(img_pil: Image.Image, max_width: int) -> Image.Image:
    """Convert PIL image to grayscale (compositing RGBA on white) and cap width."""
    if img_pil.mode in ("RGBA", "LA") or (
        img_pil.mode == "P" and "transparency" in img_pil.info
    ):
        rgba = img_pil.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        gray = Image.alpha_composite(bg, rgba).convert("L")
    else:
        gray = img_pil.convert("L")

    w, h = gray.size
    if max_width > 0 and w > max_width:
        new_h = max(1, int(round(h * (max_width / float(w)))))
        gray = gray.resize((max_width, new_h), Image.Resampling.BILINEAR)
    return gray


def _run_fraction(mask: np.ndarray, min_run: int, axis: int) -> float:
    """Return fraction of pixels in contiguous True runs of length >= min_run."""
    total = int(mask.size)
    if total == 0:
        return 0.0
    if min_run <= 1:
        return float(mask.sum()) / float(total)

    mat = mask if axis == 1 else mask.T
    padded = np.pad(mat, ((0, 0), (1, 1)), mode="constant", constant_values=False)
    flat = padded.ravel()
    diff = np.diff(flat.view(np.int8))
    starts = np.flatnonzero(diff == 1) + 1
    ends = np.flatnonzero(diff == -1) + 1
    if starts.size == 0:
        return 0.0
    lengths = ends - starts
    kept_px = int(lengths[lengths >= min_run].sum())
    return float(kept_px) / float(total)


def is_non_music_page(
    img_pil: Image.Image,
    *,
    v_thr: float = 0.0005,
    h_thr: float = 0.0015,
    max_width: int = 800,
) -> tuple[bool, dict[str, Any]]:
    """Return ``(is_non_music, metrics)`` for a page image on CPU.

    Parameters
    ----------
    img_pil:
        Input PIL image of a full page.
    v_thr:
        Minimum vertical ink-run pixel fraction required for a music page.
    h_thr:
        Minimum horizontal ink-run pixel fraction required for a music page.
    max_width:
        Maximum width to downsample to before feature extraction (preserving aspect).
    """
    t0 = time.perf_counter()
    gray = _to_grayscale_resized(img_pil, max_width=max_width)
    arr = np.asarray(gray, dtype=np.float32)
    h, w = arr.shape

    p85 = float(np.percentile(arr, 85)) if arr.size > 0 else 255.0
    ink_thr = min(p85 - 30.0, 0.80 * p85)
    ink = arr < ink_thr

    v_min_run = max(3, int(0.004 * h))
    h_min_run = max(16, int(0.04 * w))

    v_frac = _run_fraction(ink, v_min_run, axis=0)
    h_frac_0 = _run_fraction(ink, h_min_run, axis=1)
    h_frac = h_frac_0
    h_angle_deg = 0.0

    if h_frac < h_thr:
        fill_val = int(round(max(0.0, min(255.0, p85))))
        for angle in (-3.0, 3.0):
            rot_im = gray.rotate(
                angle,
                resample=Image.Resampling.BILINEAR,
                fillcolor=fill_val,
            )
            rot_arr = np.asarray(rot_im, dtype=np.float32)
            rot_ink = rot_arr < ink_thr
            cand = _run_fraction(rot_ink, h_min_run, axis=1)
            if cand > h_frac:
                h_frac = cand
                h_angle_deg = angle

    low_v = v_frac < v_thr
    low_h = h_frac < h_thr
    is_non_music = bool(low_v or low_h)

    if low_v and low_h:
        reason = "low_v_and_h_frac"
    elif low_v:
        reason = "low_v_frac"
    elif low_h:
        reason = "low_h_frac"
    else:
        reason = "ok"

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    info: dict[str, Any] = {
        "is_non_music": is_non_music,
        "reason": reason,
        "v_frac": round(v_frac, 6),
        "h_frac": round(h_frac, 6),
        "h_frac_0": round(h_frac_0, 6),
        "h_angle_deg": h_angle_deg,
        "v_thr": v_thr,
        "h_thr": h_thr,
        "v_min_run": v_min_run,
        "h_min_run": h_min_run,
        "p85": round(p85, 2),
        "ink_thr": round(ink_thr, 2),
        "width": int(w),
        "height": int(h),
        "elapsed_ms": round(elapsed_ms, 2),
    }
    return is_non_music, info
