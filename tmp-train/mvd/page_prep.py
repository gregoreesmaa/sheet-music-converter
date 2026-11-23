#!/usr/bin/env python3
"""Proposal-only page preparation: dark border crop + projection-profile deskew.

Prepares captured / photo-degraded sheet-music pages for downstream staff
detection (RowNet / ``neural_detect_staves``) without modifying clean pages:

1. **Dark desk border removal:** detects solid/dark desk margins on all 4
   sides (e.g. ``suite31/photo/*.png`` 100 px black border, or ``gate20``
   dark desk margins) by scanning inward while edge rows/columns lack
   paper-bright pixels, stopping before any paper/notation pixel is clipped.
2. **Outer desk-wedge neutralization:** when the cropped page still has dark
   triangular desk wedges in corners (from rotation/perspective warp on a
   dark desk), replaces only pixels outside the paper's 4-edge monotonic
   envelope with the page's median paper tone. Clean unbordered pages have
   bright corners and are untouched.
3. **Coarse-to-fine Radon / horizontal projection-profile deskew:** searches
   ``[-max_skew_deg, +max_skew_deg]`` (default ``[-10, +10]`` deg) to
   maximize horizontal ink projection gradient energy (crisp parallel staff
   lines), then deskews with paper-median fill when ``|skew_deg|`` exceeds
   ``skew_apply_threshold`` (default ``0.15`` deg).

Pure numpy + PIL, deterministic, fast (~100-200 ms per full page).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image

__all__ = [
    'detect_dark_border',
    'estimate_skew_deg',
    'prep_page',
]


def _paper_stats(rgb: np.ndarray, gray: np.ndarray) -> Tuple[float, Tuple[int, int, int]]:
    """Estimate paper background brightness (gray) and RGB fill color."""
    h, w = gray.shape
    y0, y1 = h // 4, max(h // 4 + 1, 3 * h // 4)
    x0, x1 = w // 4, max(w // 4 + 1, 3 * w // 4)
    center_g = gray[y0:y1, x0:x1]
    paper_gray = float(np.percentile(center_g, 85))
    center_rgb = rgb[y0:y1, x0:x1]
    paper_rgb = tuple(
        int(round(float(np.percentile(center_rgb[:, :, c], 85))))
        for c in range(3)
    )
    return paper_gray, paper_rgb


def _dark_threshold(paper_gray: float) -> float:
    """Conservative intensity ceiling separating dark desk from paper."""
    return min(135.0, max(35.0, paper_gray - 55.0))


def detect_dark_border(
    gray: np.ndarray,
    paper_gray: float | None = None,
    max_border_frac: float = 0.25,
    paper_row_frac: float = 0.10,
) -> Tuple[int, int, int, int]:
    """Return ``(x0, y0, x1, y1)`` cropping solid/dark desk borders on 4 sides.

    Scans inward from each edge while fewer than ``paper_row_frac`` of
    pixels in the row/column are paper-bright (``>= dark_thresh``). Stops
    at the first row/column reaching paper so rotated pages and notation
    are never clipped. Caps each side at ``max_border_frac`` of the axis.
    """
    h, w = gray.shape
    if h < 16 or w < 16:
        return (0, 0, w, h)
    if paper_gray is None:
        y0c, y1c = h // 4, max(h // 4 + 1, 3 * h // 4)
        x0c, x1c = w // 4, max(w // 4 + 1, 3 * w // 4)
        paper_gray = float(np.percentile(gray[y0c:y1c, x0c:x1c], 85))
    dark_thresh = _dark_threshold(paper_gray)
    is_paper = gray >= dark_thresh

    row_paper = is_paper.mean(axis=1)
    col_paper = is_paper.mean(axis=0)

    max_y = int(h * max_border_frac)
    max_x = int(w * max_border_frac)

    y0 = 0
    while y0 < max_y and row_paper[y0] < paper_row_frac:
        y0 += 1

    y1 = h
    while (h - y1) < max_y and y1 > y0 + 16 and row_paper[y1 - 1] < paper_row_frac:
        y1 -= 1

    x0 = 0
    while x0 < max_x and col_paper[x0] < paper_row_frac:
        x0 += 1

    x1 = w
    while (w - x1) < max_x and x1 > x0 + 16 and col_paper[x1 - 1] < paper_row_frac:
        x1 -= 1

    return (int(x0), int(y0), int(x1), int(y1))


def _has_dark_corner(gray: np.ndarray, dark_thresh: float, patch: int = 8) -> bool:
    """True iff at least one outer corner patch is dark desk."""
    h, w = gray.shape
    p = min(patch, max(1, min(h, w) // 8))
    corners = (
        gray[:p, :p],
        gray[:p, w - p:],
        gray[h - p:, :p],
        gray[h - p:, w - p:],
    )
    return any(float(np.median(c)) < dark_thresh for c in corners)


def _neutralize_outer_desk_wedges(
    rgb: np.ndarray,
    gray: np.ndarray,
    paper_rgb: Tuple[int, int, int],
    dark_thresh: float,
    max_depth_frac: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Replace outer dark triangular desk wedges with ``paper_rgb``.

    Uses a 4-edge monotonic paper envelope: any dark pixel with no paper
    pixel between it and at least one image edge (and within
    ``max_depth_frac`` of that edge) is strictly outside the paper page
    and replaced with paper tone. Interior ink is never touched.
    """
    if not _has_dark_corner(gray, dark_thresh):
        return rgb, gray, False
    h, w = gray.shape
    paper_mask = gray >= dark_thresh
    from_left = np.maximum.accumulate(paper_mask, axis=1)
    from_right = np.maximum.accumulate(paper_mask[:, ::-1], axis=1)[:, ::-1]
    from_top = np.maximum.accumulate(paper_mask, axis=0)
    from_bot = np.maximum.accumulate(paper_mask[::-1, :], axis=0)[::-1, :]

    max_x = int(w * max_depth_frac)
    max_y = int(h * max_depth_frac)
    outside = np.zeros((h, w), dtype=bool)
    if max_x > 0:
        outside[:, :max_x] |= ~from_left[:, :max_x]
        outside[:, w - max_x:] |= ~from_right[:, w - max_x:]
    if max_y > 0:
        outside[:max_y, :] |= ~from_top[:max_y, :]
        outside[h - max_y:, :] |= ~from_bot[h - max_y:, :]
    outside &= ~paper_mask
    if not np.any(outside):
        return rgb, gray, False

    out_rgb = rgb.copy()
    out_rgb[outside] = paper_rgb
    paper_g = int(round(0.299 * paper_rgb[0] + 0.587 * paper_rgb[1] + 0.114 * paper_rgb[2]))
    out_gray = gray.copy()
    out_gray[outside] = paper_g
    return out_rgb, out_gray, True


def _build_ink_image(gray: np.ndarray, paper_gray: float, target_width: int) -> Image.Image:
    """Downsampled, elliptically masked ink map for projection energy."""
    h, w = gray.shape
    scale = min(1.0, float(target_width) / max(1, w))
    nw = max(32, int(round(w * scale)))
    nh = max(32, int(round(h * scale)))
    pil_g = Image.fromarray(gray, mode='L')
    if nw != w or nh != h:
        pil_g = pil_g.resize((nw, nh), Image.BILINEAR)
    arr = np.asarray(pil_g, dtype=np.float32)
    ink = np.clip(paper_gray - arr, 0.0, 255.0)
    ink = np.where(ink > 25.0, ink, 0.0)

    yy, xx = np.ogrid[:nh, :nw]
    cy, cx = (nh - 1) * 0.5, (nw - 1) * 0.5
    ry, rx = max(1.0, nh * 0.46), max(1.0, nw * 0.46)
    mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    ink *= mask
    return Image.fromarray(np.clip(ink, 0, 255).astype(np.uint8), mode='L')


def _projection_energy(ink_img: Image.Image, rot_ccw_deg: float) -> float:
    """Horizontal projection-profile gradient energy after CCW rotation."""
    if abs(rot_ccw_deg) < 1e-6:
        rot = ink_img
    else:
        rot = ink_img.rotate(rot_ccw_deg, resample=Image.BILINEAR,
                             expand=False, fillcolor=0)
    arr = np.asarray(rot, dtype=np.float32)
    proj = arr.mean(axis=1)
    diff = np.diff(proj)
    return float(np.mean(diff * diff))


def estimate_skew_deg(
    gray: np.ndarray,
    paper_gray: float | None = None,
    max_skew_deg: float = 10.0,
) -> float:
    """Estimate page skew angle in ``[-max_skew_deg, +max_skew_deg]``.

    Sign convention: if a level page was rotated counter-clockwise by
    ``+theta`` degrees (as in ``PIL.Image.rotate(+theta)``), this returns
    ``skew_deg ≈ +theta``, so deskewing rotates by ``-skew_deg`` CCW.
    Returns ``0.0`` if the page has negligible ink.
    """
    h, w = gray.shape
    if h < 32 or w < 32:
        return 0.0
    if paper_gray is None:
        y0c, y1c = h // 4, max(h // 4 + 1, 3 * h // 4)
        x0c, x1c = w // 4, max(w // 4 + 1, 3 * w // 4)
        paper_gray = float(np.percentile(gray[y0c:y1c, x0c:x1c], 85))

    ink_coarse = _build_ink_image(gray, paper_gray, target_width=400)
    if float(np.asarray(ink_coarse).max()) < 1.0:
        return 0.0

    # Stage 1: coarse 0.5-deg sweep over [-max_skew_deg, +max_skew_deg]
    coarse_angles = np.arange(-max_skew_deg, max_skew_deg + 1e-6, 0.5)
    best_rot = max(coarse_angles, key=lambda a: _projection_energy(ink_coarse, float(a)))

    # Stage 2: medium 0.1-deg sweep around best_rot
    ink_med = _build_ink_image(gray, paper_gray, target_width=600)
    lo = max(-max_skew_deg, float(best_rot) - 0.8)
    hi = min(max_skew_deg, float(best_rot) + 0.8)
    med_angles = np.arange(lo, hi + 1e-6, 0.1)
    best_rot = max(med_angles, key=lambda a: _projection_energy(ink_med, float(a)))

    # Stage 3: fine 0.025-deg sweep around best_rot
    ink_fine = _build_ink_image(gray, paper_gray, target_width=720)
    lo = max(-max_skew_deg, float(best_rot) - 0.15)
    hi = min(max_skew_deg, float(best_rot) + 0.15)
    fine_angles = np.arange(lo, hi + 1e-6, 0.025)
    best_rot = max(fine_angles, key=lambda a: _projection_energy(ink_fine, float(a)))

    # best_rot is the CCW rotation that makes lines horizontal, so the
    # original page's CCW skew is -best_rot.
    skew_deg = -float(best_rot)
    if abs(skew_deg) < 1e-6:
        skew_deg = 0.0
    return round(skew_deg, 4)


def prep_page(
    img_or_path: Union[str, Path, Image.Image],
    max_skew_deg: float = 10.0,
    skew_apply_threshold: float = 0.15,
    max_border_frac: float = 0.25,
) -> Tuple[Image.Image, dict]:
    """Border-crop + deskew a page image for downstream staff detection.

    Returns ``(prepped_pil_image, prep_meta)`` where ``prep_meta`` records
    ``border_crop_box``, ``skew_deg``, ``applied``, ``border_cropped``,
    ``deskew_applied``, ``desk_wedges_filled``, ``orig_size``, and
    ``final_size``.
    """
    if isinstance(img_or_path, (str, Path)):
        with Image.open(img_or_path) as im:
            im.load()
            orig_mode = im.mode
            rgb_img = im.convert('RGB')
    else:
        orig_mode = img_or_path.mode
        rgb_img = img_or_path.convert('RGB')

    orig_w, orig_h = rgb_img.size
    rgb = np.asarray(rgb_img, dtype=np.uint8)
    gray = np.asarray(rgb_img.convert('L'), dtype=np.uint8)
    paper_gray, paper_rgb = _paper_stats(rgb, gray)
    dark_thresh = _dark_threshold(paper_gray)

    # 1. Crop solid/dark outer desk border
    x0, y0, x1, y1 = detect_dark_border(
        gray, paper_gray=paper_gray, max_border_frac=max_border_frac)
    border_cropped = (x0, y0, x1, y1) != (0, 0, orig_w, orig_h)
    if border_cropped:
        rgb = rgb[y0:y1, x0:x1]
        gray = gray[y0:y1, x0:x1]
        # Re-estimate paper stats inside the cropped page
        paper_gray, paper_rgb = _paper_stats(rgb, gray)
        dark_thresh = _dark_threshold(paper_gray)

    # 2. Neutralize dark corner desk wedges (rotated/perspective page on dark desk)
    rgb, gray, wedges_filled = _neutralize_outer_desk_wedges(
        rgb, gray, paper_rgb, dark_thresh, max_depth_frac=max_border_frac)

    # 3. Estimate skew via projection-profile gradient energy
    skew_deg = estimate_skew_deg(gray, paper_gray=paper_gray,
                                 max_skew_deg=max_skew_deg)
    deskew_applied = abs(skew_deg) >= skew_apply_threshold

    out_img = Image.fromarray(rgb, mode='RGB')
    if deskew_applied:
        out_img = out_img.rotate(
            -skew_deg,
            resample=Image.BICUBIC,
            expand=False,
            fillcolor=paper_rgb,
        )

    if orig_mode == 'L':
        out_img = out_img.convert('L')

    applied = bool(border_cropped or wedges_filled or deskew_applied)
    prep_meta = {
        'orig_size': [orig_w, orig_h],
        'border_crop_box': [x0, y0, x1, y1],
        'skew_deg': skew_deg,
        'applied': applied,
        'border_cropped': border_cropped,
        'desk_wedges_filled': wedges_filled,
        'deskew_applied': deskew_applied,
        'final_size': [out_img.size[0], out_img.size[1]],
    }
    return out_img, prep_meta
