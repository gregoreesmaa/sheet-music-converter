"""Manuscript paper synthesis, human-like horizontal spacing, and curved-staff geometry.

Pure NumPy implementation driven exclusively by a caller-provided
``numpy.random.Generator`` so every page is reproducible from its seed.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np


def _bilinear_upsample(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinearly upsample a 2D float32 array to (out_h, out_w) using pure NumPy."""
    in_h, in_w = grid.shape
    if in_h == 1 and in_w == 1:
        return np.full((out_h, out_w), float(grid[0, 0]), dtype=np.float32)

    y = np.linspace(0.0, max(in_h - 1, 0), out_h, dtype=np.float32)
    x = np.linspace(0.0, max(in_w - 1, 0), out_w, dtype=np.float32)

    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)

    wy = (y - y0)[:, None]
    wx = (x - x0)[None, :]

    top = grid[y0][:, x0] * (1.0 - wx) + grid[y0][:, x1] * wx
    bot = grid[y1][:, x0] * (1.0 - wx) + grid[y1][:, x1] * wx
    return (top * (1.0 - wy) + bot * wy).astype(np.float32, copy=False)


def make_manuscript_paper(
    width: int,
    height: int,
    style: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesize realistic float32 RGB manuscript paper in ``[0.0, 1.0]``.

    Includes laid-paper horizontal grain, multi-scale foxing/mottling,
    aged edge darkening, subtle reverse-side bleed-through ghosting,
    and optional faint eraser smudges.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Paper dimensions must be positive, got ({width}, {height})")

    base_rgb = np.asarray(style.get("paper_rgb", (0.94, 0.91, 0.85)), dtype=np.float32)
    ink_rgb = np.asarray(style.get("ink_rgb", (0.15, 0.13, 0.11)), dtype=np.float32)
    paper_grain = float(style.get("paper_grain", 0.06))
    paper_mottle = float(style.get("paper_mottle", 0.08))
    bleed_through = float(style.get("bleed_through", 0.06))
    eraser_smudge_prob = float(style.get("eraser_smudge_prob", 0.05))

    # Subtle per-page paper tone drift
    tone_shift = rng.normal(0.0, 0.012, size=3).astype(np.float32)
    page_rgb = np.clip(base_rgb + tone_shift, 0.05, 0.99)
    img = np.empty((height, width, 3), dtype=np.float32)
    img[:, :, 0] = page_rgb[0]
    img[:, :, 1] = page_rgb[1]
    img[:, :, 2] = page_rgb[2]

    # 1. Laid paper grain (fine fiber noise + subtle horizontal laid lines)
    if paper_grain > 0.0:
        fine_noise = rng.normal(0.0, paper_grain * 0.42, size=(height, width)).astype(np.float32)
        laid_period = float(rng.uniform(5.0, 9.0))
        y_idx = np.arange(height, dtype=np.float32)
        laid_wave = np.sin(2.0 * np.pi * y_idx / laid_period + float(rng.uniform(0.0, 2.0 * np.pi)))
        laid_1d = (laid_wave * (paper_grain * 0.16)).astype(np.float32)
        grain_field = fine_noise + laid_1d[:, None]
        img += grain_field[:, :, None]

    # 2. Multi-scale mottling & foxing
    if paper_mottle > 0.0:
        coarse_h = max(4, min(24, height // 64 + 4))
        coarse_w = max(4, min(24, width // 64 + 4))
        med_h = max(8, min(64, height // 24 + 6))
        med_w = max(8, min(64, width // 24 + 6))

        coarse = _bilinear_upsample(
            rng.normal(0.0, 1.0, size=(coarse_h, coarse_w)).astype(np.float32),
            height,
            width,
        )
        medium = _bilinear_upsample(
            rng.normal(0.0, 1.0, size=(med_h, med_w)).astype(np.float32),
            height,
            width,
        )
        mottle = (0.65 * coarse + 0.35 * medium) * (paper_mottle * 0.38)
        # Warm sepia tint on darker mottled areas
        img[:, :, 0] += mottle * 0.85
        img[:, :, 1] += mottle * 1.00
        img[:, :, 2] += mottle * 1.25

        # Occasional small warm-brown foxing spots on aged paper
        if paper_mottle >= 0.08:
            n_spots = int(rng.integers(3, max(4, int(28 * paper_mottle) + 4)))
            yy, xx = np.ogrid[:height, :width]
            for _ in range(n_spots):
                cy = float(rng.uniform(0.0, height))
                cx = float(rng.uniform(0.0, width))
                rad = float(rng.uniform(3.0, max(5.0, min(height, width) * 0.025)))
                y0 = max(0, int(cy - rad * 2.5))
                y1 = min(height, int(cy + rad * 2.5) + 1)
                x0 = max(0, int(cx - rad * 2.5))
                x1 = min(width, int(cx + rad * 2.5) + 1)
                if y1 <= y0 or x1 <= x0:
                    continue
                dy = (yy[y0:y1, :] - cy).astype(np.float32)
                dx = (xx[:, x0:x1] - cx).astype(np.float32)
                dist2 = (dy * dy + dx * dx) / (rad * rad + 1e-6)
                spot = np.exp(-0.5 * dist2) * float(rng.uniform(0.03, 0.14) * paper_mottle * 4.0)
                img[y0:y1, x0:x1, 0] -= spot * 0.55
                img[y0:y1, x0:x1, 1] -= spot * 0.85
                img[y0:y1, x0:x1, 2] -= spot * 1.20

    # 3. Aged sepia edge darkening (subtle vignette)
    edge_strength = float(0.02 + 0.32 * paper_mottle)
    yn = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xn = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    r2 = np.clip(0.5 * (xn * xn + yn * yn), 0.0, 1.0)
    vignette = (r2 * r2) * edge_strength
    img[:, :, 0] -= vignette * 0.72
    img[:, :, 1] -= vignette * 0.92
    img[:, :, 2] -= vignette * 1.18

    # 4. Reverse-side bleed-through ghosting
    if bleed_through > 0.01:
        ghost = np.zeros((height, width), dtype=np.float32)
        # Faint reverse-side staff bands
        n_staves = max(2, min(10, height // 140))
        sp_ghost = max(4.0, height / (n_staves * 14.0))
        margin_x = int(width * 0.08)
        for s in range(n_staves):
            y_base = (s + 0.65) * (height / (n_staves + 0.3)) + float(rng.uniform(-sp_ghost, sp_ghost))
            for li in range(-2, 3):
                yl = int(round(y_base + li * sp_ghost))
                if 1 <= yl < height - 1:
                    ghost[yl - 1 : yl + 2, margin_x : width - margin_x] += np.array(
                        [[0.25], [0.55], [0.25]], dtype=np.float32
                    )
        # Soft blurred reverse-side note blobs
        blob_h = max(8, height // 16)
        blob_w = max(8, width // 16)
        raw_blobs = (rng.random((blob_h, blob_w), dtype=np.float32) > 0.72).astype(np.float32)
        raw_blobs *= rng.uniform(0.3, 1.0, size=(blob_h, blob_w)).astype(np.float32)
        soft_blobs = _bilinear_upsample(raw_blobs, height, width)
        ghost = np.clip(0.35 * ghost + 0.65 * soft_blobs, 0.0, 1.0)
        alpha = (ghost * (bleed_through * 0.42))[:, :, None]
        img = img * (1.0 - alpha) + ink_rgb[None, None, :] * alpha

    # 5. Optional faint eraser smudges
    if eraser_smudge_prob > 0.0 and float(rng.random()) < eraser_smudge_prob:
        n_smudges = int(rng.integers(1, 5))
        yy, xx = np.ogrid[:height, :width]
        for _ in range(n_smudges):
            cy = float(rng.uniform(height * 0.12, height * 0.88))
            cx = float(rng.uniform(width * 0.10, width * 0.90))
            ry = float(rng.uniform(max(8.0, height * 0.015), max(16.0, height * 0.045)))
            rx = float(rng.uniform(max(18.0, width * 0.03), max(36.0, width * 0.10)))
            y0 = max(0, int(cy - ry * 2.5))
            y1 = min(height, int(cy + ry * 2.5) + 1)
            x0 = max(0, int(cx - rx * 2.5))
            x1 = min(width, int(cx + rx * 2.5) + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            dy = (yy[y0:y1, :] - cy).astype(np.float32) / (ry + 1e-6)
            dx = (xx[:, x0:x1] - cx).astype(np.float32) / (rx + 1e-6)
            r2_s = dx * dx + dy * dy
            # Lightened abraded core + faint smudged graphite/ink halo
            core = np.exp(-1.4 * r2_s) * float(rng.uniform(0.025, 0.065))
            halo = np.exp(-0.45 * r2_s) * float(rng.uniform(0.02, 0.055))
            img[y0:y1, x0:x1, :] += core[:, :, None]
            img[y0:y1, x0:x1, :] -= halo[:, :, None] * (1.0 - ink_rgb[None, None, :]) * 0.55

    return np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)


def compute_quirky_note_positions(
    beat_offsets: Sequence[float],
    bar_x0: float,
    bar_x1: float,
    sp: float,
    style: dict[str, Any],
    rng: np.random.Generator,
) -> list[float]:
    """Compute human-like non-uniform horizontal note positions within a measure.

    Applies organic beat stretching/compression, right-barline cramming
    (``end_bar_cram``), and per-note jitter while strictly preserving monotonic
    order (``x[i] + min_gap <= x[i+1]``) inside ``[bar_x0, bar_x1]``.
    """
    n = len(beat_offsets)
    if n == 0:
        return []
    if bar_x1 <= bar_x0:
        raise ValueError(f"bar_x1 ({bar_x1}) must be greater than bar_x0 ({bar_x0})")

    bar_w = float(bar_x1 - bar_x0)
    spacing_jitter = float(np.clip(style.get("spacing_jitter", 0.20), 0.0, 0.80))
    end_bar_cram = float(np.clip(style.get("end_bar_cram", 0.15), 0.0, 0.80))

    left_pad = min(max(0.8 * sp, 1.0), 0.14 * bar_w)
    right_pad = min(max(0.7 * sp, 1.0), 0.12 * bar_w)
    x_min = float(bar_x0 + left_pad)
    x_max = float(bar_x1 - right_pad)
    usable_w = max(x_max - x_min, 1e-3)

    if n == 1:
        center = x_min + usable_w * float(np.clip(0.22 + rng.normal(0.0, 0.08 + 0.12 * spacing_jitter), 0.05, 0.65))
        return [float(np.clip(center, x_min, x_max))]

    beats = np.asarray(beat_offsets, dtype=np.float64)
    b_min = float(beats[0])
    b_max = float(beats[-1])
    span = b_max - b_min

    if span <= 1e-9:
        u = np.linspace(0.0, 1.0, n, dtype=np.float64)
    else:
        u = (beats - b_min) / span
        # Ensure strictly non-decreasing base parameter even if caller passes ties
        for i in range(1, n):
            if u[i] <= u[i - 1]:
                u[i] = u[i - 1] + 1e-4
        u = (u - u[0]) / max(u[-1] - u[0], 1e-9)

    # 1. Right-barline cramming: early notes spread wider, late notes squeeze near bar_x1
    if end_bar_cram > 0.0:
        cram_amt = float(rng.uniform(0.35, 1.0)) * end_bar_cram
        power = 1.0 / (1.0 + 1.8 * cram_amt)
        u_warped = np.power(np.clip(u, 0.0, 1.0), power)
    else:
        u_warped = u.copy()

    # 2. Organic per-interval stretch/compression
    du = np.diff(u_warped)
    du = np.maximum(du, 1e-4)
    if spacing_jitter > 0.0:
        log_scale = rng.normal(0.0, spacing_jitter, size=n - 1)
        scale = np.clip(np.exp(log_scale), 0.35, 2.50)
        du = du * scale

    cum = np.concatenate(([0.0], np.cumsum(du)))
    cum /= max(cum[-1], 1e-9)

    # Slight random left/right margin utilization inside the bar
    start_frac = float(np.clip(rng.uniform(0.0, 0.06 + 0.06 * spacing_jitter), 0.0, 0.15))
    end_frac = float(np.clip(1.0 - rng.uniform(0.0, 0.06 + 0.08 * (1.0 - end_bar_cram)), 0.78, 1.0))
    xs = x_min + (start_frac + cum * (end_frac - start_frac)) * usable_w

    # 3. Small per-note horizontal jitter
    if spacing_jitter > 0.0:
        avg_gap = usable_w / max(n - 1, 1)
        xs = xs + rng.normal(0.0, 0.14 * spacing_jitter * avg_gap, size=n)

    # 4. Enforce strict monotonicity with non-overlapping min_gap inside [x_min, x_max]
    max_feasible_gap = (usable_w * 0.92) / max(n - 1, 1)
    min_gap = float(min(max(0.85 * sp, 1e-3), max_feasible_gap))

    xs[0] = max(xs[0], x_min)
    for i in range(1, n):
        if xs[i] < xs[i - 1] + min_gap:
            xs[i] = xs[i - 1] + min_gap

    xs[-1] = min(xs[-1], x_max)
    for i in range(n - 2, -1, -1):
        if xs[i] > xs[i + 1] - min_gap:
            xs[i] = xs[i + 1] - min_gap

    # Final safety clamp if left edge was pushed slightly below x_min
    if xs[0] < x_min:
        xs = np.linspace(x_min, x_max, n, dtype=np.float64)

    return [float(v) for v in xs]


def build_curved_staff_y(
    x0: float,
    x1: float,
    y_center: float,
    sp: float,
    style: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[Callable[[float | np.ndarray, int], float | np.ndarray], np.ndarray, float]:
    """Build a hand-drawn curved staff y-evaluator and exact cubic Bezier centerline.

    Returns ``(y_fn, ctrl_pts, sp_out)`` where:
      - ``y_fn(x, line_idx)`` evaluates the y-coordinate at horizontal position ``x``
        for staff line ``line_idx`` in ``[-2, -1, 0, 1, 2]`` (and supports ledger lines
        outside that range). For ``line_idx == 0``, ``y_fn(x, 0)`` matches the cubic
        Bezier centerline defined by ``ctrl_pts`` exactly.
      - ``ctrl_pts`` is a ``(4, 2)`` float64 array of exact cubic Bezier control points
        ``[P0, P1, P2, P3]`` for Stage-1 staff-curve supervision.
      - ``sp_out`` is the effective inter-line staff spacing ``sp``.
    """
    if x1 <= x0:
        raise ValueError(f"x1 ({x1}) must be greater than x0 ({x0})")
    if sp <= 0.0:
        raise ValueError(f"sp must be positive, got {sp}")

    span_x = float(x1 - x0)
    curve_px = abs(float(style.get("staff_curve_px", 3.5)))
    slope_max_deg = abs(float(style.get("staff_slope_deg", 0.6)))
    sp_jitter_frac = float(np.clip(style.get("sp_jitter_frac", 0.025), 0.0, 0.15))

    # Global slight staff slope
    slope_deg = float(rng.uniform(-slope_max_deg, slope_max_deg)) if slope_max_deg > 0.0 else 0.0
    dy_slope = span_x * float(np.tan(np.deg2rad(slope_deg)))

    y0 = float(y_center - 0.5 * dy_slope)
    y3 = float(y_center + 0.5 * dy_slope)

    # Smooth quadratic arch + cubic S-wave hand-drawn curve
    arch = float(rng.uniform(-curve_px, curve_px)) if curve_px > 0.0 else 0.0
    scurve = float(rng.uniform(-0.65 * curve_px, 0.65 * curve_px)) if curve_px > 0.0 else 0.0

    y1 = y0 + dy_slope / 3.0 + arch + scurve
    y2 = y0 + 2.0 * dy_slope / 3.0 + arch - scurve

    # Uniformly spaced x control points make B_x(t) = x0 + t * (x1 - x0) exact
    ctrl_pts = np.array(
        [
            [float(x0), y0],
            [float(x0 + span_x / 3.0), y1],
            [float(x0 + 2.0 * span_x / 3.0), y2],
            [float(x1), y3],
        ],
        dtype=np.float64,
    )

    # Subtle per-line sp jitter for the 5 staff lines [-2, -1, 0, 1, 2]
    # Center line (0) has zero offset so y_fn(x, 0) == exact Bezier centerline.
    max_line_shift = 0.22 * sp
    line_offsets: dict[int, float] = {0: 0.0}
    line_drifts: dict[int, float] = {0: 0.0}
    for k in (-2, -1, 1, 2):
        raw_off = float(rng.normal(0.0, sp * sp_jitter_frac))
        raw_drift = float(rng.normal(0.0, sp * sp_jitter_frac * 0.6))
        line_offsets[k] = float(np.clip(raw_off, -max_line_shift, max_line_shift))
        line_drifts[k] = float(np.clip(raw_drift, -0.5 * max_line_shift, 0.5 * max_line_shift))

    cy0, cy1, cy2, cy3 = float(y0), float(y1), float(y2), float(y3)
    inv_span = 1.0 / span_x
    sp_out = float(sp)

    def y_fn(x: float | np.ndarray, line_idx: int = 0) -> float | np.ndarray:
        is_scalar = np.ndim(x) == 0
        x_arr = np.asarray(x, dtype=np.float64)
        t = (x_arr - x0) * inv_span
        omt = 1.0 - t
        y_mid = (
            (omt * omt * omt) * cy0
            + 3.0 * (omt * omt) * t * cy1
            + 3.0 * omt * (t * t) * cy2
            + (t * t * t) * cy3
        )
        idx = int(line_idx)
        offset = line_offsets.get(idx, 0.0)
        drift = line_drifts.get(idx, 0.0)
        y_val = y_mid + idx * sp_out + offset + drift * (t - 0.5)
        if is_scalar:
            return float(y_val)
        return y_val

    return y_fn, ctrl_pts, sp_out
