"""Polymorphic handwritten musical glyph generator.

Draws rests ("pauses"), noteheads, stems/flags, beams, clefs, accidentals,
time signatures, barlines, and braces directly onto a StrokeCanvas using
parametric Bezier/polyline strokes with continuous control-point perturbation
and multi-archetype polymorphism.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from .stroke_engine import (
        RGBLike,
        StrokeCanvas,
        draw_calligraphic_polyline,
        draw_filled_blob,
    )
except ImportError:
    from stroke_engine import (
        RGBLike,
        StrokeCanvas,
        draw_calligraphic_polyline,
        draw_filled_blob,
    )


QUARTER_REST_ARCHETYPES: Tuple[str, ...] = (
    "lightning_z",
    "backwards_7_hook",
    "cursive_3",
    "hurried_slash_zig",
    "baroque_c_hook",
    "calligraphic_s_curl",
)

EIGHTH_REST_ARCHETYPES: Tuple[str, ...] = (
    "dot_head",
    "loop_head",
    "quick_tick_hook",
)

NOTEHEAD_ARCHETYPES: Tuple[str, ...] = (
    "oval",
    "slash",
    "teardrop",
    "compact",
)

CLEF_ARCHETYPES: Tuple[str, ...] = (
    "ornate",
    "simplified",
    "hurried",
)


# ---------------------------------------------------------------------------
# Geometry & Sampling Helpers
# ---------------------------------------------------------------------------


def _eval_cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, n: int = 16) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    mt = 1.0 - t
    return (mt**3) * p0 + 3.0 * (mt**2) * t * p1 + 3.0 * mt * (t**2) * p2 + (t**3) * p3


def _eval_quad(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 12) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    mt = 1.0 - t
    return (mt**2) * p0 + 2.0 * mt * t * p1 + (t**2) * p2


def _eval_catmull_rom(ctrl: np.ndarray, samples_per_seg: int = 10) -> np.ndarray:
    """Evaluate a smooth Catmull-Rom spline passing through all control points."""
    pts = np.asarray(ctrl, dtype=np.float64)
    if len(pts) <= 2:
        return pts
    # Pad endpoints by reflection for natural tangents.
    p_ext = np.vstack((2.0 * pts[0] - pts[1], pts, 2.0 * pts[-1] - pts[-2]))
    out: List[np.ndarray] = []
    t = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)[:, None]
    t2 = t * t
    t3 = t2 * t
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = p_ext[i], p_ext[i + 1], p_ext[i + 2], p_ext[i + 3]
        seg = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        out.append(seg)
    out.append(pts[-1:])
    return np.vstack(out)


def _transform_local(
    pts_norm: np.ndarray,
    cx: float,
    cy: float,
    sp: float,
    rng: np.random.Generator,
    style: dict,
    jitter_sp: float = 0.08,
    scale_x_range: Tuple[float, float] = (0.88, 1.14),
    scale_y_range: Tuple[float, float] = (0.88, 1.14),
    extra_slant_deg: float = 0.0,
) -> np.ndarray:
    """Perturb normalized (in sp units) control points with jitter, scale, slant, and offset."""
    pts = np.asarray(pts_norm, dtype=np.float64).copy()
    if jitter_sp > 0.0:
        pts += rng.normal(0.0, jitter_sp, size=pts.shape)

    sx = float(rng.uniform(*scale_x_range))
    sy = float(rng.uniform(*scale_y_range))
    slant_deg = float(style.get("slant_deg", 0.0)) + extra_slant_deg + float(rng.normal(0.0, 4.5))
    shear = math.tan(math.radians(float(np.clip(slant_deg, -28.0, 28.0))))

    x = pts[:, 0] * sx - pts[:, 1] * shear
    y = pts[:, 1] * sy

    ox = float(rng.normal(0.0, 0.04 * sp))
    oy = float(rng.normal(0.0, 0.04 * sp))
    return np.column_stack((cx + ox + x * sp, cy + oy + y * sp))


def _choose_weighted(
    choices: Sequence[str],
    weights_spec: Optional[Union[Dict[str, float], Sequence[float]]],
    rng: np.random.Generator,
    default_weights: Optional[Sequence[float]] = None,
) -> str:
    """Pick an item from choices using dict/sequence weights or defaults."""
    if isinstance(weights_spec, dict) and weights_spec:
        w = np.array([max(0.0, float(weights_spec.get(c, 0.0))) for c in choices], dtype=np.float64)
    elif isinstance(weights_spec, (list, tuple, np.ndarray)) and len(weights_spec) == len(choices):
        w = np.maximum(0.0, np.asarray(weights_spec, dtype=np.float64))
    elif default_weights is not None:
        w = np.maximum(0.0, np.asarray(default_weights, dtype=np.float64))
    else:
        w = np.ones(len(choices), dtype=np.float64)

    total = float(np.sum(w))
    if total <= 1e-12:
        w = np.ones(len(choices), dtype=np.float64)
        total = float(len(choices))
    probs = w / total
    idx = int(rng.choice(len(choices), p=probs))
    return choices[idx]


# ---------------------------------------------------------------------------
# Rests ("Pauses")
# ---------------------------------------------------------------------------


def _draw_quarter_rest_archetype(
    canvas: StrokeCanvas,
    archetype: str,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Render one of 6 distinct handwritten quarter-rest archetypes."""
    base_w = max(1.2, sp * float(style.get("glyph_stroke_ratio", 0.22)))

    if archetype == "lightning_z":
        # Classic sharp manuscript Z / lightning bolt with top flick & lower tail.
        ctrl = np.array([
            [-0.22, -1.42],
            [0.36, -0.58],
            [-0.30, 0.14],
            [0.32, 0.84],
            [-0.22, 1.38],
        ], dtype=np.float64)
        # Sharp zig with slight curved bottom tail.
        pts = _transform_local(ctrl, cx, cy, sp, rng, style, jitter_sp=0.11, scale_x_range=(0.82, 1.22), scale_y_range=(0.86, 1.18))
        # Slightly round only the final hook while keeping upper corners brisk.
        upper = pts[:4]
        tail = _eval_quad(pts[3], pts[3] + np.array([-0.38 * sp, 0.18 * sp]), pts[4], n=10)
        full = np.vstack((upper, tail[1:]))
        draw_calligraphic_polyline(canvas, full, color_rgb, base_w * 1.15, style, kind="rest")

    elif archetype == "backwards_7_hook":
        # Tall slanted backwards-7 / reversed-r stem with bottom curved hook.
        top_flick = np.array([
            [-0.36, -1.25],
            [0.24, -1.36],
            [0.02, -0.35],
            [-0.18, 0.72],
            [0.28, 1.28],
        ], dtype=np.float64)
        pts = _transform_local(top_flick, cx, cy, sp, rng, style, jitter_sp=0.10, scale_x_range=(0.80, 1.20), scale_y_range=(0.88, 1.16))
        # Draw top bar + slanted spine as one brisk stroke, lower curl as smooth curve.
        spine = np.vstack((pts[:2], _eval_quad(pts[1], pts[2], pts[3], n=12)[1:]))
        hook = _eval_quad(pts[3], pts[3] + np.array([0.05 * sp, 0.55 * sp]), pts[4], n=10)
        draw_calligraphic_polyline(canvas, np.vstack((spine, hook[1:])), color_rgb, base_w * 1.1, style, kind="rest")

    elif archetype == "cursive_3":
        # Flowing handwritten '3' / double-belly cursive pause.
        ctrl = np.array([
            [-0.28, -1.32],
            [0.36, -1.10],
            [0.32, -0.45],
            [-0.16, -0.05],
            [0.38, 0.38],
            [0.22, 1.12],
            [-0.32, 1.32],
        ], dtype=np.float64)
        pts = _transform_local(ctrl, cx, cy, sp, rng, style, jitter_sp=0.10, scale_x_range=(0.84, 1.20), scale_y_range=(0.85, 1.18))
        curve = _eval_catmull_rom(pts, samples_per_seg=8)
        draw_calligraphic_polyline(canvas, curve, color_rgb, base_w * 1.05, style, kind="rest")

    elif archetype == "hurried_slash_zig":
        # Fast Beethoven/sketch diagonal slash with tiny top nick & bottom kick.
        ctrl = np.array([
            [0.12, -1.40],
            [0.36, -1.02],
            [-0.26, 0.58],
            [0.22, 1.22],
        ], dtype=np.float64)
        pts = _transform_local(ctrl, cx, cy, sp, rng, style, jitter_sp=0.12, scale_x_range=(0.78, 1.25), scale_y_range=(0.85, 1.20), extra_slant_deg=8.0)
        draw_calligraphic_polyline(canvas, pts, color_rgb, base_w * 1.22, style, kind="rest")

    elif archetype == "baroque_c_hook":
        # Classic engraved-inspired hand: upper zig into pronounced open C-belly.
        upper_ctrl = np.array([
            [-0.16, -1.48],
            [0.34, -0.78],
            [-0.22, -0.04],
            [0.28, 0.56],
        ], dtype=np.float64)
        lower_ctrl = np.array([
            [0.28, 0.56],
            [-0.24, 0.36],
            [-0.38, 0.92],
            [0.08, 1.44],
        ], dtype=np.float64)
        all_ctrl = np.vstack((upper_ctrl, lower_ctrl[1:]))
        pts = _transform_local(all_ctrl, cx, cy, sp, rng, style, jitter_sp=0.09, scale_x_range=(0.85, 1.18), scale_y_range=(0.88, 1.16))
        # Upper zig + smooth C-hook at bottom.
        upper = pts[:4]
        lower = _eval_cubic(pts[3], pts[4], pts[5], pts[6], n=16)
        draw_calligraphic_polyline(canvas, upper, color_rgb, base_w * 1.25, style, kind="rest")
        draw_calligraphic_polyline(canvas, lower, color_rgb, base_w * 1.0, style, kind="rest")

    else:  # "calligraphic_s_curl"
        # Sweeping calligraphic S-curve with top serif flick and graceful bottom curl.
        ctrl = np.array([
            [-0.10, -1.52],
            [0.34, -0.95],
            [0.02, -0.22],
            [-0.30, 0.38],
            [0.22, 0.92],
            [-0.28, 1.42],
        ], dtype=np.float64)
        pts = _transform_local(ctrl, cx, cy, sp, rng, style, jitter_sp=0.10, scale_x_range=(0.84, 1.20), scale_y_range=(0.86, 1.18))
        curve = _eval_catmull_rom(pts, samples_per_seg=9)
        draw_calligraphic_polyline(canvas, curve, color_rgb, base_w * 1.12, style, kind="rest")


def _draw_flagged_rest(
    canvas: StrokeCanvas,
    n_hooks: int,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw an eighth (1), 16th (2), or 32nd (3) rest with 3 polymorphic archetypes."""
    archetype = _choose_weighted(
        EIGHTH_REST_ARCHETYPES,
        style.get("eighth_rest_archetype_weights"),
        rng,
        default_weights=(0.48, 0.28, 0.24),
    )
    base_w = max(1.0, sp * float(style.get("glyph_stroke_ratio", 0.19)))
    slant_deg = float(style.get("slant_deg", 0.0)) + float(rng.normal(8.0, 5.0))
    shear = math.tan(math.radians(float(np.clip(slant_deg, -20.0, 26.0))))

    top_y = cy - (0.55 + 0.35 * (n_hooks - 1)) * sp
    bot_y = cy + (1.15 + 0.25 * (n_hooks - 1)) * sp
    stem_top = np.array([cx + 0.28 * sp + rng.normal(0, 0.06 * sp), top_y + rng.normal(0, 0.06 * sp)])
    stem_bot = np.array([
        stem_top[0] - (bot_y - top_y) * (0.28 + 0.25 * shear) + rng.normal(0, 0.08 * sp),
        bot_y + rng.normal(0, 0.06 * sp),
    ])
    draw_calligraphic_polyline(canvas, np.vstack((stem_top, stem_bot)), color_rgb, base_w * 1.05, style, kind="rest")

    # Attach n_hooks along the slanted stem.
    for k in range(n_hooks):
        frac = 0.06 + k * (0.34 if n_hooks == 2 else 0.26)
        attach = stem_top * (1.0 - frac) + stem_bot * frac
        arm_dx = (0.58 + rng.uniform(-0.08, 0.10)) * sp
        arm_dy = (-0.12 + rng.uniform(-0.08, 0.08)) * sp
        head_pt = attach + np.array([-arm_dx, arm_dy])

        if archetype == "dot_head":
            # Sweeping concave arm + filled ink dot at left tip.
            mid = 0.5 * (attach + head_pt) + np.array([0.05 * sp, 0.24 * sp + rng.normal(0, 0.05 * sp)])
            arm = _eval_quad(head_pt, mid, attach, n=10)
            draw_calligraphic_polyline(canvas, arm, color_rgb, base_w * 0.9, style, kind="rest")
            dot_r = max(1.1, sp * rng.uniform(0.18, 0.25))
            draw_filled_blob(canvas, head_pt[0], head_pt[1], dot_r, dot_r * rng.uniform(0.85, 1.15), rng.uniform(-30, 30), color_rgb, style, hollow=False)

        elif archetype == "loop_head":
            # Open/closed cursive loop at tip flowing into the arm.
            r = sp * rng.uniform(0.18, 0.26)
            phis = np.linspace(math.pi * 0.2, math.pi * 2.15, 14)
            loop = np.column_stack((head_pt[0] + r * np.cos(phis), head_pt[1] + r * 0.85 * np.sin(phis)))
            mid = 0.5 * (attach + head_pt) + np.array([0.04 * sp, 0.20 * sp])
            arm = _eval_quad(loop[-1], mid, attach, n=8)
            draw_calligraphic_polyline(canvas, np.vstack((loop, arm[1:])), color_rgb, base_w * 0.88, style, kind="rest")

        else:  # "quick_tick_hook"
            # Brisk V/7 tick hook.
            tip = head_pt + np.array([0.05 * sp, -0.18 * sp + rng.normal(0, 0.05 * sp)])
            elbow = head_pt + np.array([0.14 * sp, 0.16 * sp])
            tick = np.vstack((tip, elbow, attach))
            draw_calligraphic_polyline(canvas, tick, color_rgb, base_w * 0.98, style, kind="rest")


def draw_rest(
    canvas: StrokeCanvas,
    rest_type: str,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a handwritten musical rest onto canvas.

    rest_type in {"whole", "half", "quarter", "eighth", "16th", "32nd"}.
    """
    rt = str(rest_type).strip().lower()
    alias_map = {
        "1": "whole",
        "semibreve": "whole",
        "2": "half",
        "minim": "half",
        "4": "quarter",
        "crotchet": "quarter",
        "8": "eighth",
        "quaver": "eighth",
        "16": "16th",
        "semiquaver": "16th",
        "32": "32nd",
        "demisemiquaver": "32nd",
    }
    rt = alias_map.get(rt, rt)

    if rt in ("whole", "half"):
        # Slightly uneven hand-drawn ink block hanging below (whole) or sitting above (half) the line.
        w = sp * float(rng.uniform(0.92, 1.18))
        h = sp * float(rng.uniform(0.38, 0.52))
        y_center = cy + 0.24 * sp if rt == "whole" else cy - 0.24 * sp
        y_center += float(rng.normal(0.0, 0.04 * sp))
        x_center = cx + float(rng.normal(0.0, 0.04 * sp))
        tilt = float(rng.normal(0.0, 3.5))
        dx = 0.5 * w * math.cos(math.radians(tilt))
        dy = 0.5 * w * math.sin(math.radians(tilt))
        pts = np.array([
            [x_center - dx, y_center - dy],
            [x_center, y_center + float(rng.normal(0.0, 0.04 * sp))],
            [x_center + dx, y_center + dy],
        ], dtype=np.float64)
        local_style = dict(style)
        local_style["taper_start"] = 0.04
        local_style["taper_end"] = 0.04
        local_style["nib_min_ratio"] = max(_get_style_min_ratio(style), 0.78)
        draw_calligraphic_polyline(canvas, pts, color_rgb, h, local_style, kind="beam")
        return

    if rt == "quarter":
        forced = style.get("force_rest_archetype")
        if forced in QUARTER_REST_ARCHETYPES:
            arch = str(forced)
        else:
            arch = _choose_weighted(QUARTER_REST_ARCHETYPES, style.get("rest_archetype_weights"), rng)
        _draw_quarter_rest_archetype(canvas, arch, cx, cy, sp, color_rgb, style, rng)
        return

    if rt == "eighth":
        _draw_flagged_rest(canvas, 1, cx, cy, sp, color_rgb, style, rng)
        return
    if rt == "16th":
        _draw_flagged_rest(canvas, 2, cx, cy, sp, color_rgb, style, rng)
        return
    if rt == "32nd":
        _draw_flagged_rest(canvas, 3, cx, cy, sp, color_rgb, style, rng)
        return

    # Fallback: draw quarter rest if unknown string.
    arch = _choose_weighted(QUARTER_REST_ARCHETYPES, style.get("rest_archetype_weights"), rng)
    _draw_quarter_rest_archetype(canvas, arch, cx, cy, sp, color_rgb, style, rng)


def _get_style_min_ratio(style: dict) -> float:
    for k in ("nib_min_ratio", "min_width_ratio", "min_ratio"):
        if k in style:
            return float(style[k])
    return 0.35


# ---------------------------------------------------------------------------
# Noteheads
# ---------------------------------------------------------------------------


def draw_notehead(
    canvas: StrokeCanvas,
    duration_kind: str,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a handwritten notehead ('filled', 'half', or 'whole').

    Supports 'oval', 'slash', 'teardrop', and 'compact' archetypes with
    tilt/size jitter and occasional slightly unclosed hollow loops.
    """
    dk = str(duration_kind).strip().lower()
    if dk in ("quarter", "eighth", "16th", "32nd", "black", "solid"):
        dk = "filled"
    elif dk in ("minim", "2"):
        dk = "half"
    elif dk in ("semibreve", "1"):
        dk = "whole"

    forced = style.get("notehead_archetype")
    if forced in NOTEHEAD_ARCHETYPES:
        arch = str(forced)
    else:
        arch = _choose_weighted(
            NOTEHEAD_ARCHETYPES,
            style.get("notehead_archetype_weights"),
            rng,
            default_weights=(0.48, 0.18, 0.20, 0.14),
        )

    size_j = float(rng.uniform(0.90, 1.12)) * float(style.get("notehead_scale", 1.0))
    base_angle = float(style.get("notehead_angle_deg", -30.0))
    ox = float(rng.normal(0.0, 0.03 * sp))
    oy = float(rng.normal(0.0, 0.025 * sp))
    nx, ny = cx + ox, cy + oy

    if dk == "whole":
        rx = sp * 0.72 * size_j * float(rng.uniform(0.94, 1.08))
        ry = sp * 0.48 * size_j * float(rng.uniform(0.92, 1.06))
        angle = float(rng.normal(base_angle * 0.4, 6.0))
        open_prob = float(style.get("hollow_open_prob", 0.30))
        gap = float(rng.uniform(0.05, 0.15)) if rng.random() < open_prob else 0.0
        draw_filled_blob(canvas, nx, ny, rx, ry, angle, color_rgb, style, hollow=True, open_gap=gap)
        return

    if dk == "half":
        if arch == "slash":
            # Brisk open diamond/slash loop.
            rx = sp * 0.66 * size_j
            ry = sp * 0.36 * size_j
            angle = float(rng.normal(base_angle - 8.0, 6.0))
        elif arch == "compact":
            rx = sp * 0.56 * size_j
            ry = sp * 0.44 * size_j
            angle = float(rng.normal(base_angle + 6.0, 7.0))
        else:
            rx = sp * 0.64 * size_j
            ry = sp * 0.46 * size_j
            angle = float(rng.normal(base_angle, 6.5))
        open_prob = float(style.get("hollow_open_prob", 0.38))
        gap = float(rng.uniform(0.06, 0.18)) if rng.random() < open_prob else 0.0
        draw_filled_blob(canvas, nx, ny, rx, ry, angle, color_rgb, style, hollow=True, open_gap=gap)
        return

    # Filled notehead ("filled")
    if arch == "slash":
        # Beethoven / hurried jazz dash-slash notehead!
        angle = math.radians(float(rng.normal(base_angle - 8.0, 8.0)))
        half_len = sp * 0.60 * size_j
        dx = half_len * math.cos(angle)
        dy = half_len * math.sin(angle)
        mid_off = float(rng.normal(0.0, 0.06 * sp))
        pts = np.array([
            [nx - dx, ny - dy],
            [nx - mid_off * math.sin(angle), ny + mid_off * math.cos(angle)],
            [nx + dx, ny + dy],
        ], dtype=np.float64)
        local_style = dict(style)
        local_style["taper_start"] = 0.16
        local_style["taper_end"] = 0.20
        local_style["nib_min_ratio"] = max(_get_style_min_ratio(style), 0.65)
        draw_calligraphic_polyline(canvas, pts, color_rgb, sp * 0.56 * size_j, local_style, kind="notehead_slash")
        return

    if arch == "teardrop":
        # Asymmetric tapered teardrop: main blob + slight upper-right or lower-left flick.
        rx = sp * 0.60 * size_j
        ry = sp * 0.43 * size_j
        angle = float(rng.normal(base_angle - 3.0, 6.0))
        draw_filled_blob(canvas, nx, ny, rx, ry, angle, color_rgb, style, hollow=False)
        rad = math.radians(angle)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        tip = np.array([nx + sign * rx * 1.18 * math.cos(rad), ny + sign * rx * 1.18 * math.sin(rad)])
        base = np.array([nx + sign * rx * 0.4 * math.cos(rad), ny + sign * rx * 0.4 * math.sin(rad)])
        draw_calligraphic_polyline(canvas, np.vstack((base, tip)), color_rgb, sp * 0.24 * size_j, style, kind="notehead_flick")
        return

    if arch == "compact":
        rx = sp * 0.50 * size_j * float(rng.uniform(0.92, 1.08))
        ry = sp * 0.43 * size_j * float(rng.uniform(0.92, 1.08))
        angle = float(rng.normal(base_angle + 8.0, 9.0))
        draw_filled_blob(canvas, nx, ny, rx, ry, angle, color_rgb, style, hollow=False)
        return

    # Default "oval"
    rx = sp * 0.62 * size_j * float(rng.uniform(0.93, 1.08))
    ry = sp * 0.45 * size_j * float(rng.uniform(0.92, 1.08))
    angle = float(rng.normal(base_angle, 6.5))
    draw_filled_blob(canvas, nx, ny, rx, ry, angle, color_rgb, style, hollow=False)


# ---------------------------------------------------------------------------
# Stems, Flags & Beams
# ---------------------------------------------------------------------------


def draw_stem_and_flag(
    canvas: StrokeCanvas,
    x: float,
    y0: float,
    y1: float,
    sp: float,
    flags: int,
    stem_up: bool,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a slightly slanted/bowed stem from (x, y0) to (x, y1) and curved handwritten flags."""
    stem_w = max(0.9, sp * float(style.get("stem_width_ratio", 0.135)))
    # Attachment jitter at notehead (y0) and slight lean/overshoot at tip (y1).
    ax = x + float(rng.normal(0.0, 0.05 * sp))
    ay = y0 + float(rng.normal(0.0, 0.06 * sp))
    slant_deg = float(style.get("slant_deg", 0.0)) + float(rng.normal(0.0, 2.2))
    dy = y1 - y0
    tip_x = ax - dy * math.tan(math.radians(float(np.clip(slant_deg, -12.0, 12.0)))) * 0.35 + float(rng.normal(0.0, 0.06 * sp))
    tip_y = y1 + (1.0 if dy >= 0 else -1.0) * float(rng.uniform(-0.04, 0.12)) * sp

    mid_x = 0.5 * (ax + tip_x) + float(rng.normal(0.0, 0.06 * sp))
    mid_y = 0.5 * (ay + tip_y)
    stem_pts = _eval_quad(np.array([ax, ay]), np.array([mid_x, mid_y]), np.array([tip_x, tip_y]), n=12)
    draw_calligraphic_polyline(canvas, stem_pts, color_rgb, stem_w, style, kind="stem")

    n_flags = max(0, int(flags))
    if n_flags == 0:
        return

    # Direction toward notehead along stem.
    toward_head = 1.0 if stem_up else -1.0
    flag_spacing = sp * float(rng.uniform(0.62, 0.78))
    flag_style = int(rng.integers(0, 3))  # 0: S-swoosh, 1: brisk diagonal flick, 2: curled hook
    flag_w = max(1.1, sp * float(style.get("flag_width_ratio", 0.22)))

    for i in range(n_flags):
        fy = tip_y + toward_head * (i * flag_spacing + float(rng.uniform(-0.04, 0.06)) * sp)
        fx = tip_x + float(rng.normal(0.0, 0.04 * sp))
        fw = sp * float(rng.uniform(0.85, 1.25))
        fh = sp * float(rng.uniform(1.15, 1.65)) * toward_head

        if flag_style == 0:
            # Classic calligraphic S-swoosh.
            p0 = np.array([fx, fy])
            p1 = np.array([fx + fw * 0.72, fy + fh * 0.28])
            p2 = np.array([fx + fw * 1.05, fy + fh * 0.75])
            p3 = np.array([fx + fw * 0.32, fy + fh * 1.15])
            fpts = _eval_cubic(p0, p1, p2, p3, n=14)
        elif flag_style == 1:
            # Brisk hurried diagonal flick.
            p0 = np.array([fx, fy])
            p1 = np.array([fx + fw * 0.65, fy + fh * 0.45])
            p2 = np.array([fx + fw * 0.95, fy + fh * 0.95])
            fpts = _eval_quad(p0, p1, p2, n=10)
        else:
            # Curled hook returning slightly inward.
            p0 = np.array([fx, fy])
            p1 = np.array([fx + fw * 0.95, fy + fh * 0.18])
            p2 = np.array([fx + fw * 0.85, fy + fh * 0.92])
            p3 = np.array([fx + fw * 0.15, fy + fh * 1.02])
            fpts = _eval_cubic(p0, p1, p2, p3, n=14)

        draw_calligraphic_polyline(canvas, fpts, color_rgb, flag_w, style, kind="flag")


def draw_beam(
    canvas: StrokeCanvas,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a thick variable-pressure hand-drawn beam with slight bow and overshoot."""
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-3:
        return
    ux, uy = dx / length, dy / length
    over0 = float(rng.uniform(0.02, 0.14)) * sp
    over1 = float(rng.uniform(0.02, 0.14)) * sp
    p0 = np.array([x0 - ux * over0, y0 - uy * over0])
    p2 = np.array([x1 + ux * over1, y1 + uy * over1])
    # Subtle perpendicular bow.
    bow = float(rng.normal(0.0, 0.08 * sp))
    p1 = 0.5 * (p0 + p2) + np.array([-uy * bow, ux * bow])
    pts = _eval_quad(p0, p1, p2, n=max(10, int(length / 8.0)))

    beam_w = max(1.8, sp * float(style.get("beam_width_ratio", 0.46)) * float(rng.uniform(0.92, 1.08)))
    local_style = dict(style)
    local_style["taper_start"] = 0.05
    local_style["taper_end"] = 0.05
    draw_calligraphic_polyline(canvas, pts, color_rgb, beam_w, local_style, kind="beam")


# ---------------------------------------------------------------------------
# Clefs
# ---------------------------------------------------------------------------


def draw_clef(
    canvas: StrokeCanvas,
    clef_name: str,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a handwritten clef ('G-2', 'F-4', 'C-3', 'C-4') with ornate/simplified/hurried forms."""
    name = str(clef_name).strip().upper()
    if name in ("G", "TREBLE", "G2"):
        name = "G-2"
    elif name in ("F", "BASS", "F4"):
        name = "F-4"
    elif name in ("C", "ALTO", "C3"):
        name = "C-3"
    elif name in ("TENOR", "C4"):
        name = "C-4"

    if "clef_ornate" in style:
        ornate = float(np.clip(style["clef_ornate"], 0.0, 1.0))
        default_w = (0.05 + 0.80 * ornate, 0.25, 0.05 + 0.80 * (1.0 - ornate))
    else:
        hurried = float(style.get("hurriedness", 0.25))
        default_w = (max(0.1, 0.55 - 0.4 * hurried), 0.30, max(0.1, 0.15 + 0.4 * hurried))
    arch = _choose_weighted(CLEF_ARCHETYPES, style.get("clef_archetype_weights"), rng, default_weights=default_w)
    base_w = max(1.1, sp * float(style.get("clef_stroke_ratio", 0.19)))

    if name == "G-2":
        if arch == "hurried":
            # Brisk single-sweep spiral + quick vertical slash.
            spiral = np.array([
                [0.08, 0.15],
                [0.52, -0.05],
                [0.45, 0.62],
                [-0.38, 0.55],
                [-0.55, -0.25],
                [-0.08, -1.25],
                [0.42, -2.25],
                [0.08, -2.85],
                [-0.22, -2.10],
                [-0.05, 0.10],
                [0.05, 1.85],
                [-0.45, 2.20],
            ], dtype=np.float64)
            pts = _transform_local(spiral, cx, cy, sp, rng, style, jitter_sp=0.10)
            draw_calligraphic_polyline(canvas, _eval_catmull_rom(pts, 7), color_rgb, base_w * 1.05, style, kind="clef")
        else:
            # Ornate or simplified G-clef: graceful inner G-curl + top loop + descending tail with bottom dot.
            curl = np.array([
                [-0.05, 0.12],
                [0.42, -0.18],
                [0.64, 0.35],
                [0.24, 0.88],
                [-0.48, 0.68],
                [-0.68, -0.12],
                [-0.20, -1.12],
                [0.44, -2.05],
                [0.22, -2.95],
                [-0.24, -2.35],
                [-0.10, -0.95],
                [0.04, 0.85],
                [0.10, 2.05],
                [-0.26, 2.48],
                [-0.56, 2.16],
            ], dtype=np.float64)
            if arch == "simplified":
                curl = curl[1:]  # slightly more open inner start
            pts = _transform_local(curl, cx, cy, sp, rng, style, jitter_sp=0.08)
            draw_calligraphic_polyline(canvas, _eval_catmull_rom(pts, 8), color_rgb, base_w * (1.12 if arch == "ornate" else 1.0), style, kind="clef")
            if arch == "ornate":
                end_pt = pts[-1]
                r = max(1.1, sp * rng.uniform(0.18, 0.25))
                draw_filled_blob(canvas, end_pt[0], end_pt[1], r, r, 0.0, color_rgb, style, hollow=False)

    elif name == "F-4":
        # Bass clef centered on F-line (cy).
        body = np.array([
            [-0.48, -0.05],
            [-0.36, -0.78],
            [0.28, -0.96],
            [0.62, -0.35],
            [0.42, 0.65],
            [-0.18, 1.55],
            [-0.65, 2.08],
        ], dtype=np.float64)
        if arch == "hurried":
            body = body[1:]
        pts = _transform_local(body, cx, cy, sp, rng, style, jitter_sp=0.09)
        draw_calligraphic_polyline(canvas, _eval_catmull_rom(pts, 8), color_rgb, base_w * 1.18, style, kind="clef")

        # Starting ink ball on F-line (ornate/simplified).
        if arch != "hurried":
            r0 = max(1.3, sp * rng.uniform(0.24, 0.32))
            draw_filled_blob(canvas, pts[0, 0], pts[0, 1], r0, r0 * 0.95, 0.0, color_rgb, style, hollow=False)

        # Two hand-dabbed dots straddling the F-line.
        dot_x = cx + sp * rng.uniform(0.95, 1.18)
        for sign in (-1.0, 1.0):
            dx = dot_x + float(rng.normal(0.0, 0.06 * sp))
            dy = cy + sign * sp * rng.uniform(0.42, 0.54)
            dr = max(1.1, sp * rng.uniform(0.17, 0.24))
            draw_filled_blob(canvas, dx, dy, dr, dr * rng.uniform(0.85, 1.15), rng.uniform(-25, 25), color_rgb, style, hollow=False)

    else:
        # Alto (C-3) or Tenor (C-4) C-clef centered at cy (middle C line).
        # Left vertical bar(s):
        x_bar = cx - sp * 0.62
        y_top = cy - sp * 1.85
        y_bot = cy + sp * 1.85
        bar1 = np.array([[x_bar, y_top], [x_bar + rng.normal(0, 0.06 * sp), y_bot]])
        draw_calligraphic_polyline(canvas, bar1, color_rgb, base_w * 1.7, style, kind="clef")
        if arch != "hurried":
            x_bar2 = x_bar + sp * rng.uniform(0.32, 0.42)
            bar2 = np.array([[x_bar2, y_top + 0.05 * sp], [x_bar2 + rng.normal(0, 0.05 * sp), y_bot - 0.05 * sp]])
            draw_calligraphic_polyline(canvas, bar2, color_rgb, base_w * 0.85, style, kind="clef")

        # Upper and lower opposing C/3 brackets meeting at cy.
        upper = np.array([
            [-0.12, -1.78],
            [0.52, -1.68],
            [0.62, -0.82],
            [-0.08, -0.04],
        ], dtype=np.float64)
        lower = np.array([
            [-0.08, 0.04],
            [0.62, 0.82],
            [0.52, 1.68],
            [-0.12, 1.78],
        ], dtype=np.float64)
        u_pts = _transform_local(upper, cx, cy, sp, rng, style, jitter_sp=0.08)
        l_pts = _transform_local(lower, cx, cy, sp, rng, style, jitter_sp=0.08)
        draw_calligraphic_polyline(canvas, _eval_catmull_rom(u_pts, 8), color_rgb, base_w * 1.15, style, kind="clef")
        draw_calligraphic_polyline(canvas, _eval_catmull_rom(l_pts, 8), color_rgb, base_w * 1.15, style, kind="clef")


# ---------------------------------------------------------------------------
# Accidentals
# ---------------------------------------------------------------------------


def draw_accidental(
    canvas: StrokeCanvas,
    acc: str,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a handwritten accidental ('sharp', 'flat', 'natural')."""
    kind = str(acc).strip().lower()
    if kind in ("#", "is", "s"):
        kind = "sharp"
    elif kind in ("b", "es", "f"):
        kind = "flat"
    elif kind in ("n", "nat", "♮"):
        kind = "natural"

    base_w = max(0.95, sp * float(style.get("acc_stroke_ratio", 0.145)))
    slant_deg = float(style.get("slant_deg", 0.0)) + float(rng.normal(4.0, 4.0))
    shear = math.tan(math.radians(float(np.clip(slant_deg, -20.0, 22.0))))

    if kind == "sharp":
        # Two slanted verticals + two thicker upward-tilted crossbars.
        dx_v = sp * rng.uniform(0.22, 0.30)
        h_v = sp * rng.uniform(1.05, 1.32)
        for sign in (-1.0, 1.0):
            vx = cx + sign * dx_v + float(rng.normal(0.0, 0.04 * sp))
            top = np.array([vx + h_v * shear * 0.5, cy - h_v - sign * 0.08 * sp + rng.normal(0, 0.06 * sp)])
            bot = np.array([vx - h_v * shear * 0.5, cy + h_v - sign * 0.08 * sp + rng.normal(0, 0.06 * sp)])
            draw_calligraphic_polyline(canvas, np.vstack((top, bot)), color_rgb, base_w * 0.95, style, kind="accidental")

        w_h = sp * rng.uniform(0.52, 0.68)
        dy_h = sp * rng.uniform(0.32, 0.44)
        tilt = sp * rng.uniform(0.18, 0.34)
        for sign in (-1.0, 1.0):
            hy = cy + sign * dy_h + float(rng.normal(0.0, 0.04 * sp))
            left = np.array([cx - w_h + rng.normal(0, 0.05 * sp), hy + tilt * 0.5])
            right = np.array([cx + w_h + rng.normal(0, 0.05 * sp), hy - tilt * 0.5])
            draw_calligraphic_polyline(canvas, np.vstack((left, right)), color_rgb, base_w * 1.65, style, kind="accidental")

    elif kind == "flat":
        # Tall slanted downstroke + rounded or pointed lower belly loop.
        top_y = cy - sp * rng.uniform(1.45, 1.85)
        bot_y = cy + sp * rng.uniform(0.50, 0.70)
        sx0 = cx - sp * 0.24
        top = np.array([sx0 + (cy - top_y) * shear * 0.4, top_y])
        bot = np.array([sx0 - (bot_y - cy) * shear * 0.4, bot_y])
        draw_calligraphic_polyline(canvas, np.vstack((top, bot)), color_rgb, base_w * 1.05, style, kind="accidental")

        pointed = rng.random() < 0.45
        bw = sp * rng.uniform(0.54, 0.72)
        if pointed:
            belly = np.array([
                [bot[0] + 0.03 * sp, cy - 0.28 * sp + rng.normal(0, 0.06 * sp)],
                [bot[0] + bw, cy - 0.38 * sp + rng.normal(0, 0.06 * sp)],
                [bot[0] + bw * 0.75, cy + 0.15 * sp],
                [bot[0] + rng.uniform(-0.04, 0.10) * sp, bot[1] - rng.uniform(0.0, 0.10) * sp],
            ], dtype=np.float64)
            draw_calligraphic_polyline(canvas, _eval_catmull_rom(belly, 7), color_rgb, base_w * 1.15, style, kind="accidental")
        else:
            p0 = np.array([bot[0], cy - 0.25 * sp])
            p1 = np.array([bot[0] + bw * 1.1, cy - 0.55 * sp])
            p2 = np.array([bot[0] + bw * 1.05, cy + 0.35 * sp])
            p3 = np.array([bot[0] + rng.uniform(-0.05, 0.12) * sp, bot[1]])
            draw_calligraphic_polyline(canvas, _eval_cubic(p0, p1, p2, p3, 14), color_rgb, base_w * 1.15, style, kind="accidental")

    else:  # "natural"
        # Quick 2-stroke L + inverted-L handwritten natural sign.
        dx = sp * rng.uniform(0.24, 0.34)
        dy_box = sp * rng.uniform(0.32, 0.44)
        tilt = sp * rng.uniform(0.14, 0.26)
        # Stroke 1: left stem down + lower crossbar right.
        s1 = np.array([
            [cx - dx + 0.2 * shear * sp, cy - sp * rng.uniform(1.15, 1.45)],
            [cx - dx - 0.1 * shear * sp, cy + dy_box + tilt * 0.4],
            [cx + dx + 0.06 * sp, cy + dy_box - tilt * 0.6],
        ], dtype=np.float64)
        # Stroke 2: upper crossbar left->right + right stem down.
        s2 = np.array([
            [cx - dx - 0.06 * sp, cy - dy_box + tilt * 0.5],
            [cx + dx + 0.05 * shear * sp, cy - dy_box - tilt * 0.5],
            [cx + dx - 0.2 * shear * sp, cy + sp * rng.uniform(1.15, 1.45)],
        ], dtype=np.float64)
        draw_calligraphic_polyline(canvas, s1, color_rgb, base_w * 1.1, style, kind="accidental")
        draw_calligraphic_polyline(canvas, s2, color_rgb, base_w * 1.1, style, kind="accidental")


# ---------------------------------------------------------------------------
# Time Signatures & Handwritten Numerals
# ---------------------------------------------------------------------------


def _draw_handwritten_digit(
    canvas: StrokeCanvas,
    digit: str,
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a single hand-drawn numeral ('0'..'9', 'C', 'cut') centered at (cx, cy)."""
    w = max(1.1, sp * float(style.get("timesig_stroke_ratio", 0.21)))

    def _draw_norm(ctrl: np.ndarray, smooth: bool = True, width_mult: float = 1.0) -> None:
        pts = _transform_local(ctrl, cx, cy, sp, rng, style, jitter_sp=0.07, scale_x_range=(0.86, 1.14), scale_y_range=(0.88, 1.12))
        path = _eval_catmull_rom(pts, 7) if (smooth and len(pts) >= 3) else pts
        draw_calligraphic_polyline(canvas, path, color_rgb, w * width_mult, style, kind="timesig")

    if digit == "0":
        phis = np.linspace(-0.2, 2.0 * math.pi + 0.15, 18)
        ctrl = np.column_stack((0.52 * np.cos(phis), 0.85 * np.sin(phis)))
        _draw_norm(ctrl, smooth=False)
    elif digit == "1":
        _draw_norm(np.array([[-0.22, -0.58], [0.05, -0.88], [0.02, 0.88]]), smooth=False)
        if rng.random() < 0.5:
            _draw_norm(np.array([[-0.26, 0.88], [0.28, 0.86]]), smooth=False, width_mult=0.85)
    elif digit == "2":
        ctrl = np.array([[-0.42, -0.52], [-0.08, -0.88], [0.42, -0.68], [0.36, -0.10], [-0.46, 0.84], [0.48, 0.82]])
        _draw_norm(ctrl, smooth=True)
    elif digit == "3":
        ctrl = np.array([[-0.38, -0.68], [0.10, -0.90], [0.42, -0.45], [-0.05, -0.02], [0.46, 0.42], [0.12, 0.90], [-0.44, 0.68]])
        _draw_norm(ctrl, smooth=True)
    elif digit == "4":
        if rng.random() < 0.55:
            # Closed classic 4.
            _draw_norm(np.array([[0.22, -0.88], [-0.46, 0.24], [0.48, 0.20]]), smooth=False)
            _draw_norm(np.array([[0.22, -0.86], [0.18, 0.88]]), smooth=False)
        else:
            # Open handwritten 4.
            _draw_norm(np.array([[-0.32, -0.84], [-0.40, 0.18], [0.46, 0.16]]), smooth=False)
            _draw_norm(np.array([[0.26, -0.86], [0.20, 0.88]]), smooth=False)
    elif digit == "5":
        _draw_norm(np.array([[0.42, -0.84], [-0.34, -0.82], [-0.38, -0.10]]), smooth=False)
        _draw_norm(np.array([[-0.38, -0.10], [0.26, -0.16], [0.46, 0.40], [0.06, 0.88], [-0.42, 0.68]]), smooth=True)
    elif digit == "6":
        ctrl = np.array([[0.32, -0.86], [-0.28, -0.45], [-0.46, 0.35], [-0.02, 0.88], [0.44, 0.46], [0.16, -0.06], [-0.38, 0.24]])
        _draw_norm(ctrl, smooth=True)
    elif digit == "7":
        _draw_norm(np.array([[-0.44, -0.82], [0.44, -0.86], [-0.14, 0.88]]), smooth=False)
        if rng.random() < 0.35:
            _draw_norm(np.array([[-0.22, 0.02], [0.34, -0.02]]), smooth=False, width_mult=0.8)
    elif digit == "8":
        ctrl = np.array([
            [0.02, -0.86], [-0.40, -0.48], [0.02, -0.04], [0.44, 0.44],
            [0.00, 0.88], [-0.44, 0.46], [-0.02, -0.02], [0.40, -0.48], [0.02, -0.86],
        ])
        _draw_norm(ctrl, smooth=True)
    elif digit == "9":
        ctrl = np.array([[0.34, -0.22], [0.02, -0.88], [-0.44, -0.46], [-0.02, 0.04], [0.38, -0.28], [0.22, 0.48], [-0.22, 0.88]])
        _draw_norm(ctrl, smooth=True)
    elif digit in ("C", "CUT"):
        ctrl = np.array([[0.48, -0.68], [0.02, -0.96], [-0.52, -0.42], [-0.52, 0.42], [0.04, 0.96], [0.50, 0.66]])
        _draw_norm(ctrl, smooth=True, width_mult=1.15)
        if digit == "CUT":
            _draw_norm(np.array([[0.04, -1.35], [-0.04, 1.35]]), smooth=False, width_mult=0.95)


def draw_time_sig(
    canvas: StrokeCanvas,
    num: Union[int, str],
    den: Union[int, str],
    cx: float,
    cy: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a hand-drawn time signature (e.g. 2/4, 3/4, 4/4, 6/8, 9/8, 12/8, 'C', 'cut')."""
    num_s = str(num).strip().upper()
    den_s = str(den).strip().upper()

    if num_s in ("C", "COMMON"):
        _draw_handwritten_digit(canvas, "C", cx, cy, sp * 1.1, color_rgb, style, rng)
        return
    if num_s in ("CUT", "C|", "ALLA_BREVE") or den_s in ("CUT", "0"):
        _draw_handwritten_digit(canvas, "CUT", cx, cy, sp * 1.1, color_rgb, style, rng)
        return

    def _draw_number_str(s: str, y_center: float) -> None:
        digits = [ch for ch in s if ch.isdigit()]
        if not digits:
            return
        if len(digits) == 1:
            _draw_handwritten_digit(canvas, digits[0], cx, y_center, sp, color_rgb, style, rng)
        else:
            step = sp * 0.88
            start_x = cx - 0.5 * step * (len(digits) - 1)
            for i, ch in enumerate(digits):
                _draw_handwritten_digit(canvas, ch, start_x + i * step, y_center, sp * 0.92, color_rgb, style, rng)

    _draw_number_str(num_s, cy - 1.0 * sp)
    _draw_number_str(den_s, cy + 1.0 * sp)


# ---------------------------------------------------------------------------
# Barlines & Braces
# ---------------------------------------------------------------------------


def draw_barline(
    canvas: StrokeCanvas,
    x: float,
    y_top: float,
    y_bot: float,
    sp: float,
    kind: str,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a handwritten barline ('single', 'double', 'final', 'repeat-right', 'repeat-left')."""
    k = str(kind).strip().lower().replace("_", "-")
    thin_w = max(0.95, sp * float(style.get("barline_width_ratio", 0.145)))
    thick_w = max(2.0, sp * float(style.get("barline_thick_ratio", 0.42)))
    slant_deg = float(style.get("slant_deg", 0.0)) * 0.25 + float(rng.normal(0.0, 1.2))
    shear = math.tan(math.radians(float(np.clip(slant_deg, -8.0, 8.0))))

    def _draw_vbar(bx: float, width: float) -> None:
        max_over = float(style.get("barline_overshoot_sp", 0.16))
        over_t = float(rng.uniform(-0.04, max(0.05, max_over))) * sp
        over_b = float(rng.uniform(-0.04, max(0.05, max_over))) * sp
        yt = y_top - over_t
        yb = y_bot + over_b
        ym = 0.5 * (yt + yb)
        h = yb - yt
        p0 = np.array([bx + 0.5 * h * shear, yt])
        p2 = np.array([bx - 0.5 * h * shear, yb])
        p1 = np.array([bx + float(rng.normal(0.0, 0.07 * sp)), ym])
        pts = _eval_quad(p0, p1, p2, n=max(10, int(abs(h) / 10.0)))
        draw_calligraphic_polyline(canvas, pts, color_rgb, width, style, kind="barline")

    def _draw_repeat_dots(dx: float) -> None:
        ym = 0.5 * (y_top + y_bot)
        for sign in (-1.0, 1.0):
            r = max(1.1, sp * rng.uniform(0.17, 0.23))
            draw_filled_blob(
                canvas,
                dx + float(rng.normal(0.0, 0.04 * sp)),
                ym + sign * 0.5 * sp + float(rng.normal(0.0, 0.04 * sp)),
                r,
                r * rng.uniform(0.88, 1.12),
                rng.uniform(-25, 25),
                color_rgb,
                style,
                hollow=False,
            )

    if k == "double":
        gap = sp * rng.uniform(0.36, 0.48)
        _draw_vbar(x - 0.5 * gap, thin_w)
        _draw_vbar(x + 0.5 * gap, thin_w)
    elif k == "final":
        gap = sp * rng.uniform(0.45, 0.58)
        _draw_vbar(x - gap, thin_w)
        _draw_vbar(x, thick_w)
    elif k == "repeat-right":
        gap = sp * rng.uniform(0.45, 0.58)
        _draw_vbar(x - gap, thin_w)
        _draw_vbar(x, thick_w)
        _draw_repeat_dots(x - gap - sp * rng.uniform(0.42, 0.54))
    elif k == "repeat-left":
        gap = sp * rng.uniform(0.45, 0.58)
        _draw_vbar(x, thick_w)
        _draw_vbar(x + gap, thin_w)
        _draw_repeat_dots(x + gap + sp * rng.uniform(0.42, 0.54))
    else:  # "single"
        _draw_vbar(x, thin_w)


def draw_brace(
    canvas: StrokeCanvas,
    x: float,
    y_top: float,
    y_bot: float,
    sp: float,
    color_rgb: RGBLike,
    style: dict,
    rng: np.random.Generator,
) -> None:
    """Draw a hand-drawn curly brace (or bracket if style['brace_kind'] == 'bracket')."""
    kind = str(style.get("brace_kind", "curly")).strip().lower()
    h = max(sp, abs(y_bot - y_top))
    ym = 0.5 * (y_top + y_bot)
    base_w = max(1.2, sp * float(style.get("brace_stroke_ratio", 0.22)))

    if kind == "bracket":
        bx = x - sp * 0.25
        wing = sp * rng.uniform(0.65, 0.90)
        spine = np.array([[bx, y_top - 0.1 * sp], [bx + rng.normal(0, 0.08 * sp), ym], [bx, y_bot + 0.1 * sp]])
        draw_calligraphic_polyline(canvas, _eval_quad(spine[0], spine[1], spine[2], 16), color_rgb, base_w * 1.4, style, kind="brace")
        top_w = np.array([[bx, y_top - 0.1 * sp], [bx + wing * 0.5, y_top - 0.22 * sp], [bx + wing, y_top - 0.42 * sp]])
        bot_w = np.array([[bx, y_bot + 0.1 * sp], [bx + wing * 0.5, y_bot + 0.22 * sp], [bx + wing, y_bot + 0.42 * sp]])
        draw_calligraphic_polyline(canvas, _eval_quad(*top_w, 10), color_rgb, base_w, style, kind="brace")
        draw_calligraphic_polyline(canvas, _eval_quad(*bot_w, 10), color_rgb, base_w, style, kind="brace")
        return

    # Hand-drawn curly brace: two mirrored sweeping S-curves meeting at a central point.
    bow_w = sp * rng.uniform(0.85, 1.15)
    mid_tip = np.array([x - bow_w * 1.15 + rng.normal(0, 0.08 * sp), ym + rng.normal(0, 0.08 * sp)])

    upper = _eval_cubic(
        np.array([x + rng.normal(0, 0.06 * sp), y_top]),
        np.array([x - bow_w * 1.05, y_top + 0.14 * h]),
        np.array([x + bow_w * 0.20, ym - 0.16 * h]),
        mid_tip,
        n=18,
    )
    lower = _eval_cubic(
        mid_tip,
        np.array([x + bow_w * 0.20, ym + 0.16 * h]),
        np.array([x - bow_w * 1.05, y_bot - 0.14 * h]),
        np.array([x + rng.normal(0, 0.06 * sp), y_bot]),
        n=18,
    )
    draw_calligraphic_polyline(canvas, upper, color_rgb, base_w * 1.15, style, kind="brace")
    draw_calligraphic_polyline(canvas, lower, color_rgb, base_w * 1.15, style, kind="brace")
