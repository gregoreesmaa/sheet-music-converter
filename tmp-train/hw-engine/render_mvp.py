"""Complete MuseScore (.mscx/.mscz) & MusicXML (.musicxml/.mxl) handwritten score renderer.

Integrates `musescore_compat.py`, `styles.py`, `layout_quirks.py`, `stroke_engine.py`,
and `glyph_variants.py` to render full real-world MuseScore notation (polyphony, ties/slurs,
tuplets, articulations, ornaments, grace notes, dynamics, hairpins, tempo, lyrics,
mid-measure clef/key/time changes, 8vb/8va clefs, double/courtesy accidentals,
multi-bar rests, and volta repeat brackets) into any of the 20 handwritten scribe styles,
with Stage-1 Bezier GT JSON and optional Stage-2 cropped-system + `assembly-serialize bekern`
training export.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

# Allow running both as `python -m hw_engine.render_mvp` and directly as a script
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from musescore_compat import (  # noqa: E402
    CLEF_BOTTOM_DIATONIC,
    DURATION_BEATS,
    FLAT_STEPS_BY_CLEF,
    LETTER_TO_DIATONIC,
    NATURAL_SEMITONES,
    SHARP_STEPS_BY_CLEF,
    DynamicEvent,
    HairpinSpan,
    MeasureData,
    MidMeasureChange,
    NoteEvent,
    ParsedScore,
    PartData,
    RestEvent,
    ScoreEvent,
    TempoEvent,
    TieSlurSpan,
    VoltaBracket,
    base_clef_code,
    clef_octave_shift,
    export_score_to_musicxml,
    parse_score,
    resolve_polyphony_and_collisions,
)


# ---------------------------------------------------------------------------
# Page & System Layout
# ---------------------------------------------------------------------------


@dataclass
class StaffLayout:
    """Geometry for a single 5-line staff within a system."""

    part_index: int
    part_name: str
    brace_group: Optional[int]
    x_left: float
    x_right: float
    y_center: float
    sp: float
    ctrl_pts: List[List[float]] = field(default_factory=list)


@dataclass
class SystemLayout:
    """One system (horizontal row of 1..N staves) covering measures [m_start, m_end)."""

    system_index: int
    measure_start: int
    measure_end: int
    x_left: float
    x_right: float
    bar_x_bounds: List[Tuple[float, float]]  # (x0, x1) for each measure in system
    staves: List[StaffLayout] = field(default_factory=list)
    is_blank_prerule: bool = False


def layout_score_pages(
    score: ParsedScore,
    page_w: int = 1650,
    page_h: int = 2200,
    style: Optional[Dict[str, Any]] = None,
    rng: Optional[random.Random] = None,
    max_parts: int = 4,
) -> List[SystemLayout]:
    """Wrap score measures into systems that fit on page 1."""
    style = style or {}
    rng = rng or random.Random(42)

    parts = score.parts[:max_parts]
    n_staves = max(1, len(parts))
    total_measures = score.num_measures

    sp_base = float(style.get("sp", style.get("staff_space", 16.0)))
    if n_staves >= 4:
        sp = min(sp_base, 12.0)
    elif n_staves == 3:
        sp = min(sp_base, 13.5)
    elif n_staves == 2:
        sp = min(sp_base, 15.0)
    else:
        sp = sp_base

    margin_x = float(style.get("margin_x", page_w * 0.065))
    margin_top = float(style.get("margin_top", page_h * 0.075))
    margin_bottom = float(style.get("margin_bottom", page_h * 0.065))
    x_left = margin_x
    x_right = page_w - margin_x

    staff_h = 4.0 * sp
    has_lyrics = any(
        isinstance(ev, NoteEvent) and bool(ev.lyrics)
        for p in parts
        for m in p.measures
        for ev in m.events
    )
    intra_gap = (7.2 if has_lyrics else 6.5) * sp if n_staves > 1 else 0.0
    inter_system_gap = (8.2 if has_lyrics else 7.5) * sp if n_staves == 1 else 6.2 * sp
    system_h = n_staves * staff_h + max(0, n_staves - 1) * intra_gap

    bars_per_sys_default = 4 if n_staves >= 2 else 5
    bar_width_var = float(style.get("bar_width_var", 0.12))
    prerule_prob = float(style.get("blank_prerule_prob", 0.0))

    systems: List[SystemLayout] = []
    y_cursor = margin_top + staff_h * 0.5
    m_cursor = 0
    sys_idx = 0

    while y_cursor + system_h - staff_h * 0.5 <= page_h - margin_bottom:
        if m_cursor < total_measures:
            bars_this_sys = min(
                total_measures - m_cursor,
                bars_per_sys_default
                + (rng.choice([-1, 0, 0, 1]) if total_measures - m_cursor > 5 else 0),
            )
            bars_this_sys = max(1, bars_this_sys)
            m_start = m_cursor
            m_end = m_cursor + bars_this_sys
            m_cursor = m_end

            weights: List[float] = []
            for m_i in range(m_start, m_end):
                max_ev = 4
                for p in parts:
                    if m_i < len(p.measures):
                        max_ev = max(max_ev, len(p.measures[m_i].events))
                density_w = 0.65 + 0.12 * min(max_ev, 16)
                jitter = 1.0 + rng.uniform(-bar_width_var, bar_width_var)
                prefix_bonus = 0.35 if m_i == m_start else 0.0
                weights.append(max(0.4, (density_w + prefix_bonus) * jitter))

            w_sum = sum(weights)
            bar_bounds: List[Tuple[float, float]] = []
            bx = x_left
            avail_w = x_right - x_left
            for w in weights:
                bw = avail_w * (w / w_sum)
                bar_bounds.append((bx, bx + bw))
                bx += bw

            staves: List[StaffLayout] = []
            for s_i, p in enumerate(parts):
                yc = y_cursor + s_i * (staff_h + intra_gap)
                staves.append(
                    StaffLayout(
                        part_index=s_i,
                        part_name=p.name,
                        brace_group=p.brace_group,
                        x_left=x_left,
                        x_right=x_right,
                        y_center=yc,
                        sp=sp,
                    )
                )

            systems.append(
                SystemLayout(
                    system_index=sys_idx,
                    measure_start=m_start,
                    measure_end=m_end,
                    x_left=x_left,
                    x_right=x_right,
                    bar_x_bounds=bar_bounds,
                    staves=staves,
                    is_blank_prerule=False,
                )
            )
            sys_idx += 1
            y_cursor += system_h + inter_system_gap
        else:
            if prerule_prob > 0.0 and rng.random() < prerule_prob:
                staves = [
                    StaffLayout(
                        part_index=0,
                        part_name="blank",
                        brace_group=None,
                        x_left=x_left,
                        x_right=x_right,
                        y_center=y_cursor,
                        sp=sp,
                    )
                ]
                systems.append(
                    SystemLayout(
                        system_index=sys_idx,
                        measure_start=0,
                        measure_end=0,
                        x_left=x_left,
                        x_right=x_right,
                        bar_x_bounds=[],
                        staves=staves,
                        is_blank_prerule=True,
                    )
                )
                sys_idx += 1
                y_cursor += staff_h + inter_system_gap
            else:
                break

    return systems


# ---------------------------------------------------------------------------
# Adaptive Helpers for Engine Calls
# ---------------------------------------------------------------------------


def _eval_cubic_bezier_y(ctrl_pts: Sequence[Sequence[float]], x: float) -> float:
    if not ctrl_pts:
        return 0.0
    x0 = float(ctrl_pts[0][0])
    x3 = float(ctrl_pts[-1][0])
    if abs(x3 - x0) < 1e-6:
        return float(ctrl_pts[0][1])
    t = max(0.0, min(1.0, (x - x0) / (x3 - x0)))
    if len(ctrl_pts) == 4:
        mt = 1.0 - t
        return (
            (mt ** 3) * float(ctrl_pts[0][1])
            + 3.0 * (mt ** 2) * t * float(ctrl_pts[1][1])
            + 3.0 * mt * (t ** 2) * float(ctrl_pts[2][1])
            + (t ** 3) * float(ctrl_pts[3][1])
        )
    return (1.0 - t) * float(ctrl_pts[0][1]) + t * float(ctrl_pts[-1][1])


def _get_staff_curve(
    layout_mod: Any,
    x_left: float,
    x_right: float,
    y_center: float,
    sp: float,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
) -> Tuple[Callable[[Any, int], Any], List[List[float]], float]:
    fn = getattr(layout_mod, "build_curved_staff_y", None)
    if fn is not None:
        res = fn(
            x0=x_left,
            x1=x_right,
            y_center=y_center,
            sp=sp,
            style=style,
            rng=np_rng,
        )
        if isinstance(res, tuple) and len(res) >= 3:
            y_fn, ctrl, sp_out = res[0], res[1], float(res[2])
            ctrl_list = [[float(pt[0]), float(pt[1])] for pt in ctrl]
            return y_fn, ctrl_list, sp_out
        if isinstance(res, tuple) and len(res) == 2:
            y_fn_1, ctrl = res[0], res[1]
            ctrl_list = [[float(pt[0]), float(pt[1])] for pt in ctrl]
            return (lambda x, li=0, f=y_fn_1, s=sp: f(x) + li * s), ctrl_list, sp

    sag = float(style.get("staff_curve_px", 4.0)) * float(np_rng.uniform(-1.0, 1.0))
    tilt = float(style.get("staff_slope_deg", 0.5)) * 4.0 * float(np_rng.uniform(-1.0, 1.0))
    w = x_right - x_left
    ctrl_list = [
        [x_left, y_center - tilt * 0.5],
        [x_left + w / 3.0, y_center + sag - tilt * 0.15],
        [x_left + 2.0 * w / 3.0, y_center + sag + tilt * 0.15],
        [x_right, y_center + tilt * 0.5],
    ]
    return (lambda x, li=0, c=ctrl_list, s=sp: _eval_cubic_bezier_y(c, float(x)) + li * s), ctrl_list, sp


def _compute_beat_x_map(
    layout_mod: Any,
    x_start: float,
    x_end: float,
    sp: float,
    events: List[ScoreEvent],
    extra_beats: Sequence[float],
    style: Dict[str, Any],
    np_rng: np.random.Generator,
) -> Tuple[List[float], Callable[[float], float]]:
    """Compute horizontal x positions aligned by beat_offset across all voices.

    Simultaneous voices at the same beat_offset share the same base x-position,
    while grace notes are tucked immediately left of their following main note.
    Returns (event_xs, beat_to_x_fn).
    """
    n = len(events)
    if x_end <= x_start + 4.0:
        x_end = x_start + 40.0

    # Collect unique non-grace beat positions (quantized to 1/96 beat)
    beat_keys: List[int] = sorted(
        {
            int(round(ev.beat_offset * 96.0))
            for ev in events
            if not (isinstance(ev, NoteEvent) and ev.is_grace)
        }
        | {int(round(b * 96.0)) for b in extra_beats}
    )
    if not beat_keys:
        beat_keys = [0]
    unique_beats = [k / 96.0 for k in beat_keys]

    fn = getattr(layout_mod, "compute_quirky_note_positions", None)
    beat_xs: Optional[List[float]] = None
    if fn is not None and x_end > x_start + 2.0:
        try:
            xs = fn(
                beat_offsets=unique_beats,
                bar_x0=x_start,
                bar_x1=x_end,
                sp=sp,
                style=style,
                rng=np_rng,
            )
            if xs is not None and len(xs) == len(unique_beats):
                beat_xs = [float(x) for x in xs]
        except Exception:
            beat_xs = None

    if beat_xs is None:
        pad_l = (x_end - x_start) * 0.08
        pad_r = (x_end - x_start) * 0.08
        usable = max(8.0, (x_end - x_start) - pad_l - pad_r)
        b_min = unique_beats[0]
        b_span = max(unique_beats[-1] - b_min, 1e-3)
        beat_xs = [
            x_start + pad_l + ((b - b_min) / b_span if len(unique_beats) > 1 else 0.25) * usable
            for b in unique_beats
        ]

    key_to_x = {k: x for k, x in zip(beat_keys, beat_xs)}

    def beat_to_x(b: float) -> float:
        k = int(round(b * 96.0))
        if k in key_to_x:
            return key_to_x[k]
        if len(unique_beats) == 1:
            return beat_xs[0]
        if b <= unique_beats[0]:
            return beat_xs[0]
        if b >= unique_beats[-1]:
            return min(x_end - sp * 0.3, beat_xs[-1] + sp * 0.8)
        # Piecewise linear interpolation
        for i in range(len(unique_beats) - 1):
            if unique_beats[i] <= b <= unique_beats[i + 1]:
                t = (b - unique_beats[i]) / max(1e-6, unique_beats[i + 1] - unique_beats[i])
                return beat_xs[i] + t * (beat_xs[i + 1] - beat_xs[i])
        return beat_xs[-1]

    # Count consecutive grace notes preceding each beat so they space out cleanly
    grace_run_total: Dict[int, int] = {}
    grace_run_pos: Dict[int, int] = {}
    i = 0
    while i < n:
        ev = events[i]
        if isinstance(ev, NoteEvent) and ev.is_grace:
            j = i
            while j < n and isinstance(events[j], NoteEvent) and events[j].is_grace:
                j += 1
            count = j - i
            for k in range(i, j):
                grace_run_total[k] = count
                grace_run_pos[k] = k - i
            i = j
        else:
            i += 1

    ev_xs: List[float] = []
    for idx, ev in enumerate(events):
        base_x = beat_to_x(ev.beat_offset)
        if isinstance(ev, NoteEvent) and ev.is_grace:
            tot = grace_run_total.get(idx, 1)
            pos = grace_run_pos.get(idx, 0)
            # Tuck to the left of the beat
            gx = base_x - sp * (1.05 * (tot - pos))
            ev_xs.append(max(x_start + sp * 0.4, gx))
        else:
            nudge = getattr(ev, "x_nudge_sp", 0.0) * sp
            ev_xs.append(min(x_end - sp * 0.3, base_x + nudge))

    return ev_xs, beat_to_x


# ---------------------------------------------------------------------------
# Local Calligraphic Primitives (Zero Dependency on Subagent 2 Edits)
# ---------------------------------------------------------------------------


def _eval_cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, n: int = 20) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    mt = 1.0 - t
    return (mt ** 3) * p0 + 3.0 * (mt ** 2) * t * p1 + 3.0 * mt * (t ** 2) * p2 + (t ** 3) * p3


def _eval_quad(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 14) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    mt = 1.0 - t
    return (mt ** 2) * p0 + 2.0 * mt * t * p1 + (t ** 2) * p2


def _draw_ledger_lines(
    canvas: Any,
    stroke_mod: Any,
    x: float,
    y_center: float,
    sp: float,
    steps: List[int],
    ink_rgb: Any,
    style: Dict[str, Any],
    rng: Any,
) -> None:
    if not steps:
        return
    min_s = min(steps)
    max_s = max(steps)
    half_w = sp * 0.95
    draw_poly = getattr(stroke_mod, "draw_calligraphic_polyline", None)
    if draw_poly is None:
        return

    ledger_steps: List[int] = []
    if min_s <= -2:
        ledger_steps.extend(range(-2, min_s - 1, -2))
    if max_s >= 10:
        ledger_steps.extend(range(10, max_s + 1, 2))

    w_px = max(1.0, sp * 0.09 * float(style.get("stroke_scale", 1.0)))
    for ls in ledger_steps:
        ly = y_center - (ls - 4) * (sp * 0.5)
        pts = np.asarray(
            [
                [x - half_w, ly + rng.uniform(-0.5, 0.5)],
                [x, ly + rng.uniform(-0.4, 0.4)],
                [x + half_w, ly + rng.uniform(-0.5, 0.5)],
            ],
            dtype=np.float32,
        )
        draw_poly(
            canvas=canvas,
            pts=pts,
            color_rgb=ink_rgb,
            base_width=w_px,
            style=style,
            kind="ledger",
        )


def _draw_slur_or_tie(
    canvas: Any,
    stroke_mod: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    arch_dir: int,  # -1 = arch upward (above), +1 = arch downward (below)
    sp: float,
    ink_rgb: Any,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
    is_tie: bool = False,
) -> None:
    """Draw a tapered calligraphic hand-drawn slur or tie Bezier arc."""
    span = max(sp * 0.8, abs(x1 - x0))
    if is_tie:
        height = min(sp * 1.15, max(sp * 0.55, span * 0.16))
    else:
        height = min(sp * 2.4, max(sp * 0.95, span * 0.18))
    height *= float(np_rng.uniform(0.90, 1.12))

    p0 = np.array([x0, y0], dtype=np.float64)
    p3 = np.array([x1, y1], dtype=np.float64)
    dx = x1 - x0
    dy = y1 - y0
    p1 = np.array([x0 + dx * 0.28, y0 + dy * 0.25 + arch_dir * height], dtype=np.float64)
    p2 = np.array([x0 + dx * 0.72, y0 + dy * 0.75 + arch_dir * height], dtype=np.float64)

    n_pts = max(14, min(48, int(span / 6.0)))
    outer = _eval_cubic(p0, p1, p2, p3, n=n_pts)
    local_style = dict(style)
    local_style["taper_start"] = 0.25
    local_style["taper_end"] = 0.25
    base_w = max(1.0, sp * (0.14 if is_tie else 0.16))
    stroke_mod.draw_calligraphic_polyline(canvas, outer, ink_rgb, base_w, local_style, kind="slur")

    # Subtle second inner belly stroke for authentic calligraphic thickening in the middle
    if span > sp * 1.4:
        p1_in = p1 + np.array([0.0, -arch_dir * sp * 0.12])
        p2_in = p2 + np.array([0.0, -arch_dir * sp * 0.12])
        inner = _eval_cubic(p0, p1_in, p2_in, p3, n=n_pts)
        mid_slice = inner[max(1, n_pts // 6) : max(2, n_pts - n_pts // 6)]
        if len(mid_slice) >= 2:
            stroke_mod.draw_calligraphic_polyline(
                canvas, mid_slice, ink_rgb, base_w * 0.85, local_style, kind="slur"
            )


def _draw_double_or_courtesy_accidental(
    canvas: Any,
    stroke_mod: Any,
    glyph_mod: Any,
    acc: str,
    cx: float,
    cy: float,
    sp: float,
    ink_rgb: Any,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
    courtesy: bool = False,
) -> None:
    """Draw sharp/flat/natural/double_sharp/double_flat with optional courtesy parentheses."""
    base_w = max(0.95, sp * float(style.get("acc_stroke_ratio", 0.145)))

    if acc == "double_sharp":
        # Hand-drawn ornate X with small corner serifs/blobs
        r = sp * 0.44
        s1 = np.array([[cx - r, cy - r], [cx + r, cy + r]], dtype=np.float64)
        s2 = np.array([[cx - r, cy + r], [cx + r, cy - r]], dtype=np.float64)
        stroke_mod.draw_calligraphic_polyline(canvas, s1, ink_rgb, base_w * 1.1, style, kind="accidental")
        stroke_mod.draw_calligraphic_polyline(canvas, s2, ink_rgb, base_w * 1.1, style, kind="accidental")
        dr = max(1.1, sp * 0.14)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                stroke_mod.draw_filled_blob(
                    canvas, cx + sx * r * 0.85, cy + sy * r * 0.85, dr, dr, 0.0, ink_rgb, style, hollow=False
                )
    elif acc == "double_flat":
        # Two closely-spaced overlapping flats
        glyph_mod.draw_accidental(canvas, "flat", cx - sp * 0.36, cy, sp * 0.96, ink_rgb, style, np_rng)
        glyph_mod.draw_accidental(canvas, "flat", cx + sp * 0.24, cy, sp * 0.96, ink_rgb, style, np_rng)
    else:
        glyph_mod.draw_accidental(canvas, acc, cx, cy, sp, ink_rgb, style, np_rng)

    if courtesy:
        # Small hand-drawn parentheses flanking the accidental
        px_off = sp * (0.85 if acc == "double_flat" else 0.62)
        ph = sp * 0.82
        pw = sp * 0.24
        left_p = _eval_quad(
            np.array([cx - px_off + pw, cy - ph]),
            np.array([cx - px_off - pw * 0.6, cy]),
            np.array([cx - px_off + pw, cy + ph]),
            n=10,
        )
        right_p = _eval_quad(
            np.array([cx + px_off - pw, cy - ph]),
            np.array([cx + px_off + pw * 0.6, cy]),
            np.array([cx + px_off - pw, cy + ph]),
            n=10,
        )
        stroke_mod.draw_calligraphic_polyline(canvas, left_p, ink_rgb, base_w * 0.78, style, kind="accidental")
        stroke_mod.draw_calligraphic_polyline(canvas, right_p, ink_rgb, base_w * 0.78, style, kind="accidental")


def _draw_clef_with_octave(
    canvas: Any,
    glyph_mod: Any,
    clef_name: str,
    cx: float,
    cy: float,
    sp: float,
    ink_rgb: Any,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
) -> None:
    """Draw clef plus small hand-drawn '8' below (8vb) or above (8va) when applicable."""
    base = base_clef_code(clef_name)
    glyph_mod.draw_clef(
        canvas=canvas,
        clef_name=base,
        cx=cx,
        cy=cy,
        sp=sp,
        color_rgb=ink_rgb,
        style=style,
        rng=np_rng,
    )
    shift = clef_octave_shift(clef_name)
    if shift != 0:
        y8 = cy + (sp * 3.15 if shift < 0 else -sp * 3.25)
        draw_digit = getattr(glyph_mod, "_draw_handwritten_digit", None)
        if draw_digit is not None:
            draw_digit(canvas, "8", cx, y8, sp * 0.62, ink_rgb, style, np_rng)


def _draw_articulation(
    canvas: Any,
    stroke_mod: Any,
    art: str,
    cx: float,
    cy: float,
    above: bool,
    sp: float,
    ink_rgb: Any,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
) -> None:
    """Draw staccato, accent, tenuto, fermata, trill, staccatissimo, or marcato."""
    w = max(1.0, sp * 0.14)
    if art == "staccato":
        r = max(1.4, sp * 0.17)
        stroke_mod.draw_filled_blob(canvas, cx, cy, r, r, 0.0, ink_rgb, style, hollow=False)
    elif art == "tenuto":
        hw = sp * 0.52
        pts = np.array([[cx - hw, cy], [cx + hw, cy + float(np_rng.normal(0.0, 0.04 * sp))]], dtype=np.float64)
        stroke_mod.draw_calligraphic_polyline(canvas, pts, ink_rgb, w * 1.25, style, kind="articulation")
    elif art == "accent":
        hw = sp * 0.58
        hh = sp * 0.34
        pts = np.array([[cx - hw, cy - hh], [cx + hw, cy], [cx - hw, cy + hh]], dtype=np.float64)
        stroke_mod.draw_calligraphic_polyline(canvas, pts, ink_rgb, w * 1.15, style, kind="articulation")
    elif art == "marcato":
        hw = sp * 0.36
        hh = sp * 0.52
        s = -1.0 if above else 1.0
        pts = np.array([[cx - hw, cy - s * hh * 0.5], [cx, cy + s * hh * 0.6], [cx + hw, cy - s * hh * 0.5]], dtype=np.float64)
        stroke_mod.draw_calligraphic_polyline(canvas, pts, ink_rgb, w * 1.15, style, kind="articulation")
    elif art == "staccatissimo":
        hh = sp * 0.42
        s = -1.0 if above else 1.0
        pts = np.array([[cx, cy - s * hh], [cx, cy + s * hh * 0.6]], dtype=np.float64)
        stroke_mod.draw_calligraphic_polyline(canvas, pts, ink_rgb, w * 1.5, style, kind="articulation")
    elif art == "fermata":
        # Semicircular arch + center ink dot
        hw = sp * 0.92
        hh = sp * 0.78
        s = -1.0 if above else 1.0
        arch = _eval_cubic(
            np.array([cx - hw, cy]),
            np.array([cx - hw * 0.65, cy + s * hh * 1.25]),
            np.array([cx + hw * 0.65, cy + s * hh * 1.25]),
            np.array([cx + hw, cy]),
            n=16,
        )
        stroke_mod.draw_calligraphic_polyline(canvas, arch, ink_rgb, w * 1.35, style, kind="articulation")
        r = max(1.5, sp * 0.18)
        stroke_mod.draw_filled_blob(canvas, cx, cy + s * hh * 0.25, r, r, 0.0, ink_rgb, style, hollow=False)
    elif art == "trill":
        _draw_handwritten_text(canvas, stroke_mod, "tr", cx - sp * 0.55, cy, sp * 0.95, ink_rgb, style, np_rng, italic=True)


def _draw_tuplet_bracket(
    canvas: Any,
    stroke_mod: Any,
    glyph_mod: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    num: int,
    above: bool,
    sp: float,
    ink_rgb: Any,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
) -> None:
    """Draw a hand-drawn tuplet bracket with centered handwritten numeral."""
    w = max(0.95, sp * 0.11)
    hook_dir = 1.0 if above else -1.0
    hook_len = sp * 0.55
    xm = 0.5 * (x0 + x1)
    ym = 0.5 * (y0 + y1)
    gap = sp * 0.72

    if abs(x1 - x0) > gap * 2.2:
        t_l = max(0.05, (xm - gap - x0) / max(1e-3, x1 - x0))
        t_r = min(0.95, (xm + gap - x0) / max(1e-3, x1 - x0))
        yl_g = y0 + t_l * (y1 - y0)
        yr_g = y0 + t_r * (y1 - y0)
        left_arm = np.array([[x0, y0 + hook_dir * hook_len], [x0, y0], [xm - gap, yl_g]], dtype=np.float64)
        right_arm = np.array([[xm + gap, yr_g], [x1, y1], [x1, y1 + hook_dir * hook_len]], dtype=np.float64)
        stroke_mod.draw_calligraphic_polyline(canvas, left_arm, ink_rgb, w, style, kind="tuplet")
        stroke_mod.draw_calligraphic_polyline(canvas, right_arm, ink_rgb, w, style, kind="tuplet")

    draw_digit = getattr(glyph_mod, "_draw_handwritten_digit", None)
    num_s = str(num)
    if draw_digit is not None and len(num_s) == 1:
        draw_digit(canvas, num_s, xm, ym, sp * 0.68, ink_rgb, style, np_rng)
    else:
        _draw_handwritten_text(canvas, stroke_mod, num_s, xm - sp * 0.3, ym, sp * 0.75, ink_rgb, style, np_rng)


def _draw_hairpin(
    canvas: Any,
    stroke_mod: Any,
    kind: str,  # 'crescendo' (<) or 'diminuendo' (>)
    x0: float,
    x1: float,
    yc: float,
    sp: float,
    ink_rgb: Any,
    style: Dict[str, Any],
) -> None:
    """Draw a hand-drawn crescendo (<) or diminuendo (>) hairpin wedge."""
    if x1 <= x0 + sp * 0.4:
        x1 = x0 + sp * 2.5
    half_h = sp * 0.48
    w = max(0.95, sp * 0.125)
    if kind == "crescendo":
        top = np.array([[x0, yc], [x1, yc - half_h]], dtype=np.float64)
        bot = np.array([[x0, yc], [x1, yc + half_h]], dtype=np.float64)
    else:
        top = np.array([[x0, yc - half_h], [x1, yc]], dtype=np.float64)
        bot = np.array([[x0, yc + half_h], [x1, yc]], dtype=np.float64)
    stroke_mod.draw_calligraphic_polyline(canvas, top, ink_rgb, w, style, kind="hairpin")
    stroke_mod.draw_calligraphic_polyline(canvas, bot, ink_rgb, w, style, kind="hairpin")


def _draw_multi_bar_rest(
    canvas: Any,
    stroke_mod: Any,
    glyph_mod: Any,
    x0: float,
    x1: float,
    yc: float,
    count: int,
    sp: float,
    ink_rgb: Any,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
) -> None:
    """Draw a multi-measure rest bar with end serifs and centered count numeral."""
    pad = max(sp * 0.9, (x1 - x0) * 0.18)
    rx0 = x0 + pad
    rx1 = max(rx0 + sp * 1.5, x1 - pad)
    bar_pts = np.array([[rx0, yc], [rx1, yc]], dtype=np.float64)
    stroke_mod.draw_calligraphic_polyline(canvas, bar_pts, ink_rgb, max(2.2, sp * 0.52), style, kind="beam")
    sw = max(1.0, sp * 0.14)
    for sx in (rx0, rx1):
        serif = np.array([[sx, yc - sp * 0.95], [sx, yc + sp * 0.95]], dtype=np.float64)
        stroke_mod.draw_calligraphic_polyline(canvas, serif, ink_rgb, sw, style, kind="barline")
    xm = 0.5 * (rx0 + rx1)
    ym = yc - sp * 2.85
    draw_digit = getattr(glyph_mod, "_draw_handwritten_digit", None)
    cs = str(count)
    if draw_digit is not None:
        step = sp * 0.72
        sx0 = xm - 0.5 * step * (len(cs) - 1)
        for i, ch in enumerate(cs):
            if ch.isdigit():
                draw_digit(canvas, ch, sx0 + i * step, ym, sp * 0.82, ink_rgb, style, np_rng)


def _draw_volta_bracket(
    canvas: Any,
    stroke_mod: Any,
    x0: float,
    x1: float,
    y_top: float,
    label: str,
    draw_left_hook: bool,
    draw_right_hook: bool,
    sp: float,
    ink_rgb: Any,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
) -> None:
    """Draw a volta repeat bracket ('1.', '2.') above the top staff."""
    w = max(1.0, sp * 0.12)
    hook_h = sp * 1.15
    top_line = np.array([[x0, y_top], [x1, y_top]], dtype=np.float64)
    stroke_mod.draw_calligraphic_polyline(canvas, top_line, ink_rgb, w, style, kind="volta")
    if draw_left_hook:
        lh = np.array([[x0, y_top], [x0, y_top + hook_h]], dtype=np.float64)
        stroke_mod.draw_calligraphic_polyline(canvas, lh, ink_rgb, w, style, kind="volta")
        _draw_handwritten_text(
            canvas, stroke_mod, label, x0 + sp * 0.45, y_top + sp * 0.85, sp * 0.82, ink_rgb, style, np_rng
        )
    if draw_right_hook:
        rh = np.array([[x1, y_top], [x1, y_top + hook_h]], dtype=np.float64)
        stroke_mod.draw_calligraphic_polyline(canvas, rh, ink_rgb, w, style, kind="volta")


# Compact vector stroke glyphs (normalized x in [0, 0.7], y in [-0.7, 0.4], baseline at 0)
_STROKE_FONT: Dict[str, List[List[Tuple[float, float]]]] = {
    "a": [[(0.50, -0.30), (0.20, -0.38), (0.05, -0.15), (0.20, 0.02), (0.50, -0.15)], [(0.52, -0.35), (0.50, 0.02)]],
    "b": [[(0.10, -0.78), (0.08, 0.02)], [(0.09, -0.28), (0.42, -0.36), (0.52, -0.14), (0.36, 0.02), (0.08, -0.02)]],
    "c": [[(0.50, -0.30), (0.22, -0.38), (0.06, -0.16), (0.22, 0.02), (0.52, -0.04)]],
    "d": [[(0.50, -0.78), (0.48, 0.02)], [(0.49, -0.30), (0.18, -0.36), (0.06, -0.14), (0.22, 0.02), (0.48, -0.04)]],
    "e": [[(0.08, -0.18), (0.50, -0.22), (0.42, -0.38), (0.14, -0.34), (0.08, -0.12), (0.26, 0.02), (0.52, -0.04)]],
    "f": [[(0.52, -0.74), (0.32, -0.82), (0.24, -0.58), (0.18, 0.24)], [(0.05, -0.34), (0.46, -0.36)]],
    "g": [[(0.50, -0.32), (0.20, -0.38), (0.06, -0.16), (0.22, 0.00), (0.48, -0.18)], [(0.50, -0.34), (0.46, 0.24), (0.24, 0.36), (0.06, 0.26)]],
    "h": [[(0.10, -0.78), (0.08, 0.02)], [(0.09, -0.22), (0.34, -0.38), (0.50, -0.24), (0.50, 0.02)]],
    "i": [[(0.24, -0.36), (0.22, 0.02)], [(0.25, -0.56), (0.27, -0.54)]],
    "j": [[(0.32, -0.36), (0.28, 0.24), (0.10, 0.34)], [(0.34, -0.56), (0.36, -0.54)]],
    "k": [[(0.10, -0.78), (0.08, 0.02)], [(0.48, -0.38), (0.10, -0.14), (0.50, 0.02)]],
    "l": [[(0.24, -0.78), (0.22, 0.00), (0.34, 0.02)]],
    "m": [[(0.04, -0.36), (0.04, 0.02)], [(0.04, -0.24), (0.22, -0.38), (0.34, -0.24), (0.34, 0.02)], [(0.34, -0.24), (0.52, -0.38), (0.64, -0.24), (0.64, 0.02)]],
    "n": [[(0.08, -0.36), (0.08, 0.02)], [(0.08, -0.24), (0.32, -0.38), (0.50, -0.24), (0.50, 0.02)]],
    "o": [[(0.28, -0.38), (0.06, -0.22), (0.12, 0.00), (0.36, 0.02), (0.52, -0.18), (0.28, -0.38)]],
    "p": [[(0.12, -0.36), (0.06, 0.36)], [(0.12, -0.30), (0.44, -0.38), (0.54, -0.16), (0.36, 0.02), (0.10, -0.02)]],
    "q": [[(0.48, -0.34), (0.44, 0.36)], [(0.48, -0.30), (0.18, -0.38), (0.06, -0.16), (0.22, 0.02), (0.46, -0.04)]],
    "r": [[(0.10, -0.36), (0.08, 0.02)], [(0.10, -0.22), (0.30, -0.38), (0.46, -0.32)]],
    "s": [[(0.48, -0.32), (0.22, -0.38), (0.10, -0.24), (0.44, -0.12), (0.42, 0.02), (0.10, 0.00)]],
    "t": [[(0.26, -0.68), (0.24, -0.04), (0.42, 0.02)], [(0.06, -0.38), (0.48, -0.40)]],
    "u": [[(0.08, -0.36), (0.08, -0.06), (0.26, 0.02), (0.48, -0.10)], [(0.50, -0.36), (0.48, 0.02)]],
    "v": [[(0.06, -0.36), (0.28, 0.02), (0.52, -0.36)]],
    "w": [[(0.04, -0.36), (0.18, 0.02), (0.34, -0.24), (0.50, 0.02), (0.64, -0.36)]],
    "x": [[(0.08, -0.36), (0.48, 0.02)], [(0.48, -0.36), (0.08, 0.02)]],
    "y": [[(0.08, -0.36), (0.28, -0.02)], [(0.50, -0.36), (0.26, 0.04), (0.10, 0.34)]],
    "z": [[(0.08, -0.36), (0.50, -0.36), (0.08, 0.02), (0.52, 0.02)]],
    "0": [[(0.28, -0.72), (0.08, -0.36), (0.28, 0.02), (0.48, -0.36), (0.28, -0.72)]],
    "1": [[(0.14, -0.54), (0.30, -0.72), (0.28, 0.02)]],
    "2": [[(0.10, -0.56), (0.30, -0.72), (0.48, -0.54), (0.08, 0.02), (0.52, 0.02)]],
    "3": [[(0.10, -0.64), (0.46, -0.68), (0.26, -0.36), (0.48, -0.14), (0.10, 0.00)]],
    "4": [[(0.42, -0.72), (0.08, -0.24), (0.54, -0.24)], [(0.42, -0.72), (0.40, 0.02)]],
    "5": [[(0.48, -0.70), (0.14, -0.70), (0.12, -0.38), (0.46, -0.32), (0.42, -0.04), (0.10, 0.00)]],
    "6": [[(0.44, -0.70), (0.14, -0.44), (0.10, -0.06), (0.36, 0.02), (0.48, -0.22), (0.14, -0.28)]],
    "7": [[(0.08, -0.70), (0.52, -0.70), (0.20, 0.02)]],
    "8": [[(0.28, -0.72), (0.10, -0.52), (0.46, -0.18), (0.28, 0.02), (0.10, -0.18), (0.46, -0.52), (0.28, -0.72)]],
    "9": [[(0.46, -0.48), (0.24, -0.72), (0.10, -0.48), (0.46, -0.42), (0.34, 0.02)]],
    ".": [[(0.18, -0.02), (0.22, 0.02)]],
    ",": [[(0.22, -0.02), (0.14, 0.14)]],
    "-": [[(0.08, -0.20), (0.44, -0.20)]],
    "=": [[(0.06, -0.32), (0.48, -0.32)], [(0.06, -0.14), (0.48, -0.14)]],
    "'": [[(0.22, -0.72), (0.18, -0.52)]],
    "!": [[(0.24, -0.72), (0.24, -0.18)], [(0.24, -0.02), (0.25, 0.02)]],
}


def _draw_handwritten_text(
    canvas: Any,
    stroke_mod: Any,
    text: str,
    x: float,
    y_baseline: float,
    size_px: float,
    ink_rgb: Any,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
    italic: bool = False,
    bold: bool = False,
) -> float:
    """Render handwritten text in matching ink, slant, and calligraphic stroke physics."""
    if not text:
        return x
    slant_deg = float(style.get("slant_deg", 6.0)) + (12.0 if italic else 0.0)
    shear = math.tan(math.radians(float(np.clip(slant_deg, -24.0, 28.0))))
    w = max(0.85, size_px * (0.145 if bold else 0.105))
    cursor = x

    local_style = dict(style)
    local_style["wobble_px"] = min(float(style.get("wobble_px", 0.7)), size_px * 0.06)
    local_style["tremor_px"] = min(float(style.get("tremor_px", 0.2)), size_px * 0.03)

    for ch in text:
        if ch == " ":
            cursor += size_px * 0.42
            continue
        is_upper = ch.isupper()
        key = ch.lower()
        strokes = _STROKE_FONT.get(key)
        if strokes is None:
            cursor += size_px * 0.35
            continue
        scale_y = 1.32 if (is_upper and key.isalpha()) else 1.0
        scale_x = 1.15 if (is_upper and key.isalpha()) else 1.0
        adv = size_px * (0.72 if key in ("m", "w") else 0.58) * scale_x
        for st in strokes:
            pts_list: List[List[float]] = []
            for px, py in st:
                ly = py * size_px * scale_y
                lx = px * size_px * scale_x - ly * shear
                pts_list.append([cursor + lx, y_baseline + ly])
            pts = np.asarray(pts_list, dtype=np.float64)
            stroke_mod.draw_calligraphic_polyline(canvas, pts, ink_rgb, w, local_style, kind="text")
        cursor += adv + float(np_rng.normal(0.0, size_px * 0.02))
    return cursor


def _draw_dynamic_marking(
    canvas: Any,
    stroke_mod: Any,
    text: str,
    cx: float,
    cy: float,
    sp: float,
    ink_rgb: Any,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
) -> None:
    """Draw calligraphic italic dynamics (pp, p, mp, mf, f, ff, sfz, fp, etc.)."""
    clean = text.strip().lower()
    if not clean:
        return
    size_px = sp * 1.32
    est_w = len(clean) * size_px * 0.56
    _draw_handwritten_text(
        canvas,
        stroke_mod,
        clean,
        cx - est_w * 0.45,
        cy + size_px * 0.25,
        size_px,
        ink_rgb,
        style,
        np_rng,
        italic=True,
        bold=True,
    )


# ---------------------------------------------------------------------------
# Stage-2 Training Export (`--export-train-dir`)
# ---------------------------------------------------------------------------


def _find_assembly_serialize_bin() -> Optional[Path]:
    repo_root = _THIS_DIR.parent.parent
    candidates = [
        repo_root / "converter" / "assembly-serialize" / "target" / "release" / "assembly-serialize.exe",
        repo_root / "converter" / "assembly-serialize" / "target" / "release" / "assembly-serialize",
        repo_root / "target" / "release" / "assembly-serialize.exe",
        repo_root / "target" / "release" / "assembly-serialize",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _export_training_systems(
    score: ParsedScore,
    score_path: Path,
    final_img: Image.Image,
    systems: List[SystemLayout],
    export_dir: Path,
    stem: str,
) -> List[Dict[str, Any]]:
    """Export cropped per-system PNGs and exact `assembly-serialize bekern` targets."""
    export_dir.mkdir(parents=True, exist_ok=True)
    bin_path = _find_assembly_serialize_bin()
    page_w, page_h = final_img.size

    # Prepare a MusicXML source path for `assembly-serialize bekern`
    temp_dir_obj: Optional[tempfile.TemporaryDirectory[str]] = None
    mxml_input: Optional[Path] = None
    use_sliced_export = False
    ext = score_path.suffix.lower()
    if ext in (".musicxml", ".xml"):
        mxml_input = score_path
    elif ext == ".mxl":
        temp_dir_obj = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(score_path, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith((".musicxml", ".xml")) and not name.startswith("META-INF"):
                    mxml_input = Path(temp_dir_obj.name) / "extracted.musicxml"
                    mxml_input.write_bytes(zf.read(name))
                    break
    else:
        # Export per-system ParsedScore slices from .mscx/.mscz into clean MusicXML 3.1 for bekern
        temp_dir_obj = tempfile.TemporaryDirectory()
        use_sliced_export = True

    exported: List[Dict[str, Any]] = []
    try:
        for sys_lay in systems:
            if sys_lay.is_blank_prerule or not sys_lay.staves:
                continue
            sp = sys_lay.staves[0].sp
            top_y = min(st.y_center for st in sys_lay.staves) - 5.2 * sp
            bot_y = max(st.y_center for st in sys_lay.staves) + 5.2 * sp
            left_x = max(0, int(math.floor(sys_lay.x_left - 2.5 * sp)))
            right_x = min(page_w, int(math.ceil(sys_lay.x_right + 1.5 * sp)))
            y0 = max(0, int(math.floor(top_y)))
            y1 = min(page_h, int(math.ceil(bot_y)))

            sys_stem = f"{stem}_sys{sys_lay.system_index:02d}"
            crop_png = export_dir / f"{sys_stem}.png"
            final_img.crop((left_x, y0, right_x, y1)).save(crop_png)

            m1 = sys_lay.measure_start + 1
            m2 = sys_lay.measure_end
            bekern_path = export_dir / f"{sys_stem}-bekern.txt"
            bekern_ok = False
            bekern_err: Optional[str] = None

            sys_mxml = mxml_input
            meas_arg = f"{m1}..{m2}"
            if use_sliced_export and temp_dir_obj is not None:
                sys_mxml = export_score_to_musicxml(
                    score,
                    Path(temp_dir_obj.name) / f"{sys_stem}.musicxml",
                    measure_range=(sys_lay.measure_start, sys_lay.measure_end),
                )
                meas_arg = f"1..{max(1, m2 - m1 + 1)}"

            if bin_path is not None and sys_mxml is not None and sys_mxml.exists():
                cmd = [
                    str(bin_path),
                    "bekern",
                    "--input",
                    str(sys_mxml),
                    "--measures",
                    meas_arg,
                    "--out",
                    str(export_dir),
                    "--stem",
                    sys_stem,
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if proc.returncode == 0 and bekern_path.exists():
                    bekern_ok = True
                else:
                    bekern_err = (proc.stderr or proc.stdout or "").strip()[:300]

            exported.append(
                {
                    "system_index": sys_lay.system_index,
                    "measures_1based": [m1, m2],
                    "crop_box": [left_x, y0, right_x, y1],
                    "image": str(crop_png),
                    "bekern": str(bekern_path) if bekern_ok else None,
                    "bekern_ok": bekern_ok,
                    "bekern_error": bekern_err,
                }
            )
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()

    manifest_path = export_dir / f"{stem}_train_manifest.json"
    manifest_path.write_text(
        json.dumps({"stem": stem, "source": str(score_path), "systems": exported}, indent=2),
        encoding="utf-8",
    )
    return exported


# ---------------------------------------------------------------------------
# Main Handwritten Score Renderer
# ---------------------------------------------------------------------------


def render_score_handwritten(
    score_path: Union[str, Path],
    style_name: str,
    out_png: Union[str, Path],
    seed: int = 42,
    page_w: int = 1650,
    page_h: int = 2200,
    export_train_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Render a MusicXML / MuseScore file in the given handwritten style.

    Saves `out_png` and a companion Stage-1 Bezier GT JSON at `Path(out_png).with_suffix('.json')`.
    If `export_train_dir` is provided, also exports cropped Stage-2 per-system PNGs and
    exact `assembly-serialize bekern` targets.
    """
    import glyph_variants
    import layout_quirks
    import stroke_engine
    import styles

    score = parse_score(score_path)
    style = styles.get_style(style_name) if hasattr(styles, "get_style") else styles.SCRIBE_STYLES[style_name]
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    ink_rgb = style.get("ink_rgb", (0.12, 0.10, 0.08))
    staff_ink_rgb = style.get("staff_ink_rgb", ink_rgb)
    paper_rgb = style.get("paper_rgb", (0.95, 0.92, 0.86))
    stroke_scale = float(style.get("stroke_scale", 1.0))

    # 1. Manuscript paper background
    paper_arr = layout_quirks.make_manuscript_paper(page_w, page_h, style, np_rng)
    if isinstance(paper_arr, Image.Image):
        paper_f32 = np.asarray(paper_arr.convert("RGB"), dtype=np.float32) / 255.0
    else:
        paper_f32 = np.asarray(paper_arr, dtype=np.float32)
        if float(np.max(paper_f32)) > 1.0 + 1e-3:
            paper_f32 = paper_f32 / 255.0
        paper_f32 = np.clip(paper_f32, 0.0, 1.0)

    # 2. Initialize StrokeCanvas & load textured paper
    canvas = stroke_engine.StrokeCanvas(
        width=page_w,
        height=page_h,
        bg_rgb=paper_rgb,
        rng=np_rng,
    )
    if hasattr(canvas, "buf") and canvas.buf.shape == paper_f32.shape:
        canvas.buf[:] = paper_f32

    # 3. Compute page systems & curved staff lines
    systems = layout_score_pages(score, page_w=page_w, page_h=page_h, style=style, rng=rng)
    gt_systems: List[Dict[str, Any]] = []

    for sys_lay in systems:
        sys_gt_staves: List[Dict[str, Any]] = []
        staff_curves: List[Tuple[Callable[[Any, int], Any], List[List[float]], float]] = []

        # Draw 5-line curved staves
        for st_lay in sys_lay.staves:
            y_fn_2, ctrl_pts, sp_eff = _get_staff_curve(
                layout_quirks,
                st_lay.x_left,
                st_lay.x_right,
                st_lay.y_center,
                st_lay.sp,
                style,
                np_rng,
            )
            st_lay.sp = sp_eff
            st_lay.ctrl_pts = ctrl_pts
            staff_curves.append((y_fn_2, ctrl_pts, sp_eff))

            staff_lw = max(1.0, sp_eff * 0.085 * stroke_scale)
            for line_idx in range(-2, 3):
                stroke_engine.draw_staff_line(
                    canvas=canvas,
                    x0=st_lay.x_left,
                    x1=st_lay.x_right,
                    y_fn=(lambda x, fn=y_fn_2, li=line_idx: fn(x, li)),
                    color_rgb=staff_ink_rgb,
                    base_width=staff_lw,
                    style=style,
                )

            sys_gt_staves.append(
                {
                    "part_index": st_lay.part_index,
                    "part_name": st_lay.part_name,
                    "sp": round(float(sp_eff), 3),
                    "x_left": round(float(st_lay.x_left), 2),
                    "x_right": round(float(st_lay.x_right), 2),
                    "y_center": round(float(st_lay.y_center), 2),
                    "ctrl_pts": [[round(c[0], 2), round(c[1], 2)] for c in ctrl_pts],
                    "is_blank_prerule": sys_lay.is_blank_prerule,
                }
            )

        gt_systems.append(
            {
                "system_index": sys_lay.system_index,
                "measure_start": sys_lay.measure_start,
                "measure_end": sys_lay.measure_end,
                "is_blank_prerule": sys_lay.is_blank_prerule,
                "staves": sys_gt_staves,
            }
        )

        if sys_lay.is_blank_prerule:
            continue

        # Draw system-left connecting barline & grand-staff braces
        if len(sys_lay.staves) >= 2:
            top_st = sys_lay.staves[0]
            y_top = float(staff_curves[0][0](sys_lay.x_left, -2))
            y_bot = float(staff_curves[-1][0](sys_lay.x_left, 2))
            glyph_variants.draw_barline(
                canvas=canvas,
                x=sys_lay.x_left,
                y_top=y_top,
                y_bot=y_bot,
                sp=top_st.sp,
                kind="single",
                color_rgb=ink_rgb,
                style=style,
                rng=np_rng,
            )

            seen_groups: Dict[int, List[int]] = {}
            for idx, st_lay in enumerate(sys_lay.staves):
                if st_lay.brace_group is not None:
                    seen_groups.setdefault(st_lay.brace_group, []).append(idx)
            for idxs in seen_groups.values():
                if len(idxs) >= 2:
                    i0, i1 = idxs[0], idxs[-1]
                    by0 = float(staff_curves[i0][0](sys_lay.x_left, -2))
                    by1 = float(staff_curves[i1][0](sys_lay.x_left, 2))
                    glyph_variants.draw_brace(
                        canvas=canvas,
                        x=sys_lay.x_left - sys_lay.staves[i0].sp * 0.7,
                        y_top=by0,
                        y_bot=by1,
                        sp=sys_lay.staves[i0].sp,
                        color_rgb=ink_rgb,
                        style=style,
                        rng=np_rng,
                    )

        # Render each staff's measures in this system
        for s_idx, st_lay in enumerate(sys_lay.staves):
            part = score.parts[st_lay.part_index]
            y_fn_2, _, sp = staff_curves[s_idx]

            def y_mid(x_val: float, _fn: Callable[[Any, int], Any] = y_fn_2) -> float:
                return float(_fn(x_val, 0))

            # Track rendered note coordinates in this system for ties/slurs/hairpins
            # (m_idx, ev_idx) -> (ex, {step: ny}, stem_dir, min_y, max_y)
            rendered_notes: Dict[Tuple[int, int], Tuple[float, Dict[int, float], int, float, float]] = {}
            measure_beat_to_x: Dict[int, Callable[[float], float]] = {}

            for rel_m, m_idx in enumerate(range(sys_lay.measure_start, sys_lay.measure_end)):
                if m_idx >= len(part.measures):
                    continue
                meas = part.measures[m_idx]
                bx0, bx1 = sys_lay.bar_x_bounds[rel_m]
                cursor_x = bx0 + sp * 0.7

                # Left repeat barline if present
                if meas.left_barline == "repeat_left":
                    glyph_variants.draw_barline(
                        canvas=canvas,
                        x=bx0 + sp * 0.15,
                        y_top=float(y_fn_2(bx0, -2)),
                        y_bot=float(y_fn_2(bx0, 2)),
                        sp=sp,
                        kind="repeat-left",
                        color_rgb=ink_rgb,
                        style=style,
                        rng=np_rng,
                    )
                    cursor_x += sp * 1.1

                # Clef at start of system or on clef change
                if rel_m == 0 or meas.clef_change:
                    clef_x = cursor_x + sp * 0.9
                    _draw_clef_with_octave(
                        canvas=canvas,
                        glyph_mod=glyph_variants,
                        clef_name=meas.clef,
                        cx=clef_x,
                        cy=y_mid(clef_x),
                        sp=sp,
                        ink_rgb=ink_rgb,
                        style=style,
                        np_rng=np_rng,
                    )
                    cursor_x += sp * 2.5

                # Key signature at start of system or on key change
                if (rel_m == 0 or meas.key_change) and meas.key_fifths != 0:
                    acc_kind = "sharp" if meas.key_fifths > 0 else "flat"
                    bclef = base_clef_code(meas.clef)
                    steps_tbl = (
                        SHARP_STEPS_BY_CLEF.get(bclef, SHARP_STEPS_BY_CLEF["G-2"])
                        if meas.key_fifths > 0
                        else FLAT_STEPS_BY_CLEF.get(bclef, FLAT_STEPS_BY_CLEF["G-2"])
                    )
                    for k_i in range(abs(meas.key_fifths)):
                        kx = cursor_x + k_i * sp * 0.85
                        ky = y_mid(kx) - (steps_tbl[k_i] - 4) * (sp * 0.5)
                        glyph_variants.draw_accidental(
                            canvas=canvas,
                            acc=acc_kind,
                            cx=kx,
                            cy=ky,
                            sp=sp,
                            color_rgb=ink_rgb,
                            style=style,
                            rng=np_rng,
                        )
                    cursor_x += abs(meas.key_fifths) * sp * 0.85 + sp * 0.7

                # Time signature on measure 0 or on time change
                if m_idx == 0 or meas.time_change:
                    tx = cursor_x + sp * 0.8
                    glyph_variants.draw_time_sig(
                        canvas=canvas,
                        num=meas.time_sig[0],
                        den=meas.time_sig[1],
                        cx=tx,
                        cy=y_mid(tx),
                        sp=sp,
                        color_rgb=ink_rgb,
                        style=style,
                        rng=np_rng,
                    )
                    cursor_x += sp * 2.1

                # Volta bracket on top staff of system
                if s_idx == 0 and meas.volta is not None:
                    vb = meas.volta
                    vy = float(y_fn_2(0.5 * (bx0 + bx1), -2)) - sp * 3.1
                    _draw_volta_bracket(
                        canvas=canvas,
                        stroke_mod=stroke_engine,
                        x0=bx0 + sp * 0.15,
                        x1=bx1 - sp * 0.15,
                        y_top=vy,
                        label=vb.text,
                        draw_left_hook=(m_idx == vb.start_measure),
                        draw_right_hook=(m_idx == vb.end_measure and vb.closed_right),
                        sp=sp,
                        ink_rgb=ink_rgb,
                        style=style,
                        np_rng=np_rng,
                    )

                # Multi-bar rest shortcut
                if meas.multi_bar_rest > 1:
                    _draw_multi_bar_rest(
                        canvas=canvas,
                        stroke_mod=stroke_engine,
                        glyph_mod=glyph_variants,
                        x0=cursor_x,
                        x1=bx1 - sp * 0.4,
                        yc=y_mid(0.5 * (cursor_x + bx1)),
                        count=meas.multi_bar_rest,
                        sp=sp,
                        ink_rgb=ink_rgb,
                        style=style,
                        np_rng=np_rng,
                    )
                    measure_beat_to_x[m_idx] = (lambda _b, x=0.5 * (cursor_x + bx1): x)
                else:
                    # Compute horizontal positions aligned by beat across voices
                    extra_beats = (
                        [d.beat_offset for d in meas.dynamics]
                        + [t.beat_offset for t in meas.tempos]
                        + [mc.beat_offset for mc in meas.mid_changes]
                    )
                    ev_xs, beat_to_x = _compute_beat_x_map(
                        layout_quirks, cursor_x, bx1 - sp * 0.5, sp, meas.events, extra_beats, style, np_rng
                    )
                    measure_beat_to_x[m_idx] = beat_to_x

                    # Mid-measure clef / key / time changes
                    for mc in meas.mid_changes:
                        mx = max(cursor_x + sp * 0.4, beat_to_x(mc.beat_offset) - sp * 1.15)
                        my = y_mid(mx)
                        if mc.kind == "clef":
                            _draw_clef_with_octave(
                                canvas, glyph_variants, str(mc.value), mx, my, sp * 0.80, ink_rgb, style, np_rng
                            )
                        elif mc.kind == "key" and isinstance(mc.value, int) and mc.value != 0:
                            ak = "sharp" if mc.value > 0 else "flat"
                            bclef = base_clef_code(meas.clef)
                            stbl = (
                                SHARP_STEPS_BY_CLEF.get(bclef, SHARP_STEPS_BY_CLEF["G-2"])
                                if mc.value > 0
                                else FLAT_STEPS_BY_CLEF.get(bclef, FLAT_STEPS_BY_CLEF["G-2"])
                            )
                            for k_i in range(abs(mc.value)):
                                kx = mx + k_i * sp * 0.72
                                ky = y_mid(kx) - (stbl[k_i] - 4) * (sp * 0.5)
                                glyph_variants.draw_accidental(canvas, ak, kx, ky, sp * 0.88, ink_rgb, style, np_rng)
                        elif mc.kind == "time" and isinstance(mc.value, tuple):
                            glyph_variants.draw_time_sig(
                                canvas, mc.value[0], mc.value[1], mx, my, sp * 0.90, ink_rgb, style, np_rng
                            )

                    # Pre-group beamed notes per beam_group ID
                    beam_group_dir: Dict[int, int] = {}
                    for ev in meas.events:
                        if isinstance(ev, NoteEvent) and ev.beam_group is not None:
                            if ev.beam_group not in beam_group_dir:
                                if ev.stem_dir is not None:
                                    beam_group_dir[ev.beam_group] = ev.stem_dir
                                else:
                                    beam_group_dir[ev.beam_group] = 0
                    for bg_id, cur_dir in list(beam_group_dir.items()):
                        if cur_dir != 0:
                            continue
                        steps_in_bg = [
                            s
                            for ev in meas.events
                            if isinstance(ev, NoteEvent) and ev.beam_group == bg_id
                            for s in ev.staff_steps
                        ]
                        avg_s = sum(steps_in_bg) / max(1, len(steps_in_bg))
                        beam_group_dir[bg_id] = -1 if avg_s >= 4.0 else 1

                    beam_groups: Dict[int, List[Tuple[float, float, float, int, str]]] = {}
                    tuplet_groups: Dict[int, List[Tuple[float, float, float, int, int]]] = {}

                    for ev_idx, (ev, ex) in enumerate(zip(meas.events, ev_xs)):
                        yc = y_mid(ex)

                        if isinstance(ev, RestEvent):
                            # Shift rests slightly up/down in polyphonic measures
                            ry = yc
                            if len(meas.voices_present) >= 2:
                                ry += -sp * 1.1 if (ev.voice % 2 == 1) else sp * 1.1
                            glyph_variants.draw_rest(
                                canvas=canvas,
                                rest_type=ev.duration_kind,
                                cx=ex,
                                cy=ry,
                                sp=sp,
                                color_rgb=ink_rgb,
                                style=style,
                                rng=np_rng,
                            )
                            if ev.dots > 0:
                                for d_i in range(ev.dots):
                                    stroke_engine.draw_filled_blob(
                                        canvas,
                                        ex + sp * (0.85 + d_i * 0.42),
                                        ry - sp * 0.35,
                                        max(1.4, sp * 0.14),
                                        max(1.4, sp * 0.14),
                                        0.0,
                                        ink_rgb,
                                        style,
                                        hollow=False,
                                    )
                            for art in ev.articulations:
                                if art == "fermata":
                                    _draw_articulation(
                                        canvas, stroke_engine, "fermata", ex, yc - sp * 2.8, True, sp, ink_rgb, style, np_rng
                                    )
                            if ev.tuplet_id is not None and ev.tuplet_actual:
                                tuplet_groups.setdefault(ev.tuplet_id, []).append(
                                    (ex, yc - sp * 2.4, yc + sp * 2.4, ev.tuplet_actual, 1 if ev.voice % 2 == 1 else -1)
                                )
                            continue

                        # NoteEvent / Chord / Grace Note
                        eff_sp = sp * (0.64 if ev.is_grace else 1.0)
                        _draw_ledger_lines(canvas, stroke_engine, ex, yc, sp, ev.staff_steps, ink_rgb, style, rng)

                        if ev.is_grace:
                            stem_dir = 1
                        elif ev.beam_group is not None and ev.beam_group in beam_group_dir:
                            stem_dir = beam_group_dir[ev.beam_group]
                        elif ev.stem_dir is not None:
                            stem_dir = ev.stem_dir
                        else:
                            avg_step = sum(ev.staff_steps) / len(ev.staff_steps)
                            stem_dir = -1 if avg_step >= 4.0 else 1

                        ys_note: List[float] = []
                        step_to_y: Dict[int, float] = {}
                        for h_i, (st_step, acc_kind) in enumerate(zip(ev.staff_steps, ev.accidentals)):
                            ny = yc - (st_step - 4) * (sp * 0.5)
                            ys_note.append(ny)
                            step_to_y[st_step] = ny
                            hx_off = (
                                ev.head_x_nudges_sp[h_i] * sp
                                if h_i < len(ev.head_x_nudges_sp)
                                else 0.0
                            )
                            hx = ex + hx_off

                            if acc_kind:
                                is_court = (
                                    ev.courtesy_accidentals[h_i]
                                    if h_i < len(ev.courtesy_accidentals)
                                    else False
                                )
                                _draw_double_or_courtesy_accidental(
                                    canvas=canvas,
                                    stroke_mod=stroke_engine,
                                    glyph_mod=glyph_variants,
                                    acc=acc_kind,
                                    cx=ex - eff_sp * (1.35 if is_court else 1.25),
                                    cy=ny,
                                    sp=eff_sp,
                                    ink_rgb=ink_rgb,
                                    style=style,
                                    np_rng=np_rng,
                                    courtesy=is_court,
                                )

                            glyph_variants.draw_notehead(
                                canvas=canvas,
                                duration_kind=ev.duration_kind,
                                cx=hx,
                                cy=ny,
                                sp=eff_sp,
                                color_rgb=ink_rgb,
                                style=style,
                                rng=np_rng,
                            )

                            if ev.dots > 0 and not ev.is_grace:
                                dot_y = ny - (sp * 0.28 if st_step % 2 == 0 else 0.0)
                                dot_base_x = max(ex, hx) + sp * 0.95
                                for d_i in range(ev.dots):
                                    dx = dot_base_x + d_i * sp * 0.45
                                    stroke_engine.draw_filled_blob(
                                        canvas=canvas,
                                        cx=dx,
                                        cy=dot_y,
                                        rx=max(1.5, sp * 0.14),
                                        ry=max(1.5, sp * 0.14),
                                        angle_deg=0.0,
                                        color_rgb=ink_rgb,
                                        style=style,
                                        hollow=False,
                                    )

                        # Stem & Flag
                        tip_y = min(ys_note) if stem_dir == 1 else max(ys_note)
                        if ev.duration_kind != "whole":
                            anchor_y = min(ys_note) if stem_dir == 1 else max(ys_note)
                            far_y = max(ys_note) if stem_dir == 1 else min(ys_note)
                            stem_len = max(3.3 * eff_sp, abs(far_y - anchor_y) + 2.8 * eff_sp)
                            stem_x = ex + (eff_sp * 0.52 if stem_dir == 1 else -eff_sp * 0.52)
                            tip_y = anchor_y - stem_dir * stem_len

                            if ev.is_grace:
                                n_flags = max(1, {"eighth": 1, "16th": 2, "32nd": 3}.get(ev.duration_kind, 1))
                                glyph_variants.draw_stem_and_flag(
                                    canvas=canvas,
                                    x=stem_x,
                                    y0=far_y,
                                    y1=tip_y,
                                    sp=eff_sp,
                                    flags=n_flags,
                                    stem_up=True,
                                    color_rgb=ink_rgb,
                                    style=style,
                                    rng=np_rng,
                                )
                                if ev.grace_slash:
                                    sy_mid = 0.45 * far_y + 0.55 * tip_y
                                    slash_pts = np.array(
                                        [
                                            [stem_x - eff_sp * 0.75, sy_mid + eff_sp * 0.65],
                                            [stem_x + eff_sp * 0.85, sy_mid - eff_sp * 0.55],
                                        ],
                                        dtype=np.float64,
                                    )
                                    stroke_engine.draw_calligraphic_polyline(
                                        canvas, slash_pts, ink_rgb, max(0.9, eff_sp * 0.14), style, kind="stem"
                                    )
                            elif ev.beam_group is not None:
                                beam_groups.setdefault(ev.beam_group, []).append(
                                    (stem_x, far_y, tip_y, stem_dir, ev.duration_kind)
                                )
                            else:
                                n_flags = {"eighth": 1, "16th": 2, "32nd": 3}.get(ev.duration_kind, 0)
                                glyph_variants.draw_stem_and_flag(
                                    canvas=canvas,
                                    x=stem_x,
                                    y0=far_y,
                                    y1=tip_y,
                                    sp=sp,
                                    flags=n_flags,
                                    stem_up=(stem_dir == 1),
                                    color_rgb=ink_rgb,
                                    style=style,
                                    rng=np_rng,
                                )

                        # Articulations & Ornaments
                        if ev.articulations:
                            # Place on notehead side (opposite stem), except fermata/trill always above
                            art_above = (stem_dir == -1)
                            cur_y_above = min(min(ys_note), yc - 2.0 * sp, tip_y if stem_dir == 1 else min(ys_note)) - sp * 0.95
                            cur_y_below = max(max(ys_note), yc + 2.0 * sp, tip_y if stem_dir == -1 else max(ys_note)) + sp * 0.95
                            for art in ev.articulations:
                                if art in ("fermata", "trill"):
                                    ay = cur_y_above - (sp * 0.35 if art == "fermata" else sp * 0.2)
                                    _draw_articulation(
                                        canvas, stroke_engine, art, ex, ay, True, sp, ink_rgb, style, np_rng
                                    )
                                    cur_y_above -= sp * 1.15
                                elif art_above:
                                    _draw_articulation(
                                        canvas, stroke_engine, art, ex, cur_y_above, True, sp, ink_rgb, style, np_rng
                                    )
                                    cur_y_above -= sp * 0.85
                                else:
                                    _draw_articulation(
                                        canvas, stroke_engine, art, ex, cur_y_below, False, sp, ink_rgb, style, np_rng
                                    )
                                    cur_y_below += sp * 0.85

                        # Under-staff Lyrics
                        if ev.lyrics:
                            for verse_num, syll_txt in ev.lyrics:
                                ly_y = yc + sp * (4.25 + (verse_num - 1) * 1.35)
                                _draw_handwritten_text(
                                    canvas,
                                    stroke_engine,
                                    syll_txt,
                                    ex - sp * 0.45,
                                    ly_y,
                                    sp * 0.92,
                                    ink_rgb,
                                    style,
                                    np_rng,
                                    italic=False,
                                )

                        # Track for tuplet brackets
                        if ev.tuplet_id is not None and ev.tuplet_actual and not ev.is_grace:
                            top_ref = min(min(ys_note), tip_y) - sp * 0.95
                            bot_ref = max(max(ys_note), tip_y) + sp * 0.95
                            tuplet_groups.setdefault(ev.tuplet_id, []).append(
                                (ex, top_ref, bot_ref, ev.tuplet_actual, stem_dir)
                            )

                        rendered_notes[(m_idx, ev_idx)] = (
                            ex,
                            step_to_y,
                            stem_dir,
                            min(min(ys_note), tip_y),
                            max(max(ys_note), tip_y),
                        )

                    # Draw beamed stems + beams
                    for grp_items in beam_groups.values():
                        if len(grp_items) == 1:
                            sx, fy, ty, sdir, dk = grp_items[0]
                            n_flags = {"eighth": 1, "16th": 2, "32nd": 3}.get(dk, 1)
                            glyph_variants.draw_stem_and_flag(
                                canvas=canvas,
                                x=sx,
                                y0=fy,
                                y1=ty,
                                sp=sp,
                                flags=n_flags,
                                stem_up=(sdir == 1),
                                color_rgb=ink_rgb,
                                style=style,
                                rng=np_rng,
                            )
                            continue

                        x0, _, y0_raw, sdir, _ = grp_items[0]
                        x1, _, y1_raw, _, _ = grp_items[-1]
                        max_dy = min(sp * 1.4, abs(x1 - x0) * 0.22)
                        dy = max(-max_dy, min(max_dy, y1_raw - y0_raw))
                        y0 = y0_raw
                        y1 = y0 + dy
                        dx_span = max(1e-3, x1 - x0)
                        for sx, fy, ty, _, _ in grp_items:
                            t = (sx - x0) / dx_span
                            by = y0 + t * (y1 - y0)
                            if sdir == 1 and by > ty:
                                shift = ty - by
                                y0 += shift
                                y1 += shift
                            elif sdir == -1 and by < ty:
                                shift = ty - by
                                y0 += shift
                                y1 += shift

                        for sx, fy, _, _, _ in grp_items:
                            t = (sx - x0) / dx_span
                            by = y0 + t * (y1 - y0)
                            glyph_variants.draw_stem_and_flag(
                                canvas=canvas,
                                x=sx,
                                y0=fy,
                                y1=by,
                                sp=sp,
                                flags=0,
                                stem_up=(sdir == 1),
                                color_rgb=ink_rgb,
                                style=style,
                                rng=np_rng,
                            )

                        n_beams = max(
                            {"eighth": 1, "16th": 2, "32nd": 3}.get(it[4], 1) for it in grp_items
                        )
                        for b_i in range(n_beams):
                            b_off = b_i * sp * 0.62 * sdir
                            glyph_variants.draw_beam(
                                canvas=canvas,
                                x0=x0,
                                y0=y0 + b_off,
                                x1=x1,
                                y1=y1 + b_off,
                                sp=sp,
                                color_rgb=ink_rgb,
                                style=style,
                                rng=np_rng,
                            )

                    # Draw Tuplet Brackets
                    for t_items in tuplet_groups.values():
                        if len(t_items) < 2:
                            continue
                        t_items.sort(key=lambda it: it[0])
                        tx0 = t_items[0][0] - sp * 0.35
                        tx1 = t_items[-1][0] + sp * 0.35
                        t_num = t_items[0][3]
                        sdir_avg = sum(it[4] for it in t_items)
                        above = sdir_avg >= 0
                        if above:
                            ty = min(it[1] for it in t_items) - sp * 0.35
                        else:
                            ty = max(it[2] for it in t_items) + sp * 0.35
                        _draw_tuplet_bracket(
                            canvas, stroke_engine, glyph_variants, tx0, ty, tx1, ty, t_num, above, sp, ink_rgb, style, np_rng
                        )

                    # Dynamics
                    for dyn in meas.dynamics:
                        dx = beat_to_x(dyn.beat_offset)
                        dy = y_mid(dx) + (sp * 3.25 if dyn.placement != "above" else -sp * 3.15)
                        _draw_dynamic_marking(canvas, stroke_engine, dyn.text, dx, dy, sp, ink_rgb, style, np_rng)

                    # Tempo / Staff Text (above staff)
                    for tmp in meas.tempos:
                        tx = beat_to_x(tmp.beat_offset)
                        ty = float(y_fn_2(tx, -2)) - sp * 2.15
                        _draw_handwritten_text(
                            canvas, stroke_engine, tmp.text, tx - sp * 0.3, ty, sp * 1.05, ink_rgb, style, np_rng, bold=True
                        )

                # Measure right barline
                bar_y0 = float(y_fn_2(bx1, -2))
                bar_y1 = float(y_fn_2(bx1, 2))
                glyph_variants.draw_barline(
                    canvas=canvas,
                    x=bx1,
                    y_top=bar_y0,
                    y_bot=bar_y1,
                    sp=sp,
                    kind=meas.barline,
                    color_rgb=ink_rgb,
                    style=style,
                    rng=np_rng,
                )

            # Draw Ties & Slurs for this staff within this system
            m_lo, m_hi = sys_lay.measure_start, sys_lay.measure_end
            for span in part.ties + part.slurs:
                is_tie = span.kind == "tie"
                s_in = m_lo <= span.start_measure < m_hi
                e_in = m_lo <= span.end_measure < m_hi
                if not s_in and not e_in:
                    continue

                p_start = rendered_notes.get((span.start_measure, span.start_event_idx)) if s_in else None
                p_end = rendered_notes.get((span.end_measure, span.end_event_idx)) if e_in else None
                if p_start is None and p_end is None:
                    continue

                if p_start is not None and p_end is not None:
                    sx, smap, sdir, smin_y, smax_y = p_start
                    ex, emap, edir, emin_y, emax_y = p_end
                    arch_dir = 1 if sdir == 1 else -1
                    if is_tie:
                        sy = smap.get(span.start_step, y_mid(sx)) + arch_dir * sp * 0.42
                        ey = emap.get(span.end_step, y_mid(ex)) + arch_dir * sp * 0.42
                        sx += sp * 0.42
                        ex -= sp * 0.42
                    else:
                        sy = (smax_y + sp * 0.55) if arch_dir == 1 else (smin_y - sp * 0.55)
                        ey = (emax_y + sp * 0.55) if arch_dir == 1 else (emin_y - sp * 0.55)
                    _draw_slur_or_tie(
                        canvas, stroke_engine, sx, sy, ex, ey, arch_dir, sp, ink_rgb, style, np_rng, is_tie=is_tie
                    )
                elif p_start is not None:
                    # Crosses right system edge: draw outgoing half-arc
                    sx, smap, sdir, smin_y, smax_y = p_start
                    arch_dir = 1 if sdir == 1 else -1
                    sy = smap.get(span.start_step, y_mid(sx)) + arch_dir * sp * 0.45
                    _draw_slur_or_tie(
                        canvas, stroke_engine, sx + sp * 0.35, sy, sys_lay.x_right, sy, arch_dir, sp, ink_rgb, style, np_rng, is_tie=is_tie
                    )
                elif p_end is not None:
                    # Crosses left system edge: draw incoming half-arc
                    ex, emap, edir, emin_y, emax_y = p_end
                    arch_dir = 1 if edir == 1 else -1
                    ey = emap.get(span.end_step, y_mid(ex)) + arch_dir * sp * 0.45
                    _draw_slur_or_tie(
                        canvas, stroke_engine, sys_lay.x_left + sp * 2.6, ey, ex - sp * 0.35, ey, arch_dir, sp, ink_rgb, style, np_rng, is_tie=is_tie
                    )

            # Draw Hairpins (< / >) for this staff within this system
            for hp in part.hairpins:
                if hp.end_measure < m_lo or hp.start_measure >= m_hi:
                    continue
                if hp.start_measure in measure_beat_to_x:
                    hx0 = measure_beat_to_x[hp.start_measure](hp.start_beat)
                else:
                    hx0 = sys_lay.x_left + sp * 3.0
                if hp.end_measure in measure_beat_to_x:
                    hx1 = measure_beat_to_x[hp.end_measure](hp.end_beat)
                else:
                    hx1 = sys_lay.x_right - sp * 0.5
                if hx1 <= hx0 + sp * 0.5:
                    hx1 = min(sys_lay.x_right - sp * 0.3, hx0 + sp * 3.5)
                hy = y_mid(0.5 * (hx0 + hx1)) + (sp * 3.35 if hp.placement != "above" else -sp * 3.15)
                _draw_hairpin(canvas, stroke_engine, hp.kind, hx0, hx1, hy, sp, ink_rgb, style)

    # 4. Composite & save output PNG + companion Stage-1 GT JSON
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_img = canvas.to_pil()
    final_img.save(out_path)

    gt_payload: Dict[str, Any] = {
        "score_title": score.title,
        "source_path": str(score_path),
        "style": style_name,
        "seed": seed,
        "page_w": page_w,
        "page_h": page_h,
        "systems": gt_systems,
    }

    if export_train_dir is not None:
        train_exports = _export_training_systems(
            score=score,
            score_path=Path(score_path),
            final_img=final_img,
            systems=systems,
            export_dir=Path(export_train_dir),
            stem=out_path.stem,
        )
        gt_payload["train_exports"] = train_exports

    gt_path = out_path.with_suffix(".json")
    gt_path.write_text(json.dumps(gt_payload, indent=2), encoding="utf-8")
    return gt_payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a MusicXML or MuseScore (.mscx/.mscz) file in handwritten styles."
    )
    parser.add_argument("--input", required=True, help="Path to .musicxml, .mxl, .mscx, or .mscz")
    parser.add_argument(
        "--styles",
        default="beethoven_hurried_autograph,debussy_pencil_sketch,student_ballpoint_blue",
        help="Comma-separated style names or 'all' for all 20 styles",
    )
    parser.add_argument(
        "--out-dir",
        default="tmp-train/hw-engine/mvp_samples",
        help="Directory to write rendered PNGs and GT JSON files",
    )
    parser.add_argument(
        "--export-train-dir",
        default=None,
        help="Optional directory to export cropped Stage-2 system strips + exact assembly-serialize bekern targets",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    parser.add_argument("--page-w", type=int, default=1650, help="Output page width in pixels")
    parser.add_argument("--page-h", type=int, default=2200, help="Output page height in pixels")
    args = parser.parse_args(argv)

    import styles as styles_mod

    if args.styles.strip().lower() == "all":
        style_names = list(styles_mod.SCRIBE_STYLES.keys())
    else:
        style_names = [s.strip() for s in args.styles.split(",") if s.strip()]

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, sname in enumerate(style_names):
        out_png = out_dir / f"{in_path.stem}__{sname}.png"
        gt = render_score_handwritten(
            score_path=in_path,
            style_name=sname,
            out_png=out_png,
            seed=args.seed + idx,
            page_w=args.page_w,
            page_h=args.page_h,
            export_train_dir=args.export_train_dir,
        )
        extra = ""
        if args.export_train_dir:
            n_exp = len(gt.get("train_exports", []))
            n_ok = sum(1 for e in gt.get("train_exports", []) if e.get("bekern_ok"))
            extra = f", train_strips={n_exp} (bekern_ok={n_ok})"
        print(
            f"[ok] {sname}: {out_png} "
            f"({len(gt['systems'])} systems, gt={out_png.with_suffix('.json').name}{extra})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
