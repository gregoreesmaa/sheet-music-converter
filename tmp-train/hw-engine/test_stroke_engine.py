"""Unit tests for the handwriting stroke engine and polymorphic glyph variants."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from PIL import Image

# Allow running via `python -m unittest tmp-train/hw-engine/test_stroke_engine.py`
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from stroke_engine import (  # noqa: E402
    StrokeCanvas,
    draw_calligraphic_polyline,
    draw_filled_blob,
    draw_staff_line,
)
from glyph_variants import (  # noqa: E402
    CLEF_ARCHETYPES,
    EIGHTH_REST_ARCHETYPES,
    NOTEHEAD_ARCHETYPES,
    QUARTER_REST_ARCHETYPES,
    draw_accidental,
    draw_barline,
    draw_beam,
    draw_brace,
    draw_clef,
    draw_notehead,
    draw_rest,
    draw_stem_and_flag,
    draw_time_sig,
)


BG_CREAM = np.array([0.96, 0.94, 0.89], dtype=np.float32)
INK_DARK = np.array([0.08, 0.07, 0.06], dtype=np.float32)

STYLE_BROAD_NIB = {
    "nib_angle_deg": 38.0,
    "nib_min_ratio": 0.18,
    "wobble_px": 0.8,
    "tremor_px": 0.25,
    "ink_dropout": 0.04,
    "grain_strength": 0.08,
    "ink_var": 0.06,
    "ink_pool": 0.35,
    "stroke_opacity": 0.95,
}

STYLE_PENCIL = {
    "nib_angle_deg": 30.0,
    "nib_min_ratio": 0.85,
    "wobble_px": 0.9,
    "tremor_px": 0.45,
    "ink_dropout": 0.28,
    "grain_strength": 0.62,
    "ink_var": 0.08,
    "ink_pool": 0.0,
    "stroke_opacity": 0.68,
}

STYLE_BALLPOINT = {
    "nib_angle_deg": 45.0,
    "nib_min_ratio": 0.92,
    "wobble_px": 0.5,
    "tremor_px": 0.18,
    "ink_dropout": 0.02,
    "grain_strength": 0.03,
    "ink_var": 0.03,
    "ink_pool": 0.08,
    "stroke_opacity": 0.96,
}


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


class TestStrokeEngineAndGlyphs(unittest.TestCase):
    def test_canvas_exports_and_edge_cases(self) -> None:
        rng = np.random.default_rng(42)
        canvas = StrokeCanvas(120, 80, BG_CREAM, rng)
        arr = canvas.to_uint8()
        self.assertEqual(arr.shape, (80, 120, 3))
        self.assertEqual(arr.dtype, np.uint8)
        pil_img = canvas.to_pil()
        self.assertIsInstance(pil_img, Image.Image)
        self.assertEqual(pil_img.size, (120, 80))

        # Edge cases: empty polyline, single point, out-of-bounds.
        draw_calligraphic_polyline(canvas, np.zeros((0, 2)), INK_DARK, 2.0, STYLE_BROAD_NIB)
        draw_calligraphic_polyline(canvas, np.array([[50.0, 40.0]]), INK_DARK, 3.0, STYLE_BROAD_NIB)
        draw_calligraphic_polyline(canvas, np.array([[-50.0, -50.0], [-10.0, -10.0]]), INK_DARK, 2.0, STYLE_BROAD_NIB)
        self.assertGreater(_mean_abs_diff(canvas.buf, np.broadcast_to(BG_CREAM, canvas.buf.shape)), 0.0)

    def test_quarter_rest_all_6_archetypes_unique_per_draw(self) -> None:
        self.assertGreaterEqual(len(QUARTER_REST_ARCHETYPES), 6)
        for idx, arch in enumerate(QUARTER_REST_ARCHETYPES):
            style = dict(STYLE_BROAD_NIB)
            style["force_rest_archetype"] = arch

            c1 = StrokeCanvas(96, 128, BG_CREAM, np.random.default_rng(100 + idx * 2))
            c2 = StrokeCanvas(96, 128, BG_CREAM, np.random.default_rng(101 + idx * 2))
            r1 = np.random.default_rng(200 + idx * 2)
            r2 = np.random.default_rng(201 + idx * 2)

            draw_rest(c1, "quarter", 48.0, 64.0, 14.0, INK_DARK, style, r1)
            draw_rest(c2, "quarter", 48.0, 64.0, 14.0, INK_DARK, style, r2)

            bg = np.broadcast_to(BG_CREAM, c1.buf.shape)
            self.assertGreater(_mean_abs_diff(c1.buf, bg), 1e-4, f"{arch} drew nothing on c1")
            self.assertGreater(_mean_abs_diff(c2.buf, bg), 1e-4, f"{arch} drew nothing on c2")
            diff = _mean_abs_diff(c1.buf, c2.buf)
            self.assertGreater(diff, 1e-4, f"Quarter rest archetype {arch} did not vary across RNG steps")

    def test_notehead_all_archetypes_unique_per_draw(self) -> None:
        for dur in ("filled", "half", "whole"):
            for idx, arch in enumerate(NOTEHEAD_ARCHETYPES):
                style = dict(STYLE_BROAD_NIB)
                style["notehead_archetype"] = arch
                c1 = StrokeCanvas(64, 64, BG_CREAM, np.random.default_rng(300 + idx))
                c2 = StrokeCanvas(64, 64, BG_CREAM, np.random.default_rng(400 + idx))
                r1 = np.random.default_rng(500 + idx)
                r2 = np.random.default_rng(600 + idx)

                draw_notehead(c1, dur, 32.0, 32.0, 14.0, INK_DARK, style, r1)
                draw_notehead(c2, dur, 32.0, 32.0, 14.0, INK_DARK, style, r2)

                bg = np.broadcast_to(BG_CREAM, c1.buf.shape)
                self.assertGreater(_mean_abs_diff(c1.buf, bg), 1e-4, f"{dur}/{arch} drew nothing")
                self.assertGreater(_mean_abs_diff(c1.buf, c2.buf), 1e-4, f"{dur}/{arch} was identical")

    def test_all_glyph_functions_render_cleanly(self) -> None:
        rng = np.random.default_rng(777)
        sp = 14.0

        def _assert_draws(fn, w: int = 120, h: int = 140) -> None:
            c = StrokeCanvas(w, h, BG_CREAM, np.random.default_rng(int(rng.integers(1, 100000))))
            fn(c)
            bg = np.broadcast_to(BG_CREAM, c.buf.shape)
            self.assertGreater(_mean_abs_diff(c.buf, bg), 1e-5)

        # Staff lines (hand-drawn & pre-ruled)
        _assert_draws(lambda c: draw_staff_line(c, 10, 110, lambda x: 50.0 + 0.01 * x, INK_DARK, 1.8, STYLE_BROAD_NIB))
        _assert_draws(lambda c: draw_staff_line(c, 10, 110, lambda x: 50.0, INK_DARK, 1.5, {"staff_pre_ruled": True}))

        # Filled & hollow blobs with open_gap
        _assert_draws(lambda c: draw_filled_blob(c, 60, 70, 9, 6, -30, INK_DARK, STYLE_BROAD_NIB, hollow=False))
        _assert_draws(lambda c: draw_filled_blob(c, 60, 70, 9, 6, -25, INK_DARK, STYLE_BROAD_NIB, hollow=True, open_gap=0.14))

        # All rest types + eighth-rest archetypes
        for rt in ("whole", "half", "quarter", "eighth", "16th", "32nd"):
            _assert_draws(lambda c, r=rt: draw_rest(c, r, 60, 70, sp, INK_DARK, STYLE_BROAD_NIB, rng))
        for earch in EIGHTH_REST_ARCHETYPES:
            st = dict(STYLE_BROAD_NIB, eighth_rest_archetype_weights={earch: 1.0})
            _assert_draws(lambda c, s=st: draw_rest(c, "eighth", 60, 70, sp, INK_DARK, s, rng))

        # Stems & flags (up/down, 0..3 flags)
        for up in (True, False):
            for fl in (0, 1, 2, 3):
                y0, y1 = (90.0, 40.0) if up else (40.0, 90.0)
                _assert_draws(lambda c, u=up, f=fl, a=y0, b=y1: draw_stem_and_flag(c, 50.0, a, b, sp, f, u, INK_DARK, STYLE_BROAD_NIB, rng))

        # Beams
        _assert_draws(lambda c: draw_beam(c, 20.0, 50.0, 100.0, 42.0, sp, INK_DARK, STYLE_BROAD_NIB, rng))

        # Clefs across all names and archetypes
        for clef in ("G-2", "F-4", "C-3", "C-4"):
            for carch in CLEF_ARCHETYPES:
                st = dict(STYLE_BROAD_NIB, clef_archetype_weights={carch: 1.0})
                _assert_draws(lambda c, cl=clef, s=st: draw_clef(c, cl, 60.0, 70.0, sp, INK_DARK, s, rng))

        # Accidentals
        for acc in ("sharp", "flat", "natural"):
            _assert_draws(lambda c, a=acc: draw_accidental(c, a, 60.0, 70.0, sp, INK_DARK, STYLE_BROAD_NIB, rng))

        # Time signatures
        for num, den in ((2, 4), (3, 4), (4, 4), (6, 8), (9, 8), (12, 8), ("C", 0), ("cut", 0)):
            _assert_draws(lambda c, n=num, d=den: draw_time_sig(c, n, d, 60.0, 70.0, sp, INK_DARK, STYLE_BROAD_NIB, rng))

        # Barlines
        for kind in ("single", "double", "final", "repeat-right", "repeat-left"):
            _assert_draws(lambda c, k=kind: draw_barline(c, 60.0, 30.0, 100.0, sp, k, INK_DARK, STYLE_BROAD_NIB, rng))

        # Braces (curly & bracket)
        _assert_draws(lambda c: draw_brace(c, 70.0, 20.0, 120.0, sp, INK_DARK, STYLE_BROAD_NIB, rng))
        _assert_draws(lambda c: draw_brace(c, 70.0, 20.0, 120.0, sp, INK_DARK, dict(STYLE_BROAD_NIB, brace_kind="bracket"), rng))

    def test_broad_nib_vs_ballpoint_stroke_contrast(self) -> None:
        """Broad-nib must show much higher thick-to-thin anisotropy than ballpoint."""
        # Draw two strokes: one aligned with nib angle (38 deg -> thin) and one perpendicular (128 deg -> thick).
        def _measure_anisotropy(style: dict) -> float:
            ang_thin = np.deg2rad(style["nib_angle_deg"])
            ang_thick = ang_thin + 0.5 * np.pi
            length = 70.0

            def _ink_mass(ang: float) -> float:
                c = StrokeCanvas(120, 120, BG_CREAM, np.random.default_rng(1234))
                p0 = np.array([60.0 - 0.5 * length * np.cos(ang), 60.0 - 0.5 * length * np.sin(ang)])
                p1 = np.array([60.0 + 0.5 * length * np.cos(ang), 60.0 + 0.5 * length * np.sin(ang)])
                draw_calligraphic_polyline(c, np.vstack((p0, p1)), INK_DARK, 8.0, style)
                return float(np.sum(np.mean(BG_CREAM - c.buf, axis=2)))

            thin_mass = _ink_mass(ang_thin)
            thick_mass = _ink_mass(ang_thick)
            return thick_mass / max(thin_mass, 1e-6)

        nib_ratio = _measure_anisotropy(STYLE_BROAD_NIB)
        ball_ratio = _measure_anisotropy(STYLE_BALLPOINT)
        self.assertGreater(nib_ratio, 2.2, f"Broad-nib anisotropy too low: {nib_ratio:.2f}")
        self.assertLess(ball_ratio, 1.35, f"Ballpoint anisotropy too high: {ball_ratio:.2f}")
        self.assertGreater(nib_ratio, ball_ratio * 1.8)

    def test_pencil_vs_ballpoint_texture_and_density(self) -> None:
        """Pencil must have lighter density and higher interior grain variance than ballpoint."""
        pts = np.array([[20.0, 60.0], [180.0, 60.0]])

        c_pencil = StrokeCanvas(200, 120, BG_CREAM, np.random.default_rng(99))
        c_ball = StrokeCanvas(200, 120, BG_CREAM, np.random.default_rng(99))
        draw_calligraphic_polyline(c_pencil, pts, INK_DARK, 10.0, STYLE_PENCIL)
        draw_calligraphic_polyline(c_ball, pts, INK_DARK, 10.0, STYLE_BALLPOINT)

        # Sample interior band of the horizontal stroke (y in 58..63, x in 45..155).
        band_p = np.mean(BG_CREAM - c_pencil.buf[58:63, 45:155], axis=2)
        band_b = np.mean(BG_CREAM - c_ball.buf[58:63, 45:155], axis=2)

        # Pencil deposits lighter average ink and has noticeably higher spatial grain std.
        self.assertLess(float(np.mean(band_p)), float(np.mean(band_b)) * 0.85)
        self.assertGreater(float(np.std(band_p)), float(np.std(band_b)) * 2.0)


if __name__ == "__main__":
    unittest.main()
