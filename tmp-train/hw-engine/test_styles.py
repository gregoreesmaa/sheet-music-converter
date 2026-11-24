"""Unit tests for the 20-style scribe registry and layout/paper quirks engine."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

# Ensure sibling modules in tmp-train/hw-engine are importable regardless of CWD
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from layout_quirks import (
    build_curved_staff_y,
    compute_quirky_note_positions,
    make_manuscript_paper,
)
from styles import (
    NOTEHEAD_ARCHETYPES,
    REQUIRED_STYLE_KEYS,
    REST_ARCHETYPES,
    SCRIBE_STYLES,
    get_style,
    list_styles,
    validate_style,
)

EXPECTED_20_STYLES = [
    "baroque_quill_neat",
    "beethoven_hurried_autograph",
    "mozart_flowing_cursive",
    "schubert_dense_lieder",
    "chopin_delicate_salon",
    "debussy_pencil_sketch",
    "bach_thick_iron_gall",
    "romantic_copyist_oblong",
    "modern_jazz_lead_sheet",
    "student_ballpoint_blue",
    "student_pencil_eraser",
    "fountain_pen_calligraphy",
    "faded_sepia_archival",
    "cramped_pocket_notebook",
    "wide_conductor_score",
    "left_handed_backslant",
    "tremor_elderly_hand",
    "minimalist_quick_draft",
    "vintage_purple_hectograph",
    "choral_director_red_pencil",
]


class TestScribeStylesRegistry(unittest.TestCase):
    def test_at_least_20_required_styles_present(self) -> None:
        self.assertGreaterEqual(len(SCRIBE_STYLES), 20)
        names = list_styles()
        self.assertEqual(len(names), len(SCRIBE_STYLES))
        for expected in EXPECTED_20_STYLES:
            self.assertIn(expected, SCRIBE_STYLES, f"Missing required style {expected!r}")

    def test_get_style_returns_independent_copy_and_errors_on_unknown(self) -> None:
        s1 = get_style("baroque_quill_neat")
        s1["slant_deg"] = 999.0
        s1["rest_archetype_weights"]["lightning_z"] = 999.0
        s2 = get_style("baroque_quill_neat")
        self.assertNotEqual(s2["slant_deg"], 999.0)
        self.assertNotEqual(s2["rest_archetype_weights"]["lightning_z"], 999.0)
        with self.assertRaises(KeyError):
            get_style("nonexistent_scribe_style")

    def test_every_style_has_all_required_keys_and_valid_ranges(self) -> None:
        for style_name, s in SCRIBE_STYLES.items():
            with self.subTest(style=style_name):
                self.assertEqual(s["name"], style_name)
                for key in REQUIRED_STYLE_KEYS:
                    self.assertIn(key, s, f"Style {style_name!r} missing key {key!r}")

                # RGB tuples in [0, 1]
                for rgb_key in ("ink_rgb", "staff_ink_rgb", "paper_rgb"):
                    rgb = s[rgb_key]
                    self.assertEqual(len(rgb), 3, f"{style_name}.{rgb_key} must have 3 channels")
                    for ch in rgb:
                        self.assertGreaterEqual(ch, 0.0)
                        self.assertLessEqual(ch, 1.0)

                # Nib & stroke bounds
                self.assertGreaterEqual(s["nib_min_ratio"], 0.15)
                self.assertLessEqual(s["nib_min_ratio"], 0.95)
                self.assertGreater(s["stroke_scale"], 0.3)
                self.assertLess(s["stroke_scale"], 2.5)
                self.assertGreaterEqual(s["wobble_px"], 0.0)
                self.assertGreaterEqual(s["tremor_px"], 0.0)
                self.assertTrue(0.0 <= s["ink_dropout"] <= 0.6)
                self.assertTrue(0.0 <= s["grain_strength"] <= 0.8)
                self.assertTrue(0.0 <= s["ink_var"] <= 0.6)
                self.assertTrue(0.0 <= s["ink_pooling"] <= 0.8)

                # Glyph quirks
                self.assertTrue(-14.0 <= s["slant_deg"] <= 18.0)
                self.assertIn(s["notehead_archetype"], NOTEHEAD_ARCHETYPES)
                self.assertTrue(0.50 <= s["notehead_scale"] <= 1.45)
                if "notehead_aspect" in s:
                    self.assertTrue(0.25 <= s["notehead_aspect"] <= 2.20)
                if "notehead_fill_density" in s:
                    self.assertTrue(0.25 <= s["notehead_fill_density"] <= 1.50)
                self.assertTrue(0.0 <= s["hollow_open_prob"] <= 1.0)
                self.assertTrue(0.0 <= s["clef_ornate"] <= 1.0)
                self.assertGreaterEqual(s["barline_overshoot_sp"], 0.0)

                # Rest archetype weights
                rw = s["rest_archetype_weights"]
                self.assertEqual(set(rw.keys()), set(REST_ARCHETYPES))
                for val in rw.values():
                    self.assertGreaterEqual(val, 0.0)
                self.assertAlmostEqual(sum(rw.values()), 1.0, places=4)

                # Spacing & layout quirks
                self.assertTrue(0.08 <= s["spacing_jitter"] <= 0.45)
                self.assertTrue(0.10 <= s["bar_width_var"] <= 0.35)
                self.assertTrue(0.0 <= s["end_bar_cram"] <= 0.40)
                self.assertGreaterEqual(s["staff_curve_px"], 0.0)
                self.assertGreaterEqual(s["sp_jitter_frac"], 0.0)
                self.assertTrue(0.0 <= s["blank_prerule_prob"] <= 0.60)

                # Paper & artifacts
                self.assertTrue(0.0 <= s["paper_grain"] <= 0.30)
                self.assertTrue(0.0 <= s["paper_mottle"] <= 0.40)
                self.assertTrue(0.0 <= s["bleed_through"] <= 0.25)
                self.assertTrue(0.0 <= s["eraser_smudge_prob"] <= 1.0)
                self.assertTrue(0.0 <= s["margin_annotations_prob"] <= 1.0)

    def test_pairwise_style_diversity(self) -> None:
        """No two styles may be identical or near-identical in parameter space."""
        names = list_styles()
        numeric_keys = [
            "nib_min_ratio",
            "stroke_scale",
            "wobble_px",
            "tremor_px",
            "ink_dropout",
            "grain_strength",
            "slant_deg",
            "notehead_scale",
            "clef_ornate",
            "spacing_jitter",
            "end_bar_cram",
            "staff_curve_px",
            "blank_prerule_prob",
            "paper_mottle",
            "bleed_through",
            "eraser_smudge_prob",
        ]

        # Build normalized feature vectors
        raw_vecs = []
        for name in names:
            s = SCRIBE_STYLES[name]
            vec = (
                list(s["ink_rgb"])
                + list(s["paper_rgb"])
                + [float(s[k]) for k in numeric_keys]
                + [float(s["rest_archetype_weights"][rk]) for rk in REST_ARCHETYPES]
            )
            raw_vecs.append(vec)
        mat = np.asarray(raw_vecs, dtype=np.float64)
        stds = np.std(mat, axis=0)
        stds[stds < 1e-8] = 1.0
        normed = (mat - np.mean(mat, axis=0)) / stds

        # Verify broad coverage across notehead archetypes and ink colors
        archetypes_used = {SCRIBE_STYLES[n]["notehead_archetype"] for n in names}
        self.assertTrue(archetypes_used.issubset(set(NOTEHEAD_ARCHETYPES)))
        self.assertGreaterEqual(len(archetypes_used), 6)

        min_dist = float("inf")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                dist = float(np.linalg.norm(normed[i] - normed[j]))
                min_dist = min(min_dist, dist)
                self.assertGreater(
                    dist,
                    1.0,
                    f"Styles {names[i]!r} and {names[j]!r} are too similar (norm dist={dist:.3f})",
                )
        self.assertGreater(min_dist, 1.0)

    def test_18_notehead_archetypes_and_optional_notehead_keys_validation(self) -> None:
        expected_18 = (
            "oval",
            "slash",
            "teardrop",
            "compact",
            "heavy_wedge",
            "hollow_diamond",
            "slim_oval",
            "needle_slash",
            "hairline_dash",
            "vertical_tick",
            "pinprick_dot",
            "slim_diamond",
            "boomerang_hook",
            "open_loop",
            "zigzag_fill",
            "broad_nib_lozenge",
            "ink_blot",
            "chevron_wedge",
        )
        self.assertEqual(NOTEHEAD_ARCHETYPES, expected_18)
        self.assertEqual(len(NOTEHEAD_ARCHETYPES), 18)
        self.assertNotIn("notehead_aspect", REQUIRED_STYLE_KEYS)
        self.assertNotIn("notehead_fill_density", REQUIRED_STYLE_KEYS)

        base = get_style("baroque_quill_neat")
        # Valid boundary values pass
        for arch in expected_18:
            spec = dict(base, notehead_archetype=arch, notehead_aspect=0.25, notehead_fill_density=0.25, notehead_scale=0.50)
            out = validate_style(spec, name=f"test_{arch}")
            self.assertEqual(out["notehead_archetype"], arch)
            self.assertAlmostEqual(out["notehead_aspect"], 0.25)
            self.assertAlmostEqual(out["notehead_fill_density"], 0.25)
            self.assertAlmostEqual(out["notehead_scale"], 0.50)

        out_hi = validate_style(
            dict(base, notehead_aspect=2.20, notehead_fill_density=1.50, notehead_scale=1.45),
            name="test_hi",
        )
        self.assertAlmostEqual(out_hi["notehead_aspect"], 2.20)
        self.assertAlmostEqual(out_hi["notehead_fill_density"], 1.50)
        self.assertAlmostEqual(out_hi["notehead_scale"], 1.45)

        # Out-of-range values fail validation
        for bad_kwargs in (
            {"notehead_aspect": 0.20},
            {"notehead_aspect": 2.50},
            {"notehead_fill_density": 0.10},
            {"notehead_fill_density": 1.80},
            {"notehead_scale": 0.40},
            {"notehead_scale": 1.60},
        ):
            with self.subTest(bad=bad_kwargs):
                with self.assertRaises(ValueError):
                    validate_style(dict(base, **bad_kwargs), name="test_bad")


class TestLayoutQuirks(unittest.TestCase):
    def test_make_manuscript_paper_shape_dtype_range_and_determinism(self) -> None:
        style = get_style("faded_sepia_archival")
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = make_manuscript_paper(320, 240, style, rng1)
        p2 = make_manuscript_paper(320, 240, style, rng2)

        self.assertEqual(p1.shape, (240, 320, 3))
        self.assertEqual(p1.dtype, np.float32)
        self.assertGreaterEqual(float(p1.min()), 0.0)
        self.assertLessEqual(float(p1.max()), 1.0)
        # Paper should have non-trivial texture (not a flat solid color)
        self.assertGreater(float(p1.std()), 0.01)
        np.testing.assert_array_equal(p1, p2)

    def test_make_manuscript_paper_distinct_across_styles(self) -> None:
        p_archival = make_manuscript_paper(160, 120, get_style("faded_sepia_archival"), np.random.default_rng(7))
        p_blue = make_manuscript_paper(160, 120, get_style("student_ballpoint_blue"), np.random.default_rng(7))
        # Archival paper is noticeably warmer/darker than bright student notebook paper
        self.assertLess(float(p_archival.mean()), float(p_blue.mean()) - 0.05)

    def test_compute_quirky_note_positions_non_uniform_and_strictly_monotonic(self) -> None:
        style = get_style("beethoven_hurried_autograph")
        rng = np.random.default_rng(123)
        beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        bar_x0, bar_x1, sp = 100.0, 500.0, 12.0

        xs = compute_quirky_note_positions(beats, bar_x0, bar_x1, sp, style, rng)
        self.assertEqual(len(xs), len(beats))
        self.assertGreaterEqual(xs[0], bar_x0)
        self.assertLessEqual(xs[-1], bar_x1)

        gaps = np.diff(xs)
        # Strictly monotonic with non-overlapping gap
        self.assertTrue(np.all(gaps > 0.0))
        self.assertGreaterEqual(float(gaps.min()), 0.85 * sp - 1e-6)
        # Non-uniform spacing (standard deviation of gaps clearly > 0)
        self.assertGreater(float(np.std(gaps)), 2.0)

    def test_compute_quirky_note_positions_cramming_effect_and_edge_cases(self) -> None:
        cram_style = get_style("cramped_pocket_notebook")
        beats = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        early_gaps = []
        late_gaps = []
        for seed in range(30):
            xs = compute_quirky_note_positions(beats, 0.0, 400.0, 10.0, cram_style, np.random.default_rng(seed))
            gaps = np.diff(xs)
            self.assertTrue(np.all(gaps > 0.0))
            early_gaps.append(gaps[0])
            late_gaps.append(gaps[-1])
        # On average, end_bar_cram makes early gaps wider than late gaps
        self.assertGreater(float(np.mean(early_gaps)), float(np.mean(late_gaps)))

        # Edge cases: empty and single note
        self.assertEqual(compute_quirky_note_positions([], 0.0, 200.0, 10.0, cram_style, np.random.default_rng(1)), [])
        single = compute_quirky_note_positions([0.0], 50.0, 250.0, 10.0, cram_style, np.random.default_rng(1))
        self.assertEqual(len(single), 1)
        self.assertTrue(50.0 < single[0] < 250.0)

    def test_build_curved_staff_y_exact_bezier_control_points(self) -> None:
        style = get_style("tremor_elderly_hand")
        rng = np.random.default_rng(99)
        x0, x1, y_center, sp = 80.0, 980.0, 300.0, 14.0

        y_fn, ctrl_pts, sp_out = build_curved_staff_y(x0, x1, y_center, sp, style, rng)
        self.assertEqual(ctrl_pts.shape, (4, 2))
        self.assertAlmostEqual(sp_out, sp, places=6)

        # Verify exact Bezier evaluation at multiple t values matches y_fn(x, 0)
        ts = np.linspace(0.0, 1.0, 25)
        for t in ts:
            omt = 1.0 - t
            bx = (
                omt**3 * ctrl_pts[0, 0]
                + 3 * omt**2 * t * ctrl_pts[1, 0]
                + 3 * omt * t**2 * ctrl_pts[2, 0]
                + t**3 * ctrl_pts[3, 0]
            )
            by = (
                omt**3 * ctrl_pts[0, 1]
                + 3 * omt**2 * t * ctrl_pts[1, 1]
                + 3 * omt * t**2 * ctrl_pts[2, 1]
                + t**3 * ctrl_pts[3, 1]
            )
            self.assertAlmostEqual(y_fn(bx, 0), by, places=6)

        # Verify vector evaluation and monotonic ordering across the 5 staff lines [-2..2]
        xs_vec = np.linspace(x0, x1, 50)
        line_ys = [y_fn(xs_vec, idx) for idx in (-2, -1, 0, 1, 2)]
        for i in range(4):
            diffs = line_ys[i + 1] - line_ys[i]
            self.assertTrue(np.all(diffs > 0.5 * sp), "Staff lines must remain well-separated and ordered")


if __name__ == "__main__":
    unittest.main()
