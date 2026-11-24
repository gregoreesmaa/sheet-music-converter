"""Unit tests for the pluggable style architecture and high-variance handwriting engine."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Ensure sibling modules in tmp-train/hw-engine are importable regardless of CWD
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from glyph_variants import (  # noqa: E402
    NOTEHEAD_ARCHETYPES,
    QUARTER_REST_ARCHETYPES,
    draw_correction_scratch,
    draw_dynamic_text,
    draw_fermata,
    draw_hairpin,
    draw_notehead,
    draw_rest,
    draw_slur_or_tie,
)
from layout_quirks import (  # noqa: E402
    PAPER_TYPES,
    build_curved_staff_y,
    draw_margin_annotations,
    make_manuscript_paper,
)
from stroke_engine import (  # noqa: E402
    StrokeCanvas,
    draw_calligraphic_polyline,
    draw_rastrum_staff,
)
from styles import (  # noqa: E402
    REQUIRED_STYLE_KEYS,
    REST_ARCHETYPES,
    SCRIBE_STYLES,
    get_style,
    interpolate_styles,
    list_styles,
    register_style,
    reload_style_packs,
    sample_random_scribe,
)

BG_CREAM = np.array([0.96, 0.94, 0.89], dtype=np.float32)
INK_DARK = np.array([0.08, 0.07, 0.06], dtype=np.float32)


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


class TestPluggableStyleArchitecture(unittest.TestCase):
    def tearDown(self) -> None:
        # Always restore registry to built-ins + any real style_packs on disk.
        reload_style_packs(reset_to_builtins=True)

    def test_json_style_pack_auto_loading_and_inheritance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir) / "custom_pack.json"
            pack_data = {
                "styles": {
                    "custom_ozalid_jazz": {
                        "extends": "modern_jazz_lead_sheet",
                        "description": "Custom ozalid blueprint jazz chart.",
                        "paper_type": "ozalid_blue",
                        "ink_feather": 0.35,
                        "rastrum_wobble": 1.4,
                        "tags": ["jazz", "blueprint", "custom_test"],
                        "rest_archetype_weights": {
                            "modern_tall_z": 0.65,
                            "sharp_v_tick": 0.35,
                        },
                    },
                    "custom_child_of_child": {
                        "extends": "custom_ozalid_jazz",
                        "description": "Multi-level inherited scribe.",
                        "slant_deg": 12.0,
                        "tags": ["jazz", "nested_test"],
                    },
                }
            }
            pack_path.write_text(json.dumps(pack_data), encoding="utf-8")

            loaded = reload_style_packs(packs_dir=tmpdir, reset_to_builtins=True)
            self.assertEqual(set(loaded), {"custom_ozalid_jazz", "custom_child_of_child"})

            s1 = get_style("custom_ozalid_jazz")
            for key in REQUIRED_STYLE_KEYS:
                self.assertIn(key, s1)
            self.assertEqual(s1["paper_type"], "ozalid_blue")
            self.assertAlmostEqual(s1["ink_feather"], 0.35, places=6)
            self.assertAlmostEqual(s1["rastrum_wobble"], 1.4, places=6)
            self.assertEqual(set(s1["rest_archetype_weights"].keys()), set(REST_ARCHETYPES))
            self.assertAlmostEqual(sum(s1["rest_archetype_weights"].values()), 1.0, places=4)
            self.assertNotIn("extends", s1)

            # Multi-level inheritance preserves parent overrides while applying own overrides
            s2 = get_style("custom_child_of_child")
            self.assertEqual(s2["paper_type"], "ozalid_blue")
            self.assertAlmostEqual(s2["slant_deg"], 12.0, places=6)
            self.assertAlmostEqual(s2["ink_feather"], 0.35, places=6)

            # Tag filtering works for explicit tags and era
            self.assertIn("custom_ozalid_jazz", list_styles(tag="blueprint"))
            self.assertIn("custom_child_of_child", list_styles(tag="nested_test"))
            self.assertIn("baroque_quill_neat", list_styles(tag="baroque"))

    def test_register_style_validation_and_cycle_detection(self) -> None:
        # Missing required keys without extends must fail
        with self.assertRaises(ValueError):
            register_style("bad_incomplete", {"description": "missing keys"})

        # Unknown parent must fail
        with self.assertRaises(ValueError):
            register_style("bad_parent", {"extends": "does_not_exist"})

        # Invalid notehead archetype must fail
        with self.assertRaises(ValueError):
            register_style(
                "bad_archetype",
                {"extends": "baroque_quill_neat", "notehead_archetype": "triangle_invalid"},
            )

        # Circular inheritance in a JSON pack must fail
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_pack = Path(tmpdir) / "cycle.json"
            bad_pack.write_text(
                json.dumps({
                    "cycle_a": {"extends": "cycle_b"},
                    "cycle_b": {"extends": "cycle_a"},
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                reload_style_packs(packs_dir=tmpdir, reset_to_builtins=True)

        # Valid registration via extends succeeds
        reg = register_style(
            "temp_valid_scribe",
            {
                "extends": "chopin_delicate_salon",
                "ink_feather": 0.25,
                "split_nib_prob": 0.4,
                "tags": ["salon_variant"],
            },
        )
        self.assertEqual(reg["name"], "temp_valid_scribe")
        self.assertIn("temp_valid_scribe", SCRIBE_STYLES)
        self.assertIn("temp_valid_scribe", list_styles(tag="salon_variant"))

    def test_interpolate_styles_endpoints_and_midpoint(self) -> None:
        sa = get_style("baroque_quill_neat")
        sb = get_style("student_ballpoint_blue")

        mid = interpolate_styles("baroque_quill_neat", "student_ballpoint_blue", 0.5)
        for key in REQUIRED_STYLE_KEYS:
            self.assertIn(key, mid)
        self.assertAlmostEqual(
            mid["nib_min_ratio"],
            0.5 * (sa["nib_min_ratio"] + sb["nib_min_ratio"]),
            places=5,
        )
        for ch in range(3):
            self.assertAlmostEqual(
                mid["ink_rgb"][ch],
                0.5 * (sa["ink_rgb"][ch] + sb["ink_rgb"][ch]),
                places=5,
            )
        self.assertAlmostEqual(sum(mid["rest_archetype_weights"].values()), 1.0, places=4)

        at_zero = interpolate_styles("baroque_quill_neat", "student_ballpoint_blue", 0.0)
        self.assertAlmostEqual(at_zero["stroke_scale"], sa["stroke_scale"], places=6)
        self.assertEqual(at_zero["notehead_archetype"], sa["notehead_archetype"])

        at_one = interpolate_styles("baroque_quill_neat", "student_ballpoint_blue", 1.0)
        self.assertAlmostEqual(at_one["stroke_scale"], sb["stroke_scale"], places=6)
        self.assertEqual(at_one["notehead_archetype"], sb["notehead_archetype"])

    def test_sample_random_scribe_produces_valid_diverse_unseen_scribes(self) -> None:
        rng1 = np.random.default_rng(2026)
        rng2 = np.random.default_rng(2026)
        s1 = sample_random_scribe(rng1, jitter=0.22)
        s2 = sample_random_scribe(rng2, jitter=0.22)
        # Deterministic for same seed
        self.assertEqual(s1["name"], s2["name"])
        self.assertAlmostEqual(s1["wobble_px"], s2["wobble_px"], places=6)

        # Sample 12 continuous scribes and verify validity + diversity
        samples = [sample_random_scribe(rng1, jitter=0.25) for _ in range(12)]
        names = set()
        for s in samples:
            names.add(s["name"])
            for key in REQUIRED_STYLE_KEYS:
                self.assertIn(key, s)
            self.assertIn(s["notehead_archetype"], NOTEHEAD_ARCHETYPES)
            self.assertEqual(set(s["rest_archetype_weights"].keys()), set(REST_ARCHETYPES))
            self.assertAlmostEqual(sum(s["rest_archetype_weights"].values()), 1.0, places=4)
            self.assertTrue(0.0 <= s["ink_feather"] <= 1.0)
            self.assertTrue(0.0 <= s["split_nib_prob"] <= 1.0)
            self.assertTrue(0.0 <= s["rastrum_wobble"] <= 6.0)
            self.assertTrue(0.0 <= s["stain_prob"] <= 1.0)
        self.assertGreaterEqual(len(names), 10)

        # Sampling anchored to a specific base style works
        anchored = sample_random_scribe(np.random.default_rng(9), base_style="bach_thick_iron_gall", jitter=0.15)
        self.assertIn("ink_feather", anchored)
        self.assertAlmostEqual(sum(anchored["rest_archetype_weights"].values()), 1.0, places=4)


class TestExpandedGlyphsAndStrokePhysics(unittest.TestCase):
    def test_all_10_quarter_rest_archetypes_unique_per_draw_and_distinct(self) -> None:
        self.assertEqual(len(QUARTER_REST_ARCHETYPES), 10)
        for expected in ("sharp_v_tick", "baroque_flag_loop", "modern_tall_z", "scribble_s"):
            self.assertIn(expected, QUARTER_REST_ARCHETYPES)

        base_style = get_style("baroque_quill_neat")
        bg = np.broadcast_to(BG_CREAM, (128, 96, 3))
        renders = []
        for idx, arch in enumerate(QUARTER_REST_ARCHETYPES):
            style = dict(base_style, force_rest_archetype=arch)
            c1 = StrokeCanvas(96, 128, BG_CREAM, np.random.default_rng(1000 + idx * 2))
            c2 = StrokeCanvas(96, 128, BG_CREAM, np.random.default_rng(1001 + idx * 2))
            draw_rest(c1, "quarter", 48.0, 64.0, 14.0, INK_DARK, style, np.random.default_rng(2000 + idx * 2))
            draw_rest(c2, "quarter", 48.0, 64.0, 14.0, INK_DARK, style, np.random.default_rng(2001 + idx * 2))

            self.assertGreater(_mean_abs_diff(c1.buf, bg), 1e-4, f"{arch} drew nothing")
            self.assertGreater(_mean_abs_diff(c1.buf, c2.buf), 1e-4, f"{arch} was identical across draws")
            renders.append((arch, c1.buf))

        # Every pair of archetypes should render distinctly
        for i in range(len(renders)):
            for j in range(i + 1, len(renders)):
                diff = _mean_abs_diff(renders[i][1], renders[j][1])
                self.assertGreater(
                    diff,
                    1e-4,
                    f"Archetypes {renders[i][0]} and {renders[j][0]} rendered identically",
                )

    def test_expanded_notehead_archetypes_heavy_wedge_and_hollow_diamond(self) -> None:
        self.assertIn("heavy_wedge", NOTEHEAD_ARCHETYPES)
        self.assertIn("hollow_diamond", NOTEHEAD_ARCHETYPES)
        base_style = get_style("bach_thick_iron_gall")
        bg = np.broadcast_to(BG_CREAM, (64, 64, 3))
        for arch in ("heavy_wedge", "hollow_diamond"):
            for dur in ("filled", "half", "whole"):
                style = dict(base_style, notehead_archetype=arch)
                c1 = StrokeCanvas(64, 64, BG_CREAM, np.random.default_rng(31))
                c2 = StrokeCanvas(64, 64, BG_CREAM, np.random.default_rng(32))
                draw_notehead(c1, dur, 32.0, 32.0, 14.0, INK_DARK, style, np.random.default_rng(41))
                draw_notehead(c2, dur, 32.0, 32.0, 14.0, INK_DARK, style, np.random.default_rng(42))
                self.assertGreater(_mean_abs_diff(c1.buf, bg), 1e-4, f"{arch}/{dur} drew nothing")
                self.assertGreater(_mean_abs_diff(c1.buf, c2.buf), 1e-4, f"{arch}/{dur} did not vary")

    def test_new_handwritten_primitives_render_cleanly(self) -> None:
        style = get_style("mozart_flowing_cursive")
        rng = np.random.default_rng(555)
        sp = 14.0

        def _check(fn, w: int = 180, h: int = 120) -> None:
            c = StrokeCanvas(w, h, BG_CREAM, np.random.default_rng(int(rng.integers(1, 100000))))
            fn(c)
            bg = np.broadcast_to(BG_CREAM, c.buf.shape)
            self.assertGreater(_mean_abs_diff(c.buf, bg), 1e-5)

        _check(lambda c: draw_slur_or_tie(c, 25.0, 70.0, 155.0, 68.0, sp, True, INK_DARK, style, rng))
        _check(lambda c: draw_slur_or_tie(c, 25.0, 50.0, 155.0, 52.0, sp, False, INK_DARK, style, rng))
        _check(lambda c: draw_fermata(c, 90.0, 60.0, sp, True, INK_DARK, style, rng))
        _check(lambda c: draw_fermata(c, 90.0, 60.0, sp, False, INK_DARK, style, rng))
        for dyn in ("p", "pp", "mp", "mf", "f", "ff", "sfz", "fp"):
            _check(lambda c, d=dyn: draw_dynamic_text(c, d, 90.0, 60.0, sp, INK_DARK, style, rng))
        for hk in ("cresc", "dim", "<", ">"):
            _check(lambda c, k=hk: draw_hairpin(c, 25.0, 155.0, 60.0, sp, k, INK_DARK, style, rng))
        for ck in ("scribble", "cross_x", "hatch", "auto"):
            _check(lambda c, k=ck: draw_correction_scratch(c, 50.0, 40.0, 120.0, 80.0, sp, INK_DARK, style, rng, kind=k))

    def test_stroke_physics_ink_feather_curvature_pooling_split_nib_and_rastrum(self) -> None:
        # 1. ink_feather produces a wider soft bleed halo than ink_feather=0.0
        pts = np.array([[20.0, 40.0], [100.0, 40.0]], dtype=np.float64)
        base = {
            "nib_angle_deg": 45.0,
            "nib_min_ratio": 0.9,
            "wobble_px": 0.0,
            "tremor_px": 0.0,
            "ink_dropout": 0.0,
            "grain_strength": 0.05,
            "ink_var": 0.0,
            "ink_pooling": 0.0,
            "stroke_opacity": 0.92,
        }
        c_crisp = StrokeCanvas(120, 80, BG_CREAM, np.random.default_rng(77))
        c_feath = StrokeCanvas(120, 80, BG_CREAM, np.random.default_rng(77))
        draw_calligraphic_polyline(c_crisp, pts, INK_DARK, 3.0, dict(base, ink_feather=0.0))
        draw_calligraphic_polyline(c_feath, pts, INK_DARK, 3.0, dict(base, ink_feather=0.85))
        mass_crisp = float(np.sum(np.mean(BG_CREAM - c_crisp.buf, axis=2)))
        mass_feath = float(np.sum(np.mean(BG_CREAM - c_feath.buf, axis=2)))
        self.assertGreater(mass_feath, mass_crisp * 1.15)

        # 2. Speed-dependent curvature pooling deposits more ink at a sharp 90-deg corner
        corner_pts = np.array([[20.0, 20.0], [60.0, 20.0], [60.0, 60.0]], dtype=np.float64)
        c_nopool = StrokeCanvas(100, 100, BG_CREAM, np.random.default_rng(88))
        c_pool = StrokeCanvas(100, 100, BG_CREAM, np.random.default_rng(88))
        draw_calligraphic_polyline(c_nopool, corner_pts, INK_DARK, 4.0, dict(base, ink_pooling=0.0))
        draw_calligraphic_polyline(c_pool, corner_pts, INK_DARK, 4.0, dict(base, ink_pooling=0.75))
        # Compare ink mass specifically around the corner (60, 20)
        roi_no = float(np.sum(np.mean(BG_CREAM - c_nopool.buf[14:27, 54:67], axis=2)))
        roi_pl = float(np.sum(np.mean(BG_CREAM - c_pool.buf[14:27, 54:67], axis=2)))
        self.assertGreater(roi_pl, roi_no)

        # 3. split_nib_prob=1.0 differs from split_nib_prob=0.0
        c_solid = StrokeCanvas(120, 80, BG_CREAM, np.random.default_rng(91))
        c_split = StrokeCanvas(120, 80, BG_CREAM, np.random.default_rng(91))
        draw_calligraphic_polyline(c_solid, pts, INK_DARK, 5.0, dict(base, split_nib_prob=0.0))
        draw_calligraphic_polyline(c_split, pts, INK_DARK, 5.0, dict(base, split_nib_prob=1.0))
        self.assertGreater(_mean_abs_diff(c_solid.buf, c_split.buf), 1e-3)

        # 4. draw_rastrum_staff produces coherent 5-line parallel wobble
        style_r = dict(get_style("baroque_quill_neat"), rastrum_wobble=2.5, wobble_px=0.0, tremor_px=0.0)
        y_fn, _, _ = build_curved_staff_y(20.0, 280.0, 80.0, 12.0, style_r, np.random.default_rng(12))
        c_ras = StrokeCanvas(300, 160, BG_CREAM, np.random.default_rng(13))
        wave = draw_rastrum_staff(c_ras, 20.0, 280.0, y_fn, INK_DARK, 1.5, style_r)
        self.assertGreater(len(wave), 200)
        self.assertGreater(float(np.std(wave)), 0.1)
        bg_r = np.broadcast_to(BG_CREAM, c_ras.buf.shape)
        self.assertGreater(_mean_abs_diff(c_ras.buf, bg_r), 1e-3)


class TestPaperTypesStainsAndMarginAnnotations(unittest.TestCase):
    def test_all_paper_types_render_and_are_distinct(self) -> None:
        self.assertEqual(
            set(PAPER_TYPES),
            {"aged_rag", "bright_white", "cream_manuscript", "spiral_notebook", "ozalid_blue", "soviet_green_tint"},
        )
        base = get_style("romantic_copyist_oblong")
        imgs = {}
        for ptype in PAPER_TYPES:
            st = dict(base, paper_type=ptype)
            img = make_manuscript_paper(200, 160, st, np.random.default_rng(101))
            self.assertEqual(img.shape, (160, 200, 3))
            self.assertEqual(img.dtype, np.float32)
            self.assertGreaterEqual(float(img.min()), 0.0)
            self.assertLessEqual(float(img.max()), 1.0)
            imgs[ptype] = img

        # Ozalid blue has higher blue than red; soviet green has higher green than red/blue
        oz = imgs["ozalid_blue"]
        self.assertGreater(float(oz[:, :, 2].mean()), float(oz[:, :, 0].mean()))
        sv = imgs["soviet_green_tint"]
        self.assertGreater(float(sv[:, :, 1].mean()), float(sv[:, :, 0].mean()))
        self.assertGreater(float(sv[:, :, 1].mean()), float(sv[:, :, 2].mean()))
        # Spiral notebook differs from bright_white due to ruled lines & margin/holes
        self.assertGreater(_mean_abs_diff(imgs["spiral_notebook"], imgs["bright_white"]), 1e-3)

    def test_coffee_water_stain_and_margin_annotations(self) -> None:
        base = get_style("beethoven_hurried_autograph")
        p_clean = make_manuscript_paper(220, 180, dict(base, stain_prob=0.0), np.random.default_rng(42))
        p_stain = make_manuscript_paper(220, 180, dict(base, stain_prob=1.0), np.random.default_rng(42))
        self.assertGreater(_mean_abs_diff(p_clean, p_stain), 1e-3)

        canvas = StrokeCanvas(300, 240, BG_CREAM, np.random.default_rng(55))
        drawn = draw_margin_annotations(canvas, base, np.random.default_rng(56), staff_ys=[60.0, 140.0], sp=12.0)
        self.assertGreaterEqual(len(drawn), 2)
        bg = np.broadcast_to(BG_CREAM, canvas.buf.shape)
        self.assertGreater(_mean_abs_diff(canvas.buf, bg), 1e-4)


if __name__ == "__main__":
    unittest.main()
