#!/usr/bin/env python3
"""CPU unit tests for ``dual_verifier.py`` (R_sym, R_vis, R_dual).

Run with:
    $env:CUDA_VISIBLE_DEVICES=""
    .\\.venv\\Scripts\\python.exe -m unittest tmp-train/mvd/test_dual_verifier.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import dual_verifier as dv  # noqa: E402


VALID_44_2SYS = (
    "clef-G-2 key+0 time-4-4 "
    "quarter C 4 quarter D 4 quarter E 4 quarter F 4 bar "
    "half G 4 half A 4 bar "
    "quarter B 4 quarter C 5 quarter D 5 quarter E 5 bar "
    "<sys> "
    "clef-G-2 key+0 time-4-4 "
    "quarter C 5 quarter B 4 quarter A 4 quarter G 4 bar "
    "half F 4 half E 4 bar "
    "whole C 4 bar-light-heavy"
)

INVALID_METER_44 = (
    "clef-G-2 key+0 time-4-4 "
    "quarter C 4 quarter D 4 bar "  # only 2 beats in 4/4
    "quarter E 4 quarter F 4 bar "  # only 2 beats in 4/4
    "quarter G 4 quarter A 4 bar "  # only 2 beats in 4/4
    "whole C 4 bar-light-heavy"
)

SYS_LOOP_BEKERN = " <sys> ".join(
    ["clef-G-2 key+0 time-4-4 quarter C 4 quarter D 4 quarter E 4 quarter F 4 bar"] * 4
)


def _make_synthetic_sheet(
    height: int = 240,
    width: int = 360,
    n_systems: int = 2,
    note_xs: list[float] | None = None,
    bg_level: float = 0.98,
    ink_level: float = 0.05,
    gradient: float = 0.0,
) -> np.ndarray:
    """Create a synthetic grayscale sheet-music image with staff lines and noteheads."""
    img = np.full((height, width), bg_level, dtype=np.float32)
    if gradient != 0.0:
        gx = np.linspace(0.0, gradient, width, dtype=np.float32)[None, :]
        gy = np.linspace(0.0, gradient * 0.5, height, dtype=np.float32)[:, None]
        img = np.clip(img - gx - gy, 0.25, 1.0)

    if note_xs is None:
        note_xs = [0.15, 0.28, 0.42, 0.58, 0.72, 0.86]

    margin_y = height // (n_systems + 1)
    for s in range(n_systems):
        cy = margin_y * (s + 1)
        # 5 horizontal staff lines
        for k in range(-2, 3):
            y = max(0, min(height - 1, cy + k * 5))
            img[y, int(width * 0.06) : int(width * 0.94)] = ink_level + 0.12
        # Vertical barlines
        for bx in (0.06, 0.50, 0.94):
            x = int(width * bx)
            img[max(0, cy - 10) : min(height, cy + 11), max(0, x - 1) : min(width, x + 1)] = ink_level
        # Noteheads + stems
        for idx, nx in enumerate(note_xs):
            x = int(width * nx)
            pitch_off = ((idx * 3 + s * 2) % 7 - 3) * 3
            ny = max(4, min(height - 5, cy + pitch_off))
            img[ny - 3 : ny + 4, max(0, x - 4) : min(width, x + 5)] = ink_level
            img[max(0, ny - 18) : ny + 1, min(width - 1, x + 3) : min(width, x + 5)] = ink_level
    return img


class TestSymbolicVerifier(unittest.TestCase):
    def test_valid_bekern_high_r_sym(self) -> None:
        res = dv.compute_symbolic_score(VALID_44_2SYS)
        self.assertTrue(res["parse_ok"])
        self.assertEqual(res["n_sys"], 2)
        self.assertGreaterEqual(res["r_meter"], 0.99)
        self.assertGreaterEqual(res["r_sym"], 0.95)
        self.assertFalse(res["has_tok_loop"])
        self.assertFalse(res["has_sys_loop"])

    def test_empty_and_broken_syntax_zero(self) -> None:
        self.assertEqual(dv.compute_symbolic_score("")["r_sym"], 0.0)
        self.assertEqual(dv.compute_symbolic_score("clef-G2 bar")["r_sym"], 0.0)
        self.assertEqual(dv.compute_symbolic_score("quarter C bar")["r_sym"], 0.0)

    def test_meter_violation_penalized(self) -> None:
        good = dv.compute_symbolic_score(VALID_44_2SYS)
        bad = dv.compute_symbolic_score(INVALID_METER_44)
        self.assertTrue(bad["parse_ok"])
        self.assertLess(bad["r_meter"], 0.2)
        self.assertLess(bad["r_sym"], good["r_sym"] - 0.35)

    def test_sys_loop_penalized(self) -> None:
        self.assertTrue(dv.detect_sys_loop(SYS_LOOP_BEKERN.split()))
        res = dv.compute_symbolic_score(SYS_LOOP_BEKERN)
        self.assertTrue(res["has_sys_loop"])
        self.assertLessEqual(res["r_sym"], 0.30)

    def test_empty_sys_detected(self) -> None:
        self.assertTrue(dv.detect_sys_loop(["clef-G2", "<sys>", "<sys>", "quarter", "C", "4"]))


class TestVisualCycleVerifier(unittest.TestCase):
    def test_identical_sheet_scores_near_one(self) -> None:
        img = _make_synthetic_sheet(n_systems=2)
        vis = dv.compute_visual_cycle_score(img, img)
        self.assertTrue(vis["visual_ok"])
        self.assertGreater(vis["r_vis"], 0.98)
        self.assertLess(vis["dtw_emd_dist"], 1e-4)

    def test_style_invariance_yellowed_paper_and_faint_ink(self) -> None:
        clean = _make_synthetic_sheet(n_systems=2, bg_level=0.99, ink_level=0.02, gradient=0.0)
        aged = _make_synthetic_sheet(n_systems=2, bg_level=0.82, ink_level=0.28, gradient=0.12)
        vis = dv.compute_visual_cycle_score(aged, clean)
        self.assertTrue(vis["visual_ok"])
        self.assertGreater(vis["r_vis"], 0.80)

    def test_spacing_invariance_scribal_horizontal_stretch(self) -> None:
        # Real manuscript has non-uniform scribal spacing vs engraved spacing
        engraved = _make_synthetic_sheet(
            n_systems=2, note_xs=[0.14, 0.28, 0.42, 0.56, 0.70, 0.84]
        )
        stretched = _make_synthetic_sheet(
            n_systems=2, note_xs=[0.12, 0.22, 0.34, 0.62, 0.76, 0.89]
        )
        wrong_sys = _make_synthetic_sheet(
            n_systems=4, note_xs=[0.14, 0.28, 0.42, 0.56, 0.70, 0.84]
        )
        vis_stretch = dv.compute_visual_cycle_score(stretched, engraved)
        vis_wrong = dv.compute_visual_cycle_score(wrong_sys, engraved)
        self.assertGreater(vis_stretch["r_vis"], 0.75)
        self.assertGreater(vis_stretch["r_vis"], vis_wrong["r_vis"] + 0.10)

    def test_blank_image_returns_zero(self) -> None:
        blank = np.ones((200, 200), dtype=np.float32)
        img = _make_synthetic_sheet(n_systems=2)
        vis = dv.compute_visual_cycle_score(img, blank)
        self.assertFalse(vis["visual_ok"])
        self.assertEqual(vis["r_vis"], 0.0)


class TestDualVerifierEndToEnd(unittest.TestCase):
    def test_combined_formula_prevents_empty_bar_hacking(self) -> None:
        real_img = _make_synthetic_sheet(n_systems=2)
        wrong_img = _make_synthetic_sheet(n_systems=4)

        v_match = dv.DualVerifier(render_fn=lambda _toks: real_img)
        v_mismatch = dv.DualVerifier(render_fn=lambda _toks: wrong_img)
        v_fail_render = dv.DualVerifier(render_fn=lambda _toks: None)

        s_match = v_match.score(real_img, VALID_44_2SYS)
        s_mismatch = v_mismatch.score(real_img, VALID_44_2SYS)
        s_fail = v_fail_render.score(real_img, VALID_44_2SYS)

        self.assertAlmostEqual(
            s_match["r_dual"],
            s_match["r_sym"] * (0.2 + 0.8 * s_match["r_vis"]),
            places=5,
        )
        self.assertAlmostEqual(s_fail["r_dual"], 0.2 * s_fail["r_sym"], places=5)
        self.assertGreater(s_match["r_dual"], s_mismatch["r_dual"])
        self.assertGreater(s_mismatch["r_dual"], s_fail["r_dual"])

    def test_python_musicxml_fallback_and_verovio_render(self) -> None:
        xml = dv.bekern_to_musicxml_fallback(VALID_44_2SYS)
        self.assertIsNotNone(xml)
        assert xml is not None
        self.assertIn("<score-partwise", xml)
        self.assertIn('new-system="yes"', xml)

        if dv.verovio is not None and (dv.resvg_py is not None or dv.cairosvg is not None):
            rendered = dv.render_musicxml_to_gray(xml, use_encoded_breaks=True)
            self.assertIsNotNone(rendered)
            assert rendered is not None
            self.assertEqual(rendered.ndim, 2)
            self.assertGreater(rendered.shape[0], 50)
            self.assertGreater(rendered.shape[1], 50)

            # Real end-to-end DualVerifier self-consistency check
            verifier = dv.DualVerifier()
            res = verifier.score(rendered, VALID_44_2SYS)
            self.assertTrue(res["parse_ok"])
            self.assertTrue(res["render_ok"])
            self.assertGreater(res["r_sym"], 0.90)
            self.assertGreater(res["r_vis"], 0.80)
            self.assertGreater(res["r_dual"], 0.80)

    def test_batch_scoring_caches_real_profile(self) -> None:
        real_img = _make_synthetic_sheet(n_systems=2)
        verifier = dv.DualVerifier(render_fn=lambda _toks: real_img)
        results = verifier.score_batch_for_page(
            real_img, [VALID_44_2SYS, INVALID_METER_44, ""]
        )
        self.assertEqual(len(results), 3)
        self.assertGreater(results[0]["r_dual"], results[1]["r_dual"])
        self.assertEqual(results[2]["r_dual"], 0.0)

    def test_threadpool_verovio_render_no_race(self) -> None:
        if dv.verovio is None or (dv.resvg_py is None and dv.cairosvg is None):
            self.skipTest("verovio/resvg_py not installed")
        from concurrent.futures import ThreadPoolExecutor

        def _render(_i: int) -> bool:
            img = dv.render_bekern_to_gray(VALID_44_2SYS, use_fallback=True)
            return img is not None and img.ndim == 2 and img.shape[0] > 50

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(_render, range(8)))
        self.assertTrue(all(results), f"Expected all 8 concurrent renders to succeed, got {results}")


if __name__ == "__main__":
    unittest.main()
