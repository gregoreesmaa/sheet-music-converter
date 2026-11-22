"""Unit tests for bar-anchored long-system splitting (leland cull recovery).

Contract: a >512-token system becomes measure-anchored <=512-token chunks
(`<sys>c<k>` stems) with barline-verified image cuts, or fails loud.
Pure planners here (no binaries); the bekern/image wiring is proven by
the split prototype (587-token system -> 308+306, both decode) and the
production rebuild's S2 gates.
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_sys_data as B


class TestPlanMeasureChunks(unittest.TestCase):
    def test_even_split_by_budget(self):
        # 4 measures x 200 tokens, target 448 -> [0..1] + [2..3]
        self.assertEqual(
            B.plan_measure_chunks([10, 11, 12, 13], [200, 200, 200, 200]),
            [(10, 11), (12, 13)])

    def test_exact_boundary_keeps_together(self):
        # 224 + 224 = 448 exactly fits the target
        self.assertEqual(
            B.plan_measure_chunks([1, 2], [224, 224]), [(1, 2)])

    def test_single_measure_over_hard_max_fails(self):
        self.assertIsNone(B.plan_measure_chunks([7], [600]))
        self.assertIsNone(B.plan_measure_chunks([7, 8], [100, 600]))

    def test_uneven_density_balances(self):
        # dense middle measure forces an early cut (greedy close on target)
        self.assertEqual(
            B.plan_measure_chunks([1, 2, 3, 4], [100, 400, 100, 100]),
            [(1, 1), (2, 2), (3, 4)])


class TestChooseCutX(unittest.TestCase):
    BARS = [100.0, 300.0, 500.0, 700.0]

    def test_nearest_in_bounds(self):
        self.assertEqual(
            B.choose_cut_x(310.0, self.BARS, 0.0, 800.0), 300.0)

    def test_margin_excludes_edges(self):
        # 100.0 is within the 8px margin of lo=95 -> skipped
        self.assertEqual(
            B.choose_cut_x(110.0, self.BARS, 95.0, 800.0), 300.0)

    def test_no_candidate_returns_none(self):
        self.assertIsNone(B.choose_cut_x(400.0, [], 0.0, 800.0))
        self.assertIsNone(B.choose_cut_x(400.0, [5.0], 0.0, 800.0))


class TestCutHasBarline(unittest.TestCase):
    def test_solid_column_passes(self):
        img = np.full((100, 200), 255, dtype=np.uint8)
        img[10:90, 50] = 0  # full-height barline at x=50
        self.assertTrue(B.cut_has_barline(img, 50.0, 10.0, 90.0))

    def test_empty_column_fails(self):
        img = np.full((100, 200), 255, dtype=np.uint8)
        self.assertFalse(B.cut_has_barline(img, 50.0, 10.0, 90.0))

    def test_sparse_column_fails(self):
        img = np.full((100, 200), 255, dtype=np.uint8)
        img[10:15, 50] = 0  # a dot, not a barline
        self.assertFalse(B.cut_has_barline(img, 50.0, 10.0, 90.0))


class TestFullHeightBarXs(unittest.TestCase):
    def test_selects_spanning_barlines(self):
        glyphs = [
            {'bbox': [100.0, 470.0, 104.0, 550.0], 'class': 'BarLine'},
            {'bbox': [300.0, 0.0, 304.0, 50.0], 'class': 'BarLine'},
            {'bbox': [500.0, 470.0, 504.0, 550.0], 'class': 'Lyrics'},
        ]
        self.assertEqual(
            B.full_height_bar_xs(glyphs, 0.0, 1000.0, 470.0, 550.0),
            [102.0])


def _staff_barlines(n_staves, x, top=0.0, staff_h=40.0, gap=15.0, w=4.0):
    """One per-staff BarLine glyph per staff (broken between staves)."""
    glyphs = []
    for s in range(n_staves):
        y1 = top + s * (staff_h + gap)
        glyphs.append({'bbox': [x, y1, x + w, y1 + staff_h],
                       'class': 'BarLine'})
    return glyphs


class TestBarColumns(unittest.TestCase):
    def test_per_staff_column_found(self):
        # 8 staves, per-staff barlines at the same x: column anchors the cut
        glyphs = _staff_barlines(8, 300.0)
        y_top, y_bot = 0.0, 8 * 40.0 + 7 * 15.0
        cols = B.bar_column_xs(glyphs, 0.0, 1000.0, y_top, y_bot)
        self.assertEqual(len(cols), 1)
        self.assertAlmostEqual(cols[0][0], 302.0)

    def test_truly_unbarred_stays_empty(self):
        cols = B.bar_column_xs([], 0.0, 1000.0, 0.0, 500.0)
        self.assertEqual(cols, [])
        glyphs = [{'bbox': [300.0, 0.0, 304.0, 40.0], 'class': 'NoteBlack'}]
        self.assertEqual(
            B.bar_column_xs(glyphs, 0.0, 1000.0, 0.0, 500.0), [])

    def test_half_covered_column_rejected(self):
        # barlines on only 3 of 8 staves: coverage 0.375 < 0.7 floor
        glyphs = _staff_barlines(8, 300.0)[:3]
        y_top, y_bot = 0.0, 8 * 40.0 + 7 * 15.0
        self.assertEqual(
            B.bar_column_xs(glyphs, 0.0, 1000.0, y_top, y_bot), [])

    def test_unanchored_ends_rejected(self):
        # full middle coverage but top/bottom staves lack barlines
        glyphs = _staff_barlines(8, 300.0)[1:7]
        y_top, y_bot = 0.0, 8 * 40.0 + 7 * 15.0
        self.assertEqual(
            B.bar_column_xs(glyphs, 0.0, 1000.0, y_top, y_bot), [])


if __name__ == '__main__':
    unittest.main()
