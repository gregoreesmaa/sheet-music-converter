#!/usr/bin/env python3
"""Unit tests for tmp-train/mvd/neg_control_gate.py (CPU-only)."""

from __future__ import annotations

import os
import pathlib
import sys
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

MVD_DIR = pathlib.Path(__file__).resolve().parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

from PIL import Image, ImageDraw  # noqa: E402

from neg_control_gate import is_non_music_page  # noqa: E402


def _make_blank_staves(width: int = 800, height: int = 1100) -> Image.Image:
    """Create a synthetic blank-manuscript page (horizontal staves, no notes/barlines)."""
    img = Image.new("RGB", (width, height), (250, 248, 244))
    draw = ImageDraw.Draw(img)
    for sys_idx in range(8):
        y0 = 90 + sys_idx * 120
        for line in range(5):
            y = y0 + line * 10
            draw.line([(60, y), (width - 60, y)], fill=(25, 25, 25), width=2)
    return img


def _make_text_only(width: int = 800, height: int = 1100) -> Image.Image:
    """Create a synthetic text-only program-notes page (short character strokes, no staves)."""
    img = Image.new("RGB", (width, height), (252, 250, 245))
    draw = ImageDraw.Draw(img)
    for row in range(28):
        y = 80 + row * 32
        x = 70
        while x < width - 90:
            # Draw short vertical/horizontal glyph strokes mimicking Latin text
            draw.rectangle([x, y, x + 2, y + 12], fill=(20, 20, 20))
            draw.rectangle([x + 4, y + 4, x + 9, y + 6], fill=(20, 20, 20))
            draw.rectangle([x + 10, y + 2, x + 12, y + 12], fill=(20, 20, 20))
            x += 18 if ((x + row * 7) % 5 != 0) else 28
    return img


def _make_real_staff_and_notes(width: int = 800, height: int = 1100) -> Image.Image:
    """Create a synthetic music page with 5-line staves, barlines, noteheads, and stems."""
    img = _make_blank_staves(width=width, height=height)
    draw = ImageDraw.Draw(img)
    for sys_idx in range(8):
        y0 = 90 + sys_idx * 120
        # Barlines
        for bx in (60, 240, 420, 600, width - 60):
            draw.line([(bx, y0), (bx, y0 + 40)], fill=(15, 15, 15), width=2)
        # Noteheads + vertical stems
        for nx in range(95, width - 85, 34):
            ny = y0 + ((nx // 34 + sys_idx) % 5) * 10
            draw.ellipse([nx - 5, ny - 4, nx + 5, ny + 4], fill=(15, 15, 15))
            draw.line([(nx + 5, ny), (nx + 5, ny - 32)], fill=(15, 15, 15), width=2)
    return img


class TestNegControlGate(unittest.TestCase):
    def test_blank_white_page_rejected(self) -> None:
        img = Image.new("RGB", (800, 1100), (255, 255, 255))
        rejected, info = is_non_music_page(img)
        self.assertTrue(rejected, info)
        self.assertEqual(info["v_frac"], 0.0)
        self.assertEqual(info["h_frac"], 0.0)
        self.assertEqual(info["reason"], "low_v_and_h_frac")

    def test_blank_staves_rejected_by_v_frac(self) -> None:
        img = _make_blank_staves()
        rejected, info = is_non_music_page(img)
        self.assertTrue(rejected, info)
        self.assertEqual(info["reason"], "low_v_frac")
        self.assertLess(info["v_frac"], info["v_thr"])
        self.assertGreater(info["h_frac"], info["h_thr"])

    def test_text_only_page_rejected_by_h_frac(self) -> None:
        img = _make_text_only()
        rejected, info = is_non_music_page(img)
        self.assertTrue(rejected, info)
        self.assertEqual(info["reason"], "low_h_frac")
        self.assertGreater(info["v_frac"], info["v_thr"])
        self.assertLess(info["h_frac"], info["h_thr"])

    def test_real_staff_and_notes_accepted(self) -> None:
        img = _make_real_staff_and_notes()
        rejected, info = is_non_music_page(img)
        self.assertFalse(rejected, info)
        self.assertEqual(info["reason"], "ok")
        self.assertGreater(info["v_frac"], info["v_thr"])
        self.assertGreater(info["h_frac"], info["h_thr"])
        self.assertEqual(info["h_angle_deg"], 0.0)

    def test_rotated_real_staff_and_notes_accepted(self) -> None:
        base = _make_real_staff_and_notes(width=1200, height=1650)
        for angle in (-3.0, 3.0):
            with self.subTest(angle=angle):
                rot = base.rotate(
                    angle,
                    resample=Image.Resampling.BILINEAR,
                    fillcolor=(250, 248, 244),
                )
                rejected, info = is_non_music_page(rot)
                self.assertFalse(rejected, f"angle={angle}: {info}")
                self.assertEqual(info["reason"], "ok")
                self.assertGreater(info["h_frac"], info["h_thr"])

    def test_rgba_transparent_blank_rejected(self) -> None:
        img = Image.new("RGBA", (800, 1100), (0, 0, 0, 0))
        rejected, info = is_non_music_page(img)
        self.assertTrue(rejected, info)
        self.assertEqual(info["v_frac"], 0.0)
        self.assertEqual(info["h_frac"], 0.0)


if __name__ == "__main__":
    unittest.main()
