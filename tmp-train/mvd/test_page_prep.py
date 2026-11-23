"""Unit tests for tmp-train/mvd/page_prep.py (border-crop + deskew).

Run with:
    .\\.venv\\Scripts\\python.exe -m unittest tmp-train/mvd/test_page_prep.py -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import page_prep  # noqa: E402


def _make_staff_page(w=1000, h=1400, n_staves=5, space=14, margin_x=80,
                     bg=(255, 255, 255), ink=(0, 0, 0)):
    """Synthetic clean sheet-music page with crisp 5-line staves + notes."""
    img = Image.new('RGB', (w, h), bg)
    d = ImageDraw.Draw(img)
    y_start = 150
    step_y = 220
    for s in range(n_staves):
        top = y_start + s * step_y
        for i in range(5):
            y = top + i * space
            d.line([(margin_x, y), (w - margin_x, y)], fill=ink, width=2)
        # Add some barlines and noteheads
        for x in range(margin_x + 80, w - margin_x - 40, 140):
            d.line([(x, top), (x, top + 4 * space)], fill=ink, width=2)
            d.ellipse([(x + 30, top + space - 5), (x + 46, top + space + 5)],
                      fill=ink)
            d.line([(x + 46, top + space), (x + 46, top - 20)], fill=ink, width=2)
    return img


class TestPagePrep(unittest.TestCase):
    def test_clean_unbordered_page_untouched(self):
        clean = _make_staff_page()
        out, meta = page_prep.prep_page(clean)
        self.assertFalse(meta['applied'], meta)
        self.assertFalse(meta['border_cropped'])
        self.assertFalse(meta['deskew_applied'])
        self.assertEqual(meta['border_crop_box'], [0, 0, 1000, 1400])
        self.assertLess(abs(meta['skew_deg']), 0.15)
        self.assertEqual(out.size, clean.size)

    def test_solid_black_border_removed_exact(self):
        clean = _make_staff_page(w=800, h=1100)
        bordered = Image.new('RGB', (1000, 1300), (0, 0, 0))
        bordered.paste(clean, (100, 100))
        out, meta = page_prep.prep_page(bordered)
        self.assertTrue(meta['applied'])
        self.assertTrue(meta['border_cropped'])
        self.assertEqual(meta['border_crop_box'], [100, 100, 900, 1200])
        self.assertEqual(out.size, (800, 1100))
        self.assertLess(abs(meta['skew_deg']), 0.15)

    def test_rotated_and_black_bordered_page_deskewed_accurately(self):
        clean = _make_staff_page(w=1000, h=1400, bg=(245, 242, 235))
        for true_angle in (-5.5, -2.2, 3.4, 7.0):
            with self.subTest(true_angle=true_angle):
                # Rotate page on dark desk + add 80px black border
                rot = clean.rotate(true_angle, resample=Image.BICUBIC,
                                   expand=True, fillcolor=(12, 12, 12))
                canvas = Image.new('RGB', (rot.size[0] + 160, rot.size[1] + 160),
                                   (0, 0, 0))
                canvas.paste(rot, (80, 80))
                out, meta = page_prep.prep_page(canvas)
                self.assertTrue(meta['applied'])
                self.assertTrue(meta['border_cropped'])
                self.assertTrue(meta['deskew_applied'])
                err = abs(meta['skew_deg'] - true_angle)
                self.assertLess(err, 0.25,
                                f'angle={true_angle}, est={meta["skew_deg"]}, err={err}')
                # Corners of output should be paper-bright, not black
                arr = np.asarray(out.convert('L'))
                self.assertGreater(float(np.median(arr[:10, :10])), 180.0)
                self.assertGreater(float(np.median(arr[-10:, -10:])), 180.0)

    def test_dark_desk_margin_like_gate20_does_not_clip_notation(self):
        # Desk around 60 gray, off-white paper around 225
        clean = _make_staff_page(w=900, h=1200, bg=(225, 220, 210), ink=(20, 20, 20))
        canvas = Image.new('RGB', (1020, 1320), (55, 50, 48))
        canvas.paste(clean, (60, 60))
        out, meta = page_prep.prep_page(canvas)
        self.assertTrue(meta['border_cropped'])
        self.assertEqual(meta['border_crop_box'], [60, 60, 960, 1260])
        self.assertEqual(out.size, (900, 1200))

    def test_deterministic_on_repeated_calls(self):
        clean = _make_staff_page(w=800, h=1000)
        rot = clean.rotate(2.7, resample=Image.BICUBIC, expand=True,
                           fillcolor=(0, 0, 0))
        out1, m1 = page_prep.prep_page(rot)
        out2, m2 = page_prep.prep_page(rot)
        self.assertEqual(m1, m2)
        self.assertTrue(np.array_equal(np.asarray(out1), np.asarray(out2)))


if __name__ == '__main__':
    unittest.main()
