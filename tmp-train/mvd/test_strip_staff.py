"""Unit tests for the blind windowed staff-curve detector (T2.4)."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strip_staff as ss


def synth_warped_staves(width=900, height=300, n_staves=3, sp=10.0,
                        bend=8.0, tilt_deg=3.0, seed=0):
    """White image with S warped 5-line staves + notehead dots + lyrics.

    Staff ``s`` top line: ``30 + s * 80 + bend * sin(2 pi x / W)`` tilted
    by ``tilt_deg`` about the center; lines ``sp`` apart, 2 px thick.
    Returns ``(img, truth[s][k](x))`` callables.
    """
    rng = np.random.RandomState(seed)
    img = np.full((height, width), 255, np.uint8)
    th = math.radians(tilt_deg)
    truth = []
    for s in range(n_staves):
        y0 = 30.0 + s * 80.0
        staff = []
        for k in range(5):
            def line(x, y0=y0, k=k):
                x = np.asarray(x, dtype=float)
                y = (y0 + k * sp + bend * np.sin(2 * np.pi * x / width))
                return (y + np.tan(th) * (x - width / 2)).tolist() \
                    if np.ndim(x) else float(y + np.tan(th) * (x - width / 2))
            staff.append(line)
        truth.append(staff)
    xs = np.arange(width)
    for staff in truth:
        for line in staff:
            ys = np.asarray(line(xs))
            for x, y in zip(xs, ys):
                y = int(round(y))
                img[max(0, y - 1):y + 2, x] = 0
    # Notehead dots on random lines/spaces + lyric text bars (distractors).
    for _ in range(60):
        s = rng.randint(n_staves)
        k = rng.randint(9)
        x = rng.randint(20, width - 20)
        y = int(round(30 + s * 80 + k * sp / 2
                      + bend * np.sin(2 * np.pi * x / width)
                      + np.tan(th) * (x - width / 2)))
        img[max(0, y - 3):y + 4, max(0, x - 4):x + 5] = 0
    for yb in (75, 155, 235):
        img[yb:yb + 3, 100:700:7] = 0  # dashed lyric baselines BETWEEN
        # staves (as in real vocal strips), never inside one
    return Image.fromarray(img, mode='L'), truth


class TestStripStaff(unittest.TestCase):
    def test_warped_accuracy(self):
        # Role of this module: fallback for the ~3% of clean strips
        # where RowNet fails (markers carry degraded strips; RowNet
        # carries the rest). Typical rows must be sub-pixel accurate;
        # the worst case is gap-bridging past notehead-pileup windows
        # (documented envelope, still far below raw warp amplitude).
        img, truth = synth_warped_staves()
        curves, info = ss.detect_strip_curves(img, 3)
        self.assertEqual(len(curves), 3)
        xs = np.arange(img.size[0])
        w = img.size[0]
        edge = w // 10
        worst_in, worst_all, meds = 0.0, 0.0, []
        for s in range(3):
            for k in range(5):
                ref = np.asarray(truth[s][k](xs), dtype=float)
                err = np.abs(curves[s][k] - ref)
                worst_all = max(worst_all, float(err.max()))
                worst_in = max(worst_in, float(err[edge:w - edge].max()))
                meds.append(float(np.median(err)))
        print('synth curve err interior-worst=%.2f global-worst=%.2f '
              'median=%.2f px (sp=%.1f cov=%.2f)' %
              (worst_in, worst_all, float(np.median(meds)),
               info['sp'], info['coverage']))
        self.assertLessEqual(float(np.median(meds)), 1.0)
        self.assertLessEqual(worst_in, 8.0)
        self.assertLessEqual(worst_all, 15.0)

    def test_determinism(self):
        img, _ = synth_warped_staves()
        c1, i1 = ss.detect_strip_curves(img, 3)
        c2, i2 = ss.detect_strip_curves(img, 3)
        for s in range(3):
            for k in range(5):
                np.testing.assert_array_equal(c1[s][k], c2[s][k])
        self.assertEqual(i1, i2)

    def test_table_roundtrip(self):
        # Control points are rounded to 2 decimals and window-center
        # kinks are re-interpolated, so the bound is the kink scale
        # (~0.2 px), not float epsilon.
        img, _ = synth_warped_staves()
        curves, _ = ss.detect_strip_curves(img, 3)
        back = ss.table_to_curves(ss.curves_to_table(curves))
        for s in range(3):
            for k in range(5):
                d = np.abs(back[s][k] - curves[s][k])
                self.assertLessEqual(float(d.max()), 0.25)

    def test_blank_fails(self):
        img = Image.new('L', (800, 256), 255)
        with self.assertRaises(ValueError):
            ss.detect_strip_curves(img, 2)

    def test_bad_n_staves(self):
        img, _ = synth_warped_staves()
        with self.assertRaises(ValueError):
            ss.detect_strip_curves(img, 0)
        with self.assertRaises(ValueError):
            ss.detect_strip_curves(img, 5)


if __name__ == '__main__':
    unittest.main()
