"""Unit tests for marker-track extraction (T2.4)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import marker_tracks as mt


def synth_diff(width=1200, height=400, slopes=(0.03, -0.02), seed=0):
    """Abs-diff-like image: dotted tracks (blobs every 24 px) + speckle.

    Two staves x 5 lines at sp 12 with per-line slopes; blobs are 3x3
    squares of value 60; speckle is sparse value-20 noise. Returns
    ``(diff, gray, truth)`` with ``gray`` a fake strip (dark lines on
    paper 200) and ``truth[s][k]`` dense curves.
    """
    rng = np.random.RandomState(seed)
    diff = np.zeros((height, width), np.float32)
    gray = np.full((height, width), 200, np.float32)
    truth = []
    for s in range(2):
        staff = []
        for k in range(5):
            sl = slopes[s] + 0.002 * (k - 2)
            y0 = 40.0 + s * 160.0 + k * 12.0
            xs = np.arange(width, dtype=float)
            ys = y0 + sl * (xs - width / 2)
            staff.append(ys)
            for x in range(12, width - 12, 24):
                if rng.rand() < 0.12:
                    continue  # destroyed markers
                yy = int(round(y0 + sl * (x - width / 2)))
                diff[max(0, yy - 1):yy + 2, max(0, x - 1):x + 2] = 60.0
            gray[np.clip(ys.astype(int), 0, height - 1),
                 np.arange(width)] = 60.0
        truth.append(staff)
    n = 400
    ys = rng.randint(0, height, n)
    xs = rng.randint(0, width, n)
    diff[ys, xs] = 20.0
    return diff, gray, truth


def _thick_lines(width=1200, height=400, sp=12.0):
    """Paper with 3px-thick horizontal dark lines (2 staves x 5).

    The dotted-marker rig uses 1px sloped lines, whose single-row
    surround samples miss the ink (median stays paper), so a misplaced
    curve reads contrast ~0 and never trips the anti-line floor. Thick
    horizontal lines reproduce the production anti-line signature:
    a half-spacing-offset curve sees paper on-line but dark neighbor
    lines in its +/-1.5sp surround rows. Returns ``(gray, truth)``.
    """
    gray = np.full((height, width), 200, np.float32)
    truth = []
    for s in range(2):
        staff = []
        for k in range(5):
            y = 40.0 + s * 160.0 + k * sp
            staff.append(np.full(width, y))
            gray[int(y) - 1:int(y) + 2, :] = 60.0
        truth.append(staff)
    return gray, [ln for staff in truth for ln in staff]


class TestMarkerTracks(unittest.TestCase):
    def test_exact_recovery(self):
        diff, gray, truth = synth_diff()
        curves, info = mt.extract_curves(diff, gray, 2, 12.0)
        self.assertEqual(len(curves), 10)
        worst = 0.0
        flat = [ln for staff in truth for ln in staff]
        for c, ref in zip(curves, flat):
            worst = max(worst, float(np.abs(c - ref).max()))
        print('marker worst curve err: %.2f px (info=%s)' % (worst, info))
        self.assertLessEqual(worst, 2.0)

    def test_determinism(self):
        diff, gray, _ = synth_diff()
        c1, i1 = mt.extract_curves(diff, gray, 2, 12.0)
        c2, i2 = mt.extract_curves(diff, gray, 2, 12.0)
        for a, b in zip(c1, c2):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(i1, i2)

    def test_wrong_count_fails(self):
        diff, gray, _ = synth_diff()
        with self.assertRaises(ValueError):
            mt.extract_curves(diff, gray, 3, 12.0)

    def test_blank_fails(self):
        with self.assertRaises(ValueError):
            mt.extract_curves(np.zeros((400, 1200), np.float32),
                              np.full((400, 1200), 200, np.float32), 2, 12.0)

    def test_sibling_rescue(self):
        # One smooth junk curve (misassigned slot, sits on white paper)
        # inside two otherwise strong staves: rescue re-interpolates it
        # from its good siblings and validation passes.
        gray, flat = _thick_lines()
        curves = [np.array(ln, dtype=float) for ln in flat]
        # Half-spacing offset: maximally anti-line (paper band on the
        # curve, dark neighbor lines in the surround bands), mirroring
        # production misassignments. A full-blank-paper offset reads
        # contrast ~0 and passes the hard floor by design.
        curves[2] = np.array(flat[2], dtype=float) + 6.0
        h, w = gray.shape
        blobs = [(float(x), float(ln[x]))
                 for ln in flat for x in range(0, w, 24)]
        out, info = mt._sibling_rescue(gray, curves, 12.0, blobs, 2, {})
        self.assertEqual(info.get('rescued'), 1)
        worst = 0.0
        for ref in flat:
            d = min(float(np.abs(o - ref).max()) for o in out)
            worst = max(worst, d)
        print('rescue worst curve err: %.2f px' % worst)
        self.assertLessEqual(worst, 1.0)

    def test_sibling_rescue_refuses_weak_staff(self):
        # Three junk curves in one staff (<3 good siblings): no rescue.
        gray, flat = _thick_lines()
        curves = [np.array(ln, dtype=float) for ln in flat]
        curves[0] = np.array(flat[0], dtype=float) + 6.0
        curves[1] = np.array(flat[1], dtype=float) - 6.0
        curves[2] = np.array(flat[2], dtype=float) + 6.0
        h, w = gray.shape
        blobs = [(float(x), float(ln[x]))
                 for ln in flat for x in range(0, w, 24)]
        with self.assertRaises(ValueError):
            mt._sibling_rescue(gray, curves, 12.0, blobs, 2, {})


if __name__ == '__main__':
    unittest.main()
