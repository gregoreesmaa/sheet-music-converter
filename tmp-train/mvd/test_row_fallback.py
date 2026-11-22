"""Unit tests for hardened RowNet photo fallback (row_fallback.py)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import row_fallback as rf


def _synth_probs(
    height: int,
    staff_tops: list[float],
    sp: float = 14.0,
    amp: float = 0.85,
    sigma: float = 1.0,
    noise_std: float = 0.02,
    shadow_amp: float = 0.0,
    drop_lines: set[tuple[int, int]] = frozenset(),
    seed: int = 0,
) -> np.ndarray:
    """Build a synthetic 1-D RowNet staff-probability curve."""
    rng = np.random.default_rng(seed)
    y = np.arange(height, dtype=np.float64)
    p = np.zeros(height, dtype=np.float64)
    for si, y0 in enumerate(staff_tops):
        for k in range(5):
            if (si, k) in drop_lines:
                continue
            center = y0 + k * sp
            p += amp * np.exp(-0.5 * ((y - center) / sigma) ** 2)
    if shadow_amp > 0.0:
        # Broad slow shadow hump across the middle of the strip.
        p += shadow_amp * np.exp(-0.5 * ((y - height * 0.5) / (height * 0.2)) ** 2)
    if noise_std > 0.0:
        p += rng.normal(0.0, noise_std, size=height)
    return np.clip(p, 0.0, 1.0).astype(np.float32)


class AdaptiveBinarizeTest(unittest.TestCase):
    def test_recovers_staff_lines_under_strong_shadow_gradient(self):
        h, w = 160, 240
        # Illumination drops from 240 on left to 90 on right (global <140 fails!).
        bg = np.linspace(240.0, 90.0, w, dtype=np.float32)[None, :].repeat(h, axis=0)
        img = bg.copy()
        line_rows = [40, 52, 64, 76, 88]
        for r in line_rows:
            img[r - 1:r + 1, 20:220] -= 55.0
        img = np.clip(img, 0.0, 255.0)

        # Global threshold <140 turns the entire right half black (>40% false ink).
        global_bina = (img < 140.0).astype(np.float32)
        self.assertGreater(float(global_bina[:, 180:].mean()), 0.8)

        # Adaptive binarization keeps paper clean on both sides and preserves lines.
        adap = rf.adaptive_binarize(img, block_size=31, offset=15.0)
        self.assertLess(float(adap[10:25, 180:].mean()), 0.05)
        for r in line_rows:
            self.assertGreater(float(adap[r, 30:210].mean()), 0.85)


class RowFallbackCurveTest(unittest.TestCase):
    def test_low_contrast_shadowed_below_legacy_threshold(self):
        # Peak amp 0.18 < legacy NEURAL_THRESHOLD 0.30, plus a 0.25 shadow hump!
        tops = [40.0, 170.0]
        probs = _synth_probs(
            height=280, staff_tops=tops, sp=14.0, amp=0.18,
            sigma=1.2, noise_std=0.015, shadow_amp=0.25, seed=1,
        )
        S, lines5, sp, info = rf.infer_staff_count(probs)
        self.assertEqual(S, 2)
        self.assertAlmostEqual(sp, 14.0, delta=1.0)
        for si, y0 in enumerate(tops):
            for k in range(5):
                self.assertAlmostEqual(lines5[si][k], y0 + k * 14.0, delta=1.5)

    def test_missing_occluded_line_still_recovers_exact_five_lines(self):
        # Interior lines (staff 0 line 2, staff 1 line 3) are completely occluded (0.0).
        tops = [35.0, 165.0, 295.0]
        probs = _synth_probs(
            height=420, staff_tops=tops, sp=12.0, amp=0.55,
            sigma=1.0, noise_std=0.03,
            drop_lines={(0, 2), (1, 3)}, seed=2,
        )
        S, lines5, sp, info = rf.infer_staff_count(probs)
        self.assertEqual(S, 3)
        self.assertAlmostEqual(sp, 12.0, delta=1.0)
        # Missing interior lines interpolated within 1.5 px.
        self.assertAlmostEqual(lines5[0][2], 35.0 + 2 * 12.0, delta=1.5)
        self.assertAlmostEqual(lines5[1][3], 165.0 + 3 * 12.0, delta=1.5)

        # Outer line occluded still preserves exact staff count S=3 and spacing.
        probs_outer = _synth_probs(
            height=420, staff_tops=tops, sp=12.0, amp=0.55,
            sigma=1.0, noise_std=0.03,
            drop_lines={(1, 4)}, seed=3,
        )
        S2, lines5_2, sp2, _ = rf.infer_staff_count(probs_outer)
        self.assertEqual(S2, 3)
        self.assertAlmostEqual(sp2, 12.0, delta=1.0)
        self.assertEqual(len(lines5_2[1]), 5)

    def test_smeared_tilted_photo_peaks_and_high_staff_counts_5_to_7(self):
        for n_staves in (5, 6, 7):
            sp = 11.0
            tops = [25.0 + s * 95.0 for s in range(n_staves)]
            height = int(tops[-1] + 5 * sp + 40)
            probs = _synth_probs(
                height=height, staff_tops=tops, sp=sp, amp=0.26,
                sigma=1.8, noise_std=0.02, shadow_amp=0.15, seed=10 + n_staves,
            )
            S, lines5, sp_est, _ = rf.infer_staff_count(probs, max_staves=8)
            self.assertEqual(S, n_staves, f'failed for S={n_staves}')
            self.assertAlmostEqual(sp_est, sp, delta=1.2)
            self.assertEqual(len(lines5), n_staves)

    def test_flat_noise_raises_value_error(self):
        rng = np.random.default_rng(99)
        probs = rng.uniform(0.0, 0.03, size=300).astype(np.float32)
        with self.assertRaises(ValueError):
            rf.infer_staff_count(probs)


if __name__ == '__main__':
    unittest.main()
