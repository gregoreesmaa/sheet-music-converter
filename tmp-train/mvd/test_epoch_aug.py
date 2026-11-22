"""Unit tests for per-epoch fresh redraws & on-the-fly augmentation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import epoch_aug  # noqa: E402
import train  # noqa: E402


class TestEpochAug(unittest.TestCase):
    def _sample_rgb(self, h=64, w=128):
        g = torch.Generator().manual_seed(7)
        x = torch.full((3, h, w), 0.92, dtype=torch.float32)
        # Draw dark horizontal "staff lines" at fixed rows.
        for r in (12, 22, 32, 42, 52):
            x[:, r, 8:w - 8] = 0.08
        x = (x + 0.01 * torch.randn((3, h, w), generator=g)).clamp(0.0, 1.0)
        return x

    def test_determinism_same_epoch_and_fresh_across_epochs(self):
        rgb = self._sample_rgb()
        a0 = epoch_aug.augment_rgb_01(rgb, base_seed=42, epoch=0,
                                      sample_key='ds0001-1-s0_d0.jpg')
        b0 = epoch_aug.augment_rgb_01(rgb, base_seed=42, epoch=0,
                                      sample_key='ds0001-1-s0_d0.jpg')
        a1 = epoch_aug.augment_rgb_01(rgb, base_seed=42, epoch=1,
                                      sample_key='ds0001-1-s0_d0.jpg')
        self.assertTrue(torch.equal(a0, b0), 'same (seed, epoch, key) must be bit-exact')
        self.assertFalse(torch.equal(a0, a1), 'different epochs must draw fresh optics')
        self.assertEqual(a0.shape, rgb.shape)
        self.assertGreaterEqual(float(a0.min()), 0.0)
        self.assertLessEqual(float(a0.max()), 1.0)

    def test_staff_line_rows_preserved(self):
        rgb = self._sample_rgb()
        # With geom_strength=0.0, photometric-only keeps exact 0.0-px row alignment.
        aug0 = epoch_aug.augment_rgb_01(rgb, base_seed=123, epoch=3,
                                        sample_key='sq001-2-s1_d1.jpg',
                                        geom_strength=0.0)
        luma = aug0.mean(dim=0)[:, 16:-16].mean(dim=1)
        for r in (12, 22, 32, 42, 52):
            local = luma[r - 2:r + 3]
            self.assertEqual(int(local.argmin().item()), 2,
                             f'staff row {r} shifted vertically with geom_strength=0')
        # With geom_strength=1.0, coupled staffb warps identically with rgb!
        mean = torch.tensor(epoch_aug.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(epoch_aug.IMAGENET_STD).view(3, 1, 1)
        s0 = ((rgb - mean) / std).unsqueeze(0)
        staff = torch.zeros((1, 1, 64, 128), dtype=torch.float32)
        for r in (12, 22, 32, 42, 52):
            staff[0, 0, r, :] = 1.0
        pairs = [('in/sq001-2-s1_d1.jpg', 'b', 'sq001-2-s1')]
        out_rgb, _, out_staff = epoch_aug.augment_standardized_batch(
            s0, pairs, base_seed=123, epoch=3, aug_p=1.0,
            grayb=torch.zeros((1, 1, 64, 128)), staffb=staff, geom_strength=1.0)
        # Warped staff peaks must align with warped dark lines at mid-width.
        a01 = (out_rgb[0] * std + mean).clamp(0.0, 1.0).mean(dim=0)
        col = 64
        for r in (12, 22, 32, 42, 52):
            lo, hi = max(0, r - 6), min(64, r + 7)
            r_ink = lo + int(a01[lo:hi, col].argmin().item())
            r_stf = lo + int(out_staff[0, 0, lo:hi, col].argmax().item())
            self.assertLessEqual(abs(r_ink - r_stf), 1,
                                 f'coupled staffb diverged from warped ink at row {r}')

    def test_clean_floor_preservation_and_gray_sync(self):
        rgb0 = self._sample_rgb()
        mean = torch.tensor(epoch_aug.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(epoch_aug.IMAGENET_STD).view(3, 1, 1)
        s0 = (rgb0 - mean) / std
        batch_rgb = torch.stack([s0, s0.clone()], dim=0)
        gray0 = (rgb0 * torch.tensor(epoch_aug.LUMA_WEIGHTS).view(3, 1, 1)).sum(0)
        batch_gray = torch.stack([gray0, gray0.clone()], dim=0)
        pairs = [
            ('dir/ds0001-1-s0.png', 'dir/ds0001-1-s0-bekern.txt', 'ds0001-1-s0'),
            ('dir/ds0001-1-s0_d0.jpg', 'dir/ds0001-1-s0-bekern.txt', 'ds0001-1-s0'),
        ]
        out_rgb, out_gray = epoch_aug.augment_standardized_batch(
            batch_rgb, pairs, base_seed=9, epoch=2, aug_p=1.0, clean_p=0.0,
            grayb=batch_gray)
        # Clean row untouched.
        self.assertTrue(torch.equal(out_rgb[0], batch_rgb[0]))
        self.assertTrue(torch.equal(out_gray[0], batch_gray[0]))
        # Degraded row augmented and gray synchronized with augmented RGB.
        self.assertFalse(torch.equal(out_rgb[1], batch_rgb[1]))
        self.assertFalse(torch.equal(out_gray[1], batch_gray[1]))
        expected_rgb01 = (out_rgb[1] * std + mean).clamp(0.0, 1.0)
        expected_gray = (expected_rgb01 * torch.tensor(
            epoch_aug.LUMA_WEIGHTS).view(3, 1, 1)).sum(0)
        self.assertLess(float((out_gray[1] - expected_gray).abs().max()), 1e-5)

    def test_resolve_epoch_batch_substitutes_only_degraded(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ep1 = root / 'ep1'
            ep1.mkdir()
            fresh = ep1 / 'ds0001-1-s0.jpg'
            fresh.write_bytes(b'fakejpeg')
            batch = [
                ('in/ds0001-1-s0.png', 'in/ds0001-1-s0-bekern.txt', 'ds0001-1-s0'),
                ('in/ds0001-1-s0_d0.jpg', 'in/ds0001-1-s0-bekern.txt', 'ds0001-1-s0'),
            ]
            # Epoch 0 has no dir -> falls back to original.
            r0 = epoch_aug.resolve_epoch_batch(batch, epoch=0, redraw_root=root)
            self.assertEqual(r0, batch)
            # Epoch 1 substitutes degraded only, keeps clean PNG untouched.
            r1 = epoch_aug.resolve_epoch_batch(batch, epoch=1, redraw_root=root)
            self.assertEqual(r1[0][0], 'in/ds0001-1-s0.png')
            self.assertEqual(r1[1][0], str(fresh))

    def test_epoch_seed_deterministic_and_distinct(self):
        s0a = epoch_aug.epoch_seed(42, 0)
        s0b = epoch_aug.epoch_seed(42, 0)
        s1 = epoch_aug.epoch_seed(42, 1)
        self.assertEqual(s0a, s0b)
        self.assertNotEqual(s0a, s1)

    def test_train_resume_legacy_compat_for_epoch_aug_flags(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            # Legacy meta.json predating epoch_aug_* keys.
            legacy = {k: 0 for k in train.RESUME_KEYS
                      if k not in train.LEGACY_DEFAULTS}
            (out / 'meta.json').write_text(json.dumps(legacy), encoding='utf-8')
            # Current run at legacy defaults -> allowed.
            cur_default = dict(legacy)
            cur_default.update(train.LEGACY_DEFAULTS)
            self.assertIsNone(train.check_resume(out, cur_default))
            # Current run with non-default epoch_aug_p -> refused loudly.
            cur_on = dict(cur_default)
            cur_on['epoch_aug_p'] = 0.8
            msg = train.check_resume(out, cur_on)
            self.assertIsNotNone(msg)
            self.assertIn('epoch_aug_p', msg)


if __name__ == '__main__':
    unittest.main()
