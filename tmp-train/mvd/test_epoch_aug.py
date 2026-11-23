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

    def test_calibrated_heavy_optics_determinism_and_clean_anchor_floor(self):
        rgb = self._sample_rgb()
        a0 = epoch_aug.augment_rgb_01(
            rgb, base_seed=42, epoch=0, sample_key='sq8438915-5-5-s0_d0.jpg',
            optics_profile='calibrated')
        b0 = epoch_aug.augment_rgb_01(
            rgb, base_seed=42, epoch=0, sample_key='sq8438915-5-5-s0_d0.jpg',
            heavy_optics=True)
        a1 = epoch_aug.augment_rgb_01(
            rgb, base_seed=42, epoch=1, sample_key='sq8438915-5-5-s0_d0.jpg',
            optics_profile='calibrated')
        self.assertTrue(torch.equal(a0, b0), 'calibrated and heavy_optics=True must be bit-identical')
        self.assertFalse(torch.equal(a0, a1), 'calibrated must draw fresh optics across epochs')

        # With clean_p=1.0 in calibrated mode, ~25% clean anchors stay untouched.
        mean = torch.tensor(epoch_aug.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(epoch_aug.IMAGENET_STD).view(3, 1, 1)
        s0 = (rgb - mean) / std
        n_clean = 40
        batch_rgb = s0.unsqueeze(0).repeat(n_clean, 1, 1, 1)
        pairs = [(f'dir/clean_{i:02d}.png', f'dir/clean_{i:02d}-bekern.txt', f'clean_{i:02d}')
                 for i in range(n_clean)]
        out_rgb, _ = epoch_aug.augment_standardized_batch(
            batch_rgb, pairs, base_seed=77, epoch=0,
            aug_p=1.0, clean_p=1.0, optics_profile='calibrated')
        n_untouched = sum(1 for i in range(n_clean) if torch.equal(out_rgb[i], batch_rgb[i]))
        self.assertGreaterEqual(n_untouched, 6, f'expected ~25% clean anchors untouched, got {n_untouched}/{n_clean}')
        self.assertLessEqual(n_untouched, 18, f'too many clean anchors untouched: {n_untouched}/{n_clean}')

    def test_calibrated_washout_contrast_and_dark_desk_borders(self):
        # Crisp ink (0.05) on clean paper (0.95).
        rgb = torch.full((3, 64, 256), 0.95, dtype=torch.float32)
        for r in (12, 22, 32, 42, 52):
            rgb[:, r, 24:232] = 0.05

        n_washed = 0
        n_desk = 0
        for k in range(32):
            out = epoch_aug.augment_rgb_01(
                rgb, base_seed=101, epoch=k, sample_key=f'sq{k:04d}-1-s0_d0.jpg',
                geom_strength=0.0, optics_profile='calibrated')
            luma = out.mean(dim=0)
            ink_luma = float(luma[[12, 22, 32, 42, 52], 48:208].mean())
            paper_luma = float(luma[[6, 17, 27, 37, 47, 58], 48:208].mean())
            # Ordering preserved: ink stays darker than paper.
            self.assertGreater(paper_luma, ink_luma + 0.03)
            if ink_luma >= 0.30 and (paper_luma - ink_luma) <= 0.55:
                n_washed += 1
            left_edge = float(luma[:, :4].mean())
            right_edge = float(luma[:, -4:].mean())
            if min(left_edge, right_edge) < 0.35 * paper_luma:
                n_desk += 1
        self.assertGreaterEqual(n_washed, 5, f'expected extreme washout samples, got {n_washed}/32')
        self.assertGreaterEqual(n_desk, 4, f'expected dark desk border samples, got {n_desk}/32')

    def test_calibrated_coupled_rgb_gray_staff_alignment_at_higher_geom(self):
        rgb = self._sample_rgb(h=64, w=256)
        mean = torch.tensor(epoch_aug.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(epoch_aug.IMAGENET_STD).view(3, 1, 1)
        s0 = ((rgb - mean) / std).unsqueeze(0).repeat(4, 1, 1, 1)
        gray = torch.zeros((4, 1, 64, 256), dtype=torch.float32)
        staff = torch.zeros((4, 1, 64, 256), dtype=torch.float32)
        for r in (12, 22, 32, 42, 52):
            staff[:, 0, r, :] = 1.0
        pairs = [(f'in/sq{i}-1-s0_d0.jpg', 'b', f'sq{i}-1-s0') for i in range(4)]

        out_rgb, out_gray, out_staff = epoch_aug.augment_standardized_batch(
            s0, pairs, base_seed=321, epoch=5, aug_p=1.0,
            grayb=gray, staffb=staff, geom_strength=0.35,
            optics_profile='calibrated')
        luma_w = torch.tensor(epoch_aug.LUMA_WEIGHTS).view(3, 1, 1)
        for b in range(4):
            a01 = (out_rgb[b] * std + mean).clamp(0.0, 1.0)
            exp_g = (a01 * luma_w).sum(dim=0, keepdim=True)
            self.assertLess(float((out_gray[b] - exp_g).abs().max()), 1e-5)
            luma = a01.mean(dim=0)
            # High-pass vertical contrast isolates thin dark staff lines even
            # under strong directional shadows or coffee-stain gradients.
            bg = torch.nn.functional.avg_pool1d(
                luma.T.unsqueeze(1), kernel_size=9, stride=1, padding=4
            ).squeeze(1).T
            hp = luma - bg
            for col in (64, 128, 192):
                for r in (12, 22, 32, 42, 52):
                    lo, hi = max(0, r - 5), min(64, r + 6)
                    r_ink = lo + int(hp[lo:hi, col].argmin().item())
                    r_stf = lo + int(out_staff[b, 0, lo:hi, col].argmax().item())
                    self.assertLessEqual(
                        abs(r_ink - r_stf), 1,
                        f'b={b} col={col} row={r}: ink={r_ink} vs staff={r_stf}',
                    )

    def test_calibrated_left_edge_clef_survives_dark_desk_border(self):
        # Tight-crop strip: clean white paper with a distinct dark "clef" block
        # right at the left edge (cols 2..10, rows 16..48) and dark staff lines.
        rgb = torch.full((3, 64, 256), 0.95, dtype=torch.float32)
        for r in (12, 22, 32, 42, 52):
            rgb[:, r, 2:254] = 0.05
        rgb[:, 16:48, 2:10] = 0.05  # left-edge clef

        n_left_desk = 0
        for k in range(32):
            out = epoch_aug.augment_rgb_01(
                rgb, base_seed=202, epoch=k, sample_key=f'sq{k:04d}-1-s0_d0.jpg',
                geom_strength=0.0, optics_profile='calibrated')
            luma = out.mean(dim=0)
            margin_row = luma[2:8, :32].mean(dim=0)
            clef_row = luma[18:46, :32].mean(dim=0)
            # Detect left dark desk border: sharp dark edge at col 0 that jumps up
            # by >=0.18 into the paper margin within the first 20 columns.
            col0 = float(margin_row[0].item())
            if col0 < 0.25 and float(margin_row[:24].max().item()) - col0 >= 0.18:
                n_left_desk += 1
                page_cols = torch.where(margin_row > col0 + 0.12)[0]
                self.assertGreater(page_cols.numel(), 0, f'seed k={k}: desk border swallowed left 32 px')
                page_x0 = int(page_cols[0].item())
            else:
                page_x0 = 0
            # Shifted-inward (or unshifted) clef must survive strictly inside the page
            # as a dark vertical stroke darker than the paper margin at the same x.
            x0 = page_x0 + 2
            x1 = min(32, x0 + 12)
            diff = margin_row[x0:x1] - clef_row[x0:x1]
            max_diff = float(diff.max().item())
            self.assertGreater(
                max_diff, 0.08,
                f'seed k={k}: left-edge clef erased (page_x0={page_x0}, max margin-clef={max_diff:.3f})',
            )
        self.assertGreaterEqual(n_left_desk, 4, f'expected >=4 left-desk samples, got {n_left_desk}/32')

    def test_calibrated_washout_shadow_blur_contrast_floor(self):
        # 2px-thick staff lines + continuous inked band (immune to horizontal desk shift).
        rgb = torch.full((3, 64, 256), 0.95, dtype=torch.float32)
        for r in (12, 22, 32, 42, 52):
            rgb[:, r:r + 2, 24:232] = 0.05
        # Solid beam/notehead band across central x-span.
        rgb[:, 26:31, 48:208] = 0.05

        n_washed = 0
        for k in range(64):
            out = epoch_aug.augment_rgb_01(
                rgb, base_seed=404, epoch=k, sample_key=f'sq{k:04d}-1-s0_d0.jpg',
                geom_strength=0.0, optics_profile='calibrated')
            luma = out.mean(dim=0)
            # Compare inked beam band vs adjacent paper band at identical x-columns
            # so any directional lighting gradient cancels out locally.
            note_luma = float(luma[27:30, 80:176].mean())
            paper_luma = float(luma[16:20, 80:176].mean())
            contrast = paper_luma - note_luma
            self.assertGreaterEqual(
                contrast, 0.08,
                f'seed k={k}: washout x shadow x blur collapsed contrast to {contrast:.4f} < 0.08 '
                f'(paper={paper_luma:.3f}, ink={note_luma:.3f})',
            )
            if note_luma >= 0.30:
                n_washed += 1
        self.assertGreaterEqual(n_washed, 10, f'expected >=10 washed-out samples, got {n_washed}/64')

    def test_duplicate_stem_repeat_seed_uniqueness_and_unique_determinism(self):
        rgb = self._sample_rgb()
        mean = torch.tensor(epoch_aug.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(epoch_aug.IMAGENET_STD).view(3, 1, 1)
        s0 = (rgb - mean) / std
        batch = s0.unsqueeze(0).repeat(3, 1, 1, 1)
        dup_pairs = [
            ('dir/hw-001-s0_d0.jpg', 'dir/hw-001-s0-bekern.txt', 'hw-001-s0'),
            ('dir/hw-001-s0_d0.jpg', 'dir/hw-001-s0-bekern.txt', 'hw-001-s0'),
            ('dir/hw-001-s0_d0.jpg', 'dir/hw-001-s0-bekern.txt', 'hw-001-s0'),
        ]
        out_a, _ = epoch_aug.augment_standardized_batch(
            batch, dup_pairs, base_seed=55, epoch=2, aug_p=1.0,
            optics_profile='calibrated')
        out_b, _ = epoch_aug.augment_standardized_batch(
            batch, dup_pairs, base_seed=55, epoch=2, aug_p=1.0,
            optics_profile='calibrated')
        # Deterministic across identical calls.
        self.assertTrue(torch.equal(out_a, out_b))
        # Each duplicate occurrence in the batch must get distinct augmentation.
        self.assertFalse(torch.equal(out_a[0], out_a[1]), 'dup 0 and dup 1 collided')
        self.assertFalse(torch.equal(out_a[0], out_a[2]), 'dup 0 and dup 2 collided')
        self.assertFalse(torch.equal(out_a[1], out_a[2]), 'dup 1 and dup 2 collided')

    def test_real_hw_excluded_from_clean_anchor_freeze(self):
        rgb = self._sample_rgb()
        mean = torch.tensor(epoch_aug.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(epoch_aug.IMAGENET_STD).view(3, 1, 1)
        s0 = (rgb - mean) / std
        n_hw = 40
        batch = s0.unsqueeze(0).repeat(n_hw, 1, 1, 1)
        pairs = [(f'dir/hw-real-{i:02d}-s0.png', f'dir/hw-real-{i:02d}-s0-bekern.txt', f'hw-real-{i:02d}-s0')
                 for i in range(n_hw)]
        out0, _ = epoch_aug.augment_standardized_batch(
            batch, pairs, base_seed=77, epoch=0, aug_p=1.0, clean_p=1.0,
            optics_profile='calibrated')
        out1, _ = epoch_aug.augment_standardized_batch(
            batch, pairs, base_seed=77, epoch=1, aug_p=1.0, clean_p=1.0,
            optics_profile='calibrated')
        n_frozen = sum(1 for i in range(n_hw) if torch.equal(out0[i], batch[i]))
        self.assertEqual(n_frozen, 0, f'hw-* real handwritten strips must not be frozen by clean_anchor_frac (got {n_frozen}/{n_hw})')
        n_varied = sum(1 for i in range(n_hw) if not torch.equal(out0[i], out1[i]))
        self.assertEqual(n_varied, n_hw, 'all hw-* strips must vary across epochs')

    def test_calibrated_batch_speed_budget(self):
        import time
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        b, h, w = 8, 256, 1024
        rgb = torch.randn((b, 3, h, w), device=dev, dtype=torch.float32)
        gray = torch.zeros((b, 1, h, w), device=dev, dtype=torch.float32)
        staff = torch.zeros((b, 1, h, w), device=dev, dtype=torch.float32)
        pairs = [(f'in/s{i}_d0.jpg', f'in/s{i}-bekern.txt', f's{i}') for i in range(b)]

        # Warmup
        for ep in range(3):
            epoch_aug.augment_standardized_batch(
                rgb, pairs, base_seed=42, epoch=ep, aug_p=1.0,
                grayb=gray, staffb=staff, geom_strength=0.30,
                optics_profile='calibrated')
        if dev.type == 'cuda':
            torch.cuda.synchronize(dev)

        iters = 15 if dev.type == 'cuda' else 2
        t0 = time.perf_counter()
        for ep in range(iters):
            epoch_aug.augment_standardized_batch(
                rgb, pairs, base_seed=42, epoch=ep + 10, aug_p=1.0,
                grayb=gray, staffb=staff, geom_strength=0.30,
                optics_profile='calibrated')
        if dev.type == 'cuda':
            torch.cuda.synchronize(dev)
        ms = (time.perf_counter() - t0) * 1000.0 / iters
        if dev.type == 'cuda':
            self.assertLess(ms, 8.0, f'CUDA calibrated batch too slow: {ms:.2f} ms/batch')
        else:
            self.assertLess(ms, 2000.0, f'CPU calibrated batch too slow: {ms:.2f} ms/batch')


if __name__ == '__main__':
    unittest.main()
