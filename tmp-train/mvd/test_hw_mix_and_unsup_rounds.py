"""Unit tests for HW pure/aug mix, RISM incipit mix, and unsup-round schedulers.

Covers:
  1. Exact pure vs. augmented HW split (``hw_pure_frac=0.35`` skips ``epoch_aug``
     on ~35% of HW samples and augments ~65%, deterministically).
  2. ``--real-hw-corpus`` strictly loads labeled HW (enforcing ``bekern_ok`` and
     excluding any RISM-sourced rows).
  3. ``--rism-incipit-dir`` + ``--rism-mix-ratio`` semi-supervised mix tracked
     separately from labeled HW, plus multi-dir ``--unsup-strips-dir`` and
     ``--unlabeled-dirs``.
  4. Scheduled unsupervised rounds fire on exact cadence in both ``train.py``
     (``unsup_every_steps``) and ``staff_unet2d.py`` (``unsup_round_every``),
     with lazy-import safety when optional ``unsup_*`` modules are absent.
  5. Backward-compatible ``META_KEYS`` / ``RESUME_KEYS`` / ``LEGACY_DEFAULTS``
     for all new ``train.py`` flags.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
import torch

MVD_DIR = Path(__file__).resolve().parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import staff_unet2d as su  # noqa: E402
import train  # noqa: E402


class TestHwPureAugSplit(unittest.TestCase):
    """Exact pure vs. augmented HW split in ``apply_epoch_aug_with_hw_pure``."""

    def test_hw_pure_frac_35_skips_aug_on_35pct_and_augments_65pct(self):
        aug_calls: list[dict] = []

        def _mock_aug(rgb_std, batch_pairs, base_seed, epoch, **kw):
            aug_calls.append({
                'n': len(batch_pairs),
                'aug_p': kw.get('aug_p'),
                'clean_p': kw.get('clean_p'),
                'clean_anchor_frac': kw.get('clean_anchor_frac'),
            })
            out = rgb_std + 1.0
            grayb = kw.get('grayb')
            staffb = kw.get('staffb')
            if staffb is not None:
                return out, grayb, staffb
            return out, grayb

        n_pure = 0
        n_aug = 0
        # 250 steps x 4 HW items = 1,000 HW samples.
        for step in range(1, 251):
            batch = [(f'hw-work-{step}-{j}.png', 't.txt', f'hw-{step}-{j}')
                     for j in range(4)]
            rgb = torch.zeros((4, 3, 16, 32), dtype=torch.float32)
            out_rgb, _, _, stats = train.apply_epoch_aug_with_hw_pure(
                rgb, batch, seed=42, epoch=0, step=step,
                aug_p=0.0, clean_p=0.75, geom_strength=0.15,
                optics_profile='default',
                hw_indices=frozenset(range(4)),
                hw_pure_frac=0.35,
                aug_fn=_mock_aug,
            )
            n_pure += stats['hw_pure']
            n_aug += stats['hw_aug']
            # Verify tensor-level invariance: pure items stayed 0.0, aug items became 1.0.
            for j in range(4):
                val = float(out_rgb[j].mean().item())
                self.assertIn(val, (0.0, 1.0))
            self.assertEqual(int((out_rgb.mean(dim=(1, 2, 3)) == 0.0).sum().item()),
                             stats['hw_pure'])
            self.assertEqual(int((out_rgb.mean(dim=(1, 2, 3)) == 1.0).sum().item()),
                             stats['hw_aug'])

        total = n_pure + n_aug
        self.assertEqual(total, 1000)
        pure_rate = n_pure / total
        aug_rate = n_aug / total
        self.assertAlmostEqual(pure_rate, 0.35, delta=0.04)
        self.assertAlmostEqual(aug_rate, 0.65, delta=0.04)
        # Every HW-aug call must force aug_p=1.0, clean_p=1.0, clean_anchor_frac=0.0.
        self.assertTrue(len(aug_calls) > 0)
        for c in aug_calls:
            self.assertEqual(c['aug_p'], 1.0)
            self.assertEqual(c['clean_p'], 1.0)
            self.assertEqual(c['clean_anchor_frac'], 0.0)

    def test_hw_pure_frac_extremes_and_hybrid_tensors(self):
        def _mock_aug(rgb_std, batch_pairs, base_seed, epoch, **kw):
            out = rgb_std + 2.0
            grayb = kw.get('grayb')
            staffb = kw.get('staffb')
            g_out = (grayb + 0.5) if grayb is not None else None
            s_out = staffb
            if staffb is not None:
                return out, g_out, s_out
            return out, g_out

        batch = [
            ('printed-0.png', 't.txt', 'p0'),
            ('hw-0.png', 't.txt', 'h0'),
            ('hw-1.png', 't.txt', 'h1'),
        ]
        rgb = torch.zeros((3, 3, 8, 16))
        gray = torch.zeros((3, 1, 8, 16))
        staff = torch.ones((3, 1, 8, 16))

        # hw_pure_frac=1.0 -> HW items completely untouched; printed item augmented.
        r1, g1, s1, st1 = train.apply_epoch_aug_with_hw_pure(
            rgb, batch, seed=7, epoch=0, step=1,
            aug_p=0.0, clean_p=0.75, hw_indices=frozenset((1, 2)),
            hw_pure_frac=1.0, gray_d=gray, staff_d=staff, aug_fn=_mock_aug)
        self.assertEqual(st1, {'hw_pure': 2, 'hw_aug': 0})
        self.assertAlmostEqual(float(r1[0].mean()), 2.0)
        self.assertAlmostEqual(float(r1[1].mean()), 0.0)
        self.assertAlmostEqual(float(r1[2].mean()), 0.0)
        self.assertAlmostEqual(float(g1[1].mean()), 0.0)
        self.assertTrue(torch.equal(s1, staff))

        # hw_pure_frac=0.0 -> all HW items augmented.
        r0, g0, _, st0 = train.apply_epoch_aug_with_hw_pure(
            rgb, batch, seed=7, epoch=0, step=1,
            aug_p=0.0, clean_p=0.0, hw_indices=frozenset((1, 2)),
            hw_pure_frac=0.0, gray_d=gray, staff_d=staff, aug_fn=_mock_aug)
        self.assertEqual(st0, {'hw_pure': 0, 'hw_aug': 2})
        self.assertAlmostEqual(float(r0[0].mean()), 0.0)
        self.assertAlmostEqual(float(r0[1].mean()), 2.0)
        self.assertAlmostEqual(float(r0[2].mean()), 2.0)
        self.assertAlmostEqual(float(g0[1].mean()), 0.5)


class TestHwCorpusAndRismIncipitMix(unittest.TestCase):
    """--real-hw-corpus, --rism-incipit-dir, and multi-dir --unsup-strips-dir."""

    def test_load_real_hw_corpus_filters_bekern_ok_and_blocks_rism(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            img_ok = root / 'lieder_p1.jpg'
            img_bad = root / 'bad_p1.jpg'
            img_rism = root / 'rism_p1.jpg'
            img_q = root / 'hw-mozart-s0.png'
            img_q_d = root / 'hw-mozart-s0_d0.jpg'
            for p in (img_ok, img_bad, img_rism, img_q, img_q_d):
                Image.new('RGB', (32, 32), 'white').save(p)
            bek_ok = root / 'lieder-bekern.txt'
            bek_q = root / 'hw-mozart-s0-bekern.txt'
            bek_ok.write_text('clef-G-2 quarter C 4 bar\n', encoding='utf-8')
            bek_q.write_text('clef-G-2 half D 4 bar\n', encoding='utf-8')

            rows = [
                {
                    'work_id': 'schumann-test',
                    'source': 'openscore-lieder-pdm',
                    'page_paths': [str(img_ok)],
                    'bekern_path': str(bek_ok),
                    'bekern_ok': True,
                },
                {
                    'work_id': 'broken-work',
                    'source': 'openscore-lieder-pdm',
                    'page_paths': [str(img_bad)],
                    'bekern_path': str(bek_ok),
                    'bekern_ok': False,
                },
                {
                    'work_id': 'rism-should-be-blocked',
                    'source': 'rism_hw_unlabeled',
                    'page_paths': [str(img_rism)],
                    'bekern_path': str(bek_ok),
                    'bekern_ok': True,
                },
                {
                    'work_id': 'hw-mozart-k428',
                    'source': 'real-hw-train-quartets',
                    'page_paths': [str(img_q)],
                    'degraded_strip_paths': [str(img_q_d)],
                    'bekern_paths': [str(bek_q)],
                    'bekern_ok': True,
                },
            ]
            (root / 'train_works.jsonl').write_text(
                '\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')

            pairs, _ = train.load_real_hw_corpus_pairs(str(root), include_degraded=True)
            img_names = [Path(p[0]).name for p in pairs]
            self.assertIn('lieder_p1.jpg', img_names)
            self.assertIn('hw-mozart-s0.png', img_names)
            self.assertIn('hw-mozart-s0_d0.jpg', img_names)
            self.assertNotIn('bad_p1.jpg', img_names)
            self.assertNotIn('rism_p1.jpg', img_names)

    def test_rism_incipit_dir_and_separate_mix_ratios(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rism_dir = root / 'rism_incipits'
            (rism_dir / 'incipit_strips').mkdir(parents=True)
            Image.new('RGB', (64, 32), 'white').save(
                rism_dir / 'incipit_strips' / 'rism001-s0.png')
            (rism_dir / 'incipit_strips' / 'rism001-s0-bekern.txt').write_text(
                'clef-G-2 quarter C 4 bar\n', encoding='utf-8')

            rism_pairs, _ = train.load_rism_incipit_pairs(str(rism_dir))
            self.assertEqual(len(rism_pairs), 1)
            self.assertEqual(rism_pairs[0][2], 'rism001-s0')

            base_batch = [(f'p{i}.png', 'p.txt', f'p{i}') for i in range(4)]
            hw_pairs = [(f'hw{i}.png', 'h.txt', f'hw{i}') for i in range(3)]

            # Legacy no-op when both mix ratios are 0.0.
            m0, hi0, ri0 = train.mix_supervised_batch(
                base_batch, step=1, seed=0,
                hw_pairs=hw_pairs, hw_mix_ratio=0.0,
                rism_pairs=rism_pairs, rism_mix_ratio=0.0)
            self.assertEqual(m0, base_batch)
            self.assertEqual(hi0, frozenset())
            self.assertEqual(ri0, frozenset())

            # Measure mix ratios across 500 steps x 4 items = 2,000 slots.
            hw_cnt = rism_cnt = 0
            for st in range(1, 501):
                mb, hi, ri = train.mix_supervised_batch(
                    base_batch, step=st, seed=123,
                    hw_pairs=hw_pairs, hw_mix_ratio=0.15,
                    rism_pairs=rism_pairs, rism_mix_ratio=0.10)
                self.assertTrue(hi.isdisjoint(ri))
                hw_cnt += len(hi)
                rism_cnt += len(ri)
                for idx in hi:
                    self.assertIn(mb[idx], hw_pairs)
                for idx in ri:
                    self.assertIn(mb[idx], rism_pairs)

            self.assertAlmostEqual(hw_cnt / 2000.0, 0.15, delta=0.03)
            self.assertAlmostEqual(rism_cnt / 2000.0, 0.10, delta=0.03)

    def test_multi_dir_unsup_strips_and_unlabeled_pages(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            d1 = root / 'unlabeled_printed'
            d2 = root / 'rism_unlabeled'
            d3 = root / 'intl_hw'
            for sub in (d1, d2, d3):
                sub.mkdir(parents=True)
            Image.new('RGB', (32, 32), 'white').save(d1 / 'a.png')
            Image.new('RGB', (32, 32), 'white').save(d2 / 'b.jpg')
            Image.new('RGB', (32, 32), 'white').save(d3 / 'c.png')
            (d1 / 'ignore.txt').write_text('not an image', encoding='utf-8')

            stage2_imgs = train.list_unlabeled_images(f'{d1},{d2}')
            self.assertEqual([Path(p).name for p in stage2_imgs], ['a.png', 'b.jpg'])

            stage1_imgs = su.list_unlabeled_page_images(f'{d1},{d2},{d3}')
            self.assertEqual([p.name for p in stage1_imgs], ['a.png', 'b.jpg', 'c.png'])


class TestUnsupRoundSchedulers(unittest.TestCase):
    """Exact cadence firing for Stage-1 (epoch) and Stage-2 (step) unsup rounds."""

    def test_stage2_unsup_scheduler_cadence_and_mock_modules(self):
        fired_steps = []
        calls = []

        mock_s2 = types.SimpleNamespace(
            run_unsup_round=lambda **kw: (
                calls.append(('ema_enc', kw['step'], kw['steps'], len(kw['strip_paths'])))
                or {'loss_ema': 0.12}
            )
        )
        mock_bm = types.SimpleNamespace(
            run_bar_math_round=lambda **kw: (
                calls.append(('bar_math', kw['step'], kw['steps'], len(kw['strip_paths'])))
                or {'loss_bar': 0.05}
            )
        )

        net = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(net.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        strips = ['s0.png', 's1.png', 's2.png']

        for step in range(1, 11):
            if train.unsup_round_due(step, every_steps=3):
                fired_steps.append(step)
                res = train.run_stage2_unsup_round(
                    net=net, opt=opt, scaler=scaler, strip_paths=strips,
                    step=step, round_steps=25, mode='both', weight=0.25,
                    kl_beta=0.10, height=64, device='cpu',
                    unsup_stage2_mod=mock_s2, unsup_bar_math_mod=mock_bm,
                )
                self.assertEqual(res['ran'], ['ema_enc', 'bar_math'])

        self.assertEqual(fired_steps, [3, 6, 9])
        self.assertEqual(len(calls), 6)
        self.assertEqual(calls[0], ('ema_enc', 3, 25, 3))
        self.assertEqual(calls[1], ('bar_math', 3, 25, 3))

        # Graceful skip when optional modules do not exist.
        with mock.patch.object(train, '_lazy_import_mvd_module', return_value=None):
            res_missing = train.run_stage2_unsup_round(
                net=net, opt=opt, scaler=scaler, strip_paths=strips,
                step=3, round_steps=10, mode='both',
            )
            self.assertEqual(res_missing['ran'], [])

    def test_stage1_unsup_round_cadence_and_hw_hard_negatives(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            # Minimal 2-page supervised manifest.
            rows = []
            for idx in range(2):
                arr = np.full((128, 96, 3), 245, dtype=np.uint8)
                for k in range(-2, 3):
                    arr[64 + k * 5, 10:86, :] = 20
                fname = f'p{idx}.png'
                Image.fromarray(arr, mode='RGB').save(td_path / fname)
                rows.append({
                    'image': fname,
                    'staves': [{'y_center': 64.0, 'sp': 5.0, 'x0': 10.0, 'x1': 86.0, 'system_id': 0}],
                })
            manifest = td_path / 'manifest.jsonl'
            manifest.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')

            # Two unlabeled dirs (e.g. printed + RISM HW).
            u1 = td_path / 'u_printed'
            u2 = td_path / 'u_rism_hw'
            u1.mkdir()
            u2.mkdir()
            Image.new('RGB', (96, 128), 'white').save(u1 / 'up0.png')
            Image.new('RGB', (96, 128), 'white').save(u2 / 'ur0.jpg')

            fired = []

            def _mock_stage1_round(**kw):
                fired.append({
                    'epoch': kw['epoch'],
                    'steps': kw['steps'],
                    'ema_decay': kw['ema_decay'],
                    'n_images': len(kw['image_paths']),
                    'ema_frozen': all(not p.requires_grad for p in kw['ema_model'].parameters()),
                })
                return {'loss_ema': 0.08}

            ckpt = su.train_staff_unet2d(
                manifest=manifest,
                out_dir=td_path / 'out',
                epochs=4,
                batch_size=2,
                target_h=128,
                target_w=96,
                base_channels=16,
                device='cpu',
                augment=False,
                unlabeled_dirs=f'{u1},{u2}',
                unsup_round_every=2,
                unsup_round_steps=7,
                unsup_ema_decay=0.995,
                unsup_round_fn=_mock_stage1_round,
            )
            self.assertTrue(ckpt.exists())
            self.assertEqual([f['epoch'] for f in fired], [2, 4])
            for f in fired:
                self.assertEqual(f['steps'], 7)
                self.assertAlmostEqual(f['ema_decay'], 0.995)
                self.assertEqual(f['n_images'], 2)
                self.assertTrue(f['ema_frozen'])

            # Verify hw_hard_neg_prob calls unsup_stage1.augment_hw_hard_negatives when present.
            hn_called = []
            mock_u1_mod = types.SimpleNamespace(
                augment_hw_hard_negatives=lambda img, tgt, rng=None: (
                    hn_called.append(img.shape) or (img, tgt)
                )
            )
            ds = su.StaffUNetDataset(
                manifest, target_h=128, target_w=96, augment=True,
                hw_hard_neg_prob=1.0,
            )
            with mock.patch.object(su, '_lazy_import_unsup_stage1', return_value=mock_u1_mod):
                x, y = ds[0]
                self.assertEqual(tuple(x.shape), (3, 128, 96))
                self.assertEqual(tuple(y.shape), (4, 64, 48))
            self.assertEqual(len(hn_called), 1)


class TestNewTrainFlagsResumeCompat(unittest.TestCase):
    """New train.py flags preserve legacy --resume compatibility at defaults."""

    NEW_KEYS = {
        'real_hw_corpus': None,
        'hw_mix_ratio': 0.0,
        'hw_pure_frac': 0.35,
        'rism_incipit_dir': None,
        'rism_mix_ratio': 0.0,
        'unsup_strips_dir': None,
        'unsup_every_steps': 0,
        'unsup_round_steps': 100,
        'unsup_mode': 'both',
        'unsup_weight': 0.25,
        'unsup_kl_beta': 0.10,
    }

    def test_old_meta_resumes_at_defaults_and_refuses_nondefault(self):
        for k, v in self.NEW_KEYS.items():
            self.assertIn(k, train.META_KEYS, k)
            self.assertIn(k, train.RESUME_KEYS, k)
            self.assertIn(k, train.LEGACY_DEFAULTS, k)
            self.assertEqual(train.LEGACY_DEFAULTS[k], v, k)

        base = {k: f'v-{k}' for k in train.RESUME_KEYS}
        base.update(train.LEGACY_DEFAULTS)
        old = {k: v for k, v in base.items() if k not in self.NEW_KEYS}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / 'meta.json').write_text(json.dumps(old), encoding='utf-8')
            self.assertIsNone(train.check_resume(str(out), dict(base)))

            for k, bad in (
                ('hw_mix_ratio', 0.15),
                ('hw_pure_frac', 0.50),
                ('rism_mix_ratio', 0.08),
                ('unsup_every_steps', 1000),
                ('unsup_mode', 'ema_enc'),
            ):
                with self.subTest(key=k):
                    mut = dict(base)
                    mut[k] = bad
                    msg = train.check_resume(str(out), mut)
                    self.assertIsNotNone(msg)
                    self.assertIn(repr(k), msg)


if __name__ == '__main__':
    unittest.main()
