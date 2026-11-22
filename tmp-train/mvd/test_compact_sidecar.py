"""Unit tests for compact hybrid sidecars and on-the-fly fallback.

Covers:
1. Bit-exact parity between legacy ``.hybrid.pt`` and compact
   ``.hybrid.npz`` (PNG and WebP codecs, grayscale and RGB strips).
2. Deterministic ``staff`` grid reconstruction vs ``rectify.py``.
3. Quantized ``uint8`` + zlib 7-channel HeatNet heatmap round-trip
   (max abs error <= 1/255) and ``heat_features.heat_channels_for_sidecar``
   cached vs on-the-fly fallback.
4. On-the-fly rectification (zero sidecar files on disk) bit-exactness
   and integration with ``train.filter_hybrid_pairs`` / ``train.load_sidecar``
   / ``train.collate_hybrid``.
5. ``build_hybrid_cache.stage_sidecars(..., fmt='compact')`` and
   ``compact_sidecar.convert_legacy_sidecar`` migration + backward
   compatibility with legacy ``.hybrid.pt`` files.
6. Real-corpus parity & compression ratio check (when corpus sidecars
   are present on disk).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_hybrid_cache as bhc  # noqa: E402
import compact_sidecar as cs  # noqa: E402
import heat_features as hf  # noqa: E402
import rectify as R  # noqa: E402
import train  # noqa: E402

HEIGHT = 256


def _make_strip(tmp: Path, stem: str, *, color: bool = False, staves: int = 2,
                w: int = 400, h: int = 200):
    """Create a synthetic system strip PNG + bekern + staff table row."""
    if color:
        rng = np.random.default_rng(42)
        arr = np.full((h, w, 3), 245, dtype=np.uint8)
        arr[:, :, 0] = np.clip(arr[:, :, 0] - rng.integers(0, 6, (h, w)), 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] - rng.integers(0, 12, (h, w)), 0, 255)
        for s in range(staves):
            for k in range(5):
                y = 24 + s * 76 + k * 14
                arr[y - 1:y + 2, 10:w - 10, :] = (35, 30, 28)
        img = Image.fromarray(arr, mode='RGB')
    else:
        arr = np.full((h, w), 250, dtype=np.uint8)
        for s in range(staves):
            for k in range(5):
                y = 24 + s * 76 + k * 14
                arr[y - 1:y + 2, 10:w - 10] = 40
        img = Image.fromarray(arr, mode='L').convert('RGB')
    img_path = tmp / (stem + '.png')
    img.save(img_path)
    bek_path = tmp / (stem + '-bekern.txt')
    bek_path.write_text(
        'clef-G-2 key-1 time-3-4 voice-1 bar-rest bar\n', encoding='utf-8')
    lines5 = [[float(24 + s * 76 + k * 14) for k in range(5)]
              for s in range(staves)]
    row = {
        'img': img_path.name,
        'S': staves,
        'kind': 'rows',
        'table': bhc.rows_to_table(lines5, w),
        'sp': 14.0,
        'source': 'test',
        'skip': False,
    }
    return img_path, bek_path, row


class TestCompactSidecarFormat(unittest.TestCase):
    def test_staff_grid_reconstruction_bit_exact(self):
        img = Image.new('RGB', (320, 200), (255, 255, 255))
        for s in (1, 2, 3, 4):
            lines = [[10.0 + i * 40.0 + k * 8.0 for k in range(5)]
                     for i in range(s)]
            rect_img, ref = R.rectify_system_strip(
                img, lines, target_sp=10.0, height=HEIGHT)
            rebuilt = cs.reconstruct_staff_grid(
                s, HEIGHT, rect_img.size[0], target_sp=10.0)
            self.assertTrue(
                torch.equal(ref, rebuilt),
                f'reconstructed staff grid diverged for S={s}')

    def test_png_and_webp_bit_exact_parity_gray_and_rgb(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for idx, color in enumerate((False, True)):
                stem = f'strip_{idx}'
                img_path, _, row = _make_strip(tmp, stem, color=color, staves=2)
                # Build legacy .hybrid.pt
                _, st, _ = bhc._sidecar_one((str(tmp), row, HEIGHT, 10.0))
                self.assertEqual(st, 'ok')
                leg_path = cs.legacy_sidecar_path_for(img_path)
                leg_rgb, leg_gray, leg_staff, leg_s = train.load_sidecar(
                    str(img_path), HEIGHT)

                for codec in ('png', 'webp'):
                    cpath = tmp / f'{stem}_{codec}.hybrid.npz'
                    cs.convert_legacy_sidecar(leg_path, cpath, codec=codec)
                    self.assertTrue(cpath.exists())
                    self.assertLess(cpath.stat().st_size,
                                    leg_path.stat().st_size)
                    c_rgb, c_gray, c_staff, c_s = cs.load_compact_sidecar(
                        cpath, HEIGHT, standardize_fn=train.standardize_rgb)
                    self.assertEqual(c_s, leg_s)
                    self.assertTrue(torch.equal(c_rgb, leg_rgb),
                                    f'rgb mismatch ({codec}, color={color})')
                    self.assertTrue(torch.equal(c_gray, leg_gray),
                                    f'gray mismatch ({codec}, color={color})')
                    self.assertTrue(torch.equal(c_staff, leg_staff),
                                    f'staff mismatch ({codec}, color={color})')

    def test_height_mismatch_and_slot_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rect_img = Image.new('RGB', (64, HEIGHT), (240, 240, 240))
            cpath = tmp / 'slot.hybrid.npz'
            cs.save_compact_sidecar(
                cpath, rect_img, img_name='slot.png', S=9, height=HEIGHT)
            with self.assertRaises(ValueError):
                cs.load_compact_sidecar(cpath, HEIGHT + 1)
            _, _, _, nslot = cs.load_compact_sidecar(cpath, HEIGHT)
            self.assertEqual(nslot, 4)

    def test_quantized_heat_roundtrip_error_bound(self):
        rng = torch.Generator().manual_seed(123)
        probs = torch.rand((len(hf.HEAT_CLASSES), 64, 128), generator=rng)
        # Include exact 0.0 and 1.0 extremes
        probs[:, 0, 0] = 0.0
        probs[:, 0, 1] = 1.0
        packed, shape = cs.quantize_heat(probs)
        restored = cs.dequantize_heat(packed, shape)
        self.assertEqual(tuple(restored.shape), tuple(probs.shape))
        max_err = float((probs - restored).abs().max())
        self.assertLessEqual(max_err, 0.5 / 255.0 + 1e-7)
        self.assertLessEqual(max_err, 1.0 / 255.0)
        self.assertEqual(float(restored[0, 0, 0]), 0.0)
        self.assertEqual(float(restored[0, 0, 1]), 1.0)

    def test_heat_channels_for_sidecar_cached_vs_on_the_fly(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            img_path, _, row = _make_strip(tmp, 'heat_s0', color=False, staves=2)
            hn = hf.HeatNet(device='cpu')
            # First: on-the-fly (no compact sidecar yet)
            _, gray, _, _ = cs.rectify_on_the_fly(
                img_path, row, height=HEIGHT, standardize_fn=train.standardize_rgb)
            h_otf = hf.heat_channels_for_sidecar(img_path, gray, heat_net=hn)
            self.assertEqual(tuple(h_otf.shape),
                             (len(hf.HEAT_CLASSES), HEIGHT, gray.shape[2]))

            # Save compact sidecar WITH quantized heatmaps
            rect_img = Image.fromarray(
                (gray[0].numpy() * 255.0).round().astype(np.uint8), mode='L')
            cpath = cs.compact_sidecar_path_for(img_path)
            cs.save_compact_sidecar(
                cpath, rect_img, img_name=img_path.name, S=2, height=HEIGHT,
                heat_probs=h_otf)
            # Load via heat_channels_for_sidecar WITHOUT passing heat_net
            # (proves cached heat is used, no HeatNet call needed)
            h_cached = hf.heat_channels_for_sidecar(img_path, gray, heat_net=None)
            self.assertEqual(tuple(h_cached.shape), tuple(h_otf.shape))
            max_err = float((h_otf - h_cached).abs().max())
            self.assertLessEqual(max_err, 1.0 / 255.0)


class TestOnTheFlyAndPipelineIntegration(unittest.TestCase):
    def test_on_the_fly_matches_legacy_and_compact_bit_for_bit(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            img_path, bek_path, row = _make_strip(
                tmp, 'otf-s0', color=True, staves=2)
            pair = (str(img_path), str(bek_path), 'otf-s0')
            hybrid_index = {img_path.name: row}

            # Zero sidecar files on disk
            self.assertIsNone(cs.resolve_sidecar_path(img_path))
            kept_no_otf, dc, _ = train.filter_hybrid_pairs([pair])
            self.assertEqual(len(kept_no_otf), 0)
            self.assertEqual(dc, 1)

            kept_otf, dc2, dd2 = train.filter_hybrid_pairs(
                [pair], hybrid_index=hybrid_index, allow_on_the_fly=True)
            self.assertEqual(len(kept_otf), 1)
            self.assertEqual((dc2, dd2), (0, 0))

            rgb_o, tgt_o, mask_o, gray_o, staff_o = train.collate_hybrid(
                [pair], HEIGHT, hybrid_index=hybrid_index)

            # Now materialize compact sidecar via stage_sidecars(fmt='compact')
            workdir = tmp / 'work'
            workdir.mkdir()
            (workdir / 'hybrid_staff.jsonl').write_text(
                json.dumps(row) + '\n', encoding='utf-8')
            bhc.stage_sidecars(
                workdir, tmp, height=HEIGHT, jobs=1, fmt='compact', codec='png')
            self.assertTrue(cs.compact_sidecar_path_for(img_path).exists())
            self.assertFalse(cs.legacy_sidecar_path_for(img_path).exists())

            # filter_hybrid_pairs (default args) now finds the compact sidecar
            kept_c, dc3, dd3 = train.filter_hybrid_pairs([pair])
            self.assertEqual(len(kept_c), 1)
            self.assertEqual((dc3, dd3), (0, 0))

            rgb_c, tgt_c, mask_c, gray_c, staff_c = train.collate_hybrid(
                [pair], HEIGHT)
            self.assertTrue(torch.equal(rgb_o, rgb_c))
            self.assertTrue(torch.equal(tgt_o, tgt_c))
            self.assertTrue(torch.equal(gray_o, gray_c))
            self.assertTrue(torch.equal(staff_o, staff_c))
            if mask_o is None:
                self.assertIsNone(mask_c)
            else:
                self.assertTrue(torch.equal(mask_o, mask_c))

    def test_convert_legacy_sidecar_remove_legacy(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            img_path, _, row = _make_strip(tmp, 'mig-s0', color=False, staves=1)
            _, st, _ = bhc._sidecar_one((str(tmp), row, HEIGHT, 10.0))
            self.assertEqual(st, 'ok')
            leg = cs.legacy_sidecar_path_for(img_path)
            self.assertTrue(leg.exists())
            dest = cs.convert_legacy_sidecar(leg, codec='png', remove_legacy=True)
            self.assertTrue(dest.exists())
            self.assertFalse(leg.exists())
            rgb, gray, staff, nslot = train.load_sidecar(str(img_path), HEIGHT)
            self.assertEqual(nslot, 1)
            self.assertEqual(rgb.shape[1], HEIGHT)
            self.assertEqual(gray.shape[1], HEIGHT)
            self.assertTrue(bool((staff == 1.0).any()))

    def test_real_corpus_parity_if_present(self):
        corpus = (Path(__file__).resolve().parent.parent / 'mvd-sysdata-full')
        if not corpus.exists():
            self.skipTest('mvd-sysdata-full not present')
        pts = sorted(corpus.glob('*.hybrid.pt'))[:6]
        if not pts:
            self.skipTest('no legacy .hybrid.pt files in mvd-sysdata-full')
        orig_total = 0
        comp_total = 0
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for pt in pts:
                orig_total += pt.stat().st_size
                dest = tmp / (pt.name[:-len('.hybrid.pt')] + '.hybrid.npz')
                cs.convert_legacy_sidecar(pt, dest, codec='png')
                comp_total += dest.stat().st_size
                l_rgb, l_gray, l_staff, l_s = cs.load_legacy_sidecar(
                    pt, HEIGHT, standardize_fn=train.standardize_rgb)
                c_rgb, c_gray, c_staff, c_s = cs.load_compact_sidecar(
                    dest, HEIGHT, standardize_fn=train.standardize_rgb)
                self.assertEqual(c_s, l_s, pt.name)
                self.assertTrue(torch.equal(c_rgb, l_rgb), pt.name)
                self.assertTrue(torch.equal(c_gray, l_gray), pt.name)
                self.assertTrue(torch.equal(c_staff, l_staff), pt.name)
        ratio = orig_total / max(1, comp_total)
        self.assertGreater(ratio, 5.0,
                           f'expected >5x compression, got {ratio:.2f}x')


if __name__ == '__main__':
    unittest.main()
