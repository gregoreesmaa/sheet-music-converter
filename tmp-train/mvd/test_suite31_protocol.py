"""Protocol hygiene & regression tests for Suite31 / Suite31+HW evaluation.

Verifies:
1. Frozen ``suite31_manifest.json`` SHA-256 is unchanged, has 31 pieces, and
   every referenced clean/photo/gt file exists on disk.
2. Companion ``suite31_hw_manifest.json`` (45 pieces) preserves the exact 31
   frozen pieces in identical order + 14 non-NC handwritten pieces, matching
   ``suite31_hw_only_manifest.json`` (14 pieces) with exact stratum counts:
   ``typeset=31, hw_landscape=5, hw_portrait_debussy=4, hw_portrait_pdm=5``.
3. ``trim_max_len_repeat_loop`` and ``measure_decode`` / ``eval_sys_gate``
   honor ``max_len=768`` without silently clamping at ``564``.
4. ``run_suite31`` emits ``suite31_manifest_sha256``, ``suite31_manifest``,
   per-piece ``stratum``, and structured ``suite31_strata`` (``typeset``,
   ``hw_all``, ``hw_landscape``, ``hw_portrait_debussy``, ``hw_portrait_pdm``)
   with blind misses scored as ``0.0`` (never excluded).

Run (CPU-only):
    $env:CUDA_VISIBLE_DEVICES=""; .\\.venv\\Scripts\\python.exe -m unittest tmp-train/mvd/test_suite31_protocol.py -v
"""

import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data  # noqa: E402
import eval_sys_gate as esg  # noqa: E402
import measure_decode as mdec  # noqa: E402
import staff_unet2d as su  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUITE31_DIR = REPO_ROOT / 'data-gen' / 'score-corpus' / 'suite31'
FROZEN_SHA256 = (
    'b58b4a36b70ddf553cc2bbb63396868971a6284a21d5813d6b11891fc8a7b3ac'
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestSuite31ManifestIntegrity(unittest.TestCase):
    """Frozen manifest immutability and companion HW manifest hygiene."""

    def test_frozen_suite31_manifest_sha256_and_files(self):
        p = SUITE31_DIR / 'suite31_manifest.json'
        self.assertTrue(p.exists(), 'suite31_manifest.json missing')
        self.assertEqual(
            _sha256(p), FROZEN_SHA256,
            'FROZEN suite31_manifest.json SHA-256 changed! Must never be modified.',
        )
        doc = json.loads(p.read_text(encoding='utf-8'))
        pieces = doc['pieces']
        self.assertEqual(len(pieces), 31)
        self.assertEqual(doc.get('pieces_expected'), 31)
        slugs = [x['slug'] for x in pieces]
        self.assertEqual(len(slugs), len(set(slugs)), 'duplicate slugs in suite31')
        for x in pieces:
            for k in ('clean', 'photo', 'gt'):
                f = SUITE31_DIR / x[k]['file']
                self.assertTrue(f.exists(), f'missing {k} file: {f}')
            self.assertEqual(esg.classify_suite31_stratum(x), 'typeset')

    def test_companion_hw_manifests_strata_and_non_nc_licenses(self):
        frozen = json.loads(
            (SUITE31_DIR / 'suite31_manifest.json').read_text(encoding='utf-8')
        )
        hw45 = json.loads(
            (SUITE31_DIR / 'suite31_hw_manifest.json').read_text(encoding='utf-8')
        )
        hw14 = json.loads(
            (SUITE31_DIR / 'suite31_hw_only_manifest.json').read_text(encoding='utf-8')
        )
        self.assertEqual(len(hw45['pieces']), 45)
        self.assertEqual(len(hw14['pieces']), 14)

        # First 31 pieces of suite31_hw_manifest must match frozen suite31 exactly
        self.assertEqual(
            [x['slug'] for x in hw45['pieces'][:31]],
            [x['slug'] for x in frozen['pieces']],
        )
        # Last 14 pieces must match suite31_hw_only_manifest exactly
        self.assertEqual(
            [x['slug'] for x in hw45['pieces'][31:]],
            [x['slug'] for x in hw14['pieces']],
        )
        # Zero duplicate slugs across all 45 pieces
        all_slugs = [x['slug'] for x in hw45['pieces']]
        self.assertEqual(len(all_slugs), len(set(all_slugs)))

        # Verify exact stratum counts and zero NC licenses
        counts = {}
        for x in hw45['pieces']:
            st = esg.classify_suite31_stratum(x)
            counts[st] = counts.get(st, 0) + 1
            if x.get('handwritten'):
                lic = str(x.get('license', ''))
                self.assertNotIn('NC', lic.upper(), f'NC license in {x["slug"]}: {lic}')
                for k in ('clean', 'photo', 'gt'):
                    f = SUITE31_DIR / x[k]['file']
                    self.assertTrue(f.exists(), f'missing {k} file: {f}')

        self.assertEqual(counts, {
            'typeset': 31,
            'hw_landscape': 5,
            'hw_portrait_debussy': 4,
            'hw_portrait_pdm': 5,
        })

    def test_readme_exists_and_pins_frozen_sha256(self):
        readme = SUITE31_DIR / 'README.md'
        self.assertTrue(readme.exists(), 'suite31/README.md missing')
        text = readme.read_text(encoding='utf-8')
        self.assertIn(FROZEN_SHA256, text)
        for token in ('typeset', 'hw_landscape', 'hw_portrait_debussy',
                      'hw_portrait_pdm', 'hw_all', '--blind-photo'):
            self.assertIn(token, text)


class TestMaxLenUncapped768(unittest.TestCase):
    """Ensure max_len=768 is honored end-to-end instead of clamped at 564."""

    def test_trim_max_len_repeat_loop_honors_768_not_564(self):
        prefix = ['clef-G-2', 'key+0', 'time-4-4', 'voice-1', 'half', 'G', '4', 'bar']
        loop_bar = ['voice-1', 'quarter', 'C', '4', 'quarter', 'E', '4',
                    'quarter', 'G', '4', 'quarter', 'C', '5', 'bar']  # 14 toks

        # A 600-token repeating stream did NOT hit max_len=768 -> must be untouched
        toks600 = list(prefix)
        while len(toks600) < 600:
            toks600.extend(loop_bar)
        toks600 = toks600[:600]
        self.assertEqual(
            esg.trim_max_len_repeat_loop(toks600, max_len=768),
            toks600,
            '600-tok strip must not be trimmed when max_len=768',
        )

        # A 768-token repeating stream HIT max_len=768 -> must be trimmed to 1 copy
        toks768 = list(prefix)
        while len(toks768) < 768:
            toks768.extend(loop_bar)
        toks768 = toks768[:768]
        self.assertEqual(
            esg.trim_max_len_repeat_loop(toks768, max_len=768),
            prefix + loop_bar,
        )

    def test_measure_decode_and_eval_forward_max_len_768(self):
        seen_max_lens = []
        t = data.TOKEN_TO_ID
        beam_ids = [t['clef-G-2'], t['key+0'], t['time-4-4'],
                    t['whole'], t['C'], t['4'], t['bar']]

        class SpyNet:
            def beam_search(self, images, beam=3, strict=True, max_len=564, **kw):
                seen_max_lens.append(max_len)
                return [list(beam_ids)]

        net = SpyNet()
        ten = torch.zeros(1, 3, 256, 64)
        mdec.decode_system_with_measure_fallback(net, ten, beam=3, max_len=768)
        self.assertEqual(seen_max_lens[-1], 768)

        # decode_curved_system also forwards max_len=768
        page_img = Image.new('RGB', (400, 300), (255, 255, 255))
        x_ctrl = np.linspace(20.0, 380.0, 4, dtype=np.float64)
        y_ctrl = np.full(4, 120.0, dtype=np.float64)
        csys = su.CurvedSystem(
            staves=[su.CurvedStaff(
                ctrl_pts=np.stack([x_ctrl, y_ctrl], axis=-1),
                sp_ctrl=np.full(4, 10.0, dtype=np.float64),
                x_start=20.0, x_end=380.0,
                sp_median=10.0, confidence=0.95)],
            x_start=20.0, x_end=380.0,
            polygon=np.zeros((8, 2), dtype=np.float64), confidence=0.95)
        with tempfile.TemporaryDirectory(prefix='t_ml768_') as tmp:
            esg.decode_curved_system(
                net, page_img, csys, heat_net=None, device='cpu',
                height=256, beam=3, max_len=768, workdir=Path(tmp),
                stem='c768', extra_chans=0)
        self.assertEqual(seen_max_lens[-1], 768)


class TestRunSuite31StrataAndAuditMetadata(unittest.TestCase):
    """Verify run_suite31 reports manifest SHA-256 and per-stratum breakdowns."""

    def test_run_suite31_emits_manifest_sha_and_all_strata(self):
        with tempfile.TemporaryDirectory(prefix='t_s31_strata_') as tmp:
            tmpd = Path(tmp)
            suite = tmpd / 'suite'
            suite.mkdir()
            workdir = tmpd / 'work'
            workdir.mkdir()

            specs = [
                {'slug': 'p-typeset', 'handwritten': False},
                {'slug': 'hw-mozart-k464', 'handwritten': True},
                {'slug': 'hw-debussy-lindaraja', 'handwritten': True,
                 'orientation': 'portrait', 'license': 'CC-BY-4.0 (HugoSchtr/debussy-omr)'},
                {'slug': 'hw-hensel-morgengruss-d-b', 'handwritten': True,
                 'orientation': 'portrait', 'source': 'openscore-lieder-pdm'},
            ]
            manifest_pieces = []
            for sp in specs:
                slug = sp['slug']
                for sfx in ('gt.musicxml', 'clean.png', 'photo.png'):
                    fp = suite / f'{slug}_{sfx}'
                    if sfx.endswith('.png'):
                        Image.new('RGB', (200, 300), (255, 255, 255)).save(fp)
                    else:
                        fp.write_text('<score-partwise/>', encoding='utf-8')
                entry = dict(sp)
                entry.update({
                    'gt': {'file': f'{slug}_gt.musicxml'},
                    'clean': {'file': f'{slug}_clean.png'},
                    'photo': {'file': f'{slug}_photo.png'},
                })
                manifest_pieces.append(entry)

            mpath = suite / 'suite31_hw_manifest.json'
            mpath.write_text(
                json.dumps({'pieces': manifest_pieces}), encoding='utf-8')
            expected_sha = _sha256(mpath)

            ref_toks = ['clef-G-2', 'key+0', 'time-4-4', 'whole', 'C', '4', 'bar']
            ref_notes = workdir / 'ref_notes.json'
            ref_notes.write_text('{}', encoding='utf-8')

            x_ctrl = np.linspace(20.0, 180.0, 4, dtype=np.float64)
            y_ctrl = np.full(4, 120.0, dtype=np.float64)
            csys = su.CurvedSystem(
                staves=[su.CurvedStaff(
                    ctrl_pts=np.stack([x_ctrl, y_ctrl], axis=-1),
                    sp_ctrl=np.full(4, 10.0, dtype=np.float64),
                    x_start=20.0, x_end=180.0,
                    sp_median=10.0, confidence=0.95)],
                x_start=20.0, x_end=180.0,
                polygon=np.zeros((8, 2), dtype=np.float64), confidence=0.95)

            # Make hw-debussy-lindaraja photo a blind miss (index 5 = piece 2 photo)
            call_idx = {'n': 0}

            def fake_extract(sunet, pil_img, device='cpu', meta=None):
                idx = call_idx['n']
                call_idx['n'] += 1
                if idx == 5:
                    return []
                return [csys]

            def fake_decode_curved(net, page_img, curved_sys, heat_net, device,
                                   height, beam, max_len, wd, stem, **kw):
                esg.write_hyp_bekern(wd, stem, ref_toks)
                return {'tokens': ref_toks, 'ntokens': len(ref_toks),
                        'eos_miss': False, 'vocab_valid': True,
                        'decode_secs': 0.01, 'loop_trimmed': False,
                        'geom': 'staff-unet2d', 'n_staves': 1, 'n_chunks': 1}

            def fake_run_cli(cmd, **kw):
                out_path = Path(cmd[cmd.index('--out') + 1])
                out_path.write_text('{}', encoding='utf-8')
                return mock.Mock(returncode=0, stderr='')

            args = argparse.Namespace(
                suite31_dir=str(suite),
                suite31_manifest=str(mpath),
                limit_pieces=None,
                pieces=None,
                blind_photo=True,
                no_notation=False,
                height=256,
                beam=3,
                max_len=768,
                hybrid=False,
                ink_veto=False,
                ilme_lambda=0.0,
                lattice_repair=False,
                adapt_steps=0,
                adapt_lr=0.1,
                stride='4x8',
                verify_beam=False,
                verify_alpha=2.0,
                dynamic_staff_unet=False,
            )

            with mock.patch.object(esg, 'suite31_page_refs',
                                   return_value=(ref_toks, ref_notes)), \
                 mock.patch.object(esg, 'extract_page_curved_systems',
                                   side_effect=fake_extract), \
                 mock.patch.object(esg, 'decode_curved_system',
                                   side_effect=fake_decode_curved), \
                 mock.patch.object(esg, 'run_cli', side_effect=fake_run_cli), \
                 mock.patch.object(esg, 'score_notes_pair',
                                   return_value={'f1': 0.8, 'ser': 0.2,
                                                 'proxy_fixes': 10,
                                                 'zeroed': False,
                                                 'zero_reason': None}), \
                 mock.patch.object(esg, 'score_notation_bekerns',
                                   return_value={'notation_f1': 0.75,
                                                 'notation_drift': 0.75,
                                                 'notation_zeroed': False,
                                                 'notation_zero_reason': None}):
                rows, summary = esg.run_suite31(
                    args, None, None, workdir, 'cpu', extra_chans=1,
                    staff_unet=object(),
                    staff_unet_meta={'target_h': 256, 'target_w': 192})

            # Audit metadata
            self.assertEqual(summary['suite31_manifest_sha256'], expected_sha)
            self.assertEqual(summary['suite31_manifest'], str(mpath))
            self.assertEqual(summary['n_pieces'], 4)
            self.assertEqual(summary['n_scored'], 4)
            self.assertAlmostEqual(summary['blind_photo_detected_rate'], 0.75)

            # Per-stratum structure
            strata = summary['suite31_strata']
            self.assertEqual(
                set(strata.keys()),
                {'typeset', 'hw_landscape', 'hw_portrait_debussy',
                 'hw_portrait_pdm', 'hw_all'},
            )
            self.assertEqual(strata['typeset']['n_pieces'], 1)
            self.assertAlmostEqual(strata['typeset']['photo_f1_mean'], 0.8)
            self.assertAlmostEqual(strata['typeset']['photo_ser_mean'], 0.2)
            self.assertEqual(strata['typeset']['photo_fixes_median'], 10)
            self.assertAlmostEqual(strata['typeset']['blind_photo_detected_rate'], 1.0)

            # Blind miss in hw_portrait_debussy scores 0.0 and is NOT excluded
            self.assertEqual(strata['hw_portrait_debussy']['n_scored'], 1)
            self.assertEqual(strata['hw_portrait_debussy']['photo_f1_mean'], 0.0)
            self.assertEqual(strata['hw_portrait_debussy']['photo_ser_mean'], 1.0)
            self.assertEqual(strata['hw_portrait_debussy']['photo_notation_f1_mean'], 0.0)
            self.assertEqual(strata['hw_portrait_debussy']['blind_photo_detected_rate'], 0.0)

            # hw_all averages 2 detected (0.8) + 1 miss (0.0) = 1.6 / 3
            self.assertEqual(strata['hw_all']['n_pieces'], 3)
            self.assertEqual(strata['hw_all']['n_scored'], 3)
            self.assertAlmostEqual(strata['hw_all']['photo_f1_mean'], 1.6 / 3.0)
            self.assertAlmostEqual(strata['hw_all']['blind_photo_detected_rate'], 2.0 / 3.0)

            # Backward-compatible summary keys preserved
            self.assertEqual(summary['suite31_hw_n_pieces'], 3)
            self.assertEqual(summary['suite31_typeset_n_pieces'], 1)
            self.assertAlmostEqual(summary['suite31_hw_photo_f1_mean'], 1.6 / 3.0)


if __name__ == '__main__':
    unittest.main()
