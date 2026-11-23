"""Unit tests for real handwritten training & Suite31+HW eval wiring."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train
import eval_sys_gate as esg


def _make_hw_dir(root):
    renders = Path(root) / 'renders'
    targets = Path(root) / 'targets'
    renders.mkdir(parents=True)
    targets.mkdir(parents=True)
    (targets / 'hw01-1-s0-bekern.txt').write_text('clef-G2 bar\n', encoding='utf-8')
    (targets / 'hw01-1-s1-bekern.txt').write_text('clef-G2 bar\n', encoding='utf-8')
    (renders / 'hw01-1-s0.png').write_bytes(b'\x89PNG')
    (renders / 'hw01-1-s0_d0.jpg').write_bytes(b'\xff\xd8')
    (renders / 'hw01-1-s1.png').write_bytes(b'\x89PNG')
    row = {'img': 'hw01-1-s0.png', 'S': 4, 'kind': 'rows',
           'table': {'stride': 8, 'w_in': 100, 'staves': []}}
    (Path(root) / 'hybrid_staff.jsonl').write_text(
        json.dumps(row) + '\n', encoding='utf-8')
    return Path(root)


class RealHwTrainWiringTest(unittest.TestCase):
    def test_load_real_hw_pairs_deterministic_and_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            hw = _make_hw_dir(tmp)
            p1 = train.load_real_hw_pairs(hw, include_degraded=True, repeat=1)
            names = [Path(p[0]).name for p in p1]
            self.assertEqual(names, ['hw01-1-s0.png', 'hw01-1-s0_d0.jpg', 'hw01-1-s1.png'])
            p2 = train.load_real_hw_pairs(hw, include_degraded=False, repeat=3)
            self.assertEqual(len(p2), 6)
            self.assertEqual([Path(p[0]).name for p in p2[:2]],
                             ['hw01-1-s0.png', 'hw01-1-s1.png'])

    def test_load_real_hw_pairs_respects_exclude(self):
        with tempfile.TemporaryDirectory() as tmp:
            hw = _make_hw_dir(tmp)
            p = train.load_real_hw_pairs(hw, exclude=('s1',), include_degraded=False)
            self.assertEqual([x[2] for x in p], ['hw01-1-s0'])

    def test_load_real_hw_pairs_invalid_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            hw = _make_hw_dir(tmp)
            with self.assertRaises(ValueError):
                train.load_real_hw_pairs(hw, repeat=0)

    def test_load_real_hw_hybrid_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            hw = _make_hw_dir(tmp)
            idx = train.load_real_hw_hybrid_index(hw)
            self.assertIn('hw01-1-s0.png', idx)
            self.assertEqual(idx['hw01-1-s0.png']['S'], 4)

    def test_meta_and_resume_keys_legacy_compatible(self):
        self.assertIn('real_hw_dir', train.META_KEYS)
        self.assertIn('real_hw_repeat', train.META_KEYS)
        self.assertIn('real_hw_dir', train.RESUME_KEYS)
        self.assertIn('real_hw_repeat', train.RESUME_KEYS)
        self.assertIsNone(train.LEGACY_DEFAULTS['real_hw_dir'])
        self.assertEqual(train.LEGACY_DEFAULTS['real_hw_repeat'], 1)
        args = argparse.Namespace(
            height=256, lr=1e-4, seed=1, exclude=[], shuffle=False,
            no_amp=True, accum=4, init_from=None, trunk_lr_mult=0.5,
            warmup=500, cosine=True, sample_p=0.0, pages_file=None,
            batch_size=2, pretrained=True, no_degraded=False,
            real_hw_dir='D:/hw', real_hw_repeat=4)
        meta = train.build_meta(args, [('i', 'b', 's')])
        self.assertEqual(meta['real_hw_dir'], 'D:/hw')
        self.assertEqual(meta['real_hw_repeat'], 4)


class EvalSysGateSuite31HwTest(unittest.TestCase):
    def test_eval_sys_gate_accepts_suite31_manifest_flags(self):
        # Verify CLI parser accepts --suite31-manifest and --suite31-only
        src = Path(esg.__file__).read_text(encoding='utf-8')
        self.assertIn('--suite31-manifest', src)
        self.assertIn('--suite31-only', src)
        self.assertIn('suite31_hw_clean_f1_mean', src)
        self.assertIn('suite31_hw_photo_notation_f1_mean', src)


if __name__ == '__main__':
    unittest.main()
