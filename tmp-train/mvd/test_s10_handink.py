"""Spike S10: scribble/hand-ink training variants (TDD).

RED-first tests for wiring `hand-ink` variants into training:
`list_pairs(include_handink)` pairs `<stem>_h*.png/.jpg` with the
same bekern targets (order per stem: clean, degraded, handink),
and the `handink` fingerprint key + `--handink` flag keep
resume-exactness (a run with variants must refuse `--resume`
against a variant-less meta and vice versa).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train


def _layout(tmp):
    renders = Path(tmp) / 'renders'
    targets = Path(tmp) / 'targets'
    renders.mkdir()
    targets.mkdir()
    (targets / 's1-bekern.txt').write_text('bar\n', encoding='utf-8')
    (targets / 's2-bekern.txt').write_text('bar\n', encoding='utf-8')
    (renders / 's1.png').write_bytes(b'\x89PNG')
    (renders / 's1_d0.jpg').write_bytes(b'\xff\xd8')
    (renders / 's1_h0.png').write_bytes(b'\x89PNG')
    (renders / 's2.png').write_bytes(b'\x89PNG')
    return renders, targets


class ListPairsHandinkTest(unittest.TestCase):
    def test_default_excludes_handink(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            renders, targets = _layout(tmp)
            pairs = train.list_pairs(str(renders), str(targets))
            imgs = [Path(p[0]).name for p in pairs]
            self.assertNotIn('s1_h0.png', imgs)
            self.assertIn('s1.png', imgs)
            self.assertIn('s1_d0.jpg', imgs)

    def test_opt_in_order_clean_degraded_handink(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            renders, targets = _layout(tmp)
            pairs = train.list_pairs(str(renders), str(targets),
                                     include_handink=True)
            got = [(Path(p[0]).name, p[2]) for p in pairs]
            self.assertEqual(
                got,
                [('s1.png', 's1'), ('s1_d0.jpg', 's1'),
                 ('s1_h0.png', 's1'), ('s2.png', 's2')])

    def test_handink_jpg_also_pairs(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            renders, targets = _layout(tmp)
            (renders / 's2_h1.jpg').write_bytes(b'\xff\xd8')
            pairs = train.list_pairs(str(renders), str(targets),
                                     include_handink=True)
            imgs = [Path(p[0]).name for p in pairs]
            self.assertIn('s2_h1.jpg', imgs)


class HandinkFlagTest(unittest.TestCase):
    def test_fingerprint_has_handink_key(self):
        import argparse
        args = argparse.Namespace(
            height=256, lr=1e-4, seed=1, exclude=[], shuffle=False,
            no_amp=True, accum=4, init_from=None, trunk_lr_mult=0.5,
            warmup=500, cosine=True, sample_p=0.0, pages_file=None,
            batch_size=2, pretrained=True, no_degraded=False,
            tok_weight=1.0, tok_dropout=0.0, margin_weight=0.0,
            margin_siblings=8, margin=1.0, margin_every=1,
            mwer_weight=0.0, mwer_beams=4, mwer_every=1,
            mwer_maxlen=576, action_weight=0.0, action_every=1,
            specaug_p=0.0, specaug_slope=0.02, extra_chans=8,
            hybrid_cache='hc', heat_pt='hp', handink=True)
        fp = train.build_meta(args, [('i', 'b', 's')])
        self.assertIn('handink', fp)
        self.assertTrue(fp['handink'])
        self.assertIn('handink', train.RESUME_KEYS)


if __name__ == '__main__':
    unittest.main()
