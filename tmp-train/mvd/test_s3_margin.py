"""Spike S3: sibling-margin contrastive fine-tune (TDD).

RED-first tests for the beat-cell sibling mutator (8 modifier-edited
siblings per reference) and the pure margin-loss formula. The mutator
is training-side (train.py): it edits ONLY modifier tokens inside one
randomly chosen inter-bar span, so barlines/headers/pitches outside the
span survive untouched.
"""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data
import model
import train


def _ids(toks):
    return data.encode(toks)


def _ref():
    return _ids(['clef-G-2', 'key+0', 'time-4-4',
                 'C', '4', 'quarter', 'dot', 'bar',
                 'D', '4', 'quarter', 'tie-start', 'E', '4', 'quarter',
                 'tie-stop', 'bar'])


class SiblingMutatorTest(unittest.TestCase):
    def test_eight_siblings_all_differ(self):
        ref = _ref()
        sibs = train.mutate_modifier_siblings(
            ref, 8, random.Random(0), model.MODIFIER_IDS, len(data.VOCAB))
        self.assertEqual(len(sibs), 8)
        for s in sibs:
            self.assertNotEqual(s, ref)

    def test_deterministic_in_seed(self):
        ref = _ref()
        kw = dict(n=8, mod_ids=model.MODIFIER_IDS,
                  vocab_size=len(data.VOCAB))
        a = train.mutate_modifier_siblings(ref, rng=random.Random(3),
                                           **kw)
        b = train.mutate_modifier_siblings(ref, rng=random.Random(3),
                                           **kw)
        self.assertEqual(a, b)

    def test_only_modifier_edits(self):
        ref = _ref()
        ref_set = set(ref)
        mods = set(model.MODIFIER_IDS)
        sibs = train.mutate_modifier_siblings(
            ref, 8, random.Random(1), model.MODIFIER_IDS, len(data.VOCAB))
        import difflib
        for s in sibs:
            sm = difflib.SequenceMatcher(a=ref, b=s, autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == 'equal':
                    continue
                # Every touched token on either side is a modifier
                # (bars/headers/pitches/durations never edited).
                for t in ref[i1:i2]:
                    self.assertIn(t, mods)
                for t in s[j1:j2]:
                    self.assertIn(t, mods)

    def test_bar_count_and_header_preserved(self):
        ref = _ref()
        bar = data.VOCAB.index('bar')
        n_bars = sum(1 for t in ref if t == bar)
        head = _ids(['clef-G-2', 'key+0', 'time-4-4'])
        sibs = train.mutate_modifier_siblings(
            ref, 8, random.Random(2), model.MODIFIER_IDS, len(data.VOCAB))
        for s in sibs:
            self.assertEqual(sum(1 for t in s if t == bar), n_bars)
            self.assertEqual(s[:len(head)], head)

    def test_no_bar_stream_does_not_crash(self):
        ref = _ids(['clef-G-2', 'C', '4', 'quarter', 'dot'])
        sibs = train.mutate_modifier_siblings(
            ref, 8, random.Random(4), model.MODIFIER_IDS, len(data.VOCAB))
        self.assertEqual(len(sibs), 8)
        for s in sibs:
            self.assertNotEqual(s, ref)

    def test_specials_in_ref_do_not_crash(self):
        # Training targets carry EOS (id 177, past len(VOCAB));
        # data.decode rejects it, so the mutator must tolerate
        # specials in the header scan and never edit them.
        ref = _ids(['clef-G-2', 'C', '4', 'quarter', 'dot']) + [data.EOS_ID]
        sibs = train.mutate_modifier_siblings(
            ref, 8, random.Random(5), model.MODIFIER_IDS, len(data.VOCAB))
        self.assertEqual(len(sibs), 8)
        for s in sibs:
            self.assertNotEqual(s, ref)
            self.assertEqual(s.count(data.EOS_ID), 1)


class MarginLossTest(unittest.TestCase):
    def test_zero_when_ref_wins_by_margin(self):
        import torch
        m = train.sibling_margin_loss(
            torch.tensor(-1.0),
            torch.tensor([-3.0, -5.0]), margin=1.0)
        self.assertAlmostEqual(float(m), 0.0)

    def test_positive_exact_value(self):
        import torch
        # relu(1 + (-1.5) - (-1.0)) = 0.5; relu(1-3+1)=0 -> mean 0.25
        m = train.sibling_margin_loss(
            torch.tensor(-1.0),
            torch.tensor([-1.5, -3.0]), margin=1.0)
        self.assertAlmostEqual(float(m), 0.25)

    def test_empty_siblings_zero(self):
        import torch
        m = train.sibling_margin_loss(
            torch.tensor(-1.0), torch.tensor([]), margin=1.0)
        self.assertAlmostEqual(float(m), 0.0)


class DecodeLogitsTest(unittest.TestCase):
    def test_matches_forward_minus_encode(self):
        import torch
        torch.manual_seed(0)
        net = model.MvdNet(vocab=len(data.VOCAB) + 3, d_model=64,
                           layers=1, heads=8)
        net.eval()
        imgs = torch.randn(1, 3, 64, 64)
        tgt = torch.tensor([[data.BOS_ID, 5, 10, 20]])
        with torch.no_grad():
            ref = net(imgs, tgt)
            mem = net.encode(imgs)
            got = net.decode_logits(mem, tgt)
        self.assertTrue(torch.allclose(ref, got, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
