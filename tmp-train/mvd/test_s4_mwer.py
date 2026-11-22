"""Spike S4: MWER + CE mixed fine-tune (TDD).

RED-first tests for the cost-weighted risk (3x dropped barlines and
premature EOS, 2x dropped micro-glyphs/modifiers, 1x everything else)
and the pure MWER loss (mean-risk baseline). Integration is proven by
a tiny margin-style smoke run with --mwer-weight > 0.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data
import model
import train

BAR = data.VOCAB.index('bar')
EOS = data.EOS_ID
MICRO = set(model.MODIFIER_IDS)
PITCH_C = data.VOCAB.index('C')
DOT = data.VOCAB.index('dot')


def _ids(toks):
    return data.encode(toks)


class WeightedRiskTest(unittest.TestCase):
    def test_identical_scores_zero(self):
        ref = _ids(['clef-G-2', 'C', '4', 'quarter', 'bar'])
        self.assertAlmostEqual(
            train.weighted_risk(ref, list(ref), BAR, EOS, MICRO), 0.0)

    def test_dropped_bar_costs_three(self):
        ref = _ids(['C', '4', 'quarter', 'bar', 'D', '4'])
        hyp = _ids(['C', '4', 'quarter', 'D', '4'])
        self.assertAlmostEqual(
            train.weighted_risk(ref, hyp, BAR, EOS, MICRO), 3.0)

    def test_dropped_dot_costs_two(self):
        ref = _ids(['C', '4', 'quarter', 'dot', 'bar'])
        hyp = _ids(['C', '4', 'quarter', 'bar'])
        self.assertAlmostEqual(
            train.weighted_risk(ref, hyp, BAR, EOS, MICRO), 2.0)

    def test_dropped_pitch_costs_one(self):
        ref = _ids(['C', '4', 'quarter', 'bar'])
        hyp = _ids(['4', 'quarter', 'bar'])
        self.assertAlmostEqual(
            train.weighted_risk(ref, hyp, BAR, EOS, MICRO), 1.0)

    def test_premature_eos_costs_three(self):
        ref = _ids(['C', '4', 'quarter', 'bar', 'D', '4', 'quarter',
                    'bar'])
        hyp = _ids(['C', '4', 'quarter', 'bar']) + [EOS]
        risk = train.weighted_risk(ref, hyp, BAR, EOS, MICRO)
        # replace tail (D 4 quarter bar = 1+1+1+3, hyp EOS side +1)
        # + premature EOS 3
        self.assertAlmostEqual(risk, 10.0)

    def test_eos_at_end_not_premature(self):
        ref = _ids(['C', '4', 'quarter', 'bar']) + [EOS]
        risk = train.weighted_risk(ref, list(ref), BAR, EOS, MICRO)
        self.assertAlmostEqual(risk, 0.0)


class MwerLossTest(unittest.TestCase):
    def test_equal_risks_zero(self):
        import torch
        m = train.mwer_loss(torch.tensor([-1.0, -2.0]),
                            torch.tensor([5.0, 5.0]))
        self.assertAlmostEqual(float(m), 0.0)

    def test_exact_value(self):
        import torch
        # p = softmax([-1, -2]) = [e/(e+1), 1/(e+1)]; rbar = 2.0
        # loss = p0*(0-2) + p1*(4-2) = 2*(p1 - p0)
        m = train.mwer_loss(torch.tensor([-1.0, -2.0]),
                            torch.tensor([0.0, 4.0]))
        import math
        p0 = math.exp(-1) / (math.exp(-1) + math.exp(-2))
        self.assertAlmostEqual(float(m), 2 * (1 - 2 * p0), places=5)

    def test_single_beam_zero(self):
        import torch
        m = train.mwer_loss(torch.tensor([-1.0]), torch.tensor([3.0]))
        self.assertAlmostEqual(float(m), 0.0)


class ReturnAllTest(unittest.TestCase):
    def test_best_matches_default_path(self):
        import torch
        torch.manual_seed(1)
        net = model.MvdNet(vocab=len(data.VOCAB) + 3, d_model=64,
                           layers=1, heads=8)
        net.eval()
        imgs = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            default = net.beam_search(imgs, beam=2, max_len=8,
                                      constrain=False)
            allb = net.beam_search(imgs, beam=2, max_len=8,
                                   constrain=False, return_all=True)
        self.assertEqual(allb.shape[0], 1)
        self.assertEqual(allb.shape[1], 2)
        l1 = default.shape[1]
        self.assertTrue(torch.equal(allb[0, 0, :l1], default[0]))

    def test_rank_order_covers_k_rows(self):
        import torch
        torch.manual_seed(2)
        net = model.MvdNet(vocab=len(data.VOCAB) + 3, d_model=64,
                           layers=1, heads=8)
        net.eval()
        imgs = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            allb = net.beam_search(imgs, beam=3, max_len=6,
                                   constrain=False, return_all=True)
        self.assertEqual(tuple(allb.shape[:2]), (2, 3))


class BeamHypTest(unittest.TestCase):
    def test_keeps_eos_drops_trailing_pads(self):
        hyp = train.beam_hyp([5, 10, data.EOS_ID, data.PAD_ID,
                              data.PAD_ID], data.EOS_ID, len(data.VOCAB))
        self.assertEqual(hyp, [5, 10, data.EOS_ID])

    def test_cuts_at_pad_without_eos(self):
        hyp = train.beam_hyp([5, 10, data.PAD_ID, 7], data.EOS_ID,
                             len(data.VOCAB))
        self.assertEqual(hyp, [5, 10])

    def test_pure_content_unchanged(self):
        hyp = train.beam_hyp([5, 10, 20], data.EOS_ID, len(data.VOCAB))
        self.assertEqual(hyp, [5, 10, 20])


if __name__ == '__main__':
    unittest.main()
