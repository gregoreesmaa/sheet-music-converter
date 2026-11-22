"""Spike S7: fixes-per-page action loss (TDD).

RED-first tests for the token-level expected keystroke risk
(minimum-Bayes-risk at the token level: E_{v~p}[C[v, ref]], one
forward, no beams — unlike S4's sequence MWER) plus the
bar-conservation cost matrix (bar-involved mismatches 3x,
modifier-involved 2x, EOS-involved 2x, else 1x). Integration is
proven by a smoke run with --action-weight > 0.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data
import model
import train


class CostMatrixTest(unittest.TestCase):
    def test_zero_diagonal(self):
        import torch
        c = train.keystroke_cost_matrix(len(data.VOCAB) + 3,
                                        model.MODIFIER_IDS)
        d = torch.diagonal(c)
        self.assertTrue(bool((d == 0).all()))

    def test_bar_miss_costs_three(self):
        import torch
        c = train.keystroke_cost_matrix(len(data.VOCAB) + 3,
                                        model.MODIFIER_IDS)
        bar = data.VOCAB.index('bar')
        other = data.VOCAB.index('C')
        self.assertAlmostEqual(float(c[other, bar]), 3.0)
        self.assertAlmostEqual(float(c[bar, other]), 3.0)

    def test_modifier_sub_costs_two(self):
        import torch
        c = train.keystroke_cost_matrix(len(data.VOCAB) + 3,
                                        model.MODIFIER_IDS)
        dot = data.VOCAB.index('dot')
        other = data.VOCAB.index('quarter')
        self.assertAlmostEqual(float(c[other, dot]), 2.0)

    def test_plain_sub_costs_one(self):
        import torch
        c = train.keystroke_cost_matrix(len(data.VOCAB) + 3,
                                        model.MODIFIER_IDS)
        self.assertAlmostEqual(
            float(c[data.VOCAB.index('C'), data.VOCAB.index('D')]),
            1.0)


class ExpectedRiskTest(unittest.TestCase):
    def test_confident_correct_scores_zero(self):
        import torch
        nv = len(data.VOCAB) + 3
        c = train.keystroke_cost_matrix(nv, model.MODIFIER_IDS)
        logits = torch.full((1, 3, nv), -50.0)
        logits[0, 0, 5] = 50.0
        logits[0, 1, 10] = 50.0
        logits[0, 2, data.EOS_ID] = 50.0
        tgt = torch.tensor([[5, 10, data.EOS_ID]])
        r = train.expected_keystroke_risk(logits, tgt, c)
        self.assertAlmostEqual(float(r), 0.0, places=4)

    def test_fifty_fifty_exact_value(self):
        import torch
        nv = len(data.VOCAB) + 3
        c = train.keystroke_cost_matrix(nv, model.MODIFIER_IDS)
        bar = data.VOCAB.index('bar')
        other = data.VOCAB.index('C')
        logits = torch.full((1, 1, nv), -50.0)
        logits[0, 0, bar] = 0.0
        logits[0, 0, other] = 0.0
        tgt = torch.tensor([[bar]])
        # 0.5 * C[other, bar] = 0.5 * 3 = 1.5
        r = train.expected_keystroke_risk(logits, tgt, c)
        self.assertAlmostEqual(float(r), 1.5, places=4)

    def test_pad_positions_ignored(self):
        import torch
        nv = len(data.VOCAB) + 3
        c = train.keystroke_cost_matrix(nv, model.MODIFIER_IDS)
        logits = torch.zeros(1, 2, nv)
        tgt = torch.tensor([[data.PAD_ID, data.PAD_ID]])
        r = train.expected_keystroke_risk(logits, tgt, c)
        self.assertAlmostEqual(float(r), 0.0, places=4)


if __name__ == '__main__':
    unittest.main()
