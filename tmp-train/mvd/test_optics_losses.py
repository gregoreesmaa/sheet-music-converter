"""Unit tests for optics_losses.py (Batch 4-B5 Logit-JSD)."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import optics_losses as ol


class LogitJsdConsistencyTest(unittest.TestCase):
    def test_zero_on_identical_logits(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 12, 50)
        jsd = ol.logit_jsd_consistency(logits, logits.clone())
        self.assertEqual(jsd.shape, ())
        self.assertAlmostEqual(float(jsd), 0.0, places=6)

    def test_positive_finite_and_symmetric_on_differing_logits(self):
        torch.manual_seed(1)
        a = torch.randn(2, 10, 64)
        b = torch.randn(2, 10, 64)
        j_ab = ol.logit_jsd_consistency(a, b, detach_clean=False)
        j_ba = ol.logit_jsd_consistency(b, a, detach_clean=False)
        self.assertGreater(float(j_ab), 0.0)
        self.assertLessEqual(float(j_ab), math.log(2.0) + 1e-5)
        self.assertTrue(bool(torch.isfinite(j_ab)))
        self.assertAlmostEqual(float(j_ab), float(j_ba), places=6)

    def test_exact_pad_mask_invariance_and_all_pad_safety(self):
        torch.manual_seed(2)
        clean = torch.randn(2, 8, 32)
        deg = torch.randn(2, 8, 32, requires_grad=True)
        pad_mask = torch.tensor([
            [False, False, False, False, True, True, True, True],
            [False, False, True, True, True, True, True, True],
        ])
        loss1 = ol.logit_jsd_consistency(clean, deg, pad_mask=pad_mask)
        loss1.backward()
        grad1 = deg.grad.clone()

        # Mutate masked positions to extreme/NaN values -> zero effect on loss or grads.
        clean_mut = clean.clone()
        deg_mut = deg.detach().clone()
        clean_mut[pad_mask] = float('nan')
        deg_mut[pad_mask] = 1e6
        deg_mut.requires_grad_(True)
        loss2 = ol.logit_jsd_consistency(clean_mut, deg_mut, pad_mask=pad_mask)
        loss2.backward()
        self.assertTrue(torch.equal(loss1.detach(), loss2.detach()))
        self.assertTrue(torch.equal(grad1[~pad_mask], deg_mut.grad[~pad_mask]))
        self.assertTrue(bool((deg_mut.grad[pad_mask] == 0.0).all()))

        # All-PAD batch returns exact 0.0 and backprops safely.
        deg_all = torch.randn(2, 4, 16, requires_grad=True)
        all_pad = torch.ones(2, 4, dtype=torch.bool)
        loss_zero = ol.logit_jsd_consistency(torch.randn(2, 4, 16), deg_all, pad_mask=all_pad)
        self.assertEqual(float(loss_zero), 0.0)
        loss_zero.backward()
        self.assertIsNotNone(deg_all.grad)
        self.assertTrue(bool((deg_all.grad == 0.0).all()))

    def test_bf16_stability_with_extreme_logits(self):
        torch.manual_seed(3)
        clean = (torch.randn(2, 16, 128) * 30.0).to(torch.bfloat16)
        deg = (torch.randn(2, 16, 128) * 30.0).to(torch.bfloat16).requires_grad_(True)
        loss = ol.logit_jsd_consistency(clean, deg, temperature=1.5)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(float(loss), 0.0)
        self.assertLessEqual(float(loss), math.log(2.0) + 1e-5)
        loss.backward()
        self.assertIsNotNone(deg.grad)
        self.assertTrue(bool(torch.isfinite(deg.grad).all()))

        # Identical extreme bf16 logits give 0.0.
        loss_same = ol.logit_jsd_consistency(clean, clean.clone())
        self.assertAlmostEqual(float(loss_same), 0.0, places=5)

    def test_detach_clean_gradient_routing(self):
        torch.manual_seed(4)
        clean = torch.randn(2, 6, 40, requires_grad=True)
        deg = torch.randn(2, 6, 40, requires_grad=True)
        loss = ol.logit_jsd_consistency(clean, deg, detach_clean=True)
        loss.backward()
        self.assertIsNone(clean.grad)
        self.assertIsNotNone(deg.grad)
        self.assertGreater(float(deg.grad.abs().sum()), 0.0)

        # When detach_clean=False, both receive gradients.
        clean2 = torch.randn(2, 6, 40, requires_grad=True)
        deg2 = torch.randn(2, 6, 40, requires_grad=True)
        loss2 = ol.logit_jsd_consistency(clean2, deg2, detach_clean=False)
        loss2.backward()
        self.assertIsNotNone(clean2.grad)
        self.assertIsNotNone(deg2.grad)
        self.assertGreater(float(clean2.grad.abs().sum()), 0.0)
        self.assertGreater(float(deg2.grad.abs().sum()), 0.0)

    def test_invalid_args_raise(self):
        x = torch.zeros(1, 2, 4)
        with self.assertRaises(ValueError):
            ol.logit_jsd_consistency(x, x, temperature=0.0)
        with self.assertRaises(ValueError):
            ol.logit_jsd_consistency(x, torch.zeros(1, 3, 4))
        with self.assertRaises(ValueError):
            ol.logit_jsd_consistency(x, x, pad_mask=torch.zeros(1, 3, dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
