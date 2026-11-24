"""Unit tests for stage1_academic_losses.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from stage1_academic_losses import (  # noqa: E402
    EFDMix,
    boundary_dou_loss,
    fourier_hw_amplitude_mix,
    silog_loss,
    skeleton_recall_loss,
)


class TestStage1AcademicLosses(unittest.TestCase):
    def test_srl_prefers_continuous_over_broken_staff(self) -> None:
        tgt = torch.zeros(1, 32, 64)
        tgt[0, 16, 4:60] = 1.0
        good = torch.full((1, 32, 64), -4.0, requires_grad=True)
        good_data = good.detach().clone()
        good_data[0, 16, 4:60] = 4.0
        good = good_data.requires_grad_(True)

        broken_data = good_data.detach().clone()
        broken_data[0, 16, 20:40] = -4.0
        broken = broken_data.requires_grad_(True)

        l_good = skeleton_recall_loss(good, tgt)
        l_broken = skeleton_recall_loss(broken, tgt)
        self.assertLess(float(l_good.detach()), float(l_broken.detach()))
        l_broken.backward()
        self.assertTrue(torch.isfinite(broken.grad).all())

    def test_silog_penalizes_intra_page_drift_and_is_finite(self) -> None:
        mask = torch.zeros(1, 16, 16)
        mask[0, 4:12, 2:14] = 1.0
        tgt = torch.full((1, 16, 16), 2.5)
        pred_const = torch.full((1, 16, 16), 2.7, requires_grad=True)
        pred_drift = tgt.clone()
        pred_drift[0, 4:8, :] += 0.2
        pred_drift[0, 8:12, :] -= 0.2
        pred_drift.requires_grad_(True)

        l_const = silog_loss(pred_const, tgt, mask, lam=0.5)
        l_drift = silog_loss(pred_drift, tgt, mask, lam=0.5)
        self.assertLess(float(l_const), float(l_drift))
        l_drift.backward()
        self.assertTrue(torch.isfinite(pred_drift.grad).all())

    def test_boundary_dou_loss_gradient_and_ordering(self) -> None:
        tgt = torch.zeros(1, 32, 32)
        tgt[0, 8:24, 6:26] = 1.0
        good = torch.where(tgt > 0.5, torch.tensor(4.0), torch.tensor(-4.0)).requires_grad_(True)
        bad = torch.full((1, 32, 32), -4.0, requires_grad=True)
        lg = boundary_dou_loss(good, tgt)
        lb = boundary_dou_loss(bad, tgt)
        self.assertLess(float(lg), float(lb))
        lb.backward()
        self.assertTrue(torch.isfinite(bad.grad).all())

    def test_fourier_hw_amplitude_mix_preserves_shape_and_range(self) -> None:
        g = torch.Generator().manual_seed(123)
        x_lbl = torch.rand(2, 3, 64, 96, generator=g)
        x_hw = torch.rand(2, 3, 48, 80, generator=g) * 0.6 + 0.2
        out = fourier_hw_amplitude_mix(x_lbl, x_hw, p=1.0, beta=0.15, alpha=0.5, generator=g)
        self.assertEqual(out.shape, x_lbl.shape)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)
        self.assertFalse(torch.allclose(out, x_lbl))

    def test_efdmix_identity_in_eval_and_backprop_in_train(self) -> None:
        mod = EFDMix(p=1.0, alpha=0.5)
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        mod.eval()
        self.assertTrue(torch.equal(mod(x), x))
        mod.train()
        y = mod(x)
        self.assertEqual(y.shape, x.shape)
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
