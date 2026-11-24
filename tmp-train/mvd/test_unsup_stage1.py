"""Unit tests for Stage-1 unsupervised learning primitives (unsup_stage1.py)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_MVD_DIR = str(Path(__file__).resolve().parent)
if _MVD_DIR not in sys.path:
    sys.path.insert(0, _MVD_DIR)

from staff_unet2d import StaffUNet2D  # noqa: E402
from unsup_stage1 import (  # noqa: E402
    EMAModel,
    augment_hw_hard_negatives,
    stage1_equivariance_loss,
)


class _AnalyticEquivariantPredictor(nn.Module):
    """Analytic 4-channel predictor obeying exact scale equivariance on ch3.

    Given an input image with a bright horizontal band in channel 0, outputs:
      - ch0..2 logits derived deterministically from the warped input spatial
        intensity (downsampled by 2x, equivariant to any similarity warp).
      - ch3 = ``base_log_sp + log(scale)`` where ``scale`` is injected for the
        exact equivariance test.
    """

    def __init__(self, base_log_sp: float = 2.0, current_scale: float = 1.0) -> None:
        super().__init__()
        self.base_log_sp = float(base_log_sp)
        self.current_scale = float(current_scale)
        # Dummy parameter so autograd works if needed.
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsample 2x via avg_pool2d to match StaffUNet2D out_stride=2.
        pooled = nn.functional.avg_pool2d(x[:, :1], kernel_size=2, stride=2)
        # Sharp logits: +6 on band (x > 0.5), -6 on background (x <= 0.5).
        logits = (pooled - 0.5) * 16.0
        ch0 = logits
        ch1 = logits
        ch2 = logits
        ch3 = torch.full_like(pooled, self.base_log_sp + math.log(self.current_scale)) + self.bias * 0.0
        return torch.cat([ch0, ch1, ch2, ch3], dim=1)


class TestEMAModel(unittest.TestCase):
    def test_ema_update_math_and_frozen_state(self) -> None:
        torch.manual_seed(0)
        student = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4))
        ema = EMAModel(student, decay=0.9)

        # Teacher parameters must be frozen and in eval mode.
        self.assertFalse(ema.module.training)
        for p in ema.module.parameters():
            self.assertFalse(p.requires_grad)

        orig_w = ema.module[0].weight.detach().clone()
        with torch.no_grad():
            student[0].weight.add_(2.0)

        ema.update(student, decay=0.9)
        expected = 0.9 * orig_w + 0.1 * student[0].weight.detach()
        self.assertTrue(torch.allclose(ema.module[0].weight, expected, atol=1e-6))

        # Decay=0 copies student exactly; decay=1 keeps EMA unchanged.
        prev = ema.module[0].weight.detach().clone()
        ema.update(student, decay=1.0)
        self.assertTrue(torch.equal(ema.module[0].weight, prev))
        ema.update(student, decay=0.0)
        self.assertTrue(torch.equal(ema.module[0].weight, student[0].weight.detach()))


class TestStage1EquivarianceLoss(unittest.TestCase):
    def test_exact_log_hm_scale_equivariance_zero_loss(self) -> None:
        """Exact log_hm(Scale_s(I)) == log_hm(I) + log(s) gives ~0 loss."""
        b, c, h, w = 2, 3, 64, 64
        # Smooth constant image so bilinear interpolation is exact everywhere.
        img = torch.ones((b, c, h, w), dtype=torch.float32) * 0.9
        s_val = 1.20
        teacher = _AnalyticEquivariantPredictor(base_log_sp=2.1, current_scale=1.0)
        student = _AnalyticEquivariantPredictor(base_log_sp=2.1, current_scale=s_val)

        loss, metrics = stage1_equivariance_loss(
            student=student,
            teacher=teacher,
            img_batch=img,
            sp_mode="log_hm",
            conf_thresh=0.85,
            photo_jitter=False,
            warp_params={"scales": s_val, "angles_deg": 0.0, "tx": 0.0, "ty": 0.0},
            return_metrics=True,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss.item()), 1e-6)
        self.assertLess(metrics["unsup_s1_sp"], 1e-6)
        self.assertLess(metrics["unsup_s1_ch02"], 1e-6)
        self.assertGreater(metrics["unsup_s1_sp_frac"], 0.9)

        # Perturb student's ch3 scale prediction -> positive loss & gradient.
        student_bad = _AnalyticEquivariantPredictor(base_log_sp=2.1, current_scale=1.0)
        student_bad.bias = nn.Parameter(torch.tensor([0.0]))
        # Replace forward so bias directly affects ch3.
        orig_fwd = student_bad.forward

        def _fwd_with_bias(x: torch.Tensor) -> torch.Tensor:
            out = orig_fwd(x)
            return torch.cat([out[:, :3], out[:, 3:4] + student_bad.bias], dim=1)

        student_bad.forward = _fwd_with_bias  # type: ignore[method-assign]
        loss_bad = stage1_equivariance_loss(
            student=student_bad,
            teacher=teacher,
            img_batch=img,
            sp_mode="log_hm",
            conf_thresh=0.85,
            photo_jitter=False,
            warp_params={"scales": s_val, "angles_deg": 0.0, "tx": 0.0, "ty": 0.0},
        )
        self.assertGreater(float(loss_bad.item()), 1e-3)
        loss_bad.backward()
        self.assertIsNotNone(student_bad.bias.grad)
        self.assertGreater(float(student_bad.bias.grad.abs().item()), 1e-3)

    def test_perturbed_staff_unet_positive_grad_and_frozen_teacher(self) -> None:
        torch.manual_seed(7)
        student = StaffUNet2D(in_channels=3, base_channels=16, out_stride=2)
        ema = EMAModel(student, decay=0.99)
        with torch.no_grad():
            for p in student.parameters():
                p.add_(0.05 * torch.randn_like(p))

        img = torch.rand((2, 3, 64, 64), dtype=torch.float32)
        gen = torch.Generator().manual_seed(123)
        loss = stage1_equivariance_loss(
            student=student,
            teacher=ema,
            img_batch=img,
            sp_mode="log_hm",
            conf_thresh=0.50,
            photo_jitter=True,
            hflip_weight=0.5,
            generator=gen,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.item()), 0.0)
        loss.backward()

        stu_grads = [p.grad for p in student.parameters() if p.grad is not None]
        self.assertGreater(len(stu_grads), 0)
        self.assertGreater(sum(float(g.abs().sum().item()) for g in stu_grads), 0.0)
        for p in ema.module.parameters():
            self.assertIsNone(p.grad)

    def test_oob_masking_excludes_out_of_bounds_pixels(self) -> None:
        """Zoom-out + translation masks OOB pixels cleanly with valid_frac < 1."""
        img = torch.ones((1, 3, 64, 64), dtype=torch.float32) * 0.9
        s_val = 0.80
        teacher = _AnalyticEquivariantPredictor(base_log_sp=1.8, current_scale=1.0)
        student = _AnalyticEquivariantPredictor(base_log_sp=1.8, current_scale=s_val)

        loss, metrics = stage1_equivariance_loss(
            student=student,
            teacher=teacher,
            img_batch=img,
            sp_mode="log_hm",
            conf_thresh=0.85,
            photo_jitter=False,
            warp_params={"scales": s_val, "angles_deg": 3.0, "tx": 0.15, "ty": -0.10},
            return_metrics=True,
        )
        self.assertTrue(torch.isfinite(loss))
        # Zoom-out 0.80 + translation leaves a substantial OOB border.
        self.assertLess(metrics["unsup_s1_valid_frac"], 0.70)
        self.assertGreater(metrics["unsup_s1_valid_frac"], 0.25)
        # Inside valid_mask, analytic predictor still achieves ~0 loss.
        self.assertLess(float(loss.item()), 1e-5)


class TestAugmentHwHardNegatives(unittest.TestCase):
    def _make_sample(self, h: int = 256, w: int = 192) -> tuple[np.ndarray, np.ndarray]:
        img = np.full((h, w, 3), 0.95, dtype=np.float32)
        hm = np.zeros((4, h // 2, w // 2), dtype=np.float32)
        # Single real system in the middle rows [52..76] of the heatmap (= [104..152] in image).
        hm[0, 56:72, 10:86] = 1.0
        hm[1, 62:66, 10:86] = 1.0
        hm[2, 52:76, 10:86] = 1.0
        hm[3, 56:72, 10:86] = 1.5
        return img, hm

    def test_blank_staves_modifies_empty_regions_preserves_gt(self) -> None:
        img, hm = self._make_sample()
        hm_before = hm.copy()
        img_before = img.copy()

        out_img, out_hm = augment_hw_hard_negatives(
            img, hm, rng=np.random.default_rng(42), p_blank_staves=1.0, p_cursive_text=0.0
        )
        # GT heatmaps must be bit-identical.
        self.assertTrue(np.array_equal(out_hm, hm_before))
        # Input image must not be mutated in place.
        self.assertTrue(np.array_equal(img, img_before))
        # Blank margins/gaps must have been modified.
        diff = np.abs(out_img - img_before)
        self.assertGreater(float(diff.max()), 0.05)
        # Pixels overlapping GT positive region [104..152, 20..172] must be untouched.
        self.assertEqual(float(diff[104:152, 20:172].max()), 0.0)

    def test_cursive_lyrics_modifies_empty_regions_preserves_gt(self) -> None:
        img, hm = self._make_sample()
        hm_before = hm.copy()
        img_before = img.copy()

        out_img, out_hm = augment_hw_hard_negatives(
            img, hm, rng=np.random.default_rng(99), p_blank_staves=0.0, p_cursive_text=1.0
        )
        self.assertTrue(np.array_equal(out_hm, hm_before))
        diff = np.abs(out_img - img_before)
        self.assertGreater(float(diff.max()), 0.10)
        self.assertEqual(float(diff[104:152, 20:172].max()), 0.0)

    def test_torch_tensor_chw_roundtrip(self) -> None:
        img_np, hm_np = self._make_sample()
        img_t = torch.from_numpy(img_np.transpose(2, 0, 1))
        hm_t = torch.from_numpy(hm_np)
        hm_orig = hm_t.clone()

        out_img, out_hm = augment_hw_hard_negatives(
            img_t, hm_t, rng=np.random.default_rng(7), p_blank_staves=1.0, p_cursive_text=1.0
        )
        self.assertIsInstance(out_img, torch.Tensor)
        self.assertEqual(tuple(out_img.shape), tuple(img_t.shape))
        self.assertTrue(torch.equal(out_hm, hm_orig))
        self.assertGreater(float((out_img - img_t).abs().max().item()), 0.05)


if __name__ == "__main__":
    unittest.main()
