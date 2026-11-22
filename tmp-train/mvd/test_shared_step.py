"""Verify exact forward & gradient parity of MvdNet.loss_shared vs 3 separate calls."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import model  # noqa: E402
import train  # noqa: E402


class TestLossSharedParity(unittest.TestCase):
    def test_forward_and_gradient_parity_with_separate_calls(self):
        torch.manual_seed(0)
        net = model.MvdNet(vocab=len(data.VOCAB) + 3, anisotropic=True,
                           pretrained=False, stride='4x8', extra_chans=8)
        net.eval()  # disable dropout so separate vs shared are deterministic
        imgs = torch.randn(2, 11, 64, 128)
        tgt = torch.randint(0, len(data.VOCAB), (2, 16), dtype=torch.long)
        aux_tgt = torch.rand(2, 8, 64, 128)

        # Reference: 3 separate calls (legacy path).
        net.zero_grad(set_to_none=True)
        ce_ref = net.loss(imgs, tgt, tok_weight=3.0,
                          weight_ids=train.STRUCT_WEIGHT_IDS)
        ap_ref = net.aux_pred(imgs)
        aux_ref = F.binary_cross_entropy_with_logits(
            ap_ref, F.interpolate(aux_tgt, size=ap_ref.shape[-2:], mode='area'))
        xc_ref = net.x_centroids(imgs, tgt)
        coord_ref = model.coord_monotonic_penalty(
            xc_ref, tgt, model.VOICE_IDS, net.pad_id)
        (ce_ref + 0.3 * aux_ref + 1.0 * coord_ref).backward()
        ref_grads = {n: p.grad.clone() for n, p in net.named_parameters()
                     if p.grad is not None}

        # Shared single-pass call.
        net.zero_grad(set_to_none=True)
        ce_sh, ap_sh, xc_sh = net.loss_shared(
            imgs, tgt, tok_weight=3.0, weight_ids=train.STRUCT_WEIGHT_IDS,
            need_aux=True, need_coord=True)
        aux_sh = F.binary_cross_entropy_with_logits(
            ap_sh, F.interpolate(aux_tgt, size=ap_sh.shape[-2:], mode='area'))
        coord_sh = model.coord_monotonic_penalty(
            xc_sh, tgt, model.VOICE_IDS, net.pad_id)
        (ce_sh + 0.3 * aux_sh + 1.0 * coord_sh).backward()

        self.assertLess(float((ce_ref - ce_sh).abs()), 1e-5)
        self.assertLess(float((ap_ref - ap_sh).abs().max()), 1e-5)
        self.assertLess(float((xc_ref - xc_sh).abs().max()), 1e-5)
        self.assertEqual(set(ref_grads.keys()),
                         {n for n, p in net.named_parameters() if p.grad is not None})
        max_gdiff = max(
            float((ref_grads[n] - p.grad).abs().max())
            for n, p in net.named_parameters() if p.grad is not None
        )
        self.assertLess(max_gdiff, 1e-4, f'gradient mismatch: {max_gdiff}')


if __name__ == '__main__':
    unittest.main()
