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

    def test_exp2b_s18_comer_gdcm_cpu_forward_backward_and_kv_parity(self):
        torch.manual_seed(7)
        v = len(data.VOCAB) + 3
        net = model.MvdNet(
            vocab=v, d_model=64, layers=2, heads=4, dim_ff=128,
            extra_chans=1, foveal_tap=True,
            use_comer=True, use_gdcm=True, use_s18=True)
        net.train()
        imgs = torch.randn(2, 4, 64, 128)
        seq0 = data.encode(['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4', 'dot', 'bar']) + [data.EOS_ID]
        seq1 = data.encode(['clef-F-4', 'half', 'G', '3', 'rest', 'bar']) + [data.EOS_ID]
        n = max(len(seq0), len(seq1))
        tgt = torch.full((2, n), data.PAD_ID, dtype=torch.long)
        tgt[0, :len(seq0)] = torch.tensor(seq0, dtype=torch.long)
        tgt[1, :len(seq1)] = torch.tensor(seq1, dtype=torch.long)

        # Single-pass shared forward: CE + S18 + GDCM + CoMER (1 encoder + 1 decoder).
        net.zero_grad(set_to_none=True)
        ce, ap, xc, h, gdcm_loss = net.loss_shared(
            imgs, tgt, need_aux=False, need_coord=False,
            return_hidden=True, need_gdcm=True)
        self.assertIsNone(ap)
        self.assertIsNone(xc)
        self.assertIsNotNone(gdcm_loss)
        s18_loss, s18_metrics = net.s18_head.loss_from_tokens(
            h, tgt, pad_id=net.pad_id, eos_id=net.eos_id)
        self.assertIn('s18_total', s18_metrics)
        total = ce + 0.25 * s18_loss + 0.1 * gdcm_loss
        self.assertTrue(bool(torch.isfinite(total)))
        total.backward()

        # Gradients must reach trunk, decoder, CoMER, GDCM, and S18.
        for prefix in ('encoder.', 'decoder.', 'comer_bias.', 'gdcm_head.', 's18_head.'):
            grads = [p.grad for n_p, p in net.named_parameters()
                     if n_p.startswith(prefix) and p.grad is not None]
            self.assertTrue(len(grads) > 0, f'no gradients for {prefix}')
            self.assertTrue(
                all(bool(torch.isfinite(g).all()) for g in grads),
                f'non-finite grad in {prefix}')

        # Verify CoMER batched greedy == incremental greedy_kv even with non-zero conv_out!
        with torch.no_grad():
            torch.nn.init.normal_(net.comer_bias.conv_out.weight, std=0.2)
            torch.nn.init.normal_(net.comer_bias.conv_out.bias, std=0.1)
        net.eval()
        with torch.no_grad():
            g_batched = net.greedy(imgs[:1], max_len=8, constrain=False)
            g_kv = net.greedy_kv(imgs[:1], max_len=8, constrain=False)
            self.assertTrue(torch.equal(g_batched, g_kv),
                            f'CoMER greedy vs greedy_kv mismatch: {g_batched} vs {g_kv}')
            beams = net.beam_search(imgs[:1], max_len=8, beam=2, constrain=False)
            self.assertEqual(beams.shape[0], 1)


if __name__ == '__main__':
    unittest.main()
