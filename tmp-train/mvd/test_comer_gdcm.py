"""CPU-only unit tests for Exp 2B/2C CoMER Coverage + GDCM Counting modules."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comer_gdcm as cg  # noqa: E402
import data  # noqa: E402


class TestCoverageAttentionBias(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_zero_init_warm_start_identity(self):
        mod = cg.CoverageAttentionBias(h_mem_max=64, n_heads=8, hidden_dim=32, kernel_size=5)
        b, h, hm, wm, t = 2, 8, 8, 16, 12
        cov = torch.rand(b, h, hm, wm)
        bias_2d = mod(cov)
        self.assertEqual(bias_2d.shape, (b, h, 1, hm * wm))
        self.assertTrue(torch.equal(bias_2d, torch.zeros_like(bias_2d)))

        attn_w = torch.softmax(torch.randn(b, h, t, hm * wm), dim=-1)
        bias_tf = mod.compute_shifted_coverage_bias(attn_w, hm, wm)
        self.assertEqual(bias_tf.shape, (b, h, t, hm * wm))
        self.assertTrue(torch.equal(bias_tf, torch.zeros_like(bias_tf)))

        bias_step = mod.step_coverage_bias(cov.reshape(b, h, hm * wm), hm, wm)
        self.assertEqual(bias_step.shape, (b, h, 1, hm * wm))
        self.assertTrue(torch.equal(bias_step, torch.zeros_like(bias_step)))

    def test_teacher_forced_vs_incremental_parity(self):
        mod = cg.CoverageAttentionBias(h_mem_max=64, n_heads=8, hidden_dim=32, kernel_size=5)
        # Non-zero output weights so bias is genuinely non-trivial.
        with torch.no_grad():
            mod.conv_out.weight.normal_(0.0, 0.1)
            mod.conv_out.bias.normal_(0.0, 0.05)

        b, h, hm, wm, t = 3, 8, 6, 10, 15
        s = hm * wm
        mem_mask = torch.zeros(b, s, dtype=torch.bool)
        mem_mask[0, -10:] = True
        mem_mask[2, -20:] = True

        logits = torch.randn(b, h, t, s)
        logits = logits.masked_fill(mem_mask[:, None, None, :], float("-inf"))
        attn_w = torch.softmax(logits, dim=-1)

        bias_tf = mod.compute_shifted_coverage_bias(
            attn_w, hm, wm, mem_mask=mem_mask, detach_cov=True
        )

        # Incremental rollout step-by-step.
        cum_cov = mod.init_coverage(b, hm, wm, device=attn_w.device, dtype=attn_w.dtype)
        step_biases = []
        for step in range(t):
            b_t = mod.step_coverage_bias(cum_cov, hm, wm, mem_mask=mem_mask)
            step_biases.append(b_t)
            cum_cov = mod.update_coverage(cum_cov, attn_w[:, :, step : step + 1, :])
        bias_inc = torch.cat(step_biases, dim=2)

        max_diff = (bias_tf - bias_inc).abs().max().item()
        self.assertLess(max_diff, 1e-5, f"TF vs incremental max_diff={max_diff}")
        # Confirm step 0 truly saw zero coverage (identical across batches with same mask).
        self.assertGreater(bias_tf.abs().max().item(), 1e-3)

    def test_i1_multivoice_backward_jump_soft_bias(self):
        """Coverage never hard-masks already-visited left columns (I1 backward jumps)."""
        mod = cg.CoverageAttentionBias(n_heads=8, hidden_dim=32, kernel_size=5)
        with torch.no_grad():
            mod.conv_out.weight.normal_(0.0, 0.1)
            mod.conv_out.bias.zero_()

        b, h, hm, wm = 1, 8, 4, 16
        # Simulate heavy prior coverage across the left half (voice-1 already read).
        cum_cov = torch.zeros(b, h, hm, wm)
        cum_cov[:, :, :, :8] = 5.0
        bias = mod.step_coverage_bias(cum_cov, hm, wm)
        self.assertTrue(torch.isfinite(bias).all())

        # Voice-2 query strongly targeting column 0 (backward jump to left margin).
        scores = torch.zeros(b, h, 1, hm * wm)
        scores[:, :, :, 0] = 6.0
        probs = torch.softmax(scores + bias, dim=-1)
        self.assertTrue(torch.isfinite(probs).all())
        self.assertGreater(probs[:, :, 0, 0].min().item(), 0.05)

    def test_gradient_flow(self):
        mod = cg.CoverageAttentionBias(n_heads=8, hidden_dim=32, kernel_size=5)
        b, h, hm, wm, t = 2, 8, 4, 8, 6
        attn_w = torch.softmax(torch.randn(b, h, t, hm * wm), dim=-1).requires_grad_(True)

        # Step 0 (zero-init): conv_out receives non-zero gradient immediately.
        bias0 = mod.compute_shifted_coverage_bias(attn_w, hm, wm, detach_cov=False)
        loss0 = (bias0 * torch.randn_like(bias0)).sum()
        loss0.backward()
        self.assertIsNotNone(mod.conv_out.weight.grad)
        self.assertGreater(mod.conv_out.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(mod.conv_out.bias.grad.abs().sum().item(), 0.0)

        # Step 1 (after 1 SGD step): gradients flow through conv_spatial and attn_w.
        with torch.no_grad():
            mod.conv_out.weight -= 0.1 * mod.conv_out.weight.grad
            mod.conv_out.bias -= 0.1 * mod.conv_out.bias.grad
        mod.zero_grad()
        attn_w2 = torch.softmax(torch.randn(b, h, t, hm * wm), dim=-1).requires_grad_(True)
        bias1 = mod.compute_shifted_coverage_bias(attn_w2, hm, wm, detach_cov=False)
        loss1 = (bias1 ** 2).mean()
        loss1.backward()
        self.assertGreater(mod.conv_spatial.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(attn_w2.grad.abs().sum().item(), 0.0)

    def test_bf16_and_fp32_stability(self):
        b, h, hm, wm, t = 2, 8, 8, 16, 10
        attn_w = torch.softmax(torch.randn(b, h, t, hm * wm), dim=-1)

        mod_fp32 = cg.CoverageAttentionBias(n_heads=8, hidden_dim=32, kernel_size=5)
        with torch.no_grad():
            mod_fp32.conv_out.weight.normal_(0.0, 0.05)
        out_fp32 = mod_fp32.compute_shifted_coverage_bias(attn_w, hm, wm)
        self.assertTrue(torch.isfinite(out_fp32).all())

        # Direct bf16 module + bf16 input
        mod_bf16 = cg.CoverageAttentionBias(n_heads=8, hidden_dim=32, kernel_size=5).to(
            torch.bfloat16
        )
        with torch.no_grad():
            mod_bf16.load_state_dict(
                {k: v.to(torch.bfloat16) for k, v in mod_fp32.state_dict().items()}
            )
        out_bf16 = mod_bf16.compute_shifted_coverage_bias(attn_w.to(torch.bfloat16), hm, wm)
        self.assertEqual(out_bf16.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(out_bf16).all())
        self.assertLess((out_bf16.float() - out_fp32).abs().max().item(), 5e-2)

        # CPU autocast bf16
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            out_amp = mod_fp32.compute_shifted_coverage_bias(attn_w, hm, wm)
        self.assertTrue(torch.isfinite(out_amp).all())


class TestExtractStripCounts(unittest.TestCase):
    def test_real_bekern_streams_accuracy(self):
        # Stream 1: single-voice, 2 bars, 4 notes (incl. A, chord note, grace),
        # 2 rests (1 timed rest + 1 bar-rest), 3 dots (incl. double-dot), 2 accidentals.
        s1_toks = [
            "clef-G-2", "key+1", "time-4-4",
            "quarter", "A", "4", "a1", "dot",
            "chord", "quarter", "C", "5", "dot", "dot",
            "eighth", "rest",
            "grace", "16th", "D", "5", "a-1",
            "bar",
            "bar-rest",
            "bar-light-heavy",
        ]
        # Stream 2: multi-voice (voice-1, voice-2, voice-3 -> max=3), 1 bar,
        # 3 notes, 1 rest, 0 dots, 1 double-sharp accidental.
        s2_toks = [
            "clef-F-4", "key-2", "time-3-4",
            "bar-repeat-left",  # opening barline, does NOT add a bar
            "voice-1", "half", "E", "3", "a2",
            "voice-2", "quarter", "G", "2",
            "voice-3", "quarter", "B", "2", "half", "rest",
            "bar-repeat-right",
        ]
        # Stream 3: empty (all PAD/EOS) -> all zeros.
        id1 = data.encode(s1_toks) + [data.EOS_ID]
        id2 = data.encode(s2_toks) + [data.EOS_ID]
        max_len = max(len(id1), len(id2))
        batch = torch.full((3, max_len), data.PAD_ID, dtype=torch.long)
        batch[0, : len(id1)] = torch.tensor(id1)
        batch[1, : len(id2)] = torch.tensor(id2)
        batch[2, 0] = data.EOS_ID

        counts = cg.extract_strip_counts(batch, pad_id=data.PAD_ID)
        self.assertEqual(counts.shape, (3, 6))
        # [n_bars, n_voices_max, n_notes, n_rests, n_dots, n_accidentals]
        self.assertEqual(counts[0].tolist(), [2.0, 1.0, 3.0, 2.0, 3.0, 2.0])
        self.assertEqual(counts[1].tolist(), [1.0, 3.0, 3.0, 1.0, 0.0, 1.0])
        self.assertEqual(counts[2].tolist(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # 1-D input also works.
        c1d = cg.extract_strip_counts(id1, pad_id=data.PAD_ID)
        self.assertEqual(c1d.shape, (6,))
        self.assertEqual(c1d.tolist(), [2.0, 1.0, 3.0, 2.0, 3.0, 2.0])

    def test_step_a_not_dropped_by_default_pad_id(self):
        # Token ID 0 is "A" in data.VOCAB — default pad_id must NOT drop it.
        ids = data.encode(["quarter", "A", "4", "quarter", "A", "5", "bar"])
        self.assertIn(0, ids)
        c = cg.extract_strip_counts(ids)
        self.assertEqual(c.tolist(), [1.0, 1.0, 2.0, 0.0, 0.0, 0.0])


class TestGlobalDynamicCountingHead(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_zero_init_cond_warm_start_identity(self):
        head = cg.GlobalDynamicCountingHead(d_model=256, n_classes=6)
        mem = torch.randn(2, 64, 256)
        pred = head(mem)
        self.assertEqual(pred.shape, (2, 6))
        self.assertTrue((pred >= 0).all())

        cond = head.count_to_mem_cond(pred)
        self.assertEqual(cond.shape, (2, 1, 256))
        self.assertTrue(torch.equal(cond, torch.zeros_like(cond)))

    def test_mem_mask_invariance_and_all_masked_safety(self):
        head = cg.GlobalDynamicCountingHead(d_model=256, n_classes=6)
        head.eval()
        mem_short = torch.randn(1, 32, 256)
        mem_padded = torch.cat([mem_short, torch.randn(1, 16, 256) * 100.0], dim=1)
        mask = torch.zeros(1, 48, dtype=torch.bool)
        mask[:, 32:] = True

        c_short = head(mem_short)
        c_padded = head(mem_padded, mem_mask=mask)
        self.assertLess((c_short - c_padded).abs().max().item(), 1e-5)

        # Edge case: all-masked row must remain finite (no NaN from softmax(-inf)).
        all_mask = torch.ones(1, 48, dtype=torch.bool)
        c_all = head(mem_padded, mem_mask=all_mask)
        self.assertTrue(torch.isfinite(c_all).all())

    def test_gradient_flow_and_log1p_huber_loss(self):
        head = cg.GlobalDynamicCountingHead(d_model=256, n_classes=6)
        mem = torch.randn(2, 48, 256, requires_grad=True)
        toks = data.encode(["clef-G-2", "quarter", "C", "4", "dot", "a1", "bar"])
        tgt_ids = torch.tensor([toks, toks], dtype=torch.long)

        pred, cond, loss, metrics = head.forward_with_loss(mem, tgt_ids)
        self.assertEqual(pred.shape, (2, 6))
        self.assertEqual(cond.shape, (2, 1, 256))
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("gdcm_loss", metrics)
        self.assertIn("gdcm_mae_n_notes", metrics)

        # Combine count loss + downstream decoder loss on cond.
        total = loss + (cond * torch.randn_like(cond)).sum()
        total.backward()
        self.assertIsNotNone(mem.grad)
        self.assertGreater(mem.grad.abs().sum().item(), 0.0)
        for name, p in head.named_parameters():
            self.assertIsNotNone(p.grad, f"Missing grad for {name}")
        # Output cond_proj layer receives non-zero grad at step 0.
        self.assertGreater(head.cond_proj[2].weight.grad.abs().sum().item(), 0.0)

    def test_bf16_and_fp32_stability(self):
        head_fp32 = cg.GlobalDynamicCountingHead(d_model=256, n_classes=6)
        with torch.no_grad():
            head_fp32.cond_proj[2].weight.normal_(0.0, 0.05)

        mem = torch.randn(2, 128, 256)
        mask = torch.zeros(2, 128, dtype=torch.bool)
        mask[1, 96:] = True
        tgt = torch.tensor(
            [[4.0, 2.0, 85.0, 12.0, 9.0, 6.0], [1.0, 1.0, 12.0, 2.0, 1.0, 0.0]]
        )

        pred_fp32 = head_fp32(mem, mem_mask=mask)
        cond_fp32 = head_fp32.count_to_mem_cond(pred_fp32)
        loss_fp32 = head_fp32.compute_loss(pred_fp32, tgt)
        self.assertTrue(torch.isfinite(pred_fp32).all())
        self.assertTrue(torch.isfinite(cond_fp32).all())
        self.assertTrue(torch.isfinite(loss_fp32))

        head_bf16 = cg.GlobalDynamicCountingHead(d_model=256, n_classes=6).to(torch.bfloat16)
        head_bf16.load_state_dict(
            {k: v.to(torch.bfloat16) for k, v in head_fp32.state_dict().items()}
        )
        pred_bf16 = head_bf16(mem.to(torch.bfloat16), mem_mask=mask)
        cond_bf16 = head_bf16.count_to_mem_cond(pred_bf16)
        loss_bf16 = head_bf16.compute_loss(pred_bf16, tgt)
        self.assertEqual(pred_bf16.dtype, torch.bfloat16)
        self.assertEqual(cond_bf16.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(pred_bf16).all())
        self.assertTrue(torch.isfinite(cond_bf16).all())
        self.assertTrue(torch.isfinite(loss_bf16))

        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            pred_amp = head_fp32(mem, mem_mask=mask)
            cond_amp = head_fp32.count_to_mem_cond(pred_amp)
            loss_amp = head_fp32.compute_loss(pred_amp, tgt)
        self.assertTrue(torch.isfinite(pred_amp).all())
        self.assertTrue(torch.isfinite(cond_amp).all())
        self.assertTrue(torch.isfinite(loss_amp))


if __name__ == "__main__":
    unittest.main()
