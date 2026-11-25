"""CPU unit tests for the 2027-2028 Masked Block-Diffusion Measure-Refinement prototype."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

_MVD_DIR = Path(__file__).resolve().parent
if str(_MVD_DIR) not in sys.path:
    sys.path.insert(0, str(_MVD_DIR))

from trial_2028_block_diffusion import (
    BlockDiffusionConfig,
    BlockDiffusionMeasureRefiner,
)


class TestBlockDiffusionMeasureRefiner(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1234)
        self.cfg = BlockDiffusionConfig(
            vocab_size=237,
            d_model=384,
            n_heads=6,
            n_layers=4,
        )
        self.model = BlockDiffusionMeasureRefiner(self.cfg)

    def test_parameter_count_under_budget(self) -> None:
        n_params = self.model.count_parameters()
        self.assertEqual(self.cfg.d_model, 384)
        self.assertEqual(self.cfg.n_heads, 6)
        self.assertEqual(self.cfg.n_layers, 4)
        self.assertLess(n_params, 6_000_000, f"Expected < 6M params, got {n_params:,}")
        self.assertLess(n_params, 8_000_000, f"Expected < 8M params, got {n_params:,}")

    def test_forward_and_diffusion_loss_backward(self) -> None:
        self.model.train()
        bsz, mem_len, seq_len = 2, 32, 48
        mem = torch.randn(bsz, mem_len, self.cfg.d_model, requires_grad=True)
        targets = torch.randint(1, self.cfg.vocab_size, (bsz, seq_len))

        logits = self.model(mem, targets)
        self.assertEqual(logits.shape, (bsz, seq_len, self.cfg.vocab_size))

        loss, metrics = self.model.compute_diffusion_loss(
            mem, targets, return_metrics=True
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.item()), 0.0)
        self.assertGreaterEqual(metrics["t_mean"], self.cfg.t_min)
        self.assertLessEqual(metrics["t_mean"], self.cfg.t_max)
        self.assertGreater(metrics["masked_count"], 0.0)

        self.model.zero_grad(set_to_none=True)
        loss.backward()

        self.assertIsNotNone(mem.grad)
        self.assertGreater(float(mem.grad.abs().sum().item()), 0.0)
        self.assertIsNotNone(self.model.tok_emb.weight.grad)
        self.assertGreater(float(self.model.tok_emb.weight.grad.abs().sum().item()), 0.0)
        self.assertIsNotNone(self.model.pos_emb.weight.grad)
        self.assertGreater(float(self.model.pos_emb.weight.grad.abs().sum().item()), 0.0)
        for idx, layer in enumerate(self.model.layers):
            q_grad = layer.cross_attn.q_proj.weight.grad
            self.assertIsNotNone(q_grad, f"Layer {idx} cross_attn.q_proj grad missing")
            self.assertGreater(float(q_grad.abs().sum().item()), 0.0)

    def test_denoise_measure_full_mask_4_passes(self) -> None:
        self.model.eval()
        bsz, mem_len, seq_len = 2, 24, 32
        mem = torch.randn(bsz, mem_len, self.cfg.d_model)
        init_tokens = torch.full(
            (bsz, seq_len), self.model.mask_token_id, dtype=torch.long
        )
        mask_positions = torch.ones((bsz, seq_len), dtype=torch.bool)

        pass_count = 0

        def _count_hook(_mod, _inp, _out):
            nonlocal pass_count
            pass_count += 1

        handle = self.model.layers[0].register_forward_hook(_count_hook)
        try:
            decoded, trace = self.model.denoise_measure(
                mem, init_tokens, mask_positions, steps=4, return_trace=True
            )
        finally:
            handle.remove()

        self.assertEqual(pass_count, 4, f"Expected exactly 4 forward passes, got {pass_count}")
        self.assertEqual(len(trace), 4)
        self.assertEqual(decoded.shape, (bsz, seq_len))
        self.assertEqual(
            int((decoded == self.model.mask_token_id).sum().item()),
            0,
            "All <mask> tokens must be resolved after 4 passes",
        )
        self.assertGreaterEqual(int(decoded.min().item()), 0)
        self.assertLess(int(decoded.max().item()), self.cfg.vocab_size)
        self.assertEqual(trace[-1]["remaining_masked"], 0.0)
        # Monotonic decrease in remaining masked tokens across passes
        rems = [row["remaining_masked"] for row in trace]
        self.assertEqual(rems, sorted(rems, reverse=True))

    def test_denoise_measure_partial_mask_preserves_unmasked(self) -> None:
        self.model.eval()
        bsz, mem_len, seq_len = 2, 20, 28
        mem = torch.randn(bsz, mem_len, self.cfg.d_model)
        init_tokens = torch.randint(1, self.cfg.vocab_size, (bsz, seq_len))
        mask_positions = torch.zeros((bsz, seq_len), dtype=torch.bool)
        mask_positions[:, 4:16] = True  # simulate low-confidence / duration-conflicting span

        pass_count = 0

        def _count_hook(_mod, _inp, _out):
            nonlocal pass_count
            pass_count += 1

        handle = self.model.layers[0].register_forward_hook(_count_hook)
        try:
            decoded = self.model.denoise_measure(
                mem, init_tokens, mask_positions, steps=4
            )
        finally:
            handle.remove()

        self.assertEqual(pass_count, 4)
        self.assertTrue(
            torch.equal(decoded[~mask_positions], init_tokens[~mask_positions]),
            "Unmasked high-confidence positions must remain untouched",
        )
        self.assertEqual(int((decoded == self.model.mask_token_id).sum().item()), 0)
        self.assertGreaterEqual(int(decoded.min().item()), 0)
        self.assertLess(int(decoded.max().item()), self.cfg.vocab_size)

    def test_bidirectional_right_context_sensitivity(self) -> None:
        self.model.eval()
        mem = torch.randn(1, 16, self.cfg.d_model)
        tokens_a = torch.randint(1, self.cfg.vocab_size, (1, 16))
        tokens_a[0, 5] = self.model.mask_token_id
        tokens_b = tokens_a.clone()
        # Mutate a token strictly to the RIGHT of the masked position
        tokens_b[0, 12] = (tokens_a[0, 12] % (self.cfg.vocab_size - 1)) + 1

        with torch.no_grad():
            logits_a = self.model(mem, tokens_a)
            logits_b = self.model(mem, tokens_b)

        diff = float((logits_a[0, 5] - logits_b[0, 5]).abs().max().item())
        self.assertGreater(
            diff, 1e-5, "Masked position must attend bidirectionally to right context"
        )

    def test_max_measure_len_guard(self) -> None:
        self.model.eval()
        mem = torch.randn(1, 16, self.cfg.d_model)
        ok_tokens = torch.randint(1, self.cfg.vocab_size, (1, 128))
        with torch.no_grad():
            out = self.model(mem, ok_tokens)
        self.assertEqual(out.shape, (1, 128, self.cfg.vocab_size))

        too_long = torch.randint(1, self.cfg.vocab_size, (1, 129))
        with self.assertRaises(ValueError):
            self.model(mem, too_long)


if __name__ == "__main__":
    unittest.main()
