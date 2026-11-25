#!/usr/bin/env python3
"""CPU-only unit tests for SpinQuant offline Hadamard rotation folding."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import unittest

# Enforce strictly CPU-only execution.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import spinquant_fold  # noqa: E402
from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net  # noqa: E402


def _make_tiny_v2(d_model: int, n_heads: int = 8, n_kv_heads: int = 2) -> UniOMRv2Net:
    cfg = UniOMRv2Config(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_axial_layers=1,
        n_global_layers=1,
        n_dec_layers=2,
        ffn_mult=2.0,
        axial_ffn_mult=2.0,
        dropout=0.0,
        backbone_channels=(16, 32, 48, 64),
        backbone_blocks=(1, 1, 1, 1),
        backbone_expand=2.0,
        router_hidden=16,
        max_mem_tokens=256,
        use_sys_y_aux=False,
        use_sys_count_aux=False,
        use_staff_seg_aux=False,
        mtp_k=1,
    )
    return UniOMRv2Net(cfg)


class TestSpinQuantHadamardFold(unittest.TestCase):
    def test_hadamard_orthogonality(self) -> None:
        """Verify H @ H.T == I to < 1e-6 for dims 64, 128, 384 (3*128), 512."""
        for dim in (64, 128, 384, 512):
            H = spinquant_fold.hadamard_matrix(dim)
            self.assertEqual(H.shape, (dim, dim))
            eye = torch.eye(dim, dtype=H.dtype)
            err = float((H @ H.T - eye).abs().max().item())
            sym_err = float((H - H.T).abs().max().item())
            self.assertLess(err, 1e-6, f"Non-orthogonal Hadamard for dim={dim}: max err={err}")
            self.assertLess(sym_err, 1e-6, f"Non-symmetric Hadamard for dim={dim}: max err={sym_err}")

    def test_fp32_invariance_d64_and_d96_and_d384(self) -> None:
        """Verify exact FP32 functional invariance (max_abs_diff < 1e-4) across d_model=64, 96, 384."""
        specs = [
            (64, 8, 2),
            (96, 8, 2),
            (384, 8, 2),
        ]
        for seed, (d_model, n_heads, n_kv_heads) in enumerate(specs):
            torch.manual_seed(100 + seed)
            model_orig = _make_tiny_v2(d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads).eval()
            # Perturb RMSNorm weights away from 1.0 to verify exact gamma absorption too!
            with torch.no_grad():
                for layer in model_orig.dec_layers:
                    layer.norm1.weight.uniform_(0.5, 1.8)
                    layer.norm2.weight.uniform_(0.5, 1.8)
                    layer.norm3.weight.uniform_(0.5, 1.8)
                model_orig.dec_norm.weight.uniform_(0.6, 1.5)

            model_folded = copy.deepcopy(model_orig).eval()
            summary = spinquant_fold.fold_hadamard_v2(model_folded)
            self.assertEqual(summary["d_model"], d_model)
            self.assertGreater(summary["n_r1_dec_layers"], 0)
            self.assertGreater(summary["n_r2_attn_layers"], 0)

            img = torch.rand(1, 3, 64, 128)
            ids = torch.randint(0, 64, (1, 10))
            with torch.no_grad():
                logits_orig = model_orig(img, ids, return_aux=False, return_mtp=False)["logits"]
                logits_fold = model_folded(img, ids, return_aux=False, return_mtp=False)["logits"]
            max_diff = float((logits_fold - logits_orig).abs().max().item())
            self.assertLess(
                max_diff,
                1e-4,
                f"FP32 invariance violated for d_model={d_model}: max_abs_diff={max_diff:.6e}",
            )

            # Also verify KV-cached autoregressive decode invariance
            with torch.no_grad():
                seq_orig = model_orig.generate(img, max_len=6)
                seq_fold = model_folded.generate(img, max_len=6)
            self.assertTrue(
                torch.equal(seq_orig, seq_fold),
                f"KV-cached generate() mismatch for d_model={d_model}",
            )

    def test_int4_quant_error_reduction_on_outliers(self) -> None:
        """Inject channel outliers into decoder weights and verify INT4 error drops after folding."""
        torch.manual_seed(77)
        model_orig = _make_tiny_v2(d_model=128, n_heads=8, n_kv_heads=2).eval()

        # Inject realistic Transformer outlier channels along the decoder residual stream
        # (a few channels 25x larger in weight magnitude -> dominates block_size=32 INT4 scales).
        outlier_cols = [0, 33, 65, 97]
        with torch.no_grad():
            for layer in model_orig.dec_layers:
                for lin in (
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                    layer.cross_attn.q_proj,
                    layer.ffn.w_gate_up,
                ):
                    lin.weight[:, outlier_cols] *= 25.0
                for lin in (
                    layer.self_attn.o_proj,
                    layer.cross_attn.o_proj,
                    layer.ffn.w_down,
                ):
                    lin.weight[outlier_cols, :] *= 25.0
            model_orig.lm_head.weight[:, outlier_cols] *= 25.0

        model_folded = copy.deepcopy(model_orig).eval()
        spinquant_fold.fold_hadamard_v2(model_folded)

        img = torch.rand(1, 3, 64, 128)
        ids = torch.randint(0, 64, (1, 12))
        metrics = spinquant_fold.measure_int4_quant_error(
            model_orig, model_folded, img, ids, block_size=32
        )
        print(
            f"[SpinQuant INT4 Benchmark d=128 bs=32] "
            f"fp32_max_diff={metrics['fp32_max_abs_diff']:.2e} | "
            f"orig_mse={metrics['orig_int4_mse']:.6e} -> folded_mse={metrics['folded_int4_mse']:.6e} "
            f"({metrics['mse_reduction_ratio']:.2f}x better) | "
            f"orig_max={metrics['orig_int4_max_diff']:.4f} -> folded_max={metrics['folded_int4_max_diff']:.4f} "
            f"({metrics['max_diff_reduction_ratio']:.2f}x better)"
        )

        self.assertLess(metrics["fp32_max_abs_diff"], 1e-4)
        self.assertLess(
            metrics["folded_int4_mse"],
            metrics["orig_int4_mse"],
            f"Expected INT4 MSE reduction after Hadamard folding, got {metrics}",
        )
        self.assertGreater(metrics["mse_reduction_ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
