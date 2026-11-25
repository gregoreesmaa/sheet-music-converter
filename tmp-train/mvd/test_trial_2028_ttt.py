"""CPU unit tests for 2028 Trial: On-Device Test-Time Training (TTT).

Runs strictly on CPU on a tiny ``UniOMRNet`` so it never touches the GPU.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import trial_2028_ttt_adapt as ttt_mod  # noqa: E402
import uni_omr  # noqa: E402


def _tiny_model(seed: int = 123) -> uni_omr.UniOMRNet:
    torch.manual_seed(seed)
    cfg = uni_omr.UniOMRConfig(
        d_model=64,
        n_enc_layers=2,
        n_dec_layers=2,
        n_heads=4,
        ffn_mult=2.0,
        dropout=0.0,
        keep_ratio=0.50,
        n_bg_summary=4,
        use_ink_prior=True,
        use_uv_aux=False,
        mtp_k=2,
    )
    model = uni_omr.UniOMRNet(cfg)
    model.eval()
    return model


def _synthetic_page(seed: int = 7, h: int = 96, w: int = 128) -> torch.Tensor:
    """Create a synthetic parchment-and-ink page tensor in [0, 1]."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = torch.full((1, 3, h, w), 0.92, dtype=torch.float32)
    # Draw staff-like dark horizontal bands and notehead-like ink blobs
    for y in (20, 26, 32, 38, 44, 62, 68, 74, 80, 86):
        img[:, :, y : y + 1, 12 : w - 12] = 0.15
    for x in range(20, w - 20, 14):
        y = 22 + ((x * 7) % 56)
        img[:, :, y : y + 4, x : x + 4] = 0.08
    noise = (torch.rand((1, 3, h, w), generator=g) - 0.5) * 0.03
    return (img + noise).clamp(0.0, 1.0)


class TestTrial2028TTT(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_snapshot_selects_only_rmsnorm_under_0_1_percent(self) -> None:
        model = _tiny_model()
        ttt = ttt_mod.TTTSession(model, ttt_lr=5e-5, adapt_scope="norms")

        # Verify every selected parameter belongs to an RMSNorm in backbone/enc_layers/enc_norm
        rmsnorm_param_ids = set()
        for root_name in ("backbone", "enc_layers", "enc_norm"):
            root = getattr(model, root_name)
            for mod in root.modules():
                if isinstance(mod, uni_omr.RMSNorm):
                    for p in mod.parameters(recurse=False):
                        rmsnorm_param_ids.add(id(p))

        self.assertGreater(len(ttt.adapted_param_names), 0)
        self.assertEqual(ttt._adapted_param_ids, rmsnorm_param_ids)
        # Tiny model has ~5 RMSNorms * 64 = 320 params << 0.1% of total params
        self.assertLess(ttt.adapted_param_fraction, 0.001)

    def test_adapt_updates_only_rmsnorm_decreases_consistency_and_restores_bit_exact(self) -> None:
        model = _tiny_model(seed=101)
        img = _synthetic_page(seed=11)

        # Snapshot EVERY parameter in the entire model before TTT
        full_before = {name: p.detach().clone() for name, p in model.named_parameters()}

        # Use a higher LR for the 3-step tiny-model unit test so the drop is unmistakable
        ttt = ttt_mod.TTTSession(
            model,
            ttt_lr=1e-2,
            adapt_scope="norms",
            conf_threshold=0.85,
            cons_weight=1.0,
            ent_weight=1.0,
            kl_weight=0.5,
            l2_weight=1e-4,
            grad_clip=2.0,
            max_decode_len=16,
        )

        # Measure pre-TTT consistency loss on step-0 jittered view
        img_aug0 = ttt_mod.make_jittered_view(img, step_idx=0, seed=ttt.seed)
        with torch.no_grad():
            mem_c0, _, r_c0 = model.encode_image(img, return_aux=False)
            mem_a0, _, r_a0 = model.encode_image(img_aug0, return_aux=False)
            pre_cons, _, _ = ttt_mod.compute_multiview_consistency_loss(mem_c0, r_c0, mem_a0, r_a0)
            pre_cons_val = float(pre_cons.item())

        stats = ttt.adapt_on_page(img, steps=3, max_decode_len=16)

        self.assertEqual(stats["steps"], 3)
        self.assertEqual(len(stats["cons_loss_history"]), 3)
        self.assertEqual(len(stats["grad_norm_history"]), 3)
        for gn in stats["grad_norm_history"]:
            self.assertTrue(torch.isfinite(torch.tensor(gn)).item())
            self.assertGreater(gn, 0.0)

        # Verify ONLY adapted RMSNorm params changed; all other params are bit-identical
        adapted_names = set(ttt.adapted_param_names)
        any_adapted_changed = False
        for name, p in model.named_parameters():
            diff = (p.detach() - full_before[name]).abs().max().item()
            if name in adapted_names:
                if diff > 0.0:
                    any_adapted_changed = True
            else:
                self.assertEqual(
                    diff,
                    0.0,
                    f"Non-adapted parameter {name} was unexpectedly modified (max_diff={diff})",
                )
        self.assertTrue(any_adapted_changed, "Expected adapted RMSNorm weights to update")

        # Verify consistency loss decreased on the reference jittered view
        with torch.no_grad():
            mem_c1, _, r_c1 = model.encode_image(img, return_aux=False)
            mem_a1, _, r_a1 = model.encode_image(img_aug0, return_aux=False)
            post_cons, _, _ = ttt_mod.compute_multiview_consistency_loss(mem_c1, r_c1, mem_a1, r_a1)
            post_cons_val = float(post_cons.item())
        self.assertLess(
            post_cons_val,
            pre_cons_val,
            f"Expected consistency loss to decrease: pre={pre_cons_val:.6f}, post={post_cons_val:.6f}",
        )

        # Verify restore() restores bit-exact (max_diff == 0.0) original weights across entire model
        ttt.restore()
        max_diff_all = max(
            float((p.detach() - full_before[name]).abs().max().item())
            for name, p in model.named_parameters()
        )
        self.assertEqual(max_diff_all, 0.0, f"restore() was not bit-exact: max_diff={max_diff_all}")

    def test_loss2_confidence_gated_entropy_and_kl_anchor_path(self) -> None:
        """Verify Loss 2 activates when tokens pass confidence gate and stays finite."""
        model = _tiny_model(seed=202)
        img = _synthetic_page(seed=22)

        # Use a low confidence threshold on an untrained tiny model so Loss 2 is guaranteed active
        ttt = ttt_mod.TTTSession(
            model,
            ttt_lr=5e-4,
            adapt_scope="norms",
            conf_threshold=0.0,
            cons_weight=1.0,
            ent_weight=1.0,
            kl_weight=0.5,
            max_decode_len=12,
        )
        stats = ttt.adapt_on_page(img, steps=2, max_decode_len=12)
        self.assertEqual(stats["steps"], 2)
        self.assertGreater(stats["n_confident_tokens"], 0)
        self.assertTrue(all(v > 0.0 for v in stats["ent_loss_history"]))
        self.assertTrue(all(v >= 0.0 for v in stats["kl_loss_history"]))
        ttt.restore()

    def test_detect_loop_mask_flags_single_and_ngram_loops(self) -> None:
        # 1. Single-token repeat >= 4
        seq1 = [1, 2, 5, 5, 5, 5, 5, 3]
        m1 = ttt_mod.detect_loop_mask(seq1)
        self.assertEqual(m1, [False, False, False, False, False, True, True, False])

        # 2. 2-gram repeat >= 3 (10, 11 repeated 3 times)
        seq2 = [1, 10, 11, 10, 11, 10, 11, 2]
        m2 = ttt_mod.detect_loop_mask(seq2)
        self.assertEqual(m2, [False, False, False, True, True, True, True, False])

    def test_cli_arg_parser_supports_required_flags(self) -> None:
        ap = ttt_mod.build_arg_parser()
        args = ap.parse_args([
            "--ckpt", "dummy.pt",
            "--manifest", "dummy_manifest.json",
            "--ttt-steps", "3",
            "--ttt-lr", "5e-5",
            "--device", "cpu",
            "--bf16",
            "--max-pieces", "4",
        ])
        self.assertEqual(args.ckpt, "dummy.pt")
        self.assertEqual(args.manifest, "dummy_manifest.json")
        self.assertEqual(args.ttt_steps, 3)
        self.assertAlmostEqual(args.ttt_lr, 5e-5)
        self.assertEqual(args.device, "cpu")
        self.assertTrue(args.bf16)
        self.assertEqual(args.max_pieces, 4)


if __name__ == "__main__":
    unittest.main()
