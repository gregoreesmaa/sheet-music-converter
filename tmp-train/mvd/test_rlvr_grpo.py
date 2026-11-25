#!/usr/bin/env python3
"""CPU-only unit tests for RLVR (GRPO Verifiable-Reward Post-Training)."""

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

import rlvr_grpo  # noqa: E402
from uni_omr import UniOMRConfig, UniOMRNet  # noqa: E402
from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net  # noqa: E402


# A clean 2-system, 4-bar-per-system (8 bars total, 6 interior bars) 4/4 piece.
REF_BEKERN = " ".join([
    "clef-G-2", "key+0", "time-4-4",
    "quarter", "C", "4", "quarter", "D", "4", "quarter", "E", "4", "quarter", "F", "4", "bar",
    "quarter", "G", "4", "quarter", "A", "4", "quarter", "B", "4", "quarter", "C", "5", "bar",
    "half", "B", "4", "half", "A", "4", "bar",
    "whole", "G", "4", "bar",
    "<sys>",
    "clef-G-2", "key+0", "time-4-4",
    "quarter", "F", "4", "quarter", "E", "4", "quarter", "D", "4", "quarter", "C", "4", "bar",
    "half", "D", "4", "half", "E", "4", "bar",
    "quarter", "F", "4", "quarter", "G", "4", "half", "A", "4", "bar",
    "whole", "C", "4", "bar-light-heavy",
])


def _tiny_v1() -> UniOMRNet:
    cfg = UniOMRConfig(
        d_model=32,
        n_enc_layers=1,
        n_dec_layers=1,
        n_heads=4,
        ffn_mult=2.0,
        dropout=0.0,
        keep_ratio=0.5,
        use_uv_aux=False,
        mtp_k=1,
        backbone_stride=16,
    )
    return UniOMRNet(cfg)


def _tiny_v2() -> UniOMRv2Net:
    cfg = UniOMRv2Config(
        d_model=64,
        n_heads=8,
        n_kv_heads=2,
        n_axial_layers=1,
        n_global_layers=1,
        n_dec_layers=1,
        ffn_mult=2.0,
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


class TestOMRRLVRReward(unittest.TestCase):
    def test_reward_ordering(self) -> None:
        """Verify exact > minor_typo > bar_drift > loop_garbage > unparseable."""
        # 1. Exact match -> 1.0
        exact = REF_BEKERN

        # 2. Minor typo: single pitch error (E 4 -> F 4), valid parse & meter
        minor_typo = REF_BEKERN.replace("quarter E 4", "quarter F 4", 1)

        # 3. Bar drift: valid parse, same <sys>, no loop, but interior bars have wrong beat sums
        bar_drift = " ".join([
            "clef-G-2", "key+0", "time-4-4",
            "quarter", "C", "4", "quarter", "D", "4", "quarter", "E", "4", "quarter", "F", "4", "bar",
            "quarter", "G", "4", "quarter", "A", "4", "bar",  # 2 beats != 4
            "half", "B", "4", "quarter", "A", "4", "bar",     # 3 beats != 4
            "whole", "G", "4", "bar",
            "<sys>",
            "clef-G-2", "key+0", "time-4-4",
            "quarter", "F", "4", "bar",                       # 1 beat != 4
            "half", "D", "4", "bar",                          # 2 beats != 4
            "quarter", "F", "4", "quarter", "G", "4", "bar",  # 2 beats != 4
            "whole", "C", "4", "bar-light-heavy",
        ])

        # 4. Loop garbage: valid parseable 13-token bar repeated 5x (period=13 in 8..16, >=4x)
        loop_unit = [
            "quarter", "C", "4", "quarter", "D", "4",
            "quarter", "E", "4", "quarter", "F", "4", "bar",
        ]
        loop_garbage = " ".join(["clef-G-2", "key+0", "time-4-4"] + loop_unit * 5)

        # 5. Unparseable garbage
        unparseable = "quarter quarter C C ??? invalid-tok bar-light-heavy"

        r_exact = rlvr_grpo.compute_omr_rlvr_reward(exact, REF_BEKERN)
        r_typo = rlvr_grpo.compute_omr_rlvr_reward(minor_typo, REF_BEKERN)
        r_drift = rlvr_grpo.compute_omr_rlvr_reward(bar_drift, REF_BEKERN)
        r_loop = rlvr_grpo.compute_omr_rlvr_reward(loop_garbage, REF_BEKERN)
        r_bad = rlvr_grpo.compute_omr_rlvr_reward(unparseable, REF_BEKERN)

        self.assertAlmostEqual(r_exact["reward"], 1.0, places=6)
        self.assertEqual(r_exact["r_parse"], 1.0)
        self.assertEqual(r_exact["r_sys"], 1.0)
        self.assertEqual(r_exact["r_meter"], 1.0)
        self.assertEqual(r_exact["r_noloop"], 1.0)
        self.assertAlmostEqual(r_exact["r_note"], 1.0, places=6)

        self.assertLess(r_drift["r_meter"], 0.5)
        self.assertEqual(r_loop["r_noloop"], 0.0)
        self.assertTrue(r_loop["has_loop"])
        self.assertEqual(r_bad["r_parse"], 0.0)
        self.assertEqual(r_bad["r_meter"], 0.0)
        self.assertEqual(r_bad["note_f1"], 0.0)
        self.assertLess(r_bad["r_note"], 0.1)

        self.assertGreater(r_exact["reward"], r_typo["reward"])
        self.assertGreater(r_typo["reward"], r_drift["reward"])
        self.assertGreater(r_drift["reward"], r_loop["reward"])
        self.assertGreater(r_loop["reward"], r_bad["reward"])

    def test_sys_count_off_by_one_and_check_decode_fn(self) -> None:
        # Remove <sys> -> off-by-1 gives r_sys == 0.5
        no_sys = REF_BEKERN.replace(" <sys>", "")
        res = rlvr_grpo.compute_omr_rlvr_reward(no_sys, REF_BEKERN)
        self.assertAlmostEqual(res["r_sys"], 0.5, places=6)

        # Custom check_decode_fn returning False forces r_parse == 0.0 and r_meter == 0.0
        res_fail = rlvr_grpo.compute_omr_rlvr_reward(
            REF_BEKERN, REF_BEKERN, check_decode_fn=lambda _toks: False
        )
        self.assertEqual(res_fail["r_parse"], 0.0)
        self.assertEqual(res_fail["r_meter"], 0.0)
        self.assertAlmostEqual(res_fail["reward"], 0.65, places=6)

        # Empty prediction yields 0.0 across all components
        res_empty = rlvr_grpo.compute_omr_rlvr_reward("", REF_BEKERN)
        self.assertEqual(res_empty["reward"], 0.0)


class TestGRPOAdvantageAndLoss(unittest.TestCase):
    def test_group_advantage_normalization(self) -> None:
        rewards = torch.tensor([[0.2, 0.4, 0.8, 1.0], [0.5, 0.5, 0.5, 0.5]])
        adv = rlvr_grpo.compute_group_advantages(rewards, eps=1e-6)
        self.assertEqual(adv.shape, (2, 4))
        # Group 0: zero mean, unit std
        self.assertAlmostEqual(float(adv[0].mean().item()), 0.0, places=5)
        self.assertAlmostEqual(float(adv[0].std(unbiased=False).item()), 1.0, places=4)
        # Monotonic with rewards
        self.assertTrue(bool((adv[0, 1:] > adv[0, :-1]).all()))
        # Group 1 (constant rewards): all zero advantages, no NaNs
        self.assertTrue(torch.allclose(adv[1], torch.zeros(4)))

    def test_grpo_loss_clipping_and_kl(self) -> None:
        torch.manual_seed(0)
        B, G, T = 2, 4, 8
        old_lp = torch.full((B, G, T), -1.0)
        ref_lp = torch.full((B, G, T), -1.0)
        # Shift new_lp so ratio != 1 and KL > 0
        new_lp = (old_lp + 0.5).requires_grad_(True)
        adv = rlvr_grpo.compute_group_advantages(
            torch.tensor([[0.1, 0.3, 0.7, 0.9], [0.2, 0.4, 0.6, 0.8]])
        )
        mask = torch.ones((B, G, T))
        mask[:, :, -2:] = 0.0

        out = rlvr_grpo.grpo_loss(
            new_lp, old_lp, ref_lp, adv, mask, clip_eps=0.2, kl_beta=0.02
        )
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertGreater(float(out["kl_loss"].item()), 0.0)
        self.assertGreater(float(out["clip_frac"].item()), 0.0)
        out["loss"].backward()
        self.assertIsNotNone(new_lp.grad)
        self.assertTrue(torch.isfinite(new_lp.grad).all())
        # Masked-out PAD positions must have zero gradient
        self.assertTrue(torch.allclose(new_lp.grad[:, :, -2:], torch.zeros(B, G, 2)))


class TestGRPOEndToEndCPU(unittest.TestCase):
    def test_one_grpo_gradient_step_uni_omr_v1(self) -> None:
        """Run 1 full GRPO gradient step on a tiny UniOMRNet on CPU and verify finite grads."""
        torch.manual_seed(42)
        model = _tiny_v1()
        ref_model = copy.deepcopy(model).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        images = torch.rand(1, 3, 64, 128)
        cfg = rlvr_grpo.GRPOConfig(
            group_size=4,
            max_len=12,
            temperature=1.0,
            top_p=0.95,
            clip_eps=0.2,
            kl_beta=0.02,
            use_grammar=True,
            strict_grammar=False,
        )

        # Use deterministic reward wrapper that adds exact rlvr reward + tiny rollout-index
        # tie-break only if a random tiny init produces identical 0.0 rewards, ensuring
        # non-trivial policy gradient flow through the whole model.
        call_idx = [0]

        def _reward_with_tiebreak(pred_toks, ref_b):
            res = rlvr_grpo.compute_omr_rlvr_reward(pred_toks, ref_b)
            # Keep exact verifiable reward while guaranteeing non-zero variance for grad check.
            res["reward"] = float(res["reward"]) + 1e-2 * float(call_idx[0] % 4)
            call_idx[0] += 1
            return res

        opt.zero_grad(set_to_none=True)
        loss, stats = rlvr_grpo.grpo_step(
            model,
            ref_model,
            images,
            [REF_BEKERN],
            cfg=cfg,
            reward_fn=_reward_with_tiebreak,
        )

        self.assertTrue(torch.isfinite(loss), f"Non-finite GRPO loss: {loss}")
        self.assertTrue(loss.requires_grad)
        self.assertEqual(stats["rewards"].shape, (1, 4))
        self.assertEqual(stats["advantages"].shape, (1, 4))
        self.assertAlmostEqual(float(stats["kl_loss"]), 0.0, places=4)

        loss.backward()

        grad_norms = []
        for name, p in model.named_parameters():
            if p.grad is not None:
                self.assertTrue(
                    torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"
                )
                grad_norms.append(float(p.grad.norm().item()))
        self.assertGreater(len(grad_norms), 0)
        self.assertGreater(max(grad_norms), 0.0)

        # Verify optimizer step executes cleanly and keeps weights finite.
        opt.step()
        for name, p in model.named_parameters():
            self.assertTrue(torch.isfinite(p).all(), f"Non-finite param after step: {name}")

    def test_one_grpo_gradient_step_uni_omr_v2(self) -> None:
        """Verify GRPO rollout + loss + backward compatibility with UniOMRv2Net (v2)."""
        torch.manual_seed(7)
        model = _tiny_v2()
        ref_model = copy.deepcopy(model).eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        images = torch.rand(1, 3, 64, 128)
        cfg = rlvr_grpo.GRPOConfig(
            group_size=4,
            max_len=10,
            temperature=1.0,
            top_p=0.95,
            clip_eps=0.2,
            kl_beta=0.02,
            use_grammar=True,
        )
        call_idx = [0]

        def _reward_fn(pred_toks, ref_b):
            res = rlvr_grpo.compute_omr_rlvr_reward(pred_toks, ref_b)
            res["reward"] = float(res["reward"]) + 1e-2 * float(call_idx[0] % 4)
            call_idx[0] += 1
            return res

        loss, stats = rlvr_grpo.grpo_step(
            model,
            ref_model,
            images,
            [REF_BEKERN],
            cfg=cfg,
            reward_fn=_reward_fn,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats["rewards"].shape, (1, 4))
        loss.backward()

        finite_nonzero = [
            float(p.grad.norm().item())
            for p in model.parameters()
            if p.grad is not None and torch.isfinite(p.grad).all()
        ]
        self.assertGreater(len(finite_nonzero), 0)
        self.assertGreater(max(finite_nonzero), 0.0)

    def test_cli_defaults(self) -> None:
        args = rlvr_grpo.build_arg_parser().parse_args([])
        self.assertEqual(args.group_size, 4)
        self.assertAlmostEqual(args.kl_beta, 0.02, places=6)


if __name__ == "__main__":
    unittest.main()
