"""Unit tests for verifier-guided K=4 GRPO (grpo_bar_math_step & train.py wiring)."""

from __future__ import annotations

import copy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image
import torch

_MVD_DIR = Path(__file__).resolve().parent
if str(_MVD_DIR) not in sys.path:
    sys.path.insert(0, str(_MVD_DIR))

import data  # noqa: E402
import model as mvd_model  # noqa: E402
import train  # noqa: E402
from unsup_bar_math import (  # noqa: E402
    grpo_bar_math_step,
    run_grpo_bar_math_round,
)


class TestGrpoBarMathStep(unittest.TestCase):
    """Tests for grpo_bar_math_step, run_grpo_bar_math_round, and train.py dispatch."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)
        cls.vocab = list(data.VOCAB)
        cls.vsize = len(cls.vocab) + 3
        cls.base_model = mvd_model.MvdNet(
            vocab=cls.vsize,
            d_model=32,
            layers=1,
            heads=4,
            dim_ff=64,
            dropout=0.0,
            anisotropic=True,
            pretrained=False,
        )
        cls.strips = torch.randn(2, 3, 64, 128)

    def _encode_seq(self, text: str, eos_id: int) -> list[int]:
        return data.encode(text.split()) + [eos_id]

    def _valid_ids(self, eos_id: int) -> list[int]:
        return self._encode_seq(
            "clef-G-2 key+0 time-4-4 "
            "voice-1 quarter C 4 quarter D 4 quarter E 4 quarter F 4 bar "
            "voice-1 half G 4 half A 4 bar",
            eos_id,
        )

    def _invalid_bar_ids(self, eos_id: int) -> list[int]:
        # Valid notes (not degenerate) but overfull beat sums (5.0 beats in 4/4) in both bars.
        return self._encode_seq(
            "clef-G-2 key+0 time-4-4 "
            "voice-1 whole C 4 quarter D 4 bar "
            "voice-1 whole E 4 quarter F 4 bar",
            eos_id,
        )

    def _degenerate_ids(self, eos_id: int) -> list[int]:
        return self._encode_seq(
            "clef-G-2 key+0 time-4-4 "
            "voice-1 bar-rest bar voice-1 bar-rest bar",
            eos_id,
        )

    def _pad_seqs(self, seqs: list[list[int]], pad_id: int) -> list[list[int]]:
        max_l = max(len(s) for s in seqs)
        return [s + [pad_id] * (max_l - len(s)) for s in seqs]

    def test_advantage_normalization_positive_valid_negative_degenerate(self):
        student = copy.deepcopy(self.base_model)
        ref = copy.deepcopy(self.base_model)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

        v = self._valid_ids(student.eos_id)
        inv = self._invalid_bar_ids(student.eos_id)
        deg = self._degenerate_ids(student.eos_id)
        v_p, inv_p, deg_p = self._pad_seqs([v, inv, deg], student.pad_id)

        # Strip 0: [valid, valid, invalid, degenerate] -> mixed rewards
        # Strip 1: [valid, degenerate, degenerate, degenerate] -> mixed rewards
        fake_beams = torch.tensor(
            [
                [v_p, v_p, inv_p, deg_p],
                [v_p, deg_p, deg_p, deg_p],
            ],
            dtype=torch.long,
        )

        # Optional visual_scorer adds a deterministic bonus for longer token sequences.
        def _vis(strip_t, toks):
            self.assertEqual(strip_t.ndim, 3)
            return 1.0 if "quarter" in toks else 0.0

        with mock.patch.object(student, "beam_search", return_value=fake_beams):
            loss, stats = grpo_bar_math_step(
                student,
                ref,
                self.strips,
                id2tok=self.vocab,
                num_beams=4,
                kl_beta=0.08,
                max_len=48,
                visual_scorer=_vis,
                w_bm=0.75,
                w_vis=0.25,
                w_deg=0.50,
            )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats["active_strips"], 2)
        self.assertAlmostEqual(stats["active_frac"], 1.0, places=6)

        adv0 = stats["advantages"][0]
        adv1 = stats["advantages"][1]
        self.assertIsNotNone(adv0)
        self.assertIsNotNone(adv1)
        # Valid candidates get positive advantage; invalid/degenerate get negative.
        self.assertGreater(adv0[0], 0.0)
        self.assertGreater(adv0[1], 0.0)
        self.assertLess(adv0[2], 0.0)
        self.assertLess(adv0[3], 0.0)
        self.assertGreater(adv1[0], 0.0)
        self.assertTrue(all(a < 0.0 for a in adv1[1:]))
        # Group-relative advantages sum to ~0 per strip.
        self.assertAlmostEqual(sum(adv0), 0.0, places=5)
        self.assertAlmostEqual(sum(adv1), 0.0, places=5)

    def test_flat_reward_groups_produce_zero_loss_without_nan(self):
        student = copy.deepcopy(self.base_model)
        ref = copy.deepcopy(self.base_model)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

        v = self._valid_ids(student.eos_id)
        deg = self._degenerate_ids(student.eos_id)
        v_p, deg_p = self._pad_seqs([v, deg], student.pad_id)

        # Strip 0: all identical valid -> std == 0; Strip 1: all degenerate -> std == 0.
        fake_beams = torch.tensor(
            [
                [v_p, v_p, v_p, v_p],
                [deg_p, deg_p, deg_p, deg_p],
            ],
            dtype=torch.long,
        )

        student.train()
        with mock.patch.object(student, "beam_search", return_value=fake_beams):
            loss, stats = grpo_bar_math_step(
                student,
                ref,
                self.strips,
                id2tok=self.vocab,
                num_beams=4,
                kl_beta=0.08,
                max_len=48,
            )

        self.assertTrue(student.training)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss.item()), 0.0, places=7)
        self.assertFalse(loss.requires_grad)
        self.assertEqual(stats["active_strips"], 0)
        self.assertEqual(stats["active_frac"], 0.0)
        self.assertEqual(stats["advantages"], [None, None])

    def test_gradients_flow_to_model_ref_stays_frozen_and_kl_nonnegative(self):
        student = copy.deepcopy(self.base_model)
        ref = copy.deepcopy(self.base_model)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

        v = self._valid_ids(student.eos_id)
        deg = self._degenerate_ids(student.eos_id)
        v_p, deg_p = self._pad_seqs([v, deg], student.pad_id)

        # Strip 0 active (mixed), Strip 1 flat (skipped).
        fake_beams = torch.tensor(
            [
                [v_p, v_p, deg_p, deg_p],
                [deg_p, deg_p, deg_p, deg_p],
            ],
            dtype=torch.long,
        )

        student.train()
        student.zero_grad()
        with mock.patch.object(student, "beam_search", return_value=fake_beams):
            loss, stats = grpo_bar_math_step(
                student,
                ref,
                self.strips,
                id2tok=self.vocab,
                num_beams=4,
                kl_beta=0.08,
                max_len=48,
            )

        self.assertEqual(stats["active_indices"], [0])
        self.assertAlmostEqual(stats["active_frac"], 0.5, places=6)
        self.assertGreaterEqual(stats["kl_loss"], 0.0)
        self.assertAlmostEqual(stats["kl_loss"], 0.0, places=5)
        self.assertTrue(torch.isfinite(loss))

        loss.backward()
        grad_norms = [
            float(p.grad.norm().item())
            for p in student.parameters()
            if p.grad is not None
        ]
        self.assertGreater(len(grad_norms), 0)
        self.assertTrue(all(torch.isfinite(torch.tensor(g)) for g in grad_norms))
        self.assertGreater(sum(grad_norms), 0.0)
        self.assertTrue(all(p.grad is None for p in ref.parameters()))

        # Perturb student output bias -> KL(student || ref) becomes strictly positive.
        with torch.no_grad():
            student.out.bias.add_(0.5)
        with mock.patch.object(student, "beam_search", return_value=fake_beams):
            loss2, stats2 = grpo_bar_math_step(
                student,
                ref,
                self.strips,
                id2tok=self.vocab,
                num_beams=4,
                kl_beta=0.08,
                max_len=48,
            )
        self.assertTrue(torch.isfinite(loss2))
        self.assertGreater(stats2["kl_loss"], 0.0)

    def test_greedy_kv_fallback_when_beam_search_unavailable(self):
        student = copy.deepcopy(self.base_model)
        ref = copy.deepcopy(self.base_model)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

        v = self._valid_ids(student.eos_id)
        fake_greedy = torch.tensor([v, v], dtype=torch.long)

        # Simulate a model without beam_search -> falls back to greedy_kv (K=1 -> flat -> zero loss).
        with mock.patch.object(student, "beam_search", new=None), mock.patch.object(
            student, "greedy_kv", return_value=fake_greedy
        ) as mock_greedy:
            loss, stats = grpo_bar_math_step(
                student,
                ref,
                self.strips,
                id2tok=self.vocab,
                num_beams=4,
                max_len=48,
            )
        mock_greedy.assert_called_once()
        self.assertEqual(stats["num_beams"], 1)
        self.assertEqual(stats["active_strips"], 0)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=7)

    def test_run_grpo_bar_math_round_and_train_dispatch(self):
        student = copy.deepcopy(self.base_model)
        student.in_chans = 3
        opt = torch.optim.SGD(student.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        v = self._valid_ids(student.eos_id)
        deg = self._degenerate_ids(student.eos_id)
        v_p, deg_p = self._pad_seqs([v, deg], student.pad_id)
        fake_beams = torch.tensor(
            [[v_p, v_p, deg_p, deg_p]], dtype=torch.long
        )

        with tempfile.TemporaryDirectory() as td:
            p0 = Path(td) / "strip0.png"
            Image.new("RGB", (128, 64), "white").save(p0)

            with mock.patch.object(student, "beam_search", return_value=fake_beams):
                res = run_grpo_bar_math_round(
                    net=student,
                    opt=opt,
                    scaler=scaler,
                    strip_paths=[str(p0)],
                    step=10,
                    steps=1,
                    weight=0.25,
                    kl_beta=0.08,
                    height=64,
                    device="cpu",
                    batch_size=1,
                    num_beams=4,
                    max_len=48,
                )
            self.assertEqual(res["ran_steps"], 1)
            self.assertAlmostEqual(res["active_frac"], 1.0, places=6)

        # Verify train.run_stage2_unsup_round dispatches mode='grpo_bar_math' cleanly.
        calls = []
        mock_bm = types.SimpleNamespace(
            run_grpo_bar_math_round=lambda **kw: (
                calls.append((kw["step"], kw["steps"], len(kw["strip_paths"])))
                or {"ran_steps": 1, "active_frac": 0.5, "mean_loss": 0.01}
            )
        )
        summary = train.run_stage2_unsup_round(
            net=student,
            opt=opt,
            scaler=scaler,
            strip_paths=["s0.png", "s1.png"],
            step=20,
            round_steps=5,
            mode="grpo_bar_math",
            weight=0.25,
            kl_beta=0.08,
            height=64,
            device="cpu",
            unsup_bar_math_mod=mock_bm,
        )
        self.assertEqual(summary["ran"], ["grpo_bar_math"])
        self.assertEqual(calls, [(20, 5, 2)])


if __name__ == "__main__":
    unittest.main()
