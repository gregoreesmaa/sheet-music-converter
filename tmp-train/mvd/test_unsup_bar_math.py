"""Unit tests for bar-math unsupervised training and bekern beat-sum validator."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

_MVD_DIR = Path(__file__).resolve().parent
if str(_MVD_DIR) not in sys.path:
    sys.path.insert(0, str(_MVD_DIR))

import data  # noqa: E402
import model as mvd_model  # noqa: E402
from unsup_bar_math import (  # noqa: E402
    BarMathResult,
    bar_math_unsup_step,
    score_bekern_bar_math,
)


class TestScoreBekernBarMath(unittest.TestCase):
    """Exact beat-sum validation across meters, rhythms, voices, and errors."""

    def test_4_4_basic_and_default_meter(self):
        toks = (
            "clef-G-2 key+0 time-4-4 "
            "voice-1 quarter C 4 quarter D 4 quarter E 4 quarter F 4 bar "
            "voice-1 half G 4 half A 4 bar"
        ).split()
        res = score_bekern_bar_math(toks)
        self.assertIsInstance(res, BarMathResult)
        self.assertEqual(res.n_bars, 2)
        self.assertEqual(res.n_valid_bars, 2)
        self.assertAlmostEqual(res.valid_bar_ratio, 1.0, places=6)
        self.assertAlmostEqual(res.mean_beat_err, 0.0, places=6)
        self.assertFalse(res.is_degenerate)
        self.assertIsNone(res.degenerate_reason)

        # Default 4/4 when no time-N-D header is emitted.
        no_hdr = "voice-1 quarter C 4 quarter D 4 half E 4 bar".split()
        res_def = score_bekern_bar_math(no_hdr)
        self.assertEqual(res_def.n_valid_bars, 1)
        self.assertAlmostEqual(res_def.mean_beat_err, 0.0, places=6)

    def test_meters_3_4_6_8_2_2_5_4(self):
        cases = [
            # 3/4 -> 3.0 quarter beats
            (
                "time-3-4 voice-1 quarter C 4 quarter D 4 quarter E 4 bar "
                "voice-1 half F 4 quarter G 4 bar",
                3.0,
            ),
            # 6/8 -> 3.0 quarter beats
            (
                "time-6-8 voice-1 eighth C 4 eighth D 4 eighth E 4 "
                "eighth F 4 eighth G 4 eighth A 4 bar "
                "voice-1 quarter C 5 dot quarter G 4 dot bar",
                3.0,
            ),
            # 2/2 -> 4.0 quarter beats
            (
                "time-2-2 voice-1 half C 4 half D 4 bar "
                "voice-1 whole E 4 bar",
                4.0,
            ),
            # 5/4 -> 5.0 quarter beats
            (
                "time-5-4 voice-1 half C 4 half D 4 quarter E 4 bar "
                "voice-1 quarter F 4 quarter G 4 half A 4 quarter B 4 bar",
                5.0,
            ),
        ]
        for seq, expected in cases:
            with self.subTest(expected=expected, seq=seq[:20]):
                res = score_bekern_bar_math(seq.split())
                self.assertEqual(res.n_bars, 2)
                self.assertEqual(res.n_valid_bars, 2)
                self.assertAlmostEqual(res.valid_bar_ratio, 1.0, places=6)
                self.assertAlmostEqual(res.mean_beat_err, 0.0, places=6)
                self.assertFalse(res.is_degenerate)
                self.assertAlmostEqual(
                    res.per_bar_details[0]["expected_beats"], expected, places=6
                )

    def test_single_and_double_dots_and_subdivisions(self):
        # 1.5 + 1.75 + 0.25 + 0.5 = 4.0
        # 3.0 + 0.75 + 0.125 + 0.0625 + 0.0625 = 4.0
        toks = (
            "time-4-4 "
            "voice-1 quarter C 4 dot quarter D 4 a1 dot dot 16th E 4 eighth F 4 bar "
            "voice-1 half G 4 dot eighth A 4 dot 32nd B 4 64th C 5 64th D 5 bar"
        ).split()
        res = score_bekern_bar_math(toks)
        self.assertEqual(res.n_bars, 2)
        self.assertEqual(res.n_valid_bars, 2)
        self.assertAlmostEqual(res.mean_beat_err, 0.0, places=6)
        self.assertFalse(res.is_degenerate)

    def test_chords_rests_and_grace_notes(self):
        # Bar 1: grace (0) + half chord (2) + rest quarter (1) + quarter chord (1) = 4
        # Bar 2: half (2) + rest eighth dot (0.75) + 16th (0.25) + quarter (1) = 4
        toks = (
            "time-4-4 "
            "voice-1 grace 16th B 4 half C 4 chord half E 4 chord half G 4 "
            "rest quarter quarter D 4 chord quarter F 4 a1 bar "
            "voice-1 half E 4 rest eighth dot 16th F 4 quarter G 4 bar"
        ).split()
        res = score_bekern_bar_math(toks)
        self.assertEqual(res.n_bars, 2)
        self.assertEqual(res.n_valid_bars, 2)
        self.assertAlmostEqual(res.mean_beat_err, 0.0, places=6)
        self.assertFalse(res.is_degenerate)

    def test_tuplets_canonical_and_shorthand(self):
        # Bar 1: triplet eighths (3 * 0.5 * 2/3 = 1.0) + quintuplet 16ths
        #        (5 * 0.25 * 4/5 = 1.0) + half (2.0) = 4.0
        # Bar 2: triplet quarters (3 * 1.0 * 2/3 = 2.0) + shorthand tuplet-3
        #        eighths (1.0) + quarter (1.0) = 4.0
        toks = (
            "time-4-4 "
            "voice-1 tuplet-start-3-2 eighth C 4 eighth D 4 eighth E 4 tuplet-stop "
            "tuplet-start-5-4 16th F 4 16th G 4 16th A 4 16th B 4 16th C 5 tuplet-stop "
            "half D 5 bar "
            "voice-1 tuplet-start-3-2 quarter E 4 quarter F 4 quarter G 4 tuplet-stop "
            "tuplet-3 eighth A 4 eighth B 4 eighth C 5 tuplet-stop quarter D 4 bar"
        ).split()
        res = score_bekern_bar_math(toks)
        self.assertEqual(res.n_bars, 2)
        self.assertEqual(res.n_valid_bars, 2)
        self.assertAlmostEqual(res.valid_bar_ratio, 1.0, places=6)
        self.assertAlmostEqual(res.mean_beat_err, 0.0, places=6)
        self.assertFalse(res.is_degenerate)

    def test_mid_stream_meter_changes(self):
        # Bar 1: 4/4 (4 beats), Bar 2: 3/4 (3 beats), courtesy time-6-8 mid-bar 2
        # -> Bar 3: 6/8 (3 beats).
        toks = (
            "time-4-4 voice-1 half C 4 half D 4 bar "
            "time-3-4 voice-1 quarter E 4 quarter F 4 quarter G 4 time-6-8 bar "
            "voice-1 quarter A 4 dot quarter B 4 dot bar"
        ).split()
        res = score_bekern_bar_math(toks)
        self.assertEqual(res.n_bars, 3)
        self.assertEqual(res.n_valid_bars, 3)
        self.assertAlmostEqual(res.mean_beat_err, 0.0, places=6)
        self.assertEqual(res.per_bar_details[0]["time_sig"], "4/4")
        self.assertEqual(res.per_bar_details[1]["time_sig"], "3/4")
        self.assertEqual(res.per_bar_details[2]["time_sig"], "6/8")

    def test_multi_voice_and_multi_staff(self):
        toks = (
            "time-4-4 "
            "staff-1 voice-1 half C 5 half D 5 voice-2 whole E 4 "
            "staff-2 voice-1 whole C 3 bar"
        ).split()
        res = score_bekern_bar_math(toks)
        self.assertEqual(res.n_bars, 1)
        self.assertEqual(res.n_valid_bars, 1)
        self.assertAlmostEqual(res.mean_beat_err, 0.0, places=6)
        self.assertEqual(
            res.per_bar_details[0]["voice_beats"],
            {"s1:v1": 4.0, "s1:v2": 4.0, "s2:v1": 4.0},
        )

    def test_pickup_and_final_incomplete_bars(self):
        # 1-beat pickup in 4/4, full middle bar, 3-beat final bar.
        toks = (
            "time-4-4 "
            "voice-1 quarter G 4 bar "
            "voice-1 quarter C 5 quarter D 5 quarter E 5 quarter F 5 bar "
            "voice-1 half G 5 quarter E 5 bar-light-heavy"
        ).split()
        res_allow = score_bekern_bar_math(
            toks, allow_pickup=True, allow_final_incomplete=True
        )
        self.assertEqual(res_allow.n_bars, 3)
        self.assertEqual(res_allow.n_valid_bars, 3)
        self.assertAlmostEqual(res_allow.valid_bar_ratio, 1.0, places=6)
        self.assertAlmostEqual(res_allow.mean_beat_err, 0.0, places=6)
        self.assertTrue(res_allow.per_bar_details[0]["is_pickup"])
        self.assertTrue(res_allow.per_bar_details[2]["is_final_incomplete"])

        # Strict: disallow pickup & final incomplete -> errors 3.0, 0.0, 1.0 -> mean 4/3.
        res_strict = score_bekern_bar_math(
            toks, allow_pickup=False, allow_final_incomplete=False
        )
        self.assertEqual(res_strict.n_valid_bars, 1)
        self.assertAlmostEqual(res_strict.valid_bar_ratio, 1.0 / 3.0, places=6)
        self.assertAlmostEqual(res_strict.mean_beat_err, 4.0 / 3.0, places=6)

    def test_intentional_beat_sum_errors_exact_mean_err(self):
        # 3 bars in 4/4:
        # Bar 0: 4.0 beats (err 0.0)
        # Bar 1: 3.0 beats (err 1.0)
        # Bar 2: 4.5 beats (err 0.5) -- overfull, never excused by allow_final_incomplete
        toks = (
            "time-4-4 "
            "voice-1 quarter C 4 quarter D 4 quarter E 4 quarter F 4 bar "
            "voice-1 quarter G 4 quarter A 4 quarter B 4 bar "
            "voice-1 whole C 5 eighth D 5 bar"
        ).split()
        res = score_bekern_bar_math(toks)
        self.assertEqual(res.n_bars, 3)
        self.assertEqual(res.n_valid_bars, 1)
        self.assertAlmostEqual(res.valid_bar_ratio, 1.0 / 3.0, places=6)
        self.assertAlmostEqual(res.mean_beat_err, (0.0 + 1.0 + 0.5) / 3.0, places=6)
        self.assertAlmostEqual(res.per_bar_details[1]["beat_err"], 1.0, places=6)
        self.assertAlmostEqual(res.per_bar_details[2]["beat_err"], 0.5, places=6)


class TestAntiDegeneracyGuard(unittest.TestCase):
    """Reward-hacking outputs must always be flagged as degenerate."""

    def test_zero_bars_rejected(self):
        res = score_bekern_bar_math(["clef-G-2", "key+0", "time-4-4"])
        self.assertTrue(res.is_degenerate)
        self.assertEqual(res.degenerate_reason, "zero_bars")
        self.assertEqual(res.valid_bar_ratio, 0.0)

    def test_all_rests_rejected(self):
        toks = (
            "time-4-4 "
            "voice-1 bar-rest bar "
            "voice-1 rest whole bar "
            "voice-1 rest half rest half bar"
        ).split()
        res = score_bekern_bar_math(toks)
        self.assertAlmostEqual(res.valid_bar_ratio, 1.0, places=6)
        self.assertTrue(res.is_degenerate)
        self.assertIn("excessive_rests", res.degenerate_reason or "")

    def test_single_pitch_loop_rejected(self):
        # 3 bars of repeated quarter C 4: beat math is 100% valid, but degenerate!
        bar = "voice-1 quarter C 4 quarter C 4 quarter C 4 quarter C 4 bar"
        toks = ("time-4-4 " + " ".join([bar] * 3)).split()
        res = score_bekern_bar_math(toks)
        self.assertAlmostEqual(res.valid_bar_ratio, 1.0, places=6)
        self.assertTrue(res.is_degenerate)
        self.assertIn("low_pitch_diversity", res.degenerate_reason or "")

    def test_identical_bar_loop_collapse_rejected(self):
        # 4 identical bars with >1 pitch still caught by identical_bar_loop.
        bar = "voice-1 quarter C 4 quarter D 4 quarter E 4 quarter F 4 bar"
        toks = ("time-4-4 " + " ".join([bar] * 4)).split()
        res = score_bekern_bar_math(toks)
        self.assertAlmostEqual(res.valid_bar_ratio, 1.0, places=6)
        self.assertTrue(res.is_degenerate)
        self.assertIn("identical_bar_loop", res.degenerate_reason or "")


class TestBarMathUnsupStep(unittest.TestCase):
    """End-to-end CPU tests of bar_math_unsup_step on a tiny MvdNet."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.vocab = list(data.VOCAB)
        cls.vsize = len(cls.vocab) + 3
        # Tiny MvdNet (CPU-only, fast): small decoder dims, random init.
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
        cls.imgs = torch.randn(2, 3, 64, 128)

    def _valid_seq_ids(self, eos_id: int) -> list[int]:
        toks = (
            "clef-G-2 key+0 time-4-4 "
            "voice-1 quarter C 4 quarter D 4 quarter E 4 quarter F 4 bar "
            "voice-1 half G 4 half A 4 bar"
        ).split()
        return data.encode(toks) + [eos_id]

    def _degenerate_seq_ids(self, eos_id: int) -> list[int]:
        toks = (
            "clef-G-2 key+0 time-4-4 "
            "voice-1 bar-rest bar voice-1 bar-rest bar"
        ).split()
        return data.encode(toks) + [eos_id]

    def test_zero_loss_when_all_candidates_fail_gate(self):
        student = copy.deepcopy(self.base_model)
        ref = copy.deepcopy(self.base_model)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

        # Force decoded candidates to be degenerate all-rests.
        deg_ids = self._degenerate_seq_ids(student.eos_id)
        fake_beams = torch.tensor(
            [[deg_ids, deg_ids], [deg_ids, deg_ids]], dtype=torch.long
        )
        student.train()
        with mock.patch.object(student, "beam_search", return_value=fake_beams):
            loss, stats = bar_math_unsup_step(
                student,
                ref,
                self.vocab,
                self.imgs,
                kl_beta=0.1,
                min_valid_ratio=0.95,
                max_len=48,
                num_candidates=2,
            )

        self.assertTrue(student.training)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=7)
        self.assertEqual(stats["accepted_frac"], 0.0)
        self.assertEqual(stats["n_accepted"], 0)
        self.assertEqual(stats["degenerate_frac"], 1.0)

    def test_differentiable_step_gradients_and_kl_anchor(self):
        student = copy.deepcopy(self.base_model)
        ref = copy.deepcopy(self.base_model)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

        valid_ids = self._valid_seq_ids(student.eos_id)
        deg_ids = self._degenerate_seq_ids(student.eos_id)
        # Pad shorter sequence to match beam tensor shape [B=2, K=2, L].
        L = max(len(valid_ids), len(deg_ids))
        v_pad = valid_ids + [student.pad_id] * (L - len(valid_ids))
        d_pad = deg_ids + [student.pad_id] * (L - len(deg_ids))
        # Item 0: beam 0 degenerate, beam 1 valid -> should select beam 1!
        # Item 1: both beams degenerate -> should be rejected!
        fake_beams = torch.tensor(
            [[d_pad, v_pad], [d_pad, d_pad]], dtype=torch.long
        )

        student.train()
        student.zero_grad()
        with mock.patch.object(student, "beam_search", return_value=fake_beams):
            loss, stats = bar_math_unsup_step(
                student,
                ref,
                self.vocab,
                self.imgs,
                kl_beta=0.1,
                min_valid_ratio=0.95,
                max_len=48,
                num_candidates=2,
            )

        self.assertTrue(student.training)
        self.assertAlmostEqual(stats["accepted_frac"], 0.5, places=6)
        self.assertEqual(stats["accepted_indices"], [0])
        self.assertAlmostEqual(stats["mean_weight"], 1.0, places=6)
        # Identical weights + dropout=0 -> KL(student || ref) == 0.
        self.assertAlmostEqual(stats["kl_loss"], 0.0, places=5)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.item()), 0.0)

        # Verify gradients flow to student and NOT to frozen ref_model.
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
            loss2, stats2 = bar_math_unsup_step(
                student,
                ref,
                self.vocab,
                self.imgs,
                kl_beta=0.1,
                min_valid_ratio=0.95,
                max_len=48,
                num_candidates=2,
            )
        self.assertTrue(torch.isfinite(loss2))
        self.assertGreater(stats2["kl_loss"], 0.0)

    def test_multi_view_agreement_gate(self):
        student = copy.deepcopy(self.base_model)
        ref = copy.deepcopy(self.base_model)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

        valid_ids = self._valid_seq_ids(student.eos_id)
        deg_ids = self._degenerate_seq_ids(student.eos_id)
        L = max(len(valid_ids), len(deg_ids))
        v_pad = valid_ids + [student.pad_id] * (L - len(valid_ids))
        d_pad = deg_ids + [student.pad_id] * (L - len(deg_ids))

        fake_beams = torch.tensor(
            [[v_pad, v_pad], [v_pad, v_pad]], dtype=torch.long
        )
        # Extra view agrees on item 0, disagrees on item 1.
        fake_view_greedy = torch.tensor([v_pad, d_pad], dtype=torch.long)

        with mock.patch.object(
            student, "beam_search", return_value=fake_beams
        ), mock.patch.object(
            student, "greedy_kv", return_value=fake_view_greedy
        ):
            loss, stats = bar_math_unsup_step(
                student,
                ref,
                self.vocab,
                self.imgs,
                kl_beta=0.1,
                min_valid_ratio=0.95,
                max_len=48,
                multi_view_imgs=self.imgs,
                num_candidates=2,
            )

        self.assertEqual(stats["accepted_indices"], [0])
        self.assertAlmostEqual(stats["accepted_frac"], 0.5, places=6)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
