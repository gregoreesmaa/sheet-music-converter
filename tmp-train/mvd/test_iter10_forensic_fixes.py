"""Unit tests for Iter-10 forensic scout fixes (CPU-only).

Covers:
1. MeasureBeamHeaderGuard constant-offset header guard (eval_uni_suite31.py)
2. --max-systems-per-page (default 16) full-page filter (uni_omr.py + train_uni_v2.py)
3. canonicalize_measure_voices voice-major normalization (grammar.py)
4. Beat-conservation cross-voice check & <sys>-loop rejection (rlvr_grpo.py)
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

MVD_DIR = Path(__file__).resolve().parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import eval_uni_suite31 as eval_s31  # noqa: E402
import grammar  # noqa: E402
import rlvr_grpo  # noqa: E402
import torch  # noqa: E402
import train_uni_v2  # noqa: E402
import uni_omr  # noqa: E402
import uni_omr_v2  # noqa: E402
import vir_router_v2  # noqa: E402


def _make_bar_state(tokens):
    st = grammar.PrefixState(strict=False)
    for t in tokens:
        st.step(t)
    return st


class MeasureBeamHeaderGuardTest(unittest.TestCase):
    def test_trips_on_constant_offset_wrong_header(self):
        """essen-altdeu scenario: time-3-2 (6 beats) header on 4/2 (8 beats) bars."""
        guard = eval_s31.MeasureBeamHeaderGuard(const_offset_limit=3)
        bar_toks = [
            "clef-G-2", "key+0", "time-3-2",
            "voice-1", "whole", "G", "4", "whole", "A", "4",
        ]
        st = _make_bar_state(bar_toks)
        meter_err, voice_err = eval_s31._measure_violation_errors(st)
        self.assertAlmostEqual(meter_err, 2.0, places=4)
        self.assertAlmostEqual(voice_err, 0.0, places=4)
        self.assertAlmostEqual(eval_s31._measure_signed_offset(st), 2.0, places=4)

        # First 3 bars allowed; 4th bar with identical +2.0 offset trips the guard.
        self.assertTrue(guard.should_repair(st, meter_err, voice_err))
        self.assertTrue(guard.should_repair(st, meter_err, voice_err))
        self.assertTrue(guard.should_repair(st, meter_err, voice_err))
        self.assertFalse(guard.tripped)
        self.assertFalse(guard.should_repair(st, meter_err, voice_err))
        self.assertTrue(guard.tripped)
        # Subsequent meter-only bars remain blocked.
        self.assertFalse(guard.should_repair(st, meter_err, voice_err))

    def test_cross_voice_error_still_repairs_after_trip(self):
        guard = eval_s31.MeasureBeamHeaderGuard(const_offset_limit=2)
        st_const = _make_bar_state([
            "clef-G-2", "key+0", "time-3-2",
            "voice-1", "whole", "G", "4", "whole", "A", "4",
        ])
        for _ in range(3):
            guard.should_repair(st_const, 2.0, 0.0)
        self.assertTrue(guard.tripped)

        # Genuine cross-voice mismatch (voice-1=4.0, voice-2=3.0) must still repair.
        st_voice = _make_bar_state([
            "clef-G-2", "key+0", "time-3-2",
            "voice-1", "whole", "G", "4",
            "voice-2", "half", "E", "4", "quarter", "F", "4",
        ])
        m_err, v_err = eval_s31._measure_violation_errors(st_voice)
        self.assertGreater(v_err, 0.5)
        self.assertTrue(guard.should_repair(st_voice, m_err, v_err))

    def test_varying_offsets_do_not_trip_and_tsig_change_resets(self):
        guard = eval_s31.MeasureBeamHeaderGuard(const_offset_limit=3)
        # Varying local rhythmic offsets never accumulate to > 3 on any single offset.
        bars = [
            ["clef-G-2", "key+0", "time-4-4", "voice-1", "half", "C", "4", "quarter", "D", "4"],  # -1.0
            ["clef-G-2", "key+0", "time-4-4", "voice-1", "whole", "C", "4", "eighth", "D", "4"],  # +0.5
            ["clef-G-2", "key+0", "time-4-4", "voice-1", "half", "C", "4", "eighth", "D", "4"],   # -1.5
            ["clef-G-2", "key+0", "time-4-4", "voice-1", "whole", "C", "4", "quarter", "D", "4"], # +1.0
        ]
        for b in bars:
            st = _make_bar_state(b)
            m_err, v_err = eval_s31._measure_violation_errors(st)
            self.assertTrue(guard.should_repair(st, m_err, v_err))
        self.assertFalse(guard.tripped)

        # Trip on 4/4, then switch to 3/4 -> guard must reset.
        st_44 = _make_bar_state(["clef-G-2", "key+0", "time-4-4", "voice-1", "half", "C", "4"])
        for _ in range(4):
            guard.should_repair(st_44, 2.0, 0.0)
        self.assertTrue(guard.tripped)

        st_34 = _make_bar_state(["clef-G-2", "key+0", "time-3-4", "voice-1", "half", "C", "4"])
        self.assertTrue(guard.should_repair(st_34, 1.0, 0.0))
        self.assertFalse(guard.tripped)


class MaxSystemsPerPageFilterTest(unittest.TestCase):
    def test_filter_fullpage_pairs_by_systems(self):
        pairs = [
            (Path("p1.png"), [Path(f"s{i}.txt") for i in range(4)], "printed"),
            (Path("p2.png"), [Path(f"s{i}.txt") for i in range(16)], "printed"),
            (Path("p3_vrv.png"), [Path(f"s{i}.txt") for i in range(17)], "printed"),
            (Path("p4_vrv.png"), [Path(f"s{i}.txt") for i in range(99)], "printed"),
        ]
        kept, dropped = uni_omr.filter_fullpage_pairs_by_systems(pairs, max_systems_per_page=16)
        self.assertEqual(dropped, 2)
        self.assertEqual([p[0].name for p in kept], ["p1.png", "p2.png"])

        kept_all, dropped_none = uni_omr.filter_fullpage_pairs_by_systems(pairs, max_systems_per_page=0)
        self.assertEqual(dropped_none, 0)
        self.assertEqual(len(kept_all), 4)

    def test_cli_defaults_in_uni_omr_and_train_uni_v2(self):
        args_v2 = train_uni_v2.build_arg_parser().parse_args([])
        self.assertEqual(args_v2.max_systems_per_page, 16)
        args_v2_custom = train_uni_v2.build_arg_parser().parse_args(["--max-systems-per-page", "12"])
        self.assertEqual(args_v2_custom.max_systems_per_page, 12)


class CanonicalizeMeasureVoicesTest(unittest.TestCase):
    def test_already_canonical_is_verbatim(self):
        toks = [
            "clef-G-2", "key+0", "time-4-4",
            "voice-1", "half", "E", "5", "half", "F", "5",
            "voice-2", "half", "C", "4", "half", "D", "4",
            "bar",
        ]
        self.assertEqual(grammar.canonicalize_measure_voices(toks), toks)

    def test_onset_interleaved_collapses_to_voice_major(self):
        # Mirrors Printed Scout 2's mozart-k155-2 photo trace.
        interleaved = [
            "clef-G-2", "key+0", "time-4-4",
            "dyn-f",
            "voice-1", "quarter", "E", "5", "tie-start",
            "dyn-f",
            "voice-2", "quarter", "C", "4", "a1", "tie-start",
            "voice-1", "16th", "E", "5", "tie-stop",
            "voice-2", "16th", "C", "4", "a1", "tie-stop",
            "bar",
            "<sys>",
            "clef-G-2", "key+0", "time-4-4",
            "voice-2", "whole", "C", "4",
            "voice-1", "whole", "E", "5",
            "bar-light-heavy",
        ]
        expected = [
            "clef-G-2", "key+0", "time-4-4",
            "dyn-f",
            "voice-1", "quarter", "E", "5", "tie-start", "16th", "E", "5", "tie-stop",
            "voice-2", "dyn-f", "quarter", "C", "4", "a1", "tie-start", "16th", "C", "4", "a1", "tie-stop",
            "bar",
            "<sys>",
            "clef-G-2", "key+0", "time-4-4",
            "voice-1", "whole", "E", "5",
            "voice-2", "whole", "C", "4",
            "bar-light-heavy",
        ]
        got = grammar.canonicalize_measure_voices(interleaved)
        self.assertEqual(got, expected)
        # Idempotent on second call.
        self.assertEqual(grammar.canonicalize_measure_voices(got), got)
        # Canonicalized output is 100% PrefixState-valid across <sys>.
        st = grammar.PrefixState(strict=False)
        for t in got:
            if t == "<sys>":
                self.assertTrue(st.can_emit_sys())
                st.reset_system()
            else:
                st.step(t)
        # Canonical TokLCS against voice-major reference is 1.0.
        self.assertAlmostEqual(eval_s31.compute_token_lcs_f1(expected, got), 1.0, places=6)


class RLVRBeatConservationAndSysLoopTest(unittest.TestCase):
    def test_header_only_sys_loop_rejected(self):
        spam = [
            "clef-G-2", "key+0", "time-4-4", "<sys>",
            "clef-G-2", "key+0", "time-4-4", "<sys>",
            "clef-G-2", "key+0", "time-4-4", "voice-1", "whole", "C", "4", "bar-light-heavy",
        ]
        parse_ok, _, _, _ = rlvr_grpo._walk_parse_and_meter(spam)
        self.assertFalse(parse_ok)

    def test_exceeding_max_systems_rejected(self):
        one_sys = ["clef-G-2", "key+0", "time-4-4", "voice-1", "whole", "C", "4", "bar"]
        too_many = []
        for i in range(grammar.MAX_SYSTEMS + 1):
            if i > 0:
                too_many.append("<sys>")
            too_many.extend(one_sys)
        parse_ok, _, _, _ = rlvr_grpo._walk_parse_and_meter(too_many)
        self.assertFalse(parse_ok)

    def test_cross_voice_beat_conservation_without_tsig(self):
        st_bad = _make_bar_state([
            "clef-G-2", "key+0",
            "voice-1", "half", "C", "4",
            "voice-2", "quarter", "E", "4",
        ])
        self.assertIsNone(st_bad.tsig)
        self.assertFalse(rlvr_grpo._check_measure_valid(st_bad))

        st_ok = _make_bar_state([
            "clef-G-2", "key+0",
            "voice-1", "half", "C", "4",
            "voice-2", "half", "E", "4",
        ])
        self.assertTrue(rlvr_grpo._check_measure_valid(st_ok))

    def test_wrong_header_tsig_penalized_by_r_meter(self):
        bar_42 = ["voice-1", "whole", "G", "4", "whole", "A", "4", "bar"]
        good = ["clef-G-2", "key+0", "time-4-2"] + bar_42 * 5
        bad = ["clef-G-2", "key+0", "time-3-2"] + bar_42 * 5
        res_good = rlvr_grpo.compute_omr_rlvr_reward(good, good)
        res_bad = rlvr_grpo.compute_omr_rlvr_reward(bad, good)
        self.assertAlmostEqual(res_good["r_meter"], 1.0, places=6)
        self.assertAlmostEqual(res_bad["r_meter"], 0.0, places=6)
        self.assertGreater(res_good["reward"], res_bad["reward"])


class V2Iter11ParityTest(unittest.TestCase):
    def test_v2_paper_gate_preserves_interior_shadow_and_masks_outer_border(self):
        # Synthetic 128x128 page: 20px black desk border + bright paper + dark shadowed interior patch (luma=0.20)
        img = torch.full((1, 3, 128, 128), 0.85, dtype=torch.float32)
        img[:, :, :20, :] = 0.05
        img[:, :, -20:, :] = 0.05
        img[:, :, :, :20] = 0.05
        img[:, :, :, -20:] = 0.05
        # Dark interior shadow (e.g. aged/shadowed real HW manuscript) surrounded by bright paper
        img[:, :, 48:80, 48:80] = 0.20
        # Dark ink stroke inside the interior shadow
        img[:, :, 60:64, 52:76] = 0.02

        h_p, w_p = 8, 8
        _, gate_v2, eff_v2 = uni_omr_v2.InkTokenRouterV2._compute_ink_and_paper_gate(img, h_p, w_p)
        _, gate_vir, eff_vir = vir_router_v2.compute_ink_and_paper_gate_f32(img, h_p, w_p)

        # Outer border corner MUST be suppressed (gate == 0.05)
        self.assertAlmostEqual(float(gate_v2[0, 0, 0, 0]), 0.05, places=4)
        self.assertAlmostEqual(float(gate_vir[0, 0, 0, 0]), 0.05, places=4)
        # Interior dark shadow MUST be preserved (gate == 1.0, NOT masked out!)
        self.assertAlmostEqual(float(gate_v2[0, 0, 4, 4]), 1.0, places=4)
        self.assertAlmostEqual(float(gate_vir[0, 0, 4, 4]), 1.0, places=4)
        self.assertTrue(bool(eff_v2[0, 0, 4, 4]))
        self.assertAlmostEqual(float(eff_vir[0, 0, 4, 4]), 1.0, places=4)
        # Exact parity between InkTokenRouterV2 and ViR pure-f32 paper_gate
        self.assertTrue(torch.allclose(gate_v2, gate_vir, atol=1e-5))

    def test_v2_max_mem_tokens_defaults_to_10240(self):
        cfg = uni_omr_v2.UniOMRv2Config()
        self.assertEqual(cfg.max_mem_tokens, 10240)
        router = vir_router_v2.ViRTwoTierRouter(d_model=64)
        self.assertEqual(router.max_mem_tokens, 10240)
        args_v2 = train_uni_v2.build_arg_parser().parse_args([])
        self.assertEqual(args_v2.max_mem_tokens, 10240)
        self.assertAlmostEqual(args_v2.label_smoothing, 0.05, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
