"""Unit tests for the 3-Tier Closed-Loop Verifier and eval_sys_gate.py wiring.

Run:
    .\\.venv\\Scripts\\python.exe -m unittest tmp-train/mvd/test_verifier_3tier.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data  # noqa: E402
import eval_sys_gate as esg  # noqa: E402
import staff_unet2d as su  # noqa: E402
import verifier_3tier as v3  # noqa: E402


VALID_2BAR = (
    "clef-G-2 key+0 time-4-4 "
    "quarter C 4 quarter D 4 quarter E 4 quarter F 4 bar "
    "half G 4 half A 4 bar"
)

VALID_2BAR_ALT = (
    "clef-G-2 key+0 time-4-4 "
    "quarter C 4 quarter E 4 quarter G 4 quarter C 5 bar "
    "half B 4 half A 4 bar"
)

BAD_BAR1_OF_2 = (
    "clef-G-2 key+0 time-4-4 "
    "quarter C 4 quarter D 4 quarter E 4 quarter F 4 bar "
    "quarter G 4 quarter A 4 quarter B 4 bar "
    "half C 5 half D 5 bar"
)

UNBALANCED_TIE = (
    "clef-G-2 key+0 time-4-4 "
    "quarter C 4 tie-start quarter D 4 quarter E 4 quarter F 4 bar "
    "half G 4 slur-stop half A 4 bar"
)

DEGEN_ALL_RESTS = (
    "clef-G-2 key+0 time-4-4 "
    "rest quarter rest quarter rest quarter rest quarter bar "
    "rest half rest half bar"
)


class TestTier1SymbolicGate(unittest.TestCase):
    def test_valid_stream_scores_one(self):
        res = v3.tier1_symbolic_gate(VALID_2BAR)
        self.assertEqual(res.n_bars, 2)
        self.assertEqual(res.n_bad_bars, 0)
        self.assertEqual(res.bad_bar_indices, [])
        self.assertAlmostEqual(res.valid_bar_ratio, 1.0)
        self.assertFalse(res.is_degenerate)
        self.assertEqual(res.unbalanced_spanners, 0)
        self.assertAlmostEqual(res.score, 1.0)
        self.assertTrue(res.passed)

    def test_catches_bad_bar_math(self):
        res = v3.tier1_symbolic_gate(BAD_BAR1_OF_2)
        self.assertEqual(res.n_bars, 3)
        self.assertEqual(res.n_bad_bars, 1)
        self.assertEqual(res.bad_bar_indices, [1])
        self.assertAlmostEqual(res.valid_bar_ratio, 2.0 / 3.0, places=5)
        self.assertFalse(res.is_degenerate)
        self.assertLess(res.score, 0.9)
        self.assertFalse(res.passed)

    def test_catches_degeneracy(self):
        res = v3.tier1_symbolic_gate(DEGEN_ALL_RESTS)
        self.assertTrue(res.is_degenerate)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.passed)
        is_deg, reason = v3.is_degenerate_bekern(DEGEN_ALL_RESTS)
        self.assertTrue(is_deg)
        self.assertIsNotNone(reason)

    def test_catches_unbalanced_ties_slurs_voltas(self):
        res = v3.tier1_symbolic_gate(UNBALANCED_TIE)
        self.assertEqual(res.unbalanced_spanners, 2)
        self.assertEqual(res.unbalanced_details["tie"], 1)
        self.assertEqual(res.unbalanced_details["slur"], 1)
        self.assertLess(res.score, 1.0)
        self.assertFalse(res.passed)

        volta_stream = VALID_2BAR + " volta-1 quarter C 4 quarter D 4 quarter E 4 quarter F 4 bar"
        vres = v3.tier1_symbolic_gate(volta_stream)
        self.assertEqual(vres.unbalanced_details["volta"], 1)
        self.assertEqual(vres.unbalanced_spanners, 1)


class TestTier2VisualRerankAndPreferences(unittest.TestCase):
    def test_overrules_visually_wrong_beam0_and_emits_preference_pair(self):
        strip = torch.zeros(1, 3, 64, 128)
        candidates = [
            (VALID_2BAR, -0.10),      # beam 0: valid bar-math, visually wrong
            (VALID_2BAR_ALT, -0.12),  # beam 1: valid bar-math, visually right (small LM margin)
        ]

        def fake_visual(_img, toks):
            return 0.15 if toks == VALID_2BAR.split() else 0.92

        t2 = v3.tier2_visual_rerank(strip, candidates, visual_scorer=fake_visual)
        self.assertTrue(t2.tier2_triggered)
        self.assertEqual(t2.best.index, 1)
        self.assertEqual(t2.best.tokens, VALID_2BAR_ALT.split())
        self.assertGreater(t2.margin, 0.05)

        pair = v3.maybe_build_preference_pair("strip-001", t2.ranked, greedy_idx=0, min_margin=0.05)
        self.assertIsNotNone(pair)
        self.assertEqual(pair["strip_id"], "strip-001")
        self.assertEqual(pair["winner_tokens"], VALID_2BAR_ALT.split())
        self.assertEqual(pair["loser_tokens"], VALID_2BAR.split())
        self.assertGreater(pair["winner_score"], pair["loser_score"])
        self.assertAlmostEqual(pair["t1_delta"], 0.0, places=5)
        self.assertGreater(pair["t2_delta"], 0.7)

    def test_skips_visual_on_confident_valid_beam0(self):
        strip = torch.zeros(1, 3, 64, 128)
        calls = []

        def spy_visual(_img, toks):
            calls.append(toks)
            return 0.5

        candidates = [
            (VALID_2BAR, -0.10),
            (VALID_2BAR_ALT, -0.80),  # large LM margin (0.70 >= 0.15)
        ]
        t2 = v3.tier2_visual_rerank(strip, candidates, visual_scorer=spy_visual)
        self.assertFalse(t2.tier2_triggered)
        self.assertEqual(len(calls), 0)
        self.assertEqual(t2.best.index, 0)
        self.assertIsNone(v3.maybe_build_preference_pair("s", t2.ranked))

    def test_fallback_ink_density_when_rdv_unavailable(self):
        # White strip with dark vertical note-like ink columns.
        arr = np.full((64, 128), 255, dtype=np.uint8)
        for x in (16, 36, 56, 76, 96, 112):
            arr[24:40, x : x + 4] = 10
        with mock.patch.object(v3, "_rdv", None):
            t2 = v3.tier2_visual_rerank(
                arr,
                [(BAD_BAR1_OF_2, -0.10), (VALID_2BAR, -0.11)],
            )
        self.assertTrue(t2.tier2_triggered)
        self.assertTrue(t2.used_fallback_visual)
        self.assertGreater(t2.best.t2_visual_score, 0.0)
        self.assertEqual(t2.best.index, 1)


class TestTier3SelectiveRepairAndClosedLoop(unittest.TestCase):
    def test_tier3_skipped_on_high_confidence_valid_strip(self):
        strip = torch.zeros(1, 3, 64, 128)
        decode_calls = []

        def spy_decode(x):
            decode_calls.append(x)
            return [(VALID_2BAR, -0.05)]

        verifier = v3.ClosedLoopVerifier()
        out = verifier.verify_strip(
            "s-clean",
            strip,
            [(VALID_2BAR, -0.10), (VALID_2BAR_ALT, -0.90)],
            decode_fn=spy_decode,
        )
        self.assertEqual(len(decode_calls), 0)
        self.assertFalse(out.t3_result.triggered)
        self.assertFalse(out.needs_review)
        self.assertEqual(out.low_confidence_measures, [])
        tel = verifier.telemetry_dict()
        self.assertEqual(tel["tier1_only"], 1)
        self.assertEqual(tel["tier2_triggered"], 0)
        self.assertEqual(tel["tier3_triggered"], 0)
        self.assertEqual(tel["flagged_for_review"], 0)

    def test_tier3_triggers_and_repairs_broken_strip(self):
        strip = torch.randn(1, 3, 64, 128)
        seen_shapes = []

        def repair_decode(x):
            seen_shapes.append(tuple(x.shape))
            return [(VALID_2BAR, -0.10)]

        def fake_vis(_img, toks):
            return 0.90 if toks == VALID_2BAR.split() else 0.40

        verifier = v3.ClosedLoopVerifier(visual_scorer=fake_vis)
        out = verifier.verify_strip(
            "s-broken",
            strip,
            [(BAD_BAR1_OF_2, -0.10)],
            decode_fn=repair_decode,
        )
        # Two geometric micro-perturbations, both preserving exact (1, 3, 64, 128) shape.
        self.assertEqual(len(seen_shapes), 2)
        self.assertEqual(seen_shapes[0], (1, 3, 64, 128))
        self.assertEqual(seen_shapes[1], (1, 3, 64, 128))
        self.assertTrue(out.t3_result.triggered)
        self.assertTrue(out.t3_result.repaired)
        self.assertEqual(out.tokens, VALID_2BAR.split())
        self.assertFalse(out.needs_review)
        self.assertEqual(out.low_confidence_measures, [])
        tel = verifier.telemetry_dict()
        self.assertEqual(tel["tier3_triggered"], 1)
        self.assertEqual(tel["tier3_repaired"], 1)
        self.assertEqual(tel["flagged_for_review"], 0)

    def test_tier3_flags_exact_bad_measures_when_unrepaired(self):
        strip = torch.randn(1, 3, 64, 128)

        def unchanged_decode(_x):
            return [(BAD_BAR1_OF_2, -0.10)]

        verifier = v3.ClosedLoopVerifier(visual_scorer=lambda _i, _t: 0.5)
        out = verifier.verify_strip(
            "s-unfixable",
            strip,
            [(BAD_BAR1_OF_2, -0.10)],
            decode_fn=unchanged_decode,
        )
        self.assertTrue(out.t3_result.triggered)
        self.assertFalse(out.t3_result.repaired)
        self.assertTrue(out.needs_review)
        self.assertEqual(out.low_confidence_measures, [1])
        tel = verifier.telemetry_dict()
        self.assertEqual(tel["flagged_for_review"], 1)
        self.assertEqual(tel["low_confidence_measures_total"], 1)

    def test_tier3_with_come_tta_restores_ln_weights(self):
        class TinyLNNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(16)
                self.head = nn.Linear(16, 8)

            def forward(self, x):
                # x: [B, C, H, W] -> pool to [B, 16]
                feat = x.mean(dim=(-2, -1))
                if feat.shape[-1] < 16:
                    feat = torch.nn.functional.pad(feat, (0, 16 - feat.shape[-1]))
                return self.head(self.ln(feat[:, :16]))

        net = TinyLNNet()
        w_before = net.ln.weight.detach().clone()
        b_before = net.ln.bias.detach().clone()
        strip = torch.randn(1, 3, 32, 64)

        verifier = v3.ClosedLoopVerifier(
            visual_scorer=lambda _i, _t: 0.8,
            tta_mode="come",
            tta_steps=1,
        )
        out = verifier.verify_strip(
            "s-tta",
            strip,
            [(BAD_BAR1_OF_2, -0.10)],
            decode_fn=lambda _x: [(VALID_2BAR, -0.10)],
            tta_model=net,
        )
        self.assertTrue(out.t3_result.triggered)
        self.assertTrue(torch.equal(net.ln.weight, w_before))
        self.assertTrue(torch.equal(net.ln.bias, b_before))

    def test_pref_jsonl_written_on_overrule(self):
        strip = torch.zeros(1, 3, 64, 128)
        with tempfile.TemporaryDirectory(prefix="v3_pref_") as tmp:
            pref_path = Path(tmp) / "prefs.jsonl"
            verifier = v3.ClosedLoopVerifier(
                visual_scorer=lambda _i, toks: 0.1 if toks == VALID_2BAR.split() else 0.95,
                pref_out_path=pref_path,
            )
            out = verifier.verify_strip(
                "piece-s0",
                strip,
                [(VALID_2BAR, -0.10), (VALID_2BAR_ALT, -0.11)],
            )
            self.assertIsNotNone(out.preference_pair)
            self.assertTrue(pref_path.exists())
            rows = [json.loads(ln) for ln in pref_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["strip_id"], "piece-s0")
            self.assertEqual(rows[0]["winner_tokens"], VALID_2BAR_ALT.split())
            self.assertEqual(rows[0]["loser_tokens"], VALID_2BAR.split())


class TestEvalSysGateCLIFlagsAndWiring(unittest.TestCase):
    def test_cli_flag_defaults_are_off(self):
        ap = esg.build_arg_parser()
        args = ap.parse_args(["--ckpt", "dummy.pt", "--out", "out.json"])
        self.assertFalse(args.verifier_3tier)
        self.assertIsNone(args.verifier_pref_out)
        self.assertEqual(args.page_tta, "none")

    def test_cli_flags_opt_in_parse_cleanly(self):
        ap = esg.build_arg_parser()
        args = ap.parse_args([
            "--ckpt", "dummy.pt",
            "--out", "out.json",
            "--verifier-3tier",
            "--verifier-pref-out", "prefs.jsonl",
            "--page-tta", "come_sar",
        ])
        self.assertTrue(args.verifier_3tier)
        self.assertEqual(args.verifier_pref_out, "prefs.jsonl")
        self.assertEqual(args.page_tta, "come_sar")

    def test_decode_curved_system_with_verifier_3tier(self):
        t = data.TOKEN_TO_ID
        b0_ids = [t[tok] for tok in BAD_BAR1_OF_2.split()]
        b1_ids = [t[tok] for tok in VALID_2BAR.split()]

        class FakeBeamNet:
            def beam_search(self, images, beam=3, strict=True, max_len=564, return_beams=False, **kw):
                if return_beams:
                    return ([[b0_ids, b1_ids]], [[-0.10, -0.12]])
                return [b0_ids]

        page_img = Image.new("RGB", (400, 300), (255, 255, 255))
        x_ctrl = np.linspace(20.0, 380.0, 4, dtype=np.float64)
        y_ctrl = np.full(4, 120.0, dtype=np.float64)
        csys = su.CurvedSystem(
            staves=[
                su.CurvedStaff(
                    ctrl_pts=np.stack([x_ctrl, y_ctrl], axis=-1),
                    sp_ctrl=np.full(4, 10.0, dtype=np.float64),
                    x_start=20.0,
                    x_end=380.0,
                    sp_median=10.0,
                    confidence=0.95,
                )
            ],
            x_start=20.0,
            x_end=380.0,
            polygon=np.zeros((8, 2), dtype=np.float64),
            confidence=0.95,
        )
        verifier = v3.ClosedLoopVerifier(
            visual_scorer=lambda _i, toks: 0.90 if toks == VALID_2BAR.split() else 0.20
        )
        with tempfile.TemporaryDirectory(prefix="v3_curved_") as tmp:
            dec = esg.decode_curved_system(
                FakeBeamNet(),
                page_img,
                csys,
                heat_net=None,
                device="cpu",
                height=256,
                beam=3,
                max_len=576,
                workdir=Path(tmp),
                stem="test-s0",
                extra_chans=0,
                verifier=verifier,
            )
        self.assertEqual(dec["tokens"], VALID_2BAR.split())
        self.assertFalse(dec["verifier_needs_review"])
        self.assertEqual(dec["low_confidence_measures"], [])
        self.assertEqual(verifier.preference_pairs_emitted, 1)


if __name__ == "__main__":
    unittest.main()
