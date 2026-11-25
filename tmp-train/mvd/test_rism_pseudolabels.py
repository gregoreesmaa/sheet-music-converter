#!/usr/bin/env python3
"""CPU-only unit tests for Multi-View + FSA-Gated RISM Pseudo-Label Generator."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import eval_sys_gate as esg  # noqa: E402
import generate_rism_pseudolabels as rism  # noqa: E402
import uni_omr  # noqa: E402


def _make_valid_system(pitch_seq: tuple[str, ...] = ("C", "D", "E", "F")) -> list[str]:
    toks = ["clef-G-2", "key+0", "time-4-4"]
    for p in pitch_seq:
        toks.extend(["quarter", p, "4"])
    toks.append("bar")
    for p in reversed(pitch_seq):
        toks.extend(["quarter", p, "4"])
    toks.append("bar")
    return toks


def _make_view_pred(
    systems: list[list[str]],
    decode_ok: bool = True,
    mtp_acc: float = 0.88,
    truncated: bool = False,
) -> rism.ViewPrediction:
    raw: list[str] = []
    for i, s in enumerate(systems):
        if i > 0:
            raw.append(uni_omr.SYS_TOKEN)
        raw.extend(s)
    stitched: list[str] = []
    for s in systems:
        stitched.extend(s)
    return rism.ViewPrediction(
        raw_tokens=raw,
        stitched_tokens=stitched,
        systems=systems,
        n_sys=len(systems),
        decode_ok=decode_ok,
        mtp_acc=mtp_acc,
        truncated=truncated,
    )


class TestRismPseudoLabelsCPU(unittest.TestCase):
    def test_two_view_agreement_accepts_high_confidence_page(self) -> None:
        """High-agreement, DecOK, matching n_sys, high MTP page is accepted."""
        s1 = _make_valid_system(("C", "D", "E", "F"))
        s2 = _make_valid_system(("G", "A", "B", "C"))
        v1 = _make_view_pred([s1, s2], decode_ok=True, mtp_acc=0.86)
        # Mild 1-token perturbation on View 2 -> TokLCS still >> 0.85
        s2_v2 = list(s2)
        s2_v2[4] = "B"
        v2 = _make_view_pred([s1, s2_v2], decode_ok=True, mtp_acc=0.82)

        cfg = rism.GateConfig(min_view_agree=0.85, min_mtp_acc=0.75, min_tokens=24, max_tokens=1950)
        rec = rism.evaluate_two_views("rism_001.jpg", v1, v2, cfg)

        self.assertEqual(rec["status"], "accepted")
        self.assertEqual(rec["reject_reasons"], [])
        self.assertGreaterEqual(rec["view_tok_lcs"], 0.85)
        self.assertAlmostEqual(rec["mean_mtp_acc"], 0.84, places=5)
        self.assertEqual(rec["v1_n_sys"], 2)
        self.assertEqual(rec["v2_n_sys"], 2)
        self.assertEqual(rec["tokens"], v1.stitched_tokens)
        self.assertEqual(rec["systems"], [s1, s2])

    def test_two_view_rejects_low_agreement_nsys_mismatch_low_mtp_and_length(self) -> None:
        """Each strict quality gate rejects bad predictions cleanly."""
        s1 = _make_valid_system(("C", "D", "E", "F"))
        s2 = _make_valid_system(("G", "A", "B", "C"))
        cfg = rism.GateConfig(min_view_agree=0.85, min_mtp_acc=0.75, min_tokens=24, max_tokens=100)

        # 1. Low view agreement
        v1 = _make_view_pred([s1, s2], decode_ok=True, mtp_acc=0.85)
        s_diff = _make_valid_system(("A", "B", "A", "B"))
        s_diff2 = _make_valid_system(("F", "E", "D", "C"))
        v2_bad = _make_view_pred([s_diff, s_diff2], decode_ok=True, mtp_acc=0.85)
        rec_low = rism.evaluate_two_views("low.jpg", v1, v2_bad, cfg)
        self.assertEqual(rec_low["status"], "rejected")
        self.assertTrue(any(r.startswith("low_view_agree") for r in rec_low["reject_reasons"]))

        # 2. n_sys mismatch
        v2_nsys = _make_view_pred([s1], decode_ok=True, mtp_acc=0.85)
        rec_nsys = rism.evaluate_two_views("nsys.jpg", v1, v2_nsys, cfg)
        self.assertEqual(rec_nsys["status"], "rejected")
        self.assertTrue(any(r.startswith("n_sys_mismatch") for r in rec_nsys["reject_reasons"]))

        # 3. Low MTP acceptance
        v1_mtp = _make_view_pred([s1, s2], decode_ok=True, mtp_acc=0.60)
        v2_mtp = _make_view_pred([s1, s2], decode_ok=True, mtp_acc=0.70)
        rec_mtp = rism.evaluate_two_views("mtp.jpg", v1_mtp, v2_mtp, cfg)
        self.assertEqual(rec_mtp["status"], "rejected")
        self.assertTrue(any(r.startswith("low_mtp_acc") for r in rec_mtp["reject_reasons"]))

        # 4. Too short (< min_tokens)
        short_sys = ["clef-G-2", "time-4-4", "whole", "C", "4", "bar"]
        v1_s = _make_view_pred([short_sys], decode_ok=True, mtp_acc=0.90)
        v2_s = _make_view_pred([short_sys], decode_ok=True, mtp_acc=0.90)
        rec_short = rism.evaluate_two_views("short.jpg", v1_s, v2_s, cfg)
        self.assertEqual(rec_short["status"], "rejected")
        self.assertTrue(any(r.startswith("too_short") for r in rec_short["reject_reasons"]))

        # 5. Too long (> max_tokens)
        long_sys = _make_valid_system(("C", "D", "E", "F")) * 5
        v1_l = _make_view_pred([long_sys], decode_ok=True, mtp_acc=0.90)
        v2_l = _make_view_pred([long_sys], decode_ok=True, mtp_acc=0.90)
        rec_long = rism.evaluate_two_views("long.jpg", v1_l, v2_l, cfg)
        self.assertEqual(rec_long["status"], "rejected")
        self.assertTrue(any(r.startswith("too_long") for r in rec_long["reject_reasons"]))

    def test_dec_ok_gate_synthetic_and_real_bekern_decode(self) -> None:
        """DecOK gate rejects failed views; real bekern-decode validates legal vs malformed bekern."""
        s1 = _make_valid_system(("C", "D", "E", "F"))
        s2 = _make_valid_system(("G", "A", "B", "C"))
        v1_ok = _make_view_pred([s1, s2], decode_ok=True, mtp_acc=0.90)
        v2_fail = _make_view_pred([s1, s2], decode_ok=False, mtp_acc=0.90)
        cfg = rism.GateConfig(min_view_agree=0.85, min_mtp_acc=0.75, min_tokens=24)

        rec = rism.evaluate_two_views("dec_fail.jpg", v1_ok, v2_fail, cfg)
        self.assertEqual(rec["status"], "rejected")
        self.assertIn("v2_decode_fail", rec["reject_reasons"])

        # Also exercise the real assembly-serialize bekern-decode binary if present.
        if Path(esg.ASER).is_file():
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                self.assertTrue(rism.check_bekern_decode(s1, td_path, "valid_sys"))
                malformed = ["clef-G-2", "time-4-4", "quarter", "C"]  # missing octave
                self.assertFalse(rism.check_bekern_decode(malformed, td_path, "bad_sys"))

    def test_system_level_salvage_mode_same_and_mismatched_nsys(self) -> None:
        """--salvage-systems extracts matching >=0.90 <sys> strips even when full page fails."""
        good_a = _make_valid_system(("C", "D", "E", "F"))
        good_b = _make_valid_system(("G", "A", "B", "C"))
        bad_1 = ["clef-G-2", "time-4-4", "quarter", "C", "4"] * 6 + ["bar"]
        bad_2 = ["clef-F-4", "time-3-4", "half", "G", "2"] * 6 + ["bar"]

        # Case 1: Same n_sys=3, but system 1 diverges wildly -> full page TokLCS < 0.85,
        # yet systems 0 and 2 agree 1.00 (>= 0.90) and pass sys_decode_fn!
        v1 = _make_view_pred([good_a, bad_1, good_b], decode_ok=True, mtp_acc=0.85)
        v2 = _make_view_pred([good_a, bad_2, good_b], decode_ok=True, mtp_acc=0.83)
        cfg_salv = rism.GateConfig(
            min_view_agree=0.85,
            min_mtp_acc=0.75,
            min_tokens=24,
            salvage_systems=True,
            min_sys_agree=0.90,
            min_sys_tokens=8,
        )
        # Reject system 2 via sys_decode_fn to confirm salvage enforces per-system DecOK too.
        rec = rism.evaluate_two_views(
            "salvage_same.jpg",
            v1,
            v2,
            cfg_salv,
            sys_decode_fn=lambda toks: toks != good_b,
        )
        self.assertEqual(rec["status"], "salvaged")
        self.assertEqual(len(rec["salvaged_systems"]), 1)
        self.assertEqual(rec["salvaged_systems"][0]["sys_idx_v1"], 0)
        self.assertEqual(rec["salvaged_systems"][0]["sys_idx_v2"], 0)
        self.assertAlmostEqual(rec["salvaged_systems"][0]["tok_lcs"], 1.0, places=5)
        self.assertEqual(rec["salvaged_systems"][0]["tokens"], good_a)

        # Case 2: Mismatched n_sys (3 vs 2) -> full page rejected on n_sys_mismatch,
        # salvage greedy-matches (0->0) and (2->1) cleanly!
        v1_m = _make_view_pred([good_a, bad_1, good_b], decode_ok=True, mtp_acc=0.86)
        v2_m = _make_view_pred([good_a, good_b], decode_ok=True, mtp_acc=0.84)
        rec_m = rism.evaluate_two_views(
            "salvage_mismatch.jpg",
            v1_m,
            v2_m,
            cfg_salv,
            sys_decode_fn=lambda _: True,
        )
        self.assertEqual(rec_m["status"], "salvaged")
        self.assertEqual(len(rec_m["salvaged_systems"]), 2)
        self.assertEqual(
            [(s["sys_idx_v1"], s["sys_idx_v2"]) for s in rec_m["salvaged_systems"]],
            [(0, 0), (2, 1)],
        )

        # Case 3: With salvage_systems=False, same failing page is strictly rejected.
        cfg_no_salv = rism.GateConfig(salvage_systems=False)
        rec_no = rism.evaluate_two_views("no_salv.jpg", v1_m, v2_m, cfg_no_salv)
        self.assertEqual(rec_no["status"], "rejected")
        self.assertNotIn("salvaged_systems", rec_no)

    def test_make_view_images_resume_ledger_and_cli_defaults(self) -> None:
        """View 2 scales by 0.93x + contrast shifts; resume ledger skips seen images; CLI parses."""
        im = Image.new("RGB", (2000, 1000), color=(200, 200, 200))
        v1, v2 = rism.make_view_images(im, max_dim=1280, view2_scale=0.93, view2_contrast=0.90)
        self.assertEqual(max(v1.size), 1280)
        # _resize_fullpage_pil snaps dimensions to multiples of 16 (patch size).
        expected_v2_max = (int(round(1280 * 0.93)) // 16) * 16
        self.assertEqual(max(v2.size), expected_v2_max)
        self.assertLess(max(v2.size), max(v1.size))

        # Resume ledger reads both manifest.jsonl and .processed.txt
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            mf = td_p / "manifest.jsonl"
            pl = mf.with_suffix(".jsonl.processed.txt")
            mf.write_text(json.dumps({"image": "img_a.jpg", "status": "accepted"}) + "\n", encoding="utf-8")
            pl.write_text("img_b.jpg\n", encoding="utf-8")
            seen = rism._load_seen_images(mf, pl)
            self.assertEqual(seen, {"img_a.jpg", "img_b.jpg"})

        # CLI defaults check
        ap = rism.build_arg_parser()
        ns = ap.parse_args(["--ckpt", "teacher.pt"])
        self.assertAlmostEqual(ns.min_view_agree, 0.85)
        self.assertAlmostEqual(ns.min_mtp_acc, 0.75)
        self.assertAlmostEqual(ns.view2_scale, 0.93)
        self.assertAlmostEqual(ns.view2_contrast, 0.90)
        self.assertFalse(ns.salvage_systems)
        self.assertAlmostEqual(ns.min_sys_agree, 0.90)


if __name__ == "__main__":
    unittest.main()
