"""Unit tests for tmp-train/mvd/build_w2w3_strata.py (CPU-only)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

MVD_DIR = Path(__file__).resolve().parent
REPO_ROOT = MVD_DIR.parent.parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import build_hybrid_cache  # noqa: E402
import build_unlabeled_10k  # noqa: E402
import build_w2w3_strata as w2w3  # noqa: E402


class TestFeatureDetectors(unittest.TestCase):
    def test_c_clef_detection(self) -> None:
        self.assertTrue(w2w3.has_c_clef(["clef-C-3", "key+0", "time-4-4", "quarter", "C", "4"]))
        self.assertTrue(w2w3.has_c_clef(["clef-G-2", "clef-C-4", "key-1", "time-3-4"]))
        self.assertFalse(w2w3.has_c_clef(["clef-G-2", "clef-F-4", "key+0", "time-4-4"]))

    def test_mid_bar_clef_detection(self) -> None:
        # Leading multi-staff headers are NOT mid-bar clef changes
        leading_only = [
            "clef-G-2", "key+0", "time-4-4",
            "clef-F-4", "key+0", "time-4-4",
            "voice-1", "quarter", "C", "4", "bar",
        ]
        self.assertFalse(w2w3.has_mid_bar_clef(leading_only))

        # Clef appearing in the body IS a mid-bar / mid-system clef change
        mid_change = [
            "clef-G-2", "key-1", "time-4-4",
            "voice-1", "quarter", "C", "4", "clef-F-4", "quarter", "G", "3", "bar",
        ]
        self.assertTrue(w2w3.has_mid_bar_clef(mid_change))

    def test_irregular_tuplet_detection(self) -> None:
        # Plain 3:2 triplet is NOT an irregular tuplet
        self.assertFalse(
            w2w3.has_irregular_tuplet(
                ["clef-G-2", "tuplet-start-3-2", "eighth", "C", "4", "tuplet-stop"]
            )
        )
        # Non-triplet tuplets (5, 6, 7, 9, 10, etc.) ARE irregular tuplets
        for tok in (
            "tuplet-start-5-4",
            "tuplet-start-6-4",
            "tuplet-start-7-8",
            "tuplet-start-9-8",
            "tuplet-start-10-8",
        ):
            self.assertTrue(w2w3.has_irregular_tuplet(["clef-G-2", tok, "16th", "C", "4", "tuplet-stop"]))

    def test_extreme_ledger_detection(self) -> None:
        # Mid-range notes (F2..G5) are NOT extreme ledger lines
        mid_notes = [
            "clef-G-2", "quarter", "F", "2", "quarter", "C", "4", "quarter", "G", "5", "bar",
        ]
        self.assertEqual(w2w3.count_extreme_ledger_notes(mid_notes), 0)
        self.assertFalse(w2w3.has_extreme_ledger(mid_notes, min_notes=1))

        # <= E2 (E2, D2, C1) and >= A5 (A5, B5, C6) ARE extreme ledger lines
        ext_notes = [
            "clef-F-4",
            "quarter", "E", "2",
            "quarter", "D", "2", "a-1",
            "quarter", "A", "5", "a1",
            "quarter", "C", "6",
            "bar",
        ]
        self.assertEqual(w2w3.count_extreme_ledger_notes(ext_notes), 4)
        self.assertTrue(w2w3.has_extreme_ledger(ext_notes, min_notes=4))
        self.assertFalse(w2w3.has_extreme_ledger(ext_notes, min_notes=5))

    def test_long_512_768_gate(self) -> None:
        self.assertFalse(w2w3.is_long_512_768(["bar"] * 512))
        self.assertTrue(w2w3.is_long_512_768(["bar"] * 513))
        self.assertTrue(w2w3.is_long_512_768(["bar"] * 768))
        self.assertFalse(w2w3.is_long_512_768(["bar"] * 769))

    def test_oov_and_i1_gates(self) -> None:
        good = ["clef-G-2", "key+0", "time-4-4", "voice-1", "quarter", "C", "4", "voice-2", "half", "E", "3", "bar"]
        self.assertEqual(w2w3.find_oov_tokens(good), [])
        self.assertEqual(w2w3.count_i1_bars(good), (1, 0))

        bad_oov = good + ["tuplet-start-99-99"]
        self.assertEqual(w2w3.find_oov_tokens(bad_oov), ["tuplet-start-99-99"])

        bad_i1 = [
            "clef-G-2", "key+0", "time-4-4",
            "voice-1", "quarter", "C", "4",
            "voice-2", "quarter", "E", "3",
            "voice-1", "quarter", "D", "4",
            "bar",
        ]
        self.assertEqual(w2w3.count_i1_bars(bad_i1), (1, 1))

    def test_classify_strata_combined(self) -> None:
        toks = (
            ["clef-C-3", "key+0", "time-4-4", "voice-1"]
            + ["tuplet-start-5-4", "16th", "A", "5", "16th", "B", "5", "16th", "C", "6", "16th", "E", "2", "16th", "D", "2", "tuplet-stop"]
            + ["clef-G-2"]
            + ["quarter", "C", "4"] * 170
            + ["bar"]
        )
        self.assertTrue(512 < len(toks) <= 768)
        flags = w2w3.classify_strata(toks, extreme_ledger_min=4)
        self.assertEqual(
            flags,
            {
                "long_512_768": True,
                "c_clef": True,
                "mid_bar_clef": True,
                "irregular_tuplet": True,
                "extreme_ledger": True,
            },
        )


class TestLeakageExclusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = build_unlabeled_10k.ExclusionPolicy.from_repo(REPO_ROOT)

    def test_gate20_excluded(self) -> None:
        # From gate20/exclusion.json do_not_train
        self.assertTrue(w2w3.is_excluded("ds0072-3-s0", self.policy))
        self.assertTrue(w2w3.is_excluded("ds0091-3-s1", self.policy))
        self.assertTrue(w2w3.is_excluded("ds0005-1-s0", self.policy))

    def test_suite31_and_hw_excluded(self) -> None:
        self.assertTrue(w2w3.is_excluded("repeats-melody-1-s0", self.policy))
        self.assertTrue(w2w3.is_excluded("fiddle-strathspey-1-s0", self.policy))
        self.assertTrue(w2w3.is_excluded("hw-beethoven-op135-1-s0", self.policy))
        self.assertTrue(w2w3.is_excluded("mozart-k464-1-s0", self.policy))

    def test_nc_excluded(self) -> None:
        self.assertTrue(w2w3.is_excluded(r"D:\tmp\corpus10x-NC\foo-1-s0.png", self.policy))
        self.assertTrue(w2w3.is_excluded("cvc-muscima-page1", self.policy))

    def test_permissive_pdmx_allowed(self) -> None:
        self.assertFalse(
            w2w3.is_excluded(
                "Qma3h6ha7NJJ3H4SHZPvVH4mNWxuZwqba8HCXashrYbHhy_Leland-1-s0",
                self.policy,
            )
        )


class TestSymboxStaffRowAndVerify(unittest.TestCase):
    def test_symbox_staff_row_and_verify_roundtrip(self) -> None:
        policy = build_unlabeled_10k.ExclusionPolicy.from_repo(REPO_ROOT)
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            stem = "QmPermissiveTestScore_Leland-1-s0"
            png_path = out_dir / f"{stem}.png"
            Image.new("L", (400, 160), color=255).save(png_path)

            chunk = [{"lines": [40.0, 50.0, 60.0, 70.0, 80.0]}]
            row = w2w3._build_symbox_staff_row(f"{stem}.png", chunk, 0, 0, 400, 160)
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row["S"], 1)
            self.assertEqual(row["sp"], 10.0)
            self.assertEqual(row["source"], "symbox")
            self.assertTrue(w2w3.validate_staff_row_against_png(row, png_path))

            toks = [
                "clef-C-3", "key+0", "time-4-4", "voice-1",
                "tuplet-start-5-4", "16th", "E", "2", "16th", "D", "2",
                "16th", "A", "5", "16th", "B", "5", "16th", "C", "6",
                "tuplet-stop", "clef-G-2", "quarter", "C", "4", "bar",
            ]
            (out_dir / f"{stem}-bekern.txt").write_text(" ".join(toks) + "\n", encoding="utf-8")
            (out_dir / "hybrid_staff.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            summary = w2w3.verify_output_dir(out_dir, policy, extreme_ledger_min=4)
            self.assertEqual(summary["total_strip_pngs"], 1)
            self.assertEqual(summary["leaked_eval"], 0)
            self.assertEqual(summary["nc_count"], 0)
            self.assertEqual(summary["oov_tokens"], 0)
            self.assertEqual(summary["bad_i1_bars"], 0)
            self.assertEqual(summary["bad_s_rows"], 0)
            self.assertEqual(summary["missing_pair"], 0)
            self.assertEqual(summary["strata_counts"]["c_clef"], 1)
            self.assertEqual(summary["strata_counts"]["mid_bar_clef"], 1)
            self.assertEqual(summary["strata_counts"]["irregular_tuplet"], 1)
            self.assertEqual(summary["strata_counts"]["extreme_ledger"], 1)
            self.assertEqual(summary["strata_counts"]["long_512_768"], 0)

    def test_base_score_id(self) -> None:
        self.assertEqual(
            w2w3.base_score_id("QmABC123_Leland-10-s2c1"),
            "QmABC123",
        )
        self.assertEqual(
            w2w3.base_score_id("QmABC123_Emmentaler-3-s0"),
            "QmABC123",
        )


if __name__ == "__main__":
    unittest.main()
