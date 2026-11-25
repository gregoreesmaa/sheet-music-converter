#!/usr/bin/env python3
"""CPU-only unit tests for ``align_hw_work_to_pages.py``."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import align_hw_work_to_pages as ahw  # noqa: E402
import data  # noqa: E402


def _make_distinct_measures() -> List[List[str]]:
    """Build 12 musically distinct 4/4 measures (monophonic treble) for deterministic DP tests."""
    specs = [
        # Page 1: measures 0..3 (sys 0: 0..2, sys 1: 2..4)
        ["voice-1", "quarter", "C", "4", "quarter", "D", "4", "quarter", "E", "4", "quarter", "F", "4", "bar"],
        ["voice-1", "half", "G", "4", "quarter", "A", "4", "quarter", "B", "4", "bar"],
        ["voice-1", "eighth", "C", "5", "eighth", "B", "4", "quarter", "A", "4", "half", "G", "4", "bar"],
        ["voice-1", "quarter", "F", "4", "a1", "quarter", "G", "4", "half", "A", "4", "bar"],
        # Page 2: measures 4..7 (sys 0: 4..6, sys 1: 6..8)
        ["voice-1", "quarter", "D", "5", "quarter", "C", "5", "quarter", "B", "4", "a-1", "quarter", "A", "4", "bar"],
        ["voice-1", "half", "G", "4", "a1", "quarter", "E", "4", "quarter", "F", "4", "bar"],
        ["voice-1", "16th", "E", "5", "16th", "D", "5", "eighth", "C", "5", "quarter", "B", "4", "half", "A", "4", "bar"],
        ["voice-1", "quarter", "G", "5", "quarter", "F", "5", "quarter", "E", "5", "quarter", "D", "5", "bar"],
        # Page 3: measures 8..11 (sys 0: 8..10, sys 1: 10..12)
        ["voice-1", "whole", "C", "4", "bar"],
        ["voice-1", "quarter", "E", "4", "a-1", "quarter", "G", "4", "quarter", "B", "4", "a-1", "quarter", "D", "5", "bar"],
        ["voice-1", "half", "F", "5", "quarter", "E", "5", "a-1", "quarter", "D", "5", "bar"],
        ["voice-1", "quarter", "C", "5", "quarter", "G", "4", "half", "C", "4", "bar-light-heavy"],
    ]
    return specs


def _build_full_work_tokens(measures: List[List[str]]) -> List[str]:
    toks: List[str] = ["clef-G-2", "key+0", "time-4-4"]
    for m in measures:
        toks.extend(m)
    return toks


class TestZeroLeakageExclusion(unittest.TestCase):
    def setUp(self) -> None:
        self.excl = ahw.ExclusionSet.from_files()

    def test_five_sbb_berlin_lieder_and_mozart_k458_rejected(self) -> None:
        must_reject = [
            "hensel-morgengruss-d-b",
            "mendelssohn-herbstlied-d-b",
            "beethoven-das-blumchen-wunderhold-d-b",
            "cornelius-botschaft-d-b",
            "hensel-im-herbste-d-b",
            "hw-hensel-morgengruss-d-b",
            "mozart-k458",
            "mozart-k458-1",
            "hw-mozart-k458",
            "mozart-string-quartet-no-17-k-458-d-b-999999",
        ]
        for wid in must_reject:
            is_ex, reason = self.excl.is_excluded(wid)
            self.assertTrue(is_ex, f"Expected {wid!r} to be strictly excluded (got reason={reason!r})")

    def test_catalog_metadata_exclusion(self) -> None:
        is_ex, _ = self.excl.is_excluded(
            "some-archive-id-1234",
            composer="Mozart, Wolfgang Amadeus",
            title="String Quartet in B-flat major, KV 458",
        )
        self.assertTrue(is_ex)

    def test_gate20_ids_rejected(self) -> None:
        for gid in ("ds0072-3", "ds0072", "sq7070319-1", "sq7070319", "ds0005-1"):
            is_ex, _ = self.excl.is_excluded(gid)
            self.assertTrue(is_ex, f"Expected gate20 id {gid!r} to be excluded")

    def test_non_eval_works_allowed(self) -> None:
        allowed = [
            "schumann-er-ist-gekommen-in-sturm-und-regen-d-b",
            "mozart-string-quartet-no-8-in-f-major-k-168-d-b-1001287549",
            "bach-bwv-194-06-d-b",
            "pl-kozmzk--mzk-m-135_cwiklicz-boleslaw-jan--to-sen-byl-tylko",
        ]
        for wid in allowed:
            is_ex, reason = self.excl.is_excluded(wid)
            self.assertFalse(is_ex, f"Unexpected exclusion for {wid!r}: {reason}")


class TestMonotoneDPAlignment(unittest.TestCase):
    def setUp(self) -> None:
        self.measures = _make_distinct_measures()
        self.full_tokens = _build_full_work_tokens(self.measures)
        self.parsed = ahw.parse_work_bekern(self.full_tokens)

    def test_exact_3page_partition_with_blank_cover_and_noisy_preds(self) -> None:
        # Blank cover page (p0) + 3 music pages covering [0..4), [4..8), [8..12)
        p0_blank: List[str] = []

        # Inject realistic OMR noise: wrong octave on 1 note, dropped staccato/note, extra note
        p1_sys0 = ["clef-G-2", "key+0", "time-4-4"] + list(self.measures[0]) + list(self.measures[1])
        p1_sys0[6] = "5"  # perturb octave of C4 -> C5
        p1_sys1 = ["clef-G-2", "key+0", "time-4-4"] + list(self.measures[2]) + list(self.measures[3])
        p1_pred = p1_sys0 + [ahw.SYS_TOKEN] + p1_sys1

        p2_sys0 = ["clef-G-2", "key+0", "time-4-4"] + list(self.measures[4]) + list(self.measures[5])
        p2_sys1 = ["clef-G-2", "key+0", "time-4-4"] + list(self.measures[6]) + list(self.measures[7])
        p2_sys1 = p2_sys1[:-4] + ["bar"]  # drop last note on page 2
        p2_pred = p2_sys0 + [ahw.SYS_TOKEN] + p2_sys1

        p3_sys0 = ["clef-G-2", "key+0", "time-4-4"] + list(self.measures[8]) + list(self.measures[9])
        p3_sys1 = ["clef-G-2", "key+0", "time-4-4"] + list(self.measures[10]) + list(self.measures[11])
        p3_pred = p3_sys0 + [ahw.SYS_TOKEN] + p3_sys1

        page_paths = ["cover.jpg", "p1.jpg", "p2.jpg", "p3.jpg"]
        aligned = ahw.align_work_to_pages(
            self.parsed, page_paths, [p0_blank, p1_pred, p2_pred, p3_pred]
        )

        self.assertEqual(len(aligned), 4)
        self.assertTrue(aligned[0].skipped)
        self.assertIsNone(aligned[0].measure_span)

        self.assertFalse(aligned[1].skipped)
        self.assertEqual(aligned[1].measure_span, (0, 4))
        self.assertEqual(aligned[1].system_spans, [(0, 2), (2, 4)])
        self.assertGreater(aligned[1].align_score, 0.80)

        self.assertFalse(aligned[2].skipped)
        self.assertEqual(aligned[2].measure_span, (4, 8))
        self.assertEqual(aligned[2].system_spans, [(4, 6), (6, 8)])
        self.assertGreater(aligned[2].align_score, 0.80)

        self.assertFalse(aligned[3].skipped)
        self.assertEqual(aligned[3].measure_span, (8, 12))
        self.assertEqual(aligned[3].system_spans, [(8, 10), (10, 12)])
        self.assertGreater(aligned[3].align_score, 0.90)

    def test_system_break_recovery_uneven_split(self) -> None:
        # Single page covering measures 0..7 split unevenly into 3 systems: [0..2), [2..5), [5..7)
        s0 = ["clef-G-2", "key+0", "time-4-4"] + self.measures[0] + self.measures[1]
        s1 = ["clef-G-2", "key+0", "time-4-4"] + self.measures[2] + self.measures[3] + self.measures[4]
        s2 = ["clef-G-2", "key+0", "time-4-4"] + self.measures[5] + self.measures[6]
        pred = s0 + [ahw.SYS_TOKEN] + s1 + [ahw.SYS_TOKEN] + s2

        aligned = ahw.align_work_to_pages(self.parsed, ["p1.jpg"], [pred])
        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned[0].measure_span, (0, 7))
        self.assertEqual(aligned[0].system_spans, [(0, 2), (2, 5), (5, 7)])

    def test_header_carrying_slicer_tracks_midwork_time_and_key_change(self) -> None:
        toks = [
            "clef-G-2", "key+0", "time-4-4",
            "voice-1", "whole", "C", "4", "bar",
            "key+2", "time-3-4", "voice-1", "half", "D", "4", "quarter", "E", "4", "bar",
            "voice-1", "half", "F", "4", "a1", "quarter", "G", "4", "bar-light-heavy",
        ]
        parsed = ahw.parse_work_bekern(toks)
        self.assertEqual(parsed.n_measures, 3)

        # System starting at measure 2 (0-based) must carry key+2 and time-3-4!
        s2 = parsed.slice_system_tokens(2, 3)
        self.assertEqual(s2[:3], ["clef-G-2", "key+2", "time-3-4"])
        ok, reason = ahw.validate_system_bekern(s2)
        self.assertTrue(ok, f"Expected valid sliced bekern, got {reason}")


class TestConfidenceGatingAndOOV(unittest.TestCase):
    def test_oov_token_rejected(self) -> None:
        bad_sys = ["clef-G-2", "key+0", "time-4-4", "voice-1", "quarter", "C", "4", "FAKE_OOV_TOK", "bar"]
        ok, reason = ahw.validate_system_bekern(bad_sys)
        self.assertFalse(ok)
        self.assertIn("oov_tokens", reason)

        ap = ahw.AlignedPage(
            page_index=0,
            image_path="p1.jpg",
            skipped=False,
            measure_span=(0, 1),
            system_spans=[(0, 1)],
            align_score=0.95,
            bar_ratio=1.0,
            pred_bars=1,
            gt_bars=1,
        )
        ok_gate, gate_reason = ahw.gate_aligned_page(ap, [bad_sys], min_align_score=0.45)
        self.assertFalse(ok_gate)
        self.assertIn("oov_tokens", gate_reason)

    def test_low_confidence_rejected(self) -> None:
        good_sys = ["clef-G-2", "key+0", "time-4-4", "voice-1", "whole", "C", "4", "bar"]
        ap = ahw.AlignedPage(
            page_index=0,
            image_path="p1.jpg",
            skipped=False,
            measure_span=(0, 1),
            system_spans=[(0, 1)],
            align_score=0.31,
            bar_ratio=1.0,
            pred_bars=1,
            gt_bars=1,
        )
        ok_gate, gate_reason = ahw.gate_aligned_page(ap, [good_sys], min_align_score=0.45)
        self.assertFalse(ok_gate)
        self.assertIn("low_align_score", gate_reason)

    def test_end_to_end_pipeline_emits_iter7_schema_and_blocks_eval_work(self) -> None:
        measures = _make_distinct_measures()
        full_tokens = _build_full_work_tokens(measures)

        with tempfile.TemporaryDirectory(prefix="test_align_hw_") as td:
            root = Path(td)
            bk_file = root / "clean-work-bekern.txt"
            bk_file.write_text(" ".join(full_tokens) + "\n", encoding="utf-8")

            p1 = root / "clean_p1.jpg"
            p2 = root / "clean_p2.jpg"
            p1.write_bytes(b"fakejpg1")
            p2.write_bytes(b"fakejpg2")

            works = [
                # Excluded eval work — must be blocked completely
                ahw.WorkRecord(
                    work_id="hensel-morgengruss-d-b",
                    source_corpus="pdm-openscore-hw",
                    page_paths=[str(p1)],
                    bekern_path=bk_file,
                    xml_path=None,
                ),
                # Clean training work — 2 pages covering [0..6) and [6..12)
                ahw.WorkRecord(
                    work_id="schumann-clean-train-work-d-b",
                    source_corpus="pdm-openscore-hw",
                    page_paths=[str(p1), str(p2)],
                    bekern_path=bk_file,
                    xml_path=None,
                ),
            ]

            pred_p1 = (
                ["clef-G-2", "key+0", "time-4-4"]
                + measures[0] + measures[1] + measures[2]
                + [ahw.SYS_TOKEN]
                + ["clef-G-2", "key+0", "time-4-4"]
                + measures[3] + measures[4] + measures[5]
            )
            pred_p2 = (
                ["clef-G-2", "key+0", "time-4-4"]
                + measures[6] + measures[7] + measures[8]
                + [ahw.SYS_TOKEN]
                + ["clef-G-2", "key+0", "time-4-4"]
                + measures[9] + measures[10] + measures[11]
            )
            preds_map = {str(p1): pred_p1, str(p2): pred_p2}

            out_dir = root / "aligned_out"
            excl = ahw.ExclusionSet.from_files()
            stats = ahw.align_corpus_pipeline(
                works, preds_map, out_dir, excl, min_align_score=0.45
            )

            self.assertEqual(stats["works_excluded_leakage"], 1)
            self.assertEqual(stats["works_aligned"], 1)
            self.assertEqual(stats["pages_emitted"], 2)
            self.assertEqual(stats["systems_emitted"], 4)

            manifest_lines = [
                json.loads(line)
                for line in (out_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(manifest_lines), 2)
            for row in manifest_lines:
                self.assertEqual(row["domain"], "real_hw")
                self.assertEqual(len(row["bekerns"]), 2)
                for bpath in row["bekerns"]:
                    toks = Path(bpath).read_text(encoding="utf-8").split()
                    self.assertTrue(all(t in data.TOKEN_TO_ID for t in toks))


if __name__ == "__main__":
    unittest.main()
