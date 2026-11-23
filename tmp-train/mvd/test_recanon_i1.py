"""Unit tests for tmp-train/mvd/recanon_i1.py."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recanon_i1 as rc  # noqa: E402


class TestRecanonI1(unittest.TestCase):
    def test_interleaved_three_voices_single_measure(self) -> None:
        src = (
            "clef-G-2 key+0 time-4-4 "
            "voice-1 quarter C 5 voice-2 quarter E 4 "
            "voice-1 quarter D 5 voice-3 half C 3 "
            "voice-2 quarter F 4 voice-1 half E 5 "
            "voice-2 half G 4 voice-3 half G 2 bar"
        )
        got, changed = rc.canonicalize_bekern_text(src)
        self.assertTrue(changed)
        expected = (
            "clef-G-2 key+0 time-4-4 "
            "voice-1 quarter C 5 quarter D 5 half E 5 "
            "voice-2 quarter E 4 quarter F 4 half G 4 "
            "voice-3 half C 3 half G 2 bar"
        )
        self.assertEqual(got, expected)
        self.assertTrue(rc.is_stream_canonical_i1(got))
        # Second pass is an exact no-op
        got2, changed2 = rc.canonicalize_bekern_text(got)
        self.assertFalse(changed2)
        self.assertEqual(got2, got)

    def test_already_canonical_exact_noop(self) -> None:
        src = (
            "clef-G-2 key-3 time-4-4 clef-F-4 key-3 time-4-4 "
            "voice-1 half E 4 a-1 half E 5 a-1 "
            "voice-2 half E 3 a-1 half E 4 a-1 bar "
            "voice-1 whole G 4 voice-2 whole G 3 bar-light-heavy\n"
        )
        got, changed = rc.canonicalize_bekern_text(src)
        self.assertFalse(changed)
        self.assertEqual(got, src)
        self.assertTrue(rc.is_stream_canonical_i1(got))

    def test_multi_measure_different_barlines_and_attrs(self) -> None:
        src = (
            "clef-G-2 key+1 time-3-4 "
            "bar-repeat-left dyn-p voice-1 quarter G 4 dyn-p voice-2 quarter B 3 "
            "voice-1 half A 4 voice-2 half C 4 bar-repeat-right "
            "voice-1 quarter B 4 voice-2 quarter D 4 "
            "voice-1 half C 5 voice-2 half E 4 bar-double "
            "voice-1 half D 5 dot voice-2 half F 4 a1 dot bar-light-heavy"
        )
        got, changed = rc.canonicalize_bekern_text(src)
        self.assertTrue(changed)
        expected = (
            "clef-G-2 key+1 time-3-4 "
            "bar-repeat-left dyn-p voice-1 quarter G 4 half A 4 "
            "dyn-p voice-2 quarter B 3 half C 4 bar-repeat-right "
            "voice-1 quarter B 4 half C 5 "
            "voice-2 quarter D 4 half E 4 bar-double "
            "voice-1 half D 5 dot voice-2 half F 4 a1 dot bar-light-heavy"
        )
        self.assertEqual(got, expected)
        self.assertTrue(rc.is_stream_canonical_i1(got))

    def test_tuplet_spans_across_interleaved_voices(self) -> None:
        src = (
            "clef-G-2 key+0 time-4-4 "
            "voice-1 tuplet-start-3-2 eighth C 5 "
            "voice-2 quarter E 4 "
            "voice-1 eighth D 5 eighth E 5 tuplet-stop "
            "voice-2 tuplet-start-3-2 quarter F 4 "
            "voice-1 quarter F 5 "
            "voice-2 quarter G 4 quarter A 4 tuplet-stop "
            "voice-1 half G 5 "
            "voice-2 quarter B 4 bar"
        )
        got, changed = rc.canonicalize_bekern_text(src)
        self.assertTrue(changed)
        expected = (
            "clef-G-2 key+0 time-4-4 "
            "voice-1 tuplet-start-3-2 eighth C 5 eighth D 5 eighth E 5 tuplet-stop "
            "quarter F 5 half G 5 "
            "voice-2 quarter E 4 tuplet-start-3-2 quarter F 4 quarter G 4 quarter A 4 "
            "tuplet-stop quarter B 4 bar"
        )
        self.assertEqual(got, expected)
        self.assertTrue(rc.is_stream_canonical_i1(got))

    def test_pitch_to_midi(self) -> None:
        self.assertEqual(rc.pitch_to_midi("C4"), 60)
        self.assertEqual(rc.pitch_to_midi("C#4"), 61)
        self.assertEqual(rc.pitch_to_midi("Db4"), 61)
        self.assertEqual(rc.pitch_to_midi("Ax4"), 71)
        self.assertEqual(rc.pitch_to_midi("Bbb3"), 57)
        self.assertEqual(rc.pitch_to_midi("C-1"), 0)
        self.assertIsNone(rc.pitch_to_midi(None))

    def test_live_bekern_decode_equivalence_and_hardlink_break(self) -> None:
        self.assertTrue(
            rc.DEFAULT_ASER.is_file(),
            f"assembly-serialize.exe missing at {rc.DEFAULT_ASER}",
        )
        interleaved = (
            "clef-G-2 key-3 time-4-4 clef-F-4 key-3 time-4-4 "
            "dyn-p voice-1 tuplet-start-3-2 eighth E 4 a-1 "
            "dyn-p voice-2 quarter E 3 a-1 "
            "voice-1 eighth F 4 eighth G 4 tuplet-stop "
            "voice-3 half B 2 a-1 "
            "voice-2 quarter G 3 "
            "voice-1 quarter A 4 a-1 "
            "voice-2 half B 3 a-1 "
            "voice-1 half B 4 a-1 "
            "voice-3 half E 3 a-1 bar "
            "voice-1 quarter C 5 voice-2 half E 3 a-1 "
            "voice-1 quarter B 4 a-1 voice-1 half A 4 a-1 "
            "voice-2 half G 3 bar-light-heavy\n"
        )
        with tempfile.TemporaryDirectory(prefix="test_recanon_") as td:
            td_p = Path(td)
            src_link_origin = td_p / "origin.txt"
            target = td_p / "sample-bekern.txt"
            src_link_origin.write_text(interleaved, encoding="utf-8")
            os.link(src_link_origin, target)

            scratch = td_p / "scratch"
            scratch.mkdir()
            status, detail = rc.process_one_file(target, rc.DEFAULT_ASER, scratch)
            self.assertEqual(status, "rewritten", detail)
            self.assertEqual(detail, "")

            # Target is now canonical I1
            new_text = target.read_text(encoding="utf-8")
            self.assertTrue(rc.is_stream_canonical_i1(new_text))
            # Hardlink was broken: origin still holds original pre-I1 content
            self.assertEqual(src_link_origin.read_text(encoding="utf-8"), interleaved)
            self.assertNotEqual(new_text, interleaved)

            # Live bekern-decode equivalence holds
            ok, err = rc.verify_equivalence(
                src_link_origin, target, rc.DEFAULT_ASER, scratch, "check"
            )
            self.assertTrue(ok, err)

            # Re-running is an exact no-op
            status2, detail2 = rc.process_one_file(target, rc.DEFAULT_ASER, scratch)
            self.assertEqual(status2, "noop", detail2)

    def test_page_cut_tie_stop_unmerged_fallback(self) -> None:
        # Page-cut tie-stop on voice-2 at beat 0 with same-pitch tie-start on
        # voice-1 at beat 3: `bekern-decode` tie-merger differs by scan order,
        # but unmerged tie links (`bekern2musicxml`) are 100% identical.
        src = (
            "clef-G-2 key-1 time-2-2 clef-G-2 key-1 time-2-2 "
            "voice-1 half A 4 voice-2 quarter G 4 tie-stop "
            "voice-2 quarter F 4 voice-1 quarter G 4 voice-2 quarter G 4 "
            "voice-1 quarter G 4 tie-start voice-2 quarter B 4 a-1 bar "
            "voice-1 quarter G 4 tie-stop voice-2 quarter G 4 "
            "voice-1 half C 5 voice-2 half A 4 "
            "voice-1 quarter B 4 a-1 voice-2 quarter D 4 bar"
        )
        with tempfile.TemporaryDirectory(prefix="test_recanon_tie_") as td:
            td_p = Path(td)
            target = td_p / "tie-bekern.txt"
            target.write_text(src, encoding="utf-8")
            scratch = td_p / "scratch"
            scratch.mkdir()
            status, detail = rc.process_one_file(target, rc.DEFAULT_ASER, scratch)
            self.assertEqual(status, "rewritten", detail)
            self.assertTrue(rc.is_stream_canonical_i1(target.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
