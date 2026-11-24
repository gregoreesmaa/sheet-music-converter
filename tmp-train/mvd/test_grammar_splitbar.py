"""Unit tests for octave-transposing clefs (8vb/8va) and split-bar repeat pickups.

Covers:
1. ``clef-G-2-8vb`` and ``clef-G-2-8va`` append-only placement in ``data.VOCAB``
   (preserving all pre-existing IDs 0..175 byte-identically).
2. ``grammar.py`` recognition of ``clef-G-2-8vb`` / ``clef-G-2-8va`` in leading
   headers, multi-staff headers, and mid-stream clef changes (both legacy and
   strict modes).
3. Split-bar repeat pickups / anacrusis allowance in ``grammar.py`` (both
   legacy and strict modes):
   - 1-note pickup bars with ``time-1-32`` (e.g. fiddle-strathspey m7) and
     ``time-1-16`` (e.g. oneills-tune m17),
   - partial bars closed by ``bar-repeat-right`` and opened by
     ``bar-repeat-left``,
   - grace-only pickups and measure-1 anacrusis,
   - real suite31 GT streams (fiddle-strathspey, oneills-tune, luca-gloria)
     when the ``assembly-serialize`` binary is present.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_MVD_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MVD_DIR.parent.parent
sys.path.insert(0, str(_MVD_DIR))

import data  # noqa: E402
import grammar  # noqa: E402


def _assert_stream_allowed(tc: unittest.TestCase, tokens: list[str], *, strict: bool) -> None:
    st = grammar.PrefixState(strict=strict)
    ids = data.encode(tokens)
    for k, (tok, tid) in enumerate(zip(tokens, ids)):
        allow, _ = st.allowed()
        tc.assertIn(
            tid,
            allow,
            f"strict={strict} blocked token[{k}]={tok!r} after {tokens[max(0, k - 6):k]}",
        )
        st.step(tok)
    _, eos_ok = st.allowed()
    tc.assertTrue(eos_ok, f"strict={strict} blocked EOS at end of stream")


class TestOctaveClefVocab(unittest.TestCase):
    def test_append_only_ids(self):
        self.assertEqual(len(data.VOCAB), 178)
        self.assertEqual(data.VOCAB[176], "clef-G-2-8vb")
        self.assertEqual(data.VOCAB[177], "clef-G-2-8va")
        self.assertEqual(data.TOKEN_TO_ID["clef-G-2-8vb"], 176)
        self.assertEqual(data.TOKEN_TO_ID["clef-G-2-8va"], 177)
        self.assertEqual(data.PAD_ID, 178)
        self.assertEqual(data.BOS_ID, 179)
        self.assertEqual(data.EOS_ID, 180)
        # Verify prior IDs 0..175 did not shift.
        self.assertEqual(data.VOCAB[175], "128th")
        self.assertLess(data.TOKEN_TO_ID["clef-G-2"], 133)
        self.assertEqual(len(data.VOCAB), len(set(data.VOCAB)))

    def test_encode_decode_roundtrip(self):
        toks = ["clef-G-2-8vb", "key-1", "time-6-8", "voice-1", "half", "F", "3", "dot", "bar",
                "clef-G-2-8va", "voice-1", "half", "C", "6", "dot", "bar"]
        self.assertEqual(data.decode(data.encode(toks)), toks)


class TestOctaveClefGrammar(unittest.TestCase):
    def test_clefs_set_contains_8vb_8va(self):
        self.assertIn("clef-G-2-8vb", grammar.CLEFS)
        self.assertIn("clef-G-2-8va", grammar.CLEFS)

    def test_leading_and_midstream_8vb_8va(self):
        # Multi-staff header like luca-gloria (Cantus G-2, Contratenor & Tenor G-2-8vb)
        # plus mid-stream switch to G-2-8va.
        stream = (
            "clef-G-2 key+0 time-6-8 "
            "clef-G-2-8vb key-1 time-6-8 "
            "clef-G-2-8vb key-1 time-6-8 "
            "voice-1 half C 5 dot "
            "voice-2 half F 4 dot "
            "voice-3 half F 3 dot bar "
            "clef-G-2-8va voice-1 half C 6 dot bar"
        ).split()
        for strict in (False, True):
            _assert_stream_allowed(self, stream, strict=strict)


class TestSplitBarRepeatPickups(unittest.TestCase):
    def test_fiddle_style_time_1_32_split_pickup(self):
        # Mirrors fiddle-strathspey m6-m8: 4/4 bar -> time-1-32 1-note pickup -> 4/4 bar
        stream = (
            "clef-G-2 key+0 time-4-4 "
            "voice-1 eighth E 5 bar "  # anacrusis m1
            "voice-1 half C 5 quarter G 4 eighth G 4 dot 32nd C 5 32nd B 4 tie-start bar "
            "time-1-32 voice-1 32nd B 4 tie-stop bar "
            "time-4-4 voice-1 quarter A 4 quarter B 4 quarter C 5 quarter D 5 bar"
        ).split()
        for strict in (False, True):
            _assert_stream_allowed(self, stream, strict=strict)

    def test_oneills_style_time_1_16_split_pickup(self):
        # Mirrors oneills-tune m16-m18: 2/4 bar -> time-1-16 1-note pickup -> 2/4 bar
        stream = (
            "clef-G-2 key-2 time-2-4 "
            "voice-1 eighth D 5 eighth G 5 eighth G 5 dot slur-start 16th G 5 bar "
            "time-1-16 voice-1 16th A 5 slur-stop bar "
            "time-2-4 voice-1 quarter B 5 a-1 quarter G 5 bar"
        ).split()
        for strict in (False, True):
            _assert_stream_allowed(self, stream, strict=strict)

    def test_midbar_repeat_right_and_left_partial_bars(self):
        # Partial bar closed by bar-repeat-right, followed immediately by
        # bar-repeat-left and a short pickup bar (without explicit time change).
        stream = (
            "clef-G-2 key+0 time-4-4 "
            "voice-1 half C 5 quarter D 5 eighth E 5 bar-repeat-right "
            "bar-repeat-left voice-1 eighth F 5 bar "
            "voice-1 whole G 5 bar-light-heavy"
        ).split()
        for strict in (False, True):
            _assert_stream_allowed(self, stream, strict=strict)

    def test_grace_pickup_in_short_bar(self):
        stream = (
            "clef-G-2 key+0 time-4-4 "
            "voice-1 whole C 5 bar "
            "time-1-32 voice-1 grace 16th D 5 32nd E 5 bar "
            "time-4-4 voice-1 whole F 5 bar"
        ).split()
        for strict in (False, True):
            _assert_stream_allowed(self, stream, strict=strict)

    def test_suite31_real_streams_when_binary_available(self):
        bin_path = _REPO_ROOT / "converter" / "assembly-serialize" / "target" / "release" / "assembly-serialize.exe"
        gt_dir = _REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "gt"
        if not bin_path.exists() or not gt_dir.exists():
            self.skipTest("assembly-serialize release binary or suite31 GT not present")

        with tempfile.TemporaryDirectory() as td:
            for stem in ("gt-p1-fiddle-strathspey", "gt-p1-oneills-tune", "gt-p1-luca-gloria"):
                xml_path = gt_dir / f"{stem}.musicxml"
                res = subprocess.run(
                    [str(bin_path), "bekern", "--input", str(xml_path),
                     "--out", td, "--stem", stem, "--page", "1"],
                    capture_output=True, text=True, check=True,
                )
                self.assertEqual(res.returncode, 0)
                toks = (Path(td) / f"{stem}-bekern.txt").read_text(encoding="utf-8").split()
                if stem == "gt-p1-luca-gloria":
                    self.assertIn("clef-G-2-8vb", toks)
                for strict in (False, True):
                    _assert_stream_allowed(self, toks, strict=strict)


if __name__ == "__main__":
    unittest.main()
