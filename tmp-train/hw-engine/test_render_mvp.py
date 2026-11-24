"""Unit and integration tests for `render_mvp.py` (MuseScore/MusicXML parser, layout, and 3-style MVP)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import render_mvp  # noqa: E402

SUITE31_GT = REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "gt"
CORPUS_DIR = REPO_ROOT / "data-gen" / "score-corpus"

MINIMAL_MSCX = """<?xml version="1.0" encoding="UTF-8"?>
<museScore version="3.01">
  <Score>
    <metaTag name="workTitle">Minimal Synthetic Test</metaTag>
    <Part>
      <Staff id="1"><defaultClef>G</defaultClef></Staff>
      <Staff id="2"><defaultClef>F</defaultClef></Staff>
      <trackName>Piano</trackName>
    </Part>
    <Staff id="1">
      <Measure>
        <voice>
          <KeySig><accidental>1</accidental></KeySig>
          <TimeSig><sigN>4</sigN><sigD>4</sigD></TimeSig>
          <Chord>
            <durationType>quarter</durationType>
            <Note><pitch>60</pitch><tpc>14</tpc></Note>
            <Note><pitch>64</pitch><tpc>18</tpc></Note>
          </Chord>
          <Chord>
            <durationType>eighth</durationType>
            <Note><pitch>66</pitch><tpc>20</tpc><Accidental><subtype>accidentalSharp</subtype></Accidental></Note>
          </Chord>
          <Chord>
            <durationType>eighth</durationType>
            <Note><pitch>67</pitch><tpc>15</tpc></Note>
          </Chord>
          <Rest><durationType>half</durationType></Rest>
        </voice>
      </Measure>
      <Measure>
        <voice>
          <Chord>
            <durationType>half</durationType>
            <dots>1</dots>
            <Note><pitch>72</pitch><tpc>14</tpc></Note>
          </Chord>
          <Rest><durationType>quarter</durationType></Rest>
          <BarLine><subtype>end</subtype></BarLine>
        </voice>
      </Measure>
    </Staff>
    <Staff id="2">
      <Measure>
        <voice>
          <KeySig><accidental>1</accidental></KeySig>
          <TimeSig><sigN>4</sigN><sigD>4</sigD></TimeSig>
          <Chord>
            <durationType>whole</durationType>
            <Note><pitch>48</pitch><tpc>14</tpc></Note>
          </Chord>
        </voice>
      </Measure>
      <Measure>
        <voice>
          <Rest><durationType>measure</durationType></Rest>
          <BarLine><subtype>end</subtype></BarLine>
        </voice>
      </Measure>
    </Staff>
  </Score>
</museScore>
"""


def _engine_modules_available() -> bool:
    return all(
        (THIS_DIR / name).exists()
        for name in ("stroke_engine.py", "glyph_variants.py", "styles.py", "layout_quirks.py")
    )


class TestScoreParser(unittest.TestCase):
    def test_parse_musicxml_melody(self) -> None:
        p = SUITE31_GT / "gt-p1-repeats-melody.musicxml"
        if not p.exists():
            self.skipTest(f"Fixture not found: {p}")
        score = render_mvp.parse_score(p)
        self.assertGreaterEqual(len(score.parts), 1)
        self.assertGreater(score.num_measures, 0)
        first_m = score.parts[0].measures[0]
        self.assertIn(first_m.clef, ("G-2", "F-4", "C-3", "C-4"))
        self.assertGreater(len(first_m.events), 0)

    def test_parse_musicxml_bach_chorale(self) -> None:
        p = SUITE31_GT / "gt-p1-bach-bwv66.6.musicxml"
        if not p.exists():
            self.skipTest(f"Fixture not found: {p}")
        score = render_mvp.parse_score(p)
        self.assertEqual(len(score.parts), 4)
        self.assertGreaterEqual(score.num_measures, 8)
        # Bass part should have F-4 clef
        self.assertEqual(score.parts[-1].measures[0].clef, "F-4")

    def test_parse_synthetic_mscx_and_mscz(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mscx_path = Path(td) / "mini.mscx"
            mscx_path.write_text(MINIMAL_MSCX, encoding="utf-8")

            score = render_mvp.parse_score(mscx_path)
            self.assertEqual(score.title, "Minimal Synthetic Test")
            self.assertEqual(len(score.parts), 2)
            # Both staves belong to the same Piano brace group
            self.assertIsNotNone(score.parts[0].brace_group)
            self.assertEqual(score.parts[0].brace_group, score.parts[1].brace_group)
            self.assertEqual(score.parts[0].measures[0].clef, "G-2")
            self.assertEqual(score.parts[1].measures[0].clef, "F-4")
            self.assertEqual(score.parts[0].measures[0].key_fifths, 1)
            self.assertEqual(score.parts[0].measures[0].time_sig, (4, 4))

            # First event is C4+E4 chord -> steps -2 and 0 in treble clef
            ev0 = score.parts[0].measures[0].events[0]
            self.assertIsInstance(ev0, render_mvp.NoteEvent)
            assert isinstance(ev0, render_mvp.NoteEvent)
            self.assertEqual(ev0.staff_steps, [-2, 0])
            # Eighth notes on beat 1.0 and 1.5 should be beamed together
            ev1 = score.parts[0].measures[0].events[1]
            ev2 = score.parts[0].measures[0].events[2]
            assert isinstance(ev1, render_mvp.NoteEvent) and isinstance(ev2, render_mvp.NoteEvent)
            self.assertIsNotNone(ev1.beam_group)
            self.assertEqual(ev1.beam_group, ev2.beam_group)
            self.assertEqual(ev1.accidentals, ["sharp"])

            # Also verify .mscz zip wrapper
            mscz_path = Path(td) / "mini.mscz"
            with zipfile.ZipFile(mscz_path, "w") as zf:
                zf.writestr("mini.mscx", MINIMAL_MSCX)
            score_z = render_mvp.parse_score(mscz_path)
            self.assertEqual(len(score_z.parts), 2)
            self.assertEqual(score_z.num_measures, 2)

    def test_parse_real_corpus_mscx(self) -> None:
        p = CORPUS_DIR / "sample01.mscx"
        if not p.exists():
            self.skipTest(f"Corpus file not found: {p}")
        score = render_mvp.parse_score(p)
        self.assertGreaterEqual(len(score.parts), 2)
        self.assertGreater(score.num_measures, 5)


class TestLayoutWrapping(unittest.TestCase):
    def test_layout_wraps_measures_and_prerules(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mscx_path = Path(td) / "mini.mscx"
            mscx_path.write_text(MINIMAL_MSCX, encoding="utf-8")
            score = render_mvp.parse_score(mscx_path)
            systems = render_mvp.layout_score_pages(
                score,
                page_w=1650,
                page_h=2200,
                style={"blank_prerule_prob": 1.0, "bar_width_var": 0.15},
            )
            self.assertGreaterEqual(len(systems), 2)
            self.assertFalse(systems[0].is_blank_prerule)
            self.assertEqual(systems[0].measure_start, 0)
            self.assertEqual(systems[0].measure_end, 2)
            self.assertTrue(any(s.is_blank_prerule for s in systems[1:]))


class TestThreeStyleMVPRender(unittest.TestCase):
    @unittest.skipUnless(_engine_modules_available(), "Waiting for Subagents A & B engine modules")
    def test_render_three_distinct_styles(self) -> None:
        score_path = SUITE31_GT / "gt-p1-repeats-melody.musicxml"
        if not score_path.exists():
            self.skipTest(f"Fixture not found: {score_path}")

        target_styles = [
            "beethoven_hurried_autograph",
            "debussy_pencil_sketch",
            "student_ballpoint_blue",
        ]
        arrays: dict[str, np.ndarray] = {}

        with tempfile.TemporaryDirectory() as td:
            for idx, st in enumerate(target_styles):
                out_png = Path(td) / f"{st}.png"
                gt = render_mvp.render_score_handwritten(
                    score_path=score_path,
                    style_name=st,
                    out_png=out_png,
                    seed=42 + idx,
                    page_w=1000,
                    page_h=1300,
                )
                self.assertTrue(out_png.exists())
                gt_path = out_png.with_suffix(".json")
                self.assertTrue(gt_path.exists())
                loaded = json.loads(gt_path.read_text(encoding="utf-8"))
                self.assertEqual(loaded["style"], st)
                self.assertGreater(len(loaded["systems"]), 0)
                st0 = loaded["systems"][0]["staves"][0]
                self.assertIn("ctrl_pts", st0)
                self.assertEqual(len(st0["ctrl_pts"]), 4)
                self.assertGreater(st0["sp"], 0)
                self.assertEqual(gt["style"], st)

                img = Image.open(out_png).convert("RGB")
                self.assertEqual(img.size, (1000, 1300))
                arr = np.asarray(img, dtype=np.float32)
                self.assertGreater(float(arr.mean()), 150.0, f"{st} background too dark (mean={arr.mean():.1f})")
                arrays[st] = arr

        # Pairwise pixel differences must be substantial across the 3 styles
        for i in range(len(target_styles)):
            for j in range(i + 1, len(target_styles)):
                s1, s2 = target_styles[i], target_styles[j]
                mad = float(np.mean(np.abs(arrays[s1] - arrays[s2])))
                self.assertGreater(
                    mad,
                    3.0,
                    f"Styles {s1} and {s2} look too similar (MAD={mad:.2f})",
                )


if __name__ == "__main__":
    unittest.main()
