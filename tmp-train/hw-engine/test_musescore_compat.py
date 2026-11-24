"""Comprehensive unit & integration tests for MuseScore-compatible parser & handwritten renderer."""

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

import musescore_compat as mc  # noqa: E402
import render_mvp  # noqa: E402

CORPUS_DIR = REPO_ROOT / "data-gen" / "score-corpus"
SUITE31_GT = CORPUS_DIR / "suite31" / "gt"


RICH_MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <work><work-title>Rich Compatibility Test</work-title></work>
  <part-list>
    <score-part id="P1"><part-name>Voice</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>12</divisions>
        <key><fifths>2</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line><clef-octave-change>-1</clef-octave-change></clef>
      </attributes>
      <barline location="left"><repeat direction="forward"/></barline>
      <direction placement="above">
        <direction-type><words>Allegro</words></direction-type>
      </direction>
      <direction placement="below">
        <direction-type><dynamics><mf/></dynamics></direction-type>
      </direction>
      <direction placement="below">
        <direction-type><wedge type="crescendo" number="1"/></direction-type>
      </direction>
      <!-- Voice 1: Grace note + triplet + tied note + double sharp + courtesy accidental + lyrics + articulations -->
      <note>
        <grace slash="yes"/>
        <pitch><step>B</step><octave>3</octave></pitch>
        <voice>1</voice>
        <type>eighth</type>
      </note>
      <note>
        <pitch><step>C</step><alter>2</alter><octave>4</octave></pitch>
        <duration>4</duration>
        <voice>1</voice>
        <type>eighth</type>
        <accidental>double-sharp</accidental>
        <time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>
        <notations>
          <tuplet type="start" number="1"/>
          <slur type="start" number="1"/>
          <articulations><staccato/><accent/></articulations>
        </notations>
        <lyric number="1"><syllabic>begin</syllabic><text>Hel</text></lyric>
      </note>
      <note>
        <pitch><step>D</step><alter>-2</alter><octave>4</octave></pitch>
        <duration>4</duration>
        <voice>1</voice>
        <type>eighth</type>
        <accidental parentheses="yes">flat-flat</accidental>
        <time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>
        <notations><articulations><tenuto/></articulations></notations>
        <lyric number="1"><syllabic>end</syllabic><text>lo</text></lyric>
      </note>
      <note>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>4</duration>
        <voice>1</voice>
        <type>eighth</type>
        <time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>
        <notations>
          <tuplet type="stop" number="1"/>
          <slur type="stop" number="1"/>
          <ornaments><trill-mark/></ornaments>
        </notations>
      </note>
      <!-- Mid-measure clef & key change -->
      <attributes>
        <key><fifths>-1</fifths></key>
        <clef><sign>F</sign><line>4</line></clef>
      </attributes>
      <note>
        <pitch><step>F</step><octave>3</octave></pitch>
        <duration>24</duration>
        <tie type="start"/>
        <voice>1</voice>
        <type>half</type>
        <notations><tied type="start"/><fermata/></notations>
      </note>
      <note>
        <pitch><step>G</step><octave>3</octave></pitch>
        <duration>12</duration>
        <voice>1</voice>
        <type>quarter</type>
      </note>
      <!-- Voice 2: Polyphonic second voice causing collision nudge at beat 0 -->
      <backup><duration>48</duration></backup>
      <note>
        <pitch><step>B</step><octave>3</octave></pitch>
        <duration>24</duration>
        <voice>2</voice>
        <type>half</type>
      </note>
      <note>
        <pitch><step>A</step><octave>3</octave></pitch>
        <duration>24</duration>
        <voice>2</voice>
        <type>half</type>
      </note>
      <direction placement="below">
        <direction-type><wedge type="stop" number="1"/></direction-type>
      </direction>
      <barline location="right">
        <ending number="1" type="start">1.</ending>
      </barline>
    </measure>
    <measure number="2">
      <attributes>
        <time><beats>3</beats><beat-type>4</beat-type></time>
      </attributes>
      <note>
        <pitch><step>F</step><octave>3</octave></pitch>
        <duration>36</duration>
        <tie type="stop"/>
        <voice>1</voice>
        <type>half</type>
        <dot/>
        <notations><tied type="stop"/></notations>
      </note>
      <barline location="right">
        <bar-style>light-heavy</bar-style>
        <repeat direction="backward"/>
        <ending number="1" type="stop"/>
      </barline>
    </measure>
    <measure number="3">
      <attributes>
        <measure-style><multiple-rest>4</multiple-rest></measure-style>
      </attributes>
      <note>
        <rest measure="yes"/>
        <duration>36</duration>
        <voice>1</voice>
      </note>
      <barline location="right"><bar-style>light-heavy</bar-style></barline>
    </measure>
  </part>
</score-partwise>
"""


RICH_MSCX = """<?xml version="1.0" encoding="UTF-8"?>
<museScore version="4.00">
  <Score>
    <metaTag name="workTitle">Rich MuseScore 4 Test</metaTag>
    <Part>
      <Staff id="1"><defaultClef>G8vb</defaultClef></Staff>
      <trackName>Tenor</trackName>
    </Part>
    <Staff id="1">
      <Measure>
        <startRepeat/>
        <voice>
          <KeySig><concertKey>1</concertKey></KeySig>
          <TimeSig><sigN>4</sigN><sigD>4</sigD></TimeSig>
          <Tempo><text>Andante <sym>metNoteQuarterUp</sym> = 88</text></Tempo>
          <Dynamic><subtype>p</subtype></Dynamic>
          <Spanner type="HairPin">
            <HairPin><subtype>0</subtype></HairPin>
            <next><location><measures>0</measures><fractions>1/2</fractions></location></next>
          </Spanner>
          <Tuplet><normalNotes>2</normalNotes><actualNotes>3</actualNotes></Tuplet>
          <Spanner type="Slur"><Slur/><next><location><fractions>1/4</fractions></location></next></Spanner>
          <Chord>
            <durationType>eighth</durationType>
            <Articulation><subtype>articStaccatoAbove</subtype></Articulation>
            <Lyrics><no>0</no><syllabic>begin</syllabic><text>La</text></Lyrics>
            <Note>
              <pitch>62</pitch><tpc>26</tpc>
              <Accidental><subtype>accidentalDoubleSharp</subtype><bracket>1</bracket></Accidental>
            </Note>
          </Chord>
          <Chord>
            <durationType>eighth</durationType>
            <Articulation><subtype>articAccentAbove</subtype></Articulation>
            <Lyrics><no>0</no><syllabic>end</syllabic><text>la</text></Lyrics>
            <Note><pitch>64</pitch><tpc>18</tpc></Note>
          </Chord>
          <Spanner type="Slur"><prev><location><fractions>-1/4</fractions></location></prev></Spanner>
          <Chord>
            <durationType>eighth</durationType>
            <Note><pitch>65</pitch><tpc>13</tpc></Note>
          </Chord>
          <endTuplet/>
          <Spanner type="HairPin"><prev><location><fractions>-1/2</fractions></location></prev></Spanner>
          <Fermata/>
          <Chord>
            <durationType>half</durationType>
            <Note>
              <pitch>67</pitch><tpc>15</tpc>
              <Spanner type="Tie"><Tie/><next><location><measures>1</measures></location></next></Spanner>
            </Note>
          </Chord>
          <Chord>
            <acciaccatura/>
            <durationType>eighth</durationType>
            <Note><pitch>69</pitch><tpc>17</tpc></Note>
          </Chord>
          <Chord>
            <durationType>quarter</durationType>
            <Note><pitch>67</pitch><tpc>15</tpc></Note>
          </Chord>
        </voice>
        <voice>
          <Chord>
            <durationType>half</durationType>
            <Note><pitch>60</pitch><tpc>14</tpc></Note>
          </Chord>
          <Chord>
            <durationType>half</durationType>
            <Note><pitch>59</pitch><tpc>19</tpc></Note>
          </Chord>
        </voice>
      </Measure>
      <Measure>
        <endRepeat/>
        <voice>
          <Spanner type="Volta">
            <Volta><beginText>1.</beginText></Volta>
            <next><location><measures>1</measures></location></next>
          </Spanner>
          <Chord>
            <durationType>whole</durationType>
            <Note>
              <pitch>67</pitch><tpc>15</tpc>
              <Spanner type="Tie"><prev><location><measures>-1</measures></location></prev></Spanner>
            </Note>
          </Chord>
        </voice>
      </Measure>
      <Measure>
        <multiMeasureRest>4</multiMeasureRest>
        <voice>
          <Spanner type="Volta"><prev><location><measures>-1</measures></location></prev></Spanner>
          <Rest><durationType>measure</durationType></Rest>
          <BarLine><subtype>end</subtype></BarLine>
        </voice>
      </Measure>
    </Staff>
  </Score>
</museScore>
"""


class TestMuseScoreCompatParser(unittest.TestCase):
    def test_rich_musicxml_features(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            xml_path = Path(td) / "rich.musicxml"
            xml_path.write_text(RICH_MUSICXML, encoding="utf-8")
            score = mc.parse_score(xml_path)

            self.assertEqual(score.title, "Rich Compatibility Test")
            self.assertEqual(len(score.parts), 1)
            part = score.parts[0]
            self.assertEqual(len(part.measures), 3)

            m0 = part.measures[0]
            # 8vb clef parsed accurately
            self.assertEqual(m0.clef, "F-4")  # mid-measure changed active clef to F-4
            self.assertEqual(m0.left_barline, "repeat_left")
            self.assertTrue(any(c.kind == "clef" and c.value == "F-4" for c in m0.mid_changes))
            self.assertTrue(any(c.kind == "key" and c.value == -1 for c in m0.mid_changes))

            # Polyphonic 2-voice opposing stems + collision nudge
            self.assertIn(1, m0.voices_present)
            self.assertIn(2, m0.voices_present)
            v1_notes = [e for e in m0.events if isinstance(e, mc.NoteEvent) and e.voice == 1 and not e.is_grace]
            v2_notes = [e for e in m0.events if isinstance(e, mc.NoteEvent) and e.voice == 2]
            self.assertTrue(all(e.stem_dir == 1 for e in v1_notes))
            self.assertTrue(all(e.stem_dir == -1 for e in v2_notes))
            # Beat 0 has C4 (voice 1) and B3 (voice 2) -> adjacent second -> x_nudge_sp > 0
            self.assertGreater(v2_notes[0].x_nudge_sp, 0.0)

            # Grace note with slash
            grace_notes = [e for e in m0.events if isinstance(e, mc.NoteEvent) and e.is_grace]
            self.assertEqual(len(grace_notes), 1)
            self.assertTrue(grace_notes[0].grace_slash)

            # Double sharp + courtesy double flat
            self.assertEqual(v1_notes[0].accidentals, ["double_sharp"])
            self.assertEqual(v1_notes[1].accidentals, ["double_flat"])
            self.assertEqual(v1_notes[1].courtesy_accidentals, [True])

            # Tuplet (triplet 3:2)
            self.assertEqual(v1_notes[0].tuplet_actual, 3)
            self.assertEqual(v1_notes[0].tuplet_normal, 2)
            self.assertEqual(v1_notes[0].tuplet_pos, "start")
            self.assertEqual(v1_notes[2].tuplet_pos, "stop")

            # Articulations, ornaments, fermata
            self.assertIn("staccato", v1_notes[0].articulations)
            self.assertIn("accent", v1_notes[0].articulations)
            self.assertIn("tenuto", v1_notes[1].articulations)
            self.assertIn("trill", v1_notes[2].articulations)
            self.assertIn("fermata", v1_notes[3].articulations)

            # Lyrics
            self.assertEqual(v1_notes[0].lyrics, [(1, "Hel-")])
            self.assertEqual(v1_notes[1].lyrics, [(1, "lo")])

            # Dynamics, tempo, hairpins
            self.assertEqual([d.text for d in m0.dynamics], ["mf"])
            self.assertEqual([t.text for t in m0.tempos], ["Allegro"])
            self.assertEqual(len(part.hairpins), 1)
            self.assertEqual(part.hairpins[0].kind, "crescendo")

            # Slurs & cross-measure ties resolved
            self.assertEqual(len(part.slurs), 1)
            self.assertEqual(len(part.ties), 1)
            self.assertEqual(part.ties[0].start_measure, 0)
            self.assertEqual(part.ties[0].end_measure, 1)

            # Mid-piece time change, volta, and multi-bar rest
            m1 = part.measures[1]
            self.assertTrue(m1.time_change)
            self.assertEqual(m1.time_sig, (3, 4))
            self.assertIsNotNone(m1.volta)
            self.assertEqual(m1.volta.text, "1.")  # type: ignore[union-attr]
            self.assertEqual(part.measures[2].multi_bar_rest, 4)

    def test_rich_mscx_and_mscz_features(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mscx_path = Path(td) / "rich.mscx"
            mscx_path.write_text(RICH_MSCX, encoding="utf-8")
            score = mc.parse_score(mscx_path)

            part = score.parts[0]
            m0 = part.measures[0]
            self.assertEqual(m0.clef, "G-2-8vb")
            self.assertEqual(m0.left_barline, "repeat_left")
            self.assertEqual(m0.voices_present, [1, 2])
            self.assertEqual([d.text for d in m0.dynamics], ["p"])
            self.assertTrue(any("Andante" in t.text for t in m0.tempos))
            self.assertEqual(len(part.hairpins), 1)
            self.assertEqual(len(part.slurs), 1)
            self.assertEqual(len(part.ties), 1)

            v1_notes = [e for e in m0.events if isinstance(e, mc.NoteEvent) and e.voice == 1 and not e.is_grace]
            self.assertEqual(v1_notes[0].accidentals, ["double_sharp"])
            self.assertEqual(v1_notes[0].courtesy_accidentals, [True])
            self.assertEqual(v1_notes[0].tuplet_actual, 3)
            self.assertEqual(v1_notes[0].lyrics, [(1, "La-")])
            self.assertIn("fermata", v1_notes[3].articulations)

            grace = [e for e in m0.events if isinstance(e, mc.NoteEvent) and e.is_grace]
            self.assertEqual(len(grace), 1)
            self.assertTrue(grace[0].grace_slash)

            self.assertIsNotNone(part.measures[1].volta)
            self.assertEqual(part.measures[2].multi_bar_rest, 4)

            # Verify .mscz zip container roundtrip
            mscz_path = Path(td) / "rich.mscz"
            with zipfile.ZipFile(mscz_path, "w") as zf:
                zf.writestr("rich.mscx", RICH_MSCX)
            score_z = mc.parse_score(mscz_path)
            self.assertEqual(score_z.num_measures, 3)
            self.assertEqual(len(score_z.parts[0].ties), 1)


class TestEndToEndCorpusAndTrainExport(unittest.TestCase):
    def _assert_valid_render(self, out_png: Path, gt: dict) -> None:
        self.assertTrue(out_png.exists())
        self.assertTrue(out_png.with_suffix(".json").exists())
        self.assertGreater(len(gt["systems"]), 0)
        img = Image.open(out_png).convert("RGB")
        arr = np.asarray(img, dtype=np.float32)
        # Paper should be light, with visible dark ink strokes
        self.assertGreater(float(arr.mean()), 150.0)
        self.assertLess(float(arr.min()), 140.0)

    def test_render_rich_synthetic_musicxml_and_mscx(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            xml_p = Path(td) / "rich.musicxml"
            xml_p.write_text(RICH_MUSICXML, encoding="utf-8")
            out_xml = Path(td) / "rich_xml.png"
            gt_xml = render_mvp.render_score_handwritten(
                xml_p, "fountain_pen_calligraphy", out_xml, seed=11, page_w=1200, page_h=1000
            )
            self._assert_valid_render(out_xml, gt_xml)

            mscx_p = Path(td) / "rich.mscx"
            mscx_p.write_text(RICH_MSCX, encoding="utf-8")
            out_mscx = Path(td) / "rich_mscx.png"
            gt_mscx = render_mvp.render_score_handwritten(
                mscx_p, "baroque_quill_neat", out_mscx, seed=12, page_w=1200, page_h=1000
            )
            self._assert_valid_render(out_mscx, gt_mscx)

    def test_e2e_sample01_mscx_with_train_export(self) -> None:
        p = CORPUS_DIR / "sample01.mscx"
        if not p.exists():
            self.skipTest(f"Corpus file missing: {p}")
        with tempfile.TemporaryDirectory() as td:
            out_png = Path(td) / "sample01.png"
            train_dir = Path(td) / "train_out"
            gt = render_mvp.render_score_handwritten(
                p,
                "schubert_dense_lieder",
                out_png,
                seed=101,
                page_w=1200,
                page_h=1600,
                export_train_dir=train_dir,
            )
            self._assert_valid_render(out_png, gt)
            self.assertIn("train_exports", gt)
            self.assertGreater(len(gt["train_exports"]), 0)
            exp0 = gt["train_exports"][0]
            self.assertTrue(Path(exp0["image"]).exists())
            if render_mvp._find_assembly_serialize_bin() is not None:
                self.assertTrue(exp0["bekern_ok"], f"bekern failed: {exp0.get('bekern_error')}")
                self.assertTrue(Path(exp0["bekern"]).exists())
                txt = Path(exp0["bekern"]).read_text(encoding="utf-8")
                self.assertGreater(len(txt.strip()), 0)

    def test_e2e_bach_chorale_and_chopin_mazurka(self) -> None:
        targets = [
            (SUITE31_GT / "gt-p1-bach-bwv66.6.musicxml", "bach_thick_iron_gall"),
            (SUITE31_GT / "gt-p1-chopin-mazurka06-2.musicxml", "chopin_delicate_salon"),
        ]
        with tempfile.TemporaryDirectory() as td:
            for idx, (score_p, style_name) in enumerate(targets):
                if not score_p.exists():
                    self.skipTest(f"Fixture missing: {score_p}")
                out_png = Path(td) / f"{score_p.stem}.png"
                train_dir = Path(td) / f"train_{idx}"
                gt = render_mvp.render_score_handwritten(
                    score_p,
                    style_name,
                    out_png,
                    seed=200 + idx,
                    page_w=1200,
                    page_h=1600,
                    export_train_dir=train_dir,
                )
                self._assert_valid_render(out_png, gt)
                self.assertGreater(len(gt["train_exports"]), 0)
                exp0 = gt["train_exports"][0]
                self.assertTrue(Path(exp0["image"]).exists())
                if render_mvp._find_assembly_serialize_bin() is not None:
                    self.assertTrue(exp0["bekern_ok"], f"bekern failed on {score_p.name}: {exp0.get('bekern_error')}")
                    self.assertGreater(len(Path(exp0["bekern"]).read_text(encoding="utf-8").strip()), 0)


if __name__ == "__main__":
    unittest.main()
