"""Unit tests for GT coverage extraction, sidecars, vocab stability, and Gate-D gating.

Run with the repo-root .venv python:

    .venv/Scripts/python.exe -m unittest tmp-train/mvd/test_gt_coverage.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import grammar  # noqa: E402
import gt_coverage  # noqa: E402


DYNAMICS_LYRICS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list>
  <score-part id="P1"><part-name>Voice</part-name></score-part>
</part-list>
<part id="P1">
  <measure number="1">
    <attributes>
      <divisions>4</divisions>
      <clef><sign>G</sign><line>2</line></clef>
      <key><fifths>0</fifths></key>
      <time><beats>4</beats><beat-type>4</beat-type></time>
    </attributes>
    <direction><direction-type><rehearsal>A</rehearsal></direction-type></direction>
    <direction><direction-type><words>dolce</words></direction-type></direction>
    <direction><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>96</per-minute></metronome></direction-type></direction>
    <direction><direction-type><dynamics><ppp/></dynamics></direction-type></direction>
    <direction><direction-type><wedge type="crescendo" number="1"/></direction-type></direction>
    <harmony>
      <root><root-step>C</root-step></root>
      <kind text="maj7">major-seventh</kind>
      <bass><bass-step>E</bass-step></bass>
    </harmony>
    <note>
      <pitch><step>C</step><octave>4</octave></pitch>
      <duration>8</duration><voice>1</voice>
      <lyric number="1"><syllabic>begin</syllabic><text>Hel</text></lyric>
      <lyric number="2"><syllabic>single</syllabic><text>La</text><extend/></lyric>
    </note>
    <direction><direction-type><wedge type="continue" number="1"/></direction-type></direction>
    <direction><direction-type><dynamics><f/></dynamics></direction-type></direction>
    <note>
      <pitch><step>D</step><octave>4</octave></pitch>
      <duration>8</duration><voice>1</voice>
      <lyric number="1"><syllabic>end</syllabic><text>lo</text></lyric>
    </note>
  </measure>
  <measure number="2">
    <direction><direction-type><wedge type="stop" number="1"/></direction-type></direction>
    <direction><direction-type><wedge type="diminuendo" number="2"/></direction-type></direction>
    <direction><direction-type><dynamics><rfz/></dynamics></direction-type></direction>
    <direction><direction-type><dynamics><other-dynamics>subito p</other-dynamics></dynamics></direction-type></direction>
    <note>
      <pitch><step>E</step><octave>4</octave></pitch>
      <duration>16</duration><voice>1</voice>
    </note>
  </measure>
</part>
</score-partwise>
"""

DRUM_TAB_XML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
<part-list>
  <score-part id="P1">
    <part-name>Drumset</part-name>
    <score-instrument id="P1-I36"><instrument-name>Bass Drum 1</instrument-name></score-instrument>
    <score-instrument id="P1-I42"><instrument-name>Closed Hi-Hat</instrument-name></score-instrument>
  </score-part>
  <score-part id="P2">
    <part-name>Guitar TAB</part-name>
  </score-part>
</part-list>
<part id="P1">
  <measure number="1">
    <attributes>
      <divisions>4</divisions>
      <clef><sign>percussion</sign><line>2</line></clef>
      <time><beats>4</beats><beat-type>4</beat-type></time>
      <measure-style><slash type="start"/></measure-style>
    </attributes>
    <note>
      <unpitched><display-step>F</display-step><display-octave>4</display-octave></unpitched>
      <duration>4</duration><instrument id="P1-I36"/><voice>1</voice>
    </note>
    <note>
      <chord/>
      <unpitched><display-step>G</display-step><display-octave>5</display-octave></unpitched>
      <duration>4</duration><instrument id="P1-I42"/><voice>1</voice>
      <notehead>x</notehead>
    </note>
  </measure>
</part>
<part id="P2">
  <measure number="1">
    <attributes>
      <divisions>4</divisions>
      <clef><sign>TAB</sign><line>5</line></clef>
      <time><beats>4</beats><beat-type>4</beat-type></time>
      <staff-details>
        <staff-tuning line="1"><tuning-step>E</tuning-step><tuning-octave>2</tuning-octave></staff-tuning>
        <staff-tuning line="6"><tuning-step>E</tuning-step><tuning-octave>4</tuning-octave></staff-tuning>
      </staff-details>
    </attributes>
    <note>
      <pitch><step>E</step><octave>3</octave></pitch>
      <duration>4</duration><voice>1</voice>
      <notations><technical><string>6</string><fret>0</fret></technical></notations>
    </note>
    <note>
      <pitch><step>G</step><octave>3</octave></pitch>
      <duration>4</duration><voice>1</voice>
      <notations><technical><string>6</string><fret>3</fret></technical></notations>
    </note>
  </measure>
</part>
</score-partwise>
"""


class TestVocabBackwardCompatibility(unittest.TestCase):
    """Base VOCAB IDs 0..175 must stay byte-identical for ckpt-22000."""

    def test_base_vocab_size_and_specials_unchanged(self):
        self.assertEqual(len(data.VOCAB), 176)
        self.assertEqual(data.PAD_ID, 176)
        self.assertEqual(data.BOS_ID, 177)
        self.assertEqual(data.EOS_ID, 178)
        self.assertEqual(data.VOCAB[0], "A")
        self.assertEqual(data.VOCAB[175], "128th")

    def test_build_vocab_preserves_prefix_and_appends_opt_in(self):
        v_base, m_base = data.build_vocab()
        self.assertEqual(v_base, list(data.VOCAB))
        self.assertEqual(m_base, data.TOKEN_TO_ID)

        v_all, m_all = data.build_vocab(
            include_ext_dynamics=True,
            include_lyric_markers=True,
            include_percussion_tab=True,
        )
        # Prefix 0..175 is byte-identical.
        self.assertEqual(v_all[:176], list(data.VOCAB))
        for i, tok in enumerate(data.VOCAB):
            self.assertEqual(m_all[tok], i)
        # Opt-in tokens sit strictly at >= 176.
        for tok in ("dyn-ppp", "hairpin-continue", "lyric-begin",
                    "clef-TAB-5", "unpitched", "nh-x", "str-6", "fret-0"):
            self.assertGreaterEqual(m_all[tok], 176)
            self.assertNotIn(tok, data.TOKEN_TO_ID)


class TestDynamicsHairpinsLyricsExtraction(unittest.TestCase):
    """Full dynamics, hairpins, lyrics, and text directions sidecar extraction."""

    def test_extracts_all_dynamics_and_wedges_without_overwriting(self):
        sc = gt_coverage.extract_gt_sidecar(DYNAMICS_LYRICS_XML)
        self.assertEqual(sc["schema_version"], 1)
        self.assertFalse(sc["has_percussion_or_tab"])
        self.assertEqual(sc["excluded_staves"], [])

        marks = [(d["measure"], d["onset"], d["mark"], d["bekern_token"])
                 for d in sc["dynamics"]]
        self.assertEqual(
            marks,
            [
                (1, 0.0, "ppp", None),
                (1, 2.0, "f", "dyn-f"),
                (2, 0.0, "rfz", None),
                (2, 0.0, "subito p", None),
            ],
        )

        # Simultaneous stop + diminuendo at m2 onset 0.0 both survive.
        wedges = [(h["measure"], h["onset"], h["number"], h["wedge_type"], h["bekern_token"])
                  for h in sc["hairpins"]]
        self.assertEqual(
            wedges,
            [
                (1, 0.0, 1, "crescendo", "hairpin-cresc-start"),
                (1, 2.0, 1, "continue", None),
                (2, 0.0, 1, "stop", "hairpin-stop"),
                (2, 0.0, 2, "diminuendo", "hairpin-dim-start"),
            ],
        )

    def test_extracts_aligned_lyrics_and_text_directions(self):
        sc = gt_coverage.extract_gt_sidecar(DYNAMICS_LYRICS_XML)
        self.assertEqual(len(sc["lyrics"]), 3)
        self.assertEqual(
            [(l["measure"], l["onset"], l["verse"], l["syllabic"], l["text"], l["extend"])
             for l in sc["lyrics"]],
            [
                (1, 0.0, "1", "begin", "Hel", False),
                (1, 0.0, "2", "single", "La", True),
                (1, 2.0, "1", "end", "lo", False),
            ],
        )
        tdirs = {(t["kind"], t["text"]) for t in sc["text_directions"]}
        self.assertEqual(
            tdirs,
            {
                ("rehearsal", "A"),
                ("words", "dolce"),
                ("metronome", "quarter=96"),
                ("harmony", "C:maj7/E"),
            },
        )

    def test_measure_slicing_and_determinism(self):
        sc1 = gt_coverage.extract_gt_sidecar(DYNAMICS_LYRICS_XML, measures=(2, 2))
        sc2 = gt_coverage.extract_gt_sidecar(DYNAMICS_LYRICS_XML, measures=(2, 2))
        self.assertEqual(json.dumps(sc1, sort_keys=True), json.dumps(sc2, sort_keys=True))
        self.assertEqual(len(sc1["lyrics"]), 0)
        self.assertEqual(len(sc1["dynamics"]), 2)
        self.assertEqual(len(sc1["hairpins"]), 2)

    def test_audit_reports_bekern_coverage_gaps(self):
        rep = gt_coverage.audit_bekern_coverage(DYNAMICS_LYRICS_XML)
        self.assertEqual(rep["dynamics_in_bekern"], 1)
        self.assertEqual(rep["dynamics_dropped_by_bekern"], 3)
        self.assertEqual(rep["hairpins_in_bekern"], 3)
        self.assertEqual(rep["hairpins_dropped_by_bekern"], 1)
        self.assertEqual(rep["lyrics_count"], 3)
        self.assertFalse(rep["has_percussion_or_tab"])


class TestGateDPercussionTabBoundary(unittest.TestCase):
    """Gate-D opt-in: drum/tab is OFF by default and gated in grammar."""

    def test_default_off_detects_and_excludes_without_emitting(self):
        sc = gt_coverage.extract_gt_sidecar(DRUM_TAB_XML)
        self.assertFalse(sc["include_percussion_tab"])
        self.assertTrue(sc["has_percussion_or_tab"])
        self.assertEqual(sc["excluded_staves"], [0, 1])
        self.assertEqual(sc["percussion"], [])
        self.assertEqual(sc["tab"], [])
        self.assertEqual(sc["slash_spans"], [])
        self.assertEqual(sc["staff_tunings"], [])
        with self.assertRaises(ValueError):
            gt_coverage.percussion_tab_to_tokens(sc)

    def test_opt_in_extracts_percussion_tab_and_walks_grammar(self):
        sc = gt_coverage.extract_gt_sidecar(DRUM_TAB_XML, include_percussion_tab=True)
        self.assertTrue(sc["include_percussion_tab"])
        self.assertTrue(sc["has_percussion_or_tab"])
        self.assertEqual(len(sc["percussion"]), 2)
        self.assertEqual(sc["percussion"][0]["instrument_name"], "Bass Drum 1")
        self.assertEqual(sc["percussion"][1]["notehead"], "x")
        self.assertTrue(sc["percussion"][1]["chord"])
        self.assertEqual(len(sc["tab"]), 2)
        self.assertEqual((sc["tab"][0]["string"], sc["tab"][0]["fret"]), (6, 0))
        self.assertEqual((sc["tab"][1]["string"], sc["tab"][1]["fret"]), (6, 3))
        self.assertEqual(len(sc["slash_spans"]), 1)
        self.assertEqual(len(sc["staff_tunings"]), 1)

        toks = gt_coverage.percussion_tab_to_tokens(sc)
        self.assertIn("clef-TAB-5", toks)
        self.assertIn("unpitched", toks)
        self.assertIn("nh-x", toks)
        self.assertIn("str-6", toks)
        self.assertIn("fret-3", toks)

        # Default PrefixState (Gate A-C) MUST reject Gate-D tokens.
        st_default = grammar.PrefixState(strict=False)
        with self.assertRaises(ValueError):
            st_default.step("clef-TAB-5")
        with self.assertRaises(ValueError):
            st_default.step("unpitched")

        # Opt-in PrefixState (Gate D) accepts the entire emitted stream.
        _, ext_map = data.build_vocab(include_percussion_tab=True)
        st_opt = grammar.PrefixState(
            strict=False, allow_percussion_tab=True, vocab_map=ext_map
        )
        for tok in toks:
            allow, _ = st_opt.allowed()
            self.assertIn(ext_map[tok], allow, f"Gate-D grammar blocked {tok!r}")
            st_opt.step(tok)
        _, eos_ok = st_opt.allowed()
        self.assertTrue(eos_ok)


class TestCorpusAndCli(unittest.TestCase):
    """Verify real corpus fixture ds0001.musicxml and CLI end-to-end."""

    def test_real_corpus_ds0001_sidecar_and_loader(self):
        repo_root = Path(__file__).resolve().parents[2]
        ds0001 = repo_root / "data-gen" / "score-corpus" / "big500-renders" / "ds0001.musicxml"
        if not ds0001.exists():
            self.skipTest("ds0001.musicxml fixture not present")
        xml = ds0001.read_text(encoding="utf-8")
        sc = gt_coverage.extract_gt_sidecar(xml)
        self.assertFalse(sc["has_percussion_or_tab"])
        self.assertGreater(len(sc["lyrics"]), 20)
        self.assertGreater(len(sc["dynamics"]), 0)
        self.assertGreater(len(sc["hairpins"]), 0)

    def test_cli_writes_sidecar_and_loads_via_data_py(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            xml_p = tmp / "sample.musicxml"
            out_off = tmp / "sample-off.json"
            out_on = tmp / "sample-on.json"
            xml_p.write_text(DRUM_TAB_XML, encoding="utf-8")

            rc_off = gt_coverage.main(["--input", str(xml_p), "--out", str(out_off)])
            self.assertEqual(rc_off, 0)
            sc_off = data.load_sidecar(out_off)
            self.assertFalse(sc_off["include_percussion_tab"])
            self.assertEqual(sc_off["percussion"], [])
            self.assertEqual(sc_off["tab"], [])

            rc_on = gt_coverage.main([
                "--input", str(xml_p),
                "--out", str(out_on),
                "--include-percussion-tab",
            ])
            self.assertEqual(rc_on, 0)
            sc_on = data.load_sidecar(out_on)
            self.assertTrue(sc_on["include_percussion_tab"])
            self.assertEqual(len(sc_on["percussion"]), 2)
            self.assertEqual(len(sc_on["tab"]), 2)


if __name__ == "__main__":
    unittest.main()
