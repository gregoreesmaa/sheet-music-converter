"""Unit tests for multi-system & multi-page musical state carryforward.

Covers ``ScoreContext``, ``propagate_across_systems``, and
``stitch_multipage_work`` in ``measure_decode.py`` plus their wiring in
``eval_sys_gate.py``:

1. Page 1 -> Page 2 clef/key/time carryforward when Page 2 omits redundant
   time signature or clef (or all courtesy headers).
2. Genuine mid-piece key/time/clef changes on Page 2 overriding carried state
   (both leading-header and mid-system body changes).
3. Cross-system and cross-page tie carryforward (``tie-start`` at end of
   system/page N -> ``tie-stop`` at start of system/page N+1, both explicit
   and auto-closed when N+1 omitted ``tie-stop``).
4. Multi-staff (2-staff grand staff and 4-staff string quartet) independent
   per-staff clef carryforward and single-staff clef changes.
5. Measure numbering, active voice counts, pickup flag, open slurs, open
   hairpins, open volta/repeat state, and ``init_ctx`` immutability.
6. End-to-end ``assembly-serialize bekern-decode`` verification on stitched
   multi-page streams (when the release binary is present).

Run:
    .\\.venv\\Scripts\\python.exe -m unittest tmp-train/mvd/test_multipage_carryforward.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_sys_gate as esg  # noqa: E402
import measure_decode as mdec  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASER_EXE = (
    REPO_ROOT / 'converter' / 'assembly-serialize' / 'target'
    / 'release' / 'assembly-serialize.exe'
)


class TestHeaderCarryforwardAcrossPages(unittest.TestCase):
    """Page 1 -> Page 2 clef/key/time carryforward when courtesy headers are omitted."""

    def test_page2_omits_time_sig_and_preserves_context(self):
        p1 = [
            'clef-G-2 key+2 time-3-4 quarter D 4 quarter E 4 quarter F 4 a1 bar',
            'clef-G-2 key+2 quarter G 4 half A 4 bar',
        ]
        # Page 2 omits time-3-4 on sys 0 and omits all headers on sys 1
        p2 = [
            'clef-G-2 key+2 quarter B 4 quarter A 4 quarter G 4 bar',
            'half F 4 a1 quarter D 4 bar',
        ]
        stitched, ctx = mdec.stitch_multipage_work([p1, p2])
        toks = stitched.split()

        # Opening header appears once; unchanged courtesy headers are stripped
        self.assertEqual(toks[:3], ['clef-G-2', 'key+2', 'time-3-4'])
        self.assertEqual(toks.count('clef-G-2'), 1)
        self.assertEqual(toks.count('key+2'), 1)
        self.assertEqual(toks.count('time-3-4'), 1)

        # Final context preserves carried clef, key, time, measure count, voices
        self.assertEqual(ctx.clef, {0: 'clef-G-2'})
        self.assertEqual(ctx.key_sig, {0: 'key+2'})
        self.assertEqual(ctx.time_sig, {0: 'time-3-4'})
        self.assertEqual(ctx.measure_index, 4)
        self.assertEqual(ctx.active_voices, 1)
        self.assertFalse(ctx.in_pickup)

    def test_page2_omits_clef_and_standalone_emit_header_fills_it(self):
        p1 = ['clef-F-4 key-3 time-4-4 whole C 3 bar']
        p2 = ['key-3 half E 3 a-1 half G 3 bar']

        _, ctx1 = mdec.propagate_across_systems(p1)
        self.assertEqual(ctx1.clef, {0: 'clef-F-4'})
        self.assertEqual(ctx1.key_sig, {0: 'key-3'})
        self.assertEqual(ctx1.time_sig, {0: 'time-4-4'})

        # Continuation mode (default): no duplicate headers emitted
        p2_cont, ctx2 = mdec.propagate_across_systems(p2, init_ctx=ctx1)
        self.assertEqual(p2_cont.split(), ['half', 'E', '3', 'a-1', 'half', 'G', '3', 'bar'])
        self.assertEqual(ctx2.clef, {0: 'clef-F-4'})
        self.assertEqual(ctx2.key_sig, {0: 'key-3'})
        self.assertEqual(ctx2.time_sig, {0: 'time-4-4'})
        self.assertEqual(ctx2.measure_index, 2)

        # Standalone page mode (emit_header=True): fills omitted clef + time
        p2_standalone, _ = mdec.propagate_across_systems(
            p2, init_ctx=ctx1, emit_header=True)
        self.assertEqual(
            p2_standalone.split()[:3], ['clef-F-4', 'key-3', 'time-4-4'])

        # Caller's ctx1 was NOT mutated
        self.assertEqual(ctx1.measure_index, 1)


class TestMidPieceHeaderChanges(unittest.TestCase):
    """Genuine mid-piece key/time/clef changes on Page 2 override carried state."""

    def test_page2_key_time_clef_changes_emitted_and_tracked(self):
        p1 = [
            'clef-G-2 key+0 time-4-4 whole C 4 bar',
            'clef-G-2 key+0 whole D 4 bar',
        ]
        # Page 2 switches clef to F-4, key to -2, time to 6-8
        p2 = [
            'clef-F-4 key-2 time-6-8 quarter B 3 a-1 eighth A 3 quarter G 3 eighth F 3 bar',
            # Next system repeats the new courtesy headers -> must NOT duplicate
            'clef-F-4 key-2 time-6-8 quarter E 3 a-1 eighth D 3 quarter C 3 eighth B 2 a-1 bar',
        ]
        stitched, ctx = mdec.stitch_multipage_work([p1, p2])
        toks = stitched.split()

        # Opening header from p1
        self.assertEqual(toks[:3], ['clef-G-2', 'key+0', 'time-4-4'])
        # Mid-piece change emitted exactly once at p2 boundary
        self.assertEqual(toks.count('clef-F-4'), 1)
        self.assertEqual(toks.count('key-2'), 1)
        self.assertEqual(toks.count('time-6-8'), 1)
        # Order: after p1's second bar, before p2's first note
        second_bar_idx = [i for i, t in enumerate(toks) if t == 'bar'][1]
        self.assertEqual(
            toks[second_bar_idx + 1:second_bar_idx + 4],
            ['clef-F-4', 'key-2', 'time-6-8'],
        )
        # Context updated to new state
        self.assertEqual(ctx.clef, {0: 'clef-F-4'})
        self.assertEqual(ctx.key_sig, {0: 'key-2'})
        self.assertEqual(ctx.time_sig, {0: 'time-6-8'})
        self.assertEqual(ctx.measure_index, 4)

    def test_mid_system_body_key_change_carried_to_page2(self):
        p1 = [
            'clef-G-2 key+0 time-4-4 whole C 4 bar key+3 whole A 4 bar',
        ]
        # Page 2 courtesy header matches the mid-system key+3 -> no duplicate
        p2 = [
            'clef-G-2 key+3 whole B 4 bar',
        ]
        stitched, ctx = mdec.stitch_multipage_work([p1, p2])
        toks = stitched.split()
        self.assertEqual(toks.count('key+3'), 1)
        self.assertEqual(ctx.key_sig, {0: 'key+3'})
        self.assertEqual(ctx.measure_index, 3)


class TestCrossSystemAndCrossPageTies(unittest.TestCase):
    """Cross-system and cross-page tie carryforward and clean closure."""

    def test_explicit_tie_across_page_break_closes_cleanly(self):
        p1 = ['clef-G-2 key+0 time-4-4 half C 4 half G 4 tie-start bar']
        p2 = ['clef-G-2 key+0 half G 4 tie-stop half E 4 bar']

        p1_str, ctx1 = mdec.propagate_across_systems(p1)
        self.assertEqual(ctx1.open_ties, {(0, 1, 'G4'): 1})

        p2_str, ctx2 = mdec.propagate_across_systems(p2, init_ctx=ctx1)
        self.assertEqual(ctx2.open_ties, {})
        # No duplicate tie-stop injected
        self.assertEqual(p2_str.split().count('tie-stop'), 1)

        full, full_ctx = mdec.stitch_multipage_work([p1, p2])
        self.assertEqual(full_ctx.open_ties, {})
        self.assertEqual(full, p1_str + ' ' + p2_str)

    def test_unclosed_tie_at_end_of_page1_auto_closes_on_matching_pitch(self):
        # Page 1 ends with tie-start on F#4; Page 2 starts with F#4 but OMR
        # omitted tie-stop across the page break.
        p1 = ['clef-G-2 key+1 time-4-4 half D 4 half F 4 a1 tie-start bar']
        p2 = ['clef-G-2 key+1 half F 4 a1 half G 4 bar']

        stitched, ctx = mdec.stitch_multipage_work([p1, p2])
        toks = stitched.split()
        self.assertEqual(ctx.open_ties, {})
        # tie-stop injected onto the matching F#4 at start of page 2
        self.assertIn('tie-stop', toks)
        idx_stop = toks.index('tie-stop')
        self.assertEqual(toks[idx_stop - 4:idx_stop], ['half', 'F', '4', 'a1'])

    def test_cross_system_chord_tie_auto_closes_all_tied_pitches(self):
        sys0 = [
            'clef-G-2', 'key+0', 'time-4-4',
            'whole', 'C', '4', 'tie-start',
            'chord', 'whole', 'E', '4', 'tie-start',
            'bar',
        ]
        sys1 = [
            'clef-G-2', 'key+0',
            'half', 'C', '4',
            'chord', 'half', 'E', '4',
            'half', 'G', '4',
            'bar',
        ]
        stitched, ctx = mdec.propagate_across_systems([sys0, sys1])
        toks = stitched.split()
        self.assertEqual(ctx.open_ties, {})
        self.assertEqual(toks.count('tie-stop'), 2)

    def test_non_matching_pitch_does_not_receive_spurious_tie_stop(self):
        p1 = ['clef-G-2 key+0 time-4-4 whole C 4 tie-start bar']
        p2 = ['clef-G-2 key+0 whole D 4 bar']
        stitched, ctx = mdec.stitch_multipage_work([p1, p2])
        self.assertNotIn('tie-stop', stitched.split())
        # Original unclosed tie remains recorded rather than silently corrupting D4
        self.assertEqual(ctx.open_ties, {(0, 1, 'C4'): 1})

    def test_multipage_stitched_stream_decodes_in_assembly_serialize(self):
        if not ASER_EXE.exists():
            self.skipTest('release binary missing: %s' % ASER_EXE)
        p1 = ['clef-G-2 key+1 time-4-4 half G 4 half A 4 tie-start bar']
        p2 = ['clef-G-2 key+1 half A 4 half B 4 bar']
        stitched, ctx = mdec.stitch_multipage_work([p1, p2])
        self.assertEqual(ctx.open_ties, {})
        with tempfile.TemporaryDirectory(prefix='t_mp_aser_') as tmp:
            txt = Path(tmp) / 'work.txt'
            out = Path(tmp) / 'work.json'
            txt.write_text(stitched + '\n', encoding='utf-8')
            p = esg.run_cli([ASER_EXE, 'bekern-decode', '--input', str(txt),
                             '--out', str(out)])
            self.assertEqual(p.returncode, 0, p.stderr)
            doc = json.loads(out.read_text(encoding='utf-8'))
            notes = doc['parts'][0]['notes']
            # G4 (dur 2), merged A4 (dur 2 + 2 = 4), B4 (dur 2) -> 3 merged notes
            self.assertEqual(len(notes), 3)
            self.assertEqual(notes[1]['pitch'], 'A4')
            self.assertAlmostEqual(notes[1]['dur'], 4.0)


class TestMultiStaffIndependentClefs(unittest.TestCase):
    """Grand staff (2 staves) and string quartet (4 staves) per-staff carryforward."""

    def test_grand_staff_piano_independent_clefs_and_lh_switch(self):
        p1 = [
            (
                'clef-G-2 key-1 time-4-4 clef-F-4 key-1 time-4-4 '
                'voice-1 whole F 4 voice-2 whole F 2 bar'
            ),
            # Courtesy headers omit time-4-4
            (
                'clef-G-2 key-1 clef-F-4 key-1 '
                'voice-1 whole G 4 voice-2 whole G 2 bar'
            ),
        ]
        # Page 2: LH (staff 1) switches to treble clef-G-2; RH stays clef-G-2
        p2 = [
            (
                'clef-G-2 key-1 clef-G-2 key-1 '
                'voice-1 whole A 4 voice-2 whole C 4 bar'
            ),
            # Next system omits headers entirely -> both staves stay G-2
            'voice-1 whole B 4 a-1 voice-2 whole D 4 bar',
        ]
        stitched, ctx = mdec.stitch_multipage_work([p1, p2])
        toks = stitched.split()
        self.assertEqual(ctx.clef, {0: 'clef-G-2', 1: 'clef-G-2'})
        self.assertEqual(ctx.key_sig, {0: 'key-1', 1: 'key-1'})
        self.assertEqual(ctx.time_sig, {0: 'time-4-4', 1: 'time-4-4'})
        self.assertEqual(ctx.active_voices, 2)
        self.assertEqual(ctx.measure_index, 4)
        # Staff 1's clef switch to clef-G-2 was emitted at page 2 start
        bar_positions = [i for i, t in enumerate(toks) if t == 'bar']
        self.assertEqual(toks[bar_positions[1] + 1], 'clef-G-2')

    def test_four_staff_string_quartet_independent_clefs_and_cello_tenor(self):
        p1 = [
            (
                'clef-G-2 key+2 time-3-4 '
                'clef-G-2 key+2 time-3-4 '
                'clef-C-3 key+2 time-3-4 '
                'clef-F-4 key+2 time-3-4 '
                'voice-1 half D 5 quarter E 5 '
                'voice-2 half A 4 quarter B 4 '
                'voice-3 half F 4 a1 quarter G 4 '
                'voice-4 half D 3 quarter A 2 tie-start bar'
            ),
        ]
        # Page 2: Cello (staff 3) changes from bass (F-4) to tenor (C-4);
        # viola stays alto (C-3); violins stay treble (G-2); time-3-4 omitted;
        # cello's A2 tie closes cleanly on voice-4.
        p2 = [
            (
                'clef-G-2 key+2 '
                'clef-G-2 key+2 '
                'clef-C-3 key+2 '
                'clef-C-4 key+2 '
                'voice-1 half F 5 a1 quarter D 5 '
                'voice-2 half A 4 quarter A 4 '
                'voice-3 half D 4 quarter F 4 a1 '
                'voice-4 half A 2 quarter D 3 bar'
            ),
        ]
        stitched, ctx = mdec.stitch_multipage_work([p1, p2])
        toks = stitched.split()

        self.assertEqual(ctx.clef, {
            0: 'clef-G-2',
            1: 'clef-G-2',
            2: 'clef-C-3',
            3: 'clef-C-4',
        })
        self.assertEqual(ctx.key_sig, {0: 'key+2', 1: 'key+2', 2: 'key+2', 3: 'key+2'})
        self.assertEqual(ctx.time_sig, {0: 'time-3-4', 1: 'time-3-4', 2: 'time-3-4', 3: 'time-3-4'})
        self.assertEqual(ctx.active_voices, 4)
        self.assertEqual(ctx.measure_index, 2)
        self.assertEqual(ctx.open_ties, {})

        # Only cello's clef-C-4 change is emitted between bar 0 and page 2 body
        bar0 = toks.index('bar')
        self.assertEqual(toks[bar0 + 1], 'clef-C-4')
        self.assertEqual(toks[bar0 + 2], 'voice-1')
        # Cello's A2 auto-closed with tie-stop
        self.assertIn('tie-stop', toks)


class TestSpansAndEvalSysGateWiring(unittest.TestCase):
    """Slurs, hairpins, volta/repeats, pickups, and eval_sys_gate helpers."""

    def test_slurs_hairpins_volta_and_pickup_tracking(self):
        p1 = [
            (
                'clef-G-2 key+0 time-4-4 bar-repeat-left '
                'hairpin-cresc-start quarter C 4 slur-start '
                'quarter D 4 quarter E 4 quarter F 4 bar'
            ),
        ]
        _, ctx1 = mdec.propagate_across_systems(p1)
        self.assertEqual(ctx1.open_slurs, {(0, 1): 1})
        self.assertEqual(ctx1.open_hairpins, {1: 'cresc'})
        self.assertEqual(ctx1.open_volta, 'repeat-left')
        self.assertEqual(ctx1.measure_index, 1)

        p2 = [
            (
                'clef-G-2 key+0 hairpin-stop '
                'half G 4 slur-stop half C 5 bar-repeat-right'
            ),
        ]
        _, ctx2 = mdec.propagate_across_systems(p2, init_ctx=ctx1)
        self.assertEqual(ctx2.open_slurs, {})
        self.assertEqual(ctx2.open_hairpins, {})
        self.assertIsNone(ctx2.open_volta)
        self.assertEqual(ctx2.measure_index, 2)

    def test_incomplete_trailing_measure_sets_in_pickup(self):
        _, ctx = mdec.propagate_across_systems(
            ['clef-G-2 key+0 time-4-4 quarter G 4'])
        self.assertTrue(ctx.in_pickup)
        self.assertEqual(ctx.measure_index, 1)

    def test_eval_sys_gate_stitch_multipage_toks_canonicalizes_tails(self):
        p1 = [['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'A', '4',
               'stacc', 'a-1', 'dot', 'tie-start', 'bar']]
        p2 = [['clef-G-2', 'key+0', 'quarter', 'A', '4', 'a-1', 'bar']]
        toks, ctx = esg.stitch_multipage_toks([p1, p2])
        self.assertEqual(ctx.open_ties, {})
        # Alter + dot canonicalized before stacc/tie-start
        self.assertEqual(
            toks[:10],
            ['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'A', '4',
             'a-1', 'dot', 'stacc', 'tie-start'],
        )
        self.assertIn('tie-stop', toks)


if __name__ == '__main__':
    unittest.main()
