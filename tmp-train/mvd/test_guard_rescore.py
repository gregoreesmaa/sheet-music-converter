"""Tests for guard_rescore (voice-cap, rest-guard, attachment-spray grounding).

TDD RED-first: these tests define the contract for guard_rescore.py,
guard_rescore_fliptest.py and guard_rescore_probe.py before they exist.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grammar


def _walks_loose(toks):
    st = grammar.PrefixState(strict=False)
    try:
        for t in toks:
            st.step(t)
    except ValueError:
        return False
    return True


HDR3 = ['clef-G-2', 'key+0', 'time-4-4'] * 3


class VoiceCapTest(unittest.TestCase):
    def test_cap_from_header_clefs(self):
        import guard_rescore as gr
        toks = (HDR3 + ['voice-1', 'quarter', 'C', '4', 'bar',
                        'voice-4', 'quarter', 'D', '4', 'bar'])
        out, rep = gr.voice_cap_filter(toks)
        self.assertEqual(rep['cap'], 3)
        self.assertNotIn('voice-4', out)
        # pitched content of the over-cap run is preserved (merged)
        self.assertIn('D', out)
        self.assertTrue(_walks_loose(out))

    def test_within_cap_untouched(self):
        import guard_rescore as gr
        toks = HDR3 + ['voice-1', 'quarter', 'C', '4',
                       'voice-2', 'quarter', 'D', '4',
                       'voice-3', 'quarter', 'E', '4', 'bar']
        out, rep = gr.voice_cap_filter(toks)
        self.assertEqual(out, toks)
        self.assertEqual(rep['removed_markers'], 0)

    def test_explicit_cap_override(self):
        import guard_rescore as gr
        toks = HDR3 + ['voice-1', 'quarter', 'C', '4',
                       'voice-2', 'quarter', 'D', '4', 'bar']
        out, rep = gr.voice_cap_filter(toks, cap=1)
        self.assertEqual(rep['cap'], 1)
        self.assertNotIn('voice-2', out)

    def test_header_without_clefs_fails_open(self):
        import guard_rescore as gr
        toks = ['quarter', 'C', '4', 'voice-4', 'quarter', 'D', '4']
        out, rep = gr.voice_cap_filter(toks)
        self.assertEqual(out, toks)
        self.assertIsNone(rep['cap'])


class RestGuardTest(unittest.TestCase):
    def test_rest_only_run_removed(self):
        import guard_rescore as gr
        toks = (HDR3 + ['voice-1', 'quarter', 'C', '4',
                        'voice-2', 'rest', 'quarter', 'bar'])
        out, rep = gr.rest_guard_filter(toks)
        self.assertEqual(rep['removed_rests'], 1)
        self.assertNotIn('rest', out)
        self.assertTrue(_walks_loose(out))

    def test_rest_with_notes_kept(self):
        import guard_rescore as gr
        toks = (HDR3 + ['voice-1', 'quarter', 'C', '4',
                        'voice-2', 'quarter', 'D', '4',
                        'rest', 'eighth', 'bar'])
        out, rep = gr.rest_guard_filter(toks)
        self.assertEqual(out, toks)
        self.assertEqual(rep['removed_rests'], 0)

    def test_bar_rest_kept(self):
        import guard_rescore as gr
        toks = HDR3 + ['voice-1', 'quarter', 'C', '4',
                       'voice-2', 'bar-rest', 'bar']
        out, rep = gr.rest_guard_filter(toks)
        self.assertEqual(out, toks)

    def test_dotted_rest_removed_whole(self):
        import guard_rescore as gr
        toks = (HDR3 + ['voice-1', 'quarter', 'C', '4',
                        'voice-2', 'rest', 'quarter', 'dot', 'bar'])
        out, rep = gr.rest_guard_filter(toks)
        self.assertEqual(rep['removed_rests'], 1)
        self.assertNotIn('dot', out)
        self.assertTrue(_walks_loose(out))


class SprayCapTest(unittest.TestCase):
    def test_second_artic_dropped(self):
        import guard_rescore as gr
        toks = HDR3 + ['voice-1', 'quarter', 'C', '4',
                       'stacc', 'accent', 'bar']
        out, rep = gr.spray_cap_filter(toks)
        self.assertEqual(rep['removed_artic'], 1)
        self.assertEqual(out.count('stacc') + out.count('accent'), 1)
        self.assertTrue(_walks_loose(out))

    def test_degenerate_slur_dropped(self):
        import guard_rescore as gr
        toks = (HDR3 + ['voice-1', 'quarter', 'C', '4', 'slur-start',
                        'slur-stop', 'bar'])
        out, rep = gr.spray_cap_filter(toks)
        self.assertEqual(rep['removed_slur_pairs'], 1)
        self.assertNotIn('slur-start', out)
        self.assertNotIn('slur-stop', out)
        self.assertTrue(_walks_loose(out))

    def test_covering_slur_kept(self):
        import guard_rescore as gr
        toks = (HDR3 + ['voice-1', 'quarter', 'C', '4', 'slur-start',
                        'quarter', 'D', '4', 'quarter', 'E', '4',
                        'slur-stop', 'bar'])
        out, rep = gr.spray_cap_filter(toks)
        self.assertEqual(out, toks)
        self.assertEqual(rep['removed_slur_pairs'], 0)

    def test_double_open_slur_dropped(self):
        import guard_rescore as gr
        toks = (HDR3 + ['voice-1', 'quarter', 'C', '4', 'slur-start',
                        'quarter', 'D', '4', 'slur-start',
                        'quarter', 'E', '4', 'slur-stop', 'bar'])
        out, rep = gr.spray_cap_filter(toks)
        # second start dropped; one pair still covering >= 2 notes
        self.assertEqual(out.count('slur-start'), 1)
        self.assertEqual(rep['removed_slur_double_open'], 1)
        self.assertTrue(_walks_loose(out))

    def test_slur_stop_never_dropped(self):
        import guard_rescore as gr
        # covering span (3 notes): artic_cap=0 strips nothing
        # structural, stop must survive
        toks = (HDR3 + ['voice-1', 'quarter', 'C', '4', 'slur-start',
                        'quarter', 'D', '4', 'quarter', 'E', '4',
                        'slur-stop', 'bar'])
        out, rep = gr.spray_cap_filter(toks, artic_cap=0)
        self.assertIn('slur-stop', out)
        self.assertIn('slur-start', out)
        self.assertEqual(rep['removed_slur_pairs'], 0)

    def test_peaks_hook_grounds_artic(self):
        import guard_rescore as gr
        toks = (HDR3 + ['voice-1', 'quarter', 'C', '4', 'stacc',
                        'quarter', 'D', '4', 'stacc', 'bar'])
        out, rep = gr.spray_cap_filter(toks, peaks=[0.05, 0.9],
                                       peak_min=0.5)
        # weak-peak note loses its artic, strong-peak note keeps it
        self.assertEqual(out.count('stacc'), 1)
        self.assertEqual(rep['removed_unrounded'], 1)
        self.assertTrue(_walks_loose(out))

    def test_peaks_none_is_token_path(self):
        import guard_rescore as gr
        toks = HDR3 + ['voice-1', 'quarter', 'C', '4', 'stacc', 'bar']
        out, rep = gr.spray_cap_filter(toks, peaks=None)
        self.assertEqual(out, toks)
        self.assertEqual(rep['removed_artic'], 0)


class ApplyGuardsTest(unittest.TestCase):
    def test_default_is_noop(self):
        import guard_rescore as gr
        toks = (HDR3 + ['voice-1', 'quarter', 'C', '4', 'stacc',
                        'voice-4', 'rest', 'quarter', 'bar'])
        out, rep = gr.apply_guards(toks)
        self.assertEqual(out, toks)
        self.assertFalse(rep['valid'])

    def test_combined_applies_all(self):
        import guard_rescore as gr
        toks = (HDR3 + ['voice-1', 'quarter', 'C', '4', 'stacc',
                        'accent', 'voice-4', 'rest', 'quarter', 'bar'])
        out, rep = gr.apply_guards(toks, enable_voice_cap=True,
                                   enable_rest_guard=True,
                                   enable_spray_cap=True)
        self.assertTrue(rep['valid'])
        self.assertNotIn('voice-4', out)
        self.assertNotIn('rest', out)
        self.assertEqual(out.count('stacc') + out.count('accent'), 1)

    def test_invalid_output_fails_open(self):
        import guard_rescore as gr
        # garbage that no filter can parse: returned unchanged
        toks = ['voice-1', 'chord', 'bar']
        out, rep = gr.apply_guards(toks, enable_voice_cap=True,
                                   enable_rest_guard=True,
                                   enable_spray_cap=True)
        self.assertEqual(out, toks)
        self.assertFalse(rep['valid'])


class RankBeamsTest(unittest.TestCase):
    def test_closure_ranking_prefers_closing(self):
        import guard_rescore as gr
        header = ['clef-G-2', 'key+0', 'time-4-4']
        good = ['voice-1', 'quarter', 'C', '4']
        bad = ['voice-1', 'quarter']  # truncated event
        ranked = gr.rank_beams_by_closure(header, [bad, good])
        self.assertEqual(ranked[0], good)

    def test_empty_beams(self):
        import guard_rescore as gr
        self.assertEqual(gr.rank_beams_by_closure(['bar'], []), [])


class FlipTestHarnessTest(unittest.TestCase):
    def test_inject_flip_pitch(self):
        import guard_rescore_fliptest as ft
        toks = ['voice-1', 'quarter', 'C', '4', 'bar']
        pre, old, new = ft.inject_flip(toks, 2, {'C': 'D'})
        self.assertEqual((old, new), ('C', 'D'))
        self.assertEqual(pre, ['voice-1', 'quarter', 'D'])

    def test_inject_flip_out_of_range(self):
        import guard_rescore_fliptest as ft
        with self.assertRaises(IndexError):
            ft.inject_flip(['a'], 5, {'a': 'b'})

    def test_cascade_counts_until_resync(self):
        import guard_rescore_fliptest as ft
        # stub decoder: emits junk twice, then the reference suffix
        ref = ['X', 'Y', 'Z']
        calls = {'n': 0}

        def step_fn(prefix):
            calls['n'] += 1
            if calls['n'] <= 2:
                return 'JUNK'
            return ref[calls['n'] - 3] if calls['n'] - 3 < 3 else 'Z'

        n = ft.cascade_length(step_fn, ['voice-1'], ref, max_steps=10,
                              resync_k=2)
        self.assertEqual(n, 4)  # JUNK, JUNK, X, Y -> resync at k=2

    def test_recovery_ratio_flags_s18(self):
        import guard_rescore_fliptest as ft
        res = ft.recovery_ratio(flat_cascade=12, guarded_cascade=3)
        self.assertAlmostEqual(res, 4.0)
        self.assertTrue(ft.merits_full_run(12, 3))
        self.assertFalse(ft.merits_full_run(6, 3))

    def test_compare_recovery_with_stubs(self):
        import guard_rescore_fliptest as ft
        ref = ['a', 'b', 'c', 'd']

        def flat(prefix):
            # loops on its own tail: never resyncs within budget
            return 'LOOP'

        def guarded(prefix):
            # resyncs immediately by copying reference
            return ref[len(prefix) % len(ref)]

        out = ft.compare_recovery(flat, guarded, [('case1', ['p'], ref)],
                                  max_steps=8)
        self.assertEqual(out['cases'][0]['flat'], 8)
        self.assertLessEqual(out['cases'][0]['guarded'], 8)
        self.assertIn('ratio', out['cases'][0])


class ProbeFailClosedTest(unittest.TestCase):
    def test_probe_refuses_without_flag(self):
        import guard_rescore_probe as pr
        with self.assertRaises(SystemExit):
            pr.main([])


if __name__ == '__main__':
    unittest.main()
