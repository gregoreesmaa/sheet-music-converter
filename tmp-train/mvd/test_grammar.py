"""Self-contained checks for tmp-train/mvd/grammar.py.

No torch, no GPU: pure prefix-mask logic plus a ground-truth
round-trip (every prefix of real bekern files must admit its actual
continuation — the mask never blocks valid GT). Run with the
repo-root .venv python:

    .venv/Scripts/python.exe tmp-train/mvd/test_grammar.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import grammar  # noqa: E402
import measure_decode  # noqa: E402

_T = data.TOKEN_TO_ID


def ids(*toks):
    return {_T[t] for t in toks}


def test_empty_prefix():
    allow, eos_ok = grammar.allowed_next([])
    assert _T['quarter'] in allow, 'durations must open a stream'
    assert _T['rest'] in allow and _T['bar-rest'] in allow
    assert _T['bar'] in allow and _T['voice-1'] in allow
    assert _T['clef-G-2'] in allow, 'headers legal first'
    assert _T['A'] not in allow, 'bare step must not open a stream'
    assert _T['4'] not in allow, 'bare octave must not open a stream'
    assert _T['tuplet-stop'] not in allow, 'stop with no span blocked'
    assert _T['chord'] not in allow, 'chord with no onset blocked'
    assert eos_ok, 'empty page decodes (0 notes)'
    print('PASS empty prefix')


def test_note_skeleton():
    q = [_T['quarter']]
    allow, eos_ok = grammar.allowed_next(q)
    assert allow == ids('A', 'B', 'C', 'D', 'E', 'F', 'G'), allow
    assert not eos_ok
    allow, eos_ok = grammar.allowed_next(q + [_T['C']])
    assert allow == ids('-1', '0', '1', '2', '3', '4', '5', '6', '7', '8'), allow
    assert not eos_ok
    allow, eos_ok = grammar.allowed_next(q + [_T['C'], _T['4']])
    assert _T['a1'] in allow and _T['dot'] in allow
    assert _T['tie-start'] in allow and _T['tie-stop'] in allow
    assert _T['quarter'] in allow, 'next note may follow unadorned'
    assert _T['A'] not in allow and _T['4'] not in allow
    assert eos_ok, 'a complete note may end the page'
    print('PASS note skeleton quarter/C/4')


def test_tie_chain():
    pre = [_T['quarter'], _T['C'], _T['4'], _T['tie-start']]
    allow, eos_ok = grammar.allowed_next(pre)
    assert _T['tie-stop'] in allow
    assert _T['quarter'] in allow, 'stop-less tie-start still continues'
    assert _T['dot'] not in allow, 'no dots after tie-start'
    assert eos_ok
    allow, eos_ok = grammar.allowed_next(pre + [_T['tie-stop']])
    assert _T['A'] not in allow, 'new event only after tie-stop'
    assert _T['quarter'] in allow and eos_ok
    print('PASS tie chain')


def test_rest_shape():
    allow, eos_ok = grammar.allowed_next([_T['rest']])
    assert allow == ids('whole', 'half', 'quarter', 'eighth', '16th',
                        '32nd', '64th', '128th', 'breve'), allow
    assert not eos_ok
    allow, eos_ok = grammar.allowed_next([_T['rest'], _T['quarter']])
    assert _T['dot'] in allow and _T['bar'] in allow
    assert _T['tie-start'] not in allow, 'rests take no ties'
    assert _T['A'] not in allow
    assert eos_ok
    print('PASS rest shape')


def test_tuplet_gating():
    hdr = [_T['clef-G-2'], _T['key+0'], _T['time-4-4']]
    allow, _ = grammar.allowed_next(hdr)
    assert _T['tuplet-start-3-2'] in allow
    assert _T['tuplet-stop'] not in allow, 'no open span yet'
    pre = hdr + [_T['tuplet-start-3-2'], _T['quarter'], _T['C'], _T['4']]
    allow, _ = grammar.allowed_next(pre)
    assert _T['tuplet-stop'] in allow, 'span open after one event'
    assert _T['quarter'] in allow
    pre2 = pre + [_T['tuplet-stop']]
    allow, _ = grammar.allowed_next(pre2)
    assert _T['tuplet-stop'] not in allow, 'span closed again'
    print('PASS tuplet gating')


def test_chord_gating():
    assert _T['chord'] not in grammar.allowed_next([])[0]
    pre = [_T['quarter'], _T['C'], _T['4']]
    allow, _ = grammar.allowed_next(pre)
    assert _T['chord'] in allow, 'pitched onset enables chord'
    allow, eos_ok = grammar.allowed_next(pre + [_T['chord']])
    assert _T['rest'] not in allow, 'chord on rest must stay blocked'
    assert _T['quarter'] in allow
    assert not eos_ok, 'dangling chord must not end the page'
    print('PASS chord gating')


def _gt_prefixes_ok(path):
    toks = data.load_bekern(path)
    ids_all = data.encode(toks)
    for k in range(len(ids_all)):
        allow, eos_ok = grammar.allowed_next(ids_all[:k])
        nxt = ids_all[k]
        assert nxt in allow, '%s prefix %d: %r blocked' % (
            Path(path).name, k, toks[k])
    allow, eos_ok = grammar.allowed_next(ids_all)
    assert eos_ok, '%s: EOS must be legal at end' % Path(path).name


def test_gt_round_trip():
    base = Path(__file__).resolve().parent / '..' / 'mvd-data'
    _gt_prefixes_ok(str(base / 'ds0151-6-bekern.txt'))  # shortest page
    _gt_prefixes_ok(str(base / 'ds0001-1-bekern.txt'))  # plain page
    _gt_prefixes_ok(str(base / 'ds0002-3-bekern.txt'))  # tuplets + ties
    _gt_prefixes_ok(str(base / 'ds0222-3-bekern.txt'))  # long page
    print('PASS GT round-trip (4 files, every prefix admits GT next)')


def _strict_prefixes_ok(path):
    toks = data.load_bekern(path)
    ids_all = data.encode(toks)
    st = grammar.PrefixState(strict=True)
    for k, tok in enumerate(toks):
        allow, _ = st.allowed()
        assert ids_all[k] in allow, '%s prefix %d: %r blocked (strict)' % (
            Path(path).name, k, tok)
        st.step(tok)
    allow, eos_ok = st.allowed()
    assert eos_ok, '%s: EOS must be legal at end (strict)' % Path(path).name


def test_strict_gt_round_trip():
    base = Path(__file__).resolve().parent / '..' / 'mvd-data'
    _strict_prefixes_ok(str(base / 'ds0151-6-bekern.txt'))
    _strict_prefixes_ok(str(base / 'ds0001-1-bekern.txt'))
    _strict_prefixes_ok(str(base / 'ds0002-3-bekern.txt'))
    _strict_prefixes_ok(str(base / 'ds0222-3-bekern.txt'))
    print('PASS strict GT round-trip (4 files, validator rules GT-safe)')


def test_strict_alter_rules():
    pre = data.encode('quarter C 4 a1'.split())
    allow_leg, _ = grammar.allowed_next(pre)
    allow_st, _ = grammar.allowed_next_strict(pre)
    assert _T['a-1'] in allow_leg, 'legacy keeps double-alter'
    assert _T['a-1'] not in allow_st, 'strict blocks second alter'
    assert _T['dot'] in allow_st and _T['quarter'] in allow_st
    pre2 = data.encode('quarter C 4 dot'.split())
    allow_st2, _ = grammar.allowed_next_strict(pre2)
    assert _T['a1'] not in allow_st2, 'strict blocks alter after dot'
    allow_leg2, _ = grammar.allowed_next(pre2)
    assert _T['a1'] in allow_leg2, 'legacy keeps alter after dot'
    # Single alter still fine in strict.
    allow1, _ = grammar.allowed_next_strict(data.encode('quarter C 4'.split()))
    assert _T['a1'] in allow1
    print('PASS strict alter rules (single + pre-dot only)')


def test_strict_voice_rules():
    # Non-sequential open rejected at step time (fails open at the
    # allowed_next level, so test the state directly).
    st = grammar.PrefixState(strict=True)
    for t in ['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4']:
        st.step(t)
    try:
        st.step('voice-3')
    except ValueError:
        pass
    else:
        raise AssertionError('strict must reject non-sequential voice-3')
    # Sequential open works: voice-1 then voice-2.
    st3 = grammar.PrefixState(strict=True)
    for t in ['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4',
              'bar', 'voice-1', 'quarter', 'D', '4', 'voice-2']:
        st3.step(t)
    allow, _ = st3.allowed()
    assert _T['quarter'] in allow
    # Consecutive duplicate voice marker blocked in strict only.
    st4 = grammar.PrefixState(strict=True)
    for t in ['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4',
              'bar', 'voice-1', 'voice-1']:
        try:
            st4.step(t)
        except ValueError:
            assert t == 'voice-1'
            break
    else:
        raise AssertionError('strict must reject duplicate voice-1')
    st5 = grammar.PrefixState(strict=False)
    for t in ['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4',
              'bar', 'voice-1', 'voice-1']:
        st5.step(t)  # legacy allows
    # Consecutive duplicate bar blocked in strict only.
    st6 = grammar.PrefixState(strict=True)
    st6.step('bar')
    try:
        st6.step('bar')
    except ValueError:
        pass
    else:
        raise AssertionError('strict must reject duplicate bar')
    print('PASS strict voice rules (sequential, no dup markers)')


def test_loop_caps():
    st = grammar.PrefixState()
    for t in ['clef-G-2', 'key+0', 'time-4-4', 'bar']:
        st.step(t)
    assert st.cycle_forbidden() is None, 'no cycle in header/short prefix'
    cyc = ['quarter', 'G', '4'] * 12
    for t in cyc:
        st.step(t)
    assert st.cycle_forbidden() == 'quarter', st.cycle_forbidden()
    # Unigram run tracking.
    st2 = grammar.PrefixState()
    for t in ['quarter', 'C', '4', 'dot', 'dot']:
        st2.step(t)
    assert (st2.run_tok, st2.run_len) == ('dot', 2), (st2.run_tok,
                                                     st2.run_len)
    print('PASS loop caps (cycle continuation + run tracking)')


def test_courtesy_time_sig_retunes_at_bar():
    # A courtesy time-sig before the barline must NOT retune
    # mid-measure, but MUST take effect at the barline. Today the
    # pending books never apply, so the new measure decodes under
    # the stale meter (FP3 clock crash: mid-system time-*-* leaves
    # tsig stale past the bar).
    st = grammar.PrefixState()
    for t in ['clef-G-2', 'key+0', 'time-4-4',
              'quarter', 'C', '4', 'quarter', 'D', '4']:
        st.step(t)
    assert st.tsig == 4.0, st.tsig
    st.step('time-5-4')  # courtesy: mid-measure, cursors busy
    assert st.tsig == 4.0, st.tsig
    st.step('bar')
    assert st.tsig == 5.0, \
        'courtesy time-sig must take effect at the barline, got %r' \
        % (st.tsig,)
    # Barline change with idle cursors still applies immediately
    # (legacy behavior, unchanged).
    st.step('time-3-4')
    assert st.tsig == 3.0, st.tsig
    print('PASS courtesy time-sig retunes at bar')


class StrictBarDurationTest(unittest.TestCase):
    """T2.3: PrefixState(strict=True) conserves bar duration."""

    def _walk(self, text):
        st = grammar.PrefixState(strict=True)
        for t in text.split():
            st.step(t)
        return st

    def test_time_signature_sets_expected(self):
        self.assertEqual(self._walk('clef-G-2 key+0 time-3-4').tsig, 3.0)
        self.assertAlmostEqual(
            self._walk('clef-G-2 key+0 time-6-8').tsig, 3.0)
        self.assertAlmostEqual(
            self._walk('clef-G-2 key+0 time-12-16').tsig, 3.0)

    def test_exact_fill_closes(self):
        st = self._walk('clef-G-2 key+0 time-4-4 quarter C 4 quarter D 4 '
                        'quarter E 4 quarter F 4 bar')
        self.assertEqual(st.vcurs, [0.0])

    def test_fifth_quarter_masked_then_rejected(self):
        st = self._walk('clef-G-2 key+0 time-4-4 quarter C 4 quarter D 4 '
                        'quarter E 4 quarter F 4')
        allow, _ = st.allowed()
        self.assertNotIn(_T['quarter'], allow)
        self.assertIn(_T['bar'], allow)
        self.assertNotIn(_T['bar-rest'], allow)
        with self.assertRaisesRegex(ValueError, 'overshoots bar'):
            st.step('quarter')

    def test_whole_note_overshoot_3_4(self):
        st = self._walk('clef-G-2 key+0 time-3-4')
        allow, _ = st.allowed()
        self.assertNotIn(_T['whole'], allow)
        with self.assertRaisesRegex(ValueError, 'overshoots bar'):
            self._walk('clef-G-2 key+0 time-3-4 whole C 4 bar')

    def test_rest_overshoot_blocked(self):
        st = self._walk('clef-G-2 key+0 time-4-4 rest half rest half')
        allow, _ = st.allowed()
        self.assertNotIn(_T['quarter'], allow)
        self.assertIn(_T['bar'], allow)
        st.step('rest')
        allow, _ = st.allowed()
        self.assertNotIn(_T['quarter'], allow)
        with self.assertRaisesRegex(ValueError, 'overshoots bar'):
            st.step('quarter')

    def test_bar_rest_blocked_at_full_or_nonempty_bar(self):
        # Empty bar allows bar-rest.
        st0 = self._walk('clef-G-2 key+0 time-4-4')
        self.assertIn(_T['bar-rest'], st0.allowed()[0])
        # Non-empty or full bar forbids bar-rest in strict mode (anti-loop).
        st = self._walk('clef-G-2 key+0 time-4-4 quarter C 4 quarter D 4 '
                        'quarter E 4 quarter F 4')
        allow, _ = st.allowed()
        self.assertNotIn(_T['quarter'], allow)
        self.assertNotIn(_T['bar-rest'], allow)
        with self.assertRaises(ValueError):
            st.step('bar-rest')

    def test_chord_tone_adds_zero(self):
        self._walk('clef-G-2 key+0 time-4-4 quarter C 4 chord quarter E 4 '
                   'quarter D 4 quarter F 4 quarter G 4 bar')

    def test_triplet_members_scale(self):
        self._walk('clef-G-2 key+0 time-4-4 tuplet-start-3-2 quarter C 4 '
                   'quarter D 4 quarter E 4 tuplet-stop half F 4 bar')

    def test_dotted_exact_fill_and_overdot(self):
        self._walk('clef-G-2 key+0 time-3-4 half C 4 dot bar')
        with self.assertRaisesRegex(ValueError, 'overshoots bar'):
            self._walk('clef-G-2 key+0 time-2-4 half C 4 dot')

    def test_no_time_signature_no_gating(self):
        self._walk('quarter C 4 ' * 8 + 'bar')

    def test_nonstrict_tolerates_overshoot(self):
        st = grammar.PrefixState(strict=False)
        for t in 'clef-G-2 key+0 time-3-4 whole C 4 bar'.split():
            st.step(t)

    def test_strict_gt_files_stay_clean(self):
        base = Path(__file__).resolve().parent / '..' / 'mvd-data'
        for name in ('ds0151-6-bekern.txt', 'ds0001-1-bekern.txt',
                     'ds0002-3-bekern.txt', 'ds0222-3-bekern.txt'):
            toks = data.load_bekern(str(base / name))
            ids = data.encode(toks)
            st = grammar.PrefixState(strict=True)
            for k, tok in enumerate(toks):
                allow, _ = st.allowed()
                self.assertIn(ids[k], allow, '%s prefix %d' % (name, k))
                st.step(tok)
            _, eos_ok = st.allowed()
            self.assertTrue(eos_ok)


class StrictMaskParityTest(unittest.TestCase):
    """T2.4 pipefix: strict allowed() masks exactly what step() rejects.

    Every forbidden transition is excluded under strict but admitted
    under the legacy non-strict mask; the sweep proves mask/step
    parity (no strict-allowed continuation raises at step time).
    """

    def _masks(self, prefix):
        pre = data.encode(prefix)
        allow_st, _ = grammar.allowed_next_strict(pre)
        allow_leg, _ = grammar.allowed_next(pre)
        return allow_st, allow_leg

    def test_dup_bar_masked_strict_only(self):
        allow_st, allow_leg = self._masks(
            ['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4', 'bar'])
        self.assertNotIn(_T['bar'], allow_st)
        self.assertIn(_T['bar'], allow_leg)

    def test_nonsequential_voice_masked_strict_only(self):
        allow_st, allow_leg = self._masks(
            ['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4'])
        self.assertNotIn(_T['voice-3'], allow_st)
        self.assertIn(_T['voice-1'], allow_st)
        self.assertIn(_T['voice-3'], allow_leg)

    def test_dup_voice_masked_strict_only(self):
        allow_st, allow_leg = self._masks(
            ['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4',
             'bar', 'voice-1'])
        self.assertNotIn(_T['voice-1'], allow_st)
        self.assertIn(_T['voice-2'], allow_st)
        self.assertIn(_T['voice-1'], allow_leg)

    def test_dot_overshoot_masked_strict_only(self):
        # 2/4 filled by a half note: one more dot overshoots the bar.
        allow_st, allow_leg = self._masks(
            ['clef-G-2', 'key+0', 'time-2-4', 'half', 'C', '4'])
        self.assertNotIn(_T['dot'], allow_st)
        self.assertIn(_T['dot'], allow_leg)
        # Room to dot: a half note in 4/4 stays dottable under strict.
        allow_st2, _ = self._masks(
            ['clef-G-2', 'key+0', 'time-4-4', 'half', 'C', '4'])
        self.assertIn(_T['dot'], allow_st2)

    def test_rest_dot_overshoot_masked_strict_only(self):
        allow_st, allow_leg = self._masks(
            ['clef-G-2', 'key+0', 'time-2-4', 'rest', 'half'])
        self.assertNotIn(_T['dot'], allow_st)
        self.assertIn(_T['dot'], allow_leg)

    def test_strict_allow_implies_step_accepts(self):
        prefixes = [
            ['clef-G-2', 'key+0', 'time-4-4'],
            ['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4'],
            ['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4',
             'quarter', 'D', '4', 'quarter', 'E', '4', 'quarter', 'F', '4'],
            ['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4',
             'quarter', 'D', '4', 'quarter', 'E', '4', 'quarter', 'F', '4',
             'bar'],
            ['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4',
             'quarter', 'D', '4', 'quarter', 'E', '4', 'quarter', 'F', '4',
             'bar', 'voice-1', 'quarter', 'C', '4'],
            ['clef-G-2', 'key+0', 'time-2-4', 'rest', 'half'],
            ['clef-G-2', 'key+0', 'time-3-4', 'half', 'C', '4',
             'tie-start'],
        ]
        base = Path(__file__).resolve().parent / '..' / 'mvd-data'
        gt = data.load_bekern(str(base / 'ds0001-1-bekern.txt'))
        prefixes.extend(gt[:k] for k in range(0, len(gt), 37))
        for prefix in prefixes:
            allow_st, _ = grammar.allowed_next_strict(data.encode(prefix))
            st0 = grammar.PrefixState(strict=True)
            for t in prefix:
                st0.step(t)
            for i in allow_st:
                st = st0.copy()
                try:
                    st.step(data.VOCAB[i])
                except ValueError:
                    self.fail('strict-allowed %r raises at step after '
                              'prefix ...%r' % (data.VOCAB[i], prefix[-3:]))


class BekernPlusMaskTest(unittest.TestCase):
    """T3.1 item 5: prefix states for the expanded bekern+ token list.

    Grace, slur/articulation/fermata modifiers, measure dynamics,
    hairpins, barline variants and mid-stream clef/key/time all walk
    strict and lax; strict masks stay consistent with step().
    """

    PLUS = ('clef-G-2 key+0 time-4-4 voice-1 quarter C 4 grace eighth D 5 '
            'quarter E 4 slur-start stacc quarter F 4 slur-stop tie-start '
            'bar clef-F-4 key+2 time-3-4 dyn-f hairpin-cresc-start voice-1 '
            'tuplet-start-3-2 quarter F 4 tie-stop quarter F 4 quarter F 4 '
            'tuplet-stop bar-double voice-1 half A 3 hairpin-stop '
            'rest quarter fermata bar-light-heavy').split()

    def _walk(self, toks, strict=True):
        st = grammar.PrefixState(strict=strict)
        for t in toks:
            allow, _ = st.allowed()
            self.assertIn(data.TOKEN_TO_ID[t], allow,
                          '%r not admitted in mode %r (strict=%r)'
                          % (t, st.mode, strict))
            st.step(t)
        return st

    def test_full_plus_stream_strict(self):
        st = self._walk(self.PLUS, True)
        _, eos_ok = st.allowed()
        self.assertTrue(eos_ok)
        self.assertEqual(st.tsig, 3.0)

    def test_full_plus_stream_lax(self):
        st = self._walk(self.PLUS, False)
        _, eos_ok = st.allowed()
        self.assertTrue(eos_ok)

    def test_grace_adds_zero(self):
        st = self._walk(['clef-G-2', 'key+0', 'time-4-4', 'voice-1',
                         'grace', 'eighth', 'C', '5', 'quarter'], True)
        self.assertEqual(st.vcurs, [0.0])

    def test_grace_requires_duration(self):
        for bad in (['grace', 'bar'], ['grace', 'rest'],
                    ['grace', 'quarter', 'C', '4', 'grace', 'rest']):
            st = grammar.PrefixState(strict=True)
            with self.assertRaises(ValueError, msg=' '.join(bad)):
                for t in ['clef-G-2', 'key+0', 'time-4-4', 'voice-1'] + bad:
                    st.step(t)

    def test_midstream_time_updates_only_at_barline(self):
        pre = ['clef-G-2', 'key+0', 'time-4-4', 'voice-1',
               'quarter', 'C', '4', 'quarter', 'D', '4']
        st = grammar.PrefixState(strict=True)
        for t in pre:
            st.step(t)
        # Courtesy change mid-bar: books keep the old signature.
        st.step('time-3-4')
        self.assertEqual(st.tsig, 4.0)
        # Real change at the barline: books retune.
        st.step('bar')
        st.step('time-3-4')
        self.assertEqual(st.tsig, 3.0)

    def test_midstream_headers_accepted(self):
        st = self._walk(['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4',
                         'bar', 'clef-F-4', 'key+2', 'time-3-4',
                         'quarter', 'D', '4'], True)
        self.assertEqual(st.tsig, 3.0)

    def test_bar_variants_close_measures(self):
        st = self._walk(['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4',
                         'bar-double', 'quarter', 'D', '4',
                         'bar-repeat-right'], True)
        _, eos_ok = st.allowed()
        self.assertTrue(eos_ok)
        st.step('bar-repeat-left')
        st.step('quarter')
        st.step('E')
        st.step('4')
        st.step('bar-light-heavy')
        _, eos_ok = st.allowed()
        self.assertTrue(eos_ok)

    def test_duplicate_close_rejected_strict_only(self):
        for dup in (['bar', 'bar'], ['bar', 'bar-double'],
                    ['bar-double', 'bar'], ['bar-light-heavy', 'bar-repeat-right']):
            pre = ['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4'] + dup[:1]
            st = grammar.PrefixState(strict=True)
            for t in pre:
                st.step(t)
            with self.assertRaises(ValueError, msg=' '.join(dup)):
                st.step(dup[1])
            st_lax = grammar.PrefixState(strict=False)
            for t in pre + dup[1:]:
                st_lax.step(t)

    def test_measure_tokens_are_bound_noops(self):
        # The closing 'bar' flushes the open quarter into the books.
        st = self._walk(['clef-G-2', 'key+0', 'time-4-4', 'dyn-f',
                         'hairpin-cresc-start', 'bar-repeat-left',
                         'quarter', 'C', '4', 'bar'], True)
        self.assertEqual(st.vcurs, [0.0])
        self.assertEqual(st._pending_value(), 0.0)
        pre = self._walk(['clef-G-2', 'key+0', 'time-4-4', 'dyn-f',
                          'hairpin-cresc-start', 'bar-repeat-left',
                          'quarter', 'C', '4'], True)
        self.assertEqual(pre.vcurs, [0.0])
        self.assertEqual(pre._pending_value(), 1.0)
        self.assertEqual(pre.tsig, 4.0)

    def test_strict_allow_implies_step_accepts_plus(self):
        prefixes = [
            self.PLUS[:7],
            self.PLUS[:16],
            self.PLUS[:24],
            self.PLUS[:33],
            ['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4',
             'tie-start', 'slur-start', 'stacc'],
            ['clef-G-2', 'key+0', 'time-4-4', 'rest', 'quarter', 'fermata'],
            ['clef-G-2', 'key+0', 'time-2-4', 'quarter', 'C', '4', 'bar',
             'time-3-4', 'quarter', 'D', '4'],
        ]
        for prefix in prefixes:
            allow_st, _ = grammar.allowed_next_strict(data.encode(prefix))
            st0 = grammar.PrefixState(strict=True)
            for t in prefix:
                st0.step(t)
            for i in allow_st:
                st = st0.copy()
                try:
                    st.step(data.VOCAB[i])
                except ValueError:
                    self.fail('strict-allowed %r raises at step after '
                              'prefix ...%r' % (data.VOCAB[i], prefix[-3:]))


class TestFullBarRestMasking(unittest.TestCase):
    """T2.4b: strict masks REST (+ rest-mode durations) at a full bar.

    A full bar offering REST traps the decoder in rest-mode with every
    duration overshooting (all logits -inf). Non-strict keeps the
    legacy unconditional set.
    """

    def _walk(self, text, strict=True):
        st = grammar.PrefixState(strict=strict)
        for t in text.split():
            st.step(t)
        return st

    def test_rest_masked_at_full_bar_strict_only(self):
        # Four quarters fill 4/4: cursor 3.0 + open event 1.0 = full.
        full = ('clef-G-2 key+0 time-4-4 quarter C 4 quarter D 4 '
                'quarter E 4 quarter F 4')
        st = self._walk(full, True)
        self.assertEqual(st.vcurs[st.seg] + st._pending_value(), st.tsig)
        allow, _ = st.allowed()
        self.assertNotIn(_T['rest'], allow)
        self.assertIn(_T['bar'], allow)
        self.assertNotIn(_T['bar-rest'], allow)
        allow_leg, _ = self._walk(full, False).allowed()
        self.assertIn(_T['rest'], allow_leg)

    def test_rest_masked_after_bar_rest_fills_bar(self):
        # Post-bar-rest boundary: cursor itself sits exactly on the bar.
        st = self._walk('clef-G-2 key+0 time-4-4 bar-rest', True)
        self.assertEqual(st.vcurs[st.seg], st.tsig)
        allow, _ = st.allowed()
        self.assertNotIn(_T['rest'], allow)
        self.assertIn(_T['bar'], allow)
        self.assertNotIn(_T['bar-rest'], allow)

    def test_rest_mode_durations_empty_at_full_bar_strict(self):
        st = self._walk('clef-G-2 key+0 time-4-4 quarter C 4 quarter D 4 '
                        'quarter E 4 quarter F 4 rest', True)
        self.assertEqual(st.mode, 'rest')
        self.assertEqual(st.vcurs[st.seg], st.tsig)
        allow, _ = st.allowed()
        durs = {_T[d] for d in
                ('whole', 'half', 'quarter', 'eighth', '16th', '32nd',
                 '64th', '128th', 'breve')}
        self.assertFalse(allow & durs)
        st2 = self._walk('clef-G-2 key+0 time-4-4 quarter C 4 quarter D 4 '
                         'quarter E 4 quarter F 4 rest', False)
        allow2, _ = st2.allowed()
        self.assertTrue(allow2 & durs)

    def test_rest_still_allowed_with_room(self):
        allow, _ = self._walk(
            'clef-G-2 key+0 time-4-4 quarter C 4', True).allowed()
        self.assertIn(_T['rest'], allow)


class _FakeNet:
    """Scripted beam_search stand-in (pins strict/max_len call shape)."""

    def __init__(self, script):
        self.script = [list(s) for s in script]
        self.calls = []

    def beam_search(self, images, beam=3, strict=True, max_len=564,
                    mem_mask=None):
        self.calls.append({'beam': beam, 'strict': strict,
                           'max_len': max_len,
                           'shape': tuple(images.shape),
                           'mem_mask': (None if mem_mask is None
                                        else tuple(mem_mask.shape))})
        assert strict is True and max_len == 564
        import torch
        ids = data.encode(
            self.script[min(len(self.calls) - 1, len(self.script) - 1)])
        return torch.tensor([ids + [data.EOS_ID]], dtype=torch.long)


_HDR = ['clef-G-2', 'key+0', 'time-4-4']
_M_GOOD = ['quarter', 'C', '4', 'quarter', 'D', '4',
           'quarter', 'E', '4', 'quarter', 'F', '4']
_M_BAD = _M_GOOD + ['quarter', 'G', '4']  # 5 beats in 4/4


class MeasureFallbackTest(unittest.TestCase):
    """T2.3: bekern-decode closure check + bar-rest fallback splice."""

    def test_good_measures_pass_through(self):
        toks = _HDR + _M_GOOD + ['bar'] + _M_GOOD + ['bar']
        net = _FakeNet([toks])
        import torch
        got = measure_decode.decode_system_with_measure_fallback(
            net, torch.zeros(1, 3, 64, 64), beam=3)
        self.assertEqual(got, toks)
        self.assertEqual(net.calls[0]['beam'], 3)
        self.assertTrue(net.calls[0]['strict'])
        self.assertEqual(net.calls[0]['max_len'], 564)

    def test_strict_only_violation_passes_through(self):
        # T2.4 pipefix: a 5-beat measure in 4/4 fails strict closure
        # but is non-strict-clean — the full rollout must survive, never
        # collapse to header + bar-rest on a strict-only violation.
        toks = (_HDR + _M_GOOD + ['bar'] + _M_BAD + ['bar']
                + _M_GOOD + ['bar'])
        h, ms, _tb = measure_decode.split_measures(toks)
        self.assertFalse(measure_decode.measure_closes(h, ms[1]))
        self.assertTrue(measure_decode._measure_closes_loose(h, ms[1]))
        net = _FakeNet([toks])
        import torch
        got = measure_decode.decode_system_with_measure_fallback(
            net, torch.zeros(1, 3, 64, 64))
        self.assertEqual(got, toks)
        st = grammar.PrefixState(strict=False)
        for t in got:
            st.step(t)

    def test_trailing_partial_event_trimmed(self):
        # A dangling chord stub keeps its complete prefix; only the
        # unparseable tail is dropped (no whole-measure wipe).
        toks = _HDR + _M_GOOD + ['bar', 'quarter', 'C', '4', 'chord', 'bar']
        net = _FakeNet([toks])
        import torch
        got = measure_decode.decode_system_with_measure_fallback(
            net, torch.zeros(1, 3, 64, 64))
        self.assertEqual(
            got, _HDR + _M_GOOD + ['bar', 'quarter', 'C', '4', 'bar'])

    def test_broken_header_splices_bar_rest(self):
        # bar-rest survives only for the hopeless case: even the loose
        # walk fails (broken header), so there is nothing to keep.
        bad_h = ['tuplet-stop']
        self.assertIsNone(
            measure_decode._trim_trailing_partial(bad_h, _M_GOOD))
        self.assertEqual(
            measure_decode._repair_measure(bad_h, _M_GOOD), ['bar-rest'])

    def test_split_join_roundtrip(self):
        toks = _HDR + _M_GOOD + ['bar'] + _M_GOOD + ['bar']
        h, ms, tb = measure_decode.split_measures(toks)
        self.assertEqual(h, _HDR)
        self.assertEqual(len(ms), 2)
        self.assertTrue(tb)
        self.assertEqual(measure_decode.join_measures(h, ms, tb), toks)


def _staff_lines(n, y0=20.0, dy=40.0, sp=6.0):
    return [[y0 + s * dy + k * sp for k in range(5)] for s in range(n)]


class MultiChunkMergeTest(unittest.TestCase):
    """T2.3: S>4 staff windowing with per-measure voice merging."""

    def _page(self):
        from PIL import Image
        return Image.new('RGB', (400, 300), (255, 255, 255))

    def test_single_chunk_path(self):
        m = ['voice-1', 'quarter', 'C', '4',
             'voice-2', 'quarter', 'E', '4', 'quarter', 'F', '4',
             'quarter', 'G', '4', 'quarter', 'A', '4']
        toks = _HDR + m + ['bar'] + m + ['bar']
        net = _FakeNet([toks])
        got = measure_decode.decode_multi_staff_system(
            net, self._page(), _staff_lines(2), beam=3)
        self.assertEqual(got, toks)
        self.assertEqual(len(net.calls), 1)
        self.assertEqual(net.calls[0]['shape'][0], 1)
        self.assertEqual(net.calls[0]['shape'][1], 3)
        self.assertEqual(net.calls[0]['shape'][2], 256)

    def test_six_staves_merge_4p2(self):
        h0 = ['clef-G-2', 'key+0', 'time-4-4']
        c0m = ['voice-1', 'quarter', 'C', '4',
               'voice-2', 'quarter', 'E', '4']
        c0 = h0 + c0m + ['bar'] + c0m + ['bar']
        h1 = ['clef-F-4', 'key+0', 'time-4-4']
        c1m = ['voice-1', 'half', 'G', '3']
        c1 = h1 + c1m + ['bar'] + c1m + ['bar']
        net = _FakeNet([c0, c1])
        got = measure_decode.decode_multi_staff_system(
            net, self._page(), _staff_lines(6), beam=3)
        self.assertEqual(len(net.calls), 2)
        want_m = c0m + ['voice-3', 'half', 'G', '3']
        self.assertEqual(got, h0 + h1 + want_m + ['bar'] + want_m + ['bar'])
        self.assertTrue(measure_decode.measure_closes(h0 + h1, want_m))
        st = grammar.PrefixState(strict=True)
        for t in got:
            st.step(t)

    def test_unequal_measure_counts(self):
        h0 = ['clef-G-2', 'key+0', 'time-4-4']
        c0 = (h0 + ['quarter', 'C', '4'] * 4 + ['bar']
              + ['quarter', 'D', '4'] * 4 + ['bar'])
        h1 = ['clef-F-4', 'key+0', 'time-4-4']
        c1 = h1 + ['half', 'G', '3', 'half', 'A', '3'] + ['bar']
        net = _FakeNet([c0, c1])
        got = measure_decode.decode_multi_staff_system(
            net, self._page(), _staff_lines(6))
        h, ms, _tb = measure_decode.split_measures(got)
        self.assertEqual(len(ms), 2)
        self.assertTrue(
            all(measure_decode.measure_closes(h, m) for m in ms))
        self.assertIn('voice-2', ms[0])
        self.assertNotIn('voice-3', ms[1])


class StrictTupletBankTest(unittest.TestCase):
    """omr-t24-bankfix: strict forbids chained tuplet-starts + sub-tick.

    TDD RED (pre-fix all three fail: chained/full-bar starts offered +
    stepped, sub-tick duration admitted). Non-strict paths stay legacy.
    """

    def _walk(self, text, strict=True):
        st = grammar.PrefixState(strict=strict)
        for t in text.split():
            st.step(t)
        return st

    def test_chained_tuplet_start_blocked_strict_only(self):
        pre = 'clef-G-2 key+0 time-4-4 tuplet-start-3-2'
        st = self._walk(pre, True)
        self.assertGreaterEqual(st.depth[st.seg] + st.pending, 1)
        allow, _ = st.allowed()
        self.assertNotIn(_T['tuplet-start-3-2'], allow)
        self.assertNotIn(_T['tuplet-start-6-4'], allow)
        with self.assertRaises(ValueError):
            st.step('tuplet-start-3-2')
        st_leg = self._walk(pre, False)
        allow_leg, _ = st_leg.allowed()
        self.assertIn(_T['tuplet-start-3-2'], allow_leg)
        st_leg.step('tuplet-start-3-2')  # legacy still chains
        # Drained span also blocks a second start.
        st2 = self._walk(pre + ' quarter C 4', True)
        self.assertGreaterEqual(st2.depth[st2.seg] + st2.pending, 1)
        allow2, _ = st2.allowed()
        self.assertNotIn(_T['tuplet-start-3-2'], allow2)
        with self.assertRaises(ValueError):
            st2.step('tuplet-start-3-2')

    def test_tuplet_start_blocked_at_full_bar_strict_only(self):
        full = ('clef-G-2 key+0 time-4-4 quarter C 4 quarter D 4 '
                'quarter E 4 quarter F 4')
        st = self._walk(full, True)
        self.assertTrue(st._bar_is_full())
        allow, _ = st.allowed()
        self.assertNotIn(_T['tuplet-start-3-2'], allow)
        self.assertNotIn(_T['tuplet-start-6-4'], allow)
        with self.assertRaises(ValueError):
            st.step('tuplet-start-3-2')
        allow_leg, _ = self._walk(full, False).allowed()
        self.assertIn(_T['tuplet-start-3-2'], allow_leg)

    def test_subtick_duration_blocked_strict_only(self):
        st = self._walk('clef-G-2 key+0 time-4-4', True)
        st.vscales[st.seg] = [(3, 2)] * 12  # (2/3)^12 micro scale
        self.assertLessEqual(
            round(grammar._WRITTEN['64th'] * st._open_scale() * 480.0), 0)
        self.assertTrue(st._blocks_duration(grammar._WRITTEN['64th']))
        allow, _ = st.allowed()
        self.assertNotIn(_T['64th'], allow)
        with self.assertRaises(ValueError):
            st.step('64th')
        st_leg = grammar.PrefixState(strict=False)
        for t in 'clef-G-2 key+0 time-4-4'.split():
            st_leg.step(t)
        st_leg.vscales[st_leg.seg] = [(3, 2)] * 12
        self.assertFalse(st_leg._blocks_duration(grammar._WRITTEN['64th']))
        allow_leg, _ = st_leg.allowed()
        self.assertIn(_T['64th'], allow_leg)


if __name__ == '__main__':
    test_empty_prefix()
    test_note_skeleton()
    test_tie_chain()
    test_rest_shape()
    test_tuplet_gating()
    test_chord_gating()
    test_gt_round_trip()
    test_strict_gt_round_trip()
    test_strict_alter_rules()
    test_strict_voice_rules()
    test_loop_caps()
    test_courtesy_time_sig_retunes_at_bar()
    print('ALL GRAMMAR TESTS PASSED')
    unittest.main(argv=['test_grammar'], exit=False)
