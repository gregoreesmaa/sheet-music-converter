"""Regression test for the T2.4 Gate 4 eval empty-decode bug.

``measure_decode.decode_system_with_measure_fallback`` /
``decode_multi_staff_system`` return bekern STRING token lists (via
``_ids_to_tokens``), but ``decode_system_v2`` validated them with
``data.decode``, which takes INT ids only and raises ``ValueError``
on strings — so every T2.4 hypothesis was discarded (empty hyp file,
``f1=0``) while in-memory ``ntokens`` stayed nonzero. Run with the
repo-root .venv python:

    .\\.venv\\Scripts\\python.exe -m unittest tmp-train/mvd/test_eval_sys_gate.py -v
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch  # noqa: E402

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data  # noqa: E402
import eval_sys_gate as esg  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASER_EXE = (REPO_ROOT / 'converter' / 'assembly-serialize' / 'target'
            / 'release' / 'assembly-serialize.exe')
EHARNESS_EXE = (REPO_ROOT / 'converter' / 'eval-harness' / 'target'
                / 'release' / 'eval-harness.exe')
SYS_DEV = (REPO_ROOT / 'data-gen' / 'score-corpus' / 'gate20' / 'sys-dev')


class TestNotationWiring(unittest.TestCase):
    """Page-level notation scoring rides alongside strict-F1, fail-soft."""

    def test_notation_xmls_failsoft_on_missing_binaries(self):
        with tempfile.TemporaryDirectory() as d:
            out = esg.score_notation_xmls(
                Path(d) / 'ref.xml', Path(d) / 'hyp.xml', Path(d),
                's', Path(d) / 'no-strict.exe')
            self.assertEqual(out['notation_f1'], 0.0)
            self.assertTrue(out['notation_zeroed'])
            self.assertIn('notation-f1-', out['notation_zero_reason'])

    def test_notation_bekerns_identical_scores_one(self):
        for exe in (ASER_EXE, EHARNESS_EXE):
            if not exe.exists():
                self.skipTest('release binary missing: %s' % exe)
        ref = SYS_DEV / 'refs' / 'ds0091-3-s0-bekern.txt'
        if not ref.exists():
            self.skipTest('sys-dev fixture missing')
        with tempfile.TemporaryDirectory() as d:
            out = esg.score_notation_bekerns(
                ref, ref, Path(d), 'idem', ASER_EXE, EHARNESS_EXE)
            self.assertFalse(out.get('notation_zeroed'), out)
            self.assertEqual(out['notation_f1'], 1.0)
            self.assertEqual(out['notation_drift'], 1.0)
            for k in ('notation_spell', 'notation_staff',
                      'notation_voice', 'notation_measure'):
                self.assertIn(k, out)

    def test_notation_bekerns_different_scores_below_one(self):
        for exe in (ASER_EXE, EHARNESS_EXE):
            if not exe.exists():
                self.skipTest('release binary missing: %s' % exe)
        ra = SYS_DEV / 'refs' / 'ds0091-3-s0-bekern.txt'
        rb = SYS_DEV / 'refs' / 'ds0091-3-s1-bekern.txt'
        if not (ra.exists() and rb.exists()):
            self.skipTest('sys-dev fixture missing')
        with tempfile.TemporaryDirectory() as d:
            out = esg.score_notation_bekerns(
                ra, rb, Path(d), 'diff', ASER_EXE, EHARNESS_EXE)
            self.assertFalse(out.get('notation_zeroed'), out)
            self.assertLess(out['notation_f1'], 1.0)


class TestHypTokensValid(unittest.TestCase):
    def test_string_tokens_in_vocab_are_valid(self):
        toks = ['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4',
                'bar']
        self.assertTrue(esg.hyp_tokens_valid(toks))
        # Pins the bug mechanism: the old data.decode check rejects
        # these outright (INT ids only).
        with self.assertRaises(ValueError):
            data.decode(toks)

    def test_string_tokens_outside_vocab_are_invalid(self):
        self.assertFalse(esg.hyp_tokens_valid(
            ['quarter', 'C', '4', 'not-a-bekern-token']))

    def test_empty_stream_is_valid(self):
        self.assertTrue(esg.hyp_tokens_valid([]))

    def test_int_ids_still_validate(self):
        self.assertTrue(esg.hyp_tokens_valid(
            [data.TOKEN_TO_ID['bar'], data.TOKEN_TO_ID['quarter']]))

    def test_int_ids_out_of_range_rejected(self):
        self.assertFalse(esg.hyp_tokens_valid([len(data.VOCAB) + 99]))


class FakeNet:
    """Stands in for MvdNet.beam_search (returns one row of ids)."""

    def __init__(self, ids):
        self._ids = list(ids)

    def beam_search(self, *args, **kwargs):
        return [list(self._ids)]


class TestVisualSystemGroupsSRef(unittest.TestCase):
    """T2.4c: gap grouping uses the header staff count when it divides.

    ``factor=3.5`` over-splits quartet/piano systems (10 groups where
    the header says S_ref=4); chunking by S_ref fixes it, keeping the
    factor path when the detected count is not evenly divisible.
    """

    def _staves(self, n, gap=12.0, h=40.0):
        staves = []
        y = 0.0
        for _ in range(n):
            staves.append({'x1': 0.0, 'y1': y,
                           'x2': 100.0, 'y2': y + h})
            y += h + gap
        return staves

    def test_even_split_by_s_ref(self):
        groups = esg.visual_system_groups(
            self._staves(8), space=8.0, s_ref=4)
        self.assertEqual([len(g) for g in groups], [4, 4])

    def test_uneven_count_keeps_factor_path(self):
        groups = esg.visual_system_groups(
            self._staves(5), space=8.0, s_ref=4)
        # Uniform 12px gaps never exceed 3.5*8px: one legacy group.
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 5)

    def test_factor_path_unchanged_without_s_ref(self):
        staves = self._staves(4, gap=12.0)
        staves[2]['y1'] += 200.0
        staves[2]['y2'] += 200.0
        staves[3]['y1'] += 200.0
        staves[3]['y2'] += 200.0
        groups = esg.visual_system_groups(staves, space=8.0)
        self.assertEqual([len(g) for g in groups], [2, 2])


class TestSuite31PageBoxesSRef(unittest.TestCase):
    """T2.4c: suite31 boxes forward s_ref to both clean and photo grouping."""

    def test_s_ref_chunks_detected_staves(self):
        import build_sys_data as bsd
        staves = []
        y = 0.0
        for _ in range(4):
            staves.append({'x1': 0.0, 'y1': y,
                           'x2': 100.0, 'y2': y + 40.0})
            y += 52.0
        with tempfile.TemporaryDirectory(prefix='t24_sref_') as tmp:
            clean = Path(tmp) / 'clean.png'
            photo = Path(tmp) / 'photo.png'
            Image.new('RGB', (200, 300), (255, 255, 255)).save(clean)
            Image.new('RGB', (200, 300), (255, 255, 255)).save(photo)
            with mock.patch.object(
                    esg.bgate, 'run_system_breaks',
                    side_effect=RuntimeError('no spans')), \
                 mock.patch.object(
                    bsd, 'neural_detect_staves',
                    return_value=(staves, 8.0)):
                got = esg.suite31_page_boxes(
                    str(clean), str(photo), 'dummy.musicxml', s_ref=2)
        self.assertIsNotNone(got)
        clean_boxes, deg_boxes, _src = got
        self.assertEqual(len(clean_boxes), 2)
        self.assertEqual(len(deg_boxes), 2)


class TestSparseBeamRowShed(unittest.TestCase):
    """Step 1 stitcher half: a sparse beamed-note row peaking at staff
    spacing above a staff (fiddle-strathspey, beach-prayer) must shed,
    not skip the page.

    RowNet fires ~1.0 on dense beam rows exactly one staff space above
    the top staff line; every prob-based repair rule misses (no doubled
    copy, no 2x gap, edge prob >= 0.95). The discriminator is ink
    density: an engraved staff line is a continuous rule, a beam row is
    sparse. Runs the real detector on a small synthetic page.
    """

    def _page(self, dashed=True):
        import build_sys_data as bsd
        import build_gate20_sys_dev as bgate
        bgate.ensure_weights_readable()
        from PIL import Image, ImageDraw
        w, h, space = 400, 420, 20
        img = Image.new('RGB', (w, h), (255, 255, 255))
        d = ImageDraw.Draw(img)
        top = 120
        for i in range(5):
            y = top + i * space
            d.line([(0, y), (w - 1, y)], fill=(0, 0, 0), width=2)
        if dashed:
            # Beam row one space above: thick broken beams (16 on / 14
            # off, 4px). RowNet peaks ~1.0 on it (real: 0.985/0.993 on
            # the folk pages); its +-3px max ink density is ~0.58 vs
            # 1.0 for the solid rules (real: 0.54 vs 0.95).
            y = top - space
            for x0 in range(0, w, 30):
                d.line([(x0, y - 2), (min(x0 + 16, w - 1), y - 2)],
                       fill=(0, 0, 0), width=4)
        return img, bsd

    def test_sparse_row_above_staff_sheds_to_five(self):
        img, bsd = self._page(dashed=True)
        import tempfile
        with tempfile.TemporaryDirectory(prefix='t24_shed_') as tmp:
            p = str(Path(tmp) / 'page.png')
            img.save(p)
            try:
                staves, _space = bsd.neural_detect_staves(p)
            except ValueError as e:
                self.fail('sparse beam row skipped the page: %s' % e)
        self.assertEqual(len(staves), 1)

    def test_medium_sparse_row_above_staff_sheds(self):
        # Beach-prayer boundary: annotation density ~0.66 of solid rules
        # (real: 0.57 vs 0.86) with prob ~1.0. Must shed, not skip.
        import build_sys_data as bsd
        import build_gate20_sys_dev as bgate
        bgate.ensure_weights_readable()
        from PIL import Image, ImageDraw
        import tempfile
        w, h, space = 400, 420, 20
        img = Image.new('RGB', (w, h), (255, 255, 255))
        d = ImageDraw.Draw(img)
        top = 120
        for i in range(5):
            d.line([(0, top + i * space), (w - 1, top + i * space)],
                   fill=(0, 0, 0), width=2)
        y = top - space
        for x0 in range(0, w, 30):
            d.line([(x0, y - 2), (min(x0 + 18, w - 1), y - 2)],
                   fill=(0, 0, 0), width=4)
        with tempfile.TemporaryDirectory(prefix='t24_shedmed_') as tmp:
            p = str(Path(tmp) / 'page.png')
            img.save(p)
            try:
                staves, _space = bsd.neural_detect_staves(p)
            except ValueError as e:
                self.fail('medium-sparse beam row skipped the page: %s' % e)
        self.assertEqual(len(staves), 1)

    def test_all_sparse_text_block_discarded(self):
        # Title/composer block (beach-prayer rows 59-179: 8 peaks, best
        # +-3px density ~0.49, no solid rule): the group is text, not a
        # staff — discard it, keep the real staff. Must not raise, and
        # must not shed-or-keep a phantom staff.
        import build_sys_data as bsd
        import build_gate20_sys_dev as bgate
        bgate.ensure_weights_readable()
        from PIL import Image, ImageDraw
        import tempfile
        w, h, space = 400, 640, 20
        img = Image.new('RGB', (w, h), (255, 255, 255))
        d = ImageDraw.Draw(img)
        for i, y in enumerate(range(40, 40 + 8 * 15, 15)):
            for x0 in range(0, w, 26):
                d.line([(x0, y), (min(x0 + 14, w - 1), y)],
                       fill=(0, 0, 0), width=2)
        top = 420
        for i in range(5):
            d.line([(0, top + i * space), (w - 1, top + i * space)],
                   fill=(0, 0, 0), width=2)
        with tempfile.TemporaryDirectory(prefix='t24_title_') as tmp:
            p = str(Path(tmp) / 'page.png')
            img.save(p)
            try:
                staves, _space = bsd.neural_detect_staves(p)
            except ValueError as e:
                self.fail('text block skipped the page: %s' % e)
        self.assertEqual(len(staves), 1)

    def test_solid_sixth_line_still_skips(self):
        # A solid sixth rule is genuinely ambiguous: no false repair.
        import build_sys_data as bsd
        import build_gate20_sys_dev as bgate
        bgate.ensure_weights_readable()
        from PIL import Image, ImageDraw
        import tempfile
        w, h, space = 400, 420, 20
        img = Image.new('RGB', (w, h), (255, 255, 255))
        d = ImageDraw.Draw(img)
        for i in range(6):
            y = 120 + i * space
            d.line([(0, y), (w - 1, y)], fill=(0, 0, 0), width=2)
        with tempfile.TemporaryDirectory(prefix='t24_solid6_') as tmp:
            p = str(Path(tmp) / 'page.png')
            img.save(p)
            with self.assertRaises(ValueError):
                bsd.neural_detect_staves(p)


class TestStitchHeadersFirstNonEmpty(unittest.TestCase):
    """Step 1 stitcher half: page stitch must not drop headers when the
    first system's decode is empty (decode-crash -> ht == [])."""

    def test_heads_come_from_first_nonempty_system(self):
        sys_toks = [[],
                    ['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4',
                     'bar']]
        page = esg.stitch_page_toks(sys_toks)
        self.assertEqual(page[:3], ['clef-G-2', 'key+0', 'time-4-4'])
        self.assertIn('bar', page)

    def test_all_empty_stays_empty(self):
        self.assertEqual(esg.stitch_page_toks([[], []]), [])

    def test_normal_first_system_heads_kept(self):
        sys_toks = [['clef-G-2', 'quarter', 'C', '4', 'bar'],
                    ['clef-G-2', 'half', 'D', '4', 'bar']]
        page = esg.stitch_page_toks(sys_toks)
        self.assertEqual(page[0], 'clef-G-2')
        # Second system's leading headers are dropped (bodies concat).
        self.assertEqual(page.count('clef-G-2'), 1)


class TestDecodeSystemV2KeepsStringHypotheses(unittest.TestCase):
    def test_string_hypothesis_survives_validation(self):
        t = data.TOKEN_TO_ID
        beam_ids = [t['clef-G-2'], t['key+0'], t['time-4-4'],
                    t['quarter'], t['C'], t['4'], t['bar']]
        net = FakeNet(beam_ids)
        with tempfile.TemporaryDirectory(prefix='t24_fix_') as tmp:
            tmpd = Path(tmp)
            img_path = tmpd / 'strip.png'
            Image.new('RGB', (64, 64), (255, 255, 255)).save(img_path)
            staves = [[50.0, 60.0, 70.0, 80.0, 90.0]]
            with mock.patch.object(
                    esg, 'rect_staves_for',
                    return_value=(staves, 'test')), mock.patch.object(
                    esg, 'rectified_input_tensor',
                    return_value=torch.zeros(1, 3, 256, 64)):
                dec = esg.decode_system_v2(
                    net, str(img_path), {'img': 'strip.png'}, None,
                    'cpu', 256, 3, 576, False, tmpd, 'strip-clean-hyp')
            self.assertTrue(dec['vocab_valid'])
            self.assertGreater(len(dec['tokens']), 0)
            self.assertTrue(all(isinstance(x, str) for x in dec['tokens']))
            self.assertEqual(dec['ntokens'], len(dec['tokens']))
            on_disk = (tmpd / 'strip-clean-hyp-bekern.txt').read_text(
                encoding='utf-8').split()
            self.assertEqual(on_disk, dec['tokens'])
            self.assertGreater(len(on_disk), 0)


class TestCanonicalizeBekernTailOrder(unittest.TestCase):
    """Regression tests for ryans-jig & airds-air page-decode-rc-1 crashes."""

    def test_reorders_alter_and_dot_before_note_modifiers(self):
        # Exact failing snippet from ryans-jig s1
        ryans = (
            'clef-G-2 key-2 time-2-4 voice-1 16th A 4 slur-stop stacc '
            'a-1 dot slur-start bar'
        ).split()
        fixed = esg.canonicalize_bekern_tail_order(ryans)
        self.assertEqual(
            fixed,
            'clef-G-2 key-2 time-2-4 voice-1 16th A 4 a-1 dot slur-stop '
            'stacc slur-start bar'.split(),
        )
        # Exact failing snippet from airds-air s1
        airds = (
            'clef-G-2 key+1 time-6-8 voice-1 quarter G 5 slur-stop '
            'stacc a1 bar'
        ).split()
        fixed_a = esg.canonicalize_bekern_tail_order(airds)
        self.assertEqual(
            fixed_a,
            'clef-G-2 key+1 time-6-8 voice-1 quarter G 5 a1 slur-stop '
            'stacc bar'.split(),
        )

    def test_already_canonical_stream_is_unchanged(self):
        canon = (
            'clef-G-2 key+0 time-4-4 voice-1 quarter C 4 a-1 dot '
            'slur-start stacc tie-start rest eighth dot slur-stop bar'
        ).split()
        self.assertEqual(esg.canonicalize_bekern_tail_order(canon), canon)

    def test_stitched_ryans_and_airds_snippets_decode_rc0(self):
        if not ASER_EXE.exists():
            self.skipTest('release binary missing: %s' % ASER_EXE)
        stitched = esg.stitch_page_toks([
            'clef-G-2 key-2 time-2-4 voice-1 16th A 4 slur-stop stacc a-1 dot slur-start bar'.split(),
            'clef-G-2 key+1 time-6-8 voice-1 quarter G 5 slur-stop stacc a1 bar'.split(),
        ])
        with tempfile.TemporaryDirectory(prefix='t_canon_') as tmp:
            txt = Path(tmp) / 'page.txt'
            out = Path(tmp) / 'page.json'
            txt.write_text(' '.join(stitched) + '\n', encoding='utf-8')
            p = esg.run_cli([ASER_EXE, 'bekern-decode', '--input', str(txt),
                             '--out', str(out)])
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertTrue(out.exists())


class TestTrimMaxLenRepeatLoop(unittest.TestCase):
    """Trim infinite verbatim repeating tail ONLY when strip hits max_len."""

    def test_short_repeating_stream_untouched(self):
        bar = ['voice-1', 'quarter', 'C', '4', 'quarter', 'D', '4', 'bar']
        toks = ['clef-G-2', 'key+0', 'time-2-4'] + bar * 5
        self.assertEqual(esg.trim_max_len_repeat_loop(toks, max_len=564), toks)

    def test_max_len_non_repeating_stream_untouched(self):
        import random
        rng = random.Random(42)
        toks = ['clef-G-2', 'key+0', 'time-4-4']
        steps = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        durs = ['quarter', 'eighth', 'half', '16th']
        while len(toks) < 564:
            toks.extend([rng.choice(durs), rng.choice(steps),
                         str(rng.randint(3, 5))])
        toks = toks[:564]
        self.assertEqual(esg.trim_max_len_repeat_loop(toks, max_len=564), toks)

    def test_max_len_verbatim_loop_trimmed_to_one_copy(self):
        prefix = ['clef-G-2', 'key+0', 'time-4-4', 'voice-1', 'half', 'G', '4', 'bar']
        loop_bar = ['voice-1', 'quarter', 'C', '4', 'quarter', 'E', '4',
                    'quarter', 'G', '4', 'quarter', 'C', '5', 'bar']  # 14 toks
        # Repeat until hitting 564 tokens (cut mid-bar at the end)
        toks = list(prefix)
        while len(toks) < 564:
            toks.extend(loop_bar)
        toks = toks[:564]
        trimmed = esg.trim_max_len_repeat_loop(toks, max_len=564)
        self.assertEqual(trimmed, prefix + loop_bar)


class TestDetectShortTrailingStaves(unittest.TestCase):
    """Recover short partial-width final systems without false positives."""

    def test_recovers_short_20pct_final_staff(self):
        import build_gate20_sys_dev as bgate
        from PIL import ImageDraw
        w, h, space = 1000, 1200, 17
        img = Image.new('RGB', (w, h), (255, 255, 255))
        d = ImageDraw.Draw(img)
        full_staves = []
        for idx, top in enumerate((180, 390, 600)):
            for i in range(5):
                y = top + i * space
                d.line([(48, y), (950, y)], fill=(0, 0, 0), width=2)
            full_staves.append({'x1': 48.0, 'y1': float(top),
                                'x2': 950.0, 'y2': float(top + 4 * space)})
        # Short 20% trailing staff at y=800
        for i in range(5):
            y = 800 + i * space
            d.line([(48, y), (250, y)], fill=(0, 0, 0), width=2)
        with tempfile.TemporaryDirectory(prefix='t_short_tail_') as tmp:
            p = str(Path(tmp) / 'page.png')
            img.save(p)
            got = bgate.detect_short_trailing_staves(p, full_staves, float(space))
        self.assertEqual(len(got), 4)
        self.assertAlmostEqual(got[-1]['y1'], 800.0, delta=2.0)
        self.assertLess(got[-1]['x2'], 300.0)

    def test_ignores_dashed_text_footer_below_last_staff(self):
        import build_gate20_sys_dev as bgate
        from PIL import ImageDraw
        w, h, space = 1000, 1200, 17
        img = Image.new('RGB', (w, h), (255, 255, 255))
        d = ImageDraw.Draw(img)
        full_staves = []
        for top in (180, 390, 600):
            for i in range(5):
                y = top + i * space
                d.line([(48, y), (950, y)], fill=(0, 0, 0), width=2)
            full_staves.append({'x1': 48.0, 'y1': float(top),
                                'x2': 950.0, 'y2': float(top + 4 * space)})
        # Dashed footer text (short glyphs with 10px gaps) at y=800..880
        for i in range(5):
            y = 800 + i * space
            for x0 in range(50, 400, 20):
                d.line([(x0, y), (x0 + 8, y)], fill=(0, 0, 0), width=2)
        with tempfile.TemporaryDirectory(prefix='t_footer_') as tmp:
            p = str(Path(tmp) / 'page.png')
            img.save(p)
            got = bgate.detect_short_trailing_staves(p, full_staves, float(space))
        self.assertEqual(len(got), 3)


def _make_suite31_fixture(root, slugs=('p1',)):
    """Build a minimal synthetic suite31 directory + manifest for unit tests."""
    import json
    root.mkdir(parents=True, exist_ok=True)
    pieces = []
    for slug in slugs:
        for sub in ('gt.musicxml', 'clean.png', 'photo.png'):
            p = root / ('%s_%s' % (slug, sub))
            if sub.endswith('.png'):
                Image.new('RGB', (200, 300), (255, 255, 255)).save(p)
            else:
                p.write_text('<score-partwise/>', encoding='utf-8')
        pieces.append({
            'slug': slug,
            'gt': {'file': '%s_gt.musicxml' % slug},
            'clean': {'file': '%s_clean.png' % slug},
            'photo': {'file': '%s_photo.png' % slug},
        })
    (root / 'suite31_manifest.json').write_text(
        json.dumps({'pieces': pieces}), encoding='utf-8')
    return root


def _suite31_args(suite_dir, blind_photo=False, no_notation=False):
    import types
    return types.SimpleNamespace(
        suite31_dir=str(suite_dir),
        suite31_manifest=None,
        pieces=None,
        limit_pieces=None,
        blind_photo=blind_photo,
        use_manifest_boxes=False,
        no_notation=no_notation,
        height=256,
        beam=3,
        max_len=576,
        hybrid=False,
        ink_veto=False,
        ilme_lambda=0.0,
        lattice_repair=False,
        adapt_steps=0,
        adapt_lr=0.1,
        stride='8x16',
    )


class TestNotationF1DecodeFailureZeroed(unittest.TestCase):
    """Decode failure must set notation_f1=0.0 so _mean() never drops it."""

    def test_bekern_decode_failure_zeros_notation_f1_in_mean(self):
        with tempfile.TemporaryDirectory(prefix='t_notf1_') as tmp:
            tmpd = Path(tmp)
            suite = _make_suite31_fixture(tmpd / 'suite', slugs=('ok_p', 'fail_p'))
            workdir = tmpd / 'work'
            workdir.mkdir()
            ref_toks = ['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4', 'bar']
            ref_notes = workdir / 'ref_notes.json'
            ref_notes.write_text('{}', encoding='utf-8')

            def fake_decode_row(net, img_file, row, heat_net, device, height,
                                beam, max_len, hybrid, wd, stem, **kw):
                esg.write_hyp_bekern(wd, stem, ref_toks)
                return {'tokens': ref_toks, 'ntokens': len(ref_toks),
                        'eos_miss': False, 'vocab_valid': True,
                        'decode_secs': 0.01, 'loop_trimmed': False,
                        'geom': 'test'}

            def fake_run_cli(cmd, **kw):
                out_idx = cmd.index('--out') + 1
                out_path = Path(cmd[out_idx])
                # Fail bekern-decode on fail_p photo only
                if 'fail_p-photo-page-hyp-notes.json' in str(out_path):
                    return mock.Mock(returncode=1, stderr='bad token')
                out_path.write_text('{}', encoding='utf-8')
                return mock.Mock(returncode=0, stderr='')

            with mock.patch.object(esg.page_prep, 'prep_page',
                                   side_effect=lambda p: (
                                       Image.open(p).convert('RGB'),
                                       {'skew_deg': 0.0, 'border_crop_box': None,
                                        'applied': False})), \
                 mock.patch.object(esg, 'suite31_page_refs',
                                   return_value=(ref_toks, ref_notes)), \
                 mock.patch.object(esg, 'suite31_page_boxes',
                                   return_value=([[0, 0, 200, 100]],
                                                 [[0, 0, 200, 100]], 'detected')), \
                 mock.patch.object(esg.bhc, '_sysdev_one',
                                   return_value=({'skip': True}, {'skip': True})), \
                 mock.patch.object(esg, 'decode_row_system',
                                   side_effect=fake_decode_row), \
                 mock.patch.object(esg, 'run_cli', side_effect=fake_run_cli), \
                 mock.patch.object(esg, 'score_notes_pair',
                                   return_value={'f1': 0.8, 'ser': 0.2,
                                                 'proxy_fixes': 2,
                                                 'zeroed': False,
                                                 'zero_reason': None}), \
                 mock.patch.object(esg, 'score_notation_bekerns',
                                   return_value={'notation_f1': 0.8,
                                                 'notation_drift': 0.8,
                                                 'notation_zeroed': False,
                                                 'notation_zero_reason': None}):
                rows, summary = esg.run_suite31(
                    _suite31_args(suite, blind_photo=False, no_notation=False),
                    None, None, workdir, 'cpu')

            by_slug = {r['piece']: r for r in rows}
            self.assertEqual(by_slug['fail_p']['photo_f1'], 0.0)
            self.assertEqual(by_slug['fail_p']['photo_notation_f1'], 0.0)
            self.assertTrue(by_slug['fail_p']['photo_notation_zeroed'])
            # Average of 0.8 (ok_p) and 0.0 (fail_p) MUST be 0.4, NOT 0.8
            self.assertAlmostEqual(summary['suite31_photo_notation_f1_mean'], 0.4)
            self.assertAlmostEqual(summary['suite31_clean_notation_f1_mean'], 0.8)


class TestBlindPhotoMode(unittest.TestCase):
    """--blind-photo: no s_ref, no scaled clean fallback, no clean-crop pairing."""

    def test_blind_photo_boxes_never_use_s_ref_or_scaled_fallback(self):
        # 4 staves evenly spaced: s_ref=2 would chunk into 2 groups,
        # whereas s_ref=None (factor=3.5) keeps all 4 in 1 group.
        staves = [{'x1': 0.0, 'y1': float(i * 52), 'x2': 100.0,
                   'y2': float(i * 52 + 40)} for i in range(4)]
        with tempfile.TemporaryDirectory(prefix='t_blind_box_') as tmp:
            clean = Path(tmp) / 'clean.png'
            photo = Path(tmp) / 'photo.png'
            Image.new('RGB', (200, 300), (255, 255, 255)).save(clean)
            Image.new('RGB', (200, 300), (255, 255, 255)).save(photo)

            # Case 1: photo detection succeeds -> s_ref is NOT applied to photo,
            # and mismatched count vs clean_boxes still returns 'detected'.
            with mock.patch.object(esg.bgate, 'run_system_breaks',
                                   side_effect=RuntimeError('no spans')), \
                 mock.patch.object(esg.bgate, 'detect_staves_with_short_tail',
                                   return_value=(staves, 8.0)):
                got = esg.suite31_page_boxes(
                    str(clean), str(photo), 'dummy.musicxml',
                    s_ref=2, blind_photo=True)
            self.assertIsNotNone(got)
            clean_boxes, deg_boxes, src = got
            self.assertEqual(len(clean_boxes), 2)
            self.assertEqual(len(deg_boxes), 1)
            self.assertEqual(src, 'detected')

            # Case 2: photo detection fails -> returns (clean_boxes, None, 'blind-miss'),
            # NEVER 'scaled'.
            def detect_clean_only(path):
                if 'photo' in path:
                    raise ValueError('no staves on photo')
                return staves, 8.0

            with mock.patch.object(esg.bgate, 'run_system_breaks',
                                   side_effect=RuntimeError('no spans')), \
                 mock.patch.object(esg.bgate, 'detect_staves_with_short_tail',
                                   side_effect=detect_clean_only):
                got_miss = esg.suite31_page_boxes(
                    str(clean), str(photo), 'dummy.musicxml',
                    s_ref=2, blind_photo=True)
            self.assertIsNotNone(got_miss)
            self.assertEqual(len(got_miss[0]), 2)
            self.assertIsNone(got_miss[1])
            self.assertEqual(got_miss[2], 'blind-miss')

    def test_blind_photo_run_suite31_scores_miss_zero_and_never_pairs_clean(self):
        with tempfile.TemporaryDirectory(prefix='t_blind_run_') as tmp:
            tmpd = Path(tmp)
            suite = _make_suite31_fixture(tmpd / 'suite', slugs=('hit', 'miss'))
            workdir = tmpd / 'work'
            workdir.mkdir()
            ref_toks = ['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4', 'bar']
            ref_notes = workdir / 'ref_notes.json'
            ref_notes.write_text('{}', encoding='utf-8')
            sysdev_calls = []

            def fake_boxes(cpath, ppath, gtpath, s_ref=None, blind_photo=False):
                self.assertTrue(blind_photo)
                if 'miss' in cpath:
                    return [[0, 0, 200, 100]], None, 'blind-miss'
                return [[0, 0, 200, 100]], [[0, 0, 200, 100]], 'detected'

            def fake_sysdev_one(arg):
                _wd, s = arg
                sysdev_calls.append(dict(s))
                return {'skip': True}, {'skip': True}

            def fake_decode_row(net, img_file, row, heat_net, device, height,
                                beam, max_len, hybrid, wd, stem, **kw):
                esg.write_hyp_bekern(wd, stem, ref_toks)
                return {'tokens': ref_toks, 'ntokens': len(ref_toks),
                        'eos_miss': False, 'vocab_valid': True,
                        'decode_secs': 0.01, 'loop_trimmed': False,
                        'geom': 'test'}

            def fake_run_cli(cmd, **kw):
                out_path = Path(cmd[cmd.index('--out') + 1])
                out_path.write_text('{}', encoding='utf-8')
                return mock.Mock(returncode=0, stderr='')

            with mock.patch.object(esg.page_prep, 'prep_page',
                                   side_effect=lambda p: (
                                       Image.open(p).convert('RGB'),
                                       {'skew_deg': 0.0, 'border_crop_box': None,
                                        'applied': False})), \
                 mock.patch.object(esg, 'suite31_page_refs',
                                   return_value=(ref_toks, ref_notes)), \
                 mock.patch.object(esg, 'suite31_page_boxes',
                                   side_effect=fake_boxes), \
                 mock.patch.object(esg.bhc, '_sysdev_one',
                                   side_effect=fake_sysdev_one), \
                 mock.patch.object(esg, 'decode_row_system',
                                   side_effect=fake_decode_row), \
                 mock.patch.object(esg, 'run_cli', side_effect=fake_run_cli), \
                 mock.patch.object(esg, 'score_notes_pair',
                                   return_value={'f1': 0.9, 'ser': 0.1,
                                                 'proxy_fixes': 1,
                                                 'zeroed': False,
                                                 'zero_reason': None}), \
                 mock.patch.object(esg, 'score_notation_bekerns',
                                   return_value={'notation_f1': 0.9,
                                                 'notation_drift': 0.9,
                                                 'notation_zeroed': False,
                                                 'notation_zero_reason': None}):
                rows, summary = esg.run_suite31(
                    _suite31_args(suite, blind_photo=True, no_notation=False),
                    None, None, workdir, 'cpu')

            by_slug = {r['piece']: r for r in rows}
            # 'miss' is scored (not skipped) with photo=0.0 and deg_box_source='blind-miss'
            self.assertFalse(by_slug['miss'].get('skipped'))
            self.assertEqual(by_slug['miss']['deg_box_source'], 'blind-miss')
            self.assertEqual(by_slug['miss']['photo_f1'], 0.0)
            self.assertEqual(by_slug['miss']['photo_notation_f1'], 0.0)
            self.assertAlmostEqual(summary['blind_photo_detected_rate'], 0.5)
            self.assertAlmostEqual(summary['suite31_photo_f1_mean'], 0.45)
            self.assertAlmostEqual(summary['suite31_photo_notation_f1_mean'], 0.45)
            # Photo strip in 'hit' MUST be paired with itself, never clean crop
            photo_calls = [c for c in sysdev_calls if 'photo' in c['degraded']]
            self.assertEqual(len(photo_calls), 1)
            self.assertEqual(photo_calls[0]['clean'], photo_calls[0]['degraded'])


class TestLoopTrimmedTracking(unittest.TestCase):
    """Repeat-loop trimming must be counted in decode_system_v2 and summary."""

    def test_decode_system_v2_sets_loop_trimmed_flag(self):
        prefix = ['clef-G-2', 'key+0', 'time-4-4', 'voice-1', 'half', 'G', '4', 'bar']
        loop_bar = ['voice-1', 'quarter', 'C', '4', 'quarter', 'E', '4',
                    'quarter', 'G', '4', 'quarter', 'C', '5', 'bar']
        toks = list(prefix)
        while len(toks) < 564:
            toks.extend(loop_bar)
        toks = toks[:564]
        with tempfile.TemporaryDirectory(prefix='t_loop_v2_') as tmp:
            tmpd = Path(tmp)
            img_path = tmpd / 'strip.png'
            Image.new('RGB', (64, 64), (255, 255, 255)).save(img_path)
            staves = [[50.0, 60.0, 70.0, 80.0, 90.0]]
            with mock.patch.object(esg, 'rect_staves_for',
                                   return_value=(staves, 'test')), \
                 mock.patch.object(esg, 'rectified_input_tensor',
                                   return_value=torch.zeros(1, 3, 256, 64)), \
                 mock.patch.object(esg.mdec, 'decode_system_with_measure_fallback',
                                   return_value=toks):
                dec = esg.decode_system_v2(
                    None, str(img_path), {'img': 'strip.png'}, None,
                    'cpu', 256, 3, 576, False, tmpd, 'strip-hyp')
            self.assertTrue(dec['loop_trimmed'])
            self.assertEqual(dec['tokens'], prefix + loop_bar)

            with mock.patch.object(esg, 'rect_staves_for',
                                   return_value=(staves, 'test')), \
                 mock.patch.object(esg, 'rectified_input_tensor',
                                   return_value=torch.zeros(1, 3, 256, 64)), \
                 mock.patch.object(esg.mdec, 'decode_system_with_measure_fallback',
                                   return_value=prefix):
                dec2 = esg.decode_system_v2(
                    None, str(img_path), {'img': 'strip.png'}, None,
                    'cpu', 256, 3, 576, False, tmpd, 'strip-hyp2')
            self.assertFalse(dec2['loop_trimmed'])

    def test_run_suite31_counts_clean_and_photo_loop_trimmed(self):
        with tempfile.TemporaryDirectory(prefix='t_loop_s31_') as tmp:
            tmpd = Path(tmp)
            suite = _make_suite31_fixture(tmpd / 'suite', slugs=('p1',))
            workdir = tmpd / 'work'
            workdir.mkdir()
            ref_toks = ['clef-G-2', 'key+0', 'time-4-4', 'quarter', 'C', '4', 'bar']
            ref_notes = workdir / 'ref_notes.json'
            ref_notes.write_text('{}', encoding='utf-8')

            def fake_decode_row(net, img_file, row, heat_net, device, height,
                                beam, max_len, hybrid, wd, stem, **kw):
                esg.write_hyp_bekern(wd, stem, ref_toks)
                is_photo = '-photo-' in stem
                return {'tokens': ref_toks, 'ntokens': len(ref_toks),
                        'eos_miss': False, 'vocab_valid': True,
                        'decode_secs': 0.01, 'loop_trimmed': is_photo,
                        'geom': 'test'}

            def fake_run_cli(cmd, **kw):
                out_path = Path(cmd[cmd.index('--out') + 1])
                out_path.write_text('{}', encoding='utf-8')
                return mock.Mock(returncode=0, stderr='')

            with mock.patch.object(esg.page_prep, 'prep_page',
                                   side_effect=lambda p: (
                                       Image.open(p).convert('RGB'),
                                       {'skew_deg': 0.0, 'border_crop_box': None,
                                        'applied': False})), \
                 mock.patch.object(esg, 'suite31_page_refs',
                                   return_value=(ref_toks, ref_notes)), \
                 mock.patch.object(esg, 'suite31_page_boxes',
                                   return_value=([[0, 0, 200, 100]],
                                                 [[0, 0, 200, 100], [0, 100, 200, 200]],
                                                 'detected')), \
                 mock.patch.object(esg.bhc, '_sysdev_one',
                                   return_value=({'skip': True}, {'skip': True})), \
                 mock.patch.object(esg, 'decode_row_system',
                                   side_effect=fake_decode_row), \
                 mock.patch.object(esg, 'run_cli', side_effect=fake_run_cli), \
                 mock.patch.object(esg, 'score_notes_pair',
                                   return_value={'f1': 1.0, 'ser': 0.0,
                                                 'proxy_fixes': 0,
                                                 'zeroed': False,
                                                 'zero_reason': None}), \
                 mock.patch.object(esg, 'score_notation_bekerns',
                                   return_value={'notation_f1': 1.0,
                                                 'notation_drift': 1.0,
                                                 'notation_zeroed': False,
                                                 'notation_zero_reason': None}):
                _rows, summary = esg.run_suite31(
                    _suite31_args(suite, blind_photo=True, no_notation=False),
                    None, None, workdir, 'cpu')
            self.assertEqual(summary['clean_loop_trimmed'], 0)
            self.assertEqual(summary['photo_loop_trimmed'], 2)


class TestExtraChans4chAnd3ch(unittest.TestCase):
    """Support 4ch (RGB+staff, no HeatNet) and 3ch (RGB) checkpoints."""

    def test_detect_ckpt_extra_chans(self):
        self.assertEqual(
            esg.detect_ckpt_extra_chans({'encoder.0.0.weight': torch.zeros(96, 3, 4, 4)}),
            0)
        self.assertEqual(
            esg.detect_ckpt_extra_chans({'encoder.0.0.weight': torch.zeros(96, 4, 4, 4)}),
            1)
        self.assertEqual(
            esg.detect_ckpt_extra_chans({'encoder.0.0.weight': torch.zeros(96, 11, 4, 4)}),
            8)
        self.assertIsNone(esg.detect_ckpt_extra_chans({}))

    def test_rectified_input_tensor_4ch_and_3ch_without_heatnet(self):
        rect_img = Image.new('RGB', (64, 32), (200, 200, 200))
        ref = torch.zeros(1, 32, 64)
        ref[0, 10, :] = 1.0

        ten4 = esg.rectified_input_tensor_from_rect(
            rect_img, ref, heat_net=None, device='cpu', extra_chans=1)
        self.assertEqual(tuple(ten4.shape), (1, 4, 32, 64))
        self.assertEqual(float(ten4[0, 3, 10, 0]), 1.0)
        self.assertEqual(float(ten4[0, 3, 0, 0]), 0.0)

        ten3 = esg.rectified_input_tensor_from_rect(
            rect_img, ref, heat_net=None, device='cpu', extra_chans=0)
        self.assertEqual(tuple(ten3.shape), (1, 3, 32, 64))

        fn4 = esg.hybrid_extra_fn(None, 'cpu', extra_chans=1)
        self.assertEqual(tuple(fn4(rect_img, ref).shape), (1, 4, 32, 64))

    def test_decode_system_v2_4ch_no_geometry_returns_zero(self):
        with tempfile.TemporaryDirectory(prefix='t_4ch_nogeom_') as tmp:
            tmpd = Path(tmp)
            img_path = tmpd / 'strip.png'
            Image.new('RGB', (64, 64), (255, 255, 255)).save(img_path)
            with mock.patch.object(esg, 'rect_staves_for',
                                   return_value=(None, 'legacy')):
                dec = esg.decode_system_v2(
                    None, str(img_path), None, None, 'cpu', 256, 3, 576,
                    False, tmpd, 'strip-4ch', extra_chans=1)
            self.assertEqual(dec['geom'], 'no-geometry')
            self.assertEqual(dec['ntokens'], 0)
            self.assertFalse(dec['loop_trimmed'])


if __name__ == '__main__':
    unittest.main()
