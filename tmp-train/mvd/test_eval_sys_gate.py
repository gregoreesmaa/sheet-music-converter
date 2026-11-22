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


if __name__ == '__main__':
    unittest.main()
