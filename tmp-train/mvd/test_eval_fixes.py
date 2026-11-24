"""Unit tests for the 4 zero-training eval/stitching/voice-merge fixes:

1. Mid-page header propagation in ``eval_sys_gate.stitch_page_toks``.
2. ``voice-13+`` OOV cap in ``measure_decode._shift_voices`` /
   ``merge_chunk_streams`` / ``decode_multi_staff_system`` /
   ``eval_sys_gate.decode_curved_system``.
3. Per-staff voice ID preservation across bar-rest chunks (``aloha-oe``).
4. ``eval_sys_gate.suite31_page_refs`` full-page conversion for
   ``gt-p1-*.musicxml`` (including real 34-bar check on
   ``gt-p1-luca-gloria.musicxml``).

Run (CPU-only):
    .\\.venv\\Scripts\\python.exe -m unittest tmp-train/mvd/test_eval_fixes.py -v
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data  # noqa: E402
import eval_sys_gate as esg  # noqa: E402
import measure_decode as mdec  # noqa: E402
import staff_unet2d as su  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASER_EXE = (REPO_ROOT / 'converter' / 'assembly-serialize' / 'target'
            / 'release' / 'assembly-serialize.exe')
LUCA_GLORIA_GT = (REPO_ROOT / 'data-gen' / 'score-corpus' / 'suite31'
                  / 'gt' / 'gt-p1-luca-gloria.musicxml')


class TestFix1MidPageHeaderPropagation(unittest.TestCase):
    """Fix 1: stitch_page_toks emits changed mid-page headers and strips unchanged ones."""

    def test_unchanged_courtesy_headers_stripped(self):
        s0 = 'clef-G-2 key-1 time-3-4 voice-1 half C 4 quarter D 4 bar'.split()
        s1 = 'clef-G-2 key-1 time-3-4 voice-1 half E 4 quarter F 4 bar'.split()
        stitched = esg.stitch_page_toks([s0, s1])
        self.assertEqual(stitched.count('clef-G-2'), 1)
        self.assertEqual(stitched.count('key-1'), 1)
        self.assertEqual(stitched.count('time-3-4'), 1)
        self.assertEqual(
            stitched,
            'clef-G-2 key-1 time-3-4 voice-1 half C 4 quarter D 4 bar '
            'voice-1 half E 4 quarter F 4 bar'.split(),
        )

    def test_changed_key_time_clef_propagated_and_updates_active_state(self):
        s0 = 'clef-G-2 key+0 time-4-4 voice-1 whole C 4 bar'.split()
        s1 = 'clef-G-2 key+2 time-3-4 voice-1 half D 4 quarter E 4 bar'.split()
        # System 2 repeats the new key+2 / time-3-4 courtesy header -> must NOT re-emit
        s2 = 'clef-G-2 key+2 time-3-4 voice-1 half F 4 quarter G 4 bar'.split()
        # System 3 changes clef only
        s3 = 'clef-F-4 key+2 time-3-4 voice-1 half A 3 quarter B 3 bar'.split()
        stitched = esg.stitch_page_toks([s0, s1, s2, s3])
        self.assertEqual(
            stitched,
            (
                'clef-G-2 key+0 time-4-4 voice-1 whole C 4 bar '
                'key+2 time-3-4 voice-1 half D 4 quarter E 4 bar '
                'voice-1 half F 4 quarter G 4 bar '
                'clef-F-4 voice-1 half A 3 quarter B 3 bar'
            ).split(),
        )

    def test_multistaff_header_change_deduplicated_and_staff_clef_change_kept(self):
        s0 = (
            'clef-G-2 key+0 time-4-4 clef-F-4 key+0 time-4-4 '
            'voice-1 whole C 4 voice-2 whole C 3 bar'
        ).split()
        # Both staves change to key-2 and time-3-4; staff 1 clef changes F-4 -> G-2
        s1 = (
            'clef-G-2 key-2 time-3-4 clef-G-2 key-2 time-3-4 '
            'voice-1 half D 4 quarter E 4 voice-2 half G 3 quarter A 3 bar'
        ).split()
        stitched = esg.stitch_page_toks([s0, s1])
        # Changed header between bars: clef-G-2, key-2, time-3-4 (each once)
        bar0_idx = stitched.index('bar')
        after_bar0 = stitched[bar0_idx + 1:bar0_idx + 4]
        self.assertEqual(after_bar0, ['clef-G-2', 'key-2', 'time-3-4'])
        self.assertEqual(stitched.count('key-2'), 1)
        self.assertEqual(stitched.count('time-3-4'), 1)

    def test_mid_body_change_prevents_duplicate_on_next_system(self):
        # Mid-system 0 time change to time-6-8; system 1 courtesy header has time-6-8
        s0 = (
            'clef-G-2 key+0 time-4-4 voice-1 whole C 4 bar '
            'time-6-8 voice-1 half D 4 dot bar'
        ).split()
        s1 = 'clef-G-2 key+0 time-6-8 voice-1 half E 4 dot bar'.split()
        stitched = esg.stitch_page_toks([s0, s1])
        self.assertEqual(stitched.count('time-6-8'), 1)

    def test_empty_system_between_nonempty_preserves_header_tracking(self):
        s0 = 'clef-G-2 key+0 time-4-4 voice-1 whole C 4 bar'.split()
        s1 = []
        s2 = 'clef-G-2 key+1 time-4-4 voice-1 whole D 4 bar'.split()
        stitched = esg.stitch_page_toks([s0, s1, s2])
        self.assertEqual(
            stitched,
            'clef-G-2 key+0 time-4-4 voice-1 whole C 4 bar '
            'key+1 voice-1 whole D 4 bar'.split(),
        )

    def test_stitched_midpage_changes_decode_rc0_in_aser(self):
        if not ASER_EXE.exists():
            self.skipTest('release binary missing: %s' % ASER_EXE)
        stitched = esg.stitch_page_toks([
            'clef-G-2 key+0 time-4-4 voice-1 whole C 4 bar'.split(),
            'clef-G-2 key-2 time-3-4 voice-1 half D 4 quarter E 4 bar'.split(),
            'clef-F-4 key-2 time-3-4 voice-1 half G 3 quarter A 3 bar'.split(),
        ])
        with tempfile.TemporaryDirectory(prefix='t_fix1_') as tmp:
            txt = Path(tmp) / 'page.txt'
            out = Path(tmp) / 'page.json'
            txt.write_text(' '.join(stitched) + '\n', encoding='utf-8')
            p = esg.run_cli([ASER_EXE, 'bekern-decode', '--input', str(txt),
                             '--out', str(out)])
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertTrue(out.exists())


class TestFix2Voice13CapNoOovWipeout(unittest.TestCase):
    """Fix 2: cap/drop overflow voices > voice-12 so HW mega-systems never wipe to ntok=0."""

    def test_shift_voices_drops_only_overflow_voices(self):
        meas = (
            'voice-1 whole C 4 voice-2 whole D 4 '
            'voice-3 whole E 4 voice-4 whole F 4'
        ).split()
        # Shift by +10: voice-11, voice-12 kept; voice-13, voice-14 dropped.
        shifted = mdec._shift_voices(meas, 10, max_voice=12)
        self.assertEqual(
            shifted,
            'voice-11 whole C 4 voice-12 whole D 4'.split(),
        )
        # Shift by +12: entire measure overflows -> empty list.
        self.assertEqual(mdec._shift_voices(meas, 12, max_voice=12), [])
        # Bare run with shift >= 12 -> empty list.
        self.assertEqual(mdec._shift_voices(['whole', 'C', '4'], 12, max_voice=12), [])

    def test_merge_chunk_streams_20_staff_mega_system_stays_valid(self):
        # Simulate 20-staff HW mega-system (5 chunks of 4 staves each)
        chunks = []
        for _ in range(5):
            chunks.append(
                (
                    'clef-G-2 key+0 time-4-4 '
                    'voice-1 whole C 4 voice-2 whole D 4 '
                    'voice-3 whole E 4 voice-4 whole F 4 bar'
                ).split()
            )
        merged = mdec.merge_chunk_streams(chunks, chunk_staff_counts=[4, 4, 4, 4, 4])
        self.assertTrue(esg.hyp_tokens_valid(merged))
        voices = [t for t in merged if t.startswith('voice-')]
        self.assertEqual(voices, ['voice-%d' % i for i in range(1, 13)])
        self.assertFalse(any(int(v.split('-')[1]) > 12 for v in voices))

    def test_decode_curved_system_20_staves_preserves_12_voices_and_vocab_valid(self):
        page_img = Image.new('RGB', (400, 1600), (255, 255, 255))
        staves = []
        for s in range(20):
            yc = 50.0 + s * 70.0
            x_ctrl = np.linspace(20.0, 380.0, 4, dtype=np.float64)
            y_ctrl = np.full(4, yc, dtype=np.float64)
            staves.append(su.CurvedStaff(
                ctrl_pts=np.stack([x_ctrl, y_ctrl], axis=-1),
                sp_ctrl=np.full(4, 10.0, dtype=np.float64),
                x_start=20.0, x_end=380.0,
                sp_median=10.0, confidence=0.95))
        csys20 = su.CurvedSystem(
            staves=staves, x_start=20.0, x_end=380.0,
            polygon=np.zeros((8, 2), dtype=np.float64), confidence=0.95)
        chunk_tok = (
            'clef-G-2 key+0 time-4-4 '
            'voice-1 whole C 4 voice-2 whole D 4 '
            'voice-3 whole E 4 voice-4 whole F 4 bar'
        ).split()
        calls = {'n': 0}

        def fake_fallback(net, ten, beam=3, mem_mask=None, **kw):
            calls['n'] += 1
            return list(chunk_tok)

        with tempfile.TemporaryDirectory(prefix='t_fix2_csys20_') as tmp:
            tmpd = Path(tmp)
            with mock.patch.object(
                    esg.mdec, 'decode_system_with_measure_fallback',
                    side_effect=fake_fallback):
                dec = esg.decode_curved_system(
                    None, page_img, csys20, heat_net=None, device='cpu',
                    height=256, beam=3, max_len=576, workdir=tmpd,
                    stem='hw20-hyp', extra_chans=1)
        # Only first 3 chunks (12 staves) decoded; chunks 4 & 5 skipped cleanly
        self.assertEqual(calls['n'], 3)
        self.assertTrue(dec['vocab_valid'])
        self.assertGreater(dec['ntokens'], 0)
        voices = [t for t in dec['tokens'] if t.startswith('voice-')]
        self.assertEqual(voices, ['voice-%d' % i for i in range(1, 13)])


class TestFix3PerStaffVoiceIdsAcrossBarRestChunks(unittest.TestCase):
    """Fix 3: chunk 1 shifts by max(max_voice_in_chunk0, n_staves_in_chunk0)."""

    def test_aloha_oe_collapsed_bar_rest_chunk0_preserves_voice_5_6_7(self):
        # 7-staff aloha-oe: chunk 0 has 4 staves (bar-rests collapsed to 1 voice),
        # chunk 1 has 3 staves (voice-1 bar-rest, voice-2 melody, voice-3 bass).
        chunk0 = (
            'clef-G-2 key+1 time-4-4 clef-G-2 key+1 time-4-4 '
            'clef-G-2 key+1 time-4-4 clef-F-4 key+1 time-4-4 '
            'voice-1 bar-rest bar '
            'bar-rest bar'
        ).split()
        chunk1 = (
            'clef-G-2 key+1 time-4-4 clef-G-2 key+1 time-4-4 '
            'clef-F-4 key+1 time-4-4 '
            'voice-1 bar-rest voice-2 whole B 4 voice-3 whole G 2 bar '
            'voice-1 bar-rest voice-2 whole D 4 voice-3 whole G 2 bar '
            'voice-1 bar-rest voice-2 whole E 4 voice-3 whole A 2 bar'
        ).split()
        merged = mdec.merge_chunk_streams([chunk0, chunk1], chunk_staff_counts=[4, 3])
        self.assertTrue(esg.hyp_tokens_valid(merged))
        _hdr, measures, _tb = mdec.split_measures(merged)
        self.assertEqual(len(measures), 3)
        # Measure 0: chunk 0 had 1 voice on 4 staves -> chunk 1 shifted by +4 (voice-5, 6, 7)
        self.assertEqual(
            [t for t in measures[0] if t.startswith('voice-')],
            ['voice-1', 'voice-5', 'voice-6', 'voice-7'],
        )
        # Measure 1: bare bar-rest in chunk 0 -> voice-1; chunk 1 still voice-5, 6, 7
        self.assertEqual(
            [t for t in measures[1] if t.startswith('voice-')],
            ['voice-1', 'voice-5', 'voice-6', 'voice-7'],
        )
        # Measure 2: chunk 0 had 0 measures left -> chunk 1 STILL shifts by +4 (voice-5, 6, 7)
        self.assertEqual(
            [t for t in measures[2] if t.startswith('voice-')],
            ['voice-5', 'voice-6', 'voice-7'],
        )

    def test_polyphonic_chunk0_more_voices_than_staves_never_collides(self):
        # Chunk 0 has 4 staves but 5 voices in measure 0 -> chunk 1 shifts by max(5, 4) = 5.
        chunk0 = (
            'clef-G-2 key+0 time-4-4 '
            'voice-1 whole C 5 voice-2 whole G 4 voice-3 whole E 4 '
            'voice-4 whole C 4 voice-5 whole C 3 bar'
        ).split()
        chunk1 = (
            'clef-F-4 key+0 time-4-4 '
            'voice-1 whole G 2 voice-2 whole C 2 bar'
        ).split()
        merged = mdec.merge_chunk_streams([chunk0, chunk1], chunk_staff_counts=[4, 2])
        _hdr, measures, _tb = mdec.split_measures(merged)
        self.assertEqual(
            [t for t in measures[0] if t.startswith('voice-')],
            ['voice-1', 'voice-2', 'voice-3', 'voice-4', 'voice-5',
             'voice-6', 'voice-7'],
        )

    def test_aloha_oe_merged_stream_decodes_rc0_in_aser(self):
        if not ASER_EXE.exists():
            self.skipTest('release binary missing: %s' % ASER_EXE)
        chunk0 = 'clef-G-2 key+1 time-4-4 voice-1 bar-rest bar'.split()
        chunk1 = (
            'clef-G-2 key+1 time-4-4 '
            'voice-1 bar-rest voice-2 whole B 4 voice-3 whole G 2 bar'
        ).split()
        merged = mdec.merge_chunk_streams([chunk0, chunk1], chunk_staff_counts=[4, 3])
        with tempfile.TemporaryDirectory(prefix='t_fix3_') as tmp:
            txt = Path(tmp) / 'aloha.txt'
            out = Path(tmp) / 'aloha.json'
            txt.write_text(' '.join(merged) + '\n', encoding='utf-8')
            p = esg.run_cli([ASER_EXE, 'bekern-decode', '--input', str(txt),
                             '--out', str(out)])
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertTrue(out.exists())


class TestFix4Suite31PageRefsNoGtP1Truncation(unittest.TestCase):
    """Fix 4: gt-p1-*.musicxml converts full file instead of truncating at --page 1."""

    def test_real_luca_gloria_preserves_all_34_bars(self):
        if not ASER_EXE.exists():
            self.skipTest('release binary missing: %s' % ASER_EXE)
        if not LUCA_GLORIA_GT.exists():
            self.skipTest('fixture missing: %s' % LUCA_GLORIA_GT)
        with tempfile.TemporaryDirectory(prefix='t_fix4_luca_') as tmp:
            toks, page_notes = esg.suite31_page_refs(
                LUCA_GLORIA_GT, 's31-luca-gloria', Path(tmp))
            bar_like = [
                t for t in toks
                if t == 'bar' or t in ('bar-double', 'bar-light-heavy',
                                       'bar-repeat-right')
            ]
            self.assertEqual(len(bar_like), 34,
                             'expected all 34 bars of gt-p1-luca-gloria, got %d'
                             % len(bar_like))
            self.assertGreater(len(toks), 900)
            self.assertTrue(page_notes.exists())

    def test_gt_p1_uses_measures_span_and_raw_score_keeps_page_1(self):
        seen_cmds = []

        def fake_run_cli(cmd, **kw):
            seen_cmds.append(list(cmd))
            if 'bekern' in cmd:
                out_dir = Path(cmd[cmd.index('--out') + 1])
                stem = cmd[cmd.index('--stem') + 1]
                (out_dir / (stem + '-bekern.txt')).write_text(
                    'clef-G-2 key+0 time-4-4 whole C 4 bar\n',
                    encoding='utf-8')
            elif 'bekern-decode' in cmd:
                out_path = Path(cmd[cmd.index('--out') + 1])
                out_path.write_text('{}', encoding='utf-8')
            elif 'import' in cmd:
                out_dir = Path(cmd[cmd.index('--out') + 1])
                inp = Path(cmd[cmd.index('--input') + 1])
                (out_dir / (inp.stem + '.json')).write_text('{}', encoding='utf-8')
            return mock.Mock(returncode=0, stderr='')

        with tempfile.TemporaryDirectory(prefix='t_fix4_cli_') as tmp:
            tmpd = Path(tmp)
            with mock.patch.object(esg, 'run_cli', side_effect=fake_run_cli):
                esg.suite31_page_refs(tmpd / 'gt-p1-foo.musicxml', 'foo', tmpd)
                esg.suite31_page_refs(tmpd / 'raw-multipage.musicxml', 'raw', tmpd)
        bekern_cmds = [c for c in seen_cmds if 'bekern' in c]
        self.assertEqual(len(bekern_cmds), 2)
        self.assertIn('--measures', bekern_cmds[0])
        self.assertNotIn('--page', bekern_cmds[0])
        self.assertIn('--page', bekern_cmds[1])
        self.assertEqual(bekern_cmds[1][bekern_cmds[1].index('--page') + 1], '1')
        self.assertNotIn('--measures', bekern_cmds[1])


if __name__ == '__main__':
    unittest.main()
