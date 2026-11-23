"""Unit tests for multi-engraver rendering & strip extraction (Verovio + LilyPond)."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multi_engrave  # noqa: E402
import rectify  # noqa: E402
import train  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_XML = REPO_ROOT / 'data-gen' / 'score-corpus' / 'big500-renders' / 'ds0001.musicxml'
SUITE31_XML = REPO_ROOT / 'data-gen' / 'score-corpus' / 'suite31' / 'gt' / 'gt-p1-repeats-melody.musicxml'
ASER_BIN = multi_engrave.default_assembly_serialize_bin()


class TestMultiEngrave(unittest.TestCase):
    def _ink_fraction(self, img_path: Path) -> float:
        arr = np.asarray(Image.open(img_path).convert('L'), dtype=np.float32)
        return float((arr < 200.0).mean())

    @unittest.skipUnless(SAMPLE_XML.exists(), 'ds0001.musicxml not present')
    def test_verovio_renders_nonblank_pages_and_exact_strips(self):
        with tempfile.TemporaryDirectory() as td:
            meta = multi_engrave.render_verovio(
                SAMPLE_XML, td, font='Leipzig', width_px=1240,
                extract_strips=True, max_pages=1)
            self.assertEqual(meta['engraver'], 'verovio')
            self.assertEqual(len(meta['pages']), 1)
            p0 = meta['pages'][0]
            self.assertGreaterEqual(len(p0['systems']), 2)
            # Page and strips must contain real ink (not blank white!).
            page_png = Path(td) / p0['png']
            self.assertGreater(self._ink_fraction(page_png), 0.01)
            strip0 = Path(td) / p0['systems'][0]['strip_png']
            self.assertGreater(self._ink_fraction(strip0), 0.01)
            self.assertEqual(p0['systems'][0]['n_staves'], 3)

    @unittest.skipUnless(
        SAMPLE_XML.exists() and all(multi_engrave.find_lilypond_bins()),
        'ds0001.musicxml or LilyPond not present',
    )
    def test_lilypond_renders_nonblank_pages_and_grouped_3staff_systems(self):
        with tempfile.TemporaryDirectory() as td:
            meta = multi_engrave.render_lilypond(
                SAMPLE_XML, td, dpi=200, extract_strips=True, max_pages=1)
            self.assertEqual(meta['engraver'], 'lilypond')
            self.assertEqual(len(meta['pages']), 1)
            p0 = meta['pages'][0]
            self.assertEqual(len(p0['systems']), 3)
            for s in p0['systems']:
                self.assertEqual(s['n_staves'], 3)
                strip_png = Path(td) / s['strip_png']
                self.assertGreater(self._ink_fraction(strip_png), 0.01)

    def test_lilypond_sys_starts_parser_and_rows_to_table(self):
        # Normal multi-system spans.
        spans = multi_engrave._parse_lilypond_sys_starts(
            'SYS_START 1\nSYS_START 6\nSYS_START 12\n', total_measures=16)
        self.assertEqual(spans, [(1, 5), (6, 11), (12, 16)])
        # Anacrusis (bar 0) shifts to 1-based ordinal measure index.
        spans_ana = multi_engrave._parse_lilypond_sys_starts(
            'SYS_START 0\nSYS_START 5\n', total_measures=10)
        self.assertEqual(spans_ana, [(1, 5), (6, 10)])
        # rows_to_table matches hybrid_staff.jsonl table schema.
        tbl = multi_engrave.rows_to_table([[20.0, 30.0, 40.0, 50.0, 60.0]], w_in=100)
        self.assertEqual(tbl['stride'], 8)
        self.assertEqual(tbl['w_in'], 100)
        self.assertEqual(len(tbl['staves']), 1)
        self.assertEqual(len(tbl['staves'][0]), 5)
        self.assertEqual(len(tbl['staves'][0][0]), 14)

    @unittest.skipUnless(
        SUITE31_XML.exists() and ASER_BIN.exists() and all(multi_engrave.find_lilypond_bins()),
        'gt-p1-repeats-melody.musicxml, assembly-serialize, or LilyPond not present',
    )
    def test_suite31_verovio_and_lilypond_emit_training_ready_bekern_and_staff_table(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            v_meta = multi_engrave.render_verovio(
                SUITE31_XML, out, font='Leipzig', width_px=1240,
                extract_strips=True, emit_bekern=True, write_staff_jsonl=True)
            l_meta = multi_engrave.render_lilypond(
                SUITE31_XML, out, dpi=200,
                extract_strips=True, emit_bekern=True, write_staff_jsonl=True)

            # Verify contiguous 1..16 measure coverage for both engravers.
            for label, meta in (('verovio', v_meta), ('lilypond', l_meta)):
                all_sys = [s for p in meta['pages'] for s in p['systems']]
                self.assertGreaterEqual(len(all_sys), 2, f'{label} expected >=2 systems')
                self.assertEqual(all_sys[0]['start_measure'], 1, label)
                self.assertEqual(all_sys[-1]['end_measure'], 16, label)
                for i in range(len(all_sys) - 1):
                    self.assertEqual(
                        all_sys[i + 1]['start_measure'],
                        all_sys[i]['end_measure'] + 1,
                        f'{label} non-contiguous measures between sys {i} and {i + 1}',
                    )
                for s in all_sys:
                    strip_png = out / s['strip_png']
                    self.assertTrue(strip_png.exists(), strip_png)
                    self.assertGreater(self._ink_fraction(strip_png), 0.01)
                    self.assertIsNotNone(s['bekern_txt'], f'{label} bekern missing: {s.get("bekern_error")}')
                    self.assertGreater(s['n_tokens'], 0)
                    bek_path = out / s['bekern_txt']
                    self.assertTrue(bek_path.exists(), bek_path)
                    self.assertEqual(len(bek_path.read_text(encoding='utf-8').split()), s['n_tokens'])

            # Verify unified hybrid_staff.jsonl loads via train.load_hybrid_index and rectifies.
            jsonl_path = out / 'hybrid_staff.jsonl'
            self.assertTrue(jsonl_path.exists())
            idx = train.load_hybrid_index(str(jsonl_path))
            expected_imgs = [
                s['strip_png']
                for meta in (v_meta, l_meta)
                for p in meta['pages']
                for s in p['systems']
            ]
            self.assertEqual(set(idx.keys()), set(expected_imgs))
            for img_name, row in idx.items():
                self.assertEqual(row['S'], 1)
                self.assertEqual(row['kind'], 'rows')
                self.assertFalse(row['skip'])
                im = Image.open(out / img_name).convert('RGB')
                self.assertEqual(row['table']['w_in'], im.width)
                # Strip-local staff coordinates must lie strictly inside strip height.
                ys = [y for st in row['table']['staves'] for line in st for y in line]
                self.assertGreater(min(ys), 0.0)
                self.assertLess(max(ys), float(im.height))
                # End-to-end rectification check (train.py hybrid pipeline compatibility).
                curves = [
                    [np.full(im.width, float(line[0]), dtype=np.float32) for line in st]
                    for st in row['table']['staves']
                ]
                rect_img, _ = rectify.rectify_system_strip(im, curves)
                self.assertEqual(rect_img.height, 256)
                self.assertGreater(rect_img.width, 32)

    def test_lilypond_system_count_mismatch_skips_bekern_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            fake_xml = out / 'mismatch_score.musicxml'
            fake_xml.write_text('<score-partwise/>', encoding='utf-8')
            fake_lp = out / 'lilypond.exe'
            fake_m2l = out / 'musicxml2ly.py'
            fake_lp.write_bytes(b'')
            fake_m2l.write_bytes(b'')

            def fake_subproc(cmd, **kwargs):
                cmd_s = ' '.join(str(c) for c in cmd)
                if 'musicxml2ly' in cmd_s:
                    o_idx = cmd.index('-o')
                    Path(cmd[o_idx + 1]).write_text('\\score { { c\'4 } }\n', encoding='utf-8')
                    return mock.Mock(returncode=0, stdout='', stderr='')
                # LilyPond call: emit 1 page PNG and 3 SYS_START spans.
                o_idx = cmd.index('-o')
                prefix = Path(cmd[o_idx + 1])
                Image.new('RGB', (600, 400), (255, 255, 255)).save(prefix.parent / 'page.png')
                return mock.Mock(
                    returncode=0,
                    stdout='',
                    stderr='SYS_START 1\nSYS_START 5\nSYS_START 9\n',
                )

            # Detector finds only 2 systems vs 3 LilyPond SYS_START spans.
            fake_systems = [
                {'bbox': [20, 20, 580, 140], 'staves': [[50.0, 60.0, 70.0, 80.0, 90.0]],
                 'staff_space_px': 10.0, 'n_staves': 1},
                {'bbox': [20, 200, 580, 320], 'staves': [[230.0, 240.0, 250.0, 260.0, 270.0]],
                 'staff_space_px': 10.0, 'n_staves': 1},
            ]
            stderr_buf = io.StringIO()
            with (
                mock.patch.object(multi_engrave, 'find_lilypond_bins', return_value=(fake_lp, fake_m2l)),
                mock.patch.object(multi_engrave, '_total_measures_from_breaks', return_value=(12, [])),
                mock.patch.object(multi_engrave.subprocess, 'run', side_effect=fake_subproc),
                mock.patch.object(multi_engrave, '_extract_strips_from_clean_png', return_value=fake_systems),
                mock.patch.object(multi_engrave, 'emit_and_validate_bekern') as mock_emit,
                self.assertLogs(level='WARNING') as cm,
                redirect_stderr(stderr_buf),
            ):
                meta = multi_engrave.render_lilypond(
                    fake_xml, out, dpi=200, extract_strips=True, emit_bekern=True,
                    write_staff_jsonl=True,
                )

            self.assertTrue(meta['system_count_mismatch'])
            self.assertEqual(meta['total_detected_systems'], 2)
            self.assertEqual(meta['total_lp_spans'], 3)
            # Loud logging on both logging + stderr.
            self.assertTrue(any('mismatch' in rec.lower() and '2' in rec and '3' in rec for rec in cm.output))
            self.assertIn('mismatch', stderr_buf.getvalue().lower())
            # Must NOT emit any bekern.txt files or call emit_and_validate_bekern.
            mock_emit.assert_not_called()
            self.assertEqual(list(out.glob('*-bekern.txt')), [])
            for s in meta['pages'][0]['systems']:
                self.assertIsNone(s['bekern_txt'])
                self.assertEqual(s['start_measure'], 0)
                self.assertEqual(s['end_measure'], 0)
                self.assertEqual(s['bekern_error'], 'sys-count-mismatch:2!=3')
                self.assertTrue(s['staff_row']['skip'])


if __name__ == '__main__':
    unittest.main()
