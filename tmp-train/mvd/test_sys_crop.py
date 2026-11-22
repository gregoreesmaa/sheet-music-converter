"""Unit tests for glyph-aware system-strip cropping (strip-clip fix).

Uses real pilot numbers (PDMX quota stem, staff_space 20.0, page 2481x3507):
a Tempo mark above the staff, a Lyrics line below it, and a spanning
BarLine must not drag the crop; neighbor content blocks expansion without
ever contracting the base pad box.

Run: ``.\\.venv\\Scripts\\python.exe -m unittest tmp-train/mvd/test_sys_crop.py -v``
"""

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_sys_data as B

SP = 20.0
CHUNK = [{'x1': 800.0, 'y1': 470.0, 'x2': 2360.0, 'y2': 550.0}]
IMG_W, IMG_H = 2481, 3507
TY1 = float('-inf')
TY2 = (550.0 + 726.0) / 2  # mid-gap to next staff top
NEXT_TOP = 726.0

TEMPO = {'bbox': [995.0, 405.0, 1017.0, 458.0], 'class': 'Tempo'}
LYRICS = {'bbox': [828.0, 616.0, 1339.0, 658.0], 'class': 'Lyrics'}
BARLINE = {'bbox': [799.0, 466.0, 803.0, 728.0], 'class': 'BarLine'}
NEIGHBOR_ACC = {'bbox': [1495.0, 699.0, 1509.0, 753.0], 'class': 'Accidental'}


def crop(glyphs, cap_up=None, cap_dn=699.0):
    return B.crop_box_for_system(CHUNK, SP, glyphs, TY1, TY2,
                                 cap_up, cap_dn, IMG_W, IMG_H)


class TestGlyphAwareCrop(unittest.TestCase):
    def test_expands_to_tempo_and_lyrics(self):
        self.assertEqual(crop([TEMPO, LYRICS, BARLINE, NEIGHBOR_ACC]),
                         (770.0, 405.0, 2390.0, 658.0))

    def test_spanning_barline_does_not_drag(self):
        self.assertEqual(crop([BARLINE]), (770.0, 420.0, 2390.0, 600.0))

    def test_neighbor_content_blocks_expansion_without_contraction(self):
        low_neighbor = {'bbox': [1495.0, 650.0, 1509.0, 704.0],
                        'class': 'Accidental'}
        # Lyrics bottom (658) exceeds the free-down-to cap (650): the
        # expansion is refused AND the base box is kept, never shrunk.
        self.assertEqual(crop([TEMPO, LYRICS, BARLINE, low_neighbor],
                              cap_dn=650.0),
                         (770.0, 405.0, 2390.0, 600.0))

    def test_prev_content_blocks_upward_expansion_without_contraction(self):
        prev_hanging = {'bbox': [900.0, 380.0, 1000.0, 430.0],
                        'class': 'Lyrics'}
        self.assertEqual(crop([TEMPO, LYRICS, BARLINE, prev_hanging],
                              cap_up=430.0),
                         (770.0, 420.0, 2390.0, 658.0))

    def test_glyphs_inside_base_change_nothing(self):
        note = {'bbox': [1200.0, 500.0, 1230.0, 530.0], 'class': 'NoteBlack'}
        self.assertEqual(crop([note]), (770.0, 420.0, 2390.0, 600.0))

    def test_clamps_to_image(self):
        edge = [{'x1': 2400.0, 'y1': 470.0, 'x2': 2470.0, 'y2': 550.0}]
        self.assertEqual(
            B.crop_box_for_system(edge, SP, [], TY1, TY2, None, None,
                                  IMG_W, IMG_H),
            (2370.0, 420.0, IMG_W, 600.0))

    def test_degenerate_glyph_y_clamped_to_image(self):
        # Real leland case: page-22 glyphs at y ~103644-207790 on a 3507-px
        # page (runaway SVG transform). Expansion must stay image-bounded
        # instead of producing a 600M-px PIL bomb that kills the bulk build.
        bad = {'bbox': [800.0, 103644.0, 900.0, 207790.0], 'class': 'NoteBlack'}
        self.assertEqual(
            B.crop_box_for_system(CHUNK, SP, [bad], float('-inf'),
                                  float('inf'), None, None, IMG_W, IMG_H),
            (770.0, 420.0, 2390.0, IMG_H))

    def test_degenerate_negative_glyph_y_clamped_to_zero(self):
        bad = {'bbox': [800.0, -90000.0, 900.0, -89000.0], 'class': 'NoteBlack'}
        self.assertEqual(
            B.crop_box_for_system(CHUNK, SP, [bad], float('-inf'),
                                  float('inf'), None, None, IMG_W, IMG_H),
            (770.0, 0.0, 2390.0, 600.0))


class TestCorruptPageSkip(unittest.TestCase):
    """A corrupt render PNG must skip its page, never kill the bulk build
    (quota bulk died at 6995/8000 scores on UnidentifiedImageError)."""

    def test_bomb_crop_returns_none_not_raise(self):
        # A crop box exceeding PIL's decompression-bomb limit (degenerate
        # symbox geometry) must fail the strip soft, never kill the build.
        from PIL import Image
        import tempfile
        tmp = tempfile.mkdtemp()
        png_path = os.path.join(tmp, 't-1.png')
        Image.new('RGB', (100, 100), (255, 255, 255)).save(png_path)
        orig = B.crop_box_for_system
        B.crop_box_for_system = lambda *a: (0.0, 0.0, 2481.0, 300000.0)
        try:
            n = B.emit_system('t', 1, 0, {'start_measure': 1, 'end_measure': 4},
                              [{'x1': 0.0, 'y1': 0.0, 'x2': 90.0, 'y2': 90.0}],
                              20.0, os.path.join(tmp, 't.musicxml'),
                              2481, 3507, tmp, png_path, [], 0.0, 3507.0,
                              None, None)
        finally:
            B.crop_box_for_system = orig
        self.assertIsNone(n)
        # The silent cull must become loud: one jsonl row names the system.
        with open(os.path.join(tmp, 'emit_failures.jsonl')) as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['system'], 't-1-s0')
        self.assertIn('reason', rows[0])

    def test_corrupt_png_returns_skipped_not_raise(self):
        import json
        import tempfile
        import types
        tmp = tempfile.mkdtemp()
        png_path = os.path.join(tmp, 't-1.png')
        with open(png_path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\ntruncated-garbage')
        box_path = os.path.join(tmp, 't-1-boxes.json')
        with open(box_path, 'w') as f:
            json.dump({'png_size': [2481, 3507],
                       'staves': [{'bbox': [800.0, 470.0, 2360.0, 550.0],
                                   'lines': [470.0, 490.0, 510.0, 530.0, 550.0]}],
                       'glyphs': []}, f)
        args = types.SimpleNamespace(renders=tmp, symbox=tmp)
        spans = [{'page': 1, 'start_measure': 1, 'end_measure': 4}]
        jobs, skipped = B.process_page('t', 1, png_path, spans, [1],
                                       [[True, True, True, True]], args, tmp)
        self.assertTrue(skipped)
        self.assertEqual(jobs, [])


if __name__ == '__main__':
    unittest.main()
