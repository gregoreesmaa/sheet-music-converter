"""Unit tests for multi-engraver rendering & strip extraction (Verovio + LilyPond)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multi_engrave  # noqa: E402

SAMPLE_XML = (
    Path(__file__).resolve().parents[2]
    / 'data-gen' / 'score-corpus' / 'big500-renders' / 'ds0001.musicxml'
)


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


if __name__ == '__main__':
    unittest.main()
