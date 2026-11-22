"""Cache-eviction parity: cached staff tables vs the live RowNet path.

AGY verdict ranks this Day-0: strips decoded with cached geometry tables
score F1 0.66-1.0 (D:/tmp/eyeball90 + eyeball90swap per-strip JSONs) while
the live RowNet-fallback path scores 0.0. Until the geometry contract is
unified, all other streams' metrics are confounded by silent zeros.

Two layers (both CPU-only, no checkpoints, no training):

1. ``TestRectStavesContract`` -- pins ``eval_sys_gate.rect_staves_for``
   routing with a stubbed detector: cached table wins on width match,
   width mismatch re-runs live detection, detector failure yields legacy.
2. ``TestCacheEvictionParity`` -- the agy validation itself: on
   high-scoring in-train strips, evict the cache row (``row=None``).
   Result: all live detectors fail outright (see class docstring), so
   the 0.0-F1 root cause is detection blindness, not a coordinate
   mismatch; the tests pin the resulting fail-loud contract.

Run: ``.\\.venv\\Scripts\\python.exe -m unittest tmp-train/mvd/test_cache_eviction_parity.py -v``
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_sys_gate as esg  # noqa: E402
import strip_staff as sst  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
T32_DIR = REPO_ROOT / 'tmp-train' / 'mvd-sysdata-t32'
CACHE_JSONL = Path('D:/tmp/t32hybrid-full/hybrid_staff.jsonl')

# High-scoring in-train strips (table path F1 from D:/tmp/eyeball90):
# lc5015521 degraded F1 1.0, lc5069033 degraded F1 0.8623.
PARITY_STRIPS = [
    'lc5015521_Petaluma-8-s0_d0.jpg',
    'lc5069033_Gonville-2-s0_d0.jpg',
]

def _flat_table(w_in, rows_per_staff):
    n = len(range(0, w_in, 8)) + (0 if (w_in - 1) % 8 == 0 else 1)
    return {'stride': 8, 'w_in': w_in,
            'staves': [[[round(float(y), 2)] * n for y in staff]
                       for staff in rows_per_staff]}


def _blank_png(path, size=(160, 60)):
    Image.new('RGB', size, (255, 255, 255)).save(path)


class TestRectStavesContract(unittest.TestCase):
    def test_table_hit_on_width_match(self):
        with unittest.mock.patch.object(
                esg.Image, 'open',
                return_value=mock.Mock(size=(160, 60))):
            row = {'table': _flat_table(160, [[10., 20., 30., 40., 50.]])}
            staves, geom = esg.rect_staves_for('dummy.png', row)
        self.assertEqual(geom, 'table')
        self.assertEqual(len(staves), 1)
        self.assertEqual(len(staves[0]), 5)

    def test_width_mismatch_reruns_live_detection(self):
        sentinel = [[[10.0 + 5.0 * j] * 160 for j in range(5)]]
        with unittest.mock.patch.object(
                esg.Image, 'open',
                return_value=mock.Mock(size=(160, 60))):
            with mock.patch.object(esg.bhc, 'clean_lines',
                                   return_value=(1, sentinel, 10.0, 'x')):
                row = {'table': _flat_table(999, [[10., 20., 30., 40., 50.]])}
                staves, geom = esg.rect_staves_for('dummy.png', row)
        self.assertEqual(geom, 'rownet-fallback')
        self.assertIs(staves, sentinel)

    def test_non_monotonic_fallback_geometry_yields_legacy(self):
        # Live blind fallback with a wrong staff count returns garbage
        # (measured 2026-11-21 on lc5069033_Gonville-2-s0_d0: S=1 vs
        # true S=3, rows [286.2, 675.3, 241.7, 253.3, 254.5]).
        # Decoding against it rectifies noise and scores a silent 0.0;
        # the fallback policy must reject it to legacy/no-geometry.
        garbage = [[286.2, 675.3, 241.7, 253.3, 254.5]]
        with unittest.mock.patch.object(
                esg.Image, 'open',
                return_value=mock.Mock(size=(160, 60))):
            with mock.patch.object(esg.bhc, 'clean_lines',
                                   return_value=(1, garbage, 5.0, 'stripstaff')):
                staves, geom = esg.rect_staves_for('dummy.png', None)
        self.assertEqual((staves, geom), (None, 'legacy'))

    def test_detector_failure_yields_legacy(self):
        with unittest.mock.patch.object(
                esg.Image, 'open',
                return_value=mock.Mock(size=(160, 60))):
            with mock.patch.object(esg.bhc, 'clean_lines',
                                   side_effect=ValueError('no lines')):
                staves, geom = esg.rect_staves_for('dummy.png', None)
        self.assertEqual((staves, geom), (None, 'legacy'))

    def test_no_row_attempts_live_detection(self):
        sentinel = [[[2.0 + 5.0 * j] * 160 for j in range(5)]]
        with unittest.mock.patch.object(
                esg.Image, 'open',
                return_value=mock.Mock(size=(160, 60))):
            with mock.patch.object(esg.bhc, 'clean_lines',
                                   return_value=(1, sentinel, 10.0, 'x')) as m:
                staves, geom = esg.rect_staves_for('dummy.png', None)
        m.assert_called_once_with('dummy.png')
        self.assertEqual(geom, 'rownet-fallback')


class TestCacheEvictionParity(unittest.TestCase):
    """Diagnosis (2026-11-21, D:/tmp/parity_diag.json): evicting the cache
    row on high-scoring degraded strips does NOT produce mismatched
    geometry -- it produces NO geometry at all. RowNet peaks are
    0.0001-0.0012 (threshold 0.3, zero rows above floor), the blind
    ``strip_staff`` detector fails even with the true staff count
    (sp candidates fail / coverage 0.51 < 0.6 floor), and the
    layout-staff classical fallback finds no complete staves. The
    cached marker-transferred table is the only viable geometry source,
    so the 0.0-F1 root cause is detection blindness, not a coordinate
    mismatch. These tests pin the fail-loud contract: eviction must
    surface as an explicit ``ValueError`` (recorded downstream as a
    tagged no-geometry zero), never as silent wrong-geometry decode.
    """

    @classmethod
    def setUpClass(cls):
        if not CACHE_JSONL.exists():
            raise unittest.SkipTest('no hybrid cache at %s' % CACHE_JSONL)
        try:
            import build_gate20_sys_dev as bgate
            bgate.ensure_weights_readable()
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest('RowNet weights unavailable: %s' % e)
        import build_hybrid_cache as bhc
        cls.bhc = bhc
        cls.rows = {}
        with open(CACHE_JSONL, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                if r.get('img') in PARITY_STRIPS and r.get('table'):
                    cls.rows[r['img']] = r
        missing = [s for s in PARITY_STRIPS
                   if s not in cls.rows or not (T32_DIR / s).exists()]
        if missing:
            raise unittest.SkipTest('parity strips missing: %s' % missing)

    def test_evicted_cache_neural_path_is_fully_blind(self):
        # RowNet emits zero peaks above threshold on both degraded
        # strips (probs_max 0.0001 / 0.0012 vs threshold 0.3).
        import build_sys_data as bsd
        from PIL import Image as PILImage
        for stem in PARITY_STRIPS:
            gray = PILImage.open(T32_DIR / stem).convert('L')
            probs = bsd.neural_staff_rows(gray, *gray.size)
            self.assertEqual(int((probs > bsd.NEURAL_THRESHOLD).sum()), 0,
                             stem)

    def test_evicted_cache_rect_routing_is_legacy_or_no_geometry(self):
        # Eviction must never route to a garbage-geometry decode: strip 1
        # raises inside clean_lines (-> legacy), strip 2 returns
        # non-monotonic blind rows (-> legacy after validation).
        for stem in PARITY_STRIPS:
            staves, geom = esg.rect_staves_for(str(T32_DIR / stem), None)
            self.assertIsNone(staves, stem)
            self.assertEqual(geom, 'legacy', stem)

    def test_cached_table_geometry_is_self_consistent(self):
        import numpy as np
        for stem in PARITY_STRIPS:
            row = self.rows[stem]
            curves = sst.table_to_curves(row['table'])
            self.assertEqual(len(curves), int(row['S']), stem)
            w = int(row['table']['w_in'])
            for staff in curves:
                self.assertEqual(len(staff), 5, stem)
                for line in staff:
                    self.assertEqual(len(line), w, stem)
                    self.assertTrue(np.all(np.isfinite(line)), stem)


if __name__ == '__main__':
    unittest.main()
