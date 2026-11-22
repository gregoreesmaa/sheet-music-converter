"""Focused regression tests for build_hybrid_cache._sidecar_one.

Contract under test: _sidecar_one "raises nothing; skips as rows".
A degenerate staff table (collapsed line geometry) made rectify attempt
a 1.27M-px-wide canvas, raising numpy _ArrayMemoryError (a MemoryError)
that escaped the (ValueError, TypeError)-only handler and killed the
whole sidecars stage (T2.4 chain, sq7224846-41-s3_d0.jpg).
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import build_hybrid_cache as B


def _const_table(w_in=64, y0=10.0):
    """Valid stride-8 table: S=1, five constant lines (9 points at 64px)."""
    n = B._table_npts(w_in)
    return {'w_in': w_in, 'stride': 8,
            'staves': [[[y0] * n] * 5]}


def _row(img_path, table):
    return {'img': os.path.basename(img_path), 'img_path': img_path,
            'kind': 'rows', 'S': 1, 'sp': 10.0, 'skip': False,
            'table': table}


class TestInnerPeaksStencil(unittest.TestCase):
    """The ±2 peak stencil must never read out of bounds (or wrap around).

    Production trigger: a tall leland strip (h = 3319) whose bottom-edge
    rows satisfied the peak prefix evaluated probs[h] and killed the
    whole markers stage with IndexError; row 1 silently read probs[-1].
    """

    def test_interior_peak_found(self):
        probs = [0.0] * 5 + [0.9] + [0.0] * 4  # peak at 5, n = 10
        self.assertEqual(B._inner_peaks(probs, 0, 9, 0.3), [5])

    def test_bottom_edge_rows_cannot_crash(self):
        probs = [0.0] * 3316 + [0.9, 0.1, 0.95]  # n = 3319
        self.assertEqual(B._inner_peaks(probs, 0, 3318, 0.3), [])

    def test_first_row_excluded_no_wrap_read(self):
        probs = [0.0, 0.9, 0.0, 0.0, 0.0]  # n = 5
        self.assertEqual(B._inner_peaks(probs, 0, 4, 0.3), [])


class TestSidecarOneNeverRaises(unittest.TestCase):
    def test_memory_error_becomes_skip(self):
        """A MemoryError inside rectify must become a skip row, not a crash.

        Uses a raising stub for rectify_system_strip: provoking a real
        multi-GB allocation in a unit test would be irresponsible; the
        production trigger (median staff space ~0.01px -> 1.27M-px
        canvas) is documented above and reproduced in the T2.4 log.
        """
        import rectify as R
        real = R.rectify_system_strip

        def boom(img, staves, target_sp=10.0, height=256):
            raise MemoryError("simulated degenerate-geometry blow-up")
        R.rectify_system_strip = boom
        try:
            with tempfile.TemporaryDirectory() as tmp:
                from PIL import Image
                p = os.path.join(tmp, 's.png')
                Image.new('RGB', (64, 32), (255, 255, 255)).save(p)
                table = _const_table()
                _, status, reason = B._sidecar_one((tmp, _row(p, table),
                                                      256, 10.0))
                self.assertEqual(status, 'skip')
                self.assertIn('rectify:', reason)
        finally:
            R.rectify_system_strip = real

    def test_absurd_width_becomes_skip(self):
        """A table that rectifies to > MAX_SIDECAR_WIDTH must skip.

        Production trigger: ds0402-1-s2_d2 rectified to a 162301-px
        canvas (median space ~0.08px) and the batch OOM'd the trainer.
        The stub returns a wide-but-cheap canvas (PIL allocates only
        9000x256x3 B); the point under test is the cap, not rectify.
        """
        import rectify as R
        from PIL import Image
        real = R.rectify_system_strip

        def wide(img, staves, target_sp=10.0, height=256):
            big = Image.new('RGB', (9000, height), (255, 255, 255))
            return big, torch.zeros(1, height, 9000)

        R.rectify_system_strip = wide
        try:
            with tempfile.TemporaryDirectory() as tmp:
                p = os.path.join(tmp, 's.png')
                Image.new('RGB', (64, 32), (255, 255, 255)).save(p)
                table = _const_table()
                _, status, reason = B._sidecar_one((tmp, _row(p, table),
                                                   256, 10.0))
                self.assertEqual(status, 'skip')
                self.assertIn('width', reason)
        finally:
            R.rectify_system_strip = real

    def test_collapsed_lines_skip_via_real_rectify(self):
        """All-five-lines-identical geometry skips (no raise), end to end."""
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image
            p = os.path.join(tmp, 's.png')
            Image.new('RGB', (64, 32), (255, 255, 255)).save(p)
            table = _const_table()
            _, status, _reason = B._sidecar_one((tmp, _row(p, table),
                                                   256, 10.0))
            self.assertEqual(status, 'skip')


class TestSidecarTableToCurves(unittest.TestCase):
    def test_constant_rows_bit_identical(self):
        """Constant-y rows restore bit-identically via table_to_curves."""
        import strip_staff as sst
        lines5 = [[20.0, 30.0, 40.0, 50.0, 60.0]]
        table = B.rows_to_table(lines5, 64)
        curves = sst.table_to_curves(table)
        self.assertEqual(len(curves), 1)
        self.assertEqual(len(curves[0]), 5)
        for line, y in zip(curves[0], lines5[0]):
            self.assertTrue(np.array_equal(line, np.full(64, round(y, 2))),
                            'constant-y row must restore bit-identically')

    def test_sidecar_routes_table_through_table_to_curves(self):
        """_sidecar_one rectifies table_to_curves output, not raw table."""
        import strip_staff as sst
        import rectify as R
        from PIL import Image
        real = R.rectify_system_strip
        seen = {}

        def spy(img, staves, target_sp=10.0, height=256):
            seen['staves'] = staves
            return Image.new('RGB', (16, 256)), torch.zeros(1, 256, 16)

        R.rectify_system_strip = spy
        try:
            with tempfile.TemporaryDirectory() as tmp:
                p = os.path.join(tmp, 'c.png')
                Image.new('RGB', (64, 32), (255, 255, 255)).save(p)
                table = B.rows_to_table(
                    [[20.0, 30.0, 40.0, 50.0, 60.0]], 64)
                _, status, _reason = B._sidecar_one(
                    (tmp, _row(p, table), 256, 10.0))
                self.assertEqual(status, 'ok')
        finally:
            R.rectify_system_strip = real
        want = sst.table_to_curves(table)
        got = seen['staves']
        self.assertEqual(len(got), len(want))
        for gs, ws in zip(got, want):
            self.assertEqual(len(gs), len(ws))
            for g, w in zip(gs, ws):
                self.assertTrue(np.array_equal(np.asarray(g), w))


class TestScaledCleanFallback(unittest.TestCase):
    def test_adversarial_gray_stays_on_canvas(self):
        """Heavy-degrade simulacrum (glare blotch + noise, no lines)."""
        from PIL import Image
        rng = np.random.default_rng(7)
        g = np.full((200, 320), 235.0)
        g += rng.normal(0, 18, g.shape)
        g[40:120, 60:200] = 250.0
        gray = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8))
        lines5 = [[30.0, 40.0, 50.0, 60.0, 70.0],
                  [110.0, 120.0, 130.0, 140.0, 150.0]]
        staves, sp = B._scaled_clean_fallback(gray, lines5, 200.0 / 220.0,
                                             10.0)
        self.assertEqual(len(staves), 2)
        for staff in staves:
            self.assertEqual(len(staff), 5)
            for line in staff:
                line = np.asarray(line, dtype=float)
                self.assertEqual(line.shape, (320,))
                self.assertGreaterEqual(float(line.min()), 0.0)
                self.assertLessEqual(float(line.max()), 199.0)
        self.assertTrue(np.isfinite(sp) and sp > 0)

    def test_tilt_failure_degrades_to_affine(self):
        """A failed tilt fit must still emit scaled lines, never raise."""
        from PIL import Image
        real = B._global_tilt_search

        def boom(g, seeds, sp):
            raise ValueError('simulated fit failure')
        B._global_tilt_search = boom
        try:
            gray = Image.new('L', (64, 40), 235)
            lines5 = [[8.0, 12.0, 16.0, 20.0, 24.0]]
            staves, sp = B._scaled_clean_fallback(gray, lines5, 0.5, 8.0)
        finally:
            B._global_tilt_search = real
        self.assertEqual(len(staves), 1)
        for line, y in zip(staves[0], lines5[0]):
            self.assertTrue(np.array_equal(
                np.asarray(line), np.full(64, y * 0.5)))
        self.assertAlmostEqual(sp, 2.0)


class TestSysdevDegradedCoverage(unittest.TestCase):
    def test_26_of_26_degraded_rows_kept(self):
        """T2.4 pipefix: no degraded sys-dev strip skips geometry."""
        import json
        from pathlib import Path
        import build_gate20_sys_dev as bgate
        bgate.ensure_weights_readable()
        sysdev = (Path(__file__).resolve().parent.parent.parent
                  / 'data-gen' / 'score-corpus' / 'gate20' / 'sys-dev')
        manifest = json.loads((sysdev / 'sys_dev_manifest.json').read_text(
            encoding='utf-8'))
        systems = sorted(manifest['systems'],
                         key=lambda s: (s['id'], s['sys_idx']))
        self.assertEqual(len(systems), 26)
        sources = {}
        for s in systems:
            _c, d = B._sysdev_one((str(sysdev), s))
            key = '%s-%d-s%d' % (s['id'], s['page'], s['sys_idx'])
            self.assertFalse(
                d.get('skip'),
                '%s degraded skipped: %s' % (key, d.get('reason')))
            self.assertTrue(d.get('table'), '%s empty table' % key)
            sources[d.get('source')] = sources.get(d.get('source'), 0) + 1
        print('sysdev degraded sources: %s' % (sources,))


if __name__ == '__main__':
    unittest.main()
