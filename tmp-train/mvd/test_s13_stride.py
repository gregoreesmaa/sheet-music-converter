"""S13(a) stride-4x8 tests: one feature row per diatonic step.

The 8x16 grid (32 rows at H=256) undersamples dense notation (spikes.md
S13). The 4x8 grid (64 rows, W/8 cols) doubles both axes. These tests pin
the desired API: ``stride='4x8'`` on the model, the eval dims helper, and
legacy bitwise-identical defaults when the flag is absent.
"""

import unittest

import torch

import data
import model
from measure_decode import mem_dims_for_strip, slot_mem_mask


class TestS13Stride4x8(unittest.TestCase):
    def test_default_stride_encode_shape_unchanged(self):
        net = model.MvdNet(vocab=len(data.VOCAB) + 3, anisotropic=True)
        net.eval()
        with torch.no_grad():
            mem = net.encode(torch.zeros(1, 3, 256, 512))
        self.assertEqual(list(mem.shape), [1, 32 * 32, net.d_model])

    def test_stride_4x8_encode_shape(self):
        net = model.MvdNet(vocab=len(data.VOCAB) + 3, anisotropic=True,
                           stride='4x8')
        net.eval()
        with torch.no_grad():
            mem = net.encode(torch.zeros(1, 3, 256, 512))
        self.assertEqual(list(mem.shape), [1, 64 * 64, net.d_model])

    def test_stride_4x8_weight_shapes_match_default(self):
        """Same parameter tensors: timm transfer arity is unchanged.

        Key names differ where stride-4x8 wraps a downsample in
        ``_PadConv`` (same nesting pattern as the 8x16 stage-3/4
        wraps), so compare the shape multiset + total count instead.
        """
        from collections import Counter
        dflt = model.MvdNet(vocab=len(data.VOCAB) + 3, anisotropic=True)
        fine = model.MvdNet(vocab=len(data.VOCAB) + 3, anisotropic=True,
                            stride='4x8')
        shapes = lambda sd: Counter(tuple(v.shape) for v in sd.values())
        self.assertEqual(shapes(dflt.state_dict()),
                         shapes(fine.state_dict()))
        cd = sum(v.numel() for v in dflt.state_dict().values())
        cf = sum(v.numel() for v in fine.state_dict().values())
        self.assertEqual(cd, cf)

    def test_mem_dims_legacy(self):
        self.assertEqual(mem_dims_for_strip(256, 512), (32, 32))

    def test_mem_dims_4x8(self):
        self.assertEqual(mem_dims_for_strip(256, 512, stride='4x8'),
                         (64, 64))

    def test_slot_mask_4x8_geometry(self):
        m = slot_mem_mask(2, 64, 64)
        self.assertEqual(list(m.shape), [64 * 64])
        # S=2 of 4 slots: rows >= 32 masked, rows < 32 open.
        grid = m.reshape(64, 64)
        self.assertFalse(bool(grid[:32].any()))
        self.assertTrue(bool(grid[32:].all()))


class TestS13StrideCollate(unittest.TestCase):
    """Train-side 4x8 shapes through ``collate_hybrid`` (S13a).

    Hand-written sidecars (no builder run): two S=2 strips, widths
    64 and 80 at height 256. Legacy values are pinned first so the
    flag cannot silently re-derive the default path.
    """

    HEIGHT = 256

    @classmethod
    def setUpClass(cls):
        import tempfile
        from pathlib import Path
        cls.td = Path(tempfile.mkdtemp(prefix='s13_stride_'))
        for stem, w in (('s13a-s0', 64), ('s13a-s1', 80)):
            torch.save(
                {'img': stem + '.png', 'S': 2, 'sp': 10.0,
                 'kind': 'rows', 'height': cls.HEIGHT,
                 'rgb': torch.full((3, cls.HEIGHT, w), 200,
                                   dtype=torch.uint8),
                 'gray': torch.full((cls.HEIGHT, w), 200,
                                    dtype=torch.uint8),
                 'staff': torch.zeros(1, cls.HEIGHT, w,
                                      dtype=torch.uint8)},
                str(cls.td / (stem + '.hybrid.pt')))
        import train as _train
        cls.train = _train
        cls.batch = [(str(cls.td / 's13a-s0.png'),
                      str(cls.td / 's13a-s0-bekern.txt'), 's13a-s0'),
                     (str(cls.td / 's13a-s1.png'),
                      str(cls.td / 's13a-s1-bekern.txt'), 's13a-s1')]

    def test_mem_width_legacy(self):
        self.assertEqual(self.train._mem_width_aniso(80), 5)
        self.assertEqual(self.train._mem_width_aniso(64), 4)

    def test_stride_fingerprint_keys(self):
        self.assertIn('stride', self.train.META_KEYS)
        self.assertIn('stride', self.train.RESUME_KEYS)

    def test_mem_width_4x8(self):
        self.assertEqual(
            self.train._mem_width_aniso(80, stride='4x8'), 10)
        self.assertEqual(
            self.train._mem_width_aniso(64, stride='4x8'), 8)

    def test_collate_hybrid_legacy_shapes(self):
        import data as _data
        _data.load_bekern = lambda p: ['bar']
        try:
            rgb, tgt, key_mask, gray, staff = \
                self.train.collate_hybrid(self.batch, self.HEIGHT)
        finally:
            import importlib
            importlib.reload(_data)
        self.assertEqual(list(rgb.shape), [2, 3, 256, 80])
        self.assertEqual(list(key_mask.shape), [2, 32 * 5])
        grid = key_mask[0].reshape(32, 5)
        # S=2: slot rows >= 16 masked; width 64 -> 4 valid cols.
        self.assertFalse(bool(grid[:16, :4].any()))
        self.assertTrue(bool(grid[16:].all()))
        self.assertTrue(bool(grid[:, 4:].all()))

    def test_collate_hybrid_4x8_shapes(self):
        import data as _data
        _data.load_bekern = lambda p: ['bar']
        try:
            rgb, tgt, key_mask, gray, staff = \
                self.train.collate_hybrid(self.batch, self.HEIGHT,
                                          stride='4x8')
        finally:
            import importlib
            importlib.reload(_data)
        self.assertEqual(list(rgb.shape), [2, 3, 256, 80])
        self.assertEqual(list(key_mask.shape), [2, 64 * 10])
        grid = key_mask[0].reshape(64, 10)
        # S=2 of 4 slots at 64 rows: rows >= 32 masked.
        self.assertFalse(bool(grid[:32, :8].any()))
        self.assertTrue(bool(grid[32:].all()))
        # Width 64 -> (64//4)//2 = 8 valid cols of 10.
        self.assertTrue(bool(grid[:, 8:].all()))
        grid1 = key_mask[1].reshape(64, 10)
        self.assertFalse(bool(grid1[:32].any()))
        self.assertTrue(bool(grid1[32:].all()))


class TestS13StrideCollatePlain(unittest.TestCase):
    """Unrectified ``collate`` (S13a): stride-aware mask grid.

    Two real PNGs (widths 64 and 96 at height 256) + monkeypatched
    targets. Expected grid comes from the independent
    ``measure_decode.mem_dims_for_strip`` arithmetic, and the batch
    flows through a live 4x8 ``encode`` end to end.
    """

    HEIGHT = 256

    @classmethod
    def setUpClass(cls):
        import tempfile
        from pathlib import Path
        from PIL import Image as PILImage
        import numpy as np
        cls.td = Path(tempfile.mkdtemp(prefix='s13_stride_plain_'))
        cls.widths = []
        for stem, w in (('s13p-s0', 64), ('s13p-s1', 96)):
            g = np.full((200, w), 200, np.uint8)
            PILImage.fromarray(g).convert('RGB').save(cls.td / (stem + '.png'))
            cls.widths.append(
                cls.train_load_width(str(cls.td / (stem + '.png'))))
        import train as _train
        cls.train = _train
        cls.batch = [(str(cls.td / 's13p-s0.png'),
                      str(cls.td / 's13p-s0-bekern.txt'), 's13p-s0'),
                     (str(cls.td / 's13p-s1.png'),
                      str(cls.td / 's13p-s1-bekern.txt'), 's13p-s1')]

    @classmethod
    def train_load_width(cls, path):
        import torch
        from PIL import Image as PILImage
        import numpy as np
        img = PILImage.open(path).convert('RGB')
        w, h = img.size
        new_w = round(w * cls.HEIGHT / h)
        return new_w

    def test_collate_plain_4x8_grid_and_encode(self):
        from measure_decode import mem_dims_for_strip
        import data as _data
        import model as _model
        _data.load_bekern = lambda p: ['bar']
        try:
            imgs, tgt, key_mask = self.train.collate(
                self.batch, self.HEIGHT, stride='4x8')
        finally:
            import importlib
            importlib.reload(_data)
        w0 = self.widths[0]
        w_pad = max(self.widths)
        w_pad = (w_pad + 15) // 16 * 16
        exp_h, exp_w = mem_dims_for_strip(self.HEIGHT, w_pad,
                                          stride='4x8')
        self.assertEqual(list(key_mask.shape), [2, exp_h * exp_w])
        self.assertEqual((exp_h, exp_w), (64, (w_pad // 4) // 2))
        grid0 = key_mask[0].reshape(exp_h, exp_w)
        valid0 = (w0 // 4) // 2
        self.assertFalse(bool(grid0[:, :valid0].any()))
        self.assertTrue(bool(grid0[:, valid0:].all()))
        net = _model.MvdNet(vocab=len(_data.VOCAB) + 3,
                            anisotropic=True, stride='4x8')
        net.eval()
        import torch
        with torch.no_grad():
            mem = net.encode(imgs,
                             mem_mask=key_mask)
        self.assertEqual(list(mem.shape), [2, exp_h * exp_w, net.d_model])

    def test_collate_plain_legacy_unchanged(self):
        from measure_decode import mem_dims_for_strip
        import data as _data
        _data.load_bekern = lambda p: ['bar']
        try:
            imgs, tgt, key_mask = self.train.collate(
                self.batch, self.HEIGHT)
        finally:
            import importlib
            importlib.reload(_data)
        w_pad = (max(self.widths) + 15) // 16 * 16
        exp_h, exp_w = mem_dims_for_strip(self.HEIGHT, w_pad)
        self.assertEqual((exp_h, exp_w), (32, 8))
        self.assertEqual(list(key_mask.shape), [2, exp_h * exp_w])


if __name__ == '__main__':
    unittest.main()
