"""Unit tests for T2.4 hybrid training path (train.py --hybrid-cache).

Covers: sidecar build (-> .hybrid.pt shapes/dtypes), sidecar loader
(height enforcement), collate_hybrid (padding/mask/target parity with
collate), assemble_hybrid_batch (11ch geometry), HeatNet persistent
session (probs range/shape, determinism), pair filter policy, and
meta/resume key coverage. Hermetic: synth strips + one real bekern
fixture written to a tmp dir; HeatNet uses the real heatnet.pt on CPU
(small legacy arch, fast).
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train
import data
import heat_features
import build_hybrid_cache as bhc

HEIGHT = 256


def synth_corpus(tmp, n=3):
    """Tiny corpus dir: n clean PNGs + bekern + one degraded jpg each.

    Staff lines: 2 staves x 5 constant lines at sp 20 (clean) so
    rows_to_table + rectify behave; degraded jpg = same pixels (the
    sidecar stage never inspects photometry, only widths).
    """
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(n):
        stem = 'ds0001-1-s%d' % i
        g = np.full((200, 400), 200, np.uint8)
        for s in range(2):
            for k in range(5):
                g[30 + s * 80 + k * 20 - 1:30 + s * 80 + k * 20 + 2, :] = 60
        Image.fromarray(g).convert('RGB').save(tmp / (stem + '.png'))
        Image.fromarray(g).convert('RGB').save(tmp / (stem + '_d0.jpg'))
        (tmp / (stem + '-bekern.txt')).write_text(
            'clef-G-2 key-1 time-3-4 voice-1 bar-rest bar\n',
            encoding='utf-8')
        (tmp / (stem + '_d0_params.json')).write_text(
            json.dumps({'file_seed': 1000 + i}), encoding='utf-8')
        names.append(stem)
    return tmp, names


def rows_jsonl(tmp, names):
    side = Path(tmp) / 'hybrid_staff.jsonl'
    with open(side, 'w', encoding='utf-8') as f:
        for stem in names:
            w = Image.open(tmp / (stem + '.png')).size[0]
            lines5 = [[30.0 + s * 80.0 + k * 20.0 for k in range(5)]
                      for s in range(2)]
            f.write(json.dumps(
                {'img': stem + '.png', 'S': 2, 'kind': 'rows',
                 'table': bhc.rows_to_table(lines5, w), 'sp': 20.0,
                 'source': 'test', 'skip': False}) + '\n')
            wd = Image.open(tmp / (stem + '_d0.jpg')).size[0]
            f.write(json.dumps(
                {'img': stem + '_d0.jpg', 'S': 2, 'kind': 'table',
                 'table': bhc.rows_to_table(lines5, wd), 'sp': 20.0,
                 'source': 'test', 'skip': False}) + '\n')
    return side


def _slot_blob(tmp, stem, s, w=64):
    """Hand-written sidecar with staff count ``s`` (no builder run).

    Width ``w`` is a multiple of 16, so any mask the collate returns
    is pure slot masking with no width-padding contribution.
    """
    Image.new('RGB', (w, 32), (255, 255, 255)).save(
        Path(tmp) / (stem + '.png'))
    torch.save(
        {'img': stem + '.png', 'S': s, 'sp': 20.0, 'kind': 'rows',
         'height': HEIGHT,
         'rgb': torch.full((3, HEIGHT, w), 200, dtype=torch.uint8),
         'gray': torch.full((HEIGHT, w), 200, dtype=torch.uint8),
         'staff': torch.zeros(1, HEIGHT, w, dtype=torch.uint8)},
        str(Path(tmp) / (stem + '.hybrid.pt')))


class TestHybridTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.td = Path(tempfile.mkdtemp(prefix='t24_hybtrain_'))
        cls.corpus, cls.names = synth_corpus(cls.td / 'corpus')
        cls.jsonl = rows_jsonl(cls.corpus, cls.names)
        cls.workdir = cls.td / 'hybrid'
        cls.workdir.mkdir(exist_ok=True)
        (cls.workdir / 'hybrid_staff.jsonl').write_text(
            cls.jsonl.read_text(encoding='utf-8'), encoding='utf-8')
        bhc.stage_sidecars(cls.workdir, cls.corpus, height=HEIGHT, jobs=1)
        cls.pairs = train.list_pairs(str(cls.corpus), str(cls.corpus))

    def test_sidecars_built(self):
        for stem in self.names:
            for img in (stem + '.png', stem + '_d0.jpg'):
                sc = self.corpus / (Path(img).stem + '.hybrid.pt')
                self.assertTrue(sc.exists(), sc)
                blob = torch.load(str(sc), map_location='cpu',
                                  weights_only=True)
                h, w = blob['rgb'].shape[1], blob['rgb'].shape[2]
                self.assertEqual(blob['rgb'].shape[0], 3)
                self.assertEqual(tuple(blob['gray'].shape), (h, w))
                self.assertEqual(tuple(blob['staff'].shape), (1, h, w))
                self.assertEqual(blob['rgb'].dtype, torch.uint8)
                self.assertEqual(blob['staff'].dtype, torch.uint8)
                self.assertEqual(h, HEIGHT)
                self.assertTrue(bool((blob['staff'] == 1).any()),
                                'staff grid empty for %s' % img)

    def test_index_and_filter(self):
        index = train.load_hybrid_index(str(self.jsonl))
        self.assertEqual(len(index), 2 * len(self.names))
        kept, dc, dd = train.filter_hybrid_pairs(self.pairs)
        self.assertEqual(len(kept), len(self.pairs))
        self.assertEqual((dc, dd), (0, 0))
        ghost = [(str(self.corpus / 'nope-s9_d0.jpg'),
                  str(self.corpus / (self.names[0] + '-bekern.txt')),
                  'nope-s9'),
                 (str(self.corpus / 'nope-s9.png'),
                  str(self.corpus / (self.names[0] + '-bekern.txt')),
                  'nope-s9')]
        kept2, dc2, dd2 = train.filter_hybrid_pairs(
            self.pairs + ghost)
        self.assertEqual(len(kept2), len(self.pairs))
        self.assertEqual((dc2, dd2), (1, 1))

    def test_sidecar_height_enforced(self):
        with self.assertRaises(ValueError):
            train.load_sidecar(
                str(self.corpus / (self.names[0] + '.png')), HEIGHT + 1)

    def test_collate_hybrid_shapes(self):
        batch = self.pairs[:2]
        rgb, tgt, mask, gray, staff = train.collate_hybrid(batch, HEIGHT)
        self.assertEqual(rgb.shape[0], 2)
        self.assertEqual(tuple(gray.shape)[::3], tuple(rgb.shape)[::3])
        self.assertEqual(tuple(staff.shape), tuple(gray.shape))
        # Deterministic across calls.
        rgb2, _, _, _, _ = train.collate_hybrid(batch, HEIGHT)
        self.assertTrue(torch.equal(rgb, rgb2))

    def test_collate_hybrid_targets_and_mask(self):
        # Single-width batch: no mask in either collate; targets match
        # the legacy collate on identical bekerns. (Rectified widths
        # legitimately differ from legacy resized widths, so masks are
        # only compared structurally, not cross-collate.)
        one = self.pairs[:1]
        _, tgt, _, _, _ = train.collate_hybrid(one, HEIGHT)
        _, tgt_legacy, mask_legacy = train.collate(one, HEIGHT)
        self.assertIsNone(mask_legacy)
        self.assertTrue(torch.equal(tgt, tgt_legacy))
        batch = self.pairs[:2]
        _, _, mask2, _, _ = train.collate_hybrid(batch, HEIGHT)
        self.assertIsNotNone(mask2)
        # Equal widths -> identical valid prefixes in every row.
        self.assertTrue(torch.equal(mask2[0], mask2[1]))
        valid = train._mem_width_aniso(200)
        # Fixtures are S=2 strips: 16 active slot rows (not 32).
        self.assertEqual(int((~mask2[0]).sum()), 16 * valid)

    def test_collate_hybrid_slot_mask_s1_to_s4(self):
        # Inactive staff slots (memory rows y_mem >= 8*S) are masked
        # for every S in 1..4, with no width padding involved (all
        # sidecars W=64, already a multiple of 16).
        import tempfile
        td = Path(tempfile.mkdtemp(prefix='t24_slots_'))
        bek = str(self.corpus / (self.names[0] + '-bekern.txt'))
        h_mem = HEIGHT // 8
        w_mem = train._mem_width_aniso(64)
        for s in (1, 2, 3, 4):
            stem = 'slot-s%d' % s
            _slot_blob(td, stem, s)
            pair = [(str(td / (stem + '.png')), bek, stem)]
            _, _, mask, _, _ = train.collate_hybrid(pair, HEIGHT)
            if s == 4:
                self.assertIsNone(
                    mask, 'S=4 unpadded keeps the legacy None mask')
                continue
            self.assertIsNotNone(mask, 'S=%d must mask slots' % s)
            m = mask[0].reshape(h_mem, w_mem)
            self.assertEqual(int(mask[0].sum()), (h_mem - 8 * s) * w_mem)
            self.assertFalse(bool(m[:8 * s].any()),
                             'active rows masked for S=%d' % s)
            self.assertTrue(bool(m[8 * s:].all()),
                            'inactive rows unmasked for S=%d' % s)
        # Mixed batch: slot rows are per-item, identical widths.
        items = []
        for s in (1, 4):
            stem = 'mix-s%d' % s
            _slot_blob(td, stem, s)
            items.append((str(td / (stem + '.png')), bek, stem))
        _, _, mask, _, _ = train.collate_hybrid(items, HEIGHT)
        self.assertIsNotNone(mask)
        m0 = mask[0].reshape(h_mem, w_mem)
        m1 = mask[1].reshape(h_mem, w_mem)
        self.assertTrue(bool(m0[8:].all()) and not bool(m0[:8].any()))
        self.assertFalse(bool(m1.any()))

    def test_load_sidecar_slot_fallback(self):
        # Missing or out-of-range S falls back to 4 (mask nothing),
        # preserving legacy behavior on old sidecars.
        import tempfile
        td = Path(tempfile.mkdtemp(prefix='t24_slotfb_'))
        stem = 'fb'
        _slot_blob(td, stem, 2)
        p = str(td / (stem + '.png'))
        blob = torch.load(str(td / (stem + '.hybrid.pt')),
                          map_location='cpu', weights_only=True)
        self.assertEqual(train.load_sidecar(p, HEIGHT)[3], 2)
        for bad in (0, 5, 9):
            blob['S'] = bad
            torch.save(blob, str(td / (stem + '.hybrid.pt')))
            self.assertEqual(train.load_sidecar(p, HEIGHT)[3], 4)
        del blob['S']
        torch.save(blob, str(td / (stem + '.hybrid.pt')))
        self.assertEqual(train.load_sidecar(p, HEIGHT)[3], 4)

    def test_heat_and_assemble(self):
        hn = heat_features.HeatNet(device='cpu')
        batch = self.pairs[:2]
        rgb, _, _, gray, staff = train.collate_hybrid(batch, HEIGHT)
        p1 = hn.probs(gray)
        p2 = hn.probs(gray)
        self.assertEqual(tuple(p1.shape),
                         (2, len(heat_features.HEAT_CLASSES),
                          HEIGHT, gray.shape[3]))
        self.assertTrue(bool(((p1 >= 0) & (p1 <= 1)).all()))
        self.assertTrue(torch.equal(p1, p2))
        full = train.assemble_hybrid_batch(hn, rgb, gray, staff)
        self.assertEqual(tuple(full.shape),
                         (2, 3 + heat_features.EXTRA_CHANS,
                          HEIGHT, gray.shape[3]))

    def test_meta_keys(self):
        class A:
            pass
        a = A()
        for k, v in {'height': 256, 'lr': 3e-4, 'seed': 0, 'exclude': [],
                     'shuffle': False, 'no_amp': False, 'accum': 2,
                     'init_from': None, 'trunk_lr_mult': 0.1, 'warmup': 0,
                     'cosine': False, 'sample_p': 0.0, 'pages_file': None,
                     'batch_size': 4, 'pretrained': False,
                     'no_degraded': False, 'tok_weight': 3.0,
                     'tok_dropout': 0.2, 'extra_chans': 8,
                     'hybrid_cache': 'x.jsonl',
                     'heat_pt': 'y.pt'}.items():
            setattr(a, k, v)
        meta = train.build_meta(a, self.pairs[:1])
        for k in train.META_KEYS:
            self.assertIn(k, meta)
        self.assertEqual(meta['hybrid_cache'], 'x.jsonl')
        self.assertEqual(meta['heat_pt'], 'y.pt')
        for k in train.RESUME_KEYS:
            self.assertIn(k, meta)


if __name__ == '__main__':
    unittest.main()
