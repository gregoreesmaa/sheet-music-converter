"""T0.2 safety tests: atomic saves, single-writer lock, resume refusal.

Run with the repo-root .venv python:

    .venv/Scripts/python.exe -m unittest discover -s tmp-train/mvd -p "test_*.py" -v
"""

import json
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402

TRAIN_PY = Path(__file__).resolve().parent / 'train.py'


class TestAtomicSave(unittest.TestCase):
    def test_atomic_torch_save_leaves_no_tmp(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'ckpt-1.pt'
            train.atomic_torch_save({'step': 1}, p)
            self.assertTrue(p.exists(), 'target file must exist')
            self.assertFalse(Path(str(p) + '.tmp').exists(),
                             'no .tmp file may be left behind')
            blob = torch.load(p, map_location='cpu', weights_only=True)
            self.assertEqual(blob, {'step': 1})
            t = Path(d) / 'meta.json'
            train.atomic_write_text(t, '{"a": 1}')
            self.assertEqual(
                json.loads(t.read_text(encoding='utf-8')), {'a': 1})
            self.assertFalse(Path(str(t) + '.tmp').exists(),
                             'no .tmp file may be left behind')


class TestOutLock(unittest.TestCase):
    def test_second_acquire_blocked_and_release_cleans(self):
        with tempfile.TemporaryDirectory() as d:
            train.acquire_out_lock(d)
            lock = Path(d) / '.train.lock'
            self.assertTrue(lock.exists(), 'lock file must exist')
            pid = json.loads(lock.read_text(encoding='utf-8'))['pid']
            self.assertEqual(pid, __import__('os').getpid())
            with self.assertRaises(SystemExit) as cm:
                train.acquire_out_lock(d)
            self.assertIn('refusing', str(cm.exception.code))
            self.assertTrue(lock.exists(), 'live lock must be kept')
            train.release_out_lock(d)
            self.assertFalse(lock.exists(), 'release must clean up')
            train.acquire_out_lock(d)
            self.assertTrue(lock.exists(), 're-acquire must work')
            train.release_out_lock(d)
            self.assertFalse(lock.exists())

    def test_stale_lock_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as d:
            lock = Path(d) / '.train.lock'
            lock.write_text(json.dumps({'pid': 2**31 - 1}),
                            encoding='utf-8')
            train.acquire_out_lock(d)
            self.assertTrue(lock.exists())
            pid = json.loads(lock.read_text(encoding='utf-8'))['pid']
            self.assertEqual(pid, __import__('os').getpid())
            train.release_out_lock(d)
            self.assertFalse(lock.exists())


class TestResumeRefusal(unittest.TestCase):
    def _write_pair(self, renders, targets):
        from PIL import Image
        import data
        renders.mkdir(parents=True, exist_ok=True)
        targets.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (64, 64), 'white').save(renders / 'ds0001-1.png')
        toks = list(data.VOCAB[:8])
        (targets / 'ds0001-1-bekern.txt').write_text(
            ' '.join(toks) + '\n', encoding='utf-8')

    def _run_train(self, renders, targets, out, extra=()):
        cmd = ([sys.executable, str(TRAIN_PY),
                '--renders', str(renders), '--targets', str(targets),
                '--pages', 'ds0001-1', '--steps', '1', '--eval-every', '1',
                '--height', '64', '--out', str(out)] + list(extra))
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=900)

    def test_resume_changed_height_and_lr_refused(self):
        with tempfile.TemporaryDirectory() as d:
            renders = Path(d) / 'renders'
            targets = Path(d) / 'targets'
            out = Path(d) / 'run'
            self._write_pair(renders, targets)
            first = self._run_train(renders, targets, out)
            self.assertEqual(first.returncode, 0, first.stderr[-3000:])
            self.assertTrue((out / 'meta.json').exists())
            for flag, val, key in (('--height', '96', "'height'"),
                                   ('--lr', '1e-5', "'lr'")):
                with self.subTest(change=key):
                    r = self._run_train(renders, targets, out,
                                        ['--resume', flag, val])
                    self.assertNotEqual(r.returncode, 0,
                                        'resume with changed %s must fail'
                                        % key)
                    combined = (r.stdout or '') + (r.stderr or '')
                    self.assertIn('refusing --resume: config mismatch',
                                  combined)
                    self.assertIn(key, combined)


class TestResumeLegacyKeys(unittest.TestCase):
    """Old metas predate new flags: missing key + legacy value proceeds.

    Guards the parked speed patch: its flags (`heat_cache`, `compile`,
    `grad_checkpoint`) must not brick `--resume` in existing out dirs.
    Re-apply contract: landing those flags MUST also append them to
    both RESUME_KEYS and LEGACY_DEFAULTS here and in train.py.
    """

    def _metas(self, drop=None, new_vals=None):
        base = {k: 'v-%s' % k for k in train.RESUME_KEYS}
        base.update({'lr': 0.001, 'batch_size': 8, 'accum': 1})
        old = dict(base)
        if drop is not None:
            del old[drop]
        new = dict(base)
        new.update(new_vals or {})
        return old, new

    def test_legacy_key_absent_and_default_proceeds(self):
        import train as T
        self.assertIn('heat_cache', T.LEGACY_DEFAULTS)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            old, new = self._metas()
            old = {k: v for k, v in old.items() if k != 'heat_cache'}
            (out / 'meta.json').write_text(json.dumps(old), encoding='utf-8')
            with unittest.mock.patch.object(
                    T, 'RESUME_KEYS', T.RESUME_KEYS + ('heat_cache',)):
                new = dict(new)
                new['heat_cache'] = T.LEGACY_DEFAULTS['heat_cache']
                self.assertIsNone(T.check_resume(str(out), new))

    def test_legacy_key_absent_and_nondefault_refused(self):
        import train as T
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            old, new = self._metas()
            old = {k: v for k, v in old.items() if k != 'compile'}
            (out / 'meta.json').write_text(json.dumps(old), encoding='utf-8')
            with unittest.mock.patch.object(
                    T, 'RESUME_KEYS', T.RESUME_KEYS + ('compile',)):
                new = dict(new)
                new['compile'] = not T.LEGACY_DEFAULTS['compile']
                msg = T.check_resume(str(out), new)
                self.assertIsNotNone(msg)
                self.assertIn("'compile'", msg)

    def test_nonlegacy_missing_key_still_refused(self):
        import train as T
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            old, new = self._metas(drop='lr')
            (out / 'meta.json').write_text(json.dumps(old), encoding='utf-8')
            msg = T.check_resume(str(out), new)
            self.assertIsNotNone(msg)
            self.assertIn("'lr'", msg)


class TestC2FlagsFingerprinted(unittest.TestCase):
    def _write_pair_with_degraded(self, renders, targets):
        from PIL import Image
        import data
        renders.mkdir(parents=True, exist_ok=True)
        targets.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (64, 64), 'white').save(renders / 'ds0001-1.png')
        Image.new('RGB', (64, 64), 'white').save(
            renders / 'ds0001-1_d0.jpg')
        toks = list(data.VOCAB[:8])
        (targets / 'ds0001-1-bekern.txt').write_text(
            ' '.join(toks) + '\n', encoding='utf-8')

    def _run_train(self, renders, targets, out, extra=()):
        cmd = ([sys.executable, str(TRAIN_PY),
                '--renders', str(renders), '--targets', str(targets),
                '--pages', 'ds0001-1', '--steps', '1', '--eval-every', '1',
                '--height', '64', '--out', str(out)] + list(extra))
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=900)

    def test_c2_flags_pairing_meta_and_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            renders = Path(d) / 'renders'
            targets = Path(d) / 'targets'
            out = Path(d) / 'run'
            self._write_pair_with_degraded(renders, targets)
            both = train.list_pairs(renders, targets)
            self.assertEqual(len(both), 2, 'clean + degraded expected')
            clean = train.list_pairs(renders, targets,
                                     include_degraded=False)
            self.assertEqual(len(clean), 1, 'clean-only expected')
            self.assertTrue(clean[0][0].endswith('.png'))
            first = self._run_train(renders, targets, out,
                                    ['--no-degraded', '--tok-weight', '3.0'])
            tail = ((first.stdout or '') + (first.stderr or ''))[-2000:]
            self.assertEqual(first.returncode, 0, tail)
            meta = json.loads((out / 'meta.json').read_text(
                encoding='utf-8'))
            self.assertTrue(meta['no_degraded'])
            self.assertEqual(meta['tok_weight'], 3.0)
            resumes = ((['--resume', '--tok-weight', '3.0'],
                        "'no_degraded'"),
                       (['--resume', '--no-degraded'], "'tok_weight'"))
            for extra, key in resumes:
                with self.subTest(change=key):
                    r = self._run_train(renders, targets, out, extra)
                    self.assertNotEqual(r.returncode, 0,
                                        'resume flipping %s must fail' % key)
                    combined = (r.stdout or '') + (r.stderr or '')
                    self.assertIn('refusing --resume: config mismatch',
                                  combined)
                    if key == "'no_degraded'":
                        # Flipping --no-degraded also changes the pair
                        # multiset, so 'pages' (stems with duplicates)
                        # reports first; the flag's own fingerprint is
                        # proven by its RESUME_KEYS membership below.
                        self.assertIn("'pages'", combined)
                        self.assertIn('no_degraded', train.RESUME_KEYS)
                        self.assertIn('no_degraded', train.META_KEYS)
                    else:
                        self.assertIn(key, combined)


class TestTokWeightLoss(unittest.TestCase):
    def test_structural_weight_raises_loss_and_unit_is_noop(self):
        import torch
        import data
        import model
        torch.manual_seed(0)
        net = model.MvdNet(vocab=len(data.VOCAB) + 3, d_model=64,
                           layers=1, heads=4, dim_ff=128)
        net.eval()
        bar = data.TOKEN_TO_ID['bar']
        key = data.TOKEN_TO_ID['key+0']
        q = data.TOKEN_TO_ID['quarter']
        v1 = data.TOKEN_TO_ID['voice-1']
        tgt = torch.tensor([[q, bar, key, q, bar, data.PAD_ID]])
        img = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            base = net.loss(img, tgt).item()
            one = net.loss(img, tgt, tok_weight=1.0,
                           weight_ids=[bar, key]).item()
            w_present = net.loss(img, tgt, tok_weight=3.0,
                                 weight_ids=[bar, key]).item()
            w_absent = net.loss(img, tgt, tok_weight=3.0,
                                weight_ids=[v1]).item()
        self.assertEqual(one, base, 'weight 1.0 must be bitwise legacy')
        self.assertEqual(w_absent, base,
                         'weight on ids absent from tgt must be a no-op')
        self.assertNotEqual(w_present, base,
                            'weight on present structural ids must '
                            'change the loss')


class TestTfTop1ByClass(unittest.TestCase):
    def test_counts_by_class_and_skips_pad_eos(self):
        import torch
        import data
        V = len(data.VOCAB)
        bar = data.TOKEN_TO_ID['bar']
        key = data.TOKEN_TO_ID['key+0']
        q = data.TOKEN_TO_ID['quarter']
        v1 = data.TOKEN_TO_ID['voice-1']
        step = data.TOKEN_TO_ID['C']
        oct4 = data.TOKEN_TO_ID['4']
        tgt = [q, bar, key, v1, step, oct4, data.EOS_ID, data.PAD_ID]
        T = len(tgt)
        logits = torch.zeros(T, V + 3)
        logits[0, q] = 10.0
        logits[1, bar] = 10.0
        logits[2, bar] = 10.0
        logits[3, v1] = 10.0
        logits[4, q] = 10.0
        logits[5, oct4] = 10.0
        got = train.tf_top1_by_class(logits, tgt)
        self.assertEqual(got['body'], [1, 2])
        self.assertEqual(got['bar'], [1, 1])
        self.assertEqual(got['header'], [0, 1])
        self.assertEqual(got['syntax'], [2, 2])


class TestGroundingProbe(unittest.TestCase):
    def test_grounding_probe_returns_metrics(self):
        import torch
        import data
        import model
        net = model.MvdNet(vocab=len(data.VOCAB) + 3, d_model=64,
                           layers=1, heads=4, dim_ff=128)
        net.eval()
        eimgs = torch.randn(1, 3, 128, 256)
        ref_ids = list(range(min(8, len(data.VOCAB))))
        out = train.grounding_probe(net, eimgs, ref_ids)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 4)
        hyp_real, ter_real, ter_blank, blank_diff = out
        self.assertIsInstance(hyp_real, list)
        for v in (ter_real, ter_blank, blank_diff):
            self.assertIsInstance(v, float)


class TestTokWeightIds(unittest.TestCase):
    def test_struct_weight_ids_bar_only(self):
        import data
        self.assertEqual(train.STRUCT_WEIGHT_IDS,
                         frozenset((data.TOKEN_TO_ID['bar'],)))

    def test_content_ids_composition(self):
        import data
        self.assertEqual(len(train.CONTENT_IDS), 7 + 9 + 7 + 3 + 4 + 1)
        for tok in ('quarter', 'eighth', 'whole', 'C', 'G', '4', '0',
                    'dot', 'tie-start', 'tie-stop', 'a1', 'a-2', 'rest'):
            self.assertIn(data.TOKEN_TO_ID[tok], train.CONTENT_IDS, tok)
        for tok in ('bar', 'voice-1', 'voice-12', 'clef-G-2', 'clef-C-3',
                    'key+0', 'key-1', 'time-4-4', 'time-3-8', 'chord',
                    'tuplet-stop', 'bar-rest', 'breve'):
            self.assertNotIn(data.TOKEN_TO_ID[tok], train.CONTENT_IDS,
                             tok)
        for i in train.CONTENT_IDS:
            self.assertLess(i, len(data.VOCAB))
            self.assertNotIn(i, (data.PAD_ID, data.BOS_ID, data.EOS_ID))


class TestTokDropoutResume(unittest.TestCase):
    def _write_pair(self, renders, targets):
        from PIL import Image
        import data
        renders.mkdir(parents=True, exist_ok=True)
        targets.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (64, 64), 'white').save(renders / 'ds0001-1.png')
        toks = list(data.VOCAB[:8])
        (targets / 'ds0001-1-bekern.txt').write_text(
            ' '.join(toks) + '\n', encoding='utf-8')

    def _run_train(self, renders, targets, out, extra=()):
        cmd = ([sys.executable, str(TRAIN_PY),
                '--renders', str(renders), '--targets', str(targets),
                '--pages', 'ds0001-1', '--steps', '1', '--eval-every', '1',
                '--height', '64', '--out', str(out)] + list(extra))
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=900)

    def test_resume_tok_dropout_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as d:
            renders = Path(d) / 'renders'
            targets = Path(d) / 'targets'
            out = Path(d) / 'run'
            self._write_pair(renders, targets)
            first = self._run_train(renders, targets, out)
            self.assertEqual(first.returncode, 0, first.stderr[-3000:])
            meta = json.loads((out / 'meta.json').read_text(
                encoding='utf-8'))
            self.assertEqual(meta['tok_dropout'], 0.0)
            r = self._run_train(renders, targets, out,
                                ['--resume', '--tok-dropout', '0.2'])
            self.assertNotEqual(r.returncode, 0,
                                'resume with changed tok_dropout must fail')
            combined = (r.stdout or '') + (r.stderr or '')
            self.assertIn('refusing --resume: config mismatch', combined)
            self.assertIn("'tok_dropout'", combined)

    def test_resume_tok_dropout_match_proceeds(self):
        with tempfile.TemporaryDirectory() as d:
            renders = Path(d) / 'renders'
            targets = Path(d) / 'targets'
            out = Path(d) / 'run'
            self._write_pair(renders, targets)
            first = self._run_train(renders, targets, out,
                                    ['--tok-dropout', '0.2'])
            self.assertEqual(first.returncode, 0, first.stderr[-3000:])
            meta = json.loads((out / 'meta.json').read_text(
                encoding='utf-8'))
            self.assertEqual(meta['tok_dropout'], 0.2)
            r = self._run_train(renders, targets, out,
                                ['--resume', '--tok-dropout', '0.2'])
            self.assertEqual(r.returncode, 0, (r.stdout or '')
                             + (r.stderr or '')[-3000:])


class TestListPairsPrefixIndex(unittest.TestCase):
    """list_pairs must equal per-stem glob semantics (O(stems x files)
    glob re-scanned the whole renders dir per bekern stem and hung
    startup on the 152k-file corpus; the prefix-range index over one
    sorted listing is the exact-semantics replacement)."""

    def _tree(self, root):
        r = Path(root) / 'renders'
        t = Path(root) / 'targets'
        r.mkdir()
        t.mkdir()
        # tricky names: prefix collisions, '_d' mid-stem, non-jpg tails
        files = ['a-s0.png', 'a-s0_d0.jpg', 'a-s0_d1.jpg', 'a-s0x.png',
                 'a-s0x_d0.jpg', 'b_d-s1.png', 'b_d-s1_d0.jpg',
                 'c-s2.png', 'c-s2_d0_x.jpg', 'c-s2_d0.png', 'note.txt']
        for n in files:
            (r / n).write_bytes(b'x')
        for s in ['a-s0', 'a-s0x', 'b_d-s1', 'c-s2', 'ghost-s9']:
            (t / (s + '-bekern.txt')).write_text('bar |', encoding='utf-8')
        return r, t

    def _reference(self, renders, targets):
        import fnmatch
        ref = []
        for bek in sorted(targets.glob('*-bekern.txt')):
            stem = bek.name[:-len('-bekern.txt')]
            clean = renders / (stem + '.png')
            if clean.exists():
                ref.append((str(clean), str(bek), stem))
            for dj in sorted(renders.glob(stem + '_d*.jpg'),
                             key=lambda p: p.name):
                ref.append((str(dj), str(bek), stem))
        return ref

    def test_matches_brute_force_glob(self):
        with tempfile.TemporaryDirectory() as d:
            r, t = self._tree(d)
            got = train.list_pairs(str(r), str(t))
            want = self._reference(r, t)
            self.assertEqual(got, want)
            # prefix-collision stems must not cross-match degraded files
            by_stem = {}
            for img, _bek, stem in got:
                by_stem.setdefault(stem, []).append(Path(img).name)
            self.assertEqual(sorted(by_stem['a-s0']),
                             ['a-s0.png', 'a-s0_d0.jpg', 'a-s0_d1.jpg'])
            self.assertEqual(sorted(by_stem['a-s0x']),
                             ['a-s0x.png', 'a-s0x_d0.jpg'])


class TestVocabExpansion(unittest.TestCase):
    """T3.2: loading a pre-bekern+ ckpt (133+3=136 rows, e.g. v2.4)
    under the committed vocab (178+3=181 rows) must keep every
    overlapping row byte-identical and fresh-init the 45 new rows."""

    OLD_V = 136  # sys-hybrid-v2.4 ckpt rows: 133 vocab + PAD/BOS/EOS
    NEW_ROWS = 45  # (178 + 3 specials) - 136 ckpt rows

    def _nets(self):
        import torch
        import data
        import model
        self.assertEqual(len(data.VOCAB), 178)
        new_v = len(data.VOCAB) + 3
        torch.manual_seed(0)
        old = model.MvdNet(vocab=self.OLD_V, d_model=64, layers=1,
                           heads=4, dim_ff=128)
        new = model.MvdNet(vocab=new_v, d_model=64, layers=1,
                           heads=4, dim_ff=128)
        return old, new

    def test_expand_prefix_rows_byte_identical(self):
        import torch
        old, new = self._nets()
        state = old.state_dict()
        how = train.load_weights_compat(new, state)
        self.assertEqual(how, 'vocab-expand')
        got = new.state_dict()
        for k in ('tok_emb.weight', 'out.weight', 'out.bias'):
            self.assertTrue(torch.equal(got[k][:self.OLD_V], state[k]),
                            '%s first-%d rows must equal ckpt rows exactly'
                            % (k, self.OLD_V))

    def test_expand_new_rows_finite_nonzero_distinct(self):
        import torch
        old, new = self._nets()
        state = old.state_dict()
        train.load_weights_compat(new, state)
        got = new.state_dict()
        for k in ('tok_emb.weight', 'out.weight', 'out.bias'):
            fresh = got[k][self.OLD_V:]
            self.assertEqual(fresh.shape[0], self.NEW_ROWS, k)
            self.assertTrue(torch.isfinite(fresh).all().item(),
                            '%s new rows must be finite' % k)
            self.assertTrue((fresh != 0).all().item(),
                            '%s new rows must be non-zero' % k)
            rows = [tuple(r.reshape(-1).tolist()) for r in fresh]
            self.assertEqual(len(set(rows)), len(rows),
                             '%s new rows must differ from each other'
                             % k)

    def test_same_size_ckpt_stays_strict_and_identical(self):
        import torch
        _old, new = self._nets()
        state = {k: v.clone() for k, v in new.state_dict().items()}
        how = train.load_weights_compat(new, state)
        self.assertEqual(how, 'strict')
        got = new.state_dict()
        for k in state:
            self.assertTrue(torch.equal(got[k], state[k]),
                            '%s must be unchanged by strict load' % k)


import unittest


class TestCudaMemStats(unittest.TestCase):
    def test_off_cuda_returns_nones(self):
        res, alloc = train.cuda_mem_stats('cpu')
        self.assertIsNone(res)
        self.assertIsNone(alloc)

    def test_on_cuda_returns_megabytes(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('needs CUDA')
        res, alloc = train.cuda_mem_stats('cuda')
        self.assertIsInstance(res, float)
        self.assertIsInstance(alloc, float)
        self.assertGreaterEqual(res, 0.0)
        self.assertGreaterEqual(alloc, 0.0)
        self.assertGreaterEqual(res, alloc)


class TestMaintenanceDue(unittest.TestCase):
    def test_cadence(self):
        for step in (500, 1000, 1500, 2000):
            self.assertTrue(train.maintenance_due(step), step)
        for step in (1, 50, 100, 499, 501, 3850):
            self.assertFalse(train.maintenance_due(step), step)


class TestSaveCheckpoint(unittest.TestCase):
    def test_roundtrip_schema_and_no_tmp(self):
        import torch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            net = torch.nn.Linear(4, 3)
            opt = torch.optim.SGD(net.parameters(), lr=0.01)
            args = SimpleNamespace(lr=0.01, trunk_lr_mult=0.5,
                                    warmup=500, accum=8, cosine=True)
            train.save_checkpoint(Path(d), 500, net, opt, None, [63],
                                  args)
            ckpt = Path(d) / 'ckpt-500.pt'
            side = Path(d) / 'opt-500.pt'
            self.assertTrue(ckpt.exists())
            self.assertTrue(side.exists())
            self.assertFalse(Path(str(ckpt) + '.tmp').exists())
            self.assertFalse(Path(str(side) + '.tmp').exists())
            blob = torch.load(side, map_location='cpu',
                              weights_only=False)
            self.assertEqual(
                set(blob.keys()),
                {'step', 'updates', 'opt', 'sched', 'hp'})
            self.assertEqual(blob['step'], 500)
            self.assertEqual(blob['updates'], 63)
            self.assertIsNone(blob['sched'])
            self.assertEqual(blob['hp']['accum'], 8)

import unittest


class TestBatchWidthGuard(unittest.TestCase):
    def test_cap_clears_typical_keeps_pole_out(self):
        # Typical rectified strips are ~1200-2300 px; the observed
        # 5760-px pole (step 3860) must trip the guard.
        self.assertLess(train.BATCH_WIDTH_CAP, 5760)
        self.assertGreaterEqual(train.BATCH_WIDTH_CAP, 2300)

    def test_predicate(self):
        self.assertTrue(train.batch_too_wide(5760))
        self.assertTrue(train.batch_too_wide(train.BATCH_WIDTH_CAP + 1))
        self.assertFalse(train.batch_too_wide(train.BATCH_WIDTH_CAP))
        self.assertFalse(train.batch_too_wide(2288))
        self.assertFalse(train.batch_too_wide(0))


class TestWarmStartStemSlicing(unittest.TestCase):
    """Warm-start stem-slicing (11->4 RGB+staff and 11->3 RGB-only) + aux/foveal compat."""

    def test_slice_11ch_to_4ch_copies_rgb_and_staff_skips_mismatched_aux(self):
        import torch
        import data
        import model
        v = len(data.VOCAB) + 3
        torch.manual_seed(11)
        src = model.MvdNet(vocab=v, d_model=64, layers=1, heads=4,
                           dim_ff=128, extra_chans=8)
        dst = model.MvdNet(vocab=v, d_model=64, layers=1, heads=4,
                           dim_ff=128, extra_chans=1, foveal_tap=True)
        src_sd = src.state_dict()
        how = train.load_weights_compat(dst, src_sd)
        self.assertIn('stem-slice-11to4', how)
        self.assertIn('foveal-fresh', how)
        got = dst.state_dict()
        # RGB [:3] and canonical staff grid [10:11] -> [3:4] exact copy.
        self.assertTrue(torch.equal(
            got['encoder.0.0.weight'][:, :3],
            src_sd['encoder.0.0.weight'][:, :3]))
        self.assertTrue(torch.equal(
            got['encoder.0.0.weight'][:, 3:4],
            src_sd['encoder.0.0.weight'][:, 10:11]))
        self.assertTrue(torch.equal(
            got['encoder.0.0.bias'], src_sd['encoder.0.0.bias']))
        # Trunk/decoder weights copied verbatim.
        self.assertTrue(torch.equal(got['proj.weight'], src_sd['proj.weight']))
        self.assertTrue(torch.equal(got['out.weight'], src_sd['out.weight']))
        # aux_head.2 has 1 ch on dst vs 8 ch on src -> gracefully skipped.
        self.assertEqual(tuple(got['aux_head.2.weight'].shape), (1, 128, 1, 1))
        # foveal_proj final conv stays zero-init for bit-identical step-0.
        self.assertTrue(bool((got['foveal_proj.2.weight'] == 0).all()))

    def test_slice_11ch_to_3ch_copies_rgb_and_handles_missing_aux(self):
        import torch
        import data
        import model
        v = len(data.VOCAB) + 3
        torch.manual_seed(12)
        src = model.MvdNet(vocab=v, d_model=64, layers=1, heads=4,
                           dim_ff=128, extra_chans=8)
        dst = model.MvdNet(vocab=v, d_model=64, layers=1, heads=4,
                           dim_ff=128, extra_chans=0)
        src_sd = src.state_dict()
        how = train.load_weights_compat(dst, src_sd)
        self.assertIn('stem-slice-11to3', how)
        got = dst.state_dict()
        self.assertTrue(torch.equal(
            got['encoder.0.0.weight'], src_sd['encoder.0.0.weight'][:, :3]))
        self.assertTrue(torch.equal(
            got['encoder.0.0.bias'], src_sd['encoder.0.0.bias']))
        self.assertTrue(torch.equal(got['out.weight'], src_sd['out.weight']))


class TestFourChannelNoHeatnet(unittest.TestCase):
    """extra_chans=1 4-ch assembly/forward without HeatNet + shared hidden/logits."""

    def test_assemble_and_forward_4ch_without_heatnet(self):
        import torch
        import data
        import model
        torch.manual_seed(7)
        rgb = torch.randn(2, 3, 64, 128)
        staff = torch.randint(0, 2, (2, 1, 64, 128)).float()
        imgs = train.assemble_hybrid_batch(None, rgb, None, staff)
        self.assertEqual(tuple(imgs.shape), (2, 4, 64, 128))
        self.assertTrue(torch.equal(imgs[:, :3], rgb))
        self.assertTrue(torch.equal(imgs[:, 3:4], staff))

        v = len(data.VOCAB) + 3
        net = model.MvdNet(vocab=v, d_model=64, layers=1, heads=4,
                           dim_ff=128, extra_chans=1, foveal_tap=True)
        tgt = torch.randint(0, len(data.VOCAB), (2, 10), dtype=torch.long)
        ce, ap, xc, h, logits = net.loss_shared(
            imgs, tgt, need_aux=False, need_coord=False,
            return_hidden=True, return_logits=True)
        self.assertIsNone(ap)
        self.assertIsNone(xc)
        self.assertTrue(bool(torch.isfinite(ce)))
        self.assertEqual(tuple(h.shape), (2, 10, 64))
        self.assertEqual(tuple(logits.shape), (2, 10, v))
        self.assertTrue(torch.allclose(net.out(h), logits, atol=1e-6))

        # beam_search pads with pad_id (never bos_id = eos_id - 1).
        beams = net.beam_search(imgs, max_len=6, beam=2, constrain=False)
        self.assertEqual(beams.shape[0], 2)
        self.assertFalse(bool((beams == net.bos_id).any()))


class TestNewFlagsLegacyResume(unittest.TestCase):
    """New flags must be in META_KEYS, RESUME_KEYS, and LEGACY_DEFAULTS."""

    NEW_DEFAULTS = {
        'foveal_tap': False,
        'optics_profile': 'default',
        'jsd_weight': 0.0,
        'jsd_every': 4,
        'no_heatnet': False,
        's18_weight': 0.0,
        'comer': False,
        'gdcm_weight': 0.0,
        'extra_data_dir': None,
        'max_tokens': 768,
        'rope_2d': False,
        'mtp_weight': 0.0,
    }

    def test_old_meta_without_new_keys_resumes_at_defaults_and_refuses_nondefault(self):
        for k, v in self.NEW_DEFAULTS.items():
            self.assertIn(k, train.META_KEYS, k)
            self.assertIn(k, train.RESUME_KEYS, k)
            self.assertIn(k, train.LEGACY_DEFAULTS, k)
            self.assertEqual(train.LEGACY_DEFAULTS[k], v, k)

        base = {k: 'v-%s' % k for k in train.RESUME_KEYS}
        base.update(train.LEGACY_DEFAULTS)
        old = {k: v for k, v in base.items() if k not in self.NEW_DEFAULTS}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / 'meta.json').write_text(json.dumps(old), encoding='utf-8')
            self.assertIsNone(train.check_resume(str(out), dict(base)))

            # Any non-default value must be refused.
            for k, bad_val in (('foveal_tap', True),
                               ('optics_profile', 'calibrated'),
                               ('jsd_weight', 0.2),
                               ('jsd_every', 2),
                               ('no_heatnet', True),
                               ('s18_weight', 0.25),
                               ('comer', True),
                               ('gdcm_weight', 0.1),
                               ('extra_data_dir', ['D:/tmp/extra']),
                               ('max_tokens', 512),
                               ('rope_2d', True),
                               ('mtp_weight', 0.10)):
                with self.subTest(key=k):
                    mut = dict(base)
                    mut[k] = bad_val
                    msg = train.check_resume(str(out), mut)
                    self.assertIsNotNone(msg)
                    self.assertIn(repr(k), msg)


class TestExp2BWarmStartAndOptimizer(unittest.TestCase):
    """Warm-starting CoMER + GDCM + S18 from a plain 4-ch checkpoint."""

    def test_warm_start_from_4ch_is_exact_step0_identity_and_fresh_lr_group(self):
        import torch
        import data
        import model
        v = len(data.VOCAB) + 3
        torch.manual_seed(42)
        base = model.MvdNet(
            vocab=v, d_model=64, layers=2, heads=4, dim_ff=128,
            extra_chans=1, foveal_tap=True)
        exp2b = model.MvdNet(
            vocab=v, d_model=64, layers=2, heads=4, dim_ff=128,
            extra_chans=1, foveal_tap=True,
            use_comer=True, use_gdcm=True, use_s18=True)
        how = train.load_weights_compat(exp2b, base.state_dict())
        self.assertIn('s18-fresh', how)
        self.assertIn('comer-fresh', how)
        self.assertIn('gdcm-fresh', how)

        base.eval()
        exp2b.eval()
        imgs = torch.randn(2, 4, 64, 128)
        tgt_in = torch.tensor([
            [base.bos_id, data.TOKEN_TO_ID['clef-G-2'], data.TOKEN_TO_ID['quarter'],
             data.TOKEN_TO_ID['C'], data.TOKEN_TO_ID['4'], data.TOKEN_TO_ID['bar']],
            [base.bos_id, data.TOKEN_TO_ID['clef-F-4'], data.TOKEN_TO_ID['half'],
             data.TOKEN_TO_ID['G'], data.TOKEN_TO_ID['3'], data.TOKEN_TO_ID['bar']],
        ], dtype=torch.long)
        with torch.no_grad():
            mem_base = base.encode(imgs)
            mem_2b = exp2b.encode(imgs)
            self.assertTrue(torch.allclose(mem_base, mem_2b, atol=1e-6),
                            'GDCM zero-init must be step-0 identity on encode()')
            logits_base = base(imgs, tgt_in)
            logits_2b = exp2b(imgs, tgt_in)
            max_diff = float((logits_base - logits_2b).abs().max())
            self.assertLess(max_diff, 1e-5,
                            f'CoMER+GDCM zero-init must match base logits (diff={max_diff})')
            kv_base = base.greedy_kv(imgs[:1], max_len=6, constrain=False)
            kv_2b = exp2b.greedy_kv(imgs[:1], max_len=6, constrain=False)
            self.assertTrue(torch.equal(kv_base, kv_2b),
                            'greedy_kv must match base at step 0')

        # Optimizer 3rd param group: 2.5x LR for fresh heads.
        opt_no_sched, sched_none = train.build_optimizer(
            exp2b, lr=1e-4, trunk_mult=0.5, warmup=0, cosine=False)
        self.assertIsNone(sched_none)
        lrs = sorted({round(g['lr'], 10) for g in opt_no_sched.param_groups})
        self.assertEqual(lrs, sorted([0.5e-4, 1.0e-4, 2.5e-4]))

        opt_sched, sched = train.build_optimizer(
            exp2b, lr=1e-4, trunk_mult=0.5, warmup=10, cosine=True,
            total_updates=100)
        self.assertIsNotNone(sched)
        # At step 0 of warmup (u=0): mult * 1/10 -> 2.5x group is 2.5x decoder group.
        sched_lrs = sorted({round(lr_v, 10) for lr_v in sched.get_last_lr()})
        self.assertAlmostEqual(sched_lrs[-1] / sched_lrs[1], 2.5, places=5)


class TestRoPE2DAndMTPWarmStart(unittest.TestCase):
    """Warm-start, inference auto-detect, and optimizer groups for RoPE2D + MTP."""

    def test_warm_start_from_legacy_and_inference_drop_mtp(self):
        import torch
        import data
        import model
        v = len(data.VOCAB) + 3
        torch.manual_seed(7)
        legacy = model.MvdNet(
            vocab=v, d_model=64, layers=2, heads=4, dim_ff=128,
            extra_chans=1, foveal_tap=True)
        upgraded = model.MvdNet(
            vocab=v, d_model=64, layers=2, heads=4, dim_ff=128,
            extra_chans=1, foveal_tap=True, rope_2d=True, use_mtp=True)

        how = train.load_weights_compat(upgraded, legacy.state_dict())
        self.assertIn('rope2d-fresh', how)
        self.assertIn('mtp-fresh', how)

        legacy.eval()
        upgraded.eval()
        imgs = torch.randn(2, 4, 64, 128)
        tgt_in = torch.tensor([
            [legacy.bos_id, data.TOKEN_TO_ID['clef-G-2'], data.TOKEN_TO_ID['quarter'],
             data.TOKEN_TO_ID['C'], data.TOKEN_TO_ID['4'], data.TOKEN_TO_ID['bar']],
            [legacy.bos_id, data.TOKEN_TO_ID['clef-F-4'], data.TOKEN_TO_ID['half'],
             data.TOKEN_TO_ID['G'], data.TOKEN_TO_ID['3'], data.TOKEN_TO_ID['bar']],
        ], dtype=torch.long)
        with torch.no_grad():
            self.assertTrue(torch.allclose(
                legacy.encode(imgs), upgraded.encode(imgs), atol=1e-6))
            self.assertTrue(torch.allclose(
                legacy(imgs, tgt_in), upgraded(imgs, tgt_in), atol=1e-5))
            self.assertTrue(torch.equal(
                legacy.greedy_kv(imgs[:1], max_len=6, constrain=False),
                upgraded.greedy_kv(imgs[:1], max_len=6, constrain=False)))

        # Simulate non-zero trained rope2d gate, then load into inference net
        # (use_mtp=False, rope_2d auto-detected from state_dict keys).
        with torch.no_grad():
            upgraded.rope2d.gate.fill_(0.5)
        ckpt_sd = upgraded.state_dict()
        has_rope2d = any(k.startswith('rope2d.') for k in ckpt_sd)
        self.assertTrue(has_rope2d)
        infer_net = model.MvdNet(
            vocab=v, d_model=64, layers=2, heads=4, dim_ff=128,
            extra_chans=1, foveal_tap=True, rope_2d=has_rope2d, use_mtp=False)
        how_infer = train.load_weights_compat(infer_net, ckpt_sd)
        self.assertEqual(how_infer, 'strict+mtp-dropped')
        self.assertFalse(hasattr(infer_net, 'mtp_head'))
        infer_net.eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(
                upgraded.encode(imgs), infer_net.encode(imgs), atol=1e-6))
            self.assertTrue(torch.equal(
                upgraded.greedy_kv(imgs[:1], max_len=6, constrain=False),
                infer_net.greedy_kv(imgs[:1], max_len=6, constrain=False)))

        # Fresh-head 2.5x LR param group includes rope2d and mtp_head.
        opt, _ = train.build_optimizer(
            upgraded, lr=1e-4, trunk_mult=0.5, warmup=0, cosine=False)
        lrs = sorted({round(g['lr'], 10) for g in opt.param_groups})
        self.assertEqual(lrs, sorted([0.5e-4, 1.0e-4, 2.5e-4]))


class TestExtraDataDirAndMaxTokens(unittest.TestCase):
    """--extra-data-dir parsing, pair+hybrid merging, and --max-tokens cap."""

    def test_parse_and_load_extra_data_dirs_and_max_tokens(self):
        from PIL import Image
        import data
        self.assertEqual(train.parse_extra_data_dirs(None), [])
        self.assertEqual(
            train.parse_extra_data_dirs(['a,b', ' c ', 'd,']),
            ['a', 'b', 'c', 'd'])

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            d1 = root / 'extra_flat'
            d2 = root / 'extra_split'
            d1.mkdir()
            (d2 / 'renders').mkdir(parents=True)
            (d2 / 'targets').mkdir(parents=True)

            Image.new('RGB', (64, 64), 'white').save(d1 / 'e1-s0.png')
            toks = ['clef-G-2', 'quarter', 'C', '4', 'bar'] * 4
            (d1 / 'e1-s0-bekern.txt').write_text(' '.join(toks) + '\n', encoding='utf-8')
            (d1 / 'hybrid_staff.jsonl').write_text(
                json.dumps({'img': 'e1-s0.png', 'S': 1, 'table': {'staves': [{}]}}) + '\n',
                encoding='utf-8')

            Image.new('RGB', (64, 64), 'white').save(d2 / 'renders' / 'e2-s0.png')
            (d2 / 'targets' / 'e2-s0-bekern.txt').write_text('quarter C 4 bar\n', encoding='utf-8')
            (d2 / 'hybrid_staff.jsonl').write_text(
                json.dumps({'img': 'e2-s0.png', 'S': 2, 'table': {'staves': [{}, {}]}}) + '\n',
                encoding='utf-8')

            p1, h1 = train.load_extra_data_dir(str(d1))
            p2, h2 = train.load_extra_data_dir(str(d2))
            self.assertEqual(len(p1), 1)
            self.assertEqual(p1[0][2], 'e1-s0')
            self.assertIn('e1-s0.png', h1)
            self.assertEqual(len(p2), 1)
            self.assertEqual(p2[0][2], 'e2-s0')
            self.assertIn('e2-s0.png', h2)

            merged_pairs = p1 + p2
            merged_hidx = {}
            merged_hidx.update(h1)
            merged_hidx.update(h2)
            kept, dc, dd = train.filter_hybrid_pairs(
                merged_pairs, hybrid_index=merged_hidx, allow_on_the_fly=True)
            self.assertEqual(len(kept), 2)
            self.assertEqual((dc, dd), (0, 0))

            # --max-tokens truncation preserves EOS at the end.
            tgt_capped = train.encode_targets(p1, max_tokens=8)
            self.assertEqual(tuple(tgt_capped.shape), (1, 8))
            self.assertEqual(int(tgt_capped[0, -1]), data.EOS_ID)

if __name__ == '__main__':
    unittest.main()
