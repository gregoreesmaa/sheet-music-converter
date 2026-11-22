"""Spike S9: test-time per-strip rank-8 adapter (TDD).

RED-first tests for the unsupervised test-time adapter: a low-rank
linear map on the encoder memory (identity at init), fit per strip
with a few entropy-minimization steps on the strip's own greedy
pseudo-targets. No retraining, no checkpoint change; disabled
(adapt_steps=0) keeps the legacy decode path bitwise identical and
no adapter state leaks across strips.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data
import model
import tt_adapter


def _net():
    # Live vocab: greedy/beam constrain masks use the real grammar
    # id space, so the net vocab must match data.VOCAB + specials.
    return model.MvdNet(vocab=len(data.VOCAB) + 3, anisotropic=True,
                        d_model=32, layers=1)


class StripAdapterTest(unittest.TestCase):
    def test_identity_at_init(self):
        import torch
        ad = tt_adapter.StripAdapter(32, rank=8)
        mem = torch.randn(1, 40, 32)
        self.assertTrue(torch.equal(ad(mem), mem))

    def test_param_count_is_low_rank(self):
        ad = tt_adapter.StripAdapter(32, rank=8)
        n = sum(p.numel() for p in ad.parameters())
        self.assertEqual(n, 2 * 32 * 8)

    def test_init_deterministic(self):
        import torch
        a = tt_adapter.StripAdapter(32, rank=8)
        b = tt_adapter.StripAdapter(32, rank=8)
        for pa, pb in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(pa, pb))

    def test_shape_preserved(self):
        import torch
        ad = tt_adapter.StripAdapter(32, rank=8)
        mem = torch.randn(2, 40, 32)
        self.assertEqual(tuple(ad(mem).shape), (2, 40, 32))


class AdaptStripTest(unittest.TestCase):
    def test_adapt_moves_memory(self):
        import torch
        net = _net()
        net.eval()
        img = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            mem0 = net.encode(img)
        out = tt_adapter.adapt_strip(net, img, steps=2, max_len=16)
        self.assertEqual(tuple(out['mem'].shape), tuple(mem0.shape))
        self.assertFalse(bool(torch.equal(out['mem'], mem0)))
        self.assertTrue(float(out['entropy_before']) >= 0.0)
        self.assertTrue(float(out['entropy_after']) >= 0.0)

    def test_adapt_is_deterministic(self):
        import torch
        net = _net()
        net.eval()
        img = torch.randn(1, 3, 256, 256)
        a = tt_adapter.adapt_strip(net, img, steps=2, max_len=16)
        b = tt_adapter.adapt_strip(net, img, steps=2, max_len=16)
        self.assertTrue(torch.equal(a['mem'], b['mem']))

    def test_no_state_leaks_across_strips(self):
        import torch
        net = _net()
        net.eval()
        ia = torch.randn(1, 3, 256, 256)
        ib = torch.randn(1, 3, 256, 320)
        solo = tt_adapter.adapt_strip(net, ib, steps=2, max_len=16)
        tt_adapter.adapt_strip(net, ia, steps=2, max_len=16)
        again = tt_adapter.adapt_strip(net, ib, steps=2, max_len=16)
        self.assertTrue(torch.equal(solo['mem'], again['mem']))

    def test_adapt_uses_own_greedy_pseudo_targets(self):
        import torch
        net = _net()
        net.eval()
        img = torch.randn(1, 3, 256, 256)
        out = tt_adapter.adapt_strip(net, img, steps=1, max_len=16)
        self.assertGreater(out['pseudo_len'], 0)
        self.assertLessEqual(out['pseudo_len'], 16)


class DisabledPathTest(unittest.TestCase):
    def test_zero_steps_matches_plain_beam(self):
        import torch
        net = _net()
        net.eval()
        img = torch.randn(1, 3, 256, 128)
        with torch.no_grad():
            a = net.beam_search(img, beam=1, strict=True,
                                max_len=8)
            b = net.beam_search(img, beam=1, strict=True,
                                max_len=8, adapt_steps=0)
        self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()
