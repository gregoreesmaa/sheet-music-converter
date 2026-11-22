"""S13c cross-attention x-coordinate supervision (monotonic reading).

Music reads left-to-right within a voice: the decoder's cross-attention
x-centroid should advance monotonically with target position. This
penalizes backward jumps per voice segment (voice tokens reset the
segment; PAD excluded) using only the image x-axis — no symbox rebuild.
"""

import unittest

import torch

import data
import model
import train


def tiny_net(**kw):
    params = dict(vocab=len(data.VOCAB) + 3, d_model=64,
                  layers=1, heads=4, dim_ff=128, anisotropic=False)
    params.update(kw)
    return model.MvdNet(**params)


class TestS13CoordPenalty(unittest.TestCase):
    V = frozenset({10, 11})
    PAD = 99

    def test_monotone_is_zero(self):
        xc = torch.tensor([[0.1, 0.2, 0.2, 0.5, 0.9]])
        tgt = torch.tensor([[1, 2, 3, 4, 5]])
        self.assertEqual(
            model.coord_monotonic_penalty(xc, tgt, self.V, self.PAD,
                                          margin=0.0).item(), 0.0)

    def test_drop_penalized_by_margin(self):
        xc = torch.tensor([[0.1, 0.5, 0.3, 0.6]])
        tgt = torch.tensor([[1, 2, 3, 4]])
        got = model.coord_monotonic_penalty(xc, tgt, self.V, self.PAD,
                                            margin=0.05).item()
        self.assertAlmostEqual(got, (0.5 - 0.3 + 0.05) / 3, places=6)

    def test_voice_token_resets_segment(self):
        # Jump back after a voice token is a new staff, not a regression.
        xc = torch.tensor([[0.8, 0.05, 0.4]])
        tgt = torch.tensor([[4, 10, 5]])
        self.assertEqual(
            model.coord_monotonic_penalty(xc, tgt, self.V, self.PAD,
                                          margin=0.0).item(), 0.0)

    def test_pad_excluded(self):
        xc = torch.tensor([[0.9, 0.0, 0.1]])
        tgt = torch.tensor([[4, self.PAD, 5]])
        # 0.9 -> 0.1 would penalize, but PAD breaks the segment.
        self.assertEqual(
            model.coord_monotonic_penalty(xc, tgt, self.V, self.PAD,
                                          margin=0.0).item(), 0.0)

    def test_single_valid_position_is_zero(self):
        xc = torch.tensor([[0.7]])
        tgt = torch.tensor([[3]])
        self.assertEqual(
            model.coord_monotonic_penalty(xc, tgt, self.V, self.PAD,
                                          margin=0.0).item(), 0.0)


class TestS13CoordCentroids(unittest.TestCase):
    def test_shape_range_and_determinism(self):
        torch.manual_seed(0)
        net = tiny_net()
        net.eval()
        imgs = torch.randn(2, 3, 64, 64)
        tgt = torch.randint(0, len(data.VOCAB), (2, 10))
        with torch.no_grad():
            a = net.x_centroids(imgs, tgt, None)
            b = net.x_centroids(imgs, tgt, None)
        self.assertEqual(list(a.shape), [2, 10])
        self.assertTrue(bool((a >= 0).all() and (a <= 1).all()))
        self.assertTrue(torch.equal(a, b))

    def test_no_hook_leak(self):
        torch.manual_seed(0)
        net = tiny_net()
        net.eval()
        imgs = torch.randn(1, 3, 64, 64)
        tgt = torch.randint(0, len(data.VOCAB), (1, 6))
        with torch.no_grad():
            net.x_centroids(imgs, tgt, None)
        for layer in net.decoder.layers:
            self.assertEqual(dict(layer.multihead_attn._forward_hooks),
                             {})
            self.assertEqual(dict(layer.self_attn._forward_hooks), {})

    def test_masked_columns_get_no_mass(self):
        # Centroid of a fully-right-masked row must sit left of center.
        # Anisotropic 64x64 input -> 8x4 memory grid (32 cells).
        torch.manual_seed(0)
        net = tiny_net(anisotropic=True)
        net.eval()
        imgs = torch.randn(1, 3, 64, 64)
        tgt = torch.randint(0, len(data.VOCAB), (1, 4))
        mem_mask = torch.zeros(1, 32, dtype=torch.bool)
        mem_mask = mem_mask.reshape(1, 8, 4)
        mem_mask[:, :, 2:] = True  # mask right half of the grid
        mem_mask = mem_mask.reshape(1, 32)
        with torch.no_grad():
            a = net.x_centroids(imgs, tgt, mem_mask)
        self.assertTrue(bool((a < 0.5).all()))

    def test_fingerprint_keys(self):
        self.assertIn('coord_weight', train.META_KEYS)
        self.assertIn('coord_weight', train.RESUME_KEYS)
        self.assertIn('coord_every', train.META_KEYS)
        self.assertIn('coord_every', train.RESUME_KEYS)


class TestS13CoordSpread(unittest.TestCase):
    def test_spread_uniform_is_one(self):
        w = torch.full((2, 5, 8), 1 / 8)
        s = model.attention_spread(w)
        self.assertEqual(list(s.shape), [2, 5])
        self.assertTrue(torch.allclose(s, torch.ones(2, 5)))

    def test_spread_one_hot_is_zero(self):
        w = torch.zeros(1, 3, 8)
        w[:, :, 2] = 1.0
        s = model.attention_spread(w)
        self.assertTrue(torch.allclose(s, torch.zeros(1, 3)))

    def test_spread_half_half(self):
        # 1 bit of 2 (4 cells) -> 0.5.
        w = torch.zeros(1, 1, 4)
        w[0, 0, 0] = 0.5
        w[0, 0, 1] = 0.5
        s = model.attention_spread(w)
        self.assertAlmostEqual(s.item(), 0.5, places=5)

    def test_centroids_with_spread(self):
        torch.manual_seed(0)
        net = tiny_net()
        net.eval()
        imgs = torch.randn(1, 3, 64, 64)
        tgt = torch.randint(0, len(data.VOCAB), (1, 6))
        with torch.no_grad():
            xc, spread = net.x_centroids(imgs, tgt, None,
                                         with_spread=True)
            xc2 = net.x_centroids(imgs, tgt, None)
        self.assertEqual(list(xc.shape), [1, 6])
        self.assertEqual(list(spread.shape), [1, 6])
        self.assertTrue(bool((spread >= 0).all() and (spread <= 1).all()))
        self.assertTrue(torch.equal(xc, xc2))


if __name__ == '__main__':
    unittest.main()
