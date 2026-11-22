"""S17 measure-local render-and-compare verifier, slice 1: scoring.

Projects each decoded measure's notes (via S13c cross-attention
x-centroids) back onto observed HeatNet notehead-peak x-positions and
scores the disagreement. Slice 2 (re-decode of flagged measures with
visual-prior constraints) builds on these scores.
"""

import unittest

import torch

import data
import model
import verify_measures as vm


class TestS17Peaks(unittest.TestCase):
    def test_unique_sorted_columns(self):
        m = torch.zeros(8, 32)
        m[2, 5] = 0.9
        m[6, 5] = 0.8   # same column as first: one x
        m[1, 20] = 0.7
        m[0, 0] = 0.4   # below thresh
        self.assertEqual(vm.note_peak_xs(m, thresh=0.5),
                         [5 / 32, 20 / 32])

    def test_empty_map(self):
        self.assertEqual(vm.note_peak_xs(torch.zeros(4, 16)), [])


class TestS17Residual(unittest.TestCase):
    def test_perfect_match(self):
        mean_match, miss, extra = vm.measure_residual(
            [0.1, 0.5, 0.9], [0.1, 0.5, 0.9], tol=0.02)
        self.assertEqual((mean_match, miss, extra), (0.0, 0.0, 0.0))

    def test_miss_and_extra(self):
        # Expected 0.9 has no observed partner; observed 0.3 is extra.
        mean_match, miss, extra = vm.measure_residual(
            [0.1, 0.9], [0.1, 0.3], tol=0.05)
        self.assertAlmostEqual(miss, 0.5)
        self.assertAlmostEqual(extra, 0.5)
        self.assertAlmostEqual(mean_match, (0.0 + 0.6) / 2)

    def test_empty_expected(self):
        mean_match, miss, extra = vm.measure_residual([], [0.2], tol=0.05)
        self.assertEqual((mean_match, miss, extra), (0.0, 0.0, 1.0))

    def test_empty_observed(self):
        mean_match, miss, extra = vm.measure_residual([0.2], [], tol=0.05)
        self.assertEqual(miss, 1.0)
        self.assertEqual(extra, 0.0)
        self.assertTrue(mean_match == float('inf'))


class TestS17Spans(unittest.TestCase):
    BAR = 7
    STEP = 3
    PAD = 99

    def test_splits_on_bars_skips_header(self):
        # header clef/key (no steps) skipped; two note measures kept.
        ids = [1, 2, self.BAR, 3, 3, self.BAR, 3, self.PAD, self.PAD]
        self.assertEqual(
            vm.measure_spans(ids, self.BAR, {self.STEP}, self.PAD),
            [(3, 5), (6, 7)])

    def test_no_bars_single_span(self):
        ids = [3, 3, 3]
        self.assertEqual(
            vm.measure_spans(ids, self.BAR, {self.STEP}, self.PAD), [(0, 3)])


class TestS17ScoreAndPick(unittest.TestCase):
    BAR = 7
    STEP = 3
    PAD = 99

    def test_score_flags_bad_measure(self):
        # Measure 0 matches peaks; measure 1's notes are all missed.
        ids = [self.BAR, self.STEP, self.BAR, self.STEP, self.BAR]
        xc = torch.tensor([[0.0, 0.2, 0.0, 0.8, 0.0]])
        out = vm.score_strip_measures(
            ids, xc, self.BAR, {self.STEP}, self.PAD, [0.2], tol=0.05)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]['miss'], 0.0)
        self.assertEqual(out[1]['miss'], 1.0)
        self.assertGreater(out[1]['residual'], out[0]['residual'])

    def test_pick_min_residual_hyp(self):
        obs = [0.2, 0.8]
        ids_a = [self.BAR, self.STEP, self.BAR]          # 1 note
        xc_a = torch.tensor([[0.0, 0.2, 0.0]])           # matches
        ids_b = [self.BAR, self.STEP, self.BAR]
        xc_b = torch.tensor([[0.0, 0.5, 0.0]])           # matches nothing
        best = vm.pick_min_residual_hyp(
            [(ids_a, xc_a), (ids_b, xc_b)], self.BAR, {self.STEP},
            self.PAD, obs, tol=0.05)
        self.assertEqual(best, 0)
        # Order-independent: worse hyp first still loses.
        best = vm.pick_min_residual_hyp(
            [(ids_b, xc_b), (ids_a, xc_a)], self.BAR, {self.STEP},
            self.PAD, obs, tol=0.05)
        self.assertEqual(best, 1)

    def test_live_centroids_flow(self):
        # Real x_centroids output flows through the scorer end to end.
        torch.manual_seed(0)
        net = model.MvdNet(vocab=len(data.VOCAB) + 3, d_model=64,
                           layers=1, heads=4, dim_ff=128,
                           anisotropic=False)
        net.eval()
        imgs = torch.randn(1, 3, 64, 64)
        bar = data.TOKEN_TO_ID['bar']
        step = data.TOKEN_TO_ID['quarter']
        tgt = torch.tensor([[bar, step, step, bar, step, bar]])
        with torch.no_grad():
            xc = net.x_centroids(imgs, tgt, None)
        out = vm.score_strip_measures(
            tgt[0].tolist(), xc, bar, {step}, net.pad_id,
            [0.25, 0.75], tol=0.05)
        self.assertEqual(len(out), 2)
        for row in out:
            self.assertIn('residual', row)
            self.assertGreaterEqual(row['residual'], 0.0)


if __name__ == '__main__':
    unittest.main()
