"""Spike S6: MaxSAT confusion-lattice repair (TDD).

RED-first tests for the pure pieces: ``fallback_repair`` stat
extraction (clean / trimmed / spliced), tie-balance checking, and
lexicographic candidate ranking (fewer splices > fewer trims >
closed ties > higher logprob). Net-dependent rescoring is proven by
the 20-piece repair eval, not here.

Bekern note order is duration-pitch-octave (e.g. ``quarter A 4``),
verified against sys-dev refs.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data
import measure_decode as mdec


HEAD = ['clef-G-2', 'key+0', 'time-4-4']
FULL44 = ['quarter', 'C', '4', 'quarter', 'D', '4', 'quarter', 'E', '4',
          'quarter', 'F', '4']


class FallbackRepairTest(unittest.TestCase):
    def test_clean_stream_untouched(self):
        toks = HEAD + FULL44 + ['bar']
        fixed, touched, spliced = mdec.fallback_repair(toks)
        self.assertEqual(fixed, toks)
        self.assertEqual((touched, spliced), (0, 0))

    def test_partial_measure_passes_through_untouched(self):
        # Strict-incomplete but loose-whole: trim returns the input
        # itself, so it counts as untouched (trim-to-self semantics).
        toks = HEAD + ['quarter', 'C', '4'] + ['bar']
        fixed, touched, spliced = mdec.fallback_repair(toks)
        self.assertEqual((fixed, touched, spliced), (toks, 0, 0))

    def test_garbage_measure_trimmed_not_spliced(self):
        toks = HEAD + ['grace', 'grace', 'tie-stop'] + ['bar']
        fixed, touched, spliced = mdec.fallback_repair(toks)
        self.assertEqual((touched, spliced), (1, 0))
        self.assertEqual(fixed, HEAD + ['bar'])
        self.assertNotIn('bar-rest', fixed)


class TiesBalancedTest(unittest.TestCase):
    def test_balanced(self):
        import lattice_repair as lr
        toks = ['quarter', 'C', '4', 'tie-start', 'quarter', 'C', '4',
                'tie-stop']
        self.assertTrue(lr.ties_balanced(toks))

    def test_unclosed(self):
        import lattice_repair as lr
        self.assertFalse(lr.ties_balanced(['quarter', 'C', '4',
                                           'tie-start']))

    def test_lone_stop(self):
        import lattice_repair as lr
        self.assertFalse(lr.ties_balanced(['quarter', 'C', '4',
                                           'tie-stop']))


class RankTest(unittest.TestCase):
    def test_prefers_bar_math_clean(self):
        import lattice_repair as lr
        clean = HEAD + FULL44 + ['bar']
        overfull = HEAD + FULL44 + ['quarter', 'G', '4', 'bar']
        # Overfull is invisible to trim counts (trim-to-self) but
        # strict-fails; it loses despite the better logprob.
        best = lr.rank_candidates([(overfull, 100.0), (clean, -50.0)])
        self.assertEqual(best, clean)

    def test_prefers_closed_ties(self):
        import lattice_repair as lr
        open_tie = HEAD + FULL44 + ['tie-start', 'bar']
        closed = HEAD + FULL44 + ['bar']
        best = lr.rank_candidates([(open_tie, 10.0), (closed, 9.0)])
        self.assertEqual(best, closed)

    def test_logprob_breaks_ties(self):
        import lattice_repair as lr
        a = HEAD + FULL44 + ['bar']
        b = HEAD + FULL44 + ['bar']
        best = lr.rank_candidates([(a, 1.0), (b, 2.0)])
        self.assertEqual(best, b)


class RowIdsTest(unittest.TestCase):
    def test_keeps_cutting_eos(self):
        import lattice_repair as lr
        nv = len(data.VOCAB)
        self.assertEqual(lr.row_to_ids([5, 10, data.EOS_ID,
                                        data.PAD_ID], data.EOS_ID, nv),
                         [5, 10, data.EOS_ID])

    def test_cuts_at_pad_without_eos(self):
        import lattice_repair as lr
        nv = len(data.VOCAB)
        self.assertEqual(lr.row_to_ids([5, data.PAD_ID, 7],
                                       data.EOS_ID, nv), [5])

    def test_ids_roundtrip_tokens(self):
        import lattice_repair as lr
        nv = len(data.VOCAB)
        ids = [5, 10, data.EOS_ID]
        toks = lr.ids_to_tokens(ids, nv)
        self.assertEqual(toks, [data.VOCAB[5], data.VOCAB[10]])


class RepairBeamsTest(unittest.TestCase):
    def test_tiny_net_end_to_end(self):
        import torch
        import lattice_repair as lr
        import model as model_mod
        torch.manual_seed(0)
        net = model_mod.MvdNet(vocab=len(data.VOCAB) + 3, d_model=64,
                               layers=1, heads=8)
        net.eval()
        imgs = torch.randn(1, 3, 64, 64)
        nv = len(data.VOCAB)
        rows = [[5, 10, data.EOS_ID, data.PAD_ID],
                [3, data.EOS_ID, data.PAD_ID, data.PAD_ID],
                []]
        with torch.no_grad():
            won = lr.repair_beams(net, imgs, None, rows, nv)
            again = lr.repair_beams(net, imgs, None, rows, nv)
        # Ids survive the score path (the S6 crash class errors
        # here); output is repaired token strings, deterministic.
        self.assertTrue(all(isinstance(t, str) for t in won))
        self.assertEqual(won, again)


if __name__ == '__main__':
    unittest.main()
