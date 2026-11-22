#!/usr/bin/env python3
"""T1.6b smoke tests for probe_gate_acc.py (fast: no checkpoint, no GPU,
no 26-system eval).

Covers the pure helpers the probe is built from: swap-donor rotation,
white-blank construction, the legacy state-dict gate, the never-pooled
metric summarizer, the report schema, and the run_variant wiring via a
stub net with engineered logits.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data  # noqa: E402
import train  # noqa: E402
import probe_gate_acc as probe  # noqa: E402


def _counts(body_ok, body_n, bar_ok=1, bar_n=1):
    return {'header': [0, 0], 'bar': [bar_ok, bar_n],
            'body': [body_ok, body_n], 'syntax': [0, 0]}


class TestSwapPeer(unittest.TestCase):
    def test_next_and_wrap(self):
        self.assertEqual(probe.swap_peer(0, 26), 1)
        self.assertEqual(probe.swap_peer(12, 26), 13)
        self.assertEqual(probe.swap_peer(25, 26), 0)

    def test_never_self(self):
        for n in (1, 2, 26):
            for i in range(n):
                if n > 1:
                    self.assertNotEqual(probe.swap_peer(i, n), i)
                else:
                    self.assertEqual(probe.swap_peer(0, 1), 0)


class TestMakeBlank(unittest.TestCase):
    def test_white_per_channel_and_shape(self):
        import torch
        blank = probe.make_blank(256, 100, 'cpu')
        self.assertEqual(list(blank.shape), [1, 3, 256, 100])
        for c in range(3):
            self.assertTrue(torch.equal(
                blank[0, c],
                train.WHITE[c, 0, 0].expand(256, 100)),
                'channel %d is not train.WHITE' % c)

    def test_matches_collate_fill(self):
        import torch
        blank = probe.make_blank(256, 32, 'cpu')
        fill = train.WHITE.expand(3, 256, 32)
        self.assertTrue(torch.equal(blank[0], fill))


class TestLegacyStateGate(unittest.TestCase):
    def test_exact_mem_norm_missing_ok(self):
        ok, _ = probe.check_legacy_state(
            ['mem_norm.weight', 'mem_norm.bias'], [])
        self.assertTrue(ok)

    def test_extra_missing_refused(self):
        ok, reason = probe.check_legacy_state(
            ['mem_norm.weight', 'mem_norm.bias', 'out.weight'], [])
        self.assertFalse(ok)
        self.assertIn('out.weight', reason)

    def test_unexpected_refused(self):
        ok, reason = probe.check_legacy_state(
            ['mem_norm.weight', 'mem_norm.bias'], ['rogue.weight'])
        self.assertFalse(ok)
        self.assertIn('rogue.weight', reason)

    def test_empty_missing_refused(self):
        ok, _ = probe.check_legacy_state([], [])
        self.assertFalse(ok)


class TestSummarize(unittest.TestCase):
    def test_gaps_use_separate_domains(self):
        counts = {
            'clean': _counts(80, 100, bar_ok=9, bar_n=10),
            'degraded': _counts(10, 100, bar_ok=1, bar_n=10),
            'white_blank': _counts(30, 100, bar_ok=5, bar_n=10),
            'swap': _counts(70, 100),
        }
        _acc, m = probe.summarize(counts)
        self.assertAlmostEqual(m['clean_body_acc'], 0.80)
        self.assertAlmostEqual(m['clean_body_gap'], 0.80 - 0.30)
        self.assertAlmostEqual(m['clean_swap_gap'], 0.80 - 0.70)
        self.assertAlmostEqual(m['degraded_body_acc'], 0.10)
        self.assertAlmostEqual(m['degraded_body_gap'], 0.10 - 0.30)
        self.assertAlmostEqual(m['clean_bar_acc'], 0.90)
        self.assertAlmostEqual(m['degraded_bar_acc'], 0.10)

    def test_clean_and_degraded_never_pooled(self):
        # A pooled body accuracy would be 0.45 here; the report must keep
        # 0.80 and 0.10 separate with no trace of the pooled value.
        counts = {
            'clean': _counts(80, 100),
            'degraded': _counts(10, 100),
            'white_blank': _counts(0, 100),
            'swap': _counts(0, 100),
        }
        _acc, m = probe.summarize(counts)
        self.assertNotAlmostEqual(m['clean_body_acc'],
                                  m['degraded_body_acc'])
        for v in m.values():
            self.assertNotAlmostEqual(v, 0.45)


class TestReportSchema(unittest.TestCase):
    def test_headline_keys_and_separate_sections(self):
        counts = {d: _counts(1, 2) for d in probe.DOMAINS}
        rep = probe.build_report('ckpt.pt', 123, 'abc', True, 256, 'cpu',
                                 '2.x', 26, [{'stem': 'x'}], counts)
        for key in ('clean_body_acc', 'clean_body_gap', 'clean_swap_gap',
                    'degraded_body_acc', 'degraded_body_gap',
                    'clean_bar_acc', 'degraded_bar_acc'):
            self.assertIn(key, rep['metrics'])
        for dom in ('clean', 'degraded', 'white_blank', 'swap'):
            self.assertIn(dom, rep['per_domain_class'])
            for cls in ('header', 'bar', 'body', 'syntax'):
                cell = rep['per_domain_class'][dom][cls]
                self.assertIn('acc', cell)
                self.assertIn('n', cell)
        self.assertEqual(rep['sys_dev_systems'], 26)
        self.assertEqual(len(rep['per_system']), 1)
        self.assertEqual(rep['test_pages_touched'], 0)


class TestRunVariantWiring(unittest.TestCase):
    def test_perfect_logits_score_all_correct(self):
        import torch

        class StubNet:
            def __call__(self, img, tin):
                t = tin.shape[1]
                v = len(data.VOCAB) + 3
                ref = [1, 2, 3, 4][:t]
                logits = torch.zeros(1, t, v)
                for i, tok in enumerate(ref):
                    logits[0, i, tok] = 10.0
                return logits

        got = probe.run_variant(StubNet(), torch.zeros(1, 3, 8, 8),
                                [1, 2, 3, 4], 'cpu')
        total_ok = sum(v[0] for v in got.values())
        total_n = sum(v[1] for v in got.values())
        self.assertEqual(total_ok, total_n)
        self.assertGreater(total_n, 0)


if __name__ == '__main__':
    unittest.main()
