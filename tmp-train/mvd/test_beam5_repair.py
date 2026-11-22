"""TDD for the measure-atomic beam-5 re-decode repair loop (S18-adjacent).

CPU-only: stub nets return canned id rows; GPU execution stays behind
the runner's --i-understand-gpu + GUARD_GPU_OK=1 gate (never exercised
here). Run with the repo-root .venv python:

    .\\.venv\\Scripts\\python.exe -m unittest tmp-train.mvd.test_beam5_repair -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data  # noqa: E402
import measure_decode as mdec  # noqa: E402


def toks_to_ids(toks):
    return [data.TOKEN_TO_ID[t] for t in toks]


class StubNet:
    """Stand-in with scripted beam outputs (list of id-rows per call)."""

    def __init__(self, beams_per_call):
        self.beams_per_call = beams_per_call
        self.calls = []

    def beam_search(self, ten, beam=5, **kw):
        self.calls.append({'beam': beam, 'kw': dict(kw)})
        return self.beams_per_call


def real_measure_pair():
    """One strict-closing measure + a truncated twin from a real strip."""
    ref = (Path(__file__).resolve().parent.parent.parent
           / 'data-gen' / 'score-corpus' / 'gate20' / 'sys-dev' / 'refs'
           / 'ds0091-3-s0-bekern.txt')
    toks = ref.read_text(encoding='utf-8').split()
    header, measures, _trailing = mdec.split_measures(toks)
    closing = None
    for m in measures:
        if m and mdec.measure_closes(header, m):
            closing = m
            break
    assert closing is not None, 'fixture has no closing measure'
    # Dangling duration with no pitch: an open event that cannot close.
    broken = list(closing) + ['quarter']
    assert not mdec.measure_closes(header, broken), 'twin still closes'
    return header, closing, broken


class TestFirstMeasureBody(unittest.TestCase):
    def test_cuts_at_first_bar(self):
        import beam5_repair_run as b5
        self.assertEqual(
            b5.first_measure_body(['quarter', 'C', '4', 'bar', 'half', 'D', '4']),
            ['quarter', 'C', '4'])

    def test_no_bar_keeps_all(self):
        import beam5_repair_run as b5
        self.assertEqual(b5.first_measure_body(['quarter', 'C', '4']),
                         ['quarter', 'C', '4'])


class TestRepairMeasures(unittest.TestCase):
    def test_closing_measure_kept_without_beam_call(self):
        import beam5_repair_run as b5
        header, closing, _broken = real_measure_pair()
        net = StubNet([])
        fixed, prov = b5.repair_measures(
            net, None, None, header, [closing], True)
        self.assertEqual(fixed, header + closing + ['bar'])
        self.assertEqual(prov, ['kept'])
        self.assertEqual(net.calls, [])

    def test_failed_measure_splices_first_closing_beam(self):
        import beam5_repair_run as b5
        header, closing, broken = real_measure_pair()
        garbage = toks_to_ids(['bar', 'bar'])
        net = StubNet([toks_to_ids(closing), garbage])
        fixed, prov = b5.repair_measures(
            net, None, None, header, [broken], True)
        self.assertEqual(fixed, header + closing + ['bar'])
        self.assertEqual(prov, ['beam5[0]'])

    def test_no_closing_candidate_falls_back(self):
        import beam5_repair_run as b5
        header, _closing, broken = real_measure_pair()
        net = StubNet([toks_to_ids(['bar', 'bar'])])
        fixed, prov = b5.repair_measures(
            net, None, None, header, [broken], True)
        want, _, _ = mdec.fallback_repair(header + broken + ['bar'])
        self.assertEqual(fixed, want)
        self.assertIn(prov[0], ('trimmed', 'spliced'))

    def test_span_budget_capped(self):
        import beam5_repair_run as b5
        header, closing, _broken = real_measure_pair()
        long_broken = (closing * 30)[:200] + ['quarter']
        net = StubNet([toks_to_ids(closing)])
        b5.repair_measures(net, None, None, header, [long_broken], True)
        self.assertEqual(len(net.calls), 1)
        self.assertLessEqual(net.calls[0]['kw']['max_len'], 128)
        self.assertEqual(net.calls[0]['kw']['max_len'],
                         min(len(long_broken) + 32, 128))

    def test_beam_body_cut_before_rank(self):
        import beam5_repair_run as b5
        header, closing, broken = real_measure_pair()
        # Beam 0 trails a whole second measure; only its first body
        # may be spliced (plus the single terminating bar).
        extra = toks_to_ids(closing + ['bar'] + closing)
        net = StubNet([extra])
        fixed, prov = b5.repair_measures(
            net, None, None, header, [broken], True)
        self.assertEqual(prov, ['beam5[0]'])
        self.assertEqual(fixed.count('bar'), 1)


class TestRunnerGate(unittest.TestCase):
    def test_main_refuses_without_gpu_flag(self):
        import beam5_repair_run as b5
        with self.assertRaises(SystemExit):
            b5.main(['--ckpt', 'x.pt', '--tags-json', 'y.json',
                     '--outdir', 'z'])


if __name__ == '__main__':
    unittest.main()
