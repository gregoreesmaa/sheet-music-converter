"""Unit tests for width-bucketed batching, async prefetching, and fast flags."""

from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast_loader  # noqa: E402
import train  # noqa: E402


class TestFastLoader(unittest.TestCase):
    def test_width_bucketing_cuts_padding_and_is_deterministic(self):
        rng = random.Random(7)
        pairs = [(f'dir/s{i:04d}.png', f'dir/s{i:04d}-bekern.txt', f's{i:04d}')
                 for i in range(200)]
        # Realistic rectified strip widths spanning 600..3400 px.
        widths = [int(rng.uniform(600, 3400)) for _ in pairs]
        wmap = {pairs[i][0]: widths[i] for i in range(len(pairs))}

        # Random baseline batches.
        rand_pairs = list(pairs)
        random.Random(0).shuffle(rand_pairs)
        rand_batches = [rand_pairs[k:k + 4] for k in range(0, len(rand_pairs), 4)]
        rand_ratio = fast_loader.padding_overhead_ratio(rand_batches, wmap)

        # Width-bucketed batches (same epoch -> identical; diff epoch -> fresh order).
        b0a = fast_loader.epoch_bucketed_batches(pairs, widths, 4, base_seed=42, epoch=0)
        b0b = fast_loader.epoch_bucketed_batches(pairs, widths, 4, base_seed=42, epoch=0)
        b1 = fast_loader.epoch_bucketed_batches(pairs, widths, 4, base_seed=42, epoch=1)
        self.assertEqual(b0a, b0b)
        self.assertNotEqual(b0a, b1)
        self.assertEqual(sum(len(b) for b in b0a), len(pairs))

        buck_ratio = fast_loader.padding_overhead_ratio(b0a, wmap)
        self.assertGreater(rand_ratio, 1.25)
        self.assertLess(buck_ratio, 1.08,
                        f'bucketed padding ratio too high: {buck_ratio:.3f} vs random {rand_ratio:.3f}')

    def test_async_prefetcher_exact_order_and_values(self):
        items = [(s, 0, [f'p{s}']) for s in range(1, 13)]

        def fake_collate(step, epoch, batch):
            return (torch.full((2, 3), float(step)), batch[0])

        with fast_loader.AsyncBatchPrefetcher(
            items, fake_collate, num_workers=2, prefetch=4, pin_memory=False
        ) as pf:
            got = list(pf)

        self.assertEqual(len(got), 12)
        for idx, (step, epoch, batch, (t, name)) in enumerate(got, start=1):
            self.assertEqual(step, idx)
            self.assertEqual(name, f'p{idx}')
            self.assertEqual(float(t[0, 0]), float(idx))

    def test_train_resume_legacy_compat_for_fast_flags(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            legacy = {k: 0 for k in train.RESUME_KEYS
                      if k not in train.LEGACY_DEFAULTS}
            (out / 'meta.json').write_text(json.dumps(legacy), encoding='utf-8')
            cur_default = dict(legacy)
            cur_default.update(train.LEGACY_DEFAULTS)
            self.assertIsNone(train.check_resume(out, cur_default))
            for flag, val in (('bf16', True), ('channels_last', True),
                              ('width_bucket', True), ('prefetch', 2)):
                cur_on = dict(cur_default)
                cur_on[flag] = val
                msg = train.check_resume(out, cur_on)
                self.assertIsNotNone(msg, f'{flag} non-default should refuse')
                self.assertIn(flag, msg)


if __name__ == '__main__':
    unittest.main()
