"""CPU-only unit tests for Iter 15 H100 Speed Pack:
1. FIXED_HW_BUCKETS prepends (896, 640) and (640, 896) so 896px progressive-res batches don't snap to 1024x768.
2. CLI parsing & behavior for --prog-res-schedule, --cass-start-frac, and --eval-ema-only.
3. neg_control spread guard (<= 1 zero-byte neg_control row per batch) in 2D bucket batching
   and MultiScaleBatchSampler(aspect_len_buckets=...).
4. collate_image_tensors fast stride-64 p85 quantile accuracy (abs diff < 0.02 vs full quantile).
"""

from __future__ import annotations

import os

# Force strict CPU-only execution.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
import random
import sys
import unittest
from unittest import mock

import torch

_MVD_DIR = Path(__file__).resolve().parent
if str(_MVD_DIR) not in sys.path:
    sys.path.insert(0, str(_MVD_DIR))

import pretrain_ijepa_rism_v2 as pij  # noqa: E402
import train_uni_v2 as tv2  # noqa: E402
import uni_omr as uo  # noqa: E402


class TestIter15SpeedPack(unittest.TestCase):
    # ------------------------------------------------------------------
    # 1. FIXED_HW_BUCKETS 896x640 / 640x896 snap
    # ------------------------------------------------------------------
    def test_fixed_hw_buckets_896x640_snap(self) -> None:
        self.assertIn((896, 640), uo.FIXED_HW_BUCKETS)
        self.assertIn((640, 896), uo.FIXED_HW_BUCKETS)
        self.assertEqual(uo.FIXED_HW_BUCKETS[0], (896, 640))
        # Portrait 896px page snaps cleanly to (896, 640), not (1024, 768)
        self.assertEqual(uo._snap_to_fixed_hw_bucket(896, 634), (896, 640))
        self.assertEqual(uo._snap_to_fixed_hw_bucket(880, 624), (896, 640))
        # Landscape 896px page snaps cleanly to (640, 896), not (768, 1024)
        self.assertEqual(uo._snap_to_fixed_hw_bucket(634, 896), (640, 896))
        # Slightly wider 896px portrait still snaps safely up without cropping
        self.assertEqual(uo._snap_to_fixed_hw_bucket(896, 700), (1024, 768))

    # ------------------------------------------------------------------
    # 2. CLI parsing + runtime behavior for new Iter 15 speed flags
    # ------------------------------------------------------------------
    def test_cli_parsing_prog_res_cass_start_eval_ema_only(self) -> None:
        parser = tv2.build_arg_parser()
        # Defaults remain off / backward-compatible
        args_def = parser.parse_args([])
        self.assertEqual(args_def.prog_res_schedule, "")
        self.assertAlmostEqual(args_def.cass_start_frac, 0.0)
        self.assertFalse(args_def.eval_ema_only)

        # Explicit flags parse cleanly
        args_on = parser.parse_args(
            [
                "--prog-res-schedule",
                "0.0:896,0.35:1152,0.65:1408",
                "--cass-start-frac",
                "0.35",
                "--eval-ema-only",
            ]
        )
        self.assertEqual(args_on.prog_res_schedule, "0.0:896,0.35:1152,0.65:1408")
        self.assertAlmostEqual(args_on.cass_start_frac, 0.35)
        self.assertTrue(args_on.eval_ema_only)

        sched = tv2.parse_prog_res_schedule(args_on.prog_res_schedule)
        self.assertEqual(sched, [(0.0, 896), (0.35, 1152), (0.65, 1408)])
        self.assertEqual(tv2.resolve_prog_res_dim(sched, 0.0), 896)
        self.assertEqual(tv2.resolve_prog_res_dim(sched, 0.34), 896)
        self.assertEqual(tv2.resolve_prog_res_dim(sched, 0.35), 1152)
        self.assertEqual(tv2.resolve_prog_res_dim(sched, 0.64), 1152)
        self.assertEqual(tv2.resolve_prog_res_dim(sched, 0.65), 1408)
        self.assertEqual(tv2.resolve_prog_res_dim(sched, 1.0), 1408)

    def test_cass_start_frac_skips_early_steps(self) -> None:
        parser = tv2.build_arg_parser()
        # 4 steps with cass_start_frac=0.50 -> steps 1 (0.25) skip CASS, steps 2..4 (>=0.50) enable CASS
        args = parser.parse_args(
            [
                "--arch-preset",
                "tiny",
                "--n-axial-layers",
                "1",
                "--n-global-layers",
                "1",
                "--cass-prob",
                "0.30",
                "--cass-start-frac",
                "0.50",
                "--device",
                "cpu",
                "--no-bf16",
                "--optimizer",
                "adamw",
                "--steps",
                "4",
                "--batch-size",
                "1",
                "--grad-accum",
                "1",
                "--seq-len",
                "12",
                "--prefetch-threads",
                "0",
                "--log-every",
                "4",
                "--eval-every",
                "0",
                "--gpu-ema-decay",
                "0.0",
                "--save-every-minutes",
                "0.0",
            ]
        )
        seen_cass_probs: list[float] = []
        orig_fwd = tv2.forward_and_loss_v2

        def _spy_fwd(*a, **kw):
            seen_cass_probs.append(float(kw.get("cass_prob", 0.0)))
            return orig_fwd(*a, **kw)

        with mock.patch.object(tv2, "forward_and_loss_v2", side_effect=_spy_fwd):
            tv2.run_train_v2(args)

        self.assertEqual(len(seen_cass_probs), 4)
        self.assertEqual(seen_cass_probs[0], 0.0)
        self.assertAlmostEqual(seen_cass_probs[1], 0.30)
        self.assertAlmostEqual(seen_cass_probs[2], 0.30)
        self.assertAlmostEqual(seen_cass_probs[3], 0.30)

    # ------------------------------------------------------------------
    # 3. neg_control spread guard (<= 1 zero-byte row per batch) & MultiScaleBatchSampler 2D
    # ------------------------------------------------------------------
    def test_neg_control_spread_guard_and_multiscale_2d_buckets(self) -> None:
        # 128 samples across 2 aspect buckets: 120 positive + 8 zero-byte neg_control rows
        # Even if all 8 neg_control rows are in bucket 0, every batch of size 8 must have <= 1 zero-byte row.
        n_total = 128
        bucket_ids = [0 if i < 64 else 1 for i in range(n_total)]
        sample_bytes = [0 if i < 8 else (200 + (i * 37) % 4000) for i in range(n_total)]

        batches = uo.build_2d_batches_with_neg_guard(
            epoch_indices=list(range(n_total)),
            sample_bucket_id=bucket_ids,
            sample_bytes=sample_bytes,
            n_buckets=2,
            batch_size=8,
            length_window=32,
            shuffle=True,
            rng=random.Random(123),
        )
        flat = [idx for b in batches for idx in b]
        self.assertEqual(sorted(flat), list(range(n_total)))
        for b in batches:
            self.assertLessEqual(len(b), 8)
            n_neg = sum(1 for idx in b if sample_bytes[idx] == 0)
            self.assertLessEqual(n_neg, 1, f"Batch had {n_neg} zero-byte neg_control rows: {b}")

        # Verify MultiScaleBatchSampler wires 2D aspect_len_buckets + neg_control spread guard
        sampler = uo.MultiScaleBatchSampler(
            num_samples=n_total,
            batch_size=8,
            base_dim=896,
            shuffle=True,
            seed=7,
            aspect_len_buckets=(bucket_ids, sample_bytes, 2),
            length_window=32,
        )
        sampler_batches = list(sampler)
        sampler_flat = [idx for b in sampler_batches for idx, _dim in b]
        self.assertEqual(sorted(sampler_flat), list(range(n_total)))
        for b in sampler_batches:
            dims = {dim for _idx, dim in b}
            self.assertEqual(len(dims), 1, "All items in a MultiScaleBatchSampler batch must share target_dim")
            n_neg = sum(1 for idx, _dim in b if sample_bytes[idx] == 0)
            self.assertLessEqual(n_neg, 1, f"MultiScaleBatchSampler batch had {n_neg} neg_control rows")

    # ------------------------------------------------------------------
    # 4. collate_image_tensors fast stride-64 quantile accuracy (< 0.02 diff)
    # ------------------------------------------------------------------
    def test_collate_image_tensors_fast_quantile_accuracy(self) -> None:
        g = torch.Generator().manual_seed(42)
        # Synthetic archival page: warm parchment background (~0.88, 0.83, 0.72) + dark ink strokes
        h, w = 896, 640
        img_a = torch.empty((3, h, w), dtype=torch.float32)
        img_a[0] = 0.88 + 0.04 * torch.randn((h, w), generator=g)
        img_a[1] = 0.83 + 0.04 * torch.randn((h, w), generator=g)
        img_a[2] = 0.72 + 0.04 * torch.randn((h, w), generator=g)
        # Add 12% dark ink lines/blocks
        img_a[:, 100:220, 40:600] *= 0.15
        img_a = img_a.clamp(0.0, 1.0)

        # Slightly larger second image forces padding of img_a in collate_image_tensors
        img_b = torch.ones((3, 912, 656), dtype=torch.float32) * 0.85

        full_q85 = torch.quantile(img_a.reshape(3, -1), 0.85, dim=-1)
        batched = pij.collate_image_tensors([img_a, img_b])
        self.assertEqual(tuple(batched.shape), (2, 3, 912, 656))
        # Read padded corner of img_a (outside original h x w)
        padded_fill = batched[0, :, -1, -1]
        max_abs_diff = float((padded_fill - full_q85).abs().max().item())
        self.assertLess(
            max_abs_diff,
            0.02,
            f"Fast stride-64 p85 fill diverged from full quantile: {padded_fill.tolist()} vs {full_q85.tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
