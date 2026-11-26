"""CPU-only unit tests for Iter 13 Stage 2 Tier 1 + Tier 2 training speed optimizations."""

from __future__ import annotations

import os

# Force strict CPU-only execution (never touch GPU while training runs).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

_MVD_DIR = Path(__file__).resolve().parent
if str(_MVD_DIR) not in sys.path:
    sys.path.insert(0, str(_MVD_DIR))

import train_uni_v2 as tv2  # noqa: E402
from uni_omr import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    FIXED_HW_BUCKETS,
    PAD_ID,
    VOCAB_SIZE,
    AsyncThreadPrefetcher,
    InkTokenRouter,
    UniOMRConfig,
    UniOMRNet,
    collate_uni_batch,
)
from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net  # noqa: E402


def _make_sample(h: int, w: int, u8: bool = True, sid: str = "s0") -> dict:
    if u8:
        img = torch.full((3, h, w), 42, dtype=torch.uint8)
    else:
        img = torch.full((3, h, w), 0.25, dtype=torch.float32)
    seq = torch.tensor([BOS_ID, 5, 6, EOS_ID], dtype=torch.long)
    return {"image": img, "seq": seq, "sample_id": sid, "kind": "fullpage", "domain": "printed"}


class TestIter13SpeedOpts(unittest.TestCase):
    # ------------------------------------------------------------------
    # 1. --max-mem-tokens default respects preset & explicit override clamps
    # ------------------------------------------------------------------
    def test_max_mem_tokens_default_respects_preset_and_override_clamps(self) -> None:
        parser = tv2.build_arg_parser()
        args_default = parser.parse_args([])
        self.assertIsNone(args_default.max_mem_tokens)

        # Without override, small_45m keeps its canonical 3072 cap (not 10240).
        cfg_small = tv2.build_v2_config("small_45m")
        self.assertEqual(cfg_small.max_mem_tokens, 3072)

        with tempfile.TemporaryDirectory() as tmp_dir:
            old_cfg = tv2.uni_omr_v2_tiny(max_mem_tokens=10240)
            old_model = UniOMRv2Net(old_cfg)
            ckpt_path = Path(tmp_dir) / "old_10240.pt"
            torch.save(
                {
                    "cfg": old_cfg.__dict__,
                    "config": old_cfg.__dict__,
                    "model": old_model.state_dict(),
                },
                str(ckpt_path),
            )

            args_override = parser.parse_args(
                [
                    "--preset", "tiny",
                    "--init-from", str(ckpt_path),
                    "--max-mem-tokens", "128",
                    "--device", "cpu",
                    "--no-bf16",
                    "--dry-run",
                    "--steps", "1",
                    "--batch-size", "1",
                    "--grad-accum", "1",
                    "--seq-len", "16",
                    "--prefetch-threads", "0",
                ]
            )
            self.assertEqual(args_override.max_mem_tokens, 128)

            captured_cfg: dict = {}
            orig_init = UniOMRv2Net.__init__

            def _spy_init(mod_self, cfg: UniOMRv2Config) -> None:
                captured_cfg["max_mem_tokens"] = cfg.max_mem_tokens
                orig_init(mod_self, cfg)

            with mock.patch.object(UniOMRv2Net, "__init__", _spy_init):
                res = tv2.run_train_v2(args_override)
            self.assertEqual(res["steps"], 1)
            self.assertEqual(captured_cfg["max_mem_tokens"], 128)

    # ------------------------------------------------------------------
    # 2. InkTokenRouter with compute_ink_recall=False + keep_ratio_override=0.45
    # ------------------------------------------------------------------
    def test_ink_router_no_sync_with_override_and_disabled_ink_recall(self) -> None:
        torch.manual_seed(0)
        router = InkTokenRouter(d_model=32, keep_ratio=0.40, dynamic_routing=True, n_bg_summary=4)
        router.eval()

        feat_2d = torch.randn(2, 32, 8, 12)
        images = torch.rand(2, 3, 64, 96)
        images[:, :, 16:24, 10:80] = 0.0

        with torch.no_grad():
            ref_tok, ref_coords, ref_info = router(
                feat_2d, images=images, keep_ratio_override=0.45, compute_ink_recall=True
            )

        orig_item = torch.Tensor.item

        def _forbid_item(tensor_self: torch.Tensor):
            raise AssertionError("Unexpected Tensor.item() sync in InkTokenRouter.forward")

        with torch.no_grad(), mock.patch.object(torch.Tensor, "item", _forbid_item):
            fast_tok, fast_coords, fast_info = router(
                feat_2d, images=images, keep_ratio_override=0.45, compute_ink_recall=False
            )

        self.assertEqual(fast_tok.shape, ref_tok.shape)
        self.assertEqual(fast_coords.shape, ref_coords.shape)
        self.assertTrue(torch.allclose(fast_tok, ref_tok, atol=1e-6))
        self.assertTrue(torch.equal(fast_coords, ref_coords))
        self.assertEqual(fast_info["ink_recall"], 0.0)
        self.assertGreater(ref_info["ink_recall"], 0.0)
        self.assertIs(torch.Tensor.item, orig_item)

    def test_v1_uniomr_forward_plumbs_compute_ink_recall(self) -> None:
        torch.manual_seed(1)
        t_cfg = UniOMRConfig(
            d_model=64,
            n_heads=4,
            n_enc_layers=1,
            n_dec_layers=1,
            ffn_mult=2.0,
            dropout=0.0,
            backbone_stride=16,
            use_uv_aux=False,
            mtp_k=1,
            n_bg_summary=4,
            dynamic_routing=True,
        )
        teacher = UniOMRNet(t_cfg).eval()
        images = torch.rand(1, 3, 64, 96)
        images[:, :, 20:28, 10:80] = 0.0
        tgt_in = torch.tensor([[BOS_ID, 5, 6, 7, EOS_ID]], dtype=torch.long)

        def _forbid_item(tensor_self: torch.Tensor):
            raise AssertionError("Unexpected Tensor.item() sync in frozen v1 teacher forward")

        with torch.inference_mode(), mock.patch.object(torch.Tensor, "item", _forbid_item):
            out = teacher(
                images,
                tgt_in,
                keep_ratio_override=0.45,
                return_aux=False,
                return_mtp=False,
                compute_ink_recall=False,
            )
        self.assertEqual(tuple(out["logits"].shape), (1, 5, VOCAB_SIZE))
        self.assertEqual(out["router_info"]["ink_recall"], 0.0)

    # ------------------------------------------------------------------
    # 3. Single-pass weighted F.cross_entropy matches legacy two-pass < 1e-5
    # ------------------------------------------------------------------
    def test_single_pass_weighted_ce_matches_legacy_two_pass(self) -> None:
        torch.manual_seed(7)
        cfg = tv2.uni_omr_v2_tiny()
        model = UniOMRv2Net(cfg)
        model.eval()

        B, T = 2, 18
        images = torch.rand(B, 3, 128, 192)
        seq = torch.randint(4, VOCAB_SIZE, (B, T + 1), dtype=torch.long)
        seq[:, 0] = BOS_ID
        seq[:, -1] = PAD_ID
        tgt_in = seq[:, :-1]
        tgt_out = seq[:, 1:]

        with torch.no_grad():
            out = model(images, tgt_in, return_aux=True, return_mtp=True)

        state_w = 2.5
        ls = 0.05
        ce_weights = tv2.build_state_token_weights(
            VOCAB_SIZE, state_w, device=out["logits"].device, dtype=out["logits"].dtype
        )

        # Legacy two-pass reference
        legacy_losses = model.compute_loss(
            out,
            tgt_out,
            images=images,
            label_smoothing=ls,
        )
        legacy_weighted_ce = F.cross_entropy(
            out["logits"].reshape(-1, VOCAB_SIZE),
            tgt_out.reshape(-1),
            weight=ce_weights,
            ignore_index=PAD_ID,
            label_smoothing=ls,
        )
        legacy_total = (
            legacy_weighted_ce
            + 0.30 * legacy_losses["mtp_loss"]
            + 0.10 * legacy_losses["saliency_loss"]
            + 0.25 * legacy_losses["sys_y_loss"]
            + 0.15 * legacy_losses["sys_count_loss"]
            + 0.20 * legacy_losses["staff_seg_loss"]
            + 0.15 * legacy_losses["stroke_hog_loss"]
        )

        # New single-pass weighted compute_loss
        ce_calls = {"main": 0}
        orig_ce = F.cross_entropy

        def _counting_ce(inp, tgt, *args, **kwargs):
            if inp.shape[0] == B * T:
                ce_calls["main"] += 1
            return orig_ce(inp, tgt, *args, **kwargs)

        with mock.patch.object(F, "cross_entropy", side_effect=_counting_ce):
            new_losses = model.compute_loss(
                out,
                tgt_out,
                images=images,
                label_smoothing=ls,
                ce_token_weights=ce_weights,
            )

        self.assertEqual(ce_calls["main"], 1)
        self.assertLess(abs(float(new_losses["ce_loss"] - legacy_weighted_ce)), 1e-5)
        self.assertLess(abs(float(new_losses["loss"] - legacy_total)), 1e-5)

    # ------------------------------------------------------------------
    # 4. Fixed 6-bucket (H, W) padding in collate_uni_batch & prefetcher
    # ------------------------------------------------------------------
    def test_collate_fixed_hw_buckets_snaps_and_never_crops(self) -> None:
        cases = [
            # Portrait buckets
            ((900, 700), (1152, 832)),
            ((1152, 832), (1152, 832)),
            ((1300, 950), (1408, 1024)),
            ((1400, 1100), (1408, 1152)),
            # Landscape buckets
            ((700, 900), (832, 1152)),
            ((950, 1300), (1024, 1408)),
            ((1100, 1400), (1152, 1408)),
            # Oversized fallback -> mult-64, NEVER cropped
            ((1500, 1000), (1536, 1024)),
            ((1300, 1300), (1344, 1344)),
        ]
        for (h, w), expected_hw in cases:
            with self.subTest(in_hw=(h, w), expected_hw=expected_hw):
                s = _make_sample(h, w, u8=True)
                batch = collate_uni_batch([s], fixed_hw_buckets=True)
                imgs = batch["images"]
                self.assertEqual(tuple(imgs.shape), (1, 3, expected_hw[0], expected_hw[1]))
                self.assertEqual(int(imgs[0, :, :h, :w].min()), 42)
                self.assertEqual(int(imgs[0, :, :h, :w].max()), 42)
                if expected_hw[0] > h:
                    self.assertEqual(int(imgs[0, :, h:, :].min()), 255)
                if expected_hw[1] > w:
                    self.assertEqual(int(imgs[0, :, :, w:].min()), 255)

                sf = _make_sample(h, w, u8=False)
                batch_f = collate_uni_batch([sf], fixed_hw_buckets=True)
                imgsf = batch_f["images"]
                self.assertEqual(tuple(imgsf.shape), (1, 3, expected_hw[0], expected_hw[1]))
                if expected_hw[0] > h:
                    self.assertAlmostEqual(float(imgsf[0, :, h:, :].min()), 1.0, places=5)

    def test_async_prefetcher_plumbs_fixed_hw_buckets(self) -> None:
        class _TinyDS(Dataset):
            def __len__(self) -> int:
                return 4

            def __getitem__(self, idx: int) -> dict:
                return _make_sample(900, 700, u8=True, sid=f"fp_{idx}")

        pf = AsyncThreadPrefetcher(
            _TinyDS(),
            batch_size=2,
            shuffle=False,
            max_workers=1,
            prefetch_batches=2,
            fixed_hw_buckets=True,
        )
        try:
            b = next(pf)
            self.assertEqual(tuple(b["images"].shape), (2, 3, 1152, 832))
            self.assertIn((b["images"].shape[2], b["images"].shape[3]), FIXED_HW_BUCKETS)
        finally:
            pf.close()

    # ------------------------------------------------------------------
    # 5. --kd-prob, --cass-batch-prob, and --cpu-threads parsing & gating
    # ------------------------------------------------------------------
    def test_kd_prob_and_cass_batch_prob_parse_and_gate(self) -> None:
        parser = tv2.build_arg_parser()
        args_def = parser.parse_args([])
        self.assertAlmostEqual(args_def.kd_prob, 1.0)
        self.assertAlmostEqual(args_def.cass_batch_prob, 1.0)
        self.assertFalse(args_def.fixed_hw_buckets)
        self.assertIsNone(args_def.cpu_threads)

        args_custom = parser.parse_args(
            [
                "--kd-prob", "0.5",
                "--cass-batch-prob", "0.5",
                "--fixed-hw-buckets",
                "--cpu-threads", "2",
            ]
        )
        self.assertAlmostEqual(args_custom.kd_prob, 0.5)
        self.assertAlmostEqual(args_custom.cass_batch_prob, 0.5)
        self.assertTrue(args_custom.fixed_hw_buckets)
        self.assertEqual(args_custom.cpu_threads, 2)

        torch.manual_seed(3)
        student = UniOMRv2Net(tv2.uni_omr_v2_tiny())
        teacher = UniOMRNet(
            UniOMRConfig(
                d_model=64,
                n_heads=4,
                n_enc_layers=1,
                n_dec_layers=1,
                ffn_mult=2.0,
                dropout=0.0,
                backbone_stride=16,
                use_uv_aux=False,
                mtp_k=1,
                n_bg_summary=4,
            )
        ).eval()
        images = torch.rand(1, 3, 128, 192)
        seq = torch.tensor([[BOS_ID, 5, 6, 7, 8, EOS_ID]], dtype=torch.long)

        # kd_prob=0.0 skips teacher forward completely and yields kd_loss == 0.
        teacher_calls = {"n": 0}
        orig_t_fwd = teacher.forward

        def _spy_teacher(*a, **kw):
            teacher_calls["n"] += 1
            return orig_t_fwd(*a, **kw)

        with mock.patch.object(teacher, "forward", side_effect=_spy_teacher):
            losses_0, _ = tv2.forward_and_loss_v2(
                student,
                images,
                seq,
                teacher_v1=teacher,
                kd_alpha=0.3,
                kd_prob=0.0,
                cass_prob=0.0,
                kind="fullpage",
            )
            self.assertEqual(teacher_calls["n"], 0)
            self.assertEqual(float(losses_0["kd_loss"]), 0.0)

            with mock.patch.object(random, "random", return_value=0.2):
                losses_hit, _ = tv2.forward_and_loss_v2(
                    student,
                    images,
                    seq,
                    teacher_v1=teacher,
                    kd_alpha=0.3,
                    kd_prob=0.5,
                    cass_prob=0.0,
                    kind="fullpage",
                )
                self.assertEqual(teacher_calls["n"], 1)
                self.assertGreaterEqual(float(losses_hit["kd_loss"]), 0.0)

            with mock.patch.object(random, "random", return_value=0.8):
                losses_miss, _ = tv2.forward_and_loss_v2(
                    student,
                    images,
                    seq,
                    teacher_v1=teacher,
                    kd_alpha=0.3,
                    kd_prob=0.5,
                    cass_prob=0.0,
                    kind="fullpage",
                )
                self.assertEqual(teacher_calls["n"], 1)
                self.assertEqual(float(losses_miss["kd_loss"]), 0.0)

        # cass_batch_prob gating
        dec_calls = {"n": 0}
        orig_dec = student.decode_tokens

        def _spy_dec(*a, **kw):
            dec_calls["n"] += 1
            return orig_dec(*a, **kw)

        with mock.patch.object(student, "decode_tokens", side_effect=_spy_dec):
            tv2.forward_and_loss_v2(
                student, images, seq, cass_prob=0.30, cass_batch_prob=0.0, kind="fullpage"
            )
            self.assertEqual(dec_calls["n"], 1)

            with mock.patch.object(random, "random", return_value=0.1):
                tv2.forward_and_loss_v2(
                    student, images, seq, cass_prob=0.30, cass_batch_prob=0.5, kind="fullpage"
                )
            self.assertEqual(dec_calls["n"], 3)

            with mock.patch.object(random, "random", return_value=0.9):
                tv2.forward_and_loss_v2(
                    student, images, seq, cass_prob=0.30, cass_batch_prob=0.5, kind="fullpage"
                )
            self.assertEqual(dec_calls["n"], 4)


if __name__ == "__main__":
    unittest.main()
