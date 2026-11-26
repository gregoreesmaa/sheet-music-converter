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
    # 4. Fixed 10-bucket (H, W) padding in collate_uni_batch & prefetcher
    # ------------------------------------------------------------------
    def test_collate_fixed_hw_buckets_snaps_and_never_crops(self) -> None:
        self.assertEqual(len(FIXED_HW_BUCKETS), 10)
        cases = [
            # Portrait buckets (including new 1024x768 and 1280x960)
            ((900, 700), (1024, 768)),
            ((1024, 768), (1024, 768)),
            ((1152, 832), (1152, 832)),
            ((1280, 960), (1280, 960)),
            ((1300, 950), (1408, 1024)),
            ((1400, 1100), (1408, 1152)),
            # Landscape buckets (including new 768x1024 and 960x1280)
            ((700, 900), (768, 1024)),
            ((768, 1024), (768, 1024)),
            ((960, 1280), (960, 1280)),
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
            self.assertEqual(tuple(b["images"].shape), (2, 3, 1024, 768))
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

    # ------------------------------------------------------------------
    # 6. Tier 3: InkTokenRouter clamps k <= max_kept_tokens with override
    # ------------------------------------------------------------------
    def test_tier3_ink_router_clamps_max_kept_tokens_with_override(self) -> None:
        torch.manual_seed(11)
        router = InkTokenRouter(
            d_model=32,
            keep_ratio=0.40,
            dynamic_routing=True,
            max_kept_tokens=20,
            n_bg_summary=4,
        ).eval()
        # N = 10 * 10 = 100; 0.45 * 100 = 45 > max_kept_tokens(20)
        feat_2d = torch.randn(2, 32, 10, 10)
        with torch.no_grad():
            toks, coords, info = router(feat_2d, keep_ratio_override=0.45, compute_ink_recall=False)
        self.assertEqual(info["k_kept"], 20)
        self.assertEqual(toks.shape, (2, 24, 32))
        self.assertEqual(coords.shape, (2, 24, 2))

    # ------------------------------------------------------------------
    # 7. Tier 3: RMSNorm bf16 + half-dim apply_rope + fused QKV + fused MTP-4
    # ------------------------------------------------------------------
    def test_tier3_rmsnorm_rope_fused_qkv_and_mtp_parity(self) -> None:
        from uni_omr import RMSNorm as V1RMSNorm, RoPE1D, _rotate_half, apply_rope
        from uni_omr_v2 import (
            GroupedQueryAttention,
            QATInt4ControllerV2,
            RMSNorm as V2RMSNorm,
            _is_qat_linear_active,
        )

        torch.manual_seed(17)
        # 7a. RMSNorm fp32 & bf16 parity + stays bf16 under cpu autocast
        for cls in (V1RMSNorm, V2RMSNorm):
            rn = cls(32)
            with torch.no_grad():
                rn.weight.uniform_(0.5, 1.5)
            x32 = torch.randn(2, 8, 32, dtype=torch.float32)
            ref32 = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + rn.eps) * rn.weight
            out32 = rn(x32)
            self.assertEqual(out32.dtype, torch.float32)
            self.assertTrue(torch.allclose(out32, ref32, atol=1e-5))

            x16 = x32.to(torch.bfloat16)
            with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
                out16 = rn(x16)
            self.assertEqual(out16.dtype, torch.bfloat16)
            self.assertTrue(torch.allclose(out16.float(), ref32, atol=2e-2))

        # 7b. Half-dim apply_rope matches legacy _rotate_half algebraically
        rope = RoPE1D(head_dim=16)
        cos, sin = rope(12, device=torch.device("cpu"), dtype=torch.float32)
        q = torch.randn(2, 4, 12, 16)
        k = torch.randn(2, 2, 12, 16)
        q_ref = (q * cos) + (_rotate_half(q) * sin)
        k_ref = (k * cos) + (_rotate_half(k) * sin)
        q_new, k_new = apply_rope(q, k, cos, sin)
        self.assertTrue(torch.allclose(q_new, q_ref, atol=1e-6))
        self.assertTrue(torch.allclose(k_new, k_ref, atol=1e-6))

        # 7c. Fused QKV (self-attn & cross-attn) matches unfused reference (< 1e-4)
        gqa = GroupedQueryAttention(d_model=64, n_heads=4, n_kv_heads=2, use_rope=True).eval()
        h_in = torch.randn(2, 12, 64)
        mem_in = torch.randn(2, 16, 64)
        with torch.no_grad():
            sa_fused = gqa(h_in, h_in, rope_cos=cos, rope_sin=sin, is_causal=True)
            ca_fused = gqa(h_in, mem_in, is_cross=True)
            # Force unfused path via _is_qat_linear_active=True mock
            with mock.patch("uni_omr_v2._is_qat_linear_active", return_value=True):
                sa_ref = gqa(h_in, h_in, rope_cos=cos, rope_sin=sin, is_causal=True)
                ca_ref = gqa(h_in, mem_in, is_cross=True)
        self.assertTrue(torch.allclose(sa_fused, sa_ref, atol=1e-5))
        self.assertTrue(torch.allclose(ca_fused, ca_ref, atol=1e-5))

        # 7d. Fused MTP-4 matches unfused MTPHead loop (< 1e-4) & respects QAT toggle
        cfg = tv2.uni_omr_v2_tiny(mtp_k=4)
        model = UniOMRv2Net(cfg).eval()
        imgs = torch.rand(2, 3, 128, 192)
        tgt_in = torch.randint(4, VOCAB_SIZE, (2, 16), dtype=torch.long)
        tgt_in[:, 0] = BOS_ID
        with torch.no_grad():
            mem, _, _ = model.encode_image(imgs, return_aux=False)
            logits_fused, mtp_fused, _ = model.decode_tokens(
                tgt_in, mem, return_mtp=True, return_aux=False, apply_tok_dropout=False
            )
            with mock.patch("uni_omr_v2._is_qat_linear_active", return_value=True):
                logits_ref, mtp_ref, _ = model.decode_tokens(
                    tgt_in, mem, return_mtp=True, return_aux=False, apply_tok_dropout=False
                )
        self.assertTrue(torch.allclose(logits_fused, logits_ref, atol=1e-4))
        self.assertEqual(len(mtp_fused), len(mtp_ref))
        for mf, mr in zip(mtp_fused, mtp_ref):
            self.assertTrue(torch.allclose(mf, mr, atol=1e-4))

        # Verify QATInt4ControllerV2 toggles _qat_enabled properly
        qat = QATInt4ControllerV2(model, block_size=32, enabled=False)
        self.assertFalse(_is_qat_linear_active(model.dec_layers[0].self_attn.q_proj))
        qat.enable()
        self.assertTrue(_is_qat_linear_active(model.dec_layers[0].self_attn.q_proj))
        self.assertTrue(_is_qat_linear_active(model.mtp_heads[0].proj))
        qat.disable()
        self.assertFalse(_is_qat_linear_active(model.dec_layers[0].self_attn.q_proj))
        qat.detach()

    # ------------------------------------------------------------------
    # 8. Tier 3: compute_stroke_hog_target scatter_ bit-identical to scatter_add_
    # ------------------------------------------------------------------
    def test_tier3_stroke_hog_scatter_bit_identical(self) -> None:
        import math
        from uni_omr_v2 import _get_sobel_kxy, compute_stroke_hog_target

        torch.manual_seed(23)
        img = torch.rand(2, 3, 64, 96)
        n_bins = 8
        target_hw = (16, 24)  # exact 4x factor -> avg_pool2d fast path

        out_new = compute_stroke_hog_target(img, target_hw=target_hw, n_bins=n_bins)

        # Reference using scatter_add_ + adaptive_avg_pool2d (exact same arithmetic ops)
        with torch.no_grad():
            luma_f = img[:, 0:1].to(torch.float32) * 0.299
            luma_f.add_(img[:, 1:2], alpha=0.587).add_(img[:, 2:3], alpha=0.114)
            gxy = F.conv2d(luma_f, _get_sobel_kxy(img.device), padding=1)
            gx, gy = gxy[:, 0:1], gxy[:, 1:2]
            mag = gx.mul(gx).addcmul_(gy, gy).add_(1e-8).sqrt_()
            bin_f = torch.atan2(gy, gx).remainder_(math.pi).mul_(float(n_bins) / math.pi)
            bin_floor = torch.floor(bin_f)
            bin_lo = bin_floor.to(torch.long) % n_bins
            bin_hi = (bin_lo + 1) % n_bins
            w_hi = bin_f.sub_(bin_floor)
            w_lo = 1.0 - w_hi
            hog_full = luma_f.new_zeros((2, n_bins, 64, 96), dtype=torch.float32)
            hog_full.scatter_add_(1, bin_lo, mag * w_lo)
            hog_full.scatter_add_(1, bin_hi, mag.mul_(w_hi))
            hog_pooled = F.adaptive_avg_pool2d(hog_full, target_hw)
            norm = hog_pooled.pow(2).sum(dim=1, keepdim=True).add_(1e-4).sqrt_()
            ref = hog_pooled.div_(norm)

        self.assertTrue(torch.equal(out_new, ref))

    # ------------------------------------------------------------------
    # 9. Tier 3: BatchedMuon parity with torch.optim.Muon
    # ------------------------------------------------------------------
    def test_tier3_batched_muon_matches_torch_muon(self) -> None:
        if not hasattr(torch.optim, "Muon"):
            self.skipTest("torch.optim.Muon not available in this PyTorch build")

        shapes = [(32, 32), (32, 32), (64, 32), (64, 32), (32, 64), (32, 64)]
        for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 1e-3)):
            with self.subTest(dtype=dtype):
                torch.manual_seed(31)
                params_ref = [
                    torch.nn.Parameter(torch.randn(s, dtype=dtype)) for s in shapes
                ]
                params_bat = [
                    torch.nn.Parameter(p.detach().clone()) for p in params_ref
                ]
                grads = [torch.randn(s, dtype=dtype) for s in shapes]
                for pr, pb, g in zip(params_ref, params_bat, grads):
                    pr.grad = g.clone()
                    pb.grad = g.clone()

                opt_ref = torch.optim.Muon(
                    params_ref,
                    lr=1e-3,
                    weight_decay=0.01,
                    momentum=0.95,
                    nesterov=True,
                    ns_steps=5,
                    adjust_lr_fn="match_rms_adamw",
                )
                opt_bat = tv2.BatchedMuon(
                    params_bat,
                    lr=1e-3,
                    weight_decay=0.01,
                    momentum=0.95,
                    nesterov=True,
                    ns_steps=5,
                    adjust_lr_fn="match_rms_adamw",
                )
                opt_ref.step()
                opt_bat.step()

                for pr, pb in zip(params_ref, params_bat):
                    diff = float((pr.detach().float() - pb.detach().float()).abs().max())
                    self.assertLess(diff, tol, f"shape={tuple(pr.shape)} dtype={dtype} max_diff={diff}")

    # ------------------------------------------------------------------
    # 10. Tier 3: Active-item teacher slicing, teacher-max-dim, kd-stop-frac
    # ------------------------------------------------------------------
    def test_tier3_teacher_active_slicing_max_dim_and_stop_frac(self) -> None:
        parser = tv2.build_arg_parser()
        args_def = parser.parse_args([])
        self.assertEqual(args_def.teacher_max_dim, 0)
        self.assertAlmostEqual(args_def.kd_stop_frac, 1.0)

        args_custom = parser.parse_args(["--teacher-max-dim", "1024", "--kd-stop-frac", "0.65"])
        self.assertEqual(args_custom.teacher_max_dim, 1024)
        self.assertAlmostEqual(args_custom.kd_stop_frac, 0.65)

        torch.manual_seed(41)
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

        # Batch of 2: item 0 printed (active), item 1 real_hw (weight 0 -> sliced out)
        images = torch.rand(2, 3, 192, 256)
        seq = torch.tensor(
            [
                [BOS_ID, 5, 6, 7, 8, EOS_ID],
                [BOS_ID, 9, 10, 11, 12, EOS_ID],
            ],
            dtype=torch.long,
        )
        seen_shapes: list = []
        orig_t_fwd = teacher.forward

        def _spy_t(t_imgs, t_tgt, *a, **kw):
            seen_shapes.append(tuple(t_imgs.shape))
            return orig_t_fwd(t_imgs, t_tgt, *a, **kw)

        with mock.patch.object(teacher, "forward", side_effect=_spy_t):
            losses, _ = tv2.forward_and_loss_v2(
                student,
                images,
                seq,
                teacher_v1=teacher,
                domains=["printed", "real_hw"],
                kd_alpha=0.3,
                kd_hw_scale=0.0,
                teacher_max_dim=128,
                cass_prob=0.0,
                kind="fullpage",
            )
        # Only 1 active item passed to teacher, and downscaled from (192, 256) -> (96, 128)
        self.assertEqual(seen_shapes, [(1, 3, 96, 128)])
        self.assertGreaterEqual(float(losses["kd_loss"]), 0.0)
        losses["loss"].backward()


if __name__ == "__main__":
    unittest.main()
