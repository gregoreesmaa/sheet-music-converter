"""CPU-only unit tests for Iter 15 promoted features:
1. UniOMRv2Config.edge85m() + ax1_gl5 (--arch-preset edge85m --n-axial-layers 1 --n-global-layers 5):
   - exact param count & exact packed Hybrid FP16/INT4 size < 55.0 MB (g32 and g64)
   - forward/backward finite loss & gradients on CPU
2. Late-stage INT4 QAT (--qat-int4-start-frac, default 0.0 = off, e.g. 0.85) using QATInt4ControllerV2:
   - activates at the exact start step, finite loss/grads, clean detach
3. 265-vocab UniOMRv2Net teacher KD into 265-vocab edge85m student (plus legacy v1 teacher compat)
4. --vrfm-band-alpha CLI override preserving default 0.30
"""

from __future__ import annotations

import os

# Force strict CPU-only execution (never touch GPU while Iter 14 training runs).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn

_MVD_DIR = Path(__file__).resolve().parent
if str(_MVD_DIR) not in sys.path:
    sys.path.insert(0, str(_MVD_DIR))

import train_uni_v2 as tv2  # noqa: E402
from uni_omr import BOS_ID, EOS_ID, PAD_ID, SYS_ID, VOCAB_SIZE, UniOMRConfig, UniOMRNet  # noqa: E402
from uni_omr_v2 import QATInt4ControllerV2, UniOMRv2Config, UniOMRv2Net  # noqa: E402


def _exact_packed_hybrid_int4_bytes(model: UniOMRv2Net, group_size: int = 32) -> int:
    """Compute exact on-disk packed Hybrid FP16/INT4 byte size (4-bit nibbles + FP16 scales + FP16 keep)."""
    int4_roots = (
        model.encoder.axial_layers,
        model.encoder.patch_merger,
        model.encoder.global_layers,
        model.deepstack_fusions,
        model.dec_layers,
        model.lm_head,
    )
    int4_param_ids: set[int] = set()
    int4_bytes = 0
    for root in int4_roots:
        for mod in root.modules():
            if isinstance(mod, nn.Linear):
                w = mod.weight
                int4_param_ids.add(id(w))
                n = w.numel()
                n_packed = (n + 1) // 2  # 2 INT4 nibbles per uint8 byte
                n_groups = (n + group_size - 1) // group_size
                int4_bytes += n_packed + n_groups * 2  # FP16 scale per group

    aux_and_mtp_ids: set[int] = set()
    for mod in (
        model.sys_y_head,
        model.sys_count_head,
        model.staff_seg_head,
        model.stroke_hog_head,
        model.mtp_heads,
    ):
        if mod is not None:
            for p in mod.parameters():
                aux_and_mtp_ids.add(id(p))

    fp16_keep_bytes = 0
    for p in model.parameters():
        pid = id(p)
        if pid in aux_and_mtp_ids or pid in int4_param_ids:
            continue
        fp16_keep_bytes += p.numel() * 2

    return int4_bytes + fp16_keep_bytes


class TestIter15Promotions(unittest.TestCase):
    # ------------------------------------------------------------------
    # 1. edge85m + ax1_gl5 CLI, exact packed INT4 < 55 MB, fwd/bwd
    # ------------------------------------------------------------------
    def test_edge85m_ax1_gl5_footprint_and_cli(self) -> None:
        parser = tv2.build_arg_parser()
        # Default must stay small_45m (never break Iter 14)
        args_def = parser.parse_args([])
        self.assertEqual(args_def.preset, "small_45m")
        self.assertAlmostEqual(args_def.qat_int4_start_frac, 0.0)
        self.assertIsNone(args_def.vrfm_band_alpha)

        # --arch-preset alias works identically to --preset
        args_edge = parser.parse_args(
            [
                "--arch-preset",
                "edge85m",
                "--n-axial-layers",
                "1",
                "--n-global-layers",
                "5",
                "--qat-int4-start-frac",
                "0.85",
                "--vrfm-band-alpha",
                "0.30",
            ]
        )
        self.assertEqual(args_edge.preset, "edge85m")
        self.assertEqual(args_edge.n_axial_layers, 1)
        self.assertEqual(args_edge.n_global_layers, 5)
        self.assertAlmostEqual(args_edge.qat_int4_start_frac, 0.85)
        self.assertAlmostEqual(args_edge.vrfm_band_alpha, 0.30)

        # Default edge85m (ax2_gl4) matches i14-05 < 55.0 MB budget
        cfg_edge_def = UniOMRv2Config.edge85m()
        self.assertEqual(cfg_edge_def.d_model, 512)
        self.assertEqual(cfg_edge_def.n_dec_layers, 10)
        self.assertEqual(cfg_edge_def.vocab_size, 265)
        m_edge_def = UniOMRv2Net(cfg_edge_def)
        s_def = m_edge_def.param_summary(int4_block_size=32)
        self.assertLess(s_def["hybrid_fp16_int4_with_scales_mb"], 55.0)
        exact_def_mb = _exact_packed_hybrid_int4_bytes(m_edge_def, group_size=32) / 1_000_000.0
        self.assertAlmostEqual(exact_def_mb, s_def["hybrid_fp16_int4_with_scales_mb"], places=3)
        self.assertLess(exact_def_mb, 55.0)

        # edge85m + ax1_gl5 composes cleanly and stays strictly < 55.0 MB
        cfg_ax1 = tv2.build_v2_config("edge85m", n_axial_layers=1, n_global_layers=5)
        self.assertEqual(cfg_ax1.n_axial_layers, 1)
        self.assertEqual(cfg_ax1.n_global_layers, 5)
        m_ax1 = UniOMRv2Net(cfg_ax1)
        self.assertEqual(len(m_ax1.encoder.axial_layers), 1)
        self.assertEqual(len(m_ax1.encoder.global_layers), 5)
        s_ax1 = m_ax1.param_summary(int4_block_size=32)
        exact_ax1_mb = _exact_packed_hybrid_int4_bytes(m_ax1, group_size=32) / 1_000_000.0
        self.assertAlmostEqual(exact_ax1_mb, s_ax1["hybrid_fp16_int4_with_scales_mb"], places=3)
        self.assertLess(exact_ax1_mb, 55.0)
        self.assertLess(m_ax1.param_summary(int4_block_size=64)["hybrid_fp16_int4_with_scales_mb"], 55.0)

    def test_edge85m_ax1_gl5_forward_backward_cpu(self) -> None:
        torch.manual_seed(42)
        cfg = tv2.build_v2_config("edge85m", n_axial_layers=1, n_global_layers=5, dropout=0.0)
        model = UniOMRv2Net(cfg).train()
        images = torch.rand(1, 3, 128, 128)
        seq = torch.tensor([[BOS_ID, 10, 233, 250, SYS_ID, 15, EOS_ID, PAD_ID]], dtype=torch.long)

        losses, out = tv2.forward_and_loss_v2(
            model,
            images,
            seq,
            kd_alpha=0.0,
            cass_prob=0.0,
            use_bf16=False,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(out["logits"].shape), (1, seq.shape[1] - 1, VOCAB_SIZE))
        self.assertTrue(bool(torch.isfinite(losses["loss"])))
        model.zero_grad(set_to_none=True)
        losses["loss"].backward()
        grad_count = 0
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_count += 1
                self.assertTrue(bool(torch.isfinite(p.grad).all()), f"Non-finite grad in {name}")
        self.assertGreater(grad_count, 50)

    # ------------------------------------------------------------------
    # 2. Late-stage QATInt4ControllerV2 activation via --qat-int4-start-frac
    # ------------------------------------------------------------------
    def test_qat_int4_start_frac_late_stage_activation(self) -> None:
        parser = tv2.build_arg_parser()
        # 4 synthetic steps with start_frac=0.75 -> qat_start_step = int(4 * 0.75) = 3
        args = parser.parse_args(
            [
                "--arch-preset",
                "tiny",
                "--n-axial-layers",
                "1",
                "--n-global-layers",
                "2",
                "--qat-int4-start-frac",
                "0.75",
                "--qat-int4-block-size",
                "32",
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
                "16",
                "--prefetch-threads",
                "0",
                "--log-every",
                "1",
                "--eval-every",
                "0",
                "--gpu-ema-decay",
                "0.0",
                "--save-every-minutes",
                "0.0",
            ]
        )

        enabled_at_steps: list[int] = []
        orig_enable = QATInt4ControllerV2.enable
        orig_refresh = QATInt4ControllerV2.refresh_cache
        refresh_count = {"n": 0}
        step_counter = {"cur": 0}
        orig_fwd_loss = tv2.forward_and_loss_v2

        def _spy_fwd_loss(*a, **kw):
            step_counter["cur"] += 1
            return orig_fwd_loss(*a, **kw)

        def _spy_enable(ctrl_self) -> None:
            enabled_at_steps.append(step_counter["cur"] + 1)
            orig_enable(ctrl_self)

        def _spy_refresh(ctrl_self) -> None:
            refresh_count["n"] += 1
            orig_refresh(ctrl_self)

        with (
            mock.patch.object(tv2, "forward_and_loss_v2", side_effect=_spy_fwd_loss),
            mock.patch.object(QATInt4ControllerV2, "enable", _spy_enable),
            mock.patch.object(QATInt4ControllerV2, "refresh_cache", _spy_refresh),
        ):
            res = tv2.run_train_v2(args)

        self.assertEqual(res["steps"], 4)
        self.assertEqual(enabled_at_steps, [3], f"Expected QAT to enable at step 3/4, got {enabled_at_steps}")
        self.assertEqual(
            refresh_count["n"],
            3,
            "Expected refresh_cache once inside enable() + once after opt.step() on steps 3 and 4",
        )
        for row in res["history"]:
            self.assertTrue(torch.isfinite(torch.tensor(row["loss"])), f"Non-finite loss in {row}")

    def test_qat_int4_controller_on_edge85m_finite_grad(self) -> None:
        torch.manual_seed(13)
        cfg = tv2.build_v2_config("edge85m", n_axial_layers=1, n_global_layers=5, dropout=0.0)
        model = UniOMRv2Net(cfg).train()
        ctrl = model.attach_qat_int4(block_size=32, enabled=True)
        try:
            self.assertGreater(ctrl.num_layers, 100)
            images = torch.rand(1, 3, 128, 128)
            seq = torch.tensor([[BOS_ID, 12, 240, 260, EOS_ID, PAD_ID]], dtype=torch.long)
            losses, _ = tv2.forward_and_loss_v2(
                model, images, seq, kd_alpha=0.0, cass_prob=0.0, use_bf16=False, device=torch.device("cpu")
            )
            self.assertTrue(bool(torch.isfinite(losses["loss"])))
            model.zero_grad(set_to_none=True)
            losses["loss"].backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    self.assertTrue(bool(torch.isfinite(p.grad).all()), f"Non-finite QAT grad in {name}")
        finally:
            ctrl.detach()

    # ------------------------------------------------------------------
    # 3. 265-vocab UniOMRv2Net teacher KD into 265-vocab edge85m student
    #    (+ legacy v1 teacher compatibility)
    # ------------------------------------------------------------------
    def test_kd_265_vocab_v2_teacher_into_265_vocab_edge85m_student(self) -> None:
        torch.manual_seed(17)
        # Create a 265-vocab v2 teacher checkpoint on disk (simulating Iter 14 uni_omr_v2_ema_latest.pt)
        t_cfg = tv2.build_v2_config("tiny", vocab_size=VOCAB_SIZE, dropout=0.0)
        t_model_raw = UniOMRv2Net(t_cfg).eval()
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "iter14_v2_ema_latest.pt"
            torch.save(
                {
                    "step": 16000,
                    "cfg": asdict(t_cfg),
                    "config": asdict(t_cfg),
                    "model": t_model_raw.state_dict(),
                },
                ckpt_path,
            )

            # Both load_teacher and backward-compat load_v1_teacher must auto-detect UniOMRv2Net
            teacher = tv2.load_v1_teacher(ckpt_path, device=torch.device("cpu"), use_bf16=False)
            self.assertIsInstance(teacher, UniOMRv2Net)
            self.assertFalse(teacher.training)
            self.assertEqual(int(teacher.cfg.vocab_size), VOCAB_SIZE)
            self.assertIsNone(teacher.sys_y_head)
            self.assertEqual(len(teacher.mtp_heads), 0)
            for p in teacher.parameters():
                self.assertFalse(p.requires_grad)

        # Student: 265-vocab edge85m + ax1_gl5
        s_cfg = tv2.build_v2_config("edge85m", n_axial_layers=1, n_global_layers=5, vocab_size=VOCAB_SIZE, dropout=0.0)
        student = UniOMRv2Net(s_cfg).train()

        images = torch.rand(1, 3, 128, 128)
        # Include regular (0..232), bekern+v2 Phase A (233..260), and specials (261..264)
        seq = torch.tensor([[BOS_ID, 10, 233, 248, 260, SYS_ID, 20, EOS_ID, PAD_ID]], dtype=torch.long)

        losses, _ = tv2.forward_and_loss_v2(
            student,
            images,
            seq,
            teacher_v1=teacher,
            kd_alpha=0.30,
            kd_beta=0.5,
            kd_temp=1.5,
            kd_teacher_ce_cap=0.0,
            expr_token_loss_weight=2.5,
            use_bf16=False,
            device=torch.device("cpu"),
        )
        self.assertTrue(bool(torch.isfinite(losses["loss"])))
        self.assertTrue(bool(torch.isfinite(losses["kd_loss"])))
        self.assertGreater(float(losses["kd_loss"].item()), 0.0)

        student.zero_grad(set_to_none=True)
        losses["loss"].backward()
        self.assertIsNotNone(student.lm_head.weight.grad)
        self.assertTrue(bool(torch.isfinite(student.lm_head.weight.grad).all()))
        for p in teacher.parameters():
            self.assertIsNone(p.grad)

    def test_load_teacher_still_supports_legacy_v1_checkpoint(self) -> None:
        torch.manual_seed(29)
        t_cfg = UniOMRConfig(
            vocab_size=237,
            d_model=64,
            n_heads=4,
            n_enc_layers=1,
            n_dec_layers=1,
            ffn_mult=2.0,
            mtp_k=4,
        )
        t_v1 = UniOMRNet(t_cfg).eval()
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "iter10_5_v1_ema_best.pt"
            torch.save({"config": asdict(t_cfg), "model": t_v1.state_dict()}, ckpt_path)
            loaded = tv2.load_teacher(ckpt_path, device=torch.device("cpu"), use_bf16=False)
            self.assertIsInstance(loaded, UniOMRNet)
            self.assertEqual(int(loaded.cfg.vocab_size), 237)
            self.assertFalse(loaded.training)


if __name__ == "__main__":
    unittest.main(verbosity=2)
