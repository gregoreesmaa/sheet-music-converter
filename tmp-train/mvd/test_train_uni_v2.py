"""CPU-only unit tests for ``train_uni_v2.py`` (Iter-11 UniOMR v2 training & Hybrid KD)."""

from __future__ import annotations

import os

# Force strict CPU-only execution (never touch GPU while training runs).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from uni_omr import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    GPUEmaShadow,
    PAD_ID,
    SYS_ID,
    VOCAB_SIZE,
    UniOMRConfig,
    UniOMRNet,
)
from uni_omr_v2 import QATInt4ControllerV2, UniOMRv2Net  # noqa: E402
from train_uni_v2 import (  # noqa: E402
    TopKTeacherCache,
    atomic_save_v2_latest,
    build_arg_parser,
    build_optimizer_v2,
    build_v2_config,
    compute_sparse_topk_jsd_loss,
    ensure_v2_best_checkpoints,
    extract_topk_from_logits,
    forward_and_loss_v2,
    kd_alpha_for_step,
    load_v2_checkpoint,
    partition_muon_adamw_params,
    run_train_v2,
)


def _make_tiny_v2(grad_checkpoint: bool = True) -> UniOMRv2Net:
    cfg = build_v2_config("tiny", grad_checkpoint=grad_checkpoint, dropout=0.0)
    return UniOMRv2Net(cfg)


def _make_tiny_v1_teacher() -> UniOMRNet:
    cfg = UniOMRConfig(
        d_model=64,
        n_enc_layers=1,
        n_dec_layers=1,
        n_heads=4,
        ffn_mult=2.0,
        dropout=0.0,
        backbone_stride=16,
        use_uv_aux=False,
        mtp_k=1,
    )
    teacher = UniOMRNet(cfg)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _make_dummy_batch(batch_size: int = 2, h: int = 64, w: int = 96, seq_len: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(123)
    images = torch.rand((batch_size, 3, h, w), generator=g, dtype=torch.float32)
    seq = torch.randint(0, PAD_ID, (batch_size, seq_len), generator=g, dtype=torch.long)
    seq[:, 0] = BOS_ID
    seq[:, seq_len // 2] = SYS_ID
    seq[:, -2] = EOS_ID
    seq[:, -1] = PAD_ID
    return images, seq


class TestTrainUniV2(unittest.TestCase):
    def test_1_supervised_hog_online_v1_kd_2steps(self) -> None:
        """2-step supervised + Sobel-HOG + online v1->v2 Generalized JSD KD."""
        torch.manual_seed(7)
        student = _make_tiny_v2(grad_checkpoint=True)
        student.train()
        teacher = _make_tiny_v1_teacher()

        teacher_snap = {k: v.detach().clone() for k, v in teacher.state_dict().items()}
        student_snap_before = student.lm_head.weight.detach().clone()

        opt = build_optimizer_v2(student, optimizer_name="adamw", lr=1e-3, device_type="cpu")
        images, seq = _make_dummy_batch(batch_size=2, h=64, w=96, seq_len=16)

        for step in range(1, 3):
            opt.zero_grad(set_to_none=True)
            losses, _ = forward_and_loss_v2(
                student,
                images,
                seq,
                teacher_v1=teacher,
                kd_alpha=0.35,
                kd_beta=0.5,
                kd_temp=2.0,
                cass_prob=0.25 if step == 2 else 0.0,
                stroke_hog_weight=0.15,
                use_bf16=False,
                device=torch.device("cpu"),
            )
            for name in ("loss", "ce_loss", "mtp_loss", "saliency_loss", "stroke_hog_loss", "kd_loss"):
                val = losses[name]
                self.assertTrue(torch.isfinite(val).item(), f"Non-finite {name} at step {step}: {val}")
                self.assertGreater(float(val.item()), 0.0, f"Expected positive {name} at step {step}")

            losses["loss"].backward()

            nonzero_grads = sum(
                1 for p in student.parameters()
                if p.grad is not None and float(p.grad.abs().sum().item()) > 0.0
            )
            self.assertGreater(nonzero_grads, 10, "Expected nonzero gradients across student params")

            opt.step()

        # Student updated; frozen v1 teacher completely untouched.
        self.assertFalse(
            torch.allclose(student_snap_before, student.lm_head.weight.detach()),
            "Student weights did not update after 2 steps",
        )
        for k, v in teacher.state_dict().items():
            self.assertTrue(torch.equal(v, teacher_snap[k]), f"Frozen teacher param {k} mutated!")
        for name, p in teacher.named_parameters():
            self.assertIsNone(p.grad, f"Frozen teacher param {name} unexpectedly accumulated grad")

    def test_2_topk_teacher_cache_roundtrip_and_sparse_jsd(self) -> None:
        """TopKTeacherCache write & read roundtrip + sparse Top-8 JSD loss & shape parity."""
        torch.manual_seed(11)
        teacher = _make_tiny_v1_teacher()
        student = _make_tiny_v2(grad_checkpoint=False)
        student.train()

        images, seq = _make_dummy_batch(batch_size=2, h=64, w=96, seq_len=14)
        tgt_in = seq[:, :-1]
        tgt_out = seq[:, 1:]
        T = tgt_out.shape[1]
        K = 8

        with torch.no_grad():
            t_logits = teacher(images, tgt_in, return_aux=False, return_mtp=False)["logits"]
        vals_b, idx_b = extract_topk_from_logits(t_logits, k=K)
        self.assertEqual(vals_b.dtype, np.float16)
        self.assertEqual(idx_b.dtype, np.uint8)
        self.assertEqual(vals_b.shape, (2, T, K))
        self.assertEqual(idx_b.shape, (2, T, K))

        with tempfile.TemporaryDirectory() as td:
            records = [
                ("sample_a", vals_b[0], idx_b[0]),
                ("sample_b", vals_b[1, : T - 2], idx_b[1, : T - 2]),
            ]
            cache = TopKTeacherCache.write(td, records, k=K, vocab_size=VOCAB_SIZE, teacher_ckpt="dummy_v1.pt")
            try:
                self.assertEqual(len(cache), 2)
                self.assertTrue(cache.has("sample_a"))
                self.assertTrue(cache.has("sample_b"))

                # Exact roundtrip parity
                rv0, ri0 = cache.get("sample_a")
                self.assertEqual(rv0.dtype, np.float16)
                self.assertEqual(ri0.dtype, np.uint8)
                self.assertTrue(np.array_equal(rv0, vals_b[0]))
                self.assertTrue(np.array_equal(ri0, idx_b[0]))

                rv1, ri1 = cache.get("sample_b")
                self.assertEqual(rv1.shape, (T - 2, K))
                self.assertTrue(np.array_equal(rv1, vals_b[1, : T - 2]))
                self.assertTrue(np.array_equal(ri1, idx_b[1, : T - 2]))

                # Batch fetch + sparse Top-8 JSD loss
                b_vals, b_idx, b_mask, n_hits = cache.get_batch(
                    ["sample_a", "sample_b"], seq_len=T, device=torch.device("cpu")
                )
                self.assertEqual(n_hits, 2)
                self.assertEqual(tuple(b_vals.shape), (2, T, K))
                self.assertEqual(tuple(b_idx.shape), (2, T, K))
                self.assertEqual(tuple(b_mask.shape), (2, T))
                self.assertTrue(bool(b_mask[0].all()))
                self.assertTrue(bool(b_mask[1, : T - 2].all()))
                self.assertFalse(bool(b_mask[1, T - 2 :].any()))

                s_out = student(images, tgt_in, return_aux=False, return_mtp=False)
                s_logits = s_out["logits"]
                kd_loss = compute_sparse_topk_jsd_loss(
                    s_logits, b_vals, b_idx, tgt_out, beta=0.5, temp=2.0, valid_mask=b_mask
                )
                self.assertTrue(torch.isfinite(kd_loss).item(), f"Non-finite sparse JSD: {kd_loss}")
                self.assertGreater(float(kd_loss.item()), 0.0)
                kd_loss.backward()
                self.assertIsNotNone(student.lm_head.weight.grad)
                self.assertGreater(float(student.lm_head.weight.grad.abs().sum().item()), 0.0)
            finally:
                cache.close()

    def test_3_muon_adamw_param_partition(self) -> None:
        """Verify 2D Transformer weights -> Muon, 4D convs/1D norms/embeddings/aux -> AdamW, 0 dropped/duped."""
        student = _make_tiny_v2(grad_checkpoint=False)
        muon_named, adamw_named = partition_muon_adamw_params(student)

        muon_ids = [id(p) for _, p in muon_named]
        adamw_ids = [id(p) for _, p in adamw_named]
        all_trainable_ids = [id(p) for p in student.parameters() if p.requires_grad]

        # 0 dropped, 0 duplicated
        self.assertEqual(len(muon_ids) + len(adamw_ids), len(all_trainable_ids))
        self.assertEqual(len(set(muon_ids) & set(adamw_ids)), 0)
        self.assertEqual(set(muon_ids) | set(adamw_ids), set(all_trainable_ids))
        self.assertGreater(len(muon_named), 0)
        self.assertGreater(len(adamw_named), 0)

        muon_name_set = {n for n, _ in muon_named}
        adamw_name_set = {n for n, _ in adamw_named}

        # Every Muon param must be 2D and inside Transformer blocks
        for name, p in muon_named:
            self.assertEqual(p.ndim, 2, f"Muon param {name} is not 2D (ndim={p.ndim})")
            self.assertFalse(name.startswith("backbone."), f"Backbone param {name} wrongly in Muon")
            self.assertNotEqual(name, "tok_emb.weight")
            self.assertNotEqual(name, "lm_head.weight")

        # Spot-check expected routing
        self.assertIn("encoder.axial_layers.0.row_attn.qkv_proj.weight", muon_name_set)
        self.assertIn("encoder.global_layers.0.ffn.w_gate_up.weight", muon_name_set)
        self.assertIn("dec_layers.0.self_attn.q_proj.weight", muon_name_set)

        self.assertIn("backbone.stage1.down.0.weight", adamw_name_set)
        self.assertIn("tok_emb.weight", adamw_name_set)
        self.assertIn("lm_head.weight", adamw_name_set)
        self.assertIn("dec_norm.weight", adamw_name_set)
        self.assertIn("stroke_hog_head.net.0.weight", adamw_name_set)

        # CPU fallback defaults to AdamW; explicit allow_cpu_muon builds Muon+AdamW cleanly
        opt_cpu = build_optimizer_v2(student, optimizer_name="muon", lr=3e-4, device_type="cpu")
        self.assertEqual(opt_cpu.mode, "adamw")

        opt_muon = build_optimizer_v2(
            student, optimizer_name="muon", lr=3e-4, device_type="cpu", allow_cpu_muon=True
        )
        self.assertEqual(opt_muon.mode, "muon+adamw")
        self.assertEqual(len(opt_muon.muon_param_names), len(muon_named))
        self.assertEqual(len(opt_muon.adamw_param_names), len(adamw_named))

    def test_4_atomic_autosave_and_reload(self) -> None:
        """Atomic 5-min autosave writes uni_omr_v2_latest.pt + uni_omr_v2_ema_latest.pt and reloads cleanly."""
        torch.manual_seed(19)
        student = _make_tiny_v2(grad_checkpoint=False)
        gpu_ema = GPUEmaShadow(student, decay=0.997, update_every=1)
        qat_ctrl = QATInt4ControllerV2(student, block_size=32, enabled=True)

        # Mutate student weights and update EMA so student != ema
        with torch.no_grad():
            student.lm_head.weight.add_(0.25)
        gpu_ema.update(student, step=1)
        qat_ctrl.refresh_cache()
        qat_ctrl.detach()

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            latest_p, ema_p = atomic_save_v2_latest(
                out_dir,
                student,
                student.cfg,
                step=42,
                best_val_ser=0.123,
                gpu_ema=gpu_ema,
                best_ema_ser=0.111,
            )
            self.assertTrue(latest_p.is_file())
            self.assertIsNotNone(ema_p)
            assert ema_p is not None
            self.assertTrue(ema_p.is_file())
            self.assertEqual(latest_p.name, "uni_omr_v2_latest.pt")
            self.assertEqual(ema_p.name, "uni_omr_v2_ema_latest.pt")
            # No leftover .tmp files
            self.assertEqual(list(out_dir.glob("*.tmp")), [])

            # Reload full latest checkpoint
            reloaded = _make_tiny_v2(grad_checkpoint=False)
            info_full = load_v2_checkpoint(reloaded, latest_p, encoder_only=False)
            self.assertEqual(len(info_full["missing_keys"]), 0)
            self.assertEqual(len(info_full["unexpected_keys"]), 0)
            for k, v in student.state_dict().items():
                self.assertTrue(torch.equal(v, reloaded.state_dict()[k]), f"Latest mismatch on {k}")

            # Reload EMA checkpoint
            reloaded_ema = _make_tiny_v2(grad_checkpoint=False)
            info_ema = load_v2_checkpoint(reloaded_ema, ema_p, encoder_only=False)
            self.assertEqual(len(info_ema["missing_keys"]), 0)
            for k, v in gpu_ema.cpu_state_dict().items():
                self.assertTrue(torch.equal(v, reloaded_ema.state_dict()[k]), f"EMA mismatch on {k}")

            # Verify --init-encoder-from loads only backbone.* + encoder.*
            enc_target = _make_tiny_v2(grad_checkpoint=False)
            dec_before = enc_target.lm_head.weight.detach().clone()
            info_enc = load_v2_checkpoint(enc_target, latest_p, encoder_only=True)
            self.assertGreater(info_enc["loaded_count"], 0)
            self.assertTrue(torch.equal(enc_target.lm_head.weight.detach(), dec_before))
            self.assertTrue(
                torch.equal(
                    enc_target.backbone.stage1.down[0].weight.detach(),
                    student.backbone.stage1.down[0].weight.detach(),
                )
            )

        # Verify KD cosine annealing over last 25% of steps
        self.assertAlmostEqual(kd_alpha_for_step(500, 1000, 0.35, 0.25), 0.35, places=6)
        self.assertAlmostEqual(kd_alpha_for_step(750, 1000, 0.35, 0.25), 0.35, places=6)
        self.assertAlmostEqual(kd_alpha_for_step(875, 1000, 0.35, 0.25), 0.175, places=6)
        self.assertAlmostEqual(kd_alpha_for_step(1000, 1000, 0.35, 0.25), 0.0, places=6)

    def test_5_teacher_kd_inference_mode_and_zero_alpha_skip(self) -> None:
        """Online v1 teacher runs in inference_mode when kd_alpha>0 and is skipped when kd_alpha<=0."""
        torch.manual_seed(29)
        student = _make_tiny_v2(grad_checkpoint=False)
        student.train()
        teacher = _make_tiny_v1_teacher()
        images, seq = _make_dummy_batch(batch_size=2, h=64, w=96, seq_len=14)

        call_log: list[bool] = []
        orig_forward = teacher.forward

        def _spy_forward(*args, **kwargs):
            call_log.append(torch.is_inference_mode_enabled())
            return orig_forward(*args, **kwargs)

        teacher.forward = _spy_forward  # type: ignore[method-assign]

        # 1) kd_alpha == 0.0 -> teacher forward must be skipped completely.
        losses_zero, _ = forward_and_loss_v2(
            student,
            images,
            seq,
            teacher_v1=teacher,
            kd_alpha=0.0,
            use_bf16=False,
            device=torch.device("cpu"),
        )
        self.assertEqual(len(call_log), 0, "Teacher forward should be skipped when kd_alpha <= 0")
        self.assertEqual(float(losses_zero["kd_loss"].item()), 0.0)

        # 2) kd_alpha > 0.0 -> teacher runs under torch.inference_mode() and backward succeeds.
        student.zero_grad(set_to_none=True)
        losses_kd, _ = forward_and_loss_v2(
            student,
            images,
            seq,
            teacher_v1=teacher,
            kd_alpha=0.35,
            use_bf16=False,
            device=torch.device("cpu"),
        )
        self.assertEqual(call_log, [True], "Teacher forward must run under torch.inference_mode()")
        self.assertGreater(float(losses_kd["kd_loss"].item()), 0.0)
        losses_kd["loss"].backward()
        self.assertIsNotNone(student.lm_head.weight.grad)

    def test_6_handoff_best_checkpoints_and_deferred_history_flush(self) -> None:
        """run_train_v2 flushes deferred loss history and always guarantees uni_omr_v2_ema_best.pt."""
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            parser = build_arg_parser()
            args = parser.parse_args([
                "--preset", "tiny",
                "--device", "cpu",
                "--no-bf16",
                "--steps", "3",
                "--batch-size", "2",
                "--grad-accum", "2",
                "--seq-len", "16",
                "--log-every", "10",
                "--eval-every", "0",
                "--gpu-ema-decay", "0.99",
                "--gpu-ema-every", "1",
                "--out-dir", str(out_dir),
            ])
            res = run_train_v2(args)
            self.assertEqual(len(res["history"]), 3)
            for idx, row in enumerate(res["history"], start=1):
                self.assertEqual(row["step"], idx)
                self.assertGreater(row["loss"], 0.0)

            for fname in (
                "uni_omr_v2_latest.pt",
                "uni_omr_v2_ema_latest.pt",
                "uni_omr_v2_best.pt",
                "uni_omr_v2_ema_best.pt",
                "train_log.json",
            ):
                self.assertTrue((out_dir / fname).is_file(), f"Missing expected handoff file: {fname}")

            # Verify ensure_v2_best_checkpoints never overwrites an existing ema_best.pt.
            marker_ckpt = torch.load(str(out_dir / "uni_omr_v2_ema_best.pt"), map_location="cpu", weights_only=True)
            marker_ckpt["step"] = 999
            torch.save(marker_ckpt, str(out_dir / "uni_omr_v2_ema_best.pt"))

            student = _make_tiny_v2(grad_checkpoint=False)
            ensure_v2_best_checkpoints(out_dir, student, student.cfg, step=1234)
            reloaded = torch.load(str(out_dir / "uni_omr_v2_ema_best.pt"), map_location="cpu", weights_only=True)
            self.assertEqual(reloaded["step"], 999, "Existing uni_omr_v2_ema_best.pt must not be overwritten")


if __name__ == "__main__":
    unittest.main(verbosity=2)
