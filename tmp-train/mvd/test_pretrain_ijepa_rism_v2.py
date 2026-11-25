"""CPU-only verification suite for ``pretrain_ijepa_rism_v2.py`` (V-JEPA 2.1 + Gram + Stroke-HOG)."""

from __future__ import annotations

import os

# Strictly CPU-only — never touch the GPU while training runs are active.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from dataclasses import asdict
from pathlib import Path
import random
import sys
import tempfile
from typing import List

import numpy as np
from PIL import Image, ImageDraw
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net  # noqa: E402
from pretrain_ijepa_rism_v2 import (  # noqa: E402
    DeepStackPredictors,
    EMAEncoderTeacherV2,
    StrokeHOGHead,
    UnlabeledPageBatchPrefetcher,
    build_argparser,
    build_optimizer_v2,
    compute_gram_anchoring_loss,
    cosine_ema_decay,
    discover_unlabeled_images,
    encode_student_aligned_v2,
    freeze_non_encoder_params_v2,
    run_pretrain,
    save_ijepa_v2_checkpoint,
    vjepa21_forward_loss,
)


def _make_tiny_cfg() -> UniOMRv2Config:
    return UniOMRv2Config(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_axial_layers=1,
        n_global_layers=1,
        n_dec_layers=1,
        ffn_mult=2.0,
        axial_ffn_mult=2.0,
        dropout=0.0,
        backbone_channels=(16, 32, 48, 64),
        backbone_blocks=(1, 1, 1, 1),
        backbone_expand=1.5,
        router_hidden=32,
        max_mem_tokens=256,
        n_bg_summary=4,
        mtp_k=2,
    )


def _create_synthetic_archival_pngs(out_dir: Path, count: int = 4) -> List[Path]:
    """Generate synthetic sepia archival manuscript PNGs with staff lines, stems, and noteheads."""
    rng = random.Random(1234)
    paths: List[Path] = []
    for idx in range(count):
        w, h = (192, 256) if idx % 2 == 0 else (256, 192)
        # Warm sepia parchment paper background with subtle grain
        base = np.zeros((h, w, 3), dtype=np.uint8)
        base[:, :, 0] = 228 + rng.randint(-6, 6)
        base[:, :, 1] = 214 + rng.randint(-6, 6)
        base[:, :, 2] = 186 + rng.randint(-6, 6)
        noise = np.random.default_rng(100 + idx).integers(-5, 6, size=(h, w, 3))
        arr = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(img)

        ink = (42 + rng.randint(-8, 8), 32 + rng.randint(-6, 6), 24 + rng.randint(-6, 6))
        # Draw 3 systems of 5 staff lines + stems + noteheads + barlines
        n_sys = 3
        top_margin = 24
        sys_gap = (h - 2 * top_margin) // n_sys
        for s in range(n_sys):
            y0 = top_margin + s * sys_gap
            for line_i in range(5):
                y = y0 + line_i * 6
                draw.line([(16, y), (w - 16, y)], fill=ink, width=1)
            # Barlines & notes
            for x in range(28, w - 24, 22):
                ny = y0 + rng.randint(0, 24)
                draw.ellipse([(x - 3, ny - 2), (x + 3, ny + 2)], fill=ink)
                draw.line([(x + 3, ny), (x + 3, ny - 14)], fill=ink, width=1)
                if (x // 22) % 3 == 0:
                    draw.line([(x + 11, y0), (x + 11, y0 + 24)], fill=ink, width=1)

        p = out_dir / f"rism_synth_{idx:02d}.png"
        img.save(p)
        paths.append(p)
    return paths


def test_two_training_steps_end_to_end() -> None:
    torch.manual_seed(7)
    rng = random.Random(7)

    with tempfile.TemporaryDirectory() as tmp:
        img_dir = Path(tmp) / "rism"
        img_dir.mkdir(parents=True, exist_ok=True)
        _create_synthetic_archival_pngs(img_dir, count=4)
        paths = discover_unlabeled_images(img_dir)
        assert len(paths) == 4

        loader = UnlabeledPageBatchPrefetcher(
            paths=paths, batch_size=2, max_dim=160, multi_scale=False, seed=7, num_threads=0
        )

        cfg = _make_tiny_cfg()
        student = UniOMRv2Net(cfg)
        freeze_non_encoder_params_v2(student)
        teacher = EMAEncoderTeacherV2(student)
        gram_teacher = EMAEncoderTeacherV2(student)
        predictors = DeepStackPredictors(cfg.d_model)
        stroke_hog_head = StrokeHOGHead(in_ch=student.backbone.p2_ch, n_bins=8)

        opt, all_trainable = build_optimizer_v2(
            student=student,
            predictors=predictors,
            stroke_hog_head=stroke_hog_head,
            optimizer_name="muon",  # Must cleanly fall back to AdamW on CPU
            lr=1e-3,
            device_type="cpu",
        )
        assert opt.mode == "adamw", f"Expected CPU fallback to adamw, got {opt.mode}"

        gram_snapshot_before = [p.detach().clone() for p in gram_teacher.parameters()]
        teacher_snapshot_before = [p.detach().clone() for p in teacher.parameters()]

        student.train()
        predictors.train()
        stroke_hog_head.train()
        teacher.eval()
        gram_teacher.eval()

        required_keys = ("loss", "jepa_masked", "jepa_visible", "deep_sup", "gram", "stroke_hog")
        for step in range(1, 3):
            batch = next(loader)
            opt.zero_grad(set_to_none=True)
            losses = vjepa21_forward_loss(
                student=student,
                teacher=teacher,
                gram_teacher=gram_teacher,
                predictors=predictors,
                stroke_hog_head=stroke_hog_head,
                images=batch,
                rng=rng,
                use_bf16=False,
                device_type="cpu",
                gram_weight=1.0,
                gram_max_tokens=512,
                stroke_hog_weight=0.25,
            )
            for k in required_keys:
                val = float(losses[k].detach().item())
                assert np.isfinite(val), f"Step {step}: loss '{k}' is not finite ({val})"
                assert val >= 0.0, f"Step {step}: loss '{k}' is negative ({val})"

            losses["loss"].backward()

            # Verify nonzero gradients across backbone, axial, merger, global, predictors, and Stroke-HOG
            def _grad_norm(mod: torch.nn.Module) -> float:
                return sum(float(p.grad.abs().sum().item()) for p in mod.parameters() if p.grad is not None)

            assert _grad_norm(student.backbone) > 0.0, "Backbone received zero gradient"
            assert _grad_norm(student.encoder.axial_layers) > 0.0, "Axial layers received zero gradient"
            assert _grad_norm(student.encoder.patch_merger) > 0.0, "PatchMerger received zero gradient"
            assert _grad_norm(student.encoder.global_layers) > 0.0, "Global layers received zero gradient"
            assert _grad_norm(predictors) > 0.0, "DeepStack predictors received zero gradient"
            assert _grad_norm(stroke_hog_head) > 0.0, "Stroke-HOG head received zero gradient"

            # Verify frozen decoder params have NO gradients
            assert _grad_norm(student.dec_layers) == 0.0, "Decoder should be frozen"
            assert _grad_norm(student.lm_head) == 0.0, "lm_head should be frozen"

            torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
            opt.step()
            teacher.update_from(student, decay=cosine_ema_decay(step, 2))

        # Verify EMA teacher updated while frozen gram_teacher remained bit-identical
        teacher_moved = any(
            not torch.equal(p0, p1) for p0, p1 in zip(teacher_snapshot_before, teacher.parameters())
        )
        gram_unchanged = all(
            torch.equal(p0, p1) for p0, p1 in zip(gram_snapshot_before, gram_teacher.parameters())
        )
        assert teacher_moved, "EMA teacher weights did not update"
        assert gram_unchanged, "Frozen gram_teacher weights changed unexpectedly"
        loader.close()
    print("[PASS] test_two_training_steps_end_to_end")


def test_gram_anchoring_zero_then_nonzero() -> None:
    torch.manual_seed(19)
    with tempfile.TemporaryDirectory() as tmp:
        img_dir = Path(tmp) / "rism"
        img_dir.mkdir(parents=True, exist_ok=True)
        _create_synthetic_archival_pngs(img_dir, count=2)
        loader = UnlabeledPageBatchPrefetcher(
            paths=discover_unlabeled_images(img_dir),
            batch_size=2,
            max_dim=160,
            multi_scale=False,
            seed=19,
            num_threads=0,
        )
        images = next(loader)
        loader.close()

    cfg = _make_tiny_cfg()
    student = UniOMRv2Net(cfg).eval()
    gram_teacher = EMAEncoderTeacherV2(student).eval()

    with torch.no_grad():
        t_levels, routed_coords, t_info = gram_teacher.encode(images)
        kept_idx = t_info["kept_idx"]
        k_kept = int(t_info["k_kept"])
        ink_kept = torch.gather(t_info["local_ink_grid"].flatten(1), 1, kept_idx)
        is_ink = ink_kept > 0.04

        s_levels, _, _ = encode_student_aligned_v2(
            student=student,
            student_images=images,
            teacher_kept_idx=kept_idx,
            teacher_routed_coords=routed_coords,
            teacher_info=t_info,
        )
        gram_step0 = compute_gram_anchoring_loss(
            s_levels[2][:, :k_kept, :],
            t_levels[2][:, :k_kept, :],
            ink_mask=is_ink,
            max_tokens=512,
        )
        assert float(gram_step0.item()) == 0.0, f"Expected gram_loss == 0.0 at step 0, got {gram_step0.item()}"

        # Perturb student encoder weights and verify gram_loss > 0
        for p in student.encoder.global_layers.parameters():
            p.add_(torch.randn_like(p) * 0.05)

        s_levels_pert, _, _ = encode_student_aligned_v2(
            student=student,
            student_images=images,
            teacher_kept_idx=kept_idx,
            teacher_routed_coords=routed_coords,
            teacher_info=t_info,
        )
        gram_after = compute_gram_anchoring_loss(
            s_levels_pert[2][:, :k_kept, :],
            t_levels[2][:, :k_kept, :],
            ink_mask=is_ink,
            max_tokens=512,
        )
        assert float(gram_after.item()) > 0.0, f"Expected gram_loss > 0 after perturbation, got {gram_after.item()}"
    print(f"[PASS] test_gram_anchoring_zero_then_nonzero (step0={gram_step0.item():.6f}, perturbed={gram_after.item():.6f})")


def test_atomic_checkpoint_save_and_load() -> None:
    torch.manual_seed(31)
    cfg = _make_tiny_cfg()
    student = UniOMRv2Net(cfg)
    teacher = EMAEncoderTeacherV2(student)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "uni_omr_v2_ssl.pt"
        save_ijepa_v2_checkpoint(
            ckpt_path,
            student=student,
            teacher=teacher,
            cfg=cfg,
            step=2,
            loss_val=0.42,
        )
        assert ckpt_path.is_file(), "Checkpoint file was not created"
        assert not ckpt_path.with_suffix(".pt.tmp").exists(), "Leftover .tmp file after atomic save"

        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        assert "cfg" in ck and "model" in ck and ck["step"] == 2

        fresh = UniOMRv2Net(cfg)
        inc = fresh.load_state_dict(ck["model"], strict=False)
        missing_encoder = [
            k for k in inc.missing_keys if k.startswith("backbone.") or k.startswith("encoder.")
        ]
        assert len(missing_encoder) == 0, f"Missing encoder keys: {missing_encoder}"
        assert len(inc.missing_keys) == 0, f"Unexpected missing keys: {inc.missing_keys}"
        assert len(inc.unexpected_keys) == 0, f"Unexpected extra keys (predictor/HOG leak?): {inc.unexpected_keys}"
    print("[PASS] test_atomic_checkpoint_save_and_load (0 missing encoder keys, 0 unexpected keys)")


def test_cli_run_pretrain_smoke() -> None:
    torch.manual_seed(42)
    cfg = _make_tiny_cfg()
    init_model = UniOMRv2Net(cfg)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        img_dir = root / "rism_hw"
        out_dir = root / "out"
        img_dir.mkdir(parents=True, exist_ok=True)
        _create_synthetic_archival_pngs(img_dir, count=4)

        init_ckpt = root / "init_tiny.pt"
        torch.save({"cfg": asdict(cfg), "model": init_model.state_dict(), "step": 0}, str(init_ckpt))

        args = build_argparser().parse_args([
            "--init-from", str(init_ckpt),
            "--image-dir", str(img_dir),
            "--out-dir", str(out_dir),
            "--steps", "2",
            "--batch-size", "2",
            "--grad-accum", "1",
            "--max-dim", "160",
            "--no-multi-scale",
            "--optimizer", "muon",
            "--device", "cpu",
            "--log-every", "1",
            "--save-every", "0",
            "--save-every-minutes", "0.0001",
            "--seed", "42",
        ])
        summary = run_pretrain(args)
        assert summary["steps"] == 2
        assert summary["num_images"] == 4
        assert summary["optimizer"] == "adamw"
        assert (out_dir / "uni_omr_v2_ijepa_last.pt").is_file()
        assert (out_dir / "uni_omr_v2_ijepa_autosave.pt").is_file()
        assert (out_dir / "vjepa21_train_log.json").is_file()
    print("[PASS] test_cli_run_pretrain_smoke")


def main() -> int:
    test_two_training_steps_end_to_end()
    test_gram_anchoring_zero_then_nonzero()
    test_atomic_checkpoint_save_and_load()
    test_cli_run_pretrain_smoke()
    print("ALL 4 TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
