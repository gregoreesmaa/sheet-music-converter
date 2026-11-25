"""CPU-only unit tests for ``pretrain_ijepa_rism.py``."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
from PIL import Image
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import data  # noqa: E402
from pretrain_ijepa_rism import (  # noqa: E402
    EMAEncoderTeacher,
    IJEPAPredictor,
    apply_ink_strip_mask,
    build_argparser,
    discover_unlabeled_images,
    estimate_paper_tone_p85,
    freeze_non_encoder_params,
    ijepa_forward_loss,
    run_pretrain,
)
from uni_omr import InkTokenRouter, UniOMRConfig, UniOMRNet  # noqa: E402


def _make_parchment_score_image(h: int = 128, w: int = 128) -> np.ndarray:
    """Create a synthetic warm-parchment page with dark brown horizontal staves & noteheads."""
    # Warm parchment ~(228, 214, 185)
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 228
    img[:, :, 1] = 214
    img[:, :, 2] = 185
    # Draw 2 staves with brown iron-gall ink ~(75, 58, 42)
    for base_y in (24, 76):
        for dy in range(0, 25, 6):
            img[base_y + dy : base_y + dy + 2, 12 : w - 12] = (75, 58, 42)
        for x in range(20, w - 20, 14):
            img[base_y + 4 : base_y + 16, x : x + 4] = (65, 48, 35)
    return img


def test_strip_masking_fills_with_paper_tone() -> None:
    arr = _make_parchment_score_image(128, 128)
    t = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 128, 128)
    local_ink_grid, _, _ = InkTokenRouter._compute_ink_and_paper_gate(t, 16, 16)

    expected_tone = estimate_paper_tone_p85(t[0])  # (3, 1, 1)
    # Parchment tone should be close to (228/255, 214/255, 185/255)
    assert abs(float(expected_tone[0].item()) - 228.0 / 255.0) < 0.03
    assert abs(float(expected_tone[2].item()) - 185.0 / 255.0) < 0.03

    masked, grid_mask = apply_ink_strip_mask(
        t, local_ink_grid, mask_ratio_range=(0.45, 0.55), rng=random.Random(123)
    )
    assert masked.shape == t.shape
    assert grid_mask.shape == (1, 1, 16, 16)
    assert bool(grid_mask.any()), "Expected some ink-bearing patches to be masked"

    # Every changed pixel must equal the estimated p85 paper tone (zero black/white border artifacts).
    diff = (masked[0] - t[0]).abs().max(dim=0).values > 1e-5
    assert bool(diff.any()), "Expected some ink pixels to be replaced by paper tone"
    changed_rgb = masked[0][:, diff]  # (3, N_changed)
    max_err = (changed_rgb - expected_tone.view(3, 1)).abs().max().item()
    assert max_err < 1e-5, f"Masked pixels diverged from p85 paper tone: max_err={max_err}"


def test_two_steps_decrease_loss_update_ema_and_preserve_decoder_bit_identical() -> None:
    torch.manual_seed(7)
    cfg = UniOMRConfig(
        d_model=64,
        n_enc_layers=2,
        n_dec_layers=2,
        n_heads=4,
        ffn_mult=2.0,
        dropout=0.0,
        vocab_size=32,
        keep_ratio=0.5,
        dynamic_routing=True,
        min_keep_ratio=0.25,
        max_keep_ratio=0.75,
        backbone_stride="4x8",
        use_uv_aux=True,
        mtp_k=3,
        grad_checkpoint=False,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        img_dir = tmp_dir / "rism_dummy"
        img_dir.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            arr = _make_parchment_score_image(96, 128)
            Image.fromarray(arr, mode="RGB").save(img_dir / f"page_{i}.png")

        # Save initial warm-start checkpoint
        init_model = UniOMRNet(cfg)
        init_sd = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}
        init_ckpt = tmp_dir / "init_uni_omr.pt"
        torch.save(
            {
                "step": 100,
                "config": asdict(cfg),
                "vocab": list(data.VOCAB),
                "model": init_sd,
            },
            str(init_ckpt),
        )

        # (b) Verify 2 steps decrease loss and update EMA teacher
        student = UniOMRNet(cfg)
        student.load_state_dict(init_sd, strict=True)
        trainable_enc = freeze_non_encoder_params(student)
        teacher = EMAEncoderTeacher(student)
        pred_head = IJEPAPredictor(cfg.d_model)

        teacher_before = {
            k: v.detach().clone() for k, v in teacher.backbone.state_dict().items()
        }
        opt = torch.optim.AdamW(trainable_enc + list(pred_head.parameters()), lr=1e-3, weight_decay=0.0)

        t_img = torch.from_numpy(_make_parchment_score_image(96, 128).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        losses_seq = []
        for s in range(3):
            opt.zero_grad(set_to_none=True)
            # Fixed seed per step so we measure genuine optimization decrease on the same mask
            out = ijepa_forward_loss(
                student=student,
                teacher=teacher,
                pred_head=pred_head,
                images=t_img,
                rng=random.Random(42),
                use_bf16=False,
                device_type="cpu",
            )
            out["loss"].backward()
            opt.step()
            teacher.update_from(student, decay=0.9)
            losses_seq.append(float(out["loss"].item()))

        assert losses_seq[-1] < losses_seq[0], f"Expected loss to decrease: {losses_seq}"
        ema_changed = any(
            not torch.equal(teacher_before[k], v)
            for k, v in teacher.backbone.state_dict().items()
            if v.is_floating_point()
        )
        assert ema_changed, "Expected EMA teacher weights to update"

        # (c) End-to-end CLI run_pretrain + strict=True reload + bit-identical decoder weights
        out_dir = tmp_dir / "ijepa_out"
        parser = build_argparser()
        args = parser.parse_args(
            [
                "--init-from", str(init_ckpt),
                "--image-dir", str(img_dir),
                "--out-dir", str(out_dir),
                "--steps", "2",
                "--batch-size", "2",
                "--grad-accum", "1",
                "--max-dim", "128",
                "--lr", "1e-3",
                "--device", "cpu",
                "--save-every", "2",
                "--log-every", "1",
                "--no-multi-scale",
            ]
        )
        summary = run_pretrain(args)
        assert summary["steps"] == 2
        assert summary["num_images"] == 4

        best_ckpt = out_dir / "uni_omr_ijepa_best.pt"
        assert best_ckpt.is_file(), "Expected uni_omr_ijepa_best.pt to be saved"
        ck_out = torch.load(str(best_ckpt), map_location="cpu", weights_only=True)
        loaded_cfg = UniOMRConfig(**ck_out["config"])
        reloaded = UniOMRNet(loaded_cfg)
        reloaded.load_state_dict(ck_out["model"], strict=True)

        # Verify pred_head did NOT leak into saved state_dict
        assert not any("pred_head" in k for k in ck_out["model"].keys())

        # Verify every decoder / lm_head / mtp_heads / tok_emb / uv_head weight is bit-identical
        preserved_prefixes = ("tok_emb.", "dec_layers.", "dec_norm.", "lm_head.", "mtp_heads.", "uv_head.")
        n_checked = 0
        for k, v_init in init_sd.items():
            if k.startswith(preserved_prefixes):
                assert torch.equal(v_init, ck_out["model"][k]), f"Decoder/non-encoder weight changed: {k}"
                n_checked += 1
        assert n_checked > 0, "Expected to verify preserved decoder keys"

        # Verify encoder weights DID change from init
        enc_changed = any(
            not torch.equal(init_sd[k], ck_out["model"][k])
            for k in init_sd
            if k.startswith(("backbone.", "router.", "enc_layers.", "enc_norm.")) and init_sd[k].is_floating_point()
        )
        assert enc_changed, "Expected encoder weights to be updated in saved checkpoint"


def test_discover_rism_hw_count() -> None:
    rism_dir = Path(r"D:\tmp\unlabeled-10k-real\rism_hw")
    if rism_dir.is_dir():
        paths = discover_unlabeled_images(rism_dir)
        assert len(paths) == 5707, f"Expected 5707 RISM HW images, found {len(paths)}"


if __name__ == "__main__":
    test_strip_masking_fills_with_paper_tone()
    test_two_steps_decrease_loss_update_ema_and_preserve_decoder_bit_identical()
    test_discover_rism_hw_count()
    print("ALL PRETRAIN I-JEPA TESTS PASSED", flush=True)
