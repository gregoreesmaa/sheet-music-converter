"""CPU unit tests for Gauge-RoPE (tmp-train/mvd/gauge_rope.py)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from gauge_rope import (  # noqa: E402
    GaugeFieldHead,
    GaugeRoPE2D,
    apply_synthetic_droop,
    attach_gauge_rope,
    diffeomorphic_equivariance_loss,
    get_gauge_alpha,
    get_gauge_head,
)
from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net  # noqa: E402


def _tiny_v2_cfg() -> UniOMRv2Config:
    return UniOMRv2Config(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_axial_layers=1,
        n_global_layers=2,
        n_dec_layers=2,
        ffn_mult=2.0,
        axial_ffn_mult=2.0,
        backbone_channels=(16, 32, 48, 64),
        backbone_blocks=(1, 1, 1, 1),
        max_mem_tokens=256,
        mtp_k=2,
        rope_jitter=0.0,
    )


def test_param_count_under_budget() -> None:
    head = GaugeFieldHead()
    n_params = sum(p.numel() for p in head.parameters())
    assert n_params < 250_000, f"Expected < 0.25M params, got {n_params:,}"
    assert n_params > 50_000, f"Unexpectedly tiny head: {n_params:,}"
    assert float(head.alpha.item()) == 0.0
    print(f"[PASS] test_param_count_under_budget: {n_params:,} params ({n_params / 1e6:.4f}M, ~{n_params / 1e6:.3f} MB INT8)")


def test_exact_identity_at_alpha_zero() -> None:
    torch.manual_seed(0)
    cfg = _tiny_v2_cfg()
    m_ctrl = UniOMRv2Net(cfg).eval()
    m_gauge = copy.deepcopy(m_ctrl).eval()
    attach_gauge_rope(m_gauge)

    assert get_gauge_alpha(m_gauge) == 0.0
    assert isinstance(m_gauge.encoder.rope2d, GaugeRoPE2D)

    images = torch.rand(2, 3, 128, 128)
    tgt_ids = torch.randint(0, 50, (2, 16))

    with torch.no_grad():
        out_ctrl = m_ctrl(images, tgt_ids, return_aux=False, return_mtp=False)
        out_gauge = m_gauge(images, tgt_ids, return_aux=False, return_mtp=False)

    max_diff = (out_ctrl["logits"] - out_gauge["logits"]).abs().max().item()
    assert max_diff == 0.0, f"Expected exact bit-identity at alpha=0, got max_diff={max_diff}"

    # Verify non-zero alpha actually changes logits
    with torch.no_grad():
        get_gauge_head(m_gauge).alpha.fill_(1.0)
        out_shifted = m_gauge(images, tgt_ids, return_aux=False, return_mtp=False)
    shift_diff = (out_ctrl["logits"] - out_shifted["logits"]).abs().max().item()
    assert shift_diff > 1e-5, f"Expected non-zero change when alpha=1.0, got {shift_diff}"
    print(f"[PASS] test_exact_identity_at_alpha_zero: alpha=0 diff={max_diff:.1e}, alpha=1 diff={shift_diff:.4e}")


def test_gradient_flow_alpha_and_head() -> None:
    torch.manual_seed(1)
    cfg = _tiny_v2_cfg()
    model = UniOMRv2Net(cfg).train()
    gh = attach_gauge_rope(model)

    images = torch.rand(2, 3, 128, 128)
    tgt_ids = torch.randint(0, 50, (2, 16))
    targets = torch.randint(0, 50, (2, 16))

    # Step 1: at alpha=0, alpha MUST receive non-zero gradient from main CE loss
    out = model(images, tgt_ids)
    losses = model.compute_loss(out, targets, images=images)
    losses["loss"].backward()

    assert gh.alpha.grad is not None, "alpha.grad is None at step 0"
    assert torch.isfinite(gh.alpha.grad).all(), "alpha.grad is non-finite"
    assert gh.alpha.grad.abs().item() > 0.0, "alpha.grad is zero at step 0"

    # Step 2: once alpha != 0, GaugeFieldHead conv weights MUST receive non-zero gradient from CE loss
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        gh.alpha.fill_(0.25)
    out2 = model(images, tgt_ids)
    losses2 = model.compute_loss(out2, targets, images=images)
    losses2["loss"].backward()

    weight_grads = [
        p.grad.abs().max().item()
        for name, p in gh.named_parameters()
        if name != "alpha" and p.grad is not None
    ]
    assert len(weight_grads) > 0, "No gradients reached GaugeFieldHead conv parameters"
    assert max(weight_grads) > 0.0, "All GaugeFieldHead conv gradients are zero"
    print(f"[PASS] test_gradient_flow_alpha_and_head: max_head_grad={max(weight_grads):.4e}")


def test_diffeomorphic_equivariance_loss_decreases() -> None:
    torch.manual_seed(2)
    head = GaugeFieldHead().train()
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3)

    # Synthetic score-like image with horizontal staff lines and vertical barlines
    B, C, H, W = 2, 3, 128, 128
    img = torch.ones(B, C, H, W)
    for y in (24, 30, 36, 42, 48, 80, 86, 92, 98, 104):
        img[:, :, y : y + 2, 10:118] = 0.1
    for x in (10, 46, 82, 118):
        img[:, :, 24:50, x : x + 2] = 0.1
        img[:, :, 80:106, x : x + 2] = 0.1

    # Measure initial loss on a fixed set of random warps
    g_eval = torch.Generator().manual_seed(123)
    with torch.no_grad():
        init_loss, init_stats = diffeomorphic_equivariance_loss(
            head, img, generator=g_eval, return_stats=True
        )
    assert init_loss.item() > 0.0 and torch.isfinite(init_loss)

    # Optimize for 25 steps on fixed-distribution warps
    for step in range(25):
        opt.zero_grad(set_to_none=True)
        g_step = torch.Generator().manual_seed(1000 + step)
        loss = diffeomorphic_equivariance_loss(head, img, generator=g_step)
        assert torch.isfinite(loss), f"Non-finite loss at step {step}"
        loss.backward()
        opt.step()

    g_eval2 = torch.Generator().manual_seed(123)
    with torch.no_grad():
        final_loss, final_stats = diffeomorphic_equivariance_loss(
            head, img, generator=g_eval2, return_stats=True
        )
    assert final_loss.item() < init_loss.item(), (
        f"Expected equivariance loss to decrease: init={init_loss.item():.5f}, final={final_loss.item():.5f}"
    )
    print(
        f"[PASS] test_diffeomorphic_equivariance_loss_decreases: "
        f"init={init_loss.item():.5f} -> final={final_loss.item():.5f} "
        f"(eq {init_stats['eq_loss']:.5f} -> {final_stats['eq_loss']:.5f})"
    )


def test_bf16_autocast_safety() -> None:
    torch.manual_seed(3)
    cfg = _tiny_v2_cfg()
    model = UniOMRv2Net(cfg).train()
    gh = attach_gauge_rope(model)
    with torch.no_grad():
        gh.alpha.fill_(0.5)

    images = torch.rand(2, 3, 128, 128)
    tgt_ids = torch.randint(0, 50, (2, 16))
    targets = torch.randint(0, 50, (2, 16))

    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(images, tgt_ids)
        losses = model.compute_loss(out, targets, images=images)
        eq_loss = diffeomorphic_equivariance_loss(gh, images)
        total = losses["loss"] + 0.1 * eq_loss

    assert torch.isfinite(out["logits"]).all(), "Non-finite logits under BF16 autocast"
    assert torch.isfinite(total), "Non-finite total loss under BF16 autocast"
    total.backward()
    for name, p in gh.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name} under BF16 autocast"
    print(f"[PASS] test_bf16_autocast_safety: total={total.item():.4f}")


def test_synthetic_droop_preserves_shape_and_range() -> None:
    img = torch.rand(2, 3, 128, 128)
    warped = apply_synthetic_droop(img, droop_frac=0.025)
    assert warped.shape == img.shape
    assert warped.dtype == img.dtype
    assert torch.isfinite(warped).all()
    assert (warped - img).abs().mean().item() > 1e-3
    print("[PASS] test_synthetic_droop_preserves_shape_and_range")


def main() -> None:
    test_param_count_under_budget()
    test_exact_identity_at_alpha_zero()
    test_gradient_flow_alpha_and_head()
    test_diffeomorphic_equivariance_loss_decreases()
    test_bf16_autocast_safety()
    test_synthetic_droop_preserves_shape_and_range()
    print("ALL GAUGE-ROPE UNIT TESTS PASSED!")


if __name__ == "__main__":
    main()
