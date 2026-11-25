"""Comprehensive CPU unit tests for UniOMR v2 (uni_omr_v2.py)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Enforce strictly CPU-only execution.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from uni_omr_v2 import (  # noqa: E402
    AspectNormalizedRoPE2D,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SYS_ID,
    VOCAB_SIZE,
    UniOMRv2Config,
    UniOMRv2Net,
)


def _tiny_cfg(**overrides) -> UniOMRv2Config:
    """Fast CPU-friendly config preserving every v2 architectural component."""
    params = dict(
        d_model=64,
        n_heads=8,
        n_kv_heads=2,
        n_axial_layers=2,
        n_global_layers=2,
        n_dec_layers=2,
        ffn_mult=2.0,
        dropout=0.0,
        backbone_channels=(16, 32, 48, 64),
        backbone_blocks=(1, 1, 1, 1),
        backbone_expand=2.0,
        router_hidden=32,
        max_mem_tokens=3072,
        mtp_k=4,
    )
    params.update(overrides)
    return UniOMRv2Config(**params)


def _check_presets_param_counts_and_byte_budgets() -> dict:
    """Verify small_45m and edge85m parameter counts, backbone budget, and <55 MB WebGPU size."""
    m_small = UniOMRv2Net(UniOMRv2Config.small_45m())
    s_small = m_small.param_summary()

    m_edge = UniOMRv2Net(UniOMRv2Config.edge85m())
    s_edge = m_edge.param_summary()

    # Backbone must be ~5-7M (down from 15.4M in v1 ResNet).
    assert 4_500_000 <= s_small["backbone_params"] <= 7_500_000, f"small backbone={s_small['backbone_params']}"
    assert 4_500_000 <= s_edge["backbone_params"] <= 7_500_000, f"edge backbone={s_edge['backbone_params']}"

    # small_45m: ~43M total, < 25 MB pure INT4 (even with scales!), < 55 MB hybrid.
    assert 40_000_000 <= s_small["total_params"] <= 46_000_000, f"small total={s_small['total_params']}"
    assert s_small["pure_int4_mb"] < 25.0, f"small pure_int4_mb={s_small['pure_int4_mb']:.2f}"
    assert s_small["pure_int4_with_scales_mb"] < 25.0, (
        f"small pure_int4_with_scales_mb={s_small['pure_int4_with_scales_mb']:.2f}"
    )
    assert s_small["hybrid_fp16_int4_with_scales_mb"] < 55.0

    # edge85m: d_model=512, ~80-85M total, < 55 MB Hybrid FP16/INT4-QAT (even with group-32 scales!).
    assert m_edge.cfg.d_model == 512
    assert 79_000_000 <= s_edge["total_params"] <= 86_000_000, f"edge total={s_edge['total_params']}"
    assert s_edge["hybrid_fp16_int4_mb"] < 52.0, f"edge hybrid_raw={s_edge['hybrid_fp16_int4_mb']:.2f}"
    assert s_edge["hybrid_fp16_int4_with_scales_mb"] < 55.0, (
        f"edge hybrid_scaled={s_edge['hybrid_fp16_int4_with_scales_mb']:.2f}"
    )

    # Verify strip_aux_heads() zeroes out aux params completely.
    assert s_small["aux_params"] > 0
    m_small.strip_aux_heads(strip_mtp=True)
    s_stripped = m_small.param_summary()
    assert s_stripped["aux_params"] == 0
    assert s_stripped["mtp_params"] == 0
    assert s_stripped["total_params"] == s_small["infer_params"]

    return {"small_45m": s_small, "edge85m": s_edge}


def test_presets_param_counts_and_byte_budgets() -> None:
    _check_presets_param_counts_and_byte_budgets()


def test_aspect_normalized_rope2d_resolution_invariance() -> None:
    """1280px and 1536px grids must share identical rotary phases at matching relative (y, x)."""
    rope = AspectNormalizedRoPE2D(head_dim=64, base=10000.0, coord_scale=128.0)

    # 1280x960 at /16 -> (80, 60); 1536x1152 at /16 -> (96, 72)
    h1, w1 = 80, 60
    h2, w2 = 96, 72
    rel_pts = torch.tensor([[[0.0, 0.0], [0.25, 0.5], [0.5, 0.75], [1.0, 1.0]]], dtype=torch.float32)

    coords1 = torch.stack([rel_pts[..., 0] * (h1 - 1), rel_pts[..., 1] * (w1 - 1)], dim=-1)
    coords2 = torch.stack([rel_pts[..., 0] * (h2 - 1), rel_pts[..., 1] * (w2 - 1)], dim=-1)

    cos1, sin1 = rope(coords1, grid_hw=(h1, w1), dtype=torch.float32)
    cos2, sin2 = rope(coords2, grid_hw=(h2, w2), dtype=torch.float32)

    assert (cos1 - cos2).abs().max().item() < 1e-6
    assert (sin1 - sin2).abs().max().item() < 1e-6


def test_forward_backward_portrait_landscape_and_mem_cap() -> None:
    """Test forward + backward with sys_y, sys_count, staff_seg, MTP-4 on portrait & landscape + k <= 3,072 cap."""
    torch.manual_seed(42)
    model = UniOMRv2Net(_tiny_cfg(max_mem_tokens=3072, keep_ratio=0.75, dynamic_routing=False))
    model.train()

    for shape_name, (H, W) in [("portrait", (160, 112)), ("landscape", (112, 160)), ("odd_dims", (135, 197))]:
        model.zero_grad(set_to_none=True)
        B, T = 2, 16
        images = torch.rand(B, 3, H, W)
        # Include BOS, regular tokens, and <sys> tokens.
        seq = torch.randint(0, VOCAB_SIZE - 4, (B, T + 1))
        seq[:, 0] = BOS_ID
        seq[:, 5] = SYS_ID
        seq[:, 11] = SYS_ID
        seq[:, -1] = EOS_ID
        tgt_ids = seq[:, :-1]
        targets = seq[:, 1:]

        sys_y_target = torch.full((B, T), -1.0)
        sys_y_target[:, 5] = 0.35
        sys_y_target[:, 11] = 0.72
        sys_count_target = torch.tensor([3.0, 2.0])
        staff_seg_target = (torch.rand(B, 1, H // 4, W // 4) > 0.7).float()

        out = model(images, tgt_ids)
        assert out["logits"].shape == (B, T, VOCAB_SIZE), shape_name
        assert len(out["mtp_logits"]) == 3, f"Expected 3 MTP heads (mtp_k=4), got {len(out['mtp_logits'])}"
        assert out["sys_y_pred"] is not None and out["sys_y_pred"].shape == (B, T)
        assert out["sys_count_pred"] is not None and out["sys_count_pred"].shape == (B,)
        assert out["staff_seg_logits"] is not None
        assert out["router_info"]["n_out"] <= 3072

        losses = model.compute_loss(
            out,
            targets,
            sys_y_target=sys_y_target,
            sys_count_target=sys_count_target,
            staff_seg_target=staff_seg_target,
        )
        for k in ("loss", "ce_loss", "mtp_loss", "saliency_loss", "sys_y_loss", "sys_count_loss", "staff_seg_loss"):
            assert torch.isfinite(losses[k]), f"{shape_name}: non-finite {k}={losses[k]}"
            assert float(losses[k].item()) > 0.0, f"{shape_name}: zero {k}"

        losses["loss"].backward()
        # Verify gradients flow through backbone, encoder, decoder, MTP, and all 3 aux heads.
        assert model.backbone.stage1.down[0].weight.grad is not None
        assert model.backbone.p2_to_p3.proj.weight.grad is not None
        assert model.encoder.axial_layers[0].row_attn.q_proj.weight.grad is not None
        assert model.encoder.patch_merger.fc1.weight.grad is not None
        assert model.encoder.global_layers[0].self_attn.q_proj.weight.grad is not None
        assert model.dec_layers[0].self_attn.k_proj.weight.grad is not None
        assert model.sys_y_head.net[1].weight.grad is not None
        assert model.sys_count_head.net[1].weight.grad is not None
        assert model.staff_seg_head.head[0].weight.grad is not None

    # Directly verify hard k <= 3,072 cap on a huge merged grid (> 3,072 patches, e.g. 96x72 = 6,912 patches -> 1536x1152 page).
    router = model.encoder.router
    huge_feat = torch.randn(1, model.cfg.d_model, 96, 72)  # 6,912 merged patches
    huge_img = torch.rand(1, 3, 256, 192)
    tokens, coords, info = router(huge_feat, images=huge_img, keep_ratio_override=1.0)
    assert info["n_in"] == 6912
    assert info["n_out"] <= 3072, f"Cap violated: n_out={info['n_out']} > 3072"
    assert tokens.shape[1] == info["n_out"] <= 3072
    assert coords.shape[1] == info["n_out"] <= 3072


def _check_gqa_kv_cache_decode_parity_and_4x_shrink() -> float:
    """Verify full forward vs step-by-step GQA KV-cache decode parity (max_abs_diff < 1e-4) + 4x smaller KV cache."""
    torch.manual_seed(123)
    cfg = _tiny_cfg(n_heads=8, n_kv_heads=2, dropout=0.0)
    model = UniOMRv2Net(cfg)
    model.eval()

    B, T = 2, 12
    images = torch.rand(B, 3, 128, 128)
    tgt_ids = torch.randint(0, VOCAB_SIZE - 4, (B, T))
    tgt_ids[:, 0] = BOS_ID

    with torch.no_grad():
        mem, aux_enc, _ = model.encode_image(images, return_aux=False)
        # Aux heads must be skipped at inference.
        assert aux_enc["sys_count_pred"] is None
        assert aux_enc["staff_seg_logits"] is None

        full_logits, full_mtp, full_sys_y = model.decode_tokens(
            tgt_ids, mem, return_mtp=False, return_aux=False, use_cache=False
        )
        assert full_mtp == []
        assert full_sys_y is None

        # Step-by-step KV-cached decode
        past_kvs = None
        step_logits = []
        for t in range(T):
            logits_t, _, past_kvs = model.decode_tokens(
                tgt_ids[:, t : t + 1],
                memory=mem if t == 0 else None,
                return_mtp=False,
                return_aux=False,
                past_kvs=past_kvs,
                use_cache=True,
            )
            step_logits.append(logits_t)

        cached_logits = torch.cat(step_logits, dim=1)
        max_abs_diff = float((full_logits - cached_logits).abs().max().item())
        assert max_abs_diff < 1e-4, f"GQA KV-cache parity failed: max_abs_diff={max_abs_diff}"

        # Verify KV cache stores n_kv_heads=2 (4x smaller than n_heads=8) for both self & cross attn!
        assert past_kvs is not None and len(past_kvs) == cfg.n_dec_layers
        self_kv, cross_kv = past_kvs[0]
        assert self_kv[0].shape == (B, cfg.n_kv_heads, T, cfg.d_model // cfg.n_heads)
        assert cross_kv[0].shape == (B, cfg.n_kv_heads, mem.shape[1], cfg.d_model // cfg.n_heads)

        # Smoke-test greedy generate()
        gen_seq = model.generate(images, max_len=6)
        assert gen_seq.shape[0] == B and 2 <= gen_seq.shape[1] <= 7
        assert bool((gen_seq[:, 0] == BOS_ID).all())

    return max_abs_diff


def test_gqa_kv_cache_decode_parity_and_4x_shrink() -> None:
    _check_gqa_kv_cache_decode_parity_and_4x_shrink()


def _check_qat_int4_controller_v2_and_zero_schema_change() -> int:
    """Verify QATInt4ControllerV2 attaches cleanly, modifies outputs when enabled, and preserves state_dict."""
    torch.manual_seed(7)
    model = UniOMRv2Net(_tiny_cfg())
    model.eval()
    keys_before = list(model.state_dict().keys())

    qat = model.attach_qat_int4(block_size=32, enabled=False)
    assert qat.num_layers > 0
    assert list(model.state_dict().keys()) == keys_before

    img = torch.rand(1, 3, 96, 96)
    tgt = torch.tensor([[BOS_ID, 1, 2, 3]])
    with torch.no_grad():
        out_fp = model(img, tgt)["logits"]
        qat.enable()
        out_q = model(img, tgt)["logits"]
        qat.disable()
        out_fp2 = model(img, tgt)["logits"]

    assert (out_fp - out_fp2).abs().max().item() == 0.0
    assert (out_fp - out_q).abs().max().item() > 0.0

    # Verify gradients still flow through STE when QAT is enabled.
    model.train()
    qat.enable()
    model.zero_grad(set_to_none=True)
    loss = model(img, tgt)["logits"].sum()
    loss.backward()
    assert model.dec_layers[0].ffn.w_down.weight.grad is not None

    n_patched = qat.num_layers
    qat.detach()
    assert list(model.state_dict().keys()) == keys_before
    return n_patched


def test_qat_int4_controller_v2_and_zero_schema_change() -> None:
    _check_qat_int4_controller_v2_and_zero_schema_change()


def main() -> None:
    summaries = _check_presets_param_counts_and_byte_budgets()
    test_aspect_normalized_rope2d_resolution_invariance()
    test_forward_backward_portrait_landscape_and_mem_cap()
    max_diff = _check_gqa_kv_cache_decode_parity_and_4x_shrink()
    n_qat = _check_qat_int4_controller_v2_and_zero_schema_change()

    print("=== UniOMR v2 CPU Unit Tests PASSED ===")
    for name, s in summaries.items():
        print(
            f"[{name}] total={s['total_params']:,} | infer={s['infer_params']:,} | "
            f"backbone={s['backbone_params']:,} | enc={s['encoder_params']:,} | "
            f"dec={s['decoder_params']:,} | mtp={s['mtp_params']:,} | aux={s['aux_params']:,}"
        )
        print(
            f"  sizes: FP16={s['fp16_mb']:.2f} MB | pure_INT4={s['pure_int4_mb']:.2f} MB "
            f"(w/scales={s['pure_int4_with_scales_mb']:.2f} MB) | "
            f"Hybrid_FP16/INT4={s['hybrid_fp16_int4_mb']:.2f} MB "
            f"(w/scales={s['hybrid_fp16_int4_with_scales_mb']:.2f} MB)"
        )
    print(f"[GQA KV-cache parity] max_abs_diff={max_diff:.2e} (< 1e-4)")
    print(f"[QATInt4ControllerV2] patched_linear_layers={n_qat}")


if __name__ == "__main__":
    main()
