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
    AnisotropicMusicBlock,
    AspectNormalizedRoPE1D,
    AspectNormalizedRoPE2D,
    BOS_ID,
    EOS_ID,
    GroupedQueryAttention,
    InkTokenRouterV2,
    PAD_ID,
    SYS_ID,
    StrokeHogHead,
    VOCAB_SIZE,
    UniOMRv2Config,
    UniOMRv2Net,
    compute_stroke_hog_loss,
    compute_stroke_hog_target,
    compute_sys_count_loss,
    compute_sys_y_loss,
)


def _tiny_cfg(**overrides) -> UniOMRv2Config:
    """Fast CPU-friendly config preserving every v2 architectural component."""
    params = dict(
        d_model=64,
        n_heads=8,
        n_kv_heads=2,
        n_axial_layers=2,
        n_global_layers=2,
        n_dec_layers=3,
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

    # DeepStack fusion must stay < 0.5M params for both presets.
    assert 0 < s_small["deepstack_params"] < 500_000, f"small deepstack={s_small['deepstack_params']}"
    assert 0 < s_edge["deepstack_params"] < 500_000, f"edge deepstack={s_edge['deepstack_params']}"

    # small_45m: ~41M total (trimmed axial_ffn_mult=2.0), < 25 MB pure INT4, < 55 MB hybrid.
    assert 38_000_000 <= s_small["total_params"] <= 46_000_000, f"small total={s_small['total_params']}"
    assert s_small["pure_int4_mb"] < 25.0, f"small pure_int4_mb={s_small['pure_int4_mb']:.2f}"
    assert s_small["pure_int4_with_scales_mb"] < 25.0, (
        f"small pure_int4_with_scales_mb={s_small['pure_int4_with_scales_mb']:.2f}"
    )
    assert s_small["hybrid_fp16_int4_with_scales_mb"] < 55.0

    # edge85m: d_model=512, ~78M total (trimmed axial_ffn_mult=2.0), < 55 MB Hybrid FP16/INT4-QAT.
    assert m_edge.cfg.d_model == 512
    assert 75_000_000 <= s_edge["total_params"] <= 86_000_000, f"edge total={s_edge['total_params']}"
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
    """1280px and 1536px grids must share identical rotary phases and use interleaved [y0, x0, y1, x1, ...]."""
    head_dim = 64
    rope = AspectNormalizedRoPE2D(head_dim=head_dim, base=10000.0, coord_scale=128.0)

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

    # Verify interleaved [y0, x0, y1, x1, ...] layout in emb_half (first head_dim // 2 channels):
    half = head_dim // 2
    y_only = torch.tensor([[[float(h1 - 1) * 0.37, 0.0]]], dtype=torch.float32)
    cos_y, sin_y = rope(y_only, grid_hw=(h1, w1), dtype=torch.float32)
    assert sin_y[0, 0, 0, 0:half:2].abs().max().item() > 1e-3, "Expected non-zero y phases on even indices"
    assert sin_y[0, 0, 0, 1:half:2].abs().max().item() == 0.0, "Expected zero x phases on odd indices"
    assert (cos_y[0, 0, 0, 1:half:2] - 1.0).abs().max().item() == 0.0

    x_only = torch.tensor([[[0.0, float(w1 - 1) * 0.43]]], dtype=torch.float32)
    cos_x, sin_x = rope(x_only, grid_hw=(h1, w1), dtype=torch.float32)
    assert sin_x[0, 0, 0, 1:half:2].abs().max().item() > 1e-3, "Expected non-zero x phases on odd indices"
    assert sin_x[0, 0, 0, 0:half:2].abs().max().item() == 0.0, "Expected zero y phases on even indices"
    assert (cos_x[0, 0, 0, 0:half:2] - 1.0).abs().max().item() == 0.0


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
        # Verify gradients flow through backbone, encoder, QK-Norm, DeepStack, decoder, MTP, and all 3 aux heads.
        assert model.backbone.stage1.down[0].weight.grad is not None
        assert model.backbone.p2_to_p3.proj.weight.grad is not None
        assert model.encoder.axial_layers[0].row_attn.qkv_proj.weight.grad is not None
        assert model.encoder.axial_layers[0].row_attn.q_norm.weight.grad is not None
        assert model.encoder.axial_layers[0].row_attn.k_norm.weight.grad is not None
        assert model.encoder.axial_layers[0].col_attn.q_norm.weight.grad is not None
        assert model.encoder.axial_layers[0].col_attn.k_norm.weight.grad is not None
        assert model.encoder.patch_merger.fc1.weight.grad is not None
        assert model.encoder.global_layers[0].self_attn.qkv_proj.weight.grad is not None
        assert model.encoder.global_layers[0].self_attn.q_norm.weight.grad is not None
        assert model.encoder.global_layers[0].self_attn.k_norm.weight.grad is not None
        assert model.deepstack_fusions[0].gate.grad is not None
        assert model.deepstack_fusions[0].proj[0].weight.grad is not None
        assert model.deepstack_fusions[1].gate.grad is not None
        assert model.deepstack_fusions[1].proj[0].weight.grad is not None
        assert model.dec_layers[0].self_attn.k_proj.weight.grad is not None
        assert model.dec_layers[0].self_attn.q_norm.weight.grad is not None
        assert model.dec_layers[0].self_attn.k_norm.weight.grad is not None
        assert model.dec_layers[0].cross_attn.q_norm.weight.grad is not None
        assert model.dec_layers[0].cross_attn.k_norm.weight.grad is not None
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
        mem_levels, aux_enc, _ = model.encode_image(images, return_aux=False)
        # Verify DeepStack mem_levels = (mem_post_merger, mem_global_mid, mem_final) share exact routed shape
        assert isinstance(mem_levels, tuple) and len(mem_levels) == 3
        mem_post_merger, mem_global_mid, mem_final = mem_levels
        assert mem_post_merger.shape == mem_global_mid.shape == mem_final.shape
        assert mem_final.shape[1] <= cfg.max_mem_tokens
        assert not torch.allclose(mem_post_merger, mem_final)
        assert not torch.allclose(mem_global_mid, mem_final)
        # Aux heads must be skipped at inference.
        assert aux_enc["sys_count_pred"] is None
        assert aux_enc["staff_seg_logits"] is None

        full_logits, full_mtp, full_sys_y = model.decode_tokens(
            tgt_ids, mem_levels, return_mtp=False, return_aux=False, use_cache=False
        )
        assert full_mtp == []
        assert full_sys_y is None

        # Verify DeepStack multi-level fusion actively modulates decoder outputs vs mem_final-only fallback.
        final_only_logits, _, _ = model.decode_tokens(
            tgt_ids, mem_final, return_mtp=False, return_aux=False, use_cache=False
        )
        assert float((full_logits - final_only_logits).abs().max().item()) > 1e-4

        # Step-by-step KV-cached decode with DeepStack active (0% extra KV cache!)
        past_kvs = None
        step_logits = []
        for t in range(T):
            logits_t, _, past_kvs = model.decode_tokens(
                tgt_ids[:, t : t + 1],
                memory=mem_levels if t == 0 else None,
                return_mtp=False,
                return_aux=False,
                past_kvs=past_kvs,
                use_cache=True,
            )
            step_logits.append(logits_t)

        cached_logits = torch.cat(step_logits, dim=1)
        max_abs_diff = float((full_logits - cached_logits).abs().max().item())
        assert max_abs_diff < 1e-4, f"GQA+DeepStack KV-cache parity failed: max_abs_diff={max_abs_diff}"

        # Verify KV cache stores n_kv_heads=2 (4x smaller than n_heads=8) and ONLY 1 KV cache per decoder layer!
        assert past_kvs is not None and len(past_kvs) == cfg.n_dec_layers
        for layer_idx in range(cfg.n_dec_layers):
            self_kv, cross_kv = past_kvs[layer_idx]
            assert self_kv[0].shape == (B, cfg.n_kv_heads, T, cfg.d_model // cfg.n_heads)
            assert cross_kv[0].shape == (B, cfg.n_kv_heads, mem_final.shape[1], cfg.d_model // cfg.n_heads)

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


def test_iter11_optimizations() -> None:
    """Verify Iter-11 compute/memory optimizations: 1/4-split, native GQA, branchless losses, precompute_cross_kv, RoPE1D cache."""
    torch.manual_seed(99)

    # 1. AnisotropicMusicBlock 1/4-channel split (including non-divisible-by-4 remainder on identity)
    blk = AnisotropicMusicBlock(channels=18, expand_ratio=2.0)
    assert blk.split_sizes == (4, 4, 4, 6)
    y_blk = blk(torch.randn(2, 18, 16, 16))
    assert y_blk.shape == (2, 18, 16, 16)

    # 2. Native PyTorch enable_gqa=True vs _expand_kv fallback parity
    gqa = GroupedQueryAttention(d_model=64, n_heads=8, n_kv_heads=2, use_native_gqa=True).eval()
    q_in = torch.randn(2, 10, 64)
    kv_in = torch.randn(2, 15, 64)
    with torch.no_grad():
        out_native = gqa(q_in, kv_in, is_causal=False)
        gqa.use_native_gqa = False
        out_expand = gqa(q_in, kv_in, is_causal=False)
    assert float((out_native - out_expand).abs().max().item()) < 1e-5

    # 3. Branchless aux losses return exact 0.0 when no valid entries exist
    pred_y = torch.rand(2, 8, requires_grad=True)
    tgt_y_empty = torch.full((2, 8), -1.0)
    loss_y_empty = compute_sys_y_loss(pred_y, tgt_y_empty)
    assert float(loss_y_empty.item()) == 0.0
    pred_cnt = torch.rand(2, requires_grad=True)
    tgt_cnt_empty = torch.full((2,), -1.0)
    loss_cnt_empty = compute_sys_count_loss(pred_cnt, tgt_cnt_empty)
    assert float(loss_cnt_empty.item()) == 0.0

    # 4. precompute_cross_kv parity + DeepStackFusion guard on cached steps + RoPE1D pow2 cache
    cfg = _tiny_cfg(n_heads=8, n_kv_heads=2, dropout=0.0)
    model = UniOMRv2Net(cfg).eval()
    images = torch.rand(1, 3, 128, 128)
    tgt_ids = torch.randint(0, VOCAB_SIZE - 4, (1, 10))
    tgt_ids[:, 0] = BOS_ID

    fusion_calls = [0]
    orig_f0 = model.deepstack_fusions[0].forward

    def _counted_f0(*args, **kwargs):
        fusion_calls[0] += 1
        return orig_f0(*args, **kwargs)

    model.deepstack_fusions[0].forward = _counted_f0  # type: ignore[method-assign]
    with torch.no_grad():
        mem_levels, _, _ = model.encode_image(images, return_aux=False)
        full_logits, _, _ = model.decode_tokens(tgt_ids, mem_levels, return_mtp=False, return_aux=False)
        fusion_calls[0] = 0

        past_kvs = model.precompute_cross_kv(mem_levels)
        assert fusion_calls[0] == 1, f"precompute_cross_kv should call fusion[0] once, got {fusion_calls[0]}"

        step_logits = []
        for t in range(tgt_ids.shape[1]):
            # Deliberately pass memory=mem_levels on every cached step to prove the guard skips DeepStackFusion!
            lt, _, past_kvs = model.decode_tokens(
                tgt_ids[:, t : t + 1],
                memory=mem_levels,
                return_mtp=False,
                return_aux=False,
                past_kvs=past_kvs,
                use_cache=True,
            )
            step_logits.append(lt)
        assert fusion_calls[0] == 1, f"DeepStackFusion must not re-run on cached steps, got {fusion_calls[0]}"
        precomp_logits = torch.cat(step_logits, dim=1)
        assert float((full_logits - precomp_logits).abs().max().item()) < 1e-4

        # RoPE1D power-of-2 cache pre-allocation (min 512)
        c1, _ = model.rope1d(10, device=images.device, dtype=images.dtype)
        assert c1.shape[2] == 10
        assert model.rope1d._cache[(images.device, images.dtype)][0] == 512
        c2, _ = model.rope1d(513, device=images.device, dtype=images.dtype)
        assert c2.shape[2] == 513
        assert model.rope1d._cache[(images.device, images.dtype)][0] == 1024

    # 5. InkTokenRouterV2 training-mode fixed k (zero CUDA-CPU sync) vs eval dynamic k
    router = InkTokenRouterV2(d_model=64, dynamic_routing=True, min_keep_ratio=0.15, max_keep_ratio=0.75)
    feat = torch.randn(2, 64, 8, 8)  # N=64
    img = torch.rand(2, 3, 128, 128)
    router.train()
    _, _, info_tr = router(feat, images=img)
    assert info_tr["k_kept"] == 48  # ceil(64 * 0.75) = 48
    router.eval()
    _, _, info_ev = router(feat, images=img)
    assert 10 <= info_ev["k_kept"] <= 48


def test_frontier_2025_2026_upgrades() -> None:
    """Verify iRoPE, RoPE-exempt register tokens, V2PE RoPE box jitter, and Sobel-HOG aux head."""
    torch.manual_seed(2026)

    # 1. Llama 4 iRoPE: layers where (i + 1) % irope_nope_every == 0 skip RoPE while keeping QK-Norm active.
    cfg_irope = _tiny_cfg(n_dec_layers=8, irope_nope_every=4)
    m_irope = UniOMRv2Net(cfg_irope).eval()
    expected_use_rope = [True, True, True, False, True, True, True, False]
    actual_use_rope = [layer.use_rope for layer in m_irope.dec_layers]
    assert actual_use_rope == expected_use_rope, f"iRoPE flags mismatch: {actual_use_rope}"
    assert [layer.self_attn.use_rope for layer in m_irope.dec_layers] == expected_use_rope

    # Verify NoPE layer (layer 3) ignores rope_cos/rope_sin while RoPE layer (layer 0) uses them.
    x_dec = torch.randn(1, 8, cfg_irope.d_model)
    mem_dec = torch.randn(1, 12, cfg_irope.d_model)
    cos_a, sin_a = m_irope.rope1d(8, device=x_dec.device, dtype=x_dec.dtype)
    cos_b = torch.ones_like(cos_a)
    sin_b = torch.zeros_like(sin_a)
    with torch.no_grad():
        nope_a = m_irope.dec_layers[3](x_dec, mem_dec, cos_a, sin_a)
        nope_b = m_irope.dec_layers[3](x_dec, mem_dec, cos_b, sin_b)
        rope_a = m_irope.dec_layers[0](x_dec, mem_dec, cos_a, sin_a)
        rope_b = m_irope.dec_layers[0](x_dec, mem_dec, cos_b, sin_b)
    assert float((nope_a - nope_b).abs().max().item()) == 0.0, "NoPE layer must be invariant to RoPE cos/sin"
    assert float((rope_a - rope_b).abs().max().item()) > 1e-4, "RoPE layer must depend on RoPE cos/sin"
    assert m_irope.dec_layers[3].self_attn.q_norm is not None
    assert m_irope.dec_layers[3].self_attn.k_norm is not None

    # irope_nope_every=0 disables NoPE (all layers use RoPE).
    m_all_rope = UniOMRv2Net(_tiny_cfg(n_dec_layers=4, irope_nope_every=0))
    assert all(layer.use_rope for layer in m_all_rope.dec_layers)

    # 2. ViT-5 / DINOv3 RoPE-exempt register tokens: prepended before global_layers, sliced cleanly after.
    cfg_reg = _tiny_cfg(n_register_tokens=4, n_global_layers=2, dropout=0.0)
    m_reg = UniOMRv2Net(cfg_reg).train()
    assert m_reg.encoder.register_tokens is not None
    assert m_reg.encoder.register_tokens.shape == (1, 4, cfg_reg.d_model)

    captured_rope = {}
    orig_g0 = m_reg.encoder.global_layers[0].forward

    def _capture_g0(x_in, rope_cos, rope_sin):
        captured_rope["seq_len"] = x_in.shape[1]
        captured_rope["reg_cos"] = rope_cos[:, :, :4, :].detach().clone()
        captured_rope["reg_sin"] = rope_sin[:, :, :4, :].detach().clone()
        return orig_g0(x_in, rope_cos, rope_sin)

    m_reg.encoder.global_layers[0].forward = _capture_g0  # type: ignore[method-assign]
    imgs = torch.rand(2, 3, 96, 96)
    mem_levels, aux_enc, rinfo = m_reg.encode_image(imgs, return_aux=True)
    n_out = rinfo["n_out"]
    assert captured_rope["seq_len"] == n_out + 4, "Register tokens must be prepended before global_layers"
    assert float((captured_rope["reg_cos"] - 1.0).abs().max().item()) == 0.0, "Register tokens must be RoPE-exempt (cos=1)"
    assert float(captured_rope["reg_sin"].abs().max().item()) == 0.0, "Register tokens must be RoPE-exempt (sin=0)"
    for lvl in mem_levels:
        assert lvl.shape == (2, n_out, cfg_reg.d_model), f"Register tokens must be stripped: got {lvl.shape}"

    # Verify gradients flow to register_tokens.
    m_reg.zero_grad(set_to_none=True)
    mem_levels[-1].sum().backward()
    assert m_reg.encoder.register_tokens.grad is not None
    assert float(m_reg.encoder.register_tokens.grad.abs().max().item()) > 0.0

    # n_register_tokens=0 disables register tokens cleanly.
    m_noreg = UniOMRv2Net(_tiny_cfg(n_register_tokens=0)).eval()
    assert m_noreg.encoder.register_tokens is None

    # 3. DINOv3 / InternVL 3 V2PE RoPE Box Jitter (train-only, deterministic in eval).
    rope1d_j = AspectNormalizedRoPE1D(head_dim=16, rope_jitter=0.08)
    rope1d_0 = AspectNormalizedRoPE1D(head_dim=16, rope_jitter=0.0)
    rope1d_j.eval()
    rope1d_0.eval()
    c_e1, s_e1 = rope1d_j(20, device=torch.device("cpu"), dtype=torch.float32)
    c_e2, s_e2 = rope1d_j(20, device=torch.device("cpu"), dtype=torch.float32)
    c_0, s_0 = rope1d_0(20, device=torch.device("cpu"), dtype=torch.float32)
    assert float((c_e1 - c_e2).abs().max().item()) == 0.0
    assert float((c_e1 - c_0).abs().max().item()) < 1e-6
    assert float((s_e1 - s_0).abs().max().item()) < 1e-6

    rope1d_j.train()
    c_t1, _ = rope1d_j(20, device=torch.device("cpu"), dtype=torch.float32)
    c_t2, _ = rope1d_j(20, device=torch.device("cpu"), dtype=torch.float32)
    assert float((c_t1 - c_0).abs().max().item()) > 1e-4, "RoPE1D jitter must perturb train() coordinates"
    assert float((c_t1 - c_t2).abs().max().item()) > 1e-4, "RoPE1D jitter must vary across train() batches"

    rope2d_j = AspectNormalizedRoPE2D(head_dim=16, rope_jitter=0.08)
    rope2d_0 = AspectNormalizedRoPE2D(head_dim=16, rope_jitter=0.0)
    coords = torch.tensor([[[1.0, 2.0], [5.0, 6.0], [9.0, 7.0]]], dtype=torch.float32)
    rope2d_j.eval()
    rope2d_0.eval()
    c2_e1, s2_e1 = rope2d_j(coords, (10, 10), dtype=torch.float32)
    c2_e2, s2_e2 = rope2d_j(coords, (10, 10), dtype=torch.float32)
    c2_0, s2_0 = rope2d_0(coords, (10, 10), dtype=torch.float32)
    assert float((c2_e1 - c2_e2).abs().max().item()) == 0.0
    assert float((c2_e1 - c2_0).abs().max().item()) < 1e-6
    assert float((s2_e1 - s2_0).abs().max().item()) < 1e-6

    rope2d_j.train()
    c2_t1, _ = rope2d_j(coords, (10, 10), dtype=torch.float32)
    c2_t2, _ = rope2d_j(coords, (10, 10), dtype=torch.float32)
    assert float((c2_t1 - c2_0).abs().max().item()) > 1e-4, "RoPE2D jitter must perturb train() coordinates"
    assert float((c2_t1 - c2_t2).abs().max().item()) > 1e-4, "RoPE2D jitter must vary across train() batches"

    # 4. MonkeyOCRv2 / MaskFeat Training-Only Sobel-HOG Stroke-Gradient Aux Head.
    m_hog = UniOMRv2Net(_tiny_cfg(use_stroke_hog_head=True)).train()
    assert isinstance(m_hog.stroke_hog_head, StrokeHogHead)
    tgt_ids = torch.tensor([[BOS_ID, 5, 6, 7], [BOS_ID, 8, 9, 10]])
    targets = torch.tensor([[5, 6, 7, EOS_ID], [8, 9, 10, EOS_ID]])
    out = m_hog(imgs, tgt_ids)
    assert out["stroke_hog_pred"] is not None
    assert out["stroke_hog_pred"].shape == (2, 8, 24, 24)

    hog_tgt = compute_stroke_hog_target(imgs, target_hw=(24, 24), n_bins=8)
    assert hog_tgt.shape == (2, 8, 24, 24)
    assert torch.isfinite(hog_tgt).all()

    hog_loss = compute_stroke_hog_loss(out["stroke_hog_pred"], imgs)
    assert torch.isfinite(hog_loss) and float(hog_loss.item()) > 0.0

    losses = m_hog.compute_loss(out, targets, images=imgs)
    assert torch.isfinite(losses["stroke_hog_loss"]) and float(losses["stroke_hog_loss"].item()) > 0.0
    m_hog.zero_grad(set_to_none=True)
    losses["loss"].backward()
    assert m_hog.stroke_hog_head.net[0].weight.grad is not None

    # strip_aux_heads() removes stroke_hog_head completely (0% inference cost).
    m_hog.strip_aux_heads()
    assert m_hog.stroke_hog_head is None
    assert m_hog.param_summary()["aux_params"] == 0
    m_hog.eval()
    with torch.no_grad():
        out_ev = m_hog(imgs, tgt_ids)
    assert out_ev["stroke_hog_pred"] is None


def main() -> None:
    summaries = _check_presets_param_counts_and_byte_budgets()
    test_aspect_normalized_rope2d_resolution_invariance()
    test_forward_backward_portrait_landscape_and_mem_cap()
    max_diff = _check_gqa_kv_cache_decode_parity_and_4x_shrink()
    n_qat = _check_qat_int4_controller_v2_and_zero_schema_change()
    test_iter11_optimizations()
    test_frontier_2025_2026_upgrades()

    print("=== UniOMR v2 CPU Unit Tests PASSED ===")
    for name, s in summaries.items():
        print(
            f"[{name}] total={s['total_params']:,} | infer={s['infer_params']:,} | "
            f"backbone={s['backbone_params']:,} | enc={s['encoder_params']:,} | "
            f"dec={s['decoder_params']:,} (deepstack={s['deepstack_params']:,}) | "
            f"mtp={s['mtp_params']:,} | aux={s['aux_params']:,}"
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
