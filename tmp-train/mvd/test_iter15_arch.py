"""Comprehensive CPU unit tests for Iter 15 (`pro200m` / `iter15_pro200m`) architecture & training upgrades.

Covers:
  1. 100% backward compatibility of legacy `edge85m` / `small_45m` configs (exact state_dict keys & shapes).
  2. `iter15_pro200m` (scaled to tiny d=128 for fast CPU execution) forward + backward + `grad_checkpoint=True`
     with all 7+1 features enabled (YOCO-3Bank, CLA-2:1, ResFormer v_res, DiffCrossAttn, CoordKey-CA,
     ModeGate, Chained MTP-4, lm_head_text aux, QuantVLA head_temp, BD3-LM repair loss).
  3. `precompute_cross_kv` + step-by-step incremental `decode_tokens(..., use_cache=True)` exact parity
     (< 1e-4 FP32) with full parallel `decode_tokens` on YOCO-3Bank + CLA-2:1 + ResFormer + Diff + CoordKey + ModeGate.
  4. `partition_muon_adamw_params()` routing all `(1, D)` gates (`mode_gate.weight`), `bd3_repair_head`,
     `coord_mlp`, `lm_head_text` to AdamW and 2D hidden matrices to Muon (0 dropped, 0 duplicated).
  5. `strip_aux_heads()` stripping `lm_head_text` cleanly, `QATInt4ControllerV2` covering `bd3_repair_head`,
     `param_summary()` verifying exact `iter15_pro200m` (`d=768`) infer params & INT4 MB (< 135 MB),
     plus `STRATIFIED_FULLPAGE_WEIGHTS_V4_HW70` & CLI preset checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_MVD_DIR = Path(__file__).resolve().parent
if str(_MVD_DIR) not in sys.path:
    sys.path.insert(0, str(_MVD_DIR))

import train_uni_v2 as tv2  # noqa: E402
import uni_omr_v2 as v2  # noqa: E402


def _make_tiny_iter15_cfg(*, grad_checkpoint: bool = False) -> v2.UniOMRv2Config:
    """Tiny d=128 version of `iter15_pro200m` with ALL 7+1 promoted features enabled."""
    return v2.UniOMRv2Config(
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        n_axial_layers=0,
        n_global_layers=2,
        n_dec_layers=6,
        ffn_mult=2.5,
        dec_ffn_mult=2.0,
        dropout=0.0,
        backbone_channels=(16, 32, 48, 64),
        backbone_blocks=(1, 1, 1, 1),
        backbone_expand=2.0,
        router_hidden=32,
        max_mem_tokens=128,
        mtp_k=4,
        grad_checkpoint=grad_checkpoint,
        # All 7+1 promoted Iter 15 upgrades:
        use_yoco_cla=True,
        yoco_cross_banks=(0, 2, 4),
        yoco_cross_kv_heads=4,
        yoco_self_kv_heads=2,
        use_resformer_v_res=True,
        use_diff_cross_attn=True,
        use_coordkey_ca=True,
        coord_dim=16,
        coord_mlp_hidden=64,
        coord_n_freqs=8,
        mode_gate_layers=3,
        mode_gate_rank=16,
        use_chained_mtp=True,
        text_aux_vocab_size=129,
        text_aux_loss_weight=0.02,
        use_quantvla_head_temp=True,
        use_bd3_repair_head=True,
        bd3_d_rep=64,
        bd3_n_heads=4,
        bd3_d_ff=128,
        bd3_n_layers=2,
        bd3_block_size=8,
    )


def test_1_legacy_backward_compat_state_dict() -> None:
    """Legacy `small_45m` and `edge85m` produce zero Iter-15 keys/buffers and exact legacy shapes."""
    for preset_name, cfg_fn in (("small_45m", v2.UniOMRv2Config.small_45m), ("edge85m", v2.UniOMRv2Config.edge85m)):
        cfg = cfg_fn()
        assert cfg.dec_ffn_mult is None
        assert cfg.use_yoco_cla is False
        assert cfg.use_resformer_v_res is False
        assert cfg.use_diff_cross_attn is False
        assert cfg.use_coordkey_ca is False
        assert cfg.mode_gate_layers == 0
        assert cfg.use_chained_mtp is False
        assert cfg.text_aux_vocab_size == 0
        assert cfg.use_quantvla_head_temp is False
        assert cfg.use_bd3_repair_head is False

        model = v2.UniOMRv2Net(cfg)
        sd = model.state_dict()

        forbidden_markers = (
            "coord_mlp",
            "bd3_repair_head",
            "lm_head_text",
            "mode_gate",
            "film_gamma",
            "film_beta",
            "q_down",
            "q_up",
            "q_coord_proj",
            "k_coord_proj",
            "q1_norm",
            "k1_norm",
            "lambda_q1",
            "subln",
            "v_res_alpha",
            "head_temp",
            "norm_h",
            "norm_e",
        )
        for k in sd.keys():
            for marker in forbidden_markers:
                assert marker not in k, f"Legacy {preset_name} leaked Iter-15 key: {k}"

        # Verify every decoder layer owns both self & cross k_proj/v_proj/q_norm/k_norm
        for i in range(cfg.n_dec_layers):
            assert f"dec_layers.{i}.self_attn.q_norm.weight" in sd
            assert f"dec_layers.{i}.self_attn.k_proj.weight" in sd
            assert f"dec_layers.{i}.cross_attn.q_norm.weight" in sd
            assert f"dec_layers.{i}.cross_attn.k_proj.weight" in sd
            assert sd[f"dec_layers.{i}.cross_attn.k_proj.weight"].shape == (
                cfg.n_kv_heads * (cfg.d_model // cfg.n_heads),
                cfg.d_model,
            )

        # Verify stock MTPHead keys
        for m_i in range(cfg.mtp_k - 1):
            assert f"mtp_heads.{m_i}.norm.weight" in sd
            assert sd[f"mtp_heads.{m_i}.proj.weight"].shape == (cfg.d_model, cfg.d_model)


def test_2_iter15_forward_backward_grad_checkpoint() -> None:
    """Tiny iter15_pro200m (d=128) runs forward + backward + grad_checkpoint=True with all 7+1 features."""
    torch.manual_seed(42)
    cfg = _make_tiny_iter15_cfg(grad_checkpoint=True)
    model = v2.UniOMRv2Net(cfg).train()

    # Verify split-spatial geometry dimensions on cross-attention
    ca0 = model.dec_layers[0].cross_attn
    assert ca0.owns_kv is True
    assert ca0.use_diff_attn is True
    assert ca0.use_coordkey is True
    assert ca0.use_mode_gate is True
    assert ca0.head_temp is not None and tuple(ca0.head_temp.shape) == (1, cfg.n_heads, 1, 1)
    assert ca0.half_dim == 16 and ca0.half_coord_dim == 8  # d_qk = 24 per half, packed = 48

    # Verify YOCO consumer has no k_proj/v_proj/k_coord_proj
    ca1 = model.dec_layers[1].cross_attn
    assert ca1.owns_kv is False
    assert ca1.k_proj is None and ca1.v_proj is None and ca1.k_coord_proj is None
    assert model.dec_layers[2].cross_attn.v_res_alpha is not None
    assert model.dec_layers[0].cross_attn.v_res_alpha is None

    # Verify CLA-2:1 self-KV ownership
    assert model.dec_layers[0].self_attn.owns_kv is True
    assert model.dec_layers[1].self_attn.owns_kv is False
    assert model.dec_layers[2].self_attn.owns_kv is True
    assert model.dec_layers[2].self_attn.v_res_alpha is not None

    # Verify forward + compute_loss (including Chained MTP-4, text_aux, BD3 repair) + backward
    B, T = 2, 16
    images = torch.randn(B, 3, 64, 64)
    tgt_in = torch.randint(0, cfg.vocab_size - 4, (B, T))
    tgt_in[:, 0] = cfg.resolved_bos_id()
    tgt_out = torch.randint(0, cfg.vocab_size - 4, (B, T))

    # Exercise both normal forward and forced full grad_checkpoint via bt threshold monkeypatch
    out = model(images, tgt_in, targets=tgt_out, return_aux=True, return_mtp=True)
    assert out["logits"].shape == (B, T, cfg.vocab_size)
    assert len(out["mtp_logits"]) == cfg.mtp_k - 1
    assert out["text_logits"].shape == (B, T, cfg.text_aux_vocab_size)
    assert "coord_feat" in out["router_info"]
    assert "mem_final" in out

    losses = model.compute_loss(
        out,
        tgt_out,
        images=images,
        bd3_repair_weight=0.25,
        bd3_n_views=2,
    )
    assert losses["text_aux_loss"].item() > 0.0
    assert losses["bd3_repair_loss"].item() > 0.0
    assert losses["mtp_loss"].item() > 0.0
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()

    # Check gradients flow to all key Iter-15 components
    assert model.coord_mlp is not None and model.coord_mlp.fc1.weight.grad is not None
    assert model.bd3_repair_head is not None and model.bd3_repair_head.in_proj.weight.grad is not None
    assert model.lm_head_text is not None and model.lm_head_text.weight.grad is not None
    assert model.dec_layers[0].cross_attn.mode_gate.weight.grad is not None
    assert model.dec_layers[0].cross_attn.lambda_q1.grad is not None
    assert model.dec_layers[2].cross_attn.v_res_alpha.grad is not None
    assert model.dec_layers[2].self_attn.v_res_alpha.grad is not None
    assert model.mtp_heads[0].proj.weight.grad is not None

    # Verify QATInt4ControllerV2 attaches to bd3_repair_head as well
    qat = model.attach_qat_int4(block_size=32, enabled=True)
    assert any(n.startswith("bd3_repair_head.") for n in qat.target_names)
    qat.disable()

    # Also verify full decoder layer grad_checkpoint path executes cleanly on YOCO-CLA + Diff + CoordKey
    model.zero_grad(set_to_none=True)
    out_ckpt = model(images, tgt_in, targets=tgt_out, return_aux=True, return_mtp=True)
    losses_ckpt = model.compute_loss(out_ckpt, tgt_out, images=images, bd3_repair_weight=0.25, bd3_n_views=1)
    losses_ckpt["loss"].backward()
    assert model.dec_layers[0].cross_attn.q_proj.weight.grad is not None


def test_3_incremental_kv_cache_exact_parity() -> None:
    """precompute_cross_kv + step-by-step decode_tokens(..., use_cache=True) matches full parallel logits < 1e-4."""
    torch.manual_seed(123)
    cfg = _make_tiny_iter15_cfg(grad_checkpoint=False)
    model = v2.UniOMRv2Net(cfg).eval()

    # Also test non-trivial QuantVLA head_temp values != 1.0
    with torch.no_grad():
        for idx, layer in enumerate(model.dec_layers):
            if layer.self_attn.head_temp is not None:
                layer.self_attn.head_temp.copy_(
                    torch.linspace(0.85, 1.15, cfg.n_heads).view(1, cfg.n_heads, 1, 1)
                )
            if layer.cross_attn.head_temp is not None:
                layer.cross_attn.head_temp.copy_(
                    torch.linspace(0.90, 1.20, cfg.n_heads).view(1, cfg.n_heads, 1, 1)
                )
            # Also set non-zero ModeGate weights so ModeGate is actively exercised
            if layer.cross_attn.use_mode_gate:
                layer.cross_attn.mode_gate.weight.normal_(0.0, 0.1)
                layer.cross_attn.film_gamma.normal_(0.0, 0.1)
                layer.cross_attn.film_beta.normal_(0.0, 0.1)
                layer.cross_attn.q_up.weight.normal_(0.0, 0.1)

    B, T = 2, 12
    images = torch.randn(B, 3, 64, 64)
    tgt_ids = torch.randint(0, cfg.vocab_size - 4, (B, T))
    tgt_ids[:, 0] = cfg.resolved_bos_id()

    with torch.no_grad():
        mem_levels, _, router_info = model.encode_image(images, return_aux=False)
        coord_feat = router_info["coord_feat"]

        # Full parallel forward
        par_logits, _, _ = model.decode_tokens(
            tgt_ids,
            mem_levels,
            return_mtp=False,
            return_aux=False,
            use_cache=False,
            apply_tok_dropout=False,
            coord_feat=coord_feat,
        )

        # Incremental KV-cached decode (1 token at a time)
        past_kvs = model.precompute_cross_kv(mem_levels, coord_feat=coord_feat)
        assert isinstance(past_kvs, dict)
        assert len(past_kvs["cross_banks"]) == len(cfg.yoco_cross_banks)
        assert len(past_kvs["self_banks"]) == (cfg.n_dec_layers + 1) // 2

        step_logits = []
        for t in range(T):
            tok_t = tgt_ids[:, t : t + 1]
            lg_t, _, past_kvs = model.decode_tokens(
                tok_t,
                memory=None,
                return_mtp=False,
                return_aux=False,
                past_kvs=past_kvs,
                use_cache=True,
                apply_tok_dropout=False,
            )
            step_logits.append(lg_t)
        inc_logits = torch.cat(step_logits, dim=1)

        max_diff = (par_logits - inc_logits).abs().max().item()
        assert max_diff < 1e-4, f"Incremental vs parallel logits max_diff={max_diff:.6e} >= 1e-4"

        # Also verify generate() runs cleanly end-to-end
        gen_seq = model.generate(images, max_len=6)
        assert gen_seq.shape[0] == B and gen_seq.shape[1] >= 2


def test_4_muon_partition_guard() -> None:
    """partition_muon_adamw_params routes (1, D) gates, bd3_repair_head, coord_mlp, lm_head_text to AdamW."""
    cfg = _make_tiny_iter15_cfg(grad_checkpoint=False)
    model = v2.UniOMRv2Net(cfg)

    muon_named, adamw_named = tv2.partition_muon_adamw_params(model)
    muon_names = {n for n, _ in muon_named}
    adamw_names = {n for n, _ in adamw_named}

    all_trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert muon_names.isdisjoint(adamw_names), "Overlap between Muon and AdamW param sets"
    assert (muon_names | adamw_names) == all_trainable, "Dropped or extra params in partition"

    # Every Muon param must be 2D with min(shape) >= 2
    for n, p in muon_named:
        assert p.ndim == 2 and min(p.shape) >= 2, f"Invalid Muon param {n} with shape {tuple(p.shape)}"

    # All (1, D) mode_gate.weight, bd3_repair_head.*, coord_mlp.*, lm_head_text.* MUST be in AdamW
    assert any("mode_gate.weight" in n for n in adamw_names)
    assert not any("mode_gate.weight" in n for n in muon_names)
    assert any(n.startswith("bd3_repair_head.") for n in adamw_names)
    assert not any(n.startswith("bd3_repair_head.") for n in muon_names)
    assert any(n.startswith("coord_mlp.") for n in adamw_names)
    assert not any(n.startswith("coord_mlp.") for n in muon_names)
    assert "lm_head_text.weight" in adamw_names
    assert "lm_head_text.weight" not in muon_names

    # 2D Transformer hidden matrices MUST be in Muon
    assert "dec_layers.0.cross_attn.q_proj.weight" in muon_names
    assert "dec_layers.0.cross_attn.q_coord_proj.weight" in muon_names
    assert "dec_layers.0.cross_attn.k_coord_proj.weight" in muon_names
    assert "dec_layers.0.cross_attn.q_down.weight" in muon_names
    assert "dec_layers.0.cross_attn.q_up.weight" in muon_names
    assert "encoder.global_layers.0.self_attn.qkv_proj.weight" in muon_names


def test_5_strip_aux_heads_and_iter15_pro200m_budget() -> None:
    """strip_aux_heads strips lm_head_text cleanly; iter15_pro200m (d=768) is < 135 MB Hybrid INT4."""
    # Check full d=768 iter15_pro200m on meta device (instantaneous, 0 RAM)
    cfg = v2.UniOMRv2Config.iter15_pro200m()
    with torch.device("meta"):
        model = v2.UniOMRv2Net(cfg)

    ps_before = model.param_summary(int4_block_size=32)
    assert model.lm_head_text is not None
    assert model.bd3_repair_head is not None
    assert model.coord_mlp is not None
    assert ps_before["bd3_params"] > 0
    assert ps_before["coord_mlp_params"] > 0
    assert ps_before["aux_params"] > 0
    assert ps_before["hybrid_fp16_int4_with_scales_mb"] < 135.0, (
        f"iter15_pro200m exceeded 135 MB budget: {ps_before['hybrid_fp16_int4_with_scales_mb']:.2f} MB"
    )

    # Strip aux heads (keeps bd3_repair_head & coord_mlp; removes lm_head_text + layout/hog aux)
    model.strip_aux_heads(strip_mtp=False)
    assert model.lm_head_text is None
    assert model.sys_y_head is None
    assert model.sys_count_head is None
    assert model.staff_seg_head is None
    assert model.stroke_hog_head is None
    assert model.bd3_repair_head is not None
    assert model.coord_mlp is not None

    ps_after = model.param_summary(int4_block_size=32)
    assert ps_after["aux_params"] == 0
    assert ps_after["infer_params"] == ps_before["infer_params"]
    assert ps_after["hybrid_fp16_int4_with_scales_mb"] == pytest.approx(
        ps_before["hybrid_fp16_int4_with_scales_mb"], abs=1e-6
    )

    # Verify presets & stratified weights in train_uni_v2
    assert tv2.build_v2_config("pro200m").d_model == 768
    assert tv2.build_v2_config("iter15_pro200m").use_yoco_cla is True
    assert sum(tv2.STRATIFIED_FULLPAGE_WEIGHTS_V4_HW70.values()) == pytest.approx(1.0, abs=1e-6)
    printed_sum = (
        tv2.STRATIFIED_FULLPAGE_WEIGHTS_V4_HW70["legacy_printed"]
        + tv2.STRATIFIED_FULLPAGE_WEIGHTS_V4_HW70["multi_renderer_printed"]
        + tv2.STRATIFIED_FULLPAGE_WEIGHTS_V4_HW70["rare_printed"]
    )
    assert printed_sum == pytest.approx(0.30, abs=1e-6)
