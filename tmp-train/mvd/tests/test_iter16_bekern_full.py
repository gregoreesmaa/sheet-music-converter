"""Fast CPU unit tests for Iter 16 bekern-full Pillar 3 integration (`uni_omr_v2.py` + `train_uni_v2.py`)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_MVD_DIR = Path(__file__).resolve().parent.parent
if str(_MVD_DIR) not in sys.path:
    sys.path.insert(0, str(_MVD_DIR))

import bekern_full as bf  # noqa: E402
import train_uni_v2 as tv2  # noqa: E402
import uni_omr_v2 as uv2  # noqa: E402


def _tiny_bekern_full_cfg(**overrides) -> uv2.UniOMRv2Config:
    params = dict(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_axial_layers=1,
        n_global_layers=1,
        n_dec_layers=2,
        ffn_mult=2.0,
        dec_ffn_mult=2.0,
        axial_ffn_mult=2.0,
        backbone_channels=(16, 32, 48, 64),
        backbone_blocks=(1, 1, 1, 1),
        backbone_expand=2.0,
        router_hidden=32,
        max_mem_tokens=256,
        mtp_k=4,
        n_register_tokens=8,
        use_yoco_cla=True,
        yoco_cross_banks=(0,),
        yoco_cross_kv_heads=2,
        yoco_self_kv_heads=2,
        use_resformer_v_res=True,
        use_diff_cross_attn=True,
        use_coordkey_ca=True,
        coord_dim=8,
        coord_mlp_hidden=32,
        coord_n_freqs=8,
        mode_gate_layers=2,
        mode_gate_rank=16,
        use_chained_mtp=True,
        use_quantvla_head_temp=True,
        use_bd3_repair_head=True,
        bd3_d_rep=32,
        bd3_n_heads=4,
        bd3_d_ff=64,
        bd3_n_layers=1,
        bd3_block_size=8,
        bekern_full=True,
        vocab_size=bf.FULL_VOCAB_SIZE,
        pad_id=bf.PAD_ID,
        bos_id=bf.BOS_ID,
        eos_id=bf.EOS_ID,
        sys_id=bf.SYS_ID,
        music_vocab_end=bf.MUSIC_VOCAB_END,
        use_text_aux_head=True,
        text_vocab_size=bf.TEXT_VOCAB_SIZE,
        text_aux_vocab_size=bf.TEXT_VOCAB_SIZE,
        text_aux_loss_weight=0.02,
        mode_gate_conf_thresh=0.75,
    )
    params.update(overrides)
    return uv2.UniOMRv2Config(**params)


def _tiny_v265_cfg(**overrides) -> uv2.UniOMRv2Config:
    params = dict(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_axial_layers=1,
        n_global_layers=1,
        n_dec_layers=2,
        ffn_mult=2.0,
        dec_ffn_mult=2.0,
        axial_ffn_mult=2.0,
        backbone_channels=(16, 32, 48, 64),
        backbone_blocks=(1, 1, 1, 1),
        backbone_expand=2.0,
        router_hidden=32,
        max_mem_tokens=256,
        mtp_k=4,
        n_register_tokens=4,
        use_yoco_cla=True,
        yoco_cross_banks=(0,),
        yoco_cross_kv_heads=2,
        yoco_self_kv_heads=2,
        use_resformer_v_res=True,
        use_diff_cross_attn=True,
        use_coordkey_ca=True,
        coord_dim=8,
        coord_mlp_hidden=32,
        coord_n_freqs=8,
        mode_gate_layers=2,
        mode_gate_rank=16,
        use_chained_mtp=True,
        text_aux_vocab_size=129,
        text_aux_loss_weight=0.02,
        use_quantvla_head_temp=True,
        use_bd3_repair_head=True,
        bd3_d_rep=32,
        bd3_n_heads=4,
        bd3_d_ff=64,
        bd3_n_layers=1,
        bd3_block_size=8,
        vocab_size=uv2.VOCAB_SIZE,
    )
    params.update(overrides)
    return uv2.UniOMRv2Config(**params)


# ---------------------------------------------------------------------------
# 1. Presets: iter16_pro200m, iter16_edge85m (< 55 MB INT4), and iter15 untouched
# ---------------------------------------------------------------------------

def test_iter15_preset_untouched():
    cfg15 = uv2.UniOMRv2Config.iter15_pro200m()
    assert cfg15.vocab_size == 265
    assert cfg15.bekern_full is False
    assert cfg15.n_register_tokens == 4
    assert cfg15.pad_id is None
    assert cfg15.bos_id is None
    assert cfg15.eos_id is None
    assert cfg15.sys_id is None
    assert cfg15.music_vocab_end is None
    assert cfg15.text_aux_vocab_size == 129
    assert cfg15.resolved_pad_id() == 261
    assert cfg15.resolved_bos_id() == 262
    assert cfg15.resolved_eos_id() == 263
    assert cfg15.resolved_sys_id() == 264


def test_iter16_pro200m_and_edge85m_presets_and_int4_budget():
    cfg_pro = uv2.UniOMRv2Config.iter16_pro200m()
    assert cfg_pro.d_model == 768
    assert cfg_pro.n_register_tokens == 8
    assert cfg_pro.bekern_full is True
    assert cfg_pro.vocab_size == bf.FULL_VOCAB_SIZE == 446
    assert cfg_pro.pad_id == 261
    assert cfg_pro.bos_id == 262
    assert cfg_pro.eos_id == 263
    assert cfg_pro.sys_id == 264
    assert cfg_pro.music_vocab_end == bf.MUSIC_VOCAB_END == 367
    assert cfg_pro.use_text_aux_head is True
    assert cfg_pro.text_vocab_size == bf.TEXT_VOCAB_SIZE == 259
    assert cfg_pro.mode_gate_conf_thresh == pytest.approx(0.75)

    cfg_edge = uv2.UniOMRv2Config.iter16_edge85m()
    assert cfg_edge.d_model == 512
    assert cfg_edge.n_register_tokens == 8
    assert cfg_edge.bekern_full is True
    assert cfg_edge.vocab_size == bf.FULL_VOCAB_SIZE == 446
    assert cfg_edge.text_vocab_size == bf.TEXT_VOCAB_SIZE == 259
    assert cfg_edge.mode_gate_conf_thresh == pytest.approx(0.75)

    # Construct full edge85m and verify Hybrid FP16/INT4 with scales < 55 MB
    model_edge = uv2.UniOMRv2Net(cfg_edge)
    assert model_edge.lm_head.out_features == 446
    assert model_edge.lm_head_text is not None
    assert model_edge.lm_head_text.out_features == 259
    assert model_edge.text_mode_gate is not None
    summary = model_edge.param_summary()
    assert summary["hybrid_fp16_int4_with_scales_mb"] < 55.0, (
        f"iter16_edge85m Hybrid INT4 with scales exceeded 55 MB: "
        f"{summary['hybrid_fp16_int4_with_scales_mb']:.2f} MB"
    )

    # Fast forward pass on tiny bekern-full (and verify pro200m construction via build_v2_config)
    assert tv2.build_v2_config("iter16_pro200m").vocab_size == 446
    assert tv2.build_v2_config("iter16_edge85m").vocab_size == 446

    tiny_model = uv2.UniOMRv2Net(_tiny_bekern_full_cfg())
    imgs = torch.randn(1, 3, 64, 64)
    tgt_in = torch.tensor([[bf.BOS_ID, bf.FULL_TOKEN_TO_ID["<schema:full>"], bf.FULL_TOKEN_TO_ID["quarter"], bf.FULL_TOKEN_TO_ID["C"]]])
    out = tiny_model(imgs, tgt_in, return_aux=True, return_mtp=True)
    assert out["logits"].shape == (1, 4, bf.FULL_VOCAB_SIZE)
    assert out["text_logits"].shape == (1, 4, bf.TEXT_VOCAB_SIZE)
    assert len(out["mtp_logits"]) == 3


# ---------------------------------------------------------------------------
# 2. Warm-Start Utility (v265 -> v446)
# ---------------------------------------------------------------------------

def test_warm_start_from_v265_checkpoint_exact_copy_and_fresh_tail(tmp_path):
    torch.manual_seed(42)
    src_model = uv2.UniOMRv2Net(_tiny_v265_cfg())
    dst_model = uv2.UniOMRv2Net(_tiny_bekern_full_cfg())

    ckpt_path = tmp_path / "iter15_v265.pt"
    torch.save({"model": src_model.state_dict(), "cfg": {"vocab_size": 265}}, ckpt_path)

    info = uv2.warm_start_from_v265_checkpoint(dst_model, ckpt_path, new_tok_std=0.02)
    assert "tok_emb.weight" in info["vocab_expanded"]
    assert "lm_head.weight" in info["vocab_expanded"]
    assert any("register_tokens" in s for s in info["register_expanded"])
    assert any("lm_head_text.weight" in s for s in info["shape_skipped"])

    # Exact slice copy [:265]
    assert torch.equal(dst_model.tok_emb.weight[:265], src_model.tok_emb.weight)
    assert torch.equal(dst_model.lm_head.weight[:265], src_model.lm_head.weight)
    # Register tokens [:, :4, :] copied verbatim from 4 -> 8
    assert torch.equal(dst_model.encoder.register_tokens[:, :4, :], src_model.encoder.register_tokens)

    # Fresh [265:] initialized with small std (~0.02) and non-zero
    tail_emb = dst_model.tok_emb.weight[265:]
    tail_lm = dst_model.lm_head.weight[265:]
    assert tail_emb.std().item() < 0.05 and tail_emb.std().item() > 0.005
    assert tail_lm.std().item() < 0.05 and tail_lm.std().item() > 0.005

    # Also verify train_uni_v2.load_v2_checkpoint delegates to warm_start_from_v265_checkpoint
    dst_model2 = uv2.UniOMRv2Net(_tiny_bekern_full_cfg())
    info2 = tv2.load_v2_checkpoint(dst_model2, ckpt_path, encoder_only=False)
    assert info2["old_vocab_size"] == 265
    assert info2["new_vocab_size"] == 446
    assert torch.equal(dst_model2.tok_emb.weight[:265], src_model.tok_emb.weight)
    assert torch.equal(dst_model2.lm_head.weight[:265], src_model.lm_head.weight)


# ---------------------------------------------------------------------------
# 3. 2-Stage Freeze/Unfreeze Helper (Mitigation #10)
# ---------------------------------------------------------------------------

def test_set_bekern_full_stage_1_freezes_core_and_zeroes_legacy_grads():
    torch.manual_seed(7)
    model = uv2.UniOMRv2Net(_tiny_bekern_full_cfg())
    st1 = uv2.set_bekern_full_stage(model, 1)
    assert st1["stage"] == 1
    assert st1["frozen_params"] > 0
    assert st1["n_hooks"] >= 2  # tok_emb.weight + lm_head.weight

    # Core backbone & encoder layers & decoder FFN must be frozen
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert not any(p.requires_grad for p in model.encoder.global_layers.parameters())
    assert not any(p.requires_grad for p in model.dec_layers[0].ffn.parameters())

    # New tokens, register_tokens, ModeGate, and lm_head_text must be trainable
    assert model.tok_emb.weight.requires_grad is True
    assert model.lm_head.weight.requires_grad is True
    assert model.encoder.register_tokens.requires_grad is True
    assert model.lm_head_text.weight.requires_grad is True
    assert model.text_mode_gate.weight.requires_grad is True
    assert model.dec_layers[0].cross_attn.mode_gate.weight.requires_grad is True

    # Run forward + backward with BOTH legacy (<265) and new (>=265) tokens
    imgs = torch.randn(1, 3, 64, 64)
    t = bf.FULL_TOKEN_TO_ID
    tgt_in = torch.tensor([[bf.BOS_ID, t["<schema:full>"], t["quarter"], t["marcato"]]])
    tgt_out = torch.tensor([[t["<schema:full>"], t["quarter"], t["marcato"], bf.EOS_ID]])
    out = model(imgs, tgt_in, return_aux=True, return_mtp=False)
    loss = out["logits"].sum() * 0.0 + torch.nn.functional.cross_entropy(
        out["logits"].reshape(-1, bf.FULL_VOCAB_SIZE), tgt_out.reshape(-1)
    ) + 0.01 * out["text_logits"].pow(2).mean()
    loss.backward()

    # Legacy [:265] grads MUST be identically zero; new [265:] grads MUST be non-zero!
    assert model.tok_emb.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    assert torch.all(model.tok_emb.weight.grad[:265] == 0.0)
    assert model.tok_emb.weight.grad[265:].abs().sum().item() > 0.0
    assert torch.all(model.lm_head.weight.grad[:265] == 0.0)
    assert model.lm_head.weight.grad[265:].abs().sum().item() > 0.0

    # Stage 2: unfreezes all and removes hooks so [:265] receives gradients again
    model.zero_grad(set_to_none=True)
    st2 = uv2.set_bekern_full_stage(model, 2)
    assert st2["stage"] == 2
    assert st2["frozen_params"] == 0
    assert st2["n_hooks"] == 0
    assert all(p.requires_grad for p in model.parameters())

    out2 = model(imgs, tgt_in, return_aux=True, return_mtp=False)
    loss2 = torch.nn.functional.cross_entropy(
        out2["logits"].reshape(-1, bf.FULL_VOCAB_SIZE), tgt_out.reshape(-1)
    )
    loss2.backward()
    assert model.tok_emb.weight.grad[:265].abs().sum().item() > 0.0
    assert model.lm_head.weight.grad[:265].abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# 4. 4-Bit layer_mask CE, BD3 is_music_body Mask, MTP-4 Mode-Switch & Gate
# ---------------------------------------------------------------------------

def test_compute_loss_4bit_layer_masks_zeroes_unannotated_grads():
    model = uv2.UniOMRv2Net(_tiny_bekern_full_cfg())
    t = bf.FULL_TOKEN_TO_ID
    targets = torch.tensor([[t["quarter"], t["C"], t["4"], bf.EOS_ID]], dtype=torch.long)
    layer_masks = torch.tensor([bf.SCHEMA_LAYER_BITS["v1"]], dtype=torch.long)  # L1 only

    logits = torch.zeros((1, 4, bf.FULL_VOCAB_SIZE), requires_grad=True)
    out = {"logits": logits, "mtp_logits": [], "router_info": {}}
    losses = model.compute_loss(out, targets, layer_masks=layer_masks)
    losses["ce_loss"].backward()

    grad = logits.grad[0]
    for unann in ("<part>", "dyn-f", "marcato", "slur-1-open", "<sys_lyr>", "@0"):
        assert torch.all(grad[:, t[unann]] == 0.0), f"non-zero grad on unannotated {unann}"
    assert grad[0, t["quarter"]].item() < 0.0


def test_bd3_is_music_body_mask_protects_schema_sys_and_text_spans():
    t = bf.FULL_TOKEN_TO_ID
    seq = torch.tensor([[
        bf.BOS_ID,
        t["<schema:full>"],
        t["quarter"],
        t["C"],
        t["4"],
        bf.SYS_ID,
        t["<sys_lyr>"],
        t["v=1"],
        t["@0"],
        t["lyr-beg"],
        t["</text_span>"],
        bf.EOS_ID,
    ]], dtype=torch.long)
    protected_idx = [0, 1, 5, 6, 7, 8, 9, 10, 11]

    torch.manual_seed(0)
    for _ in range(10):
        views = uv2.sample_bd3_complementary_views(
            seq,
            vocab_size=bf.FULL_VOCAB_SIZE,
            pad_id=bf.PAD_ID,
            bos_id=bf.BOS_ID,
            eos_id=bf.EOS_ID,
            n_views=2,
            music_vocab_end=bf.MUSIC_VOCAB_END,
            bekern_full_mode=True,
        )
        for corrupt_ids, mask_flags, supervise in views:
            for idx in protected_idx:
                assert corrupt_ids[0, idx].item() == seq[0, idx].item()
                assert mask_flags[0, idx].item() is False
                assert supervise[0, idx].item() is False


def test_mtp4_mode_switch_guard_and_text_mode_gate_threshold():
    torch.manual_seed(123)
    model = uv2.UniOMRv2Net(_tiny_bekern_full_cfg())
    model.eval()
    h = torch.randn(1, 1, 64)
    t = bf.FULL_TOKEN_TO_ID

    # 1. If current token is a mode-switch token -> drafting is completely disabled
    for sw_tok in bf.MODE_SWITCH_TOKENS:
        assert model.draft_mtp_tokens(h, t[sw_tok]) == []

    # 2. If a drafted token hits a mode-switch token -> drafting stops before crossing it
    with torch.no_grad():
        # Force lm_head to strongly prefer <sys_lyr>
        model.lm_head.weight.zero_()
        model.lm_head.weight[t["<sys_lyr>"]] = 10.0
    assert model.draft_mtp_tokens(h, t["quarter"]) == []

    # 3. If lm_head prefers a normal music token -> drafts all 3 MTP steps
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.weight[t["C"]] = 10.0
    drafted = model.draft_mtp_tokens(h, t["quarter"])
    assert len(drafted) == 3
    assert all(tok == t["C"] for tok in drafted)

    # 4. ModeGate confidence threshold (0.75): default bias=-2.0 -> p_text < 0.75 -> blocked
    emit_low, p_low = model.should_emit_text(h)
    assert p_low.item() < 0.75
    assert emit_low.item() is False

    # Force high confidence -> p_text >= 0.75 -> allowed
    with torch.no_grad():
        model.text_mode_gate.weight.zero_()
        model.text_mode_gate.bias.fill_(3.0)
    emit_hi, p_hi = model.should_emit_text(h)
    assert p_hi.item() >= 0.75
    assert emit_hi.item() is True


def test_cli_flags_iter16_presets_and_bekern_full_stage():
    parser = tv2.build_arg_parser()
    args1 = parser.parse_args(["--preset", "iter16_pro200m", "--bekern-full-stage", "1"])
    assert args1.preset == "iter16_pro200m"
    assert args1.bekern_full_stage == 1

    args2 = parser.parse_args(["--preset", "iter16_edge85m", "--bekern-full-stage", "2"])
    assert args2.preset == "iter16_edge85m"
    assert args2.bekern_full_stage == 2
