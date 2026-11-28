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
    assert cfg_pro.vocab_size == bf.FULL_VOCAB_SIZE
    assert cfg_pro.pad_id == 261
    assert cfg_pro.bos_id == 262
    assert cfg_pro.eos_id == 263
    assert cfg_pro.sys_id == 264
    assert cfg_pro.music_vocab_end == bf.MUSIC_VOCAB_END
    assert cfg_pro.use_text_aux_head is True
    assert cfg_pro.text_vocab_size == bf.TEXT_VOCAB_SIZE == 259
    assert cfg_pro.mode_gate_conf_thresh == pytest.approx(0.75)

    cfg_edge = uv2.UniOMRv2Config.iter16_edge85m()
    assert cfg_edge.d_model == 512
    assert cfg_edge.n_register_tokens == 8
    assert cfg_edge.bekern_full is True
    assert cfg_edge.vocab_size == bf.FULL_VOCAB_SIZE
    assert cfg_edge.text_vocab_size == bf.TEXT_VOCAB_SIZE == 259
    assert cfg_edge.mode_gate_conf_thresh == pytest.approx(0.75)

    # Construct full edge85m and verify Hybrid FP16/INT4 with scales < 55 MB
    model_edge = uv2.UniOMRv2Net(cfg_edge)
    assert model_edge.lm_head.out_features == bf.FULL_VOCAB_SIZE
    assert model_edge.lm_head_text is not None
    assert model_edge.lm_head_text.out_features == 259
    assert model_edge.tok_emb_text is not None
    assert model_edge.tok_emb_text.num_embeddings == 259
    assert model_edge.text_mode_gate is not None
    summary = model_edge.param_summary()
    assert summary["hybrid_fp16_int4_with_scales_mb"] < 55.0, (
        f"iter16_edge85m Hybrid INT4 with scales exceeded 55 MB: "
        f"{summary['hybrid_fp16_int4_with_scales_mb']:.2f} MB"
    )

    # Fast forward pass on tiny bekern-full (and verify pro200m construction via build_v2_config)
    assert tv2.build_v2_config("iter16_pro200m").vocab_size == bf.FULL_VOCAB_SIZE
    assert tv2.build_v2_config("iter16_edge85m").vocab_size == bf.FULL_VOCAB_SIZE

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
    assert info2["new_vocab_size"] == bf.FULL_VOCAB_SIZE
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


# ---------------------------------------------------------------------------
# 5. Round 1 Critical Fix Regressions (AdamW wd invariance, text gate/UTF-8,
#    generate() music<->text, label_smoothing, eval & ONNX vocab parity)
# ---------------------------------------------------------------------------

def test_stage1_adamw_weight_decay_bit_exact_legacy_invariance():
    torch.manual_seed(11)
    src_model = uv2.UniOMRv2Net(_tiny_v265_cfg())
    model = uv2.UniOMRv2Net(_tiny_bekern_full_cfg())
    uv2.warm_start_from_v265_checkpoint(model, {"model": src_model.state_dict()})

    uv2.set_bekern_full_stage(model, 1)
    emb_snap = model.tok_emb.weight[:265].detach().clone()
    lm_snap = model.lm_head.weight[:265].detach().clone()
    lm_tail_snap = model.lm_head.weight[265:].detach().clone()
    reg_snap = model.encoder.register_tokens[:, :4, :].detach().clone()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.1)
    opt.register_step_post_hook(lambda *_: uv2.restore_bekern_full_stage1_frozen_slices(model))

    imgs = torch.randn(1, 3, 64, 64)
    t = bf.FULL_TOKEN_TO_ID
    tgt_in = torch.tensor([[bf.BOS_ID, t["<schema:full>"], t["quarter"], t["marcato"]]])
    tgt_out = torch.tensor([[t["<schema:full>"], t["quarter"], t["marcato"], bf.EOS_ID]])
    out = model(imgs, tgt_in, return_aux=True, return_mtp=False)
    losses = model.compute_loss(out, tgt_out, layer_masks=torch.tensor([bf.SCHEMA_LAYER_BITS["full"]]))
    losses["loss"].backward()
    opt.step()

    # Legacy [:265] and warm-started register_tokens[:, :4, :] MUST be bit-exact invariant!
    assert torch.equal(model.tok_emb.weight[:265], emb_snap)
    assert torch.equal(model.lm_head.weight[:265], lm_snap)
    assert torch.equal(model.encoder.register_tokens[:, :4, :], reg_snap)
    # New rows [265:] MUST have updated
    assert not torch.equal(model.lm_head.weight[265:], lm_tail_snap)


def test_text_mode_gate_supervision_and_utf8_text_head_ce():
    torch.manual_seed(22)
    model = uv2.UniOMRv2Net(_tiny_bekern_full_cfg())
    imgs = torch.randn(1, 3, 64, 64)
    t = bf.FULL_TOKEN_TO_ID
    tgt_in = torch.tensor([[bf.BOS_ID, t["<schema:full>"], t["quarter"], t["<tempo>"]]])
    tgt_out = torch.tensor([[t["<schema:full>"], t["quarter"], t["<tempo>"], t["</text_span>"]]])

    # 1. With text_byte_targets=None, bekern_full must NOT run legacy build_text_aux_targets
    out = model(imgs, tgt_in, return_aux=True, return_mtp=True)
    assert "text_gate_logits" in out
    assert out["text_gate_logits"].shape == (1, 4)
    losses = model.compute_loss(
        out,
        tgt_out,
        layer_masks=torch.tensor([bf.SCHEMA_LAYER_BITS["full"]]),
        label_smoothing=0.1,
    )
    assert torch.isfinite(losses["loss"])
    assert losses["text_aux_loss"].item() == 0.0
    assert losses["text_gate_loss"].item() > 0.0
    losses["loss"].backward()
    assert model.text_mode_gate.weight.grad is not None
    assert model.text_mode_gate.weight.grad.abs().sum().item() > 0.0

    # 2. With UTF-8 text_byte_inputs & text_byte_targets, lm_head_text + tok_emb_text get UTF-8 CE grads
    model.zero_grad(set_to_none=True)
    utf8_seq = bf.encode_text_span_utf8("Allegro", add_bos_eos=True)
    tb_in = torch.tensor([utf8_seq[:-1]], dtype=torch.long)
    tb_tgt = torch.tensor([utf8_seq[1:]], dtype=torch.long)
    out2 = model(imgs, tgt_in, return_aux=True, return_mtp=False)
    losses2 = model.compute_loss(
        out2,
        tgt_out,
        layer_masks=torch.tensor([bf.SCHEMA_LAYER_BITS["full"]]),
        text_byte_inputs=tb_in,
        text_byte_targets=tb_tgt,
    )
    assert losses2["text_aux_loss"].item() > 0.0
    losses2["loss"].backward()
    assert model.lm_head_text.weight.grad is not None
    assert model.lm_head_text.weight.grad.abs().sum().item() > 0.0
    assert model.tok_emb_text.weight.grad is not None
    assert model.tok_emb_text.weight.grad.abs().sum().item() > 0.0


def test_generate_music_to_text_utf8_span_decode():
    torch.manual_seed(33)
    model = uv2.UniOMRv2Net(_tiny_bekern_full_cfg())
    model.eval()
    t = bf.FULL_TOKEN_TO_ID
    imgs = torch.randn(1, 3, 64, 64)

    # Probe exact music & context-conditioned text hidden directions so bias-free linear heads pick <tempo> and 'A'
    with torch.no_grad():
        mem, _, rinfo = model.encode_image(imgs, return_aux=False)
        coord_feat = rinfo.get("coord_feat")
        init_kvs = model.precompute_cross_kv(mem, coord_feat=coord_feat)
        extra_m = {}
        _, _, _, _, kvs_after_bos = model._decode_tokens_impl(
            torch.tensor([[bf.BOS_ID]]), None, return_mtp=False, return_aux=False,
            past_kvs=init_kvs, use_cache=True, out_extra=extra_m,
        )
        h_m = extra_m["h_last"][0, 0]
        _, _, _, _, kvs_after_open = model._decode_tokens_impl(
            torch.tensor([[t["<tempo>"]]]), None, return_mtp=False, return_aux=False,
            past_kvs=kvs_after_bos, use_cache=True, use_text_emb=False,
        )
        extra_t = {}
        model._decode_tokens_impl(
            torch.tensor([[bf.TEXT_BOS_ID]]), None, return_mtp=False, return_aux=False,
            past_kvs=kvs_after_open, use_cache=True, use_text_emb=True, out_extra=extra_t,
        )
        h_t = extra_t["h_last"][0, 0]

        model.text_mode_gate.weight.zero_()
        model.text_mode_gate.bias.fill_(5.0)
        model.lm_head.weight.zero_()
        model.lm_head.weight[t["<tempo>"]] = h_m
        model.lm_head_text.weight.zero_()
        model.lm_head_text.weight[ord("A")] = h_t
        # Ensure all text steps align with h_t by making tok_emb_text('A') == tok_emb_text(TEXT_BOS)
        model.tok_emb_text.weight[ord("A")].copy_(model.tok_emb_text.weight[bf.TEXT_BOS_ID])

    seq, spans = model.generate(
        imgs, max_len=3, return_text_spans=True, max_text_span_len=3, use_mtp_draft=True
    )
    assert seq[0, 1].item() == t["<tempo>"]
    assert seq[0, 2].item() == t["</text_span>"]
    assert len(spans) == 1 and len(spans[0]) >= 1
    assert bf.decode_text_span_utf8(spans[0][0]) == "AAA"


def test_warm_start_prefixes_ema_and_zero_match_guard():
    src = uv2.UniOMRv2Net(_tiny_v265_cfg())
    dst = uv2.UniOMRv2Net(_tiny_bekern_full_cfg())
    prefixed = {f"_orig_mod.{k}": v for k, v in src.state_dict().items()}
    prefixed["bogus_extra_key"] = torch.zeros(2)
    info = uv2.warm_start_from_v265_checkpoint(dst, {"ema_state_dict": prefixed})
    assert info["loaded_count"] > 0
    assert "bogus_extra_key" in info["unexpected_keys"]

    with pytest.raises(RuntimeError):
        uv2.warm_start_from_v265_checkpoint(dst, {"model": {"completely_unrelated": torch.zeros(1)}})


def test_eval_and_onnx_vocab_spec_for_bekern_full():
    import eval_uni_suite31 as ev31
    import export_uni_v2_onnx as exp_v2

    pad_id, bos_id, eos_id, sys_id, tok_to_id, id_to_tok = ev31._vocab_spec(bf.FULL_VOCAB_SIZE)
    assert (pad_id, bos_id, eos_id, sys_id) == (261, 262, 263, 264)
    assert tok_to_id["<schema:full>"] >= 265
    assert id_to_tok[tok_to_id["marcato"]] == "marcato"

    cfg_full = uv2.UniOMRv2Config.iter16_edge85m()
    vjson = exp_v2.build_vocab_json_v2(cfg_full)
    assert vjson["vocab_size"] == bf.FULL_VOCAB_SIZE
    assert len(vjson["vocab"]) == bf.FULL_VOCAB_SIZE
    assert (vjson["pad_id"], vjson["bos_id"], vjson["eos_id"], vjson["sys_id"]) == (261, 262, 263, 264)
    assert vjson["vocab"][261:265] == ["<pad>", "<bos>", "<eos>", "<sys>"]
    assert "<schema:full>" in vjson["vocab"]


def test_train_v2_bekern_full_helpers_and_guards(tmp_path):
    # 1. _model_vocab_list returns FULL_VOCAB for bekern_full and data.VOCAB for iter15
    assert len(tv2._model_vocab_list(uv2.UniOMRv2Config.iter16_edge85m())) == bf.FULL_VOCAB_SIZE
    assert len(tv2._model_vocab_list(uv2.UniOMRv2Config.iter15_pro200m())) == 261

    # 2. build_ce_token_weights covers >= 265 tokens for bekern-full
    w = tv2.build_ce_token_weights(bf.FULL_VOCAB_SIZE, state_weight=1.5, expr_weight=2.0)
    assert w[bf.FULL_TOKEN_TO_ID["clef-T-4"]].item() == pytest.approx(1.5)
    assert w[bf.FULL_TOKEN_TO_ID["hairpin-1-cresc"]].item() == pytest.approx(2.0)

    # 3. load_v2_checkpoint forbids mismatched non-265 vocab into bekern-full
    bad_ckpt = tmp_path / "bad_v237.pt"
    torch.save({"cfg": {"vocab_size": 237}, "model": {"tok_emb.weight": torch.zeros(237, 64)}}, bad_ckpt)
    dst = uv2.UniOMRv2Net(_tiny_bekern_full_cfg())
    with pytest.raises(ValueError):
        tv2.load_v2_checkpoint(dst, bad_ckpt, encoder_only=False)


# ---------------------------------------------------------------------------
# 6. Round 2 Regression Tests (R2-3 & R2-5: BD3 NaN, all-TEXT_PAD NaN,
#    context-conditioned text span, DeepStack mem_levels parity, v2a default,
#    resolved_*_id, Stage-1 register_tokens resume)
# ---------------------------------------------------------------------------

def test_bd3_repair_loss_finite_with_partial_layer_masks_and_unannotated_tokens():
    """R2-3: compute_bd3_repair_loss must remain finite when layer_masks suppresses target tokens."""
    torch.manual_seed(101)
    model = uv2.UniOMRv2Net(_tiny_bekern_full_cfg())
    imgs = torch.randn(1, 3, 64, 64)
    t = bf.FULL_TOKEN_TO_ID
    # Include unannotated L0 (<part>) and L2 (dyn-f, marcato) tokens under v1 (L1-only) mask
    tgt_out = torch.tensor(
        [[t["<part>"], t["quarter"], t["C"], t["4"], t["dyn-f"], t["marcato"], t["half"], bf.EOS_ID]],
        dtype=torch.long,
    )
    mem, _, _ = model.encode_image(imgs, return_aux=False)
    layer_masks = torch.tensor([bf.SCHEMA_LAYER_BITS["v1"]], dtype=torch.long)
    loss = uv2.compute_bd3_repair_loss(model, mem[-1], tgt_out, n_views=2, layer_masks=layer_masks)
    assert torch.isfinite(loss), f"BD3 repair loss was non-finite: {loss}"
    loss.backward()


def test_compute_loss_all_text_pad_targets_finite():
    """R2-3: all-TEXT_PAD_ID text_byte_targets must return finite 0 loss (never NaN) in both branches."""
    torch.manual_seed(102)
    model = uv2.UniOMRv2Net(_tiny_bekern_full_cfg())
    imgs = torch.randn(1, 3, 64, 64)
    t = bf.FULL_TOKEN_TO_ID
    tgt_in = torch.tensor([[bf.BOS_ID, t["<schema:full>"], t["quarter"], t["C"]]])
    tgt_out = torch.tensor([[t["<schema:full>"], t["quarter"], t["C"], bf.EOS_ID]])
    out = model(imgs, tgt_in, return_aux=True, return_mtp=False)

    # Branch 1: byte_logits via text_byte_inputs
    tb_in = torch.full((1, 4), bf.TEXT_PAD_ID, dtype=torch.long)
    tb_tgt = torch.full((1, 4), bf.TEXT_PAD_ID, dtype=torch.long)
    losses1 = model.compute_loss(
        out,
        tgt_out,
        layer_masks=torch.tensor([bf.SCHEMA_LAYER_BITS["full"]]),
        text_byte_inputs=tb_in,
        text_byte_targets=tb_tgt,
    )
    assert torch.isfinite(losses1["text_aux_loss"]) and losses1["text_aux_loss"].item() == 0.0
    assert torch.isfinite(losses1["loss"])

    # Branch 2: text_logits fallback (text_byte_inputs=None)
    losses2 = model.compute_loss(
        out,
        tgt_out,
        layer_masks=torch.tensor([bf.SCHEMA_LAYER_BITS["full"]]),
        text_byte_inputs=None,
        text_byte_targets=tb_tgt,
    )
    assert torch.isfinite(losses2["text_aux_loss"]) and losses2["text_aux_loss"].item() == 0.0
    assert torch.isfinite(losses2["loss"])


def test_decode_utf8_text_span_conditions_on_music_past_kvs_without_mutation_and_deepstack_parity():
    """R2-5: _decode_utf8_text_span attends to live music past_kvs without mutating it, & mem_levels parity."""
    torch.manual_seed(103)
    model = uv2.UniOMRv2Net(_tiny_bekern_full_cfg())
    model.eval()
    t = bf.FULL_TOKEN_TO_ID
    imgs = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        mem, _, rinfo = model.encode_image(imgs, return_aux=False)
        coord_feat = rinfo.get("coord_feat")

        # DeepStack mem_levels exported in forward()
        tgt_in = torch.tensor([[bf.BOS_ID, t["<schema:full>"], t["quarter"], t["C"]]])
        out = model(imgs, tgt_in, return_aux=True, return_mtp=False)
        assert "mem_levels" in out and isinstance(out["mem_levels"], tuple) and len(out["mem_levels"]) == 3

        # Build two distinct music past_kvs contexts on the SAME image
        kvs_a = model.precompute_cross_kv(mem, coord_feat=coord_feat)
        _, _, _, _, kvs_a = model._decode_tokens_impl(
            torch.tensor([[bf.BOS_ID, t["quarter"], t["C"], t["4"]]]),
            None, return_mtp=False, return_aux=False, past_kvs=kvs_a, use_cache=True,
        )
        kvs_b = model.precompute_cross_kv(mem, coord_feat=coord_feat)
        _, _, _, _, kvs_b = model._decode_tokens_impl(
            torch.tensor([[bf.BOS_ID, t["half"], t["G"], t["5"], t["bar"]]]),
            None, return_mtp=False, return_aux=False, past_kvs=kvs_b, use_cache=True,
        )

        # Snapshot kvs_a self-bank length & values to verify zero mutation
        snap_len = kvs_a["self_banks"][0][0].shape[2]
        snap_k = kvs_a["self_banks"][0][0].clone()

        # Run 1-step text logits conditioned on kvs_a+<tempo> vs kvs_b+<harm>
        _, _, _, _, sk_a = model._decode_tokens_impl(
            torch.tensor([[t["<tempo>"]]]), None, return_mtp=False, return_aux=False,
            past_kvs=model._slice_past_kvs_batch(kvs_a, 0), use_cache=True, use_text_emb=False,
        )
        _, _, _, logits_a, _ = model._decode_tokens_impl(
            torch.tensor([[bf.TEXT_BOS_ID]]), None, return_mtp=False, return_aux=False,
            past_kvs=sk_a, use_cache=True, use_text_emb=True,
        )
        _, _, _, _, sk_b = model._decode_tokens_impl(
            torch.tensor([[t["<harm>"]]]), None, return_mtp=False, return_aux=False,
            past_kvs=model._slice_past_kvs_batch(kvs_b, 0), use_cache=True, use_text_emb=False,
        )
        _, _, _, logits_b, _ = model._decode_tokens_impl(
            torch.tensor([[bf.TEXT_BOS_ID]]), None, return_mtp=False, return_aux=False,
            past_kvs=sk_b, use_cache=True, use_text_emb=True,
        )
        assert not torch.allclose(logits_a, logits_b, atol=1e-5), (
            "Text span logits did not condition on distinct music past_kvs contexts!"
        )

        # Verify _decode_utf8_text_span does NOT mutate the caller's kvs_a
        _ = model._decode_utf8_text_span(
            mem, coord_feat, max_len=4,
            past_kvs_b=model._slice_past_kvs_batch(kvs_a, 0),
            open_tok_id=t["<tempo>"],
        )
        assert kvs_a["self_banks"][0][0].shape[2] == snap_len
        assert torch.equal(kvs_a["self_banks"][0][0], snap_k)


def test_forward_and_loss_v2_defaults_unprefixed_v2a_to_0b0110_and_config_resolved_ids(tmp_path):
    """R2-5: unprefixed v2a rows default to 0b0110 (not 0b1111); resolved_*_id works with None; Stage-1 resume."""
    # 1. resolved_*_id returns bekern_full IDs when bekern_full=True and explicit IDs are None
    cfg_none = _tiny_bekern_full_cfg(pad_id=None, bos_id=None, eos_id=None, sys_id=None, music_vocab_end=None)
    assert cfg_none.resolved_pad_id() == bf.PAD_ID
    assert cfg_none.resolved_bos_id() == bf.BOS_ID
    assert cfg_none.resolved_eos_id() == bf.EOS_ID
    assert cfg_none.resolved_sys_id() == bf.SYS_ID
    assert cfg_none.resolved_music_vocab_end() == bf.MUSIC_VOCAB_END

    # 2. forward_and_loss_v2 defaults unprefixed rows to v2a (0b0110), NOT full (0b1111)!
    torch.manual_seed(104)
    model = uv2.UniOMRv2Net(_tiny_bekern_full_cfg())
    t = bf.FULL_TOKEN_TO_ID
    imgs = torch.randn(2, 3, 64, 64)
    # Row 0: unprefixed legacy v2a (starts with clef-G-2); Row 1: <schema:full>
    seq = torch.tensor(
        [
            [bf.BOS_ID, t["clef-G-2"], t["quarter"], t["C"], bf.EOS_ID],
            [bf.BOS_ID, t["<schema:full>"], t["<part>"], t["quarter"], bf.EOS_ID],
        ],
        dtype=torch.long,
    )
    captured_lm = {}
    orig_compute_loss = model.compute_loss

    def _spy_compute_loss(*args, **kwargs):
        captured_lm["layer_masks"] = kwargs.get("layer_masks")
        return orig_compute_loss(*args, **kwargs)

    model.compute_loss = _spy_compute_loss
    losses, _ = tv2.forward_and_loss_v2(model, imgs, seq, label_smoothing=0.05)
    assert torch.isfinite(losses["loss"])
    assert captured_lm["layer_masks"] is not None
    assert captured_lm["layer_masks"].tolist() == [
        bf.SCHEMA_LAYER_BITS["v2a"],
        bf.SCHEMA_LAYER_BITS["full"],
    ]

    # 3. prepare_v2_latest_payloads persists warm_start_old_registers for Stage-1 --resume
    src_model = uv2.UniOMRv2Net(_tiny_v265_cfg())
    dst_model = uv2.UniOMRv2Net(_tiny_bekern_full_cfg())
    uv2.warm_start_from_v265_checkpoint(dst_model, {"model": src_model.state_dict()})
    _, payload, _, _ = tv2.prepare_v2_latest_payloads(
        out_dir=tmp_path, model=dst_model, cfg=dst_model.cfg, step=5, include_rng_state=False
    )
    assert payload["warm_start_old_registers"] == 4
