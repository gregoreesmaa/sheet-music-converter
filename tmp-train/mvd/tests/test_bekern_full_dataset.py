"""Fast CPU unit tests for Iter 16 bekern-full Dataset & Collate Bridge (Gap 2)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import pytest
import torch

_MVD_DIR = Path(__file__).resolve().parent.parent
if str(_MVD_DIR) not in sys.path:
    sys.path.insert(0, str(_MVD_DIR))

import bekern_full as bf  # noqa: E402
import uni_omr as uo  # noqa: E402
import uni_omr_v2 as uv2  # noqa: E402


def _make_dummy_page(path: Path, w: int = 64, h: int = 80, val: int = 240) -> Path:
    arr = np.full((h, w, 3), val, dtype=np.uint8)
    arr[20:24, 8:56, :] = 20
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def _tiny_iter16_edge85m_cfg(**overrides) -> uv2.UniOMRv2Config:
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
        text_aux_loss_weight=0.05,
        mode_gate_conf_thresh=0.75,
    )
    params.update(overrides)
    return uv2.UniOMRv2Config(**params)


def test_backward_compatibility_bekern_full_false_exact_legacy_behavior(tmp_path: Path):
    """When bekern_full=False (default), dataset & collate emit exact Iter 15 keys, shapes, and IDs."""
    img1 = _make_dummy_page(tmp_path / "p1.png", w=64, h=80)
    img2 = _make_dummy_page(tmp_path / "p2.png", w=64, h=64)
    bek1 = tmp_path / "p1-bekern.txt"
    bek2 = tmp_path / "p2-bekern.txt"
    bek1.write_text("clef-G-2 quarter C 4 bar", encoding="utf-8")
    bek2.write_text("clef-F-4 half G 3", encoding="utf-8")

    ds = uo.UniFullPageDataset(
        [
            (img1, [bek1], "printed", "v2a"),
            (img2, [bek2], "real_hw", "v1"),
        ],
        fullpage_max_dim=64,
        max_seq_len=128,
        augment=False,
    )
    assert ds.bekern_full is False
    s0 = ds[0]
    s1 = ds[1]
    expected_item_keys = {"image", "seq", "sample_id", "kind", "domain", "bekern_version"}
    assert set(s0.keys()) == expected_item_keys
    assert set(s1.keys()) == expected_item_keys
    assert s0["seq"][0].item() == uo.BOS_ID
    assert s0["seq"][-1].item() == uo.EOS_ID
    assert int(s0["seq"].max().item()) < uo.VOCAB_SIZE

    batch = uo.collate_uni_batch([s0, s1])
    expected_batch_keys = {
        "images",
        "seq",
        "sample_ids",
        "domains",
        "bekern_versions",
        "is_v1_schema",
        "orig_hw",
        "orig_seq_len",
        "kind",
    }
    assert set(batch.keys()) == expected_batch_keys
    assert "layer_masks" not in batch
    assert "text_byte_inputs" not in batch
    assert "text_byte_targets" not in batch
    # Shorter sequence padded with legacy PAD_ID (261)
    assert batch["seq"][1, -1].item() == uo.PAD_ID == 261
    assert batch["is_v1_schema"].tolist() == [False, True]


def test_bekern_full_dataset_and_collate_mixed_v1_v2a_full_and_utf8_spans(tmp_path: Path):
    """Mixed v1, v2a, and full (with Layer 3 UTF-8 text spans) tokenize, mask, and pad cleanly."""
    img_v1 = _make_dummy_page(tmp_path / "v1.png", w=64, h=80)
    img_v2a = _make_dummy_page(tmp_path / "v2a.png", w=64, h=80)
    img_full = _make_dummy_page(tmp_path / "full.png", w=64, h=80)
    img_pre = _make_dummy_page(tmp_path / "pre.png", w=64, h=80)

    bek_v1 = tmp_path / "v1-bekern.txt"
    bek_v2a = tmp_path / "v2a-bekern.txt"
    bek_full_s0 = tmp_path / "full-s0.txt"
    bek_full_s1 = tmp_path / "full-s1.txt"
    bek_pre = tmp_path / "pre-bekern.txt"

    bek_v1.write_text("clef-G-2 quarter C 4 bar", encoding="utf-8")
    bek_v2a.write_text("clef-G-2 dyn-f quarter C 4 stacc bar", encoding="utf-8")
    bek_full_s0.write_text(
        "<part> clef-G-2 <tempo> Allegro </text_span> marcato quarter C 4 bar",
        encoding="utf-8",
    )
    bek_full_s1.write_text(
        "clef-G-2 half D 4 <sys_lyr v=1> Für Elise </text_span> bar",
        encoding="utf-8",
    )
    # Already has <schema:full> prefix + explicit custom layer_mask override (e.g. L0+L1+L2 = 0b1110)
    bek_pre.write_text("<schema:full> <part> clef-G-2 quarter E 4", encoding="utf-8")

    pairs: List[Tuple] = [
        (img_v1, [bek_v1], "real_hw", "v1"),
        (img_v2a, [bek_v2a], "printed", "v2a"),
        (img_full, [bek_full_s0, bek_full_s1], "printed", "full"),
        (img_pre, [bek_pre], "printed", "full", 0b1110),
    ]
    ds = uo.UniFullPageDataset(
        pairs,
        fullpage_max_dim=64,
        max_seq_len=256,
        augment=False,
        bekern_full=True,
    )

    items = [ds[i] for i in range(4)]
    t = bf.FULL_TOKEN_TO_ID

    # 1. Schema prefixes prepended once (never duplicated) & BOS/EOS present
    assert items[0]["seq"][:2].tolist() == [bf.BOS_ID, t["<schema:v1>"]]
    assert items[0]["seq"][-1].item() == bf.EOS_ID
    assert items[1]["seq"][:2].tolist() == [bf.BOS_ID, t["<schema:v2a>"]]
    assert items[2]["seq"][:2].tolist() == [bf.BOS_ID, t["<schema:full>"]]
    assert bf.SYS_ID in items[2]["seq"].tolist()
    assert t["<tempo>"] in items[2]["seq"].tolist()
    assert t["</text_span>"] in items[2]["seq"].tolist()
    # Pre-prefixed row must NOT duplicate <schema:full>
    assert items[3]["seq"][:3].tolist() == [bf.BOS_ID, t["<schema:full>"], t["<part>"]]

    # 2. Layer masks match SCHEMA_LAYER_BITS & explicit override
    assert items[0]["layer_mask"] == bf.SCHEMA_LAYER_BITS["v1"] == 0b0100
    assert items[1]["layer_mask"] == bf.SCHEMA_LAYER_BITS["v2a"] == 0b0110
    assert items[2]["layer_mask"] == bf.SCHEMA_LAYER_BITS["full"] == 0b1111
    assert items[3]["layer_mask"] == 0b1110

    # 3. UTF-8 text spans packed into [TEXT_BOS]+bytes and bytes+[TEXT_EOS]
    assert items[0]["text_byte_in"].numel() == 0
    assert items[1]["text_byte_in"].numel() == 0
    span0 = bf.encode_text_span_utf8("Allegro", add_bos_eos=True)
    span1 = bf.encode_text_span_utf8("v=1 Für Elise", add_bos_eos=True)
    assert items[2]["text_byte_in"].tolist() == span0[:-1] + span1[:-1]
    assert items[2]["text_byte_tgt"].tolist() == span0[1:] + span1[1:]

    # 4. Collate pads music seq with bf.PAD_ID (261, NOT vocab_size - 4) and text with TEXT_PAD_ID (258)
    batch = uo.collate_uni_batch(items)
    assert batch["layer_masks"].tolist() == [0b0100, 0b0110, 0b1111, 0b1110]
    assert batch["seq"][0, -1].item() == bf.PAD_ID == 261
    assert bf.FULL_VOCAB_SIZE - 4 != bf.PAD_ID
    assert (batch["seq"] == (bf.FULL_VOCAB_SIZE - 4)).sum().item() == 0

    tbi = batch["text_byte_inputs"]
    tbt = batch["text_byte_targets"]
    assert isinstance(tbi, torch.Tensor) and isinstance(tbt, torch.Tensor)
    assert tbi.shape == (4, len(span0[:-1] + span1[:-1]))
    assert tbt.shape == tbi.shape
    assert torch.all(tbi[0] == bf.TEXT_PAD_ID)
    assert torch.all(tbi[1] == bf.TEXT_PAD_ID)
    assert torch.all(tbt[0] == bf.TEXT_PAD_ID)
    assert tbi[2].tolist() == span0[:-1] + span1[:-1]
    assert tbt[2].tolist() == span0[1:] + span1[1:]

    # 5. All-empty text spans batch emits None for text_byte_inputs/targets
    no_txt_batch = uo.collate_uni_batch([items[0], items[1]])
    assert no_txt_batch["layer_masks"].tolist() == [0b0100, 0b0110]
    assert no_txt_batch["text_byte_inputs"] is None
    assert no_txt_batch["text_byte_targets"] is None

    # 6. unpack_collated_batch preserves layer_mask and unpadded text spans
    unpacked = uo.unpack_collated_batch(batch, fallback_kind="fullpage")
    assert len(unpacked) == 4
    assert unpacked[2]["layer_mask"] == 0b1111
    assert unpacked[2]["text_byte_in"].tolist() == items[2]["text_byte_in"].tolist()
    assert unpacked[2]["text_byte_tgt"].tolist() == items[2]["text_byte_tgt"].tolist()
    assert unpacked[0]["text_byte_in"].numel() == 0


def test_end_to_end_bekern_full_dataset_collate_to_compute_loss_grads(tmp_path: Path):
    """End-to-end: UniFullPageDataset(bekern_full=True) -> collate -> UniOMRv2Net.compute_loss -> backward."""
    torch.manual_seed(77)
    img_v1 = _make_dummy_page(tmp_path / "e2e_v1.png", w=64, h=64)
    img_v2a = _make_dummy_page(tmp_path / "e2e_v2a.png", w=64, h=64)
    img_full = _make_dummy_page(tmp_path / "e2e_full.png", w=64, h=64)

    bek_v1 = tmp_path / "e2e_v1.txt"
    bek_v2a = tmp_path / "e2e_v2a.txt"
    bek_full = tmp_path / "e2e_full.txt"

    bek_v1.write_text("clef-G-2 quarter C 4 bar", encoding="utf-8")
    bek_v2a.write_text("clef-G-2 dyn-p quarter E 4 stacc bar", encoding="utf-8")
    bek_full.write_text(
        "<part> clef-G-2 <tempo> Andante cantabile </text_span> marcato quarter G 4 bar",
        encoding="utf-8",
    )

    ds = uo.UniFullPageDataset(
        [
            (img_v1, [bek_v1], "real_hw", "v1"),
            (img_v2a, [bek_v2a], "printed", "v2a"),
            (img_full, [bek_full], "printed", "full"),
        ],
        fullpage_max_dim=64,
        max_seq_len=128,
        augment=False,
        bekern_full=True,
    )
    batch = uo.collate_uni_batch([ds[0], ds[1], ds[2]])

    model = uv2.UniOMRv2Net(_tiny_iter16_edge85m_cfg())
    model.train()

    imgs = batch["images"]
    seq = batch["seq"]
    tgt_in = seq[:, :-1]
    tgt_out = seq[:, 1:]

    out = model(imgs, tgt_in, return_aux=True, return_mtp=True)
    losses = model.compute_loss(
        out,
        tgt_out,
        layer_masks=batch["layer_masks"],
        text_byte_inputs=batch["text_byte_inputs"],
        text_byte_targets=batch["text_byte_targets"],
        label_smoothing=0.05,
    )

    assert torch.isfinite(losses["loss"]), f"Non-finite total loss: {losses['loss']}"
    assert torch.isfinite(losses["ce_loss"]) and losses["ce_loss"].item() > 0.0
    assert torch.isfinite(losses["text_aux_loss"]) and losses["text_aux_loss"].item() > 0.0
    assert torch.isfinite(losses["text_gate_loss"]) and losses["text_gate_loss"].item() > 0.0

    losses["loss"].backward()

    for name, param in (
        ("lm_head.weight", model.lm_head.weight),
        ("text_mode_gate.weight", model.text_mode_gate.weight),
        ("tok_emb_text.weight", model.tok_emb_text.weight),
        ("lm_head_text.weight", model.lm_head_text.weight),
    ):
        assert param.grad is not None, f"{name} received None grad"
        assert torch.isfinite(param.grad).all(), f"{name} grad has non-finite values"
        assert param.grad.abs().sum().item() > 0.0, f"{name} received zero grad"
