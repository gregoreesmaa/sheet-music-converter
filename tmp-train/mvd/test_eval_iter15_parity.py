#!/usr/bin/env python3
"""CPU parity & regression tests for Iter-15 evaluation, NorthStar runner, and ONNX guardrails.

Tests a tiny ``iter15_pro200m`` model (YOCO-CLA + ResFormer V-res + Diff Cross-Attn +
CoordKey-CA + ModeGate + Chained MTP-k + QuantVLA head_temp) on CPU through ALL 4
custom decoders in ``eval_uni_suite31.py``, plus Round-1 review fixes across
``eval_uni_suite31.py``, ``eval_northstar_runner.py``, and ``export_uni_v2_onnx.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import eval_northstar_runner  # noqa: E402
import eval_uni_suite31 as ev  # noqa: E402
import export_uni_v2_onnx  # noqa: E402
import grammar  # noqa: E402
import uni_omr  # noqa: E402
import uni_omr_v2  # noqa: E402


def _build_tiny_iter15_model(seed: int = 1234) -> Tuple[uni_omr_v2.UniOMRv2Net, uni_omr_v2.UniOMRv2Config]:
    """Create a tiny CPU-fast Iter-15 model with every Iter-15 feature enabled."""
    torch.manual_seed(seed)
    cfg = uni_omr_v2.UniOMRv2Config.iter15_pro200m(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_axial_layers=0,
        n_global_layers=1,
        n_dec_layers=4,
        ffn_mult=2.0,
        dec_ffn_mult=2.0,
        axial_ffn_mult=2.0,
        dropout=0.0,
        backbone_channels=(16, 24, 32, 48),
        backbone_blocks=(1, 1, 1, 1),
        backbone_expand=2.0,
        router_hidden=16,
        max_mem_tokens=64,
        n_bg_summary=4,
        mtp_k=3,
        yoco_cross_banks=(0, 2),
        yoco_cross_kv_heads=4,
        yoco_self_kv_heads=2,
        coord_dim=8,
        coord_mlp_hidden=32,
        coord_n_freqs=4,
        mode_gate_layers=2,
        mode_gate_rank=8,
        text_aux_vocab_size=0,
        text_aux_loss_weight=0.0,
        use_sys_y_aux=False,
        use_sys_count_aux=False,
        use_staff_seg_aux=False,
        use_stroke_hog_head=False,
        use_bd3_repair_head=False,
    )
    model = uni_omr_v2.UniOMRv2Net(cfg).eval()
    return model, cfg


def _assert_valid_token_seq(seq: torch.Tensor, batch_size: int, vocab_size: int, bos_id: int) -> None:
    assert seq.ndim == 2
    assert seq.shape[0] == batch_size
    assert seq.shape[1] >= 2
    assert seq.dtype == torch.long
    assert bool((seq[:, 0] == bos_id).all())
    assert int(seq.min().item()) >= 0
    assert int(seq.max().item()) < vocab_size


@pytest.fixture(scope="module")
def tiny_iter15() -> Tuple[uni_omr_v2.UniOMRv2Net, uni_omr_v2.UniOMRv2Config]:
    return _build_tiny_iter15_model()


def test_iter15_generate_constrained(tiny_iter15) -> None:
    model, cfg = tiny_iter15
    torch.manual_seed(11)
    img = torch.rand(1, 3, 64, 64)
    seq, rinfo = ev.generate_constrained(
        model, img, max_len=12, strict=False, return_info=True
    )
    _assert_valid_token_seq(seq, 1, int(cfg.vocab_size), cfg.resolved_bos_id())
    assert "coord_feat" in rinfo and rinfo["coord_feat"] is not None


def test_iter15_generate_mtp_spec_and_measure_beam(tiny_iter15) -> None:
    model, cfg = tiny_iter15
    torch.manual_seed(22)
    img = torch.rand(1, 3, 64, 64)
    seq, stats = ev.generate_mtp_spec(
        model,
        img,
        max_len=16,
        use_grammar=True,
        strict_grammar=False,
        return_stats=True,
        measure_beam_size=2,
        max_bar_toks=8,
    )
    _assert_valid_token_seq(seq, 1, int(cfg.vocab_size), cfg.resolved_bos_id())
    assert stats["mtp_k"] == 3
    assert stats["verify_passes"] >= 1

    # Explicitly exercise YOCO-CLA dict KV cache truncation, extraction, reordering,
    # and _beam_repair_measure rewind path directly.
    with torch.no_grad():
        mem, _, rinfo = model.encode_image(img, return_aux=False)
        coord_feat = rinfo["coord_feat"]
        init_kvs = model.precompute_cross_kv(mem, coord_feat=coord_feat)
        assert isinstance(init_kvs, dict)
        assert ev._past_kv_len(init_kvs) == 0

        bos_id = cfg.resolved_bos_id()
        _, _, kv1 = model.decode_tokens(
            torch.tensor([[bos_id, 1, 2]], dtype=torch.long),
            memory=None,
            return_mtp=False,
            past_kvs=init_kvs,
            use_cache=True,
        )
        assert isinstance(kv1, dict)
        assert ev._past_kv_len(kv1) == 3

        kv_trunc = ev._truncate_past_kvs(kv1, 1)
        assert isinstance(kv_trunc, dict)
        assert ev._past_kv_len(kv_trunc) == 1

        base_cross = ev._extract_cross_kvs(kv1, expand_beam=2)
        assert len(base_cross) == len(cfg.yoco_cross_banks)
        assert base_cross[0][0].shape[0] == 2

        _, _, _, _, tok_to_id, _ = ev._vocab_spec(int(cfg.vocab_size))
        st = grammar.PrefixState(strict=False, vocab_map=tok_to_id)
        best_toks, best_state, kv0, _ = ev._beam_repair_measure(
            model=model,
            kv_prefix=kv_trunc,
            base_cross_kvs=base_cross,
            trigger_id=1,
            bar_start_state=st,
            beam_size=2,
            max_bar_toks=6,
            vocab_size=int(cfg.vocab_size),
            use_grammar=True,
            strict_grammar=False,
            device=torch.device("cpu"),
        )
        assert isinstance(kv0, dict)
        assert ev._past_kv_len(kv0) == 2
        if best_toks is not None:
            assert best_state is not None
            assert all(0 <= t < int(cfg.vocab_size) for t in best_toks)


def test_iter15_generate_beam(tiny_iter15) -> None:
    model, cfg = tiny_iter15
    torch.manual_seed(33)
    # Test B=2 to verify coord_feat[b:b+1] per-item slicing.
    img = torch.rand(2, 3, 64, 64)
    seq, rinfo = ev.generate_beam(
        model,
        img,
        max_len=10,
        beam_size=2,
        length_penalty=0.6,
        use_grammar=True,
        strict_grammar=False,
        return_info=True,
    )
    _assert_valid_token_seq(seq, 2, int(cfg.vocab_size), cfg.resolved_bos_id())
    assert "coord_feat" in rinfo and rinfo["coord_feat"].shape[0] == 2


def test_iter15_generate_conf_gated_ilms(tiny_iter15) -> None:
    model, cfg = tiny_iter15
    torch.manual_seed(44)
    img = torch.rand(1, 3, 64, 64)
    blank = torch.ones(1, 3, 64, 64) * 0.95
    with torch.no_grad():
        mem_blank, _, blank_rinfo = model.encode_image(blank, return_aux=False)
    mem_blank_cached = (mem_blank, blank_rinfo.get("coord_feat"))

    # High conf_thr=0.9999 guarantees the lazy blank-LM branch executes on random weights.
    seq, stats = ev.generate_conf_gated_ilms(
        model,
        img,
        mem_blank=mem_blank_cached,
        max_len=10,
        alpha=0.40,
        conf_thr=0.9999,
        beta=0.10,
        use_grammar=True,
        strict_grammar=False,
        return_stats=True,
    )
    _assert_valid_token_seq(seq, 1, int(cfg.vocab_size), cfg.resolved_bos_id())
    assert stats["n_ref_calls"] >= 1
    assert stats["n_ilms_applied"] >= 1


def test_allowed_ids_blocks_eos_inside_text_span() -> None:
    vsz = uni_omr.VOCAB_SIZE
    _, _, eos_id, sys_id, tok_to_id, _ = ev._vocab_spec(vsz)
    st = grammar.PrefixState(strict=False, vocab_map=tok_to_id)
    for tok in ("clef-G-2", "time-4-4", "quarter", "C", "4", "bar"):
        st.step(tok)
    allowed_outside = ev._allowed_ids_for_state(st, vsz)
    assert eos_id in allowed_outside

    st.in_text_span = True
    allowed_inside = ev._allowed_ids_for_state(st, vsz)
    assert eos_id not in allowed_inside
    assert sys_id not in allowed_inside


def test_sys_reset_preserves_tsig_in_trim_and_fsa_beat() -> None:
    # System 1 sets time-3-4 (3 beats). System 2 omits time-3-4 and has a 3-beat bar.
    toks = [
        "clef-G-2", "time-3-4", "quarter", "C", "4", "quarter", "D", "4", "quarter", "E", "4", "bar",
        uni_omr.SYS_TOKEN,
        "clef-G-2", "quarter", "F", "4", "quarter", "G", "4", "quarter", "A", "4", "bar",
    ]
    metrics = ev.compute_fsa_beat_metrics(toks, hit_eos=True, bekern_decode_ok=True)
    assert metrics["n_bars"] == 2
    assert metrics["beat_conservation_rate"] == 1.0
    assert metrics["mean_meter_err"] == 0.0

    # Also verify trim_truncated_tokens handles multi-system without dropping system 2.
    trimmed = ev.trim_truncated_tokens(toks + ["quarter", "B"])
    assert trimmed == toks


def test_load_uni_checkpoint_yoco_cross_banks_tuple_and_no_cudnn_flag(tmp_path: Path, tiny_iter15) -> None:
    model, cfg = tiny_iter15
    cfg_dict = dict(cfg.__dict__)
    cfg_dict["yoco_cross_banks"] = [0, 2]  # Simulate JSON-deserialized list
    ckpt_path = tmp_path / "tiny_iter15.pt"
    torch.save({"arch": "uni_omr_v2", "config": cfg_dict, "model": model.state_dict()}, ckpt_path)

    loaded_model, loaded_cfg, meta = ev.load_uni_checkpoint(ckpt_path, device=torch.device("cpu"))
    assert meta["arch"] == "uni_omr_v2"
    assert isinstance(loaded_cfg.yoco_cross_banks, tuple)
    assert loaded_cfg.yoco_cross_banks == (0, 2)
    assert meta["missing_keys"] == []
    assert loaded_model is not None

    ap = ev.build_arg_parser()
    parsed = ap.parse_args(["--ckpt", str(ckpt_path), "--no-cudnn"])
    assert parsed.no_cudnn is True


def test_export_onnx_rejects_iter15_flags(tiny_iter15, tmp_path: Path) -> None:
    model, cfg = tiny_iter15
    with pytest.raises(NotImplementedError, match="use_yoco_cla=True"):
        export_uni_v2_onnx.export_all_targets(model, cfg, out_dir=tmp_path, targets=("cpu",))


def test_northstar_runner_abstain_not_counted_as_missing(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt_neg.musicxml"
    gt_path.write_text(eval_northstar_runner.EMPTY_VALID_MUSICXML, encoding="utf-8")
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    (pred_dir / "neg_page_1.abstain").write_text("abstain\n", encoding="utf-8")

    manifest = {
        "benchmark_version": "1.0.0",
        "split": "dev-v1",
        "pages": [
            {
                "page_id": "neg_page_1",
                "work_id": "w_neg_1",
                "domain": "CDR",
                "complexity_tier": "T1",
                "is_negative_control": True,
                "negative_control_type": "non_music",
                "gt_musicxml": str(gt_path),
                "image_path": str(tmp_path / "neg_page_1.png"),
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")

    report, rows, _ = eval_northstar_runner.evaluate_manifest(
        manifest_path=manifest_path,
        pred_dir=pred_dir,
        oracle=False,
        system_name="test-sys",
        strict_validate=False,
        n_boot=10,
    )
    assert rows[0]["pred_source"] == "abstain-marker"
    adv = report.get("layer2_matrix_and_deltas", {}).get("adversarial_scorecard", {})
    assert "missing_prediction_files" not in adv
