"""CPU unit tests & suite31_hw 45-piece round-trip verification for V-bekern + MaskedRefinerHead."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import torch
import torch.nn as nn

import data
import eval_sys_gate as esg
import uni_omr
import vbkern as vb


def test_clef_staff_positions() -> None:
    """Verify exact staff-line positions across Treble, Bass, Alto C3, Tenor C4, and octave clefs."""
    # Treble G-2: E4 = sp+0, G4 = sp+2, F5 = sp+8
    toks_g = ["clef-G-2", "key+0", "quarter", "E", "4", "quarter", "G", "4", "quarter", "F", "5"]
    v_g = vb.bekern_to_vbkern(toks_g)
    assert v_g == ["clef-G-2", "key+0", "quarter", "sp+0", "quarter", "sp+2", "quarter", "sp+8"], v_g
    assert vb.vbkern_to_bekern(v_g) == toks_g

    # Bass F-4: G2 = sp+0, F3 = sp+6, A3 = sp+8
    toks_f = ["clef-F-4", "key+0", "quarter", "G", "2", "quarter", "F", "3", "quarter", "A", "3"]
    v_f = vb.bekern_to_vbkern(toks_f)
    assert v_f == ["clef-F-4", "key+0", "quarter", "sp+0", "quarter", "sp+6", "quarter", "sp+8"], v_f
    assert vb.vbkern_to_bekern(v_f) == toks_f

    # Alto C-3: F3 = sp+0, C4 = sp+4 (middle line), G4 = sp+8
    toks_c3 = ["clef-C-3", "key+0", "quarter", "F", "3", "quarter", "C", "4", "quarter", "G", "4"]
    v_c3 = vb.bekern_to_vbkern(toks_c3)
    assert v_c3 == ["clef-C-3", "key+0", "quarter", "sp+0", "quarter", "sp+4", "quarter", "sp+8"], v_c3
    assert vb.vbkern_to_bekern(v_c3) == toks_c3

    # Tenor C-4: D3 = sp+0, C4 = sp+6 (line 4), E4 = sp+8
    toks_c4 = ["clef-C-4", "key+0", "quarter", "D", "3", "quarter", "C", "4", "quarter", "E", "4"]
    v_c4 = vb.bekern_to_vbkern(toks_c4)
    assert v_c4 == ["clef-C-4", "key+0", "quarter", "sp+0", "quarter", "sp+6", "quarter", "sp+8"], v_c4
    assert vb.vbkern_to_bekern(v_c4) == toks_c4

    # Octave-transposing treble G-2-8vb: E3 = sp+0, G3 = sp+2
    toks_8vb = ["clef-G-2-8vb", "key+0", "quarter", "E", "3", "quarter", "G", "3"]
    v_8vb = vb.bekern_to_vbkern(toks_8vb)
    assert v_8vb == ["clef-G-2-8vb", "key+0", "quarter", "sp+0", "quarter", "sp+2"], v_8vb
    assert vb.vbkern_to_bekern(v_8vb) == toks_8vb

    # Gauge equivariance check: same visual notehead sp+4 under G-2 vs C-3 vs F-4
    # lifts to B4 (G-2), C4 (C-3), D3 (F-4) deterministically!
    assert vb.vbkern_to_bekern(["clef-G-2", "key+0", "quarter", "sp+4"]) == ["clef-G-2", "key+0", "quarter", "B", "4"]
    assert vb.vbkern_to_bekern(["clef-C-3", "key+0", "quarter", "sp+4"]) == ["clef-C-3", "key+0", "quarter", "C", "4"]
    assert vb.vbkern_to_bekern(["clef-F-4", "key+0", "quarter", "sp+4"]) == ["clef-F-4", "key+0", "quarter", "D", "3"]


def test_keys_accidentals_carryover_barline_reset() -> None:
    """Verify sharp/flat keys, visually-drawn-only accidentals, in-measure carryover, and barline reset."""
    # Key of D major (key+2: F#, C#):
    # Bar 1: F#4 (implied -> NO acc!), F4 natural (drawn -> acc0!), F4 natural (carried -> NO acc!), F#4 (drawn -> acc+1!)
    # Bar 2 (after barline reset): F#4 (implied again -> NO acc!), C#5 (implied -> NO acc!)
    toks = [
        "clef-G-2", "key+2", "time-4-4", "voice-1",
        "quarter", "F", "4", "a1",
        "quarter", "F", "4",
        "quarter", "F", "4",
        "quarter", "F", "4", "a1",
        "bar", "voice-1",
        "half", "F", "4", "a1",
        "half", "C", "5", "a1",
    ]
    v = vb.bekern_to_vbkern(toks)
    expected = [
        "clef-G-2", "key+2", "time-4-4", "voice-1",
        "quarter", "sp+1",
        "quarter", "sp+1", "acc0",
        "quarter", "sp+1",
        "quarter", "sp+1", "acc+1",
        "bar", "voice-1",
        "half", "sp+1",
        "half", "sp+5",
    ]
    assert v == expected, f"\nGot: {v}\nExp: {expected}"
    assert vb.vbkern_to_bekern(v) == toks

    # Flat key (key-3: Bb, Eb, Ab) with double flat, chords, ties, rests, and <sys> reset
    toks_flat = [
        "clef-F-4", "key-3", "time-3-4", "voice-1",
        "quarter", "B", "2", "a-1",
        "chord", "quarter", "E", "3", "a-1", "tie-start",
        "quarter", "E", "3", "a-1", "tie-stop",
        "quarter", "E", "3", "a-2",
        "rest", "quarter",
        "<sys>",
        "clef-C-4", "key-3", "voice-1",
        "half", "E", "3", "a-1",  # reset by <sys>, implied flat -> no acc
        "quarter", "E", "3", "a1",  # sharp -> acc+1
    ]
    v_flat = vb.bekern_to_vbkern(toks_flat)
    assert "acc-1" not in v_flat  # all Eb/Bb are key-implied!
    assert "acc-2" in v_flat
    assert "acc+1" in v_flat
    assert vb.vbkern_to_bekern(v_flat) == toks_flat


def test_gauge_error_decomposition() -> None:
    """Verify decomposition cleanly separates gauge-transport vs visual-sp vs visual-acc errors."""
    # GT is in Alto clef (C-3) with key+1 (F#): sp+4 = C4, sp+0 = F#3, sp+5 = D4
    gt = [
        "clef-C-3", "key+1", "voice-1",
        "quarter", "C", "4",          # sp+4, no drawn acc
        "quarter", "F", "3", "a1",    # sp+0, no drawn acc
        "quarter", "D", "4",          # sp+5, no drawn acc
    ]
    # Suppose a semantic model misread clef-C-3 as clef-G-2 (gauge transport error!)
    # Note 1: B4 in G-2 has sp+4 (same visual sp! -> gauge_transport_error)
    # Note 2: E4 in G-2 has sp+0 (same visual sp! -> gauge_transport_error)
    # Note 3: A4 in G-2 has sp+3 (wrong visual sp! -> visual_sp_error)
    pred = [
        "clef-G-2", "key+1", "voice-1",
        "quarter", "B", "4",
        "quarter", "E", "4",
        "quarter", "A", "4",
    ]
    d = vb.gauge_error_decomposition(pred, gt)
    assert d["n_aligned"] == 3
    assert d["gauge_transport_errors"] == 2, d
    assert d["visual_sp_errors"] == 1, d
    assert d["visual_acc_errors"] == 0, d
    assert abs(d["gauge_error_share"] - (2.0 / 3.0)) < 1e-6
    assert d["gt_roundtrip_exact"] is True


def test_masked_refiner_head_cpu() -> None:
    """Verify MaskedRefinerHead param budget (<0.8M), training loss, and 2-step refinement on CPU."""
    torch.manual_seed(0)
    d_model = 384
    V = vb.VB_VOCAB_SIZE
    head = vb.MaskedRefinerHead(d_model=d_model, vocab_size=V)
    n_params = head.num_params()
    assert n_params < 800_000, f"MaskedRefinerHead too large: {n_params:,} >= 800,000"
    print(f"[test_masked_refiner_head_cpu] params={n_params:,} (< 800k OK)")

    lm_head = nn.Linear(d_model, V, bias=False)
    B, T = 2, 32
    h_dec = torch.randn(B, T, d_model)
    tgt = torch.randint(0, vb.VB_PAD_ID, (B, T))
    tgt[:, 0] = vb.VB_BOS_ID
    tgt[:, -1] = vb.VB_EOS_ID

    loss = head.compute_refinement_loss(
        h_dec, tgt, lm_head,
        pad_id=vb.VB_PAD_ID, bos_id=vb.VB_BOS_ID, eos_id=vb.VB_EOS_ID,
    )
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in head.parameters())

    # 2-step inference refinement
    draft_probs = torch.full((B, T), 0.90)
    draft_probs[:, 5:9] = 0.25  # low-confidence span
    refined, stats = head.refine_2step(
        h_dec, tgt, draft_probs, lm_head,
        pad_id=vb.VB_PAD_ID, bos_id=vb.VB_BOS_ID, eos_id=vb.VB_EOS_ID,
    )
    assert refined.shape == (B, T)
    assert stats["total_masked"] > 0


def test_suite31_hw_45_piece_roundtrip() -> None:
    """Verify bijective bekern -> V-bekern -> bekern across ALL 45 suite31_hw GT pieces (>=0.995)."""
    s31_root = _REPO_ROOT / "data-gen" / "score-corpus" / "suite31"
    manifest_path = s31_root / "suite31_hw_manifest.json"
    assert manifest_path.is_file(), f"Missing {manifest_path}"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pieces = manifest["pieces"]
    assert len(pieces) == 45

    workdir = Path(tempfile.mkdtemp(prefix="vbkern_s31_test_"))
    exact_pieces = 0
    total_gt_toks = 0
    total_vb_toks = 0
    total_matched_toks = 0
    sp_min_obs = 999
    sp_max_obs = -999

    for p in pieces:
        gt_path = s31_root / p["gt"]["file"]
        ref_toks, _ = esg.suite31_page_refs(gt_path, "vbk-" + p["slug"], workdir)
        vb_toks = vb.bekern_to_vbkern(ref_toks)
        # Verify every V-bekern token encodes & decodes cleanly
        vb_ids = vb.encode_vbkern(vb_toks, add_bos_eos=True)
        vb_dec = vb.decode_vbkern(vb_ids, skip_special=True)
        assert vb_dec == vb_toks, f"Vocab encode/decode mismatch on {p['slug']}"

        rt_toks = vb.vbkern_to_bekern(vb_toks)
        total_gt_toks += len(ref_toks)
        total_vb_toks += len(vb_toks)
        matched = sum(1 for a, b in zip(ref_toks, rt_toks) if a == b)
        total_matched_toks += matched
        if rt_toks == ref_toks:
            exact_pieces += 1
        for t in vb_toks:
            m = vb.SP_PATTERN.match(t)
            if m:
                val = int(m.group(1))
                sp_min_obs = min(sp_min_obs, val)
                sp_max_obs = max(sp_max_obs, val)

    tok_acc = total_matched_toks / max(1, total_gt_toks)
    comp_ratio = total_vb_toks / max(1, total_gt_toks)
    print(
        f"[suite31_hw 45-piece round-trip] exact_pieces={exact_pieces}/45 | "
        f"token_acc={tok_acc:.6f} ({total_matched_toks}/{total_gt_toks}) | "
        f"seq_len: bekern={total_gt_toks} -> vbkern={total_vb_toks} ({comp_ratio:.4f}x, "
        f"{(1.0 - comp_ratio)*100:.1f}% shorter!) | "
        f"vocab: bekern={uni_omr.VOCAB_SIZE} vs vbkern={vb.VB_VOCAB_SIZE} | "
        f"observed sp range=[{sp_min_obs:+d}..{sp_max_obs:+d}]"
    )
    assert tok_acc >= 0.995, f"Round-trip token acc {tok_acc:.6f} < 0.995"
    assert exact_pieces == 45, f"Expected 45/45 exact pieces, got {exact_pieces}/45"


if __name__ == "__main__":
    test_clef_staff_positions()
    test_keys_accidentals_carryover_barline_reset()
    test_gauge_error_decomposition()
    test_masked_refiner_head_cpu()
    test_suite31_hw_45_piece_roundtrip()
    print("ALL V-BEKERN TESTS PASSED!")
