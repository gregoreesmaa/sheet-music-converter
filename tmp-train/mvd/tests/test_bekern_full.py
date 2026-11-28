"""CPU unit tests for bekern-full Pillar 2 (`tmp-train/mvd/bekern_full.py`)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_MVD_DIR = Path(__file__).resolve().parent.parent
if str(_MVD_DIR) not in sys.path:
    sys.path.insert(0, str(_MVD_DIR))

import bekern_full as bf  # noqa: E402
import data  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Backward compatibility & vocab layout (0..264 frozen)
# ---------------------------------------------------------------------------

def test_legacy_vocab_unchanged_and_prefix_identical():
    assert len(data.VOCAB) == 261
    assert data.PAD_ID == 261
    assert data.BOS_ID == 262
    assert data.EOS_ID == 263
    assert data.VOCAB_SIZE == 264

    assert bf.PAD_ID == 261
    assert bf.BOS_ID == 262
    assert bf.EOS_ID == 263
    assert bf.SYS_ID == 264
    assert bf.LEGACY_VOCAB_SIZE == 265

    # Exact 0..260 identity with data.TOKEN_TO_ID / data.ID_TO_TOKEN
    for i, tok in enumerate(data.VOCAB):
        assert bf.FULL_ID_TO_TOKEN[i] == tok
        assert bf.FULL_TOKEN_TO_ID[tok] == i
        assert data.TOKEN_TO_ID[tok] == i
        assert data.ID_TO_TOKEN[i] == tok

    # Exact 261..264 special identity
    assert bf.FULL_ID_TO_TOKEN[261] == "<pad>"
    assert bf.FULL_ID_TO_TOKEN[262] == "<bos>"
    assert bf.FULL_ID_TO_TOKEN[263] == "<eos>"
    assert bf.FULL_ID_TO_TOKEN[264] == "<sys>"

    # No duplicate tokens and valid MUSIC_VOCAB_END boundary
    assert len(bf.FULL_VOCAB) == bf.FULL_VOCAB_SIZE
    assert len(set(bf.FULL_VOCAB)) == bf.FULL_VOCAB_SIZE
    assert bf.LEGACY_VOCAB_SIZE < bf.MUSIC_VOCAB_END < bf.FULL_VOCAB_SIZE


def test_all_four_layers_present_in_full_vocab():
    # Schema prefixes
    for tok in ("<schema:v1>", "<schema:v2a>", "<schema:full>"):
        assert bf.FULL_TOKEN_TO_ID[tok] >= 265
        assert bf.token_layer_of(tok) == -1

    # Layer 0
    for tok in (
        "<part>", "<staff-def>", "<group>", "<transpose>", "<pickup>",
        "<multirest>", "mrep-1", "mrep-2", "mrep-4", "<inst-change>",
        "D.C.", "D.S.", "Segno", "Coda", "Fine",
    ):
        assert bf.FULL_TOKEN_TO_ID[tok] >= 265
        assert bf.FULL_TOKEN_TO_ID[tok] < bf.MUSIC_VOCAB_END
        assert bf.token_layer_of(tok) == 0

    # Layer 1 (tup-open/tup-close keep legacy IDs; new ones live >= 265)
    assert bf.FULL_TOKEN_TO_ID["tup-open"] == data.TOKEN_TO_ID["tup-open"]
    assert bf.FULL_TOKEN_TO_ID["tup-close"] == data.TOKEN_TO_ID["tup-close"]
    for tok in (
        "q#", "qb", "3q#", "3qb", "a0", "acc-caut", "acc-paren", "<chord>",
        "stem-up", "stem-down", "beam-start", "beam-cont", "beam-end",
        "beam-hook-l", "beam-hook-r", "unpitched", "nh-x", "nh-diamond",
        "nh-slash", "nh-triangle", "nh-circle-x", "nh-square",
        "str", "fret", "bend", "hopo", "grace-slash",
        "xstaff-1", "xstaff-8",
    ):
        assert bf.FULL_TOKEN_TO_ID[tok] >= 265
        assert bf.FULL_TOKEN_TO_ID[tok] < bf.MUSIC_VOCAB_END
        assert bf.token_layer_of(tok) == 1

    # Layer 2
    for tok in (
        "marcato", "mordent-up", "mordent-lo", "turn", "turn-inv",
        "tremolo-1", "tremolo-2", "tremolo-3", "tremolo-2n",
        "15ma-open", "15ma-close", "15mb-open", "15mb-close",
        "pedal-sost-open", "pedal-sost-close", "pedal-uc-open", "pedal-uc-close",
        "arp-up", "arp-down", "arp-non", "trill-line-open", "trill-line-close",
        "gliss-open", "gliss-close",
        "slur-1-open", "slur-4-close",
        "hairpin-1-cresc", "hairpin-2-dim", "hairpin-3-close",
        "breath", "caesura", "fing-1", "fing-5", "harmonics",
    ):
        assert bf.FULL_TOKEN_TO_ID[tok] >= 265
        assert bf.FULL_TOKEN_TO_ID[tok] < bf.MUSIC_VOCAB_END
        assert bf.token_layer_of(tok) == 2

    # Layer 3 (>= MUSIC_VOCAB_END)
    for tok in (
        "<sys_lyr>", "<harm>", "<tempo>", "<dir>", "<reh>", "<fb>",
        "</text_span>", "lyr-beg", "lyr-mid", "lyr-end", "lyr-ext",
        "@0", "@63", "v=1", "v=4",
    ):
        assert bf.FULL_TOKEN_TO_ID[tok] >= bf.MUSIC_VOCAB_END
        assert bf.token_layer_of(tok) == 3


# ---------------------------------------------------------------------------
# 2. 4-Bit Layer Bitmask & Unannotated-Layer CE Loss Masking (Mitigation #1)
# ---------------------------------------------------------------------------

def test_layer_ce_masks_v1_v2a_full():
    assert bf.SCHEMA_LAYER_BITS == {"v1": 0b0100, "v2a": 0b0110, "full": 0b1111}

    # Build a 3-row batch with one token from each category:
    # [<schema:v1>, <part>(L0), quarter(L1), dyn-f(L2), marcato(L2), <sys_lyr>(L3), <eos>(univ), <pad>]
    seq = [
        bf.FULL_TOKEN_TO_ID["<schema:v1>"],
        bf.FULL_TOKEN_TO_ID["<part>"],
        bf.FULL_TOKEN_TO_ID["quarter"],
        bf.FULL_TOKEN_TO_ID["dyn-f"],
        bf.FULL_TOKEN_TO_ID["marcato"],
        bf.FULL_TOKEN_TO_ID["<sys_lyr>"],
        bf.EOS_ID,
        bf.PAD_ID,
    ]
    target_ids = torch.tensor([seq, seq, seq], dtype=torch.long)
    layer_masks = torch.tensor(
        [bf.SCHEMA_LAYER_BITS["v1"], bf.SCHEMA_LAYER_BITS["v2a"], bf.SCHEMA_LAYER_BITS["full"]],
        dtype=torch.long,
    )

    target_mask, logit_suppress = bf.build_layer_ce_masks(target_ids, layer_masks)
    assert target_mask.shape == (3, len(seq))
    assert logit_suppress.shape == (3, bf.FULL_VOCAB_SIZE)

    # Row 0 (v1 = 0b0100): keeps universal (non-PAD) + L1; masks out L0, L2, L3, PAD
    assert target_mask[0].tolist() == [True, False, True, False, False, False, True, False]
    assert logit_suppress[0, bf.FULL_TOKEN_TO_ID["<part>"]].item() is True
    assert logit_suppress[0, bf.FULL_TOKEN_TO_ID["quarter"]].item() is False
    assert logit_suppress[0, bf.FULL_TOKEN_TO_ID["dyn-f"]].item() is True
    assert logit_suppress[0, bf.FULL_TOKEN_TO_ID["marcato"]].item() is True
    assert logit_suppress[0, bf.FULL_TOKEN_TO_ID["<sys_lyr>"]].item() is True
    assert logit_suppress[0, bf.EOS_ID].item() is False

    # Row 1 (v2a = 0b0110): keeps universal (non-PAD) + L1 + L2; masks out L0, L3, PAD
    assert target_mask[1].tolist() == [True, False, True, True, True, False, True, False]
    assert logit_suppress[1, bf.FULL_TOKEN_TO_ID["<part>"]].item() is True
    assert logit_suppress[1, bf.FULL_TOKEN_TO_ID["quarter"]].item() is False
    assert logit_suppress[1, bf.FULL_TOKEN_TO_ID["dyn-f"]].item() is False
    assert logit_suppress[1, bf.FULL_TOKEN_TO_ID["marcato"]].item() is False
    assert logit_suppress[1, bf.FULL_TOKEN_TO_ID["<sys_lyr>"]].item() is True

    # Row 2 (full = 0b1111): keeps everything except PAD; suppresses zero logits
    assert target_mask[2].tolist() == [True, True, True, True, True, True, True, False]
    assert not bool(logit_suppress[2].any())


def test_masked_layer_ce_zeroes_unannotated_logit_gradients():
    # Verify that unannotated Layer 2 logits receive ZERO gradient under v1!
    t = bf.FULL_TOKEN_TO_ID
    target_ids = torch.tensor([[t["quarter"], t["C"], t["4"]]], dtype=torch.long)
    layer_masks = torch.tensor([bf.SCHEMA_LAYER_BITS["v1"]], dtype=torch.long)

    logits = torch.zeros((1, 3, bf.FULL_VOCAB_SIZE), requires_grad=True)
    loss = bf.masked_layer_cross_entropy(logits, target_ids, layer_masks)
    loss.backward()

    grad = logits.grad[0]  # (3, V)
    # Layer 2 tokens (dyn-f, marcato, slur-1-open) and Layer 0/3 tokens must have 0 gradient!
    for unann_tok in ("<part>", "dyn-f", "marcato", "slur-1-open", "<sys_lyr>", "@0"):
        assert torch.all(grad[:, t[unann_tok]] == 0.0), f"non-zero grad on {unann_tok}"
    # Layer 1 target tokens must have non-zero gradient!
    assert grad[0, t["quarter"]].item() < 0.0


# ---------------------------------------------------------------------------
# 3. MTP-4 Mode-Switch Guard & BD3-LM is_music_body Mask (Mitigations #6 & #7)
# ---------------------------------------------------------------------------

def test_mode_switch_token_guard():
    expected_switches = {
        "<sys_lyr>", "<harm>", "<tempo>", "<dir>", "<reh>", "<fb>", "</text_span>",
    }
    for tok in expected_switches:
        tid = bf.FULL_TOKEN_TO_ID[tok]
        assert bf.is_mode_switch_token(tid) is True
        assert tid in bf.MODE_SWITCH_TOKEN_IDS

    assert len(bf.MODE_SWITCH_TOKEN_IDS) == len(expected_switches)

    for non_switch in (
        "<pad>", "<bos>", "<eos>", "<sys>", "<schema:full>",
        "quarter", "C", "4", "bar", "marcato", "lyr-beg", "@0", "v=1",
    ):
        assert bf.is_mode_switch_token(bf.FULL_TOKEN_TO_ID[non_switch]) is False


def test_compute_is_music_body_mask():
    seq_toks = [
        "<bos>", "<schema:full>", "<part>", "clef-G-2", "quarter", "C", "4",
        "marcato", "bar",
        "<sys_lyr>", "v=1", "@0", "lyr-beg", "</text_span>",
        "<sys>", "half", "D", "4",
        "<harm>", "@0", "</text_span>",
        "<eos>", "<pad>",
    ]
    ids = torch.tensor([[bf.FULL_TOKEN_TO_ID[t] for t in seq_toks]], dtype=torch.long)
    mask = bf.compute_is_music_body_mask(ids)
    assert mask.shape == (1, len(seq_toks))

    expected = [
        False,  # <bos>
        False,  # <schema:full>
        True,   # <part> (L0)
        True,   # clef-G-2 (L1)
        True,   # quarter (L1)
        True,   # C (L1)
        True,   # 4 (L1)
        True,   # marcato (L2)
        True,   # bar (L1)
        False,  # <sys_lyr> (L3 open)
        False,  # v=1 (inside L3)
        False,  # @0 (inside L3)
        False,  # lyr-beg (inside L3)
        False,  # </text_span> (L3 close)
        False,  # <sys>
        True,   # half (L1)
        True,   # D (L1)
        True,   # 4 (L1)
        False,  # <harm> (L3 open)
        False,  # @0 (inside L3)
        False,  # </text_span> (L3 close)
        False,  # <eos>
        False,  # <pad>
    ]
    assert mask[0].tolist() == expected

    # 1D input works identically
    mask_1d = bf.compute_is_music_body_mask(ids[0])
    assert mask_1d.tolist() == expected


# ---------------------------------------------------------------------------
# 4. Layer 3 UTF-8 Byte-Level Text Codec (TEXT_VOCAB_SIZE = 259)
# ---------------------------------------------------------------------------

def test_utf8_text_codec_roundtrip():
    assert bf.TEXT_BOS_ID == 256
    assert bf.TEXT_EOS_ID == 257
    assert bf.TEXT_PAD_ID == 258
    assert bf.TEXT_VOCAB_SIZE == 259

    samples = [
        "",
        "Allegro ma non troppo",
        "Über allen Gipfeln — śćźżół",
        "C#m7/G# — cœur d'été, mañana!",
    ]
    for text in samples:
        enc = bf.encode_text_span_utf8(text)
        assert enc[0] == bf.TEXT_BOS_ID
        assert enc[-1] == bf.TEXT_EOS_ID
        assert all(0 <= b <= 255 for b in enc[1:-1])
        assert bf.decode_text_span_utf8(enc) == text
        # Trailing PAD and post-EOS garbage must be ignored
        assert bf.decode_text_span_utf8(enc + [bf.TEXT_PAD_ID, 65, 66]) == text


# ---------------------------------------------------------------------------
# 5. BekernFullGrammarState FSA (Mitigations #3 & #8)
# ---------------------------------------------------------------------------

def test_grammar_schema_prefix_position_zero_only():
    st = bf.BekernFullGrammarState(strict=True)
    allow0, _ = st.allowed()
    assert bf.FULL_TOKEN_TO_ID["<schema:full>"] in allow0

    st.step("<schema:full>")
    assert st.schema == "<schema:full>"

    # Second schema token at pos > 0 must be forbidden
    allow1, _ = st.allowed()
    for tok in bf.SCHEMA_TOKENS:
        assert bf.FULL_TOKEN_TO_ID[tok] not in allow1
    with pytest.raises(ValueError):
        st.step("<schema:v1>")

    # Music proceeds normally right after schema prefix
    for tok in ("clef-G-2", "key+0", "time-4-4", "quarter", "C", "4"):
        st.step(tok)


def test_grammar_numbered_slur_and_hairpin_parity():
    st = bf.BekernFullGrammarState(strict=True)
    for tok in ("<schema:full>", "clef-G-2", "time-4-4", "quarter", "C", "4"):
        st.step(tok)

    # Closing unopened slur-1 or hairpin-2 is forbidden
    allow, _ = st.allowed()
    assert bf.FULL_TOKEN_TO_ID["slur-1-open"] in allow
    assert bf.FULL_TOKEN_TO_ID["slur-1-close"] not in allow
    assert bf.FULL_TOKEN_TO_ID["hairpin-2-cresc"] in allow
    assert bf.FULL_TOKEN_TO_ID["hairpin-2-close"] not in allow
    with pytest.raises(ValueError):
        st.step("slur-1-close")
    with pytest.raises(ValueError):
        st.step("hairpin-2-close")

    # Open overlapping slur-1 and slur-2 simultaneously
    st.step("slur-1-open")
    st.step("slur-2-open")
    st.step("hairpin-1-cresc")

    # Double-opening slur-1 or hairpin-1 (even with dim) must be forbidden
    allow2, _ = st.allowed()
    assert bf.FULL_TOKEN_TO_ID["slur-1-open"] not in allow2
    assert bf.FULL_TOKEN_TO_ID["slur-1-close"] in allow2
    assert bf.FULL_TOKEN_TO_ID["hairpin-1-cresc"] not in allow2
    assert bf.FULL_TOKEN_TO_ID["hairpin-1-dim"] not in allow2
    assert bf.FULL_TOKEN_TO_ID["hairpin-1-close"] in allow2

    with pytest.raises(ValueError):
        st.step("slur-1-open")
    with pytest.raises(ValueError):
        st.step("hairpin-1-dim")

    # Crossing close order (close slur-2 before slur-1) is valid
    for tok in ("quarter", "D", "4", "slur-2-close", "quarter", "E", "4", "slur-1-close", "hairpin-1-close"):
        st.step(tok)

    assert len(st.open_slurs) == 0
    assert len(st.open_hairpins) == 0

    # Re-opening slur-1 after closing is valid
    st.step("slur-1-open")
    assert 1 in st.open_slurs


def test_grammar_pauses_rhythmic_fsa_inside_layer3_text_span():
    st = bf.BekernFullGrammarState(strict=True)
    # 4/4 measure: emit 2 beats (two quarter notes), leaving exactly 2 beats left
    for tok in ("<schema:full>", "clef-G-2", "time-4-4", "quarter", "C", "4", "quarter", "D", "4"):
        st.step(tok)

    # Open Layer 3 <sys_lyr> span; pending 2nd quarter note closes cleanly (2 beats committed),
    # rhythmic FSA pauses, and <eos>/<sys> are blocked
    st.step("<sys_lyr>")
    assert st.in_text_span is True
    assert st.music_state.mode == "bound"
    vcurs_in_span = list(st.music_state.vcurs_rat)
    assert vcurs_in_span == [(2, 1)]
    allow_txt, eos_ok = st.allowed()
    assert eos_ok is False
    assert bf.SYS_ID not in allow_txt
    assert bf.FULL_TOKEN_TO_ID["</text_span>"] in allow_txt
    assert bf.FULL_TOKEN_TO_ID["@0"] in allow_txt
    assert bf.FULL_TOKEN_TO_ID["v=1"] in allow_txt
    assert bf.FULL_TOKEN_TO_ID["lyr-beg"] in allow_txt

    for tok in ("v=1", "@0", "lyr-beg", "@1", "lyr-end"):
        st.step(tok)
    assert st.music_state.vcurs_rat == vcurs_in_span

    # Close text span -> rhythmic FSA resumes cleanly with 2 beats remaining!
    st.step("</text_span>")
    assert st.in_text_span is False
    assert st.music_state.vcurs_rat == vcurs_in_span

    # A whole note (4 beats) would overshoot the 2 remaining beats -> still blocked!
    with pytest.raises(ValueError):
        st.step("whole")

    # A half note (2 beats) fits cleanly to complete the 4/4 measure!
    for tok in ("half", "E", "4", "bar"):
        st.step(tok)

    # Copy preserves independent state
    st_copy = st.copy()
    st_copy.step("<harm>")
    assert st_copy.in_text_span is True
    assert st.in_text_span is False


# ---------------------------------------------------------------------------
# 6. Round 1 Regression Tests (label_smoothing, codec, depth, FSA hardening)
# ---------------------------------------------------------------------------

def test_masked_layer_ce_label_smoothing_finite_and_zero_sum_active_grad():
    t = bf.FULL_TOKEN_TO_ID
    target_ids = torch.tensor([[t["quarter"], t["C"], t["4"]]], dtype=torch.long)
    layer_masks = torch.tensor([bf.SCHEMA_LAYER_BITS["v1"]], dtype=torch.long)

    logits = torch.randn((1, 3, bf.FULL_VOCAB_SIZE), requires_grad=True)
    loss = bf.masked_layer_cross_entropy(logits, target_ids, layer_masks, label_smoothing=0.1)
    assert torch.isfinite(loss), f"loss was not finite: {loss}"
    loss.backward()

    _, suppress = bf.build_layer_ce_masks(target_ids, layer_masks)
    active = ~suppress[0]
    grad = logits.grad[0]  # (3, V)
    # Suppressed classes must have identically 0 gradient
    assert torch.all(grad[:, suppress[0]] == 0.0)
    # Active classes must sum to 0 per token (unbiased simplex gradient)
    active_grad_sum = grad[:, active].sum(dim=-1)
    assert torch.allclose(active_grad_sum, torch.zeros_like(active_grad_sum), atol=1e-5)

    # Also test with per-token class weights
    logits2 = torch.randn((1, 3, bf.FULL_VOCAB_SIZE), requires_grad=True)
    w = torch.ones(bf.FULL_VOCAB_SIZE)
    w[t["quarter"]] = 2.0
    loss2 = bf.masked_layer_cross_entropy(logits2, target_ids, layer_masks, label_smoothing=0.1, weight=w)
    assert torch.isfinite(loss2)
    loss2.backward()
    assert torch.all(logits2.grad[0][:, suppress[0]] == 0.0)


def test_finite_argument_tokens_and_valid_music_body_ids():
    # Layer 0 new argument tokens
    for tok in ("brace", "bracket", "barline", "lines-1", "lines-6", "-12", "-2", "+1", "+12", "9", "10", "11", "12", ":"):
        assert bf.FULL_TOKEN_TO_ID[tok] >= 265
        assert bf.token_layer_of(tok) == 0
    # Layer 1 new argument tokens
    for tok in ("str-1", "str-6", "fret-0", "fret-24", "clef-T-4", "clef-T-5", "clef-T-6", "xstaff-9", "xstaff-16"):
        assert bf.FULL_TOKEN_TO_ID[tok] >= 265
        assert bf.token_layer_of(tok) == 1
    # Layer 2 compat alias
    assert bf.FULL_TOKEN_TO_ID["15-stop"] >= 265
    assert bf.token_layer_of("15-stop") == 2
    # Layer 3 expanded anchors & verses
    for tok in ("@0", "@64", "@255", "v=1", "v=5", "v=8"):
        assert bf.FULL_TOKEN_TO_ID[tok] >= bf.MUSIC_VOCAB_END
        assert bf.token_layer_of(tok) == 3

    # VALID_MUSIC_BODY_TOKEN_IDS strictly excludes PAD/BOS/EOS/SYS, SCHEMA_TOKEN_IDS, and Layer 3
    valid_set = set(bf.VALID_MUSIC_BODY_TOKEN_IDS)
    for forbidden in (bf.PAD_ID, bf.BOS_ID, bf.EOS_ID, bf.SYS_ID, *bf.SCHEMA_TOKEN_IDS):
        assert forbidden not in valid_set
    assert all(0 <= i < bf.MUSIC_VOCAB_END for i in valid_set)


def test_dual_head_sequence_codec_roundtrip_zero_oov():
    sample = (
        "<schema:full> <part> brace lines-5 clef-G-2 time-4-4 "
        "tup-open-3:2 eighth C 4 eighth D 4 eighth E 4 tup-close "
        "multirest-16 bar "
        "<tempo> Allegro ma non troppo </text_span> "
        "<sys_lyr v=1> @0 lyr-beg Über @1 lyr-end Gipfeln </text_span> "
        "<harm> C#m7/G# </text_span>"
    )
    enc = bf.encode_bekern_full_sequence(sample, add_bos_eos=True)
    assert enc["music_ids"][0] == bf.BOS_ID
    assert enc["music_ids"][-1] == bf.EOS_ID
    assert len(enc["text_spans"]) == 3
    assert bf.decode_text_span_utf8(enc["text_spans"][0]) == "Allegro ma non troppo"
    assert bf.decode_text_span_utf8(enc["text_spans"][1]) == "v=1 @0 lyr-beg Über @1 lyr-end Gipfeln"
    assert bf.decode_text_span_utf8(enc["text_spans"][2]) == "C#m7/G#"

    dec = bf.decode_bekern_full_sequence(enc["music_ids"], enc["text_spans"])
    assert "Allegro ma non troppo" in dec
    assert "Über" in dec
    assert "C#m7/G#" in dec
    assert "tup-open-3:2" in dec
    assert "<multirest> multirest-16" in dec


def test_compute_is_music_body_mask_nested_and_stray_close():
    t = bf.FULL_TOKEN_TO_ID
    seq = torch.tensor([[
        t["</text_span>"],  # stray close before any open -> must NOT break subsequent open
        t["quarter"],       # body -> True
        t["<sys_lyr>"],     # depth 0 -> 1 -> False
        t["<harm>"],        # nested depth 1 -> 2 -> False
        t["quarter"],       # inside nested span -> False
        t["</text_span>"],  # depth 2 -> 1 -> False
        t["quarter"],       # still depth 1 -> False!
        t["</text_span>"],  # depth 1 -> 0 -> False
        t["C"],             # outside span -> True
    ]], dtype=torch.long)
    mask = bf.compute_is_music_body_mask(seq)
    assert mask[0].tolist() == [False, True, False, False, False, False, False, False, True]


def test_grammar_hardening_aliases_per_staff_spanners_caps_and_eos():
    st = bf.BekernFullGrammarState(strict=True)
    st.step("<schema:full>")
    # Right after <schema:full> with 0 music events, eos_ok must be False
    _, eos_ok = st.allowed()
    assert eos_ok is False

    for tok in ("clef-G-2", "time-4-4", "quarter", "C", "4"):
        st.step(tok)
    # <chord> delegates to chord; after <chord>, mid-note mode blocks <sys>
    allow_chord, _ = st.allowed()
    assert bf.FULL_TOKEN_TO_ID["<chord>"] in allow_chord
    st.step("<chord>")
    assert st.can_emit_sys() is False
    st.step("quarter")
    st.step("E")
    st.step("4")

    # Opening text span in pitch-tail closes pending event cleanly -> mode == "bound"
    st.step("<tempo>")
    assert st.music_state.mode == "bound"
    # Anti-loop cap MAX_LAYER3_SPAN_TOKENS = 128
    for _ in range(bf.MAX_LAYER3_SPAN_TOKENS):
        st.step("@0")
    allow_cap, _ = st.allowed()
    assert allow_cap == {bf.TEXT_SPAN_CLOSE_TOKEN_ID}
    with pytest.raises(ValueError):
        st.step("@0")
    st.step("</text_span>")

    # Per-staff slur channels: open slur-1 on staff 0, switch to staff 1 via <staff> -> slur-1 is free on staff 1!
    st.step("quarter")
    st.step("D")
    st.step("4")
    st.step("slur-1-open")
    assert 1 in st.open_slurs
    st.step("<staff>")
    assert 1 not in st.open_slurs
    st.step("clef-F-4")
    st.step("quarter")
    st.step("C")
    st.step("3")
    st.step("slur-1-open")
    assert 1 in st.open_slurs

    # Percussion/tab delegation when allow_percussion_tab=True
    st_pt = bf.BekernFullGrammarState(strict=False, allow_percussion_tab=True)
    for tok in ("clef-G-2", "time-4-4", "unpitched", "quarter", "C", "4", "nh-x"):
        st_pt.step(tok)


# ---------------------------------------------------------------------------
# 7. Round 2 Regression Tests (R2-2 & R2-4: LS NaN, Rust<->Py round-trip, FSA)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_weight", [False, True])
def test_masked_layer_ce_label_smoothing_suppressed_target_and_neg100_finite(dtype, use_weight):
    """R2-2: label_smoothing=0.1 with suppressed target + -100 + PAD must be finite (no +inf*0 NaN)."""
    t = bf.FULL_TOKEN_TO_ID
    # Row 0 is v1 (L1 only): dyn-f (L2), <part> (L0), <sys_lyr> (L3) are SUPPRESSED!
    # Also include -100 and PAD_ID alongside valid L1 tokens (quarter, C).
    target_ids = torch.tensor(
        [[t["quarter"], t["dyn-f"], t["<part>"], t["<sys_lyr>"], -100, bf.PAD_ID, t["C"]]],
        dtype=torch.long,
    )
    layer_masks = torch.tensor([bf.SCHEMA_LAYER_BITS["v1"]], dtype=torch.long)

    logits = torch.randn((1, target_ids.shape[1], bf.FULL_VOCAB_SIZE), dtype=dtype, requires_grad=True)
    w = torch.ones(bf.FULL_VOCAB_SIZE, dtype=dtype) if use_weight else None
    if w is not None:
        w[t["quarter"]] = 2.0

    loss = bf.masked_layer_cross_entropy(
        logits, target_ids, layer_masks, label_smoothing=0.1, weight=w
    )
    assert torch.isfinite(loss), f"non-finite loss ({dtype}, weight={use_weight}): {loss}"
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    _, suppress = bf.build_layer_ce_masks(target_ids, layer_masks)
    assert torch.all(logits.grad[0][:, suppress[0]] == 0.0)


def test_encode_grammar_decode_roundtrip_tuplets_multirest_transpose_and_rest_tail():
    """R2-2 & R2-4: tup-open-11:8, tup-open-3:2, <multirest> multirest-16, <transpose> -8 -14, rest-tail."""
    sample = (
        "<schema:full> <part> <transpose> -8 -14 clef-G-2 time-4-4 "
        "tup-open-11:8 eighth C 4 eighth D 4 eighth E 4 eighth F 4 tup-close bar "
        "tup-open-3:2 eighth G 4 eighth A 4 eighth B 4 tup-close bar "
        "<multirest> multirest-16 bar "
        "rest quarter <dir> dolce </text_span>"
    )
    enc = bf.encode_bekern_full_sequence(sample, add_bos_eos=True)
    toks = bf.decode_full(enc["music_ids"], skip_special=True)
    # Must NOT duplicate <multirest> when input already has `<multirest> multirest-16`!
    assert toks.count("<multirest>") == 1
    assert "-8" in toks and "-14" in toks

    # Walk every single music token through BekernFullGrammarState!
    st = bf.BekernFullGrammarState(strict=False)
    for tok in toks:
        allow, _ = st.allowed()
        assert bf.FULL_TOKEN_TO_ID[tok] in allow, f"FSA rejected {tok!r} at pos={st.pos}"
        st.step(tok)
    assert st.transpose_args_left == 0
    assert st.tup_mode is None

    # Also verify legacy v2 `tup-open 3 2` works cleanly in BekernFullGrammarState!
    st_leg = bf.BekernFullGrammarState(strict=True)
    for tok in ("clef-G-2", "time-4-4", "tup-open", "3", "2", "eighth", "C", "4", "eighth", "D", "4", "eighth", "E", "4", "tup-close"):
        allow, _ = st_leg.allowed()
        assert bf.FULL_TOKEN_TO_ID[tok] in allow, f"legacy tup-open rejected {tok!r}"
        st_leg.step(tok)

    # Verify rest-tail Layer 3 open calls _close_event(placed=False) (no pitch placed)!
    st_rest = bf.BekernFullGrammarState(strict=True)
    for tok in ("<schema:full>", "clef-G-2", "time-4-4", "rest", "quarter"):
        st_rest.step(tok)
    assert st_rest.music_state.mode == "rest-tail"
    st_rest.step("<tempo>")
    assert st_rest.in_text_span is True
    assert st_rest.music_state.mode == "bound"
    st_rest.step("</text_span>")

    # Verify decode_bekern_full_sequence recomposes tup-open-{a}:{n} and <multirest> multirest-{n}!
    dec = bf.decode_bekern_full_sequence(enc["music_ids"], enc["text_spans"])
    assert "tup-open-11:8" in dec
    assert "tup-open-3:2" in dec
    assert "<multirest> multirest-16" in dec
    assert "<multirest> <multirest>" not in dec
    assert "<transpose> -8 -14" in dec
    assert "<dir> dolce </text_span>" in dec


def test_live_rust_assembly_serialize_bekern_full_roundtrip(tmp_path):
    """R2-2: Live assembly-serialize.exe bekern --full -> Python -> bekern2musicxml round-trip."""
    import subprocess

    repo_root = _MVD_DIR.parent.parent
    exe = repo_root / "converter" / "assembly-serialize" / "target" / "release" / "assembly-serialize.exe"
    if not exe.is_file():
        exe = repo_root / "converter" / "assembly-serialize" / "target" / "debug" / "assembly-serialize.exe"
    if not exe.is_file():
        pytest.skip("assembly-serialize.exe not built")

    mxml = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1"><part-name>Clarinet</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>2</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
        <transpose><diatonic>-8</diatonic><chromatic>-14</chromatic></transpose>
      </attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>2</duration><type>quarter</type></note>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>2</duration><type>quarter</type></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration><type>quarter</type></note>
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>2</duration><type>quarter</type></note>
    </measure>
  </part>
</score-partwise>
"""
    in_xml = tmp_path / "in.musicxml"
    in_xml.write_text(mxml, encoding="utf-8")
    bk_dir = tmp_path / "bk"
    bk_dir.mkdir()

    subprocess.run(
        [
            str(exe), "bekern",
            "--input", str(in_xml),
            "--out", str(bk_dir),
            "--stem", "rt",
            "--measures", "1..1",
            "--full",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rust_bekern = (bk_dir / "rt-bekern.txt").read_text(encoding="utf-8").strip()
    assert rust_bekern, "assembly-serialize bekern --full produced empty output"

    enc = bf.encode_bekern_full_sequence(rust_bekern, add_bos_eos=True)
    dec = bf.decode_bekern_full_sequence(enc["music_ids"], enc["text_spans"])

    bk_file = tmp_path / "roundtrip-bekern.txt"
    bk_file.write_text(dec, encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    p2 = subprocess.run(
        [str(exe), "bekern2musicxml", "--input", str(bk_file), "--out", str(out_dir), "--stem", "roundtrip"],
        capture_output=True,
        text=True,
        check=True,
    )
    out_xml = out_dir / "roundtrip.musicxml"
    assert out_xml.is_file() and "<score-partwise" in out_xml.read_text(encoding="utf-8"), p2.stderr
