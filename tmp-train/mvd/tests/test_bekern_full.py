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

    # Snapshot rhythmic cursor before entering text span
    vcurs_before = list(st.music_state.vcurs_rat)

    # Open Layer 3 <sys_lyr> span; rhythmic FSA must pause and <eos>/<sys> must be blocked
    st.step("<sys_lyr>")
    assert st.in_text_span is True
    allow_txt, eos_ok = st.allowed()
    assert eos_ok is False
    assert bf.SYS_ID not in allow_txt
    assert bf.FULL_TOKEN_TO_ID["</text_span>"] in allow_txt
    assert bf.FULL_TOKEN_TO_ID["@0"] in allow_txt
    assert bf.FULL_TOKEN_TO_ID["v=1"] in allow_txt
    assert bf.FULL_TOKEN_TO_ID["lyr-beg"] in allow_txt

    for tok in ("v=1", "@0", "lyr-beg", "@1", "lyr-end"):
        st.step(tok)
    assert st.music_state.vcurs_rat == vcurs_before

    # Close text span -> rhythmic FSA resumes cleanly with 2 beats remaining!
    st.step("</text_span>")
    assert st.in_text_span is False
    assert st.music_state.vcurs_rat == vcurs_before

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
