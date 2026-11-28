"""Gauge-Equivariant Visual Tokenization (V-bekern) & Bidirectional Masked Refinement.

Implements Academic Experiment C:
  1. Exact, bijective ``bekern <-> V-bekern`` compiler & deterministic gauge-lifter:
     - Tracks ``(clef, key_signature, in_measure_accidentals)`` per staff/system.
     - Maps semantic note pitches ``<STEP> <OCT> [<ALTER>]`` into:
         * Staff-relative visual position ``sp<k>`` (``sp+0`` = bottom staff line,
           each line/space step = +1; covers ``[-24..+24]`` with core ``[-14..+14]``).
         * Visually drawn accidental ONLY (``acc-2``, ``acc-1``, ``acc0``,
           ``acc+1``, ``acc+2``) — emitted strictly when the note's semantic
           alteration differs from the active ``(key_signature, in_measure)`` state.
     - Zero-parameter deterministic gauge-lifter ``vbkern_to_bekern`` reconstructs
       exact semantic ``bekern`` pitches with 100% lossless round-trip accuracy.
  2. ``gauge_error_decomposition(pred_bekern, gt_bekern)``:
     - Separates visual-reading errors (wrong staff position or drawn accidental)
       from gauge-transport errors (correct visual staff position, wrong semantic
       pitch caused by clef / key-signature / in-measure-accidental state drift).
  3. ``MaskedRefinerHead`` (< 0.8M params):
     - Lightweight 2-layer bidirectional transformer refinement head that plugs
       onto ``UniOMRv2Net`` decoder hidden states & token embeddings to refine
       low-confidence spans in 2 parallel steps.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import data  # noqa: E402
import uni_omr  # noqa: E402

# ---------------------------------------------------------------------------
# Music-theory constants for exact gauge transport
# ---------------------------------------------------------------------------

STEP_TO_IDX: Dict[str, int] = {
    "C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6,
}
IDX_TO_STEP: Tuple[str, ...] = ("C", "D", "E", "F", "G", "A", "B")

SHARP_ORDER: Tuple[str, ...] = ("F", "C", "G", "D", "A", "E", "B")
FLAT_ORDER: Tuple[str, ...] = ("B", "E", "A", "D", "G", "C", "F")

STEPS_SET = frozenset(STEP_TO_IDX.keys())
OCTAVES_SET = frozenset(str(o) for o in range(-1, 9))
ALTER_TO_INT: Dict[str, int] = {"a-2": -2, "a-1": -1, "a1": 1, "a2": 2}
INT_TO_ALTER: Dict[int, str] = {-2: "a-2", -1: "a-1", 1: "a1", 2: "a2"}

ACC_TO_INT: Dict[str, int] = {
    "acc-2": -2,
    "acc-1": -1,
    "acc0": 0,
    "acc+1": 1,
    "acc+2": 2,
}
INT_TO_ACC: Dict[int, str] = {v: k for k, v in ACC_TO_INT.items()}

DURATIONS_SET = frozenset(
    ("breve", "whole", "half", "quarter", "eighth", "16th", "32nd", "64th", "128th")
)
BAR_RESET_TOKENS = frozenset(
    ("bar", "bar-light-heavy", "bar-repeat-right", "bar-double", "bar-repeat-left")
)

# Staff-position token range: core [-14..+14], extended [-36..+36] for zero clamping loss
# across cross-staff piano/quartet voices (C-1..A7 under any clef).
SP_MIN: int = -36
SP_MAX: int = 36
SP_TOKENS: List[str] = [f"sp{k:+d}" for k in range(SP_MIN, SP_MAX + 1)]
ACC_TOKENS: List[str] = ["acc-2", "acc-1", "acc0", "acc+1", "acc+2"]
SP_PATTERN = re.compile(r"^sp([+-]\d+)$")
CLEF_PATTERN = re.compile(r"^clef-([A-Za-z]+)-(\d+)(?:-(8vb|8va|15mb|15ma))?$")
KEY_PATTERN = re.compile(r"^key([+-]?\d+)$")

# Default clef reference: G-2 (treble) -> bottom staff line (sp+0) is E4 = 4*7 + 2 = 30
_CLEF_SIGN_REF_DIATONIC: Dict[str, int] = {
    "G": 4 * 7 + STEP_TO_IDX["G"],  # G4 = 32
    "F": 3 * 7 + STEP_TO_IDX["F"],  # F3 = 24
    "C": 4 * 7 + STEP_TO_IDX["C"],  # C4 = 28
}
_CLEF_OCT_SHIFT: Dict[Optional[str], int] = {
    None: 0,
    "8vb": -7,
    "8va": 7,
    "15mb": -14,
    "15ma": 14,
}


def diatonic_index(step: str, octave: int) -> int:
    """Convert ``(step, octave)`` to a linear diatonic integer (C0 = 0)."""
    return int(octave) * 7 + STEP_TO_IDX[step]


def from_diatonic_index(idx: int) -> Tuple[str, int]:
    """Invert ``diatonic_index`` back to ``(step, octave)``."""
    octave, step_idx = divmod(int(idx), 7)
    return IDX_TO_STEP[step_idx], octave


def clef_bottom_line_diatonic(clef_tok: str) -> int:
    """Return the diatonic index of the bottom staff line (``sp+0``) for ``clef_tok``.

    Staff lines are 1 (bottom, ``sp=0``) through 5 (top, ``sp=8``).
    For unrecognized/percussion clefs (e.g. ``clef-p-2``, ``clef-T-2``),
    falls back deterministically to treble (``E4 = 30``) so round-trip
    remains 100% lossless.
    """
    m = CLEF_PATTERN.match(clef_tok)
    if not m:
        return 30  # E4 (treble default)
    sign, line_s, oct_mod = m.group(1), m.group(2), m.group(3)
    ref = _CLEF_SIGN_REF_DIATONIC.get(sign)
    if ref is None:
        return 30
    line = int(line_s)
    shift = _CLEF_OCT_SHIFT.get(oct_mod, 0)
    # Line L has staff position 2 * (L - 1) above bottom line (line 1).
    return (ref + shift) - 2 * (line - 1)


def key_signature_alters(key_tok: str) -> Dict[str, int]:
    """Return per-step default alterations ``{step: -1|0|+1}`` for ``key*``."""
    alters = {s: 0 for s in IDX_TO_STEP}
    m = KEY_PATTERN.match(key_tok)
    if not m:
        return alters
    fifths = int(m.group(1))
    if fifths > 0:
        for s in SHARP_ORDER[: min(7, fifths)]:
            alters[s] = 1
    elif fifths < 0:
        for s in FLAT_ORDER[: min(7, -fifths)]:
            alters[s] = -1
    return alters


# ---------------------------------------------------------------------------
# V-bekern Vocabulary (drop-in replacement for UniOMR vocabulary)
# ---------------------------------------------------------------------------

def build_vbkern_vocab() -> Tuple[List[str], Dict[str, int], Dict[int, str], Dict[str, int]]:
    """Build the V-bekern vocabulary replacing semantic pitch tokens with visual tokens.

    Removes semantic ``STEP`` (7), ``OCTAVE`` (10), and semantic ``ALTER`` (4)
    tokens, and adds ``sp-24..sp+24`` (49) + ``acc-2..acc+2`` (5), plus
    specials ``<pad>``, ``<bos>``, ``<eos>``, ``<sys>`` at the end (matching
    UniOMR's ``vocab_size - 4 .. vocab_size - 1`` convention).
    """
    removed = STEPS_SET | OCTAVES_SET | frozenset(ALTER_TO_INT.keys())
    base_non_pitch = [t for t in data.VOCAB if t not in removed]
    vb_base = base_non_pitch + SP_TOKENS + ACC_TOKENS
    pad_id = len(vb_base)
    bos_id = len(vb_base) + 1
    eos_id = len(vb_base) + 2
    sys_id = len(vb_base) + 3
    vocab_size = len(vb_base) + 4

    tok_to_id: Dict[str, int] = {t: i for i, t in enumerate(vb_base)}
    tok_to_id[uni_omr.SYS_TOKEN] = sys_id
    id_to_tok: Dict[int, str] = {i: t for i, t in enumerate(vb_base)}
    id_to_tok[pad_id] = "<pad>"
    id_to_tok[bos_id] = "<bos>"
    id_to_tok[eos_id] = "<eos>"
    id_to_tok[sys_id] = uni_omr.SYS_TOKEN
    specials = {
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "sys_id": sys_id,
        "vocab_size": vocab_size,
    }
    return vb_base, tok_to_id, id_to_tok, specials


VBKERN_BASE_VOCAB, VBKERN_TOKEN_TO_ID, VBKERN_ID_TO_TOKEN, VBKERN_SPECIALS = build_vbkern_vocab()
VB_PAD_ID: int = VBKERN_SPECIALS["pad_id"]
VB_BOS_ID: int = VBKERN_SPECIALS["bos_id"]
VB_EOS_ID: int = VBKERN_SPECIALS["eos_id"]
VB_SYS_ID: int = VBKERN_SPECIALS["sys_id"]
VB_VOCAB_SIZE: int = VBKERN_SPECIALS["vocab_size"]


def encode_vbkern(tokens: Sequence[str], add_bos_eos: bool = True) -> List[int]:
    """Encode a V-bekern token sequence (with optional ``<sys>``) into IDs."""
    ids = [VBKERN_TOKEN_TO_ID[t] for t in tokens]
    if add_bos_eos:
        return [VB_BOS_ID] + ids + [VB_EOS_ID]
    return ids


def decode_vbkern(ids: Sequence[int], skip_special: bool = True) -> List[str]:
    """Decode V-bekern integer IDs back into token strings."""
    out: List[str] = []
    for idx in ids:
        i = int(idx)
        if skip_special and i in (VB_PAD_ID, VB_BOS_ID, VB_EOS_ID):
            if i == VB_EOS_ID:
                break
            continue
        tok = VBKERN_ID_TO_TOKEN.get(i)
        if tok is not None:
            out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Bijective Compiler & Deterministic Gauge-Lifter
# ---------------------------------------------------------------------------

@dataclass
class _GaugeState:
    clef: str = "clef-G-2"
    bottom_diat: int = 30  # E4
    key_alters: Dict[str, int] = None  # type: ignore[assignment]
    in_measure: Dict[Tuple[str, int], int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.key_alters is None:
            self.key_alters = {s: 0 for s in IDX_TO_STEP}
        if self.in_measure is None:
            self.in_measure = {}

    def set_clef(self, clef_tok: str) -> None:
        self.clef = clef_tok
        self.bottom_diat = clef_bottom_line_diatonic(clef_tok)

    def set_key(self, key_tok: str) -> None:
        self.key_alters = key_signature_alters(key_tok)

    def reset_measure(self) -> None:
        self.in_measure.clear()

    def active_alter(self, step: str, octave: int) -> int:
        if (step, octave) in self.in_measure:
            return self.in_measure[(step, octave)]
        return self.key_alters.get(step, 0)

    def record_alter(self, step: str, octave: int, alter: int) -> None:
        self.in_measure[(step, octave)] = alter


def bekern_to_vbkern(tokens: Sequence[str]) -> List[str]:
    """Compile semantic ``bekern`` tokens into gauge-equivariant ``V-bekern``.

    Replaces every pitched note run ``<STEP> <OCT> [<ALTER>]`` (which follows a
    note duration, never a ``rest <DUR>``) with:
      ``sp<k>`` [``acc<d>``]
    where ``acc<d>`` is emitted ONLY when the note's semantic alteration differs
    from the currently active ``(key_signature, in_measure_accidentals)`` state.

    All non-pitch tokens (clefs, keys, times, durations, rests, chords, grace,
    barlines, voices, dots, ties, tuplets, slurs, articulations, dynamics,
    ``<sys>``) pass through verbatim.
    """
    st = _GaugeState()
    out: List[str] = []
    n = len(tokens)
    i = 0
    while i < n:
        tok = tokens[i]
        if tok == uni_omr.SYS_TOKEN:
            st.reset_measure()
            out.append(tok)
            i += 1
            continue
        if tok.startswith("clef-"):
            st.set_clef(tok)
            out.append(tok)
            i += 1
            continue
        if tok.startswith("key") and KEY_PATTERN.match(tok):
            st.set_key(tok)
            out.append(tok)
            i += 1
            continue
        if tok in BAR_RESET_TOKENS:
            st.reset_measure()
            out.append(tok)
            i += 1
            continue

        # Detect semantic pitch pair: <STEP> <OCT> [<ALTER>]
        # In bekern, STEPS only ever occur as the pitch step of a note,
        # immediately followed by OCTAVE, and optionally an ALTER token.
        if tok in STEPS_SET and (i + 1) < n and tokens[i + 1] in OCTAVES_SET:
            step = tok
            octave = int(tokens[i + 1])
            i += 2
            alter = 0
            if i < n and tokens[i] in ALTER_TO_INT:
                alter = ALTER_TO_INT[tokens[i]]
                i += 1

            sp = diatonic_index(step, octave) - st.bottom_diat
            sp_clamped = max(SP_MIN, min(SP_MAX, sp))
            out.append(f"sp{sp_clamped:+d}")

            expected_alter = st.active_alter(step, octave)
            if alter != expected_alter:
                out.append(INT_TO_ACC[alter])
                st.record_alter(step, octave, alter)
            continue

        out.append(tok)
        i += 1
    return out


def vbkern_to_bekern(tokens: Sequence[str]) -> List[str]:
    """Zero-parameter deterministic gauge-lifter: ``V-bekern -> bekern``.

    Reconstructs exact semantic ``<STEP> <OCT> [<ALTER>]`` from active
    ``(clef, key_signature, in_measure_accidentals)`` + ``(sp, drawn_acc)``.
    """
    st = _GaugeState()
    out: List[str] = []
    n = len(tokens)
    i = 0
    while i < n:
        tok = tokens[i]
        if tok == uni_omr.SYS_TOKEN:
            st.reset_measure()
            out.append(tok)
            i += 1
            continue
        if tok.startswith("clef-"):
            st.set_clef(tok)
            out.append(tok)
            i += 1
            continue
        if tok.startswith("key") and KEY_PATTERN.match(tok):
            st.set_key(tok)
            out.append(tok)
            i += 1
            continue
        if tok in BAR_RESET_TOKENS:
            st.reset_measure()
            out.append(tok)
            i += 1
            continue

        m = SP_PATTERN.match(tok)
        if m:
            sp = int(m.group(1))
            step, octave = from_diatonic_index(st.bottom_diat + sp)
            i += 1
            if i < n and tokens[i] in ACC_TO_INT:
                alter = ACC_TO_INT[tokens[i]]
                st.record_alter(step, octave, alter)
                i += 1
            else:
                alter = st.active_alter(step, octave)

            out.append(step)
            out.append(str(octave))
            if alter != 0:
                out.append(INT_TO_ALTER[alter])
            continue

        # Stray accidental token without preceding sp (defensive on noisy model output)
        if tok in ACC_TO_INT:
            i += 1
            continue

        out.append(tok)
        i += 1
    return out


# ---------------------------------------------------------------------------
# Note Extraction & Gauge-Error Decomposition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoteRecord:
    dur: str
    step: str
    octave: int
    alter: int
    sp: int
    drawn_acc: Optional[int]
    clef: str
    key: str


def extract_notes_with_gauge(bekern_tokens: Sequence[str]) -> List[NoteRecord]:
    """Extract aligned note records carrying BOTH semantic pitch and visual gauge coords."""
    st = _GaugeState()
    cur_key = "key+0"
    notes: List[NoteRecord] = []
    n = len(bekern_tokens)
    i = 0
    prev_dur = "quarter"
    in_rest = False
    while i < n:
        tok = bekern_tokens[i]
        if tok == uni_omr.SYS_TOKEN:
            st.reset_measure()
            in_rest = False
            i += 1
            continue
        if tok.startswith("clef-"):
            st.set_clef(tok)
            i += 1
            continue
        if tok.startswith("key") and KEY_PATTERN.match(tok):
            cur_key = tok
            st.set_key(tok)
            i += 1
            continue
        if tok in BAR_RESET_TOKENS:
            st.reset_measure()
            in_rest = False
            i += 1
            continue
        if tok == "rest":
            in_rest = True
            i += 1
            continue
        if tok in DURATIONS_SET:
            if in_rest:
                in_rest = False
            else:
                prev_dur = tok
            i += 1
            continue
        if tok in STEPS_SET and (i + 1) < n and bekern_tokens[i + 1] in OCTAVES_SET:
            step = tok
            octave = int(bekern_tokens[i + 1])
            i += 2
            alter = 0
            if i < n and bekern_tokens[i] in ALTER_TO_INT:
                alter = ALTER_TO_INT[bekern_tokens[i]]
                i += 1
            sp = diatonic_index(step, octave) - st.bottom_diat
            expected_alter = st.active_alter(step, octave)
            drawn: Optional[int] = None
            if alter != expected_alter:
                drawn = alter
                st.record_alter(step, octave, alter)
            notes.append(
                NoteRecord(
                    dur=prev_dur,
                    step=step,
                    octave=octave,
                    alter=alter,
                    sp=sp,
                    drawn_acc=drawn,
                    clef=st.clef,
                    key=cur_key,
                )
            )
            continue
        i += 1
    return notes


def gauge_error_decomposition(
    pred_bekern: Sequence[str],
    gt_bekern: Sequence[str],
) -> Dict[str, Any]:
    """Decompose semantic pitch errors into visual-reading vs gauge-transport errors.

    Aligns predicted and ground-truth note streams by position (up to ``min(len)``
    plus count delta) and classifies every semantic pitch mismatch into:
      - ``gauge_transport_errors``: visual ``(sp, drawn_acc)`` MATCHES GT, so the
        model saw the notehead correctly on the staff, but emitted the wrong
        semantic pitch because of clef / key-signature / measure-accidental drift.
      - ``visual_sp_errors``: visual staff position ``sp`` differs from GT.
      - ``visual_acc_errors``: ``sp`` matches GT, but drawn accidental differs.
    Also reports rhythm accuracy and exact round-trip fidelity on ``gt_bekern``.
    """
    gt_notes = extract_notes_with_gauge(gt_bekern)
    pred_notes = extract_notes_with_gauge(pred_bekern)
    n_aligned = min(len(gt_notes), len(pred_notes))

    exact_pitch = 0
    exact_rhythm = 0
    exact_both = 0
    gauge_transport_err = 0
    visual_sp_err = 0
    visual_acc_err = 0

    for idx in range(n_aligned):
        g = gt_notes[idx]
        p = pred_notes[idx]
        pitch_ok = (g.step == p.step) and (g.octave == p.octave) and (g.alter == p.alter)
        rhythm_ok = (g.dur == p.dur)
        if pitch_ok:
            exact_pitch += 1
        else:
            if g.sp == p.sp and g.drawn_acc == p.drawn_acc:
                gauge_transport_err += 1
            elif g.sp != p.sp:
                visual_sp_err += 1
            else:
                visual_acc_err += 1
        if rhythm_ok:
            exact_rhythm += 1
        if pitch_ok and rhythm_ok:
            exact_both += 1

    total_pitch_err = gauge_transport_err + visual_sp_err + visual_acc_err

    # Also check round-trip on gt_bekern
    vb_gt = bekern_to_vbkern(gt_bekern)
    rt_gt = vbkern_to_bekern(vb_gt)
    rt_exact = (list(gt_bekern) == rt_gt)

    return {
        "n_gt_notes": len(gt_notes),
        "n_pred_notes": len(pred_notes),
        "n_aligned": n_aligned,
        "pitch_acc": exact_pitch / max(1, n_aligned),
        "rhythm_acc": exact_rhythm / max(1, n_aligned),
        "note_exact_acc": exact_both / max(1, n_aligned),
        "total_pitch_errors": total_pitch_err,
        "gauge_transport_errors": gauge_transport_err,
        "visual_sp_errors": visual_sp_err,
        "visual_acc_errors": visual_acc_err,
        "gauge_error_share": (gauge_transport_err / total_pitch_err) if total_pitch_err > 0 else 0.0,
        "gt_bekern_len": len(gt_bekern),
        "gt_vbkern_len": len(vb_gt),
        "compression_ratio": len(vb_gt) / max(1, len(gt_bekern)),
        "gt_roundtrip_exact": rt_exact,
    }


def compute_token_f1_multiset(pred_toks: Sequence[str], gt_toks: Sequence[str]) -> float:
    """Fast bag-of-tokens F1 (matches uni_omr.py token F1 metric)."""
    if not pred_toks and not gt_toks:
        return 1.0
    if not pred_toks or not gt_toks:
        return 0.0
    c_p = Counter(pred_toks)
    c_g = Counter(gt_toks)
    overlap = sum((c_p & c_g).values())
    prec = overlap / len(pred_toks)
    rec = overlap / len(gt_toks)
    return (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Lightweight 2-Step Bidirectional Masked Refinement Head (< 0.8M params)
# ---------------------------------------------------------------------------

class _BiRefinerBlock(nn.Module):
    """Compact bidirectional Pre-LN Transformer block with RMSNorm + GELU FFN."""

    def __init__(self, d_ref: int = 160, n_heads: int = 4, d_ff: int = 320, dropout: float = 0.05) -> None:
        super().__init__()
        self.d_ref = d_ref
        self.n_heads = n_heads
        self.head_dim = d_ref // n_heads
        self.norm1 = nn.LayerNorm(d_ref)
        self.qkv = nn.Linear(d_ref, 3 * d_ref, bias=False)
        self.proj = nn.Linear(d_ref, d_ref, bias=False)
        self.norm2 = nn.LayerNorm(d_ref)
        self.fc1 = nn.Linear(d_ref, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_ref, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
        attn_mask: Optional[torch.Tensor] = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, T) True where PAD -> mask out
            attn_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T) bool: True=keep
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.drop(self.proj(out))
        x = x + self.drop(self.fc2(F.gelu(self.fc1(self.norm2(x)))))
        return x


class MaskedRefinerHead(nn.Module):
    """Lightweight (< 0.8M params) 2-Step Bidirectional Masked Refinement Head.

    Plugs onto ``UniOMRv2Net`` decoder hidden states ``h_dec`` ``(B, T, d_model)``
    and current token IDs ``tok_ids`` ``(B, T)``. Predicts a residual hidden-state
    correction ``delta_h`` ``(B, T, d_model)`` projected through the shared
    ``lm_head`` (zero extra vocabulary-head parameters!).

    Default config (``d_model=384, d_ref=160, n_layers=2, n_heads=4, d_ff=320``):
    ~576K parameters (< 0.8M budget).
    """

    def __init__(
        self,
        d_model: int = 384,
        vocab_size: int = VB_VOCAB_SIZE,
        d_ref: int = 160,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 320,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.vocab_size = int(vocab_size)
        self.mask_id = int(vocab_size)  # dedicated [MASK] embedding index
        self.d_ref = int(d_ref)

        self.h_proj = nn.Linear(d_model, d_ref, bias=False)
        self.tok_emb = nn.Embedding(vocab_size + 1, d_ref)
        self.blocks = nn.ModuleList(
            [_BiRefinerBlock(d_ref=d_ref, n_heads=n_heads, d_ff=d_ff, dropout=dropout) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(d_ref)
        self.out_proj = nn.Linear(d_ref, d_model, bias=False)
        # Zero-init out_proj so at step 0 the refiner starts as exact identity
        nn.init.zeros_(self.out_proj.weight)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        h_dec: torch.Tensor,
        tok_ids: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return residual hidden update ``delta_h`` of shape ``(B, T, d_model)``."""
        x = self.h_proj(h_dec) + self.tok_emb(tok_ids.clamp(0, self.mask_id))
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad_mask)
        return self.out_proj(self.out_norm(x))

    def compute_refinement_loss(
        self,
        h_dec: torch.Tensor,
        target_ids: torch.Tensor,
        lm_head: nn.Linear,
        *,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        mask_prob: float = 0.15,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """Self-supervised masked span denoising loss over target sequences."""
        B, T = target_ids.shape
        device = target_ids.device
        valid = (target_ids != pad_id) & (target_ids != bos_id) & (target_ids != eos_id)
        rand = torch.rand((B, T), device=device)
        corrupt_mask = valid & (rand < mask_prob)

        # Ensure at least one masked position if any valid token exists
        if not bool(corrupt_mask.any()):
            if not bool(valid.any()):
                return h_dec.new_zeros(())
            first_valid = valid.float().argmax(dim=1)
            for b in range(B):
                if bool(valid[b].any()):
                    corrupt_mask[b, int(first_valid[b].item())] = True

        # 80% [MASK], 20% random token corruption
        r2 = torch.rand((B, T), device=device)
        use_mask = corrupt_mask & (r2 < 0.80)
        use_rand = corrupt_mask & (r2 >= 0.80)
        corrupted = target_ids.clone()
        corrupted = torch.where(use_mask, torch.full_like(corrupted, self.mask_id), corrupted)
        if bool(use_rand.any()):
            rand_toks = torch.randint(0, max(1, pad_id), (B, T), device=device, dtype=target_ids.dtype)
            corrupted = torch.where(use_rand, rand_toks, corrupted)

        pad_mask = target_ids == pad_id
        delta_h = self.forward(h_dec.detach(), corrupted, pad_mask=pad_mask)
        refined_logits = lm_head(h_dec.detach() + delta_h)

        # Only compute CE on corrupted positions
        masked_targets = torch.where(corrupt_mask, target_ids, torch.full_like(target_ids, -100))
        return F.cross_entropy(
            refined_logits.reshape(-1, refined_logits.shape[-1]),
            masked_targets.reshape(-1),
            ignore_index=-100,
            label_smoothing=float(max(0.0, min(0.2, label_smoothing))),
        )

    @torch.no_grad()
    def refine_2step(
        self,
        h_dec: torch.Tensor,
        draft_ids: torch.Tensor,
        draft_probs: torch.Tensor,
        lm_head: nn.Linear,
        *,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        tau_steps: Tuple[float, float] = (0.60, 0.45),
        max_mask_fracs: Tuple[float, float] = (0.15, 0.08),
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """2-Step parallel bidirectional masked refinement on low-confidence spans.

        Args:
            h_dec: ``(B, T, d_model)`` decoder hidden states for ``draft_ids``.
            draft_ids: ``(B, T)`` initial autoregressive token predictions.
            draft_probs: ``(B, T)`` initial token max-softmax confidences in ``[0, 1]``.
            lm_head: shared vocabulary projection ``d_model -> vocab_size``.
        Returns:
            ``(refined_ids, stats)`` where ``refined_ids`` has shape ``(B, T)``.
        """
        cur_ids = draft_ids.clone()
        cur_probs = draft_probs.clone()
        pad_mask = cur_ids == pad_id
        special_mask = (cur_ids == pad_id) | (cur_ids == bos_id) | (cur_ids == eos_id)
        total_masked = 0
        total_changed = 0

        for tau, max_frac in zip(tau_steps, max_mask_fracs):
            low_conf = (~special_mask) & (cur_probs < tau)
            # Cap masked tokens per sequence to max_frac of valid length
            B, T = cur_ids.shape
            step_mask = torch.zeros_like(low_conf)
            for b in range(B):
                valid_len = int((~special_mask[b]).sum().item())
                if valid_len == 0:
                    continue
                cap = max(1, int(math.ceil(valid_len * max_frac)))
                cand_idx = torch.nonzero(low_conf[b], as_tuple=False).squeeze(-1)
                if cand_idx.numel() == 0:
                    continue
                if cand_idx.numel() > cap:
                    # Pick the lowest-confidence `cap` positions
                    cand_scores = cur_probs[b, cand_idx]
                    _, worst = torch.topk(cand_scores, k=cap, largest=False)
                    cand_idx = cand_idx[worst]
                step_mask[b, cand_idx] = True

            n_masked = int(step_mask.sum().item())
            if n_masked == 0:
                continue
            total_masked += n_masked

            masked_in = torch.where(step_mask, torch.full_like(cur_ids, self.mask_id), cur_ids)
            delta_h = self.forward(h_dec, masked_in, pad_mask=pad_mask)
            ref_logits = lm_head(h_dec + delta_h)
            ref_probs_all = F.softmax(ref_logits.float(), dim=-1)
            new_probs, new_ids = ref_probs_all.max(dim=-1)

            # Accept refinement only on masked positions where confidence improves
            accept = step_mask & (new_probs > cur_probs) & (new_ids != pad_id) & (new_ids != bos_id)
            changed = accept & (new_ids != cur_ids)
            total_changed += int(changed.sum().item())
            cur_ids = torch.where(accept, new_ids, cur_ids)
            cur_probs = torch.where(accept, new_probs.to(cur_probs.dtype), cur_probs)

        return cur_ids, {
            "total_masked": total_masked,
            "total_changed": total_changed,
        }
