"""bekern-full Pillar 2 — 4-Layer Vocabulary, Loss Masking, Guards, UTF-8 Codec & FSA.

Implements Pillar 2 of the hardened ``bekern-full`` specification
(``docs/BEKERN_FULL_SPEC.md``) for Iter 16 while preserving **100% strict
backward compatibility** with Iter 15's frozen ``data.py`` (``len(VOCAB)=261``,
``PAD=261``, ``BOS=262``, ``EOS=263``) and ``uni_omr.py`` (``SYS=264``,
``VOCAB_SIZE=265``):

1. **Append-Only 4-Layer Vocabulary** (``build_bekern_full_vocab``):
   - Indices ``0..264`` are byte-for-byte identical to ``data.VOCAB`` +
     ``<pad>=261``, ``<bos>=262``, ``<eos>=263``, ``<sys>=264``.
   - Indices ``>= 265`` append schema prefixes, Layer 0 score structure,
     Layer 1 extended notation, Layer 2 lossless expressives & numbered
     spanners, and Layer 3 text mode-switch / anchor / verse tokens.
2. **4-Bit Layer Bitmask & Unannotated-Layer CE Loss Masking** (Mitigation #1):
   - Bit layout (MSB -> LSB): ``L0=bit3 (0b1000)``, ``L1=bit2 (0b0100)``,
     ``L2=bit1 (0b0010)``, ``L3=bit0 (0b0001)``.
   - ``SCHEMA_LAYER_BITS = {"v1": 0b0100, "v2a": 0b0110, "full": 0b1111}``.
   - ``build_layer_ce_masks`` zeroes CE loss on unannotated-layer targets AND
     builds a negative-logit suppression mask (``-inf`` before softmax) so
     unannotated real handwritten corpora never suffer false-negative poisoning.
3. **MTP-4 Mode-Switch Guard & BD3-LM ``is_music_body`` Mask** (Mitigations #6 & #7):
   - ``MODE_SWITCH_TOKEN_IDS`` / ``is_mode_switch_token`` disables MTP-4
     speculative drafting across music<->text head switches.
   - ``compute_is_music_body_mask`` restricts BD3-LM block-diffusion updates
     strictly to Layers 0-2 music body tokens (never ``<bos>``, ``<eos>``,
     ``<pad>``, ``<sys>``, ``<schema:*>``, or Layer 3 text spans).
4. **Layer 3 UTF-8 Byte-Level Text Codec** (``TEXT_VOCAB_SIZE = 259``):
   - ``0..255`` raw UTF-8 bytes + ``256=TEXT_BOS``, ``257=TEXT_EOS``,
     ``258=TEXT_PAD`` for lossless ASCII + Unicode lyric/text round-trips.
5. **``BekernFullGrammarState`` FSA** (Mitigations #3 & #8):
   - Wraps ``grammar.PrefixState`` for core music tokens, allows optional
     ``<schema:*>`` only at position 0, pauses the rhythmic bar-duration FSA
     inside Layer 3 text spans (``<sys_lyr>``/``<harm>``/``<tempo>``/``<dir>``/
     ``<reh>``/``<fb>`` ... ``</text_span>``), and enforces numbered spanner
     channel parity for ``slur-1..4`` and ``hairpin-1..3``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import data  # noqa: E402
import grammar  # noqa: E402

__all__ = [
    # Special & boundary IDs
    "PAD_ID",
    "BOS_ID",
    "EOS_ID",
    "SYS_ID",
    "LEGACY_VOCAB_SIZE",
    "MUSIC_VOCAB_END",
    "FULL_VOCAB_SIZE",
    "FULL_VOCAB",
    "FULL_TOKEN_TO_ID",
    "FULL_ID_TO_TOKEN",
    "build_bekern_full_vocab",
    # Token groups
    "SCHEMA_TOKENS",
    "SCHEMA_TOKEN_IDS",
    "LAYER0_TOKENS",
    "LAYER1_NEW_TOKENS",
    "LAYER2_NEW_TOKENS",
    "LAYER3_TOKENS",
    "TEXT_SPAN_OPEN_TOKENS",
    "TEXT_SPAN_CLOSE_TOKEN",
    "TEXT_SPAN_OPEN_TOKEN_IDS",
    "TEXT_SPAN_CLOSE_TOKEN_ID",
    "MODE_SWITCH_TOKENS",
    "MODE_SWITCH_TOKEN_IDS",
    "VALID_MUSIC_BODY_TOKEN_IDS",
    "is_mode_switch_token",
    # Layer bitmask & CE masking
    "LAYER_BIT_L0",
    "LAYER_BIT_L1",
    "LAYER_BIT_L2",
    "LAYER_BIT_L3",
    "SCHEMA_LAYER_BITS",
    "SCHEMA_TOKEN_TO_BITS",
    "TOKEN_LAYER_ID",
    "token_layer_of",
    "build_layer_ce_masks",
    "masked_layer_cross_entropy",
    # BD3-LM body mask
    "compute_is_music_body_mask",
    # Layer 3 UTF-8 byte codec
    "TEXT_BOS_ID",
    "TEXT_EOS_ID",
    "TEXT_PAD_ID",
    "TEXT_VOCAB_SIZE",
    "encode_text_span_utf8",
    "decode_text_span_utf8",
    # Encode / decode helpers
    "encode_full",
    "decode_full",
    "encode_bekern_full_sequence",
    "decode_bekern_full_sequence",
    # Grammar FSA
    "MAX_LAYER3_SPAN_TOKENS",
    "BekernFullGrammarState",
]

# ---------------------------------------------------------------------------
# 1. Exact legacy special IDs (100% identical to data.py + uni_omr.py)
# ---------------------------------------------------------------------------

PAD_ID: int = data.PAD_ID          # 261
BOS_ID: int = data.BOS_ID          # 262
EOS_ID: int = data.EOS_ID          # 263
SYS_ID: int = len(data.VOCAB) + 3  # 264
LEGACY_VOCAB_SIZE: int = len(data.VOCAB) + 4  # 265

assert PAD_ID == 261 and BOS_ID == 262 and EOS_ID == 263 and SYS_ID == 264, (
    f"Unexpected legacy special IDs: PAD={PAD_ID}, BOS={BOS_ID}, EOS={EOS_ID}, SYS={SYS_ID}"
)

# ---------------------------------------------------------------------------
# 2. Explicit 4-layer token groups
# ---------------------------------------------------------------------------

SCHEMA_TOKENS: Tuple[str, ...] = (
    "<schema:v1>",
    "<schema:v2a>",
    "<schema:full>",
)

LAYER0_TOKENS: Tuple[str, ...] = (
    "<part>",
    "<staff-def>",
    "<group>",
    "<transpose>",
    "<pickup>",
    "<multirest>",
    "mrep-1",
    "mrep-2",
    "mrep-4",
    "<inst-change>",
    "D.C.",
    "D.S.",
    "Segno",
    "Coda",
    "Fine",
    # Finite Layer 0 structural / grouping / staff-line / transpose / count arguments
    "brace",
    "bracket",
    "barline",
    "lines-1", "lines-2", "lines-3", "lines-4", "lines-5", "lines-6",
    *(str(i) for i in range(-24, -1)),
    "+1", "+2", "+3", "+4", "+5", "+6", "+7", "+8", "+9", "+10", "+11", "+12",
    *(str(i) for i in range(9, 25)),
    ":",
)

LAYER1_NEW_TOKENS: Tuple[str, ...] = (
    "tup-open",
    "tup-close",
    "q#",
    "qb",
    "3q#",
    "3qb",
    "a0",
    "acc-caut",
    "acc-paren",
    "<chord>",
    "stem-up",
    "stem-down",
    "beam-start",
    "beam-cont",
    "beam-end",
    "beam-hook-l",
    "beam-hook-r",
    "unpitched",
    "nh-x",
    "nh-diamond",
    "nh-slash",
    "nh-triangle",
    "nh-circle-x",
    "nh-square",
    "str",
    "fret",
    "bend",
    "hopo",
    "grace-slash",
    "xstaff-1",
    "xstaff-2",
    "xstaff-3",
    "xstaff-4",
    "xstaff-5",
    "xstaff-6",
    "xstaff-7",
    "xstaff-8",
    # Finite Layer 1 TAB string/fret, TAB clef, and extended cross-staff arguments
    "str-1", "str-2", "str-3", "str-4", "str-5", "str-6",
    *(f"fret-{i}" for i in range(25)),
    "clef-T-4", "clef-T-5", "clef-T-6",
    "xstaff-9", "xstaff-10", "xstaff-11", "xstaff-12",
    "xstaff-13", "xstaff-14", "xstaff-15", "xstaff-16",
)

LAYER2_NEW_TOKENS: Tuple[str, ...] = (
    "marcato",
    "mordent-up",
    "mordent-lo",
    "turn",
    "turn-inv",
    "tremolo-1",
    "tremolo-2",
    "tremolo-3",
    "tremolo-2n",
    "15ma-open",
    "15ma-close",
    "15mb-open",
    "15mb-close",
    "15-stop",
    "pedal-sost-open",
    "pedal-sost-close",
    "pedal-uc-open",
    "pedal-uc-close",
    "arp-up",
    "arp-down",
    "arp-non",
    "trill-line-open",
    "trill-line-close",
    "gliss-open",
    "gliss-close",
    # Numbered slurs: slur-1..4 open/close
    "slur-1-open", "slur-1-close",
    "slur-2-open", "slur-2-close",
    "slur-3-open", "slur-3-close",
    "slur-4-open", "slur-4-close",
    # Numbered hairpins: hairpin-1..3 cresc/dim/close
    "hairpin-1-cresc", "hairpin-1-dim", "hairpin-1-close",
    "hairpin-2-cresc", "hairpin-2-dim", "hairpin-2-close",
    "hairpin-3-cresc", "hairpin-3-dim", "hairpin-3-close",
    "breath",
    "caesura",
    "fing-1", "fing-2", "fing-3", "fing-4", "fing-5",
    "harmonics",
)

TEXT_SPAN_OPEN_TOKENS: Tuple[str, ...] = (
    "<sys_lyr>",
    "<harm>",
    "<tempo>",
    "<dir>",
    "<reh>",
    "<fb>",
)
TEXT_SPAN_CLOSE_TOKEN: str = "</text_span>"

LAYER3_LYRIC_MARKERS: Tuple[str, ...] = (
    "lyr-beg",
    "lyr-mid",
    "lyr-end",
    "lyr-ext",
)
LAYER3_ANCHORS: Tuple[str, ...] = tuple(f"@{i}" for i in range(256))
LAYER3_VERSES: Tuple[str, ...] = tuple(f"v={i}" for i in range(1, 9))

LAYER3_TOKENS: Tuple[str, ...] = (
    TEXT_SPAN_OPEN_TOKENS
    + (TEXT_SPAN_CLOSE_TOKEN,)
    + LAYER3_LYRIC_MARKERS
    + LAYER3_ANCHORS
    + LAYER3_VERSES
)

MODE_SWITCH_TOKENS: Tuple[str, ...] = TEXT_SPAN_OPEN_TOKENS + (TEXT_SPAN_CLOSE_TOKEN,)


def build_bekern_full_vocab() -> Tuple[List[str], Dict[str, int], Dict[int, str], int]:
    """Build the append-only ``bekern-full`` vocabulary preserving ``0..264``.

    Returns:
        ``(vocab_list, token_to_id, id_to_token, music_vocab_end)`` where:
        - ``vocab_list[0..260]`` == ``data.VOCAB[0..260]``
        - ``vocab_list[261..264]`` == ``["<pad>", "<bos>", "<eos>", "<sys>"]``
        - ``music_vocab_end`` is the exclusive upper bound of Layers 0-2 music
          tokens (immediately before Layer 3 mode-switch / anchor / verse tokens).
    """
    vocab: List[str] = list(data.VOCAB) + ["<pad>", "<bos>", "<eos>", "<sys>"]
    assert len(vocab) == LEGACY_VOCAB_SIZE
    seen: Set[str] = set(vocab)

    def _append_unique(tokens: Sequence[str]) -> None:
        for tok in tokens:
            if tok not in seen:
                vocab.append(tok)
                seen.add(tok)

    _append_unique(SCHEMA_TOKENS)
    _append_unique(LAYER0_TOKENS)
    _append_unique(LAYER1_NEW_TOKENS)
    _append_unique(LAYER2_NEW_TOKENS)
    music_vocab_end = len(vocab)
    _append_unique(LAYER3_TOKENS)

    token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(vocab)}
    id_to_token: Dict[int, str] = {i: tok for i, tok in enumerate(vocab)}
    assert len(token_to_id) == len(vocab), "duplicate token in bekern-full vocab"
    return vocab, token_to_id, id_to_token, music_vocab_end


FULL_VOCAB, FULL_TOKEN_TO_ID, FULL_ID_TO_TOKEN, MUSIC_VOCAB_END = build_bekern_full_vocab()
FULL_VOCAB_SIZE: int = len(FULL_VOCAB)

SCHEMA_TOKEN_IDS: FrozenSet[int] = frozenset(FULL_TOKEN_TO_ID[t] for t in SCHEMA_TOKENS)
TEXT_SPAN_OPEN_TOKEN_IDS: FrozenSet[int] = frozenset(
    FULL_TOKEN_TO_ID[t] for t in TEXT_SPAN_OPEN_TOKENS
)
TEXT_SPAN_CLOSE_TOKEN_ID: int = FULL_TOKEN_TO_ID[TEXT_SPAN_CLOSE_TOKEN]
MODE_SWITCH_TOKEN_IDS: FrozenSet[int] = frozenset(
    FULL_TOKEN_TO_ID[t] for t in MODE_SWITCH_TOKENS
)


def is_mode_switch_token(tok_id: int) -> bool:
    """Return True iff ``tok_id`` is a Layer 3 music<->text mode-switch token.

    Used by the MTP-4 speculative drafting guard (Mitigation #6) to disable
    multi-token drafting across music<->text head transitions.
    """
    return int(tok_id) in MODE_SWITCH_TOKEN_IDS


# ---------------------------------------------------------------------------
# 3. 4-Bit Layer Bitmask & Unannotated-Layer CE Loss Masking (Mitigation #1)
# ---------------------------------------------------------------------------
# Bit order (MSB -> LSB):
#   bit 3 (0b1000 = 8): Layer 0 — Score structure, multi-part & navigation
#   bit 2 (0b0100 = 4): Layer 1 — Core notation, microtones, beams, perc, tab
#   bit 1 (0b0010 = 2): Layer 2 — Expressive marks, articulations, spanners
#   bit 0 (0b0001 = 1): Layer 3 — End-of-system text, lyrics, harmony, anchors

LAYER_BIT_L0: int = 0b1000
LAYER_BIT_L1: int = 0b0100
LAYER_BIT_L2: int = 0b0010
LAYER_BIT_L3: int = 0b0001

SCHEMA_LAYER_BITS: Dict[str, int] = {
    "v1": 0b0100,    # L1 only (legacy bekern / real HW without expressive GT)
    "v2a": 0b0110,   # L1 + L2 (core notation + expressive marks)
    "full": 0b1111,  # L0 + L1 + L2 + L3 (full 4-layer supervision)
}

SCHEMA_TOKEN_TO_BITS: Dict[str, int] = {
    "<schema:v1>": SCHEMA_LAYER_BITS["v1"],
    "<schema:v2a>": SCHEMA_LAYER_BITS["v2a"],
    "<schema:full>": SCHEMA_LAYER_BITS["full"],
}

_UNIVERSAL_TOKENS: FrozenSet[str] = frozenset(
    ("<pad>", "<bos>", "<eos>", "<sys>") + SCHEMA_TOKENS
)
_LAYER0_SET: FrozenSet[str] = frozenset(LAYER0_TOKENS)
_LAYER2_LEGACY_EXACT: FrozenSet[str] = frozenset(
    ("slur-start", "slur-stop", "stacc", "accent", "tenuto", "fermata")
)
_LAYER2_NEW_SET: FrozenSet[str] = frozenset(LAYER2_NEW_TOKENS)
_LAYER3_SET: FrozenSet[str] = frozenset(LAYER3_TOKENS)


def _classify_token_layer(tok: str) -> int:
    """Return layer index in ``{0, 1, 2, 3}`` or ``-1`` for universal specials."""
    if tok in _UNIVERSAL_TOKENS:
        return -1
    if tok in _LAYER0_SET:
        return 0
    if tok in _LAYER3_SET:
        return 3
    if (
        tok in _LAYER2_NEW_SET
        or tok in _LAYER2_LEGACY_EXACT
        or tok.startswith("dyn-")
        or tok.startswith("hairpin-")
        or tok.startswith("art-")
        or tok.startswith("ottava-")
    ):
        return 2
    # All remaining tokens in 0..260 (pitches, durations, clefs, keys, times,
    # bars, voices, tuplets, <staff>, xstaff-*, cue, volta-*) plus Layer 1
    # extensions belong to Layer 1.
    return 1


TOKEN_LAYER_ID: Tuple[int, ...] = tuple(
    _classify_token_layer(tok) for tok in FULL_VOCAB
)


def token_layer_of(tok_or_id: Union[int, str]) -> int:
    """Return the layer ID (``0..3``, or ``-1`` for universal specials)."""
    if isinstance(tok_or_id, str):
        return TOKEN_LAYER_ID[FULL_TOKEN_TO_ID[tok_or_id]]
    tid = int(tok_or_id)
    if 0 <= tid < len(TOKEN_LAYER_ID):
        return TOKEN_LAYER_ID[tid]
    return -1


def build_layer_ce_masks(
    target_ids,
    layer_masks,
    vocab_size: int = FULL_VOCAB_SIZE,
):
    """Build target CE mask and negative-logit suppression mask (Mitigation #1).

    Args:
        target_ids: ``torch.Tensor`` of shape ``(B, T)`` or ``(T,)`` containing
            target vocabulary IDs (may include ``PAD_ID`` or ``-100``).
        layer_masks: ``torch.Tensor`` (or int/sequence) of 4-bit layer bitmasks
            of shape ``(B,)`` (or scalar when ``target_ids`` is 1D).
            Bit order: ``L0=bit3 (0b1000)``, ``L1=bit2 (0b0100)``,
            ``L2=bit1 (0b0010)``, ``L3=bit0 (0b0001)``.
        vocab_size: Vocabulary size of the logits tensor (default ``FULL_VOCAB_SIZE``).

    Returns:
        ``(target_mask, logit_suppress_mask)``:
        - ``target_mask`` (``BoolTensor``, same shape as ``target_ids``):
          ``True`` iff the target token is non-PAD and belongs to an annotated
          layer (or is a universal non-PAD special token such as ``<eos>``,
          ``<sys>``, ``<schema:*>``).
        - ``logit_suppress_mask`` (``BoolTensor``, shape ``(B, vocab_size)`` or
          ``(vocab_size,)``): ``True`` for every vocab ID belonging to an
          **unannotated** layer for that batch row. Apply
          ``logits.masked_fill(logit_suppress_mask[:, None, :], float('-inf'))``
          before softmax so unannotated-layer logits receive zero negative
          gradient (curing Iter 14's 43.9% false-negative poisoning).
    """
    import torch

    tgt = target_ids if isinstance(target_ids, torch.Tensor) else torch.as_tensor(target_ids, dtype=torch.long)
    squeezed = tgt.ndim == 1
    if squeezed:
        tgt_2d = tgt.unsqueeze(0)
    elif tgt.ndim == 2:
        tgt_2d = tgt
    else:
        raise ValueError(f"target_ids must be 1D or 2D, got shape {tuple(tgt.shape)}")

    bsz = tgt_2d.shape[0]
    device = tgt_2d.device

    lm = layer_masks if isinstance(layer_masks, torch.Tensor) else torch.as_tensor(layer_masks, dtype=torch.long, device=device)
    lm = lm.to(device=device, dtype=torch.long)
    if lm.ndim == 0:
        lm = lm.expand(bsz)
    elif lm.ndim == 1 and lm.shape[0] == bsz:
        pass
    else:
        raise ValueError(f"layer_masks shape {tuple(lm.shape)} incompatible with batch size {bsz}")

    v = int(vocab_size)
    if v <= len(TOKEN_LAYER_ID):
        layer_list = list(TOKEN_LAYER_ID[:v])
    else:
        layer_list = list(TOKEN_LAYER_ID) + [-1] * (v - len(TOKEN_LAYER_ID))
    vocab_layers = torch.tensor(layer_list, dtype=torch.long, device=device)  # (V,)

    # For each layer l in {0, 1, 2, 3}, bit value is 1 << (3 - l).
    # For universal tokens (layer == -1), bit_for_vocab is 0 -> always active.
    is_layered = vocab_layers >= 0  # (V,)
    vocab_bits = torch.where(
        is_layered,
        torch.bitwise_left_shift(torch.ones_like(vocab_layers), 3 - vocab_layers.clamp(min=0)),
        torch.zeros_like(vocab_layers),
    )  # (V,)

    # (B, V): True iff vocab token v is active (either universal or its layer bit is set).
    active_vocab = (~is_layered).unsqueeze(0) | ((lm.unsqueeze(1) & vocab_bits.unsqueeze(0)) != 0)
    logit_suppress_mask = ~active_vocab  # (B, V)

    # Target mask: non-PAD, valid index in [0, V), and active in its batch row.
    valid_id = (tgt_2d >= 0) & (tgt_2d < v) & (tgt_2d != PAD_ID)
    safe_tgt = tgt_2d.clamp(min=0, max=max(0, v - 1))
    tgt_active = active_vocab.gather(1, safe_tgt)
    target_mask = valid_id & tgt_active

    if squeezed:
        return target_mask.squeeze(0), logit_suppress_mask.squeeze(0)
    return target_mask, logit_suppress_mask


def masked_layer_cross_entropy(
    logits,
    target_ids,
    layer_masks,
    label_smoothing: float = 0.0,
    weight=None,
):
    """Compute layer-masked CE loss applying both target and logit-suppression masks.

    When ``label_smoothing > 0.0``, computes manual ``log_softmax`` and averages the
    uniform smoothing term **strictly over active (unsuppressed) classes** so
    ``-inf`` suppressed classes never produce ``loss = +inf`` or biased gradients.
    """
    import torch
    import torch.nn.functional as F

    bsz, seq_len, vocab_size = logits.shape
    target_mask, logit_suppress_mask = build_layer_ce_masks(
        target_ids, layer_masks, vocab_size=vocab_size
    )
    masked_logits = logits.masked_fill(logit_suppress_mask.unsqueeze(1), float("-inf"))
    if not bool(target_mask.any()):
        return logits.sum() * 0.0

    eps = float(label_smoothing)
    if eps <= 0.0:
        masked_targets = torch.where(
            target_mask,
            target_ids.to(dtype=torch.long),
            torch.full_like(target_ids, -100, dtype=torch.long),
        )
        w = weight.to(device=logits.device, dtype=logits.dtype) if weight is not None else None
        return F.cross_entropy(
            masked_logits.reshape(bsz * seq_len, vocab_size),
            masked_targets.reshape(bsz * seq_len),
            weight=w,
            ignore_index=-100,
            label_smoothing=0.0,
        )

    # Manual active-class-only label smoothing in float32 for numerical stability.
    active_mask = ~logit_suppress_mask  # (B, V)
    log_probs = F.log_softmax(masked_logits.float(), dim=-1)  # (B, T, V)
    safe_tgt = target_ids.to(dtype=torch.long).clamp(min=0, max=max(0, vocab_size - 1))

    # Zero out suppressed (-inf) entries BEFORE gathering nll or summing so suppressed
    # targets never produce +inf * 0 = NaN.
    active_lp = torch.where(active_mask.unsqueeze(1), log_probs, torch.zeros_like(log_probs))
    nll = -active_lp.gather(dim=-1, index=safe_tgt.unsqueeze(-1)).squeeze(-1)  # (B, T)
    active_cnt = active_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(log_probs.dtype)  # (B, 1)
    smooth_loss = -active_lp.sum(dim=-1) / active_cnt  # (B, T)

    per_tok = (1.0 - eps) * nll + eps * smooth_loss  # (B, T)
    per_tok = torch.where(target_mask, per_tok, torch.zeros_like(per_tok))
    valid_f = target_mask.to(per_tok.dtype)
    if weight is not None:
        w_f = weight.to(device=logits.device, dtype=per_tok.dtype)
        tok_w = torch.where(target_mask, w_f[safe_tgt], torch.zeros_like(per_tok))
        loss = (per_tok * tok_w).sum() / tok_w.sum().clamp(min=1e-12)
    else:
        loss = per_tok.sum() / valid_f.sum().clamp(min=1.0)
    return loss.to(logits.dtype)


# ---------------------------------------------------------------------------
# 4. BD3-LM `is_music_body` Mask (Mitigation #7)
# ---------------------------------------------------------------------------

_NON_BODY_SPECIAL_IDS: FrozenSet[int] = frozenset(
    (PAD_ID, BOS_ID, EOS_ID, SYS_ID)
) | SCHEMA_TOKEN_IDS

VALID_MUSIC_BODY_TOKEN_IDS: Tuple[int, ...] = tuple(
    i
    for i in range(MUSIC_VOCAB_END)
    if i not in _NON_BODY_SPECIAL_IDS and TOKEN_LAYER_ID[i] in (0, 1, 2)
)

_IS_BODY_LUT_CACHE: Dict[Any, Any] = {}
_OPEN_IDS_TENSOR_CACHE: Dict[Any, Any] = {}


def _get_body_luts(device):
    """Return device-cached ``(is_body_lut, is_open_lut)`` boolean tensors of size ``FULL_VOCAB_SIZE``."""
    import torch

    cached = _IS_BODY_LUT_CACHE.get(device)
    if cached is not None:
        return cached
    body_lut = torch.zeros(FULL_VOCAB_SIZE, dtype=torch.bool, device=device)
    if VALID_MUSIC_BODY_TOKEN_IDS:
        body_lut[list(VALID_MUSIC_BODY_TOKEN_IDS)] = True
    open_lut = torch.zeros(FULL_VOCAB_SIZE, dtype=torch.bool, device=device)
    if TEXT_SPAN_OPEN_TOKEN_IDS:
        open_lut[sorted(TEXT_SPAN_OPEN_TOKEN_IDS)] = True
    val = (body_lut, open_lut)
    _IS_BODY_LUT_CACHE[device] = val
    return val


def compute_is_music_body_mask(token_ids):
    """Return BoolTensor True ONLY for Layers 0-2 music body tokens (Mitigation #7).

    Strictly ``False`` for:
    - ``<pad>``, ``<bos>``, ``<eos>``, ``<sys>``, and ``<schema:*>`` prefixes;
    - Any token inside a Layer 3 text span (from ``<sys_lyr>`` / ``<harm>`` /
      ``<tempo>`` / ``<dir>`` / ``<reh>`` / ``<fb>`` through and including
      ``</text_span>``, tracking nested ``span_depth`` cleanly);
    - Any stray Layer 3 token outside an explicit span.

    Implemented purely on-device via vectorized PyTorch ``cumsum`` + ``cummin``
    with zero GPU->CPU synchronization.
    """
    import torch

    toks = token_ids if isinstance(token_ids, torch.Tensor) else torch.as_tensor(token_ids, dtype=torch.long)
    squeezed = toks.ndim == 1
    if squeezed:
        toks_2d = toks.unsqueeze(0)
    elif toks.ndim == 2:
        toks_2d = toks
    else:
        raise ValueError(f"token_ids must be 1D or 2D, got shape {tuple(toks.shape)}")

    device = toks_2d.device
    body_lut, open_lut = _get_body_luts(device)
    in_range = (toks_2d >= 0) & (toks_2d < FULL_VOCAB_SIZE)
    safe_ids = toks_2d.clamp(min=0, max=FULL_VOCAB_SIZE - 1)

    is_open = in_range & open_lut[safe_ids]
    is_close = in_range & (safe_ids == TEXT_SPAN_CLOSE_TOKEN_ID)

    # Exact running span depth d_t = max(0, d_{t-1} + open_t - close_t) on-device:
    delta = is_open.to(torch.int64) - is_close.to(torch.int64)
    raw_depth = delta.cumsum(dim=1)
    if raw_depth.shape[1] > 0:
        min_prefix = torch.cummin(raw_depth, dim=1).values.clamp(max=0)
        span_depth = raw_depth - min_prefix
    else:
        span_depth = raw_depth

    in_or_edge_span = (span_depth > 0) | is_open | is_close
    is_body_tok = in_range & body_lut[safe_ids]
    mask = is_body_tok & (~in_or_edge_span)
    return mask.squeeze(0) if squeezed else mask


# ---------------------------------------------------------------------------
# 5. Layer 3 UTF-8 Byte-Level Text Codec (TEXT_VOCAB_SIZE = 259)
# ---------------------------------------------------------------------------

TEXT_BOS_ID: int = 256
TEXT_EOS_ID: int = 257
TEXT_PAD_ID: int = 258
TEXT_VOCAB_SIZE: int = 259


def encode_text_span_utf8(text: str, add_bos_eos: bool = True) -> List[int]:
    """Encode a Unicode string to UTF-8 byte IDs ``0..255`` (+ optional BOS/EOS)."""
    raw = list(text.encode("utf-8"))
    if add_bos_eos:
        return [TEXT_BOS_ID] + raw + [TEXT_EOS_ID]
    return raw


def decode_text_span_utf8(ids: Sequence[int], errors: str = "replace") -> str:
    """Decode UTF-8 byte IDs (stopping at ``TEXT_EOS_ID``) back to Unicode text."""
    buf = bytearray()
    for raw_i in ids:
        i = int(raw_i)
        if i == TEXT_EOS_ID:
            break
        if i in (TEXT_BOS_ID, TEXT_PAD_ID):
            continue
        if 0 <= i <= 255:
            buf.append(i)
    return bytes(buf).decode("utf-8", errors=errors)


# ---------------------------------------------------------------------------
# 6. Encode / decode helpers for full vocabulary
# ---------------------------------------------------------------------------

def encode_full(tokens: Sequence[str], add_bos_eos: bool = True) -> List[int]:
    """Encode ``bekern-full`` token strings into vocabulary IDs."""
    if isinstance(tokens, str):
        tokens = tokens.split()
    ids: List[int] = []
    for tok in tokens:
        try:
            ids.append(FULL_TOKEN_TO_ID[tok])
        except KeyError:
            raise ValueError(f"unknown bekern-full token: {tok!r}") from None
    if add_bos_eos:
        return [BOS_ID] + ids + [EOS_ID]
    return ids


def decode_full(ids: Sequence[int], skip_special: bool = True) -> List[str]:
    """Decode ``bekern-full`` integer IDs back into token strings."""
    out: List[str] = []
    for raw_i in ids:
        i = int(raw_i)
        if skip_special and i in (PAD_ID, BOS_ID, EOS_ID):
            if i == EOS_ID:
                break
            continue
        tok = FULL_ID_TO_TOKEN.get(i)
        if tok is not None:
            out.append(tok)
    return out


_SYS_LYR_ATTR_RE = re.compile(r"<sys_lyr\s+(v=\d+)>")
_TUP_OPEN_PARAM_RE = re.compile(r"^tup-open-(\d+):(\d+)$")
_MULTIREST_PARAM_RE = re.compile(r"^multirest-(\d+)$")


def _decompose_uint_tokens(num_str: str) -> List[str]:
    """Decompose an unsigned integer string into finite vocab tokens (exact if <=12 else digits)."""
    if num_str in FULL_TOKEN_TO_ID:
        return [num_str]
    return [ch for ch in num_str]


def encode_bekern_full_sequence(
    text: str,
    add_bos_eos: bool = True,
) -> Dict[str, Any]:
    """Split a ``bekern-full`` string into dual-head ``music_ids`` + UTF-8 ``text_spans``.

    - Handles parameterized ``tup-open-{a}:{n}`` -> ``[tup-open, a, :, n]`` and
      ``multirest-{n}`` -> ``[<multirest>, digits]`` with ZERO OOVs.
    - Normalizes ``<sys_lyr v=N>`` to ``<sys_lyr>`` + ``v=N`` cleanly.
    - Extracts every Layer 3 text payload (``<sys_lyr>``, ``<harm>``, ``<tempo>``,
      ``<dir>``, ``<reh>``, ``<fb>`` ... ``</text_span>``) into UTF-8 byte ID lists
      via ``encode_text_span_utf8()``.
    """
    norm = _SYS_LYR_ATTR_RE.sub(r"<sys_lyr> \1", text)
    raw_toks = norm.split()

    music_ids: List[int] = [BOS_ID] if add_bos_eos else []
    text_spans: List[List[int]] = []
    text_span_meta: List[Dict[str, Any]] = []

    in_span = False
    cur_tag: Optional[str] = None
    cur_open_idx = -1
    cur_payload: List[str] = []

    for tok in raw_toks:
        if in_span:
            if tok == TEXT_SPAN_CLOSE_TOKEN:
                payload_str = " ".join(cur_payload)
                byte_ids = encode_text_span_utf8(payload_str, add_bos_eos=True)
                text_spans.append(byte_ids)
                text_span_meta.append(
                    {
                        "tag": cur_tag,
                        "music_open_idx": cur_open_idx,
                        "text": payload_str,
                    }
                )
                music_ids.append(TEXT_SPAN_CLOSE_TOKEN_ID)
                in_span = False
                cur_tag = None
                cur_payload = []
            else:
                cur_payload.append(tok)
            continue

        if tok in TEXT_SPAN_OPEN_TOKENS:
            cur_open_idx = len(music_ids)
            music_ids.append(FULL_TOKEN_TO_ID[tok])
            in_span = True
            cur_tag = tok
            cur_payload = []
            continue

        m_tup = _TUP_OPEN_PARAM_RE.match(tok)
        if m_tup is not None:
            a_str, n_str = m_tup.group(1), m_tup.group(2)
            parts = ["tup-open"] + _decompose_uint_tokens(a_str) + [":"] + _decompose_uint_tokens(n_str)
            for p in parts:
                music_ids.append(FULL_TOKEN_TO_ID[p])
            continue

        m_mr = _MULTIREST_PARAM_RE.match(tok)
        if m_mr is not None:
            mr_id = FULL_TOKEN_TO_ID["<multirest>"]
            if not music_ids or music_ids[-1] != mr_id:
                music_ids.append(mr_id)
            for p in _decompose_uint_tokens(m_mr.group(1)):
                music_ids.append(FULL_TOKEN_TO_ID[p])
            continue

        tid = FULL_TOKEN_TO_ID.get(tok)
        if tid is None:
            raise ValueError(f"unknown bekern-full token: {tok!r}")
        music_ids.append(tid)

    if in_span:
        payload_str = " ".join(cur_payload)
        byte_ids = encode_text_span_utf8(payload_str, add_bos_eos=True)
        text_spans.append(byte_ids)
        text_span_meta.append(
            {
                "tag": cur_tag,
                "music_open_idx": cur_open_idx,
                "text": payload_str,
            }
        )
        music_ids.append(TEXT_SPAN_CLOSE_TOKEN_ID)

    if add_bos_eos:
        music_ids.append(EOS_ID)

    return {
        "music_ids": music_ids,
        "text_spans": text_spans,
        "text_span_meta": text_span_meta,
    }


_UINT_ARG_SET: FrozenSet[str] = frozenset(
    tok for tok in (str(i) for i in range(0, 25)) if tok in FULL_TOKEN_TO_ID
)
_UINT_POS_SET: FrozenSet[str] = frozenset(
    tok for tok in (str(i) for i in range(1, 25)) if tok in FULL_TOKEN_TO_ID
)
_UINT_ARG_IDS: FrozenSet[int] = frozenset(FULL_TOKEN_TO_ID[t] for t in _UINT_ARG_SET)
_UINT_POS_IDS: FrozenSet[int] = frozenset(FULL_TOKEN_TO_ID[t] for t in _UINT_POS_SET)
_TRANSPOSE_ARG_SET: FrozenSet[str] = frozenset(
    tok
    for tok in (
        [str(i) for i in range(-24, 25)]
        + [f"+{i}" for i in range(1, 13)]
    )
    if tok in FULL_TOKEN_TO_ID
)
_TRANSPOSE_ARG_IDS: FrozenSet[int] = frozenset(FULL_TOKEN_TO_ID[t] for t in _TRANSPOSE_ARG_SET)


def decode_bekern_full_sequence(
    music_ids: Sequence[int],
    text_spans: Optional[Sequence[Union[Sequence[int], str]]] = None,
    skip_special: bool = True,
) -> str:
    """Reconstruct a ``bekern-full`` string from ``music_ids`` and UTF-8 ``text_spans``.

    Recomposes ``tup-open <a_toks...> : <n_toks...>`` back into ``tup-open-{a}:{n}``
    and ``<multirest> <digits...>`` back into ``<multirest> multirest-{n}`` so
    ``assembly-serialize bekern2musicxml`` parses the decoded sequence cleanly.
    """
    spans_list = list(text_spans) if text_spans is not None else []
    span_idx = 0
    tok_stream: List[Tuple[int, str]] = []
    for raw_i in music_ids:
        i = int(raw_i)
        if skip_special and i in (PAD_ID, BOS_ID, EOS_ID):
            if i == EOS_ID:
                break
            continue
        tok = FULL_ID_TO_TOKEN.get(i)
        if tok is not None:
            tok_stream.append((i, tok))

    out: List[str] = []
    idx = 0
    n_toks = len(tok_stream)
    while idx < n_toks:
        i, tok = tok_stream[idx]
        if tok == "tup-open":
            j = idx + 1
            a_parts: List[str] = []
            while j < n_toks and tok_stream[j][1] in _UINT_ARG_SET:
                a_parts.append(tok_stream[j][1])
                j += 1
            if a_parts and j < n_toks and tok_stream[j][1] == ":":
                k = j + 1
                n_parts: List[str] = []
                while k < n_toks and tok_stream[k][1] in _UINT_ARG_SET:
                    n_parts.append(tok_stream[k][1])
                    k += 1
                if n_parts:
                    out.append(f"tup-open-{''.join(a_parts)}:{''.join(n_parts)}")
                    idx = k
                    continue
        if tok == "<multirest>":
            j = idx + 1
            d_parts: List[str] = []
            while j < n_toks and tok_stream[j][1] in _UINT_ARG_SET:
                d_parts.append(tok_stream[j][1])
                j += 1
            out.append("<multirest>")
            if d_parts:
                out.append(f"multirest-{''.join(d_parts)}")
                idx = j
            else:
                idx += 1
            continue

        out.append(tok)
        if i in TEXT_SPAN_OPEN_TOKEN_IDS and span_idx < len(spans_list):
            sp = spans_list[span_idx]
            span_idx += 1
            decoded = sp if isinstance(sp, str) else decode_text_span_utf8(sp)
            if decoded:
                out.append(decoded)
        idx += 1
    return " ".join(out)


# ---------------------------------------------------------------------------
# 7. BekernFullGrammarState FSA (Mitigations #3 & #8)
# ---------------------------------------------------------------------------

_SLUR_OPEN_MAP: Dict[str, int] = {f"slur-{i}-open": i for i in range(1, 5)}
_SLUR_CLOSE_MAP: Dict[str, int] = {f"slur-{i}-close": i for i in range(1, 5)}
_HAIRPIN_OPEN_MAP: Dict[str, int] = {
    f"hairpin-{i}-{kind}": i
    for i in range(1, 4)
    for kind in ("cresc", "dim")
}
_HAIRPIN_CLOSE_MAP: Dict[str, int] = {f"hairpin-{i}-close": i for i in range(1, 4)}

_LEGACY_MUSIC_SET: FrozenSet[str] = frozenset(data.VOCAB)
_LAYER3_BODY_SET: FrozenSet[str] = frozenset(
    LAYER3_LYRIC_MARKERS + LAYER3_ANCHORS + LAYER3_VERSES
)
_LAYER3_BODY_IDS: FrozenSet[int] = frozenset(FULL_TOKEN_TO_ID[t] for t in _LAYER3_BODY_SET)

# Non-spanner Layer 0/1/2 extension tokens (beyond data.VOCAB) that can appear
# cleanly in music mode at boundary/tail positions without disturbing the core
# rhythmic bar-duration cursor.
_NUMBERED_SPANNER_SET: FrozenSet[str] = frozenset(
    list(_SLUR_OPEN_MAP) + list(_SLUR_CLOSE_MAP)
    + list(_HAIRPIN_OPEN_MAP) + list(_HAIRPIN_CLOSE_MAP)
)
_DELEGATED_ALIAS_MAP: Dict[str, str] = {
    "<chord>": "chord",
    "grace-slash": "grace",
}
_PERC_TAB_DELEGATE_SET: FrozenSet[str] = (
    frozenset((grammar.UNPITCHED,))
    | grammar.NOTEHEADS
    | grammar.TAB_STRINGS
    | grammar.TAB_FRETS
)
_EXT_MUSIC_PASSTHROUGH_SET: FrozenSet[str] = (
    (frozenset(LAYER0_TOKENS) | frozenset(LAYER1_NEW_TOKENS) | frozenset(LAYER2_NEW_TOKENS))
    - _LEGACY_MUSIC_SET
    - _NUMBERED_SPANNER_SET
    - frozenset(_DELEGATED_ALIAS_MAP.keys())
    - frozenset((":",))
    - _TRANSPOSE_ARG_SET
)
_EXT_MUSIC_PASSTHROUGH_IDS: FrozenSet[int] = frozenset(
    FULL_TOKEN_TO_ID[t] for t in _EXT_MUSIC_PASSTHROUGH_SET
)
_EXT_MUSIC_PASSTHROUGH_NO_PERCTAB_IDS: FrozenSet[int] = frozenset(
    FULL_TOKEN_TO_ID[t]
    for t in (_EXT_MUSIC_PASSTHROUGH_SET - _PERC_TAB_DELEGATE_SET)
)

# PrefixState modes where a boundary/tail-level extension or Layer 3 text span
# may cleanly start (never mid-pitch `dur`/`step`, mid-chord, or mid-ratio).
_SAFE_SPAN_MODES: FrozenSet[str] = frozenset(("bound", "pitch-tail", "tiestart", "rest-tail"))

MAX_LAYER3_SPAN_TOKENS: int = 128


class BekernFullGrammarState:
    """Incremental constrained-decoding FSA for ``bekern-full`` (Mitigations #3 & #8).

    Wraps ``grammar.PrefixState`` for core ``0..260`` music tokens and adds:
    1. Optional ``<schema:v1>`` / ``<schema:v2a>`` / ``<schema:full>`` prefix
       allowed **only** at position 0 (immediately after ``<bos>``).
    2. Layer 3 text-span pause: inside ``<sys_lyr>`` / ``<harm>`` / ``<tempo>``
       / ``<dir>`` / ``<reh>`` / ``<fb>`` ... ``</text_span>``, the rhythmic
       bar-duration FSA is completely paused so lyrics/text never corrupt
       measure duration tracking or note alignment (with anti-loop cap
       ``MAX_LAYER3_SPAN_TOKENS = 128``).
    3. Per-staff numbered spanner parity: tracks open channels for ``slur-1..4``
       and ``hairpin-1..3`` keyed by ``music_state.staves_in_bar``, forbidding
       double-opening an already-open channel or closing an unopened channel.
    """

    def __init__(
        self,
        strict: bool = False,
        allow_percussion_tab: bool = False,
        max_tuplet_depth: int = 1,
    ) -> None:
        self.music_state = grammar.PrefixState(
            strict=strict,
            allow_percussion_tab=allow_percussion_tab,
            vocab_map=FULL_TOKEN_TO_ID,
            max_tuplet_depth=max_tuplet_depth,
        )
        self.pos: int = 0
        self.schema: Optional[str] = None
        self.in_text_span: bool = False
        self.active_text_tag: Optional[str] = None
        self.text_span_tokens: int = 0
        self.open_slurs_by_staff: Dict[int, Set[int]] = {}
        self.open_hairpins_by_staff: Dict[int, Set[int]] = {}
        # Explicit mini-states for <transpose>, <multirest>, and tup-open (v2 + colon syntax).
        self.transpose_args_left: int = 0
        self.multirest_state: int = 0  # 0=off, 1=need >=1 digit/int, 2=optional extra digits/ints
        self.tup_mode: Optional[str] = None  # None | "a" | "a_or_legacy_n" | "n_colon"
        self.tup_a_str: str = ""
        self.tup_second_tok: str = ""
        self.tup_n_str: str = ""

    @property
    def open_slurs(self) -> Set[int]:
        """Open slur channels on the current staff (``music_state.staves_in_bar``)."""
        return self.open_slurs_by_staff.get(self.music_state.staves_in_bar, set())

    @property
    def open_hairpins(self) -> Set[int]:
        """Open hairpin channels on the current staff (``music_state.staves_in_bar``)."""
        return self.open_hairpins_by_staff.get(self.music_state.staves_in_bar, set())

    def copy(self) -> "BekernFullGrammarState":
        c = BekernFullGrammarState.__new__(BekernFullGrammarState)
        c.music_state = self.music_state.copy()
        c.pos = self.pos
        c.schema = self.schema
        c.in_text_span = self.in_text_span
        c.active_text_tag = self.active_text_tag
        c.text_span_tokens = self.text_span_tokens
        c.open_slurs_by_staff = {k: set(v) for k, v in self.open_slurs_by_staff.items()}
        c.open_hairpins_by_staff = {k: set(v) for k, v in self.open_hairpins_by_staff.items()}
        c.transpose_args_left = self.transpose_args_left
        c.multirest_state = self.multirest_state
        c.tup_mode = self.tup_mode
        c.tup_a_str = self.tup_a_str
        c.tup_second_tok = self.tup_second_tok
        c.tup_n_str = self.tup_n_str
        return c

    def _resolve_tok(self, tok_or_id: Union[int, str]) -> str:
        if isinstance(tok_or_id, str):
            if tok_or_id not in FULL_TOKEN_TO_ID:
                raise ValueError(f"unknown bekern-full token: {tok_or_id!r}")
            return tok_or_id
        tid = int(tok_or_id)
        if tid not in FULL_ID_TO_TOKEN:
            raise ValueError(f"unknown bekern-full token id: {tid!r}")
        return FULL_ID_TO_TOKEN[tid]

    def _record_ext_step(self, tok: str) -> None:
        ms = self.music_state
        ms.tokens_in_system += 1
        ms.tokens_since_bar += 1
        ms.recent.append(tok)
        if len(ms.recent) > grammar.RECENT_WINDOW_CAP:
            ms.recent = ms.recent[-grammar.RECENT_WINDOW_KEEP:]

    def _commit_tuplet_ratio(self, actual: int, normal: int) -> None:
        if actual <= 0 or normal <= 0 or actual > grammar.MAX_TUPLET_VAL or normal > grammar.MAX_TUPLET_VAL:
            raise ValueError(f"invalid tuplet ratio {actual}:{normal}")
        if actual == normal:
            raise ValueError(f"identity tuplet {actual}:{normal} forbidden")
        ms = self.music_state
        ms.pending += 1
        ms.pend_spans.append((actual, normal))
        ms.comp_num = 0
        ms.comp_digits = 0
        ms.comp_val = 0
        ms.mode = "bound"
        self.tup_mode = None
        self.tup_a_str = ""
        self.tup_second_tok = ""
        self.tup_n_str = ""

    def can_emit_sys(self) -> bool:
        if self.in_text_span or self.transpose_args_left > 0 or self.multirest_state == 1 or self.tup_mode is not None:
            return False
        if self.music_state.mode not in _SAFE_SPAN_MODES:
            return False
        return self.music_state.can_emit_sys()

    def can_emit_eos(self, cur_len: Optional[int] = None, max_len: Optional[int] = None) -> bool:
        if self.in_text_span or self.transpose_args_left > 0 or self.multirest_state == 1 or self.tup_mode is not None:
            return False
        return self.music_state.can_emit_eos(cur_len=cur_len, max_len=max_len)

    def _allowed_numbered_spanners(self) -> Set[int]:
        if not self.music_state.strict:
            return {FULL_TOKEN_TO_ID[t] for t in _NUMBERED_SPANNER_SET}
        allow: Set[int] = set()
        staff_slurs = self.open_slurs
        staff_hairpins = self.open_hairpins
        for tok, ch in _SLUR_OPEN_MAP.items():
            if ch not in staff_slurs:
                allow.add(FULL_TOKEN_TO_ID[tok])
        for tok, ch in _SLUR_CLOSE_MAP.items():
            if ch in staff_slurs:
                allow.add(FULL_TOKEN_TO_ID[tok])
        for tok, ch in _HAIRPIN_OPEN_MAP.items():
            if ch not in staff_hairpins:
                allow.add(FULL_TOKEN_TO_ID[tok])
        for tok, ch in _HAIRPIN_CLOSE_MAP.items():
            if ch in staff_hairpins:
                allow.add(FULL_TOKEN_TO_ID[tok])
        return allow

    def _allowed_normal_music(self) -> Tuple[Set[int], bool]:
        allow, eos_ok = self.music_state.allowed()
        allow = set(allow)

        # Mirror delegated aliases (<chord> <-> chord, grace-slash <-> grace).
        for alias_tok, base_tok in _DELEGATED_ALIAS_MAP.items():
            if FULL_TOKEN_TO_ID[base_tok] in allow:
                allow.add(FULL_TOKEN_TO_ID[alias_tok])

        # Optional <schema:*> token allowed ONLY at position 0.
        if self.pos == 0:
            allow.update(SCHEMA_TOKEN_IDS)
        else:
            eos_ok = bool(eos_ok and self.can_emit_eos())

        if self.can_emit_sys():
            allow.add(SYS_ID)

        not_capped = (
            (not self.music_state._sys_is_capped())
            and (self.music_state.tokens_since_bar < grammar.MAX_TOKENS_PER_BAR)
        )
        if not_capped and self.music_state.mode in _SAFE_SPAN_MODES:
            allow.update(self._allowed_numbered_spanners())
            if self.music_state.allow_percussion_tab:
                allow.update(_EXT_MUSIC_PASSTHROUGH_NO_PERCTAB_IDS)
            else:
                allow.update(_EXT_MUSIC_PASSTHROUGH_IDS)
            if not self.music_state.has_open_tuplet():
                allow.update(TEXT_SPAN_OPEN_TOKEN_IDS)

        return allow, eos_ok

    def allowed(self) -> Tuple[Set[int], bool]:
        """Return ``(allowed_token_ids, eos_ok)`` for the current FSA state."""
        if self.in_text_span:
            if self.text_span_tokens >= MAX_LAYER3_SPAN_TOKENS:
                return {TEXT_SPAN_CLOSE_TOKEN_ID}, False
            allow: Set[int] = set(_LAYER3_BODY_IDS)
            allow.add(TEXT_SPAN_CLOSE_TOKEN_ID)
            return allow, False

        if self.transpose_args_left > 0:
            return set(_TRANSPOSE_ARG_IDS), False

        if self.multirest_state == 1:
            return set(_UINT_POS_IDS), False

        if self.tup_mode == "a":
            allow_t: Set[int] = set()
            if not self.tup_a_str:
                for tok in _UINT_POS_SET:
                    if int(tok) <= grammar.MAX_TUPLET_VAL:
                        allow_t.add(FULL_TOKEN_TO_ID[tok])
            else:
                allow_t.add(FULL_TOKEN_TO_ID[":"])
                a_val = int(self.tup_a_str)
                for tok in _UINT_ARG_SET:
                    # Allow 2nd digit of multi-digit A (if <= MAX_TUPLET_VAL) OR legacy N (if != A).
                    can_extend_a = len(tok) == 1 and int(self.tup_a_str + tok) <= grammar.MAX_TUPLET_VAL
                    can_legacy_n = (tok in _UINT_POS_SET) and int(tok) <= grammar.MAX_TUPLET_VAL and int(tok) != a_val
                    if can_extend_a or can_legacy_n:
                        allow_t.add(FULL_TOKEN_TO_ID[tok])
            return allow_t, False

        if self.tup_mode == "a_or_legacy_n":
            allow_t = set()
            if len(self.tup_second_tok) == 1 and int(self.tup_a_str + self.tup_second_tok) <= grammar.MAX_TUPLET_VAL:
                allow_t.add(FULL_TOKEN_TO_ID[":"])
            a_val = int(self.tup_a_str)
            n_val = int(self.tup_second_tok)
            if n_val > 0 and n_val != a_val and n_val <= grammar.MAX_TUPLET_VAL:
                ms = self.music_state
                saved_mode = ms.mode
                ms.pending += 1
                ms.pend_spans.append((a_val, n_val))
                ms.mode = "bound"
                try:
                    sub_allow, _ = self._allowed_normal_music()
                    allow_t.update(sub_allow)
                finally:
                    ms.pending -= 1
                    ms.pend_spans.pop()
                    ms.mode = saved_mode
            return allow_t, False

        if self.tup_mode == "n_colon":
            allow_t = set()
            a_val = int(self.tup_a_str)
            if not self.tup_n_str:
                for tok in _UINT_POS_SET:
                    v = int(tok)
                    if v <= grammar.MAX_TUPLET_VAL and (v != a_val or v * 10 <= grammar.MAX_TUPLET_VAL):
                        allow_t.add(FULL_TOKEN_TO_ID[tok])
                return allow_t, False
            # Have at least 1 token in tup_n_str: allow extra digits (if <= cap) OR normal music tokens.
            for d in "0123456789":
                if d in FULL_TOKEN_TO_ID:
                    nv = int(self.tup_n_str + d)
                    if nv <= grammar.MAX_TUPLET_VAL and nv != a_val:
                        allow_t.add(FULL_TOKEN_TO_ID[d])
            n_val = int(self.tup_n_str)
            if n_val > 0 and n_val != a_val and n_val <= grammar.MAX_TUPLET_VAL:
                ms = self.music_state
                saved_mode = ms.mode
                ms.pending += 1
                ms.pend_spans.append((a_val, n_val))
                ms.mode = "bound"
                try:
                    sub_allow, _ = self._allowed_normal_music()
                    allow_t.update(sub_allow)
                finally:
                    ms.pending -= 1
                    ms.pend_spans.pop()
                    ms.mode = saved_mode
            return allow_t, False

        allow, eos_ok = self._allowed_normal_music()
        if self.multirest_state == 2:
            allow.update(_UINT_ARG_IDS)
        return allow, eos_ok

    def step(self, tok_or_id: Union[int, str]) -> None:
        """Advance the FSA by one token string or ID (raises ``ValueError`` if illegal)."""
        tok = self._resolve_tok(tok_or_id)

        # 0. Active argument mini-states (<transpose>, <multirest>, tup-open).
        if self.transpose_args_left > 0:
            if tok not in _TRANSPOSE_ARG_SET:
                raise ValueError(
                    f"expected signed transpose integer argument, got {tok!r}"
                )
            self.transpose_args_left -= 1
            self._record_ext_step(tok)
            self.pos += 1
            return

        if self.multirest_state == 1:
            if tok not in _UINT_POS_SET:
                raise ValueError(
                    f"expected positive integer/digit after <multirest>, got {tok!r}"
                )
            self.multirest_state = 2
            self._record_ext_step(tok)
            self.pos += 1
            return

        if self.multirest_state == 2:
            if tok in _UINT_ARG_SET:
                self._record_ext_step(tok)
                self.pos += 1
                return
            self.multirest_state = 0
            # Fall through to process non-digit token in normal mode.

        if self.tup_mode == "a":
            if not self.tup_a_str:
                if tok not in _UINT_POS_SET or int(tok) > grammar.MAX_TUPLET_VAL:
                    raise ValueError(f"bad tup-open actual count token {tok!r}")
                self.tup_a_str = tok
                self._record_ext_step(tok)
                self.pos += 1
                return
            if tok == ":":
                self.tup_mode = "n_colon"
                self.tup_n_str = ""
                self._record_ext_step(tok)
                self.pos += 1
                return
            if tok in _UINT_ARG_SET:
                can_extend_a = len(tok) == 1 and int(self.tup_a_str + tok) <= grammar.MAX_TUPLET_VAL
                can_legacy_n = (
                    tok in _UINT_POS_SET
                    and int(tok) <= grammar.MAX_TUPLET_VAL
                    and int(tok) != int(self.tup_a_str)
                )
                if not (can_extend_a or can_legacy_n):
                    raise ValueError(f"bad tup-open second token {tok!r} after actual {self.tup_a_str!r}")
                self.tup_second_tok = tok
                self.tup_mode = "a_or_legacy_n"
                self._record_ext_step(tok)
                self.pos += 1
                return
            raise ValueError(f"expected ':' or integer in tup-open, got {tok!r}")

        if self.tup_mode == "a_or_legacy_n":
            if tok == ":":
                combined = self.tup_a_str + self.tup_second_tok
                if len(self.tup_second_tok) != 1 or int(combined) > grammar.MAX_TUPLET_VAL:
                    raise ValueError(f"invalid multi-digit tup-open actual {combined!r}")
                self.tup_a_str = combined
                self.tup_second_tok = ""
                self.tup_mode = "n_colon"
                self.tup_n_str = ""
                self._record_ext_step(tok)
                self.pos += 1
                return
            # Legacy v2 `tup-open A N` completed -> commit (A, N) and fall through to process `tok`!
            self._commit_tuplet_ratio(int(self.tup_a_str), int(self.tup_second_tok))

        if self.tup_mode == "n_colon":
            if not self.tup_n_str:
                if tok not in _UINT_POS_SET or int(tok) > grammar.MAX_TUPLET_VAL:
                    raise ValueError(f"bad tup-open normal count token {tok!r}")
                self.tup_n_str = tok
                self._record_ext_step(tok)
                self.pos += 1
                return
            if len(tok) == 1 and tok in "0123456789":
                nv = int(self.tup_n_str + tok)
                if nv <= grammar.MAX_TUPLET_VAL and nv != int(self.tup_a_str):
                    self.tup_n_str = self.tup_n_str + tok
                    self._record_ext_step(tok)
                    self.pos += 1
                    return
                raise ValueError(
                    f"tup-open normal count {nv} invalid (cap={grammar.MAX_TUPLET_VAL}, actual={self.tup_a_str})"
                )
            # Non-digit token -> commit (A, N) and fall through to process `tok`!
            self._commit_tuplet_ratio(int(self.tup_a_str), int(self.tup_n_str))

        # 1. Schema prefix token: strictly position 0 only.
        if tok in SCHEMA_TOKENS:
            if self.pos != 0:
                raise ValueError(
                    f"schema token {tok!r} only allowed at position 0, got pos={self.pos}"
                )
            self.schema = tok
            self.pos += 1
            return

        # 2. Inside a Layer 3 text span: rhythmic FSA is paused!
        if self.in_text_span:
            if tok == TEXT_SPAN_CLOSE_TOKEN:
                self.in_text_span = False
                self.active_text_tag = None
                self.text_span_tokens = 0
                self._record_ext_step(tok)
                self.pos += 1
                return
            if tok in _LAYER3_BODY_SET:
                if self.text_span_tokens >= MAX_LAYER3_SPAN_TOKENS:
                    raise ValueError(
                        f"Layer 3 text span exceeded MAX_LAYER3_SPAN_TOKENS={MAX_LAYER3_SPAN_TOKENS}"
                    )
                self.text_span_tokens += 1
                self._record_ext_step(tok)
                self.pos += 1
                return
            raise ValueError(
                f"illegal token {tok!r} inside Layer 3 text span {self.active_text_tag!r}"
            )

        # 3. Opening a Layer 3 text span: close pending event & pause rhythmic FSA cleanly.
        if tok in TEXT_SPAN_OPEN_TOKENS:
            if self.music_state.mode not in _SAFE_SPAN_MODES or self.music_state.has_open_tuplet():
                raise ValueError(
                    f"cannot open text span {tok!r} in music mode {self.music_state.mode!r}"
                )
            if self.music_state.mode in ("pitch-tail", "tiestart", "rest-tail"):
                self.music_state._close_event(placed=(self.music_state.mode != "rest-tail"))
                self.music_state.mode = "bound"
            self.in_text_span = True
            self.active_text_tag = tok
            self.text_span_tokens = 0
            self._record_ext_step(tok)
            self.pos += 1
            return

        if tok == TEXT_SPAN_CLOSE_TOKEN:
            raise ValueError("unexpected </text_span> outside Layer 3 text span")

        if tok in _LAYER3_BODY_SET:
            raise ValueError(f"Layer 3 token {tok!r} outside text span")

        # 4. System separator.
        if tok == "<sys>":
            if not self.can_emit_sys():
                raise ValueError("<sys> not allowed in current state")
            self.music_state.reset_system()
            self.pos += 1
            return

        # 4b. Mini-state openers (<transpose>, <multirest>, tup-open).
        if tok == "<transpose>":
            if self.music_state.mode not in _SAFE_SPAN_MODES:
                raise ValueError(f"<transpose> not allowed in mode {self.music_state.mode!r}")
            self.transpose_args_left = 2
            self._record_ext_step(tok)
            self.pos += 1
            return

        if tok == "<multirest>":
            if self.music_state.mode not in _SAFE_SPAN_MODES:
                raise ValueError(f"<multirest> not allowed in mode {self.music_state.mode!r}")
            self.multirest_state = 1
            self._record_ext_step(tok)
            self.pos += 1
            return

        if tok == "tup-open":
            self.music_state.step("tup-open")
            self.tup_mode = "a"
            self.tup_a_str = ""
            self.tup_second_tok = ""
            self.tup_n_str = ""
            self.pos += 1
            return

        # 5. Per-staff numbered spanners: strict channel parity in strict=True; permissive in strict=False
        # (supports cross-staff hairpins/slurs, multi-voice channel reuse, and page-2+ continuations).
        staff_idx = self.music_state.staves_in_bar
        if tok in _SLUR_OPEN_MAP:
            if self.music_state.mode not in _SAFE_SPAN_MODES:
                raise ValueError(f"slur token {tok!r} not allowed in mode {self.music_state.mode!r}")
            ch = _SLUR_OPEN_MAP[tok]
            s_set = self.open_slurs_by_staff.setdefault(staff_idx, set())
            if self.music_state.strict and ch in s_set:
                raise ValueError(f"slur channel {ch} already open on staff {staff_idx} ({tok!r})")
            s_set.add(ch)
            self._record_ext_step(tok)
            self.pos += 1
            return

        if tok in _SLUR_CLOSE_MAP:
            if self.music_state.mode not in _SAFE_SPAN_MODES:
                raise ValueError(f"slur token {tok!r} not allowed in mode {self.music_state.mode!r}")
            ch = _SLUR_CLOSE_MAP[tok]
            s_set = self.open_slurs_by_staff.get(staff_idx)
            if self.music_state.strict and (not s_set or ch not in s_set):
                raise ValueError(f"cannot close unopened slur channel {ch} on staff {staff_idx} ({tok!r})")
            if s_set is not None:
                s_set.discard(ch)
                if not s_set:
                    self.open_slurs_by_staff.pop(staff_idx, None)
            self._record_ext_step(tok)
            self.pos += 1
            return

        if tok in _HAIRPIN_OPEN_MAP:
            if self.music_state.mode not in _SAFE_SPAN_MODES:
                raise ValueError(f"hairpin token {tok!r} not allowed in mode {self.music_state.mode!r}")
            ch = _HAIRPIN_OPEN_MAP[tok]
            h_set = self.open_hairpins_by_staff.setdefault(staff_idx, set())
            if self.music_state.strict and ch in h_set:
                raise ValueError(f"hairpin channel {ch} already open on staff {staff_idx} ({tok!r})")
            h_set.add(ch)
            self._record_ext_step(tok)
            self.pos += 1
            return

        if tok in _HAIRPIN_CLOSE_MAP:
            if self.music_state.mode not in _SAFE_SPAN_MODES:
                raise ValueError(f"hairpin token {tok!r} not allowed in mode {self.music_state.mode!r}")
            ch = _HAIRPIN_CLOSE_MAP[tok]
            h_set = self.open_hairpins_by_staff.get(staff_idx)
            if self.music_state.strict and (not h_set or ch not in h_set):
                raise ValueError(f"cannot close unopened hairpin channel {ch} on staff {staff_idx} ({tok!r})")
            if h_set is not None:
                h_set.discard(ch)
                if not h_set:
                    self.open_hairpins_by_staff.pop(staff_idx, None)
            self._record_ext_step(tok)
            self.pos += 1
            return

        # 6. Delegated aliases (<chord> -> chord, grace-slash -> grace).
        if tok in _DELEGATED_ALIAS_MAP:
            self.music_state.step(_DELEGATED_ALIAS_MAP[tok])
            self.pos += 1
            return

        # 7. Percussion / TAB tokens delegated to PrefixState when allow_percussion_tab=True.
        if self.music_state.allow_percussion_tab and tok in _PERC_TAB_DELEGATE_SET:
            self.music_state.step(tok)
            self.pos += 1
            return

        # 8. Extended Layer 0/1/2 music tokens (outside data.VOCAB).
        if tok in _EXT_MUSIC_PASSTHROUGH_SET:
            if self.music_state.mode not in _SAFE_SPAN_MODES:
                raise ValueError(
                    f"extended music token {tok!r} not allowed in mode {self.music_state.mode!r}"
                )
            self._record_ext_step(tok)
            self.pos += 1
            return

        # 9. Core legacy music tokens (0..260): delegate to grammar.PrefixState.
        if tok in _LEGACY_MUSIC_SET:
            self.music_state.step(tok)
            self.pos += 1
            return

        raise ValueError(f"unhandled token {tok!r}")
