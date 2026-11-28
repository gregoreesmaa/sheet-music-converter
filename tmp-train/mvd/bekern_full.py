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

import sys
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

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
    # Grammar FSA
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
LAYER3_ANCHORS: Tuple[str, ...] = tuple(f"@{i}" for i in range(64))
LAYER3_VERSES: Tuple[str, ...] = tuple(f"v={i}" for i in range(1, 5))

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
):
    """Compute layer-masked CE loss applying both target and logit-suppression masks."""
    import torch
    import torch.nn.functional as F

    bsz, seq_len, vocab_size = logits.shape
    target_mask, logit_suppress_mask = build_layer_ce_masks(
        target_ids, layer_masks, vocab_size=vocab_size
    )
    masked_logits = logits.masked_fill(logit_suppress_mask.unsqueeze(1), float("-inf"))
    masked_targets = torch.where(
        target_mask,
        target_ids.to(dtype=torch.long),
        torch.full_like(target_ids, -100, dtype=torch.long),
    )
    if not bool(target_mask.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(
        masked_logits.reshape(bsz * seq_len, vocab_size),
        masked_targets.reshape(bsz * seq_len),
        ignore_index=-100,
        label_smoothing=float(label_smoothing),
    )


# ---------------------------------------------------------------------------
# 4. BD3-LM `is_music_body` Mask (Mitigation #7)
# ---------------------------------------------------------------------------

_NON_BODY_SPECIAL_IDS: FrozenSet[int] = frozenset(
    (PAD_ID, BOS_ID, EOS_ID, SYS_ID)
) | SCHEMA_TOKEN_IDS


def compute_is_music_body_mask(token_ids):
    """Return BoolTensor True ONLY for Layers 0-2 music body tokens (Mitigation #7).

    Strictly ``False`` for:
    - ``<pad>``, ``<bos>``, ``<eos>``, ``<sys>``, and ``<schema:*>`` prefixes;
    - Any token inside a Layer 3 text span (from ``<sys_lyr>`` / ``<harm>`` /
      ``<tempo>`` / ``<dir>`` / ``<reh>`` / ``<fb>`` through and including
      ``</text_span>``);
    - Any stray Layer 3 token outside an explicit span.
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
    rows = toks_2d.detach().cpu().tolist()
    out_rows: List[List[bool]] = []
    for row in rows:
        in_span = False
        mask_row: List[bool] = []
        for raw_tid in row:
            tid = int(raw_tid)
            if tid in TEXT_SPAN_OPEN_TOKEN_IDS:
                in_span = True
                mask_row.append(False)
                continue
            if tid == TEXT_SPAN_CLOSE_TOKEN_ID:
                in_span = False
                mask_row.append(False)
                continue
            if in_span:
                mask_row.append(False)
                continue
            if tid in _NON_BODY_SPECIAL_IDS:
                mask_row.append(False)
                continue
            layer = token_layer_of(tid)
            mask_row.append(layer in (0, 1, 2))
        out_rows.append(mask_row)

    mask = torch.tensor(out_rows, dtype=torch.bool, device=device)
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
_EXT_MUSIC_PASSTHROUGH_SET: FrozenSet[str] = (
    (frozenset(LAYER0_TOKENS) | frozenset(LAYER1_NEW_TOKENS) | frozenset(LAYER2_NEW_TOKENS))
    - _LEGACY_MUSIC_SET
    - _NUMBERED_SPANNER_SET
)
_EXT_MUSIC_PASSTHROUGH_IDS: FrozenSet[int] = frozenset(
    FULL_TOKEN_TO_ID[t] for t in _EXT_MUSIC_PASSTHROUGH_SET
)

# PrefixState modes where a boundary/tail-level extension or Layer 3 text span
# may cleanly start (never mid-pitch `dur`/`step`, mid-chord, or mid-ratio).
_SAFE_SPAN_MODES: FrozenSet[str] = frozenset(("bound", "pitch-tail", "tiestart", "rest-tail"))


class BekernFullGrammarState:
    """Incremental constrained-decoding FSA for ``bekern-full`` (Mitigations #3 & #8).

    Wraps ``grammar.PrefixState`` for core ``0..260`` music tokens and adds:
    1. Optional ``<schema:v1>`` / ``<schema:v2a>`` / ``<schema:full>`` prefix
       allowed **only** at position 0 (immediately after ``<bos>``).
    2. Layer 3 text-span pause: inside ``<sys_lyr>`` / ``<harm>`` / ``<tempo>``
       / ``<dir>`` / ``<reh>`` / ``<fb>`` ... ``</text_span>``, the rhythmic
       bar-duration FSA is completely paused so lyrics/text never corrupt
       measure duration tracking or note alignment.
    3. Numbered spanner parity: tracks open channels for ``slur-1..4`` and
       ``hairpin-1..3``, forbidding double-opening an already-open channel or
       closing an unopened channel.
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
        self.open_slurs: Set[int] = set()
        self.open_hairpins: Set[int] = set()

    def copy(self) -> "BekernFullGrammarState":
        c = BekernFullGrammarState.__new__(BekernFullGrammarState)
        c.music_state = self.music_state.copy()
        c.pos = self.pos
        c.schema = self.schema
        c.in_text_span = self.in_text_span
        c.active_text_tag = self.active_text_tag
        c.text_span_tokens = self.text_span_tokens
        c.open_slurs = set(self.open_slurs)
        c.open_hairpins = set(self.open_hairpins)
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

    def can_emit_sys(self) -> bool:
        if self.in_text_span:
            return False
        return self.music_state.can_emit_sys()

    def can_emit_eos(self, cur_len: Optional[int] = None, max_len: Optional[int] = None) -> bool:
        if self.in_text_span:
            return False
        return self.music_state.can_emit_eos(cur_len=cur_len, max_len=max_len)

    def _allowed_numbered_spanners(self) -> Set[int]:
        allow: Set[int] = set()
        for tok, ch in _SLUR_OPEN_MAP.items():
            if ch not in self.open_slurs:
                allow.add(FULL_TOKEN_TO_ID[tok])
        for tok, ch in _SLUR_CLOSE_MAP.items():
            if ch in self.open_slurs:
                allow.add(FULL_TOKEN_TO_ID[tok])
        for tok, ch in _HAIRPIN_OPEN_MAP.items():
            if ch not in self.open_hairpins:
                allow.add(FULL_TOKEN_TO_ID[tok])
        for tok, ch in _HAIRPIN_CLOSE_MAP.items():
            if ch in self.open_hairpins:
                allow.add(FULL_TOKEN_TO_ID[tok])
        return allow

    def allowed(self) -> Tuple[Set[int], bool]:
        """Return ``(allowed_token_ids, eos_ok)`` for the current FSA state."""
        if self.in_text_span:
            allow: Set[int] = set(_LAYER3_BODY_IDS)
            allow.add(TEXT_SPAN_CLOSE_TOKEN_ID)
            return allow, False

        allow, eos_ok = self.music_state.allowed()
        allow = set(allow)

        # Optional <schema:*> token allowed ONLY at position 0.
        if self.pos == 0:
            allow.update(SCHEMA_TOKEN_IDS)

        if self.can_emit_sys():
            allow.add(SYS_ID)

        if self.music_state.mode in _SAFE_SPAN_MODES:
            allow.update(self._allowed_numbered_spanners())
            allow.update(_EXT_MUSIC_PASSTHROUGH_IDS)
            if not self.music_state.has_open_tuplet():
                allow.update(TEXT_SPAN_OPEN_TOKEN_IDS)

        return allow, eos_ok

    def step(self, tok_or_id: Union[int, str]) -> None:
        """Advance the FSA by one token string or ID (raises ``ValueError`` if illegal)."""
        tok = self._resolve_tok(tok_or_id)

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
                self.pos += 1
                return
            if tok in _LAYER3_BODY_SET:
                self.text_span_tokens += 1
                self.pos += 1
                return
            raise ValueError(
                f"illegal token {tok!r} inside Layer 3 text span {self.active_text_tag!r}"
            )

        # 3. Opening a Layer 3 text span: pause rhythmic FSA cleanly.
        if tok in TEXT_SPAN_OPEN_TOKENS:
            if self.music_state.mode not in _SAFE_SPAN_MODES or self.music_state.has_open_tuplet():
                raise ValueError(
                    f"cannot open text span {tok!r} in music mode {self.music_state.mode!r}"
                )
            self.in_text_span = True
            self.active_text_tag = tok
            self.text_span_tokens = 0
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

        # 5. Numbered spanners: slur-1..4 and hairpin-1..3 parity.
        if tok in _SLUR_OPEN_MAP:
            if self.music_state.mode not in _SAFE_SPAN_MODES:
                raise ValueError(f"slur token {tok!r} not allowed in mode {self.music_state.mode!r}")
            ch = _SLUR_OPEN_MAP[tok]
            if ch in self.open_slurs:
                raise ValueError(f"slur channel {ch} already open ({tok!r})")
            self.open_slurs.add(ch)
            self.pos += 1
            return

        if tok in _SLUR_CLOSE_MAP:
            if self.music_state.mode not in _SAFE_SPAN_MODES:
                raise ValueError(f"slur token {tok!r} not allowed in mode {self.music_state.mode!r}")
            ch = _SLUR_CLOSE_MAP[tok]
            if ch not in self.open_slurs:
                raise ValueError(f"cannot close unopened slur channel {ch} ({tok!r})")
            self.open_slurs.remove(ch)
            self.pos += 1
            return

        if tok in _HAIRPIN_OPEN_MAP:
            if self.music_state.mode not in _SAFE_SPAN_MODES:
                raise ValueError(f"hairpin token {tok!r} not allowed in mode {self.music_state.mode!r}")
            ch = _HAIRPIN_OPEN_MAP[tok]
            if ch in self.open_hairpins:
                raise ValueError(f"hairpin channel {ch} already open ({tok!r})")
            self.open_hairpins.add(ch)
            self.pos += 1
            return

        if tok in _HAIRPIN_CLOSE_MAP:
            if self.music_state.mode not in _SAFE_SPAN_MODES:
                raise ValueError(f"hairpin token {tok!r} not allowed in mode {self.music_state.mode!r}")
            ch = _HAIRPIN_CLOSE_MAP[tok]
            if ch not in self.open_hairpins:
                raise ValueError(f"cannot close unopened hairpin channel {ch} ({tok!r})")
            self.open_hairpins.remove(ch)
            self.pos += 1
            return

        # 6. Extended Layer 0/1/2 music tokens (outside data.VOCAB).
        if tok in _EXT_MUSIC_PASSTHROUGH_SET:
            if self.music_state.mode not in _SAFE_SPAN_MODES:
                raise ValueError(
                    f"extended music token {tok!r} not allowed in mode {self.music_state.mode!r}"
                )
            self.pos += 1
            return

        # 7. Core legacy music tokens (0..260): delegate to grammar.PrefixState.
        if tok in _LEGACY_MUSIC_SET:
            self.music_state.step(tok)
            self.pos += 1
            return

        raise ValueError(f"unhandled token {tok!r}")
