"""PosFormer-style musical position-forest auxiliary head & Voice-1 CTC targets.

Training-only, zero-inference-overhead structural supervision for Stage-2 MvdNet:

1. ``extract_bekern_pos_targets(tgt_ids, id2tok, pad_id)``:
   Derives deterministic per-token musical-structure targets directly from
   bekern token sequences with zero extra manual labels:
     * ``beat_phase_bin`` in ``0..15`` (16-bin beat position within the
       current measure, derived from duration/dot/tuplet tokens and reset on
       barlines and voice switches).
     * ``voice_id`` in ``0..7`` (0-indexed active voice tracked from
       ``voice-*`` tokens, clamped to ``0..7``).
     * ``flags`` ``[in_chord, in_tuplet, is_bar_close]`` (3 binary flags).

2. ``MusicalPosHead(d_model)``:
   Lightweight MLP predicting 16-bin beat phase (CE), 8-class voice ID (CE),
   and 3 binary flags (BCE) from shared decoder hidden states ``h_shared``.

3. ``extract_voice1_ctc_targets(tgt_ids, id2tok, pad_id, bos_id, eos_id)``:
   Extracts strictly the monotonic Voice-1 token skeleton (headers + Voice-1
   tokens up to any ``voice-2``+ horizontal reset, or the full single-voice
   sequence) for column-pooled auxiliary CTC supervision.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Base quarter-note beat durations for bekern duration tokens.
DURATION_BEATS = {
    'breve': 8.0,
    'whole': 4.0,
    'half': 2.0,
    'quarter': 1.0,
    'eighth': 0.5,
    '16th': 0.25,
    '32nd': 0.125,
    '64th': 0.0625,
    '128th': 0.03125,
}

NUM_BEAT_BINS = 16
NUM_VOICE_CLASSES = 8
NUM_FLAGS = 3


def _is_barline_tok(tok: str) -> bool:
    """True for barline tokens (``bar``, ``bar-*`` except ``bar-rest``, ``barline*``)."""
    return (
        tok == 'bar'
        or tok.startswith('barline')
        or (tok.startswith('bar-') and tok != 'bar-rest')
    )


def _parse_time_sig_beats(tok: str, default: float = 4.0) -> float:
    """Parse ``time-NUM-DEN`` into measure length in quarter-note beats."""
    parts = tok.split('-')
    if len(parts) == 3 and parts[0] == 'time':
        try:
            num = int(parts[1])
            den = int(parts[2])
            if num > 0 and den > 0:
                return float(num) * (4.0 / float(den))
        except ValueError:
            pass
    return default


def _parse_tuplet_scale(tok: str) -> float:
    """Parse ``tuplet-start-ACTUAL-NORMAL`` into duration scale ``NORMAL / ACTUAL``."""
    parts = tok.split('-')
    if len(parts) >= 4 and parts[0] == 'tuplet' and parts[1] == 'start':
        try:
            actual = int(parts[2])
            normal = int(parts[3])
            if actual > 0 and normal > 0:
                return float(normal) / float(actual)
        except ValueError:
            pass
    return 2.0 / 3.0


def _parse_voice_idx(tok: str) -> int | None:
    """Parse ``voice-N`` into 0-based voice index clamped to ``0..7``."""
    if not tok.startswith('voice-'):
        return None
    try:
        v = int(tok.split('-', 1)[1])
        return min(max(v - 1, 0), NUM_VOICE_CLASSES - 1)
    except ValueError:
        return None


def _lookup_tok(id2tok, tid: int) -> str | None:
    if isinstance(id2tok, dict):
        return id2tok.get(tid)
    if 0 <= tid < len(id2tok):
        return id2tok[tid]
    return None


def _beat_to_bin(beat_pos: float, measure_beats: float) -> int:
    denom = max(float(measure_beats), 1e-6)
    phase = beat_pos / denom
    if phase <= 0.0:
        return 0
    if phase >= 1.0:
        return NUM_BEAT_BINS - 1
    b = int(phase * NUM_BEAT_BINS)
    return min(max(b, 0), NUM_BEAT_BINS - 1)


def extract_bekern_pos_targets(tgt_ids, id2tok, pad_id: int):
    """Derive deterministic structural targets per token from bekern IDs.

    Args:
        tgt_ids: ``[B, T]`` integer tensor (or nested list of ints).
        id2tok: sequence or dict mapping token ID -> bekern token string.
        pad_id: padding token ID (ignored in loss via ``valid_mask``).

    Returns:
        dict with keys:
          * ``beat_phase_bin``: ``LongTensor [B, T]`` in ``0..15``
          * ``voice_id``: ``LongTensor [B, T]`` in ``0..7``
          * ``flags``: ``FloatTensor [B, T, 3]`` for
            ``[in_chord, in_tuplet, is_bar_close]``
          * ``valid_mask``: ``BoolTensor [B, T]`` (``tgt_ids != pad_id``)
    """
    is_tensor = isinstance(tgt_ids, torch.Tensor)
    device = tgt_ids.device if is_tensor else torch.device('cpu')
    rows = tgt_ids.detach().cpu().tolist() if is_tensor else tgt_ids
    bsz = len(rows)
    seq_len = len(rows[0]) if bsz > 0 else 0

    beat_bins = torch.zeros((bsz, seq_len), dtype=torch.long)
    voice_ids = torch.zeros((bsz, seq_len), dtype=torch.long)
    flags = torch.zeros((bsz, seq_len, NUM_FLAGS), dtype=torch.float32)
    valid_mask = torch.zeros((bsz, seq_len), dtype=torch.bool)

    for b, row in enumerate(rows):
        measure_beats = 4.0
        beat_pos = 0.0
        pending_dur = 0.0
        last_dur_inc = 0.0
        voice = 0
        in_chord = False
        chord_pending_dur = False
        grace_pending_dur = False
        tuplet_stack: list[float] = []

        for t, tid in enumerate(row):
            if tid == pad_id:
                continue
            valid_mask[b, t] = True
            tok = _lookup_tok(id2tok, int(tid))
            if tok is None:
                beat_bins[b, t] = _beat_to_bin(beat_pos, measure_beats)
                voice_ids[b, t] = voice
                flags[b, t, 0] = 1.0 if in_chord else 0.0
                flags[b, t, 1] = 1.0 if tuplet_stack else 0.0
                continue

            if tok.startswith('time-'):
                measure_beats = _parse_time_sig_beats(tok, measure_beats)
                in_chord = False
                chord_pending_dur = False
                beat_bins[b, t] = _beat_to_bin(beat_pos, measure_beats)
                voice_ids[b, t] = voice
                flags[b, t, 1] = 1.0 if tuplet_stack else 0.0
                continue

            if _is_barline_tok(tok):
                # Record bar-close at the barline token, then reset measure state.
                beat_pos += pending_dur
                beat_bins[b, t] = _beat_to_bin(beat_pos, measure_beats)
                voice_ids[b, t] = voice
                flags[b, t, 2] = 1.0
                beat_pos = 0.0
                pending_dur = 0.0
                last_dur_inc = 0.0
                in_chord = False
                chord_pending_dur = False
                grace_pending_dur = False
                tuplet_stack.clear()
                continue

            v_idx = _parse_voice_idx(tok)
            if v_idx is not None:
                voice = v_idx
                beat_pos = 0.0
                pending_dur = 0.0
                last_dur_inc = 0.0
                in_chord = False
                chord_pending_dur = False
                grace_pending_dur = False
                tuplet_stack.clear()
                beat_bins[b, t] = 0
                voice_ids[b, t] = voice
                continue

            if tok == 'chord':
                in_chord = True
                chord_pending_dur = True
                beat_bins[b, t] = _beat_to_bin(beat_pos, measure_beats)
                voice_ids[b, t] = voice
                flags[b, t, 0] = 1.0
                flags[b, t, 1] = 1.0 if tuplet_stack else 0.0
                continue

            if tok == 'grace':
                grace_pending_dur = True
                beat_bins[b, t] = _beat_to_bin(beat_pos, measure_beats)
                voice_ids[b, t] = voice
                flags[b, t, 0] = 1.0 if in_chord else 0.0
                flags[b, t, 1] = 1.0 if tuplet_stack else 0.0
                continue

            if tok.startswith('tuplet-start'):
                tuplet_stack.append(_parse_tuplet_scale(tok))
                beat_bins[b, t] = _beat_to_bin(beat_pos, measure_beats)
                voice_ids[b, t] = voice
                flags[b, t, 0] = 1.0 if in_chord else 0.0
                flags[b, t, 1] = 1.0
                continue

            if tok == 'tuplet-stop':
                beat_bins[b, t] = _beat_to_bin(beat_pos, measure_beats)
                voice_ids[b, t] = voice
                flags[b, t, 0] = 1.0 if in_chord else 0.0
                flags[b, t, 1] = 1.0 if tuplet_stack else 0.0
                if tuplet_stack:
                    tuplet_stack.pop()
                continue

            if tok == 'bar-rest':
                beat_pos += pending_dur
                pending_dur = measure_beats
                last_dur_inc = measure_beats
                in_chord = False
                chord_pending_dur = False
                grace_pending_dur = False
                beat_bins[b, t] = _beat_to_bin(beat_pos, measure_beats)
                voice_ids[b, t] = voice
                flags[b, t, 1] = 1.0 if tuplet_stack else 0.0
                continue

            if tok in DURATION_BEATS:
                if grace_pending_dur:
                    grace_pending_dur = False
                elif chord_pending_dur:
                    # Chord tones share the base note's onset; do not advance beat.
                    chord_pending_dur = False
                else:
                    beat_pos += pending_dur
                    in_chord = False
                    scale = 1.0
                    for s in tuplet_stack:
                        scale *= s
                    dur = DURATION_BEATS[tok] * scale
                    pending_dur = dur
                    last_dur_inc = dur
                beat_bins[b, t] = _beat_to_bin(beat_pos, measure_beats)
                voice_ids[b, t] = voice
                flags[b, t, 0] = 1.0 if in_chord else 0.0
                flags[b, t, 1] = 1.0 if tuplet_stack else 0.0
                continue

            if tok == 'dot':
                if not in_chord and last_dur_inc > 0.0:
                    last_dur_inc *= 0.5
                    pending_dur += last_dur_inc
                beat_bins[b, t] = _beat_to_bin(beat_pos, measure_beats)
                voice_ids[b, t] = voice
                flags[b, t, 0] = 1.0 if in_chord else 0.0
                flags[b, t, 1] = 1.0 if tuplet_stack else 0.0
                continue

            # Pitch steps, octaves, accidentals, rests, clefs, keys, articulations, etc.
            beat_bins[b, t] = _beat_to_bin(beat_pos, measure_beats)
            voice_ids[b, t] = voice
            flags[b, t, 0] = 1.0 if in_chord else 0.0
            flags[b, t, 1] = 1.0 if tuplet_stack else 0.0

    return {
        'beat_phase_bin': beat_bins.to(device),
        'voice_id': voice_ids.to(device),
        'flags': flags.to(device),
        'valid_mask': valid_mask.to(device),
    }


def extract_voice1_ctc_targets(tgt_ids, id2tok, pad_id: int,
                               bos_id: int, eos_id: int):
    """Extract strictly monotonic Voice-1 skeleton targets for CTC loss.

    Keeps headers + Voice-1 tokens up to the first ``voice-N`` (``N >= 2``)
    horizontal reset (or the full sequence for single-voice strips), stripping
    ``pad_id``, ``bos_id``, and ``eos_id``. Multi-voice horizontal resets
    therefore never violate CTC monotonicity.

    Returns:
        ``(targets_1d, target_lengths)`` suitable for ``F.ctc_loss``.
    """
    is_tensor = isinstance(tgt_ids, torch.Tensor)
    device = tgt_ids.device if is_tensor else torch.device('cpu')
    rows = tgt_ids.detach().cpu().tolist() if is_tensor else tgt_ids

    flat: list[int] = []
    lengths: list[int] = []
    for row in rows:
        seq: list[int] = []
        for tid in row:
            tid = int(tid)
            if tid == pad_id or tid == bos_id:
                continue
            if tid == eos_id:
                break
            tok = _lookup_tok(id2tok, tid)
            if tok is not None and tok.startswith('voice-') and tok != 'voice-1':
                break
            seq.append(tid)
        flat.extend(seq)
        lengths.append(len(seq))

    return (
        torch.tensor(flat, dtype=torch.long, device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )


class MusicalPosHead(nn.Module):
    """PosFormer-style musical position-forest auxiliary head (ECCV 2024).

    Predicts per-token 16-bin measure beat phase (CE), 8-class active voice ID
    (CE), and 3 binary structural flags ``[in_chord, in_tuplet, is_bar_close]``
    (BCE) from shared decoder hidden states ``h_shared``. Training-only;
    dropped cleanly at inference for 0% inference overhead.
    """

    def __init__(self, d_model: int, num_beat_bins: int = NUM_BEAT_BINS,
                 num_voices: int = NUM_VOICE_CLASSES,
                 num_flags: int = NUM_FLAGS):
        super().__init__()
        self.d_model = int(d_model)
        self.num_beat_bins = int(num_beat_bins)
        self.num_voices = int(num_voices)
        self.num_flags = int(num_flags)
        self.norm = nn.LayerNorm(self.d_model)
        self.fc = nn.Linear(self.d_model, self.d_model)
        self.beat_head = nn.Linear(self.d_model, self.num_beat_bins)
        self.voice_head = nn.Linear(self.d_model, self.num_voices)
        self.flag_head = nn.Linear(self.d_model, self.num_flags)

    def forward(self, h: torch.Tensor):
        z = F.gelu(self.fc(self.norm(h)))
        return self.beat_head(z), self.voice_head(z), self.flag_head(z)

    def compute_loss(self, h: torch.Tensor, targets: dict) -> torch.Tensor:
        beat_logits, voice_logits, flag_logits = self.forward(h)
        valid = targets['valid_mask']
        if not bool(valid.any()):
            return (beat_logits.sum() + voice_logits.sum() + flag_logits.sum()) * 0.0

        beat_tgt = targets['beat_phase_bin'].masked_fill(~valid, -100)
        voice_tgt = targets['voice_id'].masked_fill(~valid, -100)
        loss_beat = F.cross_entropy(
            beat_logits.float().reshape(-1, self.num_beat_bins),
            beat_tgt.reshape(-1),
            ignore_index=-100,
        )
        loss_voice = F.cross_entropy(
            voice_logits.float().reshape(-1, self.num_voices),
            voice_tgt.reshape(-1),
            ignore_index=-100,
        )
        flag_bce = F.binary_cross_entropy_with_logits(
            flag_logits.float(),
            targets['flags'].float(),
            reduction='none',
        ).mean(dim=-1)
        v_f = valid.float()
        loss_flags = (flag_bce * v_f).sum() / v_f.sum().clamp_min(1.0)
        return (loss_beat + loss_voice + loss_flags) / 3.0

    def loss_from_tokens(self, h: torch.Tensor, tgt_ids: torch.Tensor,
                         id2tok, pad_id: int) -> torch.Tensor:
        targets = extract_bekern_pos_targets(tgt_ids, id2tok=id2tok, pad_id=pad_id)
        return self.compute_loss(h, targets)
