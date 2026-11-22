"""S18: Factored Event-Tuple Decoder Head + Zero-Prefix Visual Readout Head.

Replaces unfactored flat AR bekern strings (where one missed dot or
articulation shifts every downstream token, causing voice-lane cascades,
rest floods, and 2-4.8x attachment spray) with per-event factored tuples:

  * ``kind``: categorical event kind (NOTE, CHORD_NOTE, GRACE_NOTE, REST,
    BAR_REST, BAR, VOICE, HEADER, TUPLET_START, TUPLET_STOP, DYN, HAIRPIN,
    ARTIC, TIE, SLUR, EOS, PAD)
  * ``voice``: categorical voice lane (0 = none/inherited, 1..12 = voice-1..12)
  * ``dur``: categorical duration (0 = none, 1..9 = whole..breve)
  * ``pitch``: categorical diatonic pitch (0..62 = C0..B8, 63 = NO_PITCH)
  * ``alter``: categorical accidental (0 = none, 1..4 = a-2, a-1, a1, a2)
  * ``attach``: independent multi-label Bernoulli vector (10 bits:
    dot1, dot2, tie_start, tie_stop, slur_start, slur_stop, stacc, accent,
    tenuto, fermata) — never coupled sequentially, so a missed dot or
    articulation cannot derail downstream events.

Also provides :class:`ZeroPrefixVisualReadout`: a non-autoregressive visual
readout head that predicts factored event tuples strictly from encoder
memory ``mem`` via learned visual slot queries (ZERO token-prefix
conditioning), serving both as an exposure-bias-free visual probe and an
auxiliary visual-grounding head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import tuple_events as te

EVENT_KINDS = (
    'PAD',
    'EOS',
    'NOTE',
    'CHORD_NOTE',
    'GRACE_NOTE',
    'REST',
    'BAR_REST',
    'BAR',
    'VOICE',
    'HEADER',
    'TUPLET_START',
    'TUPLET_STOP',
    'DYN',
    'HAIRPIN',
    'ARTIC',
    'TIE',
    'SLUR',
)
KIND_TO_ID = {k: i for i, k in enumerate(EVENT_KINDS)}
PAD_KIND_ID = KIND_TO_ID['PAD']
EOS_KIND_ID = KIND_TO_ID['EOS']
N_KINDS = len(EVENT_KINDS)

MAX_VOICE = 12
N_VOICES = MAX_VOICE + 1  # 0 = none/inherited, 1..12 = voice-1..voice-12

DUR_LIST = (
    None,
    'whole', 'half', 'quarter', 'eighth', '16th',
    '32nd', '64th', '128th', 'breve',
)
DUR_TO_ID = {d: i for i, d in enumerate(DUR_LIST)}
N_DURS = len(DUR_LIST)

N_PITCHES = te.NO_PITCH + 1  # 0..62 = C0..B8, 63 = NO_PITCH
STEP_LIST = ('C', 'D', 'E', 'F', 'G', 'A', 'B')

ALTER_LIST = (None, 'a-2', 'a-1', 'a1', 'a2')
ALTER_TO_ID = {a: i for i, a in enumerate(ALTER_LIST)}
N_ALTERS = len(ALTER_LIST)

ATTACH_NAMES = (
    'dot1',
    'dot2',
    'tie_start',
    'tie_stop',
    'slur_start',
    'slur_stop',
    'stacc',
    'accent',
    'tenuto',
    'fermata',
)
ATTACH_TO_IDX = {name: i for i, name in enumerate(ATTACH_NAMES)}
N_ATTACH = len(ATTACH_NAMES)

PITCH_KINDS = frozenset(
    KIND_TO_ID[k] for k in ('NOTE', 'CHORD_NOTE', 'GRACE_NOTE')
)
TIMED_KINDS = frozenset(
    KIND_TO_ID[k] for k in ('NOTE', 'CHORD_NOTE', 'GRACE_NOTE', 'REST')
)
VOICE_KINDS = frozenset(
    KIND_TO_ID[k] for k in ('NOTE', 'GRACE_NOTE', 'REST', 'BAR_REST', 'VOICE')
)


def diatonic_to_step_oct(idx: int) -> tuple[str, int]:
    """Inverse of :func:`tuple_events.diatonic_index` (0..62 -> (step, oct))."""
    if not 0 <= int(idx) < te.NO_PITCH:
        raise ValueError(f'invalid diatonic pitch index: {idx}')
    return STEP_LIST[int(idx) % 7], int(idx) // 7


def event_to_factors(ev: dict) -> dict[str, object]:
    """Convert one :mod:`tuple_events` dict into integer/binary factor targets."""
    kind = ev['kind']
    if kind not in KIND_TO_ID:
        raise ValueError(f'unknown event kind: {kind!r}')
    kind_id = KIND_TO_ID[kind]
    if kind == 'VOICE':
        voice_id = int(ev.get('n') or 0)
    else:
        voice_id = int(ev.get('voice') or 0)
    voice_id = min(max(voice_id, 0), MAX_VOICE)
    dur_id = DUR_TO_ID.get(ev.get('dur'), 0)
    pitch_id = int(ev.get('diatonic_pitch', te.NO_PITCH))
    alter_id = ALTER_TO_ID.get(ev.get('alter'), 0)

    attach = [0.0] * N_ATTACH
    dots = int(ev.get('dots', 0) or 0)
    if dots >= 1:
        attach[ATTACH_TO_IDX['dot1']] = 1.0
    if dots >= 2:
        attach[ATTACH_TO_IDX['dot2']] = 1.0
    ties = ev.get('ties', ())
    if 'start' in ties:
        attach[ATTACH_TO_IDX['tie_start']] = 1.0
    if 'stop' in ties:
        attach[ATTACH_TO_IDX['tie_stop']] = 1.0
    slurs = ev.get('slurs', ())
    if 'start' in slurs:
        attach[ATTACH_TO_IDX['slur_start']] = 1.0
    if 'stop' in slurs:
        attach[ATTACH_TO_IDX['slur_stop']] = 1.0
    for art in ev.get('articulations', ()):
        if art in ATTACH_TO_IDX:
            attach[ATTACH_TO_IDX[art]] = 1.0

    return {
        'kind': kind_id,
        'voice': voice_id,
        'dur': dur_id,
        'pitch': pitch_id,
        'alter': alter_id,
        'attach': attach,
    }


def encode_events_to_tensors(
    events: list[dict],
    append_eos: bool = True,
    device: torch.device | str = 'cpu',
) -> dict[str, torch.Tensor]:
    """Pack a list of :mod:`tuple_events` dicts into ``[T]`` / ``[T, N_ATTACH]`` tensors."""
    rows = [event_to_factors(ev) for ev in events]
    if append_eos:
        rows.append({
            'kind': EOS_KIND_ID,
            'voice': 0,
            'dur': 0,
            'pitch': te.NO_PITCH,
            'alter': 0,
            'attach': [0.0] * N_ATTACH,
        })
    return {
        'kind': torch.tensor([r['kind'] for r in rows], dtype=torch.long, device=device),
        'voice': torch.tensor([r['voice'] for r in rows], dtype=torch.long, device=device),
        'dur': torch.tensor([r['dur'] for r in rows], dtype=torch.long, device=device),
        'pitch': torch.tensor([r['pitch'] for r in rows], dtype=torch.long, device=device),
        'alter': torch.tensor([r['alter'] for r in rows], dtype=torch.long, device=device),
        'attach': torch.tensor([r['attach'] for r in rows], dtype=torch.float32, device=device),
    }


def factors_to_event(
    kind_id: int,
    voice_id: int,
    dur_id: int,
    pitch_id: int,
    alter_id: int,
    attach_bits,
) -> dict | None:
    """Reconstruct a :mod:`tuple_events`-compatible event dict from factor ids.

    Returns ``None`` on ``PAD`` or ``EOS``.
    """
    kind = EVENT_KINDS[int(kind_id)]
    if kind in ('PAD', 'EOS'):
        return None
    if kind == 'VOICE':
        return {'kind': 'VOICE', 'n': max(1, int(voice_id)),
                'diatonic_pitch': te.NO_PITCH}
    if kind == 'BAR':
        return {'kind': 'BAR', 'variant': None, 'diatonic_pitch': te.NO_PITCH}
    if kind == 'BAR_REST':
        return {'kind': 'BAR_REST',
                'voice': int(voice_id) if int(voice_id) > 0 else None,
                'diatonic_pitch': te.NO_PITCH}
    if kind in ('NOTE', 'CHORD_NOTE', 'GRACE_NOTE', 'REST'):
        dur = DUR_LIST[int(dur_id)] or 'quarter'
        alter = ALTER_LIST[int(alter_id)]
        bits = [bool(b) for b in attach_bits]
        dots = (2 if (bits[ATTACH_TO_IDX['dot1']] and bits[ATTACH_TO_IDX['dot2']])
                else (1 if bits[ATTACH_TO_IDX['dot1']] else 0))
        ties = []
        if bits[ATTACH_TO_IDX['tie_start']]:
            ties.append('start')
        if bits[ATTACH_TO_IDX['tie_stop']]:
            ties.append('stop')
        slurs = []
        if bits[ATTACH_TO_IDX['slur_start']]:
            slurs.append('start')
        if bits[ATTACH_TO_IDX['slur_stop']]:
            slurs.append('stop')
        artics = [a for a in ('stacc', 'accent', 'tenuto', 'fermata')
                  if bits[ATTACH_TO_IDX[a]]]
        mods = []
        if alter is not None:
            mods.append(alter)
        mods.extend(['dot'] * dots)
        mods.extend('tie-' + t for t in ties)
        mods.extend('slur-' + s for s in slurs)
        mods.extend(artics)
        base = {
            'kind': kind,
            'voice': None if kind == 'CHORD_NOTE' else (
                int(voice_id) if int(voice_id) > 0 else None),
            'dur': dur,
            'mods': mods,
            'alter': alter,
            'dots': dots,
            'ties': ties,
            'slurs': slurs,
            'articulations': artics,
            'dyn': None,
            'hairpins': [],
        }
        if kind == 'REST':
            base['diatonic_pitch'] = te.NO_PITCH
            return base
        pid = int(pitch_id)
        if pid >= te.NO_PITCH:
            pid = te.diatonic_index('C', 4)
        step, octv = diatonic_to_step_oct(pid)
        base.update({
            'step': step,
            'oct': octv,
            'diatonic_pitch': pid,
            'grace': kind == 'GRACE_NOTE',
        })
        return base
    return {'kind': kind, 'diatonic_pitch': te.NO_PITCH}


class FactoredEventEmbedding(nn.Module):
    """Embed factored event tuples into a single ``d_model`` vector per event."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        self.kind_emb = nn.Embedding(N_KINDS, d_model, padding_idx=PAD_KIND_ID)
        self.voice_emb = nn.Embedding(N_VOICES, d_model)
        self.dur_emb = nn.Embedding(N_DURS, d_model)
        self.pitch_emb = nn.Embedding(N_PITCHES, d_model)
        self.alter_emb = nn.Embedding(N_ALTERS, d_model)
        self.attach_proj = nn.Linear(N_ATTACH, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        kind: torch.Tensor,
        voice: torch.Tensor,
        dur: torch.Tensor,
        pitch: torch.Tensor,
        alter: torch.Tensor,
        attach: torch.Tensor,
    ) -> torch.Tensor:
        x = (
            self.kind_emb(kind)
            + self.voice_emb(voice)
            + self.dur_emb(dur)
            + self.pitch_emb(pitch)
            + self.alter_emb(alter)
            + self.attach_proj(attach.to(self.attach_proj.weight.dtype))
        )
        return self.norm(x)


class FactoredEventHead(nn.Module):
    """Decoupled S18 event-tuple prediction heads (categorical + Bernoulli).

    Predicts six independent factor heads from a ``[B, T, d_model]``
    hidden state:
      * ``kind``   ``[B, T, N_KINDS]``  (CE)
      * ``voice``  ``[B, T, N_VOICES]`` (CE, masked to voice-bearing events)
      * ``dur``    ``[B, T, N_DURS]``   (CE, masked to timed events)
      * ``pitch``  ``[B, T, N_PITCHES]`` (CE, masked to note events)
      * ``alter``  ``[B, T, N_ALTERS]`` (CE, masked to note events)
      * ``attach`` ``[B, T, N_ATTACH]`` (independent per-bit Bernoulli BCE,
        masked to timed events)

    Because each head reads directly from ``h`` in parallel, an attachment
    error (e.g. a missed dot or staccato) never alters the sequence length
    or shifts downstream pitch/duration/voice predictions.
    """

    def __init__(self, d_model: int = 256, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.drop = nn.Dropout(dropout)
        self.kind_head = nn.Linear(d_model, N_KINDS)
        self.voice_head = nn.Linear(d_model, N_VOICES)
        self.dur_head = nn.Linear(d_model, N_DURS)
        self.pitch_head = nn.Linear(d_model, N_PITCHES)
        self.alter_head = nn.Linear(d_model, N_ALTERS)
        self.attach_head = nn.Linear(d_model, N_ATTACH)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.drop(h)
        return {
            'kind': self.kind_head(h),
            'voice': self.voice_head(h),
            'dur': self.dur_head(h),
            'pitch': self.pitch_head(h),
            'alter': self.alter_head(h),
            'attach': self.attach_head(h),
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        weights: dict[str, float] | None = None,
        attach_pos_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Masked multi-head factored loss with per-factor breakdown."""
        w = {
            'kind': 1.0,
            'voice': 1.0,
            'dur': 1.0,
            'pitch': 1.5,
            'alter': 1.0,
            'attach': 1.0,
        }
        if weights:
            w.update(weights)

        kind_t = targets['kind']
        valid = kind_t != PAD_KIND_ID

        def _masked_ce(logits: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            if not bool(mask.any()):
                return logits.sum() * 0.0
            return F.cross_entropy(logits[mask], tgt[mask])

        pitch_ids = torch.tensor(sorted(PITCH_KINDS), device=kind_t.device)
        timed_ids = torch.tensor(sorted(TIMED_KINDS), device=kind_t.device)
        voice_ids = torch.tensor(sorted(VOICE_KINDS), device=kind_t.device)

        is_pitch = valid & torch.isin(kind_t, pitch_ids)
        is_timed = valid & torch.isin(kind_t, timed_ids)
        is_voice = valid & torch.isin(kind_t, voice_ids)

        losses = {
            'kind': _masked_ce(outputs['kind'], kind_t, valid),
            'voice': _masked_ce(outputs['voice'], targets['voice'], is_voice),
            'dur': _masked_ce(outputs['dur'], targets['dur'], is_timed),
            'pitch': _masked_ce(outputs['pitch'], targets['pitch'], is_pitch),
            'alter': _masked_ce(outputs['alter'], targets['alter'], is_pitch),
        }
        if bool(is_timed.any()):
            pw = (torch.full((N_ATTACH,), float(attach_pos_weight),
                             device=kind_t.device, dtype=outputs['attach'].dtype)
                  if attach_pos_weight != 1.0 else None)
            losses['attach'] = F.binary_cross_entropy_with_logits(
                outputs['attach'][is_timed],
                targets['attach'][is_timed].to(outputs['attach'].dtype),
                pos_weight=pw,
            )
        else:
            losses['attach'] = outputs['attach'].sum() * 0.0

        total = sum(w[k] * losses[k] for k in losses)
        return total, losses

    @torch.no_grad()
    def decode_greedy(
        self,
        outputs: dict[str, torch.Tensor],
        attach_thresh: float = 0.5,
    ) -> list[list[dict]]:
        """Decode ``[B, T, ...]`` logits into per-batch lists of event dicts."""
        kind_ids = outputs['kind'].argmax(dim=-1).tolist()
        voice_ids = outputs['voice'].argmax(dim=-1).tolist()
        dur_ids = outputs['dur'].argmax(dim=-1).tolist()
        pitch_ids = outputs['pitch'].argmax(dim=-1).tolist()
        alter_ids = outputs['alter'].argmax(dim=-1).tolist()
        attach_bits = (torch.sigmoid(outputs['attach']) >= attach_thresh).tolist()

        batch_events = []
        for b in range(len(kind_ids)):
            evs = []
            for t in range(len(kind_ids[b])):
                k = kind_ids[b][t]
                if k == EOS_KIND_ID:
                    break
                if k == PAD_KIND_ID:
                    continue
                ev = factors_to_event(
                    k, voice_ids[b][t], dur_ids[b][t],
                    pitch_ids[b][t], alter_ids[b][t], attach_bits[b][t],
                )
                if ev is not None:
                    evs.append(ev)
            batch_events.append(evs)
        return batch_events


class ZeroPrefixVisualReadout(nn.Module):
    """Zero-prefix non-autoregressive visual readout head (S18 probe/aux).

    Reads factored event tuples directly from encoder visual memory ``mem``
    (``[B, S, d_model]``) using ``num_queries`` learned slot queries with
    cross-attention ONLY (no token prefix input, no causal mask). This
    strictly guarantees zero language-model prior contamination: any
    change in prefix history leaves its output bitwise identical, while
    changes in visual memory directly drive the factored predictions.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_queries: int = 64,
        heads: int = 8,
        layers: int = 2,
        dim_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = int(num_queries)
        self.slot_queries = nn.Parameter(
            torch.randn(1, self.num_queries, d_model) * (d_model ** -0.5)
        )
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(
                    d_model, heads, dropout=dropout, batch_first=True),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, dim_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_ff, d_model),
                    nn.Dropout(dropout),
                ),
                'norm2': nn.LayerNorm(d_model),
            })
            for _ in range(layers)
        ])
        self.head = FactoredEventHead(d_model=d_model, dropout=dropout)

    def forward(
        self,
        mem: torch.Tensor,
        mem_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Cross-attend learned visual slots to ``mem`` with zero prefix input."""
        b = mem.shape[0]
        q = self.slot_queries.expand(b, -1, -1)
        last_attn = None
        for blk in self.layers:
            attn_out, attn_w = blk['cross_attn'](
                q, mem, mem,
                key_padding_mask=mem_mask,
                need_weights=return_attn,
                average_attn_weights=True,
            )
            q = blk['norm1'](q + attn_out)
            q = blk['norm2'](q + blk['ffn'](q))
            if return_attn:
                last_attn = attn_w
        out = self.head(q)
        if return_attn:
            return out, last_attn
        return out
