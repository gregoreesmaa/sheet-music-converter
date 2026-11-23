"""S18: Factored Event-Tuple Decoder Head + Micro-Glyph Focal Weighting + Zero-Prefix Readout.

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
  * ``subtype``: categorical non-note subtype covering all bar variants,
    clef/key/time headers, tuplet ratios, dynamics, hairpins, and standalone
    articulations/ties/slurs (lossless round-trip across the full vocab)
  * ``attach``: independent calibrated multi-label Bernoulli vector (10 bits:
    dot1, dot2, tie_start, tie_stop, slur_start, slur_stop, stacc, accent,
    tenuto, fermata) — prior-initialized, temperature-calibrated, with a
    marginal-rate calibration penalty to structurally kill ~2x attachment spray.

Also includes:
  * **Batch 4-B3 inverse-footprint micro-glyph focal loss** weighting
    (higher weight on small 3-5 px marks: dots, accidentals, flags/beams,
    articulations, ledger-line pitches).
  * **Factored-to-token consistency loss** aligning factored head marginals
    with flat bekern token logits via symmetric JSD.
  * :class:`S18FactoredHead`: clean, modular plug-in wrapper for ``model.py`` /
    ``train.py`` returning ``(total_loss, metrics_dict)`` from either event
    targets or raw bekern token IDs.
  * :class:`ZeroPrefixVisualReadout`: non-autoregressive visual readout head
    conditioned strictly on encoder memory ``mem`` (zero prefix input).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import data
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


def _build_subtype_list() -> tuple[str | None, ...]:
    """Deterministic closed list of all non-note event subtypes from data.VOCAB."""
    items: list[str | None] = [None]
    seen: set[str] = set()

    def _add(tok: str) -> None:
        if tok not in seen:
            seen.add(tok)
            items.append(tok)

    # Barline variants (plain 'bar' maps to None / id 0).
    for tok in ('bar-double', 'bar-light-heavy', 'bar-repeat-left', 'bar-repeat-right'):
        _add(tok)
    # Clef, key, and time-signature headers (in VOCAB order).
    for tok in data.VOCAB:
        if tok.startswith('clef-') or tok.startswith('key') or tok.startswith('time-'):
            _add(tok)
    # Tuplet-start ratios (in VOCAB order).
    for tok in data.VOCAB:
        if tok.startswith('tuplet-start-'):
            _add(tok)
    # Dynamics, hairpins, standalone articulations, ties, and slurs.
    for tok in ('dyn-pp', 'dyn-p', 'dyn-mp', 'dyn-mf', 'dyn-f', 'dyn-ff', 'dyn-sfz'):
        _add(tok)
    for tok in ('hairpin-cresc-start', 'hairpin-dim-start', 'hairpin-stop'):
        _add(tok)
    for tok in ('stacc', 'accent', 'tenuto', 'fermata'):
        _add(tok)
    for tok in ('tie-start', 'tie-stop', 'slur-start', 'slur-stop'):
        _add(tok)
    return tuple(items)


SUBTYPE_LIST = _build_subtype_list()
SUBTYPE_TO_ID = {s: i for i, s in enumerate(SUBTYPE_LIST)}
N_SUBTYPES = len(SUBTYPE_LIST)

PITCH_KINDS = frozenset(
    KIND_TO_ID[k] for k in ('NOTE', 'CHORD_NOTE', 'GRACE_NOTE')
)
TIMED_KINDS = frozenset(
    KIND_TO_ID[k] for k in ('NOTE', 'CHORD_NOTE', 'GRACE_NOTE', 'REST')
)
VOICE_KINDS = frozenset(
    KIND_TO_ID[k] for k in ('NOTE', 'GRACE_NOTE', 'REST', 'BAR_REST', 'VOICE')
)
SUBTYPE_KINDS = frozenset(
    KIND_TO_ID[k] for k in (
        'BAR', 'HEADER', 'TUPLET_START', 'DYN', 'HAIRPIN',
        'ARTIC', 'TIE', 'SLUR',
    )
)

# ---------------------------------------------------------------------------
# Batch 4-B3: Inverse-footprint micro-glyph weights.
# Small 3-5 px marks (augmentation dots, accidentals, flags/beams on short
# durations, staccato/articulations, ledger-line pitches) receive higher loss
# weight than large, high-footprint glyphs (quarter/half/whole notes, bars).
# ---------------------------------------------------------------------------

# Aligned with DUR_LIST:
# (None, whole, half, quarter, eighth, 16th, 32nd, 64th, 128th, breve)
MICRO_GLYPH_DUR_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.0, 1.0)

# Aligned with ALTER_LIST: (None, a-2, a-1, a1, a2)
MICRO_GLYPH_ALTER_WEIGHTS = (1.0, 2.5, 2.5, 2.5, 2.5)

# Aligned with ATTACH_NAMES:
# (dot1, dot2, tie_start, tie_stop, slur_start, slur_stop, stacc, accent, tenuto, fermata)
MICRO_GLYPH_ATTACH_WEIGHTS = (3.0, 3.5, 1.5, 1.5, 1.5, 1.5, 3.0, 2.0, 2.0, 2.0)

# Standard grand-staff in-staff range: G2 (idx 18) .. F5 (idx 38).
# Pitches outside this band require ledger lines (thin 2-3 px strokes).
LEDGER_LOW_MAX_IDX = te.diatonic_index('G', 2) - 1   # < G2 (0..17)
LEDGER_HIGH_MIN_IDX = te.diatonic_index('F', 5) + 1  # > F5 (39..62)


def build_pitch_micro_glyph_weights(
    ledger_weight: float = 2.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return ``[N_PITCHES]`` inverse-footprint weights boosting ledger-line pitches."""
    w = torch.ones(N_PITCHES, dtype=torch.float32, device=device)
    for idx in range(te.NO_PITCH):
        if idx <= LEDGER_LOW_MAX_IDX or idx >= LEDGER_HIGH_MIN_IDX:
            w[idx] = float(ledger_weight)
    return w


def token_micro_glyph_weights(
    vocab: list[str] | tuple[str, ...] | None = None,
    vocab_size: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return ``[V]`` inverse-footprint weights for flat bekern token logits.

    Useful for applying Batch 4-B3 micro-glyph weighting directly to standard
    flat token CE/focal loss as well as factored heads.
    """
    vlist = data.VOCAB if vocab is None else vocab
    vlen = max(len(vlist), int(vocab_size or 0))
    w = torch.ones(vlen, dtype=torch.float32, device=device)
    for i, tok in enumerate(vlist):
        if tok == 'dot':
            w[i] = 3.0
        elif tok == 'stacc':
            w[i] = 3.0
        elif tok in ('a-2', 'a-1', 'a1', 'a2'):
            w[i] = 2.5
        elif tok in ('64th', '128th'):
            w[i] = 3.0
        elif tok == '32nd':
            w[i] = 2.5
        elif tok == '16th':
            w[i] = 2.0
        elif tok == 'eighth':
            w[i] = 1.5
        elif tok in ('accent', 'tenuto', 'fermata'):
            w[i] = 2.0
        elif tok in ('tie-start', 'tie-stop', 'slur-start', 'slur-stop', 'grace'):
            w[i] = 1.5
        elif tok in ('0', '1', '2', '6', '7', '8'):
            # Extreme octaves strongly correlate with ledger-line pitches.
            w[i] = 1.75
    return w


def micro_glyph_focal_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal cross-entropy with inverse-footprint micro-glyph class weights.

    Args:
        logits: ``[N, C]`` unnormalized class logits.
        targets: ``[N]`` integer class targets.
        class_weights: Optional ``[C]`` per-class inverse-footprint weights.
        gamma: Focal focusing exponent (``0.0`` recovers weighted CE).

    Returns:
        Scalar loss tensor.
    """
    if targets.numel() == 0:
        return logits.float().sum() * 0.0
    ce = F.cross_entropy(logits.float(), targets, reduction='none')
    if gamma > 0.0:
        pt = torch.exp(-ce).clamp(1e-6, 1.0)
        loss = ((1.0 - pt) ** float(gamma)) * ce
    else:
        loss = ce
    if class_weights is not None:
        cw = class_weights.to(device=loss.device, dtype=loss.dtype)
        loss = loss * cw[targets]
    return loss.mean()


def diatonic_to_step_oct(idx: int) -> tuple[str, int]:
    """Inverse of :func:`tuple_events.diatonic_index` (0..62 -> (step, oct))."""
    if not 0 <= int(idx) < te.NO_PITCH:
        raise ValueError(f'invalid diatonic pitch index: {idx}')
    return STEP_LIST[int(idx) % 7], int(idx) // 7


def _event_subtype_token(ev: dict) -> str | None:
    """Extract canonical subtype token for non-note events (None if unsubtyped)."""
    kind = ev['kind']
    if kind == 'BAR':
        var = ev.get('variant')
        return None if var is None else ('bar-' + str(var))
    if kind == 'HEADER':
        return ev.get('token')
    if kind == 'TUPLET_START':
        ratio = ev.get('ratio')
        return None if ratio is None else ('tuplet-start-' + str(ratio))
    if kind in ('DYN', 'HAIRPIN', 'ARTIC'):
        return ev.get('mark')
    if kind == 'TIE':
        mark = ev.get('mark')
        return None if mark is None else ('tie-' + str(mark))
    if kind == 'SLUR':
        mark = ev.get('mark')
        return None if mark is None else ('slur-' + str(mark))
    return None


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

    sub_tok = _event_subtype_token(ev)
    if sub_tok is not None and sub_tok not in SUBTYPE_TO_ID:
        raise ValueError(f'unknown event subtype {sub_tok!r} for kind {kind!r}')
    subtype_id = SUBTYPE_TO_ID.get(sub_tok, 0)

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
        'subtype': subtype_id,
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
            'subtype': 0,
            'attach': [0.0] * N_ATTACH,
        })
    return {
        'kind': torch.tensor([r['kind'] for r in rows], dtype=torch.long, device=device),
        'voice': torch.tensor([r['voice'] for r in rows], dtype=torch.long, device=device),
        'dur': torch.tensor([r['dur'] for r in rows], dtype=torch.long, device=device),
        'pitch': torch.tensor([r['pitch'] for r in rows], dtype=torch.long, device=device),
        'alter': torch.tensor([r['alter'] for r in rows], dtype=torch.long, device=device),
        'subtype': torch.tensor([r['subtype'] for r in rows], dtype=torch.long, device=device),
        'attach': torch.tensor([r['attach'] for r in rows], dtype=torch.float32, device=device),
    }


def factors_to_event(
    kind_id: int,
    voice_id: int,
    dur_id: int,
    pitch_id: int,
    alter_id: int,
    attach_bits,
    subtype_id: int = 0,
) -> dict | None:
    """Reconstruct a :mod:`tuple_events`-compatible event dict from factor ids.

    Losslessly reconstructs all 17 event kinds including non-note subtypes
    (bar variants, clef/key/time headers, tuplet ratios, dynamics, hairpins,
    standalone articulations/ties/slurs). Returns ``None`` on ``PAD`` or ``EOS``.
    """
    kind = EVENT_KINDS[int(kind_id)]
    if kind in ('PAD', 'EOS'):
        return None
    sub_tok = (
        SUBTYPE_LIST[int(subtype_id)]
        if 0 <= int(subtype_id) < N_SUBTYPES else None
    )
    if kind == 'VOICE':
        return {'kind': 'VOICE', 'n': max(1, int(voice_id)),
                'diatonic_pitch': te.NO_PITCH}
    if kind == 'BAR':
        var = (sub_tok[4:] if (sub_tok is not None and sub_tok.startswith('bar-'))
               else None)
        return {'kind': 'BAR', 'variant': var, 'diatonic_pitch': te.NO_PITCH}
    if kind == 'BAR_REST':
        return {'kind': 'BAR_REST',
                'voice': int(voice_id) if int(voice_id) > 0 else None,
                'diatonic_pitch': te.NO_PITCH}
    if kind == 'HEADER':
        tok = (sub_tok if (sub_tok is not None and (
            sub_tok.startswith('clef-') or sub_tok.startswith('key')
            or sub_tok.startswith('time-'))) else 'clef-G-2')
        return {'kind': 'HEADER', 'token': tok, 'diatonic_pitch': te.NO_PITCH}
    if kind == 'TUPLET_START':
        ratio = (sub_tok[len('tuplet-start-'):]
                 if (sub_tok is not None and sub_tok.startswith('tuplet-start-'))
                 else '3-2')
        return {'kind': 'TUPLET_START', 'ratio': ratio,
                'diatonic_pitch': te.NO_PITCH}
    if kind == 'TUPLET_STOP':
        return {'kind': 'TUPLET_STOP', 'diatonic_pitch': te.NO_PITCH}
    if kind == 'DYN':
        mark = sub_tok if (sub_tok in te.DYNS) else 'dyn-mf'
        return {'kind': 'DYN', 'mark': mark, 'diatonic_pitch': te.NO_PITCH}
    if kind == 'HAIRPIN':
        mark = sub_tok if (sub_tok in te.HAIRPINS) else 'hairpin-cresc-start'
        return {'kind': 'HAIRPIN', 'mark': mark, 'diatonic_pitch': te.NO_PITCH}
    if kind == 'ARTIC':
        mark = sub_tok if (sub_tok in te.ARTICS) else 'stacc'
        return {'kind': 'ARTIC', 'mark': mark, 'diatonic_pitch': te.NO_PITCH}
    if kind == 'TIE':
        mark = (sub_tok[4:] if sub_tok in ('tie-start', 'tie-stop') else 'start')
        return {'kind': 'TIE', 'mark': mark, 'diatonic_pitch': te.NO_PITCH}
    if kind == 'SLUR':
        mark = (sub_tok[5:] if sub_tok in ('slur-start', 'slur-stop') else 'start')
        return {'kind': 'SLUR', 'mark': mark, 'diatonic_pitch': te.NO_PITCH}
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


def align_token_ids_to_factors(
    tgt_ids: torch.Tensor,
    pad_id: int = data.PAD_ID,
    eos_id: int = data.EOS_ID,
    vocab: list[str] | tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Align a batch of flat bekern token IDs ``[B, T]`` to per-position factor targets.

    Parses each row's bekern token stream into :mod:`tuple_events` events and
    broadcasts each event's factor tuple across the exact token span that
    emitted it, so :class:`S18FactoredHead` can supervise decoder hidden states
    ``h: [B, T, d_model]`` directly without changing sequence length.

    Truncated or malformed trailing tokens safely fall back to ``PAD_KIND_ID``
    so training batches never crash.
    """
    vlist = data.VOCAB if vocab is None else vocab
    vlen = len(vlist)
    device = tgt_ids.device
    b, t = tgt_ids.shape

    kind = torch.full((b, t), PAD_KIND_ID, dtype=torch.long)
    voice = torch.zeros((b, t), dtype=torch.long)
    dur = torch.zeros((b, t), dtype=torch.long)
    pitch = torch.full((b, t), te.NO_PITCH, dtype=torch.long)
    alter = torch.zeros((b, t), dtype=torch.long)
    subtype = torch.zeros((b, t), dtype=torch.long)
    attach = torch.zeros((b, t, N_ATTACH), dtype=torch.float32)

    rows_cpu = tgt_ids.detach().cpu().tolist()
    for bi, row in enumerate(rows_cpu):
        toks: list[str] = []
        tok_pos: list[int] = []
        eos_pos: int | None = None
        for ti, tid in enumerate(row):
            if tid == pad_id:
                continue
            if tid == eos_id:
                eos_pos = ti
                break
            if 0 <= tid < vlen:
                toks.append(vlist[tid])
                tok_pos.append(ti)

        if eos_pos is not None:
            kind[bi, eos_pos] = EOS_KIND_ID

        if not toks:
            continue

        try:
            evs = te.bekern_to_tuples(toks)
        except (ValueError, IndexError):
            # Best-effort prefix recovery if a sequence was truncated mid-note.
            evs = []
            for cut in range(len(toks) - 1, 0, -1):
                try:
                    evs = te.bekern_to_tuples(toks[:cut])
                    break
                except (ValueError, IndexError):
                    continue

        cursor = 0
        for ev in evs:
            ev_toks = te.tuples_to_bekern([ev])
            span = len(ev_toks)
            f = event_to_factors(ev)
            att_t = torch.tensor(f['attach'], dtype=torch.float32)
            for k_off in range(span):
                if cursor + k_off >= len(tok_pos):
                    break
                pos = tok_pos[cursor + k_off]
                kind[bi, pos] = f['kind']
                voice[bi, pos] = f['voice']
                dur[bi, pos] = f['dur']
                pitch[bi, pos] = f['pitch']
                alter[bi, pos] = f['alter']
                subtype[bi, pos] = f['subtype']
                attach[bi, pos] = att_t
            cursor += span

    return {
        'kind': kind.to(device=device),
        'voice': voice.to(device=device),
        'dur': dur.to(device=device),
        'pitch': pitch.to(device=device),
        'alter': alter.to(device=device),
        'subtype': subtype.to(device=device),
        'attach': attach.to(device=device),
    }


def build_vocab_factor_maps(
    vocab: list[str] | tuple[str, ...] | None = None,
    vocab_size: int | None = None,
    eos_id: int | None = data.EOS_ID,
) -> dict[str, torch.Tensor]:
    """Build ``[V, C]`` indicator matrices projecting flat token probs to factor marginals.

    Used by :func:`factored_token_consistency_loss` to enforce agreement between
    the flat bekern token logits and the factored event heads.
    """
    vlist = data.VOCAB if vocab is None else vocab
    vlen = max(len(vlist), int(vocab_size or 0), (int(eos_id) + 1) if eos_id is not None else 0)

    m_kind = torch.zeros(vlen, N_KINDS, dtype=torch.float32)
    m_voice = torch.zeros(vlen, N_VOICES, dtype=torch.float32)
    m_dur = torch.zeros(vlen, N_DURS, dtype=torch.float32)
    m_alter = torch.zeros(vlen, N_ALTERS, dtype=torch.float32)
    m_subtype = torch.zeros(vlen, N_SUBTYPES, dtype=torch.float32)

    for i, tok in enumerate(vlist):
        if tok.startswith('voice-'):
            v_idx = int(tok[6:])
            if 1 <= v_idx <= MAX_VOICE:
                m_voice[i, v_idx] = 1.0
            m_kind[i, KIND_TO_ID['VOICE']] = 1.0
        elif tok == 'bar':
            m_kind[i, KIND_TO_ID['BAR']] = 1.0
        elif tok in te.BAR_VARIANTS:
            m_kind[i, KIND_TO_ID['BAR']] = 1.0
            m_subtype[i, SUBTYPE_TO_ID[tok]] = 1.0
        elif tok == 'bar-rest':
            m_kind[i, KIND_TO_ID['BAR_REST']] = 1.0
        elif tok == 'rest':
            m_kind[i, KIND_TO_ID['REST']] = 1.0
        elif tok == 'chord':
            m_kind[i, KIND_TO_ID['CHORD_NOTE']] = 1.0
        elif tok == 'grace':
            m_kind[i, KIND_TO_ID['GRACE_NOTE']] = 1.0
        elif tok == 'tuplet-stop':
            m_kind[i, KIND_TO_ID['TUPLET_STOP']] = 1.0
        elif tok.startswith('tuplet-start-'):
            m_kind[i, KIND_TO_ID['TUPLET_START']] = 1.0
            if tok in SUBTYPE_TO_ID:
                m_subtype[i, SUBTYPE_TO_ID[tok]] = 1.0
        elif tok.startswith('clef-') or tok.startswith('key') or tok.startswith('time-'):
            m_kind[i, KIND_TO_ID['HEADER']] = 1.0
            if tok in SUBTYPE_TO_ID:
                m_subtype[i, SUBTYPE_TO_ID[tok]] = 1.0
        elif tok in te.DYNS:
            m_kind[i, KIND_TO_ID['DYN']] = 1.0
            m_subtype[i, SUBTYPE_TO_ID[tok]] = 1.0
        elif tok in te.HAIRPINS:
            m_kind[i, KIND_TO_ID['HAIRPIN']] = 1.0
            m_subtype[i, SUBTYPE_TO_ID[tok]] = 1.0
        elif tok in te.DURS:
            m_dur[i, DUR_TO_ID[tok]] = 1.0
        elif tok in te.ALTERS:
            m_alter[i, ALTER_TO_ID[tok]] = 1.0
        elif tok in te.ARTICS:
            m_subtype[i, SUBTYPE_TO_ID[tok]] = 1.0
        elif tok in ('tie-start', 'tie-stop', 'slur-start', 'slur-stop'):
            m_subtype[i, SUBTYPE_TO_ID[tok]] = 1.0

    if eos_id is not None and 0 <= eos_id < vlen:
        m_kind[eos_id, EOS_KIND_ID] = 1.0

    return {
        'kind': m_kind,
        'voice': m_voice,
        'dur': m_dur,
        'alter': m_alter,
        'subtype': m_subtype,
    }


def factored_token_consistency_loss(
    factored_outputs: dict[str, torch.Tensor],
    tok_logits: torch.Tensor,
    factor_maps: dict[str, torch.Tensor],
    pad_mask: torch.Tensor | None = None,
    detach_tok: bool = True,
    temperature: float = 1.0,
    min_mass: float = 1e-3,
    kind_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric JSD consistency loss between factored heads and flat token logits.

    Projects ``p_tok = softmax(tok_logits / T)`` onto each factor's vocabulary
    subset via ``factor_maps`` and aligns it to the corresponding factored head
    distribution ``p_fac = softmax(factored_outputs[f] / T)`` restricted to
    mapped classes, weighted by the token probability mass on that factor.

    For the ``subtype`` factor, positions are strictly masked to
    :data:`SUBTYPE_KINDS` (using ``kind_targets`` when provided, else the
    detached ``kind`` argmax) so JSD is never computed against the
    unsupervised ``subtype`` head on ``NOTE``/``REST`` tokens (whose flat
    modifiers such as ``stacc``/``tie-start`` share vocabulary strings with
    standalone subtype events).

    Gradients stop at ``tok_logits`` by default (``detach_tok=True``).
    """
    if temperature <= 0.0:
        raise ValueError(f'temperature must be > 0, got {temperature}')

    tl = tok_logits.detach() if detach_tok else tok_logits
    p_tok = F.softmax(tl.float() / float(temperature), dim=-1)  # [B, T, V]
    v_tok = p_tok.shape[-1]

    if pad_mask is not None:
        valid = (~pad_mask.to(device=p_tok.device, dtype=torch.bool)).float()
    else:
        valid = torch.ones(p_tok.shape[:-1], device=p_tok.device, dtype=torch.float32)

    if float(valid.sum()) == 0.0:
        return (tl.float().sum() + factored_outputs['kind'].float().sum()) * 0.0

    subtype_ids_t = torch.tensor(sorted(SUBTYPE_KINDS), device=p_tok.device, dtype=torch.long)
    if kind_targets is not None:
        subtype_valid = torch.isin(kind_targets.to(device=p_tok.device), subtype_ids_t).float()
    elif 'kind' in factored_outputs:
        subtype_valid = torch.isin(
            factored_outputs['kind'].detach().argmax(dim=-1), subtype_ids_t
        ).float()
    else:
        subtype_valid = torch.zeros_like(valid)

    eps = 1e-7
    per_factor_losses: list[torch.Tensor] = []

    for name, fmap in factor_maps.items():
        if name not in factored_outputs:
            continue
        m = fmap.to(device=p_tok.device, dtype=torch.float32)
        if m.shape[0] > v_tok:
            m = m[:v_tok]
        elif m.shape[0] < v_tok:
            pad_rows = torch.zeros(v_tok - m.shape[0], m.shape[1],
                                   device=p_tok.device, dtype=torch.float32)
            m = torch.cat([m, pad_rows], dim=0)

        active_cls = m.sum(dim=0) > 0  # [C]
        if int(active_cls.sum()) < 2:
            continue

        # Aggregate flat token probabilities into factor classes: [B, T, C_act]
        m_act = m[:, active_cls]
        q_raw = torch.matmul(p_tok, m_act)
        mass = q_raw.sum(dim=-1)  # [B, T]
        pos_mask = valid * subtype_valid if name == 'subtype' else valid
        w_pos = pos_mask * mass * (mass >= min_mass).float()
        w_sum = w_pos.sum()
        if float(w_sum.detach()) <= 0.0:
            continue

        q_tok = (q_raw / mass.unsqueeze(-1).clamp_min(eps)).clamp_min(eps)
        q_tok = q_tok / q_tok.sum(dim=-1, keepdim=True)

        fac_logits = factored_outputs[name][..., active_cls].float() / float(temperature)
        p_fac = F.softmax(fac_logits, dim=-1).clamp_min(eps)

        m_mix = 0.5 * (p_fac + q_tok)
        kl_pm = (p_fac * (p_fac.log() - m_mix.log())).sum(dim=-1)
        kl_qm = (q_tok * (q_tok.log() - m_mix.log())).sum(dim=-1)
        jsd = 0.5 * (kl_pm + kl_qm).clamp_min(0.0)
        per_factor_losses.append((jsd * w_pos).sum() / w_sum.clamp_min(eps))

    if not per_factor_losses:
        return (tl.float().sum() + factored_outputs['kind'].float().sum()) * 0.0
    return torch.stack(per_factor_losses).mean()


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
        self.subtype_emb = nn.Embedding(N_SUBTYPES, d_model)
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
        subtype: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = (
            self.kind_emb(kind)
            + self.voice_emb(voice)
            + self.dur_emb(dur)
            + self.pitch_emb(pitch)
            + self.alter_emb(alter)
            + self.attach_proj(attach.to(self.attach_proj.weight.dtype))
        )
        if subtype is not None:
            x = x + self.subtype_emb(subtype)
        return self.norm(x)


class FactoredEventHead(nn.Module):
    """Decoupled S18 event-tuple prediction heads (categorical + calibrated Bernoulli).

    Predicts seven independent factor heads from a ``[B, T, d_model]``
    hidden state:
      * ``kind``    ``[B, T, N_KINDS]``    (CE)
      * ``voice``   ``[B, T, N_VOICES]``   (CE, masked to voice-bearing events)
      * ``dur``     ``[B, T, N_DURS]``     (micro-glyph focal CE, timed events)
      * ``pitch``   ``[B, T, N_PITCHES]``  (ledger-line focal CE, note events)
      * ``alter``   ``[B, T, N_ALTERS]``   (micro-glyph focal CE, note events)
      * ``subtype`` ``[B, T, N_SUBTYPES]`` (CE, masked to non-note subtype events)
      * ``attach``  ``[B, T, N_ATTACH]``   (calibrated per-bit Bernoulli focal BCE
        + marginal-rate calibration penalty, masked to timed events)

    Because each head reads directly from ``h`` in parallel, an attachment
    error (e.g. a missed dot or staccato) never alters the sequence length
    or shifts downstream pitch/duration/voice predictions.
    """

    def __init__(
        self,
        d_model: int = 256,
        dropout: float = 0.0,
        attach_prior: float = 0.05,
        focal_gamma: float = 2.0,
        use_micro_glyph_weights: bool = True,
        attach_calib_weight: float = 0.25,
    ):
        super().__init__()
        if not (0.0 < attach_prior < 1.0):
            raise ValueError(f'attach_prior must be in (0, 1), got {attach_prior}')
        self.d_model = d_model
        self.focal_gamma = float(focal_gamma)
        self.use_micro_glyph_weights = bool(use_micro_glyph_weights)
        self.attach_calib_weight = float(attach_calib_weight)

        self.drop = nn.Dropout(dropout)
        self.kind_head = nn.Linear(d_model, N_KINDS)
        self.voice_head = nn.Linear(d_model, N_VOICES)
        self.dur_head = nn.Linear(d_model, N_DURS)
        self.pitch_head = nn.Linear(d_model, N_PITCHES)
        self.alter_head = nn.Linear(d_model, N_ALTERS)
        self.subtype_head = nn.Linear(d_model, N_SUBTYPES)
        self.attach_head = nn.Linear(d_model, N_ATTACH)

        # Prior-calibrated bias init for sparse Bernoulli attachments
        # (prevents 50%-prob initial spray on staccato/dots/ornaments).
        prior_logit = math.log(attach_prior / (1.0 - attach_prior))
        nn.init.constant_(self.attach_head.bias, prior_logit)

        # Learnable per-bit calibration temperature (exp(0) = 1.0 at init).
        self.attach_log_temp = nn.Parameter(torch.zeros(N_ATTACH))
        self.register_buffer('attach_thresholds', torch.full((N_ATTACH,), 0.5))

        # Batch 4-B3 inverse-footprint micro-glyph weight buffers.
        self.register_buffer(
            'dur_micro_weights',
            torch.tensor(MICRO_GLYPH_DUR_WEIGHTS, dtype=torch.float32),
        )
        self.register_buffer(
            'alter_micro_weights',
            torch.tensor(MICRO_GLYPH_ALTER_WEIGHTS, dtype=torch.float32),
        )
        self.register_buffer(
            'pitch_micro_weights',
            build_pitch_micro_glyph_weights(ledger_weight=2.0),
        )
        self.register_buffer(
            'attach_micro_weights',
            torch.tensor(MICRO_GLYPH_ATTACH_WEIGHTS, dtype=torch.float32),
        )

    def attach_temperature(self) -> torch.Tensor:
        """Per-bit positive calibration temperature clamped to ``[0.25, 4.0]``."""
        return self.attach_log_temp.exp().clamp(0.25, 4.0)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.drop(h)
        raw_attach = self.attach_head(h)
        temp = self.attach_temperature().to(dtype=raw_attach.dtype)
        return {
            'kind': self.kind_head(h),
            'voice': self.voice_head(h),
            'dur': self.dur_head(h),
            'pitch': self.pitch_head(h),
            'alter': self.alter_head(h),
            'subtype': self.subtype_head(h),
            'attach': raw_attach / temp,
        }

    @staticmethod
    @torch.no_grad()
    def calibrate_attach_thresholds(
        probs: torch.Tensor,
        targets: torch.Tensor,
        beta: float = 1.0,
        grid_steps: int = 39,
    ) -> torch.Tensor:
        """Select per-bit Bernoulli decision thresholds maximizing F_beta.

        Args:
            probs: ``[N, N_ATTACH]`` predicted attachment probabilities.
            targets: ``[N, N_ATTACH]`` binary ground-truth attachment targets.
            beta: F-beta weighting (``< 1`` favors precision to suppress spray).
            grid_steps: Number of candidate thresholds in ``[0.05, 0.95]``.

        Returns:
            ``[N_ATTACH]`` optimal threshold tensor.
        """
        probs_f = probs.float().reshape(-1, N_ATTACH)
        tgt_b = (targets.reshape(-1, N_ATTACH) > 0.5)
        cands = torch.linspace(0.05, 0.95, grid_steps, device=probs_f.device)
        b2 = float(beta) ** 2
        best_th = torch.full((N_ATTACH,), 0.5, device=probs_f.device)
        for j in range(N_ATTACH):
            tj = tgt_b[:, j]
            if not bool(tj.any()):
                best_th[j] = 0.5
                continue
            pj = probs_f[:, j]
            best_score = -1.0
            best_t = 0.5
            for th in cands:
                pred = pj >= th
                tp = float((pred & tj).sum())
                fp = float((pred & (~tj)).sum())
                fn = float(((~pred) & tj).sum())
                denom = (1.0 + b2) * tp + b2 * fn + fp
                score = ((1.0 + b2) * tp / denom) if denom > 0.0 else 0.0
                if score > best_score:
                    best_score = score
                    best_t = float(th)
            best_th[j] = best_t
        return best_th

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        weights: dict[str, float] | None = None,
        attach_pos_weight: float = 1.0,
        focal_gamma: float | None = None,
        use_micro_glyph_weights: bool | None = None,
        attach_calib_weight: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Masked multi-head factored loss with micro-glyph focal & calibration terms."""
        w = {
            'kind': 1.0,
            'voice': 1.0,
            'dur': 1.0,
            'pitch': 1.5,
            'alter': 1.0,
            'subtype': 1.0,
            'attach': 1.0,
            'attach_calib': (
                self.attach_calib_weight
                if attach_calib_weight is None else float(attach_calib_weight)
            ),
        }
        if weights:
            w.update(weights)

        gamma = self.focal_gamma if focal_gamma is None else float(focal_gamma)
        use_mg = (
            self.use_micro_glyph_weights
            if use_micro_glyph_weights is None else bool(use_micro_glyph_weights)
        )

        kind_t = targets['kind']
        valid = kind_t != PAD_KIND_ID

        def _masked_ce(
            logits: torch.Tensor,
            tgt: torch.Tensor,
            mask: torch.Tensor,
            cw: torch.Tensor | None = None,
            head_gamma: float = 0.0,
        ) -> torch.Tensor:
            if not bool(mask.any()):
                return logits.float().sum() * 0.0
            return micro_glyph_focal_ce(
                logits[mask], tgt[mask], class_weights=cw, gamma=head_gamma,
            )

        pitch_ids = torch.tensor(sorted(PITCH_KINDS), device=kind_t.device)
        timed_ids = torch.tensor(sorted(TIMED_KINDS), device=kind_t.device)
        voice_ids = torch.tensor(sorted(VOICE_KINDS), device=kind_t.device)
        subtype_ids = torch.tensor(sorted(SUBTYPE_KINDS), device=kind_t.device)

        is_pitch = valid & torch.isin(kind_t, pitch_ids)
        is_timed = valid & torch.isin(kind_t, timed_ids)
        is_voice = valid & torch.isin(kind_t, voice_ids)
        is_subtype = valid & torch.isin(kind_t, subtype_ids)

        subtype_t = targets.get('subtype')
        if subtype_t is None:
            subtype_t = torch.zeros_like(kind_t)

        losses = {
            'kind': _masked_ce(outputs['kind'], kind_t, valid),
            'voice': _masked_ce(outputs['voice'], targets['voice'], is_voice),
            'dur': _masked_ce(
                outputs['dur'], targets['dur'], is_timed,
                cw=(self.dur_micro_weights if use_mg else None),
                head_gamma=(gamma if use_mg else 0.0),
            ),
            'pitch': _masked_ce(
                outputs['pitch'], targets['pitch'], is_pitch,
                cw=(self.pitch_micro_weights if use_mg else None),
                head_gamma=(gamma if use_mg else 0.0),
            ),
            'alter': _masked_ce(
                outputs['alter'], targets['alter'], is_pitch,
                cw=(self.alter_micro_weights if use_mg else None),
                head_gamma=(gamma if use_mg else 0.0),
            ),
            'subtype': _masked_ce(outputs['subtype'], subtype_t, is_subtype),
        }

        if bool(is_timed.any()):
            att_logits = outputs['attach'][is_timed].float()
            att_tgt = targets['attach'][is_timed].float()
            pw = (
                torch.full((N_ATTACH,), float(attach_pos_weight),
                           device=kind_t.device, dtype=torch.float32)
                if attach_pos_weight != 1.0 else None
            )
            bce = F.binary_cross_entropy_with_logits(
                att_logits, att_tgt, pos_weight=pw, reduction='none',
            )
            probs = torch.sigmoid(att_logits)
            if use_mg and gamma > 0.0:
                pt = torch.where(att_tgt > 0.5, probs, 1.0 - probs).clamp(1e-6, 1.0)
                bce = ((1.0 - pt) ** gamma) * bce
            if use_mg:
                bce = bce * self.attach_micro_weights.to(
                    device=bce.device, dtype=bce.dtype
                )
            losses['attach'] = bce.mean()

            # Marginal-rate calibration penalty: aligns predicted per-bit
            # attachment rate to empirical ground-truth rate to kill ~2x spray.
            pred_rate = probs.mean(dim=0)
            tgt_rate = att_tgt.mean(dim=0)
            losses['attach_calib'] = F.mse_loss(pred_rate, tgt_rate)
        else:
            zero = outputs['attach'].float().sum() * 0.0
            losses['attach'] = zero
            losses['attach_calib'] = zero

        total = sum(w[k] * losses[k] for k in losses)
        return total, losses

    @torch.no_grad()
    def decode_greedy(
        self,
        outputs: dict[str, torch.Tensor],
        attach_thresh: float | torch.Tensor | None = None,
    ) -> list[list[dict]]:
        """Decode ``[B, T, ...]`` logits into per-batch lists of event dicts."""
        kind_ids = outputs['kind'].argmax(dim=-1).tolist()
        voice_ids = outputs['voice'].argmax(dim=-1).tolist()
        dur_ids = outputs['dur'].argmax(dim=-1).tolist()
        pitch_ids = outputs['pitch'].argmax(dim=-1).tolist()
        alter_ids = outputs['alter'].argmax(dim=-1).tolist()
        subtype_ids = outputs['subtype'].argmax(dim=-1).tolist()

        probs = torch.sigmoid(outputs['attach'])
        if attach_thresh is None:
            th = self.attach_thresholds.to(device=probs.device, dtype=probs.dtype)
        elif isinstance(attach_thresh, torch.Tensor):
            th = attach_thresh.to(device=probs.device, dtype=probs.dtype)
        else:
            th = float(attach_thresh)
        attach_bits = (probs >= th).tolist()

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
                    subtype_id=subtype_ids[b][t],
                )
                if ev is not None:
                    evs.append(ev)
            batch_events.append(evs)
        return batch_events


class S18FactoredHead(nn.Module):
    """Plug-and-play S18 factored head + consistency + micro-glyph focal wrapper.

    Designed for ~10-line integration into ``model.py`` / ``train.py``::

        self.s18_head = S18FactoredHead(d_model=d_model, vocab_size=vocab)
        # In training step:
        s18_loss, s18_metrics = self.s18_head.loss_from_tokens(
            h, tgt, tok_logits=logits, pad_id=self.pad_id, eos_id=self.eos_id
        )
        loss = ce_loss + s18_weight * s18_loss
    """

    def __init__(
        self,
        d_model: int = 256,
        vocab_size: int | None = None,
        dropout: float = 0.0,
        attach_prior: float = 0.05,
        focal_gamma: float = 2.0,
        use_micro_glyph_weights: bool = True,
        attach_calib_weight: float = 0.25,
        consistency_weight: float = 0.0,
        consistency_temp: float = 1.0,
        eos_id: int = data.EOS_ID,
    ):
        super().__init__()
        self.head = FactoredEventHead(
            d_model=d_model,
            dropout=dropout,
            attach_prior=attach_prior,
            focal_gamma=focal_gamma,
            use_micro_glyph_weights=use_micro_glyph_weights,
            attach_calib_weight=attach_calib_weight,
        )
        self.consistency_weight = float(consistency_weight)
        self.consistency_temp = float(consistency_temp)

        fmaps = build_vocab_factor_maps(vocab_size=vocab_size, eos_id=eos_id)
        for name, mat in fmaps.items():
            self.register_buffer(f'_fmap_{name}', mat, persistent=False)
        self._fmap_names = tuple(fmaps.keys())

    def _factor_maps(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, f'_fmap_{name}') for name in self._fmap_names}

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(h)

    def loss(
        self,
        h: torch.Tensor,
        targets: dict[str, torch.Tensor],
        tok_logits: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
        weights: dict[str, float] | None = None,
        attach_pos_weight: float = 1.0,
        detach_tok: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute factored + consistency loss and return ``(total_loss, metrics_dict)``."""
        outputs = self.head(h)
        total, parts = self.head.compute_loss(
            outputs, targets, weights=weights,
            attach_pos_weight=attach_pos_weight,
        )
        if tok_logits is not None and self.consistency_weight > 0.0:
            if pad_mask is None:
                pad_mask = targets['kind'] == PAD_KIND_ID
            cons = factored_token_consistency_loss(
                outputs, tok_logits, self._factor_maps(),
                pad_mask=pad_mask,
                detach_tok=detach_tok,
                temperature=self.consistency_temp,
                kind_targets=targets['kind'],
            )
            parts['consistency'] = cons
            total = total + self.consistency_weight * cons
        else:
            parts['consistency'] = total.detach() * 0.0

        metrics = {f's18_{k}': float(v.detach()) for k, v in parts.items()}
        metrics['s18_total'] = float(total.detach())
        return total, metrics

    def loss_from_tokens(
        self,
        h: torch.Tensor,
        tgt_ids: torch.Tensor,
        tok_logits: torch.Tensor | None = None,
        pad_id: int = data.PAD_ID,
        eos_id: int = data.EOS_ID,
        weights: dict[str, float] | None = None,
        attach_pos_weight: float = 1.0,
        detach_tok: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Align raw bekern token IDs ``[B, T]`` and compute ``(total_loss, metrics_dict)``."""
        targets = align_token_ids_to_factors(tgt_ids, pad_id=pad_id, eos_id=eos_id)
        pad_mask = tgt_ids == pad_id
        return self.loss(
            h, targets, tok_logits=tok_logits, pad_mask=pad_mask,
            weights=weights, attach_pos_weight=attach_pos_weight,
            detach_tok=detach_tok,
        )

    @torch.no_grad()
    def decode_greedy(
        self,
        h: torch.Tensor,
        attach_thresh: float | torch.Tensor | None = None,
    ) -> list[list[dict]]:
        return self.head.decode_greedy(self.head(h), attach_thresh=attach_thresh)


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
