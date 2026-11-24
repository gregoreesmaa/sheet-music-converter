"""Barline-anchored measure decoding + S>4 staff windowing (T2.3).

Two entry points (see ``todo/phase-2-hybrid-dewarp.md`` Task T2.3):

* :func:`decode_system_with_measure_fallback` — runs the net's
  grammar-masked beam search (``strict=True``, ``max_len=564``), then
  checks every emitted measure with a per-measure closure check that
  mirrors ``assembly-serialize`` ``decode_bekern`` (strict
  :class:`grammar.PrefixState` walk of header + measure must accept and
  end at an event boundary). A measure that fails strict closure is
  trimmed to its longest non-strict-clean prefix (trailing partial
  event dropped); a valid ``bar-rest`` fallback is spliced only when
  the non-strict walk also fails, so downstream measures stay
  bar-aligned — one bad measure can no longer shift ``onset_in_bar``
  for the rest of the system.
* :func:`decode_multi_staff_system` — decodes a system with any staff
  count. ``<= 4`` staves rectify once (via :mod:`rectify`) and decode;
  ``> 4`` staves are chunked ``[0:4]``, ``[4:8]``, ... rectified and
  decoded independently, then headers are concatenated and per-measure
  voice runs merged (chunk voices renumbered ``voice-(k+1)..`` at each
  ``bar`` token, where ``k`` is the voice count already present).

Seam note: the closure check is a pure-Python mirror of
``decode_bekern``'s per-measure requirements (complete events only,
strict duration conservation). Shelling out to the Rust binary per
measure would cost a process spawn per measure; the mirror keeps unit
tests hermetic and is pinned by ``test_grammar.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import grammar  # noqa: E402
import rectify  # noqa: E402
import render_diff_verify  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

__all__ = [
    'MAX_VOICE',
    'ScoreContext',
    'propagate_across_systems',
    'stitch_multipage_work',
    'split_measures',
    'join_measures',
    'fallback_repair',
    'measure_closes',
    'mem_dims_for_strip',
    'slot_mem_mask',
    'mask_for_strip',
    'window_staves',
    'merge_chunk_streams',
    'decode_system_with_measure_fallback',
    'decode_multi_staff_system',
]

MAX_VOICE = max(
    int(t.split('-')[1]) for t in data.VOCAB if t.startswith('voice-')
)


def mem_dims_for_strip(height, width, stride='8x16'):
    """Encoder memory grid for a ``[1, C, height, width]`` strip.

    Height is exact at ``height // 8``; width is
    ``((width // 4) // 2) // 2`` (the same conv arithmetic as
    ``train._mem_width_aniso``). Exact for every width, not just
    multiples of 16 — verified against the live encoder for widths
    64..1000 including odd values (e.g. 101, 1000) by
    ``test_mem_mask``. Lets the eval path size the slot mask without
    an extra encoder probe forward. ``stride='4x8'`` (S13) halves
    both axes: ``height // 4`` by ``(width // 4) // 2``. Default
    keeps the exact legacy values.
    """
    if stride == '4x8':
        return height // 4, (width // 4) // 2
    if stride != '8x16':
        raise ValueError('stride must be 8x16 or 4x8, got %r' % stride)
    return height // 8, ((width // 4) // 2) // 2


def slot_mem_mask(n_staves, h_mem, w_mem, n_slots=4, device='cpu'):
    """Bool ``[h_mem * w_mem]`` eval memory mask for an S-staff strip.

    Mirrors the training collates: rectified strips always canvas
    ``n_slots`` (4) staff slots, so memory rows at/after slot
    ``n_staves`` — ``y_mem >= (h_mem // n_slots) * n_staves`` — are
    inactive white paper (row-major layout ``y * w_mem + x``, matching
    :meth:`model.MvdNet.encode`). True = ignore (``nn.MultiheadAttention``
    key-padding convention). Single eval strips carry no width padding,
    so only slot rows are masked; an ``n_staves == n_slots`` strip
    yields all-False (legacy statistic, bitwise identical).
    """
    cut = (h_mem // n_slots) * n_staves
    rows = torch.arange(h_mem, device=device) >= cut
    return rows.unsqueeze(1).expand(h_mem, w_mem).reshape(-1)


def mask_for_strip(tensor, n_staves, stride='8x16'):
    """``slot_mem_mask`` sized from a ``[1, C, H, W]`` strip tensor."""
    _, _, h, w = tensor.shape
    h_mem, w_mem = mem_dims_for_strip(h, w, stride=stride)
    return slot_mem_mask(n_staves, h_mem, w_mem,
                         device=tensor.device).unsqueeze(0)


def split_measures(toks):
    """Split a bekern stream into (header, measures, trailing_bar).

    ``header`` is the leading run of clef/key/time tokens; ``measures``
    are the ``bar``-separated bodies (without the ``bar`` tokens). A
    stream ending in ``bar`` yields ``trailing_bar=True`` and no empty
    tail measure; interior empty measures (consecutive bars) are kept.
    """
    toks = list(toks)
    i = 0
    while i < len(toks) and (toks[i] in grammar.CLEFS
                             or toks[i] in grammar.KEYS
                             or toks[i] in grammar.TIMES):
        i += 1
    header = toks[:i]
    body = toks[i:]
    measures = []
    cur = []
    for t in body:
        if t == grammar.BAR:
            measures.append(cur)
            cur = []
        else:
            cur.append(t)
    trailing_bar = bool(body) and body[-1] == grammar.BAR
    if cur or not trailing_bar:
        measures.append(cur)
    return header, measures, trailing_bar


def join_measures(header, measures, trailing_bar=True):
    """Reassemble ``split_measures`` output, ``bar``-terminating runs."""
    out = list(header)
    for j, m in enumerate(measures):
        out.extend(m)
        if j < len(measures) - 1 or trailing_bar:
            out.append(grammar.BAR)
    return out


def measure_closes(header, measure):
    """True iff header + measure walks strict-clean to an event boundary.

    Mirrors ``decode_bekern``'s per-measure requirements: every token
    must extend a valid strict prefix (this enforces T2.3
    bar-duration conservation) and the measure must end on a complete
    event (``PrefixState.allowed`` EOS legality). Empty measures close.
    A sentinel ``bar`` is walked after the measure so a still-open
    event is forced to close (and checked) exactly as the real
    bar-terminated stream would.
    """
    st = grammar.PrefixState(strict=True)
    try:
        for t in header:
            st.step(t)
        for t in measure:
            st.step(t)
        st.step(grammar.BAR)
    except ValueError:
        return False
    _, eos_ok = st.allowed()
    return bool(eos_ok)


def _measure_closes_loose(header, measure):
    """Non-strict twin of :func:`measure_closes` (legacy GT-safe mask).

    Same bar-terminated walk and event-boundary requirement, but the
    prefix state runs ``strict=False`` so the validator-predicate
    rules (duration conservation, sequential voices, no duplicate
    markers) do not fail the walk. Tells a strict-only violation
    (keep the rollout) from genuinely unparseable output (splice
    ``bar-rest``).
    """
    st = grammar.PrefixState(strict=False)
    try:
        for t in header:
            st.step(t)
        for t in measure:
            st.step(t)
        st.step(grammar.BAR)
    except ValueError:
        return False
    _, eos_ok = st.allowed()
    return bool(eos_ok)


def _trim_trailing_partial(header, measure):
    """Longest leading prefix of ``measure`` that loose-closes.

    A strict failure is often just a truncated tail (early EOS left a
    mid-note stub): the longest non-strict-clean prefix keeps every
    complete event and drops only the unparseable tail. Returns the
    kept token list, or None when even the empty measure will not
    loose-close (i.e. the header itself is broken).
    """
    m = list(measure)
    for k in range(len(m), -1, -1):
        if _measure_closes_loose(header, m[:k]):
            return m[:k]
    return None


def _repair_measure(header, measure):
    """Non-destructive per-measure repair (T2.4 pipefix).

    A strict-clean measure passes through untouched. Otherwise the
    trailing incomplete event is trimmed and the remainder kept when
    it loose-closes; ``['bar-rest']`` is spliced only when the
    non-strict walk also fails, so a strict-only violation never wipes
    a full rollout down to header + bar-rest.
    """
    if measure_closes(header, measure):
        return list(measure)
    trimmed = _trim_trailing_partial(header, measure)
    if trimmed is not None:
        return trimmed
    return ['bar-rest']


def fallback_repair(toks):
    """Split + per-measure repair with splice statistics (spike S6).

    Returns ``(fixed_toks, n_touched, n_spliced)``: ``n_touched``
    counts altered measures, ``n_spliced`` the subset replaced by a
    ``bar-rest`` splice. A strict-clean stream returns identical
    tokens with ``(0, 0)``. Pure function.
    """
    header, measures, trailing_bar = split_measures(toks)
    fixed = []
    touched = spliced = 0
    for m in measures:
        r = _repair_measure(header, m)
        if r == ['bar-rest'] and m != ['bar-rest']:
            spliced += 1
            touched += 1
        elif r != m:
            touched += 1
        fixed.append(r)
    return join_measures(header, fixed, trailing_bar), touched, spliced


def _ids_to_tokens(out):
    """First beam row of ids -> bekern tokens (cut at specials)."""
    row = out[0]
    ids = row.tolist() if hasattr(row, 'tolist') else list(row)
    toks = []
    for i in ids:
        i = int(i)
        if 0 <= i < len(data.VOCAB):
            toks.append(data.VOCAB[i])
        else:
            break
    return toks


def _obs_gray_for_verify(img_tensor, obs_gray=None):
    """Extract a 2D uint8 grayscale strip for T4.1 visual reranking."""
    if obs_gray is not None:
        if isinstance(obs_gray, Image.Image):
            return np.asarray(obs_gray.convert('L'), dtype=np.uint8)
        arr = np.asarray(obs_gray)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=2)
        if arr.dtype != np.uint8:
            if arr.size > 0 and float(arr.max()) <= 1.0 and float(arr.min()) >= 0.0:
                arr = arr * 255.0
            arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
        return arr
    # Fallback: derive grayscale directly from [1, C, H, W] input tensor RGB channels.
    rgb = img_tensor[0, :3].detach().cpu().float().numpy()  # [3, H, W]
    gray = rgb.mean(axis=0)
    lo, hi = float(gray.min()), float(gray.max())
    if hi - lo > 1e-6:
        gray = (gray - lo) / (hi - lo) * 255.0
    else:
        gray = np.full_like(gray, 255.0)
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def decode_system_with_measure_fallback(net, img_tensor, beam=3,
                                        mem_mask=None,
                                        ink_eos_veto=False,
                                        ilme_lambda=0.0,
                                        lattice_repair=False,
                                        lattice_beams=8,
                                        adapt_steps=0,
                                        adapt_lr=0.1,
                                        verify_beam=False,
                                        verify_alpha=2.0,
                                        obs_gray=None,
                                        max_len=564):
    """Beam-decode one system strip with per-measure bar-rest fallback.

    Runs ``net.beam_search(img_tensor, beam=beam, strict=True,
    max_len=max_len)``; every measure is repaired by :func:`_repair_measure`
    (strict-clean passes through, trailing partial events are trimmed
    under a non-strict re-check, ``['bar-rest']`` only when the
    non-strict walk also fails) so surviving measures keep their bar
    alignment. Returns the repaired token list.
    ``mem_mask`` is an optional bool ``[1, S]`` memory key mask
    (True = ignore) forwarded to ``net.beam_search``. It is passed as
    a keyword only when given, so pre-mask stand-ins with the legacy
    ``(images, beam, strict, max_len)`` shape keep working unmasked.
    ``ink_eos_veto`` (spike S1): forbid EOS while HeatNet notehead
    peaks outnumber decoded steps; passed only when True so legacy
    stand-ins keep working. ``ilme_lambda`` (spike S2): dual-pass ILM
    debias weight; passed only when nonzero. ``lattice_repair``
    (spike S6): decode ``lattice_beams`` beams and ship the
    constraint-ranked winner instead of the top beam.
    ``adapt_steps`` (spike S9): test-time per-strip adapter steps
    (0 = off); passed only when positive so legacy stand-ins keep
    working.
    ``verify_beam`` (T4.1): when True and ``beam >= 2``, requests all
    top-K beams + length-normalized logprobs from ``net.beam_search(...,
    return_beams=True)`` and reranks them via
    :func:`render_diff_verify.rerank_beams_visual` with weight
    ``verify_alpha``. Default False preserves legacy behavior bit-for-bit.
    """
    bs_kwargs = {}
    if mem_mask is not None:
        bs_kwargs['mem_mask'] = mem_mask
    if adapt_steps:
        bs_kwargs['adapt_steps'] = adapt_steps
        bs_kwargs['adapt_lr'] = adapt_lr
    # Dropped spikes S1/S2/S6 are accepted as compat kwargs for eval_sys_gate.py.
    _ = (ink_eos_veto, ilme_lambda, lattice_beams)
    if lattice_repair:
        raise ValueError('dropped spike S6 (lattice_repair) was removed; '
                         'pass lattice_repair=False')
    if verify_beam and beam >= 2:
        all_beams, all_scores = net.beam_search(
            img_tensor, beam=beam, strict=True, max_len=int(max_len),
            return_beams=True, **bs_kwargs)
        beams_row = all_beams[0]
        scores_row = all_scores[0]
        cand_toks = []
        cand_lps = []
        for b_ids, lp in zip(beams_row, scores_row):
            toks = _ids_to_tokens([b_ids])
            fixed, _, _ = fallback_repair(toks)
            cand_toks.append(fixed)
            cand_lps.append(float(lp))
        if len(cand_toks) >= 2:
            gray = _obs_gray_for_verify(img_tensor, obs_gray=obs_gray)
            res = render_diff_verify.rerank_beams_visual(
                gray, cand_toks, cand_lps, alpha=float(verify_alpha))
            return list(res['best_toks'])
        return cand_toks[0] if cand_toks else []
    out = net.beam_search(img_tensor, beam=beam, strict=True,
                          max_len=int(max_len), **bs_kwargs)
    toks = _ids_to_tokens(out)
    fixed, _, _ = fallback_repair(toks)
    return fixed


def _tensor_from_pil(img):
    """PIL image -> float32 ``[1, 3, H, W]`` tensor in [0, 1]."""
    if not isinstance(img, Image.Image):
        raise TypeError('expected PIL Image, got %r' % type(img))
    rgb = img if img.mode == 'RGB' else img.convert('RGB')
    a = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).contiguous()


def window_staves(system_staves_lines, max_staves=rectify.MAX_STAVES):
    """Partition an S-staff system into non-overlapping <=max_staves chunks.

    Returns ``[(start_idx, end_idx, chunk_list), ...]``. For 5-7 staff
    systems (``max_staves=4``), produces ``[0:4]`` + ``[4:S]`` so every
    staff is decoded exactly once with a train-matched slot mask.
    """
    staves = list(system_staves_lines)
    if not staves:
        raise ValueError('need at least 1 staff, got 0')
    if max_staves < 1:
        raise ValueError('max_staves must be >= 1, got %r' % max_staves)
    return [
        (i, min(i + max_staves, len(staves)), staves[i:i + max_staves])
        for i in range(0, len(staves), max_staves)
    ]


def _decode_chunk(net, page_img, chunk_staves, beam, extra_fn=None,
                  mem_mask=None, ink_eos_veto=False, ilme_lambda=0.0,
                  lattice_repair=False, adapt_steps=0, adapt_lr=0.1,
                  stride='8x16', target_sp=rectify.REF_TARGET_SP,
                  height=None, verify_beam=False, verify_alpha=2.0,
                  max_len=564):
    """Rectify one <=4-staff chunk and fallback-decode it.

    ``extra_fn`` (T2.4 hybrid hook): ``(rect_img, staff_ref) ->
    [1, C, H, W]`` model input tensor replacing the plain 3-channel
    ``_tensor_from_pil`` path (e.g. rectified RGB + HeatNet 7-ch +
    staff grid). ``None`` preserves the legacy plain path exactly.
    ``mem_mask`` overrides the chunk slot mask; None auto-builds it
    from ``len(chunk_staves)`` via :func:`mask_for_strip` (train-matched
    inactive-slot masking, no encoder probe needed).
    ``target_sp`` and ``height`` forward to :func:`rectify.rectify_system_strip`
    for opt-in higher-res micro-glyph rectification (default 10.0 / 256).
    """
    rect_img, ref = rectify.rectify_system_strip(
        page_img, chunk_staves, target_sp=target_sp, height=height)
    if extra_fn is None:
        tensor = _tensor_from_pil(rect_img)
    else:
        tensor = extra_fn(rect_img, ref)
    if mem_mask is None:
        mem_mask = mask_for_strip(tensor, len(chunk_staves),
                                  stride=stride)
    return decode_system_with_measure_fallback(
        net, tensor, beam=beam, mem_mask=mem_mask,
        ink_eos_veto=ink_eos_veto, ilme_lambda=ilme_lambda,
        lattice_repair=lattice_repair, adapt_steps=adapt_steps,
        adapt_lr=adapt_lr, verify_beam=verify_beam,
        verify_alpha=verify_alpha,
        obs_gray=(rect_img if verify_beam else None),
        max_len=max_len)


def _explicit_voice(measure):
    """Make a bare (marker-less) voice run explicit as voice-1.

    Merging needs every chunk measure's voice count; a bare run is
    implicitly voice-1, and spelling it out keeps the merged stream
    sequential (a later chunk's ``voice-2`` requires ``voice-1``
    already opened since the bar).
    """
    m = list(measure)
    if m and not any(t.startswith('voice-') for t in m):
        return ['voice-1'] + m
    return m


def _measure_voices(measure):
    """Max voice index used in a measure (1 for a bare run, 0 if empty)."""
    best = 0
    for t in measure:
        if t.startswith('voice-'):
            best = max(best, int(t.split('-')[1]))
    if best == 0 and len(measure) > 0:
        return 1
    return best


def _shift_voices(measure, k, max_voice=MAX_VOICE):
    """Renumber a chunk's measure voices by +k, dropping any voice > max_voice.

    Caps shifted voices at ``max_voice`` (``data.VOCAB`` stops at
    ``voice-12``) so mega-systems (>12 staves) preserve the first 12
    voices as valid bekern instead of emitting OOV ``voice-13+`` tokens
    that trigger a full-system ``hyp_tokens_valid`` wipeout. Pass
    ``max_voice=None`` to disable capping.
    """
    m = list(measure)
    if not m:
        return m
    if max_voice is not None and k >= max_voice:
        return []
    has_voice = any(t.startswith('voice-') for t in m)
    if not has_voice:
        if k <= 0:
            return m
        v = k + 1
        if max_voice is not None and v > max_voice:
            return []
        return ['voice-%d' % v] + m
    out = []
    dropping = False
    for t in m:
        if t.startswith('voice-'):
            v = int(t.split('-')[1]) + max(0, k)
            if max_voice is not None and v > max_voice:
                dropping = True
            else:
                dropping = False
                out.append('voice-%d' % v)
        elif not dropping:
            out.append(t)
    return out


def merge_chunk_streams(chunk_streams, chunk_staff_counts=None,
                        max_voice=MAX_VOICE):
    """Merge multi-staff chunk token streams at the measure/voice level.

    When ``chunk_staff_counts`` is given (e.g. ``[4, 3]`` for a 7-staff
    system), chunk ``ci > 0`` shifts each measure's voices by
    ``max(_measure_voices(merged[j]), prev_staves_cum)`` so per-staff
    voice IDs stay aligned even when earlier chunks contain empty or
    collapsed bar-rest staves (e.g. ``aloha-oe``).

    Chunks whose cumulative preceding staff count already reaches
    ``max_voice`` (12) are dropped cleanly, and individual voices that
    would shift past ``max_voice`` are omitted, guaranteeing 100%
    in-vocab ``voice-1..voice-12`` output.
    """
    if not chunk_streams:
        return []
    if len(chunk_streams) == 1:
        return list(chunk_streams[0])
    staff_counts = (
        list(chunk_staff_counts)
        if chunk_staff_counts is not None
        else [0] * len(chunk_streams)
    )
    if len(staff_counts) != len(chunk_streams):
        raise ValueError(
            'chunk_staff_counts length %d != chunk_streams length %d'
            % (len(staff_counts), len(chunk_streams)))
    merged_header = []
    merged = []
    prev_staves_cum = 0
    for ci, (toks, n_staves) in enumerate(zip(chunk_streams, staff_counts)):
        if max_voice is not None and ci > 0 and prev_staves_cum >= max_voice:
            break
        header, measures, _tb = split_measures(toks)
        merged_header.extend(header)
        while len(merged) < len(measures):
            merged.append([])
        for j, m in enumerate(measures):
            m = _explicit_voice(m)
            if ci == 0:
                merged[j].extend(_shift_voices(m, 0, max_voice=max_voice))
            else:
                shift = max(_measure_voices(merged[j]), prev_staves_cum)
                merged[j].extend(
                    _shift_voices(m, shift, max_voice=max_voice))
        prev_staves_cum += max(0, int(n_staves))
    return join_measures(merged_header, merged, True)


def decode_multi_staff_system(net, page_img, system_staves_lines,
                              beam=3, extra_fn=None, mem_mask=None,
                              ink_eos_veto=False, ilme_lambda=0.0,
                              lattice_repair=False, adapt_steps=0,
                              adapt_lr=0.1, stride='8x16',
                              target_sp=rectify.REF_TARGET_SP,
                              height=None, verify_beam=False,
                              verify_alpha=2.0, max_len=564):
    """Decode a system of any staff count with measure-level merging.

    ``<= 4`` staves: rectify once, decode once. ``> 4`` staves: slice
    into chunks of up to 4 (``[0:4]``, ``[4:8]``, ...), rectify and
    fallback-decode each chunk, then merge headers and per-measure
    voice runs via :func:`merge_chunk_streams` (preserving per-staff
    voice offsets across bar-rest chunks and capping voices at
    :data:`MAX_VOICE`). ``extra_fn`` forwards to :func:`_decode_chunk`
    (T2.4 hybrid inputs). Returns tokens.
    ``mem_mask`` applies to the single-chunk path only (forwarded to
    :func:`_decode_chunk`); chunked decodes build a train-matched slot
    mask per chunk inside :func:`_decode_chunk` (one mask cannot span
    chunk tensors), so passing ``mem_mask`` with ``> 4`` staves raises
    ``ValueError``.
    """
    windows = window_staves(system_staves_lines,
                            max_staves=rectify.MAX_STAVES)
    chunks = [chunk for _, _, chunk in windows]
    if len(chunks) == 1:
        return _decode_chunk(net, page_img, chunks[0], beam,
                             extra_fn=extra_fn, mem_mask=mem_mask,
                             ink_eos_veto=ink_eos_veto,
                             ilme_lambda=ilme_lambda,
                             lattice_repair=lattice_repair,
                             adapt_steps=adapt_steps,
                             adapt_lr=adapt_lr, stride=stride,
                             target_sp=target_sp, height=height,
                             verify_beam=verify_beam,
                             verify_alpha=verify_alpha,
                             max_len=max_len)
    if mem_mask is not None:
        raise ValueError('mem_mask cannot span chunked (> 4 staff) '
                         'decodes; pass None so each chunk builds its '
                         'own slot mask')
    chunk_streams = []
    chunk_staff_counts = []
    prev_staves_cum = 0
    for chunk in chunks:
        if prev_staves_cum >= MAX_VOICE:
            break
        toks = _decode_chunk(net, page_img, chunk, beam,
                             extra_fn=extra_fn,
                             ink_eos_veto=ink_eos_veto,
                             ilme_lambda=ilme_lambda,
                             lattice_repair=lattice_repair,
                             adapt_steps=adapt_steps,
                             adapt_lr=adapt_lr, stride=stride,
                             target_sp=target_sp, height=height,
                             verify_beam=verify_beam,
                             verify_alpha=verify_alpha,
                             max_len=max_len)
        chunk_streams.append(toks)
        chunk_staff_counts.append(len(chunk))
        prev_staves_cum += len(chunk)
    return merge_chunk_streams(
        chunk_streams, chunk_staff_counts=chunk_staff_counts,
        max_voice=MAX_VOICE)


# ---------------------------------------------------------------------------
# Multi-system & multi-page musical state carryforward (ScoreContext).
# ---------------------------------------------------------------------------

_ALTER_SUFFIX = {
    'a-2': 'bb',
    'a-1': 'b',
    'a1': '#',
    'a2': 'x',
}

_NOTE_TAIL_NON_MOD = (
    grammar.DURATIONS | grammar.STEPS | grammar.OCTAVES
    | grammar.CLEFS | grammar.KEYS | grammar.TIMES | grammar.VOICES
    | grammar.DYNS | grammar.HAIRS | grammar.BAR_LEFT | grammar.BAR_LIKES
    | grammar.TUPLET_STARTS
    | frozenset((grammar.REST, grammar.BAR_REST, grammar.CHORD,
                 grammar.GRACE, grammar.TUPLET_STOP))
)


def _is_header_tok(tok: str) -> bool:
    return (tok.startswith('clef-')
            or tok.startswith('key')
            or tok.startswith('time-'))


def _split_head_body_toks(toks: Sequence[str]) -> tuple[list[str], list[str]]:
    toks_list = list(toks)
    i = 0
    while i < len(toks_list) and _is_header_tok(toks_list[i]):
        i += 1
    return toks_list[:i], toks_list[i:]


def _as_tokens(stream: str | Sequence[str]) -> list[str]:
    if isinstance(stream, str):
        return stream.split()
    return list(stream)


def _parse_header_slots(head_toks: Sequence[str]) -> tuple[list[str], list[str], list[str]]:
    clefs: list[str] = []
    keys: list[str] = []
    times: list[str] = []
    for t in head_toks:
        if t.startswith('clef-'):
            clefs.append(t)
        elif t.startswith('key'):
            keys.append(t)
        elif t.startswith('time-'):
            times.append(t)
    return clefs, keys, times


@dataclass
class ScoreContext:
    """Running musical state carried across systems and page boundaries.

    Tracks per-staff headers (0-based staff index -> bekern token) and
    cross-boundary musical spans so multi-system pages and multi-page
    works preserve clefs, key/time signatures, measure numbering, voice
    counts, pickups, ties, slurs, hairpins, and repeat/volta state.

    ``open_ties`` is keyed by ``(staff, voice, pitch)`` where ``staff``
    is 0-based, ``voice`` is 1-based, and ``pitch`` is canonical (e.g.
    ``'C4'``, ``'F#4'``, ``'Bb3'``).
    """

    clef: dict[int, str] = field(default_factory=dict)
    key_sig: dict[int, str] = field(default_factory=dict)
    time_sig: dict[int, str] = field(default_factory=dict)
    active_voices: int = 0
    measure_index: int = 0
    in_pickup: bool = False
    open_ties: dict[tuple[int, int, str], int] = field(default_factory=dict)
    open_slurs: dict[tuple[int, int], int] = field(default_factory=dict)
    open_hairpins: dict[int, str] = field(default_factory=dict)
    open_volta: str | None = None
    emitted_header: bool = False

    def clone(self) -> ScoreContext:
        """Return an independent deep copy of this context."""
        return ScoreContext(
            clef=dict(self.clef),
            key_sig=dict(self.key_sig),
            time_sig=dict(self.time_sig),
            active_voices=self.active_voices,
            measure_index=self.measure_index,
            in_pickup=self.in_pickup,
            open_ties=dict(self.open_ties),
            open_slurs=dict(self.open_slurs),
            open_hairpins=dict(self.open_hairpins),
            open_volta=self.open_volta,
            emitted_header=self.emitted_header,
        )

    @property
    def staff_count(self) -> int:
        """Number of staves currently tracked by header slots."""
        best = 0
        for d in (self.clef, self.key_sig, self.time_sig):
            if d:
                best = max(best, max(d.keys()) + 1)
        return best

    def staff_for_voice(self, voice: int) -> int:
        """Map a 1-based voice index to a 0-based staff index."""
        n = self.staff_count
        if n <= 1:
            return 0
        return min(max(0, voice - 1), n - 1)

    def header_tokens(self) -> list[str]:
        """Emit canonical per-staff leading header tokens from current state."""
        n = self.staff_count
        if n == 0:
            return []
        out: list[str] = []
        for s in range(n):
            if s in self.clef:
                out.append(self.clef[s])
            if s in self.key_sig:
                out.append(self.key_sig[s])
            if s in self.time_sig:
                out.append(self.time_sig[s])
        return out


def _active_barlen(ctx: ScoreContext) -> float | None:
    ts = ctx.time_sig.get(0)
    if ts is None and ctx.time_sig:
        ts = ctx.time_sig[min(ctx.time_sig.keys())]
    if ts is None:
        return None
    return grammar._beats_of(ts)


def _init_slots_from_header(
    ctx: ScoreContext,
    clefs: list[str],
    keys: list[str],
    times: list[str],
) -> None:
    """Populate initial per-staff header maps (incoming overrides defaults)."""
    for idx, tok in enumerate(clefs):
        ctx.clef[idx] = tok
    if len(keys) == 1 and len(ctx.key_sig) > 1:
        for s in list(ctx.key_sig.keys()):
            ctx.key_sig[s] = keys[0]
    else:
        for idx, tok in enumerate(keys):
            ctx.key_sig[idx] = tok
    if len(times) == 1 and len(ctx.time_sig) > 1:
        for s in list(ctx.time_sig.keys()):
            ctx.time_sig[s] = times[0]
    else:
        for idx, tok in enumerate(times):
            ctx.time_sig[idx] = tok


def _diff_staff_map(
    active: dict[int, str],
    incoming: list[str],
    *,
    broadcast_single: bool = False,
) -> list[str]:
    """Diff incoming per-staff header tokens against ``active``, updating in place.

    Omitted courtesy headers (``incoming == []``) preserve ``active``
    untouched and emit ``[]``. Genuine changes update ``active`` and
    return deduplicated changed tokens in order. When
    ``broadcast_single=True`` (used for shared key/time signatures) and
    a single incoming token arrives on a multi-staff score where all
    staves previously shared one value, all tracked staff slots update
    together so subsequent full-staff courtesy headers stay silent.
    """
    if not incoming:
        return []
    changed: list[str] = []
    if (broadcast_single and len(incoming) == 1 and len(active) > 1
            and len(set(active.values())) == 1):
        tok = incoming[0]
        prev = next(iter(active.values()))
        if tok != prev:
            for s in list(active.keys()):
                active[s] = tok
            changed.append(tok)
        return changed
    for idx, tok in enumerate(incoming):
        if idx in active:
            if active[idx] != tok:
                active[idx] = tok
                if tok not in changed:
                    changed.append(tok)
        else:
            prev_vals = [active[k] for k in sorted(active.keys())]
            prev = prev_vals[-1] if prev_vals else None
            active[idx] = tok
            if prev is None:
                if tok not in changed:
                    changed.append(tok)
            elif tok != prev and tok not in changed:
                changed.append(tok)
    return changed


def _match_open_tie(
    open_ties: dict[tuple[int, int, str], int],
    staff: int,
    voice: int,
    pitch: str,
) -> tuple[int, int, str] | None:
    exact = (staff, voice, pitch)
    if open_ties.get(exact, 0) > 0:
        return exact
    for k, cnt in open_ties.items():
        if cnt > 0 and k[1] == voice and k[2] == pitch:
            return k
    pitch_matches = [k for k, cnt in open_ties.items() if cnt > 0 and k[2] == pitch]
    if len(pitch_matches) == 1:
        return pitch_matches[0]
    return None


def _pop_open_tie(
    open_ties: dict[tuple[int, int, str], int],
    key: tuple[int, int, str],
) -> None:
    cnt = open_ties.get(key, 0)
    if cnt <= 1:
        open_ties.pop(key, None)
    else:
        open_ties[key] = cnt - 1


def _is_tail_mod_tok(tok: str) -> bool:
    if tok in _NOTE_TAIL_NON_MOD:
        return False
    return not (
        tok.startswith('clef-')
        or tok.startswith('key')
        or tok.startswith('time-')
        or tok.startswith('voice-')
        or tok.startswith('dyn-')
        or tok.startswith('hairpin-')
        or tok.startswith('tuplet-')
        or tok.startswith('bar')
        or tok.startswith('volta-')
    )


def _process_system_body(
    body: list[str],
    ctx: ScoreContext,
) -> list[str]:
    """Walk one system body, updating ``ctx`` and resolving cross-system ties."""
    out: list[str] = []
    carried_ties = bool(ctx.open_ties)
    cur_voice = 1
    cur_measure_in_sys = 0
    voice_cursor: dict[int, float] = {1: 0.0}
    voice_last_onset: dict[int, float] = {}
    measure_has_content = False
    pending_chord = False
    pending_grace = False

    i = 0
    n = len(body)
    while i < n:
        t = body[i]

        # Closing barlines advance measure_index and reset per-measure cursors.
        if t == grammar.BAR or t in grammar.BAR_RIGHT:
            out.append(t)
            if t == 'bar-repeat-right':
                ctx.open_volta = None
            ctx.measure_index += 1
            ctx.in_pickup = False
            cur_measure_in_sys += 1
            cur_voice = 1
            voice_cursor = {1: 0.0}
            voice_last_onset = {}
            measure_has_content = False
            pending_chord = False
            pending_grace = False
            i += 1
            continue

        if t == 'bar-repeat-left':
            ctx.open_volta = 'repeat-left'
            out.append(t)
            i += 1
            continue

        if t.startswith('volta-'):
            ctx.open_volta = None if t.endswith('-stop') else t
            out.append(t)
            i += 1
            continue

        if t.startswith('voice-'):
            try:
                v = int(t.split('-')[1])
            except (IndexError, ValueError):
                v = cur_voice
            if v > 0:
                cur_voice = v
                ctx.active_voices = max(ctx.active_voices, v)
                voice_cursor.setdefault(v, 0.0)
            out.append(t)
            i += 1
            continue

        if t.startswith('clef-'):
            s_idx = ctx.staff_for_voice(cur_voice)
            ctx.clef[s_idx] = t
            out.append(t)
            i += 1
            continue

        if t.startswith('key'):
            if ctx.key_sig:
                for s in list(ctx.key_sig.keys()):
                    ctx.key_sig[s] = t
            else:
                ctx.key_sig[0] = t
            out.append(t)
            i += 1
            continue

        if t.startswith('time-'):
            if ctx.time_sig:
                for s in list(ctx.time_sig.keys()):
                    ctx.time_sig[s] = t
            else:
                ctx.time_sig[0] = t
            out.append(t)
            i += 1
            continue

        if t.startswith('hairpin-'):
            if t == 'hairpin-cresc-start':
                ctx.open_hairpins[cur_voice] = 'cresc'
            elif t == 'hairpin-dim-start':
                ctx.open_hairpins[cur_voice] = 'dim'
            elif t == 'hairpin-stop':
                ctx.open_hairpins.pop(cur_voice, None)
            out.append(t)
            i += 1
            continue

        if t == grammar.CHORD:
            pending_chord = True
            out.append(t)
            i += 1
            continue

        if t == grammar.GRACE:
            pending_grace = True
            out.append(t)
            i += 1
            continue

        if t == grammar.BAR_REST:
            measure_has_content = True
            ctx.active_voices = max(ctx.active_voices, cur_voice)
            barlen = _active_barlen(ctx)
            if barlen is not None:
                voice_cursor[cur_voice] = max(
                    voice_cursor.get(cur_voice, 0.0), barlen)
            pending_chord = False
            pending_grace = False
            out.append(t)
            i += 1
            continue

        # Plain rest: rest <dur> [dot*] [mods*]
        if t == grammar.REST and i + 1 < n and body[i + 1] in grammar.DURATIONS:
            measure_has_content = True
            ctx.active_voices = max(ctx.active_voices, cur_voice)
            staff = ctx.staff_for_voice(cur_voice)
            dur_tok = body[i + 1]
            out.append(t)
            out.append(dur_tok)
            i += 2
            dots = 0
            while i < n and _is_tail_mod_tok(body[i]):
                tt = body[i]
                if tt == grammar.DOT:
                    dots += 1
                elif tt == 'slur-start':
                    ctx.open_slurs[(staff, cur_voice)] = (
                        ctx.open_slurs.get((staff, cur_voice), 0) + 1)
                elif tt == 'slur-stop':
                    d = ctx.open_slurs.get((staff, cur_voice), 0)
                    if d <= 1:
                        ctx.open_slurs.pop((staff, cur_voice), None)
                    else:
                        ctx.open_slurs[(staff, cur_voice)] = d - 1
                out.append(tt)
                i += 1
            if not pending_grace:
                base = grammar._WRITTEN.get(dur_tok, 0.0)
                val = base * (2.0 - 2.0 ** (-dots))
                voice_cursor[cur_voice] = voice_cursor.get(cur_voice, 0.0) + val
            pending_chord = False
            pending_grace = False
            continue

        # Pitched note: <dur> <step> <octave> [tail...]
        if (t in grammar.DURATIONS and i + 2 < n
                and body[i + 1] in grammar.STEPS
                and body[i + 2] in grammar.OCTAVES):
            measure_has_content = True
            ctx.active_voices = max(ctx.active_voices, cur_voice)
            staff = ctx.staff_for_voice(cur_voice)
            dur_tok, step_tok, oct_tok = t, body[i + 1], body[i + 2]
            i += 3
            tail: list[str] = []
            alter_tok = None
            dots = 0
            has_tie_start = False
            has_tie_stop = False
            while i < n and _is_tail_mod_tok(body[i]):
                tt = body[i]
                tail.append(tt)
                if tt in _ALTER_SUFFIX and alter_tok is None:
                    alter_tok = tt
                elif tt == grammar.DOT:
                    dots += 1
                elif tt == grammar.TIE_START:
                    has_tie_start = True
                elif tt == grammar.TIE_STOP:
                    has_tie_stop = True
                elif tt == 'slur-start':
                    ctx.open_slurs[(staff, cur_voice)] = (
                        ctx.open_slurs.get((staff, cur_voice), 0) + 1)
                elif tt == 'slur-stop':
                    d = ctx.open_slurs.get((staff, cur_voice), 0)
                    if d <= 1:
                        ctx.open_slurs.pop((staff, cur_voice), None)
                    else:
                        ctx.open_slurs[(staff, cur_voice)] = d - 1
                i += 1

            pitch = '%s%s%s' % (
                step_tok,
                _ALTER_SUFFIX.get(alter_tok, '') if alter_tok else '',
                oct_tok,
            )
            note_onset = (
                voice_last_onset.get(cur_voice, 0.0)
                if pending_chord
                else voice_cursor.get(cur_voice, 0.0)
            )
            at_sys_start_onset = (
                cur_measure_in_sys == 0 and note_onset == 0.0
            )
            matched_tie = _match_open_tie(ctx.open_ties, staff, cur_voice, pitch)
            if has_tie_stop:
                if matched_tie is not None:
                    _pop_open_tie(ctx.open_ties, matched_tie)
            elif carried_ties and at_sys_start_onset and matched_tie is not None:
                # Cross-system / cross-page tie continuation where the new
                # system omitted tie-stop: inject tie-stop so the carried
                # tie closes cleanly rather than stranding an unclosed tie.
                tail.append(grammar.TIE_STOP)
                _pop_open_tie(ctx.open_ties, matched_tie)

            if has_tie_start:
                tkey = (staff, cur_voice, pitch)
                ctx.open_ties[tkey] = ctx.open_ties.get(tkey, 0) + 1

            out.append(dur_tok)
            out.append(step_tok)
            out.append(oct_tok)
            out.extend(tail)

            if not pending_chord:
                voice_last_onset[cur_voice] = note_onset
                if not pending_grace:
                    base = grammar._WRITTEN.get(dur_tok, 0.0)
                    val = base * (2.0 - 2.0 ** (-dots))
                    voice_cursor[cur_voice] = note_onset + val
            pending_chord = False
            pending_grace = False
            continue

        out.append(t)
        i += 1

    if measure_has_content:
        ctx.measure_index += 1
        barlen = _active_barlen(ctx)
        max_dur = max(voice_cursor.values(), default=0.0)
        ctx.in_pickup = bool(
            barlen is not None and 0.0 < max_dur < barlen - grammar._BAR_EPS)

    return out


def propagate_across_systems(
    system_bekerns: Sequence[str | Sequence[str]],
    init_ctx: ScoreContext | None = None,
    *,
    emit_header: bool | None = None,
) -> tuple[str, ScoreContext]:
    """Stitch system bekern streams while carrying musical state forward.

    Parameters
    ----------
    system_bekerns:
        Ordered sequence of system bekern streams (each a whitespace-separated
        ``str`` or a sequence of token strings).
    init_ctx:
        Optional :class:`ScoreContext` carried from a preceding page or
        section. Never mutated in place; a fresh updated copy is returned.
    emit_header:
        If ``None`` (default), emits the opening header on the first
        non-empty system only when ``init_ctx`` has not already emitted one
        (``not ctx.emitted_header``). Pass ``True`` to force emitting the
        resolved leading header even when continuing from a prior page.

    Returns
    -------
    ``(stitched_bekern, updated_ctx)``
        Space-joined bekern string and updated :class:`ScoreContext`.
    """
    ctx = init_ctx.clone() if init_ctx is not None else ScoreContext()
    should_emit_initial = (
        (not ctx.emitted_header) if emit_header is None else bool(emit_header)
    )
    out_heads: list[str] = []
    out_body: list[str] = []
    had_nonempty_body = False

    for raw in system_bekerns:
        toks = _as_tokens(raw)
        h, b = _split_head_body_toks(toks)
        if not b:
            continue
        clefs, keys, times = _parse_header_slots(h)
        if not had_nonempty_body and should_emit_initial:
            had_nonempty_body = True
            if ctx.staff_count == 0:
                # Fresh start: preserve the exact token ordering of the
                # first non-empty system's leading header.
                _init_slots_from_header(ctx, clefs, keys, times)
                out_heads = list(h)
            else:
                # Seeded init_ctx before page 1 (or forced standalone header):
                # merge incoming header overrides onto carried defaults.
                _init_slots_from_header(ctx, clefs, keys, times)
                out_heads = ctx.header_tokens()
            ctx.emitted_header = True
        else:
            had_nonempty_body = True
            changed: list[str] = []
            changed.extend(_diff_staff_map(ctx.clef, clefs, broadcast_single=False))
            changed.extend(_diff_staff_map(ctx.key_sig, keys, broadcast_single=True))
            changed.extend(_diff_staff_map(ctx.time_sig, times, broadcast_single=True))
            out_body.extend(changed)
        processed_b = _process_system_body(b, ctx)
        out_body.extend(processed_b)

    if not had_nonempty_body and system_bekerns:
        first_h, _ = _split_head_body_toks(_as_tokens(system_bekerns[0]))
        if first_h and should_emit_initial:
            clefs, keys, times = _parse_header_slots(first_h)
            _init_slots_from_header(ctx, clefs, keys, times)
            out_heads = list(first_h)
            ctx.emitted_header = True

    stitched = out_heads + out_body
    return ' '.join(stitched), ctx


def stitch_multipage_work(
    page_system_lists: Sequence[Sequence[str | Sequence[str]]],
    init_ctx: ScoreContext | None = None,
) -> tuple[str, ScoreContext]:
    """Stitch a multi-page score (pages 1..K) sequentially with ``ScoreContext``.

    Each element of ``page_system_lists`` is the ordered list of system
    bekern streams for one page. Musical state (per-staff clefs, key/time
    signatures, open ties/slurs/hairpins/voltas, measure index, active
    voices) flows from page ``p`` into page ``p + 1``.
    """
    ctx = init_ctx.clone() if init_ctx is not None else ScoreContext()
    page_chunks: list[str] = []
    for page_systems in page_system_lists:
        chunk, ctx = propagate_across_systems(page_systems, init_ctx=ctx)
        if chunk:
            page_chunks.append(chunk)
    return ' '.join(page_chunks), ctx
