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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import grammar  # noqa: E402
import rectify  # noqa: E402
import render_diff_verify  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

__all__ = [
    'split_measures',
    'join_measures',
    'fallback_repair',
    'measure_closes',
    'mem_dims_for_strip',
    'slot_mem_mask',
    'mask_for_strip',
    'window_staves',
    'decode_system_with_measure_fallback',
    'decode_multi_staff_system',
]


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
                                        obs_gray=None):
    """Beam-decode one system strip with per-measure bar-rest fallback.

    Runs ``net.beam_search(img_tensor, beam=beam, strict=True,
    max_len=564)``; every measure is repaired by :func:`_repair_measure`
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
            img_tensor, beam=beam, strict=True, max_len=564,
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
                          max_len=564, **bs_kwargs)
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
                  height=None, verify_beam=False, verify_alpha=2.0):
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
        obs_gray=(rect_img if verify_beam else None))


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


def _shift_voices(measure, k):
    """Renumber a chunk's measure voices by +k (``voice-1..``)."""
    m = list(measure)
    if k <= 0:
        return m
    if not any(t.startswith('voice-') for t in m):
        return (['voice-%d' % (k + 1)] + m) if m else m
    out = []
    for t in m:
        if t.startswith('voice-'):
            out.append('voice-%d' % (int(t.split('-')[1]) + k))
        else:
            out.append(t)
    return out


def decode_multi_staff_system(net, page_img, system_staves_lines,
                              beam=3, extra_fn=None, mem_mask=None,
                              ink_eos_veto=False, ilme_lambda=0.0,
                              lattice_repair=False, adapt_steps=0,
                              adapt_lr=0.1, stride='8x16',
                              target_sp=rectify.REF_TARGET_SP,
                              height=None, verify_beam=False,
                              verify_alpha=2.0):
    """Decode a system of any staff count with measure-level merging.

    ``<= 4`` staves: rectify once, decode once. ``> 4`` staves: slice
    into chunks of up to 4 (``[0:4]``, ``[4:8]``, ...), rectify and
    fallback-decode each chunk, then concatenate headers and merge
    per-measure voice runs (chunk voices shift to
    ``voice-(k+1)..`` where ``k`` is the voice count already merged
    for that measure) at each ``bar`` token. ``extra_fn`` forwards
    to :func:`_decode_chunk` (T2.4 hybrid inputs). Returns tokens.
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
                             verify_alpha=verify_alpha)
    if mem_mask is not None:
        raise ValueError('mem_mask cannot span chunked (> 4 staff) '
                         'decodes; pass None so each chunk builds its '
                         'own slot mask')
    merged_header = []
    merged = []
    for ci, chunk in enumerate(chunks):
        toks = _decode_chunk(net, page_img, chunk, beam,
                             extra_fn=extra_fn,
                             ink_eos_veto=ink_eos_veto,
                             ilme_lambda=ilme_lambda,
                             lattice_repair=lattice_repair,
                             adapt_steps=adapt_steps,
                             adapt_lr=adapt_lr, stride=stride,
                             target_sp=target_sp, height=height,
                             verify_beam=verify_beam,
                             verify_alpha=verify_alpha)
        header, measures, _tb = split_measures(toks)
        merged_header.extend(header)
        while len(merged) < len(measures):
            merged.append([])
        for j, m in enumerate(measures):
            m = _explicit_voice(m)
            if ci == 0:
                merged[j].extend(m)
            else:
                merged[j].extend(
                    _shift_voices(m, _measure_voices(merged[j])))
    return join_measures(merged_header, merged, True)
