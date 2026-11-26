#!/usr/bin/env python3
"""Apples-to-apples Suite31+HW full-page evaluator for UniOMR.

Includes:
  - FSA grammar-constrained greedy decoding (``generate_constrained``)
    reusing ``grammar.PrefixState`` with clean ``<sys>`` state resets.
  - MTP-4 self-speculative decoding (``generate_mtp_spec``) using
    ``UniOMRNet.mtp_heads`` with exact token-for-token parity against
    standard greedy decoding (and optional grammar constraints).
  - Full ``suite31_hw_manifest.json`` (45 pieces: 31 printed + 14 HW)
    evaluation scoring BOTH Token F1/SER and real Note-Level F1/SER
    via ``assembly-serialize.exe`` + ``eval-harness.exe``.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe tmp-train/mvd/eval_uni_suite31.py \\
        --ckpt tmp-train/mvd-runs/uni-omr/uni_omr_best.pt \\
        --use-grammar --use-mtp-spec --device cuda \\
        --out tmp-train/mvd-runs/uni-omr/suite31_hw_eval.json
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, fields
import difflib
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from PIL import Image
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import data  # noqa: E402
import eval_sys_gate as esg  # noqa: E402
import grammar  # noqa: E402
import preprocess_dewarp  # noqa: E402
import uni_omr  # noqa: E402
import uni_omr_v2  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"


# ---------------------------------------------------------------------------
# Helpers: Backward-Compat Vocab IDs, Grammar Masking, Truncation & KV Slicing
# ---------------------------------------------------------------------------

def _vocab_spec(
    vocab_size: int,
) -> Tuple[int, int, int, int, Dict[str, int], Dict[int, str]]:
    """Derive special IDs and token maps for any append-only UniOMR ``vocab_size``.

    Keeps legacy 182-vocab checkpoints (178 base + PAD/BOS/EOS/SYS) 100%
    evaluable alongside expanded vocabs (e.g. Option D1: 233 base + 4 = 237).
    """
    n_base = int(vocab_size) - 4
    if n_base < 0 or n_base > len(data.VOCAB):
        raise ValueError(
            f"Unsupported vocab_size={vocab_size} (expected 4..{len(data.VOCAB) + 4})"
        )
    pad_id = vocab_size - 4
    bos_id = vocab_size - 3
    eos_id = vocab_size - 2
    sys_id = vocab_size - 1
    base_vocab = data.VOCAB[:n_base]
    tok_to_id = {t: i for i, t in enumerate(base_vocab)}
    id_to_tok = {i: t for i, t in enumerate(base_vocab)}
    id_to_tok[pad_id] = "<pad>"
    id_to_tok[bos_id] = "<bos>"
    id_to_tok[eos_id] = "<eos>"
    id_to_tok[sys_id] = uni_omr.SYS_TOKEN
    return pad_id, bos_id, eos_id, sys_id, tok_to_id, id_to_tok


def decode_uni_compat(
    ids: Sequence[int],
    vocab_size: Optional[int] = None,
    skip_special: bool = True,
) -> List[str]:
    """Decode UniOMR token IDs using ``vocab_size``'s special-ID layout."""
    vsz = int(vocab_size) if vocab_size is not None else uni_omr.VOCAB_SIZE
    pad_id, bos_id, eos_id, _, _, id_to_tok = _vocab_spec(vsz)
    out: List[str] = []
    for idx in ids:
        i = int(idx)
        if skip_special and i in (pad_id, bos_id, eos_id):
            if i == eos_id:
                break
            continue
        tok = id_to_tok.get(i)
        if tok is not None:
            out.append(tok)
    return out


def trim_truncated_tokens(toks: Sequence[str]) -> List[str]:
    """Trim trailing incomplete note/chord/rest/voice tokens on max_len truncation.

    Walks ``toks`` with ``grammar.PrefixState(strict=False)`` (resetting on
    ``<sys>``) and cuts back to the last complete bar or completed event
    boundary so ``bekern-decode`` does not crash on partial trailing tokens
    (e.g. ``quarter C`` -> ``missing step in measure 0``).
    """
    st = grammar.PrefixState(strict=False)
    last_complete = 0
    for i, tok in enumerate(toks):
        if tok == uni_omr.SYS_TOKEN:
            st = grammar.PrefixState(strict=False)
            continue
        try:
            st.step(tok)
        except ValueError:
            break
        _, eos_ok = st.allowed()
        if not eos_ok:
            continue
        if (
            tok in grammar.BAR_LIKES
            or tok == grammar.BAR_REST
            or tok == grammar.TIE_STOP
            or tok == grammar.TUPLET_STOP
            or st.mode in ("pitch-tail", "tiestart", "rest-tail")
        ):
            last_complete = i + 1
    if last_complete > 0:
        return list(toks[:last_complete])
    return []


def _allowed_ids_for_state(state: grammar.PrefixState, vocab_size: int) -> List[int]:
    """Return legal token IDs for ``state`` in the UniOMR vocabulary.

    Allows ``sys_id`` and ``eos_id`` whenever ``PrefixState`` permits
    end-of-system (``eos_ok`` or immediately after a barline). To avoid
    degenerate empty systems, ``sys_id`` is only offered once at least
    one token has been emitted in the current system.
    """
    pad_id, bos_id, eos_id, sys_id, tok_to_id, _ = _vocab_spec(vocab_size)
    if state.broken:
        # Fail open over all non-PAD/non-BOS tokens.
        return [
            i for i in range(vocab_size)
            if i not in (pad_id, bos_id)
        ]

    allow_set, eos_ok = state.allowed()
    allow_ids = {i for i in allow_set if 0 <= i < pad_id}

    # Suppress tight-loop cycle extension in BOTH strict=False and strict=True
    # modes whenever alternatives exist.
    cyc_tok = state.cycle_forbidden()
    if cyc_tok is not None:
        cyc_id = tok_to_id.get(cyc_tok)
        if cyc_id is not None and len(allow_ids) > 1:
            allow_ids.discard(cyc_id)

    after_bar = bool(state.recent) and (state.recent[-1] in grammar.BAR_LIKES)
    can_end_sys = bool(eos_ok or after_bar)
    if can_end_sys:
        allow_ids.add(eos_id)
        if state.can_emit_sys():
            allow_ids.add(sys_id)

    # Never emit PAD or BOS as generated tokens.
    allow_ids.discard(pad_id)
    allow_ids.discard(bos_id)
    return sorted(i for i in allow_ids if 0 <= i < vocab_size)


def _select_constrained_token(
    step_logits_1d: torch.Tensor,
    state: grammar.PrefixState,
) -> int:
    """Pick argmax token under ``state``'s grammar mask; safe fallback if empty/NaN."""
    vocab_size = int(step_logits_1d.shape[-1])
    allowed = _allowed_ids_for_state(state, vocab_size)
    if not allowed:
        return int(step_logits_1d.argmax(dim=-1).item())

    idx_t = torch.tensor(allowed, dtype=torch.long, device=step_logits_1d.device)
    sub_logits = step_logits_1d.index_select(0, idx_t)
    if not bool(torch.isfinite(sub_logits).any()):
        return int(step_logits_1d.argmax(dim=-1).item())

    best_pos = int(sub_logits.argmax(dim=-1).item())
    return int(allowed[best_pos])


def _advance_grammar_state(
    state: grammar.PrefixState,
    tok_id: int,
    strict: bool,
    vocab_size: Optional[int] = None,
) -> grammar.PrefixState:
    """Advance ``PrefixState`` by ``tok_id``; reset cleanly on ``sys_id``."""
    vsz = int(vocab_size) if vocab_size is not None else (len(state.vocab_map) + 4)
    pad_id, bos_id, eos_id, sys_id, tok_to_id, id_to_tok = _vocab_spec(vsz)
    if tok_id in (pad_id, bos_id, eos_id):
        return state
    if tok_id == sys_id:
        nxt = state.copy()
        nxt.reset_system()
        return nxt
    tok_str = id_to_tok.get(tok_id)
    if tok_str is None:
        state.broken = True
        return state
    try:
        state.step(tok_str)
    except ValueError:
        state.broken = True
    return state


def _truncate_past_kvs(
    past_kvs: List[Any],
    keep_len: int,
) -> List[Any]:
    """Slice decoder self-attention KV cache to ``keep_len`` (cross-attn untouched)."""
    trimmed: List[Any] = []
    for self_kv, cross_kv in past_kvs:
        k, v = self_kv
        trimmed.append(
            (
                (k[:, :, :keep_len, :].contiguous(), v[:, :, :keep_len, :].contiguous()),
                cross_kv,
            )
        )
    return trimmed


def compute_token_lcs_f1(ref_toks: Sequence[str], hyp_toks: Sequence[str]) -> float:
    """Ordered token LCS F1 (``2 * matches / (|ref| + |hyp|)``) via SequenceMatcher.

    Unlike bag-of-tokens F1, ordered LCS F1 penalizes token loops and
    mis-ordered repetitions directly.
    """
    if not ref_toks and not hyp_toks:
        return 1.0
    if not ref_toks or not hyp_toks:
        return 0.0
    sm = difflib.SequenceMatcher(a=list(ref_toks), b=list(hyp_toks), autojunk=False)
    matches = sum(block.size for block in sm.get_matching_blocks())
    return 2.0 * float(matches) / float(len(ref_toks) + len(hyp_toks))


def _measure_violation_errors(state: grammar.PrefixState) -> Tuple[float, float]:
    """Return ``(meter_err, voice_err)`` for the measure about to close in ``state``.

    * ``meter_err``: max ``|vcurs[v] - tsig|`` over active voices (0.0 if no tsig).
    * ``voice_err``: max ``|vcurs[v] - vcurs[active[0]]|`` over active voices
      when multiple voices are present (0.0 for single-voice bars).
    """
    eff_vcurs = list(state.vcurs)
    if 0 <= state.seg < len(eff_vcurs):
        eff_vcurs[state.seg] += state._pending_value()
    active = [
        i for i, v in enumerate(eff_vcurs)
        if v > grammar._BAR_EPS or (i + 1) in state.opened
    ]
    if not active:
        active = [0]

    meter_err = 0.0
    if state.tsig is not None:
        meter_err = max(abs(eff_vcurs[v] - state.tsig) for v in active)

    voice_err = 0.0
    if len(active) > 1:
        ref_v = eff_vcurs[active[0]]
        voice_err = max(abs(eff_vcurs[v] - ref_v) for v in active[1:])

    return float(meter_err), float(voice_err)


def _measure_signed_offset(state: grammar.PrefixState) -> Optional[float]:
    """Return signed ``vcur - tsig`` for the primary active voice, or None if no tsig."""
    if state.tsig is None:
        return None
    eff_vcurs = list(state.vcurs)
    if 0 <= state.seg < len(eff_vcurs):
        eff_vcurs[state.seg] += state._pending_value()
    active = [
        i for i, v in enumerate(eff_vcurs)
        if v > grammar._BAR_EPS or (i + 1) in state.opened
    ]
    if not active:
        active = [0]
    return round(float(eff_vcurs[active[0]] - state.tsig), 4)


class MeasureBeamHeaderGuard:
    """Abort meter-only measure-beam repair when a wrong ``time-X-Y`` header is detected.

    If ``> const_offset_limit`` (default 3, i.e. ``>= 4``) bars trigger with the exact
    same non-zero signed meter offset while all active voices agree (``voice_err <= tol``),
    the piece almost certainly has a misread time-signature header (e.g. ``time-3-2``
    on a ``4/2`` piece) rather than local intra-bar rhythm errors. Once tripped, meter-only
    repairs are disabled for the current time-signature regime while genuine cross-voice
    mismatches (``voice_err > tol``) remain eligible for repair.
    """

    def __init__(
        self,
        const_offset_limit: int = 3,
        tol: float = 1e-3,
        max_triggers: Optional[int] = None,
    ) -> None:
        self.const_offset_limit = int(const_offset_limit)
        self.tol = float(tol)
        self.max_triggers: Optional[int] = (
            int(max_triggers) if max_triggers is not None else None
        )
        self.trigger_count: int = 0
        self.last_tsig: Optional[float] = None
        self.offset_counts: Dict[float, int] = {}
        self.tripped: bool = False

    def should_repair(
        self,
        state: grammar.PrefixState,
        meter_err: float,
        voice_err: float,
    ) -> bool:
        if self.max_triggers is not None and self.trigger_count >= self.max_triggers:
            return False
        if meter_err <= self.tol and voice_err <= self.tol:
            return False
        if state.tsig != self.last_tsig:
            self.last_tsig = state.tsig
            self.offset_counts.clear()
            self.tripped = False
        # Cross-voice mismatch is independent of the time-signature header.
        if voice_err > self.tol:
            self.trigger_count += 1
            return True
        if self.tripped:
            return False
        signed_off = _measure_signed_offset(state)
        if signed_off is not None and abs(signed_off) > self.tol:
            key = round(signed_off, 3)
            cnt = self.offset_counts.get(key, 0) + 1
            self.offset_counts[key] = cnt
            if cnt > self.const_offset_limit:
                self.tripped = True
                return False
        self.trigger_count += 1
        return True


@torch.no_grad()
def _beam_repair_measure(
    model: uni_omr.UniOMRNet,
    kv_prefix: List[Any],
    base_cross_kvs: List[Tuple[torch.Tensor, torch.Tensor]],
    trigger_id: int,
    bar_start_state: grammar.PrefixState,
    beam_size: int,
    max_bar_toks: int,
    vocab_size: int,
    use_grammar: bool,
    strict_grammar: bool,
    device: torch.device,
    max_lp_drop: Optional[float] = 2.0,
    meter_w: float = 6.0,
    voice_w: float = 4.0,
) -> Tuple[Optional[List[int]], Optional[grammar.PrefixState], List[Any], float]:
    """Run a ``beam_size``-beam search over one measure starting after ``trigger_id``.

    Scores each closed measure by ``log_prob - meter_w * meter_err - voice_w * voice_err``
    (default ``6.0`` / ``4.0`` from promoted ``ov-34b`` ``mbeam2_cap4_lp2_w64``), with an
    optional log-prob drop guardrail ``max_lp_drop`` (default ``2.0``) that rejects repairs
    sacrificing more than ``max_lp_drop`` raw log-prob relative to the best-likelihood
    closed bar in the measure.
    Returns ``(best_toks, best_state, kv0, best_score)`` where ``kv0`` covers
    ``kv_prefix + [trigger_id]`` (so caller can fast-prefill ``best_toks[:-1]``).
    If no beam closes a bar within ``max_bar_toks``, returns ``(None, None, kv0, -inf)``.
    """
    _, _, eos_id, sys_id, _, id_to_tok = _vocab_spec(vocab_size)
    trig_t = torch.tensor([[trigger_id]], dtype=torch.long, device=device)
    logits0, _, kv0 = model.decode_tokens(
        trig_t,
        memory=None,
        return_mtp=False,
        past_kvs=kv_prefix,
        use_cache=True,
    )
    if max_bar_toks <= 0:
        return None, None, kv0, float("-inf")

    # Active beams: (bar_toks, raw_log_prob, grammar_state)
    active_beams: List[Tuple[List[int], float, grammar.PrefixState]] = [
        ([], 0.0, bar_start_state.copy())
    ]
    # Finished bars: (score, raw_log_prob, bar_toks, end_state)
    finished_bars: List[Tuple[float, float, List[int], grammar.PrefixState]] = []

    step_logits = logits0[:, -1, :]  # (1, V)
    past_kvs: List[Any] = kv0
    k_per_beam = min(vocab_size, beam_size + 1)

    for step_idx in range(max_bar_toks):
        if step_idx > 0:
            cur_ids = torch.tensor(
                [[b_toks[-1]] for b_toks, _, _ in active_beams],
                dtype=torch.long,
                device=device,
            )
            logits, _, past_kvs = model.decode_tokens(
                cur_ids,
                memory=None,
                return_mtp=False,
                past_kvs=past_kvs,
                use_cache=True,
            )
            step_logits = logits[:, -1, :]  # (n_active, V)

        step_lp = F.log_softmax(step_logits.float(), dim=-1)  # (n_active, V)
        candidates: List[Tuple[float, int, int]] = []  # (raw_lp, parent_idx, tok_id)

        for i, (_, beam_lp, beam_state) in enumerate(active_beams):
            row_logits = step_logits[i]
            row_lp = step_lp[i]
            used_mask = False
            if use_grammar:
                allowed = [
                    tid for tid in _allowed_ids_for_state(beam_state, vocab_size)
                    if tid != eos_id and tid != sys_id
                ]
                if allowed:
                    idx_t = torch.tensor(allowed, dtype=torch.long, device=device)
                    sub_logits = row_logits.index_select(0, idx_t)
                    if bool(torch.isfinite(sub_logits).any()):
                        sub_lp = row_lp.index_select(0, idx_t)
                        k_take = min(k_per_beam, len(allowed))
                        top_vals, top_pos = torch.topk(sub_lp, k=k_take, dim=-1)
                        for val_t, pos_t in zip(top_vals.tolist(), top_pos.tolist()):
                            tok_id = int(allowed[int(pos_t)])
                            candidates.append((beam_lp + float(val_t), i, tok_id))
                        used_mask = True
            if not used_mask:
                masked_lp = row_lp.clone()
                masked_lp[eos_id] = float("-inf")
                masked_lp[sys_id] = float("-inf")
                k_take = min(k_per_beam, max(1, vocab_size - 2))
                top_vals, top_ids = torch.topk(masked_lp, k=k_take, dim=-1)
                for val_t, tid_t in zip(top_vals.tolist(), top_ids.tolist()):
                    if float(val_t) == float("-inf"):
                        continue
                    tid_int = int(tid_t)
                    if tid_int == eos_id or tid_int == sys_id:
                        continue
                    candidates.append((beam_lp + float(val_t), i, tid_int))

        if not candidates:
            break

        candidates.sort(key=lambda c: c[0], reverse=True)
        next_active: List[Tuple[List[int], float, grammar.PrefixState]] = []
        next_parents: List[int] = []

        for rank, (raw_lp, parent_i, tok_id) in enumerate(candidates):
            if tok_id == eos_id or tok_id == sys_id:
                continue
            p_toks, _, p_state = active_beams[parent_i]
            new_toks = p_toks + [tok_id]
            tok_str = id_to_tok.get(tok_id)
            is_bar_close = tok_str in grammar.BAR_LIKES

            if is_bar_close:
                meter_err, voice_err = _measure_violation_errors(p_state)
                score = raw_lp - float(meter_w) * meter_err - float(voice_w) * voice_err
                end_state = _advance_grammar_state(
                    p_state.copy(), tok_id, strict=strict_grammar, vocab_size=vocab_size
                )
                finished_bars.append((score, raw_lp, new_toks, end_state))
                continue

            if len(next_active) < beam_size:
                c_state = _advance_grammar_state(
                    p_state.copy(), tok_id, strict=strict_grammar, vocab_size=vocab_size
                )
                next_active.append((new_toks, raw_lp, c_state))
                next_parents.append(parent_i)

            if len(next_active) >= beam_size and rank >= beam_size * 2:
                break

        active_beams = next_active
        if not active_beams:
            break

        # Exact upper bound: future log_prob <= 0 and meter_err, voice_err >= 0.
        if len(finished_bars) >= beam_size:
            best_fin_score = max(f[0] for f in finished_bars)
            best_act_lp = max(a[1] for a in active_beams)
            if best_fin_score >= best_act_lp:
                break

        if step_idx + 1 < max_bar_toks:
            parent_idx_t = torch.tensor(next_parents, dtype=torch.long, device=device)
            past_kvs = _reorder_beam_kvs(past_kvs, parent_idx_t, base_cross_kvs)

    if not finished_bars:
        return None, None, kv0, float("-inf")

    if max_lp_drop is not None and float(max_lp_drop) >= 0.0:
        ref_lp = max(f[1] for f in finished_bars)
        eligible = [f for f in finished_bars if f[1] >= ref_lp - float(max_lp_drop)]
        pool = eligible if eligible else finished_bars
    else:
        pool = finished_bars
    best_score, _, best_toks, best_state = max(
        pool, key=lambda f: (f[0], f[1])
    )
    return best_toks, best_state, kv0, float(best_score)


def _get_model_router(model: Any) -> Any:
    """Return the ink-token router module for both UniOMR v1 and v2 models."""
    router = getattr(model, "router", None)
    if router is not None:
        return router
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        return getattr(encoder, "router", None)
    return None


@contextlib.contextmanager
def override_max_keep_ratio(
    model: Any,
    max_keep_ratio: Optional[float],
) -> Iterator[None]:
    """Temporarily override router ``max_keep_ratio`` for eval-time ablations (v1 & v2)."""
    if max_keep_ratio is None:
        yield
        return
    val = float(max_keep_ratio)
    if not (0.0 < val <= 1.0):
        raise ValueError(f"max_keep_ratio_override must be in (0, 1], got {val}")
    router = _get_model_router(model)
    if router is None or not hasattr(router, "max_keep_ratio"):
        raise AttributeError(
            f"Model {type(model).__name__} has no router with max_keep_ratio"
        )
    orig = float(router.max_keep_ratio)
    router.max_keep_ratio = val
    try:
        yield
    finally:
        router.max_keep_ratio = orig


# ---------------------------------------------------------------------------
# 1. FSA Grammar-Constrained Greedy Decoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_constrained(
    model: uni_omr.UniOMRNet,
    images: torch.Tensor,
    max_len: int = 1024,
    strict: bool = False,
    keep_ratio_override: Optional[float] = None,
    return_info: bool = False,
) -> Any:
    """KV-cached greedy decoding with per-item ``grammar.PrefixState`` masking.

    Resets ``PrefixState`` cleanly whenever ``SYS_ID`` (``<sys>``) is emitted,
    and falls back safely to unconstrained argmax if a mask ever empties.
    """
    mem, _, router_info = model.encode_image(
        images,
        keep_ratio_override=keep_ratio_override,
        return_aux=False,
    )
    B = int(images.shape[0])
    device = images.device
    vocab_size = int(model.cfg.vocab_size)
    pad_id, bos_id, eos_id, _, tok_to_id, _ = _vocab_spec(vocab_size)

    cur_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    past_kvs: Optional[List[Any]] = None
    finished = [False] * B
    states = [grammar.PrefixState(strict=strict, vocab_map=tok_to_id) for _ in range(B)]
    out_tokens: List[torch.Tensor] = [cur_ids]

    for _ in range(max_len):
        logits, _, past_kvs = model.decode_tokens(
            cur_ids,
            memory=mem if past_kvs is None else None,
            return_mtp=False,
            past_kvs=past_kvs,
            use_cache=True,
        )
        step_logits = logits[:, -1, :]  # (B, V)
        next_ids: List[int] = []
        for b in range(B):
            if finished[b]:
                next_ids.append(pad_id)
                continue
            tok_id = _select_constrained_token(step_logits[b], states[b])
            next_ids.append(tok_id)
            if tok_id == eos_id:
                finished[b] = True
            else:
                states[b] = _advance_grammar_state(
                    states[b], tok_id, strict=strict, vocab_size=vocab_size
                )

        next_tok = torch.tensor(next_ids, dtype=torch.long, device=device)
        out_tokens.append(next_tok.unsqueeze(1))
        if all(finished):
            break
        cur_ids = next_tok.unsqueeze(1)

    seq = torch.cat(out_tokens, dim=1)
    if return_info:
        return seq, router_info
    return seq


# ---------------------------------------------------------------------------
# 2. MTP-4 Self-Speculative Decoding (Exact Greedy Parity)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_mtp_spec(
    model: uni_omr.UniOMRNet,
    images: torch.Tensor,
    max_len: int = 1024,
    use_grammar: bool = False,
    strict_grammar: bool = False,
    keep_ratio_override: Optional[float] = None,
    return_stats: bool = True,
    measure_beam_size: int = 0,
    max_bar_toks: int = 64,
    measure_beam_cap: Optional[int] = 4,
    measure_beam_lp: Optional[float] = 2.0,
) -> Any:
    """MTP-k self-speculative KV-cached decoding using ``model.mtp_heads``.

    Drafts ``M = len(model.mtp_heads)`` lookahead tokens in parallel, verifies
    them in a single parallel KV-cached decoder forward pass, accepts the
    longest matching prefix (truncating KV cache cleanly to the accepted
    length), and guarantees exact token-for-token parity with standard greedy
    ``model.generate()`` (or ``generate_constrained()`` when ``use_grammar=True``).

    When ``measure_beam_size > 1``, tracks measure boundaries and, whenever an
    interior closed measure violates meter or cross-voice duration conservation,
    rewinds KV cache to the bar start, runs a ``measure_beam_size``-beam search
    scored by ``log_prob - 6.0 * meter_err - 4.0 * voice_err``, and resumes
    fast MTP-k decoding from the winning measure.

    Returns:
        ``(seq_ids, stats)`` when ``return_stats=True``, else ``seq_ids``.
    """
    t0 = time.perf_counter()
    B = int(images.shape[0])
    device = images.device
    vocab_size = int(model.cfg.vocab_size)
    pad_id, bos_id, eos_id, sys_id, tok_to_id, id_to_tok = _vocab_spec(vocab_size)
    use_measure_beam = int(measure_beam_size) > 1

    # For B > 1 with measure-scoped beam search, run each item independently and pad.
    if use_measure_beam and B > 1:
        seqs_b: List[torch.Tensor] = []
        tot_prop = 0
        tot_acc = 0
        tot_passes = 0
        tot_triggers = 0
        tot_repaired = 0
        last_router_info: Dict[str, Any] = {}
        for b in range(B):
            s_b, st_b = generate_mtp_spec(
                model,
                images[b : b + 1],
                max_len=max_len,
                use_grammar=use_grammar,
                strict_grammar=strict_grammar,
                keep_ratio_override=keep_ratio_override,
                return_stats=True,
                measure_beam_size=measure_beam_size,
                max_bar_toks=max_bar_toks,
                measure_beam_cap=measure_beam_cap,
                measure_beam_lp=measure_beam_lp,
            )
            seqs_b.append(s_b[0])
            tot_prop += int(st_b["draft_proposed"])
            tot_acc += int(st_b["draft_accepted"])
            tot_passes += int(st_b["verify_passes"])
            tot_triggers += int(st_b.get("measure_beam_triggers", 0))
            tot_repaired += int(st_b.get("measure_beam_repaired", 0))
            last_router_info = st_b.get("router_info", {})
        max_out = max(int(s.shape[0]) for s in seqs_b)
        seq = torch.full((B, max_out), pad_id, dtype=torch.long, device=device)
        for b, s in enumerate(seqs_b):
            seq[b, : int(s.shape[0])] = s
        elapsed = max(1e-6, time.perf_counter() - t0)
        total_gen_toks = int((seq[:, 1:] != pad_id).sum().item())
        stats = {
            "mtp_k": 1 + len(model.mtp_heads),
            "draft_proposed": tot_prop,
            "draft_accepted": tot_acc,
            "acceptance_rate": round(tot_acc / max(1, tot_prop), 6) if tot_prop > 0 else 0.0,
            "verify_passes": tot_passes,
            "tokens_generated": total_gen_toks,
            "mean_tokens_per_pass": round(total_gen_toks / max(1, tot_passes), 3),
            "elapsed_secs": round(elapsed, 6),
            "tokens_per_sec": round(total_gen_toks / elapsed, 2),
            "measure_beam_triggers": tot_triggers,
            "measure_beam_repaired": tot_repaired,
            "router_info": last_router_info,
        }
        return (seq, stats) if return_stats else seq

    mem, _, router_info = model.encode_image(
        images,
        keep_ratio_override=keep_ratio_override,
        return_aux=False,
    )
    n_mtp = len(model.mtp_heads)

    bos = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    if max_len <= 0:
        stats = {
            "mtp_k": 1 + n_mtp,
            "draft_proposed": 0,
            "draft_accepted": 0,
            "acceptance_rate": 0.0,
            "verify_passes": 0,
            "tokens_generated": 0,
            "mean_tokens_per_pass": 0.0,
            "elapsed_secs": max(1e-6, time.perf_counter() - t0),
            "tokens_per_sec": 0.0,
            "measure_beam_triggers": 0,
            "measure_beam_repaired": 0,
            "router_info": router_info,
        }
        return (bos, stats) if return_stats else bos

    finished = [False] * B
    states = (
        [grammar.PrefixState(strict=strict_grammar, vocab_map=tok_to_id) for _ in range(B)]
        if (use_grammar or use_measure_beam)
        else None
    )
    out_tokens: List[torch.Tensor] = [bos]

    # Measure-scoped beam tracking state (B == 1 when use_measure_beam is True).
    bar_idx = 0
    bar_start_idx = 1
    bar_start_state: Optional[grammar.PrefixState] = (
        states[0].copy() if (use_measure_beam and states is not None) else None
    )
    measure_beam_triggers = 0
    measure_beam_repaired = 0
    header_guard = (
        MeasureBeamHeaderGuard(max_triggers=measure_beam_cap)
        if use_measure_beam
        else None
    )

    def _pick_step(step_logits: torch.Tensor) -> torch.Tensor:
        if not use_grammar:
            tok = step_logits.argmax(dim=-1)
            for b in range(B):
                if finished[b]:
                    tok[b] = pad_id
            return tok
        assert states is not None
        picked: List[int] = []
        for b in range(B):
            if finished[b]:
                picked.append(pad_id)
            else:
                picked.append(_select_constrained_token(step_logits[b], states[b]))
        return torch.tensor(picked, dtype=torch.long, device=device)

    def _commit_token(tok: torch.Tensor) -> bool:
        """Append ``tok``, advance grammar state, and return True if measure-beam triggers."""
        nonlocal bar_idx, bar_start_idx, bar_start_state
        out_tokens.append(tok.unsqueeze(1))
        trigger_repair = False
        for b in range(B):
            if finished[b]:
                continue
            tid = int(tok[b].item())
            if tid == eos_id:
                finished[b] = True
                continue
            if states is None:
                continue
            if use_measure_beam and b == 0:
                tok_str = id_to_tok.get(tid)
                is_bar_close = tok_str in grammar.BAR_LIKES
                if is_bar_close:
                    bar_len = len(out_tokens) - bar_start_idx
                    if (
                        bar_idx >= 1
                        and tok_str == "bar"
                        and 1 <= bar_len <= max_bar_toks
                    ):
                        meter_err, voice_err = _measure_violation_errors(states[0])
                        if header_guard is not None:
                            if header_guard.should_repair(states[0], meter_err, voice_err):
                                trigger_repair = True
                        elif meter_err > 1e-3 or voice_err > 1e-3:
                            trigger_repair = True
                    states[0] = _advance_grammar_state(
                        states[0], tid, strict=strict_grammar, vocab_size=vocab_size
                    )
                    if not trigger_repair:
                        bar_idx += 1
                        bar_start_idx = len(out_tokens)
                        bar_start_state = states[0].copy()
                else:
                    states[0] = _advance_grammar_state(
                        states[0], tid, strict=strict_grammar, vocab_size=vocab_size
                    )
                    if tid == sys_id or states[0].header_open:
                        bar_start_idx = len(out_tokens)
                        bar_start_state = states[0].copy()
            elif use_grammar:
                states[b] = _advance_grammar_state(
                    states[b], tid, strict=strict_grammar, vocab_size=vocab_size
                )
        return trigger_repair

    def _extract_drafts(mtp_logits_list: List[torch.Tensor], pos_idx: int) -> List[torch.Tensor]:
        drafts_out: List[torch.Tensor] = []
        for m_logits in mtp_logits_list:
            d = m_logits[:, pos_idx, :].argmax(dim=-1)
            for b in range(B):
                if finished[b]:
                    d[b] = pad_id
            drafts_out.append(d)
        return drafts_out

    # Step 0 (prefill BOS): get first verified token + initial M drafts.
    logits0, mtp0, past_kvs = model.decode_tokens(
        bos,
        memory=mem,
        return_mtp=(n_mtp > 0),
        past_kvs=None,
        use_cache=True,
    )
    base_cross_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    if use_measure_beam:
        for _, cross_kv in past_kvs:
            ck, cv = cross_kv
            base_cross_kvs.append((
                ck.expand(measure_beam_size, -1, -1, -1).contiguous(),
                cv.expand(measure_beam_size, -1, -1, -1).contiguous(),
            ))

    cur_tok = _pick_step(logits0[:, -1, :])
    _commit_token(cur_tok)
    gen_len = 1
    drafts = _extract_drafts(mtp0, -1) if n_mtp > 0 else []

    draft_proposed = 0
    draft_accepted = 0
    verify_passes = 1  # counts prefill pass

    def _apply_measure_repair(cur_kv: List[Any]) -> None:
        nonlocal gen_len, past_kvs, cur_tok, drafts
        nonlocal bar_idx, bar_start_idx, bar_start_state
        nonlocal measure_beam_triggers, measure_beam_repaired
        assert states is not None and bar_start_state is not None
        measure_beam_triggers += 1
        orig_bar_toks = [int(t[0, 0].item()) for t in out_tokens[bar_start_idx:]]
        kv_prefix = _truncate_past_kvs(cur_kv, bar_start_idx - 1)
        trigger_id = int(out_tokens[bar_start_idx - 1][0, 0].item())
        rem_steps = min(max_bar_toks, max_len - (bar_start_idx - 1))

        best_toks, best_state, kv0, _ = _beam_repair_measure(
            model=model,
            kv_prefix=kv_prefix,
            base_cross_kvs=base_cross_kvs,
            trigger_id=trigger_id,
            bar_start_state=bar_start_state,
            beam_size=int(measure_beam_size),
            max_bar_toks=rem_steps,
            vocab_size=vocab_size,
            use_grammar=use_grammar,
            strict_grammar=strict_grammar,
            device=device,
            max_lp_drop=measure_beam_lp,
        )
        if best_toks is not None and best_state is not None:
            if best_toks != orig_bar_toks:
                measure_beam_repaired += 1
            out_tokens[bar_start_idx:] = [
                torch.tensor([[t]], dtype=torch.long, device=device) for t in best_toks
            ]
            gen_len = len(out_tokens) - 1
            states[0] = best_state
            if best_toks[-1] == eos_id:
                finished[0] = True
                past_kvs = kv0
                drafts = []
            else:
                if len(best_toks) > 1:
                    prefill_ids = torch.tensor(
                        [best_toks[:-1]], dtype=torch.long, device=device
                    )
                    _, _, past_kvs = model.decode_tokens(
                        prefill_ids,
                        memory=None,
                        return_mtp=False,
                        past_kvs=kv0,
                        use_cache=True,
                    )
                else:
                    past_kvs = kv0
                cur_tok = torch.tensor([best_toks[-1]], dtype=torch.long, device=device)
                drafts = []
        else:
            # Fallback: keep original greedy measure and resume cleanly.
            past_kvs = _truncate_past_kvs(cur_kv, len(out_tokens) - 1)
            cur_tok = out_tokens[-1].squeeze(1)
            drafts = []

        bar_idx += 1
        bar_start_idx = len(out_tokens)
        bar_start_state = states[0].copy()

    # Invariant at top of loop:
    #   - out_tokens has [BOS, tok_1, ..., tok_{gen_len}]
    #   - past_kvs has length gen_len (covers [BOS, tok_1, ..., tok_{gen_len-1}])
    #   - cur_tok == tok_{gen_len} (verified & committed, not yet in past_kvs)
    #   - drafts has M speculative tokens for tok_{gen_len+1} .. tok_{gen_len+M}
    while gen_len < max_len and not all(finished):
        rem = max_len - gen_len
        m = min(len(drafts), rem - 1) if n_mtp > 0 else 0
        past_len = int(past_kvs[0][0][0].shape[2])

        if m > 0:
            verify_ids = torch.stack([cur_tok] + drafts[:m], dim=1)  # (B, 1 + m)
        else:
            verify_ids = cur_tok.unsqueeze(1)  # (B, 1)

        v_logits, v_mtp, new_past_kvs = model.decode_tokens(
            verify_ids,
            memory=None,
            return_mtp=(n_mtp > 0),
            past_kvs=past_kvs,
            use_cache=True,
        )
        verify_passes += 1
        draft_proposed += m

        n_acc = 0
        mismatched = False
        for j in range(m):
            target_j = _pick_step(v_logits[:, j, :])
            draft_j = drafts[j]
            # For already-finished items, force draft_j == pad_id so they match target_j.
            if any(finished):
                draft_j = draft_j.clone()
                for b in range(B):
                    if finished[b]:
                        draft_j[b] = pad_id

            if bool((target_j == draft_j).all()):
                trig = _commit_token(draft_j)
                gen_len += 1
                n_acc += 1
                draft_accepted += 1
                if trig:
                    _apply_measure_repair(new_past_kvs)
                    mismatched = True
                    break
                if all(finished):
                    past_kvs = _truncate_past_kvs(new_past_kvs, past_len + 1 + n_acc)
                    mismatched = True
                    break
            else:
                # First mismatch at draft j: commit corrected target_j and truncate KV cache.
                trig = _commit_token(target_j)
                gen_len += 1
                if trig:
                    _apply_measure_repair(new_past_kvs)
                else:
                    cur_tok = target_j
                    drafts = _extract_drafts(v_mtp, j) if n_mtp > 0 else []
                    past_kvs = _truncate_past_kvs(new_past_kvs, past_len + 1 + n_acc)
                mismatched = True
                break

        if not mismatched and not all(finished):
            # All m drafts matched: emit the free bonus token from position m!
            bonus_tok = _pick_step(v_logits[:, m, :])
            trig = _commit_token(bonus_tok)
            gen_len += 1
            if trig:
                _apply_measure_repair(new_past_kvs)
            else:
                cur_tok = bonus_tok
                drafts = _extract_drafts(v_mtp, m) if n_mtp > 0 else []
                past_kvs = new_past_kvs

    elapsed = max(1e-6, time.perf_counter() - t0)
    seq = torch.cat(out_tokens, dim=1)
    # Count non-PAD generated tokens (excluding BOS).
    total_gen_toks = int((seq[:, 1:] != pad_id).sum().item())

    stats = {
        "mtp_k": 1 + n_mtp,
        "draft_proposed": int(draft_proposed),
        "draft_accepted": int(draft_accepted),
        "acceptance_rate": round(draft_accepted / max(1, draft_proposed), 6) if draft_proposed > 0 else 0.0,
        "verify_passes": int(verify_passes),
        "tokens_generated": total_gen_toks,
        "mean_tokens_per_pass": round(total_gen_toks / max(1, verify_passes), 3),
        "elapsed_secs": round(elapsed, 6),
        "tokens_per_sec": round(total_gen_toks / elapsed, 2),
        "measure_beam_triggers": int(measure_beam_triggers),
        "measure_beam_repaired": int(measure_beam_repaired),
        "measure_beam_header_guard_tripped": bool(header_guard.tripped) if header_guard is not None else False,
        "router_info": router_info,
    }
    if return_stats:
        return seq, stats
    return seq


# ---------------------------------------------------------------------------
# 2b. KV-Cached, Length-Normalized, FSA-Grammar-Constrained Beam Search
# ---------------------------------------------------------------------------

def _reorder_beam_kvs(
    past_kvs: List[Any],
    parent_idx: torch.Tensor,
    base_cross_kvs: List[Tuple[torch.Tensor, torch.Tensor]],
) -> List[Any]:
    """Gather decoder self-attention KV cache for surviving beam parents.

    Reuses pre-expanded contiguous cross-attention K/V slices (zero re-projection).
    """
    n_active = int(parent_idx.shape[0])
    reordered: List[Any] = []
    for layer_idx, (self_kv, _) in enumerate(past_kvs):
        k_self, v_self = self_kv
        k_cross_base, v_cross_base = base_cross_kvs[layer_idx]
        reordered.append(
            (
                (
                    k_self.index_select(0, parent_idx),
                    v_self.index_select(0, parent_idx),
                ),
                (
                    k_cross_base[:n_active],
                    v_cross_base[:n_active],
                ),
            )
        )
    return reordered


def _beam_norm_score(raw_score: float, length: int, length_penalty: float) -> float:
    if length_penalty == 0.0:
        return raw_score
    return raw_score / (float(max(1, length)) ** length_penalty)


@torch.no_grad()
def generate_beam(
    model: uni_omr.UniOMRNet,
    images: torch.Tensor,
    max_len: int = 1024,
    beam_size: int = 3,
    length_penalty: float = 0.6,
    use_grammar: bool = True,
    strict_grammar: bool = False,
    keep_ratio_override: Optional[float] = None,
    return_info: bool = False,
) -> Any:
    """KV-cached, length-normalized, FSA-grammar-constrained beam search.

    Encodes ``images`` once via ``model.encode_image``, pre-expands cross-attn
    K/V caches once per item, and batches all active beams in a single
    KV-cached ``model.decode_tokens`` pass per step.

    When ``beam_size == 1`` and ``use_grammar=True``, matches
    ``generate_constrained()`` token-for-token.
    """
    if beam_size < 1:
        raise ValueError(f"beam_size must be >= 1, got {beam_size}")

    mem, _, router_info = model.encode_image(
        images,
        keep_ratio_override=keep_ratio_override,
        return_aux=False,
    )
    B = int(images.shape[0])
    device = images.device
    vocab_size = int(model.cfg.vocab_size)
    pad_id, bos_id, eos_id, _, tok_to_id, _ = _vocab_spec(vocab_size)

    if max_len <= 0:
        seq = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        return (seq, router_info) if return_info else seq

    best_seqs: List[List[int]] = []
    lp_exp = float(length_penalty)

    for b in range(B):
        if isinstance(mem, (tuple, list)):
            mem_b: Any = tuple(m[b : b + 1] for m in mem)
        else:
            mem_b = mem[b : b + 1]  # (1, L_mem, D)
        bos_b = torch.full((1, 1), bos_id, dtype=torch.long, device=device)

        # Prefill BOS once to populate cross-attn K/V cache.
        logits0, _, past_kvs0 = model.decode_tokens(
            bos_b,
            memory=mem_b,
            return_mtp=False,
            past_kvs=None,
            use_cache=True,
        )

        # Pre-expand cross-attn K/V once to (beam_size, H, L_mem, D_h) contiguous
        # so slicing [:n_active] is zero-copy and safe on all SDPA backends.
        base_cross_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _, cross_kv in past_kvs0:
            ck, cv = cross_kv
            if beam_size > 1:
                ck_exp = ck.expand(beam_size, -1, -1, -1).contiguous()
                cv_exp = cv.expand(beam_size, -1, -1, -1).contiguous()
            else:
                ck_exp, cv_exp = ck, cv
            base_cross_kvs.append((ck_exp, cv_exp))

        # Active beam tuples: (tokens, raw_score, grammar_state)
        init_state = (
            grammar.PrefixState(strict=strict_grammar, vocab_map=tok_to_id)
            if use_grammar
            else None
        )
        active_beams: List[Tuple[List[int], float, Optional[grammar.PrefixState]]] = [
            ([], 0.0, init_state)
        ]
        # Completed beam tuples: (norm_score, raw_score, tokens)
        finished_beams: List[Tuple[float, float, List[int]]] = []

        step_logits = logits0[:, -1, :]  # (1, V)
        past_kvs: List[Any] = past_kvs0

        for step_idx in range(max_len):
            if step_idx > 0:
                cur_ids = torch.tensor(
                    [[beam_toks[-1]] for beam_toks, _, _ in active_beams],
                    dtype=torch.long,
                    device=device,
                )
                logits, _, past_kvs = model.decode_tokens(
                    cur_ids,
                    memory=None,
                    return_mtp=False,
                    past_kvs=past_kvs,
                    use_cache=True,
                )
                step_logits = logits[:, -1, :]  # (n_active, V)

            step_log_probs = F.log_softmax(step_logits.float(), dim=-1)  # (n_active, V)

            # Collect top candidates across all active beams.
            # Tuple: (norm_score, raw_score, parent_beam_idx, tok_id)
            candidates: List[Tuple[float, float, int, int]] = []
            k_per_beam = 1 if beam_size == 1 else min(vocab_size, beam_size + 1)

            for i, (beam_toks, beam_score, beam_state) in enumerate(active_beams):
                next_len = len(beam_toks) + 1
                row_logits = step_logits[i]
                row_lp = step_log_probs[i]

                used_mask = False
                if use_grammar and beam_state is not None:
                    allowed = _allowed_ids_for_state(beam_state, vocab_size)
                    if allowed:
                        idx_t = torch.tensor(allowed, dtype=torch.long, device=device)
                        sub_logits = row_logits.index_select(0, idx_t)
                        if bool(torch.isfinite(sub_logits).any()):
                            sub_lp = row_lp.index_select(0, idx_t)
                            k_take = min(k_per_beam, len(allowed))
                            top_vals, top_pos = torch.topk(sub_lp, k=k_take, dim=-1)
                            for val_t, pos_t in zip(top_vals.tolist(), top_pos.tolist()):
                                tok_id = int(allowed[int(pos_t)])
                                raw_s = beam_score + float(val_t)
                                norm_s = _beam_norm_score(raw_s, next_len, lp_exp)
                                candidates.append((norm_s, raw_s, i, tok_id))
                            used_mask = True

                if not used_mask:
                    k_take = min(k_per_beam, vocab_size)
                    top_vals, top_ids = torch.topk(row_lp, k=k_take, dim=-1)
                    for val_t, tid_t in zip(top_vals.tolist(), top_ids.tolist()):
                        tok_id = int(tid_t)
                        raw_s = beam_score + float(val_t)
                        norm_s = _beam_norm_score(raw_s, next_len, lp_exp)
                        candidates.append((norm_s, raw_s, i, tok_id))

            if not candidates:
                break

            candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

            next_active: List[Tuple[List[int], float, Optional[grammar.PrefixState]]] = []
            next_parents: List[int] = []

            if beam_size == 1:
                norm_s, raw_s, parent_i, tok_id = candidates[0]
                p_toks, _, p_state = active_beams[parent_i]
                new_toks = p_toks + [tok_id]
                if tok_id == eos_id:
                    finished_beams.append((norm_s, raw_s, new_toks))
                else:
                    if use_grammar and p_state is not None:
                        c_state = _advance_grammar_state(
                            p_state.copy(), tok_id, strict=strict_grammar, vocab_size=vocab_size
                        )
                    else:
                        c_state = None
                    next_active.append((new_toks, raw_s, c_state))
                    next_parents.append(parent_i)
            else:
                for rank, (norm_s, raw_s, parent_i, tok_id) in enumerate(candidates):
                    p_toks, _, p_state = active_beams[parent_i]
                    new_toks = p_toks + [tok_id]
                    if tok_id == eos_id:
                        if rank < beam_size:
                            finished_beams.append((norm_s, raw_s, new_toks))
                        continue
                    if len(next_active) < beam_size:
                        if use_grammar and p_state is not None:
                            c_state = _advance_grammar_state(
                                p_state.copy(), tok_id, strict=strict_grammar, vocab_size=vocab_size
                            )
                        else:
                            c_state = None
                        next_active.append((new_toks, raw_s, c_state))
                        next_parents.append(parent_i)
                    if len(next_active) >= beam_size and rank >= beam_size - 1:
                        break

            active_beams = next_active
            if not active_beams:
                break

            # Early stop when we have >= beam_size finished hypotheses and even the
            # highest-scoring active beam cannot beat the best finished hypothesis.
            if len(finished_beams) >= beam_size:
                best_fin_norm = max(f[0] for f in finished_beams)
                best_act_raw = max(a[1] for a in active_beams)
                # Since future log-probs <= 0, raw score can only stay <= best_act_raw,
                # and dividing a negative score by max_len**lp_exp is an upper bound.
                upper_bound = _beam_norm_score(best_act_raw, max_len, lp_exp)
                if best_fin_norm >= upper_bound:
                    break

            if step_idx + 1 < max_len:
                parent_idx_t = torch.tensor(next_parents, dtype=torch.long, device=device)
                past_kvs = _reorder_beam_kvs(past_kvs, parent_idx_t, base_cross_kvs)

        if finished_beams:
            best_fin = max(finished_beams, key=lambda f: (f[0], f[1]))
            best_toks = best_fin[2]
        elif active_beams:
            best_act = max(
                active_beams,
                key=lambda a: (_beam_norm_score(a[1], len(a[0]), lp_exp), a[1]),
            )
            best_toks = best_act[0]
        else:
            best_toks = []

        best_seqs.append([bos_id] + best_toks)

    max_out_len = max(len(s) for s in best_seqs)
    seq_tensor = torch.full((B, max_out_len), pad_id, dtype=torch.long, device=device)
    for b, s in enumerate(best_seqs):
        seq_tensor[b, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)

    if return_info:
        return seq_tensor, router_info
    return seq_tensor


# ---------------------------------------------------------------------------
# 3. Manifest Resolution, Stitching, and Checkpoint Loading
# ---------------------------------------------------------------------------

def resolve_manifest_pieces(manifest_path: str | Path) -> Tuple[Path, List[Dict[str, Any]]]:
    """Load ``suite31_hw_manifest.json`` and resolve ``clean``/``photo``/``gt`` paths."""
    mpath = Path(manifest_path)
    base_dir = mpath.parent
    raw = json.loads(mpath.read_text(encoding="utf-8"))
    resolved: List[Dict[str, Any]] = []

    def _entry_path(entry: Any) -> Path:
        if isinstance(entry, dict):
            return base_dir / entry["file"]
        return base_dir / str(entry)

    for p in raw["pieces"]:
        resolved.append({
            "slug": str(p["slug"]),
            "handwritten": bool(p.get("handwritten", False)),
            "stratum": esg.classify_suite31_stratum(p),
            "clean": _entry_path(p["clean"]),
            "photo": _entry_path(p["photo"]),
            "gt": _entry_path(p["gt"]),
        })
    return base_dir, resolved


def stitch_uni_tokens(pred_toks: Sequence[str]) -> Tuple[List[str], int]:
    """Split UniOMR full-page tokens on ``<sys>`` and stitch into page bekern.

    Uses ``eval_sys_gate.stitch_page_toks`` so header deduplication and
    canonical tail ordering match the 2-stage ``suite31_hw`` evaluator.
    Returns ``(stitched_tokens, n_systems)``.
    """
    chunks: List[List[str]] = []
    cur: List[str] = []
    for t in pred_toks:
        if t == uni_omr.SYS_TOKEN:
            if cur:
                chunks.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        chunks.append(cur)

    if not chunks:
        return [], 0
    if len(chunks) == 1:
        return esg.canonicalize_bekern_tail_order(chunks[0]), 1
    return esg.stitch_page_toks(chunks), len(chunks)


_V2_CONFIG_KEYS = frozenset({
    "n_axial_layers",
    "n_global_layers",
    "n_kv_heads",
    "backbone_channels",
    "backbone_blocks",
    "backbone_expand",
    "axial_ffn_mult",
    "irope_nope_every",
    "n_register_tokens",
    "rope_coord_scale",
    "use_sys_y_aux",
    "use_sys_count_aux",
    "use_staff_seg_aux",
    "use_stroke_hog_head",
})

_V2_STATE_PREFIXES = (
    "encoder.axial_layers.",
    "encoder.patch_merger.",
    "encoder.global_layers.",
    "deepstack_fusions.",
    "backbone.p2_to_p3.",
    "backbone.p2_to_p4.",
)


def _is_v2_checkpoint(
    ck: Any,
    sd: Dict[str, Any],
    cfg_dict: Dict[str, Any],
) -> bool:
    """Detect whether a loaded checkpoint payload belongs to ``UniOMRv2Net``."""
    if isinstance(ck, dict):
        arch = str(ck.get("arch", "")).strip().lower()
        if arch in ("uni_omr_v2", "uniomr_v2", "v2"):
            return True
        if arch in ("uni_omr", "uni_omr_v1", "uniomr_v1", "v1"):
            return False
    if any(k in _V2_CONFIG_KEYS for k in cfg_dict):
        return True
    for k in sd.keys():
        if isinstance(k, str) and k.startswith(_V2_STATE_PREFIXES):
            return True
    return False


def load_uni_checkpoint(
    ckpt_path: str | Path,
    device: torch.device,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load a ``UniOMRNet`` (v1) or ``UniOMRv2Net`` (v2) checkpoint onto ``device`` in eval mode.

    Auto-detects ``UniOMRv2Net`` via ``ck["arch"] == "uni_omr_v2"``, v2 config keys
    (e.g. ``n_axial_layers``, ``n_kv_heads``, ``backbone_channels``), or v2 state-dict
    prefixes (e.g. ``encoder.axial_layers.*``, ``deepstack_fusions.*``).
    """
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    raw_cfg: Any = {}
    if isinstance(ck, dict):
        raw_cfg = ck.get("config") or ck.get("cfg") or {}
    if isinstance(raw_cfg, (uni_omr.UniOMRConfig, uni_omr_v2.UniOMRv2Config)):
        cfg_dict = asdict(raw_cfg)
    elif isinstance(raw_cfg, dict):
        cfg_dict = dict(raw_cfg)
    else:
        cfg_dict = {}

    if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict):
        sd = ck["model"]
    elif isinstance(ck, dict) and "state_dict" in ck and isinstance(ck["state_dict"], dict):
        sd = ck["state_dict"]
    elif isinstance(ck, dict):
        sd = ck
    else:
        raise ValueError(f"Unsupported checkpoint payload type: {type(ck)!r}")

    is_v2 = _is_v2_checkpoint(ck, sd, cfg_dict)
    if is_v2:
        valid_fields = {f.name for f in fields(uni_omr_v2.UniOMRv2Config)}
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_fields}
        for tup_k in ("backbone_channels", "backbone_blocks"):
            if tup_k in filtered_cfg and isinstance(filtered_cfg[tup_k], list):
                filtered_cfg[tup_k] = tuple(int(x) for x in filtered_cfg[tup_k])
        cfg: Any = uni_omr_v2.UniOMRv2Config(**filtered_cfg)
        model: Any = uni_omr_v2.UniOMRv2Net(cfg)
        arch_name = "uni_omr_v2"
    else:
        valid_fields = {f.name for f in fields(uni_omr.UniOMRConfig)}
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_fields}
        cfg = uni_omr.UniOMRConfig(**filtered_cfg)
        model = uni_omr.UniOMRNet(cfg)
        arch_name = "uni_omr_v1"

    inc = model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    meta = {
        "arch": arch_name,
        "step": ck.get("step") if isinstance(ck, dict) else None,
        "val_ser": ck.get("val_ser") if isinstance(ck, dict) else None,
        "missing_keys": list(inc.missing_keys),
        "unexpected_keys": list(inc.unexpected_keys),
    }
    return model, cfg, meta


# ---------------------------------------------------------------------------
# 3b. Scale-Only Multi-View TTA & best_fsa_beat View Selection
# ---------------------------------------------------------------------------

def parse_tta_scales(raw: Optional[str]) -> List[int]:
    """Parse ``--tta-scales`` (e.g. ``'1152,1280,1536'``) into unique positive ints."""
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    scales: List[int] = []
    seen: set[int] = set()
    for part in text.split(","):
        tok = part.strip()
        if not tok:
            continue
        dim = int(tok)
        if dim < 64:
            raise ValueError(f"--tta-scales dimensions must be >= 64, got {dim}")
        if dim not in seen:
            seen.add(dim)
            scales.append(dim)
    return scales


def detect_token_loop(
    toks: Sequence[str],
    min_ngram: int = 4,
    max_ngram: int = 24,
    min_repeats: int = 4,
    max_single_run: int = 8,
) -> bool:
    """Detect degenerate consecutive token loops (repeated n-grams or stuck single tokens)."""
    n = len(toks)
    if n < min_ngram * min_repeats:
        # Still check stuck single-token runs on short sequences.
        run = 1
        for i in range(1, n):
            if toks[i] == toks[i - 1] and toks[i] not in grammar.BAR_LIKES and toks[i] != grammar.BAR_REST:
                run += 1
                if run >= max_single_run:
                    return True
            else:
                run = 1
        return False

    # 1. Single non-bar/non-rest token stuck >= max_single_run consecutive times.
    run = 1
    for i in range(1, n):
        if toks[i] == toks[i - 1] and toks[i] not in grammar.BAR_LIKES and toks[i] != grammar.BAR_REST:
            run += 1
            if run >= max_single_run:
                return True
        else:
            run = 1

    # 2. Consecutive n-gram repeat >= min_repeats times (e.g. A B C D repeated 4x).
    tok_list = list(toks)
    hi_k = min(max_ngram, n // min_repeats)
    for k in range(min_ngram, hi_k + 1):
        limit = n - k * min_repeats + 1
        i = 0
        while i < limit:
            pat = tok_list[i : i + k]
            reps = 1
            j = i + k
            while j + k <= n and tok_list[j : j + k] == pat:
                reps += 1
                j += k
                if reps >= min_repeats:
                    return True
            i = j if reps > 1 else (i + 1)
    return False


def compute_fsa_beat_metrics(
    hyp_page_toks: Sequence[str],
    hit_eos: bool,
    bekern_decode_ok: bool,
) -> Dict[str, Any]:
    """Compute exact FSA beat-conservation, parse, and loop metrics for ``best_fsa_beat``.

    Matches the proven ``ov-36``/``ov-36b`` ``fsa_beat_score`` formula, augmented with
    explicit degenerate-loop detection (``has_loop``).
    """
    st = grammar.PrefixState(strict=False)
    n_bars = 0
    n_both_ok = 0
    meter_errs: List[float] = []
    voice_errs: List[float] = []
    broken_steps = 0

    for tok in hyp_page_toks:
        if tok == uni_omr.SYS_TOKEN:
            st = grammar.PrefixState(strict=False)
            continue
        if tok in grammar.BAR_LIKES:
            m_err, v_err = _measure_violation_errors(st)
            n_bars += 1
            meter_errs.append(min(4.0, float(m_err)))
            voice_errs.append(min(4.0, float(v_err)))
            if m_err <= 1e-3 and v_err <= 1e-3:
                n_both_ok += 1
        try:
            st.step(tok)
        except ValueError:
            broken_steps += 1

    has_loop = detect_token_loop(hyp_page_toks)
    beat_ok_rate = (float(n_both_ok) / float(n_bars)) if n_bars > 0 else 0.0
    mean_m_err = (sum(meter_errs) / len(meter_errs)) if meter_errs else 1.0
    mean_v_err = (sum(voice_errs) / len(voice_errs)) if voice_errs else 1.0
    broken_rate = float(broken_steps) / float(max(1, len(hyp_page_toks)))

    score = (
        (1.0 if bekern_decode_ok else -2.0)
        + (0.25 if hit_eos else -0.5)
        + (0.0 if not has_loop else -1.0)
        + 1.5 * beat_ok_rate
        - 0.5 * mean_m_err
        - 0.5 * mean_v_err
        - 0.25 * broken_rate
    )
    return {
        "n_bars": n_bars,
        "beat_conservation_rate": round(beat_ok_rate, 6),
        "mean_meter_err": round(mean_m_err, 6),
        "mean_voice_err": round(mean_v_err, 6),
        "broken_steps": broken_steps,
        "has_loop": bool(has_loop),
        "fsa_beat_score": round(float(score), 6),
    }


@torch.no_grad()
def compute_sequence_mean_logprob(
    model: Any,
    img_t: torch.Tensor,
    seq: torch.Tensor,
    pad_id: int,
) -> float:
    """Compute teacher-forced length-normalized mean log-probability for ``seq`` (B=1)."""
    if seq.ndim != 2 or seq.shape[1] < 2:
        return -10.0
    tf_out = model(
        img_t,
        seq[:, :-1],
        return_aux=False,
        return_mtp=False,
    )
    logits_f = tf_out["logits"].float()
    log_probs = F.log_softmax(logits_f, dim=-1)
    targets = seq[:, 1:]
    tok_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    valid = targets != int(pad_id)
    if not bool(valid.any()):
        return -10.0
    return float(tok_lp[valid].mean().item())


TTA_RERANK_MODES: Tuple[str, ...] = ("fsa_logprob", "fsa_logprob_mbr", "mbr")


def compute_mbr_vote_scores(candidates: Sequence[Dict[str, Any]]) -> List[float]:
    """Compute weighted pairwise token-LCS MBR vote scores across TTA candidates.

    Matches the promoted ``ov-36b`` MBR formula: for each view ``i``, average
    ``compute_token_lcs_f1(toks_i, toks_j)`` over all ``j != i`` with weight
    ``1.0`` if view ``j`` passed ``bekern-decode`` else ``0.25``.
    """
    n = len(candidates)
    if n == 0:
        return []
    if n == 1:
        if "mbr_vote" in candidates[0] and "hyp_page_toks" not in candidates[0]:
            return [float(candidates[0]["mbr_vote"])]
        return [0.0]

    has_toks = all("hyp_page_toks" in c for c in candidates)
    if not has_toks:
        return [float(c.get("mbr_vote", 0.0)) for c in candidates]

    scores: List[float] = []
    for i in range(n):
        toks_i = candidates[i]["hyp_page_toks"]
        w_sum = 0.0
        s_sum = 0.0
        for j in range(n):
            if i == j:
                continue
            w = 1.0 if bool(candidates[j].get("bekern_decode_ok", False)) else 0.25
            sim = compute_token_lcs_f1(toks_i, candidates[j]["hyp_page_toks"])
            s_sum += w * sim
            w_sum += w
        scores.append(s_sum / max(1e-6, w_sum))
    return scores


def select_best_fsa_beat_view(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Select the winning scale-TTA view using ``best_fsa_beat``.

    Ranking priority (higher is better):
      1. ``bekern_decode_ok`` (valid Rust bekern parse beats failed parse)
      2. ``not has_loop`` (non-looping decode beats degenerate token loop)
      3. ``fsa_beat_score`` (exact FSA beat validity + EOS + meter/voice errors)
      4. ``mean_logprob`` (length-normalized teacher-forced log-prob tiebreak)
      5. Earlier scale in ``--tta-scales`` (deterministic final tiebreak)
    """
    if not candidates:
        raise ValueError("select_best_fsa_beat_view requires at least one candidate")

    best_idx = max(
        range(len(candidates)),
        key=lambda i: (
            1 if bool(candidates[i].get("bekern_decode_ok", False)) else 0,
            0 if bool(candidates[i].get("has_loop", False)) else 1,
            round(float(candidates[i].get("fsa_beat_score", float("-inf"))), 6),
            float(candidates[i].get("mean_logprob", float("-inf"))),
            -i,
        ),
    )
    return candidates[best_idx]


def select_tta_view(
    candidates: Sequence[Dict[str, Any]],
    rerank: str = "fsa_logprob_mbr",
) -> Tuple[Dict[str, Any], List[float], List[float]]:
    """Select the winning multi-scale TTA view using the promoted ``ov-36b`` reranker.

    Supported ``rerank`` modes:
      - ``fsa_logprob_mbr`` (promoted ``ov-36b`` default):
        ``0.5 * fsa_beat_score + 1.5 * mean_logprob + 1.0 * mbr_vote``
      - ``fsa_logprob``:
        ``fsa_beat_score + 2.0 * mean_logprob``
      - ``mbr``:
        ``mbr_vote`` (weighted pairwise token-LCS consensus)
      - ``fsa_beat``:
        legacy ``select_best_fsa_beat_view`` priority ranking

    Deterministic tie-breaking on equal rounded rerank score:
      1. ``bekern_decode_ok`` (True > False)
      2. ``not has_loop`` (False > True)
      3. ``mean_logprob`` (higher is better)
      4. Earlier scale in ``--tta-scales`` (``-i``)

    Returns ``(winner_dict, mbr_votes, rerank_scores)``.
    """
    if not candidates:
        raise ValueError("select_tta_view requires at least one candidate")

    mode = str(rerank).strip().lower()
    mbr_votes = compute_mbr_vote_scores(candidates)

    if mode == "fsa_beat":
        rerank_scores = [float(c.get("fsa_beat_score", float("-inf"))) for c in candidates]
        return select_best_fsa_beat_view(candidates), mbr_votes, rerank_scores

    if mode == "fsa_logprob_mbr":
        rerank_scores = [
            0.5 * float(c.get("fsa_beat_score", float("-inf")))
            + 1.5 * float(c.get("mean_logprob", -10.0))
            + 1.0 * float(mbr_votes[i])
            for i, c in enumerate(candidates)
        ]
    elif mode == "fsa_logprob":
        rerank_scores = [
            float(c.get("fsa_beat_score", float("-inf")))
            + 2.0 * float(c.get("mean_logprob", -10.0))
            for c in candidates
        ]
    elif mode == "mbr":
        rerank_scores = [float(mbr_votes[i]) for i in range(len(candidates))]
    else:
        raise ValueError(
            f"Unsupported --tta-rerank={rerank!r} (expected one of {TTA_RERANK_MODES})"
        )

    best_idx = max(
        range(len(candidates)),
        key=lambda i: (
            round(float(rerank_scores[i]), 6),
            1 if bool(candidates[i].get("bekern_decode_ok", False)) else 0,
            0 if bool(candidates[i].get("has_loop", False)) else 1,
            float(candidates[i].get("mean_logprob", float("-inf"))),
            -i,
        ),
    )
    return candidates[best_idx], mbr_votes, rerank_scores


# ---------------------------------------------------------------------------
# 4. Full Suite31+HW Evaluation Loop & Reporting
# ---------------------------------------------------------------------------

def _summarize_slice(rows: Sequence[Dict[str, Any]], domain: str) -> Dict[str, Any]:
    """Aggregate Token & Note metrics for one domain ('clean' or 'photo')."""
    def _mean(vals: List[Optional[float]]) -> Optional[float]:
        xs = [float(v) for v in vals if v is not None]
        return round(sum(xs) / len(xs), 6) if xs else None

    def _med(vals: List[Optional[float]]) -> Optional[float]:
        xs = [float(v) for v in vals if v is not None]
        return round(float(statistics.median(xs)), 2) if xs else None

    ok = [r for r in rows if not r.get("skipped")]
    p = domain + "_"
    n = len(ok)
    dec_ok = sum(1 for r in ok if r.get(p + "decode_ok"))
    draft_prop = sum(int(r.get(p + "draft_proposed", 0) or 0) for r in ok)
    draft_acc = sum(int(r.get(p + "draft_accepted", 0) or 0) for r in ok)

    mbeam_trig = sum(int(r.get(p + "measure_beam_triggers", 0) or 0) for r in ok)
    mbeam_rep = sum(int(r.get(p + "measure_beam_repaired", 0) or 0) for r in ok)

    return {
        "n_pieces": len(rows),
        "n_scored": n,
        "tok_f1_mean": _mean([r.get(p + "tok_f1") for r in ok]),
        "tok_lcs_mean": _mean([r.get(p + "tok_lcs") for r in ok]),
        "tok_ser_mean": _mean([r.get(p + "tok_ser") for r in ok]),
        "note_f1_mean": _mean([r.get(p + "note_f1") for r in ok]),
        "note_ser_mean": _mean([r.get(p + "note_ser") for r in ok]),
        "note_hr_ratio_mean": _mean([r.get(p + "note_hr_ratio") for r in ok]),
        "note_hr_ratio_median": _med([r.get(p + "note_hr_ratio") for r in ok]),
        "fixes_median": _med([r.get(p + "proxy_fixes") for r in ok]),
        "decode_ok_rate": round(dec_ok / max(1, n), 4) if n else None,
        "tok_per_sec_mean": _mean([r.get(p + "tok_per_sec") for r in ok]),
        "mtp_acceptance_rate": round(draft_acc / draft_prop, 4) if draft_prop > 0 else None,
        "measure_beam_triggers": mbeam_trig,
        "measure_beam_repaired": mbeam_rep,
    }


def _decode_single_view(
    model: Any,
    cfg: Any,
    img_t: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    use_bf16: bool,
    beam_size: int,
    beam_length_penalty: float,
    measure_beam_size: int,
    measure_beam_cap: Optional[int] = 4,
    measure_beam_lp: Optional[float] = 2.0,
    measure_beam_max_window: int = 64,
    compute_logprob: bool = False,
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]], float, Optional[float]]:
    """Decode one ``(1, 3, H, W)`` image tensor using the configured decoder mode."""
    pad_id, _, _, _, _, _ = _vocab_spec(int(cfg.vocab_size))
    t0 = time.perf_counter()
    mtp_stats: Optional[Dict[str, Any]] = None
    mean_lp: Optional[float] = None
    with torch.no_grad():
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else _nullcontext()
        )
        with ctx:
            if beam_size > 1:
                seq = generate_beam(
                    model,
                    img_t,
                    max_len=args.max_len,
                    beam_size=beam_size,
                    length_penalty=beam_length_penalty,
                    use_grammar=args.use_grammar,
                    strict_grammar=args.strict_grammar,
                )
            elif args.use_mtp_spec or measure_beam_size > 1:
                seq, mtp_stats = generate_mtp_spec(
                    model,
                    img_t,
                    max_len=args.max_len,
                    use_grammar=args.use_grammar,
                    strict_grammar=args.strict_grammar,
                    return_stats=True,
                    measure_beam_size=measure_beam_size,
                    max_bar_toks=int(measure_beam_max_window),
                    measure_beam_cap=measure_beam_cap,
                    measure_beam_lp=measure_beam_lp,
                )
            elif args.use_grammar:
                seq = generate_constrained(
                    model,
                    img_t,
                    max_len=args.max_len,
                    strict=args.strict_grammar,
                )
            elif int(cfg.vocab_size) != uni_omr.VOCAB_SIZE:
                seq = generate_mtp_spec(
                    model,
                    img_t,
                    max_len=args.max_len,
                    use_grammar=False,
                    return_stats=False,
                )
            else:
                seq = model.generate(img_t, max_len=args.max_len)
            if compute_logprob:
                mean_lp = compute_sequence_mean_logprob(model, img_t, seq, pad_id=pad_id)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(1e-6, time.perf_counter() - t0)
    return seq, mtp_stats, elapsed, mean_lp


def evaluate_suite31(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)
    model, cfg, ckpt_meta = load_uni_checkpoint(args.ckpt, device=device)
    _, pieces = resolve_manifest_pieces(args.manifest)

    if args.pieces:
        wanted = {s.strip() for s in args.pieces.split(",") if s.strip()}
        pieces = [p for p in pieces if p["slug"] in wanted]
    if args.limit_pieces is not None:
        pieces = pieces[: int(args.limit_pieces)]

    use_bf16 = bool(getattr(args, "bf16", False) and device.type == "cuda")
    beam_size = int(getattr(args, "beam_size", 1))
    if beam_size < 1:
        raise ValueError(f"--beam-size must be >= 1, got {beam_size}")
    beam_length_penalty = float(getattr(args, "beam_length_penalty", 0.6))
    measure_beam_size = int(getattr(args, "measure_beam_size", 1))
    if measure_beam_size < 0:
        raise ValueError(f"--measure-beam-size must be >= 0, got {measure_beam_size}")
    measure_beam_cap = int(getattr(args, "measure_beam_cap", 4))
    measure_beam_lp = float(getattr(args, "measure_beam_lp", 2.0))
    measure_beam_max_window = int(getattr(args, "measure_beam_max_window", 64))
    use_oemer_dewarp = bool(getattr(args, "oemer_dewarp", False))
    use_unitsize_norm = bool(getattr(args, "unitsize_norm", False))
    max_keep_ratio_override = getattr(args, "max_keep_ratio_override", None)
    tta_scales = parse_tta_scales(getattr(args, "tta_scales", None))
    tta_rerank = str(getattr(args, "tta_rerank", "fsa_logprob_mbr")).strip().lower()
    if tta_scales and tta_rerank not in TTA_RERANK_MODES:
        raise ValueError(
            f"--tta-rerank must be one of {TTA_RERANK_MODES}, got {tta_rerank!r}"
        )

    if beam_size > 1:
        mode_tag = (
            f"beam{beam_size}(lp={beam_length_penalty:g})"
            + ("+grammar" if args.use_grammar else "")
            + ("(strict)" if (args.use_grammar and args.strict_grammar) else "")
        )
    else:
        mode_tag = (
            ("mtp-spec" if (args.use_mtp_spec or measure_beam_size > 1) else "greedy")
            + ("+grammar" if args.use_grammar else "")
            + ("(strict)" if (args.use_grammar and args.strict_grammar) else "")
            + (
                f"+mbeam{measure_beam_size}(cap={measure_beam_cap},lp={measure_beam_lp:g},w={measure_beam_max_window})"
                if measure_beam_size > 1
                else ""
            )
        )
    if use_oemer_dewarp:
        mode_tag += "+dewarp"
    if use_unitsize_norm:
        mode_tag += "+unitsize"
    if tta_scales:
        mode_tag += f"+tta({','.join(str(s) for s in tta_scales)}:{tta_rerank})"
    mkr_str = (
        f" | max_keep_ratio_override={float(max_keep_ratio_override):g}"
        if max_keep_ratio_override is not None
        else ""
    )
    dim_str = (
        f"tta_scales={','.join(str(s) for s in tta_scales)}"
        if tta_scales
        else f"max_dim={args.max_dim}"
    )
    print(
        f"[UniOMR suite31_hw] ckpt={args.ckpt} (arch={ckpt_meta['arch']}, step={ckpt_meta['step']}) | "
        f"device={device} | mode={mode_tag} | {dim_str} | "
        f"max_len={args.max_len} | pieces={len(pieces)}{mkr_str}",
        flush=True,
    )

    workdir = Path(tempfile.mkdtemp(prefix="uni_s31_"))
    piece_rows: List[Dict[str, Any]] = []
    _, _, eos_id, _, _, _ = _vocab_spec(int(cfg.vocab_size))

    with override_max_keep_ratio(model, max_keep_ratio_override):
        for idx, piece in enumerate(pieces, start=1):
            slug = piece["slug"]
            is_hw = piece["handwritten"]
            tag = "HW" if is_hw else "PR"
            prow: Dict[str, Any] = {
                "piece": slug,
                "handwritten": is_hw,
                "stratum": piece["stratum"],
            }
            try:
                ref_page_toks, page_ref_notes = esg.suite31_page_refs(
                    piece["gt"], "uni-" + slug, workdir
                )
                prow["ref_tokens"] = len(ref_page_toks)
            except Exception as e:  # noqa: BLE001
                prow["skipped"] = True
                prow["reason"] = f"ref-error: {str(e)[:160]}"
                piece_rows.append(prow)
                print(f"[{idx:02d}/{len(pieces):02d}] {slug:34s} [{tag}] SKIP ref-error: {e}", flush=True)
                continue

            for domain in ("clean", "photo"):
                img_path: Path = piece[domain]
                tta_selected_scale: Optional[int] = None
                tta_views_summary: Optional[List[Dict[str, Any]]] = None
                pre_meta: Dict[str, Any] = {}

                with Image.open(img_path) as raw_im:
                    work_rgb = raw_im.convert("RGB")
                if use_oemer_dewarp:
                    work_rgb, dw_meta = preprocess_dewarp.oemer_staffline_vertical_dewarp(work_rgb)
                    pre_meta.update(dw_meta)

                if not tta_scales:
                    if use_unitsize_norm:
                        target_dim, us_meta = preprocess_dewarp.estimate_unitsize_max_dim(
                            work_rgb, base_max_dim=int(args.max_dim)
                        )
                        pre_meta.update(us_meta)
                    else:
                        target_dim = int(args.max_dim)
                    pil_im = uni_omr._resize_fullpage_pil(work_rgb, target_dim)
                    img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
                    seq, mtp_stats, elapsed, _ = _decode_single_view(
                        model=model,
                        cfg=cfg,
                        img_t=img_t,
                        args=args,
                        device=device,
                        use_bf16=use_bf16,
                        beam_size=beam_size,
                        beam_length_penalty=beam_length_penalty,
                        measure_beam_size=measure_beam_size,
                        measure_beam_cap=measure_beam_cap,
                        measure_beam_lp=measure_beam_lp,
                        measure_beam_max_window=measure_beam_max_window,
                        compute_logprob=False,
                    )
                    seq_ids = seq[0].detach().cpu().tolist()
                    pred_raw_toks = decode_uni_compat(
                        seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True
                    )
                    if eos_id not in seq_ids[1:]:
                        pred_raw_toks = trim_truncated_tokens(pred_raw_toks)
                    hyp_page_toks, n_sys = stitch_uni_tokens(pred_raw_toks)

                    hyp_txt = workdir / f"uni-{slug}-{domain}-hyp-bekern.txt"
                    hyp_txt.write_text(" ".join(hyp_page_toks) + "\n", encoding="utf-8")
                    hyp_notes = workdir / f"uni-{slug}-{domain}-hyp-notes.json"
                    p_dec = esg.run_cli([
                        esg.ASER, "bekern-decode", "--input", str(hyp_txt), "--out", str(hyp_notes),
                    ])
                    bekern_decode_ok = (p_dec.returncode == 0) and hyp_notes.exists()
                    dec_rc = int(p_dec.returncode)
                else:
                    view_candidates: List[Dict[str, Any]] = []
                    total_elapsed = 0.0
                    for scale_dim in tta_scales:
                        pil_im = uni_omr._resize_fullpage_pil(work_rgb, scale_dim)
                        img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
                        v_seq, v_mtp_stats, v_elapsed, v_mean_lp = _decode_single_view(
                            model=model,
                            cfg=cfg,
                            img_t=img_t,
                            args=args,
                            device=device,
                            use_bf16=use_bf16,
                            beam_size=beam_size,
                            beam_length_penalty=beam_length_penalty,
                            measure_beam_size=measure_beam_size,
                            measure_beam_cap=measure_beam_cap,
                            measure_beam_lp=measure_beam_lp,
                            measure_beam_max_window=measure_beam_max_window,
                            compute_logprob=True,
                        )
                        total_elapsed += v_elapsed
                        v_seq_ids = v_seq[0].detach().cpu().tolist()
                        v_hit_eos = eos_id in v_seq_ids[1:]
                        v_pred_raw = decode_uni_compat(
                            v_seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True
                        )
                        if not v_hit_eos:
                            v_pred_raw = trim_truncated_tokens(v_pred_raw)
                        v_hyp_page_toks, v_n_sys = stitch_uni_tokens(v_pred_raw)

                        v_hyp_txt = workdir / f"uni-{slug}-{domain}-s{scale_dim}-hyp-bekern.txt"
                        v_hyp_txt.write_text(" ".join(v_hyp_page_toks) + "\n", encoding="utf-8")
                        v_hyp_notes = workdir / f"uni-{slug}-{domain}-s{scale_dim}-hyp-notes.json"
                        v_dec = esg.run_cli([
                            esg.ASER, "bekern-decode", "--input", str(v_hyp_txt), "--out", str(v_hyp_notes),
                        ])
                        v_bekern_ok = (v_dec.returncode == 0) and v_hyp_notes.exists()
                        v_fsa = compute_fsa_beat_metrics(
                            v_hyp_page_toks, hit_eos=v_hit_eos, bekern_decode_ok=v_bekern_ok
                        )
                        view_candidates.append({
                            "scale": int(scale_dim),
                            "pred_raw_toks": v_pred_raw,
                            "hyp_page_toks": v_hyp_page_toks,
                            "n_sys": v_n_sys,
                            "hyp_notes": v_hyp_notes,
                            "bekern_decode_ok": v_bekern_ok,
                            "dec_rc": int(v_dec.returncode),
                            "hit_eos": v_hit_eos,
                            "mtp_stats": v_mtp_stats,
                            "elapsed_secs": v_elapsed,
                            "mean_logprob": float(v_mean_lp) if v_mean_lp is not None else -10.0,
                            **v_fsa,
                        })

                    winner, mbr_votes, rerank_scores = select_tta_view(
                        view_candidates, rerank=tta_rerank
                    )
                    pred_raw_toks = winner["pred_raw_toks"]
                    hyp_page_toks = winner["hyp_page_toks"]
                    n_sys = int(winner["n_sys"])
                    hyp_notes = winner["hyp_notes"]
                    bekern_decode_ok = bool(winner["bekern_decode_ok"])
                    dec_rc = int(winner["dec_rc"])
                    mtp_stats = winner["mtp_stats"]
                    elapsed = max(1e-6, total_elapsed)
                    tta_selected_scale = int(winner["scale"])
                    tta_views_summary = [
                        {
                            "scale": c["scale"],
                            "fsa_beat_score": c["fsa_beat_score"],
                            "mean_logprob": round(float(c["mean_logprob"]), 6),
                            "mbr_vote": round(float(mbr_votes[i]), 6),
                            "tta_rerank_score": round(float(rerank_scores[i]), 6),
                            "beat_conservation_rate": c["beat_conservation_rate"],
                            "has_loop": c["has_loop"],
                            "hit_eos": c["hit_eos"],
                            "bekern_decode_ok": c["bekern_decode_ok"],
                            "n_pred_toks": len(c["hyp_page_toks"]),
                            "elapsed_secs": round(float(c["elapsed_secs"]), 4),
                        }
                        for i, c in enumerate(view_candidates)
                    ]

                # 1. Token-level F1, ordered TokLCS F1, canonical voice-major TokLCS & SER
                tok_edits = uni_omr.levenshtein_distance(ref_page_toks, hyp_page_toks)
                tok_ser = tok_edits / max(1, len(ref_page_toks))
                _, _, tok_f1 = uni_omr.compute_token_f1(ref_page_toks, hyp_page_toks)
                tok_lcs = compute_token_lcs_f1(ref_page_toks, hyp_page_toks)
                canonical_tok_lcs = compute_token_lcs_f1(
                    grammar.canonicalize_measure_voices(ref_page_toks),
                    grammar.canonicalize_measure_voices(hyp_page_toks),
                )

                # 2. Note-level F1 & SER via assembly-serialize + eval-harness
                ref_notes_cnt: Optional[int] = None
                hyp_notes_cnt: Optional[int] = None
                note_hr_ratio: Optional[float] = None
                if not bekern_decode_ok:
                    note_f1 = 0.0
                    note_ser = 1.0
                    proxy_fixes = None
                    decode_ok = False
                    zero_reason = f"bekern-decode-rc-{dec_rc}"
                else:
                    sc = esg.score_notes_pair(
                        page_ref_notes,
                        hyp_notes,
                        workdir,
                        f"uni-{slug}-{domain}",
                        esg.EHARNESS,
                        esg.ASER,
                    )
                    note_f1 = float(sc.get("f1", 0.0))
                    note_ser = float(sc.get("ser", 1.0))
                    proxy_fixes = sc.get("proxy_fixes")
                    decode_ok = not bool(sc.get("zeroed"))
                    zero_reason = sc.get("zero_reason")
                    ref_notes_cnt = sc.get("ref_len")
                    hyp_notes_cnt = sc.get("hyp_len")
                    if ref_notes_cnt is not None and ref_notes_cnt > 0 and hyp_notes_cnt is not None:
                        note_hr_ratio = float(hyp_notes_cnt) / float(ref_notes_cnt)

                n_gen = max(1, len(pred_raw_toks) + 1)
                tok_per_sec = n_gen / elapsed

                pfx = domain + "_"
                prow[pfx + "tok_f1"] = round(tok_f1, 6)
                prow[pfx + "tok_lcs"] = round(tok_lcs, 6)
                prow[pfx + "canonical_tok_lcs"] = round(canonical_tok_lcs, 6)
                prow[pfx + "tok_ser"] = round(tok_ser, 6)
                prow[pfx + "note_f1"] = round(note_f1, 6)
                prow[pfx + "note_ser"] = round(note_ser, 6)
                prow[pfx + "ref_notes"] = ref_notes_cnt
                prow[pfx + "hyp_notes"] = hyp_notes_cnt
                prow[pfx + "note_hr_ratio"] = round(note_hr_ratio, 4) if note_hr_ratio is not None else None
                prow[pfx + "proxy_fixes"] = proxy_fixes
                prow[pfx + "decode_ok"] = decode_ok
                prow[pfx + "zero_reason"] = zero_reason
                prow[pfx + "n_sys"] = n_sys
                prow[pfx + "n_pred_toks"] = len(hyp_page_toks)
                prow[pfx + "elapsed_secs"] = round(elapsed, 4)
                prow[pfx + "tok_per_sec"] = round(tok_per_sec, 2)
                if pre_meta:
                    prow[pfx + "preprocess"] = pre_meta
                if tta_selected_scale is not None:
                    prow[pfx + "tta_selected_scale"] = tta_selected_scale
                    prow[pfx + "tta_views"] = tta_views_summary
                if mtp_stats is not None:
                    prow[pfx + "draft_proposed"] = mtp_stats["draft_proposed"]
                    prow[pfx + "draft_accepted"] = mtp_stats["draft_accepted"]
                    prow[pfx + "mtp_acceptance_rate"] = mtp_stats["acceptance_rate"]
                    prow[pfx + "measure_beam_triggers"] = mtp_stats.get("measure_beam_triggers", 0)
                    prow[pfx + "measure_beam_repaired"] = mtp_stats.get("measure_beam_repaired", 0)

                mtp_str = (
                    f" mtp_acc={mtp_stats['acceptance_rate']:.2f}"
                    if mtp_stats is not None
                    else ""
                )
                mb_str = (
                    f" mb={mtp_stats.get('measure_beam_repaired', 0)}/{mtp_stats.get('measure_beam_triggers', 0)}"
                    if (mtp_stats is not None and measure_beam_size > 1)
                    else ""
                )
                tta_str = f" tta={tta_selected_scale}" if tta_selected_scale is not None else ""
                hr_str = f" H/R={note_hr_ratio:.2f}" if note_hr_ratio is not None else ""
                dec_str = "OK" if decode_ok else f"FAIL({zero_reason})"
                print(
                    f"[{idx:02d}/{len(pieces):02d}] {slug:34s} [{tag}|{domain:5s}] "
                    f"tokF1={tok_f1:.3f} tokLCS={tok_lcs:.3f} tokSER={tok_ser:.3f} | "
                    f"noteF1={note_f1:.3f} noteSER={note_ser:.3f}{hr_str} ({dec_str}) | "
                    f"sys={n_sys} tok={len(hyp_page_toks):4d} {tok_per_sec:5.1f}tok/s{mtp_str}{mb_str}{tta_str}",
                    flush=True,
                )

            piece_rows.append(prow)

    printed_rows = [r for r in piece_rows if not r["handwritten"]]
    hw_rows = [r for r in piece_rows if r["handwritten"]]

    summary = {
        "printed_clean": _summarize_slice(printed_rows, "clean"),
        "printed_photo": _summarize_slice(printed_rows, "photo"),
        "hw_clean": _summarize_slice(hw_rows, "clean"),
        "hw_photo": _summarize_slice(hw_rows, "photo"),
        "all45_clean": _summarize_slice(piece_rows, "clean"),
        "all45_photo": _summarize_slice(piece_rows, "photo"),
    }

    _print_summary_table(summary, mode_tag=mode_tag)

    report = {
        "ckpt": str(args.ckpt),
        "ckpt_meta": ckpt_meta,
        "model_config": asdict(cfg),
        "eval_args": {
            "manifest": str(args.manifest),
            "max_dim": int(args.max_dim),
            "max_len": int(args.max_len),
            "use_grammar": bool(args.use_grammar),
            "strict_grammar": bool(args.strict_grammar),
            "use_mtp_spec": bool(args.use_mtp_spec),
            "beam_size": beam_size,
            "beam_length_penalty": beam_length_penalty,
            "measure_beam_size": measure_beam_size,
            "measure_beam_cap": measure_beam_cap,
            "measure_beam_lp": measure_beam_lp,
            "measure_beam_max_window": measure_beam_max_window,
            "oemer_dewarp": use_oemer_dewarp,
            "unitsize_norm": use_unitsize_norm,
            "max_keep_ratio_override": (
                float(max_keep_ratio_override)
                if max_keep_ratio_override is not None
                else None
            ),
            "tta_scales": tta_scales if tta_scales else None,
            "tta_rerank": tta_rerank if tta_scales else None,
            "tta_selector": tta_rerank if tta_scales else None,
            "device": str(device),
        },
        "summary": summary,
        "pieces": piece_rows,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote JSON report -> {out_path}", flush=True)
    return report


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> bool:
        return False


def _fmt_f(val: Optional[float], width: int = 6, prec: int = 4) -> str:
    if val is None:
        return f"{'n/a':>{width}s}"
    return f"{val:{width}.{prec}f}"


def _print_summary_table(summary: Dict[str, Dict[str, Any]], mode_tag: str) -> None:
    print("=" * 112, flush=True)
    print(f"UniOMR Suite31+HW Summary ({mode_tag})", flush=True)
    print("=" * 112, flush=True)
    print(
        f"{'Slice':15s} | {'N':>3s} | {'Tok F1':>7s} | {'TokLCS':>7s} | {'Tok SER':>7s} | "
        f"{'Note F1':>7s} | {'Note SER':>8s} | {'H/R':>6s} | {'DecOK':>6s} | {'Tok/s':>6s} | {'MTP Acc':>7s}",
        flush=True,
    )
    print("-" * 112, flush=True)
    for key in (
        "printed_clean",
        "printed_photo",
        "hw_clean",
        "hw_photo",
        "all45_clean",
        "all45_photo",
    ):
        s = summary[key]
        dec_pct = f"{s['decode_ok_rate'] * 100.0:5.1f}%" if s["decode_ok_rate"] is not None else "   n/a"
        print(
            f"{key:15s} | {s['n_scored']:3d} | {_fmt_f(s['tok_f1_mean'], 7, 4)} | "
            f"{_fmt_f(s.get('tok_lcs_mean'), 7, 4)} | {_fmt_f(s['tok_ser_mean'], 7, 4)} | "
            f"{_fmt_f(s['note_f1_mean'], 7, 4)} | {_fmt_f(s['note_ser_mean'], 8, 4)} | "
            f"{_fmt_f(s.get('note_hr_ratio_mean'), 6, 3)} | {dec_pct:>6s} | "
            f"{_fmt_f(s['tok_per_sec_mean'], 6, 1)} | {_fmt_f(s['mtp_acceptance_rate'], 7, 3)}",
            flush=True,
        )
    print("=" * 112, flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate UniOMR on Suite31+HW (45 full pages)")
    ap.add_argument("--ckpt", required=True, help="Path to UniOMR .pt checkpoint")
    ap.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to suite31_hw_manifest.json",
    )
    ap.add_argument(
        "--max-dim",
        type=int,
        default=1408,
        help="Full-page resize max dimension (ov-35b universal default: 1408)",
    )
    ap.add_argument(
        "--max-len",
        type=int,
        default=3584,
        help="Max autoregressive decode tokens (ov-35b universal default: 3584)",
    )
    ap.add_argument("--use-grammar", action="store_true", help="Enable FSA grammar-constrained decoding")
    ap.add_argument("--strict-grammar", action="store_true", help="Enable strict bar-duration grammar rules")
    ap.add_argument("--use-mtp-spec", action="store_true", help="Enable MTP-4 self-speculative decoding")
    ap.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Beam search width (1 = existing greedy/MTP-4 fast path; >1 = KV-cached beam search)",
    )
    ap.add_argument(
        "--beam-length-penalty",
        type=float,
        default=0.6,
        help="Length normalization exponent for beam search (score / len**alpha)",
    )
    ap.add_argument(
        "--measure-beam-size",
        type=int,
        default=1,
        help="Violation-triggered measure-scoped beam search width (1 = fast default off; 2 = promoted ov-34b budget)",
    )
    ap.add_argument(
        "--measure-beam-cap",
        type=int,
        default=4,
        help="Max measure-beam repair triggers per page (ov-34b default: 4)",
    )
    ap.add_argument(
        "--measure-beam-lp",
        type=float,
        default=2.0,
        help="Max log-prob drop guardrail for measure-beam repair vs best-likelihood closed bar (ov-34b default: 2.0)",
    )
    ap.add_argument(
        "--measure-beam-max-window",
        type=int,
        default=64,
        help="Max tokens per measure-beam repair window (ov-34b default: 64)",
    )
    ap.add_argument(
        "--oemer-dewarp",
        action="store_true",
        help="Opt-in OEMER staffline-aware vertical dewarping (i13-10; auto-skips flat pages < 1.5 px)",
    )
    ap.add_argument(
        "--unitsize-norm",
        action="store_true",
        help="Opt-in Homr unit_size inter-staffline spacing adaptive max_dim normalization (i13-10)",
    )
    ap.add_argument(
        "--max-keep-ratio-override",
        type=float,
        default=None,
        help="Temporarily override model.router.max_keep_ratio at eval time (e.g. 0.55 or 0.65)",
    )
    ap.add_argument(
        "--tta-scales",
        type=str,
        default="",
        help=(
            "Opt-in comma-separated clean scale-only Multi-View TTA max dimensions "
            "(e.g. 1280,1408,1536 or 1024,1152,1280,1408,1536). Empty string = disabled "
            "(default single-scale --max-dim unchanged)."
        ),
    )
    ap.add_argument(
        "--tta-rerank",
        type=str,
        default="fsa_logprob_mbr",
        choices=TTA_RERANK_MODES,
        help=(
            "Multi-scale TTA candidate reranker when --tta-scales is non-empty "
            "(promoted ov-36b default: fsa_logprob_mbr = 0.5*fsa + 1.5*mean_logprob + 1.0*mbr)."
        ),
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
        help="Inference device (cuda or cpu)",
    )
    ap.add_argument("--bf16", action="store_true", help="Use bfloat16 autocast on CUDA")
    ap.add_argument("--out", default=None, help="Optional output JSON report path")
    ap.add_argument("--limit-pieces", type=int, default=None, help="Cap number of pieces (smoke test)")
    ap.add_argument("--pieces", default=None, help="Comma-separated slugs to evaluate")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    evaluate_suite31(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
