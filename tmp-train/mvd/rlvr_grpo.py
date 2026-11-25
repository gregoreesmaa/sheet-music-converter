#!/usr/bin/env python3
"""RLVR (GRPO Verifiable-Reward Post-Training) for UniOMR v1 & v2.

Inspired by olmOCR 2 / LightOnOCR-2 / InternVL 3.5 (2025-2026) and
DeepSeekMath / DeepSeek-R1 GRPO:

1. Exact Verifiable OMR Reward (``compute_omr_rlvr_reward``) in ``[0, 1]``:
   - ``r_parse``  (0.20): valid ``bekern-decode`` / ``PrefixState(strict=False)``
     parse with ``>= 1`` pitched note.
   - ``r_sys``    (0.15): exact ``<sys>`` count match (``1.0`` exact, ``0.5``
     off-by-1, ``0.0`` otherwise; ``0.0`` on empty output).
   - ``r_meter``  (0.15): fraction of interior measures obeying bar-duration
     conservation (``|vcurs - tsig| < 1e-3`` across all active voices).
   - ``r_noloop`` (0.15): ``1.0`` if no 8-16-token repetition loop (``>= 4x``),
     else ``0.0`` (``0.0`` on empty output).
   - ``r_note``   (0.35): ``0.5 * TokLCS_F1 + 0.5 * ExactPitchDur_F1`` against
     ``ref_bekern``.

2. GRPO Group-Relative Advantage & Clipped Policy-Gradient Loss:
   - Samples ``G`` rollouts (default ``G=4``) per page with temperature/top-p
     under ``grammar.PrefixState`` (single encoder pass per page, batched
     KV-cached decoding).
   - Group-normalized advantages: ``A_i = (R_i - mean(R)) / (std(R) + 1e-6)``.
   - Clipped GRPO surrogate + unbiased ``k3`` KL-to-reference-policy penalty
     (default ``--kl-beta 0.02``).
   - Architecture-agnostic across both ``UniOMRNet`` (v1) and ``UniOMRv2Net`` (v2)
     with zero edits to ``uni_omr.py`` or ``uni_omr_v2.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import difflib
import inspect
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import eval_uni_suite31 as eval_s31  # noqa: E402
import grammar  # noqa: E402
import uni_omr  # noqa: E402


# ---------------------------------------------------------------------------
# Reward Weights (sum to 1.00)
# ---------------------------------------------------------------------------

W_PARSE: float = 0.20
W_SYS: float = 0.15
W_METER: float = 0.15
W_NOLOOP: float = 0.15
W_NOTE: float = 0.35

METER_TOL: float = 1e-3
LOOP_MIN_PERIOD: int = 8
LOOP_MAX_PERIOD: int = 16
LOOP_MIN_REPEATS: int = 4


# ---------------------------------------------------------------------------
# Verifiable OMR Reward Helpers
# ---------------------------------------------------------------------------

def _as_tokens(seq: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(seq, str):
        return seq.split()
    return [str(t) for t in seq]


def has_repetition_loop(
    tokens: Sequence[str],
    min_period: int = LOOP_MIN_PERIOD,
    max_period: int = LOOP_MAX_PERIOD,
    min_repeats: int = LOOP_MIN_REPEATS,
) -> bool:
    """Return True if ``tokens`` contains an adjacent ``>= min_repeats`` loop of period ``min_period..max_period``."""
    n = len(tokens)
    if n < min_period * min_repeats:
        return False
    for p in range(int(min_period), int(max_period) + 1):
        target_run = (int(min_repeats) - 1) * p
        if n < p + target_run:
            continue
        run = 0
        for j in range(p, n):
            if tokens[j] == tokens[j - p]:
                run += 1
                if run >= target_run:
                    return True
            else:
                run = 0
    return False


def extract_pitch_dur_notes(tokens: Sequence[str]) -> List[Tuple[str, int, str, str, str]]:
    """Extract ordered ``(dur, n_dots, step, octave, alter)`` tuples for all pitched notes.

    Handles ``rest`` (skipped), ``chord``, ``grace``, ``<sys>``, accidentals
    (``a-2``..``a2``), and augmentation ``dot`` tokens in O(N) time.
    """
    notes: List[Tuple[str, int, str, str, str]] = []
    n = len(tokens)
    i = 0
    while i < n:
        tok = tokens[i]
        if tok == grammar.REST:
            # Skip rest + its duration (+ any trailing dots/mods)
            i += 1
            if i < n and tokens[i] in grammar.DURATIONS:
                i += 1
            while i < n and (tokens[i] == grammar.DOT or tokens[i] in grammar.MODS):
                i += 1
            continue
        if tok in grammar.DURATIONS:
            dur = tok
            if i + 2 < n and tokens[i + 1] in grammar.STEPS and tokens[i + 2] in grammar.OCTAVES:
                step = tokens[i + 1]
                octave = tokens[i + 2]
                alter = ""
                n_dots = 0
                j = i + 3
                while j < n:
                    t_tail = tokens[j]
                    if t_tail in grammar.ALTERS:
                        alter = t_tail
                        j += 1
                    elif t_tail == grammar.DOT:
                        n_dots += 1
                        j += 1
                    elif (
                        t_tail in grammar.MODS
                        or t_tail in (grammar.TIE_START, grammar.TIE_STOP)
                    ):
                        j += 1
                    else:
                        break
                notes.append((dur, n_dots, step, octave, alter))
                i = j
                continue
        i += 1
    return notes


def compute_sequence_f1(ref_items: Sequence[Any], hyp_items: Sequence[Any]) -> float:
    """Ordered LCS F1 (``2 * matches / (|ref| + |hyp|)``) over arbitrary hashable items."""
    if not ref_items and not hyp_items:
        return 1.0
    if not ref_items or not hyp_items:
        return 0.0
    sm = difflib.SequenceMatcher(a=list(ref_items), b=list(hyp_items), autojunk=False)
    matches = sum(block.size for block in sm.get_matching_blocks())
    return 2.0 * float(matches) / float(len(ref_items) + len(hyp_items))


def _check_measure_valid(state: grammar.PrefixState, tol: float = METER_TOL) -> Optional[bool]:
    """Check whether the open measure in ``state`` obeys ``|vcurs - tsig| < tol``.

    Returns ``None`` if no time signature is active or the measure is empty;
    otherwise returns ``True`` iff every active voice matches ``tsig`` within ``tol``.
    """
    if state.tsig is None:
        return None
    eff_vcurs = list(state.vcurs)
    if 0 <= state.seg < len(eff_vcurs):
        eff_vcurs[state.seg] += state._pending_value()
    active = [
        idx
        for idx, v in enumerate(eff_vcurs)
        if v > grammar._BAR_EPS or (idx + 1) in state.opened
    ]
    if not active:
        return None
    return all(abs(eff_vcurs[v] - state.tsig) < tol for v in active)


def _walk_parse_and_meter(
    tokens: Sequence[str],
    tol: float = METER_TOL,
) -> Tuple[bool, int, int, int]:
    """Walk ``tokens`` with ``PrefixState(strict=False)`` across ``<sys>`` boundaries.

    Returns ``(parse_ok, n_pitched_notes, n_valid_interior_bars, n_interior_bars)``.
    """
    if not tokens:
        return False, 0, 0, 0

    st = grammar.PrefixState(strict=False)
    last_tsig: Optional[float] = None
    parse_ok = True
    n_notes = 0
    # Each entry is (is_valid_meter: Optional[bool], closed_by_bar: bool)
    system_bars: List[List[Optional[bool]]] = [[]]

    for tok in tokens:
        if tok == uni_omr.SYS_TOKEN:
            # Current system must end cleanly and be non-empty.
            _, eos_ok = st.allowed()
            after_bar = bool(st.recent) and (st.recent[-1] in grammar.BAR_LIKES)
            if not st.recent or not (eos_ok or after_bar):
                parse_ok = False
                break
            # Record any unclosed trailing measure in this system.
            tail_valid = _check_measure_valid(st, tol=tol)
            if tail_valid is not None:
                system_bars[-1].append(tail_valid)
            if st.tsig is not None:
                last_tsig = st.tsig
            st = grammar.PrefixState(strict=False)
            st.tsig = last_tsig
            system_bars.append([])
            continue

        # Before closing a barline, evaluate bar-duration conservation.
        if tok in grammar.BAR_LIKES:
            bar_valid = _check_measure_valid(st, tol=tol)
            if bar_valid is not None:
                system_bars[-1].append(bar_valid)

        prev_mode = st.mode
        try:
            st.step(tok)
        except ValueError:
            parse_ok = False
            break

        if prev_mode == "step" and st.mode == "pitch-tail":
            n_notes += 1
        if st.tsig is not None:
            last_tsig = st.tsig

    if parse_ok:
        _, eos_ok = st.allowed()
        after_bar = bool(st.recent) and (st.recent[-1] in grammar.BAR_LIKES)
        if not st.recent or not (eos_ok or after_bar):
            parse_ok = False
        else:
            tail_valid = _check_measure_valid(st, tol=tol)
            if tail_valid is not None:
                system_bars[-1].append(tail_valid)

    if not parse_ok:
        return False, n_notes, 0, 0

    # Flatten bars across the page and evaluate interior measures (excluding
    # first pickup measure and final potentially incomplete measure when >= 3 bars).
    all_bars: List[bool] = [
        b for sys_b in system_bars for b in sys_b if b is not None
    ]
    if len(all_bars) >= 3:
        checked = all_bars[1:-1]
    else:
        checked = all_bars

    n_interior = len(checked)
    n_valid = sum(1 for ok in checked if ok)
    return True, n_notes, n_valid, n_interior


def compute_omr_rlvr_reward(
    pred_bekern: Union[str, Sequence[str]],
    ref_bekern: Union[str, Sequence[str]],
    check_decode_fn: Optional[Callable[[Sequence[str]], bool]] = None,
) -> Dict[str, Any]:
    """Compute exact verifiable OMR reward in ``[0, 1]`` for a rollout.

    Combines 5 programmatic sub-rewards:
      - ``r_parse``  (0.20): valid ``PrefixState(strict=False)`` / ``bekern-decode``
        parse with ``>= 1`` pitched note.
      - ``r_sys``    (0.15): exact ``<sys>`` count match (``1.0`` exact, ``0.5``
        off-by-1, ``0.0`` else; ``0.0`` on empty prediction).
      - ``r_meter``  (0.15): fraction of interior measures obeying
        ``|vcurs - tsig| < 1e-3``.
      - ``r_noloop`` (0.15): ``1.0`` if no 8-16-token repetition loop (``>= 4x``),
        else ``0.0`` (``0.0`` on empty prediction).
      - ``r_note``   (0.35): ``0.5 * TokLCS_F1 + 0.5 * ExactPitchDur_F1``.
    """
    pred_toks = _as_tokens(pred_bekern)
    ref_toks = _as_tokens(ref_bekern)

    # 1. Parse + meter walk
    parse_ok, n_notes, n_valid_bars, n_checked_bars = _walk_parse_and_meter(
        pred_toks, tol=METER_TOL
    )
    if parse_ok and check_decode_fn is not None:
        try:
            parse_ok = bool(check_decode_fn(pred_toks))
        except Exception:
            parse_ok = False
    r_parse = 1.0 if (parse_ok and n_notes >= 1) else 0.0

    # 2. <sys> count match
    n_sys_pred = sum(1 for t in pred_toks if t == uni_omr.SYS_TOKEN)
    n_sys_ref = sum(1 for t in ref_toks if t == uni_omr.SYS_TOKEN)
    if not pred_toks:
        r_sys = 0.0
    else:
        sys_diff = abs(n_sys_pred - n_sys_ref)
        if sys_diff == 0:
            r_sys = 1.0
        elif sys_diff == 1:
            r_sys = 0.5
        else:
            r_sys = 0.0

    # 3. Interior measure bar-duration conservation
    if parse_ok and n_checked_bars > 0:
        r_meter = float(n_valid_bars) / float(n_checked_bars)
    else:
        r_meter = 0.0

    # 4. Anti-loop reward (no 8-16 token repetition >= 4x)
    loop_detected = has_repetition_loop(
        pred_toks,
        min_period=LOOP_MIN_PERIOD,
        max_period=LOOP_MAX_PERIOD,
        min_repeats=LOOP_MIN_REPEATS,
    )
    r_noloop = 0.0 if (not pred_toks or loop_detected) else 1.0

    # 5. Note accuracy: 0.5 * TokLCS F1 + 0.5 * exact pitch/duration F1
    tok_lcs_f1 = eval_s31.compute_token_lcs_f1(ref_toks, pred_toks)
    ref_notes = extract_pitch_dur_notes(ref_toks)
    pred_notes = extract_pitch_dur_notes(pred_toks)
    note_f1 = compute_sequence_f1(ref_notes, pred_notes)
    r_note = 0.5 * tok_lcs_f1 + 0.5 * note_f1

    reward = (
        W_PARSE * r_parse
        + W_SYS * r_sys
        + W_METER * r_meter
        + W_NOLOOP * r_noloop
        + W_NOTE * r_note
    )
    reward = float(max(0.0, min(1.0, reward)))

    return {
        "reward": reward,
        "r_parse": float(r_parse),
        "r_sys": float(r_sys),
        "r_meter": float(r_meter),
        "r_noloop": float(r_noloop),
        "r_note": float(r_note),
        "tok_lcs_f1": float(tok_lcs_f1),
        "note_f1": float(note_f1),
        "parse_ok": bool(parse_ok),
        "n_notes": int(n_notes),
        "n_sys_pred": int(n_sys_pred),
        "n_sys_ref": int(n_sys_ref),
        "n_valid_interior_bars": int(n_valid_bars),
        "n_interior_bars": int(n_checked_bars),
        "has_loop": bool(loop_detected),
    }


# ---------------------------------------------------------------------------
# Model Compatibility Layer (UniOMRNet v1 & UniOMRv2Net v2)
# ---------------------------------------------------------------------------

def _model_vocab_size(model: nn.Module) -> int:
    cfg = getattr(model, "cfg", None)
    if cfg is not None and hasattr(cfg, "vocab_size"):
        return int(cfg.vocab_size)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "out_features"):
        return int(lm_head.out_features)
    return int(uni_omr.VOCAB_SIZE)


def _supports_return_aux(model: nn.Module) -> bool:
    sig = inspect.signature(model.decode_tokens)
    return "return_aux" in sig.parameters


def _encode_image_compat(
    model: nn.Module,
    images: torch.Tensor,
    keep_ratio_override: Optional[float] = None,
) -> Any:
    """Encode ``images`` into memory (tensor for v1, DeepStack tuple for v2)."""
    out = model.encode_image(
        images,
        keep_ratio_override=keep_ratio_override,
        return_aux=False,
    )
    return out[0]


def _repeat_memory(memory: Any, repeats: int) -> Any:
    """Repeat encoder memory along batch dim 0 for both single-tensor (v1) and tuple/list (v2 DeepStack)."""
    if isinstance(memory, tuple):
        return tuple(t.repeat_interleave(repeats, dim=0) for t in memory)
    if isinstance(memory, list):
        return [t.repeat_interleave(repeats, dim=0) for t in memory]
    return memory.repeat_interleave(repeats, dim=0)


def _decode_step_compat(
    model: nn.Module,
    cur_ids: torch.Tensor,
    memory: Optional[torch.Tensor],
    past_kvs: Optional[List[Any]],
) -> Tuple[torch.Tensor, List[Any]]:
    """Run one KV-cached decoder step on either ``UniOMRNet`` or ``UniOMRv2Net``."""
    kwargs: Dict[str, Any] = dict(
        memory=memory,
        return_mtp=False,
        past_kvs=past_kvs,
        use_cache=True,
        apply_tok_dropout=False,
    )
    if _supports_return_aux(model):
        kwargs["return_aux"] = False
    out = model.decode_tokens(cur_ids, **kwargs)
    return out[0], out[2]


def _decode_full_compat(
    model: nn.Module,
    tgt_ids: torch.Tensor,
    memory: torch.Tensor,
) -> torch.Tensor:
    """Run full teacher-forced decoder pass on either ``UniOMRNet`` or ``UniOMRv2Net``."""
    kwargs: Dict[str, Any] = dict(
        memory=memory,
        return_mtp=False,
        use_cache=False,
        apply_tok_dropout=False,
    )
    if _supports_return_aux(model):
        kwargs["return_aux"] = False
    out = model.decode_tokens(tgt_ids, **kwargs)
    return out[0]


# ---------------------------------------------------------------------------
# Sampling, Rollouts, Group Advantages, and Clipped GRPO Loss
# ---------------------------------------------------------------------------

def sample_top_p(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> torch.Tensor:
    """Sample token IDs from ``logits`` ``(N, V)`` with temperature & nucleus top-p."""
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    if float(temperature) <= 0.0:
        tok = logits.argmax(dim=-1)
        return tok.squeeze(0) if squeeze else tok

    scaled = logits.float() / float(temperature)
    # Guard rows where all logits are -inf/NaN (fallback to uniform).
    finite_any = torch.isfinite(scaled).any(dim=-1, keepdim=True)
    safe_logits = torch.where(finite_any, scaled, torch.zeros_like(scaled))

    p = float(top_p)
    if 0.0 < p < 1.0:
        sorted_logits, sorted_idx = torch.sort(safe_logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        # Keep the first token that pushes cumulative mass >= top_p.
        remove_mask = (cum_probs - sorted_probs) > p
        remove_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
        probs = F.softmax(sorted_logits, dim=-1)
        sampled_sorted = torch.multinomial(probs, num_samples=1)
        tok = sorted_idx.gather(-1, sampled_sorted).squeeze(-1)
    else:
        probs = F.softmax(safe_logits, dim=-1)
        tok = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return tok.squeeze(0) if squeeze else tok


@dataclass
class RolloutBatch:
    """Container for ``B x G`` sampled rollouts and behavior-policy log-probs."""

    seq_ids: torch.Tensor          # (B, G, L) long, starts with BOS_ID
    old_log_probs: torch.Tensor    # (B, G, L - 1) float32
    action_mask: torch.Tensor      # (B, G, L - 1) float32
    decoded_tokens: List[List[List[str]]]  # [B][G] decoded bekern token lists


@torch.no_grad()
def sample_rollouts(
    model: nn.Module,
    images: torch.Tensor,
    *,
    group_size: int = 4,
    max_len: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    use_grammar: bool = True,
    strict_grammar: bool = False,
    keep_ratio_override: Optional[float] = None,
) -> RolloutBatch:
    """Sample ``group_size`` rollouts per page under ``PrefixState`` with KV caching.

    Works with both ``UniOMRNet`` (v1) and ``UniOMRv2Net`` (v2).
    """
    G = int(group_size)
    if G < 1:
        raise ValueError(f"group_size must be >= 1, got {G}")
    if max_len < 1:
        raise ValueError(f"max_len must be >= 1, got {max_len}")

    was_training = model.training
    model.eval()
    try:
        B = int(images.shape[0])
        device = images.device
        vocab_size = _model_vocab_size(model)
        pad_id, bos_id, eos_id, _, tok_to_id, _ = eval_s31._vocab_spec(vocab_size)

        mem = _encode_image_compat(model, images, keep_ratio_override=keep_ratio_override)
        mem_exp = _repeat_memory(mem, G)  # (B*G, L_mem, D) or tuple thereof

        N = B * G
        cur_ids = torch.full((N, 1), bos_id, dtype=torch.long, device=device)
        past_kvs: Optional[List[Any]] = None
        finished = torch.zeros(N, dtype=torch.bool, device=device)

        states: Optional[List[grammar.PrefixState]] = None
        if use_grammar:
            states = [
                grammar.PrefixState(strict=strict_grammar, vocab_map=tok_to_id)
                for _ in range(N)
            ]

        out_tokens: List[torch.Tensor] = [cur_ids]
        old_lp_steps: List[torch.Tensor] = []
        mask_steps: List[torch.Tensor] = []

        for _ in range(int(max_len)):
            logits, past_kvs = _decode_step_compat(
                model,
                cur_ids,
                memory=mem_exp if past_kvs is None else None,
                past_kvs=past_kvs,
            )
            raw_step = logits[:, -1, :].float()  # (N, V)
            raw_log_probs = F.log_softmax(raw_step, dim=-1)

            if states is not None:
                masked_step = torch.full_like(raw_step, float("-inf"))
                for idx in range(N):
                    if bool(finished[idx]):
                        masked_step[idx, pad_id] = 0.0
                        continue
                    allowed = eval_s31._allowed_ids_for_state(states[idx], vocab_size)
                    if allowed:
                        idx_t = torch.tensor(allowed, dtype=torch.long, device=device)
                        masked_step[idx, idx_t] = raw_step[idx, idx_t]
                    else:
                        masked_step[idx] = raw_step[idx]
                sample_logits = masked_step
            else:
                sample_logits = raw_step.clone()
                sample_logits[:, pad_id] = float("-inf")
                sample_logits[:, bos_id] = float("-inf")

            next_tok = sample_top_p(sample_logits, temperature=temperature, top_p=top_p)
            active = ~finished
            next_tok = torch.where(active, next_tok, torch.full_like(next_tok, pad_id))

            step_lp = raw_log_probs.gather(-1, next_tok.unsqueeze(-1)).squeeze(-1)
            step_lp = torch.where(active, step_lp, torch.zeros_like(step_lp))

            out_tokens.append(next_tok.unsqueeze(1))
            old_lp_steps.append(step_lp.unsqueeze(1))
            mask_steps.append(active.to(torch.float32).unsqueeze(1))

            if states is not None:
                for idx in range(N):
                    if bool(active[idx]):
                        states[idx] = eval_s31._advance_grammar_state(
                            states[idx],
                            int(next_tok[idx].item()),
                            strict=strict_grammar,
                            vocab_size=vocab_size,
                        )

            finished = finished | (next_tok == eos_id)
            if bool(finished.all()):
                break
            cur_ids = next_tok.unsqueeze(1)

        seq_flat = torch.cat(out_tokens, dim=1)         # (B*G, L)
        old_lp_flat = torch.cat(old_lp_steps, dim=1)    # (B*G, L-1)
        mask_flat = torch.cat(mask_steps, dim=1)        # (B*G, L-1)

        L = int(seq_flat.shape[1])
        seq_ids = seq_flat.view(B, G, L)
        old_log_probs = old_lp_flat.view(B, G, L - 1)
        action_mask = mask_flat.view(B, G, L - 1)

        decoded: List[List[List[str]]] = []
        for b in range(B):
            row: List[List[str]] = []
            for g in range(G):
                ids_list = seq_ids[b, g].tolist()
                row.append(
                    eval_s31.decode_uni_compat(ids_list, vocab_size=vocab_size, skip_special=True)
                )
            decoded.append(row)

        return RolloutBatch(
            seq_ids=seq_ids,
            old_log_probs=old_log_probs,
            action_mask=action_mask,
            decoded_tokens=decoded,
        )
    finally:
        if was_training:
            model.train()


def compute_sequence_logprobs(
    model: nn.Module,
    images: torch.Tensor,
    seq_ids: torch.Tensor,
    *,
    keep_ratio_override: Optional[float] = None,
) -> torch.Tensor:
    """Compute per-token policy log-probs ``(B, G, L-1)`` for ``seq_ids`` ``(B, G, L)``.

    Encodes ``images`` once per page and broadcasts memory across all ``G``
    rollouts. Works with both ``UniOMRNet`` (v1) and ``UniOMRv2Net`` (v2).
    """
    if seq_ids.ndim != 3:
        raise ValueError(f"Expected seq_ids of shape (B, G, L), got {tuple(seq_ids.shape)}")
    B, G, L = seq_ids.shape
    if L < 2:
        raise ValueError(f"seq_ids length must be >= 2 (BOS + >=1 token), got {L}")

    mem = _encode_image_compat(model, images, keep_ratio_override=keep_ratio_override)
    mem_exp = _repeat_memory(mem, G)  # (B*G, L_mem, D) or tuple thereof

    flat_seq = seq_ids.reshape(B * G, L)
    inp_ids = flat_seq[:, :-1]
    tgt_ids = flat_seq[:, 1:]

    logits = _decode_full_compat(model, inp_ids, mem_exp)  # (B*G, L-1, V)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    tok_lp = log_probs.gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)  # (B*G, L-1)
    return tok_lp.view(B, G, L - 1)


def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute GRPO group-normalized advantages: ``A_i = (R_i - mean(R)) / (std(R) + eps)``.

    Accepts ``rewards`` of shape ``(G,)`` or ``(B, G)`` and normalizes along the
    last (group) dimension.
    """
    r = rewards.float()
    mean = r.mean(dim=-1, keepdim=True)
    std = r.std(dim=-1, unbiased=False, keepdim=True)
    return (r - mean) / (std + float(eps))


def grpo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    clip_eps: float = 0.2,
    kl_beta: float = 0.02,
) -> Dict[str, torch.Tensor]:
    """Compute clipped GRPO policy-gradient surrogate + unbiased ``k3`` KL penalty.

    Implements the DeepSeekMath / DeepSeek-R1 / olmOCR 2 GRPO objective:
      ``L = E[ -min(r_t * A, clip(r_t, 1-eps, 1+eps) * A) + kl_beta * KL_k3(pi || pi_ref) ]``
    where ``r_t = exp(log pi(y_t) - log pi_old(y_t))`` and
    ``KL_k3 = exp(log pi_ref - log pi) - (log pi_ref - log pi) - 1 >= 0``.

    Args:
        new_log_probs: ``(..., T)`` current policy log-probs (requires grad).
        old_log_probs: ``(..., T)`` rollout behavior policy log-probs.
        ref_log_probs: ``(..., T)`` frozen reference SFT policy log-probs.
        advantages: ``(...)`` group-normalized advantages ``A_i``.
        action_mask: ``(..., T)`` float/bool mask (``1`` on valid tokens, ``0`` on PAD).
        clip_eps: PPO/GRPO clipping ratio epsilon (default ``0.2``).
        kl_beta: KL-to-reference-policy penalty weight (default ``0.02``).

    Returns:
        Dict with ``loss``, ``pg_loss``, ``kl_loss``, ``mean_ratio``, ``clip_frac``.
    """
    new_lp = new_log_probs.float()
    old_lp = old_log_probs.detach().float()
    ref_lp = ref_log_probs.detach().float()
    adv = advantages.detach().float().unsqueeze(-1)
    mask = action_mask.to(dtype=torch.float32)

    log_ratio = (new_lp - old_lp).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))

    surr1 = ratio * adv
    surr2 = clipped_ratio * adv
    pg_per_tok = -torch.minimum(surr1, surr2)

    # Unbiased non-negative Schulman k3 KL estimator: KL(pi || pi_ref)
    ref_log_diff = (ref_lp - new_lp).clamp(-20.0, 20.0)
    kl_per_tok = torch.exp(ref_log_diff) - ref_log_diff - 1.0

    denom = mask.sum().clamp(min=1.0)
    pg_loss = (pg_per_tok * mask).sum() / denom
    kl_loss = (kl_per_tok * mask).sum() / denom
    total_loss = pg_loss + float(kl_beta) * kl_loss

    with torch.no_grad():
        mean_ratio = (ratio * mask).sum() / denom
        clipped = ((ratio - clipped_ratio).abs() > 1e-7).to(torch.float32)
        clip_frac = (clipped * mask).sum() / denom

    return {
        "loss": total_loss,
        "pg_loss": pg_loss,
        "kl_loss": kl_loss,
        "mean_ratio": mean_ratio,
        "clip_frac": clip_frac,
    }


@dataclass
class GRPOConfig:
    """Hyperparameters for RLVR GRPO post-training."""

    group_size: int = 4
    max_len: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    clip_eps: float = 0.2
    kl_beta: float = 0.02
    adv_eps: float = 1e-6
    use_grammar: bool = True
    strict_grammar: bool = False
    keep_ratio_override: Optional[float] = None


def grpo_step(
    model: nn.Module,
    ref_model: nn.Module,
    images: torch.Tensor,
    ref_bekerns: Sequence[Union[str, Sequence[str]]],
    *,
    cfg: Optional[GRPOConfig] = None,
    check_decode_fn: Optional[Callable[[Sequence[str]], bool]] = None,
    reward_fn: Optional[Callable[[Sequence[str], Union[str, Sequence[str]]], Dict[str, Any]]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run one complete RLVR GRPO step on a batch of pages.

    1. Samples ``G`` rollouts per page under ``PrefixState``.
    2. Scores every rollout with ``compute_omr_rlvr_reward`` (or custom ``reward_fn``).
    3. Computes group-normalized advantages ``A_i = (R_i - mean(R)) / (std(R) + 1e-6)``.
    4. Computes current-policy and frozen-reference-policy log-probs.
    5. Returns ``(total_loss, stats)`` ready for ``loss.backward()``.

    Compatible with both ``UniOMRNet`` (v1) and ``UniOMRv2Net`` (v2).
    """
    c = cfg if cfg is not None else GRPOConfig()
    B = int(images.shape[0])
    if len(ref_bekerns) != B:
        raise ValueError(
            f"Expected {B} reference bekern sequences to match images batch, got {len(ref_bekerns)}"
        )

    rollouts = sample_rollouts(
        model,
        images,
        group_size=c.group_size,
        max_len=c.max_len,
        temperature=c.temperature,
        top_p=c.top_p,
        use_grammar=c.use_grammar,
        strict_grammar=c.strict_grammar,
        keep_ratio_override=c.keep_ratio_override,
    )

    G = c.group_size
    rewards = torch.zeros((B, G), dtype=torch.float32, device=images.device)
    comp_sums: Dict[str, float] = {
        "r_parse": 0.0,
        "r_sys": 0.0,
        "r_meter": 0.0,
        "r_noloop": 0.0,
        "r_note": 0.0,
    }
    for b in range(B):
        ref_b = ref_bekerns[b]
        for g in range(G):
            pred_toks = rollouts.decoded_tokens[b][g]
            if reward_fn is not None:
                res = reward_fn(pred_toks, ref_b)
            else:
                res = compute_omr_rlvr_reward(
                    pred_toks, ref_b, check_decode_fn=check_decode_fn
                )
            rewards[b, g] = float(res["reward"])
            for k in comp_sums:
                comp_sums[k] += float(res.get(k, 0.0))

    advantages = compute_group_advantages(rewards, eps=c.adv_eps)

    new_log_probs = compute_sequence_logprobs(
        model,
        images,
        rollouts.seq_ids,
        keep_ratio_override=c.keep_ratio_override,
    )
    with torch.no_grad():
        ref_was_training = ref_model.training
        ref_model.eval()
        try:
            ref_log_probs = compute_sequence_logprobs(
                ref_model,
                images,
                rollouts.seq_ids,
                keep_ratio_override=c.keep_ratio_override,
            )
        finally:
            if ref_was_training:
                ref_model.train()

    loss_dict = grpo_loss(
        new_log_probs,
        rollouts.old_log_probs,
        ref_log_probs,
        advantages,
        rollouts.action_mask,
        clip_eps=c.clip_eps,
        kl_beta=c.kl_beta,
    )

    total_n = float(max(1, B * G))
    stats: Dict[str, Any] = {
        "loss": float(loss_dict["loss"].detach().item()),
        "pg_loss": float(loss_dict["pg_loss"].detach().item()),
        "kl_loss": float(loss_dict["kl_loss"].detach().item()),
        "mean_ratio": float(loss_dict["mean_ratio"].detach().item()),
        "clip_frac": float(loss_dict["clip_frac"].detach().item()),
        "mean_reward": float(rewards.mean().item()),
        "std_reward": float(rewards.std(unbiased=False).item()),
        "mean_r_parse": comp_sums["r_parse"] / total_n,
        "mean_r_sys": comp_sums["r_sys"] / total_n,
        "mean_r_meter": comp_sums["r_meter"] / total_n,
        "mean_r_noloop": comp_sums["r_noloop"] / total_n,
        "mean_r_note": comp_sums["r_note"] / total_n,
        "rewards": rewards.detach(),
        "advantages": advantages.detach(),
        "seq_len": int(rollouts.seq_ids.shape[-1]),
    }
    return loss_dict["loss"], stats


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI argument parser for RLVR GRPO post-training configuration."""
    ap = argparse.ArgumentParser(
        description="RLVR (GRPO Verifiable-Reward Post-Training) for UniOMR v1 & v2."
    )
    ap.add_argument("--group-size", type=int, default=4, help="GRPO rollout group size G (default: 4).")
    ap.add_argument("--max-len", type=int, default=256, help="Max rollout tokens per page.")
    ap.add_argument("--temperature", type=float, default=0.8, help="Rollout sampling temperature.")
    ap.add_argument("--top-p", type=float, default=0.95, help="Rollout nucleus top-p.")
    ap.add_argument("--clip-eps", type=float, default=0.2, help="GRPO clipping epsilon.")
    ap.add_argument("--kl-beta", type=float, default=0.02, help="KL-to-reference-policy penalty weight.")
    ap.add_argument("--no-grammar", action="store_true", help="Disable PrefixState rollout constraints.")
    ap.add_argument("--strict-grammar", action="store_true", help="Use PrefixState(strict=True) during rollouts.")
    return ap


__all__ = [
    "W_PARSE",
    "W_SYS",
    "W_METER",
    "W_NOLOOP",
    "W_NOTE",
    "GRPOConfig",
    "RolloutBatch",
    "has_repetition_loop",
    "extract_pitch_dur_notes",
    "compute_sequence_f1",
    "compute_omr_rlvr_reward",
    "sample_top_p",
    "sample_rollouts",
    "compute_sequence_logprobs",
    "compute_group_advantages",
    "grpo_loss",
    "grpo_step",
    "build_arg_parser",
]
