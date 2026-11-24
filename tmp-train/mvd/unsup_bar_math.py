"""Bar-math unsupervised training and exact bekern beat-sum validator (MVD).

Provides:
  - ``score_bekern_bar_math``: exact deterministic beat-sum validator for
    bekern token streams with an anti-degeneracy guard to prevent reward
    hacking (all-rests, constant-pitch loops, zero-bar stubs, tight repeats).
  - ``bar_math_unsup_step``: KL-anchored, bar-math-weighted unsupervised
    training step over unlabeled system strips with optional multi-view
    consistency gating.

Pure PyTorch + stdlib; does not modify ``model.py`` or ``train.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

try:
    import data as _mvd_data
except ImportError:  # pragma: no cover - fallback when imported as package
    _mvd_data = None


# Written duration values in quarter-note beats (mirrors bekern.rs / grammar.py).
DURATION_BEATS: Dict[str, float] = {
    "breve": 8.0,
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "16th": 0.25,
    "32nd": 0.125,
    "64th": 0.0625,
    "128th": 0.03125,
}

STEPS = frozenset(("A", "B", "C", "D", "E", "F", "G"))
OCTAVES = frozenset(str(n) for n in range(10))
ALTERS = frozenset(("a-2", "a-1", "a1", "a2"))
BAR_CLOSE_TOKENS = frozenset(
    ("bar", "bar-light-heavy", "bar-repeat-right", "bar-double")
)
TAIL_MODS = frozenset(
    (
        "tie-start",
        "tie-stop",
        "slur-start",
        "slur-stop",
        "stacc",
        "accent",
        "tenuto",
        "fermata",
    )
)

_TIME_RE = re.compile(r"^time-(\d+)-(\d+)$")
_VOICE_RE = re.compile(r"^voice-(\d+)$")
_STAFF_RE = re.compile(r"^staff-(\d+)$")
_TUPLET_START_RE = re.compile(r"^tuplet-start-(\d+)-(\d+)$")
_TUPLET_SHORT_RE = re.compile(r"^tuplet-(\d+)$")


@dataclass
class BarMathResult:
    """Result of validating bar beat sums and degeneracy on a bekern stream."""

    valid_bar_ratio: float
    n_bars: int
    n_valid_bars: int
    mean_beat_err: float
    is_degenerate: bool
    degenerate_reason: Optional[str] = None
    per_bar_details: List[Dict[str, Any]] = field(default_factory=list)


def _shorthand_tuplet_normal(actual: int) -> int:
    """Standard normal-notes count for shorthand ``tuplet-A`` (e.g. 3->2, 5->4)."""
    if actual <= 1:
        return 1
    if actual == 2:
        return 3
    p = 1
    while (p << 1) < actual:
        p <<= 1
    return p


def _parse_time_beats(tok: str) -> Optional[float]:
    m = _TIME_RE.match(tok)
    if not m:
        return None
    num, den = int(m.group(1)), int(m.group(2))
    if num <= 0 or den <= 0:
        return None
    return float(num) * (4.0 / float(den))


def _has_tight_token_cycle(
    tokens: Sequence[str], max_period: int = 8, min_repeats: int = 8
) -> bool:
    """Detect exact adjacent repeats of a short token cycle in non-header tokens."""
    n = len(tokens)
    for p in range(1, max_period + 1):
        need = p * min_repeats
        if n < need:
            continue
        # Sliding check over the sequence.
        run = 1
        for i in range(p, n - p + 1, p):
            if tokens[i : i + p] == tokens[i - p : i]:
                run += 1
                if run >= min_repeats:
                    return True
            else:
                run = 1
        # Also check anchored at the tail (catches unaligned loop starts).
        tail_base = tokens[n - p : n]
        tail_ok = True
        for k in range(1, min_repeats):
            if tokens[n - (k + 1) * p : n - k * p] != tail_base:
                tail_ok = False
                break
        if tail_ok:
            return True
    return False


def score_bekern_bar_math(
    tokens: Sequence[str],
    allow_pickup: bool = True,
    allow_final_incomplete: bool = True,
    tol: float = 1e-3,
) -> BarMathResult:
    """Validate bekern bar beat sums against active time signatures.

    Args:
        tokens: Sequence of bekern token strings (or a whitespace-separated str).
        allow_pickup: If True, an under-full first bar (``0 < beats <= expected``
            with voices agreeing) is counted as valid.
        allow_final_incomplete: If True, an under-full final bar
            (``0 < beats <= expected`` with voices agreeing) is counted as valid.
        tol: Beat-sum tolerance in quarter notes (absorbs tuplet float dust).

    Returns:
        :class:`BarMathResult` with bar validity metrics, per-bar breakdown,
        and anti-degeneracy flags.
    """
    if isinstance(tokens, str):
        tok_list: List[str] = tokens.split()
    else:
        tok_list = [str(t) for t in tokens]

    active_tsig: float = 4.0  # Default 4/4 when no time-N-D header is present.
    active_tsig_str: str = "4/4"
    pending_tsig: Optional[Tuple[float, str]] = None

    cur_staff: int = 1
    cur_voice: int = 1

    # Per-bar state keyed by (staff, voice).
    voice_beats: Dict[Tuple[int, int], float] = {}
    voice_scales: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    pend_spans: List[Tuple[int, int]] = []
    cur_bar_tokens: List[str] = []

    # Open note/rest event state.
    ev_kind: Optional[str] = None  # "note" | "rest" | None
    ev_base: Optional[float] = None
    ev_dots: int = 0
    ev_is_chord: bool = False
    ev_is_grace: bool = False
    ev_step: Optional[str] = None
    ev_oct: Optional[str] = None
    ev_alter: Optional[str] = None

    # Next-event modifier flags.
    next_is_chord: bool = False
    next_is_grace: bool = False
    expect_rest_dur: bool = False

    # Corpus-level anti-degeneracy tracking.
    n_note_events: int = 0
    n_rest_events: int = 0
    distinct_pitches: set[Tuple[str, str, Optional[str]]] = set()
    body_tokens: List[str] = []
    in_header: bool = True

    # Collected raw bars: list of (expected_beats, tsig_str, voice_beats_dict, bar_toks)
    raw_bars: List[Tuple[float, str, Dict[Tuple[int, int], float], Tuple[str, ...]]] = []

    def _cur_key() -> Tuple[int, int]:
        return (cur_staff, cur_voice)

    def _drain_scale(key: Tuple[int, int]) -> float:
        if pend_spans:
            voice_scales.setdefault(key, []).extend(pend_spans)
            pend_spans.clear()
        s = 1.0
        for a, n in voice_scales.get(key, ()):
            s *= float(n) / float(a)
        return s

    def _flush_event() -> None:
        nonlocal ev_kind, ev_base, ev_dots, ev_is_chord, ev_is_grace
        nonlocal ev_step, ev_oct, ev_alter, n_note_events, n_rest_events

        if ev_kind is None:
            return
        key = _cur_key()
        scale = _drain_scale(key)
        voice_beats.setdefault(key, 0.0)

        if ev_kind == "note":
            n_note_events += 1
            if ev_step is not None and ev_oct is not None:
                distinct_pitches.add((ev_step, ev_oct, ev_alter))
            if ev_base is not None and not ev_is_chord and not ev_is_grace:
                dur = ev_base * (2.0 - 2.0 ** (-ev_dots)) * scale
                voice_beats[key] += dur
        elif ev_kind == "rest":
            n_rest_events += 1
            if ev_base is not None:
                dur = ev_base * (2.0 - 2.0 ** (-ev_dots)) * scale
                voice_beats[key] += dur

        ev_kind = None
        ev_base = None
        ev_dots = 0
        ev_is_chord = False
        ev_is_grace = False
        ev_step = None
        ev_oct = None
        ev_alter = None

    def _close_bar() -> None:
        nonlocal active_tsig, active_tsig_str, pending_tsig
        nonlocal cur_staff, cur_voice, next_is_chord, next_is_grace, expect_rest_dur

        _flush_event()
        if voice_beats:
            raw_bars.append(
                (active_tsig, active_tsig_str, dict(voice_beats), tuple(cur_bar_tokens))
            )
        voice_beats.clear()
        voice_scales.clear()
        pend_spans.clear()
        cur_bar_tokens.clear()
        cur_staff = 1
        cur_voice = 1
        next_is_chord = False
        next_is_grace = False
        expect_rest_dur = False
        if pending_tsig is not None:
            active_tsig, active_tsig_str = pending_tsig
            pending_tsig = None

    for tok in tok_list:
        # Track body tokens (skip leading clef/key/time headers for loop check).
        is_header_tok = (
            tok.startswith("clef-")
            or tok.startswith("key")
            or tok.startswith("time-")
        )
        if not is_header_tok:
            in_header = False
        if not in_header:
            body_tokens.append(tok)

        # Time signature change.
        if tok.startswith("time-"):
            _flush_event()
            ts = _parse_time_beats(tok)
            if ts is not None:
                parts = tok.split("-")
                ts_label = f"{parts[1]}/{parts[2]}"
                if not voice_beats or all(v == 0.0 for v in voice_beats.values()):
                    active_tsig = ts
                    active_tsig_str = ts_label
                    pending_tsig = None
                else:
                    pending_tsig = (ts, ts_label)
            continue

        # Barline close.
        if tok in BAR_CLOSE_TOKENS:
            _close_bar()
            continue

        # Staff / voice switches.
        if tok.startswith("staff-"):
            m = _STAFF_RE.match(tok)
            if m:
                _flush_event()
                cur_staff = int(m.group(1))
                cur_bar_tokens.append(tok)
            continue

        if tok.startswith("voice-"):
            m = _VOICE_RE.match(tok)
            if m:
                _flush_event()
                cur_voice = int(m.group(1))
                cur_bar_tokens.append(tok)
            continue

        # Tuplet spans.
        if tok.startswith("tuplet-start-"):
            m = _TUPLET_START_RE.match(tok)
            if m:
                _flush_event()
                a, n = int(m.group(1)), int(m.group(2))
                if a > 0 and n > 0:
                    pend_spans.append((a, n))
                cur_bar_tokens.append(tok)
            continue

        if tok == "tuplet-stop":
            _flush_event()
            stk = voice_scales.get(_cur_key())
            if stk:
                stk.pop()
            elif pend_spans:
                pend_spans.pop()
            cur_bar_tokens.append(tok)
            continue

        if tok.startswith("tuplet-"):
            m = _TUPLET_SHORT_RE.match(tok)
            if m:
                _flush_event()
                a = int(m.group(1))
                if a > 0:
                    pend_spans.append((a, _shorthand_tuplet_normal(a)))
                cur_bar_tokens.append(tok)
            continue

        # Whole-bar rest.
        if tok == "bar-rest":
            _flush_event()
            key = _cur_key()
            _drain_scale(key)
            voice_beats[key] = max(voice_beats.get(key, 0.0), active_tsig)
            n_rest_events += 1
            cur_bar_tokens.append(tok)
            continue

        # Event prefixes: chord, grace, rest.
        if tok == "chord":
            _flush_event()
            next_is_chord = True
            cur_bar_tokens.append(tok)
            continue

        if tok == "grace":
            _flush_event()
            next_is_grace = True
            cur_bar_tokens.append(tok)
            continue

        if tok == "rest":
            _flush_event()
            expect_rest_dur = True
            cur_bar_tokens.append(tok)
            continue

        # Duration tokens start a new note or complete a rest's duration.
        if tok in DURATION_BEATS:
            if expect_rest_dur:
                expect_rest_dur = False
                ev_kind = "rest"
                ev_base = DURATION_BEATS[tok]
                ev_dots = 0
                ev_is_chord = False
                ev_is_grace = False
            else:
                _flush_event()
                ev_kind = "note"
                ev_base = DURATION_BEATS[tok]
                ev_dots = 0
                ev_is_chord = next_is_chord
                ev_is_grace = next_is_grace
                ev_step = None
                ev_oct = None
                ev_alter = None
                next_is_chord = False
                next_is_grace = False
            cur_bar_tokens.append(tok)
            continue

        # Intra-event pitch / accidental / dot / modifiers.
        if ev_kind == "note":
            if tok in STEPS and ev_step is None:
                ev_step = tok
                cur_bar_tokens.append(tok)
                continue
            if tok in OCTAVES and ev_oct is None:
                ev_oct = tok
                cur_bar_tokens.append(tok)
                continue
            if tok in ALTERS:
                ev_alter = tok
                cur_bar_tokens.append(tok)
                continue
            if tok == "dot":
                ev_dots += 1
                cur_bar_tokens.append(tok)
                continue
            if tok in TAIL_MODS:
                cur_bar_tokens.append(tok)
                continue
        elif ev_kind == "rest":
            if tok == "dot":
                ev_dots += 1
                cur_bar_tokens.append(tok)
                continue
            if tok in TAIL_MODS:
                cur_bar_tokens.append(tok)
                continue

        # Zero-duration boundary tokens (clef, key, dynamics, hairpins,
        # opening repeat barlines, etc.) close any open event.
        if (
            tok.startswith("clef-")
            or tok.startswith("key")
            or tok.startswith("dyn-")
            or tok.startswith("hairpin-")
            or tok == "bar-repeat-left"
        ):
            _flush_event()
            continue

    # Flush trailing unclosed bar (if events exist after the last barline).
    _flush_event()
    if voice_beats:
        raw_bars.append(
            (active_tsig, active_tsig_str, dict(voice_beats), tuple(cur_bar_tokens))
        )

    n_bars = len(raw_bars)
    if n_bars == 0:
        return BarMathResult(
            valid_bar_ratio=0.0,
            n_bars=0,
            n_valid_bars=0,
            mean_beat_err=1.0,
            is_degenerate=True,
            degenerate_reason="zero_bars",
            per_bar_details=[],
        )

    per_bar_details: List[Dict[str, Any]] = []
    n_valid_bars = 0
    total_beat_err = 0.0
    bar_signatures: List[Tuple[str, ...]] = []

    for idx, (expected, ts_label, vbeats, btoks) in enumerate(raw_bars):
        bar_signatures.append(btoks)
        vals = list(vbeats.values())
        is_first = idx == 0
        is_last = idx == n_bars - 1

        max_v = max(vals)
        min_v = min(vals)
        voices_agree = (max_v - min_v) <= tol

        is_pickup = False
        is_final_inc = False

        if all(abs(v - expected) <= tol for v in vals):
            bar_err = 0.0
            is_valid = True
        elif (
            allow_pickup
            and is_first
            and min_v > tol
            and max_v <= expected + tol
            and voices_agree
        ):
            bar_err = 0.0
            is_valid = True
            is_pickup = True
        elif (
            allow_final_incomplete
            and is_last
            and min_v > tol
            and max_v <= expected + tol
            and voices_agree
        ):
            bar_err = 0.0
            is_valid = True
            is_final_inc = True
        else:
            bar_err = max(abs(v - expected) for v in vals)
            is_valid = bar_err <= tol

        if is_valid:
            n_valid_bars += 1
        total_beat_err += bar_err

        per_bar_details.append(
            {
                "bar_index": idx,
                "time_sig": ts_label,
                "expected_beats": expected,
                "voice_beats": {
                    f"s{s}:v{v}": round(b, 6) for (s, v), b in sorted(vbeats.items())
                },
                "beat_err": round(bar_err, 6),
                "is_valid": is_valid,
                "is_pickup": is_pickup,
                "is_final_incomplete": is_final_inc,
            }
        )

    valid_bar_ratio = float(n_valid_bars) / float(n_bars)
    mean_beat_err = total_beat_err / float(n_bars)

    # Anti-degeneracy guard.
    is_degenerate = False
    degenerate_reason: Optional[str] = None
    total_events = n_note_events + n_rest_events

    if total_events == 0:
        is_degenerate = True
        degenerate_reason = "no_events"
    elif (float(n_rest_events) / float(total_events)) > 0.85:
        is_degenerate = True
        degenerate_reason = (
            f"excessive_rests ({n_rest_events}/{total_events} = "
            f"{n_rest_events / total_events:.2%})"
        )
    elif n_bars >= 3 and len(distinct_pitches) < 2:
        is_degenerate = True
        degenerate_reason = (
            f"low_pitch_diversity ({len(distinct_pitches)} distinct pitch(es) "
            f"across {n_bars} bars)"
        )
    elif _has_tight_token_cycle(body_tokens, max_period=8, min_repeats=8):
        is_degenerate = True
        degenerate_reason = "repeat_loop_collapse"
    elif n_bars >= 4 and len(set(bar_signatures)) == 1:
        is_degenerate = True
        degenerate_reason = f"identical_bar_loop ({n_bars} identical bars)"

    return BarMathResult(
        valid_bar_ratio=valid_bar_ratio,
        n_bars=n_bars,
        n_valid_bars=n_valid_bars,
        mean_beat_err=mean_beat_err,
        is_degenerate=is_degenerate,
        degenerate_reason=degenerate_reason,
        per_bar_details=per_bar_details,
    )


def _ids_to_tokens(
    seq_ids: Sequence[int],
    vocab: Sequence[str],
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> Tuple[List[str], List[int]]:
    """Convert raw decoded token IDs to (token strings, trimmed target IDs)."""
    toks: List[str] = []
    trimmed_ids: List[int] = []
    vlen = len(vocab)
    for raw_i in seq_ids:
        i = int(raw_i)
        if i == pad_id or i == bos_id:
            continue
        if i == eos_id:
            trimmed_ids.append(eos_id)
            break
        if 0 <= i < vlen:
            toks.append(vocab[i])
            trimmed_ids.append(i)
    return toks, trimmed_ids


def _token_agreement(a: Sequence[str], b: Sequence[str]) -> float:
    """Normalized Levenshtein similarity in [0, 1] between two token lists."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    # Standard O(min(la, lb)) space edit distance.
    if la < lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    dist = prev[lb]
    return max(0.0, 1.0 - float(dist) / float(max(la, lb)))


def bar_math_unsup_step(
    model,
    ref_model,
    vocab: Sequence[str],
    unlabeled_imgs: torch.Tensor,
    mem_mask: Optional[torch.Tensor] = None,
    kl_beta: float = 0.1,
    min_valid_ratio: float = 0.95,
    max_len: int = 256,
    multi_view_imgs: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    num_candidates: int = 2,
    allow_pickup: bool = True,
    allow_final_incomplete: bool = True,
    min_multi_view_agreement: float = 0.8,
    constrain: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """KL-anchored bar-math-weighted unsupervised training step.

    1. Decodes candidate hypotheses (greedy if ``num_candidates <= 1`` else
       beam search with ``beam=num_candidates``) on ``unlabeled_imgs`` under
       ``torch.no_grad()``.
    2. Scores every candidate with :func:`score_bekern_bar_math` (and optional
       multi-view agreement if ``multi_view_imgs`` is provided).
    3. Rejects degenerate or low-score (``valid_bar_ratio < min_valid_ratio``)
       hypotheses.
    4. Computes ``L = weight * CE(student_logits, accepted_ids) +
       kl_beta * KL(student_logits || ref_model_logits)`` on accepted strips,
       where ``weight = valid_bar_ratio * max(0, 1 - mean_beat_err)``.
       If no strip passes the strict gate, cleanly returns ``zero_loss`` with
       ``accepted_frac = 0.0``.

    Returns:
        ``(loss, stats)`` where ``loss`` is a scalar tensor and ``stats`` is a
        dictionary of monitoring metrics.
    """
    device = unlabeled_imgs.device
    batch_size = int(unlabeled_imgs.shape[0])
    if batch_size == 0:
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return zero, {
            "accepted_frac": 0.0,
            "n_accepted": 0,
            "batch_size": 0,
            "ce_loss": 0.0,
            "kl_loss": 0.0,
            "mean_weight": 0.0,
            "mean_valid_bar_ratio": 0.0,
            "mean_beat_err": 0.0,
            "degenerate_frac": 0.0,
            "accepted_indices": [],
        }

    # Grammar masking in model.py relies on data.VOCAB indices; only enable it
    # when the supplied vocab matches data.VOCAB in size.
    can_constrain = bool(
        constrain
        and _mvd_data is not None
        and len(vocab) == len(_mvd_data.VOCAB)
    )

    was_training = model.training
    with torch.no_grad():
        model.eval()
        if num_candidates <= 1:
            dec = model.greedy_kv(
                unlabeled_imgs,
                max_len=max_len,
                constrain=can_constrain,
                mem_mask=mem_mask,
            )  # [B, L]
            cand_ids = dec.unsqueeze(1)  # [B, 1, L]
        else:
            cand_ids = model.beam_search(
                unlabeled_imgs,
                max_len=max_len,
                constrain=can_constrain,
                strict=False,
                beam=num_candidates,
                mem_mask=mem_mask,
                return_all=True,
            )  # [B, K, L]

        # Optional multi-view decoding for consistency gating.
        view_toks_per_item: Optional[List[List[List[str]]]] = None
        if multi_view_imgs is not None:
            views = (
                [multi_view_imgs]
                if isinstance(multi_view_imgs, torch.Tensor)
                else list(multi_view_imgs)
            )
            view_toks_per_item = [[] for _ in range(batch_size)]
            for v_imgs in views:
                v_dec = model.greedy_kv(
                    v_imgs,
                    max_len=max_len,
                    constrain=can_constrain,
                    mem_mask=mem_mask,
                )
                for b in range(batch_size):
                    v_toks, _ = _ids_to_tokens(
                        v_dec[b].tolist(),
                        vocab,
                        model.pad_id,
                        model.bos_id,
                        model.eos_id,
                    )
                    view_toks_per_item[b].append(v_toks)

    model.train(was_training)

    accepted_indices: List[int] = []
    accepted_targets: List[List[int]] = []
    accepted_weights: List[float] = []

    best_ratios: List[float] = []
    best_errs: List[float] = []
    n_degenerate = 0

    k_cands = int(cand_ids.shape[1])
    for b in range(batch_size):
        best_choice: Optional[Tuple[float, float, int, List[int], BarMathResult]] = None
        top_res: Optional[BarMathResult] = None

        for k in range(k_cands):
            toks, trimmed_ids = _ids_to_tokens(
                cand_ids[b, k].tolist(),
                vocab,
                model.pad_id,
                model.bos_id,
                model.eos_id,
            )
            res = score_bekern_bar_math(
                toks,
                allow_pickup=allow_pickup,
                allow_final_incomplete=allow_final_incomplete,
            )
            if k == 0:
                top_res = res

            if not trimmed_ids or res.is_degenerate:
                continue
            if res.valid_bar_ratio < min_valid_ratio:
                continue

            if view_toks_per_item is not None and view_toks_per_item[b]:
                agrees = [
                    _token_agreement(toks, vt) for vt in view_toks_per_item[b]
                ]
                if (sum(agrees) / float(len(agrees))) < min_multi_view_agreement:
                    continue

            weight = res.valid_bar_ratio * max(0.0, 1.0 - res.mean_beat_err)
            if weight <= 0.0:
                continue

            # Prefer highest weight, then lowest beat error, then higher beam rank.
            if best_choice is None or (weight, -res.mean_beat_err, -k) > (
                best_choice[0],
                -best_choice[1],
                -best_choice[2],
            ):
                best_choice = (weight, res.mean_beat_err, k, trimmed_ids, res)

        if best_choice is not None:
            w, err, _k, t_ids, res = best_choice
            accepted_indices.append(b)
            accepted_targets.append(t_ids)
            accepted_weights.append(w)
            best_ratios.append(res.valid_bar_ratio)
            best_errs.append(err)
        else:
            if top_res is not None:
                best_ratios.append(top_res.valid_bar_ratio)
                best_errs.append(top_res.mean_beat_err)
                if top_res.is_degenerate:
                    n_degenerate += 1

    mean_valid_ratio = (
        sum(best_ratios) / float(len(best_ratios)) if best_ratios else 0.0
    )
    mean_beat_err = (
        sum(best_errs) / float(len(best_errs)) if best_errs else 1.0
    )

    if not accepted_indices:
        zero_loss = torch.zeros((), device=device, dtype=torch.float32)
        return zero_loss, {
            "accepted_frac": 0.0,
            "n_accepted": 0,
            "batch_size": batch_size,
            "ce_loss": 0.0,
            "kl_loss": 0.0,
            "mean_weight": 0.0,
            "mean_valid_bar_ratio": mean_valid_ratio,
            "mean_beat_err": mean_beat_err,
            "degenerate_frac": float(n_degenerate) / float(batch_size),
            "accepted_indices": [],
        }

    # Build padded target tensor for accepted strips.
    n_acc = len(accepted_indices)
    max_t = max(len(t) for t in accepted_targets)
    tgt = torch.full((n_acc, max_t), model.pad_id, dtype=torch.long, device=device)
    for row_i, t_ids in enumerate(accepted_targets):
        tgt[row_i, : len(t_ids)] = torch.tensor(
            t_ids, dtype=torch.long, device=device
        )

    idx_t = torch.tensor(accepted_indices, dtype=torch.long, device=device)
    acc_imgs = unlabeled_imgs.index_select(0, idx_t)
    acc_mem_mask = (
        mem_mask.index_select(0, idx_t) if mem_mask is not None else None
    )

    bos = torch.full((n_acc, 1), model.bos_id, dtype=torch.long, device=device)
    inp = torch.cat([bos, tgt[:, :-1]], dim=1)

    # Differentiable student forward pass.
    student_logits = model(acc_imgs, inp, mem_mask=acc_mem_mask)  # [N_acc, T, V]

    # Frozen reference model forward pass (no grad).
    ref_was_training = ref_model.training
    with torch.no_grad():
        ref_model.eval()
        ref_logits = ref_model(acc_imgs, inp, mem_mask=acc_mem_mask)
    ref_model.train(ref_was_training)

    v_size = student_logits.shape[-1]
    valid_mask = tgt != model.pad_id  # [N_acc, T]
    tok_counts = valid_mask.sum(dim=1).clamp_min(1).to(student_logits.dtype)

    ce_tok = F.cross_entropy(
        student_logits.reshape(-1, v_size),
        tgt.reshape(-1),
        ignore_index=model.pad_id,
        reduction="none",
    ).reshape(n_acc, max_t)
    ce_per_item = (ce_tok * valid_mask.to(ce_tok.dtype)).sum(dim=1) / tok_counts

    weights_t = torch.tensor(
        accepted_weights, device=device, dtype=ce_per_item.dtype
    )
    weighted_ce = (weights_t * ce_per_item).mean()

    # KL(student || ref) over non-PAD target positions.
    log_p = F.log_softmax(student_logits.float(), dim=-1)
    log_q = F.log_softmax(ref_logits.float(), dim=-1)
    p = log_p.exp()
    kl_tok = (p * (log_p - log_q)).sum(dim=-1)  # [N_acc, T]
    kl_per_item = (
        kl_tok * valid_mask.to(kl_tok.dtype)
    ).sum(dim=1) / tok_counts.to(kl_tok.dtype)
    kl_loss = kl_per_item.mean()

    total_loss = weighted_ce + float(kl_beta) * kl_loss

    stats = {
        "accepted_frac": float(n_acc) / float(batch_size),
        "n_accepted": n_acc,
        "batch_size": batch_size,
        "ce_loss": float(weighted_ce.detach().item()),
        "kl_loss": float(kl_loss.detach().item()),
        "mean_weight": float(weights_t.mean().item()),
        "mean_valid_bar_ratio": mean_valid_ratio,
        "mean_beat_err": mean_beat_err,
        "degenerate_frac": float(n_degenerate) / float(batch_size),
        "accepted_indices": accepted_indices,
    }
    return total_loss, stats


def run_bar_math_round(
    net: nn.Module,
    opt: torch.optim.Optimizer,
    scaler,
    strip_paths: list,
    step: int = 0,
    steps: int = 50,
    weight: float = 0.25,
    kl_beta: float = 0.10,
    height: int = 256,
    device: torch.device | str = "cpu",
    amp: bool = False,
    amp_dtype=torch.bfloat16,
    seed: int = 0,
    batch_size: int = 2,
    **_extra,
) -> dict:
    """Execute one KL-anchored bar-math-validated unsupervised round on unlabeled strips."""
    import copy
    import numpy as np
    from PIL import Image
    import data as _mvd_data

    if not strip_paths or int(steps) <= 0:
        return {"ran_steps": 0, "accepted_frac": 0.0, "mean_loss": 0.0}

    dev = torch.device(device)
    ref_model = copy.deepcopy(net).to(dev).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    rng = np.random.default_rng(int(seed))
    n_imgs = len(strip_paths)
    in_ch = int(getattr(net, "in_chans", 4))
    was_train = net.training
    net.train()
    use_amp = bool(amp and dev.type == "cuda")
    total_loss = 0.0
    total_acc = 0.0
    ran = 0
    bs = max(1, int(batch_size))

    for _ in range(int(steps)):
        idxs = rng.integers(0, n_imgs, size=bs)
        imgs = []
        max_w = 0
        for idx in idxs:
            try:
                im = Image.open(strip_paths[int(idx)]).convert("RGB")
                ow, oh = im.size
                nw = max(32, min(1536, int(round(ow * (height / max(1, oh))))))
                im = im.resize((nw, int(height)), Image.BOX)
                arr = np.asarray(im, dtype=np.float32) / 255.0
                t = (torch.from_numpy(arr).permute(2, 0, 1) - 0.85) / 0.25
                if in_ch > 3:
                    pad_c = torch.zeros((in_ch - 3, int(height), nw), dtype=torch.float32)
                    t = torch.cat([t, pad_c], dim=0)
                imgs.append(t)
                max_w = max(max_w, nw)
            except Exception:
                continue
        if not imgs:
            continue
        padded = []
        for t in imgs:
            pw = max_w - t.shape[2]
            if pw > 0:
                t = F.pad(t, (0, pw, 0, 0), value=0.6)
            padded.append(t)
        x = torch.stack(padded, dim=0).to(dev, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            res = bar_math_unsup_step(
                net,
                ref_model,
                _mvd_data.VOCAB,
                x,
                kl_beta=float(kl_beta),
                min_valid_ratio=0.95,
                max_len=192,
            )
            loss = res["loss"] * float(weight)
        acc = float(res.get("accepted_frac", 0.0))
        total_acc += acc
        if acc > 0.0 and torch.isfinite(loss) and loss.requires_grad:
            if scaler is not None and use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            total_loss += float(loss.detach().item())
        ran += 1

    net.train(was_train)
    mean_acc = total_acc / max(1, ran)
    mean_loss = total_loss / max(1, ran)
    print(f"  [unsup-bar-math step {step}] ran_steps={ran}/{steps} accepted={mean_acc:.2%} mean_loss={mean_loss:.4f}", flush=True)
    return {"ran_steps": ran, "accepted_frac": mean_acc, "mean_loss": mean_loss}


def _ids_to_tokens_flex(
    seq_ids: Sequence[int],
    id2tok: Union[Sequence[str], Dict[int, str]],
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> Tuple[List[str], List[int]]:
    """Convert raw decoded IDs to ``(token_strings, trimmed_ids)`` for list or dict vocab."""
    if isinstance(id2tok, dict):
        toks: List[str] = []
        trimmed_ids: List[int] = []
        for raw_i in seq_ids:
            i = int(raw_i)
            if i == pad_id or i == bos_id:
                continue
            if i == eos_id:
                trimmed_ids.append(eos_id)
                break
            tok = id2tok.get(i)
            if tok is not None:
                toks.append(str(tok))
                trimmed_ids.append(i)
        return toks, trimmed_ids
    return _ids_to_tokens(seq_ids, id2tok, pad_id, bos_id, eos_id)


def _score_visual(visual_scorer, strip: torch.Tensor, toks: Sequence[str]) -> float:
    """Call optional ``visual_scorer`` with ``(strip, toks)`` (fallback ``(toks,)``)."""
    if visual_scorer is None:
        return 0.0
    try:
        return float(visual_scorer(strip, toks))
    except TypeError:
        return float(visual_scorer(toks))


def grpo_bar_math_step(
    model,
    ref_model,
    strips: torch.Tensor,
    *,
    mem_mask: Optional[torch.Tensor] = None,
    id2tok: Optional[Union[Sequence[str], Dict[int, str]]] = None,
    num_beams: int = 4,
    kl_beta: float = 0.08,
    max_len: int = 192,
    visual_scorer=None,
    w_bm: float = 0.75,
    w_vis: float = 0.25,
    w_deg: float = 0.50,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Verifier-guided Group Relative Policy Optimization (GRPO) step.

    1. Samples ``K = num_beams`` candidate sequences per strip via
       ``model.beam_search(..., beam=K, return_all=True)`` (falling back to
       ``model.greedy_kv`` if ``beam_search`` is unavailable).
    2. Scores each candidate ``i`` with::

           r_i = w_bm * bar_math_ratio + w_vis * visual_score - w_deg * float(is_degenerate)

    3. Computes group-relative advantages per strip::

           A_i = (r_i - mean(r)) / (std(r) + 1e-6)

       Strips with ``std(r) < 1e-4`` (flat reward groups) are skipped so they
       do not inject gradient noise.
    4. Computes length-normalized sequence log-probabilities under ``model``
       (with grad) and frozen ``ref_model`` (no grad), returning::

           loss = -(A_i.detach() * seq_logp_i).mean() + kl_beta * kl_token_mean

    Returns:
        ``(loss, stats)`` scalar loss tensor and monitoring dict.
    """
    device = strips.device
    batch_size = int(strips.shape[0])
    if batch_size == 0:
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return zero, {
            "batch_size": 0,
            "num_beams": int(num_beams),
            "active_strips": 0,
            "active_frac": 0.0,
            "active_indices": [],
            "pg_loss": 0.0,
            "kl_loss": 0.0,
            "mean_reward": 0.0,
            "mean_valid_bar_ratio": 0.0,
            "degenerate_frac": 0.0,
            "rewards": [],
            "advantages": [],
        }

    if id2tok is None:
        if _mvd_data is None:
            raise ValueError("id2tok must be provided when data.VOCAB is unavailable")
        vocab: Union[Sequence[str], Dict[int, str]] = _mvd_data.VOCAB
    else:
        vocab = id2tok

    can_constrain = bool(
        _mvd_data is not None
        and not isinstance(vocab, dict)
        and len(vocab) == len(_mvd_data.VOCAB)
    )

    k_req = max(1, int(num_beams))
    was_training = model.training
    with torch.no_grad():
        model.eval()
        beam_fn = getattr(model, "beam_search", None)
        if k_req > 1 and callable(beam_fn):
            cand_ids = beam_fn(
                strips,
                max_len=max_len,
                constrain=can_constrain,
                strict=False,
                beam=k_req,
                mem_mask=mem_mask,
                return_all=True,
            )  # [B, K, L]
        else:
            dec = model.greedy_kv(
                strips,
                max_len=max_len,
                constrain=can_constrain,
                mem_mask=mem_mask,
            )  # [B, L]
            cand_ids = dec.unsqueeze(1)  # [B, 1, L]
        if cand_ids.ndim == 2:
            cand_ids = cand_ids.unsqueeze(1)
    model.train(was_training)

    k_cands = int(cand_ids.shape[1])
    active_indices: List[int] = []
    flat_targets: List[List[int]] = []
    flat_advantages: List[float] = []

    all_rewards: List[List[float]] = []
    all_advantages: List[Optional[List[float]]] = []
    all_bm_ratios: List[float] = []
    n_deg_total = 0
    n_cands_total = batch_size * k_cands

    for b in range(batch_size):
        strip_rewards: List[float] = []
        strip_targets: List[List[int]] = []

        for k in range(k_cands):
            toks, trimmed_ids = _ids_to_tokens_flex(
                cand_ids[b, k].tolist(),
                vocab,
                model.pad_id,
                model.bos_id,
                model.eos_id,
            )
            res = score_bekern_bar_math(toks)
            bm_ratio = float(res.valid_bar_ratio)
            is_deg = bool(res.is_degenerate)
            vis_score = _score_visual(visual_scorer, strips[b], toks)

            r_i = (
                float(w_bm) * bm_ratio
                + float(w_vis) * vis_score
                - float(w_deg) * float(is_deg)
            )
            strip_rewards.append(r_i)
            strip_targets.append(trimmed_ids if trimmed_ids else [model.eos_id])
            all_bm_ratios.append(bm_ratio)
            if is_deg:
                n_deg_total += 1

        all_rewards.append(strip_rewards)
        r_t = torch.tensor(strip_rewards, dtype=torch.float32)
        r_mean = r_t.mean()
        r_std = r_t.std(unbiased=False)

        if float(r_std.item()) < 1e-4:
            all_advantages.append(None)
            continue

        adv_t = (r_t - r_mean) / (r_std + 1e-6)
        adv_list = [float(x) for x in adv_t.tolist()]
        all_advantages.append(adv_list)
        active_indices.append(b)
        flat_targets.extend(strip_targets)
        flat_advantages.extend(adv_list)

    mean_reward = (
        sum(sum(rs) for rs in all_rewards) / float(max(1, n_cands_total))
    )
    mean_bm_ratio = (
        sum(all_bm_ratios) / float(max(1, len(all_bm_ratios)))
    )
    deg_frac = float(n_deg_total) / float(max(1, n_cands_total))

    if not active_indices:
        zero_loss = torch.zeros((), device=device, dtype=torch.float32)
        return zero_loss, {
            "batch_size": batch_size,
            "num_beams": k_cands,
            "active_strips": 0,
            "active_frac": 0.0,
            "active_indices": [],
            "pg_loss": 0.0,
            "kl_loss": 0.0,
            "mean_reward": mean_reward,
            "mean_valid_bar_ratio": mean_bm_ratio,
            "degenerate_frac": deg_frac,
            "rewards": all_rewards,
            "advantages": all_advantages,
        }

    n_total = len(flat_targets)
    max_t = max(len(t) for t in flat_targets)
    tgt = torch.full((n_total, max_t), model.pad_id, dtype=torch.long, device=device)
    for row_i, t_ids in enumerate(flat_targets):
        tgt[row_i, : len(t_ids)] = torch.tensor(t_ids, dtype=torch.long, device=device)

    idx_t = torch.tensor(active_indices, dtype=torch.long, device=device)
    act_imgs = strips.index_select(0, idx_t).repeat_interleave(k_cands, dim=0)
    act_mem_mask = (
        mem_mask.index_select(0, idx_t).repeat_interleave(k_cands, dim=0)
        if mem_mask is not None
        else None
    )

    bos = torch.full((n_total, 1), model.bos_id, dtype=torch.long, device=device)
    inp = torch.cat([bos, tgt[:, :-1]], dim=1)

    # Differentiable student forward pass.
    student_logits = model(act_imgs, inp, mem_mask=act_mem_mask)  # [N_total, T, V]

    # Frozen reference model forward pass (no grad).
    ref_was_training = ref_model.training
    with torch.no_grad():
        ref_model.eval()
        ref_logits = ref_model(act_imgs, inp, mem_mask=act_mem_mask)
    ref_model.train(ref_was_training)

    log_p = F.log_softmax(student_logits.float(), dim=-1)
    log_q = F.log_softmax(ref_logits.float(), dim=-1)

    valid_mask = tgt != model.pad_id  # [N_total, T]
    valid_f = valid_mask.to(log_p.dtype)
    tok_counts = valid_f.sum(dim=1).clamp_min(1.0)  # [N_total]

    safe_tgt = tgt.clamp(min=0, max=log_p.shape[-1] - 1)
    tok_logp = log_p.gather(-1, safe_tgt.unsqueeze(-1)).squeeze(-1)  # [N_total, T]
    seq_logp = (tok_logp * valid_f).sum(dim=1) / tok_counts  # [N_total]

    adv_vec = torch.tensor(flat_advantages, device=device, dtype=seq_logp.dtype)
    pg_loss = -(adv_vec.detach() * seq_logp).mean()

    # Token-level KL(student || ref) over non-PAD target positions.
    p = log_p.exp()
    kl_tok = (p * (log_p - log_q)).sum(dim=-1).clamp_min(0.0)  # [N_total, T]
    kl_token_mean = (kl_tok * valid_f).sum() / valid_f.sum().clamp_min(1.0)

    total_loss = pg_loss + float(kl_beta) * kl_token_mean

    stats = {
        "batch_size": batch_size,
        "num_beams": k_cands,
        "active_strips": len(active_indices),
        "active_frac": float(len(active_indices)) / float(batch_size),
        "active_indices": active_indices,
        "pg_loss": float(pg_loss.detach().item()),
        "kl_loss": float(kl_token_mean.detach().item()),
        "mean_reward": mean_reward,
        "mean_valid_bar_ratio": mean_bm_ratio,
        "degenerate_frac": deg_frac,
        "rewards": all_rewards,
        "advantages": all_advantages,
    }
    return total_loss, stats


def run_grpo_bar_math_round(
    net: nn.Module,
    opt: torch.optim.Optimizer,
    scaler,
    strip_paths: list,
    step: int = 0,
    steps: int = 50,
    weight: float = 0.25,
    kl_beta: float = 0.08,
    height: int = 256,
    device: torch.device | str = "cpu",
    amp: bool = False,
    amp_dtype=torch.bfloat16,
    seed: int = 0,
    batch_size: int = 2,
    num_beams: int = 4,
    max_len: int = 192,
    visual_scorer=None,
    w_bm: float = 0.75,
    w_vis: float = 0.25,
    w_deg: float = 0.50,
    **_extra,
) -> dict:
    """Execute one verifier-guided GRPO bar-math unsupervised round on unlabeled strips."""
    import copy
    import numpy as np
    from PIL import Image
    import data as _mvd_data

    if not strip_paths or int(steps) <= 0:
        return {
            "ran_steps": 0,
            "active_frac": 0.0,
            "mean_loss": 0.0,
            "mean_reward": 0.0,
            "mean_kl": 0.0,
        }

    dev = torch.device(device)
    ref_model = copy.deepcopy(net).to(dev).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    rng = np.random.default_rng(int(seed))
    n_imgs = len(strip_paths)
    in_ch = int(getattr(net, "in_chans", 4))
    was_train = net.training
    net.train()
    use_amp = bool(amp and dev.type == "cuda")
    total_loss = 0.0
    total_active = 0.0
    total_reward = 0.0
    total_kl = 0.0
    ran = 0
    bs = max(1, int(batch_size))

    for _ in range(int(steps)):
        idxs = rng.integers(0, n_imgs, size=bs)
        imgs = []
        max_w = 0
        for idx in idxs:
            try:
                im = Image.open(strip_paths[int(idx)]).convert("RGB")
                ow, oh = im.size
                nw = max(32, min(1536, int(round(ow * (height / max(1, oh))))))
                im = im.resize((nw, int(height)), Image.BOX)
                arr = np.asarray(im, dtype=np.float32) / 255.0
                t = (torch.from_numpy(arr).permute(2, 0, 1) - 0.85) / 0.25
                if in_ch > 3:
                    pad_c = torch.zeros((in_ch - 3, int(height), nw), dtype=torch.float32)
                    t = torch.cat([t, pad_c], dim=0)
                imgs.append(t)
                max_w = max(max_w, nw)
            except Exception:
                continue
        if not imgs:
            continue
        padded = []
        for t in imgs:
            pw = max_w - t.shape[2]
            if pw > 0:
                t = F.pad(t, (0, pw, 0, 0), value=0.6)
            padded.append(t)
        x = torch.stack(padded, dim=0).to(dev, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            step_loss, stats = grpo_bar_math_step(
                net,
                ref_model,
                x,
                id2tok=_mvd_data.VOCAB,
                num_beams=int(num_beams),
                kl_beta=float(kl_beta),
                max_len=int(max_len),
                visual_scorer=visual_scorer,
                w_bm=float(w_bm),
                w_vis=float(w_vis),
                w_deg=float(w_deg),
            )
            loss = step_loss * float(weight)
        active = float(stats.get("active_frac", 0.0))
        total_active += active
        total_reward += float(stats.get("mean_reward", 0.0))
        total_kl += float(stats.get("kl_loss", 0.0))
        if active > 0.0 and torch.isfinite(loss) and loss.requires_grad:
            if scaler is not None and use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            total_loss += float(loss.detach().item())
        ran += 1

    net.train(was_train)
    mean_active = total_active / max(1, ran)
    mean_loss = total_loss / max(1, ran)
    mean_reward = total_reward / max(1, ran)
    mean_kl = total_kl / max(1, ran)
    print(
        f"  [unsup-grpo-bar-math step {step}] ran_steps={ran}/{steps} "
        f"active={mean_active:.2%} mean_loss={mean_loss:.4f} "
        f"mean_reward={mean_reward:.4f} mean_kl={mean_kl:.4f}",
        flush=True,
    )
    return {
        "ran_steps": ran,
        "active_frac": mean_active,
        "mean_loss": mean_loss,
        "mean_reward": mean_reward,
        "mean_kl": mean_kl,
    }
