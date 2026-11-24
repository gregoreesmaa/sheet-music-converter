"""3-Tier Closed-Loop Verifier for MvdNet OMR decoding.

Implements a fast, opt-in verification & selective-repair cascade:

1. **Tier 1 (Symbolic Gate, <0.5 ms):**
   Reuses :func:`unsup_bar_math.score_bekern_bar_math` (exact beat-sum bar
   math + anti-degeneracy guard) plus balanced tie / slur / volta spanner
   verification.
2. **Tier 2 (Visual Consistency Rerank):**
   Conditionally scores top-K beam candidates with visual consistency
   (:mod:`render_diff_verify` when available, pluggable ``visual_scorer``
   callable, or fast numpy ink-density/notehead-column correlation fallback)
   ONLY when ``len(candidates) >= 2`` and either beam-0 fails Tier 1 or the
   top-2 LM log-prob margin is small.
3. **Tier 3 (Selective Geometric / COME LN-TTA Repair + Review Flagging):**
   Triggers ONLY when the Tier-2 winner still has ``valid_bar_ratio < 0.95``,
   ``is_degenerate``, or ``margin < margin_thresh``. Runs two lightweight
   geometric micro-perturbations (``+2px`` vertical shift and ``+2%``
   horizontal scale) and optional ``COME`` LayerNorm TTA through ``decode_fn``,
   adopting a repair ONLY if it strictly improves combined score. If the final
   winner still fails confidence thresholds, marks ``needs_review=True`` and
   populates ``low_confidence_measures`` with exact failing bar indices.
4. **Preference-Pair Mining:**
   :func:`maybe_build_preference_pair` emits structured ``(winner, loser)``
   records whenever Tier 1+2 overrules greedy beam-0 by ``>= min_margin``
   for offline Beam-DPO / GRPO training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

_MVD_DIR = str(Path(__file__).resolve().parent)
if _MVD_DIR not in sys.path:
    sys.path.insert(0, _MVD_DIR)

from unsup_bar_math import (  # noqa: E402
    BarMathResult,
    DURATION_BEATS,
    BAR_CLOSE_TOKENS,
    score_bekern_bar_math,
)

try:
    import render_diff_verify as _rdv
except Exception:  # pragma: no cover - optional dependency fallback
    _rdv = None

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - pure-numpy environments
    torch = None
    F = None

__all__ = [
    "Tier1Result",
    "ScoredCandidate",
    "Tier2Result",
    "Tier3Result",
    "VerifierStripOutput",
    "is_degenerate_bekern",
    "check_balanced_spanners",
    "tier1_symbolic_gate",
    "tier2_visual_rerank",
    "tier3_selective_repair",
    "maybe_build_preference_pair",
    "ClosedLoopVerifier",
]


def _normalize_tokens(bekern: Union[str, Sequence[str]]) -> list[str]:
    if isinstance(bekern, str):
        return bekern.split()
    return [str(t) for t in bekern if str(t)]


def is_degenerate_bekern(bekern: Union[str, Sequence[str]]) -> Tuple[bool, Optional[str]]:
    """Return ``(is_degenerate, reason)`` using :func:`score_bekern_bar_math`."""
    bm = score_bekern_bar_math(_normalize_tokens(bekern))
    return bool(bm.is_degenerate), bm.degenerate_reason


def check_balanced_spanners(tokens: Sequence[str]) -> Tuple[int, dict[str, int]]:
    """Count unbalanced tie, slur, and volta spanners in a bekern token stream.

    Returns ``(total_unbalanced, breakdown_dict)``.
    """
    tie_depth = 0
    slur_depth = 0
    volta_depth = 0
    tie_err = 0
    slur_err = 0
    volta_err = 0

    for tok in tokens:
        if tok == "tie-start":
            tie_depth += 1
        elif tok == "tie-stop":
            if tie_depth > 0:
                tie_depth -= 1
            else:
                tie_err += 1
        elif tok == "slur-start":
            slur_depth += 1
        elif tok == "slur-stop":
            if slur_depth > 0:
                slur_depth -= 1
            else:
                slur_err += 1
        elif tok.startswith("volta-"):
            if tok.endswith("-stop") or tok == "volta-stop":
                if volta_depth > 0:
                    volta_depth -= 1
                else:
                    volta_err += 1
            else:
                volta_depth += 1
        elif tok == "bar-repeat-right" and volta_depth > 0:
            volta_depth -= 1

    tie_unbal = tie_err + tie_depth
    slur_unbal = slur_err + slur_depth
    volta_unbal = volta_err + volta_depth
    total = tie_unbal + slur_unbal + volta_unbal
    return total, {
        "tie": tie_unbal,
        "slur": slur_unbal,
        "volta": volta_unbal,
    }


@dataclass
class Tier1Result:
    """Fast symbolic verification result for one bekern hypothesis."""

    valid_bar_ratio: float
    n_bars: int
    n_bad_bars: int
    bad_bar_indices: list[int]
    is_degenerate: bool
    unbalanced_spanners: int
    score: float
    mean_beat_err: float = 0.0
    degenerate_reason: Optional[str] = None
    unbalanced_details: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (
            not self.is_degenerate
            and self.n_bars > 0
            and self.valid_bar_ratio >= 0.95
            and self.unbalanced_spanners == 0
        )


def tier1_symbolic_gate(
    bekern_str: Union[str, Sequence[str]],
    *,
    allow_pickup: bool = True,
    allow_final_incomplete: bool = True,
) -> Tier1Result:
    """Fast (<0.5 ms) symbolic gate over bar math, degeneracy, and spanners."""
    toks = _normalize_tokens(bekern_str)
    bm: BarMathResult = score_bekern_bar_math(
        toks,
        allow_pickup=allow_pickup,
        allow_final_incomplete=allow_final_incomplete,
    )
    bad_bar_indices = [
        int(d["bar_index"])
        for d in bm.per_bar_details
        if not bool(d.get("is_valid", False))
    ]
    n_bad_bars = len(bad_bar_indices)
    unbal_total, unbal_details = check_balanced_spanners(toks)

    if bm.is_degenerate or bm.n_bars == 0:
        score = 0.0
    else:
        err_term = max(0.0, 1.0 - min(1.0, float(bm.mean_beat_err)))
        bar_score = 0.75 * float(bm.valid_bar_ratio) + 0.25 * err_term
        spanner_penalty = min(0.35, 0.10 * float(unbal_total))
        score = max(0.0, min(1.0, bar_score - spanner_penalty))

    return Tier1Result(
        valid_bar_ratio=float(bm.valid_bar_ratio),
        n_bars=int(bm.n_bars),
        n_bad_bars=n_bad_bars,
        bad_bar_indices=bad_bar_indices,
        is_degenerate=bool(bm.is_degenerate),
        unbalanced_spanners=int(unbal_total),
        score=round(float(score), 6),
        mean_beat_err=float(bm.mean_beat_err),
        degenerate_reason=bm.degenerate_reason,
        unbalanced_details=unbal_details,
    )


def _to_obs_gray(strip_tensor_or_img: Any) -> Optional[np.ndarray]:
    """Convert PIL Image, numpy array, or torch Tensor to 2D uint8 grayscale."""
    if strip_tensor_or_img is None:
        return None
    if isinstance(strip_tensor_or_img, Image.Image):
        return np.asarray(strip_tensor_or_img.convert("L"), dtype=np.uint8)
    if torch is not None and isinstance(strip_tensor_or_img, torch.Tensor):
        t = strip_tensor_or_img.detach().cpu().float()
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3:
            # [C, H, W] -> average first <=3 RGB channels
            c = min(3, int(t.shape[0]))
            arr = t[:c].mean(dim=0).numpy()
        elif t.ndim == 2:
            arr = t.numpy()
        else:
            return None
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo > 1e-6:
            arr = (arr - lo) / (hi - lo) * 255.0
        else:
            arr = np.full_like(arr, 255.0)
        return np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    arr = np.asarray(strip_tensor_or_img)
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
            arr = arr[: min(3, arr.shape[0])].mean(axis=0)
        else:
            arr = arr[..., :3].mean(axis=2)
    if arr.ndim != 2 or arr.size == 0:
        return None
    arr_f = arr.astype(np.float32)
    if float(arr_f.max()) <= 1.0 and float(arr_f.min()) >= 0.0:
        arr_f = arr_f * 255.0
    return np.clip(np.rint(arr_f), 0, 255).astype(np.uint8)


def _fallback_ink_density_score(
    obs_gray: Optional[np.ndarray],
    tokens: Sequence[str],
    n_bins: int = 32,
) -> float:
    """Fast numpy fallback visual score via ink-column / notehead density correlation."""
    if obs_gray is None or obs_gray.size == 0 or not tokens:
        return 0.0
    h, w = obs_gray.shape
    if h < 4 or w < 8:
        return 0.0

    # Estimate dark-ink mask relative to local paper quantile.
    paper = float(np.percentile(obs_gray, 85))
    thresh = max(80.0, paper - 25.0)
    ink = (obs_gray.astype(np.float32) < thresh).astype(np.float32)

    # Subtract median horizontal staff-line contribution per column so
    # vertical symbol/notehead density dominates.
    row_mean = ink.mean(axis=1)
    staff_rows = row_mean > 0.45
    if staff_rows.any() and (~staff_rows).any():
        ink_sym = ink[~staff_rows]
    else:
        ink_sym = ink
    col_profile = ink_sym.mean(axis=0)
    if float(col_profile.sum()) <= 1e-6:
        return 0.0

    # Bin observed ink profile into n_bins.
    bins = min(int(n_bins), w)
    edges = np.linspace(0, w, bins + 1, dtype=int)
    obs_vec = np.array(
        [float(col_profile[edges[i] : max(edges[i] + 1, edges[i + 1])].mean()) for i in range(bins)],
        dtype=np.float32,
    )

    # Build expected event density profile from bekern tokens.
    events: list[Tuple[float, float]] = []  # (beat_pos, weight)
    beat = 0.0
    expect_rest = False
    is_chord = False
    for tok in tokens:
        if tok == "rest":
            expect_rest = True
        elif tok == "chord":
            is_chord = True
        elif tok == "bar-rest":
            events.append((beat, 0.35))
            beat += 4.0
        elif tok in BAR_CLOSE_TOKENS:
            events.append((beat, 0.5))
        elif tok in DURATION_BEATS:
            dur = DURATION_BEATS[tok]
            if expect_rest:
                events.append((beat, 0.35))
                beat += dur
                expect_rest = False
            elif is_chord:
                events.append((beat, 0.85))
                is_chord = False
            else:
                events.append((beat, 1.0))
                beat += dur
        elif tok == "dot":
            beat += 0.25

    if not events:
        return 0.0
    total_beats = max(beat, 1.0)
    exp_vec = np.zeros(bins, dtype=np.float32)
    for pos, wt in events:
        b_idx = min(bins - 1, max(0, int((pos / total_beats) * bins)))
        exp_vec[b_idx] += wt

    norm_o = float(np.linalg.norm(obs_vec))
    norm_e = float(np.linalg.norm(exp_vec))
    if norm_o <= 1e-8 or norm_e <= 1e-8:
        return 0.0
    cos_sim = float(np.dot(obs_vec, exp_vec) / (norm_o * norm_e))
    return max(0.0, min(1.0, cos_sim))


def _default_visual_score(
    strip_tensor_or_img: Any,
    tokens: Sequence[str],
) -> Tuple[float, bool]:
    """Score visual consistency in [0, 1]. Returns ``(score, used_fallback)``."""
    obs_gray = _to_obs_gray(strip_tensor_or_img)
    if obs_gray is None:
        return 0.0, True

    if _rdv is not None:
        try:
            ren = _rdv.bekern_to_strip_png(tokens)
            if ren is not None:
                n_bars = _rdv._count_bekern_bars(tokens)
                vis = _rdv.visual_similarity(obs_gray, ren, n_bars=n_bars)
                return max(0.0, min(1.0, float(vis["iou"]))), False
        except Exception:
            pass

    return _fallback_ink_density_score(obs_gray, tokens), True


@dataclass
class ScoredCandidate:
    """Single beam hypothesis scored by Tier 1 and (optionally) Tier 2."""

    index: int
    tokens: list[str]
    logp: float
    lm_norm: float
    t1: Tier1Result
    t2_visual_score: float
    combined_score: float


@dataclass
class Tier2Result:
    """Ranked output of :func:`tier2_visual_rerank`."""

    ranked: list[ScoredCandidate]
    margin: float
    tier2_triggered: bool
    used_fallback_visual: bool = False

    @property
    def best(self) -> ScoredCandidate:
        return self.ranked[0]

    @property
    def t1(self) -> Tier1Result:
        return self.ranked[0].t1

    @property
    def combined_score(self) -> float:
        return self.ranked[0].combined_score


def tier2_visual_rerank(
    strip_tensor_or_img: Any,
    candidates: Sequence[Tuple[Union[str, Sequence[str]], float]],
    *,
    visual_scorer: Optional[Callable[[Any, list[str]], float]] = None,
    w_lm: float = 0.45,
    w_t1: float = 0.35,
    w_t2: float = 0.20,
    lm_margin_thresh: float = 0.15,
    force_visual: bool = False,
) -> Tier2Result:
    """Run Tier 1 on all candidates and conditionally run Tier 2 visual rerank.

    Visual scoring runs ONLY when ``len(candidates) >= 2`` and either beam-0
    fails Tier 1 or the top-2 LM log-prob margin is below ``lm_margin_thresh``
    (unless ``force_visual=True``).
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")

    norm_cands: list[Tuple[int, list[str], float, Tier1Result]] = []
    for idx, (toks_raw, lp) in enumerate(candidates):
        toks = _normalize_tokens(toks_raw)
        t1 = tier1_symbolic_gate(toks)
        norm_cands.append((idx, toks, float(lp), t1))

    lps = [c[2] for c in norm_cands]
    max_lp = max(lps)
    lm_norms = [math.exp(max(-20.0, lp - max_lp)) for lp in lps]

    # Decide whether Tier 2 visual consistency scoring should fire.
    beam0_t1 = norm_cands[0][3]
    if len(norm_cands) >= 2:
        sorted_lps = sorted(lps, reverse=True)
        lm_margin = sorted_lps[0] - sorted_lps[1]
        should_run_t2 = bool(
            force_visual
            or (not beam0_t1.passed)
            or (lm_margin < float(lm_margin_thresh))
        )
    else:
        should_run_t2 = bool(force_visual and strip_tensor_or_img is not None)

    used_fallback = False
    scored: list[ScoredCandidate] = []
    w_lm_f = float(w_lm)
    w_t1_f = float(w_t1)
    w_t2_f = float(w_t2)
    w_no_vis = max(1e-8, w_lm_f + w_t1_f)

    for (idx, toks, lp, t1), lm_n in zip(norm_cands, lm_norms):
        if should_run_t2:
            if visual_scorer is not None:
                v_score = max(0.0, min(1.0, float(visual_scorer(strip_tensor_or_img, toks))))
            else:
                v_score, fb = _default_visual_score(strip_tensor_or_img, toks)
                used_fallback = used_fallback or fb
            comb = w_lm_f * lm_n + w_t1_f * t1.score + w_t2_f * v_score
        else:
            v_score = 0.0
            comb = (w_lm_f * lm_n + w_t1_f * t1.score) / w_no_vis

        scored.append(
            ScoredCandidate(
                index=idx,
                tokens=toks,
                logp=lp,
                lm_norm=round(lm_n, 6),
                t1=t1,
                t2_visual_score=round(v_score, 6),
                combined_score=round(comb, 6),
            )
        )

    # Deterministic descending sort by combined_score, tie-break by lower beam index.
    scored.sort(key=lambda c: (-c.combined_score, c.index))

    if len(scored) >= 2:
        margin = round(float(scored[0].combined_score - scored[1].combined_score), 6)
    else:
        margin = 1.0 if scored[0].t1.passed else 0.0

    return Tier2Result(
        ranked=scored,
        margin=margin,
        tier2_triggered=should_run_t2,
        used_fallback_visual=used_fallback,
    )


def _geometric_micro_perturbations(strip_tensor: Any) -> list[Tuple[str, Any]]:
    """Generate 2 lightweight geometric micro-perturbations preserving (H, W).

    1. ``shift_v+2``: ``+2px`` vertical roll with top-row edge fill.
    2. ``scale_h+2pct``: ``+2%`` horizontal scale via bilinear interpolate,
       center-cropped back to original width ``W``.
    """
    if torch is None or not isinstance(strip_tensor, torch.Tensor):
        return []
    if strip_tensor.ndim not in (3, 4):
        return []

    was_3d = strip_tensor.ndim == 3
    x = strip_tensor.unsqueeze(0) if was_3d else strip_tensor
    _, _, h, w = x.shape
    if h < 4 or w < 4:
        return []

    out: list[Tuple[str, Any]] = []

    # Perturbation 1: +2px vertical shift with edge-row padding.
    shift_px = min(2, max(1, h // 16))
    p_shift = torch.roll(x, shifts=shift_px, dims=-2)
    p_shift[..., :shift_px, :] = x[..., :1, :]
    out.append(("shift_v+2", p_shift.squeeze(0) if was_3d else p_shift))

    # Perturbation 2: +2% horizontal scale, center-cropped back to W.
    new_w = max(w + 1, int(round(w * 1.02)))
    scaled = F.interpolate(
        x.float(), size=(h, new_w), mode="bilinear", align_corners=False
    ).to(dtype=x.dtype)
    left = (new_w - w) // 2
    p_scale = scaled[..., :, left : left + w]
    out.append(("scale_h+2pct", p_scale.squeeze(0) if was_3d else p_scale))

    return out


def _normalize_decode_output(raw: Any) -> list[Tuple[list[str], float]]:
    """Normalize ``decode_fn`` return value into ``[(tokens, logp), ...]``."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [(_normalize_tokens(raw), 0.0)]
    if isinstance(raw, tuple) and len(raw) == 2 and not isinstance(raw[1], (list, tuple)):
        return [(_normalize_tokens(raw[0]), float(raw[1]))]
    if isinstance(raw, list):
        if not raw:
            return []
        if isinstance(raw[0], str):
            return [(list(raw), 0.0)]
        cands: list[Tuple[list[str], float]] = []
        for item in raw:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                cands.append((_normalize_tokens(item[0]), float(item[1])))
            elif isinstance(item, str):
                cands.append((_normalize_tokens(item), 0.0))
        return cands
    return []


@dataclass
class Tier3Result:
    """Output of :func:`tier3_selective_repair`."""

    winner: ScoredCandidate
    margin: float
    triggered: bool
    repaired: bool
    needs_review: bool
    low_confidence_measures: list[int]
    repair_source: str = "none"


def _compute_review_flags(
    winner: ScoredCandidate,
    margin: float,
    margin_thresh: float,
) -> Tuple[bool, list[int]]:
    """Determine ``(needs_review, low_confidence_measures)`` for the final winner."""
    low_bar = winner.t1.valid_bar_ratio < 0.95
    degen = winner.t1.is_degenerate
    low_margin = margin < float(margin_thresh)
    if not (low_bar or degen or low_margin):
        return False, []
    if winner.t1.bad_bar_indices:
        return True, list(winner.t1.bad_bar_indices)
    if winner.t1.n_bars > 0:
        return True, list(range(winner.t1.n_bars))
    return True, [0]


def tier3_selective_repair(
    decode_fn: Optional[Callable[[Any], Any]],
    strip_tensor: Any,
    best_result: Tier2Result,
    *,
    margin_thresh: float = 0.06,
    tta_model: Any = None,
    tta_mode: str = "come",
    visual_scorer: Optional[Callable[[Any, list[str]], float]] = None,
    w_lm: float = 0.45,
    w_t1: float = 0.35,
    w_t2: float = 0.20,
    tta_steps: int = 1,
    tta_lr: float = 1e-4,
) -> Tier3Result:
    """Selective Tier-3 repair triggered only on low-confidence or broken strips.

    Triggers ONLY when ``best_result.t1.valid_bar_ratio < 0.95`` OR
    ``best_result.t1.is_degenerate`` OR ``best_result.margin < margin_thresh``.
    """
    base_winner = best_result.best
    needs_trigger = bool(
        base_winner.t1.valid_bar_ratio < 0.95
        or base_winner.t1.is_degenerate
        or best_result.margin < float(margin_thresh)
    )
    if not needs_trigger:
        return Tier3Result(
            winner=base_winner,
            margin=best_result.margin,
            triggered=False,
            repaired=False,
            needs_review=False,
            low_confidence_measures=[],
            repair_source="none",
        )

    # Establish apples-to-apples baseline score (with visual score if Tier 2 hadn't run).
    if best_result.tier2_triggered:
        baseline_cand = base_winner
    else:
        rescored_base = tier2_visual_rerank(
            strip_tensor,
            [(base_winner.tokens, base_winner.logp)],
            visual_scorer=visual_scorer,
            w_lm=w_lm,
            w_t1=w_t1,
            w_t2=w_t2,
            force_visual=True,
        )
        baseline_cand = rescored_base.best

    best_cand = baseline_cand
    repair_source = "none"
    repaired = False
    all_repair_scores: list[float] = [c.combined_score for c in best_result.ranked]

    if decode_fn is not None:
        repair_attempts: list[Tuple[str, list[Tuple[list[str], float]]]] = []

        # 1. Two lightweight geometric micro-perturbations.
        for pname, pert in _geometric_micro_perturbations(strip_tensor):
            try:
                cands = _normalize_decode_output(decode_fn(pert))
                if cands:
                    repair_attempts.append((pname, cands))
            except Exception:
                continue

        # 2. Optional COME / COME-SAR episodic LayerNorm TTA.
        if (
            tta_model is not None
            and tta_mode
            and tta_mode != "none"
            and torch is not None
            and isinstance(strip_tensor, torch.Tensor)
        ):
            try:
                from unsup_stage2 import episodic_page_ln_tta

                with episodic_page_ln_tta(
                    tta_model,
                    strip_tensor,
                    steps=int(tta_steps),
                    lr=float(tta_lr),
                    mode=str(tta_mode),
                ):
                    cands = _normalize_decode_output(decode_fn(strip_tensor))
                    if cands:
                        repair_attempts.append((f"tta_{tta_mode}", cands))
            except Exception:
                pass

        # Score all repair candidates with Tier 1 + Tier 2 and adopt strictly better ones.
        for src_name, cands in repair_attempts:
            t2_rep = tier2_visual_rerank(
                strip_tensor,
                cands,
                visual_scorer=visual_scorer,
                w_lm=w_lm,
                w_t1=w_t1,
                w_t2=w_t2,
                force_visual=True,
            )
            rep_best = t2_rep.best
            all_repair_scores.append(rep_best.combined_score)
            if rep_best.combined_score > best_cand.combined_score:
                best_cand = rep_best
                repair_source = src_name
                repaired = True

    # Compute final margin between winner and runner-up.
    if repaired:
        final_margin = round(
            float(best_cand.combined_score - baseline_cand.combined_score), 6
        )
    else:
        final_margin = best_result.margin

    needs_review, low_conf_measures = _compute_review_flags(
        best_cand, final_margin, margin_thresh
    )

    return Tier3Result(
        winner=best_cand,
        margin=final_margin,
        triggered=True,
        repaired=repaired,
        needs_review=needs_review,
        low_confidence_measures=low_conf_measures,
        repair_source=repair_source,
    )


def maybe_build_preference_pair(
    strip_id: str,
    ranked: Union[Sequence[ScoredCandidate], Tier2Result],
    greedy_idx: int = 0,
    min_margin: float = 0.05,
) -> Optional[dict[str, Any]]:
    """Build a Beam-DPO / GRPO preference pair when Tier 1+2 overrules greedy.

    Returns ``None`` if greedy (``index == greedy_idx``) already won or if
    ``winner_score - loser_score < min_margin``.
    """
    cands = ranked.ranked if isinstance(ranked, Tier2Result) else list(ranked)
    if len(cands) < 2:
        return None
    winner = cands[0]
    if winner.index == int(greedy_idx):
        return None
    loser = next((c for c in cands if c.index == int(greedy_idx)), None)
    if loser is None:
        return None
    margin = float(winner.combined_score) - float(loser.combined_score)
    if margin < float(min_margin):
        return None
    return {
        "strip_id": str(strip_id),
        "winner_tokens": list(winner.tokens),
        "loser_tokens": list(loser.tokens),
        "winner_score": round(float(winner.combined_score), 6),
        "loser_score": round(float(loser.combined_score), 6),
        "margin": round(margin, 6),
        "t1_delta": round(float(winner.t1.score - loser.t1.score), 6),
        "t2_delta": round(
            float(winner.t2_visual_score - loser.t2_visual_score), 6
        ),
    }


@dataclass
class VerifierStripOutput:
    """Complete closed-loop verifier output for one system strip."""

    tokens: list[str]
    winner: ScoredCandidate
    t2_result: Tier2Result
    t3_result: Tier3Result
    needs_review: bool
    low_confidence_measures: list[int]
    preference_pair: Optional[dict[str, Any]]
    latency_ms: float


class ClosedLoopVerifier:
    """Orchestrates Tier 1 -> Tier 2 -> selective Tier 3 with telemetry & DPO mining."""

    def __init__(
        self,
        *,
        visual_scorer: Optional[Callable[[Any, list[str]], float]] = None,
        w_lm: float = 0.45,
        w_t1: float = 0.35,
        w_t2: float = 0.20,
        lm_margin_thresh: float = 0.15,
        margin_thresh: float = 0.06,
        pref_min_margin: float = 0.05,
        tta_mode: str = "come",
        tta_steps: int = 1,
        tta_lr: float = 1e-4,
        pref_out_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.visual_scorer = visual_scorer
        self.w_lm = float(w_lm)
        self.w_t1 = float(w_t1)
        self.w_t2 = float(w_t2)
        self.lm_margin_thresh = float(lm_margin_thresh)
        self.margin_thresh = float(margin_thresh)
        self.pref_min_margin = float(pref_min_margin)
        self.tta_mode = str(tta_mode)
        self.tta_steps = int(tta_steps)
        self.tta_lr = float(tta_lr)
        self.pref_out_path = Path(pref_out_path) if pref_out_path is not None else None

        # Telemetry counters.
        self.n_strips: int = 0
        self.tier1_only: int = 0
        self.tier2_triggered: int = 0
        self.tier3_triggered: int = 0
        self.tier3_repaired: int = 0
        self.flagged_for_review: int = 0
        self.low_confidence_measures_total: int = 0
        self.preference_pairs_emitted: int = 0
        self.total_ms: float = 0.0
        self.preference_pairs: list[dict[str, Any]] = []

    def _emit_preference_pair(self, pair: dict[str, Any]) -> None:
        self.preference_pairs.append(pair)
        self.preference_pairs_emitted += 1
        if self.pref_out_path is not None:
            self.pref_out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.pref_out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(pair) + "\n")

    def verify_strip(
        self,
        strip_id: str,
        strip_tensor_or_img: Any,
        candidates: Sequence[Tuple[Union[str, Sequence[str]], float]],
        *,
        decode_fn: Optional[Callable[[Any], Any]] = None,
        tta_model: Any = None,
    ) -> VerifierStripOutput:
        """Run Tier 1 -> Tier 2 -> selective Tier 3 for one system strip."""
        t0 = time.perf_counter()

        t2_res = tier2_visual_rerank(
            strip_tensor_or_img,
            candidates,
            visual_scorer=self.visual_scorer,
            w_lm=self.w_lm,
            w_t1=self.w_t1,
            w_t2=self.w_t2,
            lm_margin_thresh=self.lm_margin_thresh,
        )

        pref_pair = maybe_build_preference_pair(
            strip_id,
            t2_res.ranked,
            greedy_idx=0,
            min_margin=self.pref_min_margin,
        )
        if pref_pair is not None:
            self._emit_preference_pair(pref_pair)

        t3_res = tier3_selective_repair(
            decode_fn,
            strip_tensor_or_img,
            t2_res,
            margin_thresh=self.margin_thresh,
            tta_model=tta_model,
            tta_mode=self.tta_mode,
            visual_scorer=self.visual_scorer,
            w_lm=self.w_lm,
            w_t1=self.w_t1,
            w_t2=self.w_t2,
            tta_steps=self.tta_steps,
            tta_lr=self.tta_lr,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.n_strips += 1
        self.total_ms += elapsed_ms
        if t2_res.tier2_triggered:
            self.tier2_triggered += 1
        if t3_res.triggered:
            self.tier3_triggered += 1
        if t3_res.repaired:
            self.tier3_repaired += 1
        if not t2_res.tier2_triggered and not t3_res.triggered:
            self.tier1_only += 1
        if t3_res.needs_review:
            self.flagged_for_review += 1
            self.low_confidence_measures_total += len(t3_res.low_confidence_measures)

        return VerifierStripOutput(
            tokens=list(t3_res.winner.tokens),
            winner=t3_res.winner,
            t2_result=t2_res,
            t3_result=t3_res,
            needs_review=t3_res.needs_review,
            low_confidence_measures=list(t3_res.low_confidence_measures),
            preference_pair=pref_pair,
            latency_ms=round(elapsed_ms, 3),
        )

    def telemetry_dict(self) -> dict[str, Any]:
        """Return JSON-serializable telemetry summary."""
        return {
            "n_strips": self.n_strips,
            "tier1_only": self.tier1_only,
            "tier2_triggered": self.tier2_triggered,
            "tier3_triggered": self.tier3_triggered,
            "tier3_repaired": self.tier3_repaired,
            "flagged_for_review": self.flagged_for_review,
            "low_confidence_measures_total": self.low_confidence_measures_total,
            "preference_pairs_emitted": self.preference_pairs_emitted,
            "mean_ms": round(self.total_ms / max(1, self.n_strips), 3),
        }
