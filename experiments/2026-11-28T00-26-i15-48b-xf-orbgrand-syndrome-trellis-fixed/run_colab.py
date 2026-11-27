#!/usr/bin/env python3
"""Colab A100 / Local RTX 4080 SUPER runner — Iter 15 Exp i15-48b: Fixed ORBGRAND/SOGRAND + Trellis-BMA.

Experiment ID: 2026-11-28T00-26-i15-48b-xf-orbgrand-syndrome-trellis-fixed
Origin: Information Theory / IDS Coding (ORBGRAND / SOGRAND IEEE TWC 2024-25 + Trellis-BMA).
Clean ITERATE successor to i15-48 fixing the two forensic findings in i15-48/OUTCOME.md:
  1. Multi-staff / multi-voice `compute_bar_duration_syndrome()`: segments each bar by
     (staff_idx, voice_id) on `<staff>`/`staff-*` and `voice-*` instead of breaking on
     leading `voice-1`, computing exact per-segment beat conservation (`bar-rest`, `chord`,
     `dot` 1.5x/chained, `tuplet` scaling, skipping `grace`) and returning non-zero syndrome
     whenever ANY (staff, voice) segment violates `bar_capacity`.
  2. Full MusicXML OMR scoring (`note_f1`, `bsvr`, `wfpp`, and `gate_pass`) using the clean
     Rust (`assembly-serialize` + `eval-harness`) + pure-Python MusicXML ElementTree scorer
     from i15-44b.

Zero-training 3-arm inference proxy on a 12-page hard printed + HW subset of suite31_hw:
  - Arm 0 (greedy_fsa):        Standard FSA-constrained greedy decode.
  - Arm 1 (orbgrand_sogrand):  CPU ORBGRAND ordered-noise search (<=64 patterns/bar) on bars
                               violating per-(staff, voice) duration conservation + SOGRAND
                               analytical "not-found" posterior P_not_found.
  - Arm 2 (trellis_bma_3view): On flagged/high-syndrome pages, decode 3 cheap contrast/scale
                               views and fuse bar-aligned streams via Trellis-BMA consensus.

Includes per-arm HF partial checkpoint sync + startup auto-resume and Windows cuDNN 9.10 safety.
"""

from __future__ import annotations

import argparse
from collections import Counter
import difflib
import gc
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-28T00-26-i15-48b-xf-orbgrand-syndrome-trellis-fixed"
EXP_CODE = "i15-48b"
ARMS = ("greedy_fsa", "orbgrand_sogrand", "trellis_bma_3view")
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_CKPT_CANDIDATES = (
    "checkpoints/iter14-v2a-final/uni_omr_v2_ema_latest.pt",
    "checkpoints/iter14-final/uni_omr_v2_ema_latest.pt",
    "checkpoints/iter14-v2a-mid/uni_omr_v2_ema_latest.pt",
    "checkpoints/iter10_soup_baseline/uni_omr_soup.pt",
)
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"
HF_CKPT_PREFIX = f"checkpoints/{EXP_ID}"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
WORK_ROOT = (
    Path("/vol/i15_48b")
    if Path("/vol").is_dir()
    else (Path("D:/tmp/i15_48b") if sys.platform == "win32" and Path("D:/tmp").is_dir() else Path("/content/i15_48b"))
)

SEED = 42
MAX_DIM = 1408
MAX_LEN = 3072
N_HARD_PRINTED = 6
N_HARD_HW = 6
ORBGRAND_MAX_PATTERNS = 64
ORBGRAND_MAX_FLIPS_PER_BAR = 12
SOGRAND_FLAG_THRESHOLD = 0.15

WRITTEN_DUR_BEATS: Dict[str, float] = {
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
BAR_DELIMS = frozenset({"bar", "bar-light-heavy", "bar-repeat-right", "bar-double"})
BAR_EPS = 1e-4


# ---------------------------------------------------------------------------
# Musical Duration Syndrome, ORBGRAND Ordered-Noise Search & SOGRAND Posterior
# ---------------------------------------------------------------------------

def parse_time_sig_beats(tok: str) -> Optional[float]:
    if not tok.startswith("time-"):
        return None
    parts = tok.split("-")
    if len(parts) != 3:
        return None
    try:
        num = float(parts[1])
        den = float(parts[2])
        if den <= 0:
            return None
        return num * (4.0 / den)
    except ValueError:
        return None


def compute_bar_duration_syndrome(
    bar_toks: Sequence[str],
    bar_capacity: float,
) -> Tuple[float, float, bool]:
    """Compute exact per-(staff, voice) duration sum and conservation syndrome for one bar.

    Segments the bar by `(staff_idx, voice_id)` whenever `<staff>`, `staff-*`, or `voice-*`
    tokens appear (never breaking on leading `voice-1`). Handles:
      - `bar-rest` = full `bar_capacity` for the current `(staff, voice)` segment
      - `chord` = 0 duration advance
      - `grace` = 0 duration advance (skipped)
      - `dot` = 1.5x (chained: +0.5x, +0.25x, ...) on the preceding non-chord/non-grace note/rest
      - `tuplet-start-A-N` ... `tuplet-stop` scaling (`normal / actual`)

    Returns `(measured_beats, abs_syndrome, is_valid)` where:
      - `is_valid` is False whenever ANY rhythmic `(staff, voice)` segment violates `bar_capacity`
      - `abs_syndrome` is the sum of per-segment `|sum - bar_capacity|` across violating segments
      - `measured_beats` is the beat sum of the worst-violating segment (or first segment if valid)
    """
    if not bar_toks:
        return 0.0, 0.0, True

    staff_idx = 0
    voice_id = "voice-1"
    # Per (staff_idx, voice_id): {"total": float, "has_rhythm": bool, "has_bar_rest": bool}
    segments: Dict[Tuple[int, str], Dict[str, Any]] = {}
    seg_order: List[Tuple[int, str]] = []

    def _get_seg(key: Tuple[int, str]) -> Dict[str, Any]:
        if key not in segments:
            segments[key] = {"total": 0.0, "has_rhythm": False, "has_bar_rest": False}
            seg_order.append(key)
        return segments[key]

    in_chord = False
    in_grace = False
    tuplet_stack: List[float] = []
    last_base_dur: Optional[float] = None
    last_dot_factor = 0.5

    for tok in bar_toks:
        if tok == "<staff>":
            staff_idx += 1
            voice_id = "voice-1"
            in_chord = False
            in_grace = False
            tuplet_stack.clear()
            last_base_dur = None
            continue
        if tok.startswith("staff-"):
            try:
                staff_idx = int(tok.split("-", 1)[1])
            except ValueError:
                staff_idx += 1
            voice_id = "voice-1"
            in_chord = False
            in_grace = False
            tuplet_stack.clear()
            last_base_dur = None
            continue
        if tok.startswith("voice-"):
            voice_id = tok
            in_chord = False
            in_grace = False
            tuplet_stack.clear()
            last_base_dur = None
            continue
        if tok == "bar-rest":
            seg = _get_seg((staff_idx, voice_id))
            seg["has_bar_rest"] = True
            seg["has_rhythm"] = True
            in_chord = False
            in_grace = False
            last_base_dur = None
            continue
        if tok == "chord":
            in_chord = True
            continue
        if tok == "grace":
            in_grace = True
            continue
        if tok.startswith("tuplet-start-"):
            parts = tok.split("-")
            if len(parts) >= 4:
                try:
                    actual, normal = float(parts[2]), float(parts[3])
                    if actual > 0:
                        tuplet_stack.append(normal / actual)
                except ValueError:
                    pass
            continue
        if tok == "tuplet-stop":
            if tuplet_stack:
                tuplet_stack.pop()
            continue
        if tok in WRITTEN_DUR_BEATS:
            scale = 1.0
            for s in tuplet_stack:
                scale *= s
            dur = WRITTEN_DUR_BEATS[tok] * scale
            if not in_chord and not in_grace:
                seg = _get_seg((staff_idx, voice_id))
                seg["total"] += dur
                seg["has_rhythm"] = True
                last_base_dur = dur
                last_dot_factor = 0.5
            else:
                last_base_dur = None
            in_chord = False
            in_grace = False
            continue
        if tok == "dot" and last_base_dur is not None:
            seg = _get_seg((staff_idx, voice_id))
            add = last_base_dur * last_dot_factor
            seg["total"] += add
            seg["has_rhythm"] = True
            last_dot_factor *= 0.5
            continue

    rhythmic_segs = [segments[k] for k in seg_order if segments[k]["has_rhythm"]]
    if not rhythmic_segs:
        # Header-only or non-rhythmic bar (no duration tokens or bar-rest)
        return 0.0, 0.0, True

    total_syndrome = 0.0
    worst_syn = -1.0
    worst_beats = 0.0
    all_valid = True

    for seg in rhythmic_segs:
        seg_beats = bar_capacity if seg["has_bar_rest"] else float(seg["total"])
        seg_syn = abs(seg_beats - bar_capacity)
        if seg_syn > BAR_EPS:
            all_valid = False
            total_syndrome += seg_syn
        if seg_syn > worst_syn:
            worst_syn = seg_syn
            worst_beats = seg_beats

    return round(worst_beats, 6), round(total_syndrome, 6), bool(all_valid)


def split_bars_with_spans(toks: Sequence[str]) -> List[Tuple[int, int, float]]:
    """Split a page token stream into (start_idx, end_idx_exclusive, bar_capacity) per bar."""
    spans: List[Tuple[int, int, float]] = []
    cap = 4.0
    bar_start = 0
    for i, t in enumerate(toks):
        ts = parse_time_sig_beats(t)
        if ts is not None:
            cap = ts
        if t in BAR_DELIMS or t == "<sys>":
            if i > bar_start:
                spans.append((bar_start, i, cap))
            bar_start = i + 1
    if bar_start < len(toks):
        spans.append((bar_start, len(toks), cap))
    return spans


def generate_orbgrand_patterns(n_pos: int, max_patterns: int = ORBGRAND_MAX_PATTERNS) -> List[Tuple[int, ...]]:
    """Generate ordered error-position index tuples sorted by ORBGRAND logistic weight sum(i+1).

    Position indices are 0-based (0 = least reliable token, n_pos-1 = most reliable).
    """
    if n_pos <= 0:
        return []
    candidates: List[Tuple[int, int, Tuple[int, ...]]] = []
    # Weight-1 flips
    for i in range(n_pos):
        candidates.append((i + 1, 1, (i,)))
    # Weight-2 flips
    for i in range(n_pos):
        for j in range(i + 1, n_pos):
            candidates.append(((i + 1) + (j + 1), 2, (i, j)))
    # Weight-3 flips (bounded to top-8 least reliable)
    lim3 = min(n_pos, 8)
    for i in range(lim3):
        for j in range(i + 1, lim3):
            for k in range(j + 1, lim3):
                candidates.append(((i + 1) + (j + 1) + (k + 1), 3, (i, j, k)))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return [c[2] for c in candidates[:max_patterns]]


def orbgrand_repair_bar(
    bar_toks: List[str],
    bar_top_cands: List[List[Tuple[str, float]]],
    bar_capacity: float,
    max_patterns: int = ORBGRAND_MAX_PATTERNS,
) -> Dict[str, Any]:
    """Run ORBGRAND ordered-noise search + SOGRAND analytical not-found posterior on one bar.

    bar_top_cands[pos] = [(tok_str, prob), ...] sorted descending (len >= 1).
    """
    _, orig_syn, orig_ok = compute_bar_duration_syndrome(bar_toks, bar_capacity)
    if orig_ok:
        return {
            "tokens": list(bar_toks),
            "repaired": False,
            "orig_syndrome": orig_syn,
            "final_syndrome": orig_syn,
            "queries": 0,
            "p_not_found": 0.0,
            "sogrand_posterior": 1.0,
        }

    # Rank rhythm-relevant positions by reliability margin (lowest top1 - alt first)
    flip_positions: List[Tuple[float, int, str, float, float]] = []
    p_orig_bar = 1.0
    for pos, tok in enumerate(bar_toks):
        cands = bar_top_cands[pos] if pos < len(bar_top_cands) else [(tok, 0.99)]
        p1 = max(1e-6, min(1.0 - 1e-6, float(cands[0][1])))
        p_orig_bar *= p1
        if len(cands) < 2:
            continue
        # Pick highest-prob top-2/top-3 rhythm-relevant alternative
        chosen_alt: Optional[Tuple[str, float]] = None
        for alt_tok, p_raw in cands[1:3]:
            is_rhythm = (
                tok in WRITTEN_DUR_BEATS
                or alt_tok in WRITTEN_DUR_BEATS
                or tok in ("dot", "bar-rest")
                or alt_tok in ("dot", "bar-rest")
            )
            if is_rhythm and alt_tok != tok:
                chosen_alt = (alt_tok, max(1e-6, min(1.0 - 1e-6, float(p_raw))))
                break
        if chosen_alt is not None:
            alt_tok, p2 = chosen_alt
            margin = math.log(p1) - math.log(p2)
            flip_positions.append((margin, pos, alt_tok, p1, p2))

    flip_positions.sort(key=lambda x: x[0])
    flip_positions = flip_positions[:ORBGRAND_MAX_FLIPS_PER_BAR]
    patterns = generate_orbgrand_patterns(len(flip_positions), max_patterns=max_patterns)

    prob_visited = p_orig_bar
    best_toks = list(bar_toks)
    best_syn = orig_syn
    found_prob = 0.0
    repaired = False
    queries = 0

    for pat in patterns:
        queries += 1
        cand_toks = list(bar_toks)
        pat_prob = p_orig_bar
        for r_idx in pat:
            _, pos, alt_tok, p1, p2 = flip_positions[r_idx]
            cand_toks[pos] = alt_tok
            pat_prob *= p2 / max(1e-6, p1)
        prob_visited += pat_prob
        _, syn, ok = compute_bar_duration_syndrome(cand_toks, bar_capacity)
        if syn < best_syn:
            best_syn = syn
            best_toks = cand_toks
        if ok:
            repaired = True
            found_prob = pat_prob
            best_toks = cand_toks
            best_syn = syn
            break

    # SOGRAND analytical "not-found" posterior (IEEE TWC 2024-25):
    p_unvisited = max(0.0, min(1.0, 1.0 - prob_visited))
    if repaired:
        p_other_valid = p_unvisited * 0.125
        denom = max(1e-12, found_prob + p_other_valid)
        sogrand_post = found_prob / denom
        p_not_found = max(0.0, min(1.0, 1.0 - sogrand_post))
    else:
        sogrand_post = 0.0
        p_not_found = 1.0

    return {
        "tokens": best_toks if repaired else list(bar_toks),
        "repaired": repaired,
        "orig_syndrome": orig_syn,
        "final_syndrome": best_syn if repaired else orig_syn,
        "queries": queries,
        "p_not_found": round(p_not_found, 5),
        "sogrand_posterior": round(sogrand_post, 5),
    }


def apply_orbgrand_page(
    toks: List[str],
    top_cands: List[List[Tuple[str, float]]],
    max_patterns: int = ORBGRAND_MAX_PATTERNS,
) -> Dict[str, Any]:
    """Apply ORBGRAND/SOGRAND bar-by-bar across a full page token stream."""
    out_toks = list(toks)
    spans = split_bars_with_spans(toks)
    n_bars = len(spans)
    n_viol_before = 0
    n_viol_after = 0
    n_repaired = 0
    total_queries = 0
    p_not_found_list: List[float] = []

    for start, end, cap in spans:
        bar_slice = toks[start:end]
        cand_slice = top_cands[start:end]
        res = orbgrand_repair_bar(bar_slice, cand_slice, cap, max_patterns=max_patterns)
        if res["orig_syndrome"] > BAR_EPS:
            n_viol_before += 1
        if res["final_syndrome"] > BAR_EPS:
            n_viol_after += 1
        if res["repaired"]:
            n_repaired += 1
            out_toks[start:end] = res["tokens"]
        total_queries += int(res["queries"])
        p_not_found_list.append(float(res["p_not_found"]))

    mean_pnf = sum(p_not_found_list) / max(1, len(p_not_found_list))
    return {
        "tokens": out_toks,
        "n_bars": n_bars,
        "n_viol_before": n_viol_before,
        "n_viol_after": n_viol_after,
        "n_repaired": n_repaired,
        "total_queries": total_queries,
        "mean_p_not_found": round(mean_pnf, 5),
        "flagged_for_3view": bool((mean_pnf > SOGRAND_FLAG_THRESHOLD) or (n_viol_after > 0)),
    }


# ---------------------------------------------------------------------------
# Trellis-BMA Multi-View IDS Consensus Fusion
# ---------------------------------------------------------------------------

def _split_stream_by_bars(toks: Sequence[str]) -> List[List[str]]:
    """Split token stream into bar units (preserving delimiter tokens at the end of each bar)."""
    bars: List[List[str]] = []
    cur: List[str] = []
    for t in toks:
        cur.append(t)
        if t in BAR_DELIMS or t == "<sys>":
            bars.append(cur)
            cur = []
    if cur:
        bars.append(cur)
    return bars


def trellis_bma_fuse_bar(views: Sequence[List[str]], bar_capacity: float) -> List[str]:
    """Fuse K=3 noisy views of a single bar via IDS Trellis-BMA majority voting."""
    non_empty = [v for v in views if v]
    if not non_empty:
        return []
    if len(non_empty) == 1:
        return list(non_empty[0])

    # Exact bar-level majority shortcut
    tuples = [tuple(v) for v in non_empty]
    counts = Counter(tuples)
    top_bar, top_cnt = counts.most_common(1)[0]
    if top_cnt >= 2:
        return list(top_bar)

    # Pick conservation-optimal pivot
    scored = []
    med_len = sorted(len(v) for v in non_empty)[len(non_empty) // 2]
    for idx, v in enumerate(non_empty):
        body = [t for t in v if t not in BAR_DELIMS and t != "<sys>"]
        _, syn, _ = compute_bar_duration_syndrome(body, bar_capacity)
        scored.append((syn, abs(len(v) - med_len), idx))
    scored.sort()
    pivot_idx = scored[0][2]
    pivot = non_empty[pivot_idx]
    L = len(pivot)

    pos_votes: List[List[Optional[str]]] = [[pivot[i]] for i in range(L)]
    ins_votes: List[List[Tuple[str, ...]]] = [[] for _ in range(L + 1)]

    for idx, v in enumerate(non_empty):
        if idx == pivot_idx:
            continue
        sm = difflib.SequenceMatcher(a=pivot, b=v, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for p, q in zip(range(i1, i2), range(j1, j2)):
                    pos_votes[p].append(v[q])
            elif tag == "replace":
                n_pair = min(i2 - i1, j2 - j1)
                for k in range(n_pair):
                    pos_votes[i1 + k].append(v[j1 + k])
                for p in range(i1 + n_pair, i2):
                    pos_votes[p].append(None)
                if j2 - j1 > n_pair:
                    ins_votes[i2].append(tuple(v[j1 + n_pair : j2]))
            elif tag == "delete":
                for p in range(i1, i2):
                    pos_votes[p].append(None)
            elif tag == "insert":
                ins_votes[i1].append(tuple(v[j1:j2]))

    fused: List[str] = []
    n_views = len(non_empty)
    thresh = (n_views // 2) + 1
    for i in range(L):
        if len(ins_votes[i]) >= thresh:
            ins_tok, ins_cnt = Counter(ins_votes[i]).most_common(1)[0]
            if ins_cnt >= thresh:
                fused.extend(ins_tok)
        votes = pos_votes[i]
        tok_counts = Counter(votes)
        winner, cnt = tok_counts.most_common(1)[0]
        if winner is not None:
            fused.append(winner)
        elif cnt < thresh:
            fused.append(pivot[i])
    if len(ins_votes[L]) >= thresh:
        ins_tok, ins_cnt = Counter(ins_votes[L]).most_common(1)[0]
        if ins_cnt >= thresh:
            fused.extend(ins_tok)

    fused_body = [t for t in fused if t not in BAR_DELIMS and t != "<sys>"]
    _, fused_syn, _ = compute_bar_duration_syndrome(fused_body, bar_capacity)
    if fused_syn > scored[0][0] + BAR_EPS:
        return list(pivot)
    return fused


def trellis_bma_fuse_views(view_streams: Sequence[List[str]]) -> List[str]:
    """Align bars across 3 decoded views and fuse each bar via Trellis-BMA consensus."""
    bar_lists = [_split_stream_by_bars(vs) for vs in view_streams]
    if not bar_lists:
        return []
    ref_idx = sorted(range(len(bar_lists)), key=lambda i: len(bar_lists[i]))[len(bar_lists) // 2]
    ref_bars = bar_lists[ref_idx]
    out: List[str] = []
    cap = 4.0
    for b_idx, rbar in enumerate(ref_bars):
        for t in rbar:
            ts = parse_time_sig_beats(t)
            if ts is not None:
                cap = ts
        cand_bars: List[List[str]] = []
        for bl in bar_lists:
            if b_idx < len(bl):
                cand_bars.append(bl[b_idx])
        out.extend(trellis_bma_fuse_bar(cand_bars, cap))
    return out


# ---------------------------------------------------------------------------
# Proper suite31_hw MusicXML Note F1, BSVR & wFPP Scoring (from i15-44b)
# ---------------------------------------------------------------------------

def _extract_musicxml_notes_headers_toks(
    xml_path: Path,
) -> Tuple[List[Tuple[str, int, str, str, str]], List[str], List[str]]:
    """Pure-Python MusicXML parser returning (ref_notes, ref_headers, approx_bekern_toks)."""
    notes: List[Tuple[str, int, str, str, str]] = []
    headers: List[str] = []
    toks: List[str] = []
    root = ET.parse(str(xml_path)).getroot()

    def _strip(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    for part in root:
        if _strip(part.tag) != "part":
            continue
        for meas in part:
            if _strip(meas.tag) != "measure":
                continue
            for child in meas:
                ctag = _strip(child.tag)
                if ctag == "attributes":
                    attr_clefs: List[str] = []
                    attr_keys: List[str] = []
                    attr_times: List[str] = []
                    for attr in child:
                        atag = _strip(attr.tag)
                        if atag == "clef":
                            sign = line = ""
                            for cc in attr:
                                cct = _strip(cc.tag)
                                txt = (cc.text or "").strip()
                                if cct == "sign":
                                    sign = txt.upper()
                                elif cct == "line":
                                    line = txt
                            if sign and line:
                                attr_clefs.append(f"clef-{sign}-{line}")
                        elif atag == "key":
                            for kc in attr:
                                if _strip(kc.tag) == "fifths":
                                    txt = (kc.text or "").strip()
                                    if txt:
                                        try:
                                            attr_keys.append(f"key{int(txt):+d}")
                                        except ValueError:
                                            pass
                        elif atag == "time":
                            beats = btype = ""
                            for tc in attr:
                                tct = _strip(tc.tag)
                                txt = (tc.text or "").strip()
                                if tct == "beats":
                                    beats = txt
                                elif tct == "beat-type":
                                    btype = txt
                            if beats and btype:
                                attr_times.append(f"time-{beats}-{btype}")
                    canon_hdrs = attr_clefs + attr_keys + attr_times
                    headers.extend(canon_hdrs)
                    toks.extend(canon_hdrs)
                elif ctag == "note":
                    is_rest = is_grace = False
                    step = octave = alter = dur = ""
                    n_dots = 0
                    for nc in child:
                        ntag = _strip(nc.tag)
                        if ntag == "rest":
                            is_rest = True
                        elif ntag == "grace":
                            is_grace = True
                        elif ntag == "pitch":
                            for pc in nc:
                                ptag = _strip(pc.tag)
                                txt = (pc.text or "").strip()
                                if ptag == "step":
                                    step = txt.upper()
                                elif ptag == "octave":
                                    octave = txt
                                elif ptag == "alter":
                                    try:
                                        av = int(round(float(txt)))
                                        if av != 0:
                                            alter = f"a{av}"
                                    except ValueError:
                                        pass
                        elif ntag == "type":
                            dur = (nc.text or "").strip().lower()
                        elif ntag == "dot":
                            n_dots += 1
                    if is_rest:
                        if dur:
                            toks.extend(["rest", dur] + ["dot"] * n_dots)
                    elif (not is_grace) and step and octave and dur:
                        notes.append((dur, n_dots, step, octave, alter))
                        toks.extend([dur, step, octave])
                        if alter:
                            toks.append(alter)
                        toks.extend(["dot"] * n_dots)
            toks.append("bar")
    return notes, headers, toks


def _extract_headers(toks: Sequence[str]) -> List[str]:
    return [t for t in toks if t.startswith(("clef-", "key-", "key", "time-"))]


def score_page_prediction(
    pred_toks: Sequence[str],
    gt_xml_path: Optional[Path],
    hit_eos: bool,
    *,
    slug: str = "page",
    workdir: Optional[Path] = None,
    prefer_rust: bool = True,
) -> Dict[str, Any]:
    """Compute real MusicXML Note F1, TokLCS F1, BSVR, and wFPP against a `.musicxml` GT file.

    Uses the exact `eval_uni_suite31` / `eval_sys_gate` Rust scorer (`assembly-serialize` +
    `eval-harness`) whenever compiled binaries are present, and cleanly falls back to
    pure-Python `xml.etree.ElementTree` MusicXML note/header matching on Colab Linux.
    """
    import eval_sys_gate as esg
    import eval_uni_suite31
    import rlvr_grpo

    if gt_xml_path is None or not gt_xml_path.is_file():
        return {
            "note_f1": 0.0,
            "py_note_f1": 0.0,
            "tok_lcs_f1": 0.0,
            "bsvr": 1.0,
            "ref_notes": 0,
            "hyp_notes": 0,
            "wfpp": 0.0,
            "bekern_decode_ok": False,
            "scorer": "missing_gt",
        }

    cleaned_toks = list(pred_toks)
    if not hit_eos and cleaned_toks:
        cleaned_toks = eval_uni_suite31.trim_truncated_tokens(cleaned_toks)
    hyp_page_toks, _ = eval_uni_suite31.stitch_uni_tokens(cleaned_toks)

    use_rust = bool(prefer_rust and esg.ASER.is_file() and esg.EHARNESS.is_file())
    if use_rust:
        tmp_ctx: Optional[tempfile.TemporaryDirectory] = None
        if workdir is None:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="i15_48b_score_")
            wd = Path(tmp_ctx.name)
        else:
            wd = workdir
            wd.mkdir(parents=True, exist_ok=True)
        try:
            ref_page_toks, page_ref_notes = esg.suite31_page_refs(gt_xml_path, f"i1548b-{slug}", wd)
            tok_lcs_f1 = eval_uni_suite31.compute_token_lcs_f1(ref_page_toks, hyp_page_toks)

            hyp_txt = wd / f"i1548b-{slug}-hyp-bekern.txt"
            hyp_txt.write_text(" ".join(hyp_page_toks) + "\n", encoding="utf-8")
            hyp_notes = wd / f"i1548b-{slug}-hyp-notes.json"
            p_dec = esg.run_cli([
                esg.ASER, "bekern-decode", "--input", str(hyp_txt), "--out", str(hyp_notes),
            ])
            bekern_ok = (p_dec.returncode == 0) and hyp_notes.exists()

            beat_m = eval_uni_suite31.compute_fsa_beat_metrics(
                hyp_page_toks, hit_eos=hit_eos, bekern_decode_ok=bekern_ok
            )
            n_bars = int(beat_m["n_bars"])
            bsvr = round(1.0 - float(beat_m["beat_conservation_rate"]), 6) if n_bars > 0 else 1.0
            n_bar_violations = int(round(bsvr * n_bars))

            ref_notes_py = rlvr_grpo.extract_pitch_dur_notes(ref_page_toks)
            pred_notes_py = rlvr_grpo.extract_pitch_dur_notes(hyp_page_toks)
            py_note_f1 = rlvr_grpo.compute_sequence_f1(ref_notes_py, pred_notes_py)

            if bekern_ok:
                sc = esg.score_notes_pair(
                    page_ref_notes, hyp_notes, wd, f"i1548b-{slug}", esg.EHARNESS, esg.ASER
                )
                note_f1 = float(sc.get("f1", 0.0))
                proxy_fixes = sc.get("proxy_fixes")
                ref_n_cnt = int(sc.get("ref_len") or len(ref_notes_py))
                hyp_n_cnt = int(sc.get("hyp_len") or len(pred_notes_py))
            else:
                note_f1 = 0.0
                proxy_fixes = None
                ref_n_cnt = len(ref_notes_py)
                hyp_n_cnt = len(pred_notes_py)

            sm_h = difflib.SequenceMatcher(
                None, _extract_headers(ref_page_toks), _extract_headers(hyp_page_toks), autojunk=False
            )
            header_edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm_h.get_opcodes() if tag != "equal")
            if proxy_fixes is not None:
                note_edits = int(proxy_fixes)
            else:
                sm_n = difflib.SequenceMatcher(None, ref_notes_py, pred_notes_py, autojunk=False)
                note_edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm_n.get_opcodes() if tag != "equal")
            wfpp = round(3.0 * header_edits + 2.0 * n_bar_violations + 1.0 * note_edits, 2)

            return {
                "note_f1": round(float(note_f1), 5),
                "py_note_f1": round(float(py_note_f1), 5),
                "tok_lcs_f1": round(float(tok_lcs_f1), 5),
                "bsvr": bsvr,
                "ref_notes": ref_n_cnt,
                "hyp_notes": hyp_n_cnt,
                "wfpp": wfpp,
                "bekern_decode_ok": bekern_ok,
                "scorer": "eval_uni_suite31_rust",
            }
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

    # Pure-Python MusicXML fallback (e.g. Colab Linux without Rust binaries)
    ref_notes, ref_headers, ref_toks = _extract_musicxml_notes_headers_toks(gt_xml_path)
    pred_notes = rlvr_grpo.extract_pitch_dur_notes(hyp_page_toks)
    note_f1 = rlvr_grpo.compute_sequence_f1(ref_notes, pred_notes)
    tok_lcs_f1 = eval_uni_suite31.compute_token_lcs_f1(ref_toks, hyp_page_toks)

    beat_m = eval_uni_suite31.compute_fsa_beat_metrics(
        hyp_page_toks, hit_eos=hit_eos, bekern_decode_ok=True
    )
    n_bars = int(beat_m["n_bars"])
    bsvr = round(1.0 - float(beat_m["beat_conservation_rate"]), 6) if n_bars > 0 else 1.0
    n_bar_violations = int(round(bsvr * n_bars))

    sm_h = difflib.SequenceMatcher(
        None, ref_headers, _extract_headers(hyp_page_toks), autojunk=False
    )
    header_edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm_h.get_opcodes() if tag != "equal")
    sm_n = difflib.SequenceMatcher(None, ref_notes, pred_notes, autojunk=False)
    note_edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm_n.get_opcodes() if tag != "equal")
    wfpp = round(3.0 * header_edits + 2.0 * n_bar_violations + 1.0 * note_edits, 2)

    return {
        "note_f1": round(float(note_f1), 5),
        "py_note_f1": round(float(note_f1), 5),
        "tok_lcs_f1": round(float(tok_lcs_f1), 5),
        "bsvr": bsvr,
        "ref_notes": len(ref_notes),
        "hyp_notes": len(pred_notes),
        "wfpp": wfpp,
        "bekern_decode_ok": True,
        "scorer": "python_musicxml_lcs",
    }


def compute_gate_pass(arm_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate PLAN.md promotion gates across greedy_fsa, orbgrand_sogrand, and trellis_bma_3view."""
    g0 = arm_results.get("greedy_fsa")
    g1 = arm_results.get("orbgrand_sogrand")
    g2 = arm_results.get("trellis_bma_3view")
    if not (g0 and g1 and g2):
        return {"overall_gate_pass": False, "complete": False}

    s0 = float(g0.get("syndrome_rate", 0.0))
    s1 = float(g1.get("syndrome_rate", 0.0))
    syn_rel_red = ((s0 - s1) / s0) if s0 > 1e-9 else 0.0

    orb_tok_delta = float(g1.get("all12_token_f1", 0.0)) - float(g0.get("all12_token_f1", 0.0))
    orb_note_delta = float(g1.get("note_f1", 0.0)) - float(g0.get("note_f1", 0.0))
    ms0 = max(1e-6, float(g0.get("mean_ms_per_page", 1.0)))
    ms1 = float(g1.get("mean_ms_per_page", ms0))
    orb_lat_pct = ((ms1 - ms0) / ms0) * 100.0

    orb_pass = bool(
        syn_rel_red >= 0.25
        and (orb_tok_delta >= 0.005 or orb_note_delta >= 0.005)
        and orb_lat_pct < 5.0
    )

    # Flagged-pages delta for Arm 2 (trellis_bma_3view vs greedy_fsa)
    g0_by_id = {r["id"]: r for r in (g0.get("per_page") or [])}
    flagged_tok_deltas: List[float] = []
    flagged_note_deltas: List[float] = []
    for r2 in (g2.get("per_page") or []):
        if r2.get("flagged", False) and r2["id"] in g0_by_id:
            r0 = g0_by_id[r2["id"]]
            flagged_tok_deltas.append(float(r2.get("token_f1", 0.0)) - float(r0.get("token_f1", 0.0)))
            flagged_note_deltas.append(float(r2.get("note_f1", 0.0)) - float(r0.get("note_f1", 0.0)))

    n_flagged = len(flagged_tok_deltas)
    tr_flagged_tok_delta = sum(flagged_tok_deltas) / max(1, n_flagged)
    tr_flagged_note_delta = sum(flagged_note_deltas) / max(1, n_flagged)
    trellis_pass = bool(
        n_flagged > 0
        and (tr_flagged_tok_delta >= 0.010 or tr_flagged_note_delta >= 0.010)
    )

    return {
        "complete": True,
        "orbgrand_syndrome_rel_reduction": round(syn_rel_red, 5),
        "orbgrand_token_f1_delta": round(orb_tok_delta, 5),
        "orbgrand_note_f1_delta": round(orb_note_delta, 5),
        "orbgrand_wfpp_delta": round(float(g1.get("wfpp", 0.0)) - float(g0.get("wfpp", 0.0)), 2),
        "orbgrand_latency_overhead_pct": round(orb_lat_pct, 3),
        "orbgrand_gate_pass": orb_pass,
        "trellis_n_flagged_pages": n_flagged,
        "trellis_flagged_token_f1_delta": round(tr_flagged_tok_delta, 5),
        "trellis_flagged_note_f1_delta": round(tr_flagged_note_delta, 5),
        "trellis_all12_note_f1_delta": round(float(g2.get("note_f1", 0.0)) - float(g0.get("note_f1", 0.0)), 5),
        "trellis_wfpp_delta": round(float(g2.get("wfpp", 0.0)) - float(g0.get("wfpp", 0.0)), 2),
        "trellis_gate_pass": trellis_pass,
        "overall_gate_pass": bool(orb_pass or trellis_pass),
    }


# ---------------------------------------------------------------------------
# Infrastructure Helpers (Windows Safety, HF Sync/Resume, Views, Decode)
# ---------------------------------------------------------------------------

def _configure_backend_safety() -> None:
    import torch

    if sys.platform == "win32":
        torch.backends.cudnn.enabled = False
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_hf_token() -> str:
    if HF_TOKEN_FILE.is_file():
        try:
            os.chmod(HF_TOKEN_FILE, 0o600)
        except OSError:
            pass
        tok = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        return tok
    cache_tok = Path.home() / ".cache" / "huggingface" / "token"
    if cache_tok.is_file():
        tok = cache_tok.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    raise RuntimeError("HF token missing: provide /tmp/.hf_token, HF_TOKEN env var, or ~/.cache/huggingface/token")


def _purge_hf_token() -> None:
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass


def _locate_mvd() -> Path:
    candidates = [
        Path.cwd() / "tmp-train" / "mvd",
        Path(__file__).resolve().parents[2] / "tmp-train" / "mvd",
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path("/workspace/mvd"),
        Path("/content/mvd"),
        WORK_ROOT / "mvd",
    ]
    for cand in candidates:
        if (cand / "uni_omr_v2.py").is_file() and (cand / "eval_uni_suite31.py").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    for bundle in (Path("/content/bundle.tar.gz"), Path("/content/mvd_bundle.tar.gz")):
        if bundle.is_file():
            WORK_ROOT.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=WORK_ROOT)
            for p in WORK_ROOT.rglob("eval_uni_suite31.py"):
                if str(p.parent) not in sys.path:
                    sys.path.insert(0, str(p.parent))
                return p.parent
    raise FileNotFoundError("Could not locate tmp-train/mvd/eval_uni_suite31.py")


def _ensure_deps() -> None:
    needed = []
    for mod, pkg in (("huggingface_hub", "huggingface_hub>=0.26"), ("PIL", "pillow"), ("numpy", "numpy")):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])


def _try_resume_from_hf(hf_token: str) -> Dict[str, Dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    recovered: Dict[str, Dict[str, Any]] = {}
    dl_dir = WORK_ROOT / "hf_resume"
    dl_dir.mkdir(parents=True, exist_ok=True)

    try:
        ev_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_EVAL_PATH,
            repo_type="dataset",
            token=hf_token,
            local_dir=str(dl_dir),
        )
        ev_data = json.loads(Path(ev_path).read_text(encoding="utf-8"))
        for arm_name, arm_res in (ev_data.get("arms") or {}).items():
            if arm_name in ARMS and isinstance(arm_res, dict) and "all12_token_f1" in arm_res and "note_f1" in arm_res:
                recovered[arm_name] = arm_res
    except Exception:
        pass

    for arm_name in ARMS:
        if arm_name in recovered:
            continue
        try:
            p = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=f"{HF_CKPT_PREFIX}/{arm_name}_arm_result.json",
                repo_type="dataset",
                token=hf_token,
                local_dir=str(dl_dir),
            )
            arm_res = json.loads(Path(p).read_text(encoding="utf-8"))
            if isinstance(arm_res, dict) and "all12_token_f1" in arm_res and "note_f1" in arm_res:
                recovered[arm_name] = arm_res
        except Exception:
            pass

    for arm_name, arm_res in recovered.items():
        cached = WORK_ROOT / "ckpts" / arm_name / "arm_result.json"
        if not cached.is_file():
            _atomic_write_json(cached, arm_res)
        print(f"[{EXP_ID}] Auto-resumed completed arm '{arm_name}' from HF Private Hub", flush=True)
    return recovered


def _resolve_checkpoint(hf_token: str) -> Tuple[Path, str]:
    from huggingface_hub import hf_hub_download

    env_ckpt = os.environ.get("UNIOMR_CKPT", "").strip()
    if env_ckpt and Path(env_ckpt).is_file():
        return Path(env_ckpt), f"local:{env_ckpt}"

    dl_dir = WORK_ROOT / "hf_dl"
    dl_dir.mkdir(parents=True, exist_ok=True)
    for rel in HF_CKPT_CANDIDATES:
        try:
            p = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=rel,
                repo_type="dataset",
                token=hf_token,
                local_dir=str(dl_dir),
            )
            return Path(p), f"hf:{rel}"
        except Exception:
            continue
    raise FileNotFoundError(f"None of HF checkpoint candidates found: {HF_CKPT_CANDIDATES}")


def _ensure_suite31_subset(hf_token: str) -> List[Dict[str, Any]]:
    from huggingface_hub import hf_hub_download
    import eval_uni_suite31

    manifest_path: Optional[Path] = None
    local_manifest = Path(__file__).resolve().parents[2] / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"
    if local_manifest.is_file():
        _, local_pieces = eval_uni_suite31.resolve_manifest_pieces(local_manifest)
        if local_pieces and local_pieces[0]["photo"].is_file():
            manifest_path = local_manifest

    if manifest_path is None:
        ds_dir = WORK_ROOT / "datasets" / "suite31_hw"
        manifest_candidates = sorted(ds_dir.rglob("suite31_hw_manifest.json")) if ds_dir.is_dir() else []
        if not manifest_candidates:
            ds_dir.mkdir(parents=True, exist_ok=True)
            dl_dir = WORK_ROOT / "hf_dl"
            dl_dir.mkdir(parents=True, exist_ok=True)
            tar_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_SUITE31_TAR,
                repo_type="dataset",
                token=hf_token,
                local_dir=str(dl_dir),
            )
            with tarfile.open(tar_path, "r:*") as tf:
                tf.extractall(path=ds_dir)
            manifest_candidates = sorted(ds_dir.rglob("suite31_hw_manifest.json"))
        if not manifest_candidates:
            raise FileNotFoundError(f"suite31_hw_manifest.json not found in {ds_dir}")
        manifest_path = manifest_candidates[0]

    _, pieces = eval_uni_suite31.resolve_manifest_pieces(manifest_path)
    printed = [p for p in pieces if not bool(p.get("handwritten", False))]
    hw = [p for p in pieces if bool(p.get("handwritten", False))]

    selected: List[Dict[str, Any]] = []
    for p in printed[:N_HARD_PRINTED]:
        pid = str(p.get("slug") or p.get("id"))
        img_p = p.get("photo") or p.get("clean")
        gt_p = p.get("gt") or p.get("gt_bekern") or p.get("bekern")
        selected.append({"id": pid, "domain": "printed", "image": Path(img_p), "gt_xml": Path(gt_p) if gt_p else None})
    for p in hw[:N_HARD_HW]:
        pid = str(p.get("slug") or p.get("id"))
        img_p = p.get("photo") or p.get("clean")
        gt_p = p.get("gt") or p.get("gt_bekern") or p.get("bekern")
        selected.append({"id": pid, "domain": "hw", "image": Path(img_p), "gt_xml": Path(gt_p) if gt_p else None})
    return selected


def _make_contrast_view(pil_im: Any) -> Any:
    from PIL import ImageOps
    return ImageOps.autocontrast(pil_im.convert("RGB"), cutoff=1)


def _pil_to_tensor(img_pil: Any, max_dim: int) -> "torch.Tensor":
    import numpy as np
    import torch
    from PIL import Image

    rgb = img_pil.convert("RGB")
    w, h = rgb.size
    scale = min(1.0, float(max_dim) / float(max(w, h)))
    nw = max(32, int(round(w * scale / 16.0)) * 16)
    nh = max(32, int(round(h * scale / 16.0)) * 16)
    if (nw, nh) != (w, h):
        rgb = rgb.resize((nw, nh), Image.BICUBIC)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return (t - 0.5) / 0.5


def _decode_fsa_with_top_cands(
    model: Any,
    img_pil: Any,
    max_dim: int,
    max_len: int,
    device: "torch.device",
    use_bf16: bool,
    top_k: int = 4,
) -> Tuple[List[str], List[List[Tuple[str, float]]], bool]:
    """FSA-constrained greedy decode returning (tokens, per-position top-K (tok_str, prob), hit_eos)."""
    import torch
    import torch.nn.functional as F
    import eval_uni_suite31
    import grammar

    inp = _pil_to_tensor(img_pil, max_dim).unsqueeze(0).to(device)
    vocab_size = int(model.cfg.vocab_size)
    pad_id, bos_id, eos_id, _, tok_to_id, id_to_tok = eval_uni_suite31._vocab_spec(vocab_size)

    toks: List[str] = []
    top_cands: List[List[Tuple[str, float]]] = []
    hit_eos = False
    state = grammar.PrefixState(strict=False, vocab_map=tok_to_id)

    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
        mem, _, _ = model.encode_image(inp, return_aux=False)
        past_kvs = model.precompute_cross_kv(mem) if hasattr(model, "precompute_cross_kv") else None
        cur_ids = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
        for _ in range(max_len):
            logits, _, past_kvs = model.decode_tokens(
                cur_ids,
                memory=mem if past_kvs is None else None,
                return_mtp=False,
                return_aux=False,
                past_kvs=past_kvs,
                use_cache=True,
                apply_tok_dropout=False,
            )
            step_logits = logits[0, -1, :].float()
            allowed = eval_uni_suite31._allowed_ids_for_state(state, vocab_size)
            if allowed:
                mask = torch.full_like(step_logits, -1e9)
                mask[allowed] = step_logits[allowed]
                probs = F.softmax(mask, dim=-1)
            else:
                probs = F.softmax(step_logits, dim=-1)

            k_eff = min(top_k, probs.numel())
            top_p, top_i = torch.topk(probs, k=k_eff)
            tok_id = int(top_i[0].item())
            if tok_id == eos_id or tok_id == pad_id:
                hit_eos = tok_id == eos_id
                break
            tok_str = id_to_tok.get(tok_id, "")
            toks.append(tok_str)
            cands = [(id_to_tok.get(int(i.item()), ""), float(p.item())) for i, p in zip(top_i, top_p)]
            top_cands.append(cands)
            state = eval_uni_suite31._advance_grammar_state(state, tok_id, strict=False, vocab_size=vocab_size)
            cur_ids = torch.tensor([[tok_id]], dtype=torch.long, device=device)

    return toks, top_cands, hit_eos


def _token_lcs_f1(ref_toks: Sequence[str], hyp_toks: Sequence[str]) -> float:
    if not ref_toks and not hyp_toks:
        return 1.0
    if not ref_toks or not hyp_toks:
        return 0.0
    sm = difflib.SequenceMatcher(a=ref_toks, b=hyp_toks, autojunk=False)
    matches = sum(b.size for b in sm.get_matching_blocks())
    prec = matches / max(1, len(hyp_toks))
    rec = matches / max(1, len(ref_toks))
    return (2.0 * prec * rec) / max(1e-12, prec + rec)


def _upload_json_to_hf(hf_token: str, local_path: Path, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"{EXP_CODE}({EXP_ID}): upload {path_in_repo}",
    )
    print(f"[{EXP_ID}] Uploaded {path_in_repo} -> {HF_REPO_ID}", flush=True)


# ---------------------------------------------------------------------------
# CPU Smoke Test (Realistic Multi-Voice/Multi-Staff + ORBGRAND + Trellis-BMA + MusicXML)
# ---------------------------------------------------------------------------

_MOCK_MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1"><part-name>Music</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration><type>half</type></note>
    </measure>
  </part>
</score-partwise>
"""


def run_smoke_test() -> int:
    """CPU unit & mock test verifying:
    1. Multi-voice (`voice-1`, `voice-2`) + multi-staff (`<staff>`) syndrome accounting
       (including `bar-rest`, `chord`, `dot` 1.5x, `tuplet`, and `grace`).
    2. Non-zero syndrome fires on corrupted leading-`voice-1`, `voice-2`, and `<staff>` bars.
    3. ORBGRAND repairs corrupted multi-voice / multi-staff bars back to exact GT.
    4. SOGRAND flags unrepairable `voice-1` pages (`flagged_for_3view == True`), and
       Trellis-BMA 3-view consensus recovers exact GT on flagged multi-voice pages.
    5. MusicXML `note_f1`, `bsvr`, `wfpp`, and `gate_pass` scoring works on CPU.
    """
    _locate_mvd()

    # 1. Valid realistic multi-voice + multi-staff 4/4 bar:
    #    Staff 0 voice-1: dotted quarter (1.5) + eighth (0.5) + triplet(3x quarter = 2.0) + grace(0) + chord(0) = 4.0
    #    Staff 0 voice-2: half (2.0) + half (2.0) = 4.0
    #    Staff 1 voice-1: bar-rest = 4.0
    gt_mv_bar = [
        "time-4-4",
        "voice-1",
        "grace", "eighth", "B", "3",
        "quarter", "dot", "C", "4",
        "chord", "quarter", "E", "4",
        "eighth", "D", "4",
        "tuplet-start-3-2", "quarter", "E", "4", "quarter", "F", "4", "quarter", "G", "4", "tuplet-stop",
        "voice-2",
        "half", "C", "3", "half", "G", "3",
        "<staff>",
        "voice-1",
        "bar-rest",
    ]
    _, gt_syn, gt_ok = compute_bar_duration_syndrome(gt_mv_bar, 4.0)
    assert gt_ok and gt_syn == 0.0, f"Valid multi-voice/multi-staff GT bar failed syndrome: {gt_syn}"

    # 2a. Corrupt leading voice-1 (the exact bug that made i15-48 a no-op!):
    corrupt_v1 = list(gt_mv_bar)
    v1_eighth_idx = corrupt_v1.index("eighth", 10)  # the eighth D 4 after chord
    corrupt_v1[v1_eighth_idx] = "16th"  # 3.75 != 4.0
    _, syn_v1, ok_v1 = compute_bar_duration_syndrome(corrupt_v1, 4.0)
    assert (not ok_v1) and abs(syn_v1 - 0.25) < 1e-4, f"Corrupted voice-1 not caught: syn={syn_v1}, ok={ok_v1}"

    # 2b. Corrupt voice-2 while voice-1 is valid:
    corrupt_v2 = list(gt_mv_bar)
    v2_half_idx = corrupt_v2.index("half")
    corrupt_v2[v2_half_idx] = "quarter"  # voice-2 sum = 3.0 != 4.0
    _, syn_v2, ok_v2 = compute_bar_duration_syndrome(corrupt_v2, 4.0)
    assert (not ok_v2) and abs(syn_v2 - 1.0) < 1e-4, f"Corrupted voice-2 not caught: syn={syn_v2}, ok={ok_v2}"

    # 2c. Corrupt staff 2 while staff 1 has bar-rest:
    corrupt_s2 = [
        "time-4-4", "voice-1", "bar-rest",
        "<staff>", "voice-1", "quarter", "C", "3", "half", "G", "3",  # 3.0 != 4.0
    ]
    _, syn_s2, ok_s2 = compute_bar_duration_syndrome(corrupt_s2, 4.0)
    assert (not ok_s2) and abs(syn_s2 - 1.0) < 1e-4, f"Corrupted staff-2 not caught: syn={syn_s2}, ok={ok_s2}"

    # 3. ORBGRAND repairs a 2-error multi-voice + multi-staff bar (1 error in voice-1, 1 in voice-2)
    noisy_mv_bar = list(gt_mv_bar)
    noisy_mv_bar[v1_eighth_idx] = "16th"
    noisy_mv_bar[v2_half_idx] = "quarter"
    top_cands: List[List[Tuple[str, float]]] = []
    for pos, tok in enumerate(noisy_mv_bar):
        if pos == v1_eighth_idx:
            top_cands.append([("16th", 0.55), ("eighth", 0.43)])
        elif pos == v2_half_idx:
            top_cands.append([("quarter", 0.53), ("half", 0.45)])
        else:
            top_cands.append([(tok, 0.98), ("whole", 0.01)])

    rep = orbgrand_repair_bar(noisy_mv_bar, top_cands, bar_capacity=4.0, max_patterns=ORBGRAND_MAX_PATTERNS)
    assert rep["repaired"], f"ORBGRAND failed to repair multi-voice 2-error bar: {rep}"
    assert rep["tokens"] == gt_mv_bar, f"ORBGRAND repaired to wrong bar: {rep['tokens']}"
    assert rep["final_syndrome"] == 0.0 and rep["queries"] > 0

    # 4. Unrepairable realistic voice-1 + <staff> page -> flagged_for_3view -> Trellis-BMA 3-view recovery
    gt_stream = [
        "time-4-4", "voice-1", "quarter", "C", "4", "quarter", "D", "4", "quarter", "E", "4", "quarter", "F", "4",
        "<staff>", "voice-1", "half", "C", "3", "half", "G", "3", "bar",
    ]
    v1_bad = [
        "time-4-4", "voice-1", "eighth", "C", "4", "quarter", "D", "4", "quarter", "E", "4", "quarter", "F", "4",
        "<staff>", "voice-1", "half", "C", "3", "half", "G", "3", "bar",
    ]  # sub in voice-1 (3.5 != 4.0)
    v2_bad = [
        "time-4-4", "voice-1", "quarter", "C", "4", "quarter", "D", "4", "quarter", "E", "4", "F", "4",
        "<staff>", "voice-1", "half", "C", "3", "half", "G", "3", "bar",
    ]  # del in voice-1 (3.0 != 4.0)
    v3_bad = [
        "time-4-4", "voice-1", "quarter", "C", "4", "dot", "quarter", "D", "4", "quarter", "E", "4", "quarter", "F", "4",
        "<staff>", "voice-1", "half", "C", "3", "half", "G", "3", "bar",
    ]  # ins dot in voice-1 (4.5 != 4.0)

    # With no valid alt in top_cands, v1_bad is unrepairable -> flagged_for_3view == True
    bad_cands = [[(t, 0.99)] for t in v1_bad]
    page_rep = apply_orbgrand_page(v1_bad, bad_cands)
    assert page_rep["n_viol_before"] == 1 and page_rep["n_viol_after"] == 1, f"Expected 1 violation: {page_rep}"
    assert page_rep["flagged_for_3view"] and page_rep["mean_p_not_found"] == 1.0, f"Expected flagged page: {page_rep}"

    # Flagged page triggers Trellis-BMA 3-view consensus -> exact GT recovery
    fused = trellis_bma_fuse_views([v1_bad, v2_bad, v3_bad])
    assert fused == gt_stream, f"Trellis-BMA mismatch: {fused} != {gt_stream}"
    f1_v1 = _token_lcs_f1(gt_stream, v1_bad)
    f1_fused = _token_lcs_f1(gt_stream, fused)
    assert f1_fused == 1.0 and f1_fused > f1_v1

    # 5. MusicXML Note F1, BSVR, wFPP, and gate_pass check
    mock_pred = ["clef-G-2", "key+0", "time-4-4", "voice-1", "quarter", "C", "4", "quarter", "D", "4", "half", "E", "4", "bar"]
    with tempfile.TemporaryDirectory(prefix="i15_48b_smoke_") as td:
        xml_p = Path(td) / "gt-p1-smoke.musicxml"
        xml_p.write_text(_MOCK_MUSICXML, encoding="utf-8")
        sc_py = score_page_prediction(mock_pred, xml_p, hit_eos=True, slug="smoke-py", prefer_rust=False)
        assert sc_py["note_f1"] == 1.0 and sc_py["wfpp"] == 0.0, f"Python MusicXML scorer failed: {sc_py}"
        sc_auto = score_page_prediction(mock_pred, xml_p, hit_eos=True, slug="smoke-auto", prefer_rust=True)
        assert sc_auto["note_f1"] > 0.99, f"Auto MusicXML scorer failed: {sc_auto}"

    gp = compute_gate_pass({
        "greedy_fsa": {"syndrome_rate": 0.40, "all12_token_f1": 0.60, "note_f1": 0.55, "wfpp": 20.0, "mean_ms_per_page": 1000.0, "per_page": [{"id": "p1", "token_f1": 0.50, "note_f1": 0.45}]},
        "orbgrand_sogrand": {"syndrome_rate": 0.20, "all12_token_f1": 0.61, "note_f1": 0.56, "wfpp": 18.0, "mean_ms_per_page": 1020.0, "per_page": [{"id": "p1", "token_f1": 0.51, "note_f1": 0.46}]},
        "trellis_bma_3view": {"syndrome_rate": 0.15, "all12_token_f1": 0.62, "note_f1": 0.57, "wfpp": 16.0, "mean_ms_per_page": 1800.0, "per_page": [{"id": "p1", "token_f1": 0.54, "note_f1": 0.49, "flagged": True}]},
    })
    assert gp["overall_gate_pass"] and gp["orbgrand_gate_pass"] and gp["trellis_gate_pass"], f"gate_pass failed: {gp}"

    print(
        f"[{EXP_CODE}][SMOKE] Multi-voice/staff syndromes: v1={syn_v1:.2f} v2={syn_v2:.2f} s2={syn_s2:.2f} | "
        f"ORBGRAND repaired 2-err mv bar in {rep['queries']} queries (p_not_found={rep['p_not_found']:.4f}) | "
        f"Flagged->Trellis-BMA 3-view F1: {f1_v1:.4f}->{f1_fused:.4f} | "
        f"MusicXML({sc_auto['scorer']}): note_f1={sc_auto['note_f1']:.4f} bsvr={sc_auto['bsvr']:.4f} wfpp={sc_auto['wfpp']:.1f}",
        flush=True,
    )
    print(f"[{EXP_CODE}][SMOKE] PASS!", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{EXP_ID} runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run CPU mock smoke test and exit")
    args = parser.parse_args()
    if args.smoke_test:
        return run_smoke_test()

    t_start = time.perf_counter()
    started_at = _utc_now()
    hf_token: Optional[str] = None
    status = "RUNNING"
    error_msg: Optional[str] = None
    arm_results: Dict[str, Dict[str, Any]] = {}

    try:
        hf_token = _read_hf_token()
        _ensure_deps()
        _locate_mvd()
        _configure_backend_safety()

        import torch
        from PIL import Image
        import eval_uni_suite31

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        print(f"[{EXP_ID}] Device={gpu_name} | bf16={use_bf16}", flush=True)

        arm_results.update(_try_resume_from_hf(hf_token))
        ckpt_path, ckpt_source = _resolve_checkpoint(hf_token)
        pages = _ensure_suite31_subset(hf_token)
        print(f"[{EXP_ID}] Loaded ckpt={ckpt_source} | pages={len(pages)}", flush=True)

        model, _, ckpt_meta = eval_uni_suite31.load_uni_checkpoint(ckpt_path, device)
        if hasattr(model, "cfg") and hasattr(model.cfg, "max_mem_tokens"):
            model.cfg.max_mem_tokens = max(int(model.cfg.max_mem_tokens), 4096)

        base_cache: Dict[str, Dict[str, Any]] = {}
        score_workdir = WORK_ROOT / "score_tmp"
        score_workdir.mkdir(parents=True, exist_ok=True)

        def _summarize_arm(arm_name: str, per_page: List[Dict[str, Any]]) -> Dict[str, Any]:
            pr_rows = [r for r in per_page if r["domain"] == "printed"]
            hw_rows = [r for r in per_page if r["domain"] == "hw"]
            tot_bars = sum(r["n_bars"] for r in per_page)
            tot_viol = sum(r["n_viol"] for r in per_page)
            tot_rep = sum(r.get("n_repaired", 0) for r in per_page)
            tot_q = sum(r.get("queries", 0) for r in per_page)

            def _mean(rows: List[Dict[str, Any]], key: str, nd: int = 5) -> float:
                return round(sum(float(r.get(key, 0.0)) for r in rows) / max(1, len(rows)), nd)

            return {
                "arm": arm_name,
                "ckpt_source": ckpt_source,
                "n_pages": len(per_page),
                "printed_token_f1": _mean(pr_rows, "token_f1"),
                "hw_token_f1": _mean(hw_rows, "token_f1"),
                "all12_token_f1": _mean(per_page, "token_f1"),
                "note_f1": _mean(per_page, "note_f1"),
                "printed_note_f1": _mean(pr_rows, "note_f1"),
                "hw_note_f1": _mean(hw_rows, "note_f1"),
                "bsvr": _mean(per_page, "bsvr"),
                "printed_bsvr": _mean(pr_rows, "bsvr"),
                "hw_bsvr": _mean(hw_rows, "bsvr"),
                "wfpp": _mean(per_page, "wfpp", 2),
                "printed_wfpp": _mean(pr_rows, "wfpp", 2),
                "hw_wfpp": _mean(hw_rows, "wfpp", 2),
                "syndrome_rate": round(tot_viol / max(1, tot_bars), 5),
                "n_bars_total": tot_bars,
                "n_bars_violating": tot_viol,
                "n_bars_repaired": tot_rep,
                "total_queries": tot_q,
                "mean_ms_per_page": _mean(per_page, "elapsed_ms", 1),
                "mean_p_not_found": _mean(per_page, "mean_p_not_found"),
                "n_flagged_pages": sum(1 for r in per_page if r.get("flagged", False)),
                "per_page": per_page,
            }

        for arm_name in ARMS:
            cached_path = WORK_ROOT / "ckpts" / arm_name / "arm_result.json"
            if arm_name in arm_results:
                continue
            per_page: List[Dict[str, Any]] = []
            for idx, p in enumerate(pages, start=1):
                pid = p["id"]
                pil_im = Image.open(p["image"]).convert("RGB")
                t0 = time.perf_counter()

                if pid not in base_cache:
                    btoks, bcands, b_eos = _decode_fsa_with_top_cands(model, pil_im, MAX_DIM, MAX_LEN, device, use_bf16)
                    base_ms = (time.perf_counter() - t0) * 1000.0
                    base_cache[pid] = {"tokens": btoks, "cands": bcands, "hit_eos": b_eos, "base_ms": base_ms}

                if arm_name == "greedy_fsa":
                    toks = base_cache[pid]["tokens"]
                    hit_eos = base_cache[pid]["hit_eos"]
                    spans = split_bars_with_spans(toks)
                    n_viol = sum(1 for s, e, c in spans if compute_bar_duration_syndrome(toks[s:e], c)[1] > BAR_EPS)
                    sc = score_page_prediction(toks, p["gt_xml"], hit_eos=hit_eos, slug=f"{arm_name}-{pid}", workdir=score_workdir)
                    row = {
                        "id": pid,
                        "domain": p["domain"],
                        "token_f1": sc["tok_lcs_f1"],
                        "n_bars": len(spans),
                        "n_viol": n_viol,
                        "elapsed_ms": round(base_cache[pid]["base_ms"], 1),
                        **sc,
                    }
                elif arm_name == "orbgrand_sogrand":
                    t_rep0 = time.perf_counter()
                    rep = apply_orbgrand_page(base_cache[pid]["tokens"], base_cache[pid]["cands"], ORBGRAND_MAX_PATTERNS)
                    rep_ms = (time.perf_counter() - t_rep0) * 1000.0
                    base_cache[pid]["orb_rep"] = rep
                    sc = score_page_prediction(
                        rep["tokens"], p["gt_xml"], hit_eos=base_cache[pid]["hit_eos"],
                        slug=f"{arm_name}-{pid}", workdir=score_workdir,
                    )
                    row = {
                        "id": pid,
                        "domain": p["domain"],
                        "token_f1": sc["tok_lcs_f1"],
                        "n_bars": rep["n_bars"],
                        "n_viol": rep["n_viol_after"],
                        "n_viol_before": rep["n_viol_before"],
                        "n_repaired": rep["n_repaired"],
                        "queries": rep["total_queries"],
                        "mean_p_not_found": rep["mean_p_not_found"],
                        "flagged": rep["flagged_for_3view"],
                        "elapsed_ms": round(base_cache[pid]["base_ms"] + rep_ms, 1),
                        **sc,
                    }
                elif arm_name == "trellis_bma_3view":
                    rep = base_cache[pid].get("orb_rep") or apply_orbgrand_page(
                        base_cache[pid]["tokens"], base_cache[pid]["cands"], ORBGRAND_MAX_PATTERNS
                    )
                    t_mv0 = time.perf_counter()
                    if rep["flagged_for_3view"]:
                        v1_toks = rep["tokens"]
                        v2_toks, _, v2_eos = _decode_fsa_with_top_cands(model, _make_contrast_view(pil_im), MAX_DIM, MAX_LEN, device, use_bf16)
                        v3_toks, _, v3_eos = _decode_fsa_with_top_cands(model, pil_im, 1280, MAX_LEN, device, use_bf16)
                        fused = trellis_bma_fuse_views([v1_toks, v2_toks, v3_toks])
                        fused_eos = bool(base_cache[pid]["hit_eos"] or v2_eos or v3_eos)
                    else:
                        fused = rep["tokens"]
                        fused_eos = bool(base_cache[pid]["hit_eos"])
                    mv_ms = (time.perf_counter() - t_mv0) * 1000.0
                    spans = split_bars_with_spans(fused)
                    n_viol = sum(1 for s, e, c in spans if compute_bar_duration_syndrome(fused[s:e], c)[1] > BAR_EPS)
                    sc = score_page_prediction(fused, p["gt_xml"], hit_eos=fused_eos, slug=f"{arm_name}-{pid}", workdir=score_workdir)
                    row = {
                        "id": pid,
                        "domain": p["domain"],
                        "token_f1": sc["tok_lcs_f1"],
                        "n_bars": len(spans),
                        "n_viol": n_viol,
                        "mean_p_not_found": rep["mean_p_not_found"],
                        "flagged": rep["flagged_for_3view"],
                        "elapsed_ms": round(base_cache[pid]["base_ms"] + mv_ms, 1),
                        **sc,
                    }
                per_page.append(row)
                print(
                    f"[{EXP_ID}][{arm_name}] ({idx}/{len(pages)}) {pid}: "
                    f"note_f1={row['note_f1']:.4f} tok_f1={row['token_f1']:.4f} "
                    f"bsvr={row['bsvr']:.4f} wfpp={row['wfpp']:.1f} "
                    f"viol={row['n_viol']}/{row['n_bars']} rep={row.get('n_repaired', 0)} "
                    f"pnf={row.get('mean_p_not_found', 0.0):.3f} flag={row.get('flagged', False)} "
                    f"ms={row['elapsed_ms']:.0f}",
                    flush=True,
                )

            res = _summarize_arm(arm_name, per_page)
            arm_results[arm_name] = res
            _atomic_write_json(cached_path, res)
            _upload_json_to_hf(hf_token, cached_path, f"{HF_CKPT_PREFIX}/{arm_name}_arm_result.json")

            eval_payload: Dict[str, Any] = {
                "exp_id": EXP_ID,
                "exp_code": EXP_CODE,
                "status": "RUNNING" if len(arm_results) < len(ARMS) else "COMPLETED",
                "started_at": started_at,
                "updated_at": _utc_now(),
                "gpu": gpu_name,
                "ckpt_source": ckpt_source,
                "ckpt_meta": ckpt_meta,
                "arms": arm_results,
                "gate_pass": compute_gate_pass(arm_results),
            }
            eval_local = WORK_ROOT / "out" / "eval.json"
            _atomic_write_json(eval_local, eval_payload)
            _upload_json_to_hf(hf_token, eval_local, HF_EVAL_PATH)

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        status = "COMPLETED"
    except Exception as exc:
        status = "FAILED"
        error_msg = f"{type(exc).__name__}: {exc}"
        print(f"[{EXP_ID}] ERROR: {error_msg}", flush=True)
        raise
    finally:
        total_sec = time.perf_counter() - t_start
        run_payload = {
            "exp_id": EXP_ID,
            "exp_code": EXP_CODE,
            "status": status,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_sec": round(total_sec, 2),
            "n_arms_completed": len(arm_results),
            "eval_path": HF_EVAL_PATH,
            "error": error_msg,
        }
        run_local = WORK_ROOT / "out" / "run.json"
        _atomic_write_json(run_local, run_payload)
        if hf_token:
            try:
                _upload_json_to_hf(hf_token, run_local, HF_RUN_PATH)
            except Exception as up_exc:
                print(f"[{EXP_ID}] WARN: failed to upload run ledger: {up_exc}", flush=True)
        _purge_hf_token()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
