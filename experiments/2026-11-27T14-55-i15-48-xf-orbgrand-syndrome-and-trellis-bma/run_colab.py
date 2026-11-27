#!/usr/bin/env python3
"""Colab A100 / Local RTX 4080 SUPER runner — Iter 15 Exp i15-48: ORBGRAND/SOGRAND + Trellis-BMA.

Experiment ID: 2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma
Origin: Information Theory / IDS Coding (ORBGRAND / SOGRAND IEEE TWC 2024-25 + Trellis-BMA).
Zero-training 3-arm inference proxy on a 12-page hard printed + HW subset of suite31_hw:
  - Arm 0 (greedy_fsa):        Standard FSA-constrained greedy decode.
  - Arm 1 (orbgrand_sogrand):  CPU ORBGRAND ordered-noise search (<=64 patterns/bar) on bars
                               violating duration/BSVR conservation (syndrome != 0) + SOGRAND
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
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma"
EXP_CODE = "i15-48"
ARMS = ("greedy_fsa", "orbgrand_sogrand", "trellis_bma_3view")
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_CKPT_CANDIDATES = (
    "checkpoints/iter14-final/uni_omr_v2_ema_latest.pt",
    "checkpoints/iter14-v2a-mid/uni_omr_v2_ema_latest.pt",
    "checkpoints/iter10_soup_baseline/uni_omr_soup.pt",
)
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"
HF_CKPT_PREFIX = f"checkpoints/{EXP_ID}"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
WORK_ROOT = Path("/vol/i15_48") if Path("/vol").is_dir() else Path("/content/i15_48")

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
    """Compute exact duration sum and conservation syndrome |sum - capacity| for one bar.

    Handles dotted notes, chords (0 duration advance), grace notes (0 advance),
    whole-bar rests (`bar-rest`), and tuplet spans (`tuplet-start-A-N` ... `tuplet-stop`).
    Returns (measured_beats, abs_syndrome, is_valid).
    """
    if not bar_toks:
        return 0.0, 0.0, True
    if "bar-rest" in bar_toks:
        return bar_capacity, 0.0, True

    total = 0.0
    in_chord = False
    in_grace = False
    tuplet_stack: List[float] = []
    last_base_dur: Optional[float] = None
    last_dot_factor = 0.5

    for tok in bar_toks:
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
                total += dur
                last_base_dur = dur
                last_dot_factor = 0.5
            else:
                last_base_dur = None
            in_chord = False
            in_grace = False
            continue
        if tok == "dot" and last_base_dur is not None:
            add = last_base_dur * last_dot_factor
            total += add
            last_dot_factor *= 0.5
            continue
        if tok.startswith("voice-") or tok == "<staff>":
            # Reset voice-level accumulator conservatively by returning best match so far
            break

    if total <= 1e-9:
        # Header-only or non-rhythmic bar (no duration tokens)
        return 0.0, 0.0, True
    syndrome = abs(total - bar_capacity)
    return round(total, 6), round(syndrome, 6), bool(syndrome <= BAR_EPS)


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

    # Rank rhythm-relevant positions by reliability margin (lowest top1 - top2 first)
    flip_positions: List[Tuple[float, int, str, float, float]] = []
    p_orig_bar = 1.0
    for pos, tok in enumerate(bar_toks):
        cands = bar_top_cands[pos] if pos < len(bar_top_cands) else [(tok, 0.99)]
        p1 = max(1e-6, min(1.0 - 1e-6, float(cands[0][1])))
        p_orig_bar *= p1
        if len(cands) < 2:
            continue
        alt_tok, p2_raw = cands[1]
        p2 = max(1e-6, min(1.0 - 1e-6, float(p2_raw)))
        # Focus flips on duration/dot/rest/bar-rest tokens that affect conservation
        is_rhythm = (
            tok in WRITTEN_DUR_BEATS
            or alt_tok in WRITTEN_DUR_BEATS
            or tok in ("dot", "bar-rest")
            or alt_tok in ("dot", "bar-rest")
        )
        if is_rhythm:
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
    # P(unvisited) = clamp(1 - prob_visited, 0, 1).
    # If a valid codeword z* was found, posterior P(z* | visited) = P(z*) / (P(z*) + P_unvisited * 2^-r),
    # and P_not_found = 1 - P(z* | visited). If not found, P_not_found = 1.0.
    p_unvisited = max(0.0, min(1.0, 1.0 - prob_visited))
    if repaired:
        # Effective redundancy factor (2^-3 = 0.125 conservative musical conservation code rate slack)
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
    """Fuse K=3 noisy views of a single bar via IDS Trellis-BMA majority voting.

    Steps:
      1. If 2+ views are identical, return the majority bar immediately.
      2. Otherwise, pick the pivot view with lowest conservation syndrome (tie-break: median length),
         align every other view to the pivot via SequenceMatcher (IDS edit operations), and take
         per-column majority vote (dropping spurious insertions, filling single-view deletions,
         resolving substitutions by majority + conservation tie-break).
    """
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

    # Align each view to pivot positions [0..L-1] plus inter-position insertion slots [0..L]
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

    # Reconstruct consensus bar via BMA column majority
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
            # No majority for deletion -> keep pivot token
            fused.append(pivot[i])
    if len(ins_votes[L]) >= thresh:
        ins_tok, ins_cnt = Counter(ins_votes[L]).most_common(1)[0]
        if ins_cnt >= thresh:
            fused.extend(ins_tok)

    # Conservation guard: if fused worsens syndrome vs best valid view, pick valid view
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
    # Use the median-bar-count view as bar-grid reference
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
            if arm_name in ARMS and isinstance(arm_res, dict) and "all12_token_f1" in arm_res:
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
            if isinstance(arm_res, dict) and "all12_token_f1" in arm_res:
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

    ds_dir = WORK_ROOT / "datasets" / "suite31_hw"
    manifest_candidates = list(ds_dir.rglob("suite31_hw_manifest.json")) if ds_dir.is_dir() else []
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
        manifest_candidates = list(ds_dir.rglob("suite31_hw_manifest.json"))
    if not manifest_candidates:
        raise FileNotFoundError(f"suite31_hw_manifest.json not found in {ds_dir}")

    _, pieces = eval_uni_suite31.resolve_manifest_pieces(manifest_candidates[0])
    printed = [p for p in pieces if str(p.get("domain", "printed")).lower() != "handwritten"]
    hw = [p for p in pieces if str(p.get("domain", "printed")).lower() == "handwritten"]

    # Deterministic hard 12-page subset: 6 printed photo + 6 handwritten photo (fallback clean)
    selected: List[Dict[str, Any]] = []
    for p in printed[:N_HARD_PRINTED]:
        img_p = p.get("photo") or p.get("clean")
        gt_p = p.get("gt_bekern") or p.get("bekern")
        selected.append({"id": p["id"], "domain": "printed", "image": Path(img_p), "gt_bekern": Path(gt_p) if gt_p else None})
    for p in hw[:N_HARD_HW]:
        img_p = p.get("photo") or p.get("clean")
        gt_p = p.get("gt_bekern") or p.get("bekern")
        selected.append({"id": p["id"], "domain": "hw", "image": Path(img_p), "gt_bekern": Path(gt_p) if gt_p else None})
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
) -> Tuple[List[str], List[List[Tuple[str, float]]]]:
    """FSA-constrained greedy decode returning tokens + per-position top-K (tok_str, prob)."""
    import torch
    import torch.nn.functional as F
    import eval_uni_suite31
    import grammar

    inp = _pil_to_tensor(img_pil, max_dim).unsqueeze(0).to(device)
    vocab_size = int(model.cfg.vocab_size)
    pad_id, bos_id, eos_id, _, tok_to_id, id_to_tok = eval_uni_suite31._vocab_spec(vocab_size)

    toks: List[str] = []
    top_cands: List[List[Tuple[str, float]]] = []
    state = grammar.PrefixState(strict=False, vocab_map=tok_to_id)

    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
        mem, _, _ = model.encode_image(inp, return_aux=False)
        cur_ids = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
        past_kvs = None
        for _ in range(max_len):
            logits, _, past_kvs = model.decode_tokens(
                cur_ids,
                memory=mem if past_kvs is None else None,
                return_mtp=False,
                past_kvs=past_kvs,
                use_cache=True,
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
                break
            tok_str = id_to_tok.get(tok_id, "")
            toks.append(tok_str)
            cands = [(id_to_tok.get(int(i.item()), ""), float(p.item())) for i, p in zip(top_i, top_p)]
            top_cands.append(cands)
            state = eval_uni_suite31._advance_grammar_state(state, tok_id, strict=False, vocab_size=vocab_size)
            cur_ids = torch.tensor([[tok_id]], dtype=torch.long, device=device)

    return toks, top_cands


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
# CPU Smoke Test (Mock ORBGRAND/SOGRAND + Trellis-BMA 3-View Consensus)
# ---------------------------------------------------------------------------

def run_smoke_test() -> int:
    """CPU unit & mock test of ORBGRAND bar repair, SOGRAND posterior, and Trellis-BMA 3-view fusion."""
    # 1. Ground-truth 4/4 bar: quarter C 4 quarter D 4 quarter E 4 quarter F 4 bar
    gt_bar = ["time-4-4", "quarter", "C", "4", "quarter", "D", "4", "quarter", "E", "4", "quarter", "F", "4"]
    _, gt_syn, gt_ok = compute_bar_duration_syndrome(gt_bar, 4.0)
    assert gt_ok and gt_syn == 0.0, f"GT bar failed syndrome: {gt_syn}"

    # Corrupt two low-margin durations: quarter -> eighth (sum = 3.0 != 4.0)
    noisy_bar = list(gt_bar)
    noisy_bar[4] = "eighth"
    noisy_bar[7] = "eighth"
    top_cands: List[List[Tuple[str, float]]] = []
    for pos, tok in enumerate(noisy_bar):
        if pos in (4, 7):
            top_cands.append([("eighth", 0.54), ("quarter", 0.44)])
        else:
            top_cands.append([(tok, 0.98), ("half", 0.01)])

    rep = orbgrand_repair_bar(noisy_bar, top_cands, bar_capacity=4.0, max_patterns=ORBGRAND_MAX_PATTERNS)
    assert rep["repaired"], f"ORBGRAND failed to repair 2-error bar: {rep}"
    assert rep["tokens"] == gt_bar, f"ORBGRAND repaired to wrong bar: {rep['tokens']}"
    assert rep["final_syndrome"] == 0.0
    assert 0.0 <= rep["p_not_found"] <= 1.0
    assert rep["queries"] <= ORBGRAND_MAX_PATTERNS

    # 2. Unrepairable bar -> SOGRAND p_not_found == 1.0 -> flagged for 3-view Trellis-BMA
    bad_bar = ["time-4-4", "eighth", "C", "4", "bar"]
    bad_cands = [[(t, 0.99)] for t in bad_bar]
    page_rep = apply_orbgrand_page(bad_bar, bad_cands)
    assert page_rep["flagged_for_3view"] and page_rep["mean_p_not_found"] == 1.0

    # 3. Trellis-BMA 3-view IDS consensus: 3 corrupted views (sub, del, ins) -> exact GT recovery
    gt_stream = ["time-4-4", "quarter", "C", "4", "quarter", "D", "4", "quarter", "E", "4", "quarter", "F", "4", "bar"]
    v1 = ["time-4-4", "eighth", "C", "4", "quarter", "D", "4", "quarter", "E", "4", "quarter", "F", "4", "bar"]  # sub
    v2 = ["time-4-4", "quarter", "C", "4", "quarter", "D", "4", "quarter", "E", "4", "F", "4", "bar"]            # del
    v3 = ["time-4-4", "quarter", "C", "4", "dot", "quarter", "D", "4", "quarter", "E", "4", "quarter", "F", "4", "bar"]  # ins
    fused = trellis_bma_fuse_views([v1, v2, v3])
    assert fused == gt_stream, f"Trellis-BMA mismatch: {fused} != {gt_stream}"
    f1_v1 = _token_lcs_f1(gt_stream, v1)
    f1_fused = _token_lcs_f1(gt_stream, fused)
    assert f1_fused == 1.0 and f1_fused > f1_v1

    print(
        f"[{EXP_CODE}][SMOKE] ORBGRAND repaired 2-err bar in {rep['queries']} queries "
        f"(p_not_found={rep['p_not_found']:.4f}) | Trellis-BMA 3-view F1: {f1_v1:.4f} -> {f1_fused:.4f}",
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

        # Shared Arm-0 cache so Arm 1 & Arm 2 avoid redundant base decoding
        base_cache: Dict[str, Dict[str, Any]] = {}
        for p in pages:
            gt_toks: List[str] = []
            if p["gt_bekern"] is not None and p["gt_bekern"].is_file():
                gt_toks = p["gt_bekern"].read_text(encoding="utf-8", errors="ignore").strip().split()
            p["gt_toks"] = gt_toks

        def _summarize_arm(arm_name: str, per_page: List[Dict[str, Any]]) -> Dict[str, Any]:
            pr_f1 = [r["token_f1"] for r in per_page if r["domain"] == "printed"]
            hw_f1 = [r["token_f1"] for r in per_page if r["domain"] == "hw"]
            all_f1 = [r["token_f1"] for r in per_page]
            tot_bars = sum(r["n_bars"] for r in per_page)
            tot_viol = sum(r["n_viol"] for r in per_page)
            mean_ms = sum(r["elapsed_ms"] for r in per_page) / max(1, len(per_page))
            return {
                "arm": arm_name,
                "ckpt_source": ckpt_source,
                "n_pages": len(per_page),
                "printed_token_f1": round(sum(pr_f1) / max(1, len(pr_f1)), 5),
                "hw_token_f1": round(sum(hw_f1) / max(1, len(hw_f1)), 5),
                "all12_token_f1": round(sum(all_f1) / max(1, len(all_f1)), 5),
                "syndrome_rate": round(tot_viol / max(1, tot_bars), 5),
                "n_bars_total": tot_bars,
                "n_bars_violating": tot_viol,
                "mean_ms_per_page": round(mean_ms, 1),
                "mean_p_not_found": round(sum(r.get("mean_p_not_found", 0.0) for r in per_page) / max(1, len(per_page)), 5),
                "n_flagged_pages": sum(1 for r in per_page if r.get("flagged", False)),
                "per_page": per_page,
            }

        for arm_name in ARMS:
            cached_path = WORK_ROOT / "ckpts" / arm_name / "arm_result.json"
            if arm_name in arm_results:
                continue
            per_page: List[Dict[str, Any]] = []
            for p in pages:
                pid = p["id"]
                pil_im = Image.open(p["image"]).convert("RGB")
                t0 = time.perf_counter()

                if pid not in base_cache:
                    btoks, bcands = _decode_fsa_with_top_cands(model, pil_im, MAX_DIM, MAX_LEN, device, use_bf16)
                    base_ms = (time.perf_counter() - t0) * 1000.0
                    base_cache[pid] = {"tokens": btoks, "cands": bcands, "base_ms": base_ms}

                if arm_name == "greedy_fsa":
                    toks = base_cache[pid]["tokens"]
                    spans = split_bars_with_spans(toks)
                    n_viol = sum(1 for s, e, c in spans if compute_bar_duration_syndrome(toks[s:e], c)[1] > BAR_EPS)
                    per_page.append({
                        "id": pid,
                        "domain": p["domain"],
                        "token_f1": round(_token_lcs_f1(p["gt_toks"], toks), 5),
                        "n_bars": len(spans),
                        "n_viol": n_viol,
                        "elapsed_ms": round(base_cache[pid]["base_ms"], 1),
                    })
                elif arm_name == "orbgrand_sogrand":
                    t_rep0 = time.perf_counter()
                    rep = apply_orbgrand_page(base_cache[pid]["tokens"], base_cache[pid]["cands"], ORBGRAND_MAX_PATTERNS)
                    rep_ms = (time.perf_counter() - t_rep0) * 1000.0
                    base_cache[pid]["orb_rep"] = rep
                    per_page.append({
                        "id": pid,
                        "domain": p["domain"],
                        "token_f1": round(_token_lcs_f1(p["gt_toks"], rep["tokens"]), 5),
                        "n_bars": rep["n_bars"],
                        "n_viol": rep["n_viol_after"],
                        "n_repaired": rep["n_repaired"],
                        "queries": rep["total_queries"],
                        "mean_p_not_found": rep["mean_p_not_found"],
                        "flagged": rep["flagged_for_3view"],
                        "elapsed_ms": round(base_cache[pid]["base_ms"] + rep_ms, 1),
                    })
                elif arm_name == "trellis_bma_3view":
                    rep = base_cache[pid].get("orb_rep") or apply_orbgrand_page(
                        base_cache[pid]["tokens"], base_cache[pid]["cands"], ORBGRAND_MAX_PATTERNS
                    )
                    t_mv0 = time.perf_counter()
                    if rep["flagged_for_3view"]:
                        v1_toks = rep["tokens"]
                        v2_toks, _ = _decode_fsa_with_top_cands(model, _make_contrast_view(pil_im), MAX_DIM, MAX_LEN, device, use_bf16)
                        v3_toks, _ = _decode_fsa_with_top_cands(model, pil_im, 1280, MAX_LEN, device, use_bf16)
                        fused = trellis_bma_fuse_views([v1_toks, v2_toks, v3_toks])
                    else:
                        fused = rep["tokens"]
                    mv_ms = (time.perf_counter() - t_mv0) * 1000.0
                    spans = split_bars_with_spans(fused)
                    n_viol = sum(1 for s, e, c in spans if compute_bar_duration_syndrome(fused[s:e], c)[1] > BAR_EPS)
                    per_page.append({
                        "id": pid,
                        "domain": p["domain"],
                        "token_f1": round(_token_lcs_f1(p["gt_toks"], fused), 5),
                        "n_bars": len(spans),
                        "n_viol": n_viol,
                        "mean_p_not_found": rep["mean_p_not_found"],
                        "flagged": rep["flagged_for_3view"],
                        "elapsed_ms": round(base_cache[pid]["base_ms"] + mv_ms, 1),
                    })

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
            }
            eval_local = WORK_ROOT / "out" / "eval.json"
            _atomic_write_json(eval_local, eval_payload)
            _upload_json_to_hf(hf_token, eval_local, HF_EVAL_PATH)

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
