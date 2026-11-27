#!/usr/bin/env python3
"""Colab L4 / Modal L4 runner for Wave 13 Exp i15-07 (bekern3 Architect 2):
2026-11-27T07-52-i15-07-bekern3-2d-cursor-and-bar-rewind

2-arm 1,200-step ablation on `iter10_5_proxy_2k` (`UniOMRv2Net(small_45m)`, BF16, EMA 0.99):
  - arm0_bekern_v2a: Control ungrounded bekern v2a + `<sys>`, standard greedy decode.
  - arm1_bekern3_spatial:
      * Quantized 2D spatial cursors `@y00..@y63` after `<bos>`/`<sys>` and `@x00..@x63`
        after `bar` (`BarClose`), with train-only CE-masked (`ignore_index=-100`) `@y?`/`@x?`
        fallbacks for noisy/unaligned handwritten staves.
      * Macro-topology tokens (`<part>`, `<s1>..<s12>`, `<pickup>`, `bar-split`) with exact
        integer `PPQ=40320` tick accounting.
      * Evaluated in two decode modes on the same trained EMA weights:
          1) `arm1_bekern3_spatial_fsa`: Strict 2D monotonic FSA (`y[s] > y[s-1]` & `x[b] > x[b-1]`,
             zero post-hoc n-gram hacks).
          2) `arm1_bekern3_spatial_kv_rewind`: Same strict 2D FSA + eval-time local single-bar
             KV-cache rewind (`beam=4`) triggered on exact `PPQ=40320` beat-sum mismatch.

Measures: `loop_truncation_count` / `loop_truncation_rate`, `sys_count_exact`,
`ryans-jig` & `mozart-k387` Note F1, `aloha-oe` cutaway & pickup `S_play`,
`pct_bars_rewound`, `latency_overhead_pct` (gate `<= +12%`), plus full `suite31_hw` slices.

Uploads `evals/<EXP_ID>.json` and `runs/<EXP_ID>.json` to HF `gregoreesmaa/sheet-music-converter-omr`.
Resume-safe (5-minute atomic checkpoints + per-arm `arm_done.json`). Never prints `HF_TOKEN`.
"""

from __future__ import annotations

from collections import Counter
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
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
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

EXP_ID = "2026-11-27T07-52-i15-07-bekern3-2d-cursor-and-bar-rewind"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
IGNORE_INDEX = -100

# 2D Spatial Cursor & Macro-Topology constants
N_Y_BINS = 64
N_X_BINS = 64
Y_TOKENS: Tuple[str, ...] = tuple(f"@y{i:02d}" for i in range(N_Y_BINS))
Y_UNK_TOKEN = "@y?"
X_TOKENS: Tuple[str, ...] = tuple(f"@x{i:02d}" for i in range(N_X_BINS))
X_UNK_TOKEN = "@x?"
PART_TOKEN = "<part>"
STAFF_TOKENS: Tuple[str, ...] = tuple(f"<s{i}>" for i in range(1, 13))
PICKUP_TOKEN = "<pickup>"
BAR_SPLIT_TOKEN = "bar-split"

BEKERN3_EXTRA_TOKENS: Tuple[str, ...] = (
    Y_TOKENS
    + (Y_UNK_TOKEN,)
    + X_TOKENS
    + (X_UNK_TOKEN, PART_TOKEN)
    + STAFF_TOKENS
    + (PICKUP_TOKEN, BAR_SPLIT_TOKEN)
)

# Exact integer PPQ = 40320 = lcm(1..10, 12, 16, 32, 64) * ... (quarter = 40320)
PPQ = 40320
DUR_TICKS: Dict[str, int] = {
    "breve": PPQ * 8,
    "whole": PPQ * 4,
    "half": PPQ * 2,
    "quarter": PPQ,
    "eighth": PPQ // 2,
    "16th": PPQ // 4,
    "32nd": PPQ // 8,
    "64th": PPQ // 16,
    "128th": PPQ // 32,
}

# Training & Eval hyperparameters
STEPS_PER_ARM = int(os.environ.get("I15_07_STEPS", "1200"))
SEED = 42
GRAD_ACCUM = 4
LR = 3.0e-4
MIN_LR = 3.0e-5
WARMUP_STEPS = 100
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
EVAL_MAX_LEN = 1536
GPU_EMA_DECAY = 0.99
SAVE_EVERY_MINUTES = 5.0
VAL_LIMIT = int(os.environ.get("I15_07_VAL_LIMIT", "48"))
SUITE31_LIMIT = int(os.environ.get("I15_07_SUITE31_LIMIT", "0"))
REWIND_BEAM_WIDTH = int(os.environ.get("I15_07_REWIND_BEAM", "4"))
MAX_BAR_TOKENS = 96
MAX_REWINDS_PER_PAGE = 12


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_vol_root() -> Path:
    if Path("/vol").is_dir():
        return Path("/vol")
    if Path("/content").is_dir():
        return Path("/content/vol")
    return Path("/tmp/omr_vol")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_hf_token() -> str:
    tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not tok and HF_TOKEN_FILE.is_file():
        try:
            os.chmod(HF_TOKEN_FILE, 0o600)
        except OSError:
            pass
        tok = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass
    if not tok:
        raise RuntimeError("HF token missing: set HF_TOKEN env var or /tmp/.hf_token")
    return tok


def _ensure_deps() -> None:
    needed = []
    for mod, pkg in (
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
    ):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])


def _locate_mvd() -> Path:
    for cand in (
        Path("/workspace/mvd"),
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path.cwd() / "tmp-train" / "mvd",
        Path(__file__).resolve().parents[2] / "tmp-train" / "mvd",
    ):
        if (cand / "uni_omr_v2.py").is_file() and (cand / "rlvr_grpo.py").is_file():
            return cand
    for bundle in (Path("/content/bundle.tar.gz"), Path("/content/repo_bundle.tar.gz")):
        if bundle.is_file():
            dest = Path("/content/sheet-music-converter")
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=dest)
            mvd = dest / "tmp-train" / "mvd"
            if (mvd / "uni_omr_v2.py").is_file():
                return mvd
    raise FileNotFoundError("Could not locate tmp-train/mvd containing uni_omr_v2.py")


def _ensure_proxy_dataset(vol_root: Path, hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_dir = vol_root / "datasets" / "iter10_5_proxy_2k"
    train_abs = ds_dir / "proxy_train_abs.jsonl"
    val_abs = ds_dir / "proxy_val_abs.jsonl"
    if train_abs.is_file() and val_abs.is_file():
        return train_abs, val_abs

    ds_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR} from {HF_REPO_ID}...", flush=True)
    tar_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_PROXY_TAR,
        repo_type="dataset",
        token=hf_token,
        local_dir=str(vol_root / "tars"),
    )
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=ds_dir)
    Path(tar_path).unlink(missing_ok=True)

    for src_name, dst_path in (("proxy_train.jsonl", train_abs), ("proxy_val.jsonl", val_abs)):
        src_path = ds_dir / src_name
        out_lines: List[str] = []
        for raw in src_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            rec["image"] = str((ds_dir / rec["image"]).resolve())
            rec["bekerns"] = [str((ds_dir / b).resolve()) for b in rec.get("bekerns", [])]
            out_lines.append(json.dumps(rec))
        dst_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return train_abs, val_abs


def _ensure_suite31(vol_root: Path, hf_token: str) -> Path:
    from huggingface_hub import hf_hub_download

    s31_dir = vol_root / "datasets" / "suite31_hw_eval"
    sentinel = s31_dir / ".extracted"
    if not sentinel.exists():
        print(f"[{EXP_ID}] Downloading {HF_SUITE31_TAR} from {HF_REPO_ID}...", flush=True)
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=HF_SUITE31_TAR,
                local_dir=str(vol_root / "tars"),
                token=hf_token,
            )
        )
        s31_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=s31_dir)
        sentinel.write_text(_utc_now() + "\n", encoding="utf-8")
        tar_p.unlink(missing_ok=True)
    manifests = sorted(s31_dir.rglob("suite31_hw_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Missing suite31_hw_manifest.json under {s31_dir}")
    return manifests[0]


def _upload_json(hf_token: str, local_path: Path, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"{EXP_ID}: upload {path_in_repo}",
    )
    print(f"[{EXP_ID}] Uploaded {path_in_repo} -> {HF_REPO_ID}", flush=True)


def _load_manifest_rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _read_page_tokens(rec: Dict[str, Any], sys_token: str) -> List[str]:
    if "tokens" in rec and isinstance(rec["tokens"], list):
        return [str(t) for t in rec["tokens"]]
    chunks: List[List[str]] = []
    for bp in rec.get("bekerns", []):
        p = Path(bp)
        if p.is_file():
            toks = p.read_text(encoding="utf-8").split()
            if toks:
                chunks.append(toks)
    out: List[str] = []
    for idx, ch in enumerate(chunks):
        if idx > 0:
            out.append(sys_token)
        out.extend(ch)
    return out


# ---------------------------------------------------------------------------
# Exact PPQ=40320 Integer Tick Accounting & Meter Tracking
# ---------------------------------------------------------------------------

def parse_time_sig_ticks(tok: str) -> Optional[int]:
    """Parse `time-NUM-DEN` into exact measure length in PPQ=40320 integer ticks."""
    if not tok.startswith("time-"):
        return None
    parts = tok.split("-")
    if len(parts) != 3:
        return None
    try:
        num = int(parts[1])
        den = int(parts[2])
    except ValueError:
        return None
    if num <= 0 or den <= 0 or (4 * PPQ) % den != 0:
        return None
    return num * ((4 * PPQ) // den)


def compute_bar_ticks_ppq(bar_toks: Sequence[str]) -> Tuple[int, bool]:
    """Compute exact PPQ=40320 ticks for a single measure's token sequence.

    Returns `(ticks, has_bar_rest)`. Handles chords (zero extra advance),
    dots (exact geometric halving), and nested `tuplet-start-A-N ... tuplet-stop`.
    """
    total = 0
    has_bar_rest = False
    tuplet_stack: List[Tuple[int, int]] = []
    in_chord = False
    n = len(bar_toks)
    i = 0
    while i < n:
        t = bar_toks[i]
        if t == "bar-rest":
            has_bar_rest = True
            i += 1
            continue
        if t == "chord":
            in_chord = True
            i += 1
            continue
        if t.startswith("tuplet-start-"):
            parts = t.split("-")
            if len(parts) == 4:
                try:
                    tuplet_stack.append((int(parts[2]), int(parts[3])))
                except ValueError:
                    pass
            i += 1
            continue
        if t == "tuplet-stop":
            if tuplet_stack:
                tuplet_stack.pop()
            i += 1
            continue
        if t in DUR_TICKS:
            base = DUR_TICKS[t]
            # Look ahead past pitch/rest/alter tokens for trailing `dot` markers
            j = i + 1
            dots = 0
            while j < n and bar_toks[j] not in DUR_TICKS and bar_toks[j] not in (
                "bar", "bar-rest", "chord", "tuplet-stop", "<sys>", PICKUP_TOKEN, BAR_SPLIT_TOKEN,
            ) and not bar_toks[j].startswith(("tuplet-start-", "@x", "@y", "time-", "clef-", "key")):
                if bar_toks[j] == "dot":
                    dots += 1
                j += 1
            dur = base
            add = base // 2
            for _ in range(dots):
                dur += add
                add //= 2
            for actual, normal in tuplet_stack:
                if actual > 0:
                    dur = (dur * normal) // actual
            if not in_chord:
                total += dur
            in_chord = False
            i += 1
            continue
        i += 1
    return total, has_bar_rest


def compute_s_play_ppq(toks: Sequence[str], sys_token: str = "<sys>") -> Tuple[float, int, int]:
    """Compute exact `PPQ=40320` playability `S_play`: fraction of closed bars that
    satisfy the active time signature (allowing valid `<pickup>`, `bar-split`, and `bar-rest`).
    """
    expected = 4 * PPQ  # default 4/4 if header omitted
    cur_bar: List[str] = []
    bar_idx = 0
    valid_bars = 0
    total_bars = 0
    is_pickup = False
    is_split = False

    for t in toks:
        ts = parse_time_sig_ticks(t)
        if ts is not None:
            expected = ts
        if t == PICKUP_TOKEN:
            is_pickup = True
            continue
        if t == BAR_SPLIT_TOKEN:
            is_split = True
            continue
        if t == sys_token:
            cur_bar = []
            is_pickup = False
            is_split = False
            continue
        if t == "bar":
            ticks, has_br = compute_bar_ticks_ppq(cur_bar)
            if ticks > 0 or has_br:
                total_bars += 1
                if has_br or ticks == expected:
                    valid_bars += 1
                elif (is_pickup or bar_idx == 0) and 0 < ticks < expected:
                    valid_bars += 1
                elif is_split and 0 < ticks < expected:
                    valid_bars += 1
                bar_idx += 1
            cur_bar = []
            is_pickup = False
            is_split = False
            continue
        cur_bar.append(t)

    s_play = float(valid_bars) / float(total_bars) if total_bars > 0 else 0.0
    return round(s_play, 4), valid_bars, total_bars


# ---------------------------------------------------------------------------
# 2D Spatial Target Synthesis (`@y00..@y63` / `@x00..@x63` + `@y?`/`@x?`)
# ---------------------------------------------------------------------------

def _is_noisy_hw_record(rec: Dict[str, Any], img_path: Path) -> bool:
    s = (str(img_path) + " " + str(rec.get("source", "")) + " " + str(rec.get("domain", ""))).lower()
    return any(k in s for k in ("hw", "handwritten", "debussy", "rism", "mozart-sq", "manuscript"))


def detect_2d_spatial_anchors(
    img_path: Path,
    sys_bar_counts: Sequence[int],
    is_noisy_hw: bool = False,
) -> Tuple[List[Optional[int]], List[List[Optional[int]]]]:
    """Synthesize strictly monotonic `@y` system anchors and per-system `@x` barline anchors
    from 2D ink-projection / StaffUNet2D-compatible band profiles.

    Returns `(y_bins, x_bins_per_sys)` where `None` triggers train-only CE-masked `@y?` / `@x?`
    on noisy/unaligned handwritten regions.
    """
    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter1d

    n_sys = max(1, min(len(sys_bar_counts), N_Y_BINS))
    with Image.open(img_path) as im:
        gray = np.asarray(im.convert("L").resize((384, 512)), dtype=np.float32) / 255.0
    ink = np.clip(1.0 - gray, 0.0, 1.0)

    # 1. Vertical system bands (horizontal ink projection)
    h_prof = ink.mean(axis=1)
    y_edges = np.linspace(0, h_prof.shape[0], N_Y_BINS + 1, dtype=int)
    y_binned = np.array(
        [h_prof[y_edges[i] : max(y_edges[i] + 1, y_edges[i + 1])].mean() for i in range(N_Y_BINS)],
        dtype=np.float32,
    )
    y_smooth = gaussian_filter1d(y_binned, sigma=1.2)
    y_peak = float(y_smooth.max()) if y_smooth.size else 0.0
    y_contrast = (y_peak - float(np.median(y_smooth))) / (y_peak + 1e-6) if y_peak > 1e-4 else 0.0

    raw_y: List[int] = []
    if y_peak > 1e-4:
        norm = y_smooth / y_peak
        thr = max(0.20, float(np.median(norm) + 0.30 * np.std(norm)))
        active = norm >= thr
        in_seg = False
        s0 = 0
        for idx in range(N_Y_BINS):
            if active[idx] and not in_seg:
                in_seg = True
                s0 = idx
            elif not active[idx] and in_seg:
                in_seg = False
                raw_y.append((s0 + idx - 1) // 2)
        if in_seg:
            raw_y.append((s0 + N_Y_BINS - 1) // 2)

    low_y_conf = is_noisy_hw and (y_contrast < 0.25 or abs(len(raw_y) - n_sys) > max(2, n_sys // 2))
    if len(raw_y) == n_sys:
        chosen_y = raw_y
    elif len(raw_y) > n_sys:
        idxs = np.linspace(0, len(raw_y) - 1, n_sys).round().astype(int)
        chosen_y = [raw_y[int(k)] for k in idxs]
    else:
        nz = np.where(y_smooth > max(1e-5, 0.15 * y_peak))[0]
        y_lo = int(nz[0]) if nz.size else int(0.08 * (N_Y_BINS - 1))
        y_hi = int(nz[-1]) if nz.size else int(0.92 * (N_Y_BINS - 1))
        if y_hi - y_lo < n_sys - 1:
            y_lo, y_hi = 0, N_Y_BINS - 1
        chosen_y = [int(round(v)) for v in np.linspace(y_lo, y_hi, n_sys)]

    mono_y: List[int] = []
    for k, b in enumerate(chosen_y):
        lo = 0 if k == 0 else mono_y[-1] + 1
        hi = (N_Y_BINS - 1) - (n_sys - 1 - k)
        mono_y.append(max(lo, min(hi, int(b))))

    y_out: List[Optional[int]] = [None if low_y_conf else y for y in mono_y]

    # 2. Horizontal barline-x peaks per system band
    H, W = ink.shape
    x_edges = np.linspace(0, W, N_X_BINS + 1, dtype=int)
    x_out_all: List[List[Optional[int]]] = []
    for s_idx in range(n_sys):
        n_bars = int(sys_bar_counts[s_idx]) if s_idx < len(sys_bar_counts) else 0
        if n_bars <= 0:
            x_out_all.append([])
            continue
        yc = mono_y[s_idx]
        half_span = max(2, (N_Y_BINS // max(2, 2 * n_sys)))
        r0 = int(max(0, (yc - half_span) * H // N_Y_BINS))
        r1 = int(min(H, (yc + half_span + 1) * H // N_Y_BINS))
        band = ink[r0:max(r0 + 1, r1), :]

        # Barlines exhibit high vertical ink continuity across the band
        v_prof = (band > 0.25).mean(axis=0).astype(np.float32)
        x_binned = np.array(
            [v_prof[x_edges[i] : max(x_edges[i] + 1, x_edges[i + 1])].max() for i in range(N_X_BINS)],
            dtype=np.float32,
        )
        x_smooth = gaussian_filter1d(x_binned, sigma=0.8)
        x_peak = float(x_smooth.max()) if x_smooth.size else 0.0
        x_med = float(np.median(x_smooth)) if x_smooth.size else 0.0

        cand_x: List[int] = []
        if x_peak > 0.15:
            thr_x = max(0.20, x_med + 0.45 * (x_peak - x_med))
            for idx in range(3, N_X_BINS - 1):
                if x_smooth[idx] >= thr_x and x_smooth[idx] >= x_smooth[idx - 1] and x_smooth[idx] >= x_smooth[idx + 1]:
                    cand_x.append(idx)

        low_x_conf = is_noisy_hw and (x_peak < 0.28 or abs(len(cand_x) - n_bars) > max(2, n_bars // 2))
        if low_x_conf:
            x_out_all.append([None] * n_bars)
            continue

        if len(cand_x) >= n_bars:
            idxs = np.linspace(0, len(cand_x) - 1, n_bars).round().astype(int)
            chosen_x = [cand_x[int(k)] for k in idxs]
        else:
            x_lo = max(4, cand_x[0] if cand_x else 8)
            x_hi = min(N_X_BINS - 1, cand_x[-1] if cand_x else N_X_BINS - 2)
            if x_hi - x_lo < n_bars - 1:
                x_lo, x_hi = 0, N_X_BINS - 1
            chosen_x = [int(round(v)) for v in np.linspace(x_lo, x_hi, n_bars)]

        mono_x: List[Optional[int]] = []
        prev_x = -1
        eff_bars = min(n_bars, N_X_BINS)
        for k in range(n_bars):
            if k >= eff_bars:
                mono_x.append(None)
                continue
            lo = prev_x + 1
            hi = (N_X_BINS - 1) - (eff_bars - 1 - k)
            val = max(lo, min(hi, int(chosen_x[k])))
            mono_x.append(val)
            prev_x = val
        x_out_all.append(mono_x)

    return y_out, x_out_all


def encode_bekern3_spatial(
    gt_toks: Sequence[str],
    y_bins: Sequence[Optional[int]],
    x_bins_per_sys: Sequence[Sequence[Optional[int]]],
    sys_token: str,
) -> List[str]:
    """Transform base `bekern` tokens into `bekern3` with `@y/@x` spatial cursors,
    `<part> <s1>`, `<pickup>`, and `bar-split` markers.
    """
    systems: List[List[str]] = []
    cur: List[str] = []
    for t in gt_toks:
        if t == sys_token:
            systems.append(cur)
            cur = []
        else:
            cur.append(t)
    systems.append(cur)

    out: List[str] = []
    active_expected = 4 * PPQ
    global_bar_idx = 0

    for s_idx, sys_toks in enumerate(systems):
        if s_idx > 0:
            out.append(sys_token)
        yb = y_bins[min(s_idx, len(y_bins) - 1)] if y_bins else None
        out.append(Y_TOKENS[yb] if yb is not None else Y_UNK_TOKEN)
        out.append(PART_TOKEN)
        out.append(STAFF_TOKENS[0])  # <s1> (multi-staff pages extend to <s2>..<s12>)

        x_list = x_bins_per_sys[s_idx] if s_idx < len(x_bins_per_sys) else []
        # Split system into bars to tag `<pickup>` and `bar-split` using exact PPQ=40320
        bars: List[List[str]] = []
        cur_bar: List[str] = []
        for t in sys_toks:
            ts = parse_time_sig_ticks(t)
            if ts is not None:
                active_expected = ts
            if t == "bar":
                bars.append(cur_bar)
                cur_bar = []
            else:
                cur_bar.append(t)
        trailing = cur_bar

        for b_idx, bar_toks in enumerate(bars):
            ticks, has_br = compute_bar_ticks_ppq(bar_toks)
            if global_bar_idx == 0 and (not has_br) and 0 < ticks < active_expected:
                out.append(PICKUP_TOKEN)
            elif b_idx == len(bars) - 1 and s_idx < len(systems) - 1 and (not has_br) and 0 < ticks < active_expected:
                out.append(BAR_SPLIT_TOKEN)
            out.extend(bar_toks)
            out.append("bar")
            xb = x_list[b_idx] if b_idx < len(x_list) else None
            out.append(X_TOKENS[xb] if xb is not None else X_UNK_TOKEN)
            global_bar_idx += 1

        if trailing:
            ticks, has_br = compute_bar_ticks_ppq(trailing)
            if s_idx < len(systems) - 1 and (not has_br) and 0 < ticks < active_expected:
                out.append(BAR_SPLIT_TOKEN)
            out.extend(trailing)

    return out


def strip_bekern3_tokens(toks: Sequence[str]) -> Tuple[List[str], List[int], List[List[int]]]:
    """Strip `@y*`, `@x*`, `<part>`, `<s1..s12>`, `<pickup>`, `bar-split` for base `bekern` eval,
    returning `(clean_bekern_tokens, y_bins, x_bins_per_sys)`.
    """
    clean: List[str] = []
    ybins: List[int] = []
    xbins_all: List[List[int]] = [[]]
    meta_set = {Y_UNK_TOKEN, X_UNK_TOKEN, PART_TOKEN, PICKUP_TOKEN, BAR_SPLIT_TOKEN} | set(STAFF_TOKENS)
    for t in toks:
        if t.startswith("@y") and len(t) == 4 and t[2:].isdigit():
            ybins.append(int(t[2:]))
        elif t.startswith("@x") and len(t) == 4 and t[2:].isdigit():
            xbins_all[-1].append(int(t[2:]))
        elif t in meta_set:
            continue
        else:
            if t == "<sys>":
                xbins_all.append([])
            clean.append(t)
    return clean, ybins, xbins_all


# ---------------------------------------------------------------------------
# Strict 2D Monotonic FSA + Eval-Time Local Single-Bar KV-Cache Rewind
# ---------------------------------------------------------------------------

@dataclass
class SpatialFSAState:
    expect_y: bool = True
    prev_y: int = -1
    expect_x: bool = False
    prev_x: int = -1

    def clone(self) -> "SpatialFSAState":
        return SpatialFSAState(self.expect_y, self.prev_y, self.expect_x, self.prev_x)


@dataclass
class VocabIds:
    bos_id: int
    eos_id: int
    pad_id: int
    sys_id: int
    bar_id: int
    y_start: Optional[int] = None
    y_unk: Optional[int] = None
    x_start: Optional[int] = None
    x_unk: Optional[int] = None
    pickup_id: Optional[int] = None
    bar_split_id: Optional[int] = None


def _apply_2d_fsa_mask(step_logits: Any, state: SpatialFSAState, v: VocabIds) -> Any:
    import torch

    if v.y_start is None or v.x_start is None:
        return step_logits
    y_lo, y_hi = v.y_start, v.y_start + N_Y_BINS
    x_lo, x_hi = v.x_start, v.x_start + N_X_BINS

    # Never emit train-only unknown cursor tokens at eval
    if v.y_unk is not None:
        step_logits[v.y_unk] = float("-inf")
    if v.x_unk is not None:
        step_logits[v.x_unk] = float("-inf")

    if state.expect_y:
        mask = torch.full_like(step_logits, float("-inf"))
        valid_lo = y_lo + max(0, state.prev_y + 1)
        if valid_lo < y_hi:
            mask[valid_lo:y_hi] = 0.0
        else:
            mask[v.eos_id] = 0.0
        return step_logits + mask

    if state.expect_x:
        mask = torch.full_like(step_logits, float("-inf"))
        valid_lo = x_lo + max(0, state.prev_x + 1)
        if valid_lo < x_hi:
            mask[valid_lo:x_hi] = 0.0
        else:
            mask[v.sys_id] = 0.0
            mask[v.eos_id] = 0.0
        return step_logits + mask

    # Regular notation positions: forbid all @y* and @x*
    step_logits[y_lo:y_hi] = float("-inf")
    step_logits[x_lo:x_hi] = float("-inf")
    if state.prev_y >= N_Y_BINS - 1:
        step_logits[v.sys_id] = float("-inf")
    if state.prev_x >= N_X_BINS - 1:
        step_logits[v.bar_id] = float("-inf")
    return step_logits


def _advance_2d_fsa(state: SpatialFSAState, nxt: int, v: VocabIds) -> None:
    if v.y_start is None or v.x_start is None:
        return
    if state.expect_y and v.y_start <= nxt < v.y_start + N_Y_BINS:
        state.prev_y = nxt - v.y_start
        state.expect_y = False
        state.prev_x = -1
    elif state.expect_x and v.x_start <= nxt < v.x_start + N_X_BINS:
        state.prev_x = nxt - v.x_start
        state.expect_x = False
    elif nxt == v.sys_id:
        state.expect_y = True
        state.expect_x = False
        state.prev_x = -1
    elif nxt == v.bar_id:
        state.expect_x = True


def _slice_past_kvs(past_kvs: List[Any], seq_len: int) -> List[Any]:
    """Slice decoder self-attention KV cache back to `seq_len`, reusing cross-attention K/V."""
    sliced: List[Any] = []
    for self_kv, cross_kv in past_kvs:
        if self_kv is None:
            sliced.append((None, cross_kv))
        else:
            sk, sv = self_kv
            sliced.append(((sk[:, :, :seq_len, :].contiguous(), sv[:, :, :seq_len, :].contiguous()), cross_kv))
    return sliced


def _kv_seq_len(past_kvs: List[Any]) -> int:
    if not past_kvs or past_kvs[0][0] is None:
        return 0
    return int(past_kvs[0][0][0].shape[2])


def _local_bar_beam_rewind(
    model: Any,
    base_past_kvs: List[Any],
    last_tok_id: int,
    base_fsa: SpatialFSAState,
    expected_ticks: int,
    is_first_bar: bool,
    v: VocabIds,
    id_to_tok: Dict[int, str],
    *,
    beam_width: int,
    max_bar_tokens: int,
    device: Any,
    use_bf16: bool,
) -> Optional[Tuple[List[int], Any, SpatialFSAState]]:
    """Run a bounded local `beam=4` search from the previous `BarClose` KV-cache snapshot,
    returning the highest-log-prob beat-consistent bar `(bar_ids, new_past_kvs, new_fsa)`.
    """
    import torch
    import torch.nn.functional as F

    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_bf16 and device.type == "cuda") else contextlib.nullcontext()

    # Each beam: (log_prob, tok_ids, past_kvs, fsa_state, last_id, finished)
    beams: List[Tuple[float, List[int], List[Any], SpatialFSAState, int, bool]] = [
        (0.0, [], _slice_past_kvs(base_past_kvs, _kv_seq_len(base_past_kvs)), base_fsa.clone(), last_tok_id, False)
    ]
    completed_valid: List[Tuple[float, List[int], List[Any], SpatialFSAState]] = []

    with torch.no_grad(), ctx:
        for _ in range(max_bar_tokens):
            next_beams: List[Tuple[float, List[int], List[Any], SpatialFSAState, int, bool]] = []
            for score, toks, kvs, fsa, cur_id, done in beams:
                if done:
                    continue
                inp = torch.tensor([[cur_id]], dtype=torch.long, device=device)
                logits, _, new_kvs = model.decode_tokens(
                    inp,
                    memory=None,
                    return_mtp=False,
                    return_aux=False,
                    past_kvs=kvs,
                    use_cache=True,
                    apply_tok_dropout=False,
                )
                step_logits = _apply_2d_fsa_mask(logits[0, -1, :].float().clone(), fsa, v)
                log_probs = F.log_softmax(step_logits, dim=-1)
                topv, topi = torch.topk(log_probs, k=min(beam_width, step_logits.numel()))
                for lp_t, tid_t in zip(topv.tolist(), topi.tolist()):
                    if math.isinf(lp_t) or math.isnan(lp_t):
                        continue
                    tid = int(tid_t)
                    new_score = score + float(lp_t)
                    new_toks = toks + [tid]
                    new_fsa = fsa.clone()
                    _advance_2d_fsa(new_fsa, tid, v)
                    if tid == v.bar_id:
                        bar_str = [id_to_tok[i] for i in toks if i in id_to_tok]
                        ticks, has_br = compute_bar_ticks_ppq(bar_str)
                        is_pu = (v.pickup_id in toks) or is_first_bar
                        is_sp = v.bar_split_id in toks
                        valid = has_br or (ticks == expected_ticks) or ((is_pu or is_sp) and 0 < ticks < expected_ticks)
                        if valid:
                            completed_valid.append((new_score / max(1, len(new_toks)), new_toks, new_kvs, new_fsa))
                    elif tid in (v.eos_id, v.pad_id, v.sys_id):
                        continue
                    else:
                        next_beams.append((new_score, new_toks, new_kvs, new_fsa, tid, False))
            if not next_beams:
                break
            next_beams.sort(key=lambda b: b[0] / max(1, len(b[1])), reverse=True)
            beams = next_beams[:beam_width]
            if len(completed_valid) >= beam_width:
                break

    if not completed_valid:
        return None
    completed_valid.sort(key=lambda c: c[0], reverse=True)
    _, best_toks, best_kvs, best_fsa = completed_valid[0]
    return best_toks, best_kvs, best_fsa


def _decode_page(
    model: Any,
    img_t: Any,
    v: VocabIds,
    id_to_tok: Dict[int, str],
    *,
    use_fsa: bool,
    use_kv_rewind: bool,
    max_len: int,
    device: Any,
    use_bf16: bool,
) -> Tuple[List[int], Dict[str, Any]]:
    import torch

    t0 = time.perf_counter()
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_bf16 and device.type == "cuda") else contextlib.nullcontext()
    fsa = SpatialFSAState(expect_y=bool(use_fsa and v.y_start is not None))

    bars_total = 0
    bars_rewound = 0
    bars_fixed = 0
    expected_ticks = 4 * PPQ
    bar_toks_ids: List[int] = []
    bar_idx = 0
    rewinds_used = 0
    hit_max_len = False

    with torch.no_grad(), ctx:
        mem, _, _ = model.encode_image(img_t, return_aux=False)
        past_kvs = model.precompute_cross_kv(mem)
        cur_id = v.bos_id
        out_ids: List[int] = []

        # Snapshot at bar boundary (starts right after initial BOS)
        snap_kv_len = 0
        snap_last_id = v.bos_id
        snap_out_len = 0
        snap_fsa = fsa.clone()

        step = 0
        while step < max_len:
            cur_ids = torch.tensor([[cur_id]], dtype=torch.long, device=device)
            logits, _, past_kvs = model.decode_tokens(
                cur_ids,
                memory=None,
                return_mtp=False,
                return_aux=False,
                past_kvs=past_kvs,
                use_cache=True,
                apply_tok_dropout=False,
            )
            step_logits = logits[0, -1, :].float().clone()
            if use_fsa:
                step_logits = _apply_2d_fsa_mask(step_logits, fsa, v)

            nxt = int(step_logits.argmax(dim=-1).item())
            if nxt == v.eos_id or nxt == v.pad_id:
                break

            out_ids.append(nxt)
            tok_str = id_to_tok.get(nxt, "")
            ts = parse_time_sig_ticks(tok_str)
            if ts is not None:
                expected_ticks = ts

            if use_fsa:
                _advance_2d_fsa(fsa, nxt, v)

            # Check measure beat-sum at BarClose (`bar`)
            if nxt == v.bar_id:
                bars_total += 1
                bar_strs = [id_to_tok[i] for i in bar_toks_ids if i in id_to_tok]
                ticks, has_br = compute_bar_ticks_ppq(bar_strs)
                is_pu = (v.pickup_id in bar_toks_ids) or (bar_idx == 0)
                is_sp = v.bar_split_id in bar_toks_ids
                beat_ok = has_br or (ticks == expected_ticks) or ((is_pu or is_sp) and 0 < ticks < expected_ticks)

                if use_kv_rewind and (not beat_ok) and rewinds_used < MAX_REWINDS_PER_PAGE and snap_kv_len > 0:
                    bars_rewound += 1
                    rewinds_used += 1
                    rewind_res = _local_bar_beam_rewind(
                        model,
                        _slice_past_kvs(past_kvs, snap_kv_len),
                        snap_last_id,
                        snap_fsa,
                        expected_ticks,
                        is_first_bar=(bar_idx == 0),
                        v=v,
                        id_to_tok=id_to_tok,
                        beam_width=REWIND_BEAM_WIDTH,
                        max_bar_tokens=MAX_BAR_TOKENS,
                        device=device,
                        use_bf16=use_bf16,
                    )
                    if rewind_res is not None:
                        new_bar_ids, new_kvs, new_fsa = rewind_res
                        out_ids = out_ids[:snap_out_len] + new_bar_ids
                        past_kvs = new_kvs
                        fsa = new_fsa
                        nxt = new_bar_ids[-1]
                        bars_fixed += 1

                bar_toks_ids = []
                bar_idx += 1
            elif nxt == v.sys_id:
                bar_toks_ids = []
            else:
                # Update bar-start snapshot right after @y or @x cursor token
                is_y_tok = v.y_start is not None and v.y_start <= nxt < v.y_start + N_Y_BINS
                is_x_tok = v.x_start is not None and v.x_start <= nxt < v.x_start + N_X_BINS
                if is_y_tok or is_x_tok:
                    snap_kv_len = _kv_seq_len(past_kvs)
                    snap_last_id = nxt
                    snap_out_len = len(out_ids)
                    snap_fsa = fsa.clone()
                else:
                    bar_toks_ids.append(nxt)

            cur_id = nxt
            step += 1
            if step >= max_len:
                hit_max_len = True

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    stats = {
        "latency_ms": elapsed_ms,
        "bars_total": bars_total,
        "bars_rewound": bars_rewound,
        "bars_fixed": bars_fixed,
        "hit_max_len": hit_max_len,
    }
    return out_ids, stats


# ---------------------------------------------------------------------------
# MusicXML & Note F1 Evaluation Helpers
# ---------------------------------------------------------------------------

def _extract_musicxml_notes_and_sys(xml_path: Path) -> Tuple[List[Tuple[str, int, str, int, int]], int]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    def _strip(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    notes: List[Tuple[str, int, str, int, int]] = []
    sys_breaks = 0
    for elem in root.iter():
        tag = _strip(elem.tag)
        if tag == "print" and elem.attrib.get("new-system") == "yes":
            sys_breaks += 1
        elif tag == "note":
            is_rest = False
            is_grace = False
            step = ""
            octave = 4
            alter = 0
            dur = "quarter"
            dots = 0
            for ch in elem:
                ctag = _strip(ch.tag)
                if ctag == "rest":
                    is_rest = True
                elif ctag == "grace":
                    is_grace = True
                elif ctag == "pitch":
                    for pch in ch:
                        ptag = _strip(pch.tag)
                        if ptag == "step" and pch.text:
                            step = pch.text.strip().upper()
                        elif ptag == "octave" and pch.text:
                            try:
                                octave = int(pch.text.strip())
                            except ValueError:
                                octave = 4
                        elif ptag == "alter" and pch.text:
                            try:
                                alter = int(round(float(pch.text.strip())))
                            except ValueError:
                                alter = 0
                elif ctag == "type" and ch.text:
                    dur = ch.text.strip().lower()
                elif ctag == "dot":
                    dots += 1
            if (not is_rest) and (not is_grace) and step:
                notes.append((dur, dots, step, octave, alter))
    return notes, max(1, 1 + sys_breaks)


def _extract_bekern_notes(toks: Sequence[str]) -> List[Tuple[str, int, str, int, int]]:
    durs = {"breve", "whole", "half", "quarter", "eighth", "16th", "32nd", "64th", "128th"}
    steps = {"A", "B", "C", "D", "E", "F", "G"}
    octs = {str(i) for i in range(-1, 9)}
    alters = {"a-2": -2, "a-1": -1, "a1": 1, "a2": 2}
    notes: List[Tuple[str, int, str, int, int]] = []
    cur_dur = "quarter"
    in_rest = False
    i = 0
    n = len(toks)
    while i < n:
        t = toks[i]
        if t == "rest":
            in_rest = True
            i += 1
            continue
        if t in durs:
            if in_rest:
                in_rest = False
            else:
                cur_dur = t
            i += 1
            continue
        if t in steps and (i + 1) < n and toks[i + 1] in octs:
            step = t
            octv = int(toks[i + 1])
            i += 2
            alt = 0
            if i < n and toks[i] in alters:
                alt = alters[toks[i]]
                i += 1
            dots = 0
            while i < n and toks[i] == "dot":
                dots += 1
                i += 1
            notes.append((cur_dur, dots, step, octv, alt))
            continue
        i += 1
    return notes


def _multiset_f1(ref: Sequence[Any], hyp: Sequence[Any]) -> float:
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    rc = Counter(ref)
    hc = Counter(hyp)
    inter = sum((rc & hc).values())
    return float(2 * inter) / float(len(ref) + len(hyp))


def _load_rgb_tensor(img_path: Path, max_dim: int, uni_omr_mod: Any) -> Any:
    from PIL import Image

    with Image.open(img_path) as im:
        pil_im = uni_omr_mod._resize_fullpage_pil(im.convert("RGB"), max_dim)
    return uni_omr_mod._pil_to_chw_tensor(pil_im).unsqueeze(0)


def _build_vocab(uni_omr_mod: Any, use_bekern3: bool) -> Tuple[Dict[str, int], Dict[int, str], int, VocabIds]:
    base_v = int(uni_omr_mod.VOCAB_SIZE)
    tok_to_id = dict(uni_omr_mod.UNI_TOKEN_TO_ID)
    id_to_tok = dict(uni_omr_mod.UNI_ID_TO_TOKEN)
    bar_id = int(tok_to_id["bar"])

    if not use_bekern3:
        v = VocabIds(
            bos_id=uni_omr_mod.BOS_ID,
            eos_id=uni_omr_mod.EOS_ID,
            pad_id=uni_omr_mod.PAD_ID,
            sys_id=uni_omr_mod.SYS_ID,
            bar_id=bar_id,
        )
        return tok_to_id, id_to_tok, base_v, v

    for idx, tok in enumerate(BEKERN3_EXTRA_TOKENS):
        tid = base_v + idx
        tok_to_id[tok] = tid
        id_to_tok[tid] = tok

    v = VocabIds(
        bos_id=uni_omr_mod.BOS_ID,
        eos_id=uni_omr_mod.EOS_ID,
        pad_id=uni_omr_mod.PAD_ID,
        sys_id=uni_omr_mod.SYS_ID,
        bar_id=bar_id,
        y_start=tok_to_id[Y_TOKENS[0]],
        y_unk=tok_to_id[Y_UNK_TOKEN],
        x_start=tok_to_id[X_TOKENS[0]],
        x_unk=tok_to_id[X_UNK_TOKEN],
        pickup_id=tok_to_id[PICKUP_TOKEN],
        bar_split_id=tok_to_id[BAR_SPLIT_TOKEN],
    )
    return tok_to_id, id_to_tok, base_v + len(BEKERN3_EXTRA_TOKENS), v


def _evaluate_arm(
    model: Any,
    use_bekern3: bool,
    use_fsa: bool,
    use_kv_rewind: bool,
    id_to_tok: Dict[int, str],
    v: VocabIds,
    val_rows: Sequence[Dict[str, Any]],
    s31_pieces: Sequence[Dict[str, Any]],
    *,
    device: Any,
    use_bf16: bool,
    uni_omr_mod: Any,
    eval_s31: Any,
    rlvr_grpo: Any,
    eval_suite31: bool = True,
) -> Dict[str, Any]:
    model.eval()

    val_tf1s: List[float] = []
    val_sys_exact = 0
    val_loop_trunc = 0
    val_lat_ms: List[float] = []

    for rec in val_rows[:VAL_LIMIT]:
        img_p = Path(rec["image"])
        if not img_p.is_file():
            continue
        gt_toks = _read_page_tokens(rec, uni_omr_mod.SYS_TOKEN)
        if not gt_toks:
            continue
        gt_sys = 1 + gt_toks.count(uni_omr_mod.SYS_TOKEN)
        img_t = _load_rgb_tensor(img_p, FULLPAGE_MAX_DIM, uni_omr_mod).to(device)
        pred_ids, st = _decode_page(
            model, img_t, v, id_to_tok,
            use_fsa=use_fsa, use_kv_rewind=use_kv_rewind,
            max_len=EVAL_MAX_LEN, device=device, use_bf16=use_bf16,
        )
        val_lat_ms.append(st["latency_ms"])
        pred_raw = [id_to_tok[i] for i in pred_ids if i in id_to_tok]
        pred_clean = strip_bekern3_tokens(pred_raw)[0] if use_bekern3 else pred_raw
        pred_sys = 1 + pred_clean.count(uni_omr_mod.SYS_TOKEN) if pred_clean else 0
        val_tf1s.append(float(eval_s31.compute_token_lcs_f1(gt_toks, pred_clean)))
        if pred_sys == gt_sys:
            val_sys_exact += 1
        if st["hit_max_len"] or rlvr_grpo.has_repetition_loop(pred_clean):
            val_loop_trunc += 1

    def _mean(xs: Sequence[float]) -> Optional[float]:
        return round(sum(xs) / len(xs), 4) if xs else None

    out: Dict[str, Any] = {
        "proxy_val": {
            "n_pages": len(val_tf1s),
            "token_f1": _mean(val_tf1s),
            "sys_count_exact": round(val_sys_exact / max(1, len(val_tf1s)), 4),
            "loop_truncation_count": val_loop_trunc,
            "loop_truncation_rate": round(val_loop_trunc / max(1, len(val_tf1s)), 4),
            "mean_latency_ms": _mean(val_lat_ms),
        }
    }
    if not eval_suite31:
        return out

    s31_slices: Dict[str, List[float]] = {"pr_clean": [], "pr_photo": [], "hw_clean": [], "hw_photo": [], "overall": []}
    s31_sys_exact = 0
    s31_sys_maes: List[int] = []
    s31_loop_trunc = 0
    s31_mono_viols = 0
    s31_pages = 0
    s31_lat_ms: List[float] = []
    tot_bars = 0
    tot_rewound = 0
    tot_fixed = 0
    s31_splay: List[float] = []

    targeted: Dict[str, Dict[str, Any]] = {
        "ryans-jig": {"note_f1": [], "s_play": [], "sys_exact": []},
        "mozart-k387": {"note_f1": [], "s_play": [], "sys_exact": []},
        "aloha-oe": {"note_f1": [], "s_play": [], "sys_exact": []},
    }

    pieces_eval = s31_pieces[:SUITE31_LIMIT] if SUITE31_LIMIT > 0 else s31_pieces
    for piece in pieces_eval:
        slug = str(piece.get("slug") or piece.get("id") or Path(piece["gt"]).stem).lower()
        gt_path = Path(piece["gt"])
        if not gt_path.is_file():
            continue
        gt_notes, gt_sys = _extract_musicxml_notes_and_sys(gt_path)
        if not gt_notes:
            continue
        is_hw = bool(piece.get("handwritten", False))
        for view in ("clean", "photo"):
            img_p = Path(piece[view])
            if not img_p.is_file():
                continue
            s31_pages += 1
            img_t = _load_rgb_tensor(img_p, FULLPAGE_MAX_DIM, uni_omr_mod).to(device)
            pred_ids, st = _decode_page(
                model, img_t, v, id_to_tok,
                use_fsa=use_fsa, use_kv_rewind=use_kv_rewind,
                max_len=EVAL_MAX_LEN, device=device, use_bf16=use_bf16,
            )
            s31_lat_ms.append(st["latency_ms"])
            tot_bars += st["bars_total"]
            tot_rewound += st["bars_rewound"]
            tot_fixed += st["bars_fixed"]

            pred_raw = [id_to_tok[i] for i in pred_ids if i in id_to_tok]
            if use_bekern3:
                pred_clean, ybins, xbins_all = strip_bekern3_tokens(pred_raw)
                y_viol = any(ybins[k] <= ybins[k - 1] for k in range(1, len(ybins)))
                x_viol = any(any(xs[k] <= xs[k - 1] for k in range(1, len(xs))) for xs in xbins_all)
                if y_viol or x_viol:
                    s31_mono_viols += 1
            else:
                pred_clean = pred_raw

            # For arm1 we intentionally do NOT apply post-hoc n-gram repetition trimming
            pred_sys = 1 + pred_clean.count(uni_omr_mod.SYS_TOKEN) if pred_clean else 0
            nf1 = _multiset_f1(gt_notes, _extract_bekern_notes(pred_clean))
            sp, _, _ = compute_s_play_ppq(pred_raw if use_bekern3 else pred_clean, uni_omr_mod.SYS_TOKEN)
            s31_splay.append(sp)

            s31_slices[f"{'hw' if is_hw else 'pr'}_{view}"].append(nf1)
            s31_slices["overall"].append(nf1)
            s31_sys_maes.append(abs(pred_sys - gt_sys))
            if pred_sys == gt_sys:
                s31_sys_exact += 1
            if st["hit_max_len"] or rlvr_grpo.has_repetition_loop(pred_clean):
                s31_loop_trunc += 1

            for tkey in targeted:
                if tkey in slug:
                    targeted[tkey]["note_f1"].append(nf1)
                    targeted[tkey]["s_play"].append(sp)
                    targeted[tkey]["sys_exact"].append(1.0 if pred_sys == gt_sys else 0.0)

    out["suite31_hw"] = {
        "n_pages": s31_pages,
        "slices": {k: {"n": len(val), "strict_note_f1": _mean(val)} for k, val in s31_slices.items()},
        "sys_count_exact": round(s31_sys_exact / max(1, s31_pages), 4),
        "sys_count_mae": _mean(s31_sys_maes),
        "loop_truncation_count": s31_loop_trunc,
        "loop_truncation_rate": round(s31_loop_trunc / max(1, s31_pages), 4),
        "monotonic_violation_rate": round(s31_mono_viols / max(1, s31_pages), 4),
        "s_play_overall": _mean(s31_splay),
        "bars_total": tot_bars,
        "bars_rewound": tot_rewound,
        "bars_fixed": tot_fixed,
        "pct_bars_rewound": round(100.0 * tot_rewound / max(1, tot_bars), 2),
        "rewind_beat_fix_rate": round(tot_fixed / max(1, tot_rewound), 4) if tot_rewound > 0 else None,
        "mean_latency_ms": _mean(s31_lat_ms),
        "targeted_pieces": {
            k: {
                "n_views": len(d["note_f1"]),
                "note_f1": _mean(d["note_f1"]),
                "s_play": _mean(d["s_play"]),
                "sys_count_exact": _mean(d["sys_exact"]),
            }
            for k, d in targeted.items()
        },
    }
    return out


# ---------------------------------------------------------------------------
# Training Loop (5-Minute Atomic Checkpoints, BF16, EMA 0.99, CE-Masked @y?/@x?)
# ---------------------------------------------------------------------------

def _train_arm(
    arm_name: str,
    use_bekern3: bool,
    train_rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    *,
    device: Any,
    use_bf16: bool,
    uni_omr_mod: Any,
    uni_omr_v2: Any,
) -> Tuple[Path, Dict[int, str], VocabIds, float, Optional[float]]:
    import torch
    import torch.nn.functional as F

    arm_dir = out_dir / arm_name
    ckpt_file = arm_dir / "ckpt_latest.pt"
    train_done = arm_dir / "train_done.json"

    tok_to_id, id_to_tok, vocab_size, v = _build_vocab(uni_omr_mod, use_bekern3)

    if train_done.is_file() and ckpt_file.is_file():
        meta = json.loads(train_done.read_text(encoding="utf-8"))
        print(f"[{EXP_ID}] Resuming trained weights for {arm_name} from {ckpt_file}", flush=True)
        return ckpt_file, id_to_tok, v, float(meta["elapsed_sec"]), meta.get("final_25step_loss")

    arm_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    random.seed(SEED)

    cfg = uni_omr_v2.UniOMRv2Config.small_45m()
    cfg.vocab_size = vocab_size
    cfg.tok_dropout = 0.0
    model = uni_omr_v2.UniOMRv2Net(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    start_step = 0
    ema_state = {k: val.detach().clone() for k, val in model.state_dict().items()}
    if ckpt_file.is_file():
        st = torch.load(ckpt_file, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        ema_state = st["ema"]
        opt.load_state_dict(st["opt"])
        start_step = int(st.get("step", 0))
        print(f"[{EXP_ID}] Resumed {arm_name} at step {start_step}/{STEPS_PER_ARM}", flush=True)

    spatial_cache: Dict[str, Tuple[List[Optional[int]], List[List[Optional[int]]]]] = {}
    t0 = time.perf_counter()
    last_save = time.monotonic()
    losses: List[float] = []
    rng = random.Random(SEED + start_step)

    for step in range(start_step, STEPS_PER_ARM):
        model.train()
        opt.zero_grad(set_to_none=True)
        if step < WARMUP_STEPS:
            lr_now = LR * float(step + 1) / float(max(1, WARMUP_STEPS))
        else:
            prog = float(step - WARMUP_STEPS) / float(max(1, STEPS_PER_ARM - WARMUP_STEPS))
            lr_now = MIN_LR + 0.5 * (LR - MIN_LR) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            rec = rng.choice(train_rows)
            img_p = Path(rec["image"])
            if not img_p.is_file():
                continue
            gt_toks = _read_page_tokens(rec, uni_omr_mod.SYS_TOKEN)
            if not gt_toks:
                continue

            if use_bekern3:
                sys_chunks = " ".join(gt_toks).split(uni_omr_mod.SYS_TOKEN)
                sys_bar_counts = [ch.split().count("bar") for ch in sys_chunks]
                cache_key = f"{img_p}:{','.join(map(str, sys_bar_counts))}"
                if cache_key not in spatial_cache:
                    spatial_cache[cache_key] = detect_2d_spatial_anchors(
                        img_p, sys_bar_counts, is_noisy_hw=_is_noisy_hw_record(rec, img_p)
                    )
                y_bins, x_bins = spatial_cache[cache_key]
                enc_toks = encode_bekern3_spatial(gt_toks, y_bins, x_bins, uni_omr_mod.SYS_TOKEN)
            else:
                enc_toks = gt_toks

            ids = [v.bos_id] + [tok_to_id[t] for t in enc_toks if t in tok_to_id][: MAX_SEQ_LEN - 2] + [v.eos_id]
            tgt_in = torch.tensor([ids[:-1]], device=device, dtype=torch.long)
            out_list = list(ids[1:])
            # Train-only CE mask for `@y?` and `@x?` (never reward predicting unknown cursor)
            if use_bekern3:
                unk_ids = {v.y_unk, v.x_unk}
                out_list = [IGNORE_INDEX if tid in unk_ids else tid for tid in out_list]
            tgt_out = torch.tensor([out_list], device=device, dtype=torch.long)
            img_t = _load_rgb_tensor(img_p, FULLPAGE_MAX_DIM, uni_omr_mod).to(device)

            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (use_bf16 and device.type == "cuda") else contextlib.nullcontext()
            with ctx:
                out = model(img_t, tgt_in, return_aux=False, return_mtp=False)
                logits = out["logits"] if isinstance(out, dict) else (out[0] if isinstance(out, tuple) else out)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), ignore_index=IGNORE_INDEX)

            (loss / float(GRAD_ACCUM)).backward()
            step_loss += float(loss.detach().item()) / float(GRAD_ACCUM)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for k, val in model.state_dict().items():
                if val.is_floating_point():
                    ema_state[k].mul_(GPU_EMA_DECAY).add_(val.detach(), alpha=1.0 - GPU_EMA_DECAY)
                else:
                    ema_state[k].copy_(val.detach())
        losses.append(step_loss)

        if (step + 1) % 100 == 0 or (step + 1) == STEPS_PER_ARM:
            print(f"[{EXP_ID}] {arm_name} step {step + 1}/{STEPS_PER_ARM} lr={lr_now:.2e} loss={step_loss:.4f}", flush=True)

        if (time.monotonic() - last_save) >= SAVE_EVERY_MINUTES * 60.0 or (step + 1) == STEPS_PER_ARM:
            tmp_ckpt = ckpt_file.with_suffix(".pt.tmp")
            torch.save(
                {"step": step + 1, "model": model.state_dict(), "ema": ema_state, "opt": opt.state_dict(), "vocab_size": vocab_size},
                tmp_ckpt,
            )
            os.replace(tmp_ckpt, ckpt_file)
            last_save = time.monotonic()

    elapsed = round(time.perf_counter() - t0, 2)
    fin_loss = round(sum(losses[-25:]) / max(1, len(losses[-25:])), 4) if losses else None
    _atomic_write_json(train_done, {"arm": arm_name, "elapsed_sec": elapsed, "final_25step_loss": fin_loss, "vocab_size": vocab_size})
    del model, opt, ema_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ckpt_file, id_to_tok, v, elapsed, fin_loss


def main() -> None:
    started_at = _utc_now()
    hf_token = _read_hf_token()
    _ensure_deps()
    mvd_dir = _locate_mvd()
    if str(mvd_dir) not in sys.path:
        sys.path.insert(0, str(mvd_dir))

    import torch
    import eval_uni_suite31 as eval_s31
    import rlvr_grpo
    import uni_omr
    import uni_omr_v2

    vol_root = _get_vol_root()
    out_dir = vol_root / "out" / EXP_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8

    train_manifest, val_manifest = _ensure_proxy_dataset(vol_root, hf_token)
    s31_manifest = _ensure_suite31(vol_root, hf_token)
    train_rows = _load_manifest_rows(train_manifest)
    val_rows = _load_manifest_rows(val_manifest)
    _, s31_pieces = eval_s31.resolve_manifest_pieces(s31_manifest)

    results: Dict[str, Any] = {}

    # Arm 0: Baseline bekern v2a
    ckpt0, id2t0, v0, el0, fl0 = _train_arm(
        "arm0_bekern_v2a", False, train_rows, out_dir,
        device=device, use_bf16=use_bf16, uni_omr_mod=uni_omr, uni_omr_v2=uni_omr_v2,
    )
    done0 = out_dir / "arm0_bekern_v2a" / "arm_done.json"
    if done0.is_file():
        results["arm0_bekern_v2a"] = json.loads(done0.read_text(encoding="utf-8"))
    else:
        st0 = torch.load(ckpt0, map_location=device, weights_only=False)
        cfg0 = uni_omr_v2.UniOMRv2Config.small_45m()
        cfg0.vocab_size = int(st0["vocab_size"])
        cfg0.tok_dropout = 0.0
        m0 = uni_omr_v2.UniOMRv2Net(cfg0).to(device)
        m0.load_state_dict(st0["ema"])
        ev0 = _evaluate_arm(
            m0, False, False, False, id2t0, v0, val_rows, s31_pieces,
            device=device, use_bf16=use_bf16, uni_omr_mod=uni_omr, eval_s31=eval_s31, rlvr_grpo=rlvr_grpo,
        )
        m0.load_state_dict(st0["model"])
        ev0_raw = _evaluate_arm(
            m0, False, False, False, id2t0, v0, val_rows, s31_pieces,
            device=device, use_bf16=use_bf16, uni_omr_mod=uni_omr, eval_s31=eval_s31, rlvr_grpo=rlvr_grpo,
            eval_suite31=False,
        )
        results["arm0_bekern_v2a"] = {
            "arm": "arm0_bekern_v2a", "steps": STEPS_PER_ARM,
            "elapsed_sec": el0, "final_25step_loss": fl0,
            "eval": ev0, "raw_proxy_val": ev0_raw["proxy_val"],
        }
        _atomic_write_json(done0, results["arm0_bekern_v2a"])
        del m0, st0
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Arm 1: bekern3 2D spatial cursor + macro-topology (evaluated with FSA-only and FSA + KV-rewind)
    ckpt1, id2t1, v1, el1, fl1 = _train_arm(
        "arm1_bekern3_spatial", True, train_rows, out_dir,
        device=device, use_bf16=use_bf16, uni_omr_mod=uni_omr, uni_omr_v2=uni_omr_v2,
    )
    for eval_arm, use_rewind in (
        ("arm1_bekern3_spatial_fsa", False),
        ("arm1_bekern3_spatial_kv_rewind", True),
    ):
        done_p = out_dir / eval_arm / "arm_done.json"
        if done_p.is_file():
            results[eval_arm] = json.loads(done_p.read_text(encoding="utf-8"))
            continue
        st1 = torch.load(ckpt1, map_location=device, weights_only=False)
        cfg1 = uni_omr_v2.UniOMRv2Config.small_45m()
        cfg1.vocab_size = int(st1["vocab_size"])
        cfg1.tok_dropout = 0.0
        m1 = uni_omr_v2.UniOMRv2Net(cfg1).to(device)
        m1.load_state_dict(st1["ema"])
        ev1 = _evaluate_arm(
            m1, True, True, use_rewind, id2t1, v1, val_rows, s31_pieces,
            device=device, use_bf16=use_bf16, uni_omr_mod=uni_omr, eval_s31=eval_s31, rlvr_grpo=rlvr_grpo,
        )
        raw_pv = None
        if not use_rewind:
            m1.load_state_dict(st1["model"])
            raw_pv = _evaluate_arm(
                m1, True, True, False, id2t1, v1, val_rows, s31_pieces,
                device=device, use_bf16=use_bf16, uni_omr_mod=uni_omr, eval_s31=eval_s31, rlvr_grpo=rlvr_grpo,
                eval_suite31=False,
            )["proxy_val"]
        results[eval_arm] = {
            "arm": eval_arm, "use_fsa": True, "use_kv_rewind": use_rewind,
            "steps": STEPS_PER_ARM, "elapsed_sec": el1, "final_25step_loss": fl1,
            "eval": ev1, "raw_proxy_val": raw_pv,
        }
        _atomic_write_json(done_p, results[eval_arm])
        del m1, st1
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Compute latency overhead of KV-rewind vs FSA-only and vs baseline
    lat_fsa = (results.get("arm1_bekern3_spatial_fsa", {}).get("eval", {}).get("suite31_hw", {}).get("mean_latency_ms") or 0.0)
    lat_rew = (results.get("arm1_bekern3_spatial_kv_rewind", {}).get("eval", {}).get("suite31_hw", {}).get("mean_latency_ms") or 0.0)
    lat_base = (results.get("arm0_bekern_v2a", {}).get("eval", {}).get("suite31_hw", {}).get("mean_latency_ms") or 0.0)
    latency_overhead_vs_fsa_pct = round(100.0 * (lat_rew - lat_fsa) / max(1e-6, lat_fsa), 2) if lat_fsa > 0 else None
    latency_overhead_vs_base_pct = round(100.0 * (lat_rew - lat_base) / max(1e-6, lat_base), 2) if lat_base > 0 else None

    eval_payload = {
        "exp_id": EXP_ID,
        "wave": 13,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "device": str(device),
        "bf16": use_bf16,
        "steps_per_arm": STEPS_PER_ARM,
        "ppq": PPQ,
        "n_y_bins": N_Y_BINS,
        "n_x_bins": N_X_BINS,
        "rewind_beam_width": REWIND_BEAM_WIDTH,
        "latency_overhead_vs_fsa_pct": latency_overhead_vs_fsa_pct,
        "latency_overhead_vs_base_pct": latency_overhead_vs_base_pct,
        "arms": results,
    }
    eval_file = out_dir / f"{EXP_ID}.eval.json"
    run_file = out_dir / f"{EXP_ID}.run.json"
    _atomic_write_json(eval_file, eval_payload)
    _atomic_write_json(run_file, {"exp_id": EXP_ID, "status": "COMPLETED", "completed_at": eval_payload["completed_at"]})
    _upload_json(hf_token, eval_file, HF_EVAL_PATH)
    _upload_json(hf_token, run_file, HF_RUN_PATH)


if __name__ == "__main__":
    main()
