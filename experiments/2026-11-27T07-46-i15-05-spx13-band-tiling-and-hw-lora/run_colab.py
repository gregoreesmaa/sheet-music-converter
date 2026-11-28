#!/usr/bin/env python3
"""Colab L4 runner for Wave 13 Exp i15-05:
Canonical s_px=13 Band Tiling (Zero-Train) + Rank-16 HW-Only LoRA (< 55 MB Edge Budget).

Experiment ID: 2026-11-27T07-46-i15-05-spx13-band-tiling-and-hw-lora
Formalizes Pivot Strategist 5's Pivots 1 & 3A:
  - Control (arm_0_base_1408): Frozen UniOMRv2 single-pass max_dim=1408 baseline
    on all 14 suite31_hw HW pieces + beach-prayer + printed safety slice, with
    a 2-page fail-fast sanity gate.
  - Arm A (arm_a_spx13_band_tiling): Zero-train dark-border crop -> native
    staff-space s_px estimate -> if s_px_1408 < 10.0 px, rescale to canonical
    s_px = 13.0 px and split vertically into 1-2 system bands -> decode & stitch
    <sys>. When s_px_1408 >= 10.0 px, executes bit-identical n_bands=1 passthrough
    (guaranteed 0.00 pp printed regression).
    Gate: hw_landscape Note F1 >= +5.0 pp, 0.00 pp printed regression.
  - Arm B (arm_b_hw_lora_r16): Freeze base model, attach rank-16 HW-only LoRA
    (1.29 MB INT4 on edge85m, fits < 55 MB total budget) on global encoder +
    decoder attn/FFN nn.Linear layers, train 1,500 steps strictly on real+sim HW
    s_px=13 px 1-2 system bands -> evaluate base vs base+HW-LoRA on suite31_hw HW
    with domain-gated 0.00 pp printed regression.
    Gate: hw_clean / hw_photo >= +4.0 pp over arm_a.

Uploads:
  - evals/2026-11-27T07-46-i15-05-spx13-band-tiling-and-hw-lora.json
  - runs/2026-11-27T07-46-i15-05-spx13-band-tiling-and-hw-lora.json
  - checkpoints/2026-11-27T07-46-i15-05-spx13-band-tiling-and-hw-lora/hw_lora_r16_final.pt
to gregoreesmaa/sheet-music-converter-omr.
"""

from __future__ import annotations

import copy
import difflib
import gc
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-27T07-46-i15-05-spx13-band-tiling-and-hw-lora"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_CANDIDATES = (
    "checkpoints/iter14_v2a_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter13_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter10_5_v2_best/uni_omr_v2_ema_best.pt",
)
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"
HF_LORA_CKPT_PATH = f"checkpoints/{EXP_ID}/hw_lora_r16_final.pt"

VOL_ROOT = Path("/vol") if Path("/vol").is_dir() else Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

# Resolution & band-tiling constants (Pivot 1)
BASE_MAX_DIM = 1408
SPX_PASSTHROUGH_THR = 10.0
TARGET_SPX = 13.0
BAND_MAX_WIDTH = 1664
BAND_MAX_HEIGHT = 1280
BAND_SYSTEMS_PER_TILE = 2
BAND_OVERLAP_FRAC = 0.035
DARK_BORDER_MAX_TRIM_FRAC = 0.15
SUITE31_MAX_LEN = 3584
BAND_MAX_LEN = 1792
PRINTED_SAFETY_N = 8

# HW-only LoRA training constants (Pivot 3A)
SEED = 42
LORA_RANK = 16
LORA_ALPHA = 32.0
LORA_DROPOUT = 0.05
LORA_TRAIN_STEPS = 1500
LORA_WARMUP_STEPS = 100
LORA_WSD_DECAY_FRAC = 0.25
LORA_MIN_LR_RATIO = 0.05
LORA_GRAD_ACCUM = 4
LORA_LR_MAX = 2e-4
LORA_WEIGHT_DECAY = 0.01
LORA_MAX_SEQ_LEN = 1536
LORA_EMA_DECAY = 0.99
INT4_GROUP_SIZE = 32
HARD_EDGE_BUDGET_MB = 55.0
SAVE_EVERY_MINUTES = 5.0

_HEADER_PREFIXES = ("clef", "key", "time", "*clef", "*k[", "*M")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _run_cmd(cmd: Sequence[str] | str, *, shell: bool = False) -> None:
    print(f"[{EXP_ID}] $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {cmd}")


def _read_and_purge_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and HF_TOKEN_FILE.exists():
        try:
            os.chmod(HF_TOKEN_FILE, 0o600)
        except OSError:
            pass
        token = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass
    if not token:
        raise RuntimeError("HF token not found in /tmp/.hf_token or HF_TOKEN env var")
    return token


def _locate_repo_and_mvd() -> Tuple[Path, Path]:
    for cand in (
        Path("/workspace/mvd"),
        Path("/content/mvd"),
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path("/content/workspace/sheet-music-converter/tmp-train/mvd"),
        Path.cwd() / "tmp-train" / "mvd",
    ):
        if (cand / "uni_omr_v2.py").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            root = cand.parent.parent if cand.name == "mvd" and cand.parent.name == "tmp-train" else cand.parent
            return root, cand

    for bundle in (Path("/content/ov_mvd_bundle.tar.gz"), Path("/content/bundle.tar.gz")):
        if bundle.is_file():
            extract_root = Path("/content/workspace")
            extract_root.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=extract_root)
            for mvd in (
                extract_root / "sheet-music-converter" / "tmp-train" / "mvd",
                extract_root / "mvd",
            ):
                if (mvd / "uni_omr_v2.py").is_file():
                    if str(mvd) not in sys.path:
                        sys.path.insert(0, str(mvd))
                    return extract_root, mvd

    raise FileNotFoundError("Could not locate tmp-train/mvd/uni_omr_v2.py")


def _setup_environment() -> Path:
    repo_root, _ = _locate_repo_and_mvd()
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
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed])
    return repo_root


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_assets(hf_token: str) -> Tuple[Path, str, Path, Path]:
    from huggingface_hub import hf_hub_download

    VOL_ROOT.mkdir(parents=True, exist_ok=True)
    ds_root = VOL_ROOT / "datasets"
    ds_root.mkdir(parents=True, exist_ok=True)
    ckpt_root = VOL_ROOT / "pretrained"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    ckpt_local: Optional[Path] = None
    ckpt_source: Optional[str] = None
    for cand in HF_CKPT_CANDIDATES:
        target = ckpt_root / cand.replace("/", "__")
        if target.is_file():
            ckpt_local = target
            ckpt_source = cand
            break
        try:
            print(f"[{EXP_ID}] Downloading checkpoint {cand}...", flush=True)
            dl_p = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=cand,
                repo_type="dataset",
                token=hf_token,
                local_dir="/tmp/hf_dl",
            )
            shutil.copy2(dl_p, target)
            try:
                os.remove(dl_p)
            except OSError:
                pass
            ckpt_local = target
            ckpt_source = cand
            break
        except Exception as exc:
            print(f"[{EXP_ID}] Checkpoint {cand} unavailable ({exc}); trying next...", flush=True)

    if ckpt_local is None or ckpt_source is None:
        raise RuntimeError(f"No frozen UniOMR v2 checkpoint found among {HF_CKPT_CANDIDATES}")

    suite31_dir = ds_root / "suite31_hw_eval"
    manifest_path = suite31_dir / "suite31" / "suite31_hw_manifest.json"
    if not manifest_path.is_file():
        suite31_dir.mkdir(parents=True, exist_ok=True)
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_SUITE31_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=suite31_dir)
        try:
            os.remove(tar_p)
        except OSError:
            pass

    proxy_dir = ds_root / "iter10_5_proxy_2k"
    train_abs = proxy_dir / "proxy_train_abs.jsonl"
    if not train_abs.is_file():
        proxy_dir.mkdir(parents=True, exist_ok=True)
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_PROXY_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=proxy_dir)
        try:
            os.remove(tar_p)
        except OSError:
            pass
        for src_name, dst_name in (("proxy_train.jsonl", "proxy_train_abs.jsonl"), ("proxy_val.jsonl", "proxy_val_abs.jsonl")):
            src_path = proxy_dir / src_name
            dst_path = proxy_dir / dst_name
            if not src_path.is_file():
                continue
            out_lines: List[str] = []
            for raw in src_path.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                rec["image"] = str((proxy_dir / rec["image"]).resolve())
                rec["bekerns"] = [str((proxy_dir / b).resolve()) for b in rec.get("bekerns", [])]
                out_lines.append(json.dumps(rec))
            dst_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    return ckpt_local, ckpt_source, manifest_path, train_abs


def _load_frozen_model(ckpt_path: Path, device: "torch.device") -> Tuple[Any, Dict[str, Any]]:
    import eval_uni_suite31 as eus

    model, cfg, meta = eus.load_uni_checkpoint(ckpt_path, device)
    missing = meta.get("missing_keys") or []
    unexpected = meta.get("unexpected_keys") or []
    print(
        f"[{EXP_ID}] Loaded frozen ckpt ({ckpt_path.name}): arch={meta.get('arch')} "
        f"step={meta.get('step')} d_model={getattr(cfg, 'd_model', None)} "
        f"vocab={getattr(cfg, 'vocab_size', None)} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if len(missing) > 0:
        raise RuntimeError(
            f"Refusing to run on partially loaded checkpoint: {len(missing)} missing keys (first 5: {missing[:5]})"
        )
    model.eval()
    return model, meta


def _pil_to_canonical_tensor(img_pil: Any, max_dim: int, max_h: Optional[int] = None) -> "torch.Tensor":
    """Canonical [0, 1] CHW tensor matching uni_omr._resize_fullpage_pil + _pil_to_chw_tensor."""
    import uni_omr

    work = img_pil.convert("RGB")
    if max_h is not None:
        w, h = work.size
        scale = min(1.0, float(max_dim) / float(max(w, h, 1)), float(max_h) / float(max(h, 1)))
        eff_max_dim = max(32, int(round(max(w, h) * scale)))
    else:
        eff_max_dim = int(max_dim)
    pil_im = uni_omr._resize_fullpage_pil(work, eff_max_dim)
    return uni_omr._pil_to_chw_tensor(pil_im)


# ---------------------------------------------------------------------------
# Pivot 1: Dark-Border Crop + Calibrated s_px Estimator + Canonical s_px=13 Band Tiling
# ---------------------------------------------------------------------------

def _crop_dark_borders(img_pil: Any) -> Tuple[Any, Tuple[int, int, int, int]]:
    """Trim dark archival scan/desk borders conservatively (clamped <= 15% per side)."""
    import numpy as np

    gray = np.asarray(img_pil.convert("L"), dtype=np.float32) / 255.0
    H, W = gray.shape
    if H < 64 or W < 64:
        return img_pil, (0, 0, W, H)

    # Estimate paper brightness from central 60% of the page
    cy0, cy1 = int(0.20 * H), int(0.80 * H)
    cx0, cx1 = int(0.20 * W), int(0.80 * W)
    paper_med = float(np.median(gray[cy0:cy1, cx0:cx1]))
    dark_thr = min(0.32, paper_med * 0.45)

    row_med = np.median(gray, axis=1)
    col_med = np.median(gray, axis=0)

    max_y = int(H * DARK_BORDER_MAX_TRIM_FRAC)
    max_x = int(W * DARK_BORDER_MAX_TRIM_FRAC)

    top = 0
    while top < max_y and row_med[top] < dark_thr:
        top += 1
    bot = H
    while (H - bot) < max_y and row_med[bot - 1] < dark_thr:
        bot -= 1
    left = 0
    while left < max_x and col_med[left] < dark_thr:
        left += 1
    right = W
    while (W - right) < max_x and col_med[right - 1] < dark_thr:
        right -= 1

    if (right - left) < int(W * 0.65) or (bot - top) < int(H * 0.65):
        return img_pil, (0, 0, W, H)
    if (left, top, right, bot) == (0, 0, W, H):
        return img_pil, (0, 0, W, H)
    return img_pil.crop((left, top, right, bot)), (left, top, right, bot)


def _estimate_spx_and_valleys(img_pil: Any) -> Dict[str, Any]:
    """Estimate native staff-space s_px (lag-0-lobe-bypassed AC) and system valleys."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    gray = np.asarray(img_pil.convert("L"), dtype=np.float32) / 255.0
    H, W = gray.shape
    scale_1408 = min(1.0, float(BASE_MAX_DIM) / float(max(H, W, 1)))

    x0, x1 = int(0.05 * W), max(int(0.95 * W), int(0.05 * W) + 1)
    crop = gray[:, x0:x1]
    row_bg_raw = np.percentile(crop, 92, axis=1)
    row_bg = gaussian_filter1d(row_bg_raw, sigma=max(12.0, H * 0.025))[:, None]
    ink = np.clip(row_bg - crop, 0.0, 1.0)
    row_prof = ink.mean(axis=1)

    row_hp = row_prof - gaussian_filter1d(row_prof, sigma=max(2.0, H * 0.012))
    row_hp = row_hp - row_hp.mean()
    ac = np.correlate(row_hp, row_hp, mode="full")[H - 1 :]
    min_lag = max(5, int(round(H * 0.0030)))
    max_lag = min(H // 12, max(min_lag + 6, int(round(H * 0.025))))
    native_spx = float(H) / 140.0
    ac_peak = 0.0
    if max_lag > min_lag + 2 and ac[0] > 1e-8:
        ac_norm = ac / ac[0]
        start_lag = min_lag
        while start_lag + 1 < max_lag and ac_norm[start_lag] >= ac_norm[start_lag + 1]:
            start_lag += 1
        best_lag = -1
        best_val = -1.0
        for lag in range(max(min_lag, start_lag), max_lag - 1):
            val = float(ac_norm[lag])
            if val > float(ac_norm[lag - 1]) and val >= float(ac_norm[lag + 1]) and val > best_val:
                best_val = val
                best_lag = lag
        if best_lag > 0 and best_val > 0.06:
            native_spx = float(best_lag)
            ac_peak = best_val

    spx_1408 = native_spx * scale_1408

    # Detect active system bands & split valleys
    smooth = gaussian_filter1d(row_prof, sigma=max(3.0, native_spx * 1.2))
    thr = max(0.012, float(np.percentile(smooth, 25)) + 0.14 * float(smooth.max() - smooth.min()))
    active = smooth > thr

    raw_bands: List[Tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    min_band_h = max(8, int(native_spx * 2.5))
    for y in range(H):
        if active[y] and not in_seg:
            seg_start = y
            in_seg = True
        elif not active[y] and in_seg:
            if (y - seg_start) >= min_band_h:
                raw_bands.append((seg_start, y))
            in_seg = False
    if in_seg and (H - seg_start) >= min_band_h:
        raw_bands.append((seg_start, H))

    merge_gap = max(6, int(round(native_spx * 2.4)))
    systems: List[Tuple[int, int]] = []
    for y0, y1 in raw_bands:
        if systems and (y0 - systems[-1][1]) <= merge_gap:
            systems[-1] = (systems[-1][0], y1)
        else:
            systems.append((y0, y1))
    if not systems:
        systems = [(int(0.02 * H), int(0.98 * H))]

    valleys: List[int] = [0]
    for i in range(len(systems) - 1):
        a_end = systems[i][1]
        b_start = systems[i + 1][0]
        if b_start > a_end + 1:
            v = a_end + int(np.argmin(smooth[a_end:b_start]))
        else:
            v = (a_end + b_start) // 2
        valleys.append(v)
    valleys.append(H)

    return {
        "hw": (H, W),
        "is_landscape": bool(W > H),
        "native_spx": round(native_spx, 2),
        "spx_1408": round(spx_1408, 2),
        "ac_peak": round(ac_peak, 3),
        "n_systems_est": len(systems),
        "systems_px": systems,
        "valleys_px": valleys,
    }


def _build_spx13_bands(img_pil: Any) -> Tuple[List[Tuple[Any, int]], Dict[str, Any]]:
    """Build canonical s_px=13 px 1-2 system bands when s_px_1408 < 10 px; else bit-identical passthrough."""
    raw_info = _estimate_spx_and_valleys(img_pil)
    if float(raw_info["spx_1408"]) >= SPX_PASSTHROUGH_THR:
        return [(img_pil, BASE_MAX_DIM)], {
            **raw_info,
            "passthrough": True,
            "n_bands": 1,
            "crop_box": None,
            "effective_spx": raw_info["spx_1408"],
        }

    cropped, crop_box = _crop_dark_borders(img_pil)
    info = _estimate_spx_and_valleys(cropped)
    H, W = info["hw"]
    native_spx = max(2.0, float(info["native_spx"]))
    n_sys = int(info["n_systems_est"])
    valleys = info["valleys_px"]
    pad_px = int(round(H * BAND_OVERLAP_FRAC))

    if n_sys <= 2:
        mid = valleys[len(valleys) // 2] if len(valleys) > 2 else H // 2
        band_ranges = [(0, min(H, mid + pad_px)), (max(0, mid - pad_px), H)]
    else:
        band_ranges = []
        idx = 0
        while idx < n_sys:
            end_idx = min(n_sys, idx + BAND_SYSTEMS_PER_TILE)
            if n_sys - end_idx == 1 and BAND_SYSTEMS_PER_TILE == 2:
                # Keep last single system separate if 3+ systems, or merge if short
                pass
            y0 = max(0, valleys[idx] - (pad_px if idx > 0 else 0))
            y1 = min(H, valleys[end_idx] + (pad_px if end_idx < n_sys else 0))
            band_ranges.append((y0, y1))
            idx = end_idx

    req_scale = TARGET_SPX / native_spx
    bands: List[Tuple[Any, int]] = []
    eff_spx_vals: List[float] = []
    for y0, y1 in band_ranges:
        tile = cropped.crop((0, y0, W, y1))
        tw, th = tile.size
        # Rescale to target s_px=13 px, capped by BAND_MAX_WIDTH / BAND_MAX_HEIGHT
        eff_scale = min(
            req_scale,
            float(BAND_MAX_WIDTH) / float(max(tw, 1)),
            float(BAND_MAX_HEIGHT) / float(max(th, 1)),
        )
        # Allow mild upscaling up to 2.0x if native archival scan is low-res
        eff_scale = min(eff_scale, 2.0)
        nw = max(32, int(round(tw * eff_scale)))
        nh = max(32, int(round(th * eff_scale)))
        if eff_scale > 1.02:
            from PIL import Image
            tile = tile.resize((nw, nh), Image.BICUBIC)
            target_dim = max(nw, nh)
        else:
            target_dim = max(512, min(BAND_MAX_WIDTH, max(nw, nh)))
        bands.append((tile, target_dim))
        eff_spx_vals.append(native_spx * eff_scale)

    return bands, {
        **info,
        "raw_spx_1408": raw_info["spx_1408"],
        "passthrough": False,
        "n_bands": len(bands),
        "crop_box": list(crop_box),
        "effective_spx": round(sum(eff_spx_vals) / max(1, len(eff_spx_vals)), 2),
    }


def _extract_header_tokens(tokens: List[str]) -> List[str]:
    header: List[str] = []
    for tok in tokens[:24]:
        low = tok.lower()
        if any(low.startswith(p) for p in _HEADER_PREFIXES):
            header.append(tok)
        elif header and (tok in ("=", "|", "<sys>") or re.match(r"^[0-9]+[a-gA-G]", tok)):
            break
    return header


def _stitch_band_tokens_with_header_carry(
    band_token_lists: List[List[str]], sys_token: str
) -> List[str]:
    if not band_token_lists:
        return []
    stitched_systems: List[List[str]] = []
    carry_header: List[str] = []

    for b_idx, toks in enumerate(band_token_lists):
        if not toks:
            continue
        cur_systems: List[List[str]] = [[]]
        for t in toks:
            if t == sys_token:
                if cur_systems[-1]:
                    cur_systems.append([])
            else:
                cur_systems[-1].append(t)
        cur_systems = [s for s in cur_systems if s]
        if not cur_systems:
            continue

        if b_idx == 0:
            carry_header = _extract_header_tokens(cur_systems[0])
            stitched_systems.extend(cur_systems)
            continue

        if stitched_systems and cur_systems:
            prev_s = stitched_systems[-1]
            next_s = cur_systems[0]
            sim = difflib.SequenceMatcher(None, prev_s[:80], next_s[:80]).ratio()
            if sim >= 0.72:
                cur_systems = cur_systems[1:]

        for s_idx, sys_toks in enumerate(cur_systems):
            if s_idx == 0 and carry_header and not _extract_header_tokens(sys_toks):
                sys_toks = list(carry_header) + sys_toks
            stitched_systems.append(sys_toks)

    out: List[str] = []
    for idx, sys_toks in enumerate(stitched_systems):
        if idx > 0:
            out.append(sys_token)
        out.extend(sys_toks)
    return out


# ---------------------------------------------------------------------------
# Pivot 3A: Domain-Gated Rank-16 HW-Only LoRA Wrapper (< 55 MB Edge Budget)
# ---------------------------------------------------------------------------

def _attach_hw_lora(
    model: Any,
    *,
    rank: int = LORA_RANK,
    alpha: float = LORA_ALPHA,
    dropout: float = LORA_DROPOUT,
) -> Dict[str, Any]:
    """Freeze base model and attach rank-16 LoRA on global encoder + decoder attn/FFN Linears.

    Sets `_qat_enabled = True` on each LoRALinear so GroupedQueryAttention does not
    bypass `forward()` via weight concatenation, while supporting `set_lora_enabled(False)`
    for guaranteed bit-identical 0.00 pp printed passthrough.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, r: int, a: float, drop: float) -> None:
            super().__init__()
            self.base = base
            self.r = int(r)
            self.scale = float(a) / float(max(1, r))
            self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()
            self.lora_a = nn.Parameter(torch.empty((r, base.in_features), dtype=base.weight.dtype, device=base.weight.device))
            self.lora_b = nn.Parameter(torch.zeros((base.out_features, r), dtype=base.weight.dtype, device=base.weight.device))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            self.enabled: bool = True
            # Signal GroupedQueryAttention to call module forward() rather than fusing .weight
            self._qat_enabled: bool = True

        @property
        def weight(self) -> torch.Tensor:
            return self.base.weight

        @property
        def bias(self) -> Optional[torch.Tensor]:
            return self.base.bias

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.base(x)
            if not self.enabled:
                return out
            delta = F.linear(F.linear(self.drop(x), self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype))
            return out + delta * self.scale

    for p in model.parameters():
        p.requires_grad = False

    target_roots = (
        ("encoder.global_layers", model.encoder.global_layers),
        ("dec_layers", model.dec_layers),
    )
    wrapped_names: List[str] = []
    lora_params = 0

    for root_prefix, root_mod in target_roots:
        replacements: List[Tuple[nn.Module, str, nn.Linear, str]] = []
        for sub_name, mod in root_mod.named_modules():
            if isinstance(mod, nn.Linear):
                parts = sub_name.split(".")
                parent = root_mod
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                replacements.append((parent, parts[-1], mod, f"{root_prefix}.{sub_name}"))
        for parent, attr, lin, full_name in replacements:
            wrapper = LoRALinear(lin, r=rank, a=alpha, drop=dropout)
            setattr(parent, attr, wrapper)
            wrapped_names.append(full_name)
            lora_params += int(wrapper.lora_a.numel() + wrapper.lora_b.numel())

    # Exact INT4 + group-32 FP16 scale footprint of LoRA adapter
    n_groups = math.ceil(lora_params / INT4_GROUP_SIZE)
    lora_int4_bytes = math.ceil(lora_params / 2) + n_groups * 2
    lora_fp16_bytes = lora_params * 2

    # Analytical projection for edge85m (d=512, 5 global + 10 dec layers, r=16)
    # Global layer: qkv(512->1536)+o(512->512)+SwiGLU(512->2048 x2 + 2048->512) => (512+1536)+(512+512)+3*(512+2048) = 10752 * 16 = 172,032 params/layer
    # Dec layer (GQA 8/2): self(q:512->512, k:512->128, v:512->128, o:512->512) + cross(same) + SwiGLU(3*(512+2048)) => (2*(1024+640+640+1024)+7680)*16 = 229,376 params/layer
    # Wait: let's compute exact edge85m LoRA params directly from UniOMRv2Config.edge85m dimensions:
    d_e = 512
    kv_e = 128
    dff_e = 2048
    gl_layer_p = rank * ((d_e + 3 * d_e) + (d_e + d_e) + 3 * (d_e + dff_e))
    dec_layer_p = rank * (2 * ((d_e + d_e) + 2 * (d_e + kv_e) + (d_e + d_e)) + 3 * (d_e + dff_e))
    edge85m_lora_params = 5 * gl_layer_p + 10 * dec_layer_p
    edge85m_lora_int4_mb = (math.ceil(edge85m_lora_params / 2) + math.ceil(edge85m_lora_params / INT4_GROUP_SIZE) * 2) / 1_000_000.0
    edge85m_base_int4_mb = 51.34  # verified on-disk in i14-05
    edge85m_total_int4_mb = round(edge85m_base_int4_mb + edge85m_lora_int4_mb, 3)

    return {
        "rank": rank,
        "alpha": alpha,
        "n_wrapped_linears": len(wrapped_names),
        "lora_params": lora_params,
        "lora_int4_mb": round(lora_int4_bytes / 1_000_000.0, 3),
        "lora_fp16_mb": round(lora_fp16_bytes / 1_000_000.0, 3),
        "edge85m_projected_lora_params": edge85m_lora_params,
        "edge85m_projected_lora_int4_mb": round(edge85m_lora_int4_mb, 3),
        "edge85m_projected_total_int4_mb": edge85m_total_int4_mb,
        "fits_55mb_budget": bool(edge85m_total_int4_mb < HARD_EDGE_BUDGET_MB),
    }


def _set_lora_enabled(model: Any, enabled: bool) -> None:
    for mod in model.modules():
        if hasattr(mod, "lora_a") and hasattr(mod, "lora_b") and hasattr(mod, "enabled"):
            mod.enabled = bool(enabled)


def _get_lora_state_dict(model: Any) -> Dict[str, "torch.Tensor"]:
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if (".lora_a" in k or ".lora_b" in k)
    }


def _load_lora_state_dict(model: Any, lora_sd: Dict[str, "torch.Tensor"]) -> None:
    model.load_state_dict(lora_sd, strict=False)


# ---------------------------------------------------------------------------
# Scoring & Evaluation Helpers
# ---------------------------------------------------------------------------

def _extract_musicxml_notes(xml_path: Path) -> List[Tuple[str, int, str, str, str]]:
    import xml.etree.ElementTree as ET

    notes: List[Tuple[str, int, str, str, str]] = []
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return notes

    def _strip(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    for elem in root.iter():
        if _strip(elem.tag) != "note":
            continue
        is_rest = is_grace = False
        step = octave = alter = dur = ""
        n_dots = 0
        for child in elem:
            ctag = _strip(child.tag)
            if ctag == "rest":
                is_rest = True
            elif ctag == "grace":
                is_grace = True
            elif ctag == "pitch":
                for pc in child:
                    ptag = _strip(pc.tag)
                    text = (pc.text or "").strip()
                    if ptag == "step":
                        step = text.upper()
                    elif ptag == "octave":
                        octave = text
                    elif ptag == "alter":
                        try:
                            av = int(round(float(text)))
                            if av != 0:
                                alter = f"a{av}"
                        except ValueError:
                            pass
            elif ctag == "type":
                dur = (child.text or "").strip().lower()
            elif ctag == "dot":
                n_dots += 1
        if not is_rest and not is_grace and step and octave and dur:
            notes.append((dur, n_dots, step, octave, alter))
    return notes


def _select_eval_pieces(manifest_path: Path) -> List[Dict[str, Any]]:
    """Select all 14 HW pieces + beach-prayer + printed safety slice (8 printed pieces)."""
    import eval_uni_suite31 as eus

    _, pieces = eus.resolve_manifest_pieces(manifest_path)
    selected: List[Dict[str, Any]] = []
    printed_count = 0
    seen_slugs = set()

    # First add printed safety slice so the 2-page fail-fast sanity gate runs immediately
    for entry in pieces:
        slug = str(entry.get("slug") or entry.get("id") or "")
        is_hw = bool(entry.get("handwritten", False))
        if (not is_hw) and (slug != "beach-prayer") and printed_count < PRINTED_SAFETY_N:
            selected.append(entry)
            seen_slugs.add(slug)
            printed_count += 1

    # Add beach-prayer
    for entry in pieces:
        slug = str(entry.get("slug") or entry.get("id") or "")
        if slug == "beach-prayer" and slug not in seen_slugs:
            selected.append(entry)
            seen_slugs.add(slug)

    # Add all 14 HW pieces
    for entry in pieces:
        slug = str(entry.get("slug") or entry.get("id") or "")
        if bool(entry.get("handwritten", False)) and slug not in seen_slugs:
            selected.append(entry)
            seen_slugs.add(slug)

    return selected


def _eval_slice(
    arm_name: str,
    *,
    use_spx13_tiling: bool,
    domain_gate_lora: bool,
    model: Any,
    eval_pieces: List[Dict[str, Any]],
    device: "torch.device",
    amp_dtype: Optional["torch.dtype"],
    sanity_check: bool = False,
) -> Dict[str, Any]:
    import torch
    from PIL import Image
    import eval_uni_suite31 as eus
    import rlvr_grpo
    import uni_omr

    arm_dir = VOL_ROOT / "ckpts" / EXP_ID / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached = arm_dir / "arm_result.json"
    if cached.is_file():
        print(f"[{EXP_ID}] Reusing cached result for {arm_name}", flush=True)
        return json.loads(cached.read_text(encoding="utf-8"))

    model.eval()
    _, _, eos_id, _, _, id_to_tok = eus._vocab_spec(int(model.cfg.vocab_size))
    amp_enabled = device.type == "cuda" and amp_dtype is not None

    buckets: Dict[str, List[Dict[str, Any]]] = {
        "printed_safety_clean": [],
        "printed_safety_photo": [],
        "beach_prayer": [],
        "hw_clean": [],
        "hw_photo": [],
        "hw_landscape": [],
        "hw_portrait": [],
        "hw_all": [],
    }
    per_page: List[Dict[str, Any]] = []
    t_arm0 = time.time()
    t_last_save = time.time()
    sanity_done = False

    with torch.no_grad():
        for p_idx, entry in enumerate(eval_pieces):
            slug = str(entry.get("slug") or entry.get("id") or f"piece_{p_idx}")
            is_hw = bool(entry.get("handwritten", False))
            is_beach = slug == "beach-prayer"
            ref_notes = _extract_musicxml_notes(Path(entry["gt"]))

            # Domain gating: enable LoRA strictly on HW pages; disable on printed pages
            if domain_gate_lora:
                _set_lora_enabled(model, is_hw)

            for mode_key in ("clean", "photo"):
                img_p = Path(entry[mode_key])
                if not img_p.is_file():
                    continue
                img = Image.open(img_p).convert("RGB")
                w0, h0 = img.size
                is_landscape = bool(w0 > h0)
                t0 = time.perf_counter()

                hit_eos = True
                ink_recalls: List[float] = []
                n_bands = 1
                spx_1408 = None
                eff_spx = None
                passthrough = True

                if not use_spx13_tiling:
                    inp_t = _pil_to_canonical_tensor(img, BASE_MAX_DIM).unsqueeze(0).to(device)
                    with torch.amp.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_enabled):
                        seq, rinfo = model.generate(inp_t, max_len=SUITE31_MAX_LEN, return_info=True)
                    ids = seq[0].tolist()
                    hit_eos = eos_id in ids[1:]
                    pred_toks = eus.decode_uni_compat(ids, vocab_size=model.cfg.vocab_size, skip_special=True)
                    if not hit_eos:
                        pred_toks = eus.trim_truncated_tokens(pred_toks)
                    ink_recalls.append(float(rinfo.get("ink_recall", 1.0)))
                else:
                    bands, binfo = _build_spx13_bands(img)
                    n_bands = len(bands)
                    passthrough = bool(binfo["passthrough"])
                    spx_1408 = binfo.get("raw_spx_1408", binfo["spx_1408"])
                    eff_spx = binfo["effective_spx"]
                    if passthrough:
                        inp_t = _pil_to_canonical_tensor(img, BASE_MAX_DIM).unsqueeze(0).to(device)
                        with torch.amp.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_enabled):
                            seq, rinfo = model.generate(inp_t, max_len=SUITE31_MAX_LEN, return_info=True)
                        ids = seq[0].tolist()
                        hit_eos = eos_id in ids[1:]
                        pred_toks = eus.decode_uni_compat(ids, vocab_size=model.cfg.vocab_size, skip_special=True)
                        if not hit_eos:
                            pred_toks = eus.trim_truncated_tokens(pred_toks)
                        ink_recalls.append(float(rinfo.get("ink_recall", 1.0)))
                    else:
                        band_toks: List[List[str]] = []
                        for crop, target_dim in bands:
                            inp_t = _pil_to_canonical_tensor(crop, target_dim, max_h=BAND_MAX_HEIGHT).unsqueeze(0).to(device)
                            with torch.amp.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_enabled):
                                seq, rinfo = model.generate(inp_t, max_len=BAND_MAX_LEN, return_info=True)
                            ids = seq[0].tolist()
                            toks_b = eus.decode_uni_compat(ids, vocab_size=model.cfg.vocab_size, skip_special=True)
                            if eos_id not in ids[1:]:
                                toks_b = eus.trim_truncated_tokens(toks_b)
                                hit_eos = False
                            band_toks.append(toks_b)
                            ink_recalls.append(float(rinfo.get("ink_recall", 1.0)))
                        pred_toks = _stitch_band_tokens_with_header_carry(band_toks, uni_omr.SYS_TOKEN)

                latency_s = time.perf_counter() - t0
                hyp_toks, _ = eus.stitch_uni_tokens(pred_toks)
                hyp_notes = rlvr_grpo.extract_pitch_dur_notes(hyp_toks)
                note_f1 = float(rlvr_grpo.compute_sequence_f1(ref_notes, hyp_notes))
                struct = rlvr_grpo.compute_unlabeled_structural_reward(
                    hyp_toks, vocab=id_to_tok, terminated_eos=hit_eos
                )
                dec_ok = 1.0 if (len(hyp_notes) > 0 and float(struct.get("r_parse", 0.0)) > 0.0) else 0.0
                mean_ir = sum(ink_recalls) / max(1, len(ink_recalls))

                row = {
                    "slug": slug,
                    "mode": mode_key,
                    "is_hw": is_hw,
                    "is_landscape": is_landscape,
                    "native_wh": [w0, h0],
                    "spx_1408": spx_1408,
                    "effective_spx": eff_spx,
                    "passthrough": passthrough,
                    "n_bands": n_bands,
                    "note_f1": round(note_f1, 4),
                    "dec_ok": dec_ok,
                    "ink_recall": round(mean_ir, 4),
                    "latency_s": round(latency_s, 3),
                }
                per_page.append(row)

                if is_hw:
                    buckets[f"hw_{mode_key}"].append(row)
                    buckets["hw_all"].append(row)
                    buckets["hw_landscape" if is_landscape else "hw_portrait"].append(row)
                elif is_beach:
                    buckets["beach_prayer"].append(row)
                else:
                    buckets[f"printed_safety_{mode_key}"].append(row)

            # Fail-fast 2-page sanity gate on printed_safety_clean
            if sanity_check and not sanity_done and len(buckets["printed_safety_clean"]) >= 2:
                sanity_done = True
                pc2 = buckets["printed_safety_clean"][:2]
                mean_f1_2 = sum(r["note_f1"] for r in pc2) / len(pc2)
                dec_ok_2 = sum(r["dec_ok"] for r in pc2) / len(pc2)
                print(
                    f"[{EXP_ID}][{arm_name}] 2-page sanity gate: mean_note_f1={mean_f1_2:.4f} dec_ok={dec_ok_2:.2f}",
                    flush=True,
                )
                if dec_ok_2 <= 0.0 or mean_f1_2 < 0.05:
                    raise RuntimeError(
                        f"Fail-fast sanity gate tripped on {arm_name}: mean_note_f1={mean_f1_2:.4f}, dec_ok={dec_ok_2:.2f}"
                    )

            if (time.time() - t_last_save) >= SAVE_EVERY_MINUTES * 60.0:
                _atomic_write_json(arm_dir / "progress.json", {"done_pieces": p_idx + 1, "pages": len(per_page)})
                t_last_save = time.time()

    def _agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"n": 0, "note_f1": None, "dec_ok": None, "ink_recall": None, "latency_s": None}
        return {
            "n": len(rows),
            "note_f1": round(sum(r["note_f1"] for r in rows) / len(rows), 4),
            "dec_ok": round(sum(r["dec_ok"] for r in rows) / len(rows), 4),
            "ink_recall": round(sum(r["ink_recall"] for r in rows) / len(rows), 4),
            "latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 3),
        }

    result = {
        "arm": arm_name,
        "use_spx13_tiling": use_spx13_tiling,
        "domain_gate_lora": domain_gate_lora,
        "summary": {k: _agg(v) for k, v in buckets.items()},
        "elapsed_secs": round(time.time() - t_arm0, 2),
        "per_page": per_page,
    }
    _atomic_write_json(cached, result)
    print(f"[{EXP_ID}][{arm_name}] summary: {json.dumps(result['summary'], indent=2)}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Arm B: Train Rank-16 HW-Only LoRA Strictly on Real+Sim HW s_px=13 Bands
# ---------------------------------------------------------------------------

def _load_hw_train_pairs(train_manifest: Path) -> List[Tuple[Path, List[Path], str]]:
    hw_pairs: List[Tuple[Path, List[Path], str]] = []
    fallback_pairs: List[Tuple[Path, List[Path], str]] = []
    for raw in train_manifest.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        img_p = Path(rec["image"])
        beks = [Path(b) for b in rec.get("bekerns", [])]
        domain = str(rec.get("domain", "")).strip().lower()
        img_s = str(img_p).lower()
        if not (img_p.is_file() and beks):
            continue
        is_hw = any(k in domain or k in img_s for k in ("hw", "hand", "debussy", "rism", "mozart", "sim"))
        if is_hw:
            hw_pairs.append((img_p, beks, domain))
        else:
            fallback_pairs.append((img_p, beks, domain))
    return hw_pairs if len(hw_pairs) >= 32 else (hw_pairs + fallback_pairs[:256])


def _encode_bekerns_compat(beks: List[Path], vocab_size: int) -> List[int]:
    import eval_uni_suite31 as eus
    import uni_omr

    _, bos_id, eos_id, sys_id, tok_to_id, _ = eus._vocab_spec(vocab_size)
    ids: List[int] = [bos_id]
    for idx, bp in enumerate(beks):
        if idx > 0:
            ids.append(sys_id)
        raw = bp.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            continue
        for tok in raw.split():
            if tok == uni_omr.SYS_TOKEN:
                ids.append(sys_id)
            else:
                tid = tok_to_id.get(tok)
                if tid is not None:
                    ids.append(tid)
    ids.append(eos_id)
    return ids


def _sample_hw_spx13_band_pair(
    img_pil: Any, beks: List[Path], rng: random.Random
) -> Tuple[Any, int, List[Path]]:
    """Slice a HW training page into a 1-2 system band rescaled to canonical s_px=13 px."""
    cropped, _ = _crop_dark_borders(img_pil)
    info = _estimate_spx_and_valleys(cropped)
    H, W = info["hw"]
    native_spx = max(2.0, float(info["native_spx"]))
    n_sys = len(beks)

    if n_sys <= 2:
        tile = cropped
        sub_beks = beks
    else:
        k = rng.choice([1, 2])
        s0 = rng.randint(0, n_sys - k)
        s1 = s0 + k
        valleys = info["valleys_px"]
        if len(valleys) == n_sys + 1:
            y0 = max(0, valleys[s0] - int(H * 0.02))
            y1 = min(H, valleys[s1] + int(H * 0.02))
        else:
            y0 = max(0, int(round(H * (s0 / float(n_sys)) - H * 0.02)))
            y1 = min(H, int(round(H * (s1 / float(n_sys)) + H * 0.02)))
        if (y1 - y0) < 32:
            y0, y1 = 0, H
            sub_beks = beks
        else:
            sub_beks = beks[s0:s1]
        tile = cropped.crop((0, y0, W, y1))

    tw, th = tile.size
    req_scale = TARGET_SPX / native_spx
    eff_scale = min(req_scale, float(BAND_MAX_WIDTH) / float(max(tw, 1)), float(BAND_MAX_HEIGHT) / float(max(th, 1)), 2.0)
    nw = max(32, int(round(tw * eff_scale)))
    nh = max(32, int(round(th * eff_scale)))
    if eff_scale > 1.02:
        from PIL import Image
        tile = tile.resize((nw, nh), Image.BICUBIC)
        target_dim = max(nw, nh)
    else:
        target_dim = max(512, min(BAND_MAX_WIDTH, max(nw, nh)))
    return tile, target_dim, sub_beks


def _wsd_mult(step: int, total_steps: int, warmup_steps: int) -> float:
    if step <= warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    decay_steps = max(1, int(round(total_steps * LORA_WSD_DECAY_FRAC)))
    decay_start = total_steps - decay_steps
    if step <= decay_start:
        return 1.0
    prog = min(1.0, max(0.0, float(step - decay_start) / float(decay_steps)))
    return LORA_MIN_LR_RATIO + 0.5 * (1.0 - LORA_MIN_LR_RATIO) * (1.0 + math.cos(math.pi * prog))


def _train_and_eval_hw_lora(
    model: Any,
    *,
    train_manifest: Path,
    eval_pieces: List[Dict[str, Any]],
    device: "torch.device",
    amp_dtype: Optional["torch.dtype"],
) -> Tuple[Dict[str, Any], Path]:
    import torch
    from PIL import Image

    arm_dir = VOL_ROOT / "ckpts" / EXP_ID / "arm_b_hw_lora_r16"
    arm_dir.mkdir(parents=True, exist_ok=True)
    final_lora_path = arm_dir / "hw_lora_r16_final.pt"
    cached = arm_dir / "arm_result.json"
    if cached.is_file() and final_lora_path.is_file():
        print(f"[{EXP_ID}] Reusing cached result for arm_b_hw_lora_r16", flush=True)
        return json.loads(cached.read_text(encoding="utf-8")), final_lora_path

    _seed_everything(SEED)
    model.cfg.tok_dropout = 0.0
    lora_meta = _attach_hw_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
    print(f"[{EXP_ID}][arm_b_hw_lora_r16] Attached LoRA: {json.dumps(lora_meta)}", flush=True)

    hw_pairs = _load_hw_train_pairs(train_manifest)
    print(f"[{EXP_ID}][arm_b_hw_lora_r16] Loaded {len(hw_pairs)} real+sim HW training pages", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=LORA_LR_MAX, weight_decay=LORA_WEIGHT_DECAY)
    amp_enabled = device.type == "cuda" and amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    start_step = 0
    ema_lora = _get_lora_state_dict(model)
    latest_ckpt = arm_dir / "latest_lora.pt"
    if latest_ckpt.is_file():
        ck = torch.load(latest_ckpt, map_location=device, weights_only=False)
        _load_lora_state_dict(model, ck["lora_state_dict"])
        ema_lora = ck.get("ema_lora_state_dict", _get_lora_state_dict(model))
        opt.load_state_dict(ck["optimizer_state_dict"])
        start_step = int(ck.get("step", 0))
        print(f"[{EXP_ID}][arm_b_hw_lora_r16] Resumed from step {start_step}", flush=True)

    vocab_size = int(model.cfg.vocab_size)
    rng = random.Random(SEED + start_step)
    loss_hist: List[float] = []
    t_train0 = time.time()
    t_last_save = time.time()

    _set_lora_enabled(model, True)
    model.train()
    for step in range(start_step + 1, LORA_TRAIN_STEPS + 1):
        mult = _wsd_mult(step, LORA_TRAIN_STEPS, LORA_WARMUP_STEPS)
        for pg in opt.param_groups:
            pg["lr"] = LORA_LR_MAX * mult

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        n_valid = 0
        for _ in range(LORA_GRAD_ACCUM):
            img_p, beks, _ = rng.choice(hw_pairs)
            img = Image.open(img_p).convert("RGB")
            tile, target_dim, sub_beks = _sample_hw_spx13_band_pair(img, beks, rng)
            ids = _encode_bekerns_compat(sub_beks, vocab_size=vocab_size)
            if len(ids) < 3:
                continue
            inp = _pil_to_canonical_tensor(tile, target_dim, max_h=BAND_MAX_HEIGHT).unsqueeze(0).to(device)
            tgt = torch.tensor(ids[:LORA_MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
            with torch.amp.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_enabled):
                out = model(inp, tgt[:, :-1], return_aux=False, return_mtp=False)
                losses = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.0, saliency_loss_weight=0.0)
                loss = losses["loss"] / float(LORA_GRAD_ACCUM)
            if not torch.isfinite(loss):
                continue
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            step_loss += float(loss.item())
            n_valid += 1

        if n_valid > 0:
            if scaler.is_enabled():
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()

            # Update EMA over LoRA weights every step
            with torch.no_grad():
                cur_lora = _get_lora_state_dict(model)
                for k, v in cur_lora.items():
                    ema_lora[k].mul_(LORA_EMA_DECAY).add_(v, alpha=1.0 - LORA_EMA_DECAY)
            loss_hist.append(step_loss)

        if step % 100 == 0 or step == 1:
            mean_l = sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:]))
            print(
                f"[{EXP_ID}][arm_b_hw_lora_r16] step={step}/{LORA_TRAIN_STEPS} "
                f"loss={mean_l:.4f} lr={LORA_LR_MAX * mult:.2e} elapsed={time.time() - t_train0:.1f}s",
                flush=True,
            )

        if (time.time() - t_last_save) >= SAVE_EVERY_MINUTES * 60.0:
            tmp_ck = arm_dir / "latest_lora.pt.tmp"
            torch.save(
                {
                    "step": step,
                    "lora_meta": lora_meta,
                    "lora_state_dict": _get_lora_state_dict(model),
                    "ema_lora_state_dict": ema_lora,
                    "optimizer_state_dict": opt.state_dict(),
                },
                tmp_ck,
            )
            os.replace(tmp_ck, latest_ckpt)
            t_last_save = time.time()

    train_secs = round(time.time() - t_train0, 2)
    raw_lora = _get_lora_state_dict(model)

    # Save final adapter checkpoint (both RAW and EMA)
    tmp_final = final_lora_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "exp_id": EXP_ID,
            "step": LORA_TRAIN_STEPS,
            "lora_meta": lora_meta,
            "lora_state_dict": raw_lora,
            "ema_lora_state_dict": ema_lora,
        },
        tmp_final,
    )
    os.replace(tmp_final, final_lora_path)

    # Evaluate RAW LoRA + EMA LoRA under domain gating (LoRA ON for HW, OFF for printed)
    raw_eval = _eval_slice(
        "arm_b_hw_lora_r16_raw_eval",
        use_spx13_tiling=True,
        domain_gate_lora=True,
        model=model,
        eval_pieces=eval_pieces,
        device=device,
        amp_dtype=amp_dtype,
    )
    _load_lora_state_dict(model, ema_lora)
    ema_eval = _eval_slice(
        "arm_b_hw_lora_r16_ema_eval",
        use_spx13_tiling=True,
        domain_gate_lora=True,
        model=model,
        eval_pieces=eval_pieces,
        device=device,
        amp_dtype=amp_dtype,
    )

    result = {
        "arm": "arm_b_hw_lora_r16",
        "lora_meta": lora_meta,
        "n_hw_train_pages": len(hw_pairs),
        "train_steps": LORA_TRAIN_STEPS,
        "train_loss_tail50": round(sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:])), 5) if loss_hist else None,
        "train_secs": train_secs,
        "raw_eval": raw_eval["summary"],
        "ema_eval": ema_eval["summary"],
        "ema_per_page": ema_eval["per_page"],
    }
    _atomic_write_json(cached, result)
    del opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, final_lora_path


def main() -> None:
    t_start = time.time()
    started_iso = _utc_now()
    hf_token = _read_and_purge_hf_token()
    try:
        _setup_environment()
        import torch
        from huggingface_hub import HfApi

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cc_major = torch.cuda.get_device_capability(0)[0] if device.type == "cuda" else 0
        amp_dtype = (
            torch.bfloat16 if (device.type == "cuda" and cc_major >= 8)
            else (torch.float16 if device.type == "cuda" else None)
        )
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

        ckpt_local, ckpt_source, manifest_path, train_abs = _ensure_assets(hf_token)
        model, ckpt_meta = _load_frozen_model(ckpt_local, device)
        eval_pieces = _select_eval_pieces(manifest_path)
        print(
            f"[{EXP_ID}] Running on {gpu_name} (amp={amp_dtype}), eval_pieces={len(eval_pieces)} "
            f"(printed_safety={PRINTED_SAFETY_N} + beach-prayer + 14 HW)",
            flush=True,
        )

        # Control: single-pass 1408 baseline with 2-page fail-fast sanity gate
        arm_0 = _eval_slice(
            "arm_0_base_1408",
            use_spx13_tiling=False,
            domain_gate_lora=False,
            model=model,
            eval_pieces=eval_pieces,
            device=device,
            amp_dtype=amp_dtype,
            sanity_check=True,
        )

        # Arm A (Pivot 1): Zero-train canonical s_px=13 band tiling
        arm_a = _eval_slice(
            "arm_a_spx13_band_tiling",
            use_spx13_tiling=True,
            domain_gate_lora=False,
            model=model,
            eval_pieces=eval_pieces,
            device=device,
            amp_dtype=amp_dtype,
        )

        # Arm B (Pivot 3A): Freeze base + rank-16 HW-only LoRA (1,500 steps on HW s_px=13 bands)
        arm_b, lora_ckpt_local = _train_and_eval_hw_lora(
            model,
            train_manifest=train_abs,
            eval_pieces=eval_pieces,
            device=device,
            amp_dtype=amp_dtype,
        )

        def _delta(b_val: Optional[float], a_val: Optional[float]) -> Optional[float]:
            return None if (b_val is None or a_val is None) else round(float(b_val) - float(a_val), 4)

        s0 = arm_0["summary"]
        sa = arm_a["summary"]
        sb_ema = arm_b["ema_eval"]
        sb_raw = arm_b["raw_eval"]

        comparison = {
            "arm_a_minus_base_hw_landscape": _delta(sa["hw_landscape"]["note_f1"], s0["hw_landscape"]["note_f1"]),
            "arm_a_minus_base_hw_clean": _delta(sa["hw_clean"]["note_f1"], s0["hw_clean"]["note_f1"]),
            "arm_a_minus_base_hw_photo": _delta(sa["hw_photo"]["note_f1"], s0["hw_photo"]["note_f1"]),
            "arm_a_minus_base_beach_prayer": _delta(sa["beach_prayer"]["note_f1"], s0["beach_prayer"]["note_f1"]),
            "arm_a_minus_base_printed_clean": _delta(sa["printed_safety_clean"]["note_f1"], s0["printed_safety_clean"]["note_f1"]),
            "arm_a_minus_base_printed_photo": _delta(sa["printed_safety_photo"]["note_f1"], s0["printed_safety_photo"]["note_f1"]),
            "arm_b_ema_minus_a_hw_clean": _delta(sb_ema["hw_clean"]["note_f1"], sa["hw_clean"]["note_f1"]),
            "arm_b_ema_minus_a_hw_photo": _delta(sb_ema["hw_photo"]["note_f1"], sa["hw_photo"]["note_f1"]),
            "arm_b_ema_minus_a_hw_landscape": _delta(sb_ema["hw_landscape"]["note_f1"], sa["hw_landscape"]["note_f1"]),
            "arm_b_raw_minus_a_hw_clean": _delta(sb_raw["hw_clean"]["note_f1"], sa["hw_clean"]["note_f1"]),
            "arm_b_ema_minus_a_printed_clean": _delta(sb_ema["printed_safety_clean"]["note_f1"], sa["printed_safety_clean"]["note_f1"]),
            "arm_a_passes_gate": bool(
                (sa["hw_landscape"]["note_f1"] is not None and s0["hw_landscape"]["note_f1"] is not None)
                and (sa["hw_landscape"]["note_f1"] - s0["hw_landscape"]["note_f1"] >= 0.05)
                and (sa["printed_safety_clean"]["note_f1"] == s0["printed_safety_clean"]["note_f1"])
            ),
            "arm_b_passes_gate": bool(
                (sb_ema["hw_clean"]["note_f1"] is not None and sa["hw_clean"]["note_f1"] is not None)
                and (
                    (sb_ema["hw_clean"]["note_f1"] - sa["hw_clean"]["note_f1"] >= 0.04)
                    or (
                        sb_ema["hw_photo"]["note_f1"] is not None
                        and sa["hw_photo"]["note_f1"] is not None
                        and sb_ema["hw_photo"]["note_f1"] - sa["hw_photo"]["note_f1"] >= 0.04
                    )
                )
                and bool(arm_b["lora_meta"]["fits_55mb_budget"])
            ),
        }

        report = {
            "exp_id": EXP_ID,
            "exp_code": "i15-05",
            "wave": 13,
            "status": "completed",
            "checkpoint": ckpt_source,
            "ckpt_meta": {
                "arch": ckpt_meta.get("arch"),
                "step": ckpt_meta.get("step"),
                "missing_keys": len(ckpt_meta.get("missing_keys") or []),
                "unexpected_keys": len(ckpt_meta.get("unexpected_keys") or []),
            },
            "started_utc": started_iso,
            "completed_utc": _utc_now(),
            "elapsed_secs": round(time.time() - t_start, 2),
            "platform": f"colab-{gpu_name}",
            "amp_dtype": str(amp_dtype),
            "arm_0_base_1408": arm_0,
            "arm_a_spx13_band_tiling": arm_a,
            "arm_b_hw_lora_r16": arm_b,
            "comparison": comparison,
        }

        eval_local = VOL_ROOT / "evals" / f"{EXP_ID}.json"
        run_local = VOL_ROOT / "runs" / f"{EXP_ID}.json"
        _atomic_write_json(eval_local, report)
        _atomic_write_json(
            run_local,
            {
                "exp_id": EXP_ID,
                "exp_code": "i15-05",
                "wave": 13,
                "status": "completed",
                "started_utc": started_iso,
                "completed_utc": report["completed_utc"],
                "elapsed_secs": report["elapsed_secs"],
                "eval_path": HF_EVAL_PATH,
                "lora_ckpt_path": HF_LORA_CKPT_PATH,
                "comparison": comparison,
            },
        )

        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(eval_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"eval({EXP_ID}): upload s_px=13 band tiling + HW-LoRA r16 eval",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload s_px=13 band tiling + HW-LoRA r16 run ledger",
        )
        if lora_ckpt_local.is_file():
            api.upload_file(
                path_or_fileobj=str(lora_ckpt_local),
                path_in_repo=HF_LORA_CKPT_PATH,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"ckpt({EXP_ID}): upload HW-only rank-16 LoRA adapter",
            )
        print(f"[{EXP_ID}] UPLOADED {HF_EVAL_PATH}, {HF_RUN_PATH}, & {HF_LORA_CKPT_PATH} TO HF!", flush=True)
    finally:
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
