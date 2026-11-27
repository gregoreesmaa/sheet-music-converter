#!/usr/bin/env python3
"""Google Colab L4 runner for Overnight Exp 28: Curved-Staff Droop, Elastic Warp & Broken-Staff-Line Augmentation Sweep.

Experiment ID: 2026-11-26T01-38-ov-28-droop-warp-broken-staff-aug
Controlled 3-Arm Sweep on UniOMR v2 small_45m (BF16, seed=42, lr=3e-4, 600 steps/arm, dim=1280):
  - Arm A (none):   Base online aug (aug_prob=0.55), no droop/elastic-warp/broken-staff aug.
  - Arm B (mild):   Curved-staff droop [0.8-1.8%] + 2D elastic warp [2-5px] + broken-staff gaps (8-20).
  - Arm C (strong): Curved-staff droop [1.8-3.5%] + 2D elastic warp [5-11px] + broken-staff gaps (20-48).

Evaluates all 3 arms on:
  1. proxy_val.jsonl (64 pages): clean val loss, CE, MTP, Token F1, Pitch F1, Rhythm F1, SER
  2. Warped+Broken-Staff stress test on proxy_val (droop_frac=0.025, elastic=6px, 24 broken-staff gaps)
  3. suite31_hw (45 pieces: 31 printed + 14 real HW, clean + photo = 90 pages):
     Strict Note F1, Token F1, Pitch-only Token F1, Rhythm-only Token F1.

Uploads results to Hugging Face Private Dataset Hub (gregoreesmaa/sheet-music-converter-omr):
  - evals/2026-11-26T01-38-ov-28-droop-warp-broken-staff-aug.json
  - runs/2026-11-26T01-38-ov-28-droop-warp-broken-staff-aug.json
"""

from __future__ import annotations

import contextlib
import difflib
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T01-38-ov-28-droop-warp-broken-staff-aug"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

WORKSPACE_ROOT = Path("/workspace")
VOL_ROOT = Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

# Matched hyperparameters across all 3 arms (~32-38 min total on Colab L4)
TRAIN_STEPS = 600
SEED = 42
LR = 3e-4
WEIGHT_DECAY = 0.04
BATCH_SIZE = 2
GRAD_ACCUM = 4
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
WARMUP_STEPS = 60
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
BASE_AUG_PROB = 0.55
VAL_EVAL_MAX_LEN = 512
SUITE31_MAX_LEN = 768
SAVE_EVERY_MINUTES = 5.0

STRENGTH_PRESETS: Dict[str, Dict[str, Any]] = {
    "none": {
        "droop_prob": 0.0,
        "droop_range": (0.0, 0.0),
        "elastic_prob": 0.0,
        "elastic_amp_px": (0.0, 0.0),
        "broken_prob": 0.0,
        "broken_count": (0, 0),
        "broken_alpha": (0.0, 0.0),
    },
    "mild": {
        "droop_prob": 0.50,
        "droop_range": (0.008, 0.018),
        "elastic_prob": 0.50,
        "elastic_amp_px": (2.0, 5.0),
        "broken_prob": 0.50,
        "broken_count": (8, 20),
        "broken_alpha": (0.45, 0.75),
    },
    "strong": {
        "droop_prob": 0.75,
        "droop_range": (0.018, 0.035),
        "elastic_prob": 0.75,
        "elastic_amp_px": (5.0, 11.0),
        "broken_prob": 0.75,
        "broken_count": (20, 48),
        "broken_alpha": (0.65, 0.92),
    },
}


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _run_cmd(cmd: Sequence[str] | str, *, shell: bool = False, env: Optional[Dict[str, str]] = None) -> None:
    print(f"[{EXP_ID}] $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        shell=shell,
        env=env,
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


def _find_or_extract_repo() -> Path:
    for cand in (
        WORKSPACE_ROOT / "sheet-music-converter",
        Path("/content/sheet-music-converter"),
        Path.cwd(),
    ):
        if (cand / "tmp-train" / "mvd" / "uni_omr_v2.py").is_file():
            return cand

    bundle_candidates = [
        Path(os.environ.get("REPO_BUNDLE_PATH", "")),
        Path("/content/repo_bundle.tar.gz"),
        Path("/content/ov_bundle.tar.gz"),
    ]
    for bp in bundle_candidates:
        if str(bp) and bp.is_file():
            WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
            print(f"[{EXP_ID}] Extracting {bp} -> {WORKSPACE_ROOT}...", flush=True)
            with tarfile.open(bp, "r:*") as tf:
                tf.extractall(path=WORKSPACE_ROOT)
            if (WORKSPACE_ROOT / "sheet-music-converter" / "tmp-train" / "mvd" / "uni_omr_v2.py").is_file():
                return WORKSPACE_ROOT / "sheet-music-converter"
            if (WORKSPACE_ROOT / "tmp-train" / "mvd" / "uni_omr_v2.py").is_file():
                return WORKSPACE_ROOT

    raise FileNotFoundError(
        "Repo not found under /workspace/sheet-music-converter or /content/sheet-music-converter, "
        "and no /content/repo_bundle.tar.gz was uploaded."
    )


def _setup_environment() -> Tuple[Path, Optional[Path], Optional[Path]]:
    repo_root = _find_or_extract_repo()

    needed_pkgs = []
    for mod, pkg in [
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            needed_pkgs.append(pkg)
    if needed_pkgs:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed_pkgs])

    aser_manifest = repo_root / "converter" / "assembly-serialize" / "Cargo.toml"
    eharness_manifest = repo_root / "converter" / "eval-harness" / "Cargo.toml"
    aser_bin = aser_manifest.parent / "target" / "release" / "assembly-serialize"
    eharness_bin = eharness_manifest.parent / "target" / "release" / "eval-harness"

    if aser_manifest.exists() and eharness_manifest.exists():
        cargo_bin_dir = Path.home() / ".cargo" / "bin"
        env = os.environ.copy()
        env["PATH"] = f"{cargo_bin_dir}:{env.get('PATH', '')}"
        if shutil.which("cargo", path=env["PATH"]) is None:
            print(f"[{EXP_ID}] Installing minimal Rust toolchain via rustup...", flush=True)
            _run_cmd(
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
                "| sh -s -- -y --default-toolchain stable --profile minimal",
                shell=True,
                env=env,
            )
        if not aser_bin.exists():
            _run_cmd(["cargo", "build", "--release", "--manifest-path", str(aser_manifest)], env=env)
        if not eharness_bin.exists():
            _run_cmd(["cargo", "build", "--release", "--manifest-path", str(eharness_manifest)], env=env)
        if aser_bin.exists() and eharness_bin.exists():
            return repo_root, aser_bin, eharness_bin
    return repo_root, None, None


def _ensure_datasets(hf_token: str) -> Tuple[Path, Path, Path]:
    from huggingface_hub import hf_hub_download

    VOL_ROOT.mkdir(parents=True, exist_ok=True)
    ds_root = VOL_ROOT / "datasets"
    ds_root.mkdir(parents=True, exist_ok=True)

    proxy_dir = ds_root / "iter10_5_proxy_2k"
    train_abs = proxy_dir / "proxy_train_abs.jsonl"
    val_abs = proxy_dir / "proxy_val_abs.jsonl"
    if not (train_abs.is_file() and val_abs.is_file()):
        proxy_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR}...", flush=True)
        t0 = time.perf_counter()
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_PROXY_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        print(f"[{EXP_ID}] Downloaded proxy tar in {time.perf_counter() - t0:.1f}s -> extracting...", flush=True)
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=proxy_dir)
        try:
            os.remove(tar_p)
        except OSError:
            pass
        for src_name, dst_path in (("proxy_train.jsonl", train_abs), ("proxy_val.jsonl", val_abs)):
            src_path = proxy_dir / src_name
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
            print(f"[{EXP_ID}] Wrote {len(out_lines)} rows -> {dst_path.name}", flush=True)

    suite31_dir = ds_root / "suite31_hw_eval"
    manifest_path = suite31_dir / "suite31" / "suite31_hw_manifest.json"
    if not manifest_path.is_file():
        suite31_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{EXP_ID}] Downloading {HF_SUITE31_TAR}...", flush=True)
        t0 = time.perf_counter()
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_SUITE31_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        print(f"[{EXP_ID}] Downloaded suite31 tar in {time.perf_counter() - t0:.1f}s -> extracting...", flush=True)
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=suite31_dir)
        try:
            os.remove(tar_p)
        except OSError:
            pass

    return train_abs, val_abs, manifest_path


def _load_manifest_pairs(manifest_path: Path) -> List[Tuple[Path, List[Path], str]]:
    pairs: List[Tuple[Path, List[Path], str]] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        img_p = Path(rec["image"])
        beks = [Path(b) for b in rec.get("bekerns", [])]
        domain = str(rec.get("domain", "")).strip().lower()
        if img_p.is_file() and beks:
            pairs.append((img_p, beks, domain))
    return pairs


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _wsd_lr(step: int, total_steps: int, base_lr: float) -> float:
    if step <= WARMUP_STEPS:
        return base_lr * float(step) / float(max(1, WARMUP_STEPS))
    decay_steps = max(1, int(round(total_steps * WSD_DECAY_FRAC)))
    decay_start = total_steps - decay_steps
    if step <= decay_start:
        return base_lr
    prog = min(1.0, max(0.0, float(step - decay_start) / float(decay_steps)))
    min_lr = base_lr * MIN_LR_RATIO
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * prog))


def apply_droop_warp_broken_staff(
    img_t: "torch.Tensor",
    preset: Dict[str, Any],
    rng: random.Random,
) -> "torch.Tensor":
    """Apply curved-staff droop + 2D elastic warp + broken-staff-line gaps to a CHW float/uint8 tensor."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    was_uint8 = img_t.dtype == torch.uint8
    x = img_t.float() / 255.0 if was_uint8 else img_t.float().clone()
    _, h, w = x.shape

    do_droop = rng.random() < float(preset["droop_prob"])
    do_elastic = rng.random() < float(preset["elastic_prob"])
    if do_droop or do_elastic:
        ys = torch.linspace(-1.0, 1.0, h, dtype=torch.float32)
        xs = torch.linspace(-1.0, 1.0, w, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        dx = torch.zeros_like(grid_x)
        dy = torch.zeros_like(grid_y)

        if do_droop:
            lo, hi = preset["droop_range"]
            droop_frac = rng.uniform(float(lo), float(hi))
            sign = rng.choice((-1.0, 1.0))
            phase = rng.uniform(0.0, math.pi)
            freq = rng.uniform(0.75, 1.5)
            # Combine parabolic sag + gentle sinusoidal staff curvature
            x_norm = 0.5 * (xs + 1.0)
            curve = 0.6 * ((x_norm - 0.5) ** 2) * 4.0 + 0.4 * torch.sin(math.pi * freq * x_norm + phase)
            curve = curve - curve.mean()
            dy = dy + (sign * 2.0 * droop_frac) * curve.unsqueeze(0)

        if do_elastic:
            lo_px, hi_px = preset["elastic_amp_px"]
            amp_px = rng.uniform(float(lo_px), float(hi_px))
            np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
            coarse = torch.from_numpy(np_rng.uniform(-1.0, 1.0, size=(1, 2, 8, 8)).astype(np.float32))
            disp = F.interpolate(coarse, size=(h, w), mode="bicubic", align_corners=True)[0]
            dx = dx + disp[0] * (2.0 * amp_px / float(max(1, w)))
            dy = dy + disp[1] * (2.0 * amp_px / float(max(1, h)))

        grid = torch.stack((grid_x + dx, grid_y + dy), dim=-1).unsqueeze(0).clamp(-1.0, 1.0)
        x = F.grid_sample(
            x.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0]

    if rng.random() < float(preset["broken_prob"]):
        n_lo, n_hi = preset["broken_count"]
        a_lo, a_hi = preset["broken_alpha"]
        n_breaks = rng.randint(int(n_lo), int(n_hi))
        # Estimate local paper color from upper-quantile pixels (subsampled for CUDA quantile safety)
        flat_x = x.reshape(3, -1)
        if flat_x.shape[-1] > 32768:
            flat_x = flat_x[:, :: max(1, flat_x.shape[-1] // 32768)]
        paper_col = torch.quantile(flat_x, 0.90, dim=1).view(3, 1, 1)
        luma = x.mean(dim=0)
        dark_ys = torch.nonzero(luma.mean(dim=1) < 0.96, as_tuple=False).flatten()
        for _ in range(n_breaks):
            if dark_ys.numel() > 0 and rng.random() < 0.85:
                cy = int(dark_ys[rng.randrange(int(dark_ys.numel()))].item())
            else:
                cy = rng.randint(max(1, int(0.05 * h)), max(2, int(0.95 * h) - 1))
            gap_h = rng.randint(1, 2)
            gap_w = rng.randint(6, max(8, min(28, w // 24)))
            cx = rng.randint(0, max(0, w - gap_w))
            y0 = max(0, cy)
            y1 = min(h, cy + gap_h)
            x0 = cx
            x1 = min(w, cx + gap_w)
            alpha = rng.uniform(float(a_lo), float(a_hi))
            x[:, y0:y1, x0:x1] = x[:, y0:y1, x0:x1] * (1.0 - alpha) + paper_col * alpha

    x = x.clamp(0.0, 1.0)
    if was_uint8:
        return (x * 255.0).round().to(torch.uint8)
    return x


_PITCH_STEPS = {"A", "B", "C", "D", "E", "F", "G"}
_PITCH_OCTAVES = {str(n) for n in range(0, 9)}
_PITCH_ALTERS = {"a-2", "a-1", "a1", "a2"}
_PITCH_TOKENS = _PITCH_STEPS | _PITCH_OCTAVES | _PITCH_ALTERS
_RHYTHM_TOKENS = {
    "whole", "half", "quarter", "eighth", "16th", "32nd", "64th", "breve",
    "dot", "rest", "bar-rest", "tie-start", "tie-stop", "tuplet-stop",
}


def _is_pitch_tok(t: str) -> bool:
    return t in _PITCH_TOKENS


def _is_rhythm_tok(t: str) -> bool:
    return (t in _RHYTHM_TOKENS) or t.startswith("tuplet-start-")


def _seq_f1(pred: Sequence[str], ref: Sequence[str]) -> float:
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    sm = difflib.SequenceMatcher(None, ref, pred, autojunk=False)
    matches = sum(b.size for b in sm.get_matching_blocks())
    return (2.0 * matches) / float(len(pred) + len(ref))


def _eval_val_and_stress(
    model: Any,
    val_pairs: List[Tuple[Path, List[Path], str]],
    device: "torch.device",
    *,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Evaluate on clean proxy_val AND deterministic warped+broken-staff stress test."""
    import torch
    from torch.utils.data import DataLoader
    import train_uni_v2 as tv2
    from uni_omr import (
        UniFullPageDataset,
        _images_to_device_float,
        collate_uni_batch,
        decode_uni,
    )

    stress_preset = {
        "droop_prob": 1.0,
        "droop_range": (0.024, 0.026),
        "elastic_prob": 1.0,
        "elastic_amp_px": (5.5, 6.5),
        "broken_prob": 1.0,
        "broken_count": (24, 24),
        "broken_alpha": (0.75, 0.85),
    }

    model.eval()
    val_ds = UniFullPageDataset(
        val_pairs,
        fullpage_max_dim=FULLPAGE_MAX_DIM,
        max_seq_len=MAX_SEQ_LEN,
        augment=False,
        return_uint8=True,
    )
    loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_uni_batch,
        pin_memory=True,
    )

    def _run_split(apply_stress: bool) -> Dict[str, Any]:
        sum_loss = 0.0
        sum_ce = 0.0
        sum_mtp = 0.0
        n_samples = 0
        f1_list: List[float] = []
        pitch_f1_list: List[float] = []
        rhythm_f1_list: List[float] = []
        tot_edits = 0
        tot_ref_len = 0
        sys_exact = 0
        stress_rng = random.Random(2028)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if (use_bf16 and device.type == "cuda")
            else contextlib.nullcontext()
        )
        with torch.no_grad():
            for batch in loader:
                bs = int(batch["seq"].shape[0])
                imgs_cpu = batch["images"]
                if apply_stress:
                    warped_list = [
                        apply_droop_warp_broken_staff(imgs_cpu[b], stress_preset, stress_rng)
                        for b in range(bs)
                    ]
                    imgs_cpu = torch.stack(warped_list, dim=0)
                images = _images_to_device_float(imgs_cpu, device, non_blocking=True)
                seq = batch["seq"].to(device, non_blocking=True)

                losses, _ = tv2.forward_and_loss_v2(
                    model,
                    images,
                    seq,
                    kind="fullpage",
                    mtp_loss_weight=0.30,
                    saliency_loss_weight=0.10,
                    sys_y_loss_weight=0.25,
                    sys_count_loss_weight=0.15,
                    staff_seg_loss_weight=0.20,
                    stroke_hog_weight=0.15,
                    label_smoothing=0.05,
                    use_bf16=use_bf16,
                    device=device,
                )
                sum_loss += float(losses["loss"].item()) * bs
                sum_ce += float(losses["ce_loss"].item()) * bs
                sum_mtp += float(losses["mtp_loss"].item()) * bs
                n_samples += bs

                with amp_ctx:
                    gen_ids = model.generate(images, max_len=VAL_EVAL_MAX_LEN)

                for b in range(bs):
                    ref_toks = decode_uni(seq[b].tolist(), skip_special=True)
                    pred_toks = decode_uni(gen_ids[b].tolist(), skip_special=True)
                    f1_list.append(_seq_f1(pred_toks, ref_toks))
                    pitch_f1_list.append(
                        _seq_f1([t for t in pred_toks if _is_pitch_tok(t)], [t for t in ref_toks if _is_pitch_tok(t)])
                    )
                    rhythm_f1_list.append(
                        _seq_f1([t for t in pred_toks if _is_rhythm_tok(t)], [t for t in ref_toks if _is_rhythm_tok(t)])
                    )
                    if ref_toks.count("<sys>") == pred_toks.count("<sys>"):
                        sys_exact += 1
                    sm = difflib.SequenceMatcher(None, ref_toks, pred_toks, autojunk=False)
                    matches = sum(blk.size for blk in sm.get_matching_blocks())
                    tot_edits += max(len(ref_toks), len(pred_toks)) - matches
                    tot_ref_len += max(1, len(ref_toks))

        return {
            "n_samples": n_samples,
            "val_loss": round(sum_loss / max(1, n_samples), 6),
            "val_ce_loss": round(sum_ce / max(1, n_samples), 6),
            "val_mtp_loss": round(sum_mtp / max(1, n_samples), 6),
            "val_token_f1": round(sum(f1_list) / max(1, len(f1_list)), 6),
            "val_pitch_token_f1": round(sum(pitch_f1_list) / max(1, len(pitch_f1_list)), 6),
            "val_rhythm_token_f1": round(sum(rhythm_f1_list) / max(1, len(rhythm_f1_list)), 6),
            "val_ser": round(tot_edits / max(1, tot_ref_len), 6),
            "val_sys_count_exact": round(sys_exact / max(1, n_samples), 4),
        }

    clean_m = _run_split(apply_stress=False)
    stress_m = _run_split(apply_stress=True)
    return {
        "clean": clean_m,
        "warped_broken_stress": stress_m,
        "stress_ce_delta": round(stress_m["val_ce_loss"] - clean_m["val_ce_loss"], 6),
        "stress_token_f1_delta": round(stress_m["val_token_f1"] - clean_m["val_token_f1"], 6),
        "stress_pitch_f1_delta": round(stress_m["val_pitch_token_f1"] - clean_m["val_pitch_token_f1"], 6),
    }


def _eval_suite31_hw(
    model: Any,
    manifest_path: Path,
    device: "torch.device",
    *,
    has_rust_scorer: bool,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    import torch
    from PIL import Image
    import eval_sys_gate as esg
    import eval_uni_suite31 as ev
    import uni_omr

    model.eval()
    _, pieces = ev.resolve_manifest_pieces(manifest_path)
    workdir = Path(tempfile.mkdtemp(prefix="ov28_s31_"))
    rows: List[Dict[str, Any]] = []

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else contextlib.nullcontext()
    )

    for idx, piece in enumerate(pieces, start=1):
        slug = piece["slug"]
        is_hw = bool(piece["handwritten"])
        try:
            ref_page_toks, page_ref_notes = esg.suite31_page_refs(piece["gt"], "ov28-" + slug, workdir)
        except Exception as exc:
            rows.append({"piece": slug, "handwritten": is_hw, "skipped": True, "reason": str(exc)[:120]})
            continue

        ref_pitch = [t for t in ref_page_toks if _is_pitch_tok(t)]
        ref_rhythm = [t for t in ref_page_toks if _is_rhythm_tok(t)]
        prow: Dict[str, Any] = {"piece": slug, "handwritten": is_hw, "ref_tokens": len(ref_page_toks)}

        for domain in ("clean", "photo"):
            img_path: Path = piece[domain]
            with Image.open(img_path) as raw_im:
                pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), FULLPAGE_MAX_DIM)
            img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)

            with torch.no_grad():
                with amp_ctx:
                    seq = ev.generate_constrained(model, img_t, max_len=SUITE31_MAX_LEN, strict=False)

            seq_ids = seq[0].detach().cpu().tolist()
            _, _, eos_id, _, _, _ = ev._vocab_spec(int(model.cfg.vocab_size))
            pred_raw_toks = ev.decode_uni_compat(seq_ids, vocab_size=int(model.cfg.vocab_size), skip_special=True)
            if eos_id not in seq_ids[1:]:
                pred_raw_toks = ev.trim_truncated_tokens(pred_raw_toks)
            hyp_page_toks, _ = ev.stitch_uni_tokens(pred_raw_toks)

            tok_f1 = _seq_f1(hyp_page_toks, ref_page_toks)
            pitch_f1 = _seq_f1([t for t in hyp_page_toks if _is_pitch_tok(t)], ref_pitch)
            rhythm_f1 = _seq_f1([t for t in hyp_page_toks if _is_rhythm_tok(t)], ref_rhythm)

            note_f1: Optional[float] = None
            decode_ok = False
            if has_rust_scorer:
                if hyp_page_toks:
                    try:
                        hyp_txt = workdir / f"ov28-{slug}-{domain}-hyp-bekern.txt"
                        hyp_txt.write_text(" ".join(hyp_page_toks) + "\n", encoding="utf-8")
                        hyp_notes = workdir / f"ov28-{slug}-{domain}-hyp-notes.json"
                        p_dec = esg.run_cli([
                            esg.ASER, "bekern-decode", "--input", str(hyp_txt), "--out", str(hyp_notes),
                        ])
                        if p_dec.returncode == 0 and hyp_notes.exists():
                            sc = esg.score_notes_pair(
                                page_ref_notes,
                                hyp_notes,
                                workdir,
                                f"ov28-{slug}-{domain}",
                                esg.EHARNESS,
                                esg.ASER,
                            )
                            note_f1 = float(sc.get("f1", 0.0))
                            decode_ok = not bool(sc.get("zeroed"))
                        else:
                            note_f1 = 0.0
                    except Exception:
                        note_f1 = 0.0
                else:
                    note_f1 = 0.0

            p = domain + "_"
            prow[p + "tok_f1"] = round(tok_f1, 6)
            prow[p + "pitch_f1"] = round(pitch_f1, 6)
            prow[p + "rhythm_f1"] = round(rhythm_f1, 6)
            prow[p + "note_f1"] = round(note_f1, 6) if note_f1 is not None else None
            prow[p + "decode_ok"] = decode_ok

        rows.append(prow)
        if idx % 15 == 0 or idx == len(pieces):
            print(f"  [suite31_hw] scored {idx}/{len(pieces)} pieces", flush=True)

    shutil.rmtree(workdir, ignore_errors=True)

    def _agg(sub: List[Dict[str, Any]], domain: str) -> Dict[str, Any]:
        ok = [r for r in sub if not r.get("skipped")]
        p = domain + "_"

        def _mean(key: str) -> Optional[float]:
            xs = [float(r[p + key]) for r in ok if r.get(p + key) is not None]
            return round(sum(xs) / len(xs), 6) if xs else None

        dec_ok = sum(1 for r in ok if r.get(p + "decode_ok"))
        return {
            "n_pieces": len(ok),
            "tok_f1": _mean("tok_f1"),
            "pitch_tok_f1": _mean("pitch_f1"),
            "rhythm_tok_f1": _mean("rhythm_f1"),
            "note_f1": _mean("note_f1"),
            "decode_ok_rate": round(dec_ok / max(1, len(ok)), 4),
        }

    pr_rows = [r for r in rows if not r.get("handwritten")]
    hw_rows = [r for r in rows if r.get("handwritten")]

    return {
        "printed_clean": _agg(pr_rows, "clean"),
        "printed_photo": _agg(pr_rows, "photo"),
        "hw_clean": _agg(hw_rows, "clean"),
        "hw_photo": _agg(hw_rows, "photo"),
        "all45_clean": _agg(rows, "clean"),
        "all45_photo": _agg(rows, "photo"),
    }


def _train_and_eval_arm(
    arm_name: str,
    strength: str,
    train_pairs: List[Tuple[Path, List[Path], str]],
    val_pairs: List[Tuple[Path, List[Path], str]],
    manifest_path: Path,
    device: "torch.device",
    *,
    has_rust_scorer: bool,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    import train_uni_v2 as tv2
    from uni_omr import (
        GPUEmaShadow,
        UniFullPageDataset,
        _images_to_device_float,
        _infinite_loader,
        collate_uni_batch,
    )
    from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net

    arm_dir = VOL_ROOT / "ckpts" / EXP_ID / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached_res = arm_dir / "arm_result.json"
    if cached_res.is_file():
        print(f"[{arm_name}] Loading cached arm_result.json", flush=True)
        return json.loads(cached_res.read_text(encoding="utf-8"))

    preset = STRENGTH_PRESETS[strength]

    class DroopWarpBrokenStaffDataset(UniFullPageDataset):
        def __getitem__(self, idx: Any) -> Dict[str, Any]:
            item = super().__getitem__(idx)
            if strength != "none":
                item["image"] = apply_droop_warp_broken_staff(item["image"], preset, random)
            return item

    _seed_everything(SEED)
    cfg = UniOMRv2Config.small_45m(grad_checkpoint=True)
    model = UniOMRv2Net(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"\n[{arm_name}] Starting | strength={strength} | total_params={total_params:,} | steps={TRAIN_STEPS}",
        flush=True,
    )

    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    opt_kwargs: Dict[str, Any] = dict(lr=LR, betas=(0.9, 0.95), eps=1e-8)
    if device.type == "cuda":
        opt_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        **opt_kwargs,
    )
    gpu_ema = GPUEmaShadow(model, decay=0.995)

    _seed_everything(SEED + 100)
    train_ds = DroopWarpBrokenStaffDataset(
        train_pairs,
        fullpage_max_dim=FULLPAGE_MAX_DIM,
        max_seq_len=MAX_SEQ_LEN,
        augment=True,
        aug_prob=BASE_AUG_PROB,
        return_uint8=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_uni_batch,
    )
    train_iter = _infinite_loader(train_loader)

    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_arm0 = time.perf_counter()
    last_save_ts = time.time()
    save_interval_s = max(60.0, SAVE_EVERY_MINUTES * 60.0)

    history_100: List[Dict[str, Any]] = []
    buf_loss: List[float] = []
    buf_ce: List[float] = []
    buf_mtp: List[float] = []
    buf_ms: List[float] = []

    for step in range(1, TRAIN_STEPS + 1):
        t_step0 = time.perf_counter()
        cur_lr = _wsd_lr(step, TRAIN_STEPS, LR)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_ce = 0.0
        step_mtp = 0.0

        for _ in range(GRAD_ACCUM):
            batch = next(train_iter)
            images = _images_to_device_float(batch["images"], device, non_blocking=True)
            seq = batch["seq"].to(device, non_blocking=True)
            losses, _ = tv2.forward_and_loss_v2(
                model,
                images,
                seq,
                kind="fullpage",
                mtp_loss_weight=0.30,
                saliency_loss_weight=0.10,
                sys_y_loss_weight=0.25,
                sys_count_loss_weight=0.15,
                staff_seg_loss_weight=0.20,
                stroke_hog_weight=0.15,
                label_smoothing=0.05,
                use_bf16=use_bf16,
                device=device,
            )
            (losses["loss"] / float(GRAD_ACCUM)).backward()
            step_loss += float(losses["loss"].detach().item()) / float(GRAD_ACCUM)
            step_ce += float(losses["ce_loss"].detach().item()) / float(GRAD_ACCUM)
            step_mtp += float(losses["mtp_loss"].detach().item()) / float(GRAD_ACCUM)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        gpu_ema.update(model, step)

        buf_loss.append(step_loss)
        buf_ce.append(step_ce)
        buf_mtp.append(step_mtp)
        buf_ms.append((time.perf_counter() - t_step0) * 1000.0)

        if step % 100 == 0 or step == TRAIN_STEPS:
            rec = {
                "step": step,
                "lr": round(cur_lr, 8),
                "loss_mean": round(float(np.mean(buf_loss)), 5),
                "ce_mean": round(float(np.mean(buf_ce)), 5),
                "mtp_mean": round(float(np.mean(buf_mtp)), 5),
                "step_ms_mean": round(float(np.mean(buf_ms)), 1),
            }
            history_100.append(rec)
            print(
                f"  [{arm_name}] step {step:4d}/{TRAIN_STEPS} | lr={cur_lr:.2e} | "
                f"loss={rec['loss_mean']:.4f} ce={rec['ce_mean']:.4f} mtp={rec['mtp_mean']:.4f} | "
                f"{rec['step_ms_mean']:.0f} ms/step",
                flush=True,
            )
            buf_loss.clear()
            buf_ce.clear()
            buf_mtp.clear()
            buf_ms.clear()

        if (time.time() - last_save_ts) >= save_interval_s:
            tv2.atomic_save_v2_latest(arm_dir, model, cfg, step=step, gpu_ema=gpu_ema)
            last_save_ts = time.time()
            print(f"  [{arm_name}] 5-min atomic checkpoint saved at step {step}", flush=True)

    train_secs = time.perf_counter() - t_arm0
    peak_vram_gb = (
        round(float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 3), 3)
        if device.type == "cuda"
        else 0.0
    )
    tv2.atomic_save_v2_latest(arm_dir, model, cfg, step=TRAIN_STEPS, gpu_ema=gpu_ema)

    ema_sd = gpu_ema.cpu_state_dict(model)
    model.load_state_dict({k: v.to(device) for k, v in ema_sd.items()}, strict=True)
    model.eval()

    t_eval0 = time.perf_counter()
    print(f"[{arm_name}] Evaluating proxy_val clean + warped/broken-staff stress...", flush=True)
    val_eval = _eval_val_and_stress(model, val_pairs, device, use_bf16=use_bf16)
    print(
        f"  [{arm_name}] val clean CE={val_eval['clean']['val_ce_loss']:.4f} F1={val_eval['clean']['val_token_f1']:.4f} | "
        f"stress CE={val_eval['warped_broken_stress']['val_ce_loss']:.4f} F1={val_eval['warped_broken_stress']['val_token_f1']:.4f}",
        flush=True,
    )

    print(f"[{arm_name}] Evaluating 45-piece suite31_hw (90 pages)...", flush=True)
    s31_eval = _eval_suite31_hw(
        model,
        manifest_path,
        device,
        has_rust_scorer=has_rust_scorer,
        use_bf16=use_bf16,
    )
    eval_secs = time.perf_counter() - t_eval0

    result = {
        "arm": arm_name,
        "strength": strength,
        "preset": preset,
        "total_params": total_params,
        "train_steps": TRAIN_STEPS,
        "train_secs": round(train_secs, 2),
        "eval_secs": round(eval_secs, 2),
        "peak_vram_gb": peak_vram_gb,
        "history_per_100": history_100,
        "proxy_val_eval": val_eval,
        "suite31_hw_eval": s31_eval,
    }
    _atomic_write_json(cached_res, result)

    del model, optimizer, gpu_ema, train_ds, train_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    try:
        hf_token = _read_and_purge_hf_token()
        repo_root, aser_bin, eharness_bin = _setup_environment()

        import torch
        from huggingface_hub import HfApi

        t_start = time.time()
        started_iso = datetime.now(timezone.utc).isoformat()

        if torch.cuda.is_available():
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
        use_bf16 = bool(torch.cuda.is_available() and cc[0] >= 8)
        print(
            f"[{EXP_ID}] Running on Colab {gpu_name} (cc={cc[0]}.{cc[1]}, bf16={use_bf16}, torch={torch.__version__})",
            flush=True,
        )

        mvd_dir = str(repo_root / "tmp-train" / "mvd")
        if mvd_dir not in sys.path:
            sys.path.insert(0, mvd_dir)

        import eval_sys_gate as esg

        has_rust = bool(aser_bin is not None and eharness_bin is not None)
        if has_rust:
            esg.ASER = aser_bin
            esg.EHARNESS = eharness_bin

        train_abs, val_abs, manifest_path = _ensure_datasets(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)
        print(f"[{EXP_ID}] Loaded train={len(train_pairs)} val={len(val_pairs)}", flush=True)

        arm_results: Dict[str, Dict[str, Any]] = {}
        for arm_name, strength in (
            ("arm_a_none", "none"),
            ("arm_b_mild", "mild"),
            ("arm_c_strong", "strong"),
        ):
            arm_results[arm_name] = _train_and_eval_arm(
                arm_name,
                strength,
                train_pairs,
                val_pairs,
                manifest_path,
                device,
                has_rust_scorer=has_rust,
                use_bf16=use_bf16,
            )

        summary_table = {
            k: {
                "strength": v["strength"],
                "val_clean_ce": v["proxy_val_eval"]["clean"]["val_ce_loss"],
                "val_clean_tok_f1": v["proxy_val_eval"]["clean"]["val_token_f1"],
                "val_stress_ce": v["proxy_val_eval"]["warped_broken_stress"]["val_ce_loss"],
                "val_stress_tok_f1": v["proxy_val_eval"]["warped_broken_stress"]["val_token_f1"],
                "stress_ce_delta": v["proxy_val_eval"]["stress_ce_delta"],
                "s31_pr_clean_tok_f1": v["suite31_hw_eval"]["printed_clean"]["tok_f1"],
                "s31_hw_clean_tok_f1": v["suite31_hw_eval"]["hw_clean"]["tok_f1"],
                "s31_hw_photo_tok_f1": v["suite31_hw_eval"]["hw_photo"]["tok_f1"],
                "s31_hw_clean_note_f1": v["suite31_hw_eval"]["hw_clean"]["note_f1"],
                "s31_hw_photo_note_f1": v["suite31_hw_eval"]["hw_photo"]["note_f1"],
            }
            for k, v in arm_results.items()
        }

        report = {
            "exp_id": EXP_ID,
            "status": "completed",
            "started_utc": started_iso,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_secs": round(time.time() - t_start, 2),
            "platform": f"colab-{gpu_name}",
            "bf16": use_bf16,
            "hyperparams": {
                "preset": "small_45m",
                "steps_per_arm": TRAIN_STEPS,
                "seed": SEED,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM,
                "fullpage_max_dim": FULLPAGE_MAX_DIM,
                "max_seq_len": MAX_SEQ_LEN,
                "base_aug_prob": BASE_AUG_PROB,
                "sweep": ["none", "mild", "strong"],
            },
            "arms": arm_results,
            "summary": summary_table,
        }

        eval_local = VOL_ROOT / "evals" / f"{EXP_ID}.json"
        run_local = VOL_ROOT / "runs" / f"{EXP_ID}.json"
        _atomic_write_json(eval_local, report)
        _atomic_write_json(
            run_local,
            {
                "exp_id": EXP_ID,
                "status": "completed",
                "platform": f"colab-{gpu_name}",
                "started_utc": started_iso,
                "completed_utc": report["completed_utc"],
                "elapsed_secs": report["elapsed_secs"],
                "eval_path": HF_EVAL_PATH,
                "summary": summary_table,
            },
        )

        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(eval_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"eval({EXP_ID}): upload droop/warp/broken-staff 3-arm sweep",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload droop/warp/broken-staff run ledger",
        )
        print(f"\n[{EXP_ID}] UPLOADED {HF_EVAL_PATH} & {HF_RUN_PATH} TO HF!", flush=True)
        print(json.dumps(summary_table, indent=2), flush=True)
    finally:
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
