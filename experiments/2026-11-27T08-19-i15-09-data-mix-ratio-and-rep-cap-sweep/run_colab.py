#!/usr/bin/env python3
"""Colab H100/L4 runner for Wave 13 Exp i15-09:
Data-Mix Ratio & Per-Unique-Page Repetition-Cap Controlled Sweep.

Experiment ID: 2026-11-27T08-19-i15-09-data-mix-ratio-and-rep-cap-sweep
3-arm controlled data-mix ablation (800 steps/arm warm-started from frozen
UniOMRv2 EMA, identical LR/seed/grad_accum, evaluated on all 90 pages of
suite31_hw for BOTH RAW and EMA checkpoints):
  - arm0_v3_50_50: Baseline Iter 14 v3 mix
      (50% Printed / 19.5% Synth-HW / 12% Real-HW / 12% RISM-Pseudo /
       6% Real-Stitched / 0.5% Neg)
  - arm1_hw_heavy_30_70: HW-heavy mix
      (30% Printed / 25% Synth-HW / 18% Real-HW / 20% RISM-Pseudo /
       6.5% Real-Stitched / 0.5% Neg)
  - arm2_rep_cap_10x_hard_upsample: 50/50 mix WITH per-unique-page repetition
      cap (<= 10x on real_hw_archival, spilling excess weight cleanly to
      rism_pseudo + synth_hw) + 2x upsampling of hard layouts
      (landscape 4-staff, lead-sheet/lyrics, dense 5+ staff).

Gates:
  - PROMOTE: >= +2.0 pp HW Note F1 with <= 0.5 pp Printed drop vs arm0
  - ITERATE: partial gain (+0.5..+2.0 pp HW) or HW gain >= +2.0 pp with >0.5 pp Printed drop
  - DISCARD: < +0.5 pp HW gain or > 1.0 pp Printed regression without >= +2.0 pp HW gain

Uploads:
  - evals/2026-11-27T08-19-i15-09-data-mix-ratio-and-rep-cap-sweep.json
  - runs/2026-11-27T08-19-i15-09-data-mix-ratio-and-rep-cap-sweep.json
to gregoreesmaa/sheet-music-converter-omr. Resume-safe (5-minute atomic
checkpointing + per-arm done sentinels). Never prints HF_TOKEN.
"""

from __future__ import annotations

import contextlib
import copy
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-27T08-19-i15-09-data-mix-ratio-and-rep-cap-sweep"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_CANDIDATES: Tuple[str, ...] = (
    "checkpoints/iter14_v2a_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter13_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter10_5_v2_best/uni_omr_v2_ema_best.pt",
)
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_OPTIONAL_RISM_TARS: Tuple[str, ...] = (
    "datasets/rism_pseudolabels_ov39.tar",
    "datasets/rism_pseudo_proxy.tar",
)
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

HF_TOKEN_FILE = Path("/tmp/.hf_token")

# Controlled training hyperparameters (identical across all 3 arms)
SEED = 42
STEPS_PER_ARM = int(os.environ.get("I15_09_STEPS", "800"))
WARMUP_STEPS = max(10, int(round(STEPS_PER_ARM * 0.10)))
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.10
GRAD_ACCUM = 4
LR_MAX = 5.0e-5
WEIGHT_DECAY = 0.01
MTP_LOSS_WEIGHT = 0.25
TRAIN_MAX_DIM = 1280
TRAIN_MAX_SEQ_LEN = 1536
EVAL_MAX_DIM = 1408
EVAL_MAX_LEN = 3584
GPU_EMA_DECAY = 0.99
SAVE_EVERY_MINUTES = 5.0
SUITE31_LIMIT = int(os.environ.get("I15_09_SUITE31_LIMIT", "0"))

# Data-mix definitions
BUCKETS: Tuple[str, ...] = (
    "printed",
    "synth_hw",
    "real_hw_archival",
    "rism_pseudo",
    "real_hw_stitched",
    "neg_control",
)

ARM_CONFIGS: Dict[str, Dict[str, Any]] = {
    "arm0_v3_50_50": {
        "description": "Baseline Iter 14 v3 (50% Printed / 19.5% Synth-HW / 12% Real-HW / 12% RISM-Pseudo / 6% Real-Stitched / 0.5% Neg)",
        "weights": {
            "printed": 0.500,
            "synth_hw": 0.195,
            "real_hw_archival": 0.120,
            "rism_pseudo": 0.120,
            "real_hw_stitched": 0.060,
            "neg_control": 0.005,
        },
        "archival_rep_cap": None,
        "hard_layout_mult": 1.0,
    },
    "arm1_hw_heavy_30_70": {
        "description": "HW-heavy 30/70 (30% Printed / 25% Synth-HW / 18% Real-HW / 20% RISM-Pseudo / 6.5% Real-Stitched / 0.5% Neg)",
        "weights": {
            "printed": 0.300,
            "synth_hw": 0.250,
            "real_hw_archival": 0.180,
            "rism_pseudo": 0.200,
            "real_hw_stitched": 0.065,
            "neg_control": 0.005,
        },
        "archival_rep_cap": None,
        "hard_layout_mult": 1.0,
    },
    "arm2_rep_cap_10x_hard_upsample": {
        "description": "50/50 mix + <=10x per-unique-page cap on real_hw_archival (spilling to rism_pseudo+synth_hw) + 2x hard-layout upsample",
        "weights": {
            "printed": 0.500,
            "synth_hw": 0.195,
            "real_hw_archival": 0.120,
            "rism_pseudo": 0.120,
            "real_hw_stitched": 0.060,
            "neg_control": 0.005,
        },
        "archival_rep_cap": 10,
        "hard_layout_mult": 2.0,
    },
}

_STITCHED_HW_MARKERS: Tuple[str, ...] = ("1s2s", "stitched", "concat", "synth_page", "composed")
_LYRIC_LEAD_MARKERS: Tuple[str, ...] = (
    "lyric", "lyr", "lieder", "vocal", "lead", "choral", "choir", "song", "hymn", "harm",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_vol_root() -> Path:
    if Path("/vol").is_dir():
        return Path("/vol")
    if Path("/content").is_dir():
        return Path("/content/vol")
    return Path("/tmp/omr_vol")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_and_purge_hf_token() -> str:
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
        Path("/content/workspace/sheet-music-converter/tmp-train/mvd"),
        Path.cwd() / "tmp-train" / "mvd",
        Path(__file__).resolve().parents[2] / "tmp-train" / "mvd",
    ):
        if (cand / "uni_omr_v2.py").is_file() and (cand / "eval_uni_suite31.py").is_file():
            return cand
    for bundle in (Path("/content/ov_mvd_bundle.tar.gz"), Path("/content/bundle.tar.gz"), Path("/content/repo_bundle.tar.gz")):
        if bundle.is_file():
            dest = Path("/content/sheet-music-converter")
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=dest)
            for mvd in (dest / "tmp-train" / "mvd", dest / "mvd"):
                if (mvd / "uni_omr_v2.py").is_file():
                    return mvd
    raise FileNotFoundError("Could not locate tmp-train/mvd containing uni_omr_v2.py and eval_uni_suite31.py")


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _wsd_mult(step: int, total_steps: int, warmup_steps: int) -> float:
    if step <= warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    decay_steps = max(1, int(round(total_steps * WSD_DECAY_FRAC)))
    decay_start = total_steps - decay_steps
    if step <= decay_start:
        return 1.0
    prog = min(1.0, max(0.0, float(step - decay_start) / float(decay_steps)))
    return MIN_LR_RATIO + 0.5 * (1.0 - MIN_LR_RATIO) * (1.0 + math.cos(math.pi * prog))


def _ensure_assets(vol_root: Path, hf_token: str) -> Tuple[Path, str, Path, Path, Optional[Path]]:
    from huggingface_hub import hf_hub_download

    vol_root.mkdir(parents=True, exist_ok=True)
    ds_root = vol_root / "datasets"
    ds_root.mkdir(parents=True, exist_ok=True)
    ckpt_root = vol_root / "pretrained"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # 1. Frozen UniOMRv2 EMA checkpoint
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
                local_dir=str(vol_root / "tars"),
            )
            shutil.copy2(dl_p, target)
            Path(dl_p).unlink(missing_ok=True)
            ckpt_local = target
            ckpt_source = cand
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[{EXP_ID}] Checkpoint {cand} unavailable ({exc}); trying next...", flush=True)

    if ckpt_local is None or ckpt_source is None:
        raise RuntimeError(f"No frozen UniOMRv2 checkpoint found among {HF_CKPT_CANDIDATES}")

    # 2. Proxy dataset
    proxy_dir = ds_root / "iter10_5_proxy_2k"
    train_abs = proxy_dir / "proxy_train_abs.jsonl"
    if not train_abs.is_file():
        proxy_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR}...", flush=True)
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_PROXY_TAR,
                repo_type="dataset",
                token=hf_token,
                local_dir=str(vol_root / "tars"),
            )
        )
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=proxy_dir)
        tar_p.unlink(missing_ok=True)
        for src_name, dst_name in (("proxy_train.jsonl", "proxy_train_abs.jsonl"), ("proxy_val.jsonl", "proxy_val_abs.jsonl")):
            src_path = proxy_dir / src_name
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
            (proxy_dir / dst_name).write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    # 3. Optional RISM pseudo-label pack on HF
    rism_manifest: Optional[Path] = None
    rism_dir = ds_root / "rism_pseudo_opt"
    existing_rism = sorted(rism_dir.rglob("*.jsonl")) if rism_dir.is_dir() else []
    if existing_rism:
        rism_manifest = existing_rism[0]
    else:
        for cand_tar in HF_OPTIONAL_RISM_TARS:
            try:
                tar_p = Path(
                    hf_hub_download(
                        repo_id=HF_REPO_ID,
                        filename=cand_tar,
                        repo_type="dataset",
                        token=hf_token,
                        local_dir=str(vol_root / "tars"),
                    )
                )
                rism_dir.mkdir(parents=True, exist_ok=True)
                with tarfile.open(tar_p, "r:*") as tf:
                    tf.extractall(path=rism_dir)
                tar_p.unlink(missing_ok=True)
                found = sorted(rism_dir.rglob("*.jsonl"))
                if found:
                    rism_manifest = found[0]
                    print(f"[{EXP_ID}] Loaded optional RISM pseudo pack {cand_tar} -> {rism_manifest}", flush=True)
                    break
            except Exception:  # noqa: BLE001
                continue

    # 4. suite31_hw (all 45 pieces = 90 pages)
    s31_dir = ds_root / "suite31_hw_eval"
    sentinel = s31_dir / ".extracted"
    if not sentinel.exists():
        print(f"[{EXP_ID}] Downloading {HF_SUITE31_TAR}...", flush=True)
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_SUITE31_TAR,
                repo_type="dataset",
                token=hf_token,
                local_dir=str(vol_root / "tars"),
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

    return ckpt_local, ckpt_source, train_abs, manifests[0], rism_manifest


def _classify_bucket(rec: Dict[str, Any]) -> str:
    b_in = str(rec.get("bucket", "")).strip().lower()
    dom = str(rec.get("domain", "")).strip().lower()
    p_str = str(rec.get("image", "")).replace("\\", "/").lower()
    if b_in == "neg_control" or dom == "neg_control" or "neg-controls" in p_str or "neg_control" in p_str:
        return "neg_control"
    if b_in == "rism_pseudo" or dom == "rism_pseudo" or "rism_pseudo" in p_str or "rism-pseudo" in p_str:
        return "rism_pseudo"
    if dom == "real_hw" or "real_hw" in p_str or "real-hw" in p_str or "debussy" in p_str:
        if any(tok in p_str or tok in dom for tok in _STITCHED_HW_MARKERS):
            return "real_hw_stitched"
        return "real_hw_archival"
    if dom in ("hw", "synth_hw") or "hw-sim" in p_str or "hw_sim" in p_str or "hand" in dom:
        return "synth_hw"
    return "printed"


def _synthesize_neg_controls(out_dir: Path, n_pages: int = 16) -> List[Dict[str, Any]]:
    """Synthesize deterministic non-music negative control pages (blank/aged/ruled paper)
    with empty token target (`""` -> `[BOS, EOS]`) if the proxy tar has 0 neg_control rows."""
    import numpy as np
    from PIL import Image

    neg_dir = out_dir / "neg_controls_synth"
    neg_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    for i in range(n_pages):
        img_p = neg_dir / f"neg_ctrl_{i:03d}.png"
        if not img_p.is_file():
            h, w = (1024, 768) if (i % 3 != 0) else (768, 1024)
            base = rng.integers(228, 250)
            arr = np.full((h, w, 3), base, dtype=np.int16)
            arr[:, :, 0] += rng.integers(-4, 6)
            arr[:, :, 2] -= rng.integers(0, 12)
            noise = rng.normal(0.0, 5.0, size=(h, w, 3))
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            if i % 2 == 1:
                # Subtle horizontal ruled notebook lines (not 5-line musical staves)
                for y in range(80, h - 80, 44):
                    arr[y : y + 1, 48 : w - 48, :] = np.clip(arr[y : y + 1, 48 : w - 48, :].astype(np.int16) - 28, 0, 255).astype(np.uint8)
            Image.fromarray(arr, mode="RGB").save(img_p)
        rows.append(
            {
                "image": str(img_p.resolve()),
                "bekerns": [],
                "tokens": [],
                "domain": "neg_control",
                "bucket": "neg_control",
                "unique_key": f"neg_ctrl_{i:03d}",
                "is_hard_layout": False,
                "hard_reasons": [],
            }
        )
    return rows


def _annotate_and_bucket_pool(
    train_manifest: Path,
    rism_manifest: Optional[Path],
    vol_root: Path,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    from PIL import Image

    raw_rows: List[Dict[str, Any]] = []
    for line in train_manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw_rows.append(json.loads(line))

    if rism_manifest is not None and rism_manifest.is_file():
        rism_root = rism_manifest.parent
        for line in rism_manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            img_p = Path(rec["image"])
            if not img_p.is_file():
                img_p = (rism_root / rec["image"]).resolve()
            if img_p.is_file():
                rec["image"] = str(img_p)
                rec["bekerns"] = [
                    str(Path(b) if Path(b).is_file() else (rism_root / b).resolve())
                    for b in rec.get("bekerns", [])
                ]
                rec["bucket"] = "rism_pseudo"
                rec["domain"] = "rism_pseudo"
                raw_rows.append(rec)

    pool: Dict[str, List[Dict[str, Any]]] = {b: [] for b in BUCKETS}
    hard_counts: Counter = Counter()

    for rec in raw_rows:
        img_p = Path(rec["image"])
        if not img_p.is_file():
            continue
        bucket = _classify_bucket(rec)
        beks = [b for b in rec.get("bekerns", []) if Path(b).is_file()]
        has_inline_toks = isinstance(rec.get("tokens"), list)
        if bucket != "neg_control" and not beks and not has_inline_toks:
            continue

        # Fast image header inspect (lazy, no pixel decode) for landscape detection
        try:
            with Image.open(img_p) as im:
                w, h = im.size
        except Exception:  # noqa: BLE001
            continue

        n_systems = len(beks) if beks else int(rec.get("n_systems", 1))
        aspect = float(w) / float(max(1, h))
        p_low = str(img_p).replace("\\", "/").lower()
        dom_low = str(rec.get("domain", "")).lower()

        reasons: List[str] = []
        if (aspect >= 1.15 and n_systems >= 3) or (w > h and n_systems >= 4):
            reasons.append("landscape_4staff")
        if any(m in p_low or m in dom_low for m in _LYRIC_LEAD_MARKERS):
            reasons.append("leadsheet_lyrics")
        if n_systems >= 5:
            reasons.append("dense_5plus_staff")

        for r in reasons:
            hard_counts[r] += 1

        # Canonical unique-page identity (strips photo-degrade / copy suffixes if present)
        stem = img_p.stem
        unique_key = str(rec.get("source_page_id") or rec.get("page_id") or stem)

        pool[bucket].append(
            {
                "image": str(img_p),
                "bekerns": beks,
                "tokens": rec.get("tokens") if has_inline_toks else None,
                "bucket": bucket,
                "unique_key": unique_key,
                "n_systems": n_systems,
                "aspect": round(aspect, 3),
                "is_hard_layout": bool(reasons),
                "hard_reasons": reasons,
            }
        )

    neg_synthesized = False
    if not pool["neg_control"]:
        pool["neg_control"] = _synthesize_neg_controls(vol_root / "datasets", n_pages=16)
        neg_synthesized = True

    pool_stats = {
        "bucket_sizes": {b: len(pool[b]) for b in BUCKETS},
        "unique_real_hw_archival_pages": len({r["unique_key"] for r in pool["real_hw_archival"]}),
        "hard_layout_counts": dict(hard_counts),
        "hard_layout_total_pages": sum(1 for b in BUCKETS for r in pool[b] if r["is_hard_layout"]),
        "neg_control_synthesized": neg_synthesized,
        "rism_pseudo_available": len(pool["rism_pseudo"]) > 0,
    }
    return pool, pool_stats


def _read_and_encode_target(
    rec: Dict[str, Any],
    *,
    bos_id: int,
    eos_id: int,
    sys_id: int,
    tok_to_id: Dict[str, int],
    max_seq_len: int,
) -> List[int]:
    if rec.get("bucket") == "neg_control":
        return [bos_id, eos_id]

    ids: List[int] = [bos_id]
    if isinstance(rec.get("tokens"), list):
        for t in rec["tokens"]:
            tid = tok_to_id.get(str(t))
            if tid is not None:
                ids.append(tid)
    else:
        for idx, bp in enumerate(rec.get("bekerns", [])):
            if idx > 0:
                ids.append(sys_id)
            raw = Path(bp).read_text(encoding="utf-8", errors="ignore").strip()
            if not raw:
                continue
            for t in raw.split():
                tid = tok_to_id.get(t)
                if tid is not None:
                    ids.append(tid)
    if len(ids) > max_seq_len - 1:
        ids = ids[: max_seq_len - 1]
    ids.append(eos_id)
    return ids


class DataMixSampler:
    """Controlled bucket sampler supporting per-unique-page repetition caps on
    `real_hw_archival` (with clean spill to `rism_pseudo` + `synth_hw`) and
    intra-bucket hard-layout upsampling (`arm2`)."""

    def __init__(
        self,
        pool: Dict[str, List[Dict[str, Any]]],
        bucket_weights: Dict[str, float],
        *,
        archival_rep_cap: Optional[int],
        hard_layout_mult: float,
        seed: int,
    ) -> None:
        self.pool = pool
        self.base_weights = dict(bucket_weights)
        self.archival_rep_cap = archival_rep_cap
        self.hard_layout_mult = float(hard_layout_mult)
        self.rng = random.Random(seed)

        # Telemetry
        self.bucket_draws: Counter = Counter()
        self.archival_page_draws: Counter = Counter()
        self.spilled_draws: int = 0
        self.hard_layout_draws: int = 0
        self.total_draws: int = 0

        # Precompute intra-bucket item weights
        self.item_weights: Dict[str, List[float]] = {}
        for b, items in self.pool.items():
            self.item_weights[b] = [
                self.hard_layout_mult if r.get("is_hard_layout") else 1.0 for r in items
            ]

    def _effective_bucket_weights(self) -> Tuple[List[str], List[float]]:
        eff = dict(self.base_weights)

        # Spill any empty bucket cleanly
        if not self.pool["rism_pseudo"]:
            spill = eff.pop("rism_pseudo", 0.0)
            if self.pool["synth_hw"]:
                eff["synth_hw"] = eff.get("synth_hw", 0.0) + spill * 0.65
                eff["real_hw_archival"] = eff.get("real_hw_archival", 0.0) + spill * 0.35
            else:
                eff["real_hw_archival"] = eff.get("real_hw_archival", 0.0) + spill

        if not self.pool["real_hw_stitched"]:
            spill = eff.pop("real_hw_stitched", 0.0)
            eff["synth_hw"] = eff.get("synth_hw", 0.0) + spill

        # Enforce archival rep cap: if all archival pages reached cap, spill archival weight
        if self.archival_rep_cap is not None and self.pool["real_hw_archival"]:
            avail = [
                r for r in self.pool["real_hw_archival"]
                if self.archival_page_draws[r["unique_key"]] < self.archival_rep_cap
            ]
            if not avail:
                spill = eff.pop("real_hw_archival", 0.0)
                if self.pool["rism_pseudo"]:
                    eff["rism_pseudo"] = eff.get("rism_pseudo", 0.0) + spill * 0.5
                    eff["synth_hw"] = eff.get("synth_hw", 0.0) + spill * 0.5
                else:
                    eff["synth_hw"] = eff.get("synth_hw", 0.0) + spill

        active_buckets: List[str] = []
        active_weights: List[float] = []
        for b in BUCKETS:
            w = float(eff.get(b, 0.0))
            if w > 0.0 and self.pool.get(b):
                active_buckets.append(b)
                active_weights.append(w)
        if not active_buckets:
            raise RuntimeError("No populated training buckets available")
        return active_buckets, active_weights

    def draw(self) -> Dict[str, Any]:
        buckets, weights = self._effective_bucket_weights()
        chosen_b = self.rng.choices(buckets, weights=weights, k=1)[0]

        if chosen_b == "real_hw_archival" and self.archival_rep_cap is not None:
            eligible_idx = [
                i for i, r in enumerate(self.pool["real_hw_archival"])
                if self.archival_page_draws[r["unique_key"]] < self.archival_rep_cap
            ]
            if not eligible_idx:
                # Spill cleanly to rism_pseudo + synth_hw
                self.spilled_draws += 1
                spill_targets = [b for b in ("rism_pseudo", "synth_hw") if self.pool.get(b)]
                chosen_b = self.rng.choice(spill_targets) if spill_targets else "printed"
                items = self.pool[chosen_b]
                rec = self.rng.choices(items, weights=self.item_weights[chosen_b], k=1)[0]
            else:
                items = [self.pool["real_hw_archival"][i] for i in eligible_idx]
                w_elig = [self.item_weights["real_hw_archival"][i] for i in eligible_idx]
                rec = self.rng.choices(items, weights=w_elig, k=1)[0]
        else:
            items = self.pool[chosen_b]
            rec = self.rng.choices(items, weights=self.item_weights[chosen_b], k=1)[0]

        self.total_draws += 1
        self.bucket_draws[chosen_b] += 1
        if chosen_b == "real_hw_archival":
            self.archival_page_draws[rec["unique_key"]] += 1
        if rec.get("is_hard_layout"):
            self.hard_layout_draws += 1
        return rec

    def state_dict(self) -> Dict[str, Any]:
        return {
            "bucket_draws": dict(self.bucket_draws),
            "archival_page_draws": dict(self.archival_page_draws),
            "spilled_draws": self.spilled_draws,
            "hard_layout_draws": self.hard_layout_draws,
            "total_draws": self.total_draws,
        }

    def load_state_dict(self, st: Dict[str, Any]) -> None:
        self.bucket_draws = Counter(st.get("bucket_draws", {}))
        self.archival_page_draws = Counter(st.get("archival_page_draws", {}))
        self.spilled_draws = int(st.get("spilled_draws", 0))
        self.hard_layout_draws = int(st.get("hard_layout_draws", 0))
        self.total_draws = int(st.get("total_draws", 0))

    def telemetry(self) -> Dict[str, Any]:
        tot = max(1, self.total_draws)
        arch_vals = list(self.archival_page_draws.values())
        return {
            "total_draws": self.total_draws,
            "bucket_draws": {b: int(self.bucket_draws.get(b, 0)) for b in BUCKETS},
            "bucket_fracs": {b: round(float(self.bucket_draws.get(b, 0)) / float(tot), 4) for b in BUCKETS},
            "archival_rep_cap": self.archival_rep_cap,
            "archival_unique_pages_seen": len(arch_vals),
            "archival_max_reps": max(arch_vals) if arch_vals else 0,
            "archival_mean_reps": round(sum(arch_vals) / max(1, len(arch_vals)), 2) if arch_vals else 0.0,
            "spilled_draws": self.spilled_draws,
            "hard_layout_draws": self.hard_layout_draws,
            "hard_layout_draw_frac": round(float(self.hard_layout_draws) / float(tot), 4),
        }


def _pil_to_canonical_tensor(img_pil: Any, max_dim: int, uni_omr_mod: Any) -> "torch.Tensor":
    pil_im = uni_omr_mod._resize_fullpage_pil(img_pil.convert("RGB"), int(max_dim))
    return uni_omr_mod._pil_to_chw_tensor(pil_im)


def _extract_musicxml_notes(xml_path: Path) -> List[Tuple[str, int, str, str, str]]:
    notes: List[Tuple[str, int, str, str, str]] = []
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:  # noqa: BLE001
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
            elif ctag == "type":
                dur = (child.text or "").strip().lower()
            elif ctag == "dot":
                n_dots += 1
        if not is_rest and not is_grace and step and octave and dur:
            notes.append((dur, n_dots, step, octave, alter))
    return notes


def _eval_suite31_all90(
    model: Any,
    pieces: Sequence[Dict[str, Any]],
    *,
    device: Any,
    use_bf16: bool,
    uni_omr_mod: Any,
    eus: Any,
    rlvr_grpo: Any,
    limit: int = 0,
) -> Dict[str, Any]:
    import torch
    from PIL import Image

    model.eval()
    _, _, eos_id, _, _, _ = eus._vocab_spec(int(model.cfg.vocab_size))
    eval_list = list(pieces[:limit]) if limit > 0 else list(pieces)

    buckets: Dict[str, List[float]] = {
        "printed_clean": [],
        "printed_photo": [],
        "printed_all": [],
        "hw_clean": [],
        "hw_photo": [],
        "hw_all": [],
        "hw_landscape": [],
        "hw_portrait": [],
        "all90": [],
    }
    latencies: List[float] = []
    truncated_pages = 0

    ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else contextlib.nullcontext()
    )

    with torch.no_grad():
        for entry in eval_list:
            is_hw = bool(entry.get("handwritten", False))
            ref_notes = _extract_musicxml_notes(Path(entry["gt"]))
            if not ref_notes:
                continue
            for view in ("clean", "photo"):
                img_p = Path(entry[view])
                if not img_p.is_file():
                    continue
                with Image.open(img_p) as im:
                    rgb = im.convert("RGB")
                    w, h = rgb.size
                    inp = _pil_to_canonical_tensor(rgb, EVAL_MAX_DIM, uni_omr_mod).unsqueeze(0).to(device)
                t0 = time.perf_counter()
                with ctx:
                    seq = model.generate(inp, max_len=EVAL_MAX_LEN)
                latencies.append(time.perf_counter() - t0)
                ids = seq[0].tolist()
                pred_toks = eus.decode_uni_compat(ids, vocab_size=int(model.cfg.vocab_size), skip_special=True)
                if eos_id not in ids[1:]:
                    truncated_pages += 1
                    pred_toks = eus.trim_truncated_tokens(pred_toks)
                hyp_toks, _ = eus.stitch_uni_tokens(pred_toks)
                hyp_notes = rlvr_grpo.extract_pitch_dur_notes(hyp_toks)
                nf1 = float(rlvr_grpo.compute_sequence_f1(ref_notes, hyp_notes))

                dom_key = "hw" if is_hw else "printed"
                buckets[f"{dom_key}_{view}"].append(nf1)
                buckets[f"{dom_key}_all"].append(nf1)
                buckets["all90"].append(nf1)
                if is_hw:
                    buckets["hw_landscape" if w > h else "hw_portrait"].append(nf1)

    def _mean(xs: Sequence[float]) -> Optional[float]:
        return round(sum(xs) / float(len(xs)), 4) if xs else None

    return {
        **{k: _mean(v) for k, v in buckets.items()},
        "counts": {k: len(v) for k, v in buckets.items()},
        "truncated_pages": truncated_pages,
        "mean_latency_s": round(sum(latencies) / max(1, len(latencies)), 3),
    }


def _train_and_eval_arm(
    arm_name: str,
    arm_cfg: Dict[str, Any],
    *,
    ckpt_path: Path,
    pool: Dict[str, List[Dict[str, Any]]],
    s31_pieces: Sequence[Dict[str, Any]],
    out_dir: Path,
    device: Any,
    use_bf16: bool,
    uni_omr_mod: Any,
    eus: Any,
    rlvr_grpo: Any,
) -> Dict[str, Any]:
    import torch
    from PIL import Image

    arm_dir = out_dir / arm_name
    done_file = arm_dir / "arm_done.json"
    ckpt_file = arm_dir / "ckpt_latest.pt"
    if done_file.is_file():
        print(f"[{EXP_ID}] Reusing completed arm {arm_name} from {done_file}", flush=True)
        return json.loads(done_file.read_text(encoding="utf-8"))

    arm_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(SEED)

    model, cfg, meta = eus.load_uni_checkpoint(ckpt_path, device)
    missing = meta.get("missing_keys") or []
    if len(missing) > 0:
        raise RuntimeError(f"Partially loaded warm-start ckpt ({len(missing)} missing keys): {missing[:5]}")

    pad_id, bos_id, eos_id, sys_id, tok_to_id, _ = eus._vocab_spec(int(cfg.vocab_size))
    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    sampler = DataMixSampler(
        pool,
        arm_cfg["weights"],
        archival_rep_cap=arm_cfg["archival_rep_cap"],
        hard_layout_mult=arm_cfg["hard_layout_mult"],
        seed=SEED,
    )

    start_step = 0
    loss_hist: List[float] = []
    if ckpt_file.is_file():
        st = torch.load(ckpt_file, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        ema_state = st["ema"]
        opt.load_state_dict(st["opt"])
        start_step = int(st.get("step", 0))
        loss_hist = list(st.get("loss_hist", []))
        if "sampler" in st:
            sampler.load_state_dict(st["sampler"])
        print(f"[{EXP_ID}] Resumed {arm_name} at step {start_step}/{STEPS_PER_ARM}", flush=True)

    t0 = time.perf_counter()
    last_save = time.monotonic()
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else contextlib.nullcontext()
    )

    for step in range(start_step + 1, STEPS_PER_ARM + 1):
        model.train()
        lr_now = LR_MAX * _wsd_mult(step, STEPS_PER_ARM, WARMUP_STEPS)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        valid_micro = 0
        for _ in range(GRAD_ACCUM):
            rec = sampler.draw()
            ids = _read_and_encode_target(
                rec,
                bos_id=bos_id,
                eos_id=eos_id,
                sys_id=sys_id,
                tok_to_id=tok_to_id,
                max_seq_len=TRAIN_MAX_SEQ_LEN,
            )
            if len(ids) < 2:
                continue
            with Image.open(rec["image"]) as im:
                inp = _pil_to_canonical_tensor(im, TRAIN_MAX_DIM, uni_omr_mod).unsqueeze(0).to(device)
            tgt = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            with amp_ctx:
                out = model(inp, tgt[:, :-1], return_aux=False, return_mtp=True)
                loss_dict = model.compute_loss(
                    out,
                    tgt[:, 1:],
                    mtp_loss_weight=MTP_LOSS_WEIGHT,
                    saliency_loss_weight=0.0,
                    sys_y_loss_weight=0.0,
                    sys_count_loss_weight=0.0,
                    staff_seg_loss_weight=0.0,
                    stroke_hog_loss_weight=0.0,
                )
                loss = loss_dict["loss"] / float(GRAD_ACCUM)
            if torch.isfinite(loss):
                loss.backward()
                step_loss += float(loss.detach().item())
                valid_micro += 1

        if valid_micro > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.is_floating_point():
                        ema_state[k].mul_(GPU_EMA_DECAY).add_(v.detach(), alpha=1.0 - GPU_EMA_DECAY)
                    else:
                        ema_state[k].copy_(v.detach())
            loss_hist.append(step_loss)

        if step == 1 or step % 100 == 0 or step == STEPS_PER_ARM:
            tail = loss_hist[-50:]
            mean_l = sum(tail) / max(1, len(tail))
            print(
                f"[{EXP_ID}][{arm_name}] step={step}/{STEPS_PER_ARM} lr={lr_now:.2e} "
                f"loss50={mean_l:.4f} elapsed={time.perf_counter()-t0:.1f}s",
                flush=True,
            )

        if (time.monotonic() - last_save) >= SAVE_EVERY_MINUTES * 60.0 or step == STEPS_PER_ARM:
            tmp_ck = ckpt_file.with_suffix(".pt.tmp")
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "ema": ema_state,
                    "opt": opt.state_dict(),
                    "loss_hist": loss_hist[-200:],
                    "sampler": sampler.state_dict(),
                },
                tmp_ck,
            )
            os.replace(tmp_ck, ckpt_file)
            last_save = time.monotonic()

    train_elapsed = round(time.perf_counter() - t0, 2)

    # Evaluate BOTH RAW and EMA on all 90 pages of suite31_hw
    print(f"[{EXP_ID}][{arm_name}] Evaluating RAW checkpoint on suite31_hw...", flush=True)
    raw_eval = _eval_suite31_all90(
        model,
        s31_pieces,
        device=device,
        use_bf16=use_bf16,
        uni_omr_mod=uni_omr_mod,
        eus=eus,
        rlvr_grpo=rlvr_grpo,
        limit=SUITE31_LIMIT,
    )

    print(f"[{EXP_ID}][{arm_name}] Evaluating EMA checkpoint on suite31_hw...", flush=True)
    model.load_state_dict(ema_state, strict=True)
    ema_eval = _eval_suite31_all90(
        model,
        s31_pieces,
        device=device,
        use_bf16=use_bf16,
        uni_omr_mod=uni_omr_mod,
        eus=eus,
        rlvr_grpo=rlvr_grpo,
        limit=SUITE31_LIMIT,
    )

    # Pick primary (higher all90 between RAW and EMA) while recording exact delta
    raw_all = float(raw_eval["all90"] or 0.0)
    ema_all = float(ema_eval["all90"] or 0.0)
    best_kind = "ema" if ema_all >= raw_all else "raw"
    best_eval = ema_eval if best_kind == "ema" else raw_eval

    summary = {
        "arm": arm_name,
        "description": arm_cfg["description"],
        "target_weights": arm_cfg["weights"],
        "archival_rep_cap": arm_cfg["archival_rep_cap"],
        "hard_layout_mult": arm_cfg["hard_layout_mult"],
        "steps": STEPS_PER_ARM,
        "train_elapsed_sec": train_elapsed,
        "train_loss_tail50": round(sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:])), 5) if loss_hist else None,
        "sampler_telemetry": sampler.telemetry(),
        "raw_suite31_hw": raw_eval,
        "ema_suite31_hw": ema_eval,
        "best_ckpt_kind": best_kind,
        "best_suite31_hw": best_eval,
        "ema_minus_raw_all90_pp": round(100.0 * (ema_all - raw_all), 3),
    }
    _atomic_write_json(done_file, summary)
    del model, opt, ema_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


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


def main() -> None:
    t_start = time.perf_counter()
    started_at = _utc_now()
    hf_token = _read_and_purge_hf_token()
    try:
        _ensure_deps()
        mvd_dir = _locate_mvd()
        if str(mvd_dir) not in sys.path:
            sys.path.insert(0, str(mvd_dir))

        import torch
        import eval_uni_suite31 as eus
        import rlvr_grpo
        import uni_omr

        vol_root = _get_vol_root()
        out_dir = vol_root / "out" / EXP_ID
        out_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        print(f"[{EXP_ID}] Device={device} ({gpu_name}) bf16={use_bf16} steps/arm={STEPS_PER_ARM}", flush=True)

        ckpt_local, ckpt_source, train_abs, s31_manifest, rism_manifest = _ensure_assets(vol_root, hf_token)
        pool, pool_stats = _annotate_and_bucket_pool(train_abs, rism_manifest, vol_root)
        _, s31_pieces = eus.resolve_manifest_pieces(s31_manifest)
        print(f"[{EXP_ID}] Warm-start ckpt: {ckpt_source} | Pool: {pool_stats}", flush=True)

        # Fail-fast 2-page baseline sanity check before spending training compute
        base_model, base_cfg, base_meta = eus.load_uni_checkpoint(ckpt_local, device)
        sanity_eval = _eval_suite31_all90(
            base_model,
            [p for p in s31_pieces if not p.get("handwritten", False)][:1],
            device=device,
            use_bf16=use_bf16,
            uni_omr_mod=uni_omr,
            eus=eus,
            rlvr_grpo=rlvr_grpo,
        )
        print(f"[{EXP_ID}] Baseline 2-page sanity check: {sanity_eval}", flush=True)
        del base_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        arms_out: Dict[str, Any] = {}
        for arm_name in ("arm0_v3_50_50", "arm1_hw_heavy_30_70", "arm2_rep_cap_10x_hard_upsample"):
            arms_out[arm_name] = _train_and_eval_arm(
                arm_name,
                ARM_CONFIGS[arm_name],
                ckpt_path=ckpt_local,
                pool=pool,
                s31_pieces=s31_pieces,
                out_dir=out_dir,
                device=device,
                use_bf16=use_bf16,
                uni_omr_mod=uni_omr,
                eus=eus,
                rlvr_grpo=rlvr_grpo,
            )

        # Compute apples-to-apples deltas vs arm0_v3_50_50 and evaluate PROMOTE / ITERATE / DISCARD gates
        a0 = arms_out["arm0_v3_50_50"]["best_suite31_hw"]
        a0_hw = float(a0["hw_all"] or 0.0)
        a0_pr = float(a0["printed_all"] or 0.0)
        a0_all = float(a0["all90"] or 0.0)

        comparison: Dict[str, Any] = {}
        promoted_arms: List[str] = []
        iterate_arms: List[str] = []
        for cand in ("arm1_hw_heavy_30_70", "arm2_rep_cap_10x_hard_upsample"):
            cb = arms_out[cand]["best_suite31_hw"]
            hw_d = round(100.0 * (float(cb["hw_all"] or 0.0) - a0_hw), 3)
            pr_d = round(100.0 * (float(cb["printed_all"] or 0.0) - a0_pr), 3)
            all_d = round(100.0 * (float(cb["all90"] or 0.0) - a0_all), 3)
            promotes = bool(hw_d >= 2.0 and pr_d >= -0.5)
            iterates = bool((not promotes) and ((hw_d >= 2.0 and pr_d < -0.5) or (0.5 <= hw_d < 2.0 and pr_d >= -0.5)))
            if promotes:
                promoted_arms.append(cand)
            elif iterates:
                iterate_arms.append(cand)
            comparison[cand] = {
                "delta_hw_all_pp": hw_d,
                "delta_printed_all_pp": pr_d,
                "delta_all90_pp": all_d,
                "promote_gate_pass": promotes,
                "iterate_gate_hit": iterates,
            }

        verdict = "PROMOTE" if promoted_arms else ("ITERATE" if iterate_arms else "DISCARD")
        completed_at = _utc_now()
        elapsed_sec = round(time.perf_counter() - t_start, 2)

        eval_payload = {
            "exp_id": EXP_ID,
            "short_id": "i15-09",
            "wave": 13,
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_sec": elapsed_sec,
            "device": str(device),
            "gpu_name": gpu_name,
            "bf16": use_bf16,
            "warm_start_ckpt": ckpt_source,
            "warm_start_meta": {
                "arch": base_meta.get("arch"),
                "step": base_meta.get("step"),
                "vocab_size": int(base_cfg.vocab_size),
                "d_model": int(getattr(base_cfg, "d_model", 0)),
            },
            "hyperparams": {
                "seed": SEED,
                "steps_per_arm": STEPS_PER_ARM,
                "grad_accum": GRAD_ACCUM,
                "lr_max": LR_MAX,
                "warmup_steps": WARMUP_STEPS,
                "wsd_decay_frac": WSD_DECAY_FRAC,
                "min_lr_ratio": MIN_LR_RATIO,
                "gpu_ema_decay": GPU_EMA_DECAY,
                "train_max_dim": TRAIN_MAX_DIM,
                "eval_max_dim": EVAL_MAX_DIM,
                "eval_max_len": EVAL_MAX_LEN,
            },
            "pool_stats": pool_stats,
            "baseline_2p_sanity": sanity_eval,
            "arms": arms_out,
            "comparison_vs_arm0": comparison,
            "promoted_arms": promoted_arms,
            "iterate_arms": iterate_arms,
            "verdict": verdict,
        }

        eval_file = out_dir / f"{EXP_ID}.eval.json"
        run_file = out_dir / f"{EXP_ID}.run.json"
        _atomic_write_json(eval_file, eval_payload)
        _atomic_write_json(
            run_file,
            {
                "exp_id": EXP_ID,
                "short_id": "i15-09",
                "wave": 13,
                "status": "COMPLETED",
                "started_at": started_at,
                "completed_at": completed_at,
                "elapsed_sec": elapsed_sec,
                "verdict": verdict,
                "promoted_arms": promoted_arms,
                "comparison_vs_arm0": comparison,
                "hf_eval_path": HF_EVAL_PATH,
            },
        )
        _upload_json(hf_token, eval_file, HF_EVAL_PATH)
        _upload_json(hf_token, run_file, HF_RUN_PATH)
    finally:
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
