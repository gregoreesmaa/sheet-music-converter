#!/usr/bin/env python3
"""Google Colab L4 runner for Overnight Exp 19: Router saliency_thr x max_keep_ratio Sweep.

Experiment ID: 2026-11-26T01-38-ov-19-router-saliency-keep-sweep
Sweeps 3 operating points of InkTokenRouterV2 (saliency_thr, max_keep_ratio) on
UniOMR v2 small_45m (BF16, seed=42, lr=3e-4, Muon+AdamW, 600 steps/arm, dim=1280)
using datasets/iter10_5_proxy_2k.tar:
  1. arm_003_055: saliency_thr=0.03, max_keep_ratio=0.55
  2. arm_005_045: saliency_thr=0.05, max_keep_ratio=0.45
  3. arm_008_035: saliency_thr=0.08, max_keep_ratio=0.35

Measures per arm on proxy_val.jsonl (raw + EMA):
  - Ink recall (mean, p10, min)
  - Token compression ratio (mean_compression_x, keep_fraction)
  - Teacher-forced val_loss / val_ce / val_mtp / val_saliency
  - Generation val_token_f1, val_ser, sys_count_exact, tokens_per_sec
  - Zero-shot cross-threshold eval matrix on arm_005_045 EMA weights

Uploads results to Hugging Face Private Dataset Hub (gregoreesmaa/sheet-music-converter-omr):
  - evals/2026-11-26T01-38-ov-19-router-saliency-keep-sweep.json
  - runs/2026-11-26T01-38-ov-19-router-saliency-keep-sweep.json
"""

from __future__ import annotations

import gc
import json
import os
import random
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T01-38-ov-19-router-saliency-keep-sweep"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

WORKSPACE_ROOT = Path("/workspace")
VOL_ROOT = Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")
BUNDLE_CANDIDATES = (
    Path("/content/bundle.tar.gz"),
    Path("/content/ov19_bundle.tar.gz"),
    Path("/content/repo_bundle.tar.gz"),
)

# Shared hyperparameters across all 3 arms
TRAIN_STEPS = 600
SEED = 42
LR = 3e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 2
GRAD_ACCUM = 4
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
VAL_EVAL_MAX_LEN = 512
SAVE_EVERY_MINUTES = 5.0

SWEEP_ARMS: Tuple[Dict[str, Any], ...] = (
    {"name": "arm_003_055", "saliency_thr": 0.03, "max_keep_ratio": 0.55},
    {"name": "arm_005_045", "saliency_thr": 0.05, "max_keep_ratio": 0.45},
    {"name": "arm_008_035", "saliency_thr": 0.08, "max_keep_ratio": 0.35},
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _run_cmd(cmd: Sequence[str]) -> None:
    print(f"[{EXP_ID}] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
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


def _resolve_mvd_dir() -> Path:
    candidates = [
        WORKSPACE_ROOT / "sheet-music-converter" / "tmp-train" / "mvd",
        WORKSPACE_ROOT / "tmp-train" / "mvd",
        WORKSPACE_ROOT / "mvd",
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path(__file__).resolve().parents[2] / "tmp-train" / "mvd",
    ]
    for c in candidates:
        if (c / "uni_omr_v2.py").is_file():
            return c
    for b in BUNDLE_CANDIDATES:
        if b.is_file():
            WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
            print(f"[{EXP_ID}] Extracting {b} -> {WORKSPACE_ROOT} ...", flush=True)
            with tarfile.open(b, "r:*") as tf:
                tf.extractall(path=WORKSPACE_ROOT)
            for c in candidates:
                if (c / "uni_omr_v2.py").is_file():
                    return c
    raise FileNotFoundError(
        "Could not locate tmp-train/mvd/uni_omr_v2.py. Upload repo bundle to /content/bundle.tar.gz."
    )


def _setup_environment() -> Path:
    needed_pkgs: List[str] = []
    for mod, pkg in (
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
    ):
        try:
            __import__(mod)
        except ImportError:
            needed_pkgs.append(pkg)
    if needed_pkgs:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed_pkgs])
    mvd_dir = _resolve_mvd_dir()
    if str(mvd_dir) not in sys.path:
        sys.path.insert(0, str(mvd_dir))
    return mvd_dir


def _ensure_proxy_dataset(hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    proxy_dir = VOL_ROOT / "datasets" / "iter10_5_proxy_2k"
    train_abs = proxy_dir / "proxy_train_abs.jsonl"
    val_abs = proxy_dir / "proxy_val_abs.jsonl"
    if train_abs.is_file() and val_abs.is_file():
        return train_abs, val_abs

    proxy_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR} from {HF_REPO_ID} ...", flush=True)
    t0 = time.perf_counter()
    tar_p = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_PROXY_TAR,
        repo_type="dataset",
        token=hf_token,
        local_dir="/tmp/hf_dl",
    )
    print(f"[{EXP_ID}] Downloaded in {time.perf_counter() - t0:.1f}s -> extracting ...", flush=True)
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
        print(f"[{EXP_ID}] Prepared {len(out_lines)} rows -> {dst_path.name}", flush=True)
    return train_abs, val_abs


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
        if img_p.is_file() and beks and all(b.is_file() for b in beks):
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


def _eval_arm(
    model: Any,
    val_ds: Any,
    device: Any,
    *,
    use_bf16: bool,
    tag: str,
) -> Dict[str, Any]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from train_uni_v2 import forward_and_loss_v2
    from uni_omr import _images_to_device_float, collate_uni_batch, eval_dataset

    model.eval()
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_uni_batch,
    )

    loss_vals: List[float] = []
    ce_vals: List[float] = []
    mtp_vals: List[float] = []
    sal_vals: List[float] = []
    ink_recalls: List[float] = []
    comp_ratios: List[float] = []
    keep_fracs: List[float] = []
    n_in_vals: List[float] = []
    n_out_vals: List[float] = []

    with torch.no_grad():
        for vb in val_loader:
            v_imgs = _images_to_device_float(vb["images"], device, non_blocking=True)
            v_seq = vb["seq"].to(device, non_blocking=True)
            v_losses, v_out = forward_and_loss_v2(
                model,
                v_imgs,
                v_seq,
                kind="fullpage",
                mtp_loss_weight=0.30,
                saliency_loss_weight=0.10,
                sys_y_loss_weight=0.25,
                sys_count_loss_weight=0.15,
                staff_seg_loss_weight=0.20,
                stroke_hog_weight=0.15,
                label_smoothing=0.0,
                use_bf16=use_bf16,
                device=device,
            )
            rinfo = v_out["router_info"]
            n_in = max(1.0, float(rinfo["n_in"]))
            n_out = max(1.0, float(rinfo["n_out"]))
            k_kept = float(rinfo.get("k_kept", n_out))
            loss_vals.append(float(v_losses["loss"].item()))
            ce_vals.append(float(v_losses["ce_loss"].item()))
            mtp_vals.append(float(v_losses["mtp_loss"].item()))
            sal_vals.append(float(v_losses["saliency_loss"].item()))
            ink_recalls.append(float(rinfo.get("ink_recall", 1.0)))
            comp_ratios.append(n_in / n_out)
            keep_fracs.append(k_kept / n_in)
            n_in_vals.append(n_in)
            n_out_vals.append(n_out)

    val_gen = eval_dataset(
        model,
        val_ds,
        device=device,
        max_len=VAL_EVAL_MAX_LEN,
        batch_size=4,
        use_bf16=use_bf16,
    )
    tf = {
        "n_pages": len(loss_vals),
        "val_loss": round(float(np.mean(loss_vals)), 5),
        "val_ce": round(float(np.mean(ce_vals)), 5),
        "val_mtp": round(float(np.mean(mtp_vals)), 5),
        "val_saliency": round(float(np.mean(sal_vals)), 5),
        "ink_recall_mean": round(float(np.mean(ink_recalls)), 5),
        "ink_recall_p10": round(float(np.percentile(ink_recalls, 10)), 5),
        "ink_recall_min": round(float(np.min(ink_recalls)), 5),
        "compression_x_mean": round(float(np.mean(comp_ratios)), 4),
        "compression_x_p50": round(float(np.percentile(comp_ratios, 50)), 4),
        "compression_x_p90": round(float(np.percentile(comp_ratios, 90)), 4),
        "keep_fraction_mean": round(float(np.mean(keep_fracs)), 5),
        "mean_n_in": round(float(np.mean(n_in_vals)), 1),
        "mean_n_out": round(float(np.mean(n_out_vals)), 1),
    }
    print(
        f"  [{tag}] val_ce={tf['val_ce']:.4f} | ink_rec={tf['ink_recall_mean']:.4f} "
        f"(p10={tf['ink_recall_p10']:.4f}) | comp={tf['compression_x_mean']:.2f}x "
        f"(keep={tf['keep_fraction_mean']:.3f}) | F1={val_gen['token_f1']:.4f} "
        f"SER={val_gen['ser']:.4f} tok/s={val_gen['tokens_per_sec']:.1f}",
        flush=True,
    )
    return {"teacher_forced": tf, "generation": val_gen}


def _train_and_eval_arm(
    arm_cfg: Dict[str, Any],
    train_pairs: List[Tuple[Path, List[Path], str]],
    val_pairs: List[Tuple[Path, List[Path], str]],
    device: Any,
    *,
    use_bf16: bool,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    import numpy as np
    import torch
    from train_uni_v2 import (
        atomic_save_v2_latest,
        build_optimizer_v2,
        forward_and_loss_v2,
    )
    from uni_omr import (
        AsyncThreadPrefetcher,
        GPUEmaShadow,
        StratifiedFullPageSampler,
        UniFullPageDataset,
        _images_to_device_float,
        _lr_for_step,
        collate_uni_batch,
    )
    from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net

    arm_name = str(arm_cfg["name"])
    saliency_thr = float(arm_cfg["saliency_thr"])
    max_keep_ratio = float(arm_cfg["max_keep_ratio"])

    arm_dir = VOL_ROOT / "ckpts" / EXP_ID / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached_path = arm_dir / "arm_result.json"
    if cached_path.is_file():
        print(f"[{arm_name}] Loading cached result from {cached_path}", flush=True)
        return json.loads(cached_path.read_text(encoding="utf-8")), None

    cfg = UniOMRv2Config.small_45m(
        grad_checkpoint=True,
        dropout=0.1,
        mixstyle_prob=0.3,
        tok_dropout=0.05,
        keep_ratio=0.40,
        dynamic_routing=True,
        min_keep_ratio=0.15,
        max_keep_ratio=max_keep_ratio,
        saliency_thr=saliency_thr,
        max_mem_tokens=3072,
        mtp_k=4,
    )

    _seed_everything(SEED)
    model = UniOMRv2Net(cfg).to(device)
    param_summary = model.param_summary()
    print(
        f"\n[{arm_name}] Starting | saliency_thr={saliency_thr} max_keep_ratio={max_keep_ratio} | "
        f"params={param_summary['total_params']:,} | steps={TRAIN_STEPS}",
        flush=True,
    )

    _seed_everything(SEED + 101)
    fp_sampler = StratifiedFullPageSampler(train_pairs)
    train_ds = UniFullPageDataset(
        train_pairs,
        fullpage_max_dim=FULLPAGE_MAX_DIM,
        max_seq_len=MAX_SEQ_LEN,
        augment=True,
        aug_prob=0.75,
        multi_scale=True,
        return_uint8=True,
    )
    val_ds = UniFullPageDataset(
        val_pairs,
        fullpage_max_dim=FULLPAGE_MAX_DIM,
        max_seq_len=MAX_SEQ_LEN,
        augment=False,
        multi_scale=False,
        return_uint8=True,
    )
    prefetcher = AsyncThreadPrefetcher(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=False,
        max_workers=4,
        prefetch_batches=3,
        collate_fn=collate_uni_batch,
        seed=SEED + 303,
        stratified_sampler=fp_sampler,
    )

    opt = build_optimizer_v2(
        model,
        optimizer_name="muon",
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device_type=device.type,
    )
    gpu_ema = GPUEmaShadow(model, decay=0.997, update_every=5)

    start_step = 1
    latest_ckpt = arm_dir / "uni_omr_v2_latest.pt"
    latest_ema = arm_dir / "uni_omr_v2_ema_latest.pt"
    if latest_ckpt.is_file():
        try:
            ck = torch.load(str(latest_ckpt), map_location="cpu", weights_only=True)
            saved_step = int(ck.get("step", 0))
            if 0 < saved_step < TRAIN_STEPS:
                model.load_state_dict(ck["model"], strict=True)
                start_step = saved_step + 1
                if latest_ema.is_file():
                    eck = torch.load(str(latest_ema), map_location="cpu", weights_only=True)
                    gpu_ema = GPUEmaShadow(model, decay=0.997, update_every=5)
                    for k, v in eck["model"].items():
                        if k in gpu_ema.shadow:
                            gpu_ema.shadow[k].copy_(v.to(device))
                print(f"[{arm_name}] Resumed from step {saved_step} -> {start_step}", flush=True)
        except Exception as exc:
            print(f"[{arm_name}] WARN resume skipped: {exc}", flush=True)

    warmup_steps = max(20, int(round(TRAIN_STEPS * 0.10)))
    save_every_sec = max(60.0, SAVE_EVERY_MINUTES * 60.0)
    last_save_ts = time.perf_counter()
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step_times: List[float] = []
    history: List[Dict[str, Any]] = []
    inv_ga = 1.0 / float(GRAD_ACCUM)

    model.train()
    try:
        for step in range(start_step, TRAIN_STEPS + 1):
            ts0 = time.perf_counter()
            cur_lr = _lr_for_step(
                step,
                TRAIN_STEPS,
                LR,
                warmup_steps,
                min_lr_ratio=0.05,
                schedule="wsd",
                wsd_decay_frac=0.25,
            )
            opt.set_lr(cur_lr / max(1e-12, LR))
            opt.zero_grad(set_to_none=True)

            acc = {"loss": 0.0, "ce_loss": 0.0, "mtp_loss": 0.0, "saliency_loss": 0.0}
            last_n_in = 0
            last_n_out = 0
            for _ in range(GRAD_ACCUM):
                b = next(prefetcher)
                images = _images_to_device_float(b["images"], device, non_blocking=True)
                seq = b["seq"].to(device, non_blocking=True)
                losses, out = forward_and_loss_v2(
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
                (losses["loss"] * inv_ga).backward()
                for k in acc:
                    acc[k] += float(losses[k].detach().item()) * inv_ga
                rinfo = out["router_info"]
                last_n_in = int(rinfo["n_in"])
                last_n_out = int(rinfo["n_out"])

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            gpu_ema.update(model, step)
            dt = time.perf_counter() - ts0
            step_times.append(dt)

            if step == 1 or step % 50 == 0 or step == TRAIN_STEPS:
                row = {
                    "step": step,
                    "lr": round(cur_lr, 8),
                    "n_in": last_n_in,
                    "n_out": last_n_out,
                    "step_sec": round(dt, 3),
                    **{k: round(v, 5) for k, v in acc.items()},
                }
                history.append(row)
                print(
                    f"[{arm_name}] step {step:3d}/{TRAIN_STEPS} | lr={cur_lr:.2e} | "
                    f"loss={acc['loss']:.4f} (ce={acc['ce_loss']:.4f}, sal={acc['saliency_loss']:.4f}) | "
                    f"toks {last_n_in}->{last_n_out} | {dt:.2f}s/step",
                    flush=True,
                )

            if (time.perf_counter() - last_save_ts) >= save_every_sec:
                last_save_ts = time.perf_counter()
                atomic_save_v2_latest(arm_dir, model, cfg, step=step, gpu_ema=gpu_ema)
                print(f"  [{arm_name}] 5-min atomic checkpoint saved @ step {step}", flush=True)
    finally:
        prefetcher.close()

    train_secs = time.perf_counter() - t0
    peak_vram_gb = (
        round(float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 3), 3)
        if device.type == "cuda"
        else 0.0
    )
    atomic_save_v2_latest(arm_dir, model, cfg, step=TRAIN_STEPS, gpu_ema=gpu_ema)

    raw_eval = _eval_arm(model, val_ds, device, use_bf16=use_bf16, tag=f"{arm_name} raw")
    with gpu_ema.swap_into(model):
        ema_eval = _eval_arm(model, val_ds, device, use_bf16=use_bf16, tag=f"{arm_name} EMA")

        cross_matrix: Optional[Dict[str, Any]] = None
        if arm_name == "arm_005_045":
            cross_matrix = {}
            router = model.encoder.router
            orig_thr, orig_max = router.saliency_thr, router.max_keep_ratio
            try:
                for probe in SWEEP_ARMS:
                    router.saliency_thr = float(probe["saliency_thr"])
                    router.max_keep_ratio = float(probe["max_keep_ratio"])
                    cross_matrix[probe["name"]] = _eval_arm(
                        model,
                        val_ds,
                        device,
                        use_bf16=use_bf16,
                        tag=f"zero-shot {probe['name']}",
                    )
            finally:
                router.saliency_thr = orig_thr
                router.max_keep_ratio = orig_max

    result = {
        "arm": arm_name,
        "saliency_thr": saliency_thr,
        "max_keep_ratio": max_keep_ratio,
        "train_steps": TRAIN_STEPS,
        "train_secs": round(train_secs, 2),
        "mean_step_sec": round(float(np.mean(step_times)) if step_times else 0.0, 3),
        "peak_vram_gb": peak_vram_gb,
        "optimizer_mode": opt.mode,
        "params": param_summary,
        "raw_eval": raw_eval,
        "ema_eval": ema_eval,
        "train_history": history,
    }
    _atomic_write_json(cached_path, result)

    del model, opt, gpu_ema, train_ds, val_ds
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, cross_matrix


def main() -> None:
    try:
        hf_token = _read_and_purge_hf_token()
        _setup_environment()

        import torch
        from huggingface_hub import HfApi

        t_start = time.perf_counter()
        started_at = _utc_now()

        if torch.cuda.is_available():
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
        use_bf16 = bool(torch.cuda.is_available() and cc[0] >= 8)
        print(
            f"[{EXP_ID}] Running on {gpu_name} (cc={cc[0]}.{cc[1]}, bf16={use_bf16}, "
            f"torch={torch.__version__})",
            flush=True,
        )

        train_abs, val_abs = _ensure_proxy_dataset(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)
        print(f"[{EXP_ID}] Loaded proxy shard: train={len(train_pairs)}, val={len(val_pairs)}", flush=True)

        arm_results: Dict[str, Any] = {}
        cross_threshold_matrix: Optional[Dict[str, Any]] = None
        for arm_cfg in SWEEP_ARMS:
            res, cm = _train_and_eval_arm(
                arm_cfg, train_pairs, val_pairs, device, use_bf16=use_bf16
            )
            arm_results[arm_cfg["name"]] = res
            if cm is not None:
                cross_threshold_matrix = cm

        summary_table: List[Dict[str, Any]] = []
        for arm_cfg in SWEEP_ARMS:
            name = arm_cfg["name"]
            ema = arm_results[name]["ema_eval"]
            tf = ema["teacher_forced"]
            gen = ema["generation"]
            summary_table.append(
                {
                    "arm": name,
                    "saliency_thr": arm_cfg["saliency_thr"],
                    "max_keep_ratio": arm_cfg["max_keep_ratio"],
                    "ink_recall_mean": tf["ink_recall_mean"],
                    "ink_recall_p10": tf["ink_recall_p10"],
                    "compression_x_mean": tf["compression_x_mean"],
                    "keep_fraction_mean": tf["keep_fraction_mean"],
                    "val_ce": tf["val_ce"],
                    "val_token_f1": gen["token_f1"],
                    "val_ser": gen["ser"],
                    "tokens_per_sec": gen["tokens_per_sec"],
                }
            )

        elapsed_sec = round(time.perf_counter() - t_start, 2)
        completed_at = _utc_now()
        report = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_sec": elapsed_sec,
            "platform": f"colab-{gpu_name}",
            "bf16": use_bf16,
            "dataset": {
                "hf_repo": HF_REPO_ID,
                "tar": HF_PROXY_TAR,
                "n_train": len(train_pairs),
                "n_val": len(val_pairs),
            },
            "hyperparams": {
                "preset": "small_45m",
                "steps": TRAIN_STEPS,
                "seed": SEED,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM,
                "fullpage_max_dim": FULLPAGE_MAX_DIM,
                "max_seq_len": MAX_SEQ_LEN,
            },
            "summary_ema": summary_table,
            "zero_shot_cross_threshold_matrix": cross_threshold_matrix,
            "arms": arm_results,
        }
        run_ledger = {
            "exp_id": EXP_ID,
            "wave": 5,
            "status": "COMPLETED",
            "platform": f"colab-{gpu_name}",
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_sec": elapsed_sec,
            "hf_eval_path": HF_EVAL_PATH,
            "summary_ema": summary_table,
        }

        eval_local = VOL_ROOT / "evals" / f"{EXP_ID}.json"
        run_local = VOL_ROOT / "runs" / f"{EXP_ID}.json"
        _atomic_write_json(eval_local, report)
        _atomic_write_json(run_local, run_ledger)

        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(eval_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"eval({EXP_ID}): upload router saliency_thr x max_keep_ratio sweep",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload run ledger",
        )
        print(f"\n[{EXP_ID}] Uploaded {HF_EVAL_PATH} & {HF_RUN_PATH} in {elapsed_sec}s!", flush=True)
        print(json.dumps(summary_table, indent=2), flush=True)
    finally:
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
