#!/usr/bin/env python3
"""Google Colab L4 runner for Overnight Exp 25: CASS Scheduled Sampling Sweep.

Experiment ID: 2026-11-26T01-38-ov-25-cass-prob-sweep
Controlled 3-Arm Sweep on UniOMR v2 small_45m (BF16, seed=42, lr=3e-4, 700 steps/arm, dim=1280):
  - Arm 1: cass_prob=0.00 (Control, pure teacher forcing, tok_dropout=0.0)
  - Arm 2: cass_prob=0.10 (Fixed no-tok-dropout CASS)
  - Arm 3: cass_prob=0.25 (Fixed no-tok-dropout CASS)

Measures exposure-bias reduction on proxy_val (64 pages, eval_max_len=1024) with
length-bucketed Token F1 / SER (short <= 256, medium 257..512, long > 512),
pred/ref length ratio, and early-EOS rate.

Uploads results to Hugging Face Private Dataset Hub (gregoreesmaa/sheet-music-converter-omr):
  - evals/2026-11-26T01-38-ov-25-cass-prob-sweep.json
  - runs/2026-11-26T01-38-ov-25-cass-prob-sweep.json
"""

from __future__ import annotations

import contextlib
import difflib
import gc
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T01-38-ov-25-cass-prob-sweep"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

WORKSPACE_ROOT = Path("/workspace")
VOL_ROOT = Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")
BUNDLE_CANDIDATES = (
    Path("/content/bundle.tar.gz"),
    Path("/content/repo_bundle.tar.gz"),
)

# Sweep configuration
ARMS: List[Tuple[str, float]] = [
    ("cass_0.00", 0.00),
    ("cass_0.10", 0.10),
    ("cass_0.25", 0.25),
]

# Matched training hyperparameters
TRAIN_STEPS = 700
SEED = 42
LR = 3e-4
WEIGHT_DECAY = 0.04
BATCH_SIZE = 2
GRAD_ACCUM = 4
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
WARMUP_STEPS = 70
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
EVAL_MAX_LEN = 1024
SAVE_EVERY_MINUTES = 5.0


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


def _setup_environment() -> Path:
    mvd_candidates = [
        WORKSPACE_ROOT / "sheet-music-converter" / "tmp-train" / "mvd",
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path.cwd() / "tmp-train" / "mvd",
    ]
    mvd_dir: Optional[Path] = next((p for p in mvd_candidates if p.is_dir()), None)

    if mvd_dir is None:
        for bpath in BUNDLE_CANDIDATES:
            if bpath.is_file():
                WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
                print(f"[{EXP_ID}] Extracting {bpath} -> {WORKSPACE_ROOT}...", flush=True)
                with tarfile.open(bpath, "r:*") as tf:
                    tf.extractall(path=WORKSPACE_ROOT)
                mvd_dir = next((p for p in mvd_candidates if p.is_dir()), None)
                if mvd_dir is not None:
                    break

    if mvd_dir is None:
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        repo_dir = WORKSPACE_ROOT / "sheet-music-converter"
        _run_cmd([
            "git", "clone", "--depth", "1",
            "https://github.com/gregoreesmaa/sheet-music-converter.git",
            str(repo_dir),
        ])
        mvd_dir = repo_dir / "tmp-train" / "mvd"

    if not mvd_dir.is_dir():
        raise FileNotFoundError(f"Could not locate tmp-train/mvd (checked {mvd_candidates})")

    needed_pkgs = []
    for mod, pkg in [
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            needed_pkgs.append(pkg)
    if needed_pkgs:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed_pkgs])

    return mvd_dir


def _ensure_proxy_dataset(hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_root = VOL_ROOT / "datasets"
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


def _seq_f1_and_edits(pred: Sequence[str], ref: Sequence[str]) -> Tuple[float, int]:
    if not pred and not ref:
        return 1.0, 0
    if not pred or not ref:
        return 0.0, max(len(pred), len(ref))
    sm = difflib.SequenceMatcher(None, ref, pred, autojunk=False)
    matches = sum(b.size for b in sm.get_matching_blocks())
    f1 = (2.0 * matches) / float(len(pred) + len(ref))
    edits = max(len(ref), len(pred)) - matches
    return f1, edits


def _eval_long_page_exposure_bias(
    model: Any,
    val_pairs: List[Tuple[Path, List[Path], str]],
    device: "torch.device",
    *,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Evaluate overall + length-bucketed proxy_val metrics to measure exposure bias."""
    import torch
    from torch.utils.data import DataLoader
    import train_uni_v2 as tv2
    from uni_omr import (
        UniFullPageDataset,
        _images_to_device_float,
        collate_uni_batch,
        decode_uni,
    )

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
        pin_memory=(device.type == "cuda"),
    )

    sum_loss = 0.0
    sum_ce = 0.0
    n_samples = 0
    sys_exact = 0
    early_eos = 0
    len_ratios: List[float] = []

    buckets: Dict[str, Dict[str, Any]] = {
        "short_le256": {"f1": [], "edits": 0, "ref_len": 0, "len_ratios": [], "early_eos": 0, "n": 0},
        "medium_257_512": {"f1": [], "edits": 0, "ref_len": 0, "len_ratios": [], "early_eos": 0, "n": 0},
        "long_gt512": {"f1": [], "edits": 0, "ref_len": 0, "len_ratios": [], "early_eos": 0, "n": 0},
    }
    all_f1: List[float] = []
    all_edits = 0
    all_ref_len = 0

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else contextlib.nullcontext()
    )

    with torch.no_grad():
        for batch in loader:
            bs = int(batch["seq"].shape[0])
            images = _images_to_device_float(batch["images"], device, non_blocking=True)
            seq = batch["seq"].to(device, non_blocking=True)

            losses, _ = tv2.forward_and_loss_v2(
                model,
                images,
                seq,
                cass_prob=0.0,
                kind="fullpage",
                label_smoothing=0.0,
                use_bf16=use_bf16,
                device=device,
            )
            sum_loss += float(losses["loss"].item()) * bs
            sum_ce += float(losses["ce_loss"].item()) * bs
            n_samples += bs

            with amp_ctx:
                gen_ids = model.generate(images, max_len=EVAL_MAX_LEN)

            for b in range(bs):
                ref_toks = decode_uni(seq[b].tolist(), skip_special=True)
                pred_toks = decode_uni(gen_ids[b].tolist(), skip_special=True)
                f1, edits = _seq_f1_and_edits(pred_toks, ref_toks)
                ref_n = max(1, len(ref_toks))
                ratio = float(len(pred_toks)) / float(ref_n)
                is_early = int(ratio < 0.75)

                all_f1.append(f1)
                all_edits += edits
                all_ref_len += ref_n
                len_ratios.append(ratio)
                early_eos += is_early
                if ref_toks.count("<sys>") == pred_toks.count("<sys>"):
                    sys_exact += 1

                if len(ref_toks) <= 256:
                    bkey = "short_le256"
                elif len(ref_toks) <= 512:
                    bkey = "medium_257_512"
                else:
                    bkey = "long_gt512"
                bk = buckets[bkey]
                bk["f1"].append(f1)
                bk["edits"] += edits
                bk["ref_len"] += ref_n
                bk["len_ratios"].append(ratio)
                bk["early_eos"] += is_early
                bk["n"] += 1

    bucket_summary: Dict[str, Any] = {}
    for bkey, bk in buckets.items():
        bn = max(1, bk["n"])
        bucket_summary[bkey] = {
            "n_samples": bk["n"],
            "token_f1": round(sum(bk["f1"]) / max(1, len(bk["f1"])), 6) if bk["n"] > 0 else None,
            "ser": round(bk["edits"] / max(1, bk["ref_len"]), 6) if bk["n"] > 0 else None,
            "mean_len_ratio": round(sum(bk["len_ratios"]) / bn, 4) if bk["n"] > 0 else None,
            "early_eos_rate": round(bk["early_eos"] / bn, 4) if bk["n"] > 0 else None,
        }

    return {
        "n_samples": n_samples,
        "val_loss": round(sum_loss / max(1, n_samples), 6),
        "val_ce_loss": round(sum_ce / max(1, n_samples), 6),
        "val_token_f1": round(sum(all_f1) / max(1, len(all_f1)), 6),
        "val_ser": round(all_edits / max(1, all_ref_len), 6),
        "val_sys_count_exact": round(sys_exact / max(1, n_samples), 4),
        "mean_len_ratio": round(sum(len_ratios) / max(1, len(len_ratios)), 4),
        "early_eos_rate": round(early_eos / max(1, n_samples), 4),
        "length_buckets": bucket_summary,
    }


def _train_and_eval_arm(
    arm_name: str,
    cass_prob: float,
    train_pairs: List[Tuple[Path, List[Path], str]],
    val_pairs: List[Tuple[Path, List[Path], str]],
    device: "torch.device",
    *,
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

    _seed_everything(SEED)
    cfg = UniOMRv2Config.small_45m(grad_checkpoint=True, tok_dropout=0.0)
    model = UniOMRv2Net(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"\n[{arm_name}] Starting | cass_prob={cass_prob:.2f} | tok_dropout=0.0 | "
        f"params={total_params:,} | steps={TRAIN_STEPS}",
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
    train_ds = UniFullPageDataset(
        train_pairs,
        fullpage_max_dim=FULLPAGE_MAX_DIM,
        max_seq_len=MAX_SEQ_LEN,
        augment=True,
        return_uint8=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
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

    history: List[Dict[str, Any]] = []
    buf_loss: List[float] = []
    buf_ce: List[float] = []
    buf_ms: List[float] = []

    for step in range(1, TRAIN_STEPS + 1):
        t_step0 = time.perf_counter()
        cur_lr = _wsd_lr(step, TRAIN_STEPS, LR)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_ce = 0.0

        for _ in range(GRAD_ACCUM):
            batch = next(train_iter)
            images = _images_to_device_float(batch["images"], device, non_blocking=True)
            seq = batch["seq"].to(device, non_blocking=True)
            losses, _ = tv2.forward_and_loss_v2(
                model,
                images,
                seq,
                cass_prob=cass_prob,
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        gpu_ema.update(model, step)

        buf_loss.append(step_loss)
        buf_ce.append(step_ce)
        buf_ms.append((time.perf_counter() - t_step0) * 1000.0)

        if step % 100 == 0 or step == TRAIN_STEPS:
            rec = {
                "step": step,
                "lr": round(cur_lr, 8),
                "loss_mean": round(float(np.mean(buf_loss)), 5),
                "ce_mean": round(float(np.mean(buf_ce)), 5),
                "step_ms_mean": round(float(np.mean(buf_ms)), 1),
            }
            history.append(rec)
            print(
                f"  [{arm_name}] step {step:4d}/{TRAIN_STEPS} | lr={cur_lr:.2e} | "
                f"loss={rec['loss_mean']:.4f} ce={rec['ce_mean']:.4f} | {rec['step_ms_mean']:.0f} ms/step",
                flush=True,
            )
            buf_loss.clear()
            buf_ce.clear()
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

    # Evaluate EMA weights
    ema_sd = gpu_ema.cpu_state_dict(model)
    model.load_state_dict({k: v.to(device) for k, v in ema_sd.items()}, strict=True)
    model.eval()

    t_eval0 = time.perf_counter()
    val_eval = _eval_long_page_exposure_bias(model, val_pairs, device, use_bf16=use_bf16)
    eval_secs = time.perf_counter() - t_eval0

    long_f1 = val_eval["length_buckets"]["long_gt512"]["token_f1"]
    print(
        f"  [{arm_name}] DONE | val_F1={val_eval['val_token_f1']:.4f} SER={val_eval['val_ser']:.4f} "
        f"long_F1={long_f1} len_ratio={val_eval['mean_len_ratio']:.3f} early_eos={val_eval['early_eos_rate']:.3f}",
        flush=True,
    )

    result = {
        "arm": arm_name,
        "cass_prob": cass_prob,
        "total_params": total_params,
        "train_steps": TRAIN_STEPS,
        "train_secs": round(train_secs, 2),
        "eval_secs": round(eval_secs, 2),
        "mean_step_ms": round(train_secs * 1000.0 / float(TRAIN_STEPS), 1),
        "peak_vram_gb": peak_vram_gb,
        "history_per_100": history,
        "proxy_val_eval": val_eval,
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
        mvd_dir = _setup_environment()
        if str(mvd_dir) not in sys.path:
            sys.path.insert(0, str(mvd_dir))

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
            f"[{EXP_ID}] Running on {gpu_name} (cc={cc[0]}.{cc[1]}, bf16={use_bf16}, torch={torch.__version__})",
            flush=True,
        )

        train_abs, val_abs = _ensure_proxy_dataset(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)
        print(f"[{EXP_ID}] Loaded train={len(train_pairs)} val={len(val_pairs)}", flush=True)

        arm_results: Dict[str, Any] = {}
        for arm_name, cprob in ARMS:
            arm_results[arm_name] = _train_and_eval_arm(
                arm_name,
                cprob,
                train_pairs,
                val_pairs,
                device,
                use_bf16=use_bf16,
            )

        best_arm = max(arm_results.values(), key=lambda r: float(r["proxy_val_eval"]["val_token_f1"]))
        ctrl = arm_results["cass_0.00"]["proxy_val_eval"]
        summary = {
            "best_arm": best_arm["arm"],
            "best_cass_prob": best_arm["cass_prob"],
            "best_val_token_f1": best_arm["proxy_val_eval"]["val_token_f1"],
            "delta_f1_vs_ctrl": round(
                float(best_arm["proxy_val_eval"]["val_token_f1"]) - float(ctrl["val_token_f1"]), 6
            ),
            "per_arm": {
                k: {
                    "cass_prob": v["cass_prob"],
                    "val_token_f1": v["proxy_val_eval"]["val_token_f1"],
                    "val_ser": v["proxy_val_eval"]["val_ser"],
                    "long_gt512_f1": v["proxy_val_eval"]["length_buckets"]["long_gt512"]["token_f1"],
                    "mean_len_ratio": v["proxy_val_eval"]["mean_len_ratio"],
                    "early_eos_rate": v["proxy_val_eval"]["early_eos_rate"],
                    "mean_step_ms": v["mean_step_ms"],
                }
                for k, v in arm_results.items()
            },
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
                "tok_dropout": 0.0,
                "steps_per_arm": TRAIN_STEPS,
                "seed": SEED,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM,
                "fullpage_max_dim": FULLPAGE_MAX_DIM,
                "max_seq_len": MAX_SEQ_LEN,
                "eval_max_len": EVAL_MAX_LEN,
            },
            "arms": arm_results,
            "summary": summary,
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
                "summary": summary,
            },
        )

        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(eval_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"eval({EXP_ID}): upload CASS probability sweep results",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload CASS probability sweep run ledger",
        )
        print(f"\n[{EXP_ID}] UPLOADED {HF_EVAL_PATH} & {HF_RUN_PATH} TO HF!", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
