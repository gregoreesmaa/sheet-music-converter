"""Experiment 4 (Modal L4): 1-Variable Ablation — Native Muon+AdamW vs Pure Fused AdamW.

Isolates ``--optimizer muon`` vs ``--optimizer adamw`` on ``UniOMR v2 small_45m``
(1,200 steps each on ``datasets/iter10_5_proxy_2k.tar``, identical seed=42,
identical B=2 GA=4 WSD schedule, BF16 native on Modal L4 24 GB).

Launch detached from repo root:
    .\\.venv\\Scripts\\modal.exe run --detach experiments/2026-11-26T00-54-ablation-muon-vs-adamw/run_modal.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import modal

EXP_ID = "2026-11-26T00-54-ablation-muon-vs-adamw"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
DATASET_TAR_PATH = "datasets/iter10_5_proxy_2k.tar"
BASELINE_SHA = "9d73170"

app = modal.App("exp4-ablation-muon-vs-adamw")
vol = modal.Volume.from_name("omr-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "tar", "libglib2.0-0", "libgl1")
    .pip_install(
        "torch>=2.10.0",
        "torchvision",
        "huggingface_hub>=0.26",
        "pillow",
        "numpy",
        "opencv-python-headless",
        "tqdm",
    )
    .add_local_dir("tmp-train/mvd", remote_path="/workspace/mvd")
)


def _ensure_proxy_dataset(vol_root: Path) -> Tuple[Path, Path]:
    """Download and unpack iter10_5_proxy_2k.tar into Modal Volume once."""
    from huggingface_hub import hf_hub_download

    ds_dir = vol_root / "datasets" / "iter10_5_proxy_2k"
    sentinel = ds_dir / ".unpacked_ok"
    train_abs = ds_dir / "proxy_train_abs.jsonl"
    val_abs = ds_dir / "proxy_val_abs.jsonl"

    if sentinel.is_file() and train_abs.is_file() and val_abs.is_file():
        print(f"[dataset] Using cached proxy dataset at {ds_dir}", flush=True)
        return train_abs, val_abs

    ds_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    print(f"[dataset] Downloading {DATASET_TAR_PATH} from {HF_REPO_ID}...", flush=True)
    t0 = time.perf_counter()
    local_tar = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=DATASET_TAR_PATH,
        repo_type="dataset",
        token=token,
        local_dir="/tmp/hf_dl",
    )
    print(f"[dataset] Downloaded in {time.perf_counter() - t0:.1f}s -> extracting...", flush=True)
    t1 = time.perf_counter()
    with tarfile.open(local_tar, "r") as tf:
        tf.extractall(path=ds_dir)
    try:
        os.remove(local_tar)
    except OSError:
        pass
    print(f"[dataset] Extracted in {time.perf_counter() - t1:.1f}s", flush=True)

    for src_name, dst_path in (("proxy_train.jsonl", train_abs), ("proxy_val.jsonl", val_abs)):
        src_path = ds_dir / src_name
        if not src_path.is_file():
            raise FileNotFoundError(f"Expected {src_path} inside {DATASET_TAR_PATH}")
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
        print(f"[dataset] Wrote {len(out_lines)} abs rows -> {dst_path.name}", flush=True)

    sentinel.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    vol.commit()
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


def _eval_proxy_val(
    ckpt_path: Path,
    val_manifest: Path,
    device: "torch.device",
    *,
    fullpage_max_dim: int = 1280,
    max_seq_len: int = 2048,
    eval_max_len: int = 512,
) -> Dict[str, Any]:
    """Evaluate a saved UniOMRv2 checkpoint on proxy_val.jsonl: CE, MTP, Total Loss, and greedy SER/F1."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    import train_uni_v2 as tv2
    from uni_omr import (
        UniFullPageDataset,
        _images_to_device_float,
        collate_uni_batch,
        eval_dataset,
    )
    from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net

    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    raw_cfg = ck.get("cfg") or ck.get("config") or {}
    valid = set(UniOMRv2Config.__dataclass_fields__.keys())
    cfg_kwargs = {k: v for k, v in dict(raw_cfg).items() if k in valid}
    # Convert lists back to tuples if serialized as JSON/lists
    for tk in ("backbone_channels", "backbone_blocks", "deepstack_layers"):
        if tk in cfg_kwargs and isinstance(cfg_kwargs[tk], list):
            cfg_kwargs[tk] = tuple(cfg_kwargs[tk])
    cfg = UniOMRv2Config(**cfg_kwargs)
    model = UniOMRv2Net(cfg).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    val_pairs = _load_manifest_pairs(val_manifest)
    val_ds = UniFullPageDataset(
        val_pairs,
        fullpage_max_dim=fullpage_max_dim,
        max_seq_len=max_seq_len,
        augment=False,
        return_uint8=True,
    )

    # 1. Teacher-forced CE / MTP / Stroke-HOG / Total loss over all 64 val pages
    loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_uni_batch,
        pin_memory=True,
    )
    sum_loss = 0.0
    sum_ce = 0.0
    sum_mtp = 0.0
    sum_hog = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            bs = int(batch["seq"].shape[0])
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
                use_bf16=True,
                device=device,
            )
            sum_loss += float(losses["loss"].item()) * bs
            sum_ce += float(losses["ce_loss"].item()) * bs
            sum_mtp += float(losses["mtp_loss"].item()) * bs
            sum_hog += float(losses["stroke_hog_loss"].item()) * bs
            n_samples += bs

    # 2. KV-cached greedy generation SER / Token F1 / Sys-Count Exact
    gen_metrics = eval_dataset(
        model,
        val_ds,
        device=device,
        max_len=eval_max_len,
        batch_size=2,
        use_bf16=True,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "n_val_samples": n_samples,
        "val_loss": round(sum_loss / max(1, n_samples), 6),
        "val_ce_loss": round(sum_ce / max(1, n_samples), 6),
        "val_mtp_loss": round(sum_mtp / max(1, n_samples), 6),
        "val_stroke_hog_loss": round(sum_hog / max(1, n_samples), 6),
        "val_ser": gen_metrics["ser"],
        "val_mean_sample_ser": gen_metrics["mean_sample_ser"],
        "val_token_f1": gen_metrics["token_f1"],
        "val_sys_count_exact": gen_metrics["sys_count_exact"],
        "val_tokens_per_sec": gen_metrics["tokens_per_sec"],
        "val_mean_compression_x": gen_metrics["mean_compression_x"],
        "eval_max_len": eval_max_len,
    }


def _summarize_per_100(
    history: List[Dict[str, Any]],
    grad_norms: List[float],
    step_times_ms: List[float],
) -> List[Dict[str, Any]]:
    """Aggregate per-100-step loss, CE, MTP, grad_norm, and step_time_ms."""
    import numpy as np

    buckets: List[Dict[str, Any]] = []
    n = len(history)
    for start in range(0, n, 100):
        end = min(n, start + 100)
        h_slice = history[start:end]
        g_slice = grad_norms[start:end]
        t_slice = step_times_ms[start:end]
        buckets.append(
            {
                "step_start": int(h_slice[0]["step"]),
                "step_end": int(h_slice[-1]["step"]),
                "lr_end": round(float(h_slice[-1]["lr"]), 8),
                "loss_mean": round(float(np.mean([r["loss"] for r in h_slice])), 5),
                "ce_loss_mean": round(float(np.mean([r["ce_loss"] for r in h_slice])), 5),
                "mtp_loss_mean": round(float(np.mean([r["mtp_loss"] for r in h_slice])), 5),
                "stroke_hog_loss_mean": round(float(np.mean([r["stroke_hog_loss"] for r in h_slice])), 5),
                "grad_norm_mean": round(float(np.mean(g_slice)) if g_slice else 0.0, 5),
                "grad_norm_p95": round(float(np.percentile(g_slice, 95)) if g_slice else 0.0, 5),
                "step_time_ms_mean": round(float(np.mean(t_slice)) if t_slice else 0.0, 2),
                "steps_per_sec": round(1000.0 / max(1e-3, float(np.mean(t_slice))) if t_slice else 0.0, 3),
            }
        )
    return buckets


def _run_arm(
    optimizer_name: str,
    train_manifest: Path,
    val_manifest: Path,
    out_root: Path,
    *,
    steps: int = 1200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run one isolated arm (muon or adamw) with identical seed, data, batch, and WSD schedule."""
    import torch

    import train_uni_v2 as tv2

    arm_dir = out_root / optimizer_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    arm_done_json = arm_dir / "arm_result.json"

    if arm_done_json.is_file():
        print(f"[{optimizer_name}] Found completed arm_result.json; loading cached result.", flush=True)
        return json.loads(arm_done_json.read_text(encoding="utf-8"))

    # Instrument clip_grad_norm_ to record per-step grad_norm & step_time_ms without altering behavior
    grad_norms: List[float] = []
    step_times_ms: List[float] = []
    last_t = [time.perf_counter()]

    orig_clip = torch.nn.utils.clip_grad_norm_

    def _wrapped_clip(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
        gn = orig_clip(
            parameters,
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )
        now = time.perf_counter()
        step_times_ms.append((now - last_t[0]) * 1000.0)
        last_t[0] = now
        try:
            grad_norms.append(float(gn.detach().item()))
        except Exception:
            grad_norms.append(float(gn))
        return gn

    orig_save_latest = tv2.atomic_save_v2_latest

    def _wrapped_save_latest(*a, **kw):
        res = orig_save_latest(*a, **kw)
        vol.commit()
        print(f"  [{optimizer_name}] vol.commit() after autosave", flush=True)
        return res

    cli_argv = [
        "--preset", "small_45m",
        "--optimizer", optimizer_name,
        "--steps", str(steps),
        "--seed", str(seed),
        "--batch-size", "2",
        "--grad-accum", "4",
        "--curriculum", "fullpage",
        "--fullpage-manifest", str(train_manifest),
        "--stratified-fullpage",
        "--fullpage-max-dim", "1280",
        "--max-seq-len", "2048",
        "--multi-scale",
        "--augment",
        "--aug-prob", "0.50",
        "--lr-schedule", "wsd",
        "--lr", "3e-4",
        "--warmup-steps", "120",
        "--wsd-decay-frac", "0.25",
        "--min-lr-ratio", "0.05",
        "--grad-checkpoint",
        "--channels-last",
        "--bf16",
        "--gpu-ema-decay", "0.997",
        "--gpu-ema-every", "5",
        "--save-every-minutes", "5.0",
        "--val-count", "0",
        "--eval-every", "0",
        "--log-every", "50",
        "--num-workers", "0",
        "--prefetch-threads", "4",
        "--device", "cuda",
        "--out-dir", str(arm_dir),
    ]

    print(f"\n==================== ARM: {optimizer_name.upper()} ====================", flush=True)
    print(f"[{optimizer_name}] CLI: {' '.join(cli_argv)}", flush=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    parser = tv2.build_arg_parser()
    args = parser.parse_args(cli_argv)

    torch.nn.utils.clip_grad_norm_ = _wrapped_clip
    tv2.atomic_save_v2_latest = _wrapped_save_latest
    t_start = time.perf_counter()
    last_t[0] = t_start
    try:
        train_res = tv2.run_train_v2(args)
    finally:
        torch.nn.utils.clip_grad_norm_ = orig_clip
        tv2.atomic_save_v2_latest = orig_save_latest

    wall_time_s = time.perf_counter() - t_start
    peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

    expected_mode = "muon+adamw" if optimizer_name == "muon" else "adamw"
    actual_mode = str(train_res.get("optimizer_mode", ""))
    if actual_mode != expected_mode:
        raise RuntimeError(
            f"Optimizer isolation violated! Expected {expected_mode!r}, got {actual_mode!r}"
        )

    per_100 = _summarize_per_100(train_res["history"], grad_norms, step_times_ms)
    print(f"[{optimizer_name}] Per-100-step summary:", flush=True)
    for b in per_100:
        print(
            f"  steps {b['step_start']:4d}-{b['step_end']:4d} | loss={b['loss_mean']:.4f} "
            f"ce={b['ce_loss_mean']:.4f} mtp={b['mtp_loss_mean']:.4f} "
            f"gnorm={b['grad_norm_mean']:.3f} | {b['step_time_ms_mean']:.1f} ms/step",
            flush=True,
        )

    # Evaluate BOTH raw latest and EMA latest on all 64 proxy_val.jsonl pages
    device = torch.device("cuda")
    raw_ckpt = arm_dir / "uni_omr_v2_latest.pt"
    ema_ckpt = arm_dir / "uni_omr_v2_ema_latest.pt"
    print(f"[{optimizer_name}] Evaluating raw checkpoint on proxy_val.jsonl...", flush=True)
    val_raw = _eval_proxy_val(raw_ckpt, val_manifest, device)
    print(f"[{optimizer_name}] Raw proxy_val: {json.dumps(val_raw)}", flush=True)

    print(f"[{optimizer_name}] Evaluating EMA checkpoint on proxy_val.jsonl...", flush=True)
    val_ema = _eval_proxy_val(ema_ckpt, val_manifest, device)
    print(f"[{optimizer_name}] EMA proxy_val: {json.dumps(val_ema)}", flush=True)

    arm_payload = {
        "optimizer": optimizer_name,
        "optimizer_mode": actual_mode,
        "steps": int(steps),
        "seed": int(seed),
        "cli_argv": cli_argv,
        "wall_time_s": round(wall_time_s, 2),
        "steps_per_sec": round(steps / max(1e-6, wall_time_s), 4),
        "peak_vram_alloc_gb": round(peak_alloc_gb, 3),
        "peak_vram_reserved_gb": round(peak_reserved_gb, 3),
        "params": train_res.get("params", {}),
        "per_100_steps": per_100,
        "final_100_train_loss": per_100[-1]["loss_mean"] if per_100 else None,
        "final_100_train_ce": per_100[-1]["ce_loss_mean"] if per_100 else None,
        "final_100_train_mtp": per_100[-1]["mtp_loss_mean"] if per_100 else None,
        "proxy_val_raw": val_raw,
        "proxy_val_ema": val_ema,
    }
    arm_done_json.write_text(json.dumps(arm_payload, indent=2), encoding="utf-8")
    vol.commit()

    gc.collect()
    torch.cuda.empty_cache()
    return arm_payload


@app.function(
    gpu="L4",
    timeout=14400,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    image=image,
)
def run_ablation(steps: int = 1200, seed: int = 42) -> Dict[str, Any]:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if "/workspace/mvd" not in sys.path:
        sys.path.insert(0, "/workspace/mvd")

    import torch
    from huggingface_hub import HfApi

    # Keep CUTLASS/Flash SDPA; never force cuDNN SDPA on variable shapes.
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)

    assert torch.cuda.is_available(), "CUDA GPU required"
    cc = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name(0)
    assert cc[0] >= 8, f"Native BF16 requires Ampere+ (cc>=8), got {gpu_name} cc={cc}"
    assert hasattr(torch.optim, "Muon"), (
        f"Native torch.optim.Muon missing in torch=={torch.__version__}!"
    )

    started_at = datetime.now(timezone.utc).isoformat()
    print(
        f"[exp4] Starting {EXP_ID} | GPU={gpu_name} (cc={cc[0]}.{cc[1]}) | "
        f"torch={torch.__version__} | native_muon=True | steps={steps} | seed={seed}",
        flush=True,
    )

    vol_root = Path("/vol")
    train_manifest, val_manifest = _ensure_proxy_dataset(vol_root)
    out_root = vol_root / "ckpts" / EXP_ID
    out_root.mkdir(parents=True, exist_ok=True)

    muon_res = _run_arm("muon", train_manifest, val_manifest, out_root, steps=steps, seed=seed)
    adamw_res = _run_arm("adamw", train_manifest, val_manifest, out_root, steps=steps, seed=seed)

    def _delta(a: float, b: float) -> float:
        return round(float(a) - float(b), 6)

    comparison = {
        "delta_muon_minus_adamw": {
            "final_100_train_loss": _delta(muon_res["final_100_train_loss"], adamw_res["final_100_train_loss"]),
            "final_100_train_ce": _delta(muon_res["final_100_train_ce"], adamw_res["final_100_train_ce"]),
            "raw_val_ce": _delta(muon_res["proxy_val_raw"]["val_ce_loss"], adamw_res["proxy_val_raw"]["val_ce_loss"]),
            "raw_val_mtp": _delta(muon_res["proxy_val_raw"]["val_mtp_loss"], adamw_res["proxy_val_raw"]["val_mtp_loss"]),
            "raw_val_ser": _delta(muon_res["proxy_val_raw"]["val_ser"], adamw_res["proxy_val_raw"]["val_ser"]),
            "raw_val_token_f1": _delta(muon_res["proxy_val_raw"]["val_token_f1"], adamw_res["proxy_val_raw"]["val_token_f1"]),
            "ema_val_ce": _delta(muon_res["proxy_val_ema"]["val_ce_loss"], adamw_res["proxy_val_ema"]["val_ce_loss"]),
            "ema_val_mtp": _delta(muon_res["proxy_val_ema"]["val_mtp_loss"], adamw_res["proxy_val_ema"]["val_mtp_loss"]),
            "ema_val_ser": _delta(muon_res["proxy_val_ema"]["val_ser"], adamw_res["proxy_val_ema"]["val_ser"]),
            "ema_val_token_f1": _delta(muon_res["proxy_val_ema"]["val_token_f1"], adamw_res["proxy_val_ema"]["val_token_f1"]),
            "wall_time_s": _delta(muon_res["wall_time_s"], adamw_res["wall_time_s"]),
            "peak_vram_alloc_gb": _delta(muon_res["peak_vram_alloc_gb"], adamw_res["peak_vram_alloc_gb"]),
        },
        "winner_by_ema_val_ce": (
            "muon"
            if muon_res["proxy_val_ema"]["val_ce_loss"] < adamw_res["proxy_val_ema"]["val_ce_loss"]
            else "adamw"
        ),
        "winner_by_ema_val_ser": (
            "muon"
            if muon_res["proxy_val_ema"]["val_ser"] < adamw_res["proxy_val_ema"]["val_ser"]
            else "adamw"
        ),
    }

    finished_at = datetime.now(timezone.utc).isoformat()
    eval_payload = {
        "exp_id": EXP_ID,
        "baseline_sha": BASELINE_SHA,
        "dataset": DATASET_TAR_PATH,
        "gpu": gpu_name,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "torch_version": torch.__version__,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "steps_per_arm": steps,
        "seed": seed,
        "arms": {
            "muon": muon_res,
            "adamw": adamw_res,
        },
        "comparison": comparison,
    }

    run_ledger = {
        "exp_id": EXP_ID,
        "status": "COMPLETED",
        "platform": "modal_l4",
        "gpu": gpu_name,
        "baseline_sha": BASELINE_SHA,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "total_wall_time_s": round(muon_res["wall_time_s"] + adamw_res["wall_time_s"], 2),
        "eval_hf_path": f"evals/{EXP_ID}.json",
        "checkpoints_hf_dir": f"checkpoints/{EXP_ID}/",
        "comparison": comparison,
    }

    eval_local = out_root / f"{EXP_ID}_eval.json"
    run_local = out_root / f"{EXP_ID}_run.json"
    eval_local.write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")
    run_local.write_text(json.dumps(run_ledger, indent=2), encoding="utf-8")
    vol.commit()

    # Upload eval JSON, run ledger JSON, and both EMA checkpoints to HF Private Dataset Hub
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    uploads = [
        (eval_local, f"evals/{EXP_ID}.json"),
        (run_local, f"runs/{EXP_ID}.json"),
        (out_root / "muon" / "uni_omr_v2_ema_latest.pt", f"checkpoints/{EXP_ID}/muon_ema_latest.pt"),
        (out_root / "adamw" / "uni_omr_v2_ema_latest.pt", f"checkpoints/{EXP_ID}/adamw_ema_latest.pt"),
    ]
    for src, dst in uploads:
        if src.is_file():
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=dst,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"exp4({EXP_ID}): upload {dst}",
            )
            print(f"[hf] Uploaded {dst}", flush=True)

    print("\n==================== EXP 4 FINAL SUMMARY ====================", flush=True)
    print(json.dumps(comparison, indent=2), flush=True)
    return eval_payload


@app.local_entrypoint()
def main(steps: int = 1200, seed: int = 42) -> None:
    fc = run_ablation.spawn(steps=steps, seed=seed)
    print(f"[spawned] function_call_id={fc.object_id}", flush=True)
