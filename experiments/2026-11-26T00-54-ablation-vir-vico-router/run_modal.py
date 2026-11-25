"""Cloud Experiment 3: 1-Variable Isolation Ablation — ViRTwoTierRouter + ViCO vs InkTokenRouterV2.

Runs on Modal L4 (24 GB, native BF16) using the stratified 2,000-page proxy shard
``datasets/iter10_5_proxy_2k.tar`` from ``gregoreesmaa/sheet-music-converter-omr``.

Both arms share:
  - Model: ``UniOMRv2Net`` (``small_45m``, ~43M params)
  - Steps: 1,200 steps (effective batch = 8 full pages/step: B=2 x grad_accum=4)
  - Optimizer: native ``torch.optim.Muon`` (2D Transformer matrices) + fused ``AdamW``
  - Schedule: WSD (``lr=3e-4``, ``warmup_steps=120``, ``wsd_decay_frac=0.25``, ``min_lr_ratio=0.05``)
  - Seed: 42 (re-seeded right before dataloader creation so both arms see bit-identical batches)
  - Resolution: ``fullpage_max_dim=1280``, ``max_seq_len=1536``, ``keep_ratio=0.40``, ``max_mem_tokens=3072``

Single isolated variable:
  - Arm 1 (``baseline``): standard binary ``InkTokenRouterV2`` (``n_bg_summary=4``)
  - Arm 2 (``vir_vico``): ``ViRTwoTierRouter`` (``k_coarse=96``, ``static_buckets=True``,
    VisionZip contextual merging) + InternVL 3.5 ViCO KL consistency loss
    (``vico_weight=0.10``, ``vico_temp=1.5``, every 4th step vs dense ``keep_ratio=0.85`` teacher)

Launch detached:
  .\\.venv\\Scripts\\modal.exe run --detach experiments/2026-11-26T00-54-ablation-vir-vico-router/run_modal.py
"""

from __future__ import annotations

import json
import os
import tarfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

EXP_ID = "2026-11-26T00-54-ablation-vir-vico-router"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_DATASET_TAR = "datasets/iter10_5_proxy_2k.tar"
VOL_DATA_DIR = Path("/vol/datasets/iter10_5_proxy_2k")
VOL_CKPT_ROOT = Path("/vol/ckpts/exp3_vir_vico")

app = modal.App("exp3-ablation-vir-vico-router")
vol = modal.Volume.from_name("omr-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-secret")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "tar")
    .pip_install(
        "torch==2.14.1",
        "huggingface_hub>=0.26",
        "pillow",
        "numpy",
        "tqdm",
    )
    .add_local_dir(
        "tmp-train/mvd",
        remote_path="/workspace/mvd",
        ignore=["__pycache__", "*.pt", "*.onnx", "*.png", "*.jpg", "*.npz", "*.npy", "*.tar"],
    )
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=1800,
    volumes={"/vol": vol},
    secrets=[hf_secret],
)
def prepare_dataset() -> Dict[str, Any]:
    """Download & unpack datasets/iter10_5_proxy_2k.tar into /vol once (race-free)."""
    from huggingface_hub import hf_hub_download

    vol.reload()
    sentinel = VOL_DATA_DIR / ".unpacked_ok"
    if sentinel.is_file() and (VOL_DATA_DIR / "proxy_train.jsonl").is_file() and (VOL_DATA_DIR / "proxy_val.jsonl").is_file():
        n_train = sum(1 for _ in (VOL_DATA_DIR / "proxy_train.jsonl").open("r", encoding="utf-8"))
        n_val = sum(1 for _ in (VOL_DATA_DIR / "proxy_val.jsonl").open("r", encoding="utf-8"))
        print(f"[prepare_dataset] Cache hit at {VOL_DATA_DIR} (train={n_train}, val={n_val})", flush=True)
        return {"status": "cached", "data_dir": str(VOL_DATA_DIR), "n_train": n_train, "n_val": n_val}

    t0 = time.perf_counter()
    VOL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_tar_dir = Path("/tmp/hf_proxy_dl")
    tmp_tar_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prepare_dataset] Downloading {HF_DATASET_TAR} from {HF_REPO_ID} ...", flush=True)
    tar_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=HF_DATASET_TAR,
        local_dir=str(tmp_tar_dir),
        token=os.environ.get("HF_TOKEN"),
    )
    t_dl = time.perf_counter() - t0
    print(f"[prepare_dataset] Downloaded in {t_dl:.1f}s -> unpacking to {VOL_DATA_DIR} ...", flush=True)
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(path=str(VOL_DATA_DIR))
    try:
        Path(tar_path).unlink(missing_ok=True)
    except Exception:
        pass

    n_train = sum(1 for _ in (VOL_DATA_DIR / "proxy_train.jsonl").open("r", encoding="utf-8"))
    n_val = sum(1 for _ in (VOL_DATA_DIR / "proxy_val.jsonl").open("r", encoding="utf-8"))
    sentinel.write_text(json.dumps({"unpacked_at": _utc_now(), "n_train": n_train, "n_val": n_val}), encoding="utf-8")
    vol.commit()
    elapsed = time.perf_counter() - t0
    print(f"[prepare_dataset] Done in {elapsed:.1f}s (train={n_train}, val={n_val})", flush=True)
    return {"status": "unpacked", "data_dir": str(VOL_DATA_DIR), "n_train": n_train, "n_val": n_val, "elapsed_sec": round(elapsed, 1)}


def _load_manifest_pairs(manifest_path: Path, data_root: Path) -> List[Tuple[Path, List[Path], str]]:
    pairs: List[Tuple[Path, List[Path], str]] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        rec = json.loads(raw_line)
        img_p = data_root / rec["image"]
        beks = [data_root / b for b in rec.get("bekerns", [])]
        domain = str(rec.get("domain", "")).strip().lower()
        if img_p.is_file() and beks and all(b.is_file() for b in beks):
            pairs.append((img_p, beks, domain))
    return pairs


@app.function(
    image=image,
    gpu="L4",
    cpu=6,
    memory=24576,
    timeout=7200,
    volumes={"/vol": vol},
    secrets=[hf_secret],
)
def train_arm(
    arm: str,
    steps: int = 1200,
    seed: int = 42,
    lr: float = 3e-4,
    batch_size: int = 2,
    grad_accum: int = 4,
    fullpage_max_dim: int = 1280,
    max_seq_len: int = 1536,
    eval_every: int = 600,
    eval_max_len: int = 512,
    vico_weight: float = 0.10,
    vico_temp: float = 1.5,
    vico_every: int = 4,
    save_every_minutes: float = 5.0,
) -> Dict[str, Any]:
    """Train and evaluate one isolated arm (``baseline`` or ``vir_vico``) on Modal L4."""
    import random
    import sys
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from huggingface_hub import HfApi

    if arm not in ("baseline", "vir_vico"):
        raise ValueError(f"Unknown arm={arm!r}; expected 'baseline' or 'vir_vico'")

    sys.path.insert(0, "/workspace/mvd")
    import data
    from uni_omr import (
        PAD_ID,
        AsyncThreadPrefetcher,
        GPUEmaShadow,
        StratifiedFullPageSampler,
        UniFullPageDataset,
        _images_to_device_float,
        _lr_for_step,
        collate_uni_batch,
        eval_dataset,
    )
    from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net
    from train_uni_v2 import (
        atomic_save_checkpoint,
        atomic_save_v2_latest,
        build_optimizer_v2,
        forward_and_loss_v2,
    )
    from vir_router_v2 import ViRTwoTierRouter, compute_vico_consistency_loss

    vol.reload()
    assert torch.cuda.is_available(), "CUDA GPU is required"
    dev_name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    use_bf16 = cc_major >= 8
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(
        f"[{arm}] Starting on {dev_name} (cc={cc_major}.{cc_minor}, bf16={use_bf16}) | "
        f"steps={steps} seed={seed} lr={lr} B={batch_size}x{grad_accum} dim={fullpage_max_dim}",
        flush=True,
    )

    out_dir = VOL_CKPT_ROOT / arm
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build identical UniOMR v2 small_45m config
    cfg = UniOMRv2Config.small_45m(
        grad_checkpoint=True,
        dropout=0.1,
        mixstyle_prob=0.3,
        tok_dropout=0.05,
        keep_ratio=0.40,
        dynamic_routing=True,
        max_mem_tokens=3072,
        mtp_k=4,
        use_sys_y_aux=True,
        use_sys_count_aux=True,
        use_staff_seg_aux=True,
        use_stroke_hog_head=True,
    )

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = UniOMRv2Net(cfg)

    if arm == "vir_vico":
        model.encoder.router = ViRTwoTierRouter(
            d_model=cfg.d_model,
            keep_ratio=cfg.keep_ratio,
            k_coarse=96,
            static_buckets=True,
            use_ink_prior=cfg.use_ink_prior,
            dynamic_routing=cfg.dynamic_routing,
            min_keep_ratio=cfg.min_keep_ratio,
            max_keep_ratio=cfg.max_keep_ratio,
            saliency_thr=cfg.saliency_thr,
            max_mem_tokens=cfg.max_mem_tokens,
            router_hidden=cfg.router_hidden,
        )
        router_name = "ViRTwoTierRouter+ViCO"
    else:
        router_name = "InkTokenRouterV2"

    model = model.to(device)
    param_summary = model.param_summary()
    print(
        f"[{arm}] router={router_name} | total_params={param_summary['total_params']:,} "
        f"| router_params={param_summary['router_params']:,}",
        flush=True,
    )

    # 2. Load proxy_train.jsonl & proxy_val.jsonl (re-seed right before so batch order is bit-identical)
    train_pairs = _load_manifest_pairs(VOL_DATA_DIR / "proxy_train.jsonl", VOL_DATA_DIR)
    val_pairs = _load_manifest_pairs(VOL_DATA_DIR / "proxy_val.jsonl", VOL_DATA_DIR)
    print(f"[{arm}] Loaded proxy shard: train={len(train_pairs)} pages, val={len(val_pairs)} pages", flush=True)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    fp_strat_sampler = StratifiedFullPageSampler(train_pairs)
    train_ds = UniFullPageDataset(
        train_pairs,
        fullpage_max_dim=fullpage_max_dim,
        max_seq_len=max_seq_len,
        augment=True,
        aug_prob=0.75,
        multi_scale=True,
        return_uint8=True,
    )
    val_ds = UniFullPageDataset(
        val_pairs,
        fullpage_max_dim=fullpage_max_dim,
        max_seq_len=max_seq_len,
        augment=False,
        multi_scale=False,
        return_uint8=True,
    )
    prefetcher = AsyncThreadPrefetcher(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        max_workers=4,
        prefetch_batches=3,
        collate_fn=collate_uni_batch,
        seed=seed + 303,
        stratified_sampler=fp_strat_sampler,
    )

    # 3. Optimizer + EMA + Optional Resume from 5-min checkpoint
    opt = build_optimizer_v2(
        model,
        optimizer_name="muon",
        lr=lr,
        weight_decay=0.01,
        device_type="cuda",
    )
    gpu_ema = GPUEmaShadow(model, decay=0.997, update_every=5)
    start_step = 1
    best_val_ser = float("inf")
    best_ema_ser = float("inf")

    latest_ckpt = out_dir / "uni_omr_v2_latest.pt"
    latest_ema = out_dir / "uni_omr_v2_ema_latest.pt"
    if latest_ckpt.is_file():
        try:
            ck = torch.load(str(latest_ckpt), map_location="cpu", weights_only=True)
            saved_step = int(ck.get("step", 0))
            if 0 < saved_step < steps:
                model.load_state_dict(ck["model"], strict=True)
                start_step = saved_step + 1
                best_val_ser = float(ck.get("best_val_ser", float("inf")))
                if latest_ema.is_file():
                    eck = torch.load(str(latest_ema), map_location="cpu", weights_only=True)
                    best_ema_ser = float(eck.get("best_val_ser", float("inf")))
                    gpu_ema = GPUEmaShadow(model, decay=0.997, update_every=5)
                    for k, v in eck["model"].items():
                        if k in gpu_ema.shadow:
                            gpu_ema.shadow[k].copy_(v.to(device))
                print(f"[{arm}] Resumed from step {saved_step} -> starting at step {start_step}", flush=True)
        except Exception as e:
            print(f"[{arm}] WARN could not resume from {latest_ckpt}: {e}", flush=True)

    @torch.no_grad()
    def _eval_val_loss_and_gen(tag: str) -> Dict[str, Any]:
        model.eval()
        val_loader = DataLoader(
            val_ds,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_uni_batch,
        )
        loss_sums = {
            "val_loss": 0.0,
            "val_ce": 0.0,
            "val_mtp": 0.0,
            "val_saliency": 0.0,
            "val_stroke_hog": 0.0,
            "mean_n_in": 0.0,
            "mean_n_out": 0.0,
            "mean_ink_recall": 0.0,
        }
        n_batches = 0
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
            loss_sums["val_loss"] += float(v_losses["loss"].item())
            loss_sums["val_ce"] += float(v_losses["ce_loss"].item())
            loss_sums["val_mtp"] += float(v_losses["mtp_loss"].item())
            loss_sums["val_saliency"] += float(v_losses["saliency_loss"].item())
            loss_sums["val_stroke_hog"] += float(v_losses["stroke_hog_loss"].item())
            loss_sums["mean_n_in"] += float(rinfo["n_in"])
            loss_sums["mean_n_out"] += float(rinfo["n_out"])
            loss_sums["mean_ink_recall"] += float(rinfo.get("ink_recall", 1.0))
            n_batches += 1

        val_tf = {k: round(v / max(1, n_batches), 5) for k, v in loss_sums.items()}
        val_gen = eval_dataset(
            model,
            val_ds,
            device=device,
            max_len=eval_max_len,
            batch_size=4,
            use_bf16=use_bf16,
        )
        model.train()
        print(
            f"  [{arm} eval {tag}] val_loss={val_tf['val_loss']:.4f} ce={val_tf['val_ce']:.4f} "
            f"mtp={val_tf['val_mtp']:.4f} | SER={val_gen['ser']:.4f} F1={val_gen['token_f1']:.4f} "
            f"sys_exact={val_gen['sys_count_exact']:.3f} comp={val_gen['mean_compression_x']:.2f}x "
            f"ink_rec={val_gen['ink_recall']:.4f} tok/s={val_gen['tokens_per_sec']:.1f}",
            flush=True,
        )
        return {"teacher_forced": val_tf, "generation": val_gen}

    history: List[Dict[str, Any]] = []
    eval_history: List[Dict[str, Any]] = []
    warmup_steps = max(20, int(round(steps * 0.10)))
    save_every_sec = max(60.0, float(save_every_minutes) * 60.0)
    last_time_save = time.perf_counter()
    t_train_start = time.perf_counter()
    step_times: List[float] = []

    model.train()
    try:
        for step in range(start_step, steps + 1):
            t_step0 = time.perf_counter()
            cur_lr = _lr_for_step(
                step,
                steps,
                lr,
                warmup_steps,
                min_lr_ratio=0.05,
                schedule="wsd",
                wsd_decay_frac=0.25,
            )
            opt.set_lr(cur_lr / max(1e-12, lr))
            opt.zero_grad(set_to_none=True)

            acc = {
                "loss": 0.0,
                "ce_loss": 0.0,
                "mtp_loss": 0.0,
                "saliency_loss": 0.0,
                "stroke_hog_loss": 0.0,
                "vico_loss": 0.0,
            }
            last_n_in = 0
            last_n_out = 0
            inv_ga = 1.0 / float(grad_accum)
            do_vico = (arm == "vir_vico") and (vico_weight > 0.0) and (step % max(1, vico_every) == 0)

            for _micro in range(grad_accum):
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

                vico_val = 0.0
                if do_vico:
                    router = model.encoder.router
                    prev_static = bool(getattr(router, "static_buckets", True))
                    router.static_buckets = False
                    try:
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            dense_out = model(
                                images,
                                seq[:, :-1],
                                keep_ratio_override=0.85,
                                return_aux=False,
                                return_mtp=False,
                            )
                    finally:
                        router.static_buckets = prev_static
                    vico_loss = compute_vico_consistency_loss(
                        dense_out["logits"],
                        out["logits"],
                        temp=vico_temp,
                        valid_mask=(seq[:, 1:] != PAD_ID),
                        symmetric=True,
                        detach_dense=True,
                    )
                    losses["loss"] = losses["loss"] + float(vico_weight) * vico_loss
                    vico_val = float(vico_loss.detach().item())

                (losses["loss"] * inv_ga).backward()
                acc["loss"] += float(losses["loss"].detach().item()) * inv_ga
                acc["ce_loss"] += float(losses["ce_loss"].detach().item()) * inv_ga
                acc["mtp_loss"] += float(losses["mtp_loss"].detach().item()) * inv_ga
                acc["saliency_loss"] += float(losses["saliency_loss"].detach().item()) * inv_ga
                acc["stroke_hog_loss"] += float(losses["stroke_hog_loss"].detach().item()) * inv_ga
                acc["vico_loss"] += vico_val * inv_ga
                rinfo = out["router_info"]
                last_n_in = int(rinfo["n_in"])
                last_n_out = int(rinfo["n_out"])

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            gpu_ema.update(model, step)
            step_dt = time.perf_counter() - t_step0
            step_times.append(step_dt)

            row = {
                "step": step,
                "lr": round(cur_lr, 8),
                "n_in": last_n_in,
                "n_out": last_n_out,
                "step_sec": round(step_dt, 3),
                **{k: round(v, 5) for k, v in acc.items()},
            }
            history.append(row)

            if step == 1 or step % 50 == 0 or step == steps:
                print(
                    f"[{arm}] step {step}/{steps} | lr={cur_lr:.2e} | loss={acc['loss']:.4f} "
                    f"(ce={acc['ce_loss']:.4f}, mtp={acc['mtp_loss']:.4f}, vico={acc['vico_loss']:.4f}) | "
                    f"toks {last_n_in}->{last_n_out} | {step_dt:.2f}s/step",
                    flush=True,
                )

            # Mandatory 5-min atomic rolling checkpoint + Modal Volume commit
            if (time.perf_counter() - last_time_save) >= save_every_sec:
                last_time_save = time.perf_counter()
                atomic_save_v2_latest(
                    out_dir,
                    model,
                    cfg,
                    step=step,
                    best_val_ser=best_val_ser,
                    gpu_ema=gpu_ema,
                    best_ema_ser=best_ema_ser,
                )
                vol.commit()
                print(f"  [{arm} autosave @ step {step}] committed to {out_dir}", flush=True)

            # Periodic validation on proxy_val.jsonl
            if step % eval_every == 0 or step == steps:
                torch.cuda.empty_cache()
                raw_ev = _eval_val_loss_and_gen(f"step {step} raw")
                raw_ser = float(raw_ev["generation"]["ser"])
                if raw_ser < best_val_ser:
                    best_val_ser = raw_ser
                    atomic_save_checkpoint(
                        out_dir / "uni_omr_v2_best.pt",
                        {
                            "arm": arm,
                            "router": router_name,
                            "step": step,
                            "val_ser": best_val_ser,
                            "cfg": asdict(cfg),
                            "vocab": list(data.VOCAB),
                            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        },
                    )
                with gpu_ema.swap_into(model):
                    ema_ev = _eval_val_loss_and_gen(f"step {step} EMA")
                ema_ser = float(ema_ev["generation"]["ser"])
                if ema_ser < best_ema_ser:
                    best_ema_ser = ema_ser
                    atomic_save_checkpoint(
                        out_dir / "uni_omr_v2_ema_best.pt",
                        {
                            "arm": arm,
                            "router": router_name,
                            "step": step,
                            "val_ser": best_ema_ser,
                            "ema_decay": gpu_ema.decay,
                            "cfg": asdict(cfg),
                            "vocab": list(data.VOCAB),
                            "model": gpu_ema.cpu_state_dict(model),
                        },
                    )
                eval_history.append({"step": step, "raw": raw_ev, "ema": ema_ev})
                vol.commit()
    finally:
        prefetcher.close()

    total_sec = time.perf_counter() - t_train_start
    atomic_save_v2_latest(
        out_dir,
        model,
        cfg,
        step=steps,
        best_val_ser=best_val_ser,
        gpu_ema=gpu_ema,
        best_ema_ser=best_ema_ser,
    )

    result = {
        "exp_id": EXP_ID,
        "arm": arm,
        "router": router_name,
        "gpu": dev_name,
        "cc": f"{cc_major}.{cc_minor}",
        "bf16": use_bf16,
        "steps": steps,
        "seed": seed,
        "optimizer_mode": opt.mode,
        "params": param_summary,
        "total_train_sec": round(total_sec, 1),
        "mean_step_sec": round(float(np.mean(step_times)) if step_times else 0.0, 3),
        "best_val_ser": round(best_val_ser, 6),
        "best_ema_ser": round(best_ema_ser, 6),
        "final_eval": eval_history[-1] if eval_history else None,
        "eval_history": eval_history,
        "tail_train_history": history[-20:],
        "completed_at": _utc_now(),
    }
    (out_dir / "arm_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    vol.commit()

    # Upload best EMA checkpoint for this arm to HF Private Dataset Hub
    try:
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        best_ema_path = out_dir / "uni_omr_v2_ema_best.pt"
        if best_ema_path.is_file():
            api.upload_file(
                path_or_fileobj=str(best_ema_path),
                path_in_repo=f"checkpoints/{EXP_ID}/{arm}/uni_omr_v2_ema_best.pt",
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"exp3({arm}): upload best EMA checkpoint (SER={best_ema_ser:.4f})",
            )
            print(f"[{arm}] Uploaded best EMA checkpoint to HF", flush=True)
    except Exception as e:
        print(f"[{arm}] WARN HF checkpoint upload failed (saved on Modal Volume): {e}", flush=True)

    return result


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=10800,
    volumes={"/vol": vol},
    secrets=[hf_secret],
)
def run_ablation_orchestrator(
    steps: int = 1200,
    seed: int = 42,
    eval_every: int = 600,
) -> Dict[str, Any]:
    """Remote detached orchestrator: prepare dataset -> fan out 2x L4 arms -> upload HF summary."""
    from huggingface_hub import HfApi

    t0 = time.perf_counter()
    started_at = _utc_now()
    print(f"[orchestrator] Starting {EXP_ID} (steps={steps}, seed={seed}) at {started_at}", flush=True)

    # 1. Prepare dataset once on shared Volume
    prep_info = prepare_dataset.remote()
    print(f"[orchestrator] Dataset ready: {prep_info}", flush=True)

    # 2. Fan out both arms in parallel on 2x L4 containers
    print("[orchestrator] Spawning baseline & vir_vico arms in parallel on 2x L4 ...", flush=True)
    h_base = train_arm.spawn(arm="baseline", steps=steps, seed=seed, eval_every=eval_every)
    h_vir = train_arm.spawn(arm="vir_vico", steps=steps, seed=seed, eval_every=eval_every)

    res_base = h_base.get()
    res_vir = h_vir.get()
    wall_sec = round(time.perf_counter() - t0, 1)

    def _extract_final(res: Dict[str, Any], branch: str) -> Dict[str, float]:
        fe = (res.get("final_eval") or {}).get(branch) or {}
        tf = fe.get("teacher_forced") or {}
        gen = fe.get("generation") or {}
        return {
            "val_loss": float(tf.get("val_loss", 0.0)),
            "val_ce": float(tf.get("val_ce", 0.0)),
            "val_mtp": float(tf.get("val_mtp", 0.0)),
            "ser": float(gen.get("ser", 1.0)),
            "token_f1": float(gen.get("token_f1", 0.0)),
            "sys_count_exact": float(gen.get("sys_count_exact", 0.0)),
            "mean_compression_x": float(gen.get("mean_compression_x", 0.0)),
            "ink_recall": float(gen.get("ink_recall", 0.0)),
            "tokens_per_sec": float(gen.get("tokens_per_sec", 0.0)),
        }

    base_ema = _extract_final(res_base, "ema")
    vir_ema = _extract_final(res_vir, "ema")
    base_raw = _extract_final(res_base, "raw")
    vir_raw = _extract_final(res_vir, "raw")

    deltas_ema = {k: round(vir_ema[k] - base_ema[k], 6) for k in base_ema}
    deltas_raw = {k: round(vir_raw[k] - base_raw[k], 6) for k in base_raw}

    comparison = {
        "exp_id": EXP_ID,
        "status": "COMPLETED",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "wall_clock_sec": wall_sec,
        "dataset": {
            "hf_repo": HF_REPO_ID,
            "tar": HF_DATASET_TAR,
            "n_train": prep_info.get("n_train"),
            "n_val": prep_info.get("n_val"),
        },
        "config": {
            "preset": "small_45m",
            "steps": steps,
            "seed": seed,
            "batch_size": 2,
            "grad_accum": 4,
            "effective_batch": 8,
            "fullpage_max_dim": 1280,
            "max_seq_len": 1536,
            "keep_ratio": 0.40,
            "max_mem_tokens": 3072,
            "vir_k_coarse": 96,
            "vir_static_buckets": True,
            "vico_weight": 0.10,
            "vico_temp": 1.5,
            "vico_every": 4,
        },
        "summary_ema": {
            "baseline_ink_router": base_ema,
            "vir_two_tier_router": vir_ema,
            "delta_vir_minus_base": deltas_ema,
        },
        "summary_raw": {
            "baseline_ink_router": base_raw,
            "vir_two_tier_router": vir_raw,
            "delta_vir_minus_base": deltas_raw,
        },
        "arms": {
            "baseline": res_base,
            "vir_vico": res_vir,
        },
    }

    run_ledger = {
        "exp_id": EXP_ID,
        "provider": "modal",
        "gpu": "2x L4 (parallel)",
        "status": "COMPLETED",
        "started_at": started_at,
        "completed_at": comparison["completed_at"],
        "wall_clock_sec": wall_sec,
        "summary_ema": comparison["summary_ema"],
        "hf_eval_path": f"evals/{EXP_ID}.json",
        "hf_checkpoints": {
            "baseline": f"checkpoints/{EXP_ID}/baseline/uni_omr_v2_ema_best.pt",
            "vir_vico": f"checkpoints/{EXP_ID}/vir_vico/uni_omr_v2_ema_best.pt",
        },
    }

    VOL_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    eval_local = VOL_CKPT_ROOT / f"{EXP_ID}.json"
    run_local = VOL_CKPT_ROOT / f"{EXP_ID}_run.json"
    eval_local.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    run_local.write_text(json.dumps(run_ledger, indent=2), encoding="utf-8")
    vol.commit()

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_file(
        path_or_fileobj=str(eval_local),
        path_in_repo=f"evals/{EXP_ID}.json",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"exp3: upload ViR+ViCO vs InkTokenRouterV2 comparison eval",
    )
    api.upload_file(
        path_or_fileobj=str(run_local),
        path_in_repo=f"runs/{EXP_ID}.json",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"exp3: upload run ledger",
    )
    print(f"[orchestrator] COMPLETE in {wall_sec}s! EMA delta (vir - base): {deltas_ema}", flush=True)
    return comparison


@app.local_entrypoint()
def main(steps: int = 1200, seed: int = 42, eval_every: int = 600, wait: bool = False) -> None:
    """Spawn the remote orchestrator so `modal run --detach` returns cleanly while Modal executes."""
    call = run_ablation_orchestrator.spawn(steps=steps, seed=seed, eval_every=eval_every)
    print(f"[exp3] Spawned detached orchestrator call_id={call.object_id}", flush=True)
    if wait:
        res = call.get()
        print(json.dumps(res.get("summary_ema", {}), indent=2), flush=True)
