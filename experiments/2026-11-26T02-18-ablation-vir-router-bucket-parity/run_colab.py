#!/usr/bin/env python3
"""Google Colab L4 runner for Follow-Up Exp: Un-Confounded ViRTwoTierRouter Bucket & Token-Budget Parity.

Experiment ID: 2026-11-26T02-18-ablation-vir-router-bucket-parity
Successor to: 2026-11-26T00-54-ablation-vir-vico-router (Exp 3, ITERATE)

Fixes BOTH confounders identified in Exp 3:
  1. Expanded static buckets: (256, 512, 768, 1024, 1536, 2048, 2560, 2976) so dense 1280px
     pages (mean_n_in=5102.5) are no longer hard-truncated at 2048 tokens (-6.28 pp ink_recall).
  2. Training token-budget parity: keep_ratio=0.75 during training (matching InkTokenRouterV2's
     max_keep_ratio=0.75) with identical dynamic_routing (0.15..0.75) at eval.
  3. Separates ViRTwoTierRouter (vico_weight=0.0) from ViCO (vico_weight=0.05, vico_every=2).

Uploads results to Hugging Face Private Dataset Hub (gregoreesmaa/sheet-music-converter-omr):
  - evals/2026-11-26T02-18-ablation-vir-router-bucket-parity.json
  - runs/2026-11-26T02-18-ablation-vir-router-bucket-parity.json
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
from typing import Any, Dict, List, Sequence, Tuple

EXP_ID = "2026-11-26T02-18-ablation-vir-router-bucket-parity"
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

TRAIN_STEPS = 600
SEED = 42
LR = 3e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 2
GRAD_ACCUM = 4
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
VAL_EVAL_MAX_LEN = 512
MAX_MEM_TOKENS = 3072
K_COARSE = 96
PARITY_BUCKETS: Tuple[int, ...] = (256, 512, 768, 1024, 1536, 2048, 2560, 2976)
VICO_TEMP = 1.5
VICO_EVERY = 2
DENSE_TEACHER_KEEP_RATIO = 0.85
EMA_DECAY = 0.99
SAVE_EVERY_MINUTES = 5.0

ARMS: Tuple[Dict[str, Any], ...] = (
    {"name": "baseline_ink_router", "use_vir": False, "vico_weight": 0.0},
    {"name": "vir_parity_no_vico", "use_vir": True, "vico_weight": 0.0},
    {"name": "vir_parity_vico005", "use_vir": True, "vico_weight": 0.05},
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
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {cmd}")


def _read_and_purge_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and HF_TOKEN_FILE.exists():
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
    here = Path(__file__).resolve()
    candidates = [
        WORKSPACE_ROOT / "sheet-music-converter" / "tmp-train" / "mvd",
        WORKSPACE_ROOT / "tmp-train" / "mvd",
        WORKSPACE_ROOT / "mvd",
        Path("/content/sheet-music-converter/tmp-train/mvd"),
    ]
    if len(here.parents) > 2:
        candidates.append(here.parents[2] / "tmp-train" / "mvd")
    for c in candidates:
        if (c / "vir_router_v2.py").is_file() and (c / "uni_omr_v2.py").is_file():
            return c
    for b in BUNDLE_CANDIDATES:
        if b.is_file():
            WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
            with tarfile.open(b, "r:*") as tf:
                tf.extractall(path=WORKSPACE_ROOT)
            for c in candidates:
                if (c / "vir_router_v2.py").is_file() and (c / "uni_omr_v2.py").is_file():
                    return c
    raise FileNotFoundError("Could not locate tmp-train/mvd/vir_router_v2.py")


def _setup_environment() -> Path:
    needed: List[str] = []
    for mod, pkg in (("huggingface_hub", "huggingface_hub>=0.26"), ("PIL", "pillow"), ("numpy", "numpy"), ("tqdm", "tqdm")):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed])
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

    for src_name, dst_path in (("proxy_train.jsonl", train_abs), ("proxy_val.jsonl", val_abs)):
        out_lines: List[str] = []
        for raw in (proxy_dir / src_name).read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            rec["image"] = str((proxy_dir / rec["image"]).resolve())
            rec["bekerns"] = [str((proxy_dir / b).resolve()) for b in rec.get("bekerns", [])]
            out_lines.append(json.dumps(rec))
        dst_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
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


def run_arm(
    arm_cfg: Dict[str, Any],
    train_pairs: List[Tuple[Path, List[Path], str]],
    val_pairs: List[Tuple[Path, List[Path], str]],
    device: Any,
    use_bf16: bool,
) -> Dict[str, Any]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from train_uni_v2 import atomic_save_v2_latest, build_optimizer_v2, forward_and_loss_v2
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
    from vir_router_v2 import ViRTwoTierRouter, compute_vico_consistency_loss

    name = str(arm_cfg["name"])
    use_vir = bool(arm_cfg["use_vir"])
    vico_weight = float(arm_cfg["vico_weight"])
    out_dir = VOL_ROOT / "ckpts" / EXP_ID / name
    out_dir.mkdir(parents=True, exist_ok=True)

    _seed_everything(SEED)
    cfg = UniOMRv2Config.small_45m(
        grad_checkpoint=True,
        dropout=0.1,
        mixstyle_prob=0.3,
        tok_dropout=0.05,
        keep_ratio=0.40,
        dynamic_routing=True,
        min_keep_ratio=0.15,
        max_keep_ratio=0.75,
        max_mem_tokens=MAX_MEM_TOKENS,
        mtp_k=4,
        use_sys_y_aux=True,
        use_sys_count_aux=True,
        use_staff_seg_aux=True,
        use_stroke_hog_head=True,
    )
    model = UniOMRv2Net(cfg)
    if use_vir:
        model.encoder.router = ViRTwoTierRouter(
            d_model=cfg.d_model,
            keep_ratio=cfg.max_keep_ratio,  # 0.75 training parity with InkTokenRouterV2
            k_coarse=K_COARSE,
            static_buckets=True,
            fine_buckets=PARITY_BUCKETS,
            use_ink_prior=cfg.use_ink_prior,
            dynamic_routing=True,
            min_keep_ratio=cfg.min_keep_ratio,
            max_keep_ratio=cfg.max_keep_ratio,
            saliency_thr=cfg.saliency_thr,
            max_mem_tokens=cfg.max_mem_tokens,
            router_hidden=cfg.router_hidden,
        )
    model = model.to(device)

    _seed_everything(SEED)
    sampler = StratifiedFullPageSampler(train_pairs)
    train_ds = UniFullPageDataset(
        train_pairs, fullpage_max_dim=FULLPAGE_MAX_DIM, max_seq_len=MAX_SEQ_LEN,
        augment=True, aug_prob=0.75, multi_scale=True, return_uint8=True,
    )
    val_ds = UniFullPageDataset(
        val_pairs, fullpage_max_dim=FULLPAGE_MAX_DIM, max_seq_len=MAX_SEQ_LEN,
        augment=False, multi_scale=False, return_uint8=True,
    )
    prefetcher = AsyncThreadPrefetcher(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False,
        max_workers=4, prefetch_batches=3, collate_fn=collate_uni_batch,
        seed=SEED + 303, stratified_sampler=sampler,
    )

    opt = build_optimizer_v2(model, optimizer_name="muon", lr=LR, weight_decay=WEIGHT_DECAY, device_type="cuda")
    gpu_ema = GPUEmaShadow(model, decay=EMA_DECAY, update_every=5)
    warmup_steps = max(20, int(round(TRAIN_STEPS * 0.10)))
    save_every_sec = SAVE_EVERY_MINUTES * 60.0
    last_save = time.perf_counter()
    t0 = time.perf_counter()
    step_times: List[float] = []
    vico_vals: List[float] = []

    @torch.no_grad()
    def _eval(tag: str) -> Dict[str, Any]:
        model.eval()
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_uni_batch)
        sums = {"val_loss": 0.0, "val_ce": 0.0, "val_mtp": 0.0, "mean_n_in": 0.0, "mean_n_out": 0.0, "mean_ink_recall": 0.0}
        nb = 0
        for vb in val_loader:
            vi = _images_to_device_float(vb["images"], device, non_blocking=True)
            vs = vb["seq"].to(device, non_blocking=True)
            vl, vo = forward_and_loss_v2(
                model, vi, vs, kind="fullpage", mtp_loss_weight=0.30, saliency_loss_weight=0.10,
                sys_y_loss_weight=0.25, sys_count_loss_weight=0.15, staff_seg_loss_weight=0.20,
                stroke_hog_weight=0.15, label_smoothing=0.0, use_bf16=use_bf16, device=device,
            )
            ri = vo["router_info"]
            sums["val_loss"] += float(vl["loss"].item())
            sums["val_ce"] += float(vl["ce_loss"].item())
            sums["val_mtp"] += float(vl["mtp_loss"].item())
            sums["mean_n_in"] += float(ri["n_in"])
            sums["mean_n_out"] += float(ri["n_out"])
            sums["mean_ink_recall"] += float(ri.get("ink_recall", 1.0))
            nb += 1
        tf = {k: round(v / max(1, nb), 5) for k, v in sums.items()}
        gen = eval_dataset(model, val_ds, device=device, max_len=VAL_EVAL_MAX_LEN, batch_size=4, use_bf16=use_bf16)
        model.train()
        print(
            f"  [{name} {tag}] val_ce={tf['val_ce']:.4f} SER={gen['ser']:.4f} F1={gen['token_f1']:.4f} "
            f"sys_exact={gen['sys_count_exact']:.3f} ink_rec={gen['ink_recall']:.4f} comp={gen['mean_compression_x']:.2f}x",
            flush=True,
        )
        return {"teacher_forced": tf, "generation": gen}

    model.train()
    try:
        for step in range(1, TRAIN_STEPS + 1):
            ts = time.perf_counter()
            cur_lr = _lr_for_step(step, TRAIN_STEPS, LR, warmup_steps, min_lr_ratio=0.05, schedule="wsd", wsd_decay_frac=0.25)
            opt.set_lr(cur_lr / max(1e-12, LR))
            opt.zero_grad(set_to_none=True)
            do_vico = use_vir and (vico_weight > 0.0) and (step % VICO_EVERY == 0)
            inv_ga = 1.0 / float(GRAD_ACCUM)
            step_loss = 0.0
            step_vico = 0.0
            last_n_in, last_n_out = 0, 0

            for _ in range(GRAD_ACCUM):
                b = next(prefetcher)
                images = _images_to_device_float(b["images"], device, non_blocking=True)
                seq = b["seq"].to(device, non_blocking=True)
                losses, out = forward_and_loss_v2(
                    model, images, seq, kind="fullpage", mtp_loss_weight=0.30, saliency_loss_weight=0.10,
                    sys_y_loss_weight=0.25, sys_count_loss_weight=0.15, staff_seg_loss_weight=0.20,
                    stroke_hog_weight=0.15, label_smoothing=0.05, use_bf16=use_bf16, device=device,
                )
                if do_vico:
                    router = model.encoder.router
                    prev_sb = bool(router.static_buckets)
                    router.static_buckets = False
                    try:
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16):
                            dense_out = model(images, seq[:, :-1], keep_ratio_override=DENSE_TEACHER_KEEP_RATIO, return_aux=False, return_mtp=False)
                    finally:
                        router.static_buckets = prev_sb
                    vl = compute_vico_consistency_loss(
                        dense_out["logits"], out["logits"], temp=VICO_TEMP,
                        valid_mask=(seq[:, 1:] != PAD_ID), symmetric=True, detach_dense=True,
                    )
                    losses["loss"] = losses["loss"] + vico_weight * vl
                    step_vico += float(vl.detach().item()) * inv_ga

                (losses["loss"] * inv_ga).backward()
                step_loss += float(losses["loss"].detach().item()) * inv_ga
                ri = out["router_info"]
                last_n_in, last_n_out = int(ri["n_in"]), int(ri["n_out"])

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            gpu_ema.update(model, step)
            dt = time.perf_counter() - ts
            step_times.append(dt)
            if do_vico:
                vico_vals.append(step_vico)

            if step == 1 or step % 100 == 0 or step == TRAIN_STEPS:
                print(
                    f"[{name}] step {step}/{TRAIN_STEPS} | lr={cur_lr:.2e} | loss={step_loss:.4f} "
                    f"| vico={step_vico:.4f} | toks {last_n_in}->{last_n_out} | {dt:.2f}s/step",
                    flush=True,
                )
            if (time.perf_counter() - last_save) >= save_every_sec:
                last_save = time.perf_counter()
                atomic_save_v2_latest(out_dir, model, cfg, step=step, best_val_ser=1.0, gpu_ema=gpu_ema, best_ema_ser=1.0)
    finally:
        prefetcher.close()

    torch.cuda.empty_cache()
    raw_ev = _eval("raw")
    with gpu_ema.swap_into(model):
        ema_ev = _eval("EMA")
    elapsed = round(time.perf_counter() - t0, 1)

    res = {
        "arm": name,
        "use_vir": use_vir,
        "vico_weight": vico_weight,
        "train_sec": elapsed,
        "mean_step_sec": round(float(np.mean(step_times)), 3),
        "mean_active_vico": round(float(np.mean(vico_vals)), 5) if vico_vals else 0.0,
        "raw": raw_ev,
        "ema": ema_ev,
    }
    del model, opt, gpu_ema
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main() -> None:
    import torch
    from huggingface_hub import HfApi

    started_at = _utc_now()
    t0 = time.perf_counter()
    hf_token = _read_and_purge_hf_token()
    _setup_environment()
    train_abs, val_abs = _ensure_proxy_dataset(hf_token)
    train_pairs = _load_manifest_pairs(train_abs)
    val_pairs = _load_manifest_pairs(val_abs)

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = torch.device("cuda")
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    use_bf16 = cc_major >= 8
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    results: Dict[str, Any] = {}
    for arm_cfg in ARMS:
        results[arm_cfg["name"]] = run_arm(arm_cfg, train_pairs, val_pairs, device, use_bf16)

    wall_sec = round(time.perf_counter() - t0, 1)
    payload = {
        "exp_id": EXP_ID,
        "predecessor": "2026-11-26T00-54-ablation-vir-vico-router",
        "status": "COMPLETED",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "wall_clock_sec": wall_sec,
        "gpu": torch.cuda.get_device_name(0),
        "cc": f"{cc_major}.{cc_minor}",
        "bf16": use_bf16,
        "config": {
            "steps": TRAIN_STEPS,
            "seed": SEED,
            "lr": LR,
            "max_mem_tokens": MAX_MEM_TOKENS,
            "k_coarse": K_COARSE,
            "parity_buckets": list(PARITY_BUCKETS),
            "ema_decay": EMA_DECAY,
        },
        "arms": results,
    }
    out_dir = VOL_ROOT / "results" / EXP_ID
    eval_path = out_dir / "eval_result.json"
    run_path = out_dir / "run_ledger.json"
    _atomic_write_json(eval_path, payload)
    _atomic_write_json(run_path, {
        "exp_id": EXP_ID,
        "provider": "colab_l4",
        "status": "COMPLETED",
        "completed_at": payload["completed_at"],
        "wall_clock_sec": wall_sec,
        "hf_eval_path": HF_EVAL_PATH,
    })

    api = HfApi(token=hf_token)
    api.upload_file(path_or_fileobj=str(eval_path), path_in_repo=HF_EVAL_PATH, repo_id=HF_REPO_ID, repo_type="dataset", commit_message=f"{EXP_ID}: upload eval")
    api.upload_file(path_or_fileobj=str(run_path), path_in_repo=HF_RUN_PATH, repo_id=HF_REPO_ID, repo_type="dataset", commit_message=f"{EXP_ID}: upload run ledger")
    print(f"[{EXP_ID}] COMPLETE in {wall_sec}s -> uploaded {HF_EVAL_PATH}", flush=True)


if __name__ == "__main__":
    main()
