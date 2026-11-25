#!/usr/bin/env python3
"""Google Colab L4 runner for Overnight Exp 31: V-JEPA 2.1 Masking Ratio Sweep.

Experiment ID: 2026-11-26T01-38-ov-31-vjepa21-mask-ratio-sweep
Sweeps V-JEPA 2.1 HTR-VT ink-span masking ratio across (0.45, 0.60, 0.75)
for 500 steps/arm on unlabeled RISM 5,707 (datasets/rism_hw_5707.tar),
evaluating loss stability and frozen-encoder linear (Ridge) + kNN (k=5)
staff-density probes on a held-out 256-page RISM split.

Uploads results to Hugging Face Private Dataset Hub:
  - evals/2026-11-26T01-38-ov-31-vjepa21-mask-ratio-sweep.json
  - runs/2026-11-26T01-38-ov-31-vjepa21-mask-ratio-sweep.json
"""

from __future__ import annotations

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

EXP_ID = "2026-11-26T01-38-ov-31-vjepa21-mask-ratio-sweep"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_RISM_TAR = "datasets/rism_hw_5707.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
WORKSPACE_ROOT = Path("/workspace")
VOL_ROOT = Path("/content/vol")

# Controlled sweep hyperparameters
MASK_RATIOS: Tuple[float, ...] = (0.45, 0.60, 0.75)
MASK_HALF_WIDTH = 0.05
STEPS_PER_ARM = 500
BATCH_SIZE = 4
GRAD_ACCUM = 2
MAX_DIM = 1024
LR = 3e-4
WEIGHT_DECAY = 0.04
WARMUP_STEPS = 50
EMA_START = 0.996
EMA_END = 0.999
LEVEL_WEIGHTS = (0.25, 0.35, 1.0)
MASKED_WEIGHT = 3.0
VISIBLE_WEIGHT = 0.5
GRAM_WEIGHT = 1.0
GRAM_MAX_TOKENS = 512
STROKE_HOG_WEIGHT = 0.25
ROUTER_CONS_WEIGHT = 0.2
SAVE_EVERY_MINUTES = 5.0
LOG_EVERY = 25
SEED = 42
VAL_PROBE_PAGES = 256
PROBE_DIM = 768


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _run_cmd(cmd: Sequence[str]) -> None:
    print(f"[{EXP_ID}] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        list(cmd),
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
        raise RuntimeError(f"Command failed (rc={rc}): {' '.join(cmd)}")


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


def _locate_or_extract_mvd_dir() -> Path:
    candidates = [
        WORKSPACE_ROOT / "sheet-music-converter" / "tmp-train" / "mvd",
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path.cwd() / "tmp-train" / "mvd",
    ]
    for cand in candidates:
        if (cand / "pretrain_ijepa_rism_v2.py").is_file():
            return cand

    for bundle in (Path("/content/repo_bundle.tar.gz"), Path("/content/bundle.tar.gz")):
        if bundle.is_file():
            WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
            print(f"[{EXP_ID}] Extracting {bundle} -> {WORKSPACE_ROOT}...", flush=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=WORKSPACE_ROOT)
            for cand in candidates:
                if (cand / "pretrain_ijepa_rism_v2.py").is_file():
                    return cand

    raise FileNotFoundError(
        "Could not locate tmp-train/mvd/pretrain_ijepa_rism_v2.py. "
        "Upload repo tarball to /content/repo_bundle.tar.gz before executing."
    )


def _ensure_packages() -> None:
    needed = []
    for mod, pkg in [
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed])


def _ensure_rism_dataset(hf_token: str) -> List[Path]:
    from huggingface_hub import hf_hub_download

    rism_dir = VOL_ROOT / "datasets" / "rism_hw_5707"
    sentinel = rism_dir / ".unpacked_ok"
    if not sentinel.is_file():
        rism_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{EXP_ID}] Downloading {HF_RISM_TAR} from {HF_REPO_ID}...", flush=True)
        t0 = time.perf_counter()
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_RISM_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        print(f"[{EXP_ID}] Downloaded in {time.perf_counter() - t0:.1f}s -> extracting...", flush=True)
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=rism_dir)
        try:
            os.remove(tar_p)
        except OSError:
            pass
        sentinel.write_text("ok\n", encoding="utf-8")

    imgs = sorted(
        p
        for p in rism_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )
    if len(imgs) < VAL_PROBE_PAGES + 32:
        raise RuntimeError(f"Expected >= {VAL_PROBE_PAGES + 32} RISM images, found {len(imgs)}")
    return imgs


def _compute_staff_density_target(img_chw: "torch.Tensor") -> Tuple[float, float]:
    """Compute unsupervised staff-line & ink density targets from a clean archival page.

    Returns:
        staff_density: fraction of horizontal rows with high Sobel-Y horizontal edge energy
        ink_density: fraction of dark ink pixels relative to per-page p85 paper tone
    """
    import torch
    import torch.nn.functional as F

    luma = (0.299 * img_chw[0] + 0.587 * img_chw[1] + 0.114 * img_chw[2]).float()
    p85 = torch.quantile(luma.reshape(-1)[:: max(1, luma.numel() // 32768)], 0.85)
    ink_mask = (p85 - luma) > 0.14
    ink_density = float(ink_mask.float().mean().item())

    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3)
    gy = F.conv2d(luma.unsqueeze(0).unsqueeze(0), ky, padding=1).abs().squeeze(0).squeeze(0)
    row_energy = gy.mean(dim=1)
    thresh = row_energy.mean() + 0.75 * row_energy.std(unbiased=False)
    staff_density = float((row_energy > thresh).float().mean().item())
    return staff_density, ink_density


def _eval_staff_density_probes(
    teacher: Any,
    val_paths: Sequence[Path],
    device: "torch.device",
    use_bf16: bool,
    pj: Any,
) -> Dict[str, Any]:
    """Extract frozen EMA teacher embeddings on held-out RISM pages and run Ridge + kNN probes."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    feats: List[np.ndarray] = []
    targets: List[float] = []
    ink_targets: List[float] = []

    teacher.eval()
    amp_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_bf16
        else pj._nullcontext()
    )
    with torch.no_grad():
        for p in val_paths:
            try:
                t_cpu = pj.load_image_resized(p, PROBE_DIM)
            except Exception:
                continue
            sd, idens = _compute_staff_density_target(t_cpu)
            img = t_cpu.unsqueeze(0).to(device, non_blocking=(device.type == "cuda"))
            with amp_ctx:
                levels, _, info = teacher.encode(img)
                k_kept = int(info["k_kept"])
                emb = levels[2][0, :k_kept, :].float().mean(dim=0)
                emb = F.normalize(emb, p=2, dim=-1, eps=1e-6)
            feats.append(emb.cpu().numpy())
            targets.append(sd)
            ink_targets.append(idens)

    X = np.stack(feats, axis=0).astype(np.float64)
    y = np.asarray(targets, dtype=np.float64)
    y_ink = np.asarray(ink_targets, dtype=np.float64)
    n = len(y)
    n_fit = int(round(0.75 * n))
    X_tr, X_te = X[:n_fit], X[n_fit:]
    y_tr, y_te = y[:n_fit], y[n_fit:]

    # 1. Closed-form Ridge regression (alpha=1.0)
    mu_x = X_tr.mean(axis=0, keepdims=True)
    mu_y = float(y_tr.mean())
    Xc_tr = X_tr - mu_x
    Xc_te = X_te - mu_x
    yc_tr = y_tr - mu_y
    d = Xc_tr.shape[1]
    reg = np.eye(d, dtype=np.float64) * 1.0
    w = np.linalg.solve(Xc_tr.T @ Xc_tr + reg, Xc_tr.T @ yc_tr)
    pred_ridge = Xc_te @ w + mu_y
    ss_res = float(np.sum((y_te - pred_ridge) ** 2))
    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2)) + 1e-12
    ridge_r2 = 1.0 - (ss_res / ss_tot)
    ridge_mae = float(np.mean(np.abs(y_te - pred_ridge)))

    # 2. k=5 cosine kNN regression + quartile classification accuracy
    sim = X_te @ X_tr.T
    k = min(5, X_tr.shape[0])
    topk_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    pred_knn = np.array([y_tr[row].mean() for row in topk_idx], dtype=np.float64)
    knn_r2 = 1.0 - float(np.sum((y_te - pred_knn) ** 2)) / ss_tot
    knn_mae = float(np.mean(np.abs(y_te - pred_knn)))

    q_edges = np.quantile(y_tr, [0.25, 0.50, 0.75])
    q_tr = np.digitize(y_tr, q_edges)
    q_te = np.digitize(y_te, q_edges)
    q_pred = np.array([int(np.bincount(q_tr[row], minlength=4).argmax()) for row in topk_idx])
    knn_q_acc = float(np.mean(q_pred == q_te))

    # Auxiliary Ridge probe on ink density
    mu_yi = float(y_ink[:n_fit].mean())
    w_i = np.linalg.solve(Xc_tr.T @ Xc_tr + reg, Xc_tr.T @ (y_ink[:n_fit] - mu_yi))
    pred_i = Xc_te @ w_i + mu_yi
    ss_tot_i = float(np.sum((y_ink[n_fit:] - y_ink[n_fit:].mean()) ** 2)) + 1e-12
    ink_ridge_r2 = 1.0 - float(np.sum((y_ink[n_fit:] - pred_i) ** 2)) / ss_tot_i

    return {
        "n_val_pages": n,
        "n_fit": n_fit,
        "n_test": n - n_fit,
        "ridge_staff_r2": round(ridge_r2, 5),
        "ridge_staff_mae": round(ridge_mae, 5),
        "knn5_staff_r2": round(knn_r2, 5),
        "knn5_staff_mae": round(knn_mae, 5),
        "knn5_staff_quartile_acc": round(knn_q_acc, 5),
        "ridge_ink_r2": round(ink_ridge_r2, 5),
    }


def _run_mask_arm(
    mask_ratio: float,
    train_paths: Sequence[Path],
    val_paths: Sequence[Path],
    device: "torch.device",
    use_bf16: bool,
    pj: Any,
) -> Dict[str, Any]:
    import numpy as np
    import torch

    arm_name = f"mask_{int(round(mask_ratio * 100)):03d}"
    mask_range = (max(0.05, mask_ratio - MASK_HALF_WIDTH), min(0.95, mask_ratio + MASK_HALF_WIDTH))
    print(f"\n[{EXP_ID}] === Starting Arm {arm_name} (mask_range={mask_range}, steps={STEPS_PER_ARM}) ===", flush=True)

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats(device)
    rng = random.Random(SEED)

    student = pj.uni_omr_v2_small_45m().to(device)
    cfg = student.cfg
    pj.freeze_non_encoder_params_v2(student)
    teacher = pj.EMAEncoderTeacherV2(student).to(device)
    gram_teacher = pj.EMAEncoderTeacherV2(student).to(device)
    predictors = pj.DeepStackPredictors(cfg.d_model).to(device)
    stroke_hog_head = pj.StrokeHOGHead(in_ch=student.backbone.p2_ch, n_bins=8).to(device)

    opt, all_trainable = pj.build_optimizer_v2(
        student=student,
        predictors=predictors,
        stroke_hog_head=stroke_hog_head,
        optimizer_name="muon",
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device_type=device.type,
    )

    loader = pj.UnlabeledPageBatchPrefetcher(
        paths=train_paths,
        batch_size=BATCH_SIZE,
        max_dim=MAX_DIM,
        multi_scale=True,
        seed=SEED,
        num_threads=4 if device.type == "cuda" else 0,
    )

    out_dir = VOL_ROOT / "ckpts" / "ov31" / arm_name
    out_dir.mkdir(parents=True, exist_ok=True)
    amp_ctx_factory = (
        (lambda: torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16))
        if use_bf16
        else pj._nullcontext
    )

    student.train()
    predictors.train()
    stroke_hog_head.train()
    teacher.eval()
    gram_teacher.eval()

    best_ema_loss = float("inf")
    ema_loss_track: Optional[float] = None
    nan_skips = 0
    oom_skips = 0
    step_losses: List[float] = []
    history: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    last_autosave_t = time.monotonic()
    autosave_interval_s = SAVE_EVERY_MINUTES * 60.0

    try:
        for step in range(1, STEPS_PER_ARM + 1):
            lr_mult = pj.cosine_lr_mult(step - 1, STEPS_PER_ARM, warmup_steps=WARMUP_STEPS)
            opt.set_lr_mult(lr_mult)
            opt.zero_grad(set_to_none=True)

            acc = {
                "loss": 0.0,
                "jepa_masked": 0.0,
                "jepa_visible": 0.0,
                "deep_sup": 0.0,
                "gram": 0.0,
                "stroke_hog": 0.0,
                "router_loss": 0.0,
                "masked_ink_frac": 0.0,
            }
            valid_micros = 0

            for _ in range(GRAD_ACCUM):
                images = next(loader).to(device, non_blocking=(device.type == "cuda"))
                try:
                    with amp_ctx_factory():
                        target_levels, routed_coords, teacher_info = teacher.encode(images)
                        local_ink_grid = teacher_info["local_ink_grid"]
                        kept_idx = teacher_info["kept_idx"]

                        gram_levels, _, _ = gram_teacher.encode(
                            images,
                            forced_kept_idx=kept_idx,
                            forced_routed_coords=routed_coords,
                            forced_info=teacher_info,
                        )
                        aug_images = pj.apply_mild_photometric_aug(images, rng=rng)
                        student_images, grid_mask, _ = pj.apply_ink_span_mask_and_corruption(
                            aug_images,
                            local_ink_grid,
                            mask_ratio_range=mask_range,
                            corrupt_ratio_range=(0.10, 0.20),
                            rng=rng,
                        )
                        mem_levels_s, sal_logits_s, p2 = pj.encode_student_aligned_v2(
                            student=student,
                            student_images=student_images,
                            teacher_kept_idx=kept_idx,
                            teacher_routed_coords=routed_coords,
                            teacher_info=teacher_info,
                        )
                        pred_levels = predictors(mem_levels_s)
                        hog_pred = stroke_hog_head(p2)
                        hog_target = pj.compute_sobel_hog_target(
                            images, target_hw=(p2.shape[2], p2.shape[3]), n_bins=8
                        )
                        losses = pj.compute_vjepa21_losses(
                            pred_levels=pred_levels,
                            target_levels=target_levels,
                            mem_final_student=mem_levels_s[2],
                            mem_final_gram=gram_levels[2],
                            hog_pred=hog_pred,
                            hog_target=hog_target,
                            grid_mask=grid_mask,
                            kept_idx=kept_idx,
                            local_ink_grid=local_ink_grid,
                            sal_logits_student=sal_logits_s,
                            sal_logits_teacher=teacher_info["saliency_logits"],
                            effective_paper=teacher_info["effective_paper"],
                            level_weights=LEVEL_WEIGHTS,
                            masked_weight=MASKED_WEIGHT,
                            visible_weight=VISIBLE_WEIGHT,
                            gram_weight=GRAM_WEIGHT,
                            gram_max_tokens=GRAM_MAX_TOKENS,
                            stroke_hog_weight=STROKE_HOG_WEIGHT,
                            router_cons_weight=ROUTER_CONS_WEIGHT,
                        )
                    scaled_loss = losses["loss"] / float(GRAD_ACCUM)
                    if not torch.isfinite(scaled_loss):
                        nan_skips += 1
                        continue
                    scaled_loss.backward()
                    for k in acc:
                        acc[k] += float(losses[k].detach().item())
                    valid_micros += 1
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        oom_skips += 1
                        opt.zero_grad(set_to_none=True)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        valid_micros = 0
                        break
                    raise

            if valid_micros == 0:
                continue

            torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
            opt.step()

            ema_decay = pj.cosine_ema_decay(step, STEPS_PER_ARM, start=EMA_START, end=EMA_END)
            teacher.update_from(student, decay=ema_decay)

            avg = {k: v / valid_micros for k, v in acc.items()}
            step_losses.append(avg["loss"])
            ema_loss_track = avg["loss"] if ema_loss_track is None else (0.95 * ema_loss_track + 0.05 * avg["loss"])
            if ema_loss_track < best_ema_loss:
                best_ema_loss = ema_loss_track

            if step == 1 or step % LOG_EVERY == 0 or step == STEPS_PER_ARM:
                elapsed = max(1e-6, time.perf_counter() - t0)
                vram_gb = (
                    torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
                    if device.type == "cuda"
                    else 0.0
                )
                rec = {
                    "step": step,
                    "loss": round(avg["loss"], 5),
                    "ema_loss": round(ema_loss_track, 5),
                    "jepa_masked": round(avg["jepa_masked"], 5),
                    "jepa_visible": round(avg["jepa_visible"], 5),
                    "deep_sup": round(avg["deep_sup"], 5),
                    "gram": round(avg["gram"], 5),
                    "stroke_hog": round(avg["stroke_hog"], 5),
                    "masked_ink_frac": round(avg["masked_ink_frac"], 3),
                    "vram_gb": round(vram_gb, 2),
                    "steps_per_sec": round(step / elapsed, 2),
                }
                history.append(rec)
                print(
                    f"[{arm_name} {step:4d}/{STEPS_PER_ARM}] loss={avg['loss']:.4f} (ema={ema_loss_track:.4f}) "
                    f"m={avg['jepa_masked']:.4f} v={avg['jepa_visible']:.4f} gram={avg['gram']:.4f} "
                    f"mask_ink={avg['masked_ink_frac']:.2f} vram={vram_gb:.2f}GB",
                    flush=True,
                )

            if (time.monotonic() - last_autosave_t) >= autosave_interval_s:
                pj.save_ijepa_v2_checkpoint(
                    out_dir / "uni_omr_v2_ijepa_autosave.pt",
                    student=student,
                    teacher=teacher,
                    cfg=cfg,
                    step=step,
                    loss_val=ema_loss_track,
                )
                last_autosave_t = time.monotonic()
    finally:
        loader.close()

    pj.save_ijepa_v2_checkpoint(
        out_dir / "uni_omr_v2_ijepa_last.pt",
        student=student,
        teacher=teacher,
        cfg=cfg,
        step=STEPS_PER_ARM,
        loss_val=ema_loss_track if ema_loss_track is not None else float("nan"),
    )

    probe = _eval_staff_density_probes(teacher, val_paths, device=device, use_bf16=use_bf16, pj=pj)
    tail = step_losses[-100:] if len(step_losses) >= 100 else step_losses
    tail_std = float(np.std(tail)) if tail else float("nan")
    elapsed_arm = round(time.perf_counter() - t0, 2)

    result = {
        "arm": arm_name,
        "mask_ratio": mask_ratio,
        "mask_ratio_range": list(mask_range),
        "steps": STEPS_PER_ARM,
        "optimizer_mode": opt.mode,
        "elapsed_secs": elapsed_arm,
        "best_ema_loss": round(best_ema_loss, 5),
        "final_ema_loss": round(ema_loss_track if ema_loss_track is not None else float("nan"), 5),
        "tail100_loss_std": round(tail_std, 5),
        "final_masked_ink_frac": history[-1]["masked_ink_frac"] if history else None,
        "nan_skips": nan_skips,
        "oom_skips": oom_skips,
        "probe": probe,
        "history": history,
    }
    print(f"[{EXP_ID}] Arm {arm_name} probe: {json.dumps(probe)}", flush=True)

    del student, teacher, gram_teacher, predictors, stroke_hog_head, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    t_start = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    try:
        hf_token = _read_and_purge_hf_token()
        _ensure_packages()
        mvd_dir = _locate_or_extract_mvd_dir()
        if str(mvd_dir) not in sys.path:
            sys.path.insert(0, str(mvd_dir))

        import torch
        from huggingface_hub import HfApi
        import pretrain_ijepa_rism_v2 as pj

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        use_bf16 = bool(device.type == "cuda" and torch.cuda.is_bf16_supported())
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
                torch.backends.cuda.enable_cudnn_sdp(False)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
        print(f"[{EXP_ID}] Device={device} ({gpu_name}), bf16={use_bf16}", flush=True)

        all_imgs = _ensure_rism_dataset(hf_token)
        split_rng = random.Random(SEED)
        shuffled = list(all_imgs)
        split_rng.shuffle(shuffled)
        val_paths = shuffled[:VAL_PROBE_PAGES]
        train_paths = shuffled[VAL_PROBE_PAGES:]
        print(f"[{EXP_ID}] RISM split: train={len(train_paths)}, val_probe={len(val_paths)}", flush=True)

        arms: Dict[str, Any] = {}
        progress_file = VOL_ROOT / "evals" / "ov31_progress.json"
        for mr in MASK_RATIOS:
            res = _run_mask_arm(mr, train_paths, val_paths, device=device, use_bf16=use_bf16, pj=pj)
            arms[res["arm"]] = res
            _atomic_write_json(progress_file, {"exp_id": EXP_ID, "arms": arms})

        best_arm = max(arms.values(), key=lambda a: a["probe"]["ridge_staff_r2"])
        summary = {
            "best_arm": best_arm["arm"],
            "best_mask_ratio": best_arm["mask_ratio"],
            "best_ridge_staff_r2": best_arm["probe"]["ridge_staff_r2"],
            "best_knn5_staff_r2": best_arm["probe"]["knn5_staff_r2"],
            "per_arm_summary": {
                k: {
                    "mask_ratio": v["mask_ratio"],
                    "final_ema_loss": v["final_ema_loss"],
                    "tail100_loss_std": v["tail100_loss_std"],
                    "ridge_staff_r2": v["probe"]["ridge_staff_r2"],
                    "knn5_staff_r2": v["probe"]["knn5_staff_r2"],
                    "knn5_staff_quartile_acc": v["probe"]["knn5_staff_quartile_acc"],
                    "nan_skips": v["nan_skips"],
                }
                for k, v in arms.items()
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
                "mask_ratios": list(MASK_RATIOS),
                "steps_per_arm": STEPS_PER_ARM,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM,
                "max_dim": MAX_DIM,
                "lr": LR,
                "gram_weight": GRAM_WEIGHT,
                "stroke_hog_weight": STROKE_HOG_WEIGHT,
                "seed": SEED,
            },
            "summary": summary,
            "arms": arms,
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
            commit_message=f"eval({EXP_ID}): upload V-JEPA 2.1 mask ratio sweep results",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload V-JEPA 2.1 mask ratio sweep run ledger",
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
