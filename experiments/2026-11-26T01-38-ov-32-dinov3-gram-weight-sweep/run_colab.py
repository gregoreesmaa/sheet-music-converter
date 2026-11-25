#!/usr/bin/env python3
"""Google Colab L4 runner for Overnight Exp 32: DINOv3 Gram Anchoring Weight Sweep.

Experiment ID: 2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep
Sweeps DINOv3 Ink-Token Gram Anchoring weight across (0.0, 0.1, 0.5) for
500 steps/arm on unlabeled RISM 5,707 (datasets/rism_hw_5707.tar), measuring
dense patch-feature effective rank, normalized spectral entropy, top-1
singular value dominance, mean off-diagonal cosine similarity, and Gram drift
on 128 held-out RISM pages.

Uploads results to Hugging Face Private Dataset Hub:
  - evals/2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep.json
  - runs/2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep.json
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

EXP_ID = "2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_RISM_TAR = "datasets/rism_hw_5707.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
WORKSPACE_ROOT = Path("/workspace")
VOL_ROOT = Path("/content/vol")

# Controlled sweep hyperparameters
GRAM_WEIGHTS: Tuple[float, ...] = (0.0, 0.1, 0.5)
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
GRAM_MAX_TOKENS = 512
STROKE_HOG_WEIGHT = 0.25
ROUTER_CONS_WEIGHT = 0.2
SAVE_EVERY_MINUTES = 5.0
LOG_EVERY = 25
SEED = 42
VAL_DIAG_PAGES = 128
DIAG_DIM = 768
DIAG_MAX_TOKENS = 512


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
    if len(imgs) < VAL_DIAG_PAGES + 32:
        raise RuntimeError(f"Expected >= {VAL_DIAG_PAGES + 32} RISM images, found {len(imgs)}")
    return imgs


def _measure_dense_feature_health(
    teacher: Any,
    gram_teacher: Any,
    val_paths: Sequence[Path],
    device: "torch.device",
    use_bf16: bool,
    pj: Any,
) -> Dict[str, Any]:
    """Measure dense ink-token feature rank, spectral entropy, cosine diversity, and Gram drift."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    eranks: List[float] = []
    norm_entropies: List[float] = []
    top1_var_fracs: List[float] = []
    offdiag_cosines: List[float] = []
    gram_drifts: List[float] = []
    pooled_embs: List[np.ndarray] = []
    ink_densities: List[float] = []

    teacher.eval()
    gram_teacher.eval()
    amp_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_bf16
        else pj._nullcontext()
    )

    with torch.no_grad():
        for p in val_paths:
            try:
                t_cpu = pj.load_image_resized(p, DIAG_DIM)
            except Exception:
                continue
            luma = (0.299 * t_cpu[0] + 0.587 * t_cpu[1] + 0.114 * t_cpu[2]).float()
            p85 = torch.quantile(luma.reshape(-1)[:: max(1, luma.numel() // 32768)], 0.85)
            ink_densities.append(float(((p85 - luma) > 0.14).float().mean().item()))

            img = t_cpu.unsqueeze(0).to(device, non_blocking=(device.type == "cuda"))
            with amp_ctx:
                levels_t, routed_coords, info = teacher.encode(img)
                k_kept = int(info["k_kept"])
                levels_g, _, _ = gram_teacher.encode(
                    img,
                    forced_kept_idx=info["kept_idx"],
                    forced_routed_coords=routed_coords,
                    forced_info=info,
                )

            feat_t = levels_t[2][0, :k_kept, :].float()
            feat_g = levels_g[2][0, :k_kept, :].float()

            # Deterministic stride subsample up to DIAG_MAX_TOKENS
            if feat_t.shape[0] > DIAG_MAX_TOKENS:
                stride = max(1, feat_t.shape[0] // DIAG_MAX_TOKENS)
                feat_t = feat_t[::stride][:DIAG_MAX_TOKENS]
                feat_g = feat_g[::stride][:DIAG_MAX_TOKENS]

            # Centered SVD for Effective Rank (Roy & Vetterli) & Spectral Entropy
            centered = feat_t - feat_t.mean(dim=0, keepdim=True)
            sv = torch.linalg.svdvals(centered)
            sv_sum = float(sv.sum().item())
            if sv_sum > 1e-8 and sv.numel() >= 2:
                prob = (sv / sv_sum).clamp(min=1e-12)
                entropy = float(-(prob * torch.log(prob)).sum().item())
                erank = math.exp(entropy)
                norm_ent = entropy / math.log(float(sv.numel()))
                sv_sq = sv * sv
                top1_var = float((sv_sq[0] / sv_sq.sum().clamp(min=1e-12)).item())
            else:
                erank, norm_ent, top1_var = 1.0, 0.0, 1.0

            eranks.append(erank)
            norm_entropies.append(norm_ent)
            top1_var_fracs.append(top1_var)

            # L2-normalized pairwise Gram matrix & off-diagonal cosine similarity
            fn_t = F.normalize(feat_t, p=2, dim=-1, eps=1e-6)
            fn_g = F.normalize(feat_g, p=2, dim=-1, eps=1e-6)
            g_t = fn_t @ fn_t.T
            g_0 = fn_g @ fn_g.T
            m = g_t.shape[0]
            if m >= 2:
                offdiag_mean = float(((g_t.sum() - g_t.diagonal().sum()) / float(m * (m - 1))).item())
            else:
                offdiag_mean = 1.0
            offdiag_cosines.append(offdiag_mean)
            gram_drifts.append(float(F.mse_loss(g_t, g_0).item()))
            pooled_embs.append(F.normalize(feat_t.mean(dim=0), p=2, dim=-1, eps=1e-6).cpu().numpy())

    # Global cross-page effective rank (checks if all pages collapse to one vector)
    X = np.stack(pooled_embs, axis=0).astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    sv_global = np.linalg.svd(Xc, compute_uv=False)
    p_glob = sv_global / (sv_global.sum() + 1e-12)
    p_glob = np.clip(p_glob, 1e-12, 1.0)
    global_erank = float(math.exp(-np.sum(p_glob * np.log(p_glob))))

    # Simple Ridge probe on ink density
    y = np.asarray(ink_densities, dtype=np.float64)
    n_fit = int(round(0.75 * len(y)))
    X_tr, X_te = X[:n_fit], X[n_fit:]
    y_tr, y_te = y[:n_fit], y[n_fit:]
    mu_x = X_tr.mean(axis=0, keepdims=True)
    mu_y = float(y_tr.mean())
    w = np.linalg.solve((X_tr - mu_x).T @ (X_tr - mu_x) + np.eye(X.shape[1]), (X_tr - mu_x).T @ (y_tr - mu_y))
    pred = (X_te - mu_x) @ w + mu_y
    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2)) + 1e-12
    ink_probe_r2 = 1.0 - float(np.sum((y_te - pred) ** 2)) / ss_tot

    return {
        "n_pages": len(eranks),
        "dense_erank_mean": round(float(np.mean(eranks)), 3),
        "dense_erank_p10": round(float(np.percentile(eranks, 10)), 3),
        "dense_norm_spectral_entropy": round(float(np.mean(norm_entropies)), 5),
        "dense_top1_var_frac": round(float(np.mean(top1_var_fracs)), 5),
        "dense_offdiag_cosine_mean": round(float(np.mean(offdiag_cosines)), 5),
        "gram_drift_vs_step0_mse": round(float(np.mean(gram_drifts)), 6),
        "global_page_erank": round(global_erank, 3),
        "ridge_ink_probe_r2": round(ink_probe_r2, 5),
    }


def _run_gram_arm(
    gram_weight: float,
    train_paths: Sequence[Path],
    val_paths: Sequence[Path],
    device: "torch.device",
    use_bf16: bool,
    pj: Any,
) -> Dict[str, Any]:
    import numpy as np
    import torch

    arm_name = f"gram_{str(gram_weight).replace('.', 'p')}"
    print(f"\n[{EXP_ID}] === Starting Arm {arm_name} (gram_weight={gram_weight}, steps={STEPS_PER_ARM}) ===", flush=True)

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

    out_dir = VOL_ROOT / "ckpts" / "ov32" / arm_name
    out_dir.mkdir(parents=True, exist_ok=True)

    student.train()
    predictors.train()
    stroke_hog_head.train()
    teacher.eval()
    gram_teacher.eval()

    best_ema_loss = float("inf")
    ema_loss_track: Optional[float] = None
    nan_skips = 0
    oom_skips = 0
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
            }
            valid_micros = 0

            for _ in range(GRAD_ACCUM):
                images = next(loader).to(device, non_blocking=(device.type == "cuda"))
                try:
                    losses = pj.vjepa21_forward_loss(
                        student=student,
                        teacher=teacher,
                        gram_teacher=gram_teacher,
                        predictors=predictors,
                        stroke_hog_head=stroke_hog_head,
                        images=images,
                        rng=rng,
                        use_bf16=use_bf16,
                        device_type=device.type,
                        level_weights=LEVEL_WEIGHTS,
                        masked_weight=MASKED_WEIGHT,
                        visible_weight=VISIBLE_WEIGHT,
                        gram_weight=gram_weight,
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
                    "vram_gb": round(vram_gb, 2),
                    "steps_per_sec": round(step / elapsed, 2),
                }
                history.append(rec)
                print(
                    f"[{arm_name} {step:4d}/{STEPS_PER_ARM}] loss={avg['loss']:.4f} (ema={ema_loss_track:.4f}) "
                    f"m={avg['jepa_masked']:.4f} gram={avg['gram']:.4f} hog={avg['stroke_hog']:.4f} "
                    f"vram={vram_gb:.2f}GB",
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

    health = _measure_dense_feature_health(
        teacher=teacher,
        gram_teacher=gram_teacher,
        val_paths=val_paths,
        device=device,
        use_bf16=use_bf16,
        pj=pj,
    )
    elapsed_arm = round(time.perf_counter() - t0, 2)

    result = {
        "arm": arm_name,
        "gram_weight": gram_weight,
        "steps": STEPS_PER_ARM,
        "optimizer_mode": opt.mode,
        "elapsed_secs": elapsed_arm,
        "best_ema_loss": round(best_ema_loss, 5),
        "final_ema_loss": round(ema_loss_track if ema_loss_track is not None else float("nan"), 5),
        "final_jepa_masked": history[-1]["jepa_masked"] if history else None,
        "nan_skips": nan_skips,
        "oom_skips": oom_skips,
        "dense_feature_health": health,
        "history": history,
    }
    print(f"[{EXP_ID}] Arm {arm_name} health: {json.dumps(health)}", flush=True)

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
        val_paths = shuffled[:VAL_DIAG_PAGES]
        train_paths = shuffled[VAL_DIAG_PAGES:]
        print(f"[{EXP_ID}] RISM split: train={len(train_paths)}, val_diag={len(val_paths)}", flush=True)

        arms: Dict[str, Any] = {}
        progress_file = VOL_ROOT / "evals" / "ov32_progress.json"
        for gw in GRAM_WEIGHTS:
            res = _run_gram_arm(gw, train_paths, val_paths, device=device, use_bf16=use_bf16, pj=pj)
            arms[res["arm"]] = res
            _atomic_write_json(progress_file, {"exp_id": EXP_ID, "arms": arms})

        best_arm = max(arms.values(), key=lambda a: a["dense_feature_health"]["dense_erank_mean"])
        summary = {
            "best_erank_arm": best_arm["arm"],
            "best_gram_weight": best_arm["gram_weight"],
            "best_dense_erank_mean": best_arm["dense_feature_health"]["dense_erank_mean"],
            "per_arm_summary": {
                k: {
                    "gram_weight": v["gram_weight"],
                    "final_jepa_masked": v["final_jepa_masked"],
                    "dense_erank_mean": v["dense_feature_health"]["dense_erank_mean"],
                    "dense_norm_spectral_entropy": v["dense_feature_health"]["dense_norm_spectral_entropy"],
                    "dense_offdiag_cosine_mean": v["dense_feature_health"]["dense_offdiag_cosine_mean"],
                    "gram_drift_vs_step0_mse": v["dense_feature_health"]["gram_drift_vs_step0_mse"],
                    "ridge_ink_probe_r2": v["dense_feature_health"]["ridge_ink_probe_r2"],
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
                "gram_weights": list(GRAM_WEIGHTS),
                "steps_per_arm": STEPS_PER_ARM,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM,
                "max_dim": MAX_DIM,
                "lr": LR,
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
            commit_message=f"eval({EXP_ID}): upload DINOv3 Gram weight sweep results",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload DINOv3 Gram weight sweep run ledger",
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
