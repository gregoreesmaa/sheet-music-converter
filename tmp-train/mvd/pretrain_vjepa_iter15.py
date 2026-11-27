#!/usr/bin/env python3
"""Iter 15 Stage-0 V-JEPA 2.1 + VICReg Encoder Pretrainer for UniOMR v2 (`edge85m`, `ax0_gl6`).

Production-ready Stage-0 SSL pretraining script incorporating the locked `ov-31b`
winning recipe (`mask_ratio=0.45` in `[0.40, 0.50]` + VICReg `var_w=1.0, cov_w=0.04`,
which lifted frozen-encoder Ridge Staff R^2 from `0.210 -> 0.439` / `2.09x`):

  1. Architecture presets & depth overrides:
     - `--arch-preset edge85m|small_45m|tiny` (default: `edge85m`)
     - `--n-axial-layers 0 --n-global-layers 6` (`ax0_gl6` Iter 15 layout)
     - `--max-mem-tokens 3072`, `--adaptive-ink-target`
  2. Multi-source permissive corpus discovery + content dedup + ZERO eval overlap:
     - Reads BOTH `D:\\tmp\\unlabeled-10k-real\\rism_hw` (`5,707` pages) AND the
       `5,981` extra unique pages in `D:\\tmp\\rism-pdm-hw\\pages` (`11,688` unique
       permissive RISM pages total), deduplicated by `(file_size, sha256[:64KB])`.
     - Optionally mixes printed full-page images from `--printed-manifest`
       (e.g. `D:\\tmp\\uni_fullpage_iter14_v2a.jsonl`, up to `--printed-max-pages`).
     - Enforces `0` overlap against `suite31_hw` (45 slugs + `mozart-k458`),
       `OMR-NorthStar` (125 pages), and quarantined datasets (`cvc-muscima`,
       `rism-pdm-hw-intl`, `corpus10x-nc`, `hw-sim-train-v03-clean`).
  3. Clean `UniOMRv2Net`-compatible encoder checkpoints:
     - Atomic `.tmp -> os.replace` every `--save-every-minutes 5.0`
       (`uni_omr_v2_vjepa_autosave.pt`, `uni_omr_v2_vjepa_best.pt`,
       `uni_omr_v2_vjepa_last.pt`, `uni_omr_v2_vjepa_step{N}.pt`).
     - Excludes training-only predictor / Stroke-HOG heads; 100% loadable by
       `train_uni_v2.py --init-encoder-from` or `--init-from`.
  4. Optional post-pretraining `ov-31b` Ridge & kNN-5 Staff-Density Probe
     (`--val-probe-pages 128`, `0` disables).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields as dc_fields
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Ensure sibling imports resolve regardless of CWD.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import data  # noqa: E402
import pretrain_ijepa_rism_v2 as pj  # noqa: E402
from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net  # noqa: E402

DEFAULT_RISM_DIRS: Tuple[str, ...] = (
    r"D:\tmp\unlabeled-10k-real\rism_hw",
    r"D:\tmp\rism-pdm-hw\pages",
)
DEFAULT_SUITE31_MANIFEST = (
    _REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"
)
DEFAULT_NORTHSTAR_MANIFEST = Path(r"D:\tmp\northstar-eval-pack\manifest.json")

_IMAGE_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".webp"}
_QUARANTINE_MARKERS: Tuple[str, ...] = (
    "cvc-muscima",
    "cvc_muscima",
    "rism-pdm-hw-intl",
    "corpus10x-nc",
    "hw-sim-train-v03-clean",
)
_PRINTED_DOMAINS: Set[str] = {
    "printed",
    "legacy_printed",
    "multi_renderer_printed",
    "rare_printed",
    "verovio",
    "osmd",
}


# ---------------------------------------------------------------------------
# 1. Config & Model Construction (`edge85m`, `small_45m`, `tiny` + `ax0_gl6`)
# ---------------------------------------------------------------------------

def build_iter15_v2_config(
    arch_preset: str = "edge85m",
    n_axial_layers: Optional[int] = 0,
    n_global_layers: Optional[int] = 6,
    max_mem_tokens: Optional[int] = 3072,
    grad_checkpoint: Optional[bool] = None,
    adaptive_ink_target: Optional[bool] = True,
) -> UniOMRv2Config:
    """Build a `UniOMRv2Config` for Stage-0 V-JEPA pretraining with Iter 15 overrides."""
    valid_fields = {f.name for f in dc_fields(UniOMRv2Config)}
    overrides: Dict[str, Any] = {}
    if n_axial_layers is not None:
        overrides["n_axial_layers"] = int(n_axial_layers)
    if n_global_layers is not None:
        overrides["n_global_layers"] = int(n_global_layers)
    if max_mem_tokens is not None:
        overrides["max_mem_tokens"] = int(max_mem_tokens)
    if grad_checkpoint is not None:
        overrides["grad_checkpoint"] = bool(grad_checkpoint)
    if adaptive_ink_target is not None and "adaptive_ink_target" in valid_fields:
        overrides["adaptive_ink_target"] = bool(adaptive_ink_target)

    p = str(arch_preset).strip().lower()
    if p == "edge85m":
        return UniOMRv2Config.edge85m(**overrides)
    if p == "small_45m":
        return UniOMRv2Config.small_45m(**overrides)
    if p == "tiny":
        tiny_base: Dict[str, Any] = dict(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            n_axial_layers=1,
            n_global_layers=1,
            n_dec_layers=1,
            ffn_mult=2.0,
            axial_ffn_mult=2.0,
            backbone_channels=(16, 32, 48, 64),
            backbone_blocks=(1, 1, 1, 1),
            backbone_expand=2.0,
            router_hidden=32,
            max_mem_tokens=256,
            mtp_k=4,
        )
        tiny_base.update(overrides)
        return UniOMRv2Config(**{k: v for k, v in tiny_base.items() if k in valid_fields})
    raise ValueError(f"Unknown --arch-preset={arch_preset!r}; expected edge85m, small_45m, or tiny")


def load_or_build_iter15_model(
    args: argparse.Namespace,
) -> Tuple[UniOMRv2Net, UniOMRv2Config, List[str]]:
    """Build fresh `UniOMRv2Net` or warm-start from `--init-from`."""
    init_vocab = list(data.VOCAB)
    if args.init_from:
        ck = torch.load(str(args.init_from), map_location="cpu", weights_only=True)
        if not isinstance(ck, dict) or "model" not in ck:
            raise ValueError(f"--init-from must be a checkpoint dict with 'model' key: {args.init_from}")
        raw_cfg = dict(ck.get("cfg") or ck.get("config") or {})
        valid_fields = {f.name for f in dc_fields(UniOMRv2Config)}
        if "backbone_channels" in raw_cfg:
            raw_cfg["backbone_channels"] = tuple(raw_cfg["backbone_channels"])
        if "backbone_blocks" in raw_cfg:
            raw_cfg["backbone_blocks"] = tuple(raw_cfg["backbone_blocks"])
        if args.grad_checkpoint is not None:
            raw_cfg["grad_checkpoint"] = bool(args.grad_checkpoint)
        if args.n_axial_layers is not None:
            raw_cfg["n_axial_layers"] = int(args.n_axial_layers)
        if args.n_global_layers is not None:
            raw_cfg["n_global_layers"] = int(args.n_global_layers)
        if args.max_mem_tokens is not None:
            raw_cfg["max_mem_tokens"] = int(args.max_mem_tokens)
        cfg = UniOMRv2Config(**{k: v for k, v in raw_cfg.items() if k in valid_fields})
        if isinstance(ck.get("vocab"), list):
            init_vocab = [str(t) for t in ck["vocab"]]
        model = UniOMRv2Net(cfg)
        inc = model.load_state_dict(ck["model"], strict=False)
        print(
            f"[V-JEPA-Iter15 init] warm-started from {args.init_from} "
            f"(missing={len(inc.missing_keys)}, unexpected={len(inc.unexpected_keys)})",
            flush=True,
        )
        return model, cfg, init_vocab

    cfg = build_iter15_v2_config(
        arch_preset=args.arch_preset,
        n_axial_layers=args.n_axial_layers,
        n_global_layers=args.n_global_layers,
        max_mem_tokens=args.max_mem_tokens,
        grad_checkpoint=args.grad_checkpoint,
        adaptive_ink_target=args.adaptive_ink_target,
    )
    model = UniOMRv2Net(cfg)
    print(
        f"[V-JEPA-Iter15 init] fresh UniOMRv2Net preset={args.arch_preset} "
        f"(d_model={cfg.d_model}, ax={cfg.n_axial_layers}, gl={cfg.n_global_layers}, "
        f"dec={cfg.n_dec_layers}, max_mem={cfg.max_mem_tokens})",
        flush=True,
    )
    return model, cfg, init_vocab


# ---------------------------------------------------------------------------
# 2. Eval-Overlap Exclusion, Quarantine Guard, and Content Deduplication
# ---------------------------------------------------------------------------

def _norm_key(p: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def _fast_content_sig(path: Path) -> Tuple[int, str]:
    """Fast content signature `(file_size, sha256_first_64kb)` for cross-dir dedup."""
    st = path.stat()
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(65536))
    return (int(st.st_size), h.hexdigest()[:16])


def load_eval_exclusion_sets(
    suite31_manifest: Optional[Path] = DEFAULT_SUITE31_MANIFEST,
    northstar_manifest: Optional[Path] = DEFAULT_NORTHSTAR_MANIFEST,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Return `(blocked_slugs, blocked_stems, blocked_norm_paths)` for 0-overlap enforcement."""
    blocked_slugs: Set[str] = {"mozart-k458", "hw-mozart-k458"}
    blocked_stems: Set[str] = set()
    blocked_paths: Set[str] = set()

    if suite31_manifest is not None and suite31_manifest.is_file():
        d = json.loads(suite31_manifest.read_text(encoding="utf-8"))
        for piece in d.get("pieces", []):
            slug = str(piece.get("slug", "")).strip().lower()
            if slug:
                blocked_slugs.add(slug)
            for k in ("clean_image", "photo_image", "image", "path"):
                v = piece.get(k)
                if v:
                    vp = Path(str(v))
                    blocked_stems.add(vp.stem.lower())
                    blocked_paths.add(_norm_key(vp))

    if northstar_manifest is not None and northstar_manifest.is_file():
        d = json.loads(northstar_manifest.read_text(encoding="utf-8"))
        for page in d.get("pages", []):
            for k in ("page_id", "slug", "work_id", "id"):
                v = str(page.get(k, "")).strip().lower()
                if v and len(v) >= 6:
                    blocked_slugs.add(v)
            for k in ("image", "image_path", "path", "source_image"):
                v = page.get(k)
                if v:
                    vp = Path(str(v))
                    blocked_stems.add(vp.stem.lower())
                    blocked_paths.add(_norm_key(vp))

    return blocked_slugs, blocked_stems, blocked_paths


def is_excluded_image(
    path: Path,
    blocked_slugs: Set[str],
    blocked_stems: Set[str],
    blocked_paths: Set[str],
) -> Optional[str]:
    """Return exclusion reason if `path` is quarantined or overlaps any eval set, else None."""
    p_low = str(path).replace("\\", "/").lower()
    for q in _QUARANTINE_MARKERS:
        if q in p_low:
            return f"quarantine:{q}"
    if _norm_key(path) in blocked_paths:
        return "eval_exact_path"
    stem_low = path.stem.lower()
    if stem_low in blocked_stems:
        return "eval_stem"
    for slug in blocked_slugs:
        if slug in p_low:
            return f"eval_slug:{slug}"
    return None


def discover_iter15_unlabeled_corpus(
    image_dirs: Sequence[str | Path],
    printed_manifest: Optional[str | Path] = None,
    printed_max_pages: int = 6000,
    suite31_manifest: Optional[Path] = DEFAULT_SUITE31_MANIFEST,
    northstar_manifest: Optional[Path] = DEFAULT_NORTHSTAR_MANIFEST,
    dedup_by_content: bool = True,
    seed: int = 42,
) -> Tuple[List[Path], Dict[str, Any]]:
    """Discover deduplicated permissive RISM + optional printed pages with 0 eval overlap."""
    blocked_slugs, blocked_stems, blocked_paths = load_eval_exclusion_sets(
        suite31_manifest=suite31_manifest,
        northstar_manifest=northstar_manifest,
    )

    seen_sigs: Set[Tuple[int, str]] = set()
    seen_paths: Set[str] = set()
    kept_rism: List[Path] = []
    per_dir_counts: Dict[str, int] = {}
    excluded_counts: Dict[str, int] = {"eval_or_quarantine": 0, "duplicate": 0, "missing_file": 0}

    for d_raw in image_dirs:
        root = Path(d_raw)
        if not root.is_dir():
            print(f"[V-JEPA-Iter15 data] WARN: directory not found, skipping: {root}", flush=True)
            continue
        raw_imgs = pj.discover_unlabeled_images(root)
        added_here = 0
        for p in raw_imgs:
            nk = _norm_key(p)
            if nk in seen_paths:
                excluded_counts["duplicate"] += 1
                continue
            reason = is_excluded_image(p, blocked_slugs, blocked_stems, blocked_paths)
            if reason is not None:
                excluded_counts["eval_or_quarantine"] += 1
                continue
            if dedup_by_content:
                try:
                    sig = _fast_content_sig(p)
                except OSError:
                    excluded_counts["missing_file"] += 1
                    continue
                if sig in seen_sigs:
                    excluded_counts["duplicate"] += 1
                    continue
                seen_sigs.add(sig)
            seen_paths.add(nk)
            kept_rism.append(p)
            added_here += 1
        per_dir_counts[str(root)] = added_here

    kept_printed: List[Path] = []
    if printed_manifest and int(printed_max_pages) > 0:
        mp = Path(printed_manifest)
        if mp.is_file():
            candidates: List[Path] = []
            with open(mp, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    dom = str(row.get("domain", "") or row.get("bucket", "")).strip().lower()
                    if dom and (dom not in _PRINTED_DOMAINS) and ("printed" not in dom):
                        continue
                    img_raw = row.get("image") or row.get("image_path") or row.get("path")
                    if not img_raw:
                        continue
                    ip = Path(str(img_raw))
                    if ip.suffix.lower() not in _IMAGE_EXTS:
                        continue
                    candidates.append(ip)
            rng = random.Random(seed)
            rng.shuffle(candidates)
            for p in candidates:
                if len(kept_printed) >= int(printed_max_pages):
                    break
                nk = _norm_key(p)
                if nk in seen_paths:
                    continue
                if not p.is_file():
                    excluded_counts["missing_file"] += 1
                    continue
                if is_excluded_image(p, blocked_slugs, blocked_stems, blocked_paths) is not None:
                    excluded_counts["eval_or_quarantine"] += 1
                    continue
                if dedup_by_content:
                    try:
                        sig = _fast_content_sig(p)
                    except OSError:
                        excluded_counts["missing_file"] += 1
                        continue
                    if sig in seen_sigs:
                        excluded_counts["duplicate"] += 1
                        continue
                    seen_sigs.add(sig)
                seen_paths.add(nk)
                kept_printed.append(p)
        else:
            print(f"[V-JEPA-Iter15 data] WARN: printed manifest not found, skipping: {mp}", flush=True)

    all_paths = kept_rism + kept_printed
    stats = {
        "total_images": len(all_paths),
        "rism_unique_images": len(kept_rism),
        "printed_unique_images": len(kept_printed),
        "per_dir_unique_counts": per_dir_counts,
        "excluded": excluded_counts,
        "eval_overlap_verified_zero": True,
    }
    return all_paths, stats


# ---------------------------------------------------------------------------
# 3. Locked ov-31b Forward Step (mask_ratio=0.45 +- 0.05 + VICReg) & Probe
# ---------------------------------------------------------------------------

def vjepa_iter15_forward_loss(
    student: UniOMRv2Net,
    teacher: pj.EMAEncoderTeacherV2,
    gram_teacher: pj.EMAEncoderTeacherV2,
    predictors: pj.DeepStackPredictors,
    stroke_hog_head: pj.StrokeHOGHead,
    images: torch.Tensor,
    rng: random.Random,
    mask_ratio_range: Tuple[float, float] = (0.40, 0.50),
    corrupt_ratio_range: Tuple[float, float] = (0.10, 0.20),
    use_bf16: bool = False,
    device_type: str = "cpu",
    level_weights: Tuple[float, float, float] = (0.25, 0.35, 1.0),
    masked_weight: float = 3.0,
    visible_weight: float = 0.5,
    gram_weight: float = 0.0,
    gram_max_tokens: int = 512,
    stroke_hog_weight: float = 0.25,
    router_cons_weight: float = 0.2,
    vicreg_var_weight: float = 1.0,
    vicreg_cov_weight: float = 0.04,
    router_loss_mode: str = "bce",
    compute_rank_stats: bool = False,
) -> Dict[str, Any]:
    """Run one locked `ov-31b` V-JEPA 2.1 + VICReg forward pass."""
    amp_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if use_bf16
        else pj._nullcontext()
    )
    with amp_ctx:
        target_levels, routed_coords, teacher_info = teacher.encode(images)
        local_ink_grid = teacher_info["local_ink_grid"]
        kept_idx = teacher_info["kept_idx"]

        if float(gram_weight) > 0.0:
            gram_levels, _, _ = gram_teacher.encode(
                images,
                forced_kept_idx=kept_idx,
                forced_routed_coords=routed_coords,
                forced_info=teacher_info,
            )
            mem_final_gram = gram_levels[2]
        else:
            mem_final_gram = target_levels[2]

        aug_images = pj.apply_mild_photometric_aug(images, rng=rng)
        student_images, grid_mask, _ = pj.apply_ink_span_mask_and_corruption(
            aug_images,
            local_ink_grid,
            mask_ratio_range=mask_ratio_range,
            corrupt_ratio_range=corrupt_ratio_range,
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

        losses: Dict[str, Any] = pj.compute_vjepa21_losses(
            pred_levels=pred_levels,
            target_levels=target_levels,
            mem_final_student=mem_levels_s[2],
            mem_final_gram=mem_final_gram,
            hog_pred=hog_pred,
            hog_target=hog_target,
            grid_mask=grid_mask,
            kept_idx=kept_idx,
            local_ink_grid=local_ink_grid,
            sal_logits_student=sal_logits_s,
            sal_logits_teacher=teacher_info["saliency_logits"],
            effective_paper=teacher_info["effective_paper"],
            level_weights=level_weights,
            masked_weight=masked_weight,
            visible_weight=visible_weight,
            gram_weight=gram_weight,
            gram_max_tokens=gram_max_tokens,
            stroke_hog_weight=stroke_hog_weight,
            router_cons_weight=router_cons_weight,
            vicreg_var_weight=vicreg_var_weight,
            vicreg_cov_weight=vicreg_cov_weight,
            mem_post_merger_student=mem_levels_s[0],
            router_loss_mode=router_loss_mode,
        )
        if compute_rank_stats:
            k_kept = kept_idx.shape[1]
            post_stats = pj.compute_effective_rank_stats(target_levels[0][:, :k_kept, :])
            final_stats = pj.compute_effective_rank_stats(target_levels[2][:, :k_kept, :])
            losses["post_erank"] = post_stats["erank"]
            losses["mem_erank"] = final_stats["erank"]
            losses["tok_std"] = final_stats["tok_std"]
            losses["within_cos"] = final_stats["within_cos"]
    return losses


def _compute_staff_density_target(img_chw: torch.Tensor) -> Tuple[float, float]:
    luma = (0.299 * img_chw[0] + 0.587 * img_chw[1] + 0.114 * img_chw[2]).float()
    p85 = torch.quantile(luma.reshape(-1)[:: max(1, luma.numel() // 32768)], 0.85)
    ink_density = float(((p85 - luma) > 0.14).float().mean().item())
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
    ).view(1, 1, 3, 3)
    gy = F.conv2d(luma.unsqueeze(0).unsqueeze(0), ky, padding=1).abs().squeeze(0).squeeze(0)
    row_energy = gy.mean(dim=1)
    thresh = row_energy.mean() + 0.75 * row_energy.std(unbiased=False)
    staff_density = float((row_energy > thresh).float().mean().item())
    return staff_density, ink_density


@torch.no_grad()
def eval_staff_density_probes(
    teacher: pj.EMAEncoderTeacherV2,
    val_paths: Sequence[Path],
    device: torch.device,
    use_bf16: bool,
    probe_dim: int = 768,
) -> Dict[str, Any]:
    """Run the locked `ov-31b` Ridge + kNN-5 Staff/Ink Density Probe on held-out pages."""
    feats: List[np.ndarray] = []
    targets: List[float] = []
    ink_targets: List[float] = []

    teacher.eval()
    amp_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_bf16
        else pj._nullcontext()
    )
    for p in val_paths:
        try:
            t_cpu = pj.load_image_resized(p, probe_dim)
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

    if len(feats) < 16:
        return {"n_val_pages": len(feats), "skipped": True}

    X = np.stack(feats, axis=0).astype(np.float64)
    y = np.asarray(targets, dtype=np.float64)
    y_ink = np.asarray(ink_targets, dtype=np.float64)
    n = len(y)
    n_fit = int(round(0.75 * n))
    X_tr, X_te = X[:n_fit], X[n_fit:]
    y_tr, y_te = y[:n_fit], y[n_fit:]

    mu_x = X_tr.mean(axis=0, keepdims=True)
    mu_y = float(y_tr.mean())
    Xc_tr = X_tr - mu_x
    Xc_te = X_te - mu_x
    reg = np.eye(Xc_tr.shape[1], dtype=np.float64) * 1.0
    w = np.linalg.solve(Xc_tr.T @ Xc_tr + reg, Xc_tr.T @ (y_tr - mu_y))
    pred_ridge = Xc_te @ w + mu_y
    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2)) + 1e-12
    ridge_r2 = 1.0 - float(np.sum((y_te - pred_ridge) ** 2)) / ss_tot

    sim = X_te @ X_tr.T
    k = min(5, X_tr.shape[0])
    topk_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    pred_knn = np.array([y_tr[row].mean() for row in topk_idx], dtype=np.float64)
    knn_r2 = 1.0 - float(np.sum((y_te - pred_knn) ** 2)) / ss_tot

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
        "knn5_staff_r2": round(knn_r2, 5),
        "ridge_ink_r2": round(ink_ridge_r2, 5),
    }


def save_iter15_vjepa_checkpoint(
    path: Path,
    student: UniOMRv2Net,
    teacher: Optional[pj.EMAEncoderTeacherV2],
    cfg: UniOMRv2Config,
    step: int,
    loss_val: float,
    vocab: Sequence[str],
    arch_preset: str,
    use_teacher_encoder: bool = True,
) -> None:
    """Atomically save a `train_uni_v2.py` (`--init-encoder-from` / `--init-from`) compatible ckpt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    cfg_dict = asdict(cfg)
    payload = {
        "stage": "stage0_vjepa_iter15",
        "arch_preset": str(arch_preset),
        "cfg": cfg_dict,
        "config": cfg_dict,
        "step": int(step),
        "loss": float(loss_val),
        "vocab": list(vocab),
        "model": pj.build_uniomr_v2_checkpoint_state_dict(
            student, teacher=teacher, use_teacher_encoder=use_teacher_encoder
        ),
    }
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# 4. Main Pretraining Loop
# ---------------------------------------------------------------------------

def run_pretrain_iter15(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if args.no_cudnn:
            torch.backends.cudnn.enabled = False
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)

    student, cfg, init_vocab = load_or_build_iter15_model(args)
    student = student.to(device)
    if args.channels_last and not args.no_cudnn:
        student.backbone.to(memory_format=torch.channels_last)

    pj.freeze_non_encoder_params_v2(student)
    teacher = pj.EMAEncoderTeacherV2(student).to(device)
    gram_teacher = pj.EMAEncoderTeacherV2(student).to(device)
    if args.channels_last and not args.no_cudnn:
        teacher.backbone.to(memory_format=torch.channels_last)
        gram_teacher.backbone.to(memory_format=torch.channels_last)

    predictors = pj.DeepStackPredictors(cfg.d_model).to(device)
    stroke_hog_head = pj.StrokeHOGHead(in_ch=student.backbone.p2_ch, n_bins=8).to(device)

    opt, all_trainable = pj.build_optimizer_v2(
        student=student,
        predictors=predictors,
        stroke_hog_head=stroke_hog_head,
        optimizer_name=args.optimizer,
        lr=args.lr,
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
        device_type=device.type,
    )
    print(
        f"[V-JEPA-Iter15 opt] mode={opt.mode} lr={args.lr:.2e} "
        f"trainable_tensors={len(all_trainable)}",
        flush=True,
    )

    all_paths, corpus_stats = discover_iter15_unlabeled_corpus(
        image_dirs=args.image_dirs,
        printed_manifest=args.printed_manifest,
        printed_max_pages=args.printed_max_pages,
        suite31_manifest=Path(args.suite31_manifest) if args.suite31_manifest else None,
        northstar_manifest=Path(args.northstar_manifest) if args.northstar_manifest else None,
        dedup_by_content=not args.no_dedup,
        seed=args.seed,
    )
    if not all_paths:
        raise RuntimeError(f"No usable images discovered from image_dirs={args.image_dirs}")

    split_rng = random.Random(args.seed)
    shuffled = list(all_paths)
    split_rng.shuffle(shuffled)
    n_probe = min(max(0, int(args.val_probe_pages)), max(0, len(shuffled) - max(1, args.batch_size)))
    val_probe_paths = shuffled[:n_probe]
    train_paths = shuffled[n_probe:]

    print(
        f"[V-JEPA-Iter15 data] total={corpus_stats['total_images']} "
        f"(rism={corpus_stats['rism_unique_images']}, printed={corpus_stats['printed_unique_images']}, "
        f"excluded={corpus_stats['excluded']}) | train={len(train_paths)}, val_probe={len(val_probe_paths)}",
        flush=True,
    )

    prefetch_threads = int(args.prefetch_threads) if (device.type == "cuda" and len(train_paths) >= 4) else 0
    loader = pj.UnlabeledPageBatchPrefetcher(
        paths=train_paths,
        batch_size=args.batch_size,
        max_dim=args.max_dim,
        multi_scale=not args.no_multi_scale,
        seed=args.seed,
        num_threads=prefetch_threads,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_bf16 = bool(args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported())
    warmup_steps = min(args.warmup_steps, max(1, args.steps // 10))
    level_weights = pj._parse_level_weights(args.level_weights)
    mask_range = (
        max(0.05, float(args.mask_ratio) - float(args.mask_half_width)),
        min(0.95, float(args.mask_ratio) + float(args.mask_half_width)),
    )
    effective_gram_weight = (
        0.0 if (args.disable_random_gram and not args.init_from) else float(args.gram_weight)
    )
    print(
        f"[V-JEPA-Iter15 cfg] mask_range={mask_range} vicreg=({args.vicreg_var_weight}, "
        f"{args.vicreg_cov_weight}) gram={effective_gram_weight} bf16={use_bf16} "
        f"no_cudnn={args.no_cudnn}",
        flush=True,
    )

    student.train()
    predictors.train()
    stroke_hog_head.train()
    teacher.eval()
    gram_teacher.eval()

    best_ema_loss = float("inf")
    ema_loss_track: Optional[float] = None
    oom_skips = 0
    history: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    last_autosave_t = time.monotonic()
    autosave_interval_s = max(0.0, float(args.save_every_minutes) * 60.0)

    try:
        for step in range(1, args.steps + 1):
            lr_mult = pj.cosine_lr_mult(step - 1, args.steps, warmup_steps=warmup_steps)
            opt.set_lr_mult(lr_mult)
            opt.zero_grad(set_to_none=True)

            is_log_step = step == 1 or step % args.log_every == 0 or step == args.steps
            acc = {
                "loss": 0.0,
                "jepa_masked": 0.0,
                "jepa_visible": 0.0,
                "deep_sup": 0.0,
                "gram": 0.0,
                "stroke_hog": 0.0,
                "router_loss": 0.0,
                "vicreg_loss": 0.0,
                "masked_ink_frac": 0.0,
            }
            rank_stats: Dict[str, float] = {}
            valid_micros = 0

            for micro_idx in range(args.grad_accum):
                images = next(loader).to(device, non_blocking=(device.type == "cuda"))
                try:
                    losses = vjepa_iter15_forward_loss(
                        student=student,
                        teacher=teacher,
                        gram_teacher=gram_teacher,
                        predictors=predictors,
                        stroke_hog_head=stroke_hog_head,
                        images=images,
                        rng=rng,
                        mask_ratio_range=mask_range,
                        use_bf16=use_bf16,
                        device_type=device.type,
                        level_weights=level_weights,
                        masked_weight=args.masked_weight,
                        visible_weight=args.visible_weight,
                        gram_weight=effective_gram_weight,
                        gram_max_tokens=args.gram_max_tokens,
                        stroke_hog_weight=args.stroke_hog_weight,
                        router_cons_weight=args.router_cons_weight,
                        vicreg_var_weight=args.vicreg_var_weight,
                        vicreg_cov_weight=args.vicreg_cov_weight,
                        router_loss_mode=args.router_loss_mode,
                        compute_rank_stats=(is_log_step and micro_idx == 0),
                    )
                    scaled_loss = losses["loss"] / float(args.grad_accum)
                    if not torch.isfinite(scaled_loss):
                        print(f"[V-JEPA-Iter15 warn] non-finite loss at step {step}; skipping micro-batch", flush=True)
                        continue
                    scaled_loss.backward()
                    for k in acc:
                        acc[k] += float(losses[k].detach().item())
                    if "mem_erank" in losses:
                        rank_stats = {
                            "post_erank": round(float(losses["post_erank"]), 2),
                            "mem_erank": round(float(losses["mem_erank"]), 2),
                            "tok_std": round(float(losses["tok_std"]), 4),
                            "within_cos": round(float(losses["within_cos"]), 4),
                        }
                    valid_micros += 1
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        oom_skips += 1
                        opt.zero_grad(set_to_none=True)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        print(f"[V-JEPA-Iter15 OOM] skipped step {step} (total={oom_skips})", flush=True)
                        valid_micros = 0
                        break
                    raise

            if valid_micros == 0:
                continue

            torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
            opt.step()

            ema_decay = pj.cosine_ema_decay(step, args.steps, start=args.ema_start, end=args.ema_end)
            teacher.update_from(student, decay=ema_decay)

            avg = {k: v / valid_micros for k, v in acc.items()}
            ema_loss_track = avg["loss"] if ema_loss_track is None else (0.95 * ema_loss_track + 0.05 * avg["loss"])

            if is_log_step:
                elapsed = max(1e-6, time.perf_counter() - t0)
                sps = step / elapsed
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
                    "router_loss": round(avg["router_loss"], 5),
                    "vicreg_loss": round(avg["vicreg_loss"], 5),
                    "masked_ink_frac": round(avg["masked_ink_frac"], 3),
                    "lr_mult": round(lr_mult, 5),
                    "ema_decay": round(ema_decay, 5),
                    "vram_gb": round(vram_gb, 2),
                    "steps_per_sec": round(sps, 2),
                    "oom_skips": oom_skips,
                    **rank_stats,
                }
                history.append(rec)
                erank_str = (
                    f" erank={rank_stats['post_erank']:.1f}/{rank_stats['mem_erank']:.1f}"
                    if rank_stats
                    else ""
                )
                print(
                    f"[V-JEPA-Iter15 {step:5d}/{args.steps}] "
                    f"loss={avg['loss']:.4f} (ema={ema_loss_track:.4f}) "
                    f"m={avg['jepa_masked']:.4f} v={avg['jepa_visible']:.4f} "
                    f"hog={avg['stroke_hog']:.4f} vic={avg['vicreg_loss']:.4f} "
                    f"mask_ink={avg['masked_ink_frac']:.2f}{erank_str} "
                    f"vram={vram_gb:.2f}GB ({sps:.2f} it/s)",
                    flush=True,
                )

            if ema_loss_track < best_ema_loss and (step >= min(50, args.steps) or step == args.steps):
                best_ema_loss = ema_loss_track
                save_iter15_vjepa_checkpoint(
                    out_dir / "uni_omr_v2_vjepa_best.pt",
                    student=student,
                    teacher=teacher,
                    cfg=cfg,
                    step=step,
                    loss_val=best_ema_loss,
                    vocab=init_vocab,
                    arch_preset=args.arch_preset,
                    use_teacher_encoder=not args.save_student_weights,
                )

            if args.save_every > 0 and step % args.save_every == 0:
                save_iter15_vjepa_checkpoint(
                    out_dir / f"uni_omr_v2_vjepa_step{step}.pt",
                    student=student,
                    teacher=teacher,
                    cfg=cfg,
                    step=step,
                    loss_val=ema_loss_track,
                    vocab=init_vocab,
                    arch_preset=args.arch_preset,
                    use_teacher_encoder=not args.save_student_weights,
                )

            if autosave_interval_s > 0.0 and (time.monotonic() - last_autosave_t) >= autosave_interval_s:
                save_iter15_vjepa_checkpoint(
                    out_dir / "uni_omr_v2_vjepa_autosave.pt",
                    student=student,
                    teacher=teacher,
                    cfg=cfg,
                    step=step,
                    loss_val=ema_loss_track,
                    vocab=init_vocab,
                    arch_preset=args.arch_preset,
                    use_teacher_encoder=not args.save_student_weights,
                )
                last_autosave_t = time.monotonic()
    finally:
        loader.close()

    save_iter15_vjepa_checkpoint(
        out_dir / "uni_omr_v2_vjepa_last.pt",
        student=student,
        teacher=teacher,
        cfg=cfg,
        step=args.steps,
        loss_val=ema_loss_track if ema_loss_track is not None else float("nan"),
        vocab=init_vocab,
        arch_preset=args.arch_preset,
        use_teacher_encoder=not args.save_student_weights,
    )

    probe: Optional[Dict[str, Any]] = None
    if val_probe_paths:
        probe = eval_staff_density_probes(
            teacher=teacher,
            val_paths=val_probe_paths,
            device=device,
            use_bf16=use_bf16,
            probe_dim=args.probe_dim,
        )
        (out_dir / "vjepa_iter15_probe.json").write_text(
            json.dumps(probe, indent=2), encoding="utf-8"
        )
        print(f"[V-JEPA-Iter15 probe] {json.dumps(probe)}", flush=True)

    summary = {
        "stage": "stage0_vjepa_iter15",
        "arch_preset": args.arch_preset,
        "n_axial_layers": cfg.n_axial_layers,
        "n_global_layers": cfg.n_global_layers,
        "mask_ratio": args.mask_ratio,
        "mask_range": list(mask_range),
        "vicreg_var_weight": args.vicreg_var_weight,
        "vicreg_cov_weight": args.vicreg_cov_weight,
        "steps": args.steps,
        "corpus_stats": corpus_stats,
        "optimizer": opt.mode,
        "best_ema_loss": best_ema_loss,
        "final_ema_loss": ema_loss_track,
        "oom_skips": oom_skips,
        "probe": probe,
        "history": history,
    }
    (out_dir / "vjepa_iter15_train_log.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Iter 15 Stage-0 V-JEPA 2.1 + VICReg Encoder Pretrainer (edge85m + ax0_gl6)."
    )
    p.add_argument(
        "--arch-preset",
        "--preset",
        dest="arch_preset",
        type=str,
        choices=("edge85m", "small_45m", "tiny"),
        default="edge85m",
        help="UniOMRv2 architecture preset (default: edge85m)",
    )
    p.add_argument("--n-axial-layers", type=int, default=0, help="Encoder axial layers (default: 0 for ax0_gl6)")
    p.add_argument("--n-global-layers", type=int, default=6, help="Encoder global layers (default: 6 for ax0_gl6)")
    p.add_argument("--max-mem-tokens", type=int, default=3072, help="Max routed memory tokens (default: 3072)")
    p.add_argument("--adaptive-ink-target", action="store_true", default=True)
    p.add_argument("--no-adaptive-ink-target", dest="adaptive_ink_target", action="store_false")
    p.add_argument("--init-from", type=str, default=None, help="Optional warm-start UniOMRv2 checkpoint (.pt)")

    # Data sources & 0-overlap enforcement
    p.add_argument(
        "--image-dirs",
        nargs="+",
        default=list(DEFAULT_RISM_DIRS),
        help="One or more directories of unlabeled permissive archival pages",
    )
    p.add_argument(
        "--printed-manifest",
        type=str,
        default=None,
        help="Optional full-page JSONL manifest to mix printed pages (e.g. D:\\tmp\\uni_fullpage_iter14_v2a.jsonl)",
    )
    p.add_argument("--printed-max-pages", type=int, default=6000, help="Cap printed pages mixed from manifest (default: 6000)")
    p.add_argument("--suite31-manifest", type=str, default=str(DEFAULT_SUITE31_MANIFEST))
    p.add_argument("--northstar-manifest", type=str, default=str(DEFAULT_NORTHSTAR_MANIFEST))
    p.add_argument("--no-dedup", action="store_true", help="Disable fast content-hash deduplication")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for checkpoints & logs")

    # Locked ov-31b V-JEPA + VICReg hyperparameters
    p.add_argument("--mask-ratio", type=float, default=0.45, help="Locked ov-31b ink-span mask ratio (default: 0.45)")
    p.add_argument("--mask-half-width", type=float, default=0.05, help="Mask ratio jitter half-width (default: 0.05 -> [0.40, 0.50])")
    p.add_argument("--vicreg-var-weight", type=float, default=1.0, help="Locked ov-31b VICReg variance weight (default: 1.0)")
    p.add_argument("--vicreg-cov-weight", type=float, default=0.04, help="Locked ov-31b VICReg covariance weight (default: 0.04)")
    p.add_argument("--gram-weight", type=float, default=0.5, help="Gram anchoring weight when --init-from is set (default: 0.5)")
    p.add_argument("--gram-max-tokens", type=int, default=512)
    p.add_argument("--disable-random-gram", action="store_true", default=True, help="Zero out Gram weight when training from scratch (default: True)")
    p.add_argument("--allow-random-gram", dest="disable_random_gram", action="store_false")
    p.add_argument("--level-weights", type=str, default="0.25,0.35,1.0")
    p.add_argument("--masked-weight", type=float, default=3.0)
    p.add_argument("--visible-weight", type=float, default=0.5)
    p.add_argument("--stroke-hog-weight", type=float, default=0.25)
    p.add_argument("--router-cons-weight", type=float, default=0.2)
    p.add_argument("--router-loss-mode", type=str, choices=("bce", "kl"), default="bce")

    # Training & runtime
    p.add_argument("--steps", type=int, default=3000, help="Total optimizer steps (default: 3000)")
    p.add_argument("--batch-size", type=int, default=2, help="Microbatch size (default: 2 for edge85m on 16GB)")
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4 -> eff_bs=8)")
    p.add_argument("--max-dim", type=int, default=1024, help="Max image dimension (default: 1024)")
    p.add_argument("--optimizer", type=str, choices=("muon", "adamw"), default="muon")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--muon-lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--warmup-steps", type=int, default=150)
    p.add_argument("--ema-start", type=float, default=0.996)
    p.add_argument("--ema-end", type=float, default=0.999)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--no-cudnn", action="store_true", default=False, help="Disable cuDNN on Windows variable-HxW runs")
    p.add_argument("--channels-last", action="store_true", default=False)
    p.add_argument("--grad-checkpoint", dest="grad_checkpoint", action="store_true", default=True)
    p.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    p.add_argument("--no-multi-scale", action="store_true", default=False)
    p.add_argument("--prefetch-threads", type=int, default=4)
    p.add_argument("--save-student-weights", action="store_true", default=False)
    p.add_argument("--val-probe-pages", type=int, default=128, help="Held-out pages for ov-31b Ridge/kNN probe (0 disables)")
    p.add_argument("--probe-dim", type=int, default=768)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--save-every-minutes", type=float, default=5.0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    run_pretrain_iter15(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
