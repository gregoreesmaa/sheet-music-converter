#!/usr/bin/env python3
"""Google Colab L4 Cloud Benchmark — Experiment 8: SpinQuant INT4 Parity & Multi-Target ONNX Export.

Mirrors the exact Experiment 8 protocol on full-size UniOMR v2 presets (`small_45m`, `edge85m`):
  1. Exact FP32 functional invariance (`max_abs_diff`, `mse`) of `fold_hadamard_v2`.
  2. Exact INT4 (`block_size=32`) logit MSE, KL divergence, max-abs diff, and top-1
     token agreement BEFORE vs AFTER SpinQuant Hadamard folding on 12 diverse real `suite31_hw`
     pages (`datasets/suite31_hw_eval.tar`) — under BOTH clean weights and
     12x outlier-stressed weights.
  3. Exact multi-target ONNX export (`export_uni_v2_onnx.py --targets all`) file sizes
     for `webgpu` (INT4 MatMulNBits), `npu` (INT8 DequantizeLinear), and `cpu`
     (INT8 dynamic), confirming the `< 55 MB` WebGPU budget.

Executed on a Google Colab L4 session (`exp8-spinquant`) via `colab exec -f`.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List, Tuple

EXP_ID = "2026-11-26T00-55-ablation-spinquant-int4-parity"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
BASELINE_SHA = "9d73170"

BUNDLE_PATH = Path("/content/exp8_bundle.tar.gz")
APP_MVD_DIR = Path("/content/app/mvd")
WORK_ROOT = Path("/content/work")
HF_TOKEN_PATH = Path("/tmp/.hf_token")


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_hf_token() -> str:
    if HF_TOKEN_PATH.exists():
        try:
            os.chmod(HF_TOKEN_PATH, 0o600)
        except OSError:
            pass
        tok = HF_TOKEN_PATH.read_text(encoding="utf-8").strip()
        if tok:
            os.environ["HF_TOKEN"] = tok
            return tok
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        raise RuntimeError("Missing HF_TOKEN (neither /tmp/.hf_token nor env HF_TOKEN is set).")
    return tok


def _extract_bundle() -> None:
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Expected uploaded bundle at {BUNDLE_PATH}")
    app_root = APP_MVD_DIR.parent
    if APP_MVD_DIR.exists():
        shutil.rmtree(APP_MVD_DIR)
    app_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(BUNDLE_PATH, "r:gz") as tf:
        tf.extractall(path=app_root)
    if not (APP_MVD_DIR / "uni_omr_v2.py").exists():
        raise RuntimeError(f"Bundle did not extract uni_omr_v2.py into {APP_MVD_DIR}")
    print(f"[exp8] Extracted bundle -> {APP_MVD_DIR}", flush=True)


def _hf_upload_json(api: Any, payload: Dict[str, Any], path_in_repo: str) -> None:
    tmp_path = Path("/tmp") / Path(path_in_repo).name
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(tmp_path),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO,
        repo_type="dataset",
        commit_message=f"exp8: upload {path_in_repo}",
    )


def _ensure_suite31_hw_pages(work_root: Path, token: str) -> List[Tuple[str, str, Path]]:
    """Download/extract datasets/suite31_hw_eval.tar from HF and select 12 diverse pages."""
    from huggingface_hub import hf_hub_download

    extract_dir = work_root / "datasets" / "suite31_hw_eval"
    marker = extract_dir / ".extracted_ok"
    if not marker.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        tar_cache = work_root / "datasets" / "_tar_cache"
        tar_cache.mkdir(parents=True, exist_ok=True)
        print("[exp8] Downloading datasets/suite31_hw_eval.tar from HF...", flush=True)
        tar_path = hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename="datasets/suite31_hw_eval.tar",
            local_dir=str(tar_cache),
            token=token,
        )
        print(f"[exp8] Extracting {tar_path} -> {extract_dir}...", flush=True)
        with tarfile.open(tar_path, "r:*") as tf:
            tf.extractall(path=extract_dir)
        marker.write_text(_utc_now() + "\n", encoding="utf-8")

    manifest_candidates = list(extract_dir.rglob("suite31_hw_manifest.json"))
    pages_by_bucket: Dict[str, List[Tuple[str, Path]]] = {
        "printed_clean": [],
        "printed_photo": [],
        "hw_clean": [],
        "hw_photo": [],
    }

    if manifest_candidates:
        mf_path = manifest_candidates[0]
        mf_dir = mf_path.parent
        mf = json.loads(mf_path.read_text(encoding="utf-8"))
        for piece in mf.get("pieces", []):
            slug = str(piece.get("slug", ""))
            is_hw = slug.startswith("hw-") or ("handwritten" in slug.lower()) or bool(piece.get("is_hw", False))
            for split in ("clean", "photo"):
                rel = piece.get(split, {}).get("file")
                if not rel:
                    continue
                cand = mf_dir / rel
                if not cand.exists():
                    hits = list(extract_dir.rglob(Path(rel).name))
                    cand = hits[0] if hits else cand
                if cand.exists():
                    bucket = f"{'hw' if is_hw else 'printed'}_{split}"
                    pages_by_bucket[bucket].append((f"{slug}:{split}", cand))

    total_found = sum(len(v) for v in pages_by_bucket.values())
    if total_found == 0:
        all_imgs = sorted(
            [p for p in extract_dir.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        )
        for p in all_imgs:
            low = str(p).lower()
            is_hw = "hw-" in p.name.lower() or "handwritten" in low
            split = "photo" if "photo" in low else "clean"
            bucket = f"{'hw' if is_hw else 'printed'}_{split}"
            pages_by_bucket[bucket].append((f"{p.stem}:{split}", p))

    selected: List[Tuple[str, str, Path]] = []
    for bucket in ("printed_clean", "printed_photo", "hw_clean", "hw_photo"):
        items = pages_by_bucket[bucket][:3]
        for label, path in items:
            selected.append((bucket, label, path))

    if not selected:
        raise RuntimeError(f"No images found under {extract_dir}")

    print(
        f"[exp8] Selected {len(selected)} real suite31_hw pages: "
        + ", ".join(f"{b}={sum(1 for x in selected if x[0] == b)}" for b in pages_by_bucket),
        flush=True,
    )
    return selected


def _load_image_tensor(path: Path, max_dim: int = 1024) -> Any:
    """Load RGB image, resize aspect-preserving to <= max_dim with H,W multiples of 16."""
    import numpy as np
    from PIL import Image
    import torch

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, float(max_dim) / float(max(h, w)))
        nh = max(128, int(round((h * scale) / 16.0)) * 16)
        nw = max(128, int(round((w * scale) / 16.0)) * 16)
        if (nw, nh) != (w, h):
            im = im.resize((nw, nh), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor


def _inject_realistic_outliers(
    model: Any,
    seed: int = 1234,
    gamma_scale: float = 12.0,
    weight_spike_mult: float = 12.0,
) -> Dict[str, Any]:
    """Inject deterministic channel-wise activation/weight outliers simulating converged Transformers."""
    import torch
    import torch.nn as nn

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    d_model = int(model.cfg.d_model)
    n_out_ch = max(2, int(round(d_model * 0.02)))
    outlier_idx = torch.randperm(d_model, generator=g)[:n_out_ch]

    n_norms_stressed = 0
    n_linears_stressed = 0

    with torch.no_grad():
        for layer in model.dec_layers:
            for norm in (layer.norm1, layer.norm2, layer.norm3):
                norm.weight.data[outlier_idx] *= gamma_scale
                n_norms_stressed += 1
        model.dec_norm.weight.data[outlier_idx] *= gamma_scale
        n_norms_stressed += 1

        for layer in model.dec_layers:
            in_linears: List[nn.Linear] = [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.cross_attn.q_proj,
                layer.ffn.w_gate_up,
            ]
            for lin in in_linears:
                std = float(lin.weight.data.std().item())
                signs = (torch.randint(0, 2, (lin.weight.shape[0], n_out_ch), generator=g) * 2 - 1).to(lin.weight.dtype)
                lin.weight.data[:, outlier_idx] += signs * (weight_spike_mult * std)
                n_linears_stressed += 1

        std_lm = float(model.lm_head.weight.data.std().item())
        signs_lm = (torch.randint(0, 2, (model.lm_head.weight.shape[0], n_out_ch), generator=g) * 2 - 1).to(
            model.lm_head.weight.dtype
        )
        model.lm_head.weight.data[:, outlier_idx] += signs_lm * (weight_spike_mult * std_lm)
        n_linears_stressed += 1

        head_dim = int(d_model // int(model.cfg.n_heads))
        hd_idx = torch.randperm(head_dim, generator=g)[: max(1, head_dim // 16)]
        for mod in model.encoder.modules():
            if hasattr(mod, "qkv_proj") and hasattr(mod, "o_proj") and hasattr(mod, "n_heads"):
                nh = int(mod.n_heads)
                std_o = float(mod.o_proj.weight.data.std().item())
                w_o = mod.o_proj.weight.data.view(d_model, nh, head_dim)
                for hi in hd_idx.tolist():
                    signs = (torch.randint(0, 2, (d_model, nh), generator=g) * 2 - 1).to(w_o.dtype)
                    w_o[:, :, hi] += signs * (weight_spike_mult * std_o)
                n_linears_stressed += 1

    return {
        "seed": seed,
        "gamma_scale": gamma_scale,
        "weight_spike_mult": weight_spike_mult,
        "n_outlier_channels": int(n_out_ch),
        "n_norms_stressed": n_norms_stressed,
        "n_linears_stressed": n_linears_stressed,
    }


def _eval_spinquant_parity(
    model_orig: Any,
    model_folded: Any,
    page_tensors: List[Tuple[str, str, Any]],
    sample_ids: Any,
    device: Any,
    block_size: int = 32,
) -> Dict[str, Any]:
    """Measure exact FP32 invariance + INT4 (block_size=32) MSE, KL, max_diff, top-1 agree."""
    import torch
    import torch.nn.functional as F

    from uni_omr_v2 import QATInt4ControllerV2

    m_orig = copy.deepcopy(model_orig).to(device=device, dtype=torch.float32).eval()
    m_fold = copy.deepcopy(model_folded).to(device=device, dtype=torch.float32).eval()
    ids_dev = sample_ids.to(device=device)

    per_page: List[Dict[str, Any]] = []

    with torch.no_grad():
        ref_cache: List[Tuple[str, str, Any, float, float]] = []
        for bucket, label, img_cpu in page_tensors:
            img_dev = img_cpu.to(device=device, dtype=torch.float32)
            logits_ref = m_orig(img_dev, ids_dev, return_aux=False, return_mtp=False)["logits"].float()
            logits_fold_fp32 = m_fold(img_dev, ids_dev, return_aux=False, return_mtp=False)["logits"].float()
            fp32_max = float((logits_fold_fp32 - logits_ref).abs().max().item())
            fp32_mse = float((logits_fold_fp32 - logits_ref).pow(2).mean().item())
            ref_cache.append((bucket, label, logits_ref.cpu(), fp32_max, fp32_mse))

        qat_orig = QATInt4ControllerV2(m_orig, block_size=block_size, enabled=True)
        qat_fold = QATInt4ControllerV2(m_fold, block_size=block_size, enabled=True)
        try:
            for (bucket, label, img_cpu), (_, _, logits_ref_cpu, fp32_max, fp32_mse) in zip(page_tensors, ref_cache):
                img_dev = img_cpu.to(device=device, dtype=torch.float32)
                logits_ref = logits_ref_cpu.to(device=device, dtype=torch.float32)
                logits_orig_q = m_orig(img_dev, ids_dev, return_aux=False, return_mtp=False)["logits"].float()
                logits_fold_q = m_fold(img_dev, ids_dev, return_aux=False, return_mtp=False)["logits"].float()

                err_orig = logits_orig_q - logits_ref
                err_fold = logits_fold_q - logits_ref

                orig_mse = float(err_orig.pow(2).mean().item())
                fold_mse = float(err_fold.pow(2).mean().item())
                orig_max = float(err_orig.abs().max().item())
                fold_max = float(err_fold.abs().max().item())

                p_ref = F.softmax(logits_ref, dim=-1)
                log_p_ref = F.log_softmax(logits_ref, dim=-1)
                log_p_orig = F.log_softmax(logits_orig_q, dim=-1)
                log_p_fold = F.log_softmax(logits_fold_q, dim=-1)
                orig_kl = float((p_ref * (log_p_ref - log_p_orig)).sum(dim=-1).mean().item())
                fold_kl = float((p_ref * (log_p_ref - log_p_fold)).sum(dim=-1).mean().item())

                top1_ref = logits_ref.argmax(dim=-1)
                orig_top1 = float((logits_orig_q.argmax(dim=-1) == top1_ref).float().mean().item())
                fold_top1 = float((logits_fold_q.argmax(dim=-1) == top1_ref).float().mean().item())

                per_page.append(
                    {
                        "bucket": bucket,
                        "page": label,
                        "hw": [int(img_cpu.shape[2]), int(img_cpu.shape[3])],
                        "fp32_max_abs_diff": fp32_max,
                        "fp32_mse": fp32_mse,
                        "orig_int4_mse": orig_mse,
                        "folded_int4_mse": fold_mse,
                        "orig_int4_max_diff": orig_max,
                        "folded_int4_max_diff": fold_max,
                        "orig_int4_kl": orig_kl,
                        "folded_int4_kl": fold_kl,
                        "orig_int4_top1_agree": orig_top1,
                        "folded_int4_top1_agree": fold_top1,
                    }
                )
        finally:
            qat_orig.disable()
            qat_fold.disable()

    del m_orig, m_fold
    torch.cuda.empty_cache()

    def _agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = max(1, len(rows))
        orig_mse = sum(r["orig_int4_mse"] for r in rows) / n
        fold_mse = sum(r["folded_int4_mse"] for r in rows) / n
        orig_kl = sum(r["orig_int4_kl"] for r in rows) / n
        fold_kl = sum(r["folded_int4_kl"] for r in rows) / n
        orig_top1 = sum(r["orig_int4_top1_agree"] for r in rows) / n
        fold_top1 = sum(r["folded_int4_top1_agree"] for r in rows) / n
        orig_max = max(r["orig_int4_max_diff"] for r in rows)
        fold_max = max(r["folded_int4_max_diff"] for r in rows)
        return {
            "n_pages": len(rows),
            "fp32_max_abs_diff_worst": max(r["fp32_max_abs_diff"] for r in rows),
            "fp32_mse_mean": sum(r["fp32_mse"] for r in rows) / n,
            "orig_int4_mse_mean": orig_mse,
            "folded_int4_mse_mean": fold_mse,
            "mse_reduction_ratio": orig_mse / max(fold_mse, 1e-12),
            "orig_int4_max_diff_worst": orig_max,
            "folded_int4_max_diff_worst": fold_max,
            "max_diff_reduction_ratio": orig_max / max(fold_max, 1e-12),
            "orig_int4_kl_mean": orig_kl,
            "folded_int4_kl_mean": fold_kl,
            "kl_reduction_ratio": orig_kl / max(fold_kl, 1e-12),
            "orig_int4_top1_agree_mean": orig_top1,
            "folded_int4_top1_agree_mean": fold_top1,
            "top1_agree_delta": fold_top1 - orig_top1,
        }

    by_bucket: Dict[str, Any] = {}
    for bucket in ("printed_clean", "printed_photo", "hw_clean", "hw_photo"):
        b_rows = [r for r in per_page if r["bucket"] == bucket]
        if b_rows:
            by_bucket[bucket] = _agg(b_rows)

    return {
        "block_size": int(block_size),
        "summary": _agg(per_page),
        "by_bucket": by_bucket,
        "per_page": per_page,
    }


def _run_onnx_export_all_targets(preset: str, model_folded_cpu: Any, cfg: Any) -> Dict[str, Any]:
    """Save folded checkpoint and invoke export_uni_v2_onnx.py --targets all in clean CPU subprocess."""
    import torch

    ckpt_dir = Path("/tmp/exp8_ckpts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{preset}_folded.pt"
    out_dir = Path(f"/tmp/exp8_onnx/{preset}")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    torch.save(
        {
            "arch": "uni_omr_v2",
            "preset": preset,
            "config": asdict(cfg),
            "model": model_folded_cpu.state_dict(),
        },
        str(ckpt_path),
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [
        sys.executable,
        str(APP_MVD_DIR / "export_uni_v2_onnx.py"),
        "--ckpt",
        str(ckpt_path),
        "--targets",
        "all",
        "--out-dir",
        str(out_dir),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    wall_s = time.perf_counter() - t0
    if proc.stdout:
        print(proc.stdout[-4000:], end="", flush=True)
    if proc.returncode != 0:
        if proc.stderr:
            print(f"[export-v2 ERROR stderr ({preset})]\n{proc.stderr[-4000:]}", flush=True)
        raise RuntimeError(f"export_uni_v2_onnx.py failed (rc={proc.returncode}) for preset={preset}")
    if proc.stderr:
        print(f"[export-v2 stderr ({preset})]\n{proc.stderr[-1000:]}", flush=True)

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    targets_summary: Dict[str, Any] = {}
    for tname, entry in manifest.get("targets", {}).items():
        enc_mb = entry["encoder_bytes"] / (1024.0 * 1024.0)
        dec_mb = entry["decoder_bytes"] / (1024.0 * 1024.0)
        tot_mb = entry["total_bytes"] / (1024.0 * 1024.0)
        targets_summary[tname] = {
            "quant_format": entry.get("quant_format"),
            "encoder_bytes": int(entry["encoder_bytes"]),
            "decoder_bytes": int(entry["decoder_bytes"]),
            "total_bytes": int(entry["total_bytes"]),
            "encoder_mb": round(enc_mb, 3),
            "decoder_mb": round(dec_mb, 3),
            "total_mb": round(tot_mb, 3),
            "under_55mb_budget": bool(tot_mb < 55.0),
        }

    shutil.rmtree(out_dir, ignore_errors=True)
    ckpt_path.unlink(missing_ok=True)

    return {
        "export_wall_s": round(wall_s, 2),
        "targets": targets_summary,
    }


def main() -> None:
    try:
        token = _load_hf_token()
        _extract_bundle()

        import onnx
        import onnxruntime as ort
        import torch
        from huggingface_hub import HfApi

        if str(APP_MVD_DIR) not in sys.path:
            sys.path.insert(0, str(APP_MVD_DIR))
        from spinquant_fold import fold_hadamard_v2
        import uni_omr_v2

        t_start = time.perf_counter()
        started_at = _utc_now()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        gpu_cc = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else [0, 0]
        gpu_vram_gb = (
            round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            if torch.cuda.is_available()
            else 0.0
        )

        print(
            f"[exp8] Started {EXP_ID} on Google Colab {gpu_name} ({gpu_vram_gb} GB, cc={gpu_cc}) "
            f"| torch={torch.__version__} onnx={onnx.__version__} ort={ort.__version__}",
            flush=True,
        )

        api = HfApi(token=token)
        run_ledger: Dict[str, Any] = {
            "experiment_id": EXP_ID,
            "status": "RUNNING",
            "baseline_sha": BASELINE_SHA,
            "platform": "Google Colab L4 (24 GB)",
            "colab_session": "exp8-spinquant",
            "gpu_name": gpu_name,
            "gpu_cc": gpu_cc,
            "gpu_vram_gb": gpu_vram_gb,
            "started_at": started_at,
            "updated_at": started_at,
        }
        _hf_upload_json(api, run_ledger, f"runs/{EXP_ID}.json")

        selected_pages = _ensure_suite31_hw_pages(WORK_ROOT, token=token)
        page_tensors = [(b, lbl, _load_image_tensor(p, max_dim=1024)) for b, lbl, p in selected_pages]

        g_tok = torch.Generator(device="cpu")
        g_tok.manual_seed(2026)
        n_base = uni_omr_v2.VOCAB_SIZE - 4
        sample_ids = torch.randint(0, n_base, (1, 128), generator=g_tok, dtype=torch.long)
        sample_ids[0, 0] = uni_omr_v2.BOS_ID
        sample_ids[0, 32] = uni_omr_v2.SYS_ID
        sample_ids[0, 64] = uni_omr_v2.SYS_ID
        sample_ids[0, 96] = uni_omr_v2.SYS_ID

        results_by_preset: Dict[str, Any] = {}

        for preset in ("small_45m", "edge85m"):
            t_p = time.perf_counter()
            print(f"\n[exp8] === Benchmarking preset: {preset} ===", flush=True)
            torch.manual_seed(42)
            cfg = (
                uni_omr_v2.UniOMRv2Config.edge85m()
                if preset == "edge85m"
                else uni_omr_v2.UniOMRv2Config.small_45m()
            )
            model_clean = uni_omr_v2.UniOMRv2Net(cfg)
            model_clean.strip_aux_heads(strip_mtp=False)
            model_clean.eval()

            total_params = sum(p.numel() for p in model_clean.parameters())
            print(f"[exp8] {preset}: {total_params:,} params ({total_params / 1e6:.2f}M)", flush=True)

            # 1. Clean weights
            model_clean_folded = copy.deepcopy(model_clean).eval()
            fold_summary = fold_hadamard_v2(model_clean_folded)
            print(f"[exp8] {preset} fold_hadamard_v2: {fold_summary}", flush=True)

            clean_parity = _eval_spinquant_parity(
                model_clean,
                model_clean_folded,
                page_tensors,
                sample_ids,
                device=device,
                block_size=32,
            )
            cs = clean_parity["summary"]
            print(
                f"[exp8] {preset} [clean] FP32 max_diff={cs['fp32_max_abs_diff_worst']:.3e} | "
                f"INT4 MSE: {cs['orig_int4_mse_mean']:.4e} -> {cs['folded_int4_mse_mean']:.4e} ({cs['mse_reduction_ratio']:.2f}x) | "
                f"KL: {cs['orig_int4_kl_mean']:.4e} -> {cs['folded_int4_kl_mean']:.4e} ({cs['kl_reduction_ratio']:.2f}x) | "
                f"Top1: {cs['orig_int4_top1_agree_mean']:.4f} -> {cs['folded_int4_top1_agree_mean']:.4f}",
                flush=True,
            )

            # 2. Outlier-stressed weights
            model_outlier = copy.deepcopy(model_clean).eval()
            outlier_meta = _inject_realistic_outliers(
                model_outlier, seed=1234, gamma_scale=12.0, weight_spike_mult=12.0
            )
            model_outlier_folded = copy.deepcopy(model_outlier).eval()
            fold_hadamard_v2(model_outlier_folded)

            outlier_parity = _eval_spinquant_parity(
                model_outlier,
                model_outlier_folded,
                page_tensors,
                sample_ids,
                device=device,
                block_size=32,
            )
            os_sum = outlier_parity["summary"]
            print(
                f"[exp8] {preset} [outlier] FP32 max_diff={os_sum['fp32_max_abs_diff_worst']:.3e} | "
                f"INT4 MSE: {os_sum['orig_int4_mse_mean']:.4e} -> {os_sum['folded_int4_mse_mean']:.4e} ({os_sum['mse_reduction_ratio']:.2f}x) | "
                f"KL: {os_sum['orig_int4_kl_mean']:.4e} -> {os_sum['folded_int4_kl_mean']:.4e} ({os_sum['kl_reduction_ratio']:.2f}x) | "
                f"Top1: {os_sum['orig_int4_top1_agree_mean']:.4f} -> {os_sum['folded_int4_top1_agree_mean']:.4f}",
                flush=True,
            )

            # 3. Multi-target ONNX export (--targets all)
            print(f"[exp8] {preset}: running export_uni_v2_onnx.py --targets all...", flush=True)
            onnx_export = _run_onnx_export_all_targets(preset, model_clean_folded, cfg)
            for tname, tinfo in onnx_export["targets"].items():
                print(
                    f"[exp8] {preset} ONNX [{tname}]: enc={tinfo['encoder_mb']:.2f}MB "
                    f"dec={tinfo['decoder_mb']:.2f}MB total={tinfo['total_mb']:.2f}MB "
                    f"(<55MB={tinfo['under_55mb_budget']}, quant={tinfo['quant_format']})",
                    flush=True,
                )

            results_by_preset[preset] = {
                "total_params": int(total_params),
                "param_millions": round(total_params / 1e6, 2),
                "d_model": int(cfg.d_model),
                "n_heads": int(cfg.n_heads),
                "n_kv_heads": int(cfg.n_kv_heads),
                "n_axial_layers": int(cfg.n_axial_layers),
                "n_global_layers": int(cfg.n_global_layers),
                "n_dec_layers": int(cfg.n_dec_layers),
                "fold_summary": fold_summary,
                "clean_weights": clean_parity,
                "outlier_stressed_weights": {
                    "injection_meta": outlier_meta,
                    **outlier_parity,
                },
                "onnx_export": onnx_export,
                "preset_wall_s": round(time.perf_counter() - t_p, 2),
            }

            del model_clean, model_clean_folded, model_outlier, model_outlier_folded
            torch.cuda.empty_cache()

        total_wall_s = round(time.perf_counter() - t_start, 2)
        finished_at = _utc_now()

        report: Dict[str, Any] = {
            "experiment_id": EXP_ID,
            "baseline_sha": BASELINE_SHA,
            "platform": "Google Colab L4 (24 GB)",
            "colab_session": "exp8-spinquant",
            "gpu_name": gpu_name,
            "gpu_cc": gpu_cc,
            "gpu_vram_gb": gpu_vram_gb,
            "torch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "started_at": started_at,
            "finished_at": finished_at,
            "total_wall_s": total_wall_s,
            "n_suite31_hw_pages": len(page_tensors),
            "pages": [{"bucket": b, "page": lbl} for b, lbl, _ in page_tensors],
            "presets": results_by_preset,
        }

        eval_dir = WORK_ROOT / "evals"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_path = eval_dir / f"{EXP_ID}.json"
        eval_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        _hf_upload_json(api, report, f"evals/{EXP_ID}.json")
        run_ledger.update(
            {
                "status": "COMPLETED",
                "updated_at": finished_at,
                "finished_at": finished_at,
                "total_wall_s": total_wall_s,
                "eval_artifact": f"evals/{EXP_ID}.json",
                "headline": {
                    p: {
                        "param_millions": results_by_preset[p]["param_millions"],
                        "clean_fp32_max_diff": results_by_preset[p]["clean_weights"]["summary"]["fp32_max_abs_diff_worst"],
                        "clean_mse_ratio": round(results_by_preset[p]["clean_weights"]["summary"]["mse_reduction_ratio"], 3),
                        "outlier_fp32_max_diff": results_by_preset[p]["outlier_stressed_weights"]["summary"]["fp32_max_abs_diff_worst"],
                        "outlier_mse_ratio": round(results_by_preset[p]["outlier_stressed_weights"]["summary"]["mse_reduction_ratio"], 3),
                        "outlier_kl_ratio": round(results_by_preset[p]["outlier_stressed_weights"]["summary"]["kl_reduction_ratio"], 3),
                        "outlier_top1_before": round(results_by_preset[p]["outlier_stressed_weights"]["summary"]["orig_int4_top1_agree_mean"], 4),
                        "outlier_top1_after": round(results_by_preset[p]["outlier_stressed_weights"]["summary"]["folded_int4_top1_agree_mean"], 4),
                        "webgpu_total_mb": results_by_preset[p]["onnx_export"]["targets"]["webgpu"]["total_mb"],
                        "npu_total_mb": results_by_preset[p]["onnx_export"]["targets"]["npu"]["total_mb"],
                        "cpu_total_mb": results_by_preset[p]["onnx_export"]["targets"]["cpu"]["total_mb"],
                        "webgpu_under_55mb": results_by_preset[p]["onnx_export"]["targets"]["webgpu"]["under_55mb_budget"],
                    }
                    for p in ("small_45m", "edge85m")
                },
            }
        )
        _hf_upload_json(api, run_ledger, f"runs/{EXP_ID}.json")

        print(f"\n[exp8] COMPLETED in {total_wall_s:.1f}s! Uploaded evals/{EXP_ID}.json & runs/{EXP_ID}.json to HF.", flush=True)
        print("[exp8] HEADLINE_JSON_START", flush=True)
        print(json.dumps(run_ledger["headline"], indent=2), flush=True)
        print("[exp8] HEADLINE_JSON_END", flush=True)
    finally:
        try:
            HF_TOKEN_PATH.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
