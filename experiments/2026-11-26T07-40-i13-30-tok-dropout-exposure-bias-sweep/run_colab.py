#!/usr/bin/env python3
"""Iter 13 Exp 30 (Colab L4): Decoder Token Dropout (--tok-dropout) Zero-Cost Exposure-Bias Sweep.

Experiment ID: 2026-11-26T07-40-i13-30-tok-dropout-exposure-bias-sweep
Isolated variable: --tok-dropout in {0.00, 0.05, 0.15, 0.25} on UniOMR v2 small_45m
(0% training-time compute overhead vs CASS's 2x decoder forward cost).

Evaluates long-horizon proxy_val (eval_max_len=1024) with length-bucketed Token F1
(short <= 256, medium 257..512, long > 512), mean_len_ratio, and early_eos_rate.

Uploads to Hugging Face Private Dataset Hub (gregoreesmaa/sheet-music-converter-omr):
  - evals/2026-11-26T07-40-i13-30-tok-dropout-exposure-bias-sweep.json
  - runs/2026-11-26T07-40-i13-30-tok-dropout-exposure-bias-sweep.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T07-40-i13-30-tok-dropout-exposure-bias-sweep"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
DATASET_TAR_PATH = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"
BASELINE_SHA = "9303e51"

BUNDLE_CANDIDATES = (
    Path("/content/bundle.tar.gz"),
    Path("/content/ov_bundle.tar.gz"),
    Path("/content/repo_bundle.tar.gz"),
)
WORKSPACE_ROOT = Path("/workspace/sheet-music-converter")
VOL_ROOT = Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

TOKDROP_ARMS: Tuple[Tuple[str, float], ...] = (
    ("tokdrop_0p00", 0.00),
    ("tokdrop_0p05", 0.05),
    ("tokdrop_0p15", 0.15),
    ("tokdrop_0p25", 0.25),
)

DEFAULT_STEPS = 600
DEFAULT_SEED = 42
BASE_LR = 3.0e-4
WARMUP_STEPS = 60
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 2048
EVAL_MAX_LEN = 1024
SAVE_EVERY_MINUTES = 5.0


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _purge_hf_token_file() -> None:
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
            print(f"[{EXP_ID}] Deleted {HF_TOKEN_FILE}", flush=True)
        except OSError:
            pass


def _read_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and HF_TOKEN_FILE.exists():
        try:
            os.chmod(HF_TOKEN_FILE, 0o600)
        except OSError:
            pass
        token = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(
            "HF token not found in /tmp/.hf_token or HF_TOKEN environment variable."
        )
    return token


def _locate_or_extract_mvd() -> Path:
    direct_candidates = (
        WORKSPACE_ROOT / "tmp-train" / "mvd",
        Path("/workspace/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path.cwd() / "tmp-train" / "mvd",
    )
    for cand in direct_candidates:
        if (cand / "train_uni_v2.py").is_file():
            return cand

    for bundle in BUNDLE_CANDIDATES:
        if bundle.is_file():
            WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
            print(f"[{EXP_ID}] Extracting {bundle} -> {WORKSPACE_ROOT}...", flush=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=WORKSPACE_ROOT)
            for cand in (
                WORKSPACE_ROOT / "tmp-train" / "mvd",
                WORKSPACE_ROOT / "sheet-music-converter" / "tmp-train" / "mvd",
                WORKSPACE_ROOT / "mvd",
            ):
                if (cand / "train_uni_v2.py").is_file():
                    return cand

    raise FileNotFoundError(
        "Could not locate tmp-train/mvd/train_uni_v2.py. Upload code bundle to /content/bundle.tar.gz."
    )


def _ensure_deps() -> None:
    needed: List[str] = []
    for mod, pkg in (
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("cv2", "opencv-python-headless"),
        ("tqdm", "tqdm"),
    ):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        cmd = [sys.executable, "-m", "pip", "install", "-q", *needed]
        print(f"[{EXP_ID}] Installing lightweight deps: {' '.join(needed)}", flush=True)
        subprocess.check_call(cmd)


def _ensure_proxy_dataset(vol_root: Path, hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_dir = vol_root / "datasets" / "iter10_5_proxy_2k"
    sentinel = ds_dir / ".unpacked_ok"
    train_abs = ds_dir / "proxy_train_abs.jsonl"
    val_abs = ds_dir / "proxy_val_abs.jsonl"

    if sentinel.is_file() and train_abs.is_file() and val_abs.is_file():
        print(f"[{EXP_ID}] Using cached proxy dataset at {ds_dir}", flush=True)
        return train_abs, val_abs

    ds_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{EXP_ID}] Downloading {DATASET_TAR_PATH} from {HF_REPO_ID}...", flush=True)
    t0 = time.perf_counter()
    local_tar = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=DATASET_TAR_PATH,
        repo_type="dataset",
        token=hf_token,
        local_dir="/tmp/hf_dl",
    )
    print(f"[{EXP_ID}] Downloaded in {time.perf_counter() - t0:.1f}s -> extracting...", flush=True)
    with tarfile.open(local_tar, "r:*") as tf:
        tf.extractall(path=ds_dir)
    try:
        os.remove(local_tar)
    except OSError:
        pass

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
        print(f"[{EXP_ID}] Wrote {len(out_lines)} abs rows -> {dst_path.name}", flush=True)

    sentinel.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
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


def _eval_proxy_val_exposure_bias(
    ckpt_path: Path,
    val_manifest: Path,
    device: Any,
    *,
    fullpage_max_dim: int = FULLPAGE_MAX_DIM,
    max_seq_len: int = MAX_SEQ_LEN,
    eval_max_len: int = EVAL_MAX_LEN,
) -> Dict[str, Any]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    import train_uni_v2 as tv2
    from uni_omr import (
        SYS_TOKEN,
        UniFullPageDataset,
        _images_to_device_float,
        collate_uni_batch,
        compute_token_f1,
        decode_uni,
        levenshtein_distance,
    )
    from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net

    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    raw_cfg = ck.get("cfg") or ck.get("config") or {}
    valid = set(UniOMRv2Config.__dataclass_fields__.keys())
    cfg_kwargs = {k: v for k, v in dict(raw_cfg).items() if k in valid}
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
    n_samples = 0
    total_edits = 0
    total_ref = 0
    sys_exact = 0
    all_f1: List[float] = []
    len_ratios: List[float] = []
    early_eos = 0
    bucket_f1: Dict[str, List[float]] = {"short_le_256": [], "medium_257_512": [], "long_gt_512": []}

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
                label_smoothing=0.0,
                use_bf16=True,
                device=device,
            )
            sum_loss += float(losses["loss"].item()) * bs
            sum_ce += float(losses["ce_loss"].item()) * bs
            sum_mtp += float(losses["mtp_loss"].item()) * bs

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_ids = tv2.generate_mtp_speculative_v2(
                    model, images, max_len=eval_max_len, return_info=False
                )
            pred_cpu = pred_ids.detach().cpu().tolist()
            ref_cpu = batch["seq"].detach().cpu().tolist()

            for p_row, r_row in zip(pred_cpu, ref_cpu):
                pred_toks = decode_uni(p_row, skip_special=True)
                ref_toks = decode_uni(r_row, skip_special=True)
                ref_len = max(1, len(ref_toks))
                pred_len = len(pred_toks)
                edits = levenshtein_distance(ref_toks, pred_toks)
                total_edits += edits
                total_ref += len(ref_toks)
                _, _, f1 = compute_token_f1(ref_toks, pred_toks)
                all_f1.append(float(f1))
                if pred_toks.count(SYS_TOKEN) == ref_toks.count(SYS_TOKEN):
                    sys_exact += 1
                ratio = float(pred_len) / float(min(ref_len, eval_max_len))
                len_ratios.append(ratio)
                if ratio < 0.75:
                    early_eos += 1
                if ref_len <= 256:
                    bucket_f1["short_le_256"].append(float(f1))
                elif ref_len <= 512:
                    bucket_f1["medium_257_512"].append(float(f1))
                else:
                    bucket_f1["long_gt_512"].append(float(f1))
                n_samples += 1

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "n_val_samples": n_samples,
        "val_loss": round(sum_loss / max(1, n_samples), 6),
        "val_ce_loss": round(sum_ce / max(1, n_samples), 6),
        "val_mtp_loss": round(sum_mtp / max(1, n_samples), 6),
        "val_ser": round(total_edits / max(1, total_ref), 6),
        "val_token_f1": round(float(np.mean(all_f1)) if all_f1 else 0.0, 6),
        "val_sys_count_exact": round(sys_exact / max(1, n_samples), 6),
        "mean_len_ratio": round(float(np.mean(len_ratios)) if len_ratios else 0.0, 4),
        "early_eos_rate": round(early_eos / max(1, n_samples), 4),
        "bucket_token_f1": {
            k: round(float(np.mean(v)) if v else 0.0, 6) for k, v in bucket_f1.items()
        },
        "bucket_counts": {k: len(v) for k, v in bucket_f1.items()},
        "eval_max_len": eval_max_len,
    }


def _summarize_per_100(
    history: List[Dict[str, Any]],
    grad_norms: List[float],
    step_times_ms: List[float],
) -> List[Dict[str, Any]]:
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
                "grad_norm_mean": round(float(np.mean(g_slice)) if g_slice else 0.0, 5),
                "grad_norm_p95": round(float(np.percentile(g_slice, 95)) if g_slice else 0.0, 5),
                "step_time_ms_mean": round(float(np.mean(t_slice)) if t_slice else 0.0, 2),
            }
        )
    return buckets


def _run_arm(
    arm_name: str,
    tok_dropout: float,
    train_manifest: Path,
    val_manifest: Path,
    out_root: Path,
    *,
    steps: int,
    seed: int,
) -> Dict[str, Any]:
    import numpy as np
    import torch
    import train_uni_v2 as tv2

    arm_dir = out_root / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    arm_done_json = arm_dir / "arm_result.json"
    if arm_done_json.is_file():
        print(f"[{arm_name}] Found cached arm_result.json; loading.", flush=True)
        return json.loads(arm_done_json.read_text(encoding="utf-8"))

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

    cli_argv = [
        "--preset", "small_45m",
        "--optimizer", "muon",
        "--steps", str(steps),
        "--seed", str(seed),
        "--batch-size", "2",
        "--grad-accum", "4",
        "--curriculum", "fullpage",
        "--fullpage-manifest", str(train_manifest),
        "--stratified-fullpage",
        "--fullpage-max-dim", str(FULLPAGE_MAX_DIM),
        "--max-seq-len", str(MAX_SEQ_LEN),
        "--multi-scale",
        "--augment",
        "--aug-prob", "0.50",
        "--tok-dropout", f"{tok_dropout:.4f}",
        "--cass-prob", "0.0",
        "--lr-schedule", "wsd",
        "--lr", f"{BASE_LR:.6g}",
        "--warmup-steps", str(WARMUP_STEPS),
        "--wsd-decay-frac", str(WSD_DECAY_FRAC),
        "--min-lr-ratio", str(MIN_LR_RATIO),
        "--grad-checkpoint",
        "--channels-last",
        "--bf16",
        "--gpu-ema-decay", "0.99",
        "--gpu-ema-every", "1",
        "--save-every-minutes", str(SAVE_EVERY_MINUTES),
        "--val-count", "0",
        "--eval-every", "0",
        "--log-every", "50",
        "--num-workers", "0",
        "--prefetch-threads", "4",
        "--device", "cuda",
        "--out-dir", str(arm_dir),
    ]

    print(
        f"\n==================== ARM: {arm_name} (tok_dropout={tok_dropout:.2f}) ====================",
        flush=True,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    args = tv2.build_arg_parser().parse_args(cli_argv)

    torch.nn.utils.clip_grad_norm_ = _wrapped_clip
    t_start = time.perf_counter()
    last_t[0] = t_start
    try:
        train_res = tv2.run_train_v2(args)
    finally:
        torch.nn.utils.clip_grad_norm_ = orig_clip

    wall_time_s = time.perf_counter() - t_start
    peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    per_100 = _summarize_per_100(train_res["history"], grad_norms, step_times_ms)
    device = torch.device("cuda")

    val_raw = _eval_proxy_val_exposure_bias(arm_dir / "uni_omr_v2_latest.pt", val_manifest, device)
    val_ema = _eval_proxy_val_exposure_bias(arm_dir / "uni_omr_v2_ema_latest.pt", val_manifest, device)

    arm_payload: Dict[str, Any] = {
        "arm": arm_name,
        "tok_dropout": float(tok_dropout),
        "steps": int(steps),
        "seed": int(seed),
        "wall_time_s": round(wall_time_s, 2),
        "steps_per_sec": round(steps / max(1e-6, wall_time_s), 4),
        "peak_vram_alloc_gb": round(peak_alloc_gb, 3),
        "grad_norm_p95_overall": round(float(np.percentile(grad_norms, 95)) if grad_norms else 0.0, 5),
        "per_100_steps": per_100,
        "proxy_val_raw": val_raw,
        "proxy_val_ema": val_ema,
    }
    _atomic_write_json(arm_done_json, arm_payload)

    gc.collect()
    torch.cuda.empty_cache()
    return arm_payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=f"Colab L4 runner for {EXP_ID}")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        hf_token = _read_hf_token()
        _ensure_deps()
        mvd_dir = _locate_or_extract_mvd()
        if str(mvd_dir) not in sys.path:
            sys.path.insert(0, str(mvd_dir))

        import torch
        from huggingface_hub import HfApi

        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)

        assert torch.cuda.is_available(), "CUDA GPU required"
        cc = torch.cuda.get_device_capability()
        gpu_name = torch.cuda.get_device_name(0)
        assert cc[0] >= 8, f"Native BF16 requires Ampere+ (cc>=8), got {gpu_name} cc={cc}"
        assert hasattr(torch.optim, "Muon"), f"Native torch.optim.Muon missing in torch=={torch.__version__}"

        print(
            f"[{EXP_ID}] GPU={gpu_name} (cc={cc[0]}.{cc[1]}) | torch={torch.__version__} | "
            f"steps={args.steps} | seed={args.seed}",
            flush=True,
        )

        train_manifest, val_manifest = _ensure_proxy_dataset(VOL_ROOT, hf_token)
        out_root = VOL_ROOT / "ckpts" / EXP_ID
        out_root.mkdir(parents=True, exist_ok=True)

        arms_out: Dict[str, Any] = {}
        for arm_name, td in TOKDROP_ARMS:
            arms_out[arm_name] = _run_arm(
                arm_name,
                td,
                train_manifest,
                val_manifest,
                out_root,
                steps=args.steps,
                seed=args.seed,
            )

        ctrl = arms_out["tokdrop_0p05"]["proxy_val_ema"]
        best_ema_f1 = max(arms_out.keys(), key=lambda k: arms_out[k]["proxy_val_ema"]["val_token_f1"])
        best_long_f1 = max(
            arms_out.keys(),
            key=lambda k: arms_out[k]["proxy_val_ema"]["bucket_token_f1"]["long_gt_512"],
        )
        deltas_vs_ctrl = {
            k: {
                "ema_delta_val_f1": round(v["proxy_val_ema"]["val_token_f1"] - ctrl["val_token_f1"], 6),
                "ema_delta_long_f1": round(
                    v["proxy_val_ema"]["bucket_token_f1"]["long_gt_512"]
                    - ctrl["bucket_token_f1"]["long_gt_512"],
                    6,
                ),
                "ema_delta_val_ser": round(v["proxy_val_ema"]["val_ser"] - ctrl["val_ser"], 6),
                "ema_early_eos_rate": v["proxy_val_ema"]["early_eos_rate"],
            }
            for k, v in arms_out.items()
        }

        finished_at = datetime.now(timezone.utc).isoformat()
        total_wall_s = round(sum(v["wall_time_s"] for v in arms_out.values()), 2)
        summary = {
            "winner_by_ema_token_f1": best_ema_f1,
            "winner_by_long_gt_512_f1": best_long_f1,
            "control_arm": "tokdrop_0p05",
            "deltas_vs_control_ema": deltas_vs_ctrl,
        }

        eval_payload = {
            "exp_id": EXP_ID,
            "baseline_sha": BASELINE_SHA,
            "dataset": DATASET_TAR_PATH,
            "provider": "colab_l4",
            "gpu": gpu_name,
            "compute_capability": f"{cc[0]}.{cc[1]}",
            "torch_version": torch.__version__,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "steps_per_arm": args.steps,
            "seed": args.seed,
            "arms": arms_out,
            "summary": summary,
        }
        run_ledger = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "provider": "colab_l4",
            "gpu": gpu_name,
            "baseline_sha": BASELINE_SHA,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "total_wall_time_s": total_wall_s,
            "eval_hf_path": HF_EVAL_PATH,
            "summary": summary,
        }

        eval_local = out_root / f"{EXP_ID}_eval.json"
        run_local = out_root / f"{EXP_ID}_run.json"
        _atomic_write_json(eval_local, eval_payload)
        _atomic_write_json(run_local, run_ledger)

        api = HfApi(token=hf_token)
        for src, dst in ((eval_local, HF_EVAL_PATH), (run_local, HF_RUN_PATH)):
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=dst,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"exp({EXP_ID}): upload {dst}",
            )
            print(f"[{EXP_ID}] Uploaded {dst} to {HF_REPO_ID}", flush=True)

        print(f"\n==================== {EXP_ID} SUMMARY ====================", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    finally:
        _purge_hf_token_file()


if __name__ == "__main__":
    raise SystemExit(main())
