#!/usr/bin/env python3
"""Google Colab L4 runner for Overnight Exp 22: fullpage_frac Curriculum Sweep.

Experiment ID: 2026-11-26T01-38-ov-22-fullpage-frac-sweep
Controlled 3-Arm Ablation on UniOMR v2 small_45m (BF16, seed=42, lr=3e-4, 800 steps):
  - Arm A (fp_070): --curriculum mixed --fullpage-frac 0.70
  - Arm B (fp_085): --curriculum mixed --fullpage-frac 0.85
  - Arm C (fp_100): --curriculum fullpage --fullpage-frac 1.00

Uploads results to Hugging Face Private Dataset Hub (gregoreesmaa/sheet-music-converter-omr):
  - evals/2026-11-26T01-38-ov-22-fullpage-frac-sweep.json
  - runs/2026-11-26T01-38-ov-22-fullpage-frac-sweep.json
"""

from __future__ import annotations

import gc
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T01-38-ov-22-fullpage-frac-sweep"
BASELINE_SHA = "164852f"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

BUNDLE_CANDIDATES = (
    Path("/content/repo_bundle.tar.gz"),
    Path("/content/bundle.tar.gz"),
)
WORKSPACE_ROOT = Path("/workspace")
VOL_ROOT = Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

# Matched training hyperparameters across all 3 arms
TRAIN_STEPS = 800
SEED = 42
LR = 3e-4
WARMUP_STEPS = 80
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
BATCH_SIZE = 2
STRIP_BATCH_SIZE = 4
GRAD_ACCUM = 4
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
VAL_EVAL_MAX_LEN = 512
SAVE_EVERY_MINUTES = 5.0
MAX_STRIP_SOURCE_PAGES = 600

ARMS: List[Tuple[str, str, float]] = [
    ("fp_070", "mixed", 0.70),
    ("fp_085", "mixed", 0.85),
    ("fp_100", "fullpage", 1.00),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _purge_hf_token_file() -> None:
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass


def _read_and_purge_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and HF_TOKEN_FILE.exists():
        try:
            os.chmod(HF_TOKEN_FILE, 0o600)
        except OSError:
            pass
        token = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
    _purge_hf_token_file()
    if not token:
        raise RuntimeError("HF token not found in /tmp/.hf_token or HF_TOKEN env var")
    return token


def _find_mvd_dir() -> Path:
    for cand in (
        WORKSPACE_ROOT / "sheet-music-converter" / "tmp-train" / "mvd",
        WORKSPACE_ROOT / "tmp-train" / "mvd",
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path.cwd() / "tmp-train" / "mvd",
    ):
        if (cand / "train_uni_v2.py").is_file():
            return cand
    for bundle in BUNDLE_CANDIDATES:
        if bundle.is_file():
            WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
            print(f"[{EXP_ID}] Extracting {bundle} -> {WORKSPACE_ROOT}...", flush=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=WORKSPACE_ROOT)
            for p in WORKSPACE_ROOT.rglob("train_uni_v2.py"):
                return p.parent
    raise FileNotFoundError(
        "Could not locate tmp-train/mvd/train_uni_v2.py. Upload repo bundle to /content/repo_bundle.tar.gz."
    )


def _setup_environment() -> Path:
    mvd_dir = _find_mvd_dir()
    needed_pkgs: List[str] = []
    for mod, pkg in (
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
    ):
        try:
            __import__(mod)
        except ImportError:
            needed_pkgs.append(pkg)
    if needed_pkgs:
        print(f"[{EXP_ID}] Installing lightweight packages: {needed_pkgs}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed_pkgs])
    if str(mvd_dir) not in sys.path:
        sys.path.insert(0, str(mvd_dir))
    return mvd_dir


def _ensure_proxy_dataset(hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_dir = VOL_ROOT / "datasets" / "iter10_5_proxy_2k"
    train_abs = ds_dir / "proxy_train_abs.jsonl"
    val_abs = ds_dir / "proxy_val_abs.jsonl"
    if train_abs.is_file() and val_abs.is_file():
        print(f"[{EXP_ID}] Using cached proxy dataset at {ds_dir}", flush=True)
        return train_abs, val_abs

    ds_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR}...", flush=True)
    t0 = time.perf_counter()
    tar_p = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_PROXY_TAR,
        repo_type="dataset",
        token=hf_token,
        local_dir="/tmp/hf_dl",
    )
    print(f"[{EXP_ID}] Downloaded in {time.perf_counter() - t0:.1f}s -> extracting...", flush=True)
    with tarfile.open(tar_p, "r:*") as tf:
        tf.extractall(path=ds_dir)
    try:
        os.remove(tar_p)
    except OSError:
        pass

    for src_name, dst_path in (("proxy_train.jsonl", train_abs), ("proxy_val.jsonl", val_abs)):
        src_path = ds_dir / src_name
        if not src_path.is_file():
            raise FileNotFoundError(f"Expected {src_path} inside {HF_PROXY_TAR}")
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
        print(f"[{EXP_ID}] Wrote {len(out_lines)} rows -> {dst_path.name}", flush=True)
    return train_abs, val_abs


def _ensure_derived_strips(train_manifest: Path) -> Tuple[Path, Path, Dict[str, int]]:
    """Slice deterministic aligned system strips from proxy_train.jsonl for mixed-curriculum arms."""
    from PIL import Image

    strips_root = VOL_ROOT / "datasets" / "iter10_5_proxy_2k_strips"
    pr_dir = strips_root / "printed"
    hw_dir = strips_root / "hw"
    sentinel = strips_root / ".strips_ok.json"
    if sentinel.is_file() and pr_dir.is_dir() and hw_dir.is_dir():
        meta = json.loads(sentinel.read_text(encoding="utf-8"))
        print(f"[{EXP_ID}] Using cached derived strips: {meta}", flush=True)
        return pr_dir, hw_dir, meta

    pr_dir.mkdir(parents=True, exist_ok=True)
    hw_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for raw in train_manifest.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            rows.append(json.loads(raw))

    rng = random.Random(SEED)
    rng.shuffle(rows)
    selected = rows[:MAX_STRIP_SOURCE_PAGES]

    n_pr = 0
    n_hw = 0
    t0 = time.perf_counter()
    for page_idx, rec in enumerate(selected):
        img_p = Path(rec["image"])
        beks = [Path(b) for b in rec.get("bekerns", []) if Path(b).is_file()]
        if not img_p.is_file() or not beks:
            continue
        domain = str(rec.get("domain", "")).strip().lower()
        is_hw = ("hw" in domain) or ("hand" in domain) or ("rism" in domain) or ("debussy" in domain)
        target_dir = hw_dir if is_hw else pr_dir

        with Image.open(img_p) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            n_sys = len(beks)
            band_h = float(h) / float(max(1, n_sys))
            pad_y = int(round(band_h * 0.08))
            for s_idx, bek_p in enumerate(beks):
                y0 = max(0, int(round(s_idx * band_h)) - pad_y)
                y1 = min(h, int(round((s_idx + 1) * band_h)) + pad_y)
                if y1 - y0 < 16:
                    continue
                stem = f"p{page_idx:04d}_s{s_idx:02d}"
                crop = rgb.crop((0, y0, w, y1))
                crop.save(target_dir / f"{stem}.png")
                shutil.copyfile(bek_p, target_dir / f"{stem}-bekern.txt")
                if is_hw:
                    n_hw += 1
                else:
                    n_pr += 1

    meta = {
        "source_pages": len(selected),
        "printed_strips": n_pr,
        "hw_strips": n_hw,
        "build_sec": round(time.perf_counter() - t0, 2),
    }
    _atomic_write_json(sentinel, meta)
    print(f"[{EXP_ID}] Built derived strips: {meta}", flush=True)
    return pr_dir, hw_dir, meta


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


def _is_hw_domain(domain: str) -> bool:
    d = domain.lower()
    return ("hw" in d) or ("hand" in d) or ("rism" in d) or ("debussy" in d)


def _eval_subset(
    model: Any,
    pairs: List[Tuple[Path, List[Path], str]],
    device: Any,
) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    import train_uni_v2 as tv2
    from uni_omr import UniFullPageDataset, _images_to_device_float, collate_uni_batch, eval_dataset

    if not pairs:
        return {"n_pages": 0}

    val_ds = UniFullPageDataset(
        pairs,
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
    sum_mtp = 0.0
    sum_hog = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            bs = int(batch["seq"].shape[0])
            images = _images_to_device_float(batch["images"], device, non_blocking=(device.type == "cuda"))
            seq = batch["seq"].to(device, non_blocking=(device.type == "cuda"))
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
                use_bf16=(device.type == "cuda"),
                device=device,
            )
            sum_loss += float(losses["loss"].item()) * bs
            sum_ce += float(losses["ce_loss"].item()) * bs
            sum_mtp += float(losses["mtp_loss"].item()) * bs
            sum_hog += float(losses["stroke_hog_loss"].item()) * bs
            n_samples += bs

    greedy = eval_dataset(
        model,
        val_ds,
        device=device,
        max_len=VAL_EVAL_MAX_LEN,
        batch_size=2,
        use_bf16=(device.type == "cuda"),
    )
    denom = float(max(1, n_samples))
    return {
        "n_pages": n_samples,
        "val_loss": round(sum_loss / denom, 6),
        "val_ce": round(sum_ce / denom, 6),
        "val_mtp": round(sum_mtp / denom, 6),
        "val_stroke_hog": round(sum_hog / denom, 6),
        "ser": round(float(greedy["ser"]), 6),
        "token_f1": round(float(greedy["token_f1"]), 6),
        "sys_count_exact": round(float(greedy["sys_count_exact"]), 6),
        "mean_compression_x": round(float(greedy["mean_compression_x"]), 4),
    }


def _eval_checkpoint(ckpt_path: Path, val_manifest: Path, device: Any) -> Dict[str, Any]:
    import torch
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

    all_pairs = _load_manifest_pairs(val_manifest)
    pr_pairs = [p for p in all_pairs if not _is_hw_domain(p[2])]
    hw_pairs = [p for p in all_pairs if _is_hw_domain(p[2])]

    res = {
        "all": _eval_subset(model, all_pairs, device),
        "printed": _eval_subset(model, pr_pairs, device),
        "hw": _eval_subset(model, hw_pairs, device),
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def _upload_json(api: Any, path_in_repo: str, payload: Dict[str, Any], message: str) -> None:
    tmp = Path("/tmp") / f"{EXP_ID}_{Path(path_in_repo).name}"
    _atomic_write_json(tmp, payload)
    api.upload_file(
        path_or_fileobj=str(tmp),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=message,
    )
    try:
        tmp.unlink()
    except OSError:
        pass


def main() -> int:
    t_exp_start = time.perf_counter()
    hf_token = ""
    try:
        hf_token = _read_and_purge_hf_token()
        _setup_environment()

        import torch
        from huggingface_hub import HfApi
        import train_uni_v2 as tv2

        api = HfApi(token=hf_token)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        gpu_vram_gb = (
            round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
            if device.type == "cuda"
            else 0.0
        )
        print(f"[{EXP_ID}] Device: {gpu_name} ({gpu_vram_gb} GB) | torch={torch.__version__}", flush=True)

        run_ledger: Dict[str, Any] = {
            "exp_id": EXP_ID,
            "baseline_sha": BASELINE_SHA,
            "platform": f"Google Colab ({gpu_name}, {gpu_vram_gb} GB)",
            "status": "RUNNING",
            "started_at_utc": _utc_now(),
            "arms": [a[0] for a in ARMS],
            "completed_arms": [],
        }
        _upload_json(api, HF_RUN_PATH, run_ledger, f"{EXP_ID}: mark RUNNING on Colab L4")

        train_abs, val_abs = _ensure_proxy_dataset(hf_token)
        pr_strip_dir, hw_strip_dir, strip_meta = _ensure_derived_strips(train_abs)

        exp_ckpt_root = VOL_ROOT / "ckpts" / EXP_ID
        arm_results: Dict[str, Any] = {}

        for arm_name, curriculum, fp_frac in ARMS:
            arm_dir = exp_ckpt_root / arm_name
            arm_dir.mkdir(parents=True, exist_ok=True)
            cached_res = arm_dir / "arm_result.json"
            if cached_res.is_file():
                print(f"[{EXP_ID}] Reusing completed arm result: {arm_name}", flush=True)
                arm_results[arm_name] = json.loads(cached_res.read_text(encoding="utf-8"))
                continue

            print(
                f"\n[{EXP_ID}] === Arm {arm_name}: curriculum={curriculum}, fullpage_frac={fp_frac:.2f} ===",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            argv: List[str] = [
                "--preset", "small_45m",
                "--optimizer", "muon",
                "--steps", str(TRAIN_STEPS),
                "--seed", str(SEED),
                "--batch-size", str(BATCH_SIZE),
                "--strip-batch-size", str(STRIP_BATCH_SIZE),
                "--grad-accum", str(GRAD_ACCUM),
                "--curriculum", curriculum,
                "--fullpage-frac", str(fp_frac),
                "--fullpage-manifest", str(train_abs),
                "--fullpage-max-dim", str(FULLPAGE_MAX_DIM),
                "--max-seq-len", str(MAX_SEQ_LEN),
                "--augment", "--aug-prob", "0.50",
                "--lr-schedule", "wsd",
                "--lr", str(LR),
                "--warmup-steps", str(WARMUP_STEPS),
                "--wsd-decay-frac", str(WSD_DECAY_FRAC),
                "--min-lr-ratio", str(MIN_LR_RATIO),
                "--grad-checkpoint", "--channels-last", "--bf16",
                "--gpu-ema-decay", "0.997", "--gpu-ema-every", "5",
                "--save-every-minutes", str(SAVE_EVERY_MINUTES),
                "--val-count", "0", "--eval-every", "0",
                "--log-every", "50",
                "--num-workers", "0", "--prefetch-threads", "4",
                "--out-dir", str(arm_dir),
            ]
            if curriculum == "mixed":
                argv.extend(["--strip-dirs", str(pr_strip_dir)])
                if any(hw_strip_dir.glob("*-bekern.txt")):
                    argv.extend(["--hw-strip-dirs", str(hw_strip_dir)])

            args = tv2.build_arg_parser().parse_args(argv)
            t0 = time.perf_counter()
            train_out = tv2.run_train_v2(args)
            train_sec = round(time.perf_counter() - t0, 2)
            peak_vram_gb = (
                round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
                if device.type == "cuda"
                else 0.0
            )

            history = train_out.get("history", [])
            kind_counts: Dict[str, int] = {}
            for row in history:
                k = str(row.get("kind", "unknown"))
                kind_counts[k] = kind_counts.get(k, 0) + 1
            tail_rows = history[-50:] if history else []
            final_train_loss = round(
                sum(float(r["loss"]) for r in tail_rows) / float(max(1, len(tail_rows))), 6
            )
            final_train_ce = round(
                sum(float(r["ce_loss"]) for r in tail_rows) / float(max(1, len(tail_rows))), 6
            )

            raw_ckpt = arm_dir / "uni_omr_v2_latest.pt"
            ema_ckpt = arm_dir / "uni_omr_v2_ema_latest.pt"
            print(f"[{EXP_ID}] Evaluating {arm_name} Raw & EMA checkpoints on proxy_val...", flush=True)
            raw_eval = _eval_checkpoint(raw_ckpt, val_abs, device) if raw_ckpt.is_file() else {}
            ema_eval = _eval_checkpoint(ema_ckpt, val_abs, device) if ema_ckpt.is_file() else {}

            arm_rec: Dict[str, Any] = {
                "arm": arm_name,
                "curriculum": curriculum,
                "fullpage_frac": fp_frac,
                "steps": TRAIN_STEPS,
                "train_sec": train_sec,
                "steps_per_sec": round(TRAIN_STEPS / max(1e-6, train_sec), 3),
                "peak_vram_gb": peak_vram_gb,
                "optimizer_mode": train_out.get("optimizer_mode"),
                "step_kind_counts": kind_counts,
                "final_50step_train_loss": final_train_loss,
                "final_50step_train_ce": final_train_ce,
                "proxy_val_raw": raw_eval,
                "proxy_val_ema": ema_eval,
            }
            _atomic_write_json(cached_res, arm_rec)
            arm_results[arm_name] = arm_rec

            run_ledger["completed_arms"] = list(arm_results.keys())
            run_ledger["updated_at_utc"] = _utc_now()
            _upload_json(api, HF_RUN_PATH, run_ledger, f"{EXP_ID}: completed arm {arm_name}")

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Select winning arm by EMA proxy_val overall SER (tie-break: Token F1 desc, Val CE asc)
        best_arm = min(
            arm_results.keys(),
            key=lambda k: (
                arm_results[k].get("proxy_val_ema", {}).get("all", {}).get("ser", 999.0),
                -arm_results[k].get("proxy_val_ema", {}).get("all", {}).get("token_f1", 0.0),
                arm_results[k].get("proxy_val_ema", {}).get("all", {}).get("val_ce", 999.0),
            ),
        )

        final_eval: Dict[str, Any] = {
            "exp_id": EXP_ID,
            "baseline_sha": BASELINE_SHA,
            "completed_at_utc": _utc_now(),
            "total_wall_sec": round(time.perf_counter() - t_exp_start, 2),
            "platform": f"Google Colab ({gpu_name}, {gpu_vram_gb} GB)",
            "derived_strips_meta": strip_meta,
            "winning_arm_by_ema_ser": best_arm,
            "arms": arm_results,
        }
        _upload_json(api, HF_EVAL_PATH, final_eval, f"{EXP_ID}: upload final fullpage_frac sweep eval")

        run_ledger["status"] = "COMPLETED"
        run_ledger["completed_at_utc"] = _utc_now()
        run_ledger["total_wall_sec"] = final_eval["total_wall_sec"]
        run_ledger["winning_arm"] = best_arm
        run_ledger["eval_path"] = HF_EVAL_PATH
        _upload_json(api, HF_RUN_PATH, run_ledger, f"{EXP_ID}: mark COMPLETED")

        print(f"\n[{EXP_ID}] COMPLETED! Winner={best_arm} | Eval uploaded -> {HF_EVAL_PATH}", flush=True)
        return 0
    finally:
        _purge_hf_token_file()


if __name__ == "__main__":
    raise SystemExit(main())
