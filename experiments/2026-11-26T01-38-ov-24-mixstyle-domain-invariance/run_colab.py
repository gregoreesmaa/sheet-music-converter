#!/usr/bin/env python3
"""Google Colab L4 runner for Overnight Exp 24: MixStyle2d Domain-Invariance Sweep.

Experiment ID: 2026-11-26T01-38-ov-24-mixstyle-domain-invariance
Controlled 3-Arm Ablation on UniOMR v2 small_45m (BF16, seed=42, lr=3e-4, 800 steps):
  - Arm A (ms_00): --mixstyle-prob 0.00
  - Arm B (ms_20): --mixstyle-prob 0.20
  - Arm C (ms_40): --mixstyle-prob 0.40

Trains strictly on the printed-only subset of datasets/iter10_5_proxy_2k.tar so HW is unseen,
then measures in-domain printed retention vs zero-shot printed->HW cross-domain generalization
on proxy_val.jsonl (printed vs hw) and suite31_hw (45 pieces x 2 = 90 pages).

Uploads results to Hugging Face Private Dataset Hub (gregoreesmaa/sheet-music-converter-omr):
  - evals/2026-11-26T01-38-ov-24-mixstyle-domain-invariance.json
  - runs/2026-11-26T01-38-ov-24-mixstyle-domain-invariance.json
"""

from __future__ import annotations

import difflib
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

EXP_ID = "2026-11-26T01-38-ov-24-mixstyle-domain-invariance"
BASELINE_SHA = "164852f"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
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
GRAD_ACCUM = 4
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
VAL_EVAL_MAX_LEN = 512
SUITE31_MAX_LEN = 768
SAVE_EVERY_MINUTES = 5.0

ARMS: List[Tuple[str, float]] = [
    ("ms_00", 0.00),
    ("ms_20", 0.20),
    ("ms_40", 0.40),
]

_PITCH_TOKENS = {"A", "B", "C", "D", "E", "F", "G", "a-2", "a-1", "a1", "a2"} | {
    str(n) for n in range(0, 9)
}
_RHYTHM_TOKENS = {
    "whole", "half", "quarter", "eighth", "16th", "32nd", "64th", "breve",
    "dot", "rest", "bar-rest", "tie-start", "tie-stop", "tuplet-stop",
}


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


def _download_and_unpack(hf_token: str, hf_filename: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    sentinel = dest_dir / ".unpacked_ok"
    if sentinel.is_file():
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{EXP_ID}] Downloading {hf_filename}...", flush=True)
    t0 = time.perf_counter()
    tar_p = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=hf_filename,
        repo_type="dataset",
        token=hf_token,
        local_dir="/tmp/hf_dl",
    )
    print(f"[{EXP_ID}] Downloaded {hf_filename} in {time.perf_counter() - t0:.1f}s -> extracting...", flush=True)
    with tarfile.open(tar_p, "r:*") as tf:
        tf.extractall(path=dest_dir)
    try:
        os.remove(tar_p)
    except OSError:
        pass
    sentinel.write_text(_utc_now() + "\n", encoding="utf-8")
    return dest_dir


def _is_hw_domain(domain: str) -> bool:
    d = domain.lower()
    return ("hw" in d) or ("hand" in d) or ("rism" in d) or ("debussy" in d)


def _ensure_datasets(hf_token: str) -> Tuple[Path, Path, Path, int]:
    ds_root = VOL_ROOT / "datasets"

    proxy_dir = _download_and_unpack(hf_token, HF_PROXY_TAR, ds_root / "iter10_5_proxy_2k")
    proxy_train_abs = proxy_dir / "proxy_train_abs.jsonl"
    proxy_val_abs = proxy_dir / "proxy_val_abs.jsonl"
    for src_name, dst_path in (("proxy_train.jsonl", proxy_train_abs), ("proxy_val.jsonl", proxy_val_abs)):
        if not dst_path.is_file():
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

    # Filter strictly printed-only training manifest for clean printed->HW generalization test
    printed_only_path = proxy_dir / "proxy_train_printed_only.jsonl"
    pr_lines: List[str] = []
    for raw in proxy_train_abs.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        if not _is_hw_domain(str(rec.get("domain", ""))):
            pr_lines.append(json.dumps(rec))
    printed_only_path.write_text("\n".join(pr_lines) + "\n", encoding="utf-8")
    print(f"[{EXP_ID}] Printed-only train manifest: {len(pr_lines)} rows", flush=True)

    suite31_dir = _download_and_unpack(hf_token, HF_SUITE31_TAR, ds_root / "suite31_hw_eval")
    s31_manifests = sorted(suite31_dir.rglob("suite31_hw_manifest.json"))
    if not s31_manifests:
        raise FileNotFoundError(f"suite31_hw_manifest.json not found under {suite31_dir}")

    return printed_only_path, proxy_val_abs, s31_manifests[0], len(pr_lines)


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


def _seq_f1_ser(pred_toks: Sequence[str], ref_toks: Sequence[str]) -> Tuple[float, float]:
    if not ref_toks and not pred_toks:
        return 1.0, 0.0
    if not ref_toks:
        return 0.0, 1.0
    sm = difflib.SequenceMatcher(a=ref_toks, b=pred_toks, autojunk=False)
    matches = sum(blk.size for blk in sm.get_matching_blocks())
    prec = matches / float(max(1, len(pred_toks)))
    rec = matches / float(max(1, len(ref_toks)))
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
    edits = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            edits += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            edits += i2 - i1
        elif tag == "insert":
            edits += j2 - j1
    ser = edits / float(max(1, len(ref_toks)))
    return f1, ser


def _eval_subset(model: Any, pairs: List[Tuple[Path, List[Path], str]], device: Any) -> Dict[str, Any]:
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
        "ser": round(float(greedy["ser"]), 6),
        "token_f1": round(float(greedy["token_f1"]), 6),
        "sys_count_exact": round(float(greedy["sys_count_exact"]), 6),
    }


def _eval_suite31_hw(model: Any, manifest_path: Path, device: Any) -> Dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image
    import data as mvd_data
    from uni_omr import BOS_ID, EOS_ID, PAD_ID

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    pieces = manifest.get("pieces", manifest if isinstance(manifest, list) else [])

    buckets: Dict[str, List[Dict[str, float]]] = {
        "printed_clean": [],
        "printed_photo": [],
        "hw_clean": [],
        "hw_photo": [],
    }
    id_to_tok = list(mvd_data.VOCAB)

    def _ids_to_toks(ids: Sequence[int]) -> List[str]:
        out: List[str] = []
        for tid in ids:
            if tid == EOS_ID:
                break
            if tid in (BOS_ID, PAD_ID):
                continue
            if 0 <= tid < len(id_to_tok):
                out.append(id_to_tok[tid])
        return out

    for piece in pieces:
        is_hw = bool(piece.get("is_handwritten", False)) or ("hw" in str(piece.get("category", "")).lower())
        gt_rel = piece.get("gt_bekern") or piece.get("bekern")
        if not gt_rel:
            continue
        gt_path = (base_dir / gt_rel).resolve()
        if not gt_path.is_file():
            continue
        ref_toks = [t for t in gt_path.read_text(encoding="utf-8", errors="ignore").split() if t]
        ref_pitch = [t for t in ref_toks if t in _PITCH_TOKENS]
        ref_rhythm = [t for t in ref_toks if t in _RHYTHM_TOKENS]

        for variant in ("clean", "photo"):
            img_rel = piece.get(f"{variant}_image") or piece.get(variant)
            if not img_rel:
                continue
            img_path = (base_dir / img_rel).resolve()
            if not img_path.is_file():
                continue

            with Image.open(img_path) as im:
                rgb = im.convert("RGB")
                w, h = rgb.size
                scale = min(1.0, float(FULLPAGE_MAX_DIM) / float(max(w, h)))
                nw = max(32, (int(round(w * scale)) // 16) * 16)
                nh = max(32, (int(round(h * scale)) // 16) * 16)
                if (nw, nh) != (w, h):
                    rgb = rgb.resize((nw, nh), Image.BICUBIC)
                arr = np.asarray(rgb, dtype=np.float32) / 255.0
                img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pred_ids = model.greedy_decode(img_t, max_len=SUITE31_MAX_LEN)[0].detach().cpu().tolist()

            pred_toks = _ids_to_toks(pred_ids)
            pred_pitch = [t for t in pred_toks if t in _PITCH_TOKENS]
            pred_rhythm = [t for t in pred_toks if t in _RHYTHM_TOKENS]

            tok_f1, ser = _seq_f1_ser(pred_toks, ref_toks)
            pitch_f1, _ = _seq_f1_ser(pred_pitch, ref_pitch)
            rhythm_f1, _ = _seq_f1_ser(pred_rhythm, ref_rhythm)

            bkey = f"{'hw' if is_hw else 'printed'}_{variant}"
            buckets[bkey].append({
                "token_f1": tok_f1,
                "pitch_f1": pitch_f1,
                "rhythm_f1": rhythm_f1,
                "ser": ser,
            })

    summary: Dict[str, Any] = {}
    for bkey, rows in buckets.items():
        if not rows:
            summary[bkey] = {"n_pages": 0}
            continue
        summary[bkey] = {
            "n_pages": len(rows),
            "token_f1": round(sum(r["token_f1"] for r in rows) / len(rows), 6),
            "pitch_f1": round(sum(r["pitch_f1"] for r in rows) / len(rows), 6),
            "rhythm_f1": round(sum(r["rhythm_f1"] for r in rows) / len(rows), 6),
            "ser": round(sum(r["ser"] for r in rows) / len(rows), 6),
        }

    pr_mean = 0.5 * (
        float(summary.get("printed_clean", {}).get("token_f1", 0.0))
        + float(summary.get("printed_photo", {}).get("token_f1", 0.0))
    )
    hw_mean = 0.5 * (
        float(summary.get("hw_clean", {}).get("token_f1", 0.0))
        + float(summary.get("hw_photo", {}).get("token_f1", 0.0))
    )
    summary["printed_mean_token_f1"] = round(pr_mean, 6)
    summary["hw_mean_token_f1"] = round(hw_mean, 6)
    summary["hw_minus_printed_gap"] = round(hw_mean - pr_mean, 6)
    return summary


def _eval_arm_checkpoint(
    ckpt_path: Path,
    proxy_val_abs: Path,
    suite31_manifest: Path,
    device: Any,
) -> Dict[str, Any]:
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

    all_proxy = _load_manifest_pairs(proxy_val_abs)
    pr_proxy = [p for p in all_proxy if not _is_hw_domain(p[2])]
    hw_proxy = [p for p in all_proxy if _is_hw_domain(p[2])]

    pr_eval = _eval_subset(model, pr_proxy, device)
    hw_eval = _eval_subset(model, hw_proxy, device)
    res = {
        "proxy_val_printed": pr_eval,
        "proxy_val_hw_zero_shot": hw_eval,
        "proxy_val_hw_minus_printed_f1": round(
            float(hw_eval.get("token_f1", 0.0)) - float(pr_eval.get("token_f1", 0.0)), 6
        ),
        "suite31_hw_cross_domain": _eval_suite31_hw(model, suite31_manifest, device),
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

        printed_train_manifest, proxy_val_abs, suite31_manifest, n_pr_train = _ensure_datasets(hf_token)

        exp_ckpt_root = VOL_ROOT / "ckpts" / EXP_ID
        arm_results: Dict[str, Any] = {}

        for arm_name, ms_prob in ARMS:
            arm_dir = exp_ckpt_root / arm_name
            arm_dir.mkdir(parents=True, exist_ok=True)
            cached_res = arm_dir / "arm_result.json"
            if cached_res.is_file():
                print(f"[{EXP_ID}] Reusing completed arm result: {arm_name}", flush=True)
                arm_results[arm_name] = json.loads(cached_res.read_text(encoding="utf-8"))
                continue

            print(f"\n[{EXP_ID}] === Arm {arm_name}: mixstyle_prob={ms_prob:.2f} ===", flush=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            argv: List[str] = [
                "--preset", "small_45m",
                "--optimizer", "muon",
                "--mixstyle-prob", str(ms_prob),
                "--steps", str(TRAIN_STEPS),
                "--seed", str(SEED),
                "--batch-size", str(BATCH_SIZE),
                "--grad-accum", str(GRAD_ACCUM),
                "--curriculum", "fullpage",
                "--fullpage-manifest", str(printed_train_manifest),
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
            args = tv2.build_arg_parser().parse_args(argv)
            args.stratified_fullpage = False

            t0 = time.perf_counter()
            train_out = tv2.run_train_v2(args)
            train_sec = round(time.perf_counter() - t0, 2)
            peak_vram_gb = (
                round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
                if device.type == "cuda"
                else 0.0
            )

            history = train_out.get("history", [])
            tail_rows = history[-50:] if history else []
            final_train_loss = round(
                sum(float(r["loss"]) for r in tail_rows) / float(max(1, len(tail_rows))), 6
            )

            ema_ckpt = arm_dir / "uni_omr_v2_ema_latest.pt"
            eval_ckpt = ema_ckpt if ema_ckpt.is_file() else (arm_dir / "uni_omr_v2_latest.pt")
            print(f"[{EXP_ID}] Evaluating {arm_name} ({eval_ckpt.name}) on proxy_val + suite31_hw...", flush=True)
            eval_out = _eval_arm_checkpoint(eval_ckpt, proxy_val_abs, suite31_manifest, device)

            arm_rec: Dict[str, Any] = {
                "arm": arm_name,
                "mixstyle_prob": ms_prob,
                "printed_train_rows": n_pr_train,
                "steps": TRAIN_STEPS,
                "train_sec": train_sec,
                "steps_per_sec": round(TRAIN_STEPS / max(1e-6, train_sec), 3),
                "peak_vram_gb": peak_vram_gb,
                "final_50step_train_loss": final_train_loss,
                "eval_ema": eval_out,
            }
            _atomic_write_json(cached_res, arm_rec)
            arm_results[arm_name] = arm_rec

            run_ledger["completed_arms"] = list(arm_results.keys())
            run_ledger["updated_at_utc"] = _utc_now()
            _upload_json(api, HF_RUN_PATH, run_ledger, f"{EXP_ID}: completed arm {arm_name}")

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        best_arm = max(
            arm_results.keys(),
            key=lambda k: float(
                arm_results[k].get("eval_ema", {}).get("suite31_hw_cross_domain", {}).get("hw_mean_token_f1", 0.0)
            ),
        )

        final_eval: Dict[str, Any] = {
            "exp_id": EXP_ID,
            "baseline_sha": BASELINE_SHA,
            "completed_at_utc": _utc_now(),
            "total_wall_sec": round(time.perf_counter() - t_exp_start, 2),
            "platform": f"Google Colab ({gpu_name}, {gpu_vram_gb} GB)",
            "winning_arm_by_suite31_hw_f1": best_arm,
            "arms": arm_results,
        }
        _upload_json(api, HF_EVAL_PATH, final_eval, f"{EXP_ID}: upload final MixStyle2d sweep eval")

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
