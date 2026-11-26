#!/usr/bin/env python3
"""Google Colab L4 runner for ov-19b: Converged InkTokenRouterV2 Sweep on Full suite31_hw.

Experiment ID: 2026-11-26T14-30-ov-19b-router-saliency-keep-converged
Sweeps InkTokenRouterV2 `(saliency_thr, max_keep_ratio)` in:
  1. arm_005_075: (0.05, 0.75)  [baseline default]
  2. arm_002_065: (0.02, 0.65)  [high-recall dense-page safety band]
  3. arm_003_055: (0.03, 0.55)  [ov-19 top compression candidate]
with the promoted `p10 + 0.15` paper-baseline threshold in `InkTokenRouterV2`.

Model resolution:
  - Primary: warm-starts from HF `checkpoints/iter12_v2_best.pt`
    (fallback candidates: `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt`,
    `checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt`).
  - Fallback: if no converged UniOMR v2 checkpoint is present on HF, trains a
    2,500-step bias-corrected EMA `UniOMR v2 small_45m` proxy on
    `datasets/iter10_5_proxy_2k.tar`.

Evaluation:
  - Full `suite31_hw` (`45` pieces = `31` printed + `14` HW × `{clean, photo}` = `90` pages)
  - `assert total_scored == 90` on every arm
  - `max_dim=1280`, `max_len=3584`, MTP-4 + FSA
  - Pure-Python MusicXML Strict Note F1 (zero `cargo` dependency)
  - Per-page router telemetry (`ink_recall`, `compression_x`, `keep_frac`) + dense-page audit.

Expects on Colab VM:
  - /content/bundle.tar.gz (or pre-extracted /workspace/sheet-music-converter/tmp-train/mvd)
  - /tmp/.hf_token (Hugging Face token; chmod 600, deleted immediately and in finally)
"""

from __future__ import annotations

import copy
import gc
import json
import os
import random
import statistics
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

EXP_ID = "2026-11-26T14-30-ov-19b-router-saliency-keep-converged"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_CANDIDATES = (
    "checkpoints/iter12_v2_best.pt",
    "checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt",
    "checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt",
)
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

WORKSPACE_ROOT = Path("/workspace")
VOL_ROOT = Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")
BUNDLE_CANDIDATES = (
    Path("/content/bundle.tar.gz"),
    Path("/content/ov19b_bundle.tar.gz"),
    Path("/content/repo_bundle.tar.gz"),
)

# Training / proxy fallback hyperparameters
PROXY_TRAIN_STEPS = 2500
SEED = 42
LR = 3e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 2
GRAD_ACCUM = 4
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
EVAL_MAX_LEN = 3584
SAVE_EVERY_MINUTES = 5.0
EMA_MAX_DECAY = 0.997
EMA_UPDATE_EVERY = 5

SWEEP_ARMS: Tuple[Dict[str, Any], ...] = (
    {"name": "arm_005_075", "saliency_thr": 0.05, "max_keep_ratio": 0.75},
    {"name": "arm_002_065", "saliency_thr": 0.02, "max_keep_ratio": 0.65},
    {"name": "arm_003_055", "saliency_thr": 0.03, "max_keep_ratio": 0.55},
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
    proc = subprocess.Popen(
        cmd,
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
        raise RuntimeError(f"Command failed (rc={rc}): {cmd}")


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


def _resolve_mvd_dir() -> Path:
    candidates = [
        WORKSPACE_ROOT / "sheet-music-converter" / "tmp-train" / "mvd",
        WORKSPACE_ROOT / "tmp-train" / "mvd",
        WORKSPACE_ROOT / "mvd",
        Path("/content/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path(__file__).resolve().parents[2] / "tmp-train" / "mvd",
    ]
    for c in candidates:
        if (c / "uni_omr_v2.py").is_file():
            return c
    for b in BUNDLE_CANDIDATES:
        if b.is_file():
            WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
            print(f"[{EXP_ID}] Extracting {b} -> {WORKSPACE_ROOT} ...", flush=True)
            with tarfile.open(b, "r:*") as tf:
                tf.extractall(path=WORKSPACE_ROOT)
            for c in candidates:
                if (c / "uni_omr_v2.py").is_file():
                    return c
    raise FileNotFoundError(
        "Could not locate tmp-train/mvd/uni_omr_v2.py. Upload repo bundle to /content/bundle.tar.gz."
    )


def _setup_environment() -> Path:
    VOL_ROOT.mkdir(parents=True, exist_ok=True)
    needed_pkgs: List[str] = []
    for mod, pkg in (
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
    ):
        try:
            __import__(mod)
        except ImportError:
            needed_pkgs.append(pkg)
    if needed_pkgs:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed_pkgs])
    mvd_dir = _resolve_mvd_dir()
    if str(mvd_dir) not in sys.path:
        sys.path.insert(0, str(mvd_dir))
    return mvd_dir


def _try_download_v2_ckpt(hf_token: str) -> Tuple[Optional[Path], Optional[str]]:
    from huggingface_hub import hf_hub_download

    for cand in HF_CKPT_CANDIDATES:
        local_p = VOL_ROOT / cand
        if local_p.is_file():
            print(f"[{EXP_ID}] Using cached converged v2 checkpoint: {cand}", flush=True)
            return local_p, cand
        try:
            print(f"[{EXP_ID}] Checking HF for {cand} ...", flush=True)
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=cand,
                local_dir=str(VOL_ROOT),
                token=hf_token,
            )
            if local_p.is_file():
                print(f"[{EXP_ID}] Downloaded converged v2 checkpoint: {cand}", flush=True)
                return local_p, cand
        except Exception as exc:
            print(f"[{EXP_ID}] {cand} not available on HF ({exc}); trying next...", flush=True)
    return None, None


def _ensure_suite31_manifest(hf_token: str) -> Path:
    from huggingface_hub import hf_hub_download

    s31_dir = VOL_ROOT / "datasets" / "suite31_hw_eval"
    sentinel = s31_dir / ".extracted"
    if not sentinel.exists():
        print(f"[{EXP_ID}] Downloading {HF_SUITE31_TAR} ...", flush=True)
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=HF_SUITE31_TAR,
                local_dir=str(VOL_ROOT / "tars"),
                token=hf_token,
            )
        )
        s31_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=s31_dir)
        sentinel.write_text(_utc_now() + "\n", encoding="utf-8")
        tar_p.unlink(missing_ok=True)

    manifests = sorted(s31_dir.rglob("suite31_hw_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Missing suite31_hw_manifest.json under {s31_dir}")
    return manifests[0]


def _ensure_proxy_dataset(hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    proxy_dir = VOL_ROOT / "datasets" / "iter10_5_proxy_2k"
    train_abs = proxy_dir / "proxy_train_abs.jsonl"
    val_abs = proxy_dir / "proxy_val_abs.jsonl"
    if train_abs.is_file() and val_abs.is_file():
        return train_abs, val_abs

    proxy_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR} from {HF_REPO_ID} ...", flush=True)
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


def _extract_musicxml_notes(xml_path: Path) -> List[Tuple[str, int, str, str, str]]:
    """Pure-Python MusicXML pitched-note extractor (zero cargo dependency)."""
    notes: List[Tuple[str, int, str, str, str]] = []
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return notes

    def _strip(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    for elem in root.iter():
        if _strip(elem.tag) != "note":
            continue
        is_rest = is_grace = False
        step = octave = alter = dur = ""
        n_dots = 0
        for child in elem:
            ctag = _strip(child.tag)
            if ctag == "rest":
                is_rest = True
            elif ctag == "grace":
                is_grace = True
            elif ctag == "pitch":
                for pc in child:
                    ptag = _strip(pc.tag)
                    text = (pc.text or "").strip()
                    if ptag == "step":
                        step = text.upper()
                    elif ptag == "octave":
                        octave = text
                    elif ptag == "alter":
                        try:
                            av = int(round(float(text)))
                            if av != 0:
                                alter = f"a{av}"
                        except ValueError:
                            pass
            elif ctag == "type":
                dur = (child.text or "").strip().lower()
            elif ctag == "dot":
                n_dots += 1
        if not is_rest and not is_grace and step and octave and dur:
            notes.append((dur, n_dots, step, octave, alter))
    return notes


class _BiasCorrectedGPUEma:
    """EMA shadow with PyTorch/timm bias-corrected ramp: decay_t = min(max_decay, (1+u)/(10+u))."""

    def __init__(self, model: Any, max_decay: float = EMA_MAX_DECAY, update_every: int = EMA_UPDATE_EVERY) -> None:
        import torch

        self.max_decay = float(max_decay)
        self.update_every = max(1, int(update_every))
        self.num_updates = 0
        self.shadow: Dict[str, Any] = {
            k: v.detach().clone().to(dtype=torch.float32)
            for k, v in model.state_dict().items()
            if v.is_floating_point()
        }

    def update(self, model: Any, step: int) -> None:
        if step % self.update_every != 0:
            return
        self.num_updates += 1
        d = min(self.max_decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        one_minus_d = 1.0 - d
        sd = model.state_dict()
        for k, s_val in self.shadow.items():
            p_val = sd[k].detach()
            s_val.mul_(d).add_(p_val.to(dtype=s_val.dtype), alpha=one_minus_d)

    def load_into(self, model: Any) -> None:
        sd = model.state_dict()
        for k, s_val in self.shadow.items():
            if k in sd:
                sd[k].copy_(s_val.to(device=sd[k].device, dtype=sd[k].dtype))


def _train_fallback_proxy_v2(
    hf_token: str,
    device: Any,
    *,
    use_bf16: bool,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Train a 2,500-step bias-corrected EMA UniOMR v2 small_45m proxy when HF v2 ckpt is absent."""
    import numpy as np
    import torch
    from train_uni_v2 import atomic_save_v2_latest, build_optimizer_v2, forward_and_loss_v2
    from uni_omr import (
        AsyncThreadPrefetcher,
        StratifiedFullPageSampler,
        UniFullPageDataset,
        _images_to_device_float,
        _lr_for_step,
        collate_uni_batch,
    )
    from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net

    train_abs, _ = _ensure_proxy_dataset(hf_token)
    train_pairs = _load_manifest_pairs(train_abs)
    proxy_dir = VOL_ROOT / "ckpts" / EXP_ID / "fallback_proxy_2500"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    done_ckpt = proxy_dir / "uni_omr_v2_ema_2500.pt"

    cfg = UniOMRv2Config.small_45m(
        grad_checkpoint=True,
        dropout=0.1,
        mixstyle_prob=0.3,
        tok_dropout=0.05,
        keep_ratio=0.40,
        dynamic_routing=True,
        min_keep_ratio=0.15,
        max_keep_ratio=0.75,
        saliency_thr=0.05,
        max_mem_tokens=3072,
        mtp_k=4,
    )
    _seed_everything(SEED)
    model = UniOMRv2Net(cfg).to(device)

    if done_ckpt.is_file():
        ck = torch.load(str(done_ckpt), map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"], strict=True)
        model.eval()
        print(f"[{EXP_ID}] Loaded cached 2,500-step fallback proxy from {done_ckpt}", flush=True)
        return model, cfg, {"source": "fallback_proxy_2500_cached", "steps": PROXY_TRAIN_STEPS}

    print(
        f"[{EXP_ID}] No converged HF v2 checkpoint found -> training {PROXY_TRAIN_STEPS}-step "
        f"bias-corrected EMA UniOMR v2 proxy on {len(train_pairs)} pages...",
        flush=True,
    )
    _seed_everything(SEED + 101)
    fp_sampler = StratifiedFullPageSampler(train_pairs)
    train_ds = UniFullPageDataset(
        train_pairs,
        fullpage_max_dim=FULLPAGE_MAX_DIM,
        max_seq_len=MAX_SEQ_LEN,
        augment=True,
        aug_prob=0.75,
        multi_scale=True,
        return_uint8=True,
    )
    prefetcher = AsyncThreadPrefetcher(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=False,
        max_workers=4,
        prefetch_batches=3,
        collate_fn=collate_uni_batch,
        seed=SEED + 303,
        stratified_sampler=fp_sampler,
    )
    opt = build_optimizer_v2(
        model,
        optimizer_name="muon",
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device_type=device.type,
    )
    ema = _BiasCorrectedGPUEma(model, max_decay=EMA_MAX_DECAY, update_every=EMA_UPDATE_EVERY)

    start_step = 1
    latest_ckpt = proxy_dir / "uni_omr_v2_latest.pt"
    if latest_ckpt.is_file():
        try:
            ck = torch.load(str(latest_ckpt), map_location="cpu", weights_only=True)
            saved_step = int(ck.get("step", 0))
            if 0 < saved_step < PROXY_TRAIN_STEPS:
                model.load_state_dict(ck["model"], strict=True)
                start_step = saved_step + 1
                ema = _BiasCorrectedGPUEma(model, max_decay=EMA_MAX_DECAY, update_every=EMA_UPDATE_EVERY)
                ema.num_updates = saved_step // EMA_UPDATE_EVERY
                print(f"[{EXP_ID}] Resumed proxy from step {saved_step} -> {start_step}", flush=True)
        except Exception as exc:
            print(f"[{EXP_ID}] WARN resume skipped: {exc}", flush=True)

    warmup_steps = max(50, int(round(PROXY_TRAIN_STEPS * 0.08)))
    save_every_sec = max(60.0, SAVE_EVERY_MINUTES * 60.0)
    last_save_ts = time.perf_counter()
    t0 = time.perf_counter()
    inv_ga = 1.0 / float(GRAD_ACCUM)
    step_times: List[float] = []

    model.train()
    try:
        for step in range(start_step, PROXY_TRAIN_STEPS + 1):
            ts0 = time.perf_counter()
            cur_lr = _lr_for_step(
                step,
                PROXY_TRAIN_STEPS,
                LR,
                warmup_steps,
                min_lr_ratio=0.05,
                schedule="wsd",
                wsd_decay_frac=0.25,
            )
            opt.set_lr(cur_lr / max(1e-12, LR))
            opt.zero_grad(set_to_none=True)

            acc_loss = 0.0
            acc_ce = 0.0
            for _ in range(GRAD_ACCUM):
                b = next(prefetcher)
                images = _images_to_device_float(b["images"], device, non_blocking=True)
                seq = b["seq"].to(device, non_blocking=True)
                losses, _ = forward_and_loss_v2(
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
                (losses["loss"] * inv_ga).backward()
                acc_loss += float(losses["loss"].detach().item()) * inv_ga
                acc_ce += float(losses["ce_loss"].detach().item()) * inv_ga

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model, step)
            dt = time.perf_counter() - ts0
            step_times.append(dt)

            if step == 1 or step % 100 == 0 or step == PROXY_TRAIN_STEPS:
                print(
                    f"[proxy] step {step:4d}/{PROXY_TRAIN_STEPS} | lr={cur_lr:.2e} | "
                    f"loss={acc_loss:.4f} ce={acc_ce:.4f} | ema_u={ema.num_updates} | {dt:.2f}s/step",
                    flush=True,
                )
            if (time.perf_counter() - last_save_ts) >= save_every_sec:
                last_save_ts = time.perf_counter()
                atomic_save_v2_latest(proxy_dir, model, cfg, step=step, gpu_ema=None)
                print(f"  [proxy] 5-min atomic checkpoint saved @ step {step}", flush=True)
    finally:
        prefetcher.close()

    ema.load_into(model)
    model.eval()
    tmp_ckpt = done_ckpt.with_suffix(".pt.tmp")
    torch.save({"arch": "uni_omr_v2", "step": PROXY_TRAIN_STEPS, "model": model.state_dict()}, str(tmp_ckpt))
    os.replace(tmp_ckpt, done_ckpt)

    train_secs = round(time.perf_counter() - t0, 2)
    meta = {
        "source": "fallback_proxy_2500_bias_corrected_ema",
        "steps": PROXY_TRAIN_STEPS,
        "train_secs": train_secs,
        "mean_step_sec": round(float(np.mean(step_times)) if step_times else 0.0, 3),
        "ema_updates": ema.num_updates,
    }
    del opt, ema, train_ds
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, cfg, meta


def _eval_full_suite31_arm(
    model: Any,
    cfg: Any,
    all_pieces: Sequence[Dict[str, Any]],
    arm_cfg: Dict[str, Any],
    device: Any,
    *,
    use_bf16: bool,
    eval_s31: Any,
    rlvr_grpo: Any,
    uni_omr: Any,
    partial_path: Path,
) -> Dict[str, Any]:
    """Evaluate one `(saliency_thr, max_keep_ratio)` arm on all 90 suite31_hw pages."""
    import numpy as np
    import torch
    from PIL import Image

    arm_name = str(arm_cfg["name"])
    sal_thr = float(arm_cfg["saliency_thr"])
    max_keep = float(arm_cfg["max_keep_ratio"])
    effective_p10_delta = max(sal_thr, 0.15)

    router = eval_s31._get_model_router(model)
    if router is None:
        raise RuntimeError("Model has no InkTokenRouterV2!")
    orig_thr = float(router.saliency_thr)
    orig_max = float(router.max_keep_ratio)
    orig_dyn = bool(router.dynamic_routing)
    router.saliency_thr = sal_thr
    router.max_keep_ratio = max_keep
    router.dynamic_routing = True

    _, _, eos_id, _, _, id_to_tok = eval_s31._vocab_spec(int(cfg.vocab_size))
    rows: List[Dict[str, Any]] = []
    t_arm = time.perf_counter()
    last_save_t = time.perf_counter()

    try:
        for idx, piece in enumerate(all_pieces, 1):
            slug = piece["slug"]
            is_hw = bool(piece["handwritten"])
            ref_notes = _extract_musicxml_notes(piece["gt"])
            for domain in ("clean", "photo"):
                with Image.open(piece[domain]) as raw_im:
                    pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), FULLPAGE_MAX_DIM)
                img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)

                t0 = time.perf_counter()
                with torch.no_grad():
                    if use_bf16 and device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            _, rinfo, _ = model.encode_image(img_t, return_aux=False)
                            seq, mtp_stats = eval_s31.generate_mtp_spec(
                                model,
                                img_t,
                                max_len=EVAL_MAX_LEN,
                                use_grammar=True,
                                strict_grammar=False,
                                return_stats=True,
                            )
                    else:
                        _, rinfo, _ = model.encode_image(img_t, return_aux=False)
                        seq, mtp_stats = eval_s31.generate_mtp_spec(
                            model,
                            img_t,
                            max_len=EVAL_MAX_LEN,
                            use_grammar=True,
                            strict_grammar=False,
                            return_stats=True,
                        )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = max(1e-6, time.perf_counter() - t0)

                seq_ids = seq[0].detach().cpu().tolist()
                has_eos = eos_id in seq_ids[1:]
                pred_raw = eval_s31.decode_uni_compat(
                    seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True
                )
                if not has_eos:
                    pred_raw = eval_s31.trim_truncated_tokens(pred_raw)
                hyp_toks, n_sys = eval_s31.stitch_uni_tokens(pred_raw)
                struct = rlvr_grpo.compute_unlabeled_structural_reward(
                    pred_raw, vocab=id_to_tok, terminated_eos=has_eos
                )
                hyp_notes = rlvr_grpo.extract_pitch_dur_notes(hyp_toks)
                note_f1 = rlvr_grpo.compute_sequence_f1(ref_notes, hyp_notes)

                n_in = max(1.0, float(rinfo.get("n_in", 1.0)))
                n_out = max(1.0, float(rinfo.get("n_out", 1.0)))
                k_kept = float(rinfo.get("k_kept", n_out))
                ink_rec = float(rinfo.get("ink_recall", 1.0))

                rows.append(
                    {
                        "piece": slug,
                        "handwritten": is_hw,
                        "domain": domain,
                        "note_f1_py": round(float(note_f1), 6),
                        "struct_reward": round(float(struct["reward"]), 6),
                        "has_eos": bool(has_eos),
                        "n_sys": int(n_sys),
                        "n_hyp_notes": len(hyp_notes),
                        "n_toks": len(seq_ids),
                        "elapsed_secs": round(elapsed, 3),
                        "mtp_acc": round(float(mtp_stats.get("acceptance_rate", 0.0)), 4),
                        "ink_recall": round(ink_rec, 5),
                        "compression_x": round(n_in / n_out, 4),
                        "keep_frac": round(k_kept / n_in, 5),
                        "n_in": int(n_in),
                        "n_out": int(n_out),
                    }
                )
            if idx % 5 == 0 or idx == len(all_pieces):
                cur_f1 = sum(r["note_f1_py"] for r in rows) / max(1, len(rows))
                cur_ir = sum(r["ink_recall"] for r in rows) / max(1, len(rows))
                print(
                    f"  [{arm_name}] {idx:02d}/{len(all_pieces):02d} pieces ({len(rows)}/90 pages) | "
                    f"mean_f1={cur_f1:.4f} mean_ink_rec={cur_ir:.4f}",
                    flush=True,
                )
            if (time.perf_counter() - last_save_t) >= SAVE_EVERY_MINUTES * 60.0:
                _atomic_write_json(partial_path, {"exp_id": EXP_ID, "arm": arm_name, "n_rows": len(rows), "rows": rows})
                last_save_t = time.perf_counter()
    finally:
        router.saliency_thr = orig_thr
        router.max_keep_ratio = orig_max
        router.dynamic_routing = orig_dyn

    total_scored = len(rows)
    assert total_scored == 90, f"[{arm_name}] Expected 90 scored suite31_hw pages, got {total_scored}!"

    quadrants: Dict[str, Any] = {}
    for split_key, is_hw_flag in (("printed", False), ("hw", True)):
        for dom in ("clean", "photo"):
            qrows = [r for r in rows if r["handwritten"] == is_hw_flag and r["domain"] == dom]
            quadrants[f"{split_key}_{dom}"] = {
                "n": len(qrows),
                "note_f1_py": round(float(np.mean([r["note_f1_py"] for r in qrows])), 6),
                "eos_rate": round(sum(1 for r in qrows if r["has_eos"]) / max(1, len(qrows)), 4),
                "ink_recall_mean": round(float(np.mean([r["ink_recall"] for r in qrows])), 5),
                "ink_recall_p10": round(float(np.percentile([r["ink_recall"] for r in qrows], 10)), 5),
                "ink_recall_min": round(float(np.min([r["ink_recall"] for r in qrows])), 5),
                "compression_x_mean": round(float(np.mean([r["compression_x"] for r in qrows])), 4),
                "keep_frac_mean": round(float(np.mean([r["keep_frac"] for r in qrows])), 5),
                "median_secs": round(float(statistics.median([r["elapsed_secs"] for r in qrows])), 3),
            }

    all_ir = [r["ink_recall"] for r in rows]
    all_comp = [r["compression_x"] for r in rows]
    all_kf = [r["keep_frac"] for r in rows]
    dense_audit = sorted(rows, key=lambda r: r["ink_recall"])[:10]

    return {
        "arm": arm_name,
        "saliency_thr": sal_thr,
        "effective_p10_delta": effective_p10_delta,
        "max_keep_ratio": max_keep,
        "total_scored": total_scored,
        "elapsed_secs": round(time.perf_counter() - t_arm, 2),
        "overall": {
            "note_f1_py": round(float(np.mean([r["note_f1_py"] for r in rows])), 6),
            "ink_recall_mean": round(float(np.mean(all_ir)), 5),
            "ink_recall_p10": round(float(np.percentile(all_ir, 10)), 5),
            "ink_recall_min": round(float(np.min(all_ir)), 5),
            "compression_x_mean": round(float(np.mean(all_comp)), 4),
            "keep_frac_mean": round(float(np.mean(all_kf)), 5),
            "median_secs": round(float(statistics.median([r["elapsed_secs"] for r in rows])), 3),
        },
        "quadrants": quadrants,
        "dense_page_audit_bottom10": dense_audit,
        "rows": rows,
    }


def main() -> None:
    try:
        hf_token = _read_and_purge_hf_token()
        _setup_environment()

        import torch
        from huggingface_hub import HfApi
        import eval_uni_suite31 as eval_s31
        import rlvr_grpo
        import uni_omr

        t_start = time.perf_counter()
        started_at = _utc_now()

        if torch.cuda.is_available():
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
        use_bf16 = bool(torch.cuda.is_available() and cc[0] >= 8)
        platform_str = f"Colab {gpu_name} (cc={cc[0]}.{cc[1]}, bf16={use_bf16})"
        print(f"[{EXP_ID}] Running on {platform_str}", flush=True)

        api = HfApi(token=hf_token)
        run_local = VOL_ROOT / "runs" / f"{EXP_ID}.json"
        _atomic_write_json(
            run_local,
            {"exp_id": EXP_ID, "status": "RUNNING", "started_at": started_at, "platform": platform_str},
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: mark RUNNING",
        )

        manifest_path = _ensure_suite31_manifest(hf_token)
        _, all_pieces = eval_s31.resolve_manifest_pieces(manifest_path)
        assert len(all_pieces) == 45, f"Expected 45 suite31_hw pieces, got {len(all_pieces)}"

        ckpt_path, ckpt_hf_used = _try_download_v2_ckpt(hf_token)
        if ckpt_path is not None:
            model, cfg, ckpt_meta = eval_s31.load_uni_checkpoint(str(ckpt_path), device=device)
            model.eval()
            model_source = {"mode": "warmstart_hf_v2", "ckpt": ckpt_hf_used, "meta": ckpt_meta}
        else:
            model, cfg, proxy_meta = _train_fallback_proxy_v2(hf_token, device, use_bf16=use_bf16)
            model_source = {"mode": "fallback_proxy_2500", "ckpt": None, "meta": proxy_meta}

        arm_results: Dict[str, Any] = {}
        partial_path = VOL_ROOT / "evals" / f"{EXP_ID}_partial.json"
        for arm_cfg in SWEEP_ARMS:
            print(
                f"\n[{EXP_ID}] Evaluating {arm_cfg['name']} "
                f"(saliency_thr={arm_cfg['saliency_thr']}, p10+0.15, max_keep_ratio={arm_cfg['max_keep_ratio']}) "
                f"on 90 suite31_hw pages (max_len={EVAL_MAX_LEN})...",
                flush=True,
            )
            res = _eval_full_suite31_arm(
                model,
                cfg,
                all_pieces,
                arm_cfg,
                device,
                use_bf16=use_bf16,
                eval_s31=eval_s31,
                rlvr_grpo=rlvr_grpo,
                uni_omr=uni_omr,
                partial_path=partial_path,
            )
            arm_results[arm_cfg["name"]] = res

        base_quads = arm_results["arm_005_075"]["quadrants"]
        summary_table: List[Dict[str, Any]] = []
        for arm_cfg in SWEEP_ARMS:
            name = arm_cfg["name"]
            ar = arm_results[name]
            ov = ar["overall"]
            q = ar["quadrants"]
            summary_table.append(
                {
                    "arm": name,
                    "saliency_thr": arm_cfg["saliency_thr"],
                    "effective_p10_delta": ar["effective_p10_delta"],
                    "max_keep_ratio": arm_cfg["max_keep_ratio"],
                    "total_scored": ar["total_scored"],
                    "ink_recall_mean": ov["ink_recall_mean"],
                    "ink_recall_p10": ov["ink_recall_p10"],
                    "ink_recall_min": ov["ink_recall_min"],
                    "compression_x_mean": ov["compression_x_mean"],
                    "keep_frac_mean": ov["keep_frac_mean"],
                    "pr_clean_f1": q["printed_clean"]["note_f1_py"],
                    "delta_pr_clean_f1": round(q["printed_clean"]["note_f1_py"] - base_quads["printed_clean"]["note_f1_py"], 6),
                    "pr_photo_f1": q["printed_photo"]["note_f1_py"],
                    "delta_pr_photo_f1": round(q["printed_photo"]["note_f1_py"] - base_quads["printed_photo"]["note_f1_py"], 6),
                    "hw_clean_f1": q["hw_clean"]["note_f1_py"],
                    "delta_hw_clean_f1": round(q["hw_clean"]["note_f1_py"] - base_quads["hw_clean"]["note_f1_py"], 6),
                    "hw_photo_f1": q["hw_photo"]["note_f1_py"],
                    "delta_hw_photo_f1": round(q["hw_photo"]["note_f1_py"] - base_quads["hw_photo"]["note_f1_py"], 6),
                    "median_secs": ov["median_secs"],
                }
            )

        elapsed_sec = round(time.perf_counter() - t_start, 2)
        completed_at = _utc_now()
        report = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_sec": elapsed_sec,
            "platform": platform_str,
            "bf16": use_bf16,
            "model_source": model_source,
            "config": {
                "arms": list(SWEEP_ARMS),
                "paper_baseline_threshold": "p10 + max(saliency_thr, 0.15)",
                "fullpage_max_dim": FULLPAGE_MAX_DIM,
                "eval_max_len": EVAL_MAX_LEN,
                "suite31_hw_pages": 90,
            },
            "summary": summary_table,
            "arms": arm_results,
        }
        run_ledger = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "platform": platform_str,
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_sec": elapsed_sec,
            "model_source_mode": model_source["mode"],
            "hf_eval_path": HF_EVAL_PATH,
            "summary": summary_table,
        }

        eval_local = VOL_ROOT / "evals" / f"{EXP_ID}.json"
        _atomic_write_json(eval_local, report)
        _atomic_write_json(run_local, run_ledger)

        api.upload_file(
            path_or_fileobj=str(eval_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"eval({EXP_ID}): upload converged InkTokenRouterV2 suite31_hw sweep",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload run ledger",
        )
        print(f"\n[{EXP_ID}] COMPLETED & uploaded {HF_EVAL_PATH} in {elapsed_sec}s!", flush=True)
        print(json.dumps(summary_table, indent=2), flush=True)
    finally:
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
