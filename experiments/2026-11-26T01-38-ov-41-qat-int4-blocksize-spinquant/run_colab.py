#!/usr/bin/env python3
"""Google Colab L4 runner for Overnight Exp 41: QAT-INT4 block_size x SpinQuant Hadamard Sweep.

Experiment ID: 2026-11-26T01-38-ov-41-qat-int4-blocksize-spinquant
Protocol on UniOMR v2 small_45m (BF16, iter10_5_proxy_2k.tar):
  1. Train 1 shared BF16 base checkpoint (600 steps, seed=42) and measure FP32 reference
     proxy_val Token F1, SER, CE loss, and reference logits.
  2. Branch into 3 x 2 factorial matrix:
       block_size in {32, 64, 128} x spinquant in {False, True}
     For each of the 6 arms:
       - If spinquant=True: apply fold_hadamard_v2(model) and verify FP32 invariance.
       - Measure 0-step PTQ INT4 proxy_val F1 & logit KL/MSE/top1 vs FP32 reference.
       - Run 200 steps of QAT-INT4 fine-tuning (QATInt4ControllerV2, lr=1e-4 cosine).
       - Measure Post-QAT INT4 proxy_val F1, F1 retention, logit KL/MSE/top1, & WebGPU MB.

Uploads to Hugging Face Private Dataset Hub (gregoreesmaa/sheet-music-converter-omr):
  - evals/2026-11-26T01-38-ov-41-qat-int4-blocksize-spinquant.json
  - runs/2026-11-26T01-38-ov-41-qat-int4-blocksize-spinquant.json
"""

from __future__ import annotations

import contextlib
import copy
import difflib
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

EXP_ID = "2026-11-26T01-38-ov-41-qat-int4-blocksize-spinquant"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

BUNDLE_CANDIDATES = (
    Path("/content/ov_bundle.tar.gz"),
    Path("/content/bundle.tar.gz"),
    Path("/content/repo_bundle.tar.gz"),
)
WORKSPACE_ROOT = Path("/workspace")
VOL_ROOT = Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

# Hyperparameters
SEED = 42
BASE_STEPS = 600
BASE_LR = 3e-4
QAT_STEPS = 200
QAT_LR = 1e-4
WEIGHT_DECAY = 0.04
BATCH_SIZE = 2
GRAD_ACCUM = 4
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
VAL_EVAL_MAX_LEN = 512
KL_MAX_PAGES = 24
SAVE_EVERY_MINUTES = 5.0
BLOCK_SIZES = (32, 64, 128)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _run_cmd(cmd: Sequence[str] | str, *, shell: bool = False) -> None:
    print(f"[{EXP_ID}] $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        shell=shell,
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


def _read_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and HF_TOKEN_FILE.exists():
        try:
            os.chmod(HF_TOKEN_FILE, 0o600)
        except OSError:
            pass
        token = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("HF token not found in /tmp/.hf_token or HF_TOKEN env var.")
    return token


def _resolve_repo_root() -> Path:
    candidates = [
        WORKSPACE_ROOT / "sheet-music-converter",
        Path("/content/sheet-music-converter"),
        Path.cwd(),
    ]
    for c in candidates:
        if (c / "tmp-train" / "mvd" / "spinquant_fold.py").is_file():
            return c
    for b in BUNDLE_CANDIDATES:
        if b.is_file():
            WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
            print(f"[{EXP_ID}] Extracting {b} -> {WORKSPACE_ROOT}...", flush=True)
            with tarfile.open(b, "r:*") as tf:
                tf.extractall(path=WORKSPACE_ROOT)
            for c in candidates:
                if (c / "tmp-train" / "mvd" / "spinquant_fold.py").is_file():
                    return c
            if (WORKSPACE_ROOT / "tmp-train" / "mvd" / "spinquant_fold.py").is_file():
                return WORKSPACE_ROOT
    raise FileNotFoundError("Could not locate tmp-train/mvd/spinquant_fold.py.")


def _setup_environment() -> Path:
    repo_root = _resolve_repo_root()
    needed = []
    for mod, pkg in [
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed])
    return repo_root


def _ensure_proxy_dataset(hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_root = VOL_ROOT / "datasets"
    proxy_dir = ds_root / "iter10_5_proxy_2k"
    train_abs = proxy_dir / "proxy_train_abs.jsonl"
    val_abs = proxy_dir / "proxy_val_abs.jsonl"
    if not (train_abs.is_file() and val_abs.is_file()):
        proxy_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR}...", flush=True)
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
        if img_p.is_file() and beks:
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


def _cosine_lr(step: int, total_steps: int, base_lr: float, warmup: int = 30, min_ratio: float = 0.1) -> float:
    if step <= warmup:
        return base_lr * float(step) / float(max(1, warmup))
    prog = min(1.0, max(0.0, float(step - warmup) / float(max(1, total_steps - warmup))))
    return base_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog)))


def _seq_f1(pred: Sequence[str], ref: Sequence[str]) -> float:
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    sm = difflib.SequenceMatcher(None, ref, pred, autojunk=False)
    matches = sum(blk.size for blk in sm.get_matching_blocks())
    return float(2.0 * matches) / float(len(pred) + len(ref))


def _build_optimizer(model: Any, lr: float, device: Any) -> Any:
    import torch

    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    kwargs: Dict[str, Any] = dict(lr=lr, betas=(0.9, 0.95), eps=1e-8)
    if device.type == "cuda":
        kwargs["fused"] = True
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        **kwargs,
    )


def _capture_fp32_ref_logits(
    model: Any,
    val_pairs: List[Tuple[Path, List[Path], str]],
    device: Any,
    *,
    use_bf16: bool,
) -> List[Tuple[Any, Any, Any]]:
    """Cache (images_cpu_u8, seq_cpu, ref_logits_cpu_fp16) on KL_MAX_PAGES val pages."""
    import torch
    from torch.utils.data import DataLoader
    from uni_omr import UniFullPageDataset, _images_to_device_float, collate_uni_batch

    model.eval()
    sub_pairs = val_pairs[:KL_MAX_PAGES]
    ds = UniFullPageDataset(
        sub_pairs,
        fullpage_max_dim=FULLPAGE_MAX_DIM,
        max_seq_len=MAX_SEQ_LEN,
        augment=False,
        return_uint8=True,
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_uni_batch)
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else contextlib.nullcontext()
    )
    cached: List[Tuple[Any, Any, Any]] = []
    with torch.no_grad():
        for batch in loader:
            imgs_u8 = batch["images"]
            seq = batch["seq"]
            imgs_dev = _images_to_device_float(imgs_u8, device, non_blocking=False)
            tgt_in = seq[:, :-1].to(device)
            with amp_ctx:
                logits = model(imgs_dev, tgt_in, return_aux=False, return_mtp=False)["logits"]
            cached.append((imgs_u8, seq, logits.detach().to(torch.float16).cpu()))
    return cached


def _eval_val_and_logit_fidelity(
    model: Any,
    val_pairs: List[Tuple[Path, List[Path], str]],
    ref_logit_batches: List[Tuple[Any, Any, Any]],
    device: Any,
    *,
    use_bf16: bool,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import train_uni_v2 as tv2
    from uni_omr import PAD_ID, UniFullPageDataset, _images_to_device_float, collate_uni_batch, decode_uni

    model.eval()
    val_ds = UniFullPageDataset(
        val_pairs,
        fullpage_max_dim=FULLPAGE_MAX_DIM,
        max_seq_len=MAX_SEQ_LEN,
        augment=False,
        return_uint8=True,
    )
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_uni_batch)
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else contextlib.nullcontext()
    )

    sum_loss = 0.0
    sum_ce = 0.0
    n_samples = 0
    f1_list: List[float] = []
    tot_edits = 0
    tot_ref_len = 0

    with torch.no_grad():
        for batch in val_loader:
            images = _images_to_device_float(batch["images"], device, non_blocking=True)
            seq = batch["seq"].to(device, non_blocking=True)
            bs = int(images.shape[0])
            losses, _ = tv2.forward_and_loss_v2(
                model,
                images,
                seq,
                kind="fullpage",
                label_smoothing=0.0,
                use_bf16=use_bf16,
                device=device,
            )
            sum_loss += float(losses["loss"].item()) * bs
            sum_ce += float(losses["ce_loss"].item()) * bs
            n_samples += bs

            with amp_ctx:
                gen_ids = model.generate(images, max_len=VAL_EVAL_MAX_LEN)
            for b in range(bs):
                ref_toks = decode_uni(seq[b].tolist(), skip_special=True)
                pred_toks = decode_uni(gen_ids[b].tolist(), skip_special=True)
                f1_list.append(_seq_f1(pred_toks, ref_toks))
                sm = difflib.SequenceMatcher(None, ref_toks, pred_toks, autojunk=False)
                matches = sum(blk.size for blk in sm.get_matching_blocks())
                tot_edits += max(len(ref_toks), len(pred_toks)) - matches
                tot_ref_len += max(1, len(ref_toks))

    # Measure logit KL(P_ref || P_q), MSE, max_abs_diff, and top-1 token agreement on ref_logit_batches
    kl_sum = 0.0
    mse_sum = 0.0
    max_abs = 0.0
    agree_sum = 0.0
    tok_count = 0.0
    with torch.no_grad():
        for imgs_u8, seq, ref_logits_cpu in ref_logit_batches:
            imgs_dev = _images_to_device_float(imgs_u8, device, non_blocking=False)
            tgt_in = seq[:, :-1].to(device)
            tgt_out = seq[:, 1:].to(device)
            valid = (tgt_out != PAD_ID).float()
            with amp_ctx:
                q_logits = model(imgs_dev, tgt_in, return_aux=False, return_mtp=False)["logits"].float()
            r_logits = ref_logits_cpu.to(device=device, dtype=torch.float32)

            log_p_r = F.log_softmax(r_logits, dim=-1)
            log_p_q = F.log_softmax(q_logits, dim=-1)
            p_r = log_p_r.exp()
            kl_tok = (p_r * (log_p_r - log_p_q)).sum(dim=-1)
            mse_tok = (q_logits - r_logits).pow(2).mean(dim=-1)
            agree_tok = (q_logits.argmax(dim=-1) == r_logits.argmax(dim=-1)).float()
            diff_max = float((q_logits - r_logits).abs().max().item())

            kl_sum += float((kl_tok * valid).sum().item())
            mse_sum += float((mse_tok * valid).sum().item())
            agree_sum += float((agree_tok * valid).sum().item())
            tok_count += float(valid.sum().item())
            if diff_max > max_abs:
                max_abs = diff_max

    return {
        "n_val_samples": n_samples,
        "val_loss": round(sum_loss / max(1, n_samples), 6),
        "val_ce_loss": round(sum_ce / max(1, n_samples), 6),
        "val_token_f1": round(sum(f1_list) / max(1, len(f1_list)), 6),
        "val_ser": round(tot_edits / max(1, tot_ref_len), 6),
        "logit_kl_vs_fp32": round(kl_sum / max(1.0, tok_count), 6),
        "logit_mse_vs_fp32": round(mse_sum / max(1.0, tok_count), 6),
        "logit_max_abs_diff": round(max_abs, 6),
        "top1_token_agree": round(agree_sum / max(1.0, tok_count), 6),
    }


def _train_or_load_base(
    train_pairs: List[Tuple[Path, List[Path], str]],
    device: Any,
    *,
    use_bf16: bool,
) -> Tuple[Any, Dict[str, Any]]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import train_uni_v2 as tv2
    from uni_omr import GPUEmaShadow, UniFullPageDataset, _images_to_device_float, _infinite_loader, collate_uni_batch
    from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net

    base_dir = VOL_ROOT / "ckpts" / EXP_ID / "base_fp"
    base_dir.mkdir(parents=True, exist_ok=True)
    base_ckpt = base_dir / "base_fp_ema_600.pt"

    _seed_everything(SEED)
    cfg = UniOMRv2Config.small_45m(grad_checkpoint=True)
    model = UniOMRv2Net(cfg).to(device)

    if base_ckpt.is_file():
        print(f"[{EXP_ID}] Loading cached shared base checkpoint: {base_ckpt}", flush=True)
        sd = torch.load(str(base_ckpt), map_location=device, weights_only=True)
        model.load_state_dict(sd["model"], strict=True)
        return model, {"loaded_from_cache": True, "train_secs": 0.0}

    optimizer = _build_optimizer(model, BASE_LR, device)
    gpu_ema = GPUEmaShadow(model, decay=0.995)

    _seed_everything(SEED + 100)
    train_ds = UniFullPageDataset(
        train_pairs,
        fullpage_max_dim=FULLPAGE_MAX_DIM,
        max_seq_len=MAX_SEQ_LEN,
        augment=True,
        return_uint8=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_uni_batch,
    )
    train_iter = _infinite_loader(train_loader)

    model.train()
    t0 = time.perf_counter()
    last_save = time.time()
    buf_ce: List[float] = []

    for step in range(1, BASE_STEPS + 1):
        cur_lr = _cosine_lr(step, BASE_STEPS, BASE_LR, warmup=60, min_ratio=0.1)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr
        optimizer.zero_grad(set_to_none=True)
        step_ce = 0.0
        for _ in range(GRAD_ACCUM):
            batch = next(train_iter)
            images = _images_to_device_float(batch["images"], device, non_blocking=True)
            seq = batch["seq"].to(device, non_blocking=True)
            losses, _ = tv2.forward_and_loss_v2(
                model,
                images,
                seq,
                kind="fullpage",
                label_smoothing=0.05,
                use_bf16=use_bf16,
                device=device,
            )
            (losses["loss"] / float(GRAD_ACCUM)).backward()
            step_ce += float(losses["ce_loss"].detach().item()) / float(GRAD_ACCUM)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        gpu_ema.update(model, step)
        buf_ce.append(step_ce)

        if step % 100 == 0 or step == BASE_STEPS:
            print(
                f"  [base_fp] step {step:4d}/{BASE_STEPS} | lr={cur_lr:.2e} | ce={float(np.mean(buf_ce)):.4f}",
                flush=True,
            )
            buf_ce.clear()
        if (time.time() - last_save) >= SAVE_EVERY_MINUTES * 60.0:
            tv2.atomic_save_v2_latest(base_dir, model, cfg, step=step, gpu_ema=gpu_ema)
            last_save = time.time()

    train_secs = time.perf_counter() - t0
    ema_sd = gpu_ema.cpu_state_dict(model)
    model.load_state_dict({k: v.to(device) for k, v in ema_sd.items()}, strict=True)
    tv2.atomic_save_checkpoint(base_ckpt, {"step": BASE_STEPS, "model": ema_sd})

    del optimizer, gpu_ema, train_ds, train_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, {"loaded_from_cache": False, "train_secs": round(train_secs, 2)}


def _run_qat_arm(
    base_state_cpu: Dict[str, Any],
    block_size: int,
    use_spinquant: bool,
    train_pairs: List[Tuple[Path, List[Path], str]],
    val_pairs: List[Tuple[Path, List[Path], str]],
    ref_logit_batches: List[Tuple[Any, Any, Any]],
    fp32_ref_metrics: Dict[str, Any],
    device: Any,
    *,
    use_bf16: bool,
) -> Dict[str, Any]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import spinquant_fold as sq
    import train_uni_v2 as tv2
    from uni_omr import UniFullPageDataset, _images_to_device_float, _infinite_loader, collate_uni_batch
    from uni_omr_v2 import QATInt4ControllerV2, UniOMRv2Config, UniOMRv2Net

    arm_name = f"bs{block_size}_{'spinquant' if use_spinquant else 'no_spinquant'}"
    arm_dir = VOL_ROOT / "ckpts" / EXP_ID / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached = arm_dir / "arm_result.json"
    if cached.is_file():
        print(f"[{arm_name}] Loading cached arm_result.json", flush=True)
        return json.loads(cached.read_text(encoding="utf-8"))

    cfg = UniOMRv2Config.small_45m(grad_checkpoint=True)
    model = UniOMRv2Net(cfg).to(device)
    model.load_state_dict({k: v.to(device) for k, v in base_state_cpu.items()}, strict=True)
    ps = model.param_summary(int4_block_size=block_size)

    fold_info: Optional[Dict[str, Any]] = None
    fp32_fold_max_abs = 0.0
    if use_spinquant:
        fold_info = sq.fold_hadamard_v2(model, fold_r1_decoder=True, fold_r2_attn=True)
        # Verify FP32 functional invariance on first cached val batch
        model.eval()
        with torch.no_grad():
            imgs_u8, seq, ref_l = ref_logit_batches[0]
            imgs_dev = _images_to_device_float(imgs_u8, device, non_blocking=False)
            tgt_in = seq[:, :-1].to(device)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if (use_bf16 and device.type == "cuda")
                else contextlib.nullcontext()
            )
            with amp_ctx:
                fold_l = model(imgs_dev, tgt_in, return_aux=False, return_mtp=False)["logits"].float()
            fp32_fold_max_abs = float((fold_l - ref_l.to(device=device, dtype=torch.float32)).abs().max().item())

    # 1. Enable QAT-INT4 controller and measure 0-step PTQ fidelity
    qat_ctrl = QATInt4ControllerV2(model, block_size=block_size, enabled=True)
    print(
        f"\n[{arm_name}] block_size={block_size} spinquant={use_spinquant} | "
        f"quant_layers={qat_ctrl.num_layers} | hybrid_mb={ps['hybrid_fp16_int4_with_scales_mb']:.2f} | "
        f"fp32_fold_max_abs={fp32_fold_max_abs:.5f}",
        flush=True,
    )
    ptq_eval = _eval_val_and_logit_fidelity(
        model, val_pairs, ref_logit_batches, device, use_bf16=use_bf16
    )
    print(
        f"  [{arm_name} PTQ-0step] F1={ptq_eval['val_token_f1']:.4f} "
        f"KL={ptq_eval['logit_kl_vs_fp32']:.5f} MSE={ptq_eval['logit_mse_vs_fp32']:.5f} "
        f"top1={ptq_eval['top1_token_agree']:.4f}",
        flush=True,
    )

    # 2. Run 200 steps of QAT-INT4 fine-tuning with identical batch stream across arms
    optimizer = _build_optimizer(model, QAT_LR, device)
    _seed_everything(SEED + 500)
    train_ds = UniFullPageDataset(
        train_pairs,
        fullpage_max_dim=FULLPAGE_MAX_DIM,
        max_seq_len=MAX_SEQ_LEN,
        augment=True,
        return_uint8=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_uni_batch,
    )
    train_iter = _infinite_loader(train_loader)

    model.train()
    t0 = time.perf_counter()
    last_save = time.time()
    buf_ce: List[float] = []

    for step in range(1, QAT_STEPS + 1):
        cur_lr = _cosine_lr(step, QAT_STEPS, QAT_LR, warmup=20, min_ratio=0.1)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr
        optimizer.zero_grad(set_to_none=True)
        step_ce = 0.0
        for _ in range(GRAD_ACCUM):
            batch = next(train_iter)
            images = _images_to_device_float(batch["images"], device, non_blocking=True)
            seq = batch["seq"].to(device, non_blocking=True)
            losses, _ = tv2.forward_and_loss_v2(
                model,
                images,
                seq,
                kind="fullpage",
                label_smoothing=0.05,
                use_bf16=use_bf16,
                device=device,
            )
            (losses["loss"] / float(GRAD_ACCUM)).backward()
            step_ce += float(losses["ce_loss"].detach().item()) / float(GRAD_ACCUM)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        qat_ctrl.refresh_cache()
        buf_ce.append(step_ce)

        if step % 50 == 0 or step == QAT_STEPS:
            print(
                f"  [{arm_name} QAT] step {step:3d}/{QAT_STEPS} | lr={cur_lr:.2e} | ce={float(np.mean(buf_ce)):.4f}",
                flush=True,
            )
            buf_ce.clear()
        if (time.time() - last_save) >= SAVE_EVERY_MINUTES * 60.0:
            tv2.atomic_save_v2_latest(arm_dir, model, cfg, step=step)
            last_save = time.time()

    qat_secs = time.perf_counter() - t0
    tv2.atomic_save_v2_latest(arm_dir, model, cfg, step=QAT_STEPS)

    # 3. Post-QAT INT4 evaluation
    qat_eval = _eval_val_and_logit_fidelity(
        model, val_pairs, ref_logit_batches, device, use_bf16=use_bf16
    )
    ref_f1 = max(1e-6, float(fp32_ref_metrics["val_token_f1"]))
    ptq_f1_ret = round(float(ptq_eval["val_token_f1"]) / ref_f1, 6)
    qat_f1_ret = round(float(qat_eval["val_token_f1"]) / ref_f1, 6)
    print(
        f"  [{arm_name} QAT-200] F1={qat_eval['val_token_f1']:.4f} (ret={qat_f1_ret:.4f}) "
        f"KL={qat_eval['logit_kl_vs_fp32']:.5f} top1={qat_eval['top1_token_agree']:.4f}",
        flush=True,
    )

    qat_ctrl.detach()
    result = {
        "arm": arm_name,
        "block_size": block_size,
        "use_spinquant": use_spinquant,
        "fold_info": fold_info,
        "fp32_fold_max_abs_diff": round(fp32_fold_max_abs, 6),
        "hybrid_fp16_int4_with_scales_mb": round(float(ps["hybrid_fp16_int4_with_scales_mb"]), 4),
        "pure_int4_with_scales_mb": round(float(ps["pure_int4_with_scales_mb"]), 4),
        "qat_train_secs": round(qat_secs, 2),
        "ptq_0step": {**ptq_eval, "f1_retention_vs_fp32": ptq_f1_ret},
        "qat_200step": {**qat_eval, "f1_retention_vs_fp32": qat_f1_ret},
    }
    _atomic_write_json(cached, result)

    del model, optimizer, train_ds, train_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    try:
        hf_token = _read_hf_token()
        repo_root = _setup_environment()

        import torch
        from huggingface_hub import HfApi

        t_start = time.time()
        started_iso = datetime.now(timezone.utc).isoformat()

        if torch.cuda.is_available():
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
        use_bf16 = bool(torch.cuda.is_available() and cc[0] >= 8)
        print(f"[{EXP_ID}] Running on Colab {gpu_name} (cc={cc[0]}.{cc[1]}, bf16={use_bf16})", flush=True)

        mvd_dir = str(repo_root / "tmp-train" / "mvd")
        if mvd_dir not in sys.path:
            sys.path.insert(0, mvd_dir)

        train_abs, val_abs = _ensure_proxy_dataset(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)

        base_model, base_meta = _train_or_load_base(train_pairs, device, use_bf16=use_bf16)
        ref_logit_batches = _capture_fp32_ref_logits(base_model, val_pairs, device, use_bf16=use_bf16)
        fp32_ref = _eval_val_and_logit_fidelity(
            base_model, val_pairs, ref_logit_batches, device, use_bf16=use_bf16
        )
        print(
            f"[{EXP_ID}] FP32/BF16 Base Ref: val_token_f1={fp32_ref['val_token_f1']:.4f} "
            f"val_ce={fp32_ref['val_ce_loss']:.4f} val_ser={fp32_ref['val_ser']:.4f}",
            flush=True,
        )
        base_state_cpu = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        del base_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        arms: List[Dict[str, Any]] = []
        for bs in BLOCK_SIZES:
            for use_sq in (False, True):
                res = _run_qat_arm(
                    base_state_cpu,
                    bs,
                    use_sq,
                    train_pairs,
                    val_pairs,
                    ref_logit_batches,
                    fp32_ref,
                    device,
                    use_bf16=use_bf16,
                )
                arms.append(res)

        best_arm = max(
            arms,
            key=lambda a: (a["qat_200step"]["val_token_f1"], -a["qat_200step"]["logit_kl_vs_fp32"]),
        )
        summary = {
            "fp32_ref_val_token_f1": fp32_ref["val_token_f1"],
            "best_arm": best_arm["arm"],
            "best_block_size": best_arm["block_size"],
            "best_use_spinquant": best_arm["use_spinquant"],
            "best_qat_val_token_f1": best_arm["qat_200step"]["val_token_f1"],
            "best_qat_f1_retention": best_arm["qat_200step"]["f1_retention_vs_fp32"],
            "best_qat_logit_kl": best_arm["qat_200step"]["logit_kl_vs_fp32"],
            "best_hybrid_mb": best_arm["hybrid_fp16_int4_with_scales_mb"],
            "table": [
                {
                    "arm": a["arm"],
                    "block_size": a["block_size"],
                    "spinquant": a["use_spinquant"],
                    "hybrid_mb": a["hybrid_fp16_int4_with_scales_mb"],
                    "ptq_f1": a["ptq_0step"]["val_token_f1"],
                    "ptq_kl": a["ptq_0step"]["logit_kl_vs_fp32"],
                    "qat_f1": a["qat_200step"]["val_token_f1"],
                    "qat_f1_ret": a["qat_200step"]["f1_retention_vs_fp32"],
                    "qat_kl": a["qat_200step"]["logit_kl_vs_fp32"],
                }
                for a in arms
            ],
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
                "seed": SEED,
                "base_steps": BASE_STEPS,
                "base_lr": BASE_LR,
                "qat_steps": QAT_STEPS,
                "qat_lr": QAT_LR,
                "block_sizes": list(BLOCK_SIZES),
            },
            "base_meta": base_meta,
            "fp32_reference": fp32_ref,
            "arms": arms,
            "summary": summary,
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
            commit_message=f"eval({EXP_ID}): upload QAT-INT4 block_size x SpinQuant sweep",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload QAT-INT4 sweep run ledger",
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
