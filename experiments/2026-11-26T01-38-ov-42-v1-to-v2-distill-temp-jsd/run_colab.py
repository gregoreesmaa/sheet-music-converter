#!/usr/bin/env python3
"""Google Colab L4 runner for Overnight Exp 42: v1 -> v2 Online Distillation Temp x G-JSD/KL x Alpha Sweep.

Experiment ID: 2026-11-26T01-38-ov-42-v1-to-v2-distill-temp-jsd
Controlled 2-Stage Sweep on UniOMR v2 small_45m with frozen v1 teacher (uni_omr_soup.pt):
  - Control: alpha=0.0 (pure supervised small_45m, 350 steps, seed=42)
  - Stage 1 (6 arms x 350 steps at alpha=0.50):
      T in {1.0, 2.0, 4.0} x objective in {G-JSD (beta=0.5), Forward KL (beta=1.0)}
  - Stage 2 (2 arms x 350 steps at winning (T*, beta*)):
      alpha in {0.25, 0.75} (completing alpha in {0.0, 0.25, 0.50, 0.75})

Includes automatic vocab-size alignment between frozen v1 teacher and v2 student.

Uploads to Hugging Face Private Dataset Hub (gregoreesmaa/sheet-music-converter-omr):
  - evals/2026-11-26T01-38-ov-42-v1-to-v2-distill-temp-jsd.json
  - runs/2026-11-26T01-38-ov-42-v1-to-v2-distill-temp-jsd.json
"""

from __future__ import annotations

import contextlib
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

EXP_ID = "2026-11-26T01-38-ov-42-v1-to-v2-distill-temp-jsd"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_TEACHER_CKPT = "checkpoints/iter10_soup_baseline/uni_omr_soup.pt"
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

# Matched training hyperparameters
SEED = 42
ARM_STEPS = 350
LR = 3e-4
WEIGHT_DECAY = 0.04
BATCH_SIZE = 2
GRAD_ACCUM = 4
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
WARMUP_STEPS = 40
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
VAL_EVAL_MAX_LEN = 512
SAVE_EVERY_MINUTES = 5.0

TEMPS = (1.0, 2.0, 4.0)
OBJECTIVES = (("jsd", 0.5), ("fkl", 1.0))
STAGE1_ALPHA = 0.50
STAGE2_ALPHAS = (0.25, 0.75)


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
        if (c / "tmp-train" / "mvd" / "train_uni_v2.py").is_file():
            return c
    for b in BUNDLE_CANDIDATES:
        if b.is_file():
            WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
            print(f"[{EXP_ID}] Extracting {b} -> {WORKSPACE_ROOT}...", flush=True)
            with tarfile.open(b, "r:*") as tf:
                tf.extractall(path=WORKSPACE_ROOT)
            for c in candidates:
                if (c / "tmp-train" / "mvd" / "train_uni_v2.py").is_file():
                    return c
            if (WORKSPACE_ROOT / "tmp-train" / "mvd" / "train_uni_v2.py").is_file():
                return WORKSPACE_ROOT
    raise FileNotFoundError("Could not locate tmp-train/mvd/train_uni_v2.py.")


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


def _ensure_assets(hf_token: str) -> Tuple[Path, Path, Path]:
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

    teacher_dir = VOL_ROOT / "teachers"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    teacher_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_TEACHER_CKPT,
            repo_type="dataset",
            token=hf_token,
            local_dir=str(teacher_dir),
        )
    )
    return train_abs, val_abs, teacher_path


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


def _wsd_lr(step: int, total_steps: int, base_lr: float) -> float:
    if step <= WARMUP_STEPS:
        return base_lr * float(step) / float(max(1, WARMUP_STEPS))
    decay_steps = max(1, int(round(total_steps * WSD_DECAY_FRAC)))
    decay_start = total_steps - decay_steps
    if step <= decay_start:
        return base_lr
    prog = min(1.0, max(0.0, float(step - decay_start) / float(decay_steps)))
    return base_lr * (MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * 0.5 * (1.0 + math.cos(math.pi * prog)))


def _seq_f1(pred: Sequence[str], ref: Sequence[str]) -> float:
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    sm = difflib.SequenceMatcher(None, ref, pred, autojunk=False)
    matches = sum(blk.size for blk in sm.get_matching_blocks())
    return float(2.0 * matches) / float(len(pred) + len(ref))


def _map_student_ids_to_teacher(
    tgt_in: Any,
    s_vocab: int,
    t_vocab: int,
) -> Any:
    """Remap append-only student vocab IDs (base + 4 specials) into teacher vocab space."""
    import torch

    if s_vocab == t_vocab:
        return tgt_in
    s_base = s_vocab - 4
    t_base = t_vocab - 4
    t_pad, t_bos, t_eos, t_sys = t_base, t_base + 1, t_base + 2, t_base + 3
    s_pad, s_bos, s_eos, s_sys = s_base, s_base + 1, s_base + 2, s_base + 3
    out = tgt_in.clone()
    # Any newly appended base token >= t_base maps safely to t_pad for teacher context
    new_base_mask = (tgt_in >= t_base) & (tgt_in < s_base)
    out = torch.where(new_base_mask, torch.full_like(out, t_pad), out)
    out = torch.where(tgt_in == s_pad, torch.full_like(out, t_pad), out)
    out = torch.where(tgt_in == s_bos, torch.full_like(out, t_bos), out)
    out = torch.where(tgt_in == s_eos, torch.full_like(out, t_eos), out)
    out = torch.where(tgt_in == s_sys, torch.full_like(out, t_sys), out)
    return out


def _align_student_teacher_logits(
    s_logits: Any,
    t_logits: Any,
) -> Tuple[Any, Any]:
    """Align logits over shared append-only vocabulary (base[:min_base] + 4 specials)."""
    import torch

    s_v = int(s_logits.shape[-1])
    t_v = int(t_logits.shape[-1])
    if s_v == t_v:
        return s_logits, t_logits
    n_base = min(s_v - 4, t_v - 4)
    s_aligned = torch.cat([s_logits[..., :n_base], s_logits[..., s_v - 4 :]], dim=-1)
    t_aligned = torch.cat([t_logits[..., :n_base], t_logits[..., t_v - 4 :]], dim=-1)
    return s_aligned, t_aligned


def _eval_proxy_val(
    model: Any,
    val_pairs: List[Tuple[Path, List[Path], str]],
    device: Any,
    *,
    use_bf16: bool,
) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    import train_uni_v2 as tv2
    from uni_omr import UniFullPageDataset, _images_to_device_float, collate_uni_batch, decode_uni

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
    sys_exact = 0
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
                if ref_toks.count("<sys>") == pred_toks.count("<sys>"):
                    sys_exact += 1
                sm = difflib.SequenceMatcher(None, ref_toks, pred_toks, autojunk=False)
                matches = sum(blk.size for blk in sm.get_matching_blocks())
                tot_edits += max(len(ref_toks), len(pred_toks)) - matches
                tot_ref_len += max(1, len(ref_toks))

    return {
        "n_samples": n_samples,
        "val_loss": round(sum_loss / max(1, n_samples), 6),
        "val_ce_loss": round(sum_ce / max(1, n_samples), 6),
        "val_token_f1": round(sum(f1_list) / max(1, len(f1_list)), 6),
        "val_ser": round(tot_edits / max(1, tot_ref_len), 6),
        "val_sys_count_exact": round(sys_exact / max(1, n_samples), 4),
    }


def _run_distill_arm(
    arm_name: str,
    kd_temp: float,
    kd_beta: float,
    kd_alpha: float,
    teacher_v1: Any,
    train_pairs: List[Tuple[Path, List[Path], str]],
    val_pairs: List[Tuple[Path, List[Path], str]],
    device: Any,
    *,
    use_bf16: bool,
) -> Dict[str, Any]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import train_uni_v2 as tv2
    from uni_omr import (
        GPUEmaShadow,
        PAD_ID,
        UniFullPageDataset,
        _images_to_device_float,
        _infinite_loader,
        collate_uni_batch,
    )
    from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net

    arm_dir = VOL_ROOT / "ckpts" / EXP_ID / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached = arm_dir / "arm_result.json"
    if cached.is_file():
        print(f"[{arm_name}] Loading cached arm_result.json", flush=True)
        return json.loads(cached.read_text(encoding="utf-8"))

    _seed_everything(SEED)
    cfg = UniOMRv2Config.small_45m(grad_checkpoint=True)
    model = UniOMRv2Net(cfg).to(device)

    s_vocab = int(cfg.vocab_size)
    t_vocab = int(teacher_v1.cfg.vocab_size) if teacher_v1 is not None else s_vocab
    use_kd = bool(kd_alpha > 0.0 and teacher_v1 is not None)

    print(
        f"\n[{arm_name}] T={kd_temp} beta={kd_beta} alpha={kd_alpha} | "
        f"s_vocab={s_vocab} t_vocab={t_vocab} | steps={ARM_STEPS}",
        flush=True,
    )

    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    opt_kwargs: Dict[str, Any] = dict(lr=LR, betas=(0.9, 0.95), eps=1e-8)
    if device.type == "cuda":
        opt_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        **opt_kwargs,
    )
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

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else contextlib.nullcontext()
    )

    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    last_save = time.time()
    all_kd: List[float] = []
    all_ms: List[float] = []
    buf_ce: List[float] = []
    buf_kd: List[float] = []

    for step in range(1, ARM_STEPS + 1):
        ts0 = time.perf_counter()
        cur_lr = _wsd_lr(step, ARM_STEPS, LR)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr
        cur_alpha = tv2.kd_alpha_for_step(step, ARM_STEPS, kd_alpha, anneal_last_frac=0.25) if use_kd else 0.0

        optimizer.zero_grad(set_to_none=True)
        step_ce = 0.0
        step_kd = 0.0

        for _ in range(GRAD_ACCUM):
            batch = next(train_iter)
            images = _images_to_device_float(batch["images"], device, non_blocking=True)
            seq = batch["seq"].to(device, non_blocking=True)

            if use_kd and s_vocab == t_vocab:
                losses, _ = tv2.forward_and_loss_v2(
                    model,
                    images,
                    seq,
                    teacher_v1=teacher_v1,
                    kd_alpha=cur_alpha,
                    kd_beta=kd_beta,
                    kd_temp=kd_temp,
                    kind="fullpage",
                    label_smoothing=0.05,
                    use_bf16=use_bf16,
                    device=device,
                )
                total_loss = losses["loss"]
                kd_val = float(losses["kd_loss"].detach().item())
            else:
                losses, out = tv2.forward_and_loss_v2(
                    model,
                    images,
                    seq,
                    kind="fullpage",
                    label_smoothing=0.05,
                    use_bf16=use_bf16,
                    device=device,
                )
                total_loss = losses["loss"]
                kd_val = 0.0
                if use_kd and cur_alpha > 0.0:
                    tgt_in = seq[:, :-1]
                    tgt_out = seq[:, 1:]
                    t_in = _map_student_ids_to_teacher(tgt_in, s_vocab, t_vocab)
                    with amp_ctx:
                        with torch.no_grad():
                            t_out = teacher_v1(images, t_in, return_aux=False, return_mtp=False)
                        s_al, t_al = _align_student_teacher_logits(out["logits"], t_out["logits"])
                        kd_loss = tv2.compute_generalized_jsd_loss(
                            s_al,
                            t_al,
                            tgt_out,
                            beta=kd_beta,
                            temp=kd_temp,
                            ignore_index=PAD_ID,
                        )
                        total_loss = total_loss + float(cur_alpha) * kd_loss
                    kd_val = float(kd_loss.detach().item())

            (total_loss / float(GRAD_ACCUM)).backward()
            step_ce += float(losses["ce_loss"].detach().item()) / float(GRAD_ACCUM)
            step_kd += kd_val / float(GRAD_ACCUM)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        gpu_ema.update(model, step)

        step_ms = (time.perf_counter() - ts0) * 1000.0
        buf_ce.append(step_ce)
        buf_kd.append(step_kd)
        all_kd.append(step_kd)
        all_ms.append(step_ms)

        if step % 70 == 0 or step == ARM_STEPS:
            print(
                f"  [{arm_name}] step {step:3d}/{ARM_STEPS} | lr={cur_lr:.2e} alpha={cur_alpha:.3f} | "
                f"ce={float(np.mean(buf_ce)):.4f} kd={float(np.mean(buf_kd)):.4f} | {step_ms:.0f} ms",
                flush=True,
            )
            buf_ce.clear()
            buf_kd.clear()

        if (time.time() - last_save) >= SAVE_EVERY_MINUTES * 60.0:
            tv2.atomic_save_v2_latest(arm_dir, model, cfg, step=step, gpu_ema=gpu_ema)
            last_save = time.time()

    train_secs = time.perf_counter() - t0
    peak_vram_gb = (
        round(float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 3), 3)
        if device.type == "cuda"
        else 0.0
    )
    tv2.atomic_save_v2_latest(arm_dir, model, cfg, step=ARM_STEPS, gpu_ema=gpu_ema)

    ema_sd = gpu_ema.cpu_state_dict(model)
    model.load_state_dict({k: v.to(device) for k, v in ema_sd.items()}, strict=True)
    val_eval = _eval_proxy_val(model, val_pairs, device, use_bf16=use_bf16)
    print(
        f"  [{arm_name} VAL] F1={val_eval['val_token_f1']:.4f} CE={val_eval['val_ce_loss']:.4f} "
        f"SER={val_eval['val_ser']:.4f} sys_exact={val_eval['val_sys_count_exact']:.3f}",
        flush=True,
    )

    result = {
        "arm": arm_name,
        "kd_temp": kd_temp,
        "kd_beta": kd_beta,
        "kd_objective": "none" if kd_alpha <= 0.0 else ("jsd" if kd_beta < 0.9 else "fkl"),
        "kd_alpha": kd_alpha,
        "train_steps": ARM_STEPS,
        "train_secs": round(train_secs, 2),
        "mean_step_ms": round(float(np.mean(all_ms)), 2),
        "mean_kd_loss": round(float(np.mean(all_kd)), 6),
        "peak_vram_gb": peak_vram_gb,
        "proxy_val_eval": val_eval,
    }
    _atomic_write_json(cached, result)

    del model, optimizer, gpu_ema, train_ds, train_loader
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

        import train_uni_v2 as tv2

        train_abs, val_abs, teacher_path = _ensure_assets(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)
        teacher_v1 = tv2.load_v1_teacher(teacher_path, device)

        # 0. Pure supervised control (alpha = 0.0)
        ctrl_arm = _run_distill_arm(
            "ctrl_no_kd",
            kd_temp=2.0,
            kd_beta=0.5,
            kd_alpha=0.0,
            teacher_v1=teacher_v1,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            device=device,
            use_bf16=use_bf16,
        )

        # 1. Stage 1: Temp x Objective sweep at alpha = 0.50
        stage1_arms: List[Dict[str, Any]] = []
        for temp in TEMPS:
            for obj_name, beta in OBJECTIVES:
                arm_name = f"t{temp:.1f}_{obj_name}_a{STAGE1_ALPHA:.2f}"
                res = _run_distill_arm(
                    arm_name,
                    kd_temp=temp,
                    kd_beta=beta,
                    kd_alpha=STAGE1_ALPHA,
                    teacher_v1=teacher_v1,
                    train_pairs=train_pairs,
                    val_pairs=val_pairs,
                    device=device,
                    use_bf16=use_bf16,
                )
                stage1_arms.append(res)

        best_s1 = max(
            stage1_arms,
            key=lambda a: (a["proxy_val_eval"]["val_token_f1"], -a["proxy_val_eval"]["val_ce_loss"]),
        )
        best_temp = float(best_s1["kd_temp"])
        best_beta = float(best_s1["kd_beta"])
        best_obj = str(best_s1["kd_objective"])
        print(
            f"\n[{EXP_ID}] Stage 1 Winner: {best_s1['arm']} "
            f"(T={best_temp}, obj={best_obj}, F1={best_s1['proxy_val_eval']['val_token_f1']:.4f})",
            flush=True,
        )

        # 2. Stage 2: Alpha sweep at winning (T*, beta*)
        stage2_arms: List[Dict[str, Any]] = []
        for alpha in STAGE2_ALPHAS:
            arm_name = f"t{best_temp:.1f}_{best_obj}_a{alpha:.2f}"
            res = _run_distill_arm(
                arm_name,
                kd_temp=best_temp,
                kd_beta=best_beta,
                kd_alpha=alpha,
                teacher_v1=teacher_v1,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                device=device,
                use_bf16=use_bf16,
            )
            stage2_arms.append(res)

        all_arms = [ctrl_arm, *stage1_arms, *stage2_arms]
        ctrl_f1 = float(ctrl_arm["proxy_val_eval"]["val_token_f1"])
        ctrl_ce = float(ctrl_arm["proxy_val_eval"]["val_ce_loss"])
        best_overall = max(
            all_arms,
            key=lambda a: (a["proxy_val_eval"]["val_token_f1"], -a["proxy_val_eval"]["val_ce_loss"]),
        )

        summary = {
            "ctrl_val_token_f1": ctrl_f1,
            "ctrl_val_ce_loss": ctrl_ce,
            "stage1_winner": best_s1["arm"],
            "best_overall_arm": best_overall["arm"],
            "best_kd_temp": best_overall["kd_temp"],
            "best_kd_beta": best_overall["kd_beta"],
            "best_kd_alpha": best_overall["kd_alpha"],
            "best_val_token_f1": best_overall["proxy_val_eval"]["val_token_f1"],
            "delta_val_tok_f1_vs_ctrl": round(best_overall["proxy_val_eval"]["val_token_f1"] - ctrl_f1, 6),
            "delta_val_ce_vs_ctrl": round(best_overall["proxy_val_eval"]["val_ce_loss"] - ctrl_ce, 6),
            "table": [
                {
                    "arm": a["arm"],
                    "T": a["kd_temp"],
                    "obj": a["kd_objective"],
                    "alpha": a["kd_alpha"],
                    "val_f1": a["proxy_val_eval"]["val_token_f1"],
                    "delta_f1": round(a["proxy_val_eval"]["val_token_f1"] - ctrl_f1, 6),
                    "val_ce": a["proxy_val_eval"]["val_ce_loss"],
                    "val_ser": a["proxy_val_eval"]["val_ser"],
                    "step_ms": a["mean_step_ms"],
                }
                for a in all_arms
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
            "teacher_ckpt": HF_TEACHER_CKPT,
            "hyperparams": {
                "preset": "small_45m",
                "seed": SEED,
                "arm_steps": ARM_STEPS,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM,
                "temps": list(TEMPS),
                "objectives": [o[0] for o in OBJECTIVES],
                "alphas": [0.0, 0.25, 0.50, 0.75],
            },
            "control_arm": ctrl_arm,
            "stage1_arms": stage1_arms,
            "stage2_arms": stage2_arms,
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
            commit_message=f"eval({EXP_ID}): upload v1->v2 distillation sweep results",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload v1->v2 distillation run ledger",
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
