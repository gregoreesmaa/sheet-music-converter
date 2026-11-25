#!/usr/bin/env python3
"""Google Colab L4 runner for Overnight Exp 27: MTP-k x MTP Loss Weight Decoder Sweep.

Experiment ID: 2026-11-26T01-38-ov-27-mtp-k-weight-sweep
Controlled 3x3 Full-Factorial Sweep on UniOMR v2 small_45m (BF16, seed=42, lr=3e-4, 300 steps/arm):
  - mtp_k in {2, 4, 6}
  - mtp_loss_weight in {0.15, 0.30, 0.50}

Measures for every arm:
  1. proxy_val (64 pages): val_token_f1, val_ser, val_ce_loss, val_mtp_loss, mtp_params
  2. Lossless self-speculative MTP decoding benchmark (16 val pages, B=1, max_len=384):
     bit-exact parity vs greedy generate(), draft_acceptance_rate, mean_accepted_tokens_per_step,
     greedy_tok_per_sec, spec_tok_per_sec, and speedup_x.

Uploads results to Hugging Face Private Dataset Hub (gregoreesmaa/sheet-music-converter-omr):
  - evals/2026-11-26T01-38-ov-27-mtp-k-weight-sweep.json
  - runs/2026-11-26T01-38-ov-27-mtp-k-weight-sweep.json
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

EXP_ID = "2026-11-26T01-38-ov-27-mtp-k-weight-sweep"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

WORKSPACE_ROOT = Path("/workspace")
VOL_ROOT = Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")
BUNDLE_CANDIDATES = (
    Path("/content/bundle.tar.gz"),
    Path("/content/repo_bundle.tar.gz"),
)

# 3x3 full-factorial grid: (arm_name, mtp_k, mtp_loss_weight)
ARMS: List[Tuple[str, int, float]] = [
    ("mtp2_w0.15", 2, 0.15),
    ("mtp2_w0.30", 2, 0.30),
    ("mtp2_w0.50", 2, 0.50),
    ("mtp4_w0.15", 4, 0.15),
    ("mtp4_w0.30", 4, 0.30),
    ("mtp4_w0.50", 4, 0.50),
    ("mtp6_w0.15", 6, 0.15),
    ("mtp6_w0.30", 6, 0.30),
    ("mtp6_w0.50", 6, 0.50),
]

# Matched training hyperparameters
TRAIN_STEPS = 300
SEED = 42
LR = 3e-4
WEIGHT_DECAY = 0.04
BATCH_SIZE = 2
GRAD_ACCUM = 4
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
WARMUP_STEPS = 40
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
EVAL_MAX_LEN = 512
SPEC_BENCH_PAGES = 16
SPEC_MAX_LEN = 384
SAVE_EVERY_MINUTES = 5.0


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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


def _setup_environment() -> Path:
    mvd_candidates = [
        WORKSPACE_ROOT / "sheet-music-converter" / "tmp-train" / "mvd",
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path.cwd() / "tmp-train" / "mvd",
    ]
    mvd_dir: Optional[Path] = next((p for p in mvd_candidates if p.is_dir()), None)

    if mvd_dir is None:
        for bpath in BUNDLE_CANDIDATES:
            if bpath.is_file():
                WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
                print(f"[{EXP_ID}] Extracting {bpath} -> {WORKSPACE_ROOT}...", flush=True)
                with tarfile.open(bpath, "r:*") as tf:
                    tf.extractall(path=WORKSPACE_ROOT)
                mvd_dir = next((p for p in mvd_candidates if p.is_dir()), None)
                if mvd_dir is not None:
                    break

    if mvd_dir is None:
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        repo_dir = WORKSPACE_ROOT / "sheet-music-converter"
        _run_cmd([
            "git", "clone", "--depth", "1",
            "https://github.com/gregoreesmaa/sheet-music-converter.git",
            str(repo_dir),
        ])
        mvd_dir = repo_dir / "tmp-train" / "mvd"

    if not mvd_dir.is_dir():
        raise FileNotFoundError(f"Could not locate tmp-train/mvd (checked {mvd_candidates})")

    needed_pkgs = []
    for mod, pkg in [
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            needed_pkgs.append(pkg)
    if needed_pkgs:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed_pkgs])

    return mvd_dir


def _ensure_proxy_dataset(hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_root = VOL_ROOT / "datasets"
    proxy_dir = ds_root / "iter10_5_proxy_2k"
    train_abs = proxy_dir / "proxy_train_abs.jsonl"
    val_abs = proxy_dir / "proxy_val_abs.jsonl"
    if not (train_abs.is_file() and val_abs.is_file()):
        proxy_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR}...", flush=True)
        t0 = time.perf_counter()
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_PROXY_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        print(f"[{EXP_ID}] Downloaded proxy tar in {time.perf_counter() - t0:.1f}s -> extracting...", flush=True)
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
            print(f"[{EXP_ID}] Wrote {len(out_lines)} rows -> {dst_path.name}", flush=True)
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


def _wsd_lr(step: int, total_steps: int, base_lr: float) -> float:
    if step <= WARMUP_STEPS:
        return base_lr * float(step) / float(max(1, WARMUP_STEPS))
    decay_steps = max(1, int(round(total_steps * WSD_DECAY_FRAC)))
    decay_start = total_steps - decay_steps
    if step <= decay_start:
        return base_lr
    prog = min(1.0, max(0.0, float(step - decay_start) / float(decay_steps)))
    min_lr = base_lr * MIN_LR_RATIO
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * prog))


def _seq_f1_and_edits(pred: Sequence[str], ref: Sequence[str]) -> Tuple[float, int]:
    if not pred and not ref:
        return 1.0, 0
    if not pred or not ref:
        return 0.0, max(len(pred), len(ref))
    sm = difflib.SequenceMatcher(None, ref, pred, autojunk=False)
    matches = sum(b.size for b in sm.get_matching_blocks())
    f1 = (2.0 * matches) / float(len(pred) + len(ref))
    edits = max(len(ref), len(pred)) - matches
    return f1, edits


def _trim_self_kv(past_kvs: List[Any], keep_len: int) -> List[Any]:
    trimmed: List[Any] = []
    for self_kv, cross_kv in past_kvs:
        sk, sv = self_kv
        trimmed.append(((sk[:, :, :keep_len, :], sv[:, :, :keep_len, :]), cross_kv))
    return trimmed


def _speculative_generate_b1(
    model: Any,
    image: "torch.Tensor",
    *,
    max_len: int,
    use_bf16: bool,
) -> Tuple[List[int], Dict[str, Any]]:
    """Lossless B=1 MTP self-speculative greedy decoding on UniOMRv2Net."""
    import torch
    from uni_omr import BOS_ID, EOS_ID

    device = image.device
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else contextlib.nullcontext()
    )

    with torch.no_grad(), amp_ctx:
        mem, _, _ = model.encode_image(image, return_aux=False)
        past_kvs = model.precompute_cross_kv(mem)

        # Prefill BOS to obtain first greedy token d_0 and (k-1) MTP drafts
        bos = torch.full((1, 1), BOS_ID, dtype=torch.long, device=device)
        logits, mtp_logits, past_kvs = model.decode_tokens(
            bos, memory=None, return_mtp=True, return_aux=False, past_kvs=past_kvs, use_cache=True
        )
        d0 = int(logits[0, -1].argmax(dim=-1).item())
        draft_tail = [int(ml[0, -1].argmax(dim=-1).item()) for ml in mtp_logits]

        generated: List[int] = [BOS_ID]
        n_verify_steps = 0
        total_draft_proposed = 0
        total_draft_accepted = 0
        total_committed = 0

        if d0 == EOS_ID or max_len <= 1:
            generated.append(d0)
            return generated, {
                "n_tokens": len(generated) - 1,
                "n_verify_steps": 1,
                "draft_proposed": 0,
                "draft_accepted": 0,
                "accepted_per_step": 1.0,
            }

        k_draft = len(draft_tail)
        cur_chunk = [d0] + draft_tail
        kv_len = 1  # [BOS] is in KV cache

        while len(generated) - 1 < max_len:
            rem = max_len - (len(generated) - 1)
            chunk = cur_chunk[: min(len(cur_chunk), rem)]
            chunk_t = torch.tensor([chunk], dtype=torch.long, device=device)
            logits_c, mtp_c, new_past_kvs = model.decode_tokens(
                chunk_t, memory=None, return_mtp=True, return_aux=False, past_kvs=past_kvs, use_cache=True
            )
            n_verify_steps += 1
            verified = logits_c[0].argmax(dim=-1).tolist()  # len == len(chunk)

            # chunk[0] was already greedy from previous step; verify chunk[1..]
            n_match = 0
            n_check = len(chunk) - 1
            total_draft_proposed += n_check
            for m in range(n_check):
                if chunk[m + 1] == verified[m]:
                    n_match += 1
                else:
                    break
            total_draft_accepted += n_match

            committed = chunk[: 1 + n_match]
            hit_eos = False
            for tok in committed:
                generated.append(tok)
                total_committed += 1
                if tok == EOS_ID or (len(generated) - 1) >= max_len:
                    hit_eos = True
                    break
            if hit_eos:
                break

            # Trim self-KV cache to include exactly the committed prefix
            kv_len = kv_len + 1 + n_match
            past_kvs = _trim_self_kv(new_past_kvs, kv_len)

            # Next greedy token + its (k-1) MTP drafts from position n_match
            next_d0 = int(verified[n_match])
            if next_d0 == EOS_ID:
                generated.append(EOS_ID)
                total_committed += 1
                break
            next_tail = [int(ml[0, n_match].argmax(dim=-1).item()) for ml in mtp_c]
            cur_chunk = [next_d0] + next_tail[:k_draft]

    return generated, {
        "n_tokens": len(generated) - 1,
        "n_verify_steps": max(1, n_verify_steps),
        "draft_proposed": total_draft_proposed,
        "draft_accepted": total_draft_accepted,
        "accepted_per_step": float(total_committed) / float(max(1, n_verify_steps)),
    }


def _eval_val_and_speculative(
    model: Any,
    val_pairs: List[Tuple[Path, List[Path], str]],
    device: "torch.device",
    *,
    mtp_loss_weight: float,
    use_bf16: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader
    import train_uni_v2 as tv2
    from uni_omr import (
        UniFullPageDataset,
        _images_to_device_float,
        collate_uni_batch,
        decode_uni,
    )

    model.eval()
    val_ds = UniFullPageDataset(
        val_pairs,
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

    sum_ce = 0.0
    sum_mtp = 0.0
    n_samples = 0
    all_f1: List[float] = []
    tot_edits = 0
    tot_ref_len = 0
    sys_exact = 0

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else contextlib.nullcontext()
    )

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
                mtp_loss_weight=mtp_loss_weight,
                label_smoothing=0.0,
                use_bf16=use_bf16,
                device=device,
            )
            sum_ce += float(losses["ce_loss"].item()) * bs
            sum_mtp += float(losses["mtp_loss"].item()) * bs
            n_samples += bs

            with amp_ctx:
                gen_ids = model.generate(images, max_len=EVAL_MAX_LEN)

            for b in range(bs):
                ref_toks = decode_uni(seq[b].tolist(), skip_special=True)
                pred_toks = decode_uni(gen_ids[b].tolist(), skip_special=True)
                f1, edits = _seq_f1_and_edits(pred_toks, ref_toks)
                all_f1.append(f1)
                tot_edits += edits
                tot_ref_len += max(1, len(ref_toks))
                if ref_toks.count("<sys>") == pred_toks.count("<sys>"):
                    sys_exact += 1

    val_metrics = {
        "n_samples": n_samples,
        "val_ce_loss": round(sum_ce / max(1, n_samples), 6),
        "val_mtp_loss": round(sum_mtp / max(1, n_samples), 6),
        "val_token_f1": round(sum(all_f1) / max(1, len(all_f1)), 6),
        "val_ser": round(tot_edits / max(1, tot_ref_len), 6),
        "val_sys_count_exact": round(sys_exact / max(1, n_samples), 4),
    }

    # Benchmark B=1 greedy vs lossless MTP speculative decoding on first SPEC_BENCH_PAGES pages
    n_bench = min(SPEC_BENCH_PAGES, len(val_ds))
    greedy_toks = 0
    greedy_secs = 0.0
    spec_toks = 0
    spec_secs = 0.0
    tot_prop = 0
    tot_acc = 0
    tot_steps = 0
    parity_checked = 0
    parity_exact = 0

    for i in range(n_bench):
        item = val_ds[i]
        b1 = collate_uni_batch([item])
        img1 = _images_to_device_float(b1["images"], device, non_blocking=False)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad(), amp_ctx:
            g_seq = model.generate(img1, max_len=SPEC_MAX_LEN)[0].detach().cpu().tolist()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        greedy_secs += time.perf_counter() - t0
        greedy_toks += max(1, len(g_seq) - 1)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        s_seq, st = _speculative_generate_b1(model, img1, max_len=SPEC_MAX_LEN, use_bf16=use_bf16)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        spec_secs += time.perf_counter() - t1
        spec_toks += max(1, int(st["n_tokens"]))
        tot_prop += int(st["draft_proposed"])
        tot_acc += int(st["draft_accepted"])
        tot_steps += int(st["n_verify_steps"])

        if i < 4:
            parity_checked += 1
            if g_seq == s_seq:
                parity_exact += 1

    greedy_tps = float(greedy_toks) / max(1e-6, greedy_secs)
    spec_tps = float(spec_toks) / max(1e-6, spec_secs)
    spec_metrics = {
        "n_pages": n_bench,
        "spec_max_len": SPEC_MAX_LEN,
        "parity_exact_of_checked": f"{parity_exact}/{parity_checked}",
        "draft_acceptance_rate": round(float(tot_acc) / float(max(1, tot_prop)), 4),
        "mean_accepted_tokens_per_step": round(float(spec_toks) / float(max(1, tot_steps)), 4),
        "greedy_tok_per_sec": round(greedy_tps, 2),
        "spec_tok_per_sec": round(spec_tps, 2),
        "speedup_x": round(spec_tps / max(1e-6, greedy_tps), 4),
    }
    return val_metrics, spec_metrics


def _train_and_eval_arm(
    arm_name: str,
    mtp_k: int,
    mtp_loss_weight: float,
    train_pairs: List[Tuple[Path, List[Path], str]],
    val_pairs: List[Tuple[Path, List[Path], str]],
    device: "torch.device",
    *,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import train_uni_v2 as tv2
    from uni_omr import (
        GPUEmaShadow,
        UniFullPageDataset,
        _images_to_device_float,
        _infinite_loader,
        collate_uni_batch,
    )
    from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net

    arm_dir = VOL_ROOT / "ckpts" / EXP_ID / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached_res = arm_dir / "arm_result.json"
    if cached_res.is_file():
        print(f"[{arm_name}] Loading cached arm_result.json", flush=True)
        return json.loads(cached_res.read_text(encoding="utf-8"))

    _seed_everything(SEED)
    cfg = UniOMRv2Config.small_45m(grad_checkpoint=True, mtp_k=mtp_k)
    model = UniOMRv2Net(cfg).to(device)
    ps = model.param_summary()
    print(
        f"\n[{arm_name}] Starting | mtp_k={mtp_k} mtp_w={mtp_loss_weight:.2f} | "
        f"total={ps['total_params']:,} mtp={ps['mtp_params']:,} | steps={TRAIN_STEPS}",
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
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate_uni_batch,
    )
    train_iter = _infinite_loader(train_loader)

    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_arm0 = time.perf_counter()
    last_save_ts = time.time()
    save_interval_s = max(60.0, SAVE_EVERY_MINUTES * 60.0)

    history: List[Dict[str, Any]] = []
    buf_loss: List[float] = []
    buf_ce: List[float] = []
    buf_mtp: List[float] = []
    buf_ms: List[float] = []

    for step in range(1, TRAIN_STEPS + 1):
        t_step0 = time.perf_counter()
        cur_lr = _wsd_lr(step, TRAIN_STEPS, LR)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_ce = 0.0
        step_mtp = 0.0

        for _ in range(GRAD_ACCUM):
            batch = next(train_iter)
            images = _images_to_device_float(batch["images"], device, non_blocking=True)
            seq = batch["seq"].to(device, non_blocking=True)
            losses, _ = tv2.forward_and_loss_v2(
                model,
                images,
                seq,
                kind="fullpage",
                mtp_loss_weight=mtp_loss_weight,
                saliency_loss_weight=0.10,
                sys_y_loss_weight=0.25,
                sys_count_loss_weight=0.15,
                staff_seg_loss_weight=0.20,
                stroke_hog_weight=0.15,
                label_smoothing=0.05,
                use_bf16=use_bf16,
                device=device,
            )
            (losses["loss"] / float(GRAD_ACCUM)).backward()
            step_loss += float(losses["loss"].detach().item()) / float(GRAD_ACCUM)
            step_ce += float(losses["ce_loss"].detach().item()) / float(GRAD_ACCUM)
            step_mtp += float(losses["mtp_loss"].detach().item()) / float(GRAD_ACCUM)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        gpu_ema.update(model)

        buf_loss.append(step_loss)
        buf_ce.append(step_ce)
        buf_mtp.append(step_mtp)
        buf_ms.append((time.perf_counter() - t_step0) * 1000.0)

        if step % 150 == 0 or step == TRAIN_STEPS:
            rec = {
                "step": step,
                "lr": round(cur_lr, 8),
                "loss_mean": round(float(np.mean(buf_loss)), 5),
                "ce_mean": round(float(np.mean(buf_ce)), 5),
                "mtp_mean": round(float(np.mean(buf_mtp)), 5),
                "step_ms_mean": round(float(np.mean(buf_ms)), 1),
            }
            history.append(rec)
            print(
                f"  [{arm_name}] step {step:4d}/{TRAIN_STEPS} | lr={cur_lr:.2e} | "
                f"loss={rec['loss_mean']:.4f} ce={rec['ce_mean']:.4f} mtp={rec['mtp_mean']:.4f} | "
                f"{rec['step_ms_mean']:.0f} ms/step",
                flush=True,
            )
            buf_loss.clear()
            buf_ce.clear()
            buf_mtp.clear()
            buf_ms.clear()

        if (time.time() - last_save_ts) >= save_interval_s:
            tv2.atomic_save_v2_latest(arm_dir, model, cfg, step=step, gpu_ema=gpu_ema)
            last_save_ts = time.time()
            print(f"  [{arm_name}] 5-min atomic checkpoint saved at step {step}", flush=True)

    train_secs = time.perf_counter() - t_arm0
    peak_vram_gb = (
        round(float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 3), 3)
        if device.type == "cuda"
        else 0.0
    )
    tv2.atomic_save_v2_latest(arm_dir, model, cfg, step=TRAIN_STEPS, gpu_ema=gpu_ema)

    ema_sd = gpu_ema.cpu_state_dict(model)
    model.load_state_dict({k: v.to(device) for k, v in ema_sd.items()}, strict=True)
    model.eval()

    t_eval0 = time.perf_counter()
    val_eval, spec_eval = _eval_val_and_speculative(
        model, val_pairs, device, mtp_loss_weight=mtp_loss_weight, use_bf16=use_bf16
    )
    eval_secs = time.perf_counter() - t_eval0

    print(
        f"  [{arm_name}] DONE | val_F1={val_eval['val_token_f1']:.4f} CE={val_eval['val_ce_loss']:.4f} | "
        f"accept={spec_eval['draft_acceptance_rate']:.3f} tok/step={spec_eval['mean_accepted_tokens_per_step']:.2f} "
        f"spec_tps={spec_eval['spec_tok_per_sec']:.1f} ({spec_eval['speedup_x']:.2f}x) "
        f"parity={spec_eval['parity_exact_of_checked']}",
        flush=True,
    )

    result = {
        "arm": arm_name,
        "mtp_k": mtp_k,
        "mtp_loss_weight": mtp_loss_weight,
        "total_params": ps["total_params"],
        "mtp_params": ps["mtp_params"],
        "train_steps": TRAIN_STEPS,
        "train_secs": round(train_secs, 2),
        "eval_secs": round(eval_secs, 2),
        "peak_vram_gb": peak_vram_gb,
        "history": history,
        "proxy_val_eval": val_eval,
        "speculative_eval": spec_eval,
    }
    _atomic_write_json(cached_res, result)

    del model, optimizer, gpu_ema, train_ds, train_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    try:
        hf_token = _read_and_purge_hf_token()
        mvd_dir = _setup_environment()
        if str(mvd_dir) not in sys.path:
            sys.path.insert(0, str(mvd_dir))

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
        print(
            f"[{EXP_ID}] Running on {gpu_name} (cc={cc[0]}.{cc[1]}, bf16={use_bf16}, torch={torch.__version__})",
            flush=True,
        )

        train_abs, val_abs = _ensure_proxy_dataset(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)
        print(f"[{EXP_ID}] Loaded train={len(train_pairs)} val={len(val_pairs)}", flush=True)

        arm_results: Dict[str, Any] = {}
        for arm_name, mk, mw in ARMS:
            arm_results[arm_name] = _train_and_eval_arm(
                arm_name,
                mk,
                mw,
                train_pairs,
                val_pairs,
                device,
                use_bf16=use_bf16,
            )

        best_f1_arm = max(arm_results.values(), key=lambda r: float(r["proxy_val_eval"]["val_token_f1"]))
        fastest_arm = max(arm_results.values(), key=lambda r: float(r["speculative_eval"]["spec_tok_per_sec"]))
        summary = {
            "best_f1_arm": best_f1_arm["arm"],
            "best_val_token_f1": best_f1_arm["proxy_val_eval"]["val_token_f1"],
            "fastest_spec_arm": fastest_arm["arm"],
            "fastest_spec_tok_per_sec": fastest_arm["speculative_eval"]["spec_tok_per_sec"],
            "fastest_speedup_x": fastest_arm["speculative_eval"]["speedup_x"],
            "grid": {
                k: {
                    "mtp_k": v["mtp_k"],
                    "mtp_loss_weight": v["mtp_loss_weight"],
                    "val_token_f1": v["proxy_val_eval"]["val_token_f1"],
                    "val_ce_loss": v["proxy_val_eval"]["val_ce_loss"],
                    "draft_acceptance_rate": v["speculative_eval"]["draft_acceptance_rate"],
                    "mean_accepted_tokens_per_step": v["speculative_eval"]["mean_accepted_tokens_per_step"],
                    "spec_tok_per_sec": v["speculative_eval"]["spec_tok_per_sec"],
                    "speedup_x": v["speculative_eval"]["speedup_x"],
                    "parity": v["speculative_eval"]["parity_exact_of_checked"],
                }
                for k, v in arm_results.items()
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
                "steps_per_arm": TRAIN_STEPS,
                "seed": SEED,
                "lr": LR,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM,
                "fullpage_max_dim": FULLPAGE_MAX_DIM,
                "max_seq_len": MAX_SEQ_LEN,
                "eval_max_len": EVAL_MAX_LEN,
                "spec_bench_pages": SPEC_BENCH_PAGES,
                "spec_max_len": SPEC_MAX_LEN,
            },
            "arms": arm_results,
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
            commit_message=f"eval({EXP_ID}): upload MTP-k x mtp_loss_weight sweep results",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload MTP-k x mtp_loss_weight run ledger",
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
