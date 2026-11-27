#!/usr/bin/env python3
"""Colab H100 / Modal runner for Exp i15-40:
`pro200m` (d=768, ~120 MB INT4) vs `edge85m_ctrl` Capacity Tier Proxy.

Experiment ID: 2026-11-27T10-30-i15-40-pro200m-d768-capacity-proxy
Controlled 2-Arm Proxy (BF16, seed=42, 600 steps/arm, EMA=0.99, QAT-INT4 @ step>=420):
  - Arm 0 (`edge85m_ctrl`): edge85m + ax0_gl6 (d_model=512, ~78M infer params, ~53 MB INT4).
  - Arm 1 (`pro200m`):      pro200m + ax0_gl6 (d_model=768, n_heads=12, n_kv_heads=4,
                            n_axial_layers=0, n_global_layers=8, n_dec_layers=12,
                            ffn_mult=3.5 -> ~185-205M infer params, ~115-128 MB INT4).

Self-contained: does NOT edit tmp-train/mvd/uni_omr_v2.py.
Uploads evals/<EXP_ID>.json and runs/<EXP_ID>.json to gregoreesmaa/sheet-music-converter-omr.
"""

from __future__ import annotations

import argparse
import copy
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
from typing import Any, Dict, List, Optional, Tuple

EXP_ID = "2026-11-27T10-30-i15-40-pro200m-d768-capacity-proxy"
EXP_CODE = "i15-40"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

VOL_ROOT = Path("/vol") if Path("/vol").is_dir() else Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

SEED = 42
TRAIN_STEPS = 600
WARMUP_STEPS = 60
QAT_START_STEP = 420
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
GRAD_ACCUM = 4
LR_MAX = 3e-4
WEIGHT_DECAY = 0.04
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
EMA_DECAY = 0.99
INT4_GROUP_SIZE = 32
EDGE85M_BUDGET_MB = 55.0
PRO200M_BUDGET_MB = 135.0
SAVE_EVERY_MINUTES = 5.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


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


def _locate_or_unpack_mvd() -> Path:
    candidates = (
        Path.cwd() / "tmp-train" / "mvd",
        Path("/workspace/mvd"),
        Path("/content/mvd"),
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path("/content/workspace/sheet-music-converter/tmp-train/mvd"),
    )
    for cand in candidates:
        if (cand / "uni_omr_v2.py").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand

    for bundle in (Path("/content/bundle.tar.gz"), Path("/content/mvd_bundle.tar.gz")):
        if bundle.is_file():
            extract_dir = Path("/content/mvd_unpacked")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=extract_dir)
            for p in extract_dir.rglob("uni_omr_v2.py"):
                if str(p.parent) not in sys.path:
                    sys.path.insert(0, str(p.parent))
                return p.parent

    raise FileNotFoundError("Could not locate tmp-train/mvd/uni_omr_v2.py")


def _ensure_deps() -> None:
    needed = []
    for mod, pkg in (("huggingface_hub", "huggingface_hub>=0.26"), ("PIL", "pillow"), ("numpy", "numpy")):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])


def _ensure_proxy_dataset(hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_dir = VOL_ROOT / "datasets" / "iter10_5_proxy_2k"
    train_abs = ds_dir / "proxy_train_abs.jsonl"
    val_abs = ds_dir / "proxy_val_abs.jsonl"
    if train_abs.is_file() and val_abs.is_file():
        return train_abs, val_abs

    ds_dir.mkdir(parents=True, exist_ok=True)
    tar_p = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_PROXY_TAR,
        repo_type="dataset",
        token=hf_token,
        local_dir="/tmp/hf_dl",
    )
    with tarfile.open(tar_p, "r:*") as tf:
        tf.extractall(path=ds_dir)
    try:
        os.remove(tar_p)
    except OSError:
        pass
    for src_name, dst_path in (("proxy_train.jsonl", train_abs), ("proxy_val.jsonl", val_abs)):
        src_path = ds_dir / src_name
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
    return train_abs, val_abs


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _wsd_mult(step: int, total_steps: int, warmup_steps: int) -> float:
    if step <= warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    decay_steps = max(1, int(round(total_steps * WSD_DECAY_FRAC)))
    decay_start = total_steps - decay_steps
    if step <= decay_start:
        return 1.0
    prog = min(1.0, max(0.0, float(step - decay_start) / float(decay_steps)))
    return MIN_LR_RATIO + 0.5 * (1.0 - MIN_LR_RATIO) * (1.0 + math.cos(math.pi * prog))


def _load_manifest_pairs(manifest_path: Path) -> List[Tuple[Path, List[Path], str, float]]:
    pairs: List[Tuple[Path, List[Path], str, float]] = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)
        img_p = Path(rec["image"])
        beks = [Path(b) for b in rec.get("bekerns", [])]
        domain = str(rec.get("domain", "")).strip().lower()
        if img_p.is_file() and beks:
            is_dense_or_hw = (len(beks) >= 4) or ("hw" in domain) or ("hand" in domain) or ("debussy" in domain)
            weight = 2.0 if is_dense_or_hw else 1.0
            pairs.append((img_p, beks, domain, weight))
    return pairs


def _encode_bekerns_full(beks: List[Path]) -> List[int]:
    import uni_omr

    toks: List[str] = []
    for idx, bp in enumerate(beks):
        if idx > 0:
            toks.append(uni_omr.SYS_TOKEN)
        raw = bp.read_text(encoding="utf-8", errors="ignore").strip()
        if raw:
            toks.extend(raw.split())
    return uni_omr.encode_uni(toks)


def _pil_to_tensor(img_pil: Any, max_dim: int) -> "torch.Tensor":
    import numpy as np
    import torch
    from PIL import Image

    rgb = img_pil.convert("RGB")
    w, h = rgb.size
    scale = min(1.0, float(max_dim) / float(max(w, h)))
    nw = max(32, int(round(w * scale / 16.0)) * 16)
    nh = max(32, int(round(h * scale / 16.0)) * 16)
    if (nw, nh) != (w, h):
        rgb = rgb.resize((nw, nh), Image.BICUBIC)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return (t - 0.5) / 0.5


def _build_arm_config(arm_name: str, grad_checkpoint: bool = True) -> Any:
    import uni_omr_v2 as v2

    if arm_name == "pro200m":
        return v2.UniOMRv2Config.edge85m(
            d_model=768,
            n_heads=12,
            n_kv_heads=4,
            n_axial_layers=0,
            n_global_layers=8,
            n_dec_layers=12,
            ffn_mult=3.5,
            grad_checkpoint=grad_checkpoint,
        )
    return v2.UniOMRv2Config.edge85m(
        n_axial_layers=0,
        n_global_layers=6,
        grad_checkpoint=grad_checkpoint,
    )


def _pack_and_verify_hybrid_int4(
    model: Any,
    out_bin: Path,
    budget_mb: float,
    group_size: int = INT4_GROUP_SIZE,
) -> Dict[str, Any]:
    import torch
    import torch.nn as nn

    infer_model = copy.deepcopy(model).cpu().eval()
    infer_model.strip_aux_heads(strip_mtp=True)

    int4_roots = (
        infer_model.encoder.axial_layers,
        infer_model.encoder.patch_merger,
        infer_model.encoder.global_layers,
        infer_model.deepstack_fusions,
        infer_model.dec_layers,
        infer_model.lm_head,
    )
    int4_module_ids = set()
    for root in int4_roots:
        for mod in root.modules():
            if isinstance(mod, nn.Linear):
                int4_module_ids.add(id(mod))

    int4_param_names = set()
    for name, mod in infer_model.named_modules():
        if id(mod) in int4_module_ids:
            int4_param_names.add(f"{name}.weight" if name else "weight")

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    total_int4_bytes = 0
    total_scale_bytes = 0
    total_fp16_bytes = 0
    dequant_sd: Dict[str, torch.Tensor] = {}

    with open(out_bin, "wb") as f:
        for name, param in infer_model.named_parameters():
            t = param.detach().float()
            if name in int4_param_names:
                flat = t.reshape(-1)
                orig_n = flat.numel()
                pad = (group_size - (orig_n % group_size)) % group_size
                if pad:
                    flat = torch.nn.functional.pad(flat, (0, pad))
                g = flat.reshape(-1, group_size)
                amax = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                scale = (amax / 7.0).to(torch.float16)
                q = torch.clamp(torch.round(g / scale.float()), -8, 7).to(torch.int8)
                u = (q.to(torch.int16) + 8).to(torch.uint8).reshape(-1)
                if u.numel() % 2 == 1:
                    u = torch.nn.functional.pad(u, (0, 1))
                packed = (u[0::2] | (u[1::2] << 4)).numpy()
                scales_np = scale.squeeze(-1).numpy()
                f.write(packed.tobytes())
                f.write(scales_np.tobytes())
                total_int4_bytes += int(packed.nbytes)
                total_scale_bytes += int(scales_np.nbytes)
                dq = (q.float() * scale.float()).reshape(-1)[:orig_n].reshape(t.shape)
                dequant_sd[name] = dq.to(param.dtype)
            else:
                fp16_np = t.to(torch.float16).numpy()
                f.write(fp16_np.tobytes())
                total_fp16_bytes += int(fp16_np.nbytes)
                dequant_sd[name] = t.to(torch.float16).to(param.dtype)

    disk_bytes = out_bin.stat().st_size
    disk_mb = disk_bytes / 1_000_000.0
    infer_model.load_state_dict(dequant_sd, strict=False)
    return {
        "disk_bytes": int(disk_bytes),
        "packed_int4_mb": round(disk_mb, 3),
        "budget_mb": float(budget_mb),
        "passes_budget": bool(disk_mb <= budget_mb),
        "int4_weight_mb": round(total_int4_bytes / 1_000_000.0, 3),
        "int4_scales_mb": round(total_scale_bytes / 1_000_000.0, 3),
        "fp16_keep_mb": round(total_fp16_bytes / 1_000_000.0, 3),
        "dequant_model": infer_model,
    }


def _eval_val_ce(
    model: Any,
    val_pairs: List[Tuple[Path, List[Path], str, float]],
    device: "torch.device",
    use_bf16: bool,
) -> float:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import uni_omr

    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for img_p, beks, _, _ in val_pairs:
            ids = _encode_bekerns_full(beks)
            if len(ids) < 3:
                continue
            img = Image.open(img_p).convert("RGB")
            inp = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
            tgt = torch.tensor(ids[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                mem, _, _ = model.encode_image(inp, return_aux=False)
                logits, _, _ = model.decode_tokens(
                    tgt[:, :-1], mem, return_mtp=False, return_aux=False, apply_tok_dropout=False
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=uni_omr.PAD_ID
                )
            losses.append(float(loss.item()))
    return round(sum(losses) / max(1, len(losses)), 5)


def _train_arm(
    arm_name: str,
    *,
    train_pairs: List[Tuple[Path, List[Path], str, float]],
    val_pairs: List[Tuple[Path, List[Path], str, float]],
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, Any]:
    import torch
    from PIL import Image
    import uni_omr_v2 as v2

    _seed_everything(SEED)
    arm_dir = VOL_ROOT / "ckpts" / EXP_ID / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached = arm_dir / "arm_result.json"
    if cached.is_file():
        print(f"[{EXP_ID}] Reusing cached result for {arm_name}", flush=True)
        return json.loads(cached.read_text(encoding="utf-8"))

    cfg = _build_arm_config(arm_name, grad_checkpoint=True)
    model = v2.UniOMRv2Net(cfg).to(device)
    psummary = model.param_summary(int4_block_size=INT4_GROUP_SIZE)
    qat = v2.QATInt4ControllerV2(model, block_size=INT4_GROUP_SIZE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
    ema_state = copy.deepcopy(model.state_dict())

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    weights = [p[3] for p in train_pairs]
    rng = random.Random(SEED)
    t_arm0 = time.time()
    t_last_save = time.time()
    loss_hist: List[float] = []
    step_times_ms: List[float] = []

    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        if step == QAT_START_STEP:
            qat.enable()
            print(f"[{EXP_ID}][{arm_name}] Enabled QATInt4ControllerV2 at step {step}", flush=True)

        mult = _wsd_mult(step, TRAIN_STEPS, WARMUP_STEPS)
        for pg in opt.param_groups:
            pg["lr"] = LR_MAX * mult

        t_s0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            img_p, beks, _, _ = rng.choices(train_pairs, weights=weights, k=1)[0]
            ids = _encode_bekerns_full(beks)
            if len(ids) < 3:
                continue
            img = Image.open(img_p).convert("RGB")
            inp = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
            tgt = torch.tensor(ids[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(inp, tgt[:, :-1], return_aux=False, return_mtp=True)
                losses = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)
                loss = losses["loss"] / GRAD_ACCUM
            loss.backward()
            step_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times_ms.append((time.perf_counter() - t_s0) * 1000.0)

        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema_state[k].mul_(EMA_DECAY).add_(v, alpha=1.0 - EMA_DECAY)
                else:
                    ema_state[k].copy_(v)

        loss_hist.append(step_loss)
        if step % 100 == 0 or step == 1:
            mean_l = sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:]))
            mean_ms = sum(step_times_ms[-50:]) / max(1, len(step_times_ms[-50:]))
            print(
                f"[{EXP_ID}][{arm_name}] step={step}/{TRAIN_STEPS} loss={mean_l:.4f} "
                f"step_ms={mean_ms:.1f} elapsed={time.time()-t_arm0:.1f}s",
                flush=True,
            )

        if (time.time() - t_last_save) >= SAVE_EVERY_MINUTES * 60.0:
            tmp_ck = arm_dir / "latest.pt.tmp"
            torch.save({"step": step, "model_state_dict": model.state_dict(), "ema_state_dict": ema_state}, tmp_ck)
            os.replace(tmp_ck, arm_dir / "latest.pt")
            t_last_save = time.time()

    qat.disable()
    peak_vram_gb = (
        round(float(torch.cuda.max_memory_allocated()) / (1024.0 ** 3), 3) if device.type == "cuda" else 0.0
    )
    raw_val_ce = _eval_val_ce(model, val_pairs, device, use_bf16)

    model.load_state_dict(ema_state, strict=True)
    ema_val_ce = _eval_val_ce(model, val_pairs, device, use_bf16)

    budget_mb = PRO200M_BUDGET_MB if arm_name == "pro200m" else EDGE85M_BUDGET_MB
    packed_info = _pack_and_verify_hybrid_int4(
        model, arm_dir / f"{arm_name}_hybrid_int4.bin", budget_mb=budget_mb, group_size=INT4_GROUP_SIZE
    )
    dq_model = packed_info.pop("dequant_model").to(device)
    int4_val_ce = _eval_val_ce(dq_model, val_pairs, device, use_bf16)

    steady_ms = step_times_ms[min(10, len(step_times_ms) - 1) :]
    result = {
        "arm": arm_name,
        "param_summary": psummary,
        "packed_int4_verification": packed_info,
        "packed_int4_mb": packed_info["packed_int4_mb"],
        "step_ms": round(sum(steady_ms) / max(1, len(steady_ms)), 2),
        "vram_gb": peak_vram_gb,
        "train_loss_tail50": round(sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:])), 5),
        "raw_val_ce": raw_val_ce,
        "ema_val_ce": ema_val_ce,
        "int4_dequant_val_ce": int4_val_ce,
        "elapsed_secs": round(time.time() - t_arm0, 2),
    }
    _atomic_write_json(cached, result)
    del model, dq_model, opt, ema_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _run_cpu_smoke_test() -> int:
    _locate_or_unpack_mvd()
    import tempfile
    import torch
    import uni_omr
    import uni_omr_v2 as v2

    device = torch.device("cpu")
    print(f"[{EXP_ID}] Running 1-step tiny CPU forward+backward smoke test...", flush=True)
    for arm_name in ("edge85m_ctrl", "pro200m"):
        _seed_everything(SEED)
        cfg = _build_arm_config(arm_name, grad_checkpoint=False)
        model = v2.UniOMRv2Net(cfg).to(device)
        ps = model.param_summary(int4_block_size=INT4_GROUP_SIZE)
        model.train()
        img = torch.randn(1, 3, 128, 128, device=device)
        tgt = torch.randint(0, 32, (1, 16), device=device)
        tgt[:, 0] = uni_omr.BOS_ID
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        opt.zero_grad(set_to_none=True)
        out = model(img, tgt[:, :-1], return_aux=False, return_mtp=True)
        losses = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)
        losses["loss"].backward()
        opt.step()

        budget_mb = PRO200M_BUDGET_MB if arm_name == "pro200m" else EDGE85M_BUDGET_MB
        with tempfile.TemporaryDirectory() as td:
            pack = _pack_and_verify_hybrid_int4(
                model, Path(td) / f"{arm_name}.bin", budget_mb=budget_mb, group_size=INT4_GROUP_SIZE
            )
            dq = pack.pop("dequant_model")
            del dq
        print(
            f"[{EXP_ID}][smoke][{arm_name}] OK | total_params={ps['total_params']/1e6:.2f}M "
            f"| infer_params={ps['infer_params']/1e6:.2f}M | packed_int4_mb={pack['packed_int4_mb']:.2f} MB "
            f"| passes_budget={pack['passes_budget']} | loss={float(losses['loss'].item()):.4f}",
            flush=True,
        )
    return 0


def _restore_partial_arms_from_hf(hf_token: str) -> Dict[str, Any]:
    """Best-effort download of partial/completed `evals/<EXP_ID>.json` from HF for arm-level auto-resume."""
    try:
        from huggingface_hub import hf_hub_download

        dl_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_EVAL_PATH,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        payload = json.loads(Path(dl_p).read_text(encoding="utf-8"))
        arms = payload.get("arms")
        if isinstance(arms, dict) and arms:
            print(f"[{EXP_ID}][resume] Restored completed arms from HF {HF_EVAL_PATH}: {sorted(arms.keys())}", flush=True)
            return dict(arms)
    except Exception as exc:
        print(f"[{EXP_ID}][resume] No partial HF eval restored ({type(exc).__name__}: {str(exc)[:120]})", flush=True)
    return {}


def _sync_partial_arms_to_hf(
    *,
    api: Any,
    eval_local: Path,
    arms_done: Dict[str, Any],
    started_iso: str,
    t_start: float,
    gpu_name: str,
    use_bf16: bool,
) -> None:
    """Best-effort upload of partial arm results to `evals/<EXP_ID>.json` so completed arms survive preemption."""
    try:
        partial_report = {
            "exp_id": EXP_ID,
            "exp_code": EXP_CODE,
            "status": "partial",
            "started_utc": started_iso,
            "updated_utc": _utc_now(),
            "elapsed_secs": round(time.time() - t_start, 2),
            "platform": f"colab-or-modal-{gpu_name}",
            "bf16": use_bf16,
            "steps_per_arm": TRAIN_STEPS,
            "arms": arms_done,
        }
        _atomic_write_json(eval_local, partial_report)
        api.upload_file(
            path_or_fileobj=str(eval_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"ckpt({EXP_ID}): partial arm results ({','.join(sorted(arms_done.keys()))})",
        )
        print(f"[{EXP_ID}][hf] Synced partial {HF_EVAL_PATH} (arms={sorted(arms_done.keys())})", flush=True)
    except Exception as exc:
        print(f"[{EXP_ID}][hf] WARN: partial arm sync failed ({type(exc).__name__}: {str(exc)[:120]})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true", help="Run 1-step tiny CPU forward+backward smoke test")
    args = ap.parse_args()
    if args.smoke_test:
        return _run_cpu_smoke_test()

    t_start = time.time()
    started_iso = _utc_now()
    hf_token = _read_and_purge_hf_token()
    try:
        _locate_or_unpack_mvd()
        _ensure_deps()
        import torch
        from huggingface_hub import HfApi

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

        train_abs, val_abs = _ensure_proxy_dataset(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)
        print(f"[{EXP_ID}] Loaded {len(train_pairs)} train / {len(val_pairs)} val on {gpu_name}", flush=True)

        api = HfApi(token=hf_token)
        eval_local = VOL_ROOT / "evals" / f"{EXP_ID}.json"
        run_local = VOL_ROOT / "runs" / f"{EXP_ID}.json"
        resumed_arms = _restore_partial_arms_from_hf(hf_token)

        if "edge85m_ctrl" in resumed_arms:
            print(f"[{EXP_ID}][resume] Skipping edge85m_ctrl (already recorded in HF {HF_EVAL_PATH})", flush=True)
            arm_ctrl = resumed_arms["edge85m_ctrl"]
        else:
            arm_ctrl = _train_arm(
                "edge85m_ctrl", train_pairs=train_pairs, val_pairs=val_pairs, device=device, use_bf16=use_bf16
            )
            resumed_arms["edge85m_ctrl"] = arm_ctrl
            _sync_partial_arms_to_hf(
                api=api,
                eval_local=eval_local,
                arms_done=resumed_arms,
                started_iso=started_iso,
                t_start=t_start,
                gpu_name=gpu_name,
                use_bf16=use_bf16,
            )

        if "pro200m" in resumed_arms:
            print(f"[{EXP_ID}][resume] Skipping pro200m (already recorded in HF {HF_EVAL_PATH})", flush=True)
            arm_pro = resumed_arms["pro200m"]
        else:
            arm_pro = _train_arm(
                "pro200m", train_pairs=train_pairs, val_pairs=val_pairs, device=device, use_bf16=use_bf16
            )

        comparison = {
            "ema_val_ce_delta": round(arm_pro["ema_val_ce"] - arm_ctrl["ema_val_ce"], 5),
            "int4_val_ce_delta": round(arm_pro["int4_dequant_val_ce"] - arm_ctrl["int4_dequant_val_ce"], 5),
            "pro200m_int4_drop": round(arm_pro["int4_dequant_val_ce"] - arm_pro["ema_val_ce"], 5),
            "step_ms_ratio": round(arm_pro["step_ms"] / max(1e-6, arm_ctrl["step_ms"]), 3),
            "edge85m_packed_int4_mb": arm_ctrl["packed_int4_mb"],
            "pro200m_packed_int4_mb": arm_pro["packed_int4_mb"],
            "pro200m_passes_budget": arm_pro["packed_int4_verification"]["passes_budget"],
            "pro200m_vram_gb": arm_pro["vram_gb"],
        }
        report = {
            "exp_id": EXP_ID,
            "exp_code": EXP_CODE,
            "status": "completed",
            "started_utc": started_iso,
            "completed_utc": _utc_now(),
            "elapsed_secs": round(time.time() - t_start, 2),
            "platform": f"colab-or-modal-{gpu_name}",
            "bf16": use_bf16,
            "steps_per_arm": TRAIN_STEPS,
            "arms": {"edge85m_ctrl": arm_ctrl, "pro200m": arm_pro},
            "comparison_pro200m_minus_edge85m": comparison,
        }
        _atomic_write_json(eval_local, report)
        _atomic_write_json(
            run_local,
            {
                "exp_id": EXP_ID,
                "status": "completed",
                "started_utc": started_iso,
                "completed_utc": report["completed_utc"],
                "elapsed_secs": report["elapsed_secs"],
                "eval_path": HF_EVAL_PATH,
                "comparison": comparison,
            },
        )
        api.upload_file(
            path_or_fileobj=str(eval_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"eval({EXP_ID}): upload i15-40 pro200m d=768 capacity proxy",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload i15-40 run ledger",
        )
        print(f"[{EXP_ID}] UPLOADED {HF_EVAL_PATH} & {HF_RUN_PATH} TO HF!", flush=True)
    finally:
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
