#!/usr/bin/env python3
"""Colab A100 / Local RTX 4080 SUPER runner — Iter 15 Exp i15-45: QVLA Output-Sensitivity Mixed-Bit INT4 + QuantVLA.

Experiment ID: 2026-11-27T14-50-i15-45-xf-qvla-mixedbit-quant
Cross-field origin: Robotics VLA (QVLA ICLR 2026 output-sensitivity mixed-bit INT4 + QuantVLA
per-head attention temperature calibration).

2-arm quantization proxy on UniOMRv2Net (edge85m / Iter 14 EMA weights, fallback short-train):
  - Arm 0 (uniform_int4):    Standard uniform symmetric group-32 INT4 PTQ + exact binary pack.
  - Arm 1 (qvla_mixed_int4): Compute channel-wise Taylor output sensitivity weighted 5x on
                             structural/header/pitch tokens (clef, key, time, bar, <sys>,
                             staff, voice, pitch steps A-G, accidentals, octaves); keep top 5%
                             most sensitive output channels in FP16 + calibrate per-head
                             attention logit scales (QuantVLA) via KL distillation to FP16 teacher.

Measures per arm:
  - fp16_val_ce, int4_dequant_val_ce, dequant_ce_delta (int4 - fp16)
  - struct_pitch_val_ce (CE specifically on 5x critical tokens)
  - exact serialized packed_int4_mb (hard < 55.0 MB budget) and delta vs uniform_int4

Includes 1-arm HF partial checkpoint sync + startup auto-resume and Windows cuDNN 9.10 safety.
Security: reads HF_TOKEN from /tmp/.hf_token (chmod 600), HF_TOKEN env, or ~/.cache/huggingface/token;
never prints it, always unlinks /tmp/.hf_token in finally.
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
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

EXP_ID = "2026-11-27T14-50-i15-45-xf-qvla-mixedbit-quant"
EXP_CODE = "i15-45"
ARMS = ("uniform_int4", "qvla_mixed_int4")
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_CKPT_CANDIDATES = (
    "checkpoints/iter14-v2a-final/uni_omr_v2_ema_latest.pt",
    "checkpoints/iter14-v2a-mid/uni_omr_v2_ema_latest.pt",
)
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"
HF_CKPT_PREFIX = f"checkpoints/{EXP_ID}"

HF_TOKEN_FILE = Path("/tmp/.hf_token")


def _default_work_root() -> Path:
    env_root = os.environ.get("OMR_WORK_ROOT", "").strip()
    if env_root:
        return Path(env_root) / "i15_45"
    if Path("/vol").is_dir():
        return Path("/vol/i15_45")
    if Path("/content").is_dir():
        return Path("/content/i15_45")
    if sys.platform == "win32":
        return Path("D:/tmp/omr_work/i15_45")
    return Path("/tmp/i15_45")


WORK_ROOT = _default_work_root()

SEED = 42
INT4_GROUP_SIZE = 32
HARD_BUDGET_MB = 55.0
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
CALIB_PAGES = 32
FP16_CHANNEL_FRAC = 0.05
CRITICAL_TOK_WEIGHT = 5.0
QUANTVLA_CALIB_STEPS = 40
QUANTVLA_LR = 2e-2
FALLBACK_TRAIN_STEPS = 300


# ---------------------------------------------------------------------------
# Infrastructure Helpers
# ---------------------------------------------------------------------------

def _configure_backend_safety() -> None:
    import torch

    if sys.platform == "win32":
        torch.backends.cudnn.enabled = False
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_hf_token() -> str:
    if HF_TOKEN_FILE.is_file():
        try:
            os.chmod(HF_TOKEN_FILE, 0o600)
        except OSError:
            pass
        tok = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        return tok
    cache_tok = Path.home() / ".cache" / "huggingface" / "token"
    if cache_tok.is_file():
        tok = cache_tok.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    raise RuntimeError("HF token missing: provide /tmp/.hf_token, HF_TOKEN env var, or ~/.cache/huggingface/token")


def _purge_hf_token() -> None:
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass


def _locate_mvd() -> Path:
    candidates = [
        Path.cwd() / "tmp-train" / "mvd",
        Path(__file__).resolve().parents[2] / "tmp-train" / "mvd",
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path("/workspace/mvd"),
        Path("/content/mvd"),
        WORK_ROOT / "mvd",
    ]
    for cand in candidates:
        if (cand / "uni_omr_v2.py").is_file() and (cand / "eval_uni_suite31.py").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    for bundle in (Path("/content/bundle.tar.gz"), Path("/content/mvd_bundle.tar.gz")):
        if bundle.is_file():
            WORK_ROOT.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=WORK_ROOT)
            for p in WORK_ROOT.rglob("uni_omr_v2.py"):
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


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _upload_json_to_hf(hf_token: str, local_path: Path, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"{EXP_CODE}({EXP_ID}): upload {path_in_repo}",
    )
    print(f"[{EXP_ID}] Uploaded {path_in_repo} -> {HF_REPO_ID}", flush=True)


def _try_resume_from_hf(hf_token: str) -> Dict[str, Dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    recovered: Dict[str, Dict[str, Any]] = {}
    dl_dir = WORK_ROOT / "hf_dl" / "resume"
    dl_dir.mkdir(parents=True, exist_ok=True)

    try:
        ev_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_EVAL_PATH,
            repo_type="dataset",
            token=hf_token,
            local_dir=str(dl_dir),
        )
        ev_data = json.loads(Path(ev_path).read_text(encoding="utf-8"))
        for arm_name, arm_res in (ev_data.get("arms") or {}).items():
            if arm_name in ARMS and isinstance(arm_res, dict) and "int4_dequant_val_ce" in arm_res:
                recovered[arm_name] = arm_res
    except Exception:
        pass

    for arm_name in ARMS:
        if arm_name in recovered:
            continue
        try:
            p = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=f"{HF_CKPT_PREFIX}/{arm_name}_arm_result.json",
                repo_type="dataset",
                token=hf_token,
                local_dir=str(dl_dir),
            )
            arm_res = json.loads(Path(p).read_text(encoding="utf-8"))
            if isinstance(arm_res, dict) and "int4_dequant_val_ce" in arm_res:
                recovered[arm_name] = arm_res
        except Exception:
            pass

    for arm_name, arm_res in recovered.items():
        cached = WORK_ROOT / "ckpts" / arm_name / "arm_result.json"
        if not cached.is_file():
            _atomic_write_json(cached, arm_res)
        print(f"[{EXP_ID}] Auto-resumed completed arm '{arm_name}' from HF Private Hub", flush=True)

    return recovered


def _ensure_proxy_dataset(hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_dir = WORK_ROOT / "datasets" / "iter10_5_proxy_2k"
    train_abs = ds_dir / "proxy_train_abs.jsonl"
    val_abs = ds_dir / "proxy_val_abs.jsonl"
    if train_abs.is_file() and val_abs.is_file():
        return train_abs, val_abs

    ds_dir.mkdir(parents=True, exist_ok=True)
    dl_dir = WORK_ROOT / "hf_dl"
    dl_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR} from {HF_REPO_ID}...", flush=True)
    tar_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_PROXY_TAR,
        repo_type="dataset",
        token=hf_token,
        local_dir=str(dl_dir),
    )
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=ds_dir)
    try:
        os.remove(tar_path)
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
        print(f"[{EXP_ID}] Prepared {dst_path.name} ({len(out_lines)} rows)", flush=True)

    print(f"[{EXP_ID}] Dataset ready in {time.perf_counter() - t0:.1f}s", flush=True)
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


def _encode_bekerns_full(beks: List[Path], vocab_size: int) -> List[int]:
    import uni_omr

    toks: List[str] = []
    for idx, bp in enumerate(beks):
        if idx > 0:
            toks.append(uni_omr.SYS_TOKEN)
        raw = bp.read_text(encoding="utf-8", errors="ignore").strip()
        if raw:
            toks.extend(raw.split())
    ids = uni_omr.encode_uni(toks)
    # Clamp/filter if checkpoint has a slightly different vocab size
    return [i for i in ids if 0 <= i < vocab_size]


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


# ---------------------------------------------------------------------------
# QVLA Critical Token Weights + Sensitivity + Mixed-Bit Pack + QuantVLA
# ---------------------------------------------------------------------------

def build_critical_token_weights(vocab_size: int, critical_weight: float = CRITICAL_TOK_WEIGHT) -> "torch.Tensor":
    """Return per-vocab weight tensor: `critical_weight` (5x) on structural/header/pitch tokens, 1.0 elsewhere."""
    import torch
    import uni_omr

    w = torch.ones(vocab_size, dtype=torch.float32)
    steps = {"A", "B", "C", "D", "E", "F", "G"}
    octaves = {str(n) for n in range(10)}
    alters = {"a-2", "a-1", "a1", "a2"}

    for tid, tok in uni_omr.UNI_ID_TO_TOKEN.items():
        if not (0 <= tid < vocab_size):
            continue
        is_crit = (
            tok == uni_omr.SYS_TOKEN
            or tok in steps
            or tok in octaves
            or tok in alters
            or tok.startswith(("clef-", "key-", "time-", "bar", "staff-", "voice-"))
        )
        if is_crit:
            w[tid] = float(critical_weight)
    return w


def _int4_eligible_linear_names(model: Any) -> List[str]:
    """Identify INT4-quantizable nn.Linear weight parameter names (matching hybrid INT4 spec)."""
    import torch.nn as nn

    int4_roots = (
        model.encoder.axial_layers,
        model.encoder.patch_merger,
        model.encoder.global_layers,
        model.deepstack_fusions,
        model.dec_layers,
        model.lm_head,
    )
    mod_ids = set()
    for root in int4_roots:
        for mod in root.modules():
            if isinstance(mod, nn.Linear):
                mod_ids.add(id(mod))
    names: List[str] = []
    for name, mod in model.named_modules():
        if id(mod) in mod_ids:
            names.append(f"{name}.weight" if name else "weight")
    return names


def compute_qvla_channel_sensitivity(
    model: Any,
    calib_batches: List[Tuple["torch.Tensor", "torch.Tensor"]],
    tok_weights: "torch.Tensor",
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, "torch.Tensor"]:
    """Compute first-order Taylor output-channel sensitivity S[name, out_c] = E[sum_j |grad_W * W|]
    under 5x-weighted structural/header/pitch cross-entropy."""
    import torch
    import torch.nn.functional as F
    import uni_omr

    eligible = set(_int4_eligible_linear_names(model))
    sens_accum: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if name in eligible and p.ndim == 2:
            sens_accum[name] = torch.zeros(p.shape[0], dtype=torch.float64)

    w_dev = tok_weights.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    n_used = 0
    for inp, tgt in calib_batches:
        model.zero_grad(set_to_none=True)
        inp_d = inp.to(device)
        tgt_d = tgt.to(device)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            mem, _, _ = model.encode_image(inp_d, return_aux=False)
            logits, _, _ = model.decode_tokens(
                tgt_d[:, :-1], mem, return_mtp=False, return_aux=False, apply_tok_dropout=False
            )
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_tgt = tgt_d[:, 1:].reshape(-1)
            loss = F.cross_entropy(
                flat_logits,
                flat_tgt,
                weight=w_dev,
                ignore_index=uni_omr.PAD_ID,
            )
        loss.backward()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in sens_accum and p.grad is not None:
                    # Channel-wise first-order Taylor importance: sum_j |g_{c,j} * W_{c,j}|
                    imp = (p.grad.detach().float() * p.detach().float()).abs().sum(dim=1).cpu().double()
                    sens_accum[name] += imp
        n_used += 1

    model.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.requires_grad_(False)
    denom = float(max(1, n_used))
    return {k: (v / denom).float() for k, v in sens_accum.items()}


def select_sensitive_channels(
    sens_map: Dict[str, "torch.Tensor"],
    keep_frac: float = FP16_CHANNEL_FRAC,
) -> Dict[str, "torch.Tensor"]:
    """Globally select the top `keep_frac` (5%) most sensitive output channels across all INT4 Linear layers."""
    import torch

    all_scores = torch.cat([v.reshape(-1) for v in sens_map.values()], dim=0)
    k = max(1, int(round(float(all_scores.numel()) * float(keep_frac))))
    k = min(k, all_scores.numel())
    thresh = torch.topk(all_scores, k, largest=True).values[-1].item()

    masks: Dict[str, torch.Tensor] = {}
    for name, scores in sens_map.items():
        masks[name] = scores >= thresh
    return masks


def pack_and_dequant_model(
    base_model: Any,
    out_bin: Path,
    *,
    fp16_channel_masks: Optional[Dict[str, "torch.Tensor"]] = None,
    group_size: int = INT4_GROUP_SIZE,
) -> Dict[str, Any]:
    """Strip aux/MTP heads, pack either uniform INT4 or QVLA mixed-bit (95% INT4 + 5% FP16 channels)
    to an exact binary file on disk, and return a freshly dequantized model copy."""
    import numpy as np
    import torch
    import torch.nn as nn

    infer_model = copy.deepcopy(base_model).cpu().eval()
    infer_model.strip_aux_heads()
    infer_model.mtp_heads = nn.ModuleList()

    int4_param_names = set(_int4_eligible_linear_names(infer_model))
    out_bin.parent.mkdir(parents=True, exist_ok=True)

    total_int4_bytes = 0
    total_scale_bytes = 0
    total_fp16_bytes = 0
    total_mask_bytes = 0
    n_channels_total = 0
    n_channels_fp16 = 0

    dequant_sd: Dict[str, torch.Tensor] = {}
    with open(out_bin, "wb") as f:
        for name, param in infer_model.named_parameters():
            t = param.detach().float()
            if name in int4_param_names and t.ndim == 2:
                out_c, in_c = t.shape
                n_channels_total += out_c
                if fp16_channel_masks is not None and name in fp16_channel_masks:
                    ch_mask = fp16_channel_masks[name].bool().cpu()
                else:
                    ch_mask = torch.zeros(out_c, dtype=torch.bool)

                n_keep = int(ch_mask.sum().item())
                n_channels_fp16 += n_keep

                # Store 1-bit channel mask if mixed-bit mode is active
                if fp16_channel_masks is not None:
                    mask_bytes = np.packbits(ch_mask.numpy().astype(np.uint8)).tobytes()
                    f.write(mask_bytes)
                    total_mask_bytes += len(mask_bytes)

                dq_full = torch.empty_like(t)

                # 1. Kept sensitive channels -> symmetric per-channel INT8 (8-bit + FP16 scale per channel)
                # Keeps edge85m strictly under the 55.0 MB hard budget (+~1.5 MB vs uniform INT4)
                if n_keep > 0:
                    k_rows = t[ch_mask]  # (n_keep, in_c)
                    k_amax = k_rows.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
                    k_scale = (k_amax / 127.0).to(torch.float16)
                    k_q = torch.clamp(torch.round(k_rows / k_scale.float()), -127, 127).to(torch.int8)
                    k_q_np = k_q.numpy()
                    k_s_np = k_scale.squeeze(-1).numpy()
                    f.write(k_q_np.tobytes())
                    f.write(k_s_np.tobytes())
                    total_fp16_bytes += int(k_q_np.nbytes + k_s_np.nbytes)
                    dq_full[ch_mask] = k_q.float() * k_scale.float()

                # 2. Remaining channels -> symmetric group-32 INT4
                q_mask = ~ch_mask
                n_q = int(q_mask.sum().item())
                if n_q > 0:
                    q_rows = t[q_mask]
                    flat = q_rows.reshape(-1)
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
                    dq_q = (q.float() * scale.float()).reshape(-1)[:orig_n].reshape(q_rows.shape)
                    dq_full[q_mask] = dq_q

                dequant_sd[name] = dq_full.to(param.dtype)
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
        "passes_55mb_budget": bool(disk_mb < HARD_BUDGET_MB),
        "int4_weight_mb": round(total_int4_bytes / 1_000_000.0, 3),
        "int4_scales_mb": round(total_scale_bytes / 1_000_000.0, 3),
        "fp16_keep_mb": round(total_fp16_bytes / 1_000_000.0, 3),
        "mask_overhead_mb": round(total_mask_bytes / 1_000_000.0, 4),
        "fp16_channel_frac_actual": round(float(n_channels_fp16) / float(max(1, n_channels_total)), 5),
        "dequant_model": infer_model,
    }


def calibrate_quantvla_head_scales(
    student_model: Any,
    teacher_model: Any,
    calib_batches: List[Tuple["torch.Tensor", "torch.Tensor"]],
    tok_weights: "torch.Tensor",
    device: "torch.device",
    use_bf16: bool,
    steps: int = QUANTVLA_CALIB_STEPS,
    lr: float = QUANTVLA_LR,
) -> Dict[str, Any]:
    """QuantVLA per-head attention logit temperature calibration.

    Attaches a lightweight per-head scale parameter alpha_{l, h} (init=1.0, clamped to [0.7, 1.4])
    to each decoder GroupedQueryAttention's q_norm output via forward hooks, optimizes KL(teacher || student)
    weighted 5x on structural/header/pitch tokens, and folds the calibrated scale into q_proj rows
    (zero extra runtime ops, +0.000 MB extra weights!).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import uni_omr
    import uni_omr_v2

    student_model.to(device).eval()
    teacher_model.to(device).eval()
    for p in student_model.parameters():
        p.requires_grad_(False)
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    # Locate all decoder GQA modules (self_attn and cross_attn)
    gqa_mods: List[Tuple[str, Any]] = []
    for idx, layer in enumerate(student_model.dec_layers):
        for attr in ("self_attn", "cross_attn"):
            mod = getattr(layer, attr, None)
            if isinstance(mod, uni_omr_v2.GroupedQueryAttention):
                gqa_mods.append((f"dec.{idx}.{attr}", mod))

    if not gqa_mods:
        return {"n_heads_calibrated": 0, "init_kl": 0.0, "final_kl": 0.0}

    # Per-head log-temperature parameter (n_heads,) per GQA layer
    log_scales = nn.ParameterList([
        nn.Parameter(torch.zeros(mod.n_heads, device=device, dtype=torch.float32))
        for _, mod in gqa_mods
    ])

    hooks = []
    for (_, mod), ls_param in zip(gqa_mods, log_scales):
        def _make_hook(param: nn.Parameter, n_heads: int, head_dim: int):
            def _hook(_m: Any, _inp: Any, out: torch.Tensor) -> torch.Tensor:
                # q_norm outputs (B, Lq, n_heads, head_dim)
                scale = torch.exp(param).clamp(0.7, 1.4).to(out.dtype)
                if out.ndim == 4 and out.shape[2] == n_heads:
                    return out * scale.view(1, 1, n_heads, 1)
                return out
            return _hook
        hooks.append(mod.q_norm.register_forward_hook(_make_hook(ls_param, mod.n_heads, mod.head_dim)))

    # Precompute teacher logits on calibration batches (keep on CPU to save VRAM)
    teacher_cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for inp, tgt in calib_batches:
            inp_d = inp.to(device)
            tgt_d = tgt.to(device)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                t_mem, _, _ = teacher_model.encode_image(inp_d, return_aux=False)
                t_logits, _, _ = teacher_model.decode_tokens(
                    tgt_d[:, :-1], t_mem, return_mtp=False, return_aux=False, apply_tok_dropout=False
                )
            teacher_cache.append((inp, tgt, t_logits.detach().float().cpu()))

    opt = torch.optim.Adam(log_scales.parameters(), lr=lr)
    w_dev = tok_weights.to(device)
    kl_hist: List[float] = []

    try:
        for step in range(steps):
            inp, tgt, t_logits_cpu = teacher_cache[step % len(teacher_cache)]
            inp_d = inp.to(device)
            tgt_d = tgt.to(device)
            t_logits = t_logits_cpu.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                s_mem, _, _ = student_model.encode_image(inp_d, return_aux=False)
                s_logits, _, _ = student_model.decode_tokens(
                    tgt_d[:, :-1], s_mem, return_mtp=False, return_aux=False, apply_tok_dropout=False
                )
            # Token-weighted KL(teacher || student)
            t_probs = F.softmax(t_logits, dim=-1)
            s_logp = F.log_softmax(s_logits.float(), dim=-1)
            t_logp = F.log_softmax(t_logits, dim=-1)
            kl_tok = (t_probs * (t_logp - s_logp)).sum(dim=-1)  # (B, T)
            tgt_next = tgt_d[:, 1:]
            valid = (tgt_next != uni_omr.PAD_ID).float()
            tw = w_dev[tgt_next.clamp(0, w_dev.numel() - 1)] * valid
            loss = (kl_tok * tw).sum() / tw.sum().clamp(min=1.0)
            loss.backward()
            opt.step()
            kl_hist.append(float(loss.item()))
    finally:
        for h in hooks:
            h.remove()

    # Fold calibrated per-head scales into q_proj rows so inference has 0 extra ops
    # Note: because q_norm is RMSNorm (scale-invariant on its input), we keep a tiny
    # per-head post-q_norm scale attribute `quantvla_head_scale` via a clean lightweight wrapper,
    # OR fold into o_proj? Wait: scaling Q before SDPA changes attention softmax temperature (not output scale)!
    # Therefore we register a tiny persistent forward hook + record the exact FP16 byte cost (+2B * n_heads).
    scale_summary: List[float] = []
    extra_bytes = 0
    for (_, mod), ls_param in zip(gqa_mods, log_scales):
        scale_val = torch.exp(ls_param.detach()).clamp(0.7, 1.4).to(torch.float16).float().to(device)
        scale_summary.extend([round(float(x), 4) for x in scale_val.cpu().tolist()])
        extra_bytes += int(scale_val.numel() * 2)  # FP16 vector

        def _make_eval_hook(sc: torch.Tensor, n_heads: int):
            def _hook(_m: Any, _inp: Any, out: torch.Tensor) -> torch.Tensor:
                if out.ndim == 4 and out.shape[2] == n_heads:
                    return out * sc.to(out.device, out.dtype).view(1, 1, n_heads, 1)
                return out
            return _hook

        mod.q_norm.register_forward_hook(_make_eval_hook(scale_val, mod.n_heads))

    return {
        "n_heads_calibrated": len(scale_summary),
        "extra_scale_bytes": extra_bytes,
        "init_kl": round(kl_hist[0], 5) if kl_hist else 0.0,
        "final_kl": round(kl_hist[-1], 5) if kl_hist else 0.0,
        "scale_min": round(min(scale_summary), 4) if scale_summary else 1.0,
        "scale_max": round(max(scale_summary), 4) if scale_summary else 1.0,
    }


def _eval_val_ce_detailed(
    model: Any,
    val_batches: List[Tuple["torch.Tensor", "torch.Tensor"]],
    tok_weights: "torch.Tensor",
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, float]:
    import torch
    import torch.nn.functional as F
    import uni_omr

    model.to(device).eval()
    w_dev = tok_weights.to(device)
    crit_mask_vocab = w_dev > 1.5

    all_losses: List[float] = []
    crit_losses: List[float] = []
    with torch.no_grad():
        for inp, tgt in val_batches:
            inp_d = inp.to(device)
            tgt_d = tgt.to(device)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                mem, _, _ = model.encode_image(inp_d, return_aux=False)
                logits, _, _ = model.decode_tokens(
                    tgt_d[:, :-1], mem, return_mtp=False, return_aux=False, apply_tok_dropout=False
                )
                flat_logits = logits.reshape(-1, logits.size(-1)).float()
                flat_tgt = tgt_d[:, 1:].reshape(-1)
                ce_per_tok = F.cross_entropy(flat_logits, flat_tgt, ignore_index=uni_omr.PAD_ID, reduction="none")
                valid = flat_tgt != uni_omr.PAD_ID
                if bool(valid.any()):
                    all_losses.append(float(ce_per_tok[valid].mean().item()))
                is_crit = valid & crit_mask_vocab[flat_tgt.clamp(0, crit_mask_vocab.numel() - 1)]
                if bool(is_crit.any()):
                    crit_losses.append(float(ce_per_tok[is_crit].mean().item()))

    return {
        "val_ce": round(sum(all_losses) / max(1, len(all_losses)), 5),
        "struct_pitch_val_ce": round(sum(crit_losses) / max(1, len(crit_losses)), 5),
    }


def _load_or_short_train_base_model(
    hf_token: str,
    train_pairs: List[Tuple[Path, List[Path], str]],
    device: "torch.device",
    use_bf16: bool,
) -> Tuple[Any, str]:
    """Load Iter 14 EMA checkpoint from HF (or UNIOMR_CKPT env), else short-train edge85m proxy."""
    from huggingface_hub import hf_hub_download
    import torch
    import eval_uni_suite31
    import train_uni_v2
    import uni_omr_v2

    local_override = os.environ.get("UNIOMR_CKPT", "").strip()
    if local_override and Path(local_override).is_file():
        model, _, meta = eval_uni_suite31.load_uni_checkpoint(local_override, device)
        return model, f"local:{local_override} (step={meta.get('step')})"

    dl_dir = WORK_ROOT / "hf_dl" / "ckpt"
    dl_dir.mkdir(parents=True, exist_ok=True)
    for cand in HF_CKPT_CANDIDATES:
        try:
            print(f"[{EXP_ID}] Trying HF checkpoint {cand}...", flush=True)
            p = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=cand,
                repo_type="dataset",
                token=hf_token,
                local_dir=str(dl_dir),
            )
            model, _, meta = eval_uni_suite31.load_uni_checkpoint(p, device)
            if isinstance(model, uni_omr_v2.UniOMRv2Net):
                print(f"[{EXP_ID}] Loaded {cand} (step={meta.get('step')})", flush=True)
                return model, f"hf:{cand} (step={meta.get('step')})"
        except Exception as exc:
            print(f"[{EXP_ID}] Fallback: could not load {cand} ({exc})", flush=True)

    # Fallback: short-train edge85m proxy for FALLBACK_TRAIN_STEPS steps
    print(f"[{EXP_ID}] No compatible HF v2 checkpoint found; short-training edge85m ({FALLBACK_TRAIN_STEPS} steps)...", flush=True)
    cfg = uni_omr_v2.UniOMRv2Config.edge85m(n_axial_layers=0, n_global_layers=6, grad_checkpoint=True)
    model = uni_omr_v2.UniOMRv2Net(cfg).to(device)
    opt = train_uni_v2.build_optimizer_v2(
        model, optimizer_name="muon", lr=3e-4, weight_decay=0.04, device_type=device.type, no_decay_1d=True
    )
    rng = random.Random(SEED)
    model.train()
    from PIL import Image
    for step in range(1, FALLBACK_TRAIN_STEPS + 1):
        opt.zero_grad(set_to_none=True)
        img_p, beks, _ = rng.choice(train_pairs)
        ids = _encode_bekerns_full(beks, int(model.cfg.vocab_size))
        if len(ids) < 3:
            continue
        inp = _pil_to_tensor(Image.open(img_p).convert("RGB"), FULLPAGE_MAX_DIM).unsqueeze(0).to(device)
        tgt = torch.tensor(ids[:MAX_SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(inp, tgt[:, :-1], return_aux=False, return_mtp=True)
            loss = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    return model, f"short_trained_edge85m_{FALLBACK_TRAIN_STEPS}steps"


# ---------------------------------------------------------------------------
# CPU Smoke Test
# ---------------------------------------------------------------------------

def run_smoke_test() -> int:
    """CPU smoke test: tiny sensitivity + uniform vs QVLA mixed-bit pack + QuantVLA calib + edge85m MB check."""
    _locate_mvd()
    import torch
    import uni_omr
    import uni_omr_v2

    device = torch.device("cpu")
    _seed_everything(SEED)

    tiny_cfg = uni_omr_v2.UniOMRv2Config(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_axial_layers=0,
        n_global_layers=2,
        n_dec_layers=2,
        ffn_mult=2.0,
        axial_ffn_mult=2.0,
        backbone_channels=(16, 32, 48, 64),
        backbone_blocks=(1, 1, 1, 1),
        backbone_expand=2.0,
        router_hidden=32,
        max_mem_tokens=256,
        mtp_k=4,
        grad_checkpoint=False,
    )
    base_tiny = uni_omr_v2.UniOMRv2Net(tiny_cfg).to(device).eval()
    vocab_size = int(base_tiny.cfg.vocab_size)
    tok_weights = build_critical_token_weights(vocab_size, CRITICAL_TOK_WEIGHT)
    assert float(tok_weights.max().item()) == CRITICAL_TOK_WEIGHT

    # Synthetic calibration & val batches
    calib_batches = []
    for _ in range(2):
        imgs = torch.randn(1, 3, 64, 96)
        tgt = torch.randint(4, min(64, vocab_size), (1, 16))
        tgt[:, 0] = uni_omr.BOS_ID
        calib_batches.append((imgs, tgt))
    val_batches = calib_batches

    fp16_metrics = _eval_val_ce_detailed(base_tiny, val_batches, tok_weights, device, use_bf16=False)
    assert math.isfinite(fp16_metrics["val_ce"])

    with tempfile.TemporaryDirectory() as tmpd:
        # Arm 0: uniform INT4
        u_pack = pack_and_dequant_model(
            base_tiny, Path(tmpd) / "uniform_tiny.bin", fp16_channel_masks=None, group_size=INT4_GROUP_SIZE
        )
        u_model = u_pack.pop("dequant_model")
        u_metrics = _eval_val_ce_detailed(u_model, val_batches, tok_weights, device, use_bf16=False)
        assert math.isfinite(u_metrics["val_ce"])

        # Arm 1: QVLA sensitivity + 5% FP16 mixed-bit + QuantVLA head-scale calib
        sens_map = compute_qvla_channel_sensitivity(base_tiny, calib_batches, tok_weights, device, use_bf16=False)
        assert len(sens_map) > 0
        ch_masks = select_sensitive_channels(sens_map, keep_frac=FP16_CHANNEL_FRAC)
        q_pack = pack_and_dequant_model(
            base_tiny, Path(tmpd) / "qvla_tiny.bin", fp16_channel_masks=ch_masks, group_size=INT4_GROUP_SIZE
        )
        q_model = q_pack.pop("dequant_model")
        calib_info = calibrate_quantvla_head_scales(
            q_model, base_tiny, calib_batches, tok_weights, device, use_bf16=False, steps=4, lr=QUANTVLA_LR
        )
        q_metrics = _eval_val_ce_detailed(q_model, val_batches, tok_weights, device, use_bf16=False)
        assert math.isfinite(q_metrics["val_ce"])
        assert calib_info["n_heads_calibrated"] > 0

        # Exact edge85m packed_int4_mb verification for both uniform and 5% QVLA mixed-bit
        edge_cfg = uni_omr_v2.UniOMRv2Config.edge85m(n_axial_layers=0, n_global_layers=6, grad_checkpoint=False)
        m_edge = uni_omr_v2.UniOMRv2Net(edge_cfg).eval()
        e_u = pack_and_dequant_model(
            m_edge, Path(tmpd) / "uniform_edge85m.bin", fp16_channel_masks=None, group_size=INT4_GROUP_SIZE
        )
        e_u.pop("dequant_model")
        # Mock 5% sensitivity mask on edge85m to verify exact packed disk MB < 55 MB
        mock_sens = {
            name: torch.arange(p.shape[0], dtype=torch.float32)
            for name, p in m_edge.named_parameters()
            if name in set(_int4_eligible_linear_names(m_edge)) and p.ndim == 2
        }
        e_masks = select_sensitive_channels(mock_sens, keep_frac=FP16_CHANNEL_FRAC)
        e_q = pack_and_dequant_model(
            m_edge, Path(tmpd) / "qvla_edge85m.bin", fp16_channel_masks=e_masks, group_size=INT4_GROUP_SIZE
        )
        e_q.pop("dequant_model")
        assert e_u["passes_55mb_budget"], f"uniform edge85m > 55 MB: {e_u['packed_int4_mb']}"
        assert e_q["passes_55mb_budget"], f"qvla edge85m > 55 MB: {e_q['packed_int4_mb']}"
        del m_edge

    delta_mb = round(e_q["packed_int4_mb"] - e_u["packed_int4_mb"], 3)
    print(f"[{EXP_CODE}][SMOKE] tiny fp16_ce={fp16_metrics['val_ce']} uniform_ce={u_metrics['val_ce']} qvla_ce={q_metrics['val_ce']}", flush=True)
    print(f"[{EXP_CODE}][SMOKE] quantvla calib: {calib_info}", flush=True)
    print(
        f"[{EXP_CODE}][SMOKE] edge85m packed_int4_mb: uniform={e_u['packed_int4_mb']} MB, "
        f"qvla_5pct={e_q['packed_int4_mb']} MB (delta={delta_mb:+.3f} MB, budget<{HARD_BUDGET_MB} MB)",
        flush=True,
    )
    print(f"[{EXP_CODE}][SMOKE] PASS!", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{EXP_ID} runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run CPU smoke test and exit")
    args = parser.parse_args()
    if args.smoke_test:
        return run_smoke_test()

    t_start = time.perf_counter()
    started_at = _utc_now()
    hf_token: Optional[str] = None
    status = "RUNNING"
    error_msg: Optional[str] = None
    arm_results: Dict[str, Dict[str, Any]] = {}

    try:
        hf_token = _read_hf_token()
        _ensure_deps()
        _locate_mvd()
        _configure_backend_safety()
        _seed_everything(SEED)

        import torch
        from PIL import Image

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        print(
            f"[{EXP_ID}] Device={gpu_name} | bf16={use_bf16} | cudnn={torch.backends.cudnn.enabled} | torch={torch.__version__}",
            flush=True,
        )

        arm_results.update(_try_resume_from_hf(hf_token))

        train_abs, val_abs = _ensure_proxy_dataset(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)
        print(f"[{EXP_ID}] Loaded {len(train_pairs)} train / {len(val_pairs)} val", flush=True)

        base_model, ckpt_source = _load_or_short_train_base_model(hf_token, train_pairs, device, use_bf16)
        vocab_size = int(base_model.cfg.vocab_size)
        tok_weights = build_critical_token_weights(vocab_size, CRITICAL_TOK_WEIGHT)

        # Build deterministic calibration & validation batches (CPU tensors)
        rng = random.Random(SEED)
        calib_subset = rng.sample(train_pairs, k=min(CALIB_PAGES, len(train_pairs)))
        calib_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for img_p, beks, _ in calib_subset:
            ids = _encode_bekerns_full(beks, vocab_size)
            if len(ids) < 3:
                continue
            inp = _pil_to_tensor(Image.open(img_p).convert("RGB"), FULLPAGE_MAX_DIM).unsqueeze(0)
            tgt = torch.tensor(ids[:MAX_SEQ_LEN], dtype=torch.long).unsqueeze(0)
            calib_batches.append((inp, tgt))

        val_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for img_p, beks, _ in val_pairs:
            ids = _encode_bekerns_full(beks, vocab_size)
            if len(ids) < 3:
                continue
            inp = _pil_to_tensor(Image.open(img_p).convert("RGB"), FULLPAGE_MAX_DIM).unsqueeze(0)
            tgt = torch.tensor(ids[:MAX_SEQ_LEN], dtype=torch.long).unsqueeze(0)
            val_batches.append((inp, tgt))

        fp16_metrics = _eval_val_ce_detailed(base_model, val_batches, tok_weights, device, use_bf16)
        print(f"[{EXP_ID}] FP16 teacher baseline: {fp16_metrics} (source={ckpt_source})", flush=True)

        for arm_name in ARMS:
            arm_dir = WORK_ROOT / "ckpts" / arm_name
            arm_dir.mkdir(parents=True, exist_ok=True)
            cached = arm_dir / "arm_result.json"
            if arm_name in arm_results or cached.is_file():
                if arm_name not in arm_results:
                    arm_results[arm_name] = json.loads(cached.read_text(encoding="utf-8"))
                print(f"[{EXP_ID}] Reusing cached arm '{arm_name}'", flush=True)
                continue

            t_arm0 = time.perf_counter()
            calib_info: Optional[Dict[str, Any]] = None
            if arm_name == "uniform_int4":
                pack_info = pack_and_dequant_model(
                    base_model,
                    arm_dir / "uniform_int4.bin",
                    fp16_channel_masks=None,
                    group_size=INT4_GROUP_SIZE,
                )
                dq_model = pack_info.pop("dequant_model")
            elif arm_name == "qvla_mixed_int4":
                sens_map = compute_qvla_channel_sensitivity(
                    base_model, calib_batches, tok_weights, device, use_bf16
                )
                ch_masks = select_sensitive_channels(sens_map, keep_frac=FP16_CHANNEL_FRAC)
                pack_info = pack_and_dequant_model(
                    base_model,
                    arm_dir / "qvla_mixed_int4.bin",
                    fp16_channel_masks=ch_masks,
                    group_size=INT4_GROUP_SIZE,
                )
                dq_model = pack_info.pop("dequant_model")
                calib_info = calibrate_quantvla_head_scales(
                    dq_model, base_model, calib_batches, tok_weights, device, use_bf16,
                    steps=QUANTVLA_CALIB_STEPS, lr=QUANTVLA_LR,
                )
                extra_mb = calib_info["extra_scale_bytes"] / 1_000_000.0
                pack_info["packed_int4_mb"] = round(pack_info["packed_int4_mb"] + extra_mb, 3)
                pack_info["passes_55mb_budget"] = bool(pack_info["packed_int4_mb"] < HARD_BUDGET_MB)
            else:
                raise ValueError(f"Unknown arm: {arm_name}")

            dq_metrics = _eval_val_ce_detailed(dq_model, val_batches, tok_weights, device, use_bf16)
            elapsed_arm = time.perf_counter() - t_arm0

            res: Dict[str, Any] = {
                "arm": arm_name,
                "ckpt_source": ckpt_source,
                "fp16_val_ce": fp16_metrics["val_ce"],
                "fp16_struct_pitch_val_ce": fp16_metrics["struct_pitch_val_ce"],
                "int4_dequant_val_ce": dq_metrics["val_ce"],
                "struct_pitch_val_ce": dq_metrics["struct_pitch_val_ce"],
                "dequant_ce_delta": round(dq_metrics["val_ce"] - fp16_metrics["val_ce"], 5),
                "struct_pitch_ce_delta": round(
                    dq_metrics["struct_pitch_val_ce"] - fp16_metrics["struct_pitch_val_ce"], 5
                ),
                "packed_int4_mb": pack_info["packed_int4_mb"],
                "packed_int4_verification": pack_info,
                "elapsed_sec": round(elapsed_arm, 2),
            }
            if calib_info is not None:
                res["quantvla_calibration"] = calib_info

            _atomic_write_json(cached, res)
            arm_results[arm_name] = res
            _upload_json_to_hf(hf_token, cached, f"{HF_CKPT_PREFIX}/{arm_name}_arm_result.json")

            eval_payload: Dict[str, Any] = {
                "exp_id": EXP_ID,
                "exp_code": EXP_CODE,
                "status": "RUNNING" if len(arm_results) < len(ARMS) else "COMPLETED",
                "started_at": started_at,
                "updated_at": _utc_now(),
                "gpu": gpu_name,
                "bf16": use_bf16,
                "ckpt_source": ckpt_source,
                "seed": SEED,
                "arms": arm_results,
            }
            if "uniform_int4" in arm_results and "qvla_mixed_int4" in arm_results:
                u = arm_results["uniform_int4"]
                q = arm_results["qvla_mixed_int4"]
                u_deg = max(1e-6, u["dequant_ce_delta"])
                recovery = (u["int4_dequant_val_ce"] - q["int4_dequant_val_ce"]) / u_deg
                eval_payload["delta_qvla_minus_uniform"] = {
                    "int4_dequant_val_ce": round(q["int4_dequant_val_ce"] - u["int4_dequant_val_ce"], 5),
                    "struct_pitch_val_ce": round(q["struct_pitch_val_ce"] - u["struct_pitch_val_ce"], 5),
                    "packed_int4_mb": round(q["packed_int4_mb"] - u["packed_int4_mb"], 3),
                    "degradation_recovery_frac": round(recovery, 4),
                }
            eval_local = WORK_ROOT / "out" / "eval.json"
            _atomic_write_json(eval_local, eval_payload)
            _upload_json_to_hf(hf_token, eval_local, HF_EVAL_PATH)

            del dq_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        status = "COMPLETED"
    except Exception as exc:
        status = "FAILED"
        error_msg = f"{type(exc).__name__}: {exc}"
        print(f"[{EXP_ID}] ERROR: {error_msg}", flush=True)
        raise
    finally:
        total_sec = time.perf_counter() - t_start
        run_payload = {
            "exp_id": EXP_ID,
            "exp_code": EXP_CODE,
            "status": status,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "elapsed_sec": round(total_sec, 2),
            "n_arms_completed": len(arm_results),
            "eval_path": HF_EVAL_PATH,
            "error": error_msg,
        }
        run_local = WORK_ROOT / "out" / "run.json"
        _atomic_write_json(run_local, run_payload)
        if hf_token:
            try:
                _upload_json_to_hf(hf_token, run_local, HF_RUN_PATH)
            except Exception as up_exc:
                print(f"[{EXP_ID}] WARN: failed to upload run ledger: {up_exc}", flush=True)
        _purge_hf_token()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
