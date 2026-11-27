#!/usr/bin/env python3
"""Colab A100 / Local RTX 4080 SUPER runner — Iter 15 Exp i15-43: MRFP + DELTA Debiasing.

Experiment ID: 2026-11-27T14-50-i15-43-xf-mrfp-and-delta-debias
Cross-field origin:
  - Sim2Real: MRFP (CVPR 2024) parameter-free Multi-Resolution Feature Perturbation
  - Medical Pathology: DELTA (2026) counterfactual nuisance-subspace projection

Controlled 2-arm proxy on edge85m + ax0_gl6 (600 steps/arm, BF16, seed=42):
  - Arm 0 (ctrl):       Stock edge85m + ax0_gl6.
  - Arm 1 (mrfp_delta):
      (a) Training-only parameter-free MRFP across multi-scale AnisotropicMusicBackbone
          stages (fine-scale channel stats on stage1/stage2 + coarse-scale style &
          low-frequency spatial envelope perturbation on stage3/stage4), linearly
          annealed 1.0 -> 0.0 over steps 420..600 (0.000 MB inference cost).
      (b) Post-training DELTA counterfactual nuisance-subspace estimation from paired
          clean-vs-degraded validation pages, folded with 0.000 MB / 0% FLOP cost into
          decoder cross-attention k_proj & v_proj weights (W' = W @ (I - U_k U_k^T)).

Measures per arm:
  - raw_val_ce, ema_val_ce, ema_val_ce_degraded
  - delta_val_ce_clean, delta_val_ce_degraded (and full k-sweep on held-out val)
  - int4_dequant_val_ce, int4_dequant_val_ce_degraded
  - step_ms, train_elapsed_sec, peak_vram_mb, exact packed_int4_mb (< 55.0 MB)

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
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EXP_ID = "2026-11-27T14-50-i15-43-xf-mrfp-and-delta-debias"
EXP_CODE = "i15-43"
ARMS = ("ctrl", "mrfp_delta")
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"
HF_CKPT_PREFIX = f"checkpoints/{EXP_ID}"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
WORK_ROOT = Path("/vol/i15_43") if Path("/vol").is_dir() else Path("/content/i15_43")

SEED = 42
TRAIN_STEPS = 600
WARMUP_STEPS = 60
QAT_START_STEP = 420
MRFP_ANNEAL_START = 420
WSD_DECAY_FRAC = 0.25
MIN_LR_RATIO = 0.05
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR_MAX = 3e-4
WEIGHT_DECAY = 0.04
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
EMA_DECAY = 0.99
INT4_GROUP_SIZE = 32
HARD_BUDGET_MB = 55.0
SAVE_EVERY_MINUTES = 5.0
DELTA_K_CANDIDATES = (2, 4, 8, 16)


# ---------------------------------------------------------------------------
# Experimental Module A: Parameter-Free MRFP (CVPR 2024)
# ---------------------------------------------------------------------------

class MRFPState:
    """Mutable training-only schedule holder for parameter-free MRFP (0 extra params)."""

    def __init__(self, prob: float = 0.65, fine_sigma: float = 0.18, coarse_sigma: float = 0.14) -> None:
        self.prob = float(prob)
        self.fine_sigma = float(fine_sigma)
        self.coarse_sigma = float(coarse_sigma)
        self.strength: float = 1.0  # Annealed 1.0 -> 0.0 late in training


def _mrfp_fine_perturb(x: "torch.Tensor", state: MRFPState) -> "torch.Tensor":
    """Fine-scale MRFP (stage1 /2, stage2 /4): channel-wise mean/std randomization + style mix."""
    import torch

    if state.strength <= 0.0 or random.random() > state.prob:
        return x
    B, C, _, _ = x.shape
    eps = 1e-5
    orig_dtype = x.dtype
    xf = x.float()
    mu = xf.mean(dim=(2, 3), keepdim=True)
    var = xf.var(dim=(2, 3), unbiased=False, keepdim=True)
    sig = (var + eps).sqrt()
    x_norm = (xf - mu) / sig

    s = state.fine_sigma * state.strength
    if B >= 2 and random.random() < 0.5:
        perm = torch.randperm(B, device=x.device)
        lam = torch.empty((B, 1, 1, 1), device=x.device).uniform_(0.65, 1.0)
        mu_mix = lam * mu + (1.0 - lam) * mu[perm]
        sig_mix = lam * sig + (1.0 - lam) * sig[perm]
    else:
        mu_mix = mu
        sig_mix = sig

    scale_jit = 1.0 + torch.randn((B, C, 1, 1), device=x.device).clamp(-2.5, 2.5) * s
    shift_jit = torch.randn((B, C, 1, 1), device=x.device).clamp(-2.5, 2.5) * (s * sig_mix)
    out = x_norm * (sig_mix * scale_jit.clamp(min=0.5)) + (mu_mix + shift_jit)
    return out.to(orig_dtype)


def _mrfp_coarse_perturb(x: "torch.Tensor", state: MRFPState) -> "torch.Tensor":
    """Coarse-scale MRFP (stage3 /8, stage4 /8): channel style + low-freq spatial envelope."""
    import torch
    import torch.nn.functional as F

    if state.strength <= 0.0 or random.random() > state.prob:
        return x
    B, C, H, W = x.shape
    eps = 1e-5
    orig_dtype = x.dtype
    xf = x.float()
    mu = xf.mean(dim=(2, 3), keepdim=True)
    sig = (xf.var(dim=(2, 3), unbiased=False, keepdim=True) + eps).sqrt()
    x_norm = (xf - mu) / sig

    s = state.coarse_sigma * state.strength
    # Low-frequency 4x4 spatial illumination/paper style envelope upsampled to (H, W)
    grid_h = max(2, min(4, H))
    grid_w = max(2, min(4, W))
    env = 1.0 + torch.randn((B, 1, grid_h, grid_w), device=x.device).clamp(-2.0, 2.0) * s
    env_up = F.interpolate(env, size=(H, W), mode="bilinear", align_corners=False)

    ch_scale = (1.0 + torch.randn((B, C, 1, 1), device=x.device).clamp(-2.0, 2.0) * s).clamp(min=0.5)
    ch_shift = torch.randn((B, C, 1, 1), device=x.device).clamp(-2.0, 2.0) * (s * 0.5 * sig)

    out = x_norm * (sig * ch_scale * env_up) + (mu + ch_shift)
    return out.to(orig_dtype)


def attach_mrfp_to_backbone(model: Any) -> MRFPState:
    """Wrap model.backbone.forward with parameter-free training-only MRFP (0 extra params)."""
    import torch.nn.functional as F

    mrfp_state = MRFPState()
    model._mrfp_state = mrfp_state

    def _mrfp_backbone_forward(self: Any, x: Any, return_pyramid: bool = False) -> Any:
        h1 = self.mixstyle(self.stage1(x))
        if self.training:
            h1 = _mrfp_fine_perturb(h1, mrfp_state)
        p2 = self.mixstyle(self.stage2(h1))
        if self.training:
            p2 = _mrfp_fine_perturb(p2, mrfp_state)
        _, _, h2, w2 = p2.shape
        pad_h = h2 % 2
        pad_w = w2 % 2
        p2_padded = F.pad(p2, (0, pad_w, 0, pad_h), mode="replicate") if (pad_h or pad_w) else p2
        p2_unshuffled = self.p2_unshuffle(p2_padded)
        p3 = self.stage3(p2)
        p3 = p3 + self.p2_to_p3(p2_unshuffled, (p3.shape[2], p3.shape[3]))
        if self.training:
            p3 = _mrfp_coarse_perturb(p3, mrfp_state)
        p4 = self.stage4(p3)
        p4 = p4 + self.p2_to_p4(p2_unshuffled, (p4.shape[2], p4.shape[3]))
        if self.training:
            p4 = _mrfp_coarse_perturb(p4, mrfp_state)
        out = self.out_norm(self.proj(p4))
        if return_pyramid:
            return out, p2, p3
        return out

    model.backbone.forward = types.MethodType(_mrfp_backbone_forward, model.backbone)
    return mrfp_state


def _mrfp_strength_for_step(step: int) -> float:
    if step <= MRFP_ANNEAL_START:
        return 1.0
    rem = max(1, TRAIN_STEPS - MRFP_ANNEAL_START)
    prog = min(1.0, max(0.0, float(step - MRFP_ANNEAL_START) / float(rem)))
    return 1.0 - prog


# ---------------------------------------------------------------------------
# Experimental Module B: DELTA Counterfactual Nuisance-Subspace Projection
# ---------------------------------------------------------------------------

def _synthetically_degrade_tensor(inp: "torch.Tensor", sample_idx: int) -> "torch.Tensor":
    """Deterministic paired phone-photo/scan degradation on normalized [-1, 1] RGB tensor (1, 3, H, W)."""
    import torch
    import torch.nn.functional as F

    g = torch.Generator(device="cpu").manual_seed(SEED * 10007 + sample_idx * 97 + 13)
    # Convert [-1, 1] -> [0, 1]
    x = (inp.float() * 0.5 + 0.5).clamp(0.0, 1.0)
    _, _, H, W = x.shape

    # 1. Mild box/Gaussian-like defocus blur
    k = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], device=x.device, dtype=torch.float32) / 16.0
    k = k.view(1, 1, 3, 3).expand(3, 1, 3, 3)
    x_blur = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k, groups=3)
    blur_mix = float(torch.empty(()).uniform_(0.35, 0.85, generator=g).item())
    x = (1.0 - blur_mix) * x + blur_mix * x_blur

    # 2. Uneven illumination ramp (diagonal lighting gradient + vignette)
    yy = torch.linspace(-1.0, 1.0, H, device=x.device).view(1, 1, H, 1)
    xx = torch.linspace(-1.0, 1.0, W, device=x.device).view(1, 1, 1, W)
    gx = float(torch.empty(()).uniform_(-0.22, 0.22, generator=g).item())
    gy = float(torch.empty(()).uniform_(-0.22, 0.22, generator=g).item())
    illum = (1.0 + gx * xx + gy * yy - 0.08 * (xx * xx + yy * yy)).clamp(0.60, 1.15)
    x = x * illum

    # 3. Paper yellowing / warm cast + contrast/gamma shift
    tint = torch.tensor(
        [
            float(torch.empty(()).uniform_(0.95, 1.02, generator=g).item()),
            float(torch.empty(()).uniform_(0.88, 0.96, generator=g).item()),
            float(torch.empty(()).uniform_(0.74, 0.88, generator=g).item()),
        ],
        device=x.device,
        dtype=torch.float32,
    ).view(1, 3, 1, 1)
    gamma = float(torch.empty(()).uniform_(0.80, 1.35, generator=g).item())
    contrast = float(torch.empty(()).uniform_(0.72, 0.92, generator=g).item())
    x = ((x.clamp(1e-4, 1.0) ** gamma) * tint - 0.5) * contrast + 0.5

    # 4. Additive sensor noise
    noise_std = float(torch.empty(()).uniform_(0.03, 0.065, generator=g).item())
    noise = torch.randn(x.shape, generator=g, dtype=torch.float32).to(x.device) * noise_std
    x = (x + noise).clamp(0.0, 1.0)
    return ((x - 0.5) / 0.5).to(inp.dtype)


def estimate_delta_nuisance_basis(
    model: Any,
    calib_tensors: List["torch.Tensor"],
    device: "torch.device",
    use_bf16: bool,
    max_k: int = 16,
) -> Tuple["torch.Tensor", List[float]]:
    """Estimate top-k orthonormal nuisance subspace U (D, max_k) via SVD of paired (mem_deg - mem_clean)."""
    import torch

    model.eval()
    diffs: List[torch.Tensor] = []
    with torch.no_grad():
        for idx, inp_clean in enumerate(calib_tensors):
            inp_c = inp_clean.to(device)
            inp_d = _synthetically_degrade_tensor(inp_c, sample_idx=idx)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                mem_c_lvls, _, _ = model.encode_image(inp_c, return_aux=False)
                mem_d_lvls, _, _ = model.encode_image(inp_d, return_aux=False)
            mem_c = mem_c_lvls[-1].float()  # (1, Lc, D)
            mem_d = mem_d_lvls[-1].float()  # (1, Ld, D)
            # Global pooled + 4-band vertical strip pooled diffs (robust to router index shifts)
            diffs.append((mem_d.mean(dim=1) - mem_c.mean(dim=1)).cpu())
            n_bands = 4
            Lc, Ld = mem_c.shape[1], mem_d.shape[1]
            for b in range(n_bands):
                c0, c1 = (b * Lc) // n_bands, max(1, ((b + 1) * Lc) // n_bands)
                d0, d1 = (b * Ld) // n_bands, max(1, ((b + 1) * Ld) // n_bands)
                if c1 > c0 and d1 > d0:
                    diffs.append((mem_d[:, d0:d1, :].mean(dim=1) - mem_c[:, c0:c1, :].mean(dim=1)).cpu())

    mat = torch.cat(diffs, dim=0)  # (N_diffs, D)
    mat = mat - mat.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    k_eff = min(max_k, Vh.shape[0])
    basis = Vh[:k_eff, :].T.contiguous()  # (D, k_eff) orthonormal columns
    explained = (S[:k_eff].pow(2) / S.pow(2).sum().clamp(min=1e-12)).tolist()
    return basis, [round(float(v), 5) for v in explained]


def fold_delta_projection_into_model(
    model: Any,
    basis_u: "torch.Tensor",
    k: int,
    alpha: float = 1.0,
) -> None:
    """Zero-cost linear fold: W' = W @ (I - alpha * U_k U_k^T) into all decoder cross-attn k_proj & v_proj."""
    import torch

    if k <= 0 or alpha <= 0.0:
        return
    u_k = basis_u[:, : min(k, basis_u.shape[1])].float()  # (D, k)
    D = u_k.shape[0]
    proj = torch.eye(D, dtype=torch.float32) - float(alpha) * (u_k @ u_k.T)  # (D, D) symmetric

    with torch.no_grad():
        for layer in model.dec_layers:
            ca = layer.cross_attn
            for lin in (ca.k_proj, ca.v_proj):
                w = lin.weight.data
                p_dev = proj.to(device=w.device, dtype=torch.float32)
                lin.weight.data.copy_((w.float() @ p_dev).to(w.dtype))


def build_arm_model(arm_name: str, grad_checkpoint: bool = True, tiny: bool = False) -> Any:
    """Build edge85m + ax0_gl6 (or tiny d=64 for smoke test) for 'ctrl' or 'mrfp_delta'."""
    import uni_omr_v2

    if tiny:
        cfg = uni_omr_v2.UniOMRv2Config(
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
            grad_checkpoint=grad_checkpoint,
        )
    else:
        cfg = uni_omr_v2.UniOMRv2Config.edge85m(
            n_axial_layers=0,
            n_global_layers=6,
            grad_checkpoint=grad_checkpoint,
        )
    model = uni_omr_v2.UniOMRv2Net(cfg)
    if arm_name == "mrfp_delta":
        attach_mrfp_to_backbone(model)
    elif arm_name != "ctrl":
        raise ValueError(f"Unknown arm_name: {arm_name}")
    return model


# ---------------------------------------------------------------------------
# Infrastructure Helpers (Windows Safety, HF Auth & Partial Resume, Dataset, INT4)
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
        if (cand / "uni_omr_v2.py").is_file() and (cand / "train_uni_v2.py").is_file():
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


def _try_resume_from_hf(hf_token: str) -> Dict[str, Dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    recovered: Dict[str, Dict[str, Any]] = {}
    dl_dir = WORK_ROOT / "hf_resume"
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
            if arm_name in ARMS and isinstance(arm_res, dict) and "ema_val_ce" in arm_res:
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
            if isinstance(arm_res, dict) and "ema_val_ce" in arm_res:
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


def _pack_and_verify_hybrid_int4(
    model: Any,
    arm_name: str,
    out_bin: Path,
    group_size: int = INT4_GROUP_SIZE,
    tiny: bool = False,
) -> Dict[str, Any]:
    import torch
    import torch.nn as nn

    infer_model = build_arm_model(arm_name, grad_checkpoint=False, tiny=tiny).cpu().eval()
    infer_model.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()}, strict=True)
    infer_model.strip_aux_heads()
    infer_model.mtp_heads = nn.ModuleList()

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
    n_int4_tensors = 0
    n_fp16_tensors = 0

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
                n_int4_tensors += 1
                dq = (q.float() * scale.float()).reshape(-1)[:orig_n].reshape(t.shape)
                dequant_sd[name] = dq.to(param.dtype)
            else:
                fp16_np = t.to(torch.float16).numpy()
                f.write(fp16_np.tobytes())
                total_fp16_bytes += int(fp16_np.nbytes)
                n_fp16_tensors += 1
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
        "n_int4_tensors": n_int4_tensors,
        "n_fp16_tensors": n_fp16_tensors,
        "dequant_model": infer_model,
    }


def _preload_val_samples(
    val_pairs: List[Tuple[Path, List[Path], str, float]],
) -> List[Tuple["torch.Tensor", "torch.Tensor"]]:
    import torch
    from PIL import Image

    loaded: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for img_p, beks, _, _ in val_pairs:
        ids = _encode_bekerns_full(beks)
        if len(ids) < 3:
            continue
        img = Image.open(img_p).convert("RGB")
        inp = _pil_to_tensor(img, FULLPAGE_MAX_DIM).unsqueeze(0)
        tgt = torch.tensor(ids[:MAX_SEQ_LEN], dtype=torch.long).unsqueeze(0)
        loaded.append((inp, tgt))
    return loaded


def _eval_val_ce_samples(
    model: Any,
    val_samples: List[Tuple["torch.Tensor", "torch.Tensor"]],
    device: "torch.device",
    use_bf16: bool,
    degraded: bool = False,
) -> float:
    import torch
    import torch.nn.functional as F
    import uni_omr

    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for idx, (inp_cpu, tgt_cpu) in enumerate(val_samples):
            inp = inp_cpu.to(device)
            if degraded:
                inp = _synthetically_degrade_tensor(inp, sample_idx=idx)
            tgt = tgt_cpu.to(device)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                mem, _, _ = model.encode_image(inp, return_aux=False)
                logits, _, _ = model.decode_tokens(
                    tgt[:, :-1], mem, return_mtp=False, return_aux=False, apply_tok_dropout=False
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=uni_omr.PAD_ID
                )
            losses.append(float(loss.item()))
    return round(sum(losses) / max(1, len(losses)), 5)


def _collate_microbatch(
    samples: List[Tuple[Path, List[Path], str, float]],
    device: "torch.device",
) -> Optional[Tuple["torch.Tensor", "torch.Tensor"]]:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import uni_omr

    imgs: List[torch.Tensor] = []
    seqs: List[List[int]] = []
    for img_p, beks, _, _ in samples:
        ids = _encode_bekerns_full(beks)
        if len(ids) < 3:
            continue
        img = Image.open(img_p).convert("RGB")
        imgs.append(_pil_to_tensor(img, FULLPAGE_MAX_DIM))
        seqs.append(ids[:MAX_SEQ_LEN])
    if not imgs:
        return None

    max_h = max(t.shape[1] for t in imgs)
    max_w = max(t.shape[2] for t in imgs)
    padded_imgs = [
        F.pad(t, (0, max_w - t.shape[2], 0, max_h - t.shape[1]), value=1.0)
        for t in imgs
    ]
    max_t = max(len(s) for s in seqs)
    tgt = torch.full((len(seqs), max_t), uni_omr.PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        tgt[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return torch.stack(padded_imgs, dim=0).to(device), tgt.to(device)


def _train_arm(
    arm_name: str,
    *,
    train_pairs: List[Tuple[Path, List[Path], str, float]],
    val_samples: List[Tuple["torch.Tensor", "torch.Tensor"]],
    device: "torch.device",
    use_bf16: bool,
) -> Dict[str, Any]:
    import torch
    import train_uni_v2
    import uni_omr_v2

    _seed_everything(SEED)
    arm_dir = WORK_ROOT / "ckpts" / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    cached = arm_dir / "arm_result.json"
    if cached.is_file():
        print(f"[{EXP_ID}] Reusing cached result for {arm_name}", flush=True)
        return json.loads(cached.read_text(encoding="utf-8"))

    model = build_arm_model(arm_name, grad_checkpoint=True).to(device)
    psummary = model.param_summary(int4_block_size=INT4_GROUP_SIZE)
    qat = uni_omr_v2.QATInt4ControllerV2(model, block_size=INT4_GROUP_SIZE)
    opt = train_uni_v2.build_optimizer_v2(
        model,
        optimizer_name="muon",
        lr=LR_MAX,
        weight_decay=WEIGHT_DECAY,
        device_type=device.type,
        no_decay_1d=True,
    )
    if device.type == "cuda" and opt.mode != "muon+adamw":
        raise RuntimeError(f"[{EXP_ID}][{arm_name}] Silent fallback detected: expected muon+adamw, got {opt.mode}")
    print(f"[{EXP_ID}][{arm_name}] Optimizer verified: mode={opt.mode}", flush=True)

    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    weights = [p[3] for p in train_pairs]
    rng = random.Random(SEED)
    start_step = 1
    loss_hist: List[float] = []
    step_ms_hist: List[float] = []

    latest_pt = arm_dir / "latest.pt"
    if latest_pt.is_file():
        ck = torch.load(str(latest_pt), map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        ema_state = {k: v.to(device) for k, v in ck["ema_state_dict"].items()}
        if "opt_state_dicts" in ck and len(ck["opt_state_dicts"]) == len(opt.optimizers):
            for o_obj, sd in zip(opt.optimizers, ck["opt_state_dicts"]):
                o_obj.load_state_dict(sd)
        if "rng_state" in ck:
            rng.setstate(ck["rng_state"])
        start_step = int(ck["step"]) + 1
        loss_hist = list(ck.get("loss_hist", []))
        step_ms_hist = list(ck.get("step_ms_hist", []))
        if start_step > QAT_START_STEP:
            qat.enable()
        print(f"[{EXP_ID}][{arm_name}] Resumed local checkpoint at step {start_step}/{TRAIN_STEPS}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t_arm0 = time.perf_counter()
    t_last_save = time.time()

    model.train()
    for step in range(start_step, TRAIN_STEPS + 1):
        if step == QAT_START_STEP:
            qat.enable()
            print(f"[{EXP_ID}][{arm_name}] Enabled QATInt4ControllerV2 at step {step}", flush=True)

        if hasattr(model, "_mrfp_state"):
            model._mrfp_state.strength = _mrfp_strength_for_step(step)

        mult = _wsd_mult(step, TRAIN_STEPS, WARMUP_STEPS)
        opt.set_lr(mult)

        t_step0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            batch_samples = rng.choices(train_pairs, weights=weights, k=BATCH_SIZE)
            collated = _collate_microbatch(batch_samples, device)
            if collated is None:
                continue
            inp, tgt = collated
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                out = model(inp, tgt[:, :-1], return_aux=False, return_mtp=True)
                losses = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)
                loss = losses["loss"] / GRAD_ACCUM
            loss.backward()
            step_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema_state[k].mul_(EMA_DECAY).add_(v, alpha=1.0 - EMA_DECAY)
                else:
                    ema_state[k].copy_(v)

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_ms_hist.append((time.perf_counter() - t_step0) * 1000.0)
        loss_hist.append(step_loss)

        if step % 100 == 0 or step == 1:
            mean_l = sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:]))
            mean_ms = sum(step_ms_hist[-50:]) / max(1, len(step_ms_hist[-50:]))
            mrfp_str = f" mrfp={model._mrfp_state.strength:.2f}" if hasattr(model, "_mrfp_state") else ""
            print(
                f"[{EXP_ID}][{arm_name}] step={step}/{TRAIN_STEPS} "
                f"loss={mean_l:.4f} step_ms={mean_ms:.1f}{mrfp_str} elapsed={time.perf_counter() - t_arm0:.1f}s",
                flush=True,
            )

        if (time.time() - t_last_save) >= SAVE_EVERY_MINUTES * 60.0:
            tmp_ck = arm_dir / "latest.pt.tmp"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema_state,
                    "opt_state_dicts": [o.state_dict() for o in opt.optimizers],
                    "rng_state": rng.getstate(),
                    "loss_hist": loss_hist,
                    "step_ms_hist": step_ms_hist,
                },
                tmp_ck,
            )
            os.replace(tmp_ck, latest_pt)
            t_last_save = time.time()

    train_elapsed_sec = time.perf_counter() - t_arm0
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
    )

    qat.disable()
    raw_val_ce = _eval_val_ce_samples(model, val_samples, device, use_bf16, degraded=False)

    model.load_state_dict(ema_state, strict=True)
    ema_val_ce = _eval_val_ce_samples(model, val_samples, device, use_bf16, degraded=False)
    ema_val_ce_degraded = _eval_val_ce_samples(model, val_samples, device, use_bf16, degraded=True)

    # Post-training DELTA counterfactual nuisance-subspace estimation on first half of val,
    # evaluated on held-out second half + full val
    split_n = max(1, len(val_samples) // 2)
    calib_imgs = [s[0] for s in val_samples[:split_n]]
    holdout_samples = val_samples[split_n:] if len(val_samples) > split_n else val_samples
    basis_u, explained_var = estimate_delta_nuisance_basis(
        model, calib_imgs, device, use_bf16, max_k=max(DELTA_K_CANDIDATES)
    )

    base_ema_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    delta_sweep: Dict[str, Dict[str, float]] = {}
    best_k = 0
    best_score = ema_val_ce + ema_val_ce_degraded
    for k_cand in DELTA_K_CANDIDATES:
        model.load_state_dict({k: v.to(device) for k, v in base_ema_cpu.items()}, strict=True)
        fold_delta_projection_into_model(model, basis_u, k=k_cand, alpha=1.0)
        ho_clean = _eval_val_ce_samples(model, holdout_samples, device, use_bf16, degraded=False)
        ho_deg = _eval_val_ce_samples(model, holdout_samples, device, use_bf16, degraded=True)
        delta_sweep[f"k={k_cand}"] = {"holdout_clean_ce": ho_clean, "holdout_degraded_ce": ho_deg}
        if (ho_clean + ho_deg) < best_score:
            best_score = ho_clean + ho_deg
            best_k = k_cand

    # Apply best_k (or k=4 if mrfp_delta wants explicit DELTA eval) onto full val
    eval_k = best_k if best_k > 0 else DELTA_K_CANDIDATES[1]
    model.load_state_dict({k: v.to(device) for k, v in base_ema_cpu.items()}, strict=True)
    fold_delta_projection_into_model(model, basis_u, k=eval_k, alpha=1.0)
    delta_val_ce_clean = _eval_val_ce_samples(model, val_samples, device, use_bf16, degraded=False)
    delta_val_ce_degraded = _eval_val_ce_samples(model, val_samples, device, use_bf16, degraded=True)

    # Restore pre-DELTA EMA for ctrl INT4 packing; keep DELTA-folded model for mrfp_delta INT4 packing
    if arm_name == "ctrl":
        model.load_state_dict({k: v.to(device) for k, v in base_ema_cpu.items()}, strict=True)

    packed_info = _pack_and_verify_hybrid_int4(
        model, arm_name, arm_dir / f"{arm_name}_hybrid_int4.bin", INT4_GROUP_SIZE
    )
    dq_model = packed_info.pop("dequant_model").to(device)
    int4_dequant_val_ce = _eval_val_ce_samples(dq_model, val_samples, device, use_bf16, degraded=False)
    int4_dequant_val_ce_degraded = _eval_val_ce_samples(dq_model, val_samples, device, use_bf16, degraded=True)

    steady_ms = step_ms_hist[10:] if len(step_ms_hist) > 10 else step_ms_hist
    result: Dict[str, Any] = {
        "arm": arm_name,
        "optimizer_mode": opt.mode,
        "steps": TRAIN_STEPS,
        "param_summary": psummary,
        "packed_int4_mb": packed_info["packed_int4_mb"],
        "packed_int4_verification": packed_info,
        "train_loss_tail50": round(sum(loss_hist[-50:]) / max(1, len(loss_hist[-50:])), 5),
        "raw_val_ce": raw_val_ce,
        "ema_val_ce": ema_val_ce,
        "ema_val_ce_degraded": ema_val_ce_degraded,
        "delta_best_k": eval_k,
        "delta_val_ce_clean": delta_val_ce_clean,
        "delta_val_ce_degraded": delta_val_ce_degraded,
        "delta_sweep": delta_sweep,
        "delta_explained_var_top8": explained_var[:8],
        "int4_dequant_val_ce": int4_dequant_val_ce,
        "int4_dequant_val_ce_degraded": int4_dequant_val_ce_degraded,
        "step_ms": round(sum(steady_ms) / max(1, len(steady_ms)), 2),
        "train_elapsed_sec": round(train_elapsed_sec, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
    }
    _atomic_write_json(cached, result)
    del model, dq_model, opt, ema_state, base_ema_cpu
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


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


# ---------------------------------------------------------------------------
# CPU 2-Step Smoke Test
# ---------------------------------------------------------------------------

def run_smoke_test() -> int:
    """2-step tiny CPU forward+backward+step + MRFP/DELTA projection + edge85m INT4 size verification."""
    _locate_mvd()
    import tempfile
    import torch
    import train_uni_v2
    import uni_omr

    device = torch.device("cpu")
    _seed_everything(SEED)

    tiny_results: Dict[str, Dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as tmpd:
        for arm_name in ARMS:
            model = build_arm_model(arm_name, grad_checkpoint=False, tiny=True).to(device)
            opt = train_uni_v2.build_optimizer_v2(
                model,
                optimizer_name="muon",
                lr=LR_MAX,
                weight_decay=WEIGHT_DECAY,
                device_type="cpu",
                allow_cpu_muon=True,
                no_decay_1d=True,
            )
            assert opt.mode == "muon+adamw", f"{arm_name}: expected muon+adamw, got {opt.mode}"
            model.train()

            losses_seen: List[float] = []
            calib_imgs: List[torch.Tensor] = []
            for step in range(2):
                if hasattr(model, "_mrfp_state"):
                    model._mrfp_state.strength = 1.0 if step == 0 else 0.5
                opt.zero_grad(set_to_none=True)
                imgs = torch.randn(2, 3, 64, 96, device=device)
                calib_imgs.append(imgs[:1].detach().clone())
                tgt = torch.randint(4, min(64, uni_omr.VOCAB_SIZE), (2, 14), device=device)
                tgt[:, 0] = uni_omr.BOS_ID
                out = model(imgs, tgt[:, :-1], return_aux=False, return_mtp=True)
                loss = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)["loss"]
                assert torch.isfinite(loss), f"{arm_name} step {step}: non-finite loss {loss}"
                loss.backward()
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all(), f"{arm_name} non-finite grad in {n}"
                opt.step()
                losses_seen.append(round(float(loss.item()), 4))

            # Verify MRFP is deterministic/off in eval() and DELTA projection folds cleanly
            model.eval()
            with torch.no_grad():
                m1, _, _ = model.encode_image(calib_imgs[0], return_aux=False)
                m2, _, _ = model.encode_image(calib_imgs[0], return_aux=False)
                assert torch.allclose(m1[-1], m2[-1], atol=1e-6), f"{arm_name}: eval() not deterministic!"

            basis_u, exp_var = estimate_delta_nuisance_basis(model, calib_imgs, device, use_bf16=False, max_k=4)
            assert basis_u.shape == (model.cfg.d_model, min(4, basis_u.shape[1]))
            assert torch.isfinite(basis_u).all()
            if arm_name == "mrfp_delta":
                fold_delta_projection_into_model(model, basis_u, k=2, alpha=1.0)

            pack_tiny = _pack_and_verify_hybrid_int4(
                model, arm_name, Path(tmpd) / f"{arm_name}_tiny.bin", INT4_GROUP_SIZE, tiny=True
            )
            dq_model = pack_tiny.pop("dequant_model")
            with torch.no_grad():
                mem, _, _ = dq_model.encode_image(calib_imgs[0], return_aux=False)
                dq_logits, _, _ = dq_model.decode_tokens(
                    tgt[:1, :-1], mem, return_mtp=False, return_aux=False, apply_tok_dropout=False
                )
                seq = dq_model.generate(calib_imgs[0], max_len=4)
            assert dq_logits.shape == (1, 13, uni_omr.VOCAB_SIZE) and torch.isfinite(dq_logits).all()
            assert seq.shape[0] == 1 and seq.shape[1] >= 2

            tiny_results[arm_name] = {
                "opt_mode": opt.mode,
                "losses_2step": losses_seen,
                "tiny_params": sum(p.numel() for p in model.parameters()),
                "delta_explained_var": exp_var[:2],
            }

        edge_mb: Dict[str, float] = {}
        for arm_name in ARMS:
            m_edge = build_arm_model(arm_name, grad_checkpoint=False, tiny=False)
            pack_edge = _pack_and_verify_hybrid_int4(
                m_edge, arm_name, Path(tmpd) / f"{arm_name}_edge85m.bin", INT4_GROUP_SIZE, tiny=False
            )
            pack_edge.pop("dequant_model")
            assert pack_edge["passes_55mb_budget"], f"{arm_name} exceeded 55 MB budget: {pack_edge['packed_int4_mb']}"
            edge_mb[arm_name] = pack_edge["packed_int4_mb"]
            del m_edge

    delta_mb = round(edge_mb["mrfp_delta"] - edge_mb["ctrl"], 3)
    assert abs(delta_mb) < 1e-6, f"Expected 0.000 MB inference cost for mrfp_delta, got {delta_mb}"
    print(f"[{EXP_CODE}][SMOKE] tiny ctrl:       {tiny_results['ctrl']}", flush=True)
    print(f"[{EXP_CODE}][SMOKE] tiny mrfp_delta: {tiny_results['mrfp_delta']}", flush=True)
    print(
        f"[{EXP_CODE}][SMOKE] edge85m packed_int4_mb: ctrl={edge_mb['ctrl']} MB, "
        f"mrfp_delta={edge_mb['mrfp_delta']} MB (delta={delta_mb:+.3f} MB)",
        flush=True,
    )
    print(f"[{EXP_CODE}][SMOKE] PASS!", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{EXP_ID} runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-step tiny CPU smoke test and exit")
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

        import torch

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
        val_samples = _preload_val_samples(val_pairs)
        print(f"[{EXP_ID}] Loaded {len(train_pairs)} train / {len(val_samples)} val", flush=True)

        for arm_name in ARMS:
            res = _train_arm(
                arm_name,
                train_pairs=train_pairs,
                val_samples=val_samples,
                device=device,
                use_bf16=use_bf16,
            )
            arm_results[arm_name] = res

            arm_json_local = WORK_ROOT / "ckpts" / arm_name / "arm_result.json"
            _upload_json_to_hf(hf_token, arm_json_local, f"{HF_CKPT_PREFIX}/{arm_name}_arm_result.json")

            eval_payload: Dict[str, Any] = {
                "exp_id": EXP_ID,
                "exp_code": EXP_CODE,
                "status": "RUNNING" if len(arm_results) < len(ARMS) else "COMPLETED",
                "started_at": started_at,
                "updated_at": _utc_now(),
                "gpu": gpu_name,
                "bf16": use_bf16,
                "steps_per_arm": TRAIN_STEPS,
                "seed": SEED,
                "arms": arm_results,
            }
            if "ctrl" in arm_results and "mrfp_delta" in arm_results:
                c = arm_results["ctrl"]
                t = arm_results["mrfp_delta"]
                eval_payload["delta_mrfp_delta_minus_ctrl"] = {
                    "ema_val_ce": round(t["ema_val_ce"] - c["ema_val_ce"], 5),
                    "ema_val_ce_degraded": round(t["ema_val_ce_degraded"] - c["ema_val_ce_degraded"], 5),
                    "delta_val_ce_clean": round(t["delta_val_ce_clean"] - c["ema_val_ce"], 5),
                    "delta_val_ce_degraded": round(t["delta_val_ce_degraded"] - c["ema_val_ce_degraded"], 5),
                    "int4_dequant_val_ce": round(t["int4_dequant_val_ce"] - c["int4_dequant_val_ce"], 5),
                    "int4_dequant_val_ce_degraded": round(
                        t["int4_dequant_val_ce_degraded"] - c["int4_dequant_val_ce_degraded"], 5
                    ),
                    "step_ms": round(t["step_ms"] - c["step_ms"], 2),
                    "packed_int4_mb": round(t["packed_int4_mb"] - c["packed_int4_mb"], 3),
                }
            eval_local = WORK_ROOT / "out" / "eval.json"
            _atomic_write_json(eval_local, eval_payload)
            _upload_json_to_hf(hf_token, eval_local, HF_EVAL_PATH)

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
