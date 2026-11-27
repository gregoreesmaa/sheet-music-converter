#!/usr/bin/env python3
"""Colab A100 / Local RTX 4080 SUPER runner — Iter 15 Exp i15-33: Differential Cross-Attention.

Experiment ID: 2026-11-27T14-15-i15-33-diff-cross-attn
Controlled 2-arm weight-incompatible proxy on edge85m + ax0_gl6 (600 steps/arm, BF16, seed=42):
  - Arm 0 (ctrl):    edge85m + ax0_gl6 stock GQA cross-attention (d_qk=64, d_v=64).
  - Arm 1 (diff_ca): Differential Cross-Attention (Ye et al. 2024) in all 10 decoder cross-attn
                     layers: split Q and K head_dim into two 32-dim halves (Q1, K1) and (Q2, K2),
                     learnable per-layer lambda = exp(lam_q1·lam_k1) - exp(lam_q2·lam_k2) + lam_init,
                     RMSNorm(Attn(Q1, K1, V) - lambda * Attn(Q2, K2, V)) * (1 - lam_init).
                     (+0.004 MB INT4, 0% extra KV-cache RAM).

Measures per arm:
  - raw_val_ce, ema_val_ce, int4_dequant_val_ce
  - step_ms, train_elapsed_sec, peak_vram_mb
  - exact serialized packed_int4_mb (hard < 55.0 MB budget)

Includes 1-arm HF partial checkpoint sync + startup auto-resume and Windows cuDNN 9.10 safety.
Security: reads HF_TOKEN from /tmp/.hf_token (chmod 600), HF_TOKEN env, or ~/.cache/huggingface/token;
never prints it, always unlinks /tmp/.hf_token in finally.
"""

from __future__ import annotations

import argparse
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

EXP_ID = "2026-11-27T14-15-i15-33-diff-cross-attn"
EXP_CODE = "i15-33"
ARMS = ("ctrl", "diff_ca")
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"
HF_CKPT_PREFIX = f"checkpoints/{EXP_ID}"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
WORK_ROOT = Path("/vol/i15_33") if Path("/vol").is_dir() else Path("/content/i15_33")

SEED = 42
TRAIN_STEPS = 600
WARMUP_STEPS = 60
QAT_START_STEP = 420
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


# ---------------------------------------------------------------------------
# Self-Contained Experimental Module: DiffCrossAttention (Ye et al. 2024)
# ---------------------------------------------------------------------------

def _lambda_init_fn(layer_idx: int) -> float:
    return 0.8 - 0.6 * math.exp(-0.3 * float(layer_idx))


def _build_diff_cross_attn_class() -> Any:
    """Factory returning DiffCrossAttention using uni_omr_v2 primitives."""
    import torch
    import torch.nn as nn
    import uni_omr_v2

    class DiffCrossAttention(nn.Module):
        """Differential GQA Cross-Attention (Ye et al. 2024) for noise-canceling OMR decoding.

        Splits head_dim into two equal halves (half_dim = head_dim // 2) for Q and K while
        keeping V at full head_dim:
            Attn1 = SDPA(Q1, K1, V),  Attn2 = SDPA(Q2, K2, V)
            lambda = exp(lam_q1 · lam_k1) - exp(lam_q2 · lam_k2) + lambda_init(layer_idx)
            out = RMSNorm(Attn1 - lambda * Attn2) * (1 - lambda_init)
        Caches packed K = cat([K1, K2], dim=-1) so KV-cache shape is identical to stock GQA.
        """

        def __init__(
            self,
            d_model: int,
            n_heads: int = 8,
            n_kv_heads: int = 2,
            layer_idx: int = 0,
            use_native_gqa: bool = True,
        ) -> None:
            super().__init__()
            if d_model % n_heads != 0:
                raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
            if n_heads % n_kv_heads != 0:
                raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
            self.d_model = int(d_model)
            self.n_heads = int(n_heads)
            self.n_kv_heads = int(n_kv_heads)
            self.n_rep = self.n_heads // self.n_kv_heads
            self.head_dim = self.d_model // self.n_heads
            if self.head_dim % 2 != 0:
                raise ValueError(f"head_dim ({self.head_dim}) must be even for DiffCrossAttention")
            self.half_dim = self.head_dim // 2
            self.kv_dim = self.n_kv_heads * self.head_dim
            self.layer_idx = int(layer_idx)
            self.lambda_init = float(_lambda_init_fn(self.layer_idx))
            self.use_native_gqa = bool(use_native_gqa)
            self.use_rope = False

            # Identical Linear shapes to GroupedQueryAttention (0 extra matmul params)
            self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.k_proj = nn.Linear(self.d_model, self.kv_dim, bias=False)
            self.v_proj = nn.Linear(self.d_model, self.kv_dim, bias=False)
            self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

            # Per-half QK-Norm (same total param count as stock q_norm + k_norm)
            self.q1_norm = uni_omr_v2.RMSNorm(self.half_dim)
            self.q2_norm = uni_omr_v2.RMSNorm(self.half_dim)
            self.k1_norm = uni_omr_v2.RMSNorm(self.half_dim)
            self.k2_norm = uni_omr_v2.RMSNorm(self.half_dim)

            # Learnable per-layer differential scalar parameters (1D -> AdamW no_decay)
            self.lambda_q1 = nn.Parameter(torch.zeros(self.half_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1))
            self.lambda_k1 = nn.Parameter(torch.zeros(self.half_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1))
            self.lambda_q2 = nn.Parameter(torch.zeros(self.half_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1))
            self.lambda_k2 = nn.Parameter(torch.zeros(self.half_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1))

            # Per-head SubLayer RMSNorm (Ye et al. 2024 GroupNorm/RMSNorm equivalent)
            self.subln = uni_omr_v2.RMSNorm(self.head_dim)

        def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
            if self.n_rep == 1:
                return x
            B, n_kv, L, hd = x.shape
            return x.unsqueeze(2).expand(B, n_kv, self.n_rep, L, hd).reshape(B, self.n_heads, L, hd)

        def project_kv(self, kv_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            B, Lk, _ = kv_in.shape
            k_raw = self.k_proj(kv_in).view(B, Lk, self.n_kv_heads, 2, self.half_dim)
            k1 = self.k1_norm(k_raw[:, :, :, 0, :]).transpose(1, 2)  # (B, n_kv, Lk, half_dim)
            k2 = self.k2_norm(k_raw[:, :, :, 1, :]).transpose(1, 2)  # (B, n_kv, Lk, half_dim)
            k_packed = torch.cat([k1, k2], dim=-1)  # (B, n_kv, Lk, head_dim)
            v = self.v_proj(kv_in).view(B, Lk, self.n_kv_heads, self.head_dim).transpose(1, 2)
            return k_packed, v

        def compute_lambda(self, dtype: torch.dtype) -> torch.Tensor:
            lam_1 = torch.exp(torch.sum(self.lambda_q1.float() * self.lambda_k1.float()))
            lam_2 = torch.exp(torch.sum(self.lambda_q2.float() * self.lambda_k2.float()))
            return (lam_1 - lam_2 + self.lambda_init).to(dtype)

        def forward(
            self,
            q_in: torch.Tensor,
            kv_in: Optional[torch.Tensor],
            rope_cos: Optional[torch.Tensor] = None,
            rope_sin: Optional[torch.Tensor] = None,
            is_causal: bool = False,
            attn_mask: Optional[torch.Tensor] = None,
            past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            is_cross: bool = True,
        ) -> Any:
            B, Lq, _ = q_in.shape
            q_raw = self.q_proj(q_in).view(B, Lq, self.n_heads, 2, self.half_dim)
            q1 = self.q1_norm(q_raw[:, :, :, 0, :]).transpose(1, 2)  # (B, n_heads, Lq, half_dim)
            q2 = self.q2_norm(q_raw[:, :, :, 1, :]).transpose(1, 2)  # (B, n_heads, Lq, half_dim)

            if past_kv is not None:
                k_packed, v = past_kv
            else:
                if kv_in is None:
                    raise ValueError("kv_in must be provided when past_kv is None")
                k_packed, v = self.project_kv(kv_in)

            present_kv = (k_packed, v) if use_cache else None
            k1, k2 = k_packed.chunk(2, dim=-1)  # each (B, n_kv, Lk, half_dim)

            # Expand KV for broad SDPA compatibility with asymmetric d_qk=half_dim, d_v=head_dim
            k1_exp = self._expand_kv(k1)
            k2_exp = self._expand_kv(k2)
            v_exp = self._expand_kv(v)

            attn1 = uni_omr_v2._sdpa(q1, k1_exp, v_exp, attn_mask=attn_mask, is_causal=False, enable_gqa=False)
            attn2 = uni_omr_v2._sdpa(q2, k2_exp, v_exp, attn_mask=attn_mask, is_causal=False, enable_gqa=False)

            lam = self.compute_lambda(q1.dtype)
            diff = attn1 - lam * attn2  # (B, n_heads, Lq, head_dim)
            out = self.subln(diff) * (1.0 - self.lambda_init)
            out = out.transpose(1, 2).reshape(B, Lq, self.d_model)
            out = self.o_proj(out)
            if use_cache:
                return out, present_kv
            return out

    return DiffCrossAttention


def apply_diff_cross_attn(model: Any) -> Any:
    """Replace all decoder cross_attn layers with DiffCrossAttention in-place."""
    DiffCrossAttention = _build_diff_cross_attn_class()
    c = model.cfg

    for idx, layer in enumerate(model.dec_layers):
        layer.cross_attn = DiffCrossAttention(
            d_model=c.d_model,
            n_heads=c.n_heads,
            n_kv_heads=c.n_kv_heads,
            layer_idx=idx,
            use_native_gqa=True,
        )

    def _precompute_cross_kv_diff(self: Any, mem_levels: Any) -> Any:
        if isinstance(mem_levels, (tuple, list)):
            mem_post_merger, mem_global_mid, mem_final = mem_levels
            level_mems = (mem_post_merger, mem_global_mid)
        else:
            mem_final = mem_levels
            level_mems = None
        past_kvs = []
        for idx, layer in enumerate(self.dec_layers):
            if level_mems is not None and idx < len(self.deepstack_fusions):
                layer_mem = self.deepstack_fusions[idx](mem_final, level_mems[idx])
            else:
                layer_mem = mem_final
            k_packed, v = layer.cross_attn.project_kv(layer_mem)
            past_kvs.append((None, (k_packed, v)))
        return past_kvs

    model.precompute_cross_kv = types.MethodType(_precompute_cross_kv_diff, model)
    return model


def build_arm_model(arm_name: str, grad_checkpoint: bool = True, tiny: bool = False) -> Any:
    """Build edge85m + ax0_gl6 (or tiny d=64 for smoke test) model for 'ctrl' or 'diff_ca'."""
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
    if arm_name == "diff_ca":
        apply_diff_cross_attn(model)
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
    """Download any already-completed arm results from HF Private Hub into local cache."""
    from huggingface_hub import hf_hub_download

    recovered: Dict[str, Dict[str, Any]] = {}
    dl_dir = WORK_ROOT / "hf_resume"
    dl_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check partial/complete evals/<EXP_ID>.json
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

    # 2. Check per-arm partial checkpoint JSONs under checkpoints/<EXP_ID>/
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
    """Strip aux/MTP heads, pack QAT-INT4 Linear weights into 4-bit nibbles + FP16 scales,
    write exact binary file to disk, and dequantize onto a fresh model copy."""
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
    val_pairs: List[Tuple[Path, List[Path], str, float]],
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
            print(
                f"[{EXP_ID}][{arm_name}] step={step}/{TRAIN_STEPS} "
                f"loss={mean_l:.4f} step_ms={mean_ms:.1f} elapsed={time.perf_counter() - t_arm0:.1f}s",
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
    raw_val_ce = _eval_val_ce(model, val_pairs, device, use_bf16)

    model.load_state_dict(ema_state, strict=True)
    ema_val_ce = _eval_val_ce(model, val_pairs, device, use_bf16)

    # Record learned per-layer lambda values for forensic inspection
    learned_lambdas: Optional[List[float]] = None
    if arm_name == "diff_ca":
        with torch.no_grad():
            learned_lambdas = [
                round(float(layer.cross_attn.compute_lambda(torch.float32).item()), 4)
                for layer in model.dec_layers
            ]

    packed_info = _pack_and_verify_hybrid_int4(
        model, arm_name, arm_dir / f"{arm_name}_hybrid_int4.bin", INT4_GROUP_SIZE
    )
    dq_model = packed_info.pop("dequant_model").to(device)
    int4_dequant_val_ce = _eval_val_ce(dq_model, val_pairs, device, use_bf16)

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
        "int4_dequant_val_ce": int4_dequant_val_ce,
        "step_ms": round(sum(steady_ms) / max(1, len(steady_ms)), 2),
        "train_elapsed_sec": round(train_elapsed_sec, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
    }
    if learned_lambdas is not None:
        result["learned_lambdas"] = learned_lambdas
    _atomic_write_json(cached, result)
    del model, dq_model, opt, ema_state
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
    """2-step tiny CPU forward+backward+step + KV-cache + edge85m packed_int4_mb verification."""
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
                model, optimizer_name="muon", lr=LR_MAX, weight_decay=WEIGHT_DECAY,
                device_type="cpu", allow_cpu_muon=True, no_decay_1d=True,
            )
            assert opt.mode == "muon+adamw", f"{arm_name}: expected muon+adamw, got {opt.mode}"
            model.train()

            losses_seen: List[float] = []
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                imgs = torch.randn(1, 3, 64, 96, device=device)
                tgt = torch.randint(4, min(64, uni_omr.VOCAB_SIZE), (1, 14), device=device)
                tgt[:, 0] = uni_omr.BOS_ID
                out = model(imgs, tgt[:, :-1], return_aux=False, return_mtp=True)
                loss = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)["loss"]
                assert torch.isfinite(loss), f"{arm_name} step {step}: non-finite loss {loss}"
                loss.backward()

                assert out["logits"].shape == (1, 13, uni_omr.VOCAB_SIZE)
                if arm_name == "diff_ca":
                    for i, layer in enumerate(model.dec_layers):
                        ca = layer.cross_attn
                        for pname in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"):
                            p = getattr(ca, pname)
                            assert p.grad is not None and torch.isfinite(p.grad).all(), f"layer {i} {pname} bad grad"
                            assert p.grad.abs().sum().item() > 0.0, f"layer {i} {pname} zero grad"
                        assert ca.subln.weight.grad is not None and ca.subln.weight.grad.abs().sum().item() > 0.0
                opt.step()
                losses_seen.append(round(float(loss.item()), 4))

            # Verify KV-cached generate() and dequantized INT4 parity
            pack_tiny = _pack_and_verify_hybrid_int4(
                model, arm_name, Path(tmpd) / f"{arm_name}_tiny.bin", INT4_GROUP_SIZE, tiny=True
            )
            dq_model = pack_tiny.pop("dequant_model")
            with torch.no_grad():
                mem, _, _ = dq_model.encode_image(imgs, return_aux=False)
                dq_logits, _, _ = dq_model.decode_tokens(
                    tgt[:, :-1], mem, return_mtp=False, return_aux=False, apply_tok_dropout=False
                )
                seq = dq_model.generate(imgs, max_len=4)
            assert dq_logits.shape == (1, 13, uni_omr.VOCAB_SIZE) and torch.isfinite(dq_logits).all()
            assert seq.shape[0] == 1 and seq.shape[1] >= 2

            tiny_results[arm_name] = {
                "opt_mode": opt.mode,
                "losses_2step": losses_seen,
                "tiny_params": sum(p.numel() for p in model.parameters()),
            }

        # Exact edge85m packed_int4_mb verification
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

    delta_mb = round(edge_mb["diff_ca"] - edge_mb["ctrl"], 3)
    print(f"[{EXP_CODE}][SMOKE] tiny ctrl:    {tiny_results['ctrl']}", flush=True)
    print(f"[{EXP_CODE}][SMOKE] tiny diff_ca: {tiny_results['diff_ca']}", flush=True)
    print(f"[{EXP_CODE}][SMOKE] edge85m packed_int4_mb: ctrl={edge_mb['ctrl']} MB, diff_ca={edge_mb['diff_ca']} MB (delta={delta_mb:+.3f} MB)", flush=True)
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

        # Startup auto-resume from HF Private Hub
        arm_results.update(_try_resume_from_hf(hf_token))

        train_abs, val_abs = _ensure_proxy_dataset(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)
        print(f"[{EXP_ID}] Loaded {len(train_pairs)} train / {len(val_pairs)} val", flush=True)

        for arm_name in ARMS:
            res = _train_arm(
                arm_name,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                device=device,
                use_bf16=use_bf16,
            )
            arm_results[arm_name] = res

            # 1-arm HF partial checkpoint sync
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
            if "ctrl" in arm_results and "diff_ca" in arm_results:
                c = arm_results["ctrl"]
                t = arm_results["diff_ca"]
                eval_payload["delta_diff_ca_minus_ctrl"] = {
                    "ema_val_ce": round(t["ema_val_ce"] - c["ema_val_ce"], 5),
                    "int4_dequant_val_ce": round(t["int4_dequant_val_ce"] - c["int4_dequant_val_ce"], 5),
                    "raw_val_ce": round(t["raw_val_ce"] - c["raw_val_ce"], 5),
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
