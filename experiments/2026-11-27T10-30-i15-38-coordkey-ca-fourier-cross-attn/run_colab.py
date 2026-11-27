#!/usr/bin/env python3
"""Colab H100 runner — Iter 15 Exp i15-38: Decoupled 2D Fourier CoordKey-CA Cross-Attention.

Experiment ID: 2026-11-27T10-30-i15-38-coordkey-ca-fourier-cross-attn
Controlled 2-arm weight-incompatible proxy on edge85m + ax0_gl6 (600 steps/arm, BF16, seed=42):
  - Arm 0 (ctrl):        edge85m + ax0_gl6 stock cross-attention (d_qk=64, d_v=64, additive pos2d).
  - Arm 1 (coordkey_ca): shared 2D Fourier CoordMLP (66->128->128) + dedicated 16-dim RMSNorm'd
                         spatial subspace concatenated onto Q & K (d_qk=64+16=80, d_v=64) in all
                         10 decoder cross-attention layers.

Measures per arm:
  - raw_val_ce, ema_val_ce, int4_dequant_val_ce
  - step_ms, train_elapsed_sec, peak_vram_mb
  - exact serialized packed_int4_mb (hard < 55.0 MB budget)

Uploads evals/<exp_id>.json and runs/<exp_id>.json to gregoreesmaa/sheet-music-converter-omr.
Security: reads HF_TOKEN from /tmp/.hf_token (chmod 600) or env, never prints it, always unlinks in finally.
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

EXP_ID = "2026-11-27T10-30-i15-38-coordkey-ca-fourier-cross-attn"
EXP_CODE = "i15-38"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
WORK_ROOT = Path("/vol/i15_38") if Path("/vol").is_dir() else Path("/content/i15_38")

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
# Self-Contained Experimental Modules: FourierCoordMLP + CoordKeyCrossAttention
# ---------------------------------------------------------------------------

def _build_coordkey_classes() -> Tuple[Any, Any]:
    """Factory returning (FourierCoordMLP, CoordKeyCrossAttention) using uni_omr_v2 primitives."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import uni_omr_v2

    class FourierCoordMLP(nn.Module):
        """Shared 2D Fourier coordinate encoder: (y_norm, x_norm) -> 66-dim -> 128 -> 128.

        66 input dims = 2 raw normalized coords + 16 log-spaced (sin, cos) pairs for y (32)
        + 16 log-spaced (sin, cos) pairs for x (32).
        """

        def __init__(self, n_freqs: int = 16, hidden_dim: int = 128, out_dim: int = 128) -> None:
            super().__init__()
            self.n_freqs = int(n_freqs)
            in_dim = 2 + 4 * self.n_freqs  # 2 + 64 = 66
            freqs = (2.0 ** torch.linspace(0.0, 6.0, self.n_freqs, dtype=torch.float32)) * math.pi
            self.register_buffer("freqs", freqs, persistent=False)
            self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
            self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)

        def forward(
            self,
            kept_idx: torch.Tensor,
            grid_hw: Tuple[int, int],
            n_bg_summary: int,
            dtype: torch.dtype,
            is_full_grid: bool = False,
        ) -> torch.Tensor:
            """Return shared spatial coordinate features ``(B, Lk, 128)``."""
            h_p, w_p = int(grid_hw[0]), int(grid_hw[1])
            device = kept_idx.device
            B = kept_idx.shape[0]

            if is_full_grid and n_bg_summary == 0:
                ys = torch.arange(h_p, device=device, dtype=torch.float32) / float(max(1, h_p - 1))
                xs = torch.arange(w_p, device=device, dtype=torch.float32) / float(max(1, w_p - 1))
                gy, gx = torch.meshgrid(ys, xs, indexing="ij")
                y_norm = gy.reshape(1, -1).expand(B, -1)
                x_norm = gx.reshape(1, -1).expand(B, -1)
            else:
                y_idx = torch.div(kept_idx, w_p, rounding_mode="floor").to(torch.float32)
                x_idx = (kept_idx % w_p).to(torch.float32)
                y_norm = y_idx / float(max(1, h_p - 1))
                x_norm = x_idx / float(max(1, w_p - 1))
                if n_bg_summary > 0:
                    cy = torch.full((B, n_bg_summary), 0.5, device=device, dtype=torch.float32)
                    cx = torch.full((B, n_bg_summary), 0.5, device=device, dtype=torch.float32)
                    y_norm = torch.cat([y_norm, cy], dim=1)
                    x_norm = torch.cat([x_norm, cx], dim=1)

            f = self.freqs if self.freqs.device == device else self.freqs.to(device)
            fy = y_norm.unsqueeze(-1) * f  # (B, Lk, 16)
            fx = x_norm.unsqueeze(-1) * f  # (B, Lk, 16)
            feats = torch.cat(
                [y_norm.unsqueeze(-1), x_norm.unsqueeze(-1), fy.sin(), fy.cos(), fx.sin(), fx.cos()],
                dim=-1,
            ).to(dtype)  # (B, Lk, 66)
            return self.fc2(F.silu(self.fc1(feats)))

    class CoordKeyCrossAttention(nn.Module):
        """GQA Cross-Attention with decoupled 16-dim RMSNorm'd 2D Fourier spatial QK subspace.

        Content Q/K/V have head_dim=64 (8 Q-heads, 2 KV-heads).
        Spatial Q/K have coord_dim=16 (8 Q-heads, 2 KV-heads), projected from decoder hidden
        state (q_coord_proj: d_model -> n_heads*16) and shared CoordMLP output (k_coord_proj:
        coord_in_dim -> n_kv_heads*16). Concatenated along head_dim -> d_qk=80, d_v=64.
        """

        def __init__(
            self,
            d_model: int,
            n_heads: int = 8,
            n_kv_heads: int = 2,
            coord_in_dim: int = 128,
            coord_dim: int = 16,
            use_native_gqa: bool = True,
        ) -> None:
            super().__init__()
            self.d_model = int(d_model)
            self.n_heads = int(n_heads)
            self.n_kv_heads = int(n_kv_heads)
            self.n_rep = self.n_heads // self.n_kv_heads
            self.head_dim = self.d_model // self.n_heads
            self.kv_dim = self.n_kv_heads * self.head_dim
            self.coord_dim = int(coord_dim)
            self.use_native_gqa = bool(use_native_gqa)
            self.use_rope = False

            # Content projections (exact match to GroupedQueryAttention)
            self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.k_proj = nn.Linear(self.d_model, self.kv_dim, bias=False)
            self.v_proj = nn.Linear(self.d_model, self.kv_dim, bias=False)
            self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            self.q_norm = uni_omr_v2.RMSNorm(self.head_dim)
            self.k_norm = uni_omr_v2.RMSNorm(self.head_dim)

            # Decoupled 16-dim RMSNorm'd spatial Q/K projections
            self.q_coord_proj = nn.Linear(self.d_model, self.n_heads * self.coord_dim, bias=False)
            self.k_coord_proj = nn.Linear(coord_in_dim, self.n_kv_heads * self.coord_dim, bias=False)
            self.q_coord_norm = uni_omr_v2.RMSNorm(self.coord_dim)
            self.k_coord_norm = uni_omr_v2.RMSNorm(self.coord_dim)

            # Ephemeral per-forward coordinate feature reference (set by encode_image / precompute)
            self._coord_feat: Optional[torch.Tensor] = None

        def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
            if self.n_rep == 1:
                return x
            B, n_kv, L, hd = x.shape
            return x.unsqueeze(2).expand(B, n_kv, self.n_rep, L, hd).reshape(B, self.n_heads, L, hd)

        def project_kv(
            self,
            kv_in: torch.Tensor,
            coord_feat: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            B, Lk, _ = kv_in.shape
            k_c = self.k_norm(self.k_proj(kv_in).view(B, Lk, self.n_kv_heads, self.head_dim)).transpose(1, 2)
            v = self.v_proj(kv_in).view(B, Lk, self.n_kv_heads, self.head_dim).transpose(1, 2)
            k_s = self.k_coord_norm(
                self.k_coord_proj(coord_feat.to(kv_in.dtype)).view(B, Lk, self.n_kv_heads, self.coord_dim)
            ).transpose(1, 2)
            k_cat = torch.cat([k_c, k_s], dim=-1)  # (B, n_kv_heads, Lk, head_dim + coord_dim = 80)
            return k_cat, v

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
            q_c = self.q_norm(self.q_proj(q_in).view(B, Lq, self.n_heads, self.head_dim)).transpose(1, 2)
            q_s = self.q_coord_norm(
                self.q_coord_proj(q_in).view(B, Lq, self.n_heads, self.coord_dim)
            ).transpose(1, 2)
            q_cat = torch.cat([q_c, q_s], dim=-1)  # (B, n_heads, Lq, 80)

            if past_kv is not None:
                k_cat, v = past_kv
            else:
                if kv_in is None:
                    raise ValueError("kv_in must be provided when past_kv is None")
                if self._coord_feat is None:
                    raise RuntimeError("CoordKeyCrossAttention._coord_feat was not populated before decode_tokens")
                k_cat, v = self.project_kv(kv_in, self._coord_feat)

            present_kv = (k_cat, v) if use_cache else None

            # Expand KV for broad SDPA compatibility with asymmetric d_qk=80, d_v=64
            k_exp = self._expand_kv(k_cat)
            v_exp = self._expand_kv(v)
            out = uni_omr_v2._sdpa(q_cat, k_exp, v_exp, attn_mask=attn_mask, is_causal=False, enable_gqa=False)
            out = out.transpose(1, 2).reshape(B, Lq, self.d_model)
            out = self.o_proj(out)
            if use_cache:
                return out, present_kv
            return out

    return FourierCoordMLP, CoordKeyCrossAttention


def apply_coordkey_ca(model: Any) -> Any:
    """Attach shared FourierCoordMLP and replace all 10 decoder cross_attn layers in-place."""
    FourierCoordMLP, CoordKeyCrossAttention = _build_coordkey_classes()
    c = model.cfg

    model.coord_mlp = FourierCoordMLP(n_freqs=16, hidden_dim=128, out_dim=128)
    for layer in model.dec_layers:
        layer.cross_attn = CoordKeyCrossAttention(
            d_model=c.d_model,
            n_heads=c.n_heads,
            n_kv_heads=c.n_kv_heads,
            coord_in_dim=128,
            coord_dim=16,
            use_native_gqa=True,
        )

    orig_encode_image = model.encode_image

    def _encode_image_with_coords(
        self: Any,
        images: Any,
        keep_ratio_override: Optional[float] = None,
        return_aux: Optional[bool] = None,
    ) -> Any:
        mem_levels, aux_enc, router_info = orig_encode_image(
            images, keep_ratio_override=keep_ratio_override, return_aux=return_aux
        )
        mem_final = mem_levels[-1]
        coord_feat = self.coord_mlp(
            router_info["kept_idx"],
            grid_hw=router_info["grid_hw"],
            n_bg_summary=int(router_info["n_bg_summary"]),
            dtype=mem_final.dtype,
            is_full_grid=bool(router_info.get("is_full_grid", False)),
        )
        self._last_coord_feat = coord_feat
        for layer in self.dec_layers:
            layer.cross_attn._coord_feat = coord_feat
        return mem_levels, aux_enc, router_info

    def _precompute_cross_kv_with_coords(self: Any, mem_levels: Any) -> Any:
        if isinstance(mem_levels, (tuple, list)):
            mem_post_merger, mem_global_mid, mem_final = mem_levels
            level_mems = (mem_post_merger, mem_global_mid)
        else:
            mem_final = mem_levels
            level_mems = None
        coord_feat = getattr(self, "_last_coord_feat", None)
        if coord_feat is None:
            raise RuntimeError("_last_coord_feat missing; call encode_image before precompute_cross_kv")
        past_kvs = []
        for idx, layer in enumerate(self.dec_layers):
            if level_mems is not None and idx < len(self.deepstack_fusions):
                layer_mem = self.deepstack_fusions[idx](mem_final, level_mems[idx])
            else:
                layer_mem = mem_final
            k_cat, v = layer.cross_attn.project_kv(layer_mem, coord_feat)
            past_kvs.append((None, (k_cat, v)))
        return past_kvs

    model.encode_image = types.MethodType(_encode_image_with_coords, model)
    model.precompute_cross_kv = types.MethodType(_precompute_cross_kv_with_coords, model)
    return model


def build_arm_model(arm_name: str, grad_checkpoint: bool = True) -> Any:
    """Build edge85m + ax0_gl6 model for either 'ctrl' or 'coordkey_ca'."""
    import uni_omr_v2

    cfg = uni_omr_v2.UniOMRv2Config.edge85m(
        n_axial_layers=0,
        n_global_layers=6,
        grad_checkpoint=grad_checkpoint,
    )
    model = uni_omr_v2.UniOMRv2Net(cfg)
    if arm_name == "coordkey_ca":
        apply_coordkey_ca(model)
    elif arm_name != "ctrl":
        raise ValueError(f"Unknown arm_name: {arm_name}")
    return model


# ---------------------------------------------------------------------------
# Infrastructure Helpers (HF Auth, Dataset, Optimizer, INT4 Packing, Eval)
# ---------------------------------------------------------------------------

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
    if not tok:
        raise RuntimeError("HF token missing: provide /tmp/.hf_token or HF_TOKEN env var")
    return tok


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


def _ensure_proxy_dataset(hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_dir = WORK_ROOT / "datasets" / "iter10_5_proxy_2k"
    train_abs = ds_dir / "proxy_train_abs.jsonl"
    val_abs = ds_dir / "proxy_val_abs.jsonl"
    if train_abs.is_file() and val_abs.is_file():
        return train_abs, val_abs

    ds_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR} from {HF_REPO_ID}...", flush=True)
    tar_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_PROXY_TAR,
        repo_type="dataset",
        token=hf_token,
        local_dir="/tmp/hf_dl",
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
) -> Dict[str, Any]:
    """Strip aux/MTP heads, pack QAT-INT4 Linear weights into 4-bit nibbles + FP16 scales,
    write exact binary file to disk, and dequantize onto a fresh model copy."""
    import torch
    import torch.nn as nn

    # Build fresh inference copy to preserve bound methods cleanly across deepcopy
    infer_model = build_arm_model(arm_name, grad_checkpoint=False).cpu().eval()
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
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    weights = [p[3] for p in train_pairs]
    rng = random.Random(SEED)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t_arm0 = time.perf_counter()
    t_last_save = time.time()
    loss_hist: List[float] = []
    step_ms_hist: List[float] = []

    model.train()
    for step in range(1, TRAIN_STEPS + 1):
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
            torch.save({"step": step, "model_state_dict": model.state_dict(), "ema_state_dict": ema_state}, tmp_ck)
            os.replace(tmp_ck, arm_dir / "latest.pt")
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

    packed_info = _pack_and_verify_hybrid_int4(
        model, arm_name, arm_dir / f"{arm_name}_hybrid_int4.bin", INT4_GROUP_SIZE
    )
    dq_model = packed_info.pop("dequant_model").to(device)
    int4_dequant_val_ce = _eval_val_ce(dq_model, val_pairs, device, use_bf16)

    steady_ms = step_ms_hist[10:] if len(step_ms_hist) > 10 else step_ms_hist
    result = {
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
# CPU 1-Step Smoke Test
# ---------------------------------------------------------------------------

def run_smoke_test() -> int:
    """1-step tiny CPU forward + backward + KV-cache + packed_int4_mb verification for both arms."""
    _locate_mvd()
    import tempfile
    import torch
    import uni_omr
    device = torch.device("cpu")
    _seed_everything(SEED)

    results: Dict[str, Dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as tmpd:
        for arm_name in ("ctrl", "coordkey_ca"):
            model = build_arm_model(arm_name, grad_checkpoint=False).to(device)
            model.train()
            imgs = torch.randn(1, 3, 64, 96, device=device)
            tgt = torch.randint(4, min(64, uni_omr.VOCAB_SIZE), (1, 12), device=device)
            tgt[:, 0] = uni_omr.BOS_ID

            out = model(imgs, tgt[:, :-1], return_aux=False, return_mtp=True)
            losses = model.compute_loss(out, tgt[:, 1:], mtp_loss_weight=0.25, saliency_loss_weight=0.10)
            loss = losses["loss"]
            assert torch.isfinite(loss), f"{arm_name}: non-finite loss {loss}"
            loss.backward()

            assert out["logits"].shape == (1, 11, uni_omr.VOCAB_SIZE), f"bad logits shape {out['logits'].shape}"
            if arm_name == "coordkey_ca":
                coord_mlp_grads = [
                    p.grad.abs().sum().item()
                    for p in model.coord_mlp.parameters()
                    if p.grad is not None
                ]
                assert len(coord_mlp_grads) == 4 and sum(coord_mlp_grads) > 0.0, "zero coord_mlp grads"
                for i, layer in enumerate(model.dec_layers):
                    assert layer.cross_attn.q_coord_proj.weight.grad is not None, f"layer {i} missing q_coord grad"
                    assert layer.cross_attn.k_coord_proj.weight.grad is not None, f"layer {i} missing k_coord grad"
                    assert layer.cross_attn.q_coord_proj.weight.grad.abs().sum().item() > 0.0
                    assert layer.cross_attn.k_coord_proj.weight.grad.abs().sum().item() > 0.0

            # Also verify KV-cached generate() and dequantized INT4 forward
            pack = _pack_and_verify_hybrid_int4(model, arm_name, Path(tmpd) / f"{arm_name}.bin", INT4_GROUP_SIZE)
            dq_model = pack.pop("dequant_model")
            with torch.no_grad():
                mem, _, _ = dq_model.encode_image(imgs, return_aux=False)
                dq_logits, _, _ = dq_model.decode_tokens(
                    tgt[:, :-1], mem, return_mtp=False, return_aux=False, apply_tok_dropout=False
                )
                seq = dq_model.generate(imgs, max_len=4)
            assert dq_logits.shape == out["logits"].shape and torch.isfinite(dq_logits).all()
            assert seq.shape[0] == 1 and seq.shape[1] >= 2

            results[arm_name] = {
                "total_params": sum(p.numel() for p in model.parameters()),
                "packed_int4_mb": pack["packed_int4_mb"],
                "passes_55mb_budget": pack["passes_55mb_budget"],
                "loss": round(float(loss.item()), 4),
            }

    delta_mb = round(results["coordkey_ca"]["packed_int4_mb"] - results["ctrl"]["packed_int4_mb"], 3)
    print(f"[{EXP_CODE}][SMOKE] ctrl:        {results['ctrl']}", flush=True)
    print(f"[{EXP_CODE}][SMOKE] coordkey_ca: {results['coordkey_ca']}", flush=True)
    print(f"[{EXP_CODE}][SMOKE] packed_int4_mb delta (coordkey - ctrl): {delta_mb:+.3f} MB", flush=True)
    assert results["coordkey_ca"]["passes_55mb_budget"], "coordkey_ca exceeded 55 MB budget!"
    print(f"[{EXP_CODE}][SMOKE] PASS!", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{EXP_ID} runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-step tiny CPU smoke test and exit")
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

        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        print(f"[{EXP_ID}] Device={gpu_name} | bf16={use_bf16} | torch={torch.__version__}", flush=True)

        train_abs, val_abs = _ensure_proxy_dataset(hf_token)
        train_pairs = _load_manifest_pairs(train_abs)
        val_pairs = _load_manifest_pairs(val_abs)
        print(f"[{EXP_ID}] Loaded {len(train_pairs)} train / {len(val_pairs)} val", flush=True)

        for arm_name in ("ctrl", "coordkey_ca"):
            res = _train_arm(
                arm_name,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                device=device,
                use_bf16=use_bf16,
            )
            arm_results[arm_name] = res

            eval_payload: Dict[str, Any] = {
                "exp_id": EXP_ID,
                "exp_code": EXP_CODE,
                "status": "RUNNING" if len(arm_results) < 2 else "COMPLETED",
                "started_at": started_at,
                "updated_at": _utc_now(),
                "gpu": gpu_name,
                "bf16": use_bf16,
                "steps_per_arm": TRAIN_STEPS,
                "seed": SEED,
                "arms": arm_results,
            }
            if "ctrl" in arm_results and "coordkey_ca" in arm_results:
                c = arm_results["ctrl"]
                t = arm_results["coordkey_ca"]
                eval_payload["delta_coordkey_minus_ctrl"] = {
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
