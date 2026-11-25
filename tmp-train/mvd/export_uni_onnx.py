#!/usr/bin/env python3
"""Export UniOMR checkpoint to browser-ready ONNX models (FP32 + INT8).

Produces two ONNX models in ``--out-dir`` (default: ``web/models``):
  1. ``uni_omr_encoder.onnx``:
       Input:  ``image`` float32 ``(1, 3, H, W)`` in ``[0, 1]`` (H, W multiples of 16)
       Outputs:
         - ``cross_k_0`` .. ``cross_k_{L-1}`` float32 ``(1, n_heads, N_mem, head_dim)``
         - ``cross_v_0`` .. ``cross_v_{L-1}`` float32 ``(1, n_heads, N_mem, head_dim)``
     (Pre-projecting cross-attention K/V inside the encoder avoids recomputing
     them on every autoregressive decode step — ~5x faster in the browser.)
  2. ``uni_omr_decoder_step.onnx``:
       Inputs:
         - ``token_id`` int64 ``(1, 1)``
         - ``step_pos`` int64 ``(1,)``
         - ``past_k_0`` .. ``past_k_{L-1}`` float32 ``(1, n_heads, past_len, head_dim)``
         - ``past_v_0`` .. ``past_v_{L-1}`` float32 ``(1, n_heads, past_len, head_dim)``
         - ``cross_k_0`` .. ``cross_k_{L-1}`` float32 ``(1, n_heads, N_mem, head_dim)``
         - ``cross_v_0`` .. ``cross_v_{L-1}`` float32 ``(1, n_heads, N_mem, head_dim)``
       Outputs:
         - ``logits`` float32 ``(1, vocab_size)``
         - ``present_k_0`` .. ``present_k_{L-1}`` float32 ``(1, n_heads, past_len + 1, head_dim)``
         - ``present_v_0`` .. ``present_v_{L-1}`` float32 ``(1, n_heads, past_len + 1, head_dim)``
  3. ``vocab.json``:
       Vocab list, special token IDs, and model metadata for the web frontend.

Runs strictly on CPU (never touches CUDA).
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

# Force CPU-only execution before importing torch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import data  # noqa: E402
import uni_omr  # noqa: E402


DEFAULT_CKPT = Path(r"D:\tmp\mvd-runs\uni-omr-iter7plus-v1-part2\uni_omr_best.pt")
DEFAULT_OUT_DIR = REPO_ROOT / "web" / "models"
DEFAULT_TEST_IMAGE = (
    REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "clean" / "repeats-melody.png"
)


# ---------------------------------------------------------------------------
# ONNX-Safe Primitives (exact math parity with uni_omr.py, opset 17 friendly)
# ---------------------------------------------------------------------------

def _rms_norm_onnx(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Explicit RMSNorm using standard opset-17 ops (avoids custom RMSNorm op)."""
    norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * norm * weight


def _manual_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Explicit scaled dot-product attention (opset-17 clean, no custom op)."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _stride_hw(backbone_stride: Any) -> Tuple[int, int]:
    if backbone_stride == "4x8":
        return 4, 8
    s = int(backbone_stride)
    return s, s


# ---------------------------------------------------------------------------
# 1. ONNX Encoder Wrapper (Backbone + ONNX-safe InkTokenRouter + Encoder + Cross-KV)
# ---------------------------------------------------------------------------

class UniOMREncoderONNX(nn.Module):
    """ONNX-exportable UniOMR encoder with pre-projected decoder cross-attention K/V."""

    def __init__(self, model: uni_omr.UniOMRNet) -> None:
        super().__init__()
        self.cfg = model.cfg
        self.backbone = model.backbone
        self.router = model.router
        self.rope2d = model.rope2d
        self.pos2d_proj = model.pos2d_proj
        self.enc_layers = model.enc_layers
        self.enc_norm = model.enc_norm

        # Store references to decoder cross-attention K/V projections so they
        # are computed once per image instead of once per decode token!
        self.cross_k_projs = nn.ModuleList(
            [layer.cross_attn.k_proj for layer in model.dec_layers]
        )
        self.cross_v_projs = nn.ModuleList(
            [layer.cross_attn.v_proj for layer in model.dec_layers]
        )

        self.d_model = int(model.cfg.d_model)
        self.n_heads = int(model.cfg.n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.n_dec_layers = int(model.cfg.n_dec_layers)
        self.stride_h, self.stride_w = _stride_hw(model.cfg.backbone_stride)

        # Precompute 2D sinusoidal frequencies buffer for _sinusoidal_pos2d.
        q = self.d_model // 4
        freqs = torch.exp(
            torch.arange(0, q, dtype=torch.float32) * (-math.log(10000.0) / max(1, q))
        )
        self.register_buffer("pos2d_freqs", freqs, persistent=False)

    def _compute_ink_and_paper_gate_onnx(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact ONNX-safe equivalent of InkTokenRouter._compute_ink_and_paper_gate.

        Uses fixed-stride max_pool2d / avg_pool2d matching the backbone stride
        (bit-identical to adaptive_*_pool2d when H % stride_h == 0, W % stride_w == 0).
        """
        luma = (
            0.299 * images[:, 0:1]
            + 0.587 * images[:, 1:2]
            + 0.114 * images[:, 2:3]
        )
        local_bg = F.avg_pool2d(
            F.avg_pool2d(
                luma,
                kernel_size=(1, 31),
                stride=1,
                padding=(0, 15),
                count_include_pad=False,
            ),
            kernel_size=(31, 1),
            stride=1,
            padding=(15, 0),
            count_include_pad=False,
        )
        paper_mask = (local_bg > 0.35).to(torch.float32)
        local_ink = (
            (local_bg - luma) / local_bg.clamp(min=0.15)
        ).clamp(0.0, 1.0) * paper_mask

        ks = (self.stride_h, self.stride_w)
        ink_max = F.max_pool2d(local_ink, kernel_size=ks, stride=ks)
        ink_avg = F.avg_pool2d(local_ink, kernel_size=ks, stride=ks)
        local_ink_grid = 0.5 * ink_max + 0.5 * ink_avg

        local_bg_grid = F.avg_pool2d(local_bg, kernel_size=ks, stride=ks)
        paper_mask_grid = local_bg_grid > 0.35
        paper_gate = torch.where(
            paper_mask_grid,
            torch.ones_like(local_bg_grid),
            torch.full_like(local_bg_grid, 0.05),
        )
        return local_ink_grid, paper_gate

    def _route_onnx(
        self,
        feat_2d: torch.Tensor,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ONNX-safe InkTokenRouter forward (supports dynamic H, W and dynamic_routing)."""
        r = self.router
        shape_t = torch._shape_as_tensor(feat_2d)
        h_p_t = shape_t[2]  # scalar int64 tensor
        w_p_t = shape_t[3]  # scalar int64 tensor
        n_t = h_p_t * w_p_t  # scalar int64 tensor

        saliency_logits = r.saliency_net(feat_2d)  # (1, 1, H', W')
        saliency_prob = torch.sigmoid(saliency_logits)

        local_ink_grid, paper_gate = self._compute_ink_and_paper_gate_onnx(images)
        saliency_prob = saliency_prob * paper_gate

        if r.use_ink_prior:
            flat = local_ink_grid.flatten(1)  # (1, N)
            # Exact 0.99 quantile via sort + linear interpolation (ONNX-safe).
            flat_sorted, _ = torch.sort(flat, dim=-1)
            n_m1 = (n_t - 1).to(torch.float32)
            q_pos = n_m1 * 0.99
            lo_idx = torch.floor(q_pos).to(torch.long).view(1, 1)
            hi_idx = torch.ceil(q_pos).to(torch.long).view(1, 1)
            frac = (q_pos - lo_idx.to(torch.float32)).view(1, 1)
            v_lo = torch.gather(flat_sorted, 1, lo_idx)
            v_hi = torch.gather(flat_sorted, 1, hi_idx)
            denom = (v_lo + frac * (v_hi - v_lo)).clamp(min=1e-4).view(1, 1, 1, 1)
            local_ink_norm = (local_ink_grid / denom).clamp(0.0, 1.0)
            score = saliency_prob + torch.sigmoid(r.ink_prior_scale) * local_ink_norm
        else:
            score = saliency_prob

        score_flat = score.flatten(1)  # (1, N)
        tokens_flat = feat_2d.flatten(2).transpose(1, 2)  # (1, N, D)

        # Build dynamic 2D coordinate grid (1, N, 2).
        ys = torch.arange(feat_2d.shape[2], device=feat_2d.device, dtype=torch.float32)
        xs = torch.arange(feat_2d.shape[3], device=feat_2d.device, dtype=torch.float32)
        grid_y = ys.unsqueeze(1).expand(feat_2d.shape[2], feat_2d.shape[3])
        grid_x = xs.unsqueeze(0).expand(feat_2d.shape[2], feat_2d.shape[3])
        grid_coords = torch.stack(
            [grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1
        ).unsqueeze(0)  # (1, N, 2)

        # Dynamic or fixed k selection.
        if r.dynamic_routing:
            score_sorted, _ = torch.sort(score_flat, dim=1)
            med_idx = ((n_t - 1) // 2).view(1, 1)
            med = torch.gather(score_sorted, 1, med_idx)
            above = (score_flat > (med + r.saliency_thr)).to(torch.long).sum(dim=1)
            k_min = torch.ceil(n_t.to(torch.float32) * r.min_keep_ratio).to(torch.long)
            k_min = torch.clamp(k_min, min=1)
            k_max = torch.ceil(n_t.to(torch.float32) * r.max_keep_ratio).to(torch.long)
            k_max = torch.minimum(n_t, torch.maximum(k_min, k_max))
            k_t = torch.minimum(k_max, torch.maximum(k_min, above[0]))
        else:
            k_t = torch.ceil(n_t.to(torch.float32) * r.keep_ratio).to(torch.long)
            k_t = torch.minimum(n_t, torch.clamp(k_t, min=1))

        # Top-k selection, then sort ascending for reading order.
        _, topk_idx = torch.topk(score_flat, k=k_t, dim=1, largest=True, sorted=False)
        kept_idx, _ = torch.sort(topk_idx, dim=1)  # (1, k)

        kept_scores = torch.gather(score_flat, 1, kept_idx)
        kept_gate = (0.5 + 0.5 * torch.sigmoid(kept_scores)).unsqueeze(-1)

        idx_feat = kept_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
        kept_tokens = torch.gather(tokens_flat, 1, idx_feat) * kept_gate

        idx_coords = kept_idx.unsqueeze(-1).expand(-1, -1, 2)
        kept_coords = torch.gather(grid_coords, 1, idx_coords)

        # Pool background tokens into n_bg_summary tokens.
        kept_mask = torch.zeros_like(score_flat, dtype=torch.bool).scatter(
            1, kept_idx, torch.ones_like(kept_idx, dtype=torch.bool)
        )
        bg_q = r.bg_queries  # (1, M, D)
        bg_k = r.bg_key_proj(tokens_flat)  # (1, N, D)
        bg_v = r.bg_val_proj(tokens_flat)  # (1, N, D)
        bg_logits = torch.bmm(bg_q, bg_k.transpose(1, 2)) / math.sqrt(self.d_model)
        bg_logits = bg_logits - score_flat.unsqueeze(1)
        bg_logits = torch.where(
            kept_mask.unsqueeze(1),
            torch.full_like(bg_logits, -1e4),
            bg_logits,
        )
        bg_attn = F.softmax(bg_logits, dim=-1)
        bg_tokens = _rms_norm_onnx(
            torch.bmm(bg_attn, bg_v), r.bg_out_norm.weight, r.bg_out_norm.eps
        )

        cy = (h_p_t.to(torch.float32) - 1.0) * 0.5
        cx = (w_p_t.to(torch.float32) - 1.0) * 0.5
        center_coord = torch.stack([cy, cx]).view(1, 1, 2).expand(1, r.n_bg_summary, 2)

        out_tokens = torch.cat([kept_tokens, bg_tokens], dim=1)
        out_coords = torch.cat([kept_coords, center_coord], dim=1)
        return out_tokens, out_coords, h_p_t, w_p_t

    def _sinusoidal_pos2d_onnx(
        self,
        coords: torch.Tensor,
        h_p_t: torch.Tensor,
        w_p_t: torch.Tensor,
    ) -> torch.Tensor:
        y_denom = torch.clamp(h_p_t - 1, min=1).to(torch.float32)
        x_denom = torch.clamp(w_p_t - 1, min=1).to(torch.float32)
        y = (coords[..., 0] / y_denom) * 128.0
        x = (coords[..., 1] / x_denom) * 128.0
        fy = y.unsqueeze(-1) * self.pos2d_freqs
        fx = x.unsqueeze(-1) * self.pos2d_freqs
        return torch.cat([fy.sin(), fy.cos(), fx.sin(), fx.cos()], dim=-1)

    def _enc_layer_onnx(
        self,
        layer: uni_omr.EncoderLayer,
        x: torch.Tensor,
        cos2d: torch.Tensor,
        sin2d: torch.Tensor,
    ) -> torch.Tensor:
        h = _rms_norm_onnx(x, layer.norm1.weight, layer.norm1.eps)
        sa = layer.self_attn
        q = sa.q_proj(h).view(1, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = sa.k_proj(h).view(1, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = sa.v_proj(h).view(1, -1, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos2d, sin2d)
        attn_out = _manual_sdpa(q, k, v).transpose(1, 2).contiguous().view(1, -1, self.d_model)
        x = x + sa.o_proj(attn_out)

        h2 = _rms_norm_onnx(x, layer.norm2.weight, layer.norm2.eps)
        gate, up = layer.ffn.w_gate_up(h2).chunk(2, dim=-1)
        x = x + layer.ffn.w_down(F.silu(gate) * up)
        return x

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        feat_2d = self.backbone(image)
        routed_tokens, routed_coords, h_p_t, w_p_t = self._route_onnx(feat_2d, image)

        pos2d = self.pos2d_proj(
            self._sinusoidal_pos2d_onnx(routed_coords, h_p_t, w_p_t)
        )
        cos2d, sin2d = self.rope2d(routed_coords, dtype=routed_tokens.dtype)

        mem = routed_tokens + pos2d
        for layer in self.enc_layers:
            mem = self._enc_layer_onnx(layer, mem, cos2d, sin2d)
        mem = _rms_norm_onnx(mem, self.enc_norm.weight, self.enc_norm.eps) + pos2d

        # Pre-project cross-attention K and V for every decoder layer.
        cross_ks: List[torch.Tensor] = []
        cross_vs: List[torch.Tensor] = []
        for k_proj, v_proj in zip(self.cross_k_projs, self.cross_v_projs):
            ck = k_proj(mem).view(1, -1, self.n_heads, self.head_dim).transpose(1, 2)
            cv = v_proj(mem).view(1, -1, self.n_heads, self.head_dim).transpose(1, 2)
            cross_ks.append(ck)
            cross_vs.append(cv)

        return tuple(cross_ks + cross_vs)


# ---------------------------------------------------------------------------
# 2. ONNX Single-Step KV-Cached Decoder Wrapper
# ---------------------------------------------------------------------------

class UniOMRDecoderStepONNX(nn.Module):
    """Fast single-step KV-cached decoder accepting pre-projected cross-attn K/V."""

    def __init__(self, model: uni_omr.UniOMRNet) -> None:
        super().__init__()
        self.d_model = int(model.cfg.d_model)
        self.n_heads = int(model.cfg.n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.n_dec_layers = int(model.cfg.n_dec_layers)

        self.tok_emb = model.tok_emb
        self.dec_layers = model.dec_layers
        self.dec_norm = model.dec_norm
        self.lm_head = model.lm_head
        self.register_buffer(
            "rope1d_inv_freq",
            model.rope1d.inv_freq.detach().clone().to(torch.float32),
            persistent=False,
        )

    def forward(
        self,
        token_id: torch.Tensor,
        step_pos: torch.Tensor,
        *kv_tensors: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        L = self.n_dec_layers
        past_ks = kv_tensors[0:L]
        past_vs = kv_tensors[L : 2 * L]
        cross_ks = kv_tensors[2 * L : 3 * L]
        cross_vs = kv_tensors[3 * L : 4 * L]

        x = self.tok_emb(token_id)  # (1, 1, D)

        # 1D RoPE for single step position.
        pos = step_pos.to(torch.float32).view(1, 1)  # (1, 1)
        freqs = pos * self.rope1d_inv_freq.unsqueeze(0)  # (1, head_dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (1, head_dim)
        cos1d = emb.cos().view(1, 1, 1, self.head_dim)
        sin1d = emb.sin().view(1, 1, 1, self.head_dim)

        present_ks: List[torch.Tensor] = []
        present_vs: List[torch.Tensor] = []

        for idx, layer in enumerate(self.dec_layers):
            # 1. Causal self-attention with KV cache (Lq=1 -> attends to all past + cur).
            h = _rms_norm_onnx(x, layer.norm1.weight, layer.norm1.eps)
            sa = layer.self_attn
            q = sa.q_proj(h).view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)
            k = sa.k_proj(h).view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)
            v = sa.v_proj(h).view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)
            q, k = _apply_rope(q, k, cos1d, sin1d)

            k_cat = torch.cat([past_ks[idx], k], dim=2)
            v_cat = torch.cat([past_vs[idx], v], dim=2)
            present_ks.append(k_cat)
            present_vs.append(v_cat)

            sa_out = (
                _manual_sdpa(q, k_cat, v_cat)
                .transpose(1, 2)
                .contiguous()
                .view(1, 1, self.d_model)
            )
            x = x + sa.o_proj(sa_out)

            # 2. Cross-attention using pre-projected K/V from encoder.
            h2 = _rms_norm_onnx(x, layer.norm2.weight, layer.norm2.eps)
            ca = layer.cross_attn
            cq = ca.q_proj(h2).view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)
            ca_out = (
                _manual_sdpa(cq, cross_ks[idx], cross_vs[idx])
                .transpose(1, 2)
                .contiguous()
                .view(1, 1, self.d_model)
            )
            x = x + ca.o_proj(ca_out)

            # 3. SwiGLU FFN.
            h3 = _rms_norm_onnx(x, layer.norm3.weight, layer.norm3.eps)
            gate, up = layer.ffn.w_gate_up(h3).chunk(2, dim=-1)
            x = x + layer.ffn.w_down(F.silu(gate) * up)

        h_out = _rms_norm_onnx(x, self.dec_norm.weight, self.dec_norm.eps)
        logits = self.lm_head(h_out).squeeze(1)  # (1, vocab_size)
        return (logits, *present_ks, *present_vs)


# ---------------------------------------------------------------------------
# Checkpoint Loading, Export, Quantization, and Parity Verification
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: Path) -> Tuple[uni_omr.UniOMRNet, uni_omr.UniOMRConfig, Dict[str, Any]]:
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg_dict = ck.get("config", {}) if isinstance(ck, dict) else {}
    valid_fields = {f.name for f in fields(uni_omr.UniOMRConfig)}
    cfg = uni_omr.UniOMRConfig(**{k: v for k, v in cfg_dict.items() if k in valid_fields})
    model = uni_omr.UniOMRNet(cfg)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    inc = model.load_state_dict(sd, strict=False)
    model.eval()
    return model, cfg, {
        "step": ck.get("step") if isinstance(ck, dict) else None,
        "val_ser": ck.get("val_ser") if isinstance(ck, dict) else None,
        "vocab": ck.get("vocab") if isinstance(ck, dict) else None,
        "missing_keys": list(inc.missing_keys),
        "unexpected_keys": list(inc.unexpected_keys),
    }


def build_vocab_json(cfg: uni_omr.UniOMRConfig, ckpt_vocab: Optional[List[str]]) -> Dict[str, Any]:
    vocab_size = int(cfg.vocab_size)
    n_base = vocab_size - 4
    base_vocab = list(ckpt_vocab) if ckpt_vocab and len(ckpt_vocab) == n_base else list(data.VOCAB[:n_base])
    pad_id = vocab_size - 4
    bos_id = vocab_size - 3
    eos_id = vocab_size - 2
    sys_id = vocab_size - 1
    full_vocab = base_vocab + ["<pad>", "<bos>", "<eos>", uni_omr.SYS_TOKEN]
    assert len(full_vocab) == vocab_size
    return {
        "vocab": full_vocab,
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "sys_id": sys_id,
        "vocab_size": vocab_size,
        "max_dim": 1280,
        "d_model": int(cfg.d_model),
        "n_enc_layers": int(cfg.n_enc_layers),
        "n_dec_layers": int(cfg.n_dec_layers),
        "n_heads": int(cfg.n_heads),
        "head_dim": int(cfg.d_model) // int(cfg.n_heads),
        "backbone_stride": cfg.backbone_stride,
    }


def export_models(
    model: uni_omr.UniOMRNet,
    cfg: uni_omr.UniOMRConfig,
    out_dir: Path,
    opset: int = 17,
    quantize: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_dec = int(cfg.n_dec_layers)
    n_heads = int(cfg.n_heads)
    head_dim = int(cfg.d_model) // n_heads

    enc_wrapper = UniOMREncoderONNX(model).eval()
    dec_wrapper = UniOMRDecoderStepONNX(model).eval()

    enc_fp32_path = out_dir / "uni_omr_encoder_fp32.onnx"
    dec_fp32_path = out_dir / "uni_omr_decoder_step_fp32.onnx"
    enc_final_path = out_dir / "uni_omr_encoder.onnx"
    dec_final_path = out_dir / "uni_omr_decoder_step.onnx"

    # --- 1. Export Encoder ---
    dummy_img = torch.rand(1, 3, 256, 512, dtype=torch.float32)
    enc_out_names = (
        [f"cross_k_{i}" for i in range(n_dec)]
        + [f"cross_v_{i}" for i in range(n_dec)]
    )
    enc_dyn_axes: Dict[str, Dict[int, str]] = {"image": {2: "height", 3: "width"}}
    for name in enc_out_names:
        enc_dyn_axes[name] = {2: "n_mem"}

    print(f"[export] Exporting encoder (opset={opset}) -> {enc_fp32_path}", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            enc_wrapper,
            (dummy_img,),
            str(enc_fp32_path),
            input_names=["image"],
            output_names=enc_out_names,
            dynamic_axes=enc_dyn_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    print(
        f"[export] Encoder FP32 exported in {time.perf_counter() - t0:.1f}s "
        f"({enc_fp32_path.stat().st_size / (1024 * 1024):.2f} MB)",
        flush=True,
    )

    # --- 2. Export Single-Step KV-Cached Decoder ---
    dummy_tok = torch.tensor([[int(cfg.vocab_size) - 3]], dtype=torch.long)
    dummy_pos = torch.tensor([2], dtype=torch.long)
    dummy_past_ks = [torch.zeros(1, n_heads, 2, head_dim, dtype=torch.float32) for _ in range(n_dec)]
    dummy_past_vs = [torch.zeros(1, n_heads, 2, head_dim, dtype=torch.float32) for _ in range(n_dec)]
    dummy_cross_ks = [torch.zeros(1, n_heads, 64, head_dim, dtype=torch.float32) for _ in range(n_dec)]
    dummy_cross_vs = [torch.zeros(1, n_heads, 64, head_dim, dtype=torch.float32) for _ in range(n_dec)]

    dec_in_names = (
        ["token_id", "step_pos"]
        + [f"past_k_{i}" for i in range(n_dec)]
        + [f"past_v_{i}" for i in range(n_dec)]
        + [f"cross_k_{i}" for i in range(n_dec)]
        + [f"cross_v_{i}" for i in range(n_dec)]
    )
    dec_out_names = (
        ["logits"]
        + [f"present_k_{i}" for i in range(n_dec)]
        + [f"present_v_{i}" for i in range(n_dec)]
    )
    dec_dyn_axes: Dict[str, Dict[int, str]] = {}
    for i in range(n_dec):
        dec_dyn_axes[f"past_k_{i}"] = {2: "past_len"}
        dec_dyn_axes[f"past_v_{i}"] = {2: "past_len"}
        dec_dyn_axes[f"cross_k_{i}"] = {2: "n_mem"}
        dec_dyn_axes[f"cross_v_{i}"] = {2: "n_mem"}
        dec_dyn_axes[f"present_k_{i}"] = {2: "total_len"}
        dec_dyn_axes[f"present_v_{i}"] = {2: "total_len"}

    print(f"[export] Exporting decoder step (opset={opset}) -> {dec_fp32_path}", flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            dec_wrapper,
            (dummy_tok, dummy_pos, *dummy_past_ks, *dummy_past_vs, *dummy_cross_ks, *dummy_cross_vs),
            str(dec_fp32_path),
            input_names=dec_in_names,
            output_names=dec_out_names,
            dynamic_axes=dec_dyn_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    print(
        f"[export] Decoder FP32 exported in {time.perf_counter() - t0:.1f}s "
        f"({dec_fp32_path.stat().st_size / (1024 * 1024):.2f} MB)",
        flush=True,
    )

    # --- 3. Quantize to INT8 (dynamic QUInt8 weights for MatMul/Gemm/Conv) ---
    quantized = False
    if quantize:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            print("[quantize] Applying dynamic INT8 (QUInt8) quantization...", flush=True)
            t0 = time.perf_counter()
            quantize_dynamic(
                model_input=str(enc_fp32_path),
                model_output=str(enc_final_path),
                weight_type=QuantType.QUInt8,
                op_types_to_quantize=["MatMul", "Gemm", "Conv"],
            )
            quantize_dynamic(
                model_input=str(dec_fp32_path),
                model_output=str(dec_final_path),
                weight_type=QuantType.QUInt8,
                op_types_to_quantize=["MatMul", "Gemm"],
            )
            quantized = True
            print(f"[quantize] Done in {time.perf_counter() - t0:.1f}s", flush=True)
        except Exception as exc:
            print(f"[quantize] WARNING: quantization failed ({exc}); keeping FP32 models.", flush=True)

    if not quantized:
        enc_fp32_path.replace(enc_final_path)
        dec_fp32_path.replace(dec_final_path)
    else:
        # Remove temporary FP32 intermediates so web/models/ stays clean.
        enc_fp32_path.unlink(missing_ok=True)
        dec_fp32_path.unlink(missing_ok=True)

    return {
        "encoder_path": enc_final_path,
        "decoder_path": dec_final_path,
        "quantized": quantized,
        "enc_out_names": enc_out_names,
        "dec_in_names": dec_in_names,
        "dec_out_names": dec_out_names,
    }


def verify_parity(
    model: uni_omr.UniOMRNet,
    cfg: uni_omr.UniOMRConfig,
    enc_path: Path,
    dec_path: Path,
    image_path: Path,
    max_dim: int = 1280,
    n_tokens: int = 30,
) -> Dict[str, Any]:
    """Compare PyTorch UniOMRNet vs ONNX Runtime greedy tokens on a real image."""
    import onnxruntime as ort

    with Image.open(image_path) as raw_im:
        pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), max_dim)
    img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0)  # (1, 3, H, W)
    _, _, H, W = img_t.shape
    print(f"[parity] Image: {image_path.name} -> resized ({H}x{W})", flush=True)

    vocab_size = int(cfg.vocab_size)
    bos_id = vocab_size - 3
    eos_id = vocab_size - 2
    n_dec = int(cfg.n_dec_layers)
    n_heads = int(cfg.n_heads)
    head_dim = int(cfg.d_model) // n_heads

    # 1. PyTorch greedy reference
    t0 = time.perf_counter()
    with torch.no_grad():
        pt_mem, _, pt_rinfo = model.encode_image(img_t, return_aux=False)
        pt_enc_ms = (time.perf_counter() - t0) * 1000.0
        pt_tokens: List[int] = []
        pt_past = None
        cur_id = torch.tensor([[bos_id]], dtype=torch.long)
        for step in range(n_tokens):
            logits, _, pt_past = model.decode_tokens(
                cur_id,
                memory=pt_mem if pt_past is None else None,
                return_mtp=False,
                past_kvs=pt_past,
                use_cache=True,
            )
            nxt = int(logits[0, -1].argmax().item())
            pt_tokens.append(nxt)
            if nxt == eos_id:
                break
            cur_id = torch.tensor([[nxt]], dtype=torch.long)

    # 2. ONNX Runtime CPU inference
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    enc_sess = ort.InferenceSession(str(enc_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(dec_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])

    t0 = time.perf_counter()
    enc_outs = enc_sess.run(None, {"image": img_t.numpy()})
    ort_enc_ms = (time.perf_counter() - t0) * 1000.0

    cross_ks = enc_outs[:n_dec]
    cross_vs = enc_outs[n_dec : 2 * n_dec]
    n_mem = int(cross_ks[0].shape[2])

    # Initialize empty past KV caches (past_len = 0).
    past_ks = [np.zeros((1, n_heads, 0, head_dim), dtype=np.float32) for _ in range(n_dec)]
    past_vs = [np.zeros((1, n_heads, 0, head_dim), dtype=np.float32) for _ in range(n_dec)]

    ort_tokens: List[int] = []
    cur_tok = bos_id
    t_dec0 = time.perf_counter()
    for step in range(n_tokens):
        feed: Dict[str, np.ndarray] = {
            "token_id": np.array([[cur_tok]], dtype=np.int64),
            "step_pos": np.array([step], dtype=np.int64),
        }
        for i in range(n_dec):
            feed[f"past_k_{i}"] = past_ks[i]
            feed[f"past_v_{i}"] = past_vs[i]
            feed[f"cross_k_{i}"] = cross_ks[i]
            feed[f"cross_v_{i}"] = cross_vs[i]
        dec_outs = dec_sess.run(None, feed)
        step_logits = dec_outs[0]  # (1, vocab_size)
        past_ks = dec_outs[1 : 1 + n_dec]
        past_vs = dec_outs[1 + n_dec : 1 + 2 * n_dec]
        nxt = int(np.argmax(step_logits[0]))
        ort_tokens.append(nxt)
        if nxt == eos_id:
            break
        cur_tok = nxt
    ort_dec_ms = (time.perf_counter() - t_dec0) * 1000.0

    n_cmp = min(len(pt_tokens), len(ort_tokens))
    n_match = sum(1 for a, b in zip(pt_tokens[:n_cmp], ort_tokens[:n_cmp]) if a == b)
    first_diff = next((i for i, (a, b) in enumerate(zip(pt_tokens, ort_tokens)) if a != b), None)

    print(
        f"[parity] N_mem: PyTorch={pt_rinfo['n_out']} vs ORT={n_mem} | "
        f"Tokens matched: {n_match}/{n_cmp} (first_diff={first_diff})",
        flush=True,
    )
    print(
        f"[parity] CPU timing: PT_enc={pt_enc_ms:.0f}ms | ORT_enc={ort_enc_ms:.0f}ms | "
        f"ORT_dec({len(ort_tokens)} toks)={ort_dec_ms:.0f}ms "
        f"({ort_dec_ms / max(1, len(ort_tokens)):.1f} ms/tok)",
        flush=True,
    )
    return {
        "image": str(image_path),
        "shape_hw": [H, W],
        "pt_n_mem": int(pt_rinfo["n_out"]),
        "ort_n_mem": n_mem,
        "n_compared": n_cmp,
        "n_matched": n_match,
        "exact_match": (pt_tokens == ort_tokens),
        "first_diff": first_diff,
        "pt_tokens": pt_tokens,
        "ort_tokens": ort_tokens,
        "ort_enc_ms": round(ort_enc_ms, 1),
        "ort_dec_ms_per_tok": round(ort_dec_ms / max(1, len(ort_tokens)), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export UniOMR checkpoint to browser-ready ONNX.")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--test-image", type=Path, default=DEFAULT_TEST_IMAGE)
    parser.add_argument("--parity-tokens", type=int, default=30)
    args = parser.parse_args()

    print(f"[load] Loading checkpoint {args.ckpt} on CPU...", flush=True)
    model, cfg, meta = load_checkpoint(args.ckpt)
    print(
        f"[load] step={meta['step']} val_ser={meta['val_ser']} "
        f"d_model={cfg.d_model} enc={cfg.n_enc_layers} dec={cfg.n_dec_layers} "
        f"stride={cfg.backbone_stride} dynamic_routing={cfg.dynamic_routing} "
        f"vocab={cfg.vocab_size}",
        flush=True,
    )

    info = export_models(
        model,
        cfg,
        args.out_dir,
        opset=args.opset,
        quantize=(not args.no_quantize),
    )

    vocab_meta = build_vocab_json(cfg, meta.get("vocab"))
    vocab_path = args.out_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab_meta, indent=2) + "\n", encoding="utf-8")
    print(f"[vocab] Wrote {vocab_path} ({vocab_meta['vocab_size']} tokens)", flush=True)

    enc_mb = info["encoder_path"].stat().st_size / (1024 * 1024)
    dec_mb = info["decoder_path"].stat().st_size / (1024 * 1024)
    print(
        f"[sizes] {info['encoder_path'].name}: {enc_mb:.2f} MB | "
        f"{info['decoder_path'].name}: {dec_mb:.2f} MB | "
        f"TOTAL: {enc_mb + dec_mb:.2f} MB (quantized={info['quantized']})",
        flush=True,
    )

    if args.test_image.exists():
        parity = verify_parity(
            model,
            cfg,
            info["encoder_path"],
            info["decoder_path"],
            args.test_image,
            max_dim=vocab_meta["max_dim"],
            n_tokens=args.parity_tokens,
        )
        print("[parity] Result:", json.dumps({k: v for k, v in parity.items() if k not in ("pt_tokens", "ort_tokens")}), flush=True)
    else:
        print(f"[parity] Skipping: test image not found at {args.test_image}", flush=True)


if __name__ == "__main__":
    main()
