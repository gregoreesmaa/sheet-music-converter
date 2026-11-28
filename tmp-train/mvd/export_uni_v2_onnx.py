#!/usr/bin/env python3
"""Multi-target ONNX exporter for UniOMR v2 (WebGPU, NPU, CPU/WASM).

Exports a ``UniOMRv2Net`` checkpoint (or preset ``small_45m`` / ``edge85m`` /
``--tiny-test``) into ``<out-dir>/{webgpu,npu,cpu}/`` plus ``vocab.json`` and
a top-level ``manifest.json`` describing every exported target.

Targets:
  - ``webgpu``: Dynamic ``(H, W)`` (multiples of 16), dynamic ``k_mem`` capped
    at ``max_mem_tokens``, zero-copy on-GPU KV-cache slicing via ``step_pos``,
    all-T MTP draft heads, 4D Q-fold GQA, pure float32 additive causal mask,
    and INT4 ``MatMulNBits`` quantization (with per-channel INT8 Conv weights).
  - ``npu``: 100% static shapes (zero recompiles for WebNN NPU / CoreML ANE /
    Snapdragon QNN / Intel OpenVINO / AMD XDNA), fixed encoder input bucket
    ``--npu-hw`` and fixed ``--npu-k-mem``, static ring-buffer KV cache
    ``(1, n_kv_heads, kv_max_len, head_dim)``, and standard ONNX opset-17
    ``DequantizeLinear`` INT8 weight-only quantization (zero contrib ops).
  - ``cpu``: Dynamic ``(H, W)`` + dynamic ``k_mem`` + 32-bit WASM-safe tiled
    Q-chunked SDPA (``chunk_q=256``, < 200 MB RAM), pure FP32 activations,
    and optional dynamic INT8 or INT4 quantization.

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
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Force CPU-only execution before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import data  # noqa: E402
import uni_omr_v2  # noqa: E402


DEFAULT_OUT_DIR = REPO_ROOT / "web" / "models_v2"
VALID_TARGETS = ("webgpu", "npu", "cpu")


def _assert_no_unsupported_iter15_flags(cfg: uni_omr_v2.UniOMRv2Config) -> None:
    """Fail fast if any Iter-15-only architecture flag is enabled (ONNX lowering not yet implemented)."""
    unsupported: List[str] = []
    if bool(getattr(cfg, "use_yoco_cla", False)):
        unsupported.append("use_yoco_cla=True")
    else:
        yoco_self_kv = getattr(cfg, "yoco_self_kv_heads", None)
        if yoco_self_kv is not None and int(yoco_self_kv) != int(cfg.n_kv_heads):
            unsupported.append(
                f"yoco_self_kv_heads={int(yoco_self_kv)} (n_kv_heads={int(cfg.n_kv_heads)})"
            )
        yoco_cross_kv = getattr(cfg, "yoco_cross_kv_heads", None)
        if yoco_cross_kv is not None and int(yoco_cross_kv) != int(cfg.n_kv_heads):
            unsupported.append(
                f"yoco_cross_kv_heads={int(yoco_cross_kv)} (n_kv_heads={int(cfg.n_kv_heads)})"
            )
    if bool(getattr(cfg, "use_diff_cross_attn", False)):
        unsupported.append("use_diff_cross_attn=True")
    if bool(getattr(cfg, "use_coordkey_ca", False)):
        unsupported.append("use_coordkey_ca=True")
    if bool(getattr(cfg, "use_chained_mtp", False)):
        unsupported.append("use_chained_mtp=True")
    if int(getattr(cfg, "mode_gate_layers", 0)) > 0:
        unsupported.append(f"mode_gate_layers={int(cfg.mode_gate_layers)}")
    if bool(getattr(cfg, "use_quantvla_head_temp", False)):
        unsupported.append("use_quantvla_head_temp=True")
    if unsupported:
        raise NotImplementedError(
            "Iter-15 ONNX lowering is not yet implemented for config flags: "
            + ", ".join(unsupported)
            + ". Wait for Iter-15 ONNX export support before exporting this model."
        )


# ---------------------------------------------------------------------------
# Shared Clean ONNX-Safe Primitives (Zero WebGPU/NPU CPU-Fallback Ops)
# ---------------------------------------------------------------------------

def _rms_norm_onnx(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Explicit opset-17 RMSNorm (avoids custom RMSNorm contrib op)."""
    norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * norm * weight


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


def _manual_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Standard 4D opset-17 scaled dot-product attention (no custom SDPA op)."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


def _gqa_sdpa_qfold(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_kv_heads: int,
    n_rep: int,
    causal_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Zero-copy 4D Q-fold Grouped-Query Attention.

    Keeps ``k, v`` in compact ``(1, n_kv_heads, L_k, hd)`` and folds
    ``q`` from ``(1, n_heads, T, hd)`` to ``(1, n_kv_heads, n_rep * T, hd)``
    so K/V are never copied or broadcast-expanded ``n_rep`` times.
    """
    hd = q.shape[-1]
    t_len = q.shape[2]
    scale = 1.0 / math.sqrt(hd)
    if n_rep == 1:
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if causal_bias is not None:
            scores = scores + causal_bias
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    q_fold = q.reshape(1, n_kv_heads, n_rep * t_len, hd)
    scores = torch.matmul(q_fold, k.transpose(-2, -1)) * scale
    if causal_bias is not None:
        # Repeat causal_bias (1, 1, T, L_k) -> (1, 1, n_rep * T, L_k) in 4D.
        bias_rep = torch.cat([causal_bias] * n_rep, dim=2)
        scores = scores + bias_rep
    attn = F.softmax(scores, dim=-1)
    out_fold = torch.matmul(attn, v)  # (1, n_kv_heads, n_rep * T, hd)
    return out_fold.reshape(1, n_kv_heads * n_rep, t_len, hd)


@torch.jit.script
def _chunked_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_q: int = 256,
) -> torch.Tensor:
    """32-bit WASM-safe Q-chunked SDPA exported as an ONNX opset-17 Loop.

    Caps peak attention buffer size at ``O(H * chunk_q * Lk)`` (< 200 MB RAM)
    while remaining mathematically identical to full SDPA.
    """
    lq = q.shape[2]
    d = q.shape[3]
    scale = 1.0 / math.sqrt(float(d))
    k_t = k.transpose(-2, -1)
    chunks: List[torch.Tensor] = []
    i = 0
    while i < lq:
        end = i + chunk_q
        if end > lq:
            end = lq
        q_chunk = q[:, :, i:end, :]
        scores = torch.matmul(q_chunk, k_t) * scale
        probs = F.softmax(scores, dim=-1)
        chunks.append(torch.matmul(probs, v))
        i = end
    return torch.cat(chunks, dim=2)


def _pixel_unshuffle_2x2_4d(x: torch.Tensor, c: int, h_out: Any, w_out: Any) -> torch.Tensor:
    """4D-only equivalent of ``nn.PixelUnshuffle(2)`` for ``B=1`` (no SpaceToDepth).

    Maps ``(1, C, 2*h_out, 2*w_out)`` -> ``(1, 4*C, h_out, w_out)`` using only
    <= 4D reshapes and transposes, bit-identical to ``nn.PixelUnshuffle(2)``.
    """
    w_in = w_out * 2
    t = x.reshape(c, h_out, 2, w_in)
    t = t.transpose(1, 2)                    # (C, 2, h_out, 2*w_out)
    t = t.reshape(2 * c, h_out, w_out, 2)    # (2*C, h_out, w_out, 2)
    t = t.permute(0, 3, 1, 2)                # (2*C, 2, h_out, w_out)
    return t.reshape(1, 4 * c, h_out, w_out)


# ---------------------------------------------------------------------------
# 1. Shared ONNX Encoder Wrapper (Dynamic for WebGPU/CPU, Static for NPU)
# ---------------------------------------------------------------------------

class UniOMRv2EncoderONNX(nn.Module):
    """ONNX-friendly UniOMR v2 encoder with pre-projected DeepStack Cross-KV.

    When ``static_hw`` and ``static_k_mem`` are provided (``npu`` target),
    every tensor shape in the graph is 100% static (zero dynamic spatial or
    sequence dimensions, zero ONNX Loop ops).
    """

    def __init__(
        self,
        model: uni_omr_v2.UniOMRv2Net,
        use_chunked_sdpa: bool = False,
        attn_chunk_q: int = 256,
        static_hw: Optional[Tuple[int, int]] = None,
        static_k_mem: Optional[int] = None,
    ) -> None:
        super().__init__()
        _assert_no_unsupported_iter15_flags(model.cfg)
        self.cfg = model.cfg
        self.use_chunked_sdpa = bool(use_chunked_sdpa) and (static_hw is None)
        self.attn_chunk_q = int(attn_chunk_q)
        self.static_hw = static_hw
        self.static_k_mem = static_k_mem

        # Backbone submodules
        bb = model.backbone
        self.stage1 = bb.stage1
        self.stage2 = bb.stage2
        self.stage3 = bb.stage3
        self.stage4 = bb.stage4
        self.p2_ch = int(bb.p2_ch)
        self.p2_to_p3 = bb.p2_to_p3
        self.p2_to_p4 = bb.p2_to_p4
        self.bb_proj = bb.proj
        self.bb_out_norm = bb.out_norm

        # Hybrid encoder submodules
        enc = model.encoder
        self.axial_layers = enc.axial_layers
        self.patch_merger = enc.patch_merger
        self.router = enc.router
        self.pos2d_proj = enc.pos2d_proj
        self.n_register_tokens = int(model.cfg.n_register_tokens) if enc.register_tokens is not None else 0
        self.register_tokens = enc.register_tokens
        self.global_layers = enc.global_layers
        self.enc_norm = enc.enc_norm

        # DeepStack fusions + pre-projected decoder cross-attention K/V (compact n_kv_heads)
        self.deepstack_fusions = model.deepstack_fusions
        self.cross_k_projs = nn.ModuleList([l.cross_attn.k_proj for l in model.dec_layers])
        self.cross_k_norms = nn.ModuleList([l.cross_attn.k_norm for l in model.dec_layers])
        self.cross_v_projs = nn.ModuleList([l.cross_attn.v_proj for l in model.dec_layers])

        self.d_model = int(model.cfg.d_model)
        self.n_heads = int(model.cfg.n_heads)
        self.n_kv_heads = int(model.cfg.n_kv_heads)
        self.head_dim = self.d_model // self.n_heads
        self.n_dec_layers = int(model.cfg.n_dec_layers)
        self.irope_nope_every = int(model.cfg.irope_nope_every)
        self.rope_coord_scale = float(model.cfg.rope_coord_scale)

        # RoPE / sinusoidal frequency buffers
        self.register_buffer(
            "axial_inv_freq",
            enc.axial_rope.inv_freq.detach().clone().to(torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "rope2d_inv_freq",
            enc.rope2d.inv_freq.detach().clone().to(torch.float32),
            persistent=False,
        )
        q = self.d_model // 4
        pos2d_freqs = torch.exp(
            torch.arange(0, q, dtype=torch.float32) * (-math.log(10000.0) / max(1, q))
        )
        self.register_buffer("pos2d_freqs", pos2d_freqs, persistent=False)

    def _backbone_onnx(self, images: torch.Tensor) -> torch.Tensor:
        h1 = self.stage1(images)
        p2 = self.stage2(h1)  # (1, c2, H/4, W/4)
        p3 = self.stage3(p2)  # (1, c3, H/8, W/8)

        if self.static_hw is not None:
            h8 = self.static_hw[0] // 8
            w8 = self.static_hw[1] // 8
        else:
            p3_shape = torch._shape_as_tensor(p3)
            h8 = p3_shape[2]
            w8 = p3_shape[3]

        p2_unshuffled = _pixel_unshuffle_2x2_4d(p2, self.p2_ch, h8, w8)
        p3 = p3 + F.silu(self.p2_to_p3.norm(self.p2_to_p3.proj(p2_unshuffled)))
        p4 = self.stage4(p3)
        p4 = p4 + F.silu(self.p2_to_p4.norm(self.p2_to_p4.proj(p2_unshuffled)))
        return self.bb_out_norm(self.bb_proj(p4))

    def _axial_rope_onnx(self, seq_len: Any, seq_len_t: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len_t is None:
            L = int(seq_len)
            denom = float(max(1, L - 1))
            pos = torch.arange(L, dtype=torch.float32) * (self.rope_coord_scale / denom)
            freqs = pos.unsqueeze(-1) * self.axial_inv_freq.unsqueeze(0)
            emb = torch.cat([freqs, freqs], dim=-1)
            return emb.cos().view(1, 1, L, self.head_dim), emb.sin().view(1, 1, L, self.head_dim)
        denom = torch.clamp(seq_len_t - 1, min=1).to(torch.float32)
        pos = torch.arange(seq_len, dtype=torch.float32) * (self.rope_coord_scale / denom)
        freqs = pos.unsqueeze(-1) * self.axial_inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().view(1, 1, -1, self.head_dim), emb.sin().view(1, 1, -1, self.head_dim)

    def _self_attn_onnx(
        self,
        sa: uni_omr_v2._SelfAttention,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        use_chunked: bool = False,
    ) -> torch.Tensor:
        b_dim = x.shape[0]
        l_dim = x.shape[1]
        q, k, v = sa.qkv_proj(x).chunk(3, dim=-1)
        q = _rms_norm_onnx(
            q.reshape(b_dim, l_dim, self.n_heads, self.head_dim),
            sa.q_norm.weight,
            sa.q_norm.eps,
        ).transpose(1, 2)
        k = _rms_norm_onnx(
            k.reshape(b_dim, l_dim, self.n_heads, self.head_dim),
            sa.k_norm.weight,
            sa.k_norm.eps,
        ).transpose(1, 2)
        v = v.reshape(b_dim, l_dim, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, rope_cos, rope_sin)
        if use_chunked:
            out = _chunked_sdpa(q, k, v, self.attn_chunk_q)
        else:
            out = _manual_sdpa(q, k, v)
        out = out.transpose(1, 2).contiguous().reshape(b_dim, l_dim, self.d_model)
        return sa.o_proj(out)

    def _axial_layer_onnx(
        self,
        layer: uni_omr_v2.AxialEncoderLayer,
        x: torch.Tensor,
        h_p: Any,
        w_p: Any,
        row_cos: torch.Tensor,
        row_sin: torch.Tensor,
        col_cos: torch.Tensor,
        col_sin: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Row attention: (H', W', D)
        x_row = _rms_norm_onnx(x, layer.norm_row.weight, layer.norm_row.eps).reshape(h_p, w_p, self.d_model)
        row_out = self._self_attn_onnx(layer.row_attn, x_row, row_cos, row_sin, use_chunked=False)
        x = x + row_out.reshape(1, h_p, w_p, self.d_model)

        # 2. Column attention: (W', H', D)
        x_col = _rms_norm_onnx(x, layer.norm_col.weight, layer.norm_col.eps).permute(0, 2, 1, 3).reshape(w_p, h_p, self.d_model)
        col_out = self._self_attn_onnx(layer.col_attn, x_col, col_cos, col_sin, use_chunked=False)
        x = x + col_out.reshape(1, w_p, h_p, self.d_model).permute(0, 2, 1, 3)

        # 3. SwiGLU FFN
        h_ffn = _rms_norm_onnx(x, layer.norm_ffn.weight, layer.norm_ffn.eps)
        gate, up = layer.ffn.w_gate_up(h_ffn).chunk(2, dim=-1)
        x = x + layer.ffn.w_down(F.silu(gate) * up)
        return x

    def _patch_merger_4d_onnx(self, x: torch.Tensor, h_m: Any, w_m: Any) -> torch.Tensor:
        """4D PatchMerger2x2 for B=1 (zero 6D permute)."""
        pm = self.patch_merger
        x = _rms_norm_onnx(x, pm.norm.weight, pm.norm.eps)
        x = x.reshape(h_m, 2, w_m, 2 * self.d_model).transpose(1, 2).reshape(1, h_m, w_m, 4 * self.d_model)
        return pm.fc2(F.silu(pm.fc1(x)))

    def _compute_ink_and_paper_gate_f32(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure float32 InkTokenRouterV2 edge-connected dark-border ink map + paper_gate.

        Exact parity with ``InkTokenRouterV2._compute_ink_and_paper_gate`` using
        zero ``bool`` ``~`` / ``&`` / ``|`` / ``Where`` ops.
        """
        luma = (
            0.299 * images[:, 0:1]
            + 0.587 * images[:, 1:2]
            + 0.114 * images[:, 2:3]
        )

        # Per-image adaptive dark-border thresholds from 90th-percentile luma (4x downsampled).
        luma_ds = F.avg_pool2d(luma, kernel_size=4, stride=4)
        luma_flat = luma_ds.flatten(1)
        luma_sorted, _ = torch.sort(luma_flat, dim=1)
        n_ds = torch.ones_like(luma_flat, dtype=torch.float32).sum(dim=1)
        k90_idx = torch.round((n_ds - 1.0).clamp(min=0.0) * 0.90).to(torch.long).view(-1, 1)
        p90_luma = torch.gather(luma_sorted, 1, k90_idx).view(-1, 1, 1, 1)
        bright_thr = (0.85 * p90_luma).clamp(min=0.05, max=0.35)
        dark_thr = (0.65 * p90_luma).clamp(min=0.05, max=0.22)

        # Edge-connected dark-border mask via float32 cumsums (exact for H, W <= 2^24).
        bright_f = (luma >= bright_thr).to(torch.float32)
        cs_y = bright_f.cumsum(dim=2)
        cs_x = bright_f.cumsum(dim=3)
        top_dark = (cs_y < 0.5).to(torch.float32)
        bot_dark = ((cs_y[:, :, -1:, :] - cs_y) < 0.5).to(torch.float32)
        left_dark = (cs_x < 0.5).to(torch.float32)
        right_dark = ((cs_x[:, :, :, -1:] - cs_x) < 0.5).to(torch.float32)
        edge_dark_f = (top_dark + bot_dark + left_dark + right_dark).clamp(0.0, 1.0)
        border_px = edge_dark_f * (luma < dark_thr).to(torch.float32)

        border_dilated_px = F.max_pool2d(
            F.max_pool2d(border_px, kernel_size=(1, 13), stride=1, padding=(0, 6)),
            kernel_size=(13, 1),
            stride=1,
            padding=(6, 0),
        )
        paper_px = (1.0 - border_dilated_px).clamp(0.0, 1.0)

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
        local_ink = ((local_bg - luma) / local_bg.clamp(min=0.15)).clamp(0.0, 1.0) * paper_px

        # Merged grid is /16 (H, W are multiples of 16 -> exact fixed-stride pooling).
        ks = (16, 16)
        ink_max = F.max_pool2d(local_ink, kernel_size=ks, stride=ks)
        ink_avg = F.avg_pool2d(local_ink, kernel_size=ks, stride=ks)
        local_ink_grid = 0.5 * ink_max + 0.5 * ink_avg

        paper_grid_f = F.avg_pool2d(paper_px, kernel_size=ks, stride=ks)
        effective_paper_f = (paper_grid_f > 0.5).to(torch.float32)

        local_ink_grid = local_ink_grid * effective_paper_f
        paper_gate = 0.05 + 0.95 * effective_paper_f
        return local_ink_grid, paper_gate, effective_paper_f

    def _route_v2_onnx(
        self,
        merged_2d: torch.Tensor,
        images: torch.Tensor,
        h_m_t: Optional[torch.Tensor],
        w_m_t: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pure float32 InkTokenRouterV2 forward (dynamic or static k_mem)."""
        r = self.router
        saliency_logits = r.saliency_net(merged_2d)
        if r.vrfm_band_alpha > 0.0 and merged_2d.shape[2] >= 3:
            row_mean = saliency_logits.mean(dim=3, keepdim=True)
            row_smooth = F.avg_pool2d(
                row_mean,
                kernel_size=(3, 1),
                stride=1,
                padding=(1, 0),
                count_include_pad=False,
            )
            saliency_logits = saliency_logits + r.vrfm_band_alpha * row_smooth
        saliency_prob = torch.sigmoid(saliency_logits)

        local_ink_grid, paper_gate, effective_paper_f = self._compute_ink_and_paper_gate_f32(images)
        saliency_prob = saliency_prob * paper_gate

        if r.use_ink_prior:
            # Sort-free mean + 2.5*std normalization (exact parity with uni_omr_v2.py).
            flat_f = local_ink_grid.flatten(1)
            mean_f = flat_f.mean(dim=-1, keepdim=True)
            var_f = ((flat_f - mean_f) * (flat_f - mean_f)).mean(dim=-1, keepdim=True)
            std_f = torch.sqrt(var_f.clamp(min=0.0))
            denom = (mean_f + 2.5 * std_f).clamp(min=1e-4).view(1, 1, 1, 1)
            local_ink_norm = (local_ink_grid / denom).clamp(0.0, 1.0)
            score = saliency_prob + torch.sigmoid(r.ink_prior_scale) * local_ink_norm
        else:
            score = saliency_prob

        score_flat = score.flatten(1)  # (1, N)
        tokens_flat = merged_2d.permute(0, 2, 3, 1).reshape(1, -1, self.d_model)  # (1, N, D)

        if self.static_hw is not None:
            h_m = self.static_hw[0] // 16
            w_m = self.static_hw[1] // 16
            n_total = h_m * w_m
            ys = torch.arange(h_m, dtype=torch.float32)
            xs = torch.arange(w_m, dtype=torch.float32)
            grid_y = ys.unsqueeze(1).expand(h_m, w_m)
            grid_x = xs.unsqueeze(0).expand(h_m, w_m)
            grid_coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1).unsqueeze(0)

            target_k_mem = self.static_k_mem if self.static_k_mem is not None else min(n_total, r.max_mem_tokens)
            k_kept_int = max(1, min(n_total - 1 if n_total > 1 else 1, target_k_mem - r.n_bg_summary))
            _, topk_idx = torch.topk(score_flat, k=k_kept_int, dim=1, largest=True, sorted=False)
            kept_idx, _ = torch.sort(topk_idx, dim=1)
            cy = torch.tensor((h_m - 1) * 0.5, dtype=torch.float32)
            cx = torch.tensor((w_m - 1) * 0.5, dtype=torch.float32)
        else:
            assert h_m_t is not None and w_m_t is not None
            n_t = h_m_t * w_m_t
            ys = torch.arange(merged_2d.shape[2], dtype=torch.float32)
            xs = torch.arange(merged_2d.shape[3], dtype=torch.float32)
            grid_y = ys.unsqueeze(1).expand(merged_2d.shape[2], merged_2d.shape[3])
            grid_x = xs.unsqueeze(0).expand(merged_2d.shape[2], merged_2d.shape[3])
            grid_coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1).unsqueeze(0)

            if r.dynamic_routing:
                flat_paper_f = effective_paper_f.flatten(1)
                n_valid_f = flat_paper_f.sum(dim=1)
                has_valid_f = (n_valid_f > 0.5).to(torch.float32).view(1, 1)
                score_masked_sorted, _ = torch.sort(score_flat + (1.0 - flat_paper_f) * 1e6, dim=1)
                pos_m = (n_valid_f - 1.0).clamp(min=0.0) * 0.10
                lo_m = torch.floor(pos_m)
                hi_m = torch.ceil(pos_m)
                frac_m = (pos_m - lo_m).view(1, 1)
                v_lo_m = torch.gather(score_masked_sorted, 1, lo_m.to(torch.long).view(1, 1))
                v_hi_m = torch.gather(score_masked_sorted, 1, hi_m.to(torch.long).view(1, 1))
                p10_masked = v_lo_m + frac_m * (v_hi_m - v_lo_m)

                score_sorted, _ = torch.sort(score_flat, dim=1)
                pos = (n_t - 1).clamp(min=0).to(torch.float32) * 0.10
                lo_f = torch.floor(pos)
                hi_f = torch.ceil(pos)
                frac = (pos - lo_f).view(1, 1)
                v_lo = torch.gather(score_sorted, 1, lo_f.to(torch.long).view(1, 1))
                v_hi = torch.gather(score_sorted, 1, hi_f.to(torch.long).view(1, 1))
                p10_fallback = v_lo + frac * (v_hi - v_lo)

                p10 = has_valid_f * p10_masked + (1.0 - has_valid_f) * p10_fallback
                thr = max(float(r.saliency_thr), 0.15)
                above = (score_flat > (p10 + thr)).to(torch.long).sum(dim=1)
                k_min = torch.clamp(torch.ceil(n_t.to(torch.float32) * r.min_keep_ratio).to(torch.long), min=1)
                k_max = torch.minimum(
                    n_t,
                    torch.maximum(k_min, torch.ceil(n_t.to(torch.float32) * r.max_keep_ratio).to(torch.long)),
                )
                k_t = torch.minimum(k_max, torch.maximum(k_min, above[0]))
            else:
                k_t = torch.minimum(
                    n_t,
                    torch.clamp(torch.ceil(n_t.to(torch.float32) * r.keep_ratio).to(torch.long), min=1),
                )

            max_kept = max(1, r.max_mem_tokens - r.n_bg_summary)
            n_cap = torch.clamp(n_t - 1, min=1)
            k_t = torch.clamp(
                torch.minimum(k_t, torch.minimum(n_cap, torch.tensor(max_kept, dtype=torch.long))),
                min=1,
            )
            _, topk_idx = torch.topk(score_flat, k=k_t, dim=1, largest=True, sorted=False)
            kept_idx, _ = torch.sort(topk_idx, dim=1)
            cy = (h_m_t.to(torch.float32) - 1.0) * 0.5
            cx = (w_m_t.to(torch.float32) - 1.0) * 0.5

        kept_scores = torch.gather(score_flat, 1, kept_idx)
        kept_gate = (0.5 + 0.5 * torch.sigmoid(kept_scores)).unsqueeze(-1)

        idx_feat = kept_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
        kept_tokens = torch.gather(tokens_flat, 1, idx_feat) * kept_gate

        idx_coords = kept_idx.unsqueeze(-1).expand(-1, -1, 2)
        kept_coords = torch.gather(grid_coords, 1, idx_coords)

        # Pure float32 ScatterElements + additive (-1e4) mask + bg_q_proj on 4 bg_queries.
        kept_mask_f = torch.zeros_like(score_flat).scatter(
            1, kept_idx, torch.ones_like(kept_idx, dtype=score_flat.dtype)
        )
        bg_q_proj = F.linear(r.bg_queries, r.bg_key_proj.weight.t())  # (1, M, D)
        bg_logits = torch.bmm(bg_q_proj, tokens_flat.transpose(1, 2)) / math.sqrt(self.d_model)
        bg_logits = bg_logits - score_flat.unsqueeze(1) + kept_mask_f.unsqueeze(1) * (-1e4)
        bg_attn = F.softmax(bg_logits, dim=-1)
        pooled = torch.bmm(bg_attn, tokens_flat)
        bg_tokens = _rms_norm_onnx(
            r.bg_val_proj(pooled), r.bg_out_norm.weight, r.bg_out_norm.eps
        )

        center_coord = torch.stack([cy, cx]).view(1, 1, 2).expand(1, r.n_bg_summary, 2)
        out_tokens = torch.cat([kept_tokens, bg_tokens], dim=1)
        out_coords = torch.cat([kept_coords, center_coord], dim=1)
        return out_tokens, out_coords

    def _sinusoidal_pos2d_onnx(
        self,
        coords: torch.Tensor,
        y_denom: Any,
        x_denom: Any,
    ) -> torch.Tensor:
        y = (coords[..., 0] / y_denom) * self.rope_coord_scale
        x = (coords[..., 1] / x_denom) * self.rope_coord_scale
        fy = y.unsqueeze(-1) * self.pos2d_freqs
        fx = x.unsqueeze(-1) * self.pos2d_freqs
        return torch.cat([fy.sin(), fy.cos(), fx.sin(), fx.cos()], dim=-1)

    def _rope2d_onnx(
        self,
        coords: torch.Tensor,
        y_denom: Any,
        x_denom: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y_norm = coords[..., 0] / y_denom
        x_norm = coords[..., 1] / x_denom
        fy = (y_norm * self.rope_coord_scale).unsqueeze(-1) * self.rope2d_inv_freq
        fx = (x_norm * self.rope_coord_scale).unsqueeze(-1) * self.rope2d_inv_freq
        emb_half = torch.stack([fy, fx], dim=-1).flatten(-2)
        emb = torch.cat([emb_half, emb_half], dim=-1)
        return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)

    def _global_layer_onnx(
        self,
        layer: uni_omr_v2.GlobalEncoderLayer,
        x: torch.Tensor,
        cos2d: torch.Tensor,
        sin2d: torch.Tensor,
    ) -> torch.Tensor:
        h = _rms_norm_onnx(x, layer.norm1.weight, layer.norm1.eps)
        x = x + self._self_attn_onnx(layer.self_attn, h, cos2d, sin2d, use_chunked=self.use_chunked_sdpa)
        h2 = _rms_norm_onnx(x, layer.norm2.weight, layer.norm2.eps)
        gate, up = layer.ffn.w_gate_up(h2).chunk(2, dim=-1)
        x = x + layer.ffn.w_down(F.silu(gate) * up)
        return x

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        feat_2d = self._backbone_onnx(image)  # (1, D, H/8, W/8)

        if self.static_hw is not None:
            h_p = self.static_hw[0] // 8
            w_p = self.static_hw[1] // 8
            h_m = self.static_hw[0] // 16
            w_m = self.static_hw[1] // 16
            h_m_t = None
            w_m_t = None
            row_cos, row_sin = self._axial_rope_onnx(w_p, None)
            col_cos, col_sin = self._axial_rope_onnx(h_p, None)
            y_denom: Any = float(max(1, h_m - 1))
            x_denom: Any = float(max(1, w_m - 1))
        else:
            f_shape = torch._shape_as_tensor(feat_2d)
            h_p = feat_2d.shape[2]
            w_p = feat_2d.shape[3]
            h_p_t = f_shape[2]
            w_p_t = f_shape[3]
            h_m = h_p // 2
            w_m = w_p // 2
            h_m_t = h_p_t // 2
            w_m_t = w_p_t // 2
            row_cos, row_sin = self._axial_rope_onnx(w_p, w_p_t)
            col_cos, col_sin = self._axial_rope_onnx(h_p, h_p_t)
            y_denom = torch.clamp(h_m_t - 1, min=1).to(torch.float32)
            x_denom = torch.clamp(w_m_t - 1, min=1).to(torch.float32)

        x = feat_2d.permute(0, 2, 3, 1).contiguous()  # (1, H/8, W/8, D)
        for layer in self.axial_layers:
            x = self._axial_layer_onnx(layer, x, h_p, w_p, row_cos, row_sin, col_cos, col_sin)

        merged = self._patch_merger_4d_onnx(x, h_m, w_m)  # (1, H/16, W/16, D)
        merged_2d = merged.permute(0, 3, 1, 2).contiguous()

        routed_tokens, routed_coords = self._route_v2_onnx(merged_2d, image, h_m_t, w_m_t)
        mem_post_merger = routed_tokens

        pos2d = self.pos2d_proj(self._sinusoidal_pos2d_onnx(routed_coords, y_denom, x_denom))
        cos2d, sin2d = self._rope2d_onnx(routed_coords, y_denom, x_denom)
        mem = routed_tokens + pos2d

        n_reg = self.n_register_tokens
        if n_reg > 0 and self.register_tokens is not None:
            reg = self.register_tokens.to(mem.dtype)
            mem = torch.cat([reg, mem], dim=1)
            ones_c = torch.ones((1, 1, n_reg, self.head_dim), dtype=cos2d.dtype)
            zeros_s = torch.zeros((1, 1, n_reg, self.head_dim), dtype=sin2d.dtype)
            cos2d = torch.cat([ones_c, cos2d], dim=2)
            sin2d = torch.cat([zeros_s, sin2d], dim=2)

        n_global = len(self.global_layers)
        mid_idx = max(0, (n_global // 2) - 1)
        mem_global_mid = mem_post_merger
        for i, layer in enumerate(self.global_layers):
            mem = self._global_layer_onnx(layer, mem, cos2d, sin2d)
            if i == mid_idx:
                mem_global_mid = mem[:, n_reg:, :] if n_reg > 0 else mem

        if n_reg > 0:
            mem = mem[:, n_reg:, :]
        mem_final = _rms_norm_onnx(mem, self.enc_norm.weight, self.enc_norm.eps) + pos2d

        # Encoder-side DeepStackFusion + pre-projected Cross-K/V (compact n_kv_heads).
        level_mems = (mem_post_merger, mem_global_mid)
        cross_ks: List[torch.Tensor] = []
        cross_vs: List[torch.Tensor] = []
        for idx in range(self.n_dec_layers):
            if idx < len(self.deepstack_fusions):
                ds = self.deepstack_fusions[idx]
                delta = ds.proj(_rms_norm_onnx(level_mems[idx], ds.norm.weight, ds.norm.eps))
                layer_mem = mem_final + torch.tanh(ds.gate) * delta
            else:
                layer_mem = mem_final
            ck = _rms_norm_onnx(
                self.cross_k_projs[idx](layer_mem).reshape(1, -1, self.n_kv_heads, self.head_dim),
                self.cross_k_norms[idx].weight,
                self.cross_k_norms[idx].eps,
            ).transpose(1, 2)
            cv = self.cross_v_projs[idx](layer_mem).reshape(1, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)
            cross_ks.append(ck)
            cross_vs.append(cv)

        return tuple(cross_ks + cross_vs)


# ---------------------------------------------------------------------------
# 2. Dynamic-KV Decoder Wrapper (WebGPU + CPU: Zero-Copy Slice + All-T MTP)
# ---------------------------------------------------------------------------

class UniOMRv2DecoderDynamicONNX(nn.Module):
    """Variable-T GQA KV-cached decoder with on-GPU ``step_pos`` KV slicing & all-T MTP.

    Inputs:
      - ``token_ids``: int64 ``(1, T)``
      - ``step_pos``: int64 ``(1,)`` (slices ``past_k/v[:, :, :step_pos, :]`` on-device)
      - ``past_k_0..L-1``, ``past_v_0..L-1``: float32 ``(1, n_kv_heads, past_len, head_dim)``
      - ``cross_k_0..L-1``, ``cross_v_0..L-1``: float32 ``(1, n_kv_heads, n_mem, head_dim)``
    Outputs:
      - ``logits``: float32 ``(1, T, vocab_size)``
      - ``mtp_logits_0..K-2``: float32 ``(1, T, vocab_size)`` (all-T positions!)
      - ``present_k_0..L-1``, ``present_v_0..L-1``: float32 ``(1, n_kv_heads, step_pos + T, head_dim)``
    """

    def __init__(self, model: uni_omr_v2.UniOMRv2Net) -> None:
        super().__init__()
        _assert_no_unsupported_iter15_flags(model.cfg)
        self.d_model = int(model.cfg.d_model)
        self.n_heads = int(model.cfg.n_heads)
        self.n_kv_heads = int(model.cfg.n_kv_heads)
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = self.d_model // self.n_heads
        self.n_dec_layers = int(model.cfg.n_dec_layers)
        self.irope_nope_every = int(model.cfg.irope_nope_every)

        self.tok_emb = model.tok_emb
        self.dec_layers = model.dec_layers
        self.dec_norm = model.dec_norm
        self.lm_head = model.lm_head
        self.mtp_heads = model.mtp_heads
        self.register_buffer(
            "rope1d_inv_freq",
            model.rope1d.inv_freq.detach().clone().to(torch.float32),
            persistent=False,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        step_pos: torch.Tensor,
        *kv_tensors: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        L = self.n_dec_layers
        past_ks = kv_tensors[0:L]
        past_vs = kv_tensors[L : 2 * L]
        cross_ks = kv_tensors[2 * L : 3 * L]
        cross_vs = kv_tensors[3 * L : 4 * L]

        x = self.tok_emb(token_ids)  # (1, T, D)
        t_len = token_ids.shape[1]
        pos0 = step_pos[0]  # scalar int64 tensor

        # 1D RoPE for positions [step_pos, ..., step_pos + T - 1].
        t_arange = torch.arange(t_len, dtype=torch.float32)
        pos = step_pos.to(torch.float32).view(1, 1) + t_arange.view(1, -1)  # (1, T)
        freqs = pos.unsqueeze(-1) * self.rope1d_inv_freq.view(1, 1, -1)     # (1, T, hd // 2)
        emb = torch.cat([freqs, freqs], dim=-1)                             # (1, T, hd)
        cos1d = emb.cos().unsqueeze(1)                                      # (1, 1, T, hd)
        sin1d = emb.sin().unsqueeze(1)                                      # (1, 1, T, hd)

        # Pure float32 additive causal bias over [0 .. step_pos + T - 1].
        # Slice past_k/v on-device to :step_pos so partial-accept rewind is zero-copy!
        pos0_f = pos0.to(torch.float32)
        row_pos = (pos0_f + t_arange).view(1, 1, -1, 1)
        total_len = pos0 + t_len
        col_pos = torch.arange(total_len, dtype=torch.float32).view(1, 1, 1, -1)
        causal_bias = torch.clamp(col_pos - row_pos, min=0.0, max=1.0) * (-1e9)

        present_ks: List[torch.Tensor] = []
        present_vs: List[torch.Tensor] = []

        for idx, layer in enumerate(self.dec_layers):
            # 1. Causal GQA self-attention with on-device KV slice + QK-Norm.
            h = _rms_norm_onnx(x, layer.norm1.weight, layer.norm1.eps)
            sa = layer.self_attn
            q = _rms_norm_onnx(
                sa.q_proj(h).reshape(1, t_len, self.n_heads, self.head_dim),
                sa.q_norm.weight,
                sa.q_norm.eps,
            ).transpose(1, 2)
            k = _rms_norm_onnx(
                sa.k_proj(h).reshape(1, t_len, self.n_kv_heads, self.head_dim),
                sa.k_norm.weight,
                sa.k_norm.eps,
            ).transpose(1, 2)
            v = sa.v_proj(h).reshape(1, t_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
            use_rope = getattr(
                layer,
                "use_rope",
                (self.irope_nope_every <= 0) or (((idx + 1) % self.irope_nope_every) != 0),
            )
            if use_rope:
                q, k = _apply_rope(q, k, cos1d, sin1d)

            pk = past_ks[idx][:, :, :pos0, :]
            pv = past_vs[idx][:, :, :pos0, :]
            k_cat = torch.cat([pk, k], dim=2)
            v_cat = torch.cat([pv, v], dim=2)
            present_ks.append(k_cat)
            present_vs.append(v_cat)

            sa_out = (
                _gqa_sdpa_qfold(q, k_cat, v_cat, self.n_kv_heads, self.n_rep, causal_bias=causal_bias)
                .transpose(1, 2)
                .contiguous()
                .reshape(1, t_len, self.d_model)
            )
            x = x + sa.o_proj(sa_out)

            # 2. Cross-attention using pre-projected DeepStack K/V (4D Q-fold GQA).
            h2 = _rms_norm_onnx(x, layer.norm2.weight, layer.norm2.eps)
            ca = layer.cross_attn
            cq = _rms_norm_onnx(
                ca.q_proj(h2).reshape(1, t_len, self.n_heads, self.head_dim),
                ca.q_norm.weight,
                ca.q_norm.eps,
            ).transpose(1, 2)
            ca_out = (
                _gqa_sdpa_qfold(cq, cross_ks[idx], cross_vs[idx], self.n_kv_heads, self.n_rep, causal_bias=None)
                .transpose(1, 2)
                .contiguous()
                .reshape(1, t_len, self.d_model)
            )
            x = x + ca.o_proj(ca_out)

            # 3. SwiGLU FFN.
            h3 = _rms_norm_onnx(x, layer.norm3.weight, layer.norm3.eps)
            gate, up = layer.ffn.w_gate_up(h3).chunk(2, dim=-1)
            x = x + layer.ffn.w_down(F.silu(gate) * up)

        h_out = _rms_norm_onnx(x, self.dec_norm.weight, self.dec_norm.eps)  # (1, T, D)
        logits = self.lm_head(h_out)  # (1, T, vocab_size)

        # All-T MTP draft heads (eliminates post-mismatch T=1 fallback steps!).
        mtp_logits: List[torch.Tensor] = []
        for head in self.mtp_heads:
            h_mtp = h_out + F.silu(
                head.proj(_rms_norm_onnx(h_out, head.norm.weight, head.norm.eps))
            )
            mtp_logits.append(self.lm_head(h_mtp))  # (1, T, vocab_size)

        return (logits, *mtp_logits, *present_ks, *present_vs)


# ---------------------------------------------------------------------------
# 3. Static Ring-Buffer KV Decoder Wrapper (NPU: 100% Constant Shapes)
# ---------------------------------------------------------------------------

class UniOMRv2DecoderNPUONNX(nn.Module):
    """100% static-shape ring-buffer KV-cache GQA decoder for NPU compilers.

    Inputs (ALL static shapes):
      - ``token_ids``: int64 ``(1, T)`` (fixed ``T = static_t``)
      - ``step_pos``: int64 ``(1,)`` (write offset in ``[0, kv_max_len - T]``)
      - ``past_k_0..L-1``, ``past_v_0..L-1``: float32 ``(1, n_kv_heads, kv_max_len, head_dim)``
      - ``cross_k_0..L-1``, ``cross_v_0..L-1``: float32 ``(1, n_kv_heads, static_k_mem, head_dim)``
    Outputs (ALL static shapes):
      - ``logits``: float32 ``(1, T, vocab_size)``
      - ``mtp_logits_0..K-2``: float32 ``(1, T, vocab_size)``
      - ``present_k_0..L-1``, ``present_v_0..L-1``: float32 ``(1, n_kv_heads, kv_max_len, head_dim)``
    """

    def __init__(
        self,
        model: uni_omr_v2.UniOMRv2Net,
        kv_max_len: int = 2048,
        static_t: int = 1,
    ) -> None:
        super().__init__()
        _assert_no_unsupported_iter15_flags(model.cfg)
        self.d_model = int(model.cfg.d_model)
        self.n_heads = int(model.cfg.n_heads)
        self.n_kv_heads = int(model.cfg.n_kv_heads)
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = self.d_model // self.n_heads
        self.n_dec_layers = int(model.cfg.n_dec_layers)
        self.irope_nope_every = int(model.cfg.irope_nope_every)
        self.kv_max_len = int(kv_max_len)
        self.static_t = int(static_t)

        self.tok_emb = model.tok_emb
        self.dec_layers = model.dec_layers
        self.dec_norm = model.dec_norm
        self.lm_head = model.lm_head
        self.mtp_heads = model.mtp_heads
        self.register_buffer(
            "rope1d_inv_freq",
            model.rope1d.inv_freq.detach().clone().to(torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "kv_pos_idx",
            torch.arange(self.kv_max_len, dtype=torch.float32).view(1, 1, 1, self.kv_max_len),
            persistent=False,
        )
        self.register_buffer(
            "t_arange_f",
            torch.arange(self.static_t, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "t_arange_i",
            torch.arange(self.static_t, dtype=torch.long).view(1, 1, self.static_t, 1),
            persistent=False,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        step_pos: torch.Tensor,
        *kv_tensors: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        L = self.n_dec_layers
        T = self.static_t
        past_ks = kv_tensors[0:L]
        past_vs = kv_tensors[L : 2 * L]
        cross_ks = kv_tensors[2 * L : 3 * L]
        cross_vs = kv_tensors[3 * L : 4 * L]

        x = self.tok_emb(token_ids)  # (1, T, D)

        # 1D RoPE for positions [step_pos .. step_pos + T - 1].
        pos = step_pos.to(torch.float32).view(1, 1) + self.t_arange_f.view(1, T)  # (1, T)
        freqs = pos.unsqueeze(-1) * self.rope1d_inv_freq.view(1, 1, -1)          # (1, T, hd // 2)
        emb = torch.cat([freqs, freqs], dim=-1)                                  # (1, T, hd)
        cos1d = emb.cos().unsqueeze(1)                                           # (1, 1, T, hd)
        sin1d = emb.sin().unsqueeze(1)                                           # (1, 1, T, hd)

        # Static pure-float32 causal + valid-ring-buffer mask:
        # Allowed iff kv_pos_idx <= step_pos + t (masks future tokens AND unwritten slots >= step_pos + T).
        row_pos = pos.view(1, 1, T, 1)
        causal_bias = torch.clamp(self.kv_pos_idx - row_pos, min=0.0, max=1.0) * (-1e9)

        # Static ScatterElements write indices for slots [step_pos .. step_pos + T - 1].
        write_idx = (step_pos.view(1, 1, 1, 1) + self.t_arange_i).expand(
            1, self.n_kv_heads, T, self.head_dim
        )

        present_ks: List[torch.Tensor] = []
        present_vs: List[torch.Tensor] = []

        for idx, layer in enumerate(self.dec_layers):
            h = _rms_norm_onnx(x, layer.norm1.weight, layer.norm1.eps)
            sa = layer.self_attn
            q = _rms_norm_onnx(
                sa.q_proj(h).reshape(1, T, self.n_heads, self.head_dim),
                sa.q_norm.weight,
                sa.q_norm.eps,
            ).transpose(1, 2)
            k = _rms_norm_onnx(
                sa.k_proj(h).reshape(1, T, self.n_kv_heads, self.head_dim),
                sa.k_norm.weight,
                sa.k_norm.eps,
            ).transpose(1, 2)
            v = sa.v_proj(h).reshape(1, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            use_rope = getattr(
                layer,
                "use_rope",
                (self.irope_nope_every <= 0) or (((idx + 1) % self.irope_nope_every) != 0),
            )
            if use_rope:
                q, k = _apply_rope(q, k, cos1d, sin1d)

            k_buf = past_ks[idx].scatter(2, write_idx, k)
            v_buf = past_vs[idx].scatter(2, write_idx, v)
            present_ks.append(k_buf)
            present_vs.append(v_buf)

            sa_out = (
                _gqa_sdpa_qfold(q, k_buf, v_buf, self.n_kv_heads, self.n_rep, causal_bias=causal_bias)
                .transpose(1, 2)
                .contiguous()
                .reshape(1, T, self.d_model)
            )
            x = x + sa.o_proj(sa_out)

            # Cross-attention using pre-projected DeepStack K/V.
            h2 = _rms_norm_onnx(x, layer.norm2.weight, layer.norm2.eps)
            ca = layer.cross_attn
            cq = _rms_norm_onnx(
                ca.q_proj(h2).reshape(1, T, self.n_heads, self.head_dim),
                ca.q_norm.weight,
                ca.q_norm.eps,
            ).transpose(1, 2)
            ca_out = (
                _gqa_sdpa_qfold(cq, cross_ks[idx], cross_vs[idx], self.n_kv_heads, self.n_rep, causal_bias=None)
                .transpose(1, 2)
                .contiguous()
                .reshape(1, T, self.d_model)
            )
            x = x + ca.o_proj(ca_out)

            # SwiGLU FFN.
            h3 = _rms_norm_onnx(x, layer.norm3.weight, layer.norm3.eps)
            gate, up = layer.ffn.w_gate_up(h3).chunk(2, dim=-1)
            x = x + layer.ffn.w_down(F.silu(gate) * up)

        h_out = _rms_norm_onnx(x, self.dec_norm.weight, self.dec_norm.eps)
        logits = self.lm_head(h_out)  # (1, T, vocab_size)

        mtp_logits: List[torch.Tensor] = []
        for head in self.mtp_heads:
            h_mtp = h_out + F.silu(
                head.proj(_rms_norm_onnx(h_out, head.norm.weight, head.norm.eps))
            )
            mtp_logits.append(self.lm_head(h_mtp))

        return (logits, *mtp_logits, *present_ks, *present_vs)


# ---------------------------------------------------------------------------
# Quantization Helpers (WebGPU MatMulNBits, NPU Opset-17 DequantizeLinear, CPU)
# ---------------------------------------------------------------------------

def _compress_weights_int8_dequant(
    model_proto: Any,
    op_types: Sequence[str] = ("Conv",),
    min_elems: int = 256,
) -> int:
    """Compress FP32 weight initializers of ``op_types`` to per-channel INT8 + opset-17 DequantizeLinear.

    Produces 100% standard ONNX opset-17 nodes (zero ``com.microsoft`` contrib ops)
    so every NPU compiler (WebNN, CoreML ANE, QNN, OpenVINO, XDNA) and WebGPU EP
    can ingest it cleanly.
    """
    import onnx
    from onnx import helper, numpy_helper

    op_set = set(op_types)
    init_map = {init.name: idx for idx, init in enumerate(model_proto.graph.initializer)}
    compressed = 0
    new_inits: List[Any] = []
    new_nodes: List[Any] = []
    replaced_init_names: set[str] = set()
    dq_cache: Dict[str, str] = {}

    for node in model_proto.graph.node:
        if node.op_type in op_set and len(node.input) >= 2:
            w_name = node.input[1]
            if w_name in dq_cache:
                node.input[1] = dq_cache[w_name]
            elif w_name in init_map and w_name not in replaced_init_names:
                init = model_proto.graph.initializer[init_map[w_name]]
                if init.data_type == onnx.TensorProto.FLOAT:
                    w = numpy_helper.to_array(init)
                    if w.size >= min_elems and w.ndim in (2, 4):
                        if node.op_type == "Conv" and w.ndim == 4:
                            axis = 0
                            reduce_axes = (1, 2, 3)
                            bcast_shape = (-1, 1, 1, 1)
                        elif node.op_type == "MatMul" and w.ndim == 2:
                            axis = 1
                            reduce_axes = (0,)
                            bcast_shape = (1, -1)
                        elif node.op_type == "Gemm" and w.ndim == 2:
                            trans_b = 0
                            for attr in node.attribute:
                                if attr.name == "transB":
                                    trans_b = int(attr.i)
                            axis = 0 if trans_b else 1
                            reduce_axes = (1,) if axis == 0 else (0,)
                            bcast_shape = (-1, 1) if axis == 0 else (1, -1)
                        else:
                            axis = -1

                        if axis >= 0:
                            max_abs = np.max(np.abs(w), axis=reduce_axes)
                            scale = np.maximum(max_abs / 127.0, 1e-8).astype(np.float32)
                            w_q = np.clip(
                                np.round(w / scale.reshape(bcast_shape)), -127, 127
                            ).astype(np.int8)
                            zp = np.zeros(w.shape[axis], dtype=np.int8)

                            q_name = f"{w_name}_i8"
                            s_name = f"{w_name}_scale"
                            z_name = f"{w_name}_zp"
                            dq_out = f"{w_name}_dequant"

                            new_inits.append(numpy_helper.from_array(w_q, name=q_name))
                            new_inits.append(numpy_helper.from_array(scale, name=s_name))
                            new_inits.append(numpy_helper.from_array(zp, name=z_name))
                            dq_node = helper.make_node(
                                "DequantizeLinear",
                                inputs=[q_name, s_name, z_name],
                                outputs=[dq_out],
                                name=f"Dequant_{w_name}",
                                axis=axis,
                            )
                            new_nodes.append(dq_node)
                            node.input[1] = dq_out
                            dq_cache[w_name] = dq_out
                            replaced_init_names.add(w_name)
                            compressed += 1
        new_nodes.append(node)

    if compressed > 0:
        still_used = {inp for node in new_nodes for inp in node.input}
        kept_inits = [
            init
            for init in model_proto.graph.initializer
            if (init.name not in replaced_init_names) or (init.name in still_used)
        ]
        del model_proto.graph.initializer[:]
        model_proto.graph.initializer.extend(kept_inits)
        model_proto.graph.initializer.extend(new_inits)
        del model_proto.graph.node[:]
        model_proto.graph.node.extend(new_nodes)

    return compressed


def _convert_fp16_mixed(fp32_path: Path, out_path: Path) -> bool:
    """Convert FP32 model to mixed FP16 while keeping norms/softmax/router in FP32."""
    try:
        from onnxconverter_common import float16
        import onnx

        proto = onnx.load(str(fp32_path))
        fp16_proto = float16.convert_float_to_float16(
            proto,
            keep_io_types=True,
            op_block_list=[
                "Softmax",
                "ReduceMean",
                "Pow",
                "Sqrt",
                "Rsqrt",
                "GroupNormalization",
                "LayerNormalization",
                "TopK",
                "ScatterElements",
                "AveragePool",
                "MaxPool",
            ],
        )
        onnx.save(fp16_proto, str(out_path))
        return True
    except Exception:
        return False


def _quantize_webgpu_int4(
    fp32_path: Path,
    out_path: Path,
    block_size: int = 32,
    is_symmetric: bool = False,
    compress_conv_int8: bool = False,
) -> None:
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    quantizer = MatMulNBitsQuantizer(
        model=str(fp32_path),
        bits=4,
        block_size=block_size,
        is_symmetric=is_symmetric,
        accuracy_level=4,
    )
    quantizer.process()
    if compress_conv_int8:
        _compress_weights_int8_dequant(quantizer.model.model, op_types=("Conv",), min_elems=1024)
    quantizer.model.save_model_to_file(str(out_path))


def _quantize_npu_int8_dequant(fp32_path: Path, out_path: Path) -> int:
    """Quantize Conv/MatMul/Gemm weights to opset-17 INT8 DequantizeLinear (zero contrib ops)."""
    import onnx

    proto = onnx.load(str(fp32_path))
    n_comp = _compress_weights_int8_dequant(proto, op_types=("Conv", "MatMul", "Gemm"), min_elems=256)
    onnx.save(proto, str(out_path))
    return n_comp


def _quantize_cpu_dynamic_int8(
    fp32_path: Path,
    out_path: Path,
    op_types: Sequence[str],
) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(out_path),
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=list(op_types),
    )


# ---------------------------------------------------------------------------
# Model Loading & Vocab JSON
# ---------------------------------------------------------------------------

def build_tiny_test_model(seed: int = 42) -> Tuple[uni_omr_v2.UniOMRv2Net, uni_omr_v2.UniOMRv2Config]:
    """Deterministic tiny UniOMRv2Net for fast CPU export & parity unit tests."""
    torch.manual_seed(seed)
    cfg = uni_omr_v2.UniOMRv2Config(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_axial_layers=1,
        n_global_layers=1,
        n_dec_layers=2,
        ffn_mult=2.0,
        axial_ffn_mult=2.0,
        dropout=0.0,
        backbone_channels=(16, 24, 32, 48),
        backbone_blocks=(1, 1, 1, 1),
        backbone_expand=2.0,
        router_hidden=32,
        max_mem_tokens=64,
        n_bg_summary=4,
        mtp_k=2,
        irope_nope_every=2,
        n_register_tokens=2,
        use_sys_y_aux=False,
        use_sys_count_aux=False,
        use_staff_seg_aux=False,
        use_stroke_hog_head=False,
    )
    model = uni_omr_v2.UniOMRv2Net(cfg).eval()
    return model, cfg


def load_v2_model(
    ckpt_path: Optional[Path] = None,
    preset: Optional[str] = None,
    tiny_test: bool = False,
) -> Tuple[uni_omr_v2.UniOMRv2Net, uni_omr_v2.UniOMRv2Config, Dict[str, Any]]:
    if tiny_test:
        model, cfg = build_tiny_test_model()
        return model, cfg, {"source": "tiny_test", "vocab": None}

    if ckpt_path is not None:
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg_dict = ck.get("config", {}) if isinstance(ck, dict) else {}
        valid_fields = {f.name for f in fields(uni_omr_v2.UniOMRv2Config)}
        filtered: Dict[str, Any] = {}
        for k, v in cfg_dict.items():
            if k in valid_fields:
                if k in ("backbone_channels", "backbone_blocks", "yoco_cross_banks") and isinstance(v, list):
                    v = tuple(v)
                filtered[k] = v
        cfg = uni_omr_v2.UniOMRv2Config(**filtered)
        model = uni_omr_v2.UniOMRv2Net(cfg)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        inc = model.load_state_dict(sd, strict=False)
        model.strip_aux_heads(strip_mtp=False)
        model.eval()
        return model, cfg, {
            "source": str(ckpt_path),
            "step": ck.get("step") if isinstance(ck, dict) else None,
            "val_ser": ck.get("val_ser") if isinstance(ck, dict) else None,
            "vocab": ck.get("vocab") if isinstance(ck, dict) else None,
            "missing_keys": list(inc.missing_keys),
            "unexpected_keys": list(inc.unexpected_keys),
        }

    if preset == "edge85m":
        cfg = uni_omr_v2.UniOMRv2Config.edge85m()
    else:
        cfg = uni_omr_v2.UniOMRv2Config.small_45m()
    model = uni_omr_v2.UniOMRv2Net(cfg)
    model.strip_aux_heads(strip_mtp=False)
    model.eval()
    return model, cfg, {"source": f"preset:{preset or 'small_45m'}", "vocab": None}


def build_vocab_json_v2(
    cfg: uni_omr_v2.UniOMRv2Config,
    ckpt_vocab: Optional[List[str]] = None,
) -> Dict[str, Any]:
    vocab_size = int(cfg.vocab_size)
    n_base = vocab_size - 4
    base_vocab = list(ckpt_vocab) if ckpt_vocab and len(ckpt_vocab) == n_base else list(data.VOCAB[:n_base])
    pad_id = vocab_size - 4
    bos_id = vocab_size - 3
    eos_id = vocab_size - 2
    sys_id = vocab_size - 1
    full_vocab = base_vocab + ["<pad>", "<bos>", "<eos>", uni_omr_v2.SYS_TOKEN]
    assert len(full_vocab) == vocab_size
    return {
        "arch": "uni_omr_v2",
        "vocab": full_vocab,
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "sys_id": sys_id,
        "vocab_size": vocab_size,
        "max_dim": 1280,
        "d_model": int(cfg.d_model),
        "n_axial_layers": int(cfg.n_axial_layers),
        "n_global_layers": int(cfg.n_global_layers),
        "n_dec_layers": int(cfg.n_dec_layers),
        "n_heads": int(cfg.n_heads),
        "n_kv_heads": int(cfg.n_kv_heads),
        "head_dim": int(cfg.d_model) // int(cfg.n_heads),
        "mtp_k": int(cfg.mtp_k),
        "max_mem_tokens": int(cfg.max_mem_tokens),
    }


# ---------------------------------------------------------------------------
# Target Export Functions
# ---------------------------------------------------------------------------

def _export_dynamic_target(
    model: uni_omr_v2.UniOMRv2Net,
    cfg: uni_omr_v2.UniOMRv2Config,
    target_dir: Path,
    target_name: str,
    opset: int = 17,
    use_chunked_sdpa: bool = False,
    attn_chunk_q: int = 256,
    quantize: bool = True,
    quant_mode: str = "int4",
    block_size: int = 32,
    is_symmetric: bool = False,
    dummy_hw: Tuple[int, int] = (128, 128),
) -> Dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    n_dec = int(cfg.n_dec_layers)
    n_heads = int(cfg.n_heads)
    n_kv_heads = int(cfg.n_kv_heads)
    head_dim = int(cfg.d_model) // n_heads
    n_mtp_heads = len(model.mtp_heads)
    mtp_k = n_mtp_heads + 1

    enc_wrapper = UniOMRv2EncoderONNX(
        model,
        use_chunked_sdpa=use_chunked_sdpa,
        attn_chunk_q=attn_chunk_q,
        static_hw=None,
        static_k_mem=None,
    ).eval()
    dec_wrapper = UniOMRv2DecoderDynamicONNX(model).eval()

    enc_fp32_path = target_dir / "uni_omr_v2_encoder_fp32.onnx"
    dec_fp32_path = target_dir / "uni_omr_v2_decoder_fp32.onnx"
    enc_final_path = target_dir / "uni_omr_v2_encoder.onnx"
    dec_final_path = target_dir / "uni_omr_v2_decoder.onnx"

    # 1. Export Encoder
    dummy_img = torch.rand(1, 3, dummy_hw[0], dummy_hw[1], dtype=torch.float32)
    enc_out_names = [f"cross_k_{i}" for i in range(n_dec)] + [f"cross_v_{i}" for i in range(n_dec)]
    enc_dyn_axes: Dict[str, Dict[int, str]] = {"image": {2: "height", 3: "width"}}
    for name in enc_out_names:
        enc_dyn_axes[name] = {2: "n_mem"}

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

    # 2. Export Dynamic-KV All-T MTP Decoder
    # Trace with past_len=3, step_pos=2, T=4 so both dynamic seq_len and KV slicing stay dynamic!
    bos_id = int(cfg.vocab_size) - 3
    dummy_toks = torch.full((1, 4), bos_id, dtype=torch.long)
    dummy_pos = torch.tensor([2], dtype=torch.long)
    dummy_past_ks = [torch.zeros(1, n_kv_heads, 3, head_dim, dtype=torch.float32) for _ in range(n_dec)]
    dummy_past_vs = [torch.zeros(1, n_kv_heads, 3, head_dim, dtype=torch.float32) for _ in range(n_dec)]
    dummy_cross_ks = [torch.zeros(1, n_kv_heads, 16, head_dim, dtype=torch.float32) for _ in range(n_dec)]
    dummy_cross_vs = [torch.zeros(1, n_kv_heads, 16, head_dim, dtype=torch.float32) for _ in range(n_dec)]

    dec_in_names = (
        ["token_ids", "step_pos"]
        + [f"past_k_{i}" for i in range(n_dec)]
        + [f"past_v_{i}" for i in range(n_dec)]
        + [f"cross_k_{i}" for i in range(n_dec)]
        + [f"cross_v_{i}" for i in range(n_dec)]
    )
    mtp_out_names = [f"mtp_logits_{i}" for i in range(n_mtp_heads)]
    dec_out_names = (
        ["logits"]
        + mtp_out_names
        + [f"present_k_{i}" for i in range(n_dec)]
        + [f"present_v_{i}" for i in range(n_dec)]
    )
    dec_dyn_axes: Dict[str, Dict[int, str]] = {
        "token_ids": {1: "seq_len"},
        "logits": {1: "seq_len"},
    }
    for m_name in mtp_out_names:
        dec_dyn_axes[m_name] = {1: "seq_len"}
    for i in range(n_dec):
        dec_dyn_axes[f"past_k_{i}"] = {2: "past_len"}
        dec_dyn_axes[f"past_v_{i}"] = {2: "past_len"}
        dec_dyn_axes[f"cross_k_{i}"] = {2: "n_mem"}
        dec_dyn_axes[f"cross_v_{i}"] = {2: "n_mem"}
        dec_dyn_axes[f"present_k_{i}"] = {2: "total_len"}
        dec_dyn_axes[f"present_v_{i}"] = {2: "total_len"}

    with torch.no_grad():
        torch.onnx.export(
            dec_wrapper,
            (dummy_toks, dummy_pos, *dummy_past_ks, *dummy_past_vs, *dummy_cross_ks, *dummy_cross_vs),
            str(dec_fp32_path),
            input_names=dec_in_names,
            output_names=dec_out_names,
            dynamic_axes=dec_dyn_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    # 3. Quantization
    quant_format = "fp32"
    if quantize:
        if quant_mode == "int4":
            _quantize_webgpu_int4(
                enc_fp32_path,
                enc_final_path,
                block_size=block_size,
                is_symmetric=is_symmetric,
                compress_conv_int8=True,
            )
            _quantize_webgpu_int4(
                dec_fp32_path,
                dec_final_path,
                block_size=block_size,
                is_symmetric=is_symmetric,
                compress_conv_int8=False,
            )
            quant_format = f"int4_matmul_nbits_b{block_size}"
            enc_fp32_path.unlink(missing_ok=True)
            dec_fp32_path.unlink(missing_ok=True)
        elif quant_mode == "fp16":
            ok_e = _convert_fp16_mixed(enc_fp32_path, enc_final_path)
            ok_d = _convert_fp16_mixed(dec_fp32_path, dec_final_path)
            if ok_e and ok_d:
                quant_format = "fp16_mixed"
                enc_fp32_path.unlink(missing_ok=True)
                dec_fp32_path.unlink(missing_ok=True)
            else:
                enc_fp32_path.replace(enc_final_path)
                dec_fp32_path.replace(dec_final_path)
        elif quant_mode == "int8_dynamic":
            _quantize_cpu_dynamic_int8(enc_fp32_path, enc_final_path, op_types=("MatMul", "Gemm", "Conv"))
            _quantize_cpu_dynamic_int8(dec_fp32_path, dec_final_path, op_types=("MatMul", "Gemm"))
            quant_format = "int8_dynamic_quint8"
            enc_fp32_path.unlink(missing_ok=True)
            dec_fp32_path.unlink(missing_ok=True)
        else:
            enc_fp32_path.replace(enc_final_path)
            dec_fp32_path.replace(dec_final_path)
    else:
        enc_fp32_path.replace(enc_final_path)
        dec_fp32_path.replace(dec_final_path)

    enc_bytes = enc_final_path.stat().st_size
    dec_bytes = dec_final_path.stat().st_size
    return {
        "target": target_name,
        "encoder_file": f"{target_name}/{enc_final_path.name}",
        "decoder_file": f"{target_name}/{dec_final_path.name}",
        "encoder_bytes": enc_bytes,
        "decoder_bytes": dec_bytes,
        "total_bytes": enc_bytes + dec_bytes,
        "d_model": int(cfg.d_model),
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "n_dec_layers": n_dec,
        "mtp_k": mtp_k,
        "static_hw": None,
        "static_k_mem": None,
        "static_dec_t": None,
        "kv_max_len": None,
        "kv_mode": "dynamic_slice",
        "quant_format": quant_format,
        "chunked_sdpa_q": attn_chunk_q if use_chunked_sdpa else None,
    }


def _export_npu_target(
    model: uni_omr_v2.UniOMRv2Net,
    cfg: uni_omr_v2.UniOMRv2Config,
    target_dir: Path,
    opset: int = 17,
    npu_hw: Tuple[int, int] = (1280, 960),
    npu_k_mem: int = 2048,
    npu_kv_max_len: int = 2048,
    npu_dec_t: int = 1,
    quantize: bool = True,
) -> Dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    h_st, w_st = int(npu_hw[0]), int(npu_hw[1])
    if h_st % 16 != 0 or w_st % 16 != 0:
        raise ValueError(f"--npu-hw must be multiples of 16, got {h_st}x{w_st}")

    n_dec = int(cfg.n_dec_layers)
    n_heads = int(cfg.n_heads)
    n_kv_heads = int(cfg.n_kv_heads)
    head_dim = int(cfg.d_model) // n_heads
    n_mtp_heads = len(model.mtp_heads)
    mtp_k = n_mtp_heads + 1

    # Compute exact static k_mem produced by encoder for this bucket.
    n_grid = (h_st // 16) * (w_st // 16)
    n_bg = int(model.encoder.router.n_bg_summary)
    eff_k_kept = max(1, min(n_grid - 1 if n_grid > 1 else 1, int(npu_k_mem) - n_bg))
    eff_k_mem = eff_k_kept + n_bg

    enc_wrapper = UniOMRv2EncoderONNX(
        model,
        use_chunked_sdpa=False,
        static_hw=(h_st, w_st),
        static_k_mem=eff_k_mem,
    ).eval()
    dec_wrapper = UniOMRv2DecoderNPUONNX(
        model,
        kv_max_len=npu_kv_max_len,
        static_t=npu_dec_t,
    ).eval()

    enc_fp32_path = target_dir / "uni_omr_v2_encoder_fp32.onnx"
    dec_fp32_path = target_dir / "uni_omr_v2_decoder_fp32.onnx"
    enc_final_path = target_dir / "uni_omr_v2_encoder.onnx"
    dec_final_path = target_dir / "uni_omr_v2_decoder.onnx"

    # 1. Export 100% Static Encoder (NO dynamic_axes!)
    dummy_img = torch.rand(1, 3, h_st, w_st, dtype=torch.float32)
    enc_out_names = [f"cross_k_{i}" for i in range(n_dec)] + [f"cross_v_{i}" for i in range(n_dec)]
    with torch.no_grad():
        torch.onnx.export(
            enc_wrapper,
            (dummy_img,),
            str(enc_fp32_path),
            input_names=["image"],
            output_names=enc_out_names,
            dynamic_axes=None,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    # 2. Export 100% Static Ring-Buffer KV Decoder (NO dynamic_axes!)
    bos_id = int(cfg.vocab_size) - 3
    dummy_toks = torch.full((1, npu_dec_t), bos_id, dtype=torch.long)
    dummy_pos = torch.tensor([0], dtype=torch.long)
    dummy_past_ks = [torch.zeros(1, n_kv_heads, npu_kv_max_len, head_dim, dtype=torch.float32) for _ in range(n_dec)]
    dummy_past_vs = [torch.zeros(1, n_kv_heads, npu_kv_max_len, head_dim, dtype=torch.float32) for _ in range(n_dec)]
    dummy_cross_ks = [torch.zeros(1, n_kv_heads, eff_k_mem, head_dim, dtype=torch.float32) for _ in range(n_dec)]
    dummy_cross_vs = [torch.zeros(1, n_kv_heads, eff_k_mem, head_dim, dtype=torch.float32) for _ in range(n_dec)]

    dec_in_names = (
        ["token_ids", "step_pos"]
        + [f"past_k_{i}" for i in range(n_dec)]
        + [f"past_v_{i}" for i in range(n_dec)]
        + [f"cross_k_{i}" for i in range(n_dec)]
        + [f"cross_v_{i}" for i in range(n_dec)]
    )
    mtp_out_names = [f"mtp_logits_{i}" for i in range(n_mtp_heads)]
    dec_out_names = (
        ["logits"]
        + mtp_out_names
        + [f"present_k_{i}" for i in range(n_dec)]
        + [f"present_v_{i}" for i in range(n_dec)]
    )
    with torch.no_grad():
        torch.onnx.export(
            dec_wrapper,
            (dummy_toks, dummy_pos, *dummy_past_ks, *dummy_past_vs, *dummy_cross_ks, *dummy_cross_vs),
            str(dec_fp32_path),
            input_names=dec_in_names,
            output_names=dec_out_names,
            dynamic_axes=None,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    # 3. Quantize with standard opset-17 DequantizeLinear (zero MatMulNBits contrib ops!)
    quant_format = "fp32"
    if quantize:
        _quantize_npu_int8_dequant(enc_fp32_path, enc_final_path)
        _quantize_npu_int8_dequant(dec_fp32_path, dec_final_path)
        quant_format = "int8_dequantize_linear"
        enc_fp32_path.unlink(missing_ok=True)
        dec_fp32_path.unlink(missing_ok=True)
    else:
        enc_fp32_path.replace(enc_final_path)
        dec_fp32_path.replace(dec_final_path)

    enc_bytes = enc_final_path.stat().st_size
    dec_bytes = dec_final_path.stat().st_size
    return {
        "target": "npu",
        "encoder_file": f"npu/{enc_final_path.name}",
        "decoder_file": f"npu/{dec_final_path.name}",
        "encoder_bytes": enc_bytes,
        "decoder_bytes": dec_bytes,
        "total_bytes": enc_bytes + dec_bytes,
        "d_model": int(cfg.d_model),
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "n_dec_layers": n_dec,
        "mtp_k": mtp_k,
        "static_hw": [h_st, w_st],
        "static_k_mem": eff_k_mem,
        "static_dec_t": int(npu_dec_t),
        "kv_max_len": int(npu_kv_max_len),
        "kv_mode": "static_ring_buffer",
        "quant_format": quant_format,
    }


def parse_hw(spec: str) -> Tuple[int, int]:
    parts = spec.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid HW spec '{spec}' (expected HxW, e.g. 1280x960)")
    return int(parts[0]), int(parts[1])


def parse_targets(spec: str) -> List[str]:
    raw = spec.strip().lower()
    if raw == "all":
        return list(VALID_TARGETS)
    out: List[str] = []
    for item in raw.split(","):
        t = item.strip()
        if not t:
            continue
        if t not in VALID_TARGETS:
            raise ValueError(f"Unknown target '{t}' (valid: {', '.join(VALID_TARGETS)}, all)")
        if t not in out:
            out.append(t)
    if not out:
        raise ValueError("At least one target must be specified")
    return out


def export_all_targets(
    model: uni_omr_v2.UniOMRv2Net,
    cfg: uni_omr_v2.UniOMRv2Config,
    out_dir: Path,
    targets: Sequence[str] = VALID_TARGETS,
    opset: int = 17,
    quantize: bool = True,
    attn_chunk_q: int = 256,
    block_size: int = 32,
    is_symmetric: bool = False,
    fp16: bool = False,
    cpu_quant: str = "int8_dynamic",
    npu_hw: Tuple[int, int] = (1280, 960),
    npu_k_mem: int = 2048,
    npu_kv_max_len: int = 2048,
    npu_dec_t: int = 1,
    dummy_hw: Tuple[int, int] = (128, 128),
    ckpt_vocab: Optional[List[str]] = None,
) -> Dict[str, Any]:
    _assert_no_unsupported_iter15_flags(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    target_entries: Dict[str, Any] = {}
    for target in targets:
        t0 = time.perf_counter()
        if target == "webgpu":
            entry = _export_dynamic_target(
                model,
                cfg,
                out_dir / "webgpu",
                target_name="webgpu",
                opset=opset,
                use_chunked_sdpa=False,
                attn_chunk_q=attn_chunk_q,
                quantize=quantize,
                quant_mode="fp16" if fp16 else "int4",
                block_size=block_size,
                is_symmetric=is_symmetric,
                dummy_hw=dummy_hw,
            )
        elif target == "npu":
            entry = _export_npu_target(
                model,
                cfg,
                out_dir / "npu",
                opset=opset,
                npu_hw=npu_hw,
                npu_k_mem=npu_k_mem,
                npu_kv_max_len=npu_kv_max_len,
                npu_dec_t=npu_dec_t,
                quantize=quantize,
            )
        elif target == "cpu":
            entry = _export_dynamic_target(
                model,
                cfg,
                out_dir / "cpu",
                target_name="cpu",
                opset=opset,
                use_chunked_sdpa=True,
                attn_chunk_q=attn_chunk_q,
                quantize=quantize,
                quant_mode=cpu_quant,
                block_size=block_size,
                is_symmetric=is_symmetric,
                dummy_hw=dummy_hw,
            )
        else:
            raise ValueError(f"Unsupported target: {target}")
        dt = time.perf_counter() - t0
        print(
            f"[export-v2] target={target} quant={entry['quant_format']} "
            f"enc={entry['encoder_bytes'] / (1024 * 1024):.2f}MB "
            f"dec={entry['decoder_bytes'] / (1024 * 1024):.2f}MB ({dt:.1f}s)",
            flush=True,
        )
        target_entries[target] = entry

    vocab_meta = build_vocab_json_v2(cfg, ckpt_vocab=ckpt_vocab)
    vocab_path = out_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab_meta, indent=2) + "\n", encoding="utf-8")

    manifest: Dict[str, Any] = {
        "schema_version": 2,
        "arch": "uni_omr_v2",
        "opset": int(opset),
        "vocab_file": "vocab.json",
        "model": {
            "d_model": int(cfg.d_model),
            "n_heads": int(cfg.n_heads),
            "n_kv_heads": int(cfg.n_kv_heads),
            "head_dim": int(cfg.d_model) // int(cfg.n_heads),
            "n_axial_layers": int(cfg.n_axial_layers),
            "n_global_layers": int(cfg.n_global_layers),
            "n_dec_layers": int(cfg.n_dec_layers),
            "vocab_size": int(cfg.vocab_size),
            "mtp_k": int(cfg.mtp_k),
            "max_mem_tokens": int(cfg.max_mem_tokens),
        },
        "targets": target_entries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export UniOMR v2 to WebGPU, NPU, and CPU ONNX targets.")
    parser.add_argument("--ckpt", type=Path, default=None, help="Path to UniOMRv2Net checkpoint (.pt).")
    parser.add_argument("--preset", choices=("small_45m", "edge85m"), default="small_45m")
    parser.add_argument("--tiny-test", action="store_true", help="Export tiny UniOMRv2Net for fast CPU verification.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--targets", type=str, default="all", help="Comma-separated targets: webgpu,npu,cpu or all.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-quantize", action="store_true", help="Export pure FP32 models (for parity testing).")
    parser.add_argument("--attn-chunk-q", type=int, default=256, help="Q-chunk size for CPU/WASM tiled SDPA.")
    parser.add_argument("--nbits-block-size", type=int, default=32, help="Block size for WebGPU INT4 MatMulNBits.")
    parser.add_argument("--nbits-symmetric", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision (with FP32 norms/softmax) for WebGPU.")
    parser.add_argument("--cpu-quant", choices=("int8_dynamic", "int4", "fp32"), default="int8_dynamic")
    parser.add_argument("--npu-hw", type=str, default="1280x960", help="Static HxW bucket for NPU encoder.")
    parser.add_argument("--npu-k-mem", type=int, default=2048, help="Static memory token count for NPU encoder.")
    parser.add_argument("--npu-kv-max-len", type=int, default=2048, help="Static KV ring-buffer length for NPU decoder.")
    parser.add_argument("--npu-dec-t", type=int, default=1, help="Static decode step length T for NPU decoder (1 or 4).")
    args = parser.parse_args()

    targets = parse_targets(args.targets)
    npu_hw = parse_hw(args.npu_hw)
    model, cfg, meta = load_v2_model(ckpt_path=args.ckpt, preset=args.preset, tiny_test=args.tiny_test)
    print(
        f"[export-v2] source={meta['source']} d_model={cfg.d_model} "
        f"heads={cfg.n_heads}Q/{cfg.n_kv_heads}KV dec={cfg.n_dec_layers} "
        f"mtp_k={cfg.mtp_k} targets={targets}",
        flush=True,
    )
    export_all_targets(
        model=model,
        cfg=cfg,
        out_dir=args.out_dir,
        targets=targets,
        opset=args.opset,
        quantize=(not args.no_quantize),
        attn_chunk_q=args.attn_chunk_q,
        block_size=args.nbits_block_size,
        is_symmetric=args.nbits_symmetric,
        fp16=args.fp16,
        cpu_quant=args.cpu_quant,
        npu_hw=npu_hw,
        npu_k_mem=args.npu_k_mem,
        npu_kv_max_len=args.npu_kv_max_len,
        npu_dec_t=args.npu_dec_t,
        ckpt_vocab=meta.get("vocab"),
    )


if __name__ == "__main__":
    main()
