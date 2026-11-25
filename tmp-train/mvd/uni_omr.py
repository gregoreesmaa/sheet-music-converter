"""UniOMR — True Unified Full-Page OMR Experiment Harness.

Single-model architecture capable of reading both single-staff strips
(e.g. 256x1024) and full pages (e.g. 1024x768, 1280x960) natively without
fixed-resolution distortion, designed to fit < 55 MB quantized (INT8/INT4)
for client-side in-browser WASM/WebGPU execution.

Key components:
  - ``DynamicConvBackbone``: Stride-16 GroupNorm residual conv stem + stages
    accepting arbitrary ``(B, 3, H, W)`` inputs.
  - ``RoPE2D``: 2D Rotary Position Embeddings on the native ``(H', W')``
    feature grid, preserving aspect ratio and spatial coordinates.
  - ``InkTokenRouter``: Lightweight conv/MLP saliency router (optionally
    guided by ink darkness ``1 - luma``) that keeps top-k ink-rich visual
    tokens (``keep_ratio=0.40``) and pools background paper tokens into 4
    global summary tokens (~2.5x fewer visual tokens).
  - ``UVFlowHead``: Auxiliary 3-channel ``(u, v, valid)`` dewarping head
    supervised by ``photo-degrade --export-masks`` UV maps during training,
    completely skipped at inference (0% inference cost).
  - ``MTPDecoder``: Causal Transformer decoder over ``data.VOCAB`` + ``<sys>``
    with Multi-Token Prediction (``mtp_k=4``) for richer training signal and
    optional speculative 4-token decoding (0% extra cost in standard decode).
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
import contextlib
import inspect
import io
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_ckpt
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from torch.nn.attention import SDPBackend as _SDPBackend, sdpa_kernel as _sdpa_kernel
    _SDPA_HAS_SET_PRIORITY: bool = "set_priority" in inspect.signature(_sdpa_kernel).parameters
    # Prefer EFFICIENT_ATTENTION (CUTLASS) over CUDNN_ATTENTION: cuDNN 9.10 SDPA
    # compiles a new execution plan on CPU (~640 ms/shape) whenever (mem, T) changes,
    # causing a 49x slowdown under dynamic routing + multi-scale + variable seq len.
    _CUDA_SDPA_BACKENDS = [
        _SDPBackend.EFFICIENT_ATTENTION,
        _SDPBackend.MATH,
    ]
except Exception:
    _SDPBackend = None  # type: ignore[assignment]
    _sdpa_kernel = None  # type: ignore[assignment]
    _SDPA_HAS_SET_PRIORITY = False
    _CUDA_SDPA_BACKENDS = []

# Ensure sibling imports (e.g. ``data.py``) resolve regardless of CWD.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import data  # noqa: E402

# ---------------------------------------------------------------------------
# Vocabulary: preserves all data.VOCAB + PAD/BOS/EOS IDs verbatim, and appends
# <sys> (multi-system full-page separator) at the end.
# ---------------------------------------------------------------------------

PAD_ID: int = data.PAD_ID
BOS_ID: int = data.BOS_ID
EOS_ID: int = data.EOS_ID
SYS_TOKEN: str = "<sys>"
SYS_ID: int = len(data.VOCAB) + 3
VOCAB_SIZE: int = len(data.VOCAB) + 4

UNI_TOKEN_TO_ID: Dict[str, int] = dict(data.TOKEN_TO_ID)
UNI_TOKEN_TO_ID[SYS_TOKEN] = SYS_ID
UNI_ID_TO_TOKEN: Dict[int, str] = {i: t for i, t in enumerate(data.VOCAB)}
UNI_ID_TO_TOKEN[PAD_ID] = "<pad>"
UNI_ID_TO_TOKEN[BOS_ID] = "<bos>"
UNI_ID_TO_TOKEN[EOS_ID] = "<eos>"
UNI_ID_TO_TOKEN[SYS_ID] = SYS_TOKEN


def encode_uni(tokens: Sequence[str], add_bos_eos: bool = True) -> List[int]:
    """Encode a bekern + ``<sys>`` token sequence into integer IDs."""
    ids = [UNI_TOKEN_TO_ID[t] for t in tokens]
    if add_bos_eos:
        return [BOS_ID] + ids + [EOS_ID]
    return ids


def decode_uni(ids: Sequence[int], skip_special: bool = True) -> List[str]:
    """Decode integer IDs back into bekern + ``<sys>`` tokens."""
    out: List[str] = []
    for idx in ids:
        i = int(idx)
        if skip_special and i in (PAD_ID, BOS_ID, EOS_ID):
            if i == EOS_ID:
                break
            continue
        tok = UNI_ID_TO_TOKEN.get(i)
        if tok is not None:
            out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def normalize_backbone_stride(stride: Any) -> Any:
    """Normalize backbone stride to ``8``, ``16``, or ``'4x8'``."""
    if isinstance(stride, int):
        if stride in (8, 16):
            return stride
        raise ValueError(f"Unsupported backbone_stride={stride!r}; expected 8, 16, or '4x8'")
    s = str(stride).strip().lower()
    if s in ("8", "8x8"):
        return 8
    if s in ("16", "16x16"):
        return 16
    if s == "4x8":
        return "4x8"
    raise ValueError(f"Unsupported backbone_stride={stride!r}; expected 8, 16, or '4x8'")


@dataclass
class UniOMRConfig:
    d_model: int = 384
    n_enc_layers: int = 4
    n_dec_layers: int = 6
    n_heads: int = 6
    ffn_mult: float = 4.0
    dropout: float = 0.1
    vocab_size: int = VOCAB_SIZE
    keep_ratio: float = 0.40
    dynamic_routing: bool = False
    min_keep_ratio: float = 0.15
    max_keep_ratio: float = 0.75
    saliency_thr: float = 0.05
    n_bg_summary: int = 4
    use_ink_prior: bool = True
    use_uv_aux: bool = True
    backbone_stride: Any = 16
    mixstyle_prob: float = 0.0
    tok_dropout: float = 0.0
    mtp_k: int = 4
    max_dec_len: int = 4096
    rope_base: float = 10000.0
    grad_checkpoint: bool = False

    def __post_init__(self) -> None:
        self.backbone_stride = normalize_backbone_stride(self.backbone_stride)
        self.mixstyle_prob = float(max(0.0, min(1.0, self.mixstyle_prob)))


# ---------------------------------------------------------------------------
# Normalization & Feed-Forward Building Blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (self.weight.shape[0],), self.weight, self.eps)
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, d_ff * 2, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(self.drop(F.silu(gate) * up))


# ---------------------------------------------------------------------------
# 1D & 2D Rotary Position Embeddings (RoPE1D / RoPE2D)
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings.

    q, k: ``(B, H, L, D)``
    cos, sin: ``(B, 1, L, D)`` or ``(1, 1, L, D)``
    """
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class RoPE1D(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: Dict[Tuple[torch.device, torch.dtype], Tuple[int, torch.Tensor, torch.Tensor]] = {}

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype)
        entry = self._cache.get(key)
        if entry is None or entry[0] < seq_len:
            pos = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(pos, self.inv_freq.to(device))  # (L, D/2)
            emb = torch.cat([freqs, freqs], dim=-1)  # (L, D)
            cos = emb.cos().to(dtype)[None, None, :, :]
            sin = emb.sin().to(dtype)[None, None, :, :]
            self._cache[key] = (seq_len, cos, sin)
            return cos, sin
        _, cos, sin = entry
        return cos[:, :, :seq_len, :], sin[:, :, :seq_len, :]


class RoPE2D(nn.Module):
    """2D Rotary Position Embeddings over arbitrary ``(y, x)`` coordinates.

    Splits ``head_dim`` equally between Y (row) and X (col) frequencies so
    arbitrary aspect ratios and sparse routed visual tokens retain exact 2D
    spatial geometry.
    """

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"RoPE2D requires head_dim % 4 == 0, got {head_dim}")
        half_dim = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2, dtype=torch.float32) / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, coords: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 2D rotary cos/sin tables.

        Args:
            coords: ``(B, L, 2)`` float tensor of ``(y, x)`` coordinates.
            dtype: target tensor dtype.

        Returns:
            ``cos, sin`` each of shape ``(B, 1, L, head_dim)``.
        """
        inv_f = self.inv_freq.to(coords.device, dtype=torch.float32)  # (D/4,)
        y = coords[..., 0].to(torch.float32)  # (B, L)
        x = coords[..., 1].to(torch.float32)  # (B, L)

        fy = y.unsqueeze(-1) * inv_f  # (B, L, D/4)
        fx = x.unsqueeze(-1) * inv_f  # (B, L, D/4)

        # Interleave half-blocks to match _rotate_half: [fy, fx, fy, fx]
        emb_half = torch.cat([fy, fx], dim=-1)  # (B, L, D/2)
        emb = torch.cat([emb_half, emb_half], dim=-1)  # (B, L, D)

        cos = emb.cos().to(dtype).unsqueeze(1)  # (B, 1, L, D)
        sin = emb.sin().to(dtype).unsqueeze(1)  # (B, 1, L, D)
        return cos, sin


# ---------------------------------------------------------------------------
# Native Dynamic Resolution Conv Stem + Stages (stride 16)
# ---------------------------------------------------------------------------

class ResConvBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 16) -> None:
        super().__init__()
        g = min(groups, channels)
        while channels % g != 0:
            g -= 1
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(g, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return F.silu(x + h)


class ConvDownStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_blocks: int, groups: int = 16, stride: Any = 2) -> None:
        super().__init__()
        g = min(groups, out_ch)
        while out_ch % g != 0:
            g -= 1
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResConvBlock(out_ch, groups=g) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class MixStyle2d(nn.Module):
    """Lightweight early-layer MixStyle (0 extra params, 0% inference cost).

    During training with ``p > 0`` and ``B >= 2``, mixes per-channel instance
    mean/std across shuffled batch items using ``Beta(alpha, alpha)`` weights
    to improve OOD phone-photo and handwriting generalization.
    """

    def __init__(self, p: float = 0.0, alpha: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = float(max(0.0, min(1.0, p)))
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0.0:
            return x
        B = x.shape[0]
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        sig = (var + self.eps).sqrt()
        mu_d, sig_d = mu.detach(), sig.detach()

        if B < 2:
            c = int(x.shape[1])
            cache = getattr(self, "_prev_stats", None)
            if cache is None:
                cache = {}
                self._prev_stats = cache
            prev = cache.get(c)
            cache[c] = (mu_d.clone(), sig_d.clone())
            if prev is None or prev[0].device != mu_d.device or random.random() >= self.p:
                return x
            lmda = (
                torch.distributions.Beta(self.alpha, self.alpha)
                .sample((1, 1, 1, 1))
                .to(device=x.device, dtype=x.dtype)
            )
            mu_mix = mu_d * lmda + prev[0].to(dtype=x.dtype) * (1.0 - lmda)
            sig_mix = sig_d * lmda + prev[1].to(dtype=x.dtype) * (1.0 - lmda)
            return ((x - mu_d) / sig_d) * sig_mix + mu_mix

        if random.random() >= self.p:
            return x

        x_normed = (x - mu_d) / sig_d

        lmda = (
            torch.distributions.Beta(self.alpha, self.alpha)
            .sample((B, 1, 1, 1))
            .to(device=x.device, dtype=x.dtype)
        )
        perm = torch.randperm(B, device=x.device)
        mu_mix = mu_d * lmda + mu_d[perm] * (1.0 - lmda)
        sig_mix = sig_d * lmda + sig_d[perm] * (1.0 - lmda)
        return x_normed * sig_mix + mu_mix


class DynamicConvBackbone(nn.Module):
    """Dynamic-resolution convolutional feature extractor (stride 16, 8, or '4x8')."""

    STRIDE: int = 16

    def __init__(self, d_model: int = 384, stride: Any = 16, mixstyle_prob: float = 0.0) -> None:
        super().__init__()
        self.stride = normalize_backbone_stride(stride)
        s3_stride: Any = (1, 2) if self.stride == "4x8" else 2
        s4_stride: Any = 2 if self.stride == 16 else 1
        self.stage1 = ConvDownStage(3, 64, n_blocks=1, stride=2)             # /2
        self.stage2 = ConvDownStage(64, 128, n_blocks=2, stride=2)           # /4
        self.stage3 = ConvDownStage(128, 256, n_blocks=3, stride=s3_stride)  # /8 or /(4,8)
        self.stage4 = ConvDownStage(256, 512, n_blocks=2, stride=s4_stride)  # /16, /8, or /(4,8)
        self.mixstyle = MixStyle2d(p=mixstyle_prob, alpha=0.1)
        self.proj = nn.Conv2d(512, d_model, kernel_size=1, bias=False)
        self.out_norm = nn.GroupNorm(16 if d_model % 16 == 0 else 8, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mixstyle(self.stage1(x))
        h = self.mixstyle(self.stage2(h))
        h = self.stage3(h)
        h = self.stage4(h)
        return self.out_norm(self.proj(h))


# ---------------------------------------------------------------------------
# Ink-Saliency Visual Token Router
# ---------------------------------------------------------------------------

class InkTokenRouter(nn.Module):
    """Selects top-k ink-rich visual tokens and pools background into summary tokens.

    - Deterministic top-k selection sorted in row-major reading order.
    - Differentiable gating on kept tokens + soft background pooling so
      gradients flow cleanly to both the router and backbone.
    - Preserves exact 2D ``(y, x)`` coordinates for downstream ``RoPE2D``.
    """

    def __init__(
        self,
        d_model: int,
        keep_ratio: float = 0.40,
        n_bg_summary: int = 4,
        use_ink_prior: bool = True,
        dynamic_routing: bool = False,
        min_keep_ratio: float = 0.15,
        max_keep_ratio: float = 0.75,
        saliency_thr: float = 0.05,
    ) -> None:
        super().__init__()
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        self.d_model = d_model
        self.keep_ratio = float(keep_ratio)
        self.dynamic_routing = bool(dynamic_routing)
        self.min_keep_ratio = float(min_keep_ratio)
        self.max_keep_ratio = float(max_keep_ratio)
        self.saliency_thr = float(saliency_thr)
        self.n_bg_summary = int(n_bg_summary)
        self.use_ink_prior = bool(use_ink_prior)

        hidden = max(64, d_model // 4)
        self.saliency_net = nn.Sequential(
            nn.Conv2d(d_model, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        # Learnable weight on the local high-pass ink prior (bounded via sigmoid).
        self.ink_prior_scale = nn.Parameter(torch.tensor(1.0))

        # Learned queries that pool background paper tokens into n_bg_summary tokens.
        self.bg_queries = nn.Parameter(torch.randn(1, n_bg_summary, d_model) * 0.02)
        self.bg_key_proj = nn.Linear(d_model, d_model, bias=False)
        self.bg_val_proj = nn.Linear(d_model, d_model, bias=False)
        self.bg_out_norm = RMSNorm(d_model)
        self._grid_cache: Dict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}

    def reset_saliency(self) -> None:
        """Re-initialize saliency_net weights and reset ink_prior_scale to 1.0."""
        for m in self.saliency_net.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        with torch.no_grad():
            self.ink_prior_scale.fill_(1.0)

    @staticmethod
    def _compute_ink_and_paper_gate(
        images: torch.Tensor, h_p: int, w_p: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute border-masked relative high-pass ink map, soft paper_gate, and effective_paper.

        Suppresses dark exterior desk/scanner border rings (``local_bg <= 0.35``)
        and erodes ``paper_mask_grid`` by 2 patches (5x5 max-pool dilation of
        non-paper) to eliminate the 2-patch (~32px) bright-paper-to-dark-desk
        transition halo both in ``local_ink_grid`` (before ``q99`` normalization)
        and via a detached soft ``paper_gate`` (``1.0`` on interior paper, ``0.05``
        outside), while normalizing by local paper brightness so faint brown
        iron-gall ink on warm parchment remains strongly salient.
        """
        # Standard Rec.601 luma for RGB in [0, 1], computed in float32 for precision.
        img_f = images.float()
        if img_f.shape[1] == 3:
            luma = (
                0.299 * img_f[:, 0:1]
                + 0.587 * img_f[:, 1:2]
                + 0.114 * img_f[:, 2:3]
            )
        else:
            luma = img_f.mean(dim=1, keepdim=True)
        local_bg = F.avg_pool2d(
            F.avg_pool2d(luma, kernel_size=(1, 31), stride=1, padding=(0, 15), count_include_pad=False),
            kernel_size=(31, 1),
            stride=1,
            padding=(15, 0),
            count_include_pad=False,
        )
        paper_mask = (local_bg > 0.35).to(luma.dtype)
        local_ink = ((local_bg - luma) / local_bg.clamp(min=0.15)).clamp(0.0, 1.0) * paper_mask
        ink_max = F.adaptive_max_pool2d(local_ink, (h_p, w_p))
        ink_avg = F.adaptive_avg_pool2d(local_ink, (h_p, w_p))
        local_ink_grid = 0.5 * ink_max + 0.5 * ink_avg

        local_bg_grid = F.adaptive_avg_pool2d(local_bg, (h_p, w_p))
        paper_mask_grid = (local_bg_grid > 0.35).detach()
        non_paper = (~paper_mask_grid).to(torch.float32)
        border_dilated = F.max_pool2d(non_paper, kernel_size=5, stride=1, padding=2) > 0.5
        interior_paper = paper_mask_grid & (~border_dilated)
        # Fallback if entire image is dark/eroded away:
        has_interior = interior_paper.flatten(1).any(dim=1).view(-1, 1, 1, 1)
        effective_paper = torch.where(has_interior, interior_paper, paper_mask_grid).detach()

        local_ink_grid = local_ink_grid * effective_paper.to(local_ink_grid.dtype)
        paper_gate = torch.where(
            effective_paper,
            torch.ones_like(local_bg_grid),
            torch.full_like(local_bg_grid, 0.05),
        ).detach()
        return local_ink_grid, paper_gate, effective_paper

    @staticmethod
    def _compute_ink_darkness(images: torch.Tensor, h_p: int, w_p: int) -> torch.Tensor:
        """Compute border-masked relative high-pass ink map ``(B, 1, H', W')``."""
        local_ink_grid, _, _ = InkTokenRouter._compute_ink_and_paper_gate(images, h_p, w_p)
        return local_ink_grid

    def forward(
        self,
        feat_2d: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        keep_ratio_override: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Route 2D feature grid into compact token sequence.

        Args:
            feat_2d: ``(B, D, H', W')`` encoder feature map.
            images: optional ``(B, 3, H, W)`` input images for local high-pass ink prior.
            keep_ratio_override: optional runtime keep_ratio in ``(0, 1]``.

        Returns:
            tokens: ``(B, K + n_bg_summary, D)`` (or ``(B, N, D)`` if ``keep_ratio == 1``).
            coords: ``(B, L_out, 2)`` float ``(y, x)`` coordinates for RoPE2D.
            info: metadata dict (``n_in``, ``n_out``, ``k_kept``, ``kept_idx``, ``saliency_map``,
                ``saliency_logits``, ``ink_target``, ``ink_recall``).
        """
        B, D, H_p, W_p = feat_2d.shape
        N = H_p * W_p
        ratio = self.keep_ratio if keep_ratio_override is None else float(keep_ratio_override)
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {ratio}")

        saliency_logits = self.saliency_net(feat_2d)  # (B, 1, H', W')
        saliency_prob = torch.sigmoid(saliency_logits)

        ink_target: Optional[torch.Tensor] = None
        if images is not None:
            local_ink_grid, paper_gate, paper_mask_grid = self._compute_ink_and_paper_gate(images, H_p, W_p)
            local_ink_grid = local_ink_grid.to(saliency_logits.dtype)
            paper_gate = paper_gate.to(saliency_logits.dtype)
            saliency_prob = saliency_prob * paper_gate
            ink_target = ((local_ink_grid > 0.04) & paper_mask_grid).to(saliency_logits.dtype)
            if self.use_ink_prior:
                flat = local_ink_grid.flatten(1)
                denom = (
                    torch.quantile(flat.float(), 0.99, dim=-1)
                    .clamp(min=1e-4)
                    .view(B, 1, 1, 1)
                    .to(local_ink_grid.dtype)
                )
                local_ink_norm = (local_ink_grid / denom).clamp(0.0, 1.0)
                score = saliency_prob + torch.sigmoid(self.ink_prior_scale).to(saliency_logits.dtype) * local_ink_norm
            else:
                score = saliency_prob
        else:
            score = saliency_prob

        score_flat = score.flatten(1)  # (B, N), bounded in [0, 2]
        tokens_flat = feat_2d.permute(0, 2, 3, 1).reshape(B, N, D)  # (B, N, D) zero-copy for NHWC

        # Build (or reuse cached) 2D spatial coordinate grid (y, x).
        gkey = (H_p, W_p, feat_2d.device)
        cached_grid = self._grid_cache.get(gkey)
        if cached_grid is None:
            ys = torch.arange(H_p, device=feat_2d.device, dtype=torch.float32)
            xs = torch.arange(W_p, device=feat_2d.device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            base_coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1).unsqueeze(0)  # (1, N, 2)
            base_center = torch.tensor(
                [[[(H_p - 1) * 0.5, (W_p - 1) * 0.5]]],
                device=feat_2d.device,
                dtype=torch.float32,
            )  # (1, 1, 2)
            self._grid_cache[gkey] = (base_coords, base_center)
        else:
            base_coords, base_center = cached_grid
        grid_coords = base_coords.expand(B, -1, -1)  # (B, N, 2)

        if self.dynamic_routing and keep_ratio_override is None:
            with torch.no_grad():
                # Threshold relative to per-image paper baseline (median bounded score in [0, 2]) so
                # sparse pages drop to min_keep_ratio (~15%) while dense pages expand up to max_keep_ratio (~75%).
                med = score_flat.median(dim=1, keepdim=True).values
                above = (score_flat > (med + self.saliency_thr)).sum(dim=1)
                k_dyn = int(above.max().item())
                k_min = max(1, int(math.ceil(N * self.min_keep_ratio)))
                k_max = max(k_min, min(N, int(math.ceil(N * self.max_keep_ratio))))
                k = max(k_min, min(k_max, k_dyn))
        else:
            k = max(1, min(N, int(math.ceil(N * ratio))))

        if k >= N:
            gate = (0.5 + 0.5 * torch.sigmoid(score_flat)).unsqueeze(-1)
            out_tokens = tokens_flat * gate
            kept_idx = torch.arange(N, device=feat_2d.device).unsqueeze(0).expand(B, -1)
            info = {
                "n_in": N,
                "n_out": N,
                "k_kept": N,
                "n_bg_summary": 0,
                "kept_idx": kept_idx,
                "saliency_map": score,
                "saliency_logits": saliency_logits,
                "ink_target": ink_target,
                "ink_recall": 1.0,
                "grid_hw": (H_p, W_p),
            }
            return out_tokens, grid_coords, info

        # Deterministic top-k selection, then sort ascending to preserve reading order.
        _, topk_idx = torch.topk(score_flat, k=k, dim=1, largest=True, sorted=False)
        kept_idx, _ = torch.sort(topk_idx, dim=1)  # (B, k)

        kept_scores = torch.gather(score_flat, 1, kept_idx)  # (B, k)
        kept_gate = (0.5 + 0.5 * torch.sigmoid(kept_scores)).unsqueeze(-1)  # (B, k, 1)

        idx_feat = kept_idx.unsqueeze(-1).expand(-1, -1, D)
        kept_tokens = torch.gather(tokens_flat, 1, idx_feat) * kept_gate  # (B, k, D)

        idx_coords = kept_idx.unsqueeze(-1).expand(-1, -1, 2)
        kept_coords = torch.gather(grid_coords, 1, idx_coords)  # (B, k, 2)

        # Pool background tokens into n_bg_summary global summary tokens.
        kept_mask = torch.zeros(B, N, device=feat_2d.device, dtype=torch.bool)
        kept_mask.scatter_(1, kept_idx, True)

        if ink_target is not None and not self.training:
            with torch.no_grad():
                tgt_mask = ink_target.flatten(1) > 0.5  # (B, N)
                n_ink = tgt_mask.sum(dim=1)  # (B,)
                kept_ink = (tgt_mask & kept_mask).sum(dim=1)
                has_ink = n_ink > 0
                if bool(has_ink.any()):
                    per_sample = torch.where(
                        has_ink,
                        kept_ink.float() / n_ink.clamp(min=1).float(),
                        torch.ones_like(n_ink, dtype=torch.float32),
                    )
                    ink_recall = float(per_sample.mean().item())
                else:
                    ink_recall = 1.0
        else:
            ink_recall = 1.0

        bg_q = self.bg_queries.expand(B, -1, -1)  # (B, M, D)
        # Algebraic transpose: Q @ (X @ W_k^T)^T == (Q @ W_k) @ X^T, and (A @ X) @ W_v^T == A @ (X @ W_v^T).
        # Avoids O(N * D^2) projections over all visual tokens while staying mathematically identical.
        bg_q_proj = F.linear(bg_q, self.bg_key_proj.weight.t()).to(tokens_flat.dtype)  # (B, M, D)

        # Soft background affinity biases attention toward non-kept, low-score tokens.
        bg_logits = torch.bmm(bg_q_proj, tokens_flat.transpose(1, 2)) / math.sqrt(D)  # (B, M, N)
        bg_logits = bg_logits - score_flat.unsqueeze(1)  # prefer background
        bg_logits = bg_logits.masked_fill(kept_mask.unsqueeze(1), -1e4)
        bg_attn = F.softmax(bg_logits, dim=-1)
        pooled = torch.bmm(bg_attn, tokens_flat)  # (B, M, D)
        bg_tokens = self.bg_out_norm(self.bg_val_proj(pooled))  # (B, M, D)

        # Assign center-of-page coordinates to background summary tokens.
        center_coord = base_center.expand(B, self.n_bg_summary, 2)

        out_tokens = torch.cat([kept_tokens, bg_tokens], dim=1)  # (B, k + M, D)
        out_coords = torch.cat([kept_coords, center_coord], dim=1)  # (B, k + M, 2)

        info = {
            "n_in": N,
            "n_out": k + self.n_bg_summary,
            "k_kept": k,
            "n_bg_summary": self.n_bg_summary,
            "kept_idx": kept_idx,
            "saliency_map": score,
            "saliency_logits": saliency_logits,
            "ink_target": ink_target,
            "ink_recall": ink_recall,
            "grid_hw": (H_p, W_p),
        }
        return out_tokens, out_coords, info


# ---------------------------------------------------------------------------
# Transformer Encoder (with RoPE2D) & Decoder (with RoPE1D + Cross-Attention)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

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
        is_cross: bool = False,
    ) -> Any:
        B, Lq, _ = q_in.shape
        q = self.q_proj(q_in).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)

        if is_cross and past_kv is not None:
            k, v = past_kv
        else:
            if kv_in is None:
                raise ValueError("kv_in must be provided when past_kv is None")
            Lk = kv_in.shape[1]
            k = self.k_proj(kv_in).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(kv_in).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
            if rope_cos is not None and rope_sin is not None:
                q, k = apply_rope(q, k, rope_cos, rope_sin)
            if not is_cross and past_kv is not None:
                k = torch.cat([past_kv[0], k], dim=2)
                v = torch.cat([past_kv[1], v], dim=2)

        present_kv = (k, v) if use_cache else None

        use_causal_flag = is_causal and (attn_mask is None)
        eff_mask = attn_mask
        if use_causal_flag and not is_cross and past_kv is not None:
            past_len = past_kv[0].shape[2]
            if Lq == 1:
                # Single new query attends to all past + current keys.
                use_causal_flag = False
            else:
                total_k = past_len + Lq
                row_idx = torch.arange(Lq, device=q.device).unsqueeze(1) + past_len
                col_idx = torch.arange(total_k, device=q.device).unsqueeze(0)
                eff_mask = col_idx <= row_idx
                use_causal_flag = False

        # Pass dropout_p=0.0 to SDPA (keeping FFN/residual dropout at cfg.dropout) so
        # cuDNN 9.10 Flash/CUDNN_ATTENTION is never disqualified by attention dropout.
        if q.device.type == "cuda" and _sdpa_kernel is not None:
            if _SDPA_HAS_SET_PRIORITY:
                with _sdpa_kernel(_CUDA_SDPA_BACKENDS, set_priority=True):
                    out = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=eff_mask,
                        dropout_p=0.0,
                        is_causal=use_causal_flag,
                    )
            elif eff_mask is None:
                with _sdpa_kernel(_CUDA_SDPA_BACKENDS):
                    out = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=use_causal_flag,
                    )
            else:
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=eff_mask,
                    dropout_p=0.0,
                    is_causal=use_causal_flag,
                )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=eff_mask,
                dropout_p=0.0,
                is_causal=use_causal_flag,
            )
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.o_proj(out)
        if use_cache:
            return out, present_kv
        return out


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, h, rope_cos=rope_cos, rope_sin=rope_sin))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm3 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor],
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        past_kv: Optional[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ) -> Any:
        h = self.norm1(x)
        if use_cache:
            self_past = past_kv[0] if past_kv is not None else None
            cross_past = past_kv[1] if past_kv is not None else None
            sa_out, new_self_kv = self.self_attn(
                h, h,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                is_causal=True,
                past_kv=self_past,
                use_cache=True,
                is_cross=False,
            )
            x = x + self.drop(sa_out)
            ca_out, new_cross_kv = self.cross_attn(
                self.norm2(x), memory,
                past_kv=cross_past,
                use_cache=True,
                is_cross=True,
            )
            x = x + self.drop(ca_out)
            x = x + self.drop(self.ffn(self.norm3(x)))
            return x, (new_self_kv, new_cross_kv)

        x = x + self.drop(self.self_attn(h, h, rope_cos=rope_cos, rope_sin=rope_sin, is_causal=True))
        x = x + self.drop(self.cross_attn(self.norm2(x), memory))
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x


# ---------------------------------------------------------------------------
# Auxiliary UV-Flow Dewarping Head (0% Inference Cost)
# ---------------------------------------------------------------------------

class UVFlowHead(nn.Module):
    """Predicts dense 3-channel ``(u, v, valid_logit)`` map from encoder features.

    Supervised during training by ``photo-degrade --export-masks`` UV maps;
    completely bypassed when ``use_uv_aux=False`` or in inference mode.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        mid = max(128, d_model // 2)
        self.net = nn.Sequential(
            nn.Conv2d(d_model, mid, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, mid // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid // 2, 3, kernel_size=1, bias=True),
        )

    def forward(self, feat_2d: torch.Tensor) -> torch.Tensor:
        """Returns ``(B, 3, H', W')`` where ``[0:2]`` are sigmoid ``(u, v)`` and ``[2]`` is ``valid_logit``."""
        raw = self.net(feat_2d)
        uv = torch.sigmoid(raw[:, 0:2])
        valid_logit = raw[:, 2:3]
        return torch.cat([uv, valid_logit], dim=1)


def compute_uv_flow_loss(
    uv_pred: torch.Tensor,
    uv_target: torch.Tensor,
) -> torch.Tensor:
    """Compute masked Smooth-L1 loss on ``(u, v)`` + BCE loss on ``valid`` channel.

    Args:
        uv_pred: ``(B, 3, H', W')`` from ``UVFlowHead`` (channels 0..1 in [0,1], ch 2 logit).
        uv_target: ``(B, 3, H_t, W_t)`` float tensor with ``(u, v, valid)`` in ``[0, 1]``.
    """
    if uv_target.shape[-2:] != uv_pred.shape[-2:]:
        uv_target = F.interpolate(
            uv_target.to(uv_pred.dtype),
            size=uv_pred.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    else:
        uv_target = uv_target.to(uv_pred.dtype)

    pred_uv = uv_pred[:, 0:2]
    pred_valid_logit = uv_pred[:, 2:3]
    tgt_uv = uv_target[:, 0:2]
    tgt_valid = uv_target[:, 2:3].clamp(0.0, 1.0)

    valid_mask = (tgt_valid > 0.5).to(uv_pred.dtype)
    denom = valid_mask.sum().clamp(min=1.0) * 2.0
    uv_l1 = (F.smooth_l1_loss(pred_uv, tgt_uv, reduction="none") * valid_mask).sum() / denom
    valid_bce = F.binary_cross_entropy_with_logits(pred_valid_logit, tgt_valid)
    return uv_l1 + 0.5 * valid_bce


def load_uv_flow_mask(path: Path) -> np.ndarray:
    """Load a ``photo-degrade --export-masks`` UV flow mask (``.npy`` or 16-bit PNG).

    Returns a float32 ``(3, H, W)`` array with channels ``(u, v, valid)`` in ``[0, 1]``.
    """
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(p).astype(np.float32)
        if arr.ndim == 3 and arr.shape[2] == 3:
            return np.transpose(arr, (2, 0, 1))
        if arr.ndim == 3 and arr.shape[0] == 3:
            return arr
        raise ValueError(f"Unexpected UV .npy shape {arr.shape} in {p}")
    # 16-bit PNG (or 8-bit fallback)
    img = Image.open(p)
    arr = np.asarray(img)
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, np.ones_like(arr)], axis=-1)
    return np.transpose(arr[..., :3], (2, 0, 1))


# ---------------------------------------------------------------------------
# Multi-Token Prediction (MTP) Head
# ---------------------------------------------------------------------------

class MTPHead(nn.Module):
    """Lightweight parallel Multi-Token Prediction projection head.

    Maps decoder hidden state ``h_t`` to a latent state for predicting token
    ``t + offset`` via the shared ``lm_head`` weights.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + F.silu(self.proj(self.norm(h)))


# ---------------------------------------------------------------------------
# Native Symmetric Group-Wise INT4 QAT (Zero Schema Change)
# ---------------------------------------------------------------------------

def fake_quant_int4_ste(w: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Symmetric group-wise INT4 weight fake-quantization with Straight-Through Estimator (STE).

    Matches ONNX Runtime ``MatMulNBits`` symmetric INT4 (``[-8, 7]`` per group of
    ``block_size`` along ``in_features``), padding non-multiples of ``block_size``
    with zeros before grouping and slicing back to the original shape.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    orig_shape = w.shape
    in_f = orig_shape[-1]
    pad = (block_size - (in_f % block_size)) % block_size
    w_f = w.float()
    if pad:
        w_f = F.pad(w_f, (0, pad))
    w_g = w_f.reshape(-1, block_size)
    scale = (w_g.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=1e-8)
    q = torch.round(w_g / scale).clamp(-8.0, 7.0)
    w_dq = (q * scale).reshape(*orig_shape[:-1], -1)
    if pad:
        w_dq = w_dq[..., :in_f]
    w_dq = w_dq.to(w.dtype)
    return w + (w_dq - w).detach()


class QATInt4Controller:
    """Zero-schema-change symmetric group-wise INT4 QAT controller for ``UniOMRNet``.

    Monkey-patches ``forward`` on Transformer ``nn.Linear`` layers inside
    ``enc_layers``, ``dec_layers``, ``mtp_heads``, and ``lm_head`` (skipping
    ``backbone``, ``router``, ``tok_emb``, ``pos2d_proj``, and norms) without
    adding parameters or buffers, preserving 100% ``state_dict()`` compatibility
    (``strict=True``).
    """

    TARGET_ROOTS: Tuple[str, ...] = ("enc_layers", "dec_layers", "mtp_heads", "lm_head")

    def __init__(self, model: nn.Module, block_size: int = 32, enabled: bool = False) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.model = model
        self.block_size = int(block_size)
        self.enabled = bool(enabled)
        self._orig_forwards: Dict[nn.Linear, Any] = {}
        self.target_names: List[str] = []
        self._attach()

    def _attach(self) -> None:
        for root_name in self.TARGET_ROOTS:
            root_mod = getattr(self.model, root_name, None)
            if root_mod is None:
                continue
            for sub_name, mod in root_mod.named_modules():
                if isinstance(mod, nn.Linear) and mod not in self._orig_forwards:
                    full_name = f"{root_name}.{sub_name}" if sub_name else root_name
                    self._orig_forwards[mod] = mod.forward
                    self.target_names.append(full_name)
                    self._patch_linear(mod)

    def _patch_linear(self, mod: nn.Linear) -> None:
        controller = self

        def _qat_forward(x: torch.Tensor) -> torch.Tensor:
            if controller.enabled:
                w_q = fake_quant_int4_ste(mod.weight, block_size=controller.block_size)
                return F.linear(x, w_q, mod.bias)
            return F.linear(x, mod.weight, mod.bias)

        mod.forward = _qat_forward  # type: ignore[method-assign]

    @property
    def num_layers(self) -> int:
        return len(self.target_names)

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def detach(self) -> None:
        """Restore original ``forward`` methods on all patched ``nn.Linear`` modules."""
        for mod, orig_fwd in self._orig_forwards.items():
            mod.forward = orig_fwd  # type: ignore[method-assign]
        self._orig_forwards.clear()
        self.target_names.clear()


# ---------------------------------------------------------------------------
# UniOMRNet — Full Unified Model
# ---------------------------------------------------------------------------

class UniOMRNet(nn.Module):
    """Unified Full-Page & Single-Staff OMR Network."""

    def __init__(self, cfg: Optional[UniOMRConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else UniOMRConfig()
        c = self.cfg
        d_ff = int(c.d_model * c.ffn_mult)
        head_dim = c.d_model // c.n_heads

        # 1. Native dynamic-resolution conv backbone (stride 16, 8, or 4x8)
        self.backbone = DynamicConvBackbone(
            d_model=c.d_model,
            stride=c.backbone_stride,
            mixstyle_prob=c.mixstyle_prob,
        )

        # 2. Ink-saliency visual token router
        self.router = InkTokenRouter(
            d_model=c.d_model,
            keep_ratio=c.keep_ratio,
            n_bg_summary=c.n_bg_summary,
            use_ink_prior=c.use_ink_prior,
            dynamic_routing=c.dynamic_routing,
            min_keep_ratio=c.min_keep_ratio,
            max_keep_ratio=c.max_keep_ratio,
            saliency_thr=c.saliency_thr,
        )

        # 3. 2D RoPE Transformer Encoder + explicit 2D sinusoidal coords into memory
        #    (RoPE2D only rotates Q/K inside encoder self-attention; adding explicit 2D
        #    Fourier/sinusoidal (y, x) embeddings directly into routed_tokens and mem
        #    ensures decoder cross-attention knows exact reading order (x) & pitch height (y)
        #    of every sparse routed token!)
        self.rope2d = RoPE2D(head_dim=head_dim, base=c.rope_base)
        self.pos2d_proj = nn.Linear(c.d_model, c.d_model, bias=False)
        nn.init.zeros_(self.pos2d_proj.weight)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(c.d_model, c.n_heads, d_ff, dropout=c.dropout)
            for _ in range(c.n_enc_layers)
        ])
        self.enc_norm = RMSNorm(c.d_model)

        # 4. Auxiliary UV-flow head (training-only, 0% infer cost when skipped/stripped)
        self.uv_head: Optional[UVFlowHead] = UVFlowHead(c.d_model) if c.use_uv_aux else None

        # 5. Causal Transformer Decoder with 1D RoPE + MTP-k heads
        pad_idx = PAD_ID if PAD_ID < c.vocab_size else (c.vocab_size - 4 if c.vocab_size >= 4 else None)
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model, padding_idx=pad_idx)
        self.rope1d = RoPE1D(head_dim=head_dim, base=c.rope_base)
        self.dec_layers = nn.ModuleList([
            DecoderLayer(c.d_model, c.n_heads, d_ff, dropout=c.dropout)
            for _ in range(c.n_dec_layers)
        ])
        self.dec_norm = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

        # MTP auxiliary heads for offsets +2 .. +mtp_k (shared lm_head projection)
        n_extra_mtp = max(0, int(c.mtp_k) - 1)
        self.mtp_heads = nn.ModuleList([MTPHead(c.d_model) for _ in range(n_extra_mtp)])
        self._struct_weight_cache: Dict[Tuple[int, float, torch.device, torch.dtype], torch.Tensor] = {}

    def _sinusoidal_pos2d(
        self,
        coords: torch.Tensor,
        dtype: torch.dtype,
        grid_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Build (B, L, D) 2D sinusoidal positional embeddings from normalized (y, x) coords."""
        d = self.cfg.d_model
        q = d // 4
        device = coords.device
        freqs = torch.exp(
            torch.arange(0, q, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(1, q))
        )
        y_raw = coords[..., 0].to(torch.float32)
        x_raw = coords[..., 1].to(torch.float32)
        if grid_hw is not None:
            h_p, w_p = int(grid_hw[0]), int(grid_hw[1])
            y_norm = y_raw / float(max(1, h_p - 1))
            x_norm = x_raw / float(max(1, w_p - 1))
        else:
            y_norm = y_raw
            x_norm = x_raw
        # Scale normalized [0, 1] coords to [0, 128] so multi-scale sin/cos resolve fine pitch & measure x
        y = y_norm * 128.0
        x = x_norm * 128.0
        fy = y.unsqueeze(-1) * freqs
        fx = x.unsqueeze(-1) * freqs
        pe = torch.cat([fy.sin(), fy.cos(), fx.sin(), fx.cos()], dim=-1)
        if pe.shape[-1] < d:
            pe = F.pad(pe, (0, d - pe.shape[-1]))
        return pe.to(dtype)

    def encode_image(
        self,
        images: torch.Tensor,
        keep_ratio_override: Optional[float] = None,
        return_aux: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Encode arbitrary-resolution ``(B, 3, H, W)`` images into routed memory tokens."""
        backbone_in = (
            images.to(memory_format=torch.channels_last)
            if self.backbone.stage1.down[0].weight.is_contiguous(memory_format=torch.channels_last)
            else images
        )
        feat_2d = self.backbone(backbone_in)  # (B, D, H', W')
        routed_tokens, routed_coords, router_info = self.router(
            feat_2d, images=images, keep_ratio_override=keep_ratio_override
        )

        pos2d = self.pos2d_proj(
            self._sinusoidal_pos2d(
                routed_coords,
                dtype=routed_tokens.dtype,
                grid_hw=router_info["grid_hw"],
            )
        )
        cos2d, sin2d = self.rope2d(routed_coords, dtype=routed_tokens.dtype)
        mem = routed_tokens + pos2d
        if (
            self.training
            and self.cfg.grad_checkpoint
            and torch.is_grad_enabled()
            and (mem.shape[0] * mem.shape[1] > 8000)
        ):
            for layer in self.enc_layers:
                mem = _grad_ckpt(
                    lambda _m, _c, _s, _l=layer: _l(_m, rope_cos=_c, rope_sin=_s),
                    mem, cos2d, sin2d,
                    use_reentrant=False,
                )
        else:
            for layer in self.enc_layers:
                mem = layer(mem, rope_cos=cos2d, rope_sin=sin2d)
        mem = self.enc_norm(mem) + pos2d

        # Auxiliary UV-flow head: skipped at inference unless explicitly requested.
        want_aux = self.training if return_aux is None else bool(return_aux)
        uv_pred: Optional[torch.Tensor] = None
        if want_aux and self.uv_head is not None:
            B, D, H_p, W_p = feat_2d.shape
            k_kept = int(router_info["k_kept"])
            kept_idx = router_info["kept_idx"]  # (B, k_kept)
            # Scatter encoder-refined kept tokens back onto the dense 2D grid.
            dense_flat = feat_2d.permute(0, 2, 3, 1).reshape(B, H_p * W_p, D).clone()  # (B, N, D)
            idx_exp = kept_idx.unsqueeze(-1).expand(-1, -1, D)
            dense_flat.scatter_add_(1, idx_exp, mem[:, :k_kept, :])
            dense_2d = dense_flat.transpose(1, 2).view(B, D, H_p, W_p)
            uv_pred = self.uv_head(dense_2d)

        return mem, uv_pred, router_info

    def decode_tokens(
        self,
        tgt_ids: torch.Tensor,
        memory: Optional[torch.Tensor],
        return_mtp: Optional[bool] = None,
        past_kvs: Optional[List[Any]] = None,
        use_cache: bool = False,
        apply_tok_dropout: bool = True,
    ) -> Any:
        """Run causal decoder over ``tgt_ids`` ``(B, T)`` conditioned on ``memory``.

        When ``use_cache=False`` (default), returns ``(logits, mtp_logits)``.
        When ``use_cache=True``, returns ``(logits, mtp_logits, new_past_kvs)``.
        """
        _, T = tgt_ids.shape
        inp_ids = tgt_ids
        if self.training and apply_tok_dropout and self.cfg.tok_dropout > 0.0 and not use_cache:
            drop_mask = (
                (torch.rand_like(inp_ids, dtype=torch.float32) < self.cfg.tok_dropout)
                & (inp_ids != PAD_ID)
                & (inp_ids != BOS_ID)
            )
            rand_toks = torch.randint(0, len(data.VOCAB), inp_ids.shape, device=inp_ids.device, dtype=inp_ids.dtype)
            inp_ids = torch.where(drop_mask, rand_toks, inp_ids)
        x = self.tok_emb(inp_ids)
        past_len = 0
        if past_kvs is not None and len(past_kvs) > 0 and past_kvs[0] is not None and past_kvs[0][0] is not None:
            past_len = int(past_kvs[0][0][0].shape[2])

        cos_all, sin_all = self.rope1d(past_len + T, device=x.device, dtype=x.dtype)
        cos1d = cos_all[:, :, past_len : past_len + T, :]
        sin1d = sin_all[:, :, past_len : past_len + T, :]

        use_dec_ckpt = (
            self.training
            and self.cfg.grad_checkpoint
            and torch.is_grad_enabled()
            and (x.shape[0] * x.shape[1] > 2000)
        )
        new_past_kvs: List[Any] = []
        for idx, layer in enumerate(self.dec_layers):
            layer_past = past_kvs[idx] if past_kvs is not None else None
            if use_cache:
                x, layer_kv = layer(
                    x, memory=memory, rope_cos=cos1d, rope_sin=sin1d,
                    past_kv=layer_past, use_cache=True,
                )
                new_past_kvs.append(layer_kv)
            elif use_dec_ckpt:
                x = _grad_ckpt(
                    lambda _x, _m, _c, _s, _l=layer: _l(_x, memory=_m, rope_cos=_c, rope_sin=_s),
                    x, memory, cos1d, sin1d,
                    use_reentrant=False,
                )
            else:
                x = layer(x, memory=memory, rope_cos=cos1d, rope_sin=sin1d)
        h = self.dec_norm(x)

        logits = self.lm_head(h)
        want_mtp = self.training if return_mtp is None else bool(return_mtp)
        mtp_logits: List[torch.Tensor] = []
        if want_mtp and len(self.mtp_heads) > 0:
            for head in self.mtp_heads:
                mtp_logits.append(self.lm_head(head(h)))
        if use_cache:
            return logits, mtp_logits, new_past_kvs
        return logits, mtp_logits

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_len: int = 256,
        keep_ratio_override: Optional[float] = None,
        return_info: bool = False,
        return_conf: bool = False,
    ) -> Any:
        """Fast KV-cached greedy autoregressive decoding (0% aux/MTP overhead).

        Args:
            images: ``(B, 3, H, W)`` float tensor in ``[0, 1]``.
            max_len: maximum number of generated tokens after ``BOS_ID``.
            keep_ratio_override: optional runtime keep_ratio for ``InkTokenRouter``.
            return_info: if True, include ``router_info`` in the return tuple.
            return_conf: if True, include ``(mean_conf, reached_eos)`` in the return tuple.

        Returns:
            ``seq_ids`` of shape ``(B, L)`` starting with ``BOS_ID`` (or tuple when
            ``return_info`` / ``return_conf`` are enabled).
        """
        mem, _, router_info = self.encode_image(
            images,
            keep_ratio_override=keep_ratio_override,
            return_aux=False,
        )
        B = images.shape[0]
        device = images.device
        cur_ids = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        past_kvs: Optional[List[Any]] = None
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        out_tokens: List[torch.Tensor] = [cur_ids]
        prob_sum = torch.zeros(B, dtype=torch.float32, device=device) if return_conf else None
        tok_cnt = torch.zeros(B, dtype=torch.float32, device=device) if return_conf else None

        for _ in range(max_len):
            logits, _, past_kvs = self.decode_tokens(
                cur_ids,
                memory=mem if past_kvs is None else None,
                return_mtp=False,
                past_kvs=past_kvs,
                use_cache=True,
            )
            step_logits = logits[:, -1, :]
            if return_conf:
                step_probs = F.softmax(step_logits.float(), dim=-1)
                next_prob, next_tok = step_probs.max(dim=-1)
                active_f = (~finished).to(torch.float32)
                assert prob_sum is not None and tok_cnt is not None
                prob_sum = prob_sum + next_prob * active_f
                tok_cnt = tok_cnt + active_f
            else:
                next_tok = step_logits.argmax(dim=-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, PAD_ID), next_tok)
            out_tokens.append(next_tok.unsqueeze(1))
            finished = finished | (next_tok == EOS_ID)
            if bool(finished.all()):
                break
            cur_ids = next_tok.unsqueeze(1)

        seq = torch.cat(out_tokens, dim=1)
        if return_conf:
            assert prob_sum is not None and tok_cnt is not None
            mean_conf = prob_sum / tok_cnt.clamp(min=1.0)
            if return_info:
                return seq, router_info, mean_conf, finished
            return seq, mean_conf, finished
        if return_info:
            return seq, router_info
        return seq

    def forward(
        self,
        images: torch.Tensor,
        tgt_ids: torch.Tensor,
        keep_ratio_override: Optional[float] = None,
        return_aux: Optional[bool] = None,
        return_mtp: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Full encoder-decoder forward pass.

        Returns a dict with:
          - ``logits``: ``(B, T, V)`` next-token logits
          - ``mtp_logits``: list of ``(B, T, V)`` auxiliary lookahead logits (empty at infer)
          - ``uv_pred``: ``(B, 3, H', W')`` or ``None`` (skipped at infer)
          - ``router_info``: token compression metadata
        """
        mem, uv_pred, router_info = self.encode_image(
            images,
            keep_ratio_override=keep_ratio_override,
            return_aux=return_aux,
        )
        logits, mtp_logits = self.decode_tokens(tgt_ids, mem, return_mtp=return_mtp)
        return {
            "logits": logits,
            "mtp_logits": mtp_logits,
            "uv_pred": uv_pred,
            "router_info": router_info,
        }

    def _get_struct_token_weights(
        self,
        vocab_size: int,
        weight: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build and cache 1D per-token CE weights upweighting header/structural and micro-rhythm tokens."""
        w_val = float(weight)
        micro_w = 0.5 * (1.0 + w_val)
        key = (int(vocab_size), w_val, device, dtype)
        cached = self._struct_weight_cache.get(key)
        if cached is not None:
            return cached
        tok_w = torch.ones(int(vocab_size), device=device, dtype=dtype)
        for tok, tid in UNI_TOKEN_TO_ID.items():
            if tid >= vocab_size:
                continue
            if (
                tok.startswith("clef-")
                or tok.startswith("key")
                or tok.startswith("time-")
                or tok == SYS_TOKEN
                or tok == "grace"
                or tok.startswith("bar")
            ):
                tok_w[tid] = w_val
            elif (
                tok.startswith("tuplet-start-")
                or tok == "tuplet-stop"
                or tok in ("32nd", "64th", "a2", "a-2")
            ):
                tok_w[tid] = micro_w
        if vocab_size != VOCAB_SIZE and vocab_size >= 4:
            # Ensure tail <sys> token is upweighted even on legacy vocab sizes (e.g. 182)
            tok_w[vocab_size - 1] = w_val
        self._struct_weight_cache[key] = tok_w
        return tok_w

    def compute_loss(
        self,
        out: Dict[str, Any],
        targets: torch.Tensor,
        uv_target: Optional[torch.Tensor] = None,
        mtp_loss_weight: float = 0.30,
        uv_loss_weight: float = 0.20,
        saliency_loss_weight: float = 0.10,
        struct_tok_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined CE + MTP-k + auxiliary UV-flow + router saliency BCE losses.

        Args:
            out: dict returned by ``forward(images, tgt_ids)`` where ``tgt_ids``
                is ``seq[:, :-1]`` and ``targets`` is ``seq[:, 1:]`` of shape ``(B, T)``.
            targets: ``(B, T)`` next-token target IDs (with ``PAD_ID`` ignored).
            uv_target: optional ``(B, 3, H, W)`` ground-truth ``(u, v, valid)`` map.
            saliency_loss_weight: weight on lightweight BCE supervision of ``saliency_logits``.
            struct_tok_weight: optional loss multiplier (> 1.0) on header/structural tokens
                (``clef-*``, ``key*``, ``time-*``, ``<sys>``, ``grace``, ``bar*``).
        """
        logits: torch.Tensor = out["logits"]
        V = logits.shape[-1]
        ce_weight: Optional[torch.Tensor] = None
        if float(struct_tok_weight) > 1.0:
            ce_weight = self._get_struct_token_weights(
                V, float(struct_tok_weight), logits.device, logits.dtype
            )
        ce_loss = F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1),
            weight=ce_weight,
            ignore_index=PAD_ID,
        )

        # Multi-Token Prediction auxiliary losses for offsets +2 .. +mtp_k
        mtp_logits: List[torch.Tensor] = out.get("mtp_logits", [])
        T = targets.shape[1]
        mtp_terms: List[torch.Tensor] = []
        for idx, m_logits in enumerate(mtp_logits):
            shift = idx + 1  # shift=1 -> predicts targets[:, 1:] from pos t (i.e. t+2)
            if T > shift:
                m_pred = m_logits[:, :-shift, :].reshape(-1, V)
                m_tgt = targets[:, shift:].reshape(-1)
                m_loss = F.cross_entropy(m_pred, m_tgt, ignore_index=PAD_ID)
                mtp_terms.append(torch.nan_to_num(m_loss, nan=0.0))
        if mtp_terms:
            mtp_loss = torch.stack(mtp_terms).mean()
        else:
            mtp_loss = logits.new_zeros(())

        # Auxiliary UV-flow dewarping loss
        uv_pred: Optional[torch.Tensor] = out.get("uv_pred")
        if uv_pred is not None and uv_target is not None and uv_loss_weight > 0.0:
            uv_loss = compute_uv_flow_loss(uv_pred, uv_target)
        else:
            uv_loss = logits.new_zeros(())

        # Auxiliary InkTokenRouter saliency BCE loss (0% inference cost)
        rinfo = out.get("router_info") or {}
        sal_logits: Optional[torch.Tensor] = rinfo.get("saliency_logits")
        ink_target: Optional[torch.Tensor] = rinfo.get("ink_target")
        if sal_logits is not None and ink_target is not None and saliency_loss_weight > 0.0:
            saliency_loss = F.binary_cross_entropy_with_logits(
                sal_logits.float(), ink_target.float()
            ).to(logits.dtype)
        else:
            saliency_loss = logits.new_zeros(())

        total_loss = (
            ce_loss
            + mtp_loss_weight * mtp_loss
            + uv_loss_weight * uv_loss
            + saliency_loss_weight * saliency_loss
        )
        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mtp_loss": mtp_loss,
            "uv_loss": uv_loss,
            "saliency_loss": saliency_loss,
        }

    def param_summary(self) -> Dict[str, Any]:
        """Return exact parameter counts and quantized byte sizes (FP16/INT8/INT4)."""
        total_params = sum(p.numel() for p in self.parameters())
        uv_params = sum(p.numel() for p in self.uv_head.parameters()) if self.uv_head is not None else 0
        mtp_params = sum(p.numel() for p in self.mtp_heads.parameters())
        infer_params = total_params - uv_params - mtp_params
        infer_with_mtp_params = total_params - uv_params

        def _mb(n_params: int, bytes_per_param: float) -> float:
            return (n_params * bytes_per_param) / 1_000_000.0

        return {
            "total_params": total_params,
            "infer_params": infer_params,
            "infer_with_mtp_params": infer_with_mtp_params,
            "uv_aux_params": uv_params,
            "mtp_aux_params": mtp_params,
            "fp32_mb": _mb(infer_params, 4.0),
            "fp16_mb": _mb(infer_params, 2.0),
            "int8_mb": _mb(infer_params, 1.0),
            "int4_mb": _mb(infer_params, 0.5),
            "total_fp16_mb": _mb(total_params, 2.0),
            "total_int8_mb": _mb(total_params, 1.0),
            "total_int4_mb": _mb(total_params, 0.5),
        }


# ---------------------------------------------------------------------------
# Real Strip & Full-Page Data Loaders + Online Degradation
# ---------------------------------------------------------------------------

_STRIP_SYS_RE = re.compile(r"^(.+)_s(\d+)-bekern\.txt$")
_IMG_EXTS = (".png", ".jpg", ".jpeg")


def _round_mult16(val: float, min_val: int = 16, max_val: Optional[int] = None) -> int:
    r = max(min_val, int(round(val / 16.0)) * 16)
    if max_val is not None:
        max_m16 = max(min_val, (int(max_val) // 16) * 16)
        r = min(r, max_m16)
    return r


def _find_matching_image(directory: Path, stem: str) -> Optional[Path]:
    for ext in _IMG_EXTS:
        cand = directory / f"{stem}{ext}"
        if cand.is_file():
            return cand
    return None


def _strip_group_key(stem: str) -> str:
    m = re.match(r"^(.+)_s\d+$", stem)
    base = m.group(1) if m else stem
    return data.piece_key(base)


def discover_strip_pairs(
    strip_dirs: Sequence[str | Path],
    max_samples: Optional[int] = None,
) -> List[Tuple[Path, Path]]:
    """Discover ``(image_path, bekern_path)`` strip pairs in ``strip_dirs``."""
    pairs: List[Tuple[Path, Path]] = []
    for raw_dir in strip_dirs:
        d = Path(raw_dir)
        if not d.is_dir():
            continue
        scan_dir = d
        # Convenience: allow passing dataset root containing stage2_strips/
        if (d / "stage2_strips").is_dir():
            has_direct = any(
                e.name.endswith("-bekern.txt") for _, e in zip(range(32), os.scandir(d))
            )
            if not has_direct:
                scan_dir = d / "stage2_strips"

        found: List[Tuple[Path, Path]] = []
        with os.scandir(scan_dir) as it:
            for entry in it:
                if not entry.is_file() or not entry.name.endswith("-bekern.txt"):
                    continue
                stem = entry.name[: -len("-bekern.txt")]
                img_path = _find_matching_image(scan_dir, stem)
                if img_path is not None:
                    found.append((img_path, Path(entry.path)))
                    if max_samples is not None and len(pairs) + len(found) >= max_samples:
                        break
        found.sort(key=lambda p: p[0].name)
        pairs.extend(found)
        if max_samples is not None and len(pairs) >= max_samples:
            pairs = pairs[:max_samples]
            break
    return pairs


def discover_fullpage_pairs(
    fullpage_dirs: Sequence[str | Path],
    max_samples: Optional[int] = None,
) -> List[Tuple[Path, List[Path]]]:
    """Discover ``(page_image_path, [ordered_system_bekern_paths])`` full-page pairs.

    Accepts either dataset roots containing ``pages/`` + ``stage2_strips/``,
    or ``pages/`` directories directly (resolving sibling ``stage2_strips/``
    or direct ``<stem>-bekern.txt`` files).
    """
    pairs: List[Tuple[Path, List[Path]]] = []
    for raw_dir in fullpage_dirs:
        d = Path(raw_dir)
        if not d.is_dir():
            continue
        pages_dir = (d / "pages") if (d / "pages").is_dir() else d
        strips_dir: Optional[Path] = None
        for cand in (
            pages_dir.parent / "stage2_strips",
            d / "stage2_strips",
            pages_dir.parent / "strips",
            d / "strips",
        ):
            if cand.is_dir():
                strips_dir = cand
                break

        if max_samples is not None:
            # Fast bounded discovery for smoke tests: probe page images directly.
            found_fast: List[Tuple[Path, List[Path]]] = []
            with os.scandir(pages_dir) as it:
                for entry in it:
                    if not entry.is_file():
                        continue
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext not in _IMG_EXTS:
                        continue
                    stem = os.path.splitext(entry.name)[0]
                    sys_files: List[Tuple[int, Path]] = []
                    if strips_dir is not None:
                        for k in range(24):
                            cand_b = strips_dir / f"{stem}_s{k}-bekern.txt"
                            if cand_b.is_file():
                                sys_files.append((k, cand_b))
                    if not sys_files:
                        direct_b = pages_dir / f"{stem}-bekern.txt"
                        if direct_b.is_file():
                            sys_files.append((0, direct_b))
                    if sys_files:
                        sys_files.sort(key=lambda t: t[0])
                        found_fast.append((Path(entry.path), [p for _, p in sys_files]))
                        if len(pairs) + len(found_fast) >= max_samples:
                            break
            found_fast.sort(key=lambda t: t[0].name)
            pairs.extend(found_fast)
            if len(pairs) >= max_samples:
                pairs = pairs[:max_samples]
                break
            continue

        # Full directory scan: index all <stem>_s<k>-bekern.txt once.
        by_stem: Dict[str, List[Tuple[int, Path]]] = {}
        if strips_dir is not None:
            with os.scandir(strips_dir) as it:
                for entry in it:
                    if not entry.is_file() or not entry.name.endswith("-bekern.txt"):
                        continue
                    m = _STRIP_SYS_RE.match(entry.name)
                    if m is not None:
                        by_stem.setdefault(m.group(1), []).append((int(m.group(2)), Path(entry.path)))

        found_full: List[Tuple[Path, List[Path]]] = []
        with os.scandir(pages_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext not in _IMG_EXTS:
                    continue
                stem = os.path.splitext(entry.name)[0]
                sys_list = by_stem.get(stem)
                if sys_list:
                    sys_list_sorted = sorted(sys_list, key=lambda t: t[0])
                    found_full.append((Path(entry.path), [p for _, p in sys_list_sorted]))
                else:
                    direct_b = pages_dir / f"{stem}-bekern.txt"
                    if direct_b.is_file():
                        found_full.append((Path(entry.path), [direct_b]))
        found_full.sort(key=lambda t: t[0].name)
        pairs.extend(found_full)

    return pairs


def split_train_val(
    items: Sequence[Any],
    key_fn: Any,
    val_count: int = 100,
    seed: int = 42,
) -> Tuple[List[Any], List[Any]]:
    """Deterministic group-aware train/val split."""
    n = len(items)
    if n <= 1 or val_count <= 0:
        return list(items), []

    target_val = min(val_count, max(1, n // 5)) if n >= 2 else 0
    groups: Dict[str, List[Any]] = {}
    for item in items:
        groups.setdefault(str(key_fn(item)), []).append(item)

    keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    val_items: List[Any] = []
    train_items: List[Any] = []
    for k in keys:
        grp = groups[k]
        if len(val_items) < target_val and (len(train_items) > 0 or len(keys) > 1):
            # Ensure train stays non-empty even when n == 2.
            if len(val_items) + len(grp) <= n - 1:
                val_items.extend(grp)
                continue
        train_items.extend(grp)

    if not train_items and val_items:
        train_items.append(val_items.pop())
    return train_items, val_items


def discover_unlabeled_images(
    unlabeled_dirs: Sequence[str | Path],
    max_samples: Optional[int] = None,
) -> List[Path]:
    """Recursively discover unlabeled ``.png/.jpg/.jpeg`` images."""
    found: List[Path] = []
    for raw_dir in unlabeled_dirs:
        d = Path(raw_dir)
        if not d.is_dir():
            continue
        dir_imgs: List[Path] = []
        for root_str, dirs, files in os.walk(d):
            dirs.sort()
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() in _IMG_EXTS:
                    dir_imgs.append(Path(root_str) / fname)
                    if max_samples is not None and len(found) + len(dir_imgs) >= max_samples:
                        break
            if max_samples is not None and len(found) + len(dir_imgs) >= max_samples:
                break
        found.extend(dir_imgs)
        if max_samples is not None and len(found) >= max_samples:
            found = found[:max_samples]
            break
    return found


def apply_weak_augment(img: Image.Image, rng: Optional[random.Random] = None) -> Image.Image:
    """Mild brightness/contrast perturbation for EMA teacher weak view."""
    r = rng if rng is not None else random
    out = img
    if r.random() < 0.50:
        out = ImageEnhance.Brightness(out).enhance(r.uniform(0.94, 1.06))
    if r.random() < 0.50:
        out = ImageEnhance.Contrast(out).enhance(r.uniform(0.94, 1.06))
    return out


def _find_perspective_coeffs(
    dst_pts: Sequence[Tuple[float, float]],
    src_pts: Sequence[Tuple[float, float]],
) -> List[float]:
    """Solve 8-DOF homography coefficients mapping output ``dst_pts`` -> input ``src_pts`` for PIL."""
    rows: List[List[float]] = []
    rhs: List[float] = []
    for (x, y), (u, v) in zip(dst_pts, src_pts):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        rhs.append(v)
    A = np.asarray(rows, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    coeffs = np.linalg.solve(A, b)
    return [float(c) for c in coeffs]


def apply_clean_scale_jitter(img: Image.Image, rng: Optional[random.Random] = None) -> Image.Image:
    """Mild clean-page scale jitter (0.90-1.00x) with white paper padding.

    Matches photo scale variation on clean unbordered pages without introducing
    dark borders. Preserves exact ``(W, H)`` dimensions.
    """
    r = rng if rng is not None else random
    out = img if img.mode == "RGB" else img.convert("RGB")
    orig_w, orig_h = out.size
    if orig_w < 32 or orig_h < 32:
        return out
    scale = r.uniform(0.90, 1.00)
    new_w = max(16, int(round(orig_w * scale)))
    new_h = max(16, int(round(orig_h * scale)))
    if new_w >= orig_w and new_h >= orig_h:
        return out
    resized = out.resize((new_w, new_h), Image.Resampling.BILINEAR)
    bg_v = r.randint(246, 255)
    paper_fill = (bg_v, max(242, bg_v - r.randint(0, 3)), max(238, bg_v - r.randint(0, 6)))
    canvas = Image.new("RGB", (orig_w, orig_h), paper_fill)
    dx = r.randint(0, max(0, orig_w - new_w))
    dy = r.randint(0, max(0, orig_h - new_h))
    canvas.paste(resized, (dx, dy))
    return canvas


def _add_archival_empty_staves(arr: np.ndarray, r: Any) -> None:
    """Draw 1-2 faint empty pre-ruled 5-line staves in quiet top/bottom margins or inter-system gaps (in-place)."""
    h, w = arr.shape[:2]
    if h < 128 or w < 64:
        return
    spacing = max(2, int(round(h * r.uniform(0.0045, 0.0075))))
    staff_h = spacing * 4 + 1
    x0 = int(round(w * r.uniform(0.06, 0.12)))
    x1 = int(round(w * r.uniform(0.88, 0.94)))
    if x1 <= x0 + 16:
        return

    candidates: List[int] = []
    # Top quiet margin band
    top_y0 = max(2, int(round(h * 0.025)))
    top_y1 = min(h - staff_h - 2, int(round(h * 0.12)) - staff_h)
    if top_y1 >= top_y0:
        band = arr[top_y0 : top_y1 + staff_h, x0:x1]
        if float(band.mean()) > 0.85:
            candidates.append(r.randint(top_y0, top_y1))
    # Bottom quiet margin band
    bot_y0 = max(2, int(round(h * 0.88)))
    bot_y1 = min(h - staff_h - 2, int(round(h * 0.975)) - staff_h)
    if bot_y1 >= bot_y0:
        band = arr[bot_y0 : bot_y1 + staff_h, x0:x1]
        if float(band.mean()) > 0.85:
            candidates.append(r.randint(bot_y0, bot_y1))
    # Inter-system quiet gaps (between 14% and 86% of page height)
    mid_y0 = max(2, int(round(h * 0.14)))
    mid_y1 = min(h - staff_h - 2, int(round(h * 0.86)) - staff_h)
    if mid_y1 >= mid_y0:
        row_means = arr[mid_y0 : mid_y1 + staff_h, x0:x1].mean(axis=1)
        pad_gap = max(3, spacing)
        min_gap = staff_h + 2 * pad_gap
        quiet = row_means > 0.94
        run_start: Optional[int] = None
        inter_cands: List[int] = []
        for idx_q, is_q in enumerate(quiet):
            if is_q:
                if run_start is None:
                    run_start = idx_q
            else:
                if run_start is not None:
                    if (idx_q - run_start) >= min_gap:
                        lo_y = mid_y0 + run_start + pad_gap
                        hi_y = mid_y0 + idx_q - staff_h - pad_gap
                        if hi_y >= lo_y:
                            inter_cands.append(r.randint(lo_y, hi_y))
                    run_start = None
        if run_start is not None and (len(quiet) - run_start) >= min_gap:
            lo_y = mid_y0 + run_start + pad_gap
            hi_y = mid_y0 + len(quiet) - staff_h - pad_gap
            if hi_y >= lo_y:
                inter_cands.append(r.randint(lo_y, hi_y))
        if inter_cands:
            r.shuffle(inter_cands)
            candidates.extend(inter_cands[:2])

    if not candidates:
        return
    r.shuffle(candidates)
    n_draw = 1 if (len(candidates) == 1 or r.random() < 0.55) else 2
    line_luma = r.uniform(0.62, 0.82)
    for y_start in candidates[:n_draw]:
        for k in range(5):
            y = y_start + k * spacing
            if 0 <= y < h:
                arr[y, x0:x1] = np.minimum(arr[y, x0:x1], line_luma)


def _apply_system1_left_indent(
    img: Image.Image,
    rng: Any,
    paper_fill: Optional[Tuple[int, int, int]] = None,
) -> Image.Image:
    """Apply 15-35% left indentation to System 1 (compressing horizontally so 0 notes are clipped)."""
    out = img.copy() if img.mode == "RGB" else img.convert("RGB")
    w, h = out.size
    if w < 128 or h < 256:
        return out
    gray = np.asarray(out.convert("L"), dtype=np.float32) * (1.0 / 255.0)
    row_means = gray.mean(axis=1)
    # Use relative paper-baseline threshold so off-white / cream paper works cleanly
    paper_level = float(np.percentile(row_means, 90.0))
    ink_thr = paper_level - 0.025
    # Locate the first ink band (System 1) in the upper half of the page
    ink_rows = np.where(row_means < ink_thr)[0]
    if len(ink_rows) == 0:
        return out
    y0 = int(ink_rows[0])
    if y0 > int(h * 0.35):
        return out
    min_gap = max(14, int(round(h * 0.035)))
    max_y1 = min(h, y0 + int(round(h * 0.36)))
    quiet_run = 0
    y1 = max_y1
    found_gap = False
    for y in range(y0, max_y1):
        if row_means[y] >= ink_thr:
            quiet_run += 1
            if quiet_run >= min_gap:
                y1 = y - quiet_run + 1
                found_gap = True
                break
        else:
            quiet_run = 0
    band_h = y1 - y0
    if not found_gap or band_h < max(12, int(round(h * 0.03))):
        return out
    pad_y = max(2, min(4, y0))
    crop_y0 = max(0, y0 - pad_y)
    crop_y1 = min(h, y1 + pad_y)
    crop_h = crop_y1 - crop_y0
    indent_px = max(8, int(round(w * rng.uniform(0.15, 0.35))))
    new_band_w = w - indent_px
    if new_band_w < 32:
        return out
    band = out.crop((0, crop_y0, w, crop_y1)).resize((new_band_w, crop_h), Image.Resampling.BILINEAR)
    if paper_fill is None:
        # Estimate paper tone from top margin
        top_strip = np.asarray(out.crop((0, 0, w, max(2, crop_y0))), dtype=np.float32)
        if top_strip.size >= 3:
            med = np.median(top_strip.reshape(-1, 3), axis=0)
            paper_fill = (int(round(med[0])), int(round(med[1])), int(round(med[2])))
        else:
            paper_fill = (248, 246, 240)
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, crop_y0, indent_px, crop_y1 - 1], fill=paper_fill)
    out.paste(band, (indent_px, crop_y0))
    return out


def _apply_corner_sticky_note(img: Image.Image, rng: Any) -> Image.Image:
    """Draw a small opaque corner sticky-note / archival label occlusion (5-12% W x H)."""
    out = img.copy() if img.mode == "RGB" else img.convert("RGB")
    w, h = out.size
    if w < 64 or h < 64:
        return out
    nw = max(6, int(round(w * rng.uniform(0.05, 0.12))))
    nh = max(6, int(round(h * rng.uniform(0.05, 0.12))))
    margin_x = int(round(w * rng.uniform(0.0, 0.025)))
    margin_y = int(round(h * rng.uniform(0.0, 0.025)))
    corner = rng.randint(0, 3)
    x0 = margin_x if corner in (0, 2) else max(0, w - nw - margin_x)
    y0 = margin_y if corner in (0, 1) else max(0, h - nh - margin_y)
    x1 = min(w - 1, x0 + nw)
    y1 = min(h - 1, y0 + nh)

    # Pale yellow, pale pink, or pale kraft/brown archival label
    palette = (
        (246, 236, 168),
        (240, 202, 210),
        (214, 188, 150),
    )
    br, bg, bb = rng.choice(palette)
    fill = (
        max(0, min(255, br + rng.randint(-8, 8))),
        max(0, min(255, bg + rng.randint(-8, 8))),
        max(0, min(255, bb + rng.randint(-8, 8))),
    )
    edge = (max(0, fill[0] - 28), max(0, fill[1] - 28), max(0, fill[2] - 28))
    draw = ImageDraw.Draw(out)
    draw.rectangle([x0, y0, x1, y1], fill=fill, outline=edge, width=1)
    return out


def _apply_hw_lowres_defocus_compress(img: Image.Image, rng: Any) -> Image.Image:
    """Low-res downscale (800-1150px round-trip) + mild defocus blur + contrast compression."""
    out = img if img.mode == "RGB" else img.convert("RGB")
    orig_w, orig_h = out.size
    if orig_w < 48 or orig_h < 48:
        return out
    max_side = max(orig_w, orig_h)
    if max_side > 800:
        rt_max = rng.randint(800, min(1150, max_side))
        scale = float(rt_max) / float(max_side)
    else:
        scale = rng.uniform(0.65, 0.85)
    dw = max(32, int(round(orig_w * scale)))
    dh = max(32, int(round(orig_h * scale)))
    if (dw, dh) != (orig_w, orig_h):
        out = out.resize((dw, dh), Image.Resampling.BILINEAR).resize(
            (orig_w, orig_h), Image.Resampling.BILINEAR
        )
    out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.1)))
    # Compress ink-to-paper contrast toward page paper tone (targeting faint InkMn ~ 0.06-0.10)
    arr = np.asarray(out, dtype=np.float32) * (1.0 / 255.0)
    paper_ref = np.percentile(arr.reshape(-1, 3), 85.0, axis=0).reshape(1, 1, 3)
    c_fac = rng.uniform(0.48, 0.74)
    arr = paper_ref + (arr - paper_ref) * c_fac
    np.clip(arr, 0.0, 1.0, out=arr)
    return Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")


def _apply_archival_border_and_colorbar(
    img: Image.Image,
    r: Any,
    dark_fill: Tuple[int, int, int],
) -> Image.Image:
    """Add archival black scan border + small colorbar strip inside the dark margin."""
    orig_w, orig_h = img.size
    if orig_w < 48 or orig_h < 48:
        return img
    pad_l = max(4, int(round(orig_w * r.uniform(0.025, 0.085))))
    pad_r = max(4, int(round(orig_w * r.uniform(0.025, 0.085))))
    pad_t = max(4, int(round(orig_h * r.uniform(0.025, 0.085))))
    pad_b = max(4, int(round(orig_h * r.uniform(0.025, 0.085))))
    cw = orig_w + pad_l + pad_r
    ch = orig_h + pad_t + pad_b
    canvas = Image.new("RGB", (cw, ch), dark_fill)
    canvas.paste(img, (pad_l, pad_t))

    # Paint a small archival colorbar strip strictly inside one padded border (zero-copy ImageDraw)
    swatches = (
        (20, 20, 20),
        (120, 120, 120),
        (235, 235, 230),
        (195, 50, 45),
        (55, 145, 70),
        (45, 80, 175),
        (215, 190, 50),
    )
    draw = ImageDraw.Draw(canvas)
    side = r.choice(("left", "right", "bottom"))
    if side in ("left", "right"):
        margin_w = pad_l if side == "left" else pad_r
        bar_w = max(2, min(margin_w - 2, int(round(margin_w * 0.60))))
        if bar_w >= 2:
            bar_h = max(len(swatches) * 3, int(round(ch * r.uniform(0.18, 0.36))))
            y0 = r.randint(pad_t, max(pad_t, ch - pad_b - bar_h))
            x0 = (max(1, (pad_l - bar_w) // 2)) if side == "left" else (cw - pad_r + max(1, (pad_r - bar_w) // 2))
            step_h = max(1, bar_h // len(swatches))
            for idx, col in enumerate(swatches):
                sy0 = y0 + idx * step_h
                sy1 = min(ch - 1, sy0 + step_h)
                if sy1 > sy0:
                    draw.rectangle([x0, sy0, min(cw - 1, x0 + bar_w), sy1], fill=col)
    else:
        bar_h = max(2, min(pad_b - 2, int(round(pad_b * 0.60))))
        if bar_h >= 2:
            bar_w = max(len(swatches) * 3, int(round(cw * r.uniform(0.18, 0.36))))
            x0 = r.randint(pad_l, max(pad_l, cw - pad_r - bar_w))
            y0 = ch - pad_b + max(1, (pad_b - bar_h) // 2)
            step_w = max(1, bar_w // len(swatches))
            for idx, col in enumerate(swatches):
                sx0 = x0 + idx * step_w
                sx1 = min(cw - 1, sx0 + step_w)
                if sx1 > sx0:
                    draw.rectangle([sx0, y0, sx1, min(ch - 1, y0 + bar_h)], fill=col)

    return canvas.resize((orig_w, orig_h), Image.Resampling.BILINEAR)


def _apply_composer_scratchouts(
    img: Image.Image,
    rng: Any,
    is_hw: bool = True,
) -> Image.Image:
    """Simulate compact composer/copyist deletions (phantom ink + heavy cancellation).

    Draws 1-3 compact (15-55 x 12-35 px) phantom notehead/stem fragments and cancels
    them with diagonal cross-hatching, zig-zag strikethroughs, or smudged ink blotches.
    Uses ``ImageDraw.Draw(out, "RGBA")`` directly on the RGB canvas for <1 ms speed.
    """
    out = img.copy() if img.mode == "RGB" else img.convert("RGB")
    w, h = out.size
    if w < 48 or h < 48:
        return out

    draw = ImageDraw.Draw(out, "RGBA")
    n_patches = rng.randint(1, 3)
    max_bw = min(55, max(16, w // 4))
    max_bh = min(35, max(13, h // 4))
    min_bw = min(15, max_bw)
    min_bh = min(12, max_bh)

    def _sample_ink(alpha_lo: int = 185, alpha_hi: int = 245) -> Tuple[int, int, int, int]:
        if is_hw:
            ir = rng.randint(36, 98)
            ig = max(18, ir - rng.randint(6, 22))
            ib = max(10, ig - rng.randint(6, 20))
        else:
            base = rng.randint(14, 64)
            ir = max(0, min(85, base + rng.randint(-4, 6)))
            ig = max(0, min(85, base + rng.randint(-4, 6)))
            ib = max(0, min(95, base + rng.randint(-4, 16)))
        return (ir, ig, ib, rng.randint(alpha_lo, alpha_hi))

    for _ in range(n_patches):
        bw = rng.randint(min_bw, max_bw)
        bh = rng.randint(min_bh, max_bh)
        # Prefer quiet margins or inter-staff bands without covering large page areas
        zone = rng.random()
        if zone < 0.25:
            # Top or bottom margin
            x0 = rng.randint(max(2, int(w * 0.04)), max(3, w - bw - max(2, int(w * 0.04))))
            if rng.random() < 0.5:
                y0 = rng.randint(2, max(2, min(h - bh - 2, int(h * 0.14))))
            else:
                y_lo = max(2, int(h * 0.84) - bh)
                y0 = rng.randint(min(y_lo, h - bh - 2), max(2, h - bh - 2))
        elif zone < 0.45:
            # Left or right margin
            y0 = rng.randint(max(2, int(h * 0.06)), max(3, h - bh - max(2, int(h * 0.06))))
            if rng.random() < 0.5:
                x0 = rng.randint(2, max(2, min(w - bw - 2, int(w * 0.12))))
            else:
                x_lo = max(2, int(w * 0.86) - bw)
                x0 = rng.randint(min(x_lo, w - bw - 2), max(2, w - bw - 2))
        else:
            # Interior inter-staff / near-ink area
            x0 = rng.randint(max(2, int(w * 0.06)), max(3, w - bw - max(2, int(w * 0.06))))
            y0 = rng.randint(max(2, int(h * 0.10)), max(3, h - bh - max(2, int(h * 0.10))))
        x1 = x0 + bw
        y1 = y0 + bh

        # 1. Phantom ink cluster: 1-3 tiny noteheads + stems + optional beam/scribble
        ink_phantom = _sample_ink(170, 230)
        n_notes = rng.randint(1, 3)
        stem_tops: List[Tuple[int, int]] = []
        for ni in range(n_notes):
            nx = x0 + max(3, int(round((ni + 0.5) * bw / max(1, n_notes)))) + rng.randint(-2, 2)
            nx = max(x0 + 3, min(x1 - 3, nx))
            ny = rng.randint(y0 + max(3, bh // 3), max(y0 + max(3, bh // 3), y1 - 3))
            rx = rng.randint(2, 4)
            ry = rng.randint(2, 3)
            draw.ellipse([nx - rx, ny - ry, nx + rx, ny + ry], fill=ink_phantom)
            stem_x = min(x1 - 1, nx + rx)
            stem_y = max(y0 + 1, ny - rng.randint(max(6, bh // 2), max(7, bh - 2)))
            draw.line([(stem_x, ny), (stem_x, stem_y)], fill=ink_phantom, width=rng.randint(1, 2))
            stem_tops.append((stem_x, stem_y))
        if len(stem_tops) >= 2:
            draw.line([stem_tops[0], stem_tops[-1]], fill=ink_phantom, width=2)

        # 2. Heavy cancellation stroke pattern
        style = rng.choice(("crosshatch", "zigzag", "blotch_smudge"))
        if style == "crosshatch":
            n_fwd = rng.randint(4, 8)
            for k in range(n_fwd):
                frac = (k + 0.5) / n_fwd
                cx = x0 + int(round(frac * bw))
                dx = rng.randint(max(3, bw // 6), max(4, bw // 3))
                draw.line(
                    [(max(0, cx - dx), y1 + rng.randint(-2, 2)), (min(w - 1, cx + dx), y0 + rng.randint(-2, 2))],
                    fill=_sample_ink(190, 245),
                    width=rng.randint(1, 3),
                )
            n_bwd = rng.randint(3, 6)
            for k in range(n_bwd):
                frac = (k + 0.5) / n_bwd
                cx = x0 + int(round(frac * bw))
                dx = rng.randint(max(3, bw // 6), max(4, bw // 3))
                draw.line(
                    [(max(0, cx - dx), y0 + rng.randint(-2, 2)), (min(w - 1, cx + dx), y1 + rng.randint(-2, 2))],
                    fill=_sample_ink(185, 240),
                    width=rng.randint(1, 2),
                )
        elif style == "zigzag":
            n_pts = rng.randint(7, 13)
            pts: List[Tuple[int, int]] = []
            for k in range(n_pts):
                px = x0 + int(round((k / max(1, n_pts - 1)) * bw)) + rng.randint(-2, 2)
                py = (y0 + rng.randint(0, max(1, bh // 4))) if (k % 2 == 0) else (y1 - rng.randint(0, max(1, bh // 4)))
                pts.append((max(0, min(w - 1, px)), max(0, min(h - 1, py))))
            draw.line(pts, fill=_sample_ink(200, 250), width=rng.randint(2, 3))
            # Reinforcing horizontal cancel slash
            ym = (y0 + y1) // 2
            draw.line(
                [(max(0, x0 - 2), ym + rng.randint(-3, 3)), (min(w - 1, x1 + 2), ym + rng.randint(-3, 3))],
                fill=_sample_ink(190, 245),
                width=rng.randint(2, 3),
            )
        else:
            # Smudged erasure / ink blotch + heavy scribble strokes
            n_blobs = rng.randint(2, 4)
            for _ in range(n_blobs):
                bx0 = rng.randint(x0, max(x0, x1 - max(6, bw // 3)))
                by0 = rng.randint(y0, max(y0, y1 - max(5, bh // 3)))
                bx1 = min(x1, bx0 + rng.randint(max(6, bw // 3), max(7, int(bw * 0.75))))
                by1 = min(y1, by0 + rng.randint(max(5, bh // 3), max(6, int(bh * 0.75))))
                draw.ellipse([bx0, by0, bx1, by1], fill=_sample_ink(105, 175))
            for _ in range(rng.randint(3, 5)):
                sx0 = rng.randint(x0, x1)
                sy0 = rng.randint(y0, y1)
                sx1 = rng.randint(x0, x1)
                sy1 = rng.randint(y0, y1)
                draw.line([(sx0, sy0), (sx1, sy1)], fill=_sample_ink(200, 250), width=rng.randint(2, 4))

    return out


def _apply_performer_pencil_markup(
    img: Image.Image,
    rng: Any,
) -> Image.Image:
    """Simulate realistic musicians' pencil & translucent highlighter markings on printed scores.

    Draws thin/translucent graphite ovals, underlines, brackets/cut arrows, bowing marks
    (``V`` / ``Π``) / breath commas, and/or 1-2 translucent highlighter bands while keeping
    underlying printed notation 100% legible.
    """
    out = img.copy() if img.mode == "RGB" else img.convert("RGB")
    w, h = out.size
    if w < 48 or h < 48:
        return out

    draw = ImageDraw.Draw(out, "RGBA")
    drew_any = False

    def _sample_graphite(alpha_lo: int = 110, alpha_hi: int = 175) -> Tuple[int, int, int, int]:
        g = rng.randint(75, 135)
        gr = max(70, min(140, g + rng.randint(-5, 5)))
        gg = max(70, min(140, g + rng.randint(-4, 4)))
        gb = max(70, min(142, g + rng.randint(-5, 7)))
        return (gr, gg, gb, rng.randint(alpha_lo, alpha_hi))

    def _draw_hand_oval() -> None:
        rx = rng.randint(max(6, min(10, w // 16)), max(8, min(28, w // 8)))
        ry = rng.randint(max(5, min(8, h // 16)), max(7, min(22, h // 8)))
        cx = rng.randint(rx + 4, max(rx + 4, w - rx - 4))
        cy = rng.randint(ry + 4, max(ry + 4, h - ry - 4))
        n_seg = rng.randint(12, 16)
        pts: List[Tuple[int, int]] = []
        phase = rng.uniform(0.0, 2.0 * math.pi)
        for k in range(n_seg + 2):
            ang = phase + (2.0 * math.pi * k) / n_seg
            jr = rng.uniform(0.88, 1.12)
            px = int(round(cx + rx * jr * math.cos(ang)))
            py = int(round(cy + ry * jr * math.sin(ang)))
            pts.append((max(0, min(w - 1, px)), max(0, min(h - 1, py))))
        draw.line(pts, fill=_sample_graphite(), width=rng.randint(1, 2))

    # 1. Translucent yellow/pink/green highlighter bands (1-2 bands, alpha 0.18-0.32)
    if rng.random() < 0.45:
        palette = ((255, 245, 70), (255, 135, 190), (120, 240, 130))
        for _ in range(rng.randint(1, 2)):
            hr, hg, hb = rng.choice(palette)
            ha = int(round(rng.uniform(0.18, 0.32) * 255.0))
            band_len = rng.randint(max(24, int(w * 0.08)), max(28, int(w * 0.28)))
            band_h = rng.randint(max(8, min(12, h // 24)), max(10, min(24, h // 12)))
            x0 = rng.randint(max(4, int(w * 0.05)), max(5, w - band_len - max(4, int(w * 0.05))))
            y0 = rng.randint(max(4, int(h * 0.08)), max(5, h - band_h - max(4, int(h * 0.08))))
            dy = rng.randint(-4, 4)
            poly = [
                (x0, y0),
                (x0 + band_len, max(0, min(h - 1, y0 + dy))),
                (x0 + band_len, max(0, min(h - 1, y0 + band_h + dy))),
                (x0, min(h - 1, y0 + band_h)),
            ]
            draw.polygon(poly, fill=(hr, hg, hb, ha))
        drew_any = True

    # 2. Graphite hand-drawn ovals/circles (1-3)
    if rng.random() < 0.75:
        for _ in range(rng.randint(1, 3)):
            _draw_hand_oval()
        drew_any = True

    # 3. Underlines, brackets, or cut arrows (1-2)
    if rng.random() < 0.65:
        for _ in range(rng.randint(1, 2)):
            col = _sample_graphite()
            kind = rng.choice(("underline", "bracket", "arrow"))
            if kind == "underline":
                ulen = rng.randint(max(20, min(35, w // 6)), max(25, min(140, w // 3)))
                ux0 = rng.randint(4, max(4, w - ulen - 4))
                uy0 = rng.randint(8, max(8, h - 8))
                n_u = rng.randint(5, 8)
                pts = [
                    (
                        ux0 + int(round(ulen * k / (n_u - 1))),
                        max(0, min(h - 1, uy0 + rng.randint(-2, 2))),
                    )
                    for k in range(n_u)
                ]
                draw.line(pts, fill=col, width=rng.randint(1, 2))
            elif kind == "bracket":
                bh = rng.randint(14, min(42, max(15, h // 6)))
                bw = rng.randint(5, 10)
                bx = rng.randint(12, max(12, w - 12))
                by = rng.randint(8, max(8, h - bh - 8))
                sgn = 1 if rng.random() < 0.5 else -1
                draw.line(
                    [(bx + sgn * bw, by), (bx, by), (bx, by + bh), (bx + sgn * bw, by + bh)],
                    fill=col,
                    width=rng.randint(1, 2),
                )
            else:
                ax0 = rng.randint(12, max(12, w - 24))
                ay0 = rng.randint(12, max(12, h - 24))
                adx = rng.choice((-1, 1)) * rng.randint(12, 28)
                ady = rng.choice((-1, 1)) * rng.randint(10, 22)
                ax1 = max(2, min(w - 3, ax0 + adx))
                ay1 = max(2, min(h - 3, ay0 + ady))
                draw.line([(ax0, ay0), (ax1, ay1)], fill=col, width=rng.randint(1, 2))
                # Arrowhead ticks at (ax1, ay1)
                sx = -1 if ax1 >= ax0 else 1
                sy = -1 if ay1 >= ay0 else 1
                draw.line([(ax1, ay1), (max(0, min(w - 1, ax1 + sx * 6)), ay1)], fill=col, width=1)
                draw.line([(ax1, ay1), (ax1, max(0, min(h - 1, ay1 + sy * 6)))], fill=col, width=1)
        drew_any = True

    # 4. Bowing symbols (V, Π) and breath commas (1-3)
    if rng.random() < 0.65:
        for _ in range(rng.randint(1, 3)):
            col = _sample_graphite()
            sym = rng.choice(("upbow", "downbow", "comma"))
            mx = rng.randint(10, max(10, w - 14))
            my = rng.randint(10, max(10, h - 16))
            if sym == "upbow":
                # V shape
                draw.line([(mx - 5, my - 6), (mx, my + 6), (mx + 5, my - 6)], fill=col, width=rng.randint(1, 2))
            elif sym == "downbow":
                # Π shape (thick top bar + 2 legs)
                draw.line([(mx - 5, my - 5), (mx + 5, my - 5)], fill=col, width=3)
                draw.line([(mx - 5, my - 5), (mx - 5, my + 5)], fill=col, width=1)
                draw.line([(mx + 5, my - 5), (mx + 5, my + 5)], fill=col, width=1)
            else:
                # Breath comma
                draw.line([(mx + 1, my - 4), (mx + 2, my), (mx - 2, my + 5)], fill=col, width=rng.randint(1, 2))
        drew_any = True

    if not drew_any:
        _draw_hand_oval()

    return out


def _apply_archival_verse_block(
    img: Image.Image,
    rng: Any,
    ink_rgb_u8: Tuple[int, int, int],
) -> Image.Image:
    """Draw a fast handwritten-style verse/prose cursive squiggle block in bottom/top margin."""
    out = img.copy() if img.mode == "RGB" else img.convert("RGB")
    w, h = out.size
    if w < 64 or h < 64:
        return out
    draw = ImageDraw.Draw(out, "RGBA")
    place_bottom = rng.random() < 0.78
    n_lines = rng.randint(2, 5)
    line_gap = max(5, min(14, int(round(h * rng.uniform(0.014, 0.024)))))
    block_h = n_lines * line_gap
    if place_bottom:
        y_start = max(int(h * 0.76), h - block_h - max(6, int(h * rng.uniform(0.02, 0.07))))
    else:
        y_start = max(4, int(h * rng.uniform(0.02, 0.06)))
    x_left = int(w * rng.uniform(0.10, 0.28))
    max_span = max(32, int(w * rng.uniform(0.30, 0.62)))
    ir, ig, ib = ink_rgb_u8
    alpha = rng.randint(120, 195)
    col = (ir, ig, ib, alpha)
    step_x = max(4, min(7, w // 120))
    amp = max(1.5, min(3.5, line_gap * 0.28))

    for li in range(n_lines):
        y0 = y_start + li * line_gap
        if y0 < 3 or y0 >= h - 3:
            continue
        x_end = min(w - 6, x_left + int(max_span * rng.uniform(0.65, 1.0)))
        x = x_left + rng.randint(-4, 6)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        while x < x_end - 8:
            word_len = rng.randint(max(12, step_x * 3), max(22, step_x * 9))
            wx_end = min(x_end, x + word_len)
            pts: List[Tuple[int, int]] = []
            cx = x
            k = 0
            while cx <= wx_end:
                dy = math.sin(phase + k * 1.15) * amp + rng.uniform(-1.0, 1.0)
                pts.append((max(0, min(w - 1, cx)), max(0, min(h - 1, int(round(y0 + dy))))))
                cx += step_x
                k += 1
            if len(pts) >= 2:
                draw.line(pts, fill=col, width=1)
            x = wx_end + rng.randint(5, 12)
    return out


def _apply_archival_watermark_stain(
    img: Image.Image,
    rng: Any,
    ink_rgb_u8: Tuple[int, int, int],
) -> Image.Image:
    """Draw a faint semi-transparent archival watermark or corner botanical/ink-stain blob."""
    out = img.copy() if img.mode == "RGB" else img.convert("RGB")
    w, h = out.size
    if w < 64 or h < 64:
        return out
    draw = ImageDraw.Draw(out, "RGBA")
    alpha = int(round(rng.uniform(0.12, 0.22) * 255.0))
    ir, ig, ib = ink_rgb_u8
    # Warm sepia/amber stain or faint archival stamp tint
    sr = max(40, min(195, ir + rng.randint(10, 45)))
    sg = max(35, min(175, ig + rng.randint(5, 35)))
    sb = max(25, min(145, ib + rng.randint(-5, 25)))
    col = (sr, sg, sb, alpha)

    if rng.random() < 0.55:
        # Corner botanical / ink-stain cluster (2-4 soft overlapping ellipses)
        corner = rng.randint(0, 3)
        rx_base = max(12, int(w * rng.uniform(0.05, 0.13)))
        ry_base = max(12, int(h * rng.uniform(0.05, 0.13)))
        cx = int(w * rng.uniform(0.06, 0.20)) if corner in (0, 2) else int(w * rng.uniform(0.80, 0.94))
        cy = int(h * rng.uniform(0.06, 0.20)) if corner in (0, 1) else int(h * rng.uniform(0.80, 0.94))
        for _ in range(rng.randint(2, 4)):
            ox = rng.randint(-rx_base // 2, rx_base // 2)
            oy = rng.randint(-ry_base // 2, ry_base // 2)
            rx = max(6, int(rx_base * rng.uniform(0.55, 1.15)))
            ry = max(6, int(ry_base * rng.uniform(0.55, 1.15)))
            draw.ellipse([cx + ox - rx, cy + oy - ry, cx + ox + rx, cy + oy + ry], fill=col)
    else:
        # Faint semi-transparent library watermark oval/ring + inner cross-bar
        rx = max(18, int(w * rng.uniform(0.08, 0.18)))
        ry = max(14, int(h * rng.uniform(0.06, 0.14)))
        cx = rng.randint(rx + 6, max(rx + 6, w - rx - 6))
        cy = rng.randint(ry + 6, max(ry + 6, h - ry - 6))
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=col, width=max(2, min(4, w // 220)))
        irx, iry = int(rx * 0.72), int(ry * 0.72)
        draw.ellipse([cx - irx, cy - iry, cx + irx, cy + iry], outline=col, width=1)
        draw.line([(cx - irx, cy), (cx + irx, cy)], fill=col, width=1)
    return out


def apply_online_augment(
    img: Image.Image,
    rng: Optional[random.Random] = None,
    is_hw: bool = False,
    hw_archival_prob: Optional[float] = None,
) -> Image.Image:
    """Online phone-photo, scanner, and historical manuscript degradation.

    Supports dark desk border, perspective, rotation, shadows/vignette/parchment,
    brightness/contrast, 1-bit scanner threshold, blur, JPEG, sensor noise, and
    Historical Manuscript / Archival Mode (parchment + brown iron-gall ink
    transfer, mottled foxing, verso bleed-through, optional archival border +
    colorbar strip, and faint pre-ruled empty margin staves).

    Preserves exact ``(W, H)`` dimensions and is deterministic per ``rng`` seed.
    """
    r = rng if rng is not None else random
    out = img if img.mode == "RGB" else img.convert("RGB")
    orig_w, orig_h = out.size
    has_dark_border = False
    dark_fill = (25, 22, 20)

    archival_p = (0.55 if is_hw else 0.12) if hw_archival_prob is None else float(hw_archival_prob)
    do_archival = r.random() < archival_p

    def _sample_dark_desk() -> Tuple[int, int, int]:
        base = r.randint(4, 62)
        dr = max(0, min(70, base + r.randint(-5, 8)))
        dg = max(0, min(70, base + r.randint(-5, 6)))
        db = max(0, min(70, base + r.randint(-6, 5)))
        return (dr, dg, db)

    def _sample_paper_fill() -> Tuple[int, int, int]:
        bg_v = r.randint(235, 255)
        return (bg_v, max(225, bg_v - r.randint(0, 5)), max(215, bg_v - r.randint(0, 12)))

    # Random top/bottom white margin padding (p=0.25): pad 2-8% white margin on top/bottom
    # so top-aligned clean scores don't behave differently from bordered photos.
    if r.random() < 0.25 and orig_w >= 32 and orig_h >= 32:
        pad_t = max(2, int(round(orig_h * r.uniform(0.02, 0.08))))
        pad_b = max(2, int(round(orig_h * r.uniform(0.02, 0.08))))
        canvas = Image.new("RGB", (orig_w, orig_h + pad_t + pad_b), _sample_paper_fill())
        canvas.paste(out, (0, pad_t))
        out = canvas.resize((orig_w, orig_h), Image.Resampling.BILINEAR)

    if do_archival:
        # Historical Manuscript / Archival Mode:
        #   1) Optional System-1 left indentation (15-35%, p=0.10)
        #   2) Optional faint empty pre-ruled 5-line staves in margins / inter-system gaps (p=0.25)
        #   3) Parchment + brown iron-gall ink color transfer + mottled foxing + verso bleed
        #   4) Optional corner sticky-note/blot occlusion (p=0.12), archival border + colorbar (p=0.40)
        #   5) Optional low-res round-trip downscale + defocus blur + contrast compression
        if r.random() < 0.10 and orig_w >= 128 and orig_h >= 256:
            out = _apply_system1_left_indent(out, r)
        luma = np.asarray(out.convert("L"), dtype=np.float32) * (1.0 / 255.0)
        if r.random() < 0.25:
            _add_archival_empty_staves(luma, r)

        # Sample warm parchment paper RGB ~(215..245, 200..235, 170..215) with R > G > B
        pr = r.uniform(215.0, 245.0)
        pg = min(pr - r.uniform(4.0, 14.0), r.uniform(200.0, 235.0))
        pb = min(pg - r.uniform(10.0, 28.0), r.uniform(170.0, 215.0))
        paper_rgb = np.array([pr, max(190.0, pg), max(155.0, pb)], dtype=np.float32).reshape(1, 1, 3) / 255.0

        # Sample brown iron-gall ink RGB ~(70..140, 60..125, 45..105)
        ir = r.uniform(70.0, 140.0)
        ig = min(ir - r.uniform(3.0, 15.0), r.uniform(60.0, 125.0))
        ib = min(ig - r.uniform(5.0, 20.0), r.uniform(45.0, 105.0))
        ink_rgb = np.array([ir, max(50.0, ig), max(35.0, ib)], dtype=np.float32).reshape(1, 1, 3) / 255.0

        # Faint low-contrast sepia ink (p=0.25): compress ink-to-paper contrast for faint autographs
        if r.random() < 0.25:
            fade = r.uniform(0.35, 0.62)
            ink_rgb = ink_rgb * (1.0 - fade) + paper_rgb * fade

        ink_rgb_u8 = (
            int(round(float(ink_rgb[0, 0, 0]) * 255.0)),
            int(round(float(ink_rgb[0, 0, 1]) * 255.0)),
            int(round(float(ink_rgb[0, 0, 2]) * 255.0)),
        )

        h, w = luma.shape
        # Optional faint horizontally-flipped verso bleed-through (6-14% opacity) folded into luma
        if r.random() < 0.65:
            alpha = r.uniform(0.06, 0.14)
            bleed_luma = luma[:, ::-1]
            # Blend verso ink into luma directly (faster & equivalent to blending toward mid-brown ink)
            luma = luma * (1.0 - alpha * (1.0 - bleed_luma))

        # Low-frequency mottled foxing / aging via upsampled 16x16 noise field
        np_rng = np.random.default_rng(r.randint(0, 2**31 - 1))
        mottle_small = np.clip(
            128.0 + np_rng.standard_normal((16, 16), dtype=np.float32) * r.uniform(16.0, 32.0),
            0.0,
            255.0,
        ).astype(np.uint8)
        mottle = (
            np.asarray(
                Image.fromarray(mottle_small, mode="L").resize((w, h), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            - 128.0
        ) * (r.uniform(0.04, 0.11) / 128.0)

        # Gentle uneven illumination / edge darkening folded into a single scalar map
        illum_map = mottle + 1.0
        if r.random() < 0.60:
            gx = r.uniform(-0.16, 0.16)
            gy = r.uniform(-0.16, 0.16)
            vig = r.uniform(0.02, 0.18)
            xs = np.linspace(-0.5, 0.5, w, dtype=np.float32).reshape(1, w)
            ys = np.linspace(-0.5, 0.5, h, dtype=np.float32).reshape(h, 1)
            illum_map += gx * xs + gy * ys - vig * (xs * xs + ys * ys) * 2.0
            np.clip(illum_map, 0.62, 1.12, out=illum_map)

        # Fused sensor noise folded into luma/illum before RGB expansion
        if r.random() < 0.45:
            sigma = r.uniform(2.0, 7.5) / 255.0
            illum_map += np_rng.standard_normal((h, w), dtype=np.float32) * sigma

        # Single RGB materialization: ink_rgb * (1 - L) + paper_rgb * L, modulated by illum_map
        arr = ((paper_rgb - ink_rgb) * luma[..., None] + ink_rgb) * illum_map[..., None]
        np.clip(arr, 0.0, 1.0, out=arr)
        out = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")

        # Composer/copyist scratch-outs on archival pages (p=0.22 HW, p=0.06 printed)
        if r.random() < (0.22 if is_hw else 0.06):
            out = _apply_composer_scratchouts(out, r, is_hw=True)

        # Handwritten-style verse/prose text block in bottom/top margin quiet area (p=0.20)
        if r.random() < 0.20 and orig_w >= 64 and orig_h >= 64:
            out = _apply_archival_verse_block(out, r, ink_rgb_u8)

        # Faint semi-transparent watermark or corner botanical/ink-stain blob (p=0.15)
        if r.random() < 0.15 and orig_w >= 64 and orig_h >= 64:
            out = _apply_archival_watermark_stain(out, r, ink_rgb_u8)

        # Corner opaque sticky-note / archival label occlusion (p=0.12)
        if r.random() < 0.12 and orig_w >= 64 and orig_h >= 64:
            out = _apply_corner_sticky_note(out, r)

        if r.random() < 0.40 and orig_w >= 48 and orig_h >= 48:
            has_dark_border = True
            base_d = r.randint(6, 36)
            dark_fill = (base_d, max(4, base_d - 1), max(4, base_d - 2))
            out = _apply_archival_border_and_colorbar(out, r, dark_fill)

        # Mild rotation +/- 3.5 deg (p=0.35)
        if r.random() < 0.35:
            angle = r.uniform(-3.5, 3.5)
            rot_fill = (
                dark_fill
                if has_dark_border
                else (int(round(pr)), int(round(max(190.0, pg))), int(round(max(155.0, pb))))
            )
            out = out.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=rot_fill)

        # Low-res round-trip downscale + defocus blur + contrast compression (dense/landscape HW)
        lowres_p = 0.25 if orig_w > orig_h else 0.15
        if r.random() < lowres_p and orig_w >= 48 and orig_h >= 48:
            out = _apply_hw_lowres_defocus_compress(out, r)
        elif r.random() < 0.25:
            out = out.filter(ImageFilter.GaussianBlur(radius=r.uniform(0.3, 0.8)))
        if r.random() < 0.40:
            buf = io.BytesIO()
            out.save(buf, format="JPEG", quality=r.randint(45, 88))
            buf.seek(0)
            out = Image.open(buf).convert("RGB")
        return out

    # 0. Pre-warp paper-surface manual corrections, HW layout/sticky-note, & performer markup
    if is_hw and r.random() < 0.10 and orig_w >= 128 and orig_h >= 256:
        out = _apply_system1_left_indent(out, r)
    if r.random() < (0.22 if is_hw else 0.06):
        out = _apply_composer_scratchouts(out, r, is_hw=is_hw)
    if is_hw and r.random() < 0.12 and orig_w >= 64 and orig_h >= 64:
        out = _apply_corner_sticky_note(out, r)
    if not is_hw and r.random() < 0.18:
        out = _apply_performer_pencil_markup(out, r)

    # 1. Dark desk inset/border framing (p=0.50): pad 2-10% per side with dark desk RGB 0..70
    if r.random() < 0.50 and orig_w >= 32 and orig_h >= 32:
        has_dark_border = True
        dark_fill = _sample_dark_desk()
        pad_l = max(1, int(round(orig_w * r.uniform(0.02, 0.10))))
        pad_r = max(1, int(round(orig_w * r.uniform(0.02, 0.10))))
        pad_t = max(1, int(round(orig_h * r.uniform(0.02, 0.10))))
        pad_b = max(1, int(round(orig_h * r.uniform(0.02, 0.10))))
        cw = orig_w + pad_l + pad_r
        ch = orig_h + pad_t + pad_b
        canvas = Image.new("RGB", (cw, ch), dark_fill)
        if r.random() < 0.50:
            # Fast subtle desk gradient/noise on low-res grid upsampled to canvas
            np_rng = np.random.default_rng(r.randint(0, 2**31 - 1))
            gw, gh = 32, 32
            base_arr = np.array(dark_fill, dtype=np.float32).reshape(1, 1, 3)
            gx = r.uniform(-10.0, 10.0)
            gy = r.uniform(-10.0, 10.0)
            xs = np.linspace(-0.5, 0.5, gw, dtype=np.float32).reshape(1, gw, 1)
            ys = np.linspace(-0.5, 0.5, gh, dtype=np.float32).reshape(gh, 1, 1)
            noise = np_rng.uniform(-4.0, 4.0, size=(gh, gw, 3)).astype(np.float32)
            desk_small = np.clip(base_arr + gx * xs + gy * ys + noise, 0.0, 70.0).astype(np.uint8)
            canvas = Image.fromarray(desk_small, mode="RGB").resize((cw, ch), Image.Resampling.BILINEAR)
        canvas.paste(out, (pad_l, pad_t))
        out = canvas.resize((orig_w, orig_h), Image.Resampling.BILINEAR)

    # 2. Perspective quad warp (p=0.40): gentle 1-4% inward corner jitter
    if r.random() < 0.40 and orig_w >= 32 and orig_h >= 32:
        w_f, h_f = float(orig_w), float(orig_h)
        src_pts = [(0.0, 0.0), (w_f, 0.0), (w_f, h_f), (0.0, h_f)]
        dst_pts = [
            (w_f * r.uniform(0.01, 0.04), h_f * r.uniform(0.01, 0.04)),
            (w_f * (1.0 - r.uniform(0.01, 0.04)), h_f * r.uniform(0.01, 0.04)),
            (w_f * (1.0 - r.uniform(0.01, 0.04)), h_f * (1.0 - r.uniform(0.01, 0.04))),
            (w_f * r.uniform(0.01, 0.04), h_f * (1.0 - r.uniform(0.01, 0.04))),
        ]
        coeffs = _find_perspective_coeffs(dst_pts, src_pts)
        p_fill = dark_fill if (has_dark_border or r.random() < 0.50) else _sample_paper_fill()
        out = out.transform(
            (orig_w, orig_h),
            Image.Transform.PERSPECTIVE,
            coeffs,
            resample=Image.Resampling.BILINEAR,
            fillcolor=p_fill,
        )

    # 3. Rotation +/- 6 deg (p=0.50): dark desk fill half the time when dark-bordered, else paper tone
    if r.random() < 0.50:
        angle = r.uniform(-6.0, 6.0)
        if has_dark_border and r.random() < 0.50:
            rot_fill = (r.randint(0, 55), r.randint(0, 55), r.randint(0, 55))
        else:
            rot_fill = _sample_paper_fill()
        out = out.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=rot_fill,
        )

    # 4. Photometric pass (fused numpy for speed): uneven shadows/vignette/parchment + brightness/contrast
    illum_roll = r.random()
    do_bright = r.random() < 0.70
    do_contrast = r.random() < 0.70
    do_1bit = r.random() < 0.15
    if illum_roll < 0.70 or do_bright or do_contrast:
        arr = np.asarray(out, dtype=np.float32) * (1.0 / 255.0)
        h, w = arr.shape[:2]
        if illum_roll < 0.35:
            # Stronger uneven shadows / vignette / warm cream-tan parchment cast (illum down to 0.40)
            tint = np.array(
                [r.uniform(0.92, 1.02), r.uniform(0.86, 0.98), r.uniform(0.72, 0.94)],
                dtype=np.float32,
            ).reshape(1, 1, 3)
            gx = r.uniform(-0.45, 0.45)
            gy = r.uniform(-0.45, 0.45)
            vig = r.uniform(0.0, 0.35)
            xs = np.linspace(-0.5, 0.5, w, dtype=np.float32).reshape(1, w, 1)
            ys = np.linspace(-0.5, 0.5, h, dtype=np.float32).reshape(h, 1, 1)
            r2 = (xs * xs + ys * ys) * 2.0
            illum = np.clip(1.0 + gx * xs + gy * ys - vig * r2, 0.40, 1.15)
            arr *= tint * illum
        elif illum_roll < 0.70:
            tint = np.array(
                [r.uniform(0.95, 1.02), r.uniform(0.93, 1.00), r.uniform(0.87, 0.99)],
                dtype=np.float32,
            ).reshape(1, 1, 3)
            gx = r.uniform(-0.18, 0.18)
            gy = r.uniform(-0.18, 0.18)
            xs = np.linspace(-0.5, 0.5, w, dtype=np.float32).reshape(1, w, 1)
            ys = np.linspace(-0.5, 0.5, h, dtype=np.float32).reshape(h, 1, 1)
            illum = np.clip(1.0 + gx * xs + gy * ys, 0.75, 1.15)
            arr *= tint * illum

        if do_bright:
            arr *= r.uniform(0.80, 1.15)
        if do_contrast:
            c_fac = r.uniform(0.75, 1.25)
            arr = (arr - 0.5) * c_fac + 0.5

        if do_1bit:
            # 5. 1-bit / clean scanner threshold (p=0.15) matching crisp binary scans (e.g. suite31/photo)
            luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            thr = r.uniform(0.47, 0.76)
            bw = np.where(luma >= thr, np.uint8(255), np.uint8(0))
            return Image.fromarray(np.repeat(bw[..., None], 3, axis=2), mode="RGB")

        np.clip(arr, 0.0, 1.0, out=arr)
        out = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")
    elif do_1bit:
        gray = out.convert("L")
        thr = r.randint(120, 195)
        bw = gray.point(lambda px, t=thr: 255 if px >= t else 0, mode="L")
        return bw.convert("RGB")

    # 6. Mild Gaussian blur (or HW low-res round-trip + defocus + contrast compression)
    if is_hw and r.random() < (0.25 if orig_w > orig_h else 0.15) and orig_w >= 48 and orig_h >= 48:
        out = _apply_hw_lowres_defocus_compress(out, r)
    elif r.random() < 0.30:
        out = out.filter(ImageFilter.GaussianBlur(radius=r.uniform(0.3, 1.0)))

    # 7. JPEG compression artifacts
    if r.random() < 0.35:
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=r.randint(35, 85))
        buf.seek(0)
        out = Image.open(buf).convert("RGB")

    # 8. Additive Gaussian sensor noise (fast luma-dominant float32 noise)
    if r.random() < 0.45:
        arr = np.asarray(out, dtype=np.float32)
        sigma = r.uniform(2.5, 9.0)
        np_rng = np.random.default_rng(r.randint(0, 2**31 - 1))
        noise = np_rng.standard_normal((arr.shape[0], arr.shape[1], 1), dtype=np.float32) * sigma
        arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
        out = Image.fromarray(arr, mode="RGB")

    return out


def _resize_strip_pil(img: Image.Image, strip_height: int = 192, max_strip_width: int = 1024) -> Image.Image:
    orig_w, orig_h = img.size
    target_h = _round_mult16(strip_height, min_val=16)
    scale = target_h / max(1, orig_h)
    target_w = _round_mult16(orig_w * scale, min_val=16, max_val=max_strip_width)
    if (orig_w, orig_h) != (target_w, target_h):
        return img.resize((target_w, target_h), Image.Resampling.BILINEAR)
    return img


def _resize_fullpage_pil(img: Image.Image, fullpage_max_dim: int = 768) -> Image.Image:
    orig_w, orig_h = img.size
    max_side = max(orig_w, orig_h, 1)
    scale = min(1.0, float(fullpage_max_dim) / float(max_side))
    target_w = _round_mult16(orig_w * scale, min_val=16, max_val=fullpage_max_dim)
    target_h = _round_mult16(orig_h * scale, min_val=16, max_val=fullpage_max_dim)
    if (orig_w, orig_h) != (target_w, target_h):
        return img.resize((target_w, target_h), Image.Resampling.BILINEAR)
    return img


def _pil_to_chw_tensor(img: Image.Image) -> torch.Tensor:
    rgb = img if img.mode == "RGB" else img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32) * (1.0 / 255.0)
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)


def _pil_to_chw_uint8(img: Image.Image) -> torch.Tensor:
    rgb = img if img.mode == "RGB" else img.convert("RGB")
    arr = np.array(rgb, dtype=np.uint8, copy=True)
    return torch.from_numpy(arr).permute(2, 0, 1)


def _images_to_device_float(
    images: torch.Tensor,
    device: torch.device,
    non_blocking: bool = True,
) -> torch.Tensor:
    t = images.to(device, non_blocking=non_blocking)
    if t.dtype == torch.uint8:
        return t.float().div_(255.0)
    return t


def _cap_token_ids(ids: List[int], max_seq_len: int) -> List[int]:
    if len(ids) <= max_seq_len:
        return ids
    return ids[: max(2, max_seq_len) - 1] + [EOS_ID]


class UniStripDataset(Dataset):
    """Real single-staff strip dataset ``(png, *-bekern.txt)``."""

    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        strip_height: int = 192,
        max_strip_width: int = 1024,
        max_seq_len: int = 1024,
        augment: bool = False,
        aug_prob: float = 0.75,
        kind: str = "strip",
        return_uint8: bool = False,
        is_hw: Optional[bool] = None,
    ) -> None:
        self.pairs = list(pairs)
        self.strip_height = int(strip_height)
        self.max_strip_width = int(max_strip_width)
        self.max_seq_len = int(max_seq_len)
        self.augment = bool(augment)
        self.aug_prob = float(max(0.0, min(1.0, aug_prob)))
        self.kind = str(kind)
        self.return_uint8 = bool(return_uint8)
        self.is_hw = bool(is_hw) if is_hw is not None else (self.kind == "hw_strip")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, bekern_path = self.pairs[idx]
        with Image.open(img_path) as raw:
            rgb = raw if raw.mode == "RGB" else raw.convert("RGB")
            img = _resize_strip_pil(rgb, self.strip_height, self.max_strip_width)
            if img is raw:
                raw.load()
                img = raw.copy()
        if self.augment and self.aug_prob > 0.0:
            if random.random() < self.aug_prob:
                img = apply_online_augment(img, is_hw=self.is_hw)
            elif random.random() < 0.25:
                img = apply_clean_scale_jitter(img)
        toks = data.load_bekern(bekern_path)
        ids = _cap_token_ids([BOS_ID] + data.encode(toks) + [EOS_ID], self.max_seq_len)
        img_t = _pil_to_chw_uint8(img) if self.return_uint8 else _pil_to_chw_tensor(img)
        return {
            "image": img_t,
            "seq": torch.tensor(ids, dtype=torch.long),
            "sample_id": img_path.stem,
            "kind": self.kind,
        }


def _sample_weighted_fullpage_dim(base_dim: int, rng: Optional[random.Random] = None) -> int:
    """Weighted realistic multi-scale max_dim (multiple of 16):
    15% low-res (512..768px), 60% sweet-spot (896..1152px), 25% high-res (1280..1536px).
    """
    r = rng if rng is not None else random
    u = r.random()
    if u < 0.15:
        choices = (512, 576, 640, 704, 768)
    elif u < 0.75:
        choices = (896, 960, 1024, 1088, 1152)
    else:
        choices = (1280, 1344, 1408, 1536)
    hi = max(512, int(base_dim))
    valid = [c for c in choices if c <= hi]
    return int(r.choice(valid)) if valid else _round_mult16(hi, min_val=512)


class UniFullPageDataset(Dataset):
    """Real full-page dataset stitching ordered systems with ``<sys>``."""

    def __init__(
        self,
        pairs: Sequence[Tuple[Any, ...]],
        fullpage_max_dim: int = 768,
        max_seq_len: int = 1024,
        augment: bool = False,
        aug_prob: float = 0.75,
        multi_scale: bool = False,
        return_uint8: bool = False,
        domains: Optional[Sequence[str]] = None,
    ) -> None:
        self.pairs = list(pairs)
        self.fullpage_max_dim = int(fullpage_max_dim)
        self.max_seq_len = int(max_seq_len)
        self.augment = bool(augment)
        self.aug_prob = float(max(0.0, min(1.0, aug_prob)))
        self.multi_scale = bool(multi_scale)
        self.return_uint8 = bool(return_uint8)
        self.domains = [str(d).strip().lower() for d in domains] if domains is not None else None
        self._batch_target_dim: Optional[int] = None

    def set_target_dim(self, dim: Optional[int]) -> None:
        """Set a shared per-batch target max_dim (pass None to clear)."""
        self._batch_target_dim = int(dim) if dim is not None else None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict[str, Any]:
        if isinstance(idx, tuple):
            sample_idx, target_dim = int(idx[0]), int(idx[1])
        else:
            sample_idx = int(idx)
            if self._batch_target_dim is not None:
                target_dim = self._batch_target_dim
            elif self.multi_scale:
                target_dim = _sample_weighted_fullpage_dim(self.fullpage_max_dim)
            else:
                target_dim = self.fullpage_max_dim

        entry = self.pairs[sample_idx]
        img_path, bekern_paths = entry[0], entry[1]
        if len(entry) > 2:
            domain = str(entry[2]).strip().lower()
        elif self.domains is not None:
            domain = self.domains[sample_idx]
        else:
            domain = ""
        is_hw = domain in ("hw", "real_hw")

        with Image.open(img_path) as raw:
            rgb = raw if raw.mode == "RGB" else raw.convert("RGB")
            img = _resize_fullpage_pil(rgb, target_dim)
            if img is raw:
                raw.load()
                img = raw.copy()
        if self.augment and self.aug_prob > 0.0:
            if random.random() < self.aug_prob:
                img = apply_online_augment(img, is_hw=is_hw)
            elif random.random() < 0.25:
                img = apply_clean_scale_jitter(img)

        stitched: List[str] = []
        for s_idx, b_path in enumerate(bekern_paths):
            if s_idx > 0:
                stitched.append(SYS_TOKEN)
            stitched.extend(data.load_bekern(b_path))

        ids = _cap_token_ids(encode_uni(stitched, add_bos_eos=True), self.max_seq_len)
        img_t = _pil_to_chw_uint8(img) if self.return_uint8 else _pil_to_chw_tensor(img)
        return {
            "image": img_t,
            "seq": torch.tensor(ids, dtype=torch.long),
            "sample_id": img_path.stem,
            "kind": "fullpage",
            "domain": domain,
        }


class MultiScaleBatchSampler:
    """Batch sampler that assigns one shared ``target_dim`` to every item in a batch.

    Yields lists of ``(sample_idx, target_dim)`` tuples so ``UniFullPageDataset``
    resizes all batch items to the exact same resolution with zero white-pad waste
    and full thread/worker safety.
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        base_dim: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.num_samples = int(num_samples)
        self.batch_size = max(1, int(batch_size))
        self.base_dim = int(base_dim)
        self.shuffle = bool(shuffle)
        self._rng = random.Random(seed) if seed is not None else None

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        indices = list(range(self.num_samples))
        if self.shuffle:
            if self._rng is not None:
                self._rng.shuffle(indices)
            else:
                random.shuffle(indices)
        for i in range(0, self.num_samples, self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            batch_dim = _sample_weighted_fullpage_dim(self.base_dim, rng=self._rng)
            yield [(idx, batch_dim) for idx in batch_idx]


def collate_uni_batch(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad variable-resolution images (white paper=1.0 or 255) and sequences (``PAD_ID``)."""
    B = len(samples)
    max_h = _round_mult16(max(int(s["image"].shape[1]) for s in samples), min_val=16)
    max_w = _round_mult16(max(int(s["image"].shape[2]) for s in samples), min_val=16)
    max_t = max(int(s["seq"].shape[0]) for s in samples)

    is_u8 = B > 0 and samples[0]["image"].dtype == torch.uint8
    if is_u8:
        images = torch.full((B, 3, max_h, max_w), 255, dtype=torch.uint8)
    else:
        images = torch.ones((B, 3, max_h, max_w), dtype=torch.float32)
    seqs = torch.full((B, max_t), PAD_ID, dtype=torch.long)
    sample_ids: List[str] = []
    for i, s in enumerate(samples):
        im = s["image"]
        _, h, w = im.shape
        images[i, :, :h, :w] = im
        sq = s["seq"]
        seqs[i, : sq.shape[0]] = sq
        sample_ids.append(str(s.get("sample_id", i)))

    return {
        "images": images,
        "seq": seqs,
        "sample_ids": sample_ids,
        "kind": samples[0].get("kind", "unknown") if B > 0 else "empty",
    }


def _fast_image_is_landscape(path: Any) -> Optional[bool]:
    """Header-only orientation probe (True if W > H, False if H >= W, None on failure)."""
    try:
        with Image.open(path) as im:
            w, h = im.size
            return bool(w > h)
    except Exception:
        return None


class AsyncThreadPrefetcher:
    """Lightweight infinite multi-threaded batch prefetcher (zero Windows spawn overhead).

    Prefetches ``prefetch_batches`` (default 3) batches ahead using a background
    ``ThreadPoolExecutor``. Because PIL image decode/resize and NumPy augmentation
    release the Python GIL, this overlaps CPU I/O + augmentation with GPU compute
    without duplicating dataset state across spawned processes. For ``multi_scale``
    ``UniFullPageDataset``, samples a single shared ``target_dim`` per batch.
    When ``batch_size >= 2`` on ``UniFullPageDataset``, partitions shuffled indices
    into portrait (``H >= W``) and landscape (``W > H``) buckets so batches never
    mix orientations into a giant padded square.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        pin_memory: bool = False,
        max_workers: int = 6,
        prefetch_batches: int = 3,
        collate_fn: Any = collate_uni_batch,
        seed: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.n_samples = len(dataset)  # type: ignore[arg-type]
        if self.n_samples <= 0:
            raise ValueError("AsyncThreadPrefetcher requires a non-empty dataset")
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.pin_memory = bool(pin_memory)
        self.max_workers = max(1, int(max_workers))
        self.prefetch_batches = max(6, min(16, max(int(prefetch_batches), self.max_workers * 2)))
        self.collate_fn = collate_fn
        self._rng = random.Random(seed) if seed is not None else random.Random()
        self._is_multiscale_fp = bool(
            getattr(dataset, "multi_scale", False)
            and hasattr(dataset, "fullpage_max_dim")
        )
        self._use_orient_buckets = bool(
            self.batch_size >= 2
            and isinstance(dataset, UniFullPageDataset)
            and hasattr(dataset, "pairs")
        )
        self._portrait_indices: List[int] = []
        self._landscape_indices: List[int] = []
        self._batches: List[List[int]] = []
        self._batch_pos: int = 0
        if self._use_orient_buckets:
            for i, entry in enumerate(dataset.pairs):  # type: ignore[attr-defined]
                if _fast_image_is_landscape(entry[0]):
                    self._landscape_indices.append(i)
                else:
                    self._portrait_indices.append(i)
            if not self._portrait_indices or not self._landscape_indices:
                self._use_orient_buckets = False
        self._order: List[int] = []
        self._pos: int = 0
        self._closed: bool = False
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="uni_prefetch",
        )
        self._queue: deque[Future[Dict[str, Any]]] = deque()
        for _ in range(self.prefetch_batches):
            self._enqueue_one()

    def _next_batch_spec(self) -> List[Any]:
        if self._use_orient_buckets:
            if self._batch_pos >= len(self._batches):
                p_idx = list(self._portrait_indices)
                l_idx = list(self._landscape_indices)
                if self.shuffle:
                    self._rng.shuffle(p_idx)
                    self._rng.shuffle(l_idx)
                batches: List[List[int]] = []
                for i in range(0, len(p_idx), self.batch_size):
                    batches.append(p_idx[i : i + self.batch_size])
                for i in range(0, len(l_idx), self.batch_size):
                    batches.append(l_idx[i : i + self.batch_size])
                if self.shuffle:
                    self._rng.shuffle(batches)
                self._batches = batches
                self._batch_pos = 0
            idxs = self._batches[self._batch_pos]
            self._batch_pos += 1
        else:
            if self._pos >= len(self._order):
                self._order = list(range(self.n_samples))
                if self.shuffle:
                    self._rng.shuffle(self._order)
                self._pos = 0
            end = min(self._pos + self.batch_size, len(self._order))
            idxs = self._order[self._pos : end]
            self._pos = end
        if self._is_multiscale_fp:
            batch_dim = _sample_weighted_fullpage_dim(
                int(getattr(self.dataset, "fullpage_max_dim")), rng=self._rng
            )
            return [(i, batch_dim) for i in idxs]
        return idxs

    def _load_batch(self, spec: List[Any]) -> Dict[str, Any]:
        samples = [self.dataset[item] for item in spec]
        # Never call .pin_memory() from background threads on Windows WDDM:
        # cudaHostAlloc takes a global CUDA driver lock and stalls the main GPU thread.
        return self.collate_fn(samples)

    def _enqueue_one(self) -> None:
        if not self._closed:
            spec = self._next_batch_spec()
            self._queue.append(self._executor.submit(self._load_batch, spec))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self

    def __next__(self) -> Dict[str, Any]:
        if self._closed:
            raise StopIteration
        if not self._queue:
            self._enqueue_one()
        fut = self._queue.popleft()
        self._enqueue_one()
        return fut.result()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.clear()
            self._executor.shutdown(wait=False, cancel_futures=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _make_synthetic_batch(
    curriculum: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    step: int = 0,
    with_uv: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Generate deterministic synthetic strip/fullpage batch for dry-run & smoke tests."""
    g = torch.Generator(device="cpu").manual_seed(1337 + step)
    if curriculum == "strip":
        H, W = 128, 384
    elif curriculum == "fullpage":
        H, W = 384, 288
    else:  # mixed
        H, W = (128, 384) if (step % 2 == 0) else (384, 288)

    images = torch.rand((batch_size, 3, H, W), generator=g).to(device)
    # Valid bekern token IDs in [0, len(data.VOCAB)-1], plus occasional <sys> on fullpage
    max_tok = len(data.VOCAB)
    body = torch.randint(0, max_tok, (batch_size, seq_len - 2), generator=g)
    if curriculum in ("fullpage", "mixed") and seq_len > 6:
        body[:, seq_len // 2] = SYS_ID
    bos = torch.full((batch_size, 1), BOS_ID, dtype=torch.long)
    eos = torch.full((batch_size, 1), EOS_ID, dtype=torch.long)
    seq = torch.cat([bos, body, eos], dim=1).to(device)

    uv_target: Optional[torch.Tensor] = None
    if with_uv:
        ys = torch.linspace(0.0, 1.0, H // 16).view(1, 1, H // 16, 1).expand(batch_size, 1, H // 16, W // 16)
        xs = torch.linspace(0.0, 1.0, W // 16).view(1, 1, 1, W // 16).expand(batch_size, 1, H // 16, W // 16)
        valid = torch.ones((batch_size, 1, H // 16, W // 16))
        uv_target = torch.cat([xs, ys, valid], dim=1).to(device)
    return images, seq, uv_target


# ---------------------------------------------------------------------------
# Evaluation Metrics (Levenshtein Token SER, Token F1, System Count, Speed)
# ---------------------------------------------------------------------------

def levenshtein_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Exact Levenshtein edit distance between two token sequences."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    if n < m:
        ref, hyp = hyp, ref
        n, m = m, n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            a = prev[j] + 1
            b = curr[j - 1] + 1
            c = prev[j - 1] + cost
            curr[j] = a if a <= b and a <= c else (b if b <= c else c)
        prev, curr = curr, prev
    return prev[m]


def compute_token_f1(ref: Sequence[str], hyp: Sequence[str]) -> Tuple[float, float, float]:
    """Multiset precision, recall, and F1 over token sequences."""
    c_ref = Counter(ref)
    c_hyp = Counter(hyp)
    overlap = sum((c_ref & c_hyp).values())
    n_hyp = len(hyp)
    n_ref = len(ref)
    if n_hyp == 0 and n_ref == 0:
        return 1.0, 1.0, 1.0
    prec = overlap / n_hyp if n_hyp > 0 else 0.0
    rec = overlap / n_ref if n_ref > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
    return prec, rec, f1


@torch.no_grad()
def eval_dataset(
    model: UniOMRNet,
    val_samples: Any,
    device: torch.device,
    max_len: int = 256,
    batch_size: int = 4,
    use_bf16: bool = False,
) -> Dict[str, Any]:
    """Evaluate ``model`` with KV-cached ``generate()`` on held-out strips or full pages.

    Computes Levenshtein Token SER, multiset Token F1, exact ``<sys>`` count
    accuracy, mean tokens/sec, and mean ``InkTokenRouter`` compression.
    """
    was_training = model.training
    model.eval()

    if isinstance(val_samples, Dataset):
        eval_ds: Dataset = val_samples
        if batch_size > 1 and hasattr(val_samples, "pairs") and len(val_samples) > 1:  # type: ignore[arg-type]
            sort_keys: List[Tuple[int, int, int]] = []
            for idx, entry in enumerate(val_samples.pairs):  # type: ignore[attr-defined]
                is_land = 1 if _fast_image_is_landscape(entry[0]) else 0
                bek_part = entry[1]
                try:
                    if isinstance(bek_part, (list, tuple)):
                        seq_len_est = sum(len(data.load_bekern(p)) for p in bek_part)
                    else:
                        seq_len_est = len(data.load_bekern(bek_part))
                except Exception:
                    seq_len_est = 0
                sort_keys.append((is_land, seq_len_est, idx))
            sort_keys.sort()
            eval_ds = Subset(val_samples, [idx for _, _, idx in sort_keys])
        loader = DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_uni_batch,
            pin_memory=(device.type == "cuda"),
        )
    elif isinstance(val_samples, DataLoader):
        loader = val_samples
    else:
        sample_list = list(val_samples)
        if not sample_list:
            if was_training:
                model.train()
            return {
                "n_samples": 0,
                "ser": 0.0,
                "mean_sample_ser": 0.0,
                "token_f1": 0.0,
                "token_precision": 0.0,
                "token_recall": 0.0,
                "sys_count_exact": 0.0,
                "tokens_per_sec": 0.0,
                "mean_compression_x": 0.0,
                "ink_recall": 0.0,
            }
        if batch_size > 1 and len(sample_list) > 1:
            sample_list = sorted(
                sample_list,
                key=lambda s: (
                    1 if int(s["image"].shape[2]) > int(s["image"].shape[1]) else 0,
                    int(s["seq"].shape[0]),
                ),
            )
        loader = [
            collate_uni_batch(sample_list[i : i + batch_size])
            for i in range(0, len(sample_list), batch_size)
        ]

    total_edits = 0
    total_ref_toks = 0
    total_pred_toks = 0
    total_gen_steps = 0
    sample_sers: List[float] = []
    f1s: List[float] = []
    precs: List[float] = []
    recs: List[float] = []
    sys_exact_hits = 0
    n_samples = 0
    compressions: List[float] = []
    ink_recalls: List[float] = []

    use_cuda_bf16 = bool(use_bf16 and device.type == "cuda")
    t_start = time.perf_counter()
    for batch in loader:
        images = _images_to_device_float(
            batch["images"], device, non_blocking=(device.type == "cuda")
        )
        seqs = batch["seq"]
        if use_cuda_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_ids, rinfo = model.generate(images, max_len=max_len, return_info=True)
        else:
            pred_ids, rinfo = model.generate(images, max_len=max_len, return_info=True)
        if device.type == "cuda":
            torch.cuda.synchronize()

        n_in = int(rinfo["n_in"])
        n_out = int(rinfo["n_out"])
        comp = n_in / max(1, n_out)
        ink_rec = float(rinfo.get("ink_recall", 1.0))

        pred_cpu = pred_ids.detach().cpu().tolist()
        ref_cpu = seqs.detach().cpu().tolist()
        for p_row, r_row in zip(pred_cpu, ref_cpu):
            pred_toks = decode_uni(p_row, skip_special=True)
            ref_toks = decode_uni(r_row, skip_special=True)
            edits = levenshtein_distance(ref_toks, pred_toks)
            total_edits += edits
            total_ref_toks += len(ref_toks)
            total_pred_toks += len(pred_toks)
            total_gen_steps += max(1, len(pred_toks) + 1)
            sample_sers.append(edits / max(1, len(ref_toks)))

            p, r, f1 = compute_token_f1(ref_toks, pred_toks)
            precs.append(p)
            recs.append(r)
            f1s.append(f1)

            if pred_toks.count(SYS_TOKEN) == ref_toks.count(SYS_TOKEN):
                sys_exact_hits += 1
            compressions.append(comp)
            ink_recalls.append(ink_rec)
            n_samples += 1

    elapsed = max(1e-6, time.perf_counter() - t_start)
    if was_training:
        model.train()

    return {
        "n_samples": n_samples,
        "ser": round(total_edits / max(1, total_ref_toks), 6),
        "mean_sample_ser": round(float(np.mean(sample_sers)) if sample_sers else 0.0, 6),
        "token_f1": round(float(np.mean(f1s)) if f1s else 0.0, 6),
        "token_precision": round(float(np.mean(precs)) if precs else 0.0, 6),
        "token_recall": round(float(np.mean(recs)) if recs else 0.0, 6),
        "sys_count_exact": round(sys_exact_hits / max(1, n_samples), 6),
        "tokens_per_sec": round(total_gen_steps / elapsed, 2),
        "mean_compression_x": round(float(np.mean(compressions)) if compressions else 0.0, 3),
        "ink_recall": round(float(np.mean(ink_recalls)) if ink_recalls else 0.0, 4),
        "total_edits": total_edits,
        "total_ref_tokens": total_ref_toks,
        "total_pred_tokens": total_pred_toks,
    }


# ---------------------------------------------------------------------------
# Browser Deployment Benchmark
# ---------------------------------------------------------------------------

def run_bench_browser(
    cfg: UniOMRConfig,
    device_str: str = "cpu",
    warmup: int = 1,
    iters: int = 3,
    dec_len: int = 64,
    ar_tokens: int = 32,
) -> Dict[str, Any]:
    """Benchmark parameter counts, quantized sizes, token compression, and latency.

    Includes both parallel teacher-forced latency and real KV-cached
    autoregressive ``ms/token`` on CPU/GPU for honest browser estimates.
    """
    device = torch.device(device_str)
    model = UniOMRNet(cfg).to(device)
    model.eval()

    summary = model.param_summary()
    shapes = [
        ("strip_256x1024", (1, 3, 256, 1024)),
        ("fullpage_1024x768", (1, 3, 1024, 768)),
    ]
    benchmarks: Dict[str, Dict[str, Any]] = {}

    with torch.no_grad():
        tgt = torch.full((1, dec_len), BOS_ID, dtype=torch.long, device=device)
        for name, shape in shapes:
            x = torch.rand(shape, device=device)
            # Warmup parallel forward
            for _ in range(warmup):
                out = model(x, tgt, return_aux=False, return_mtp=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                out = model(x, tgt, return_aux=False, return_mtp=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0 / max(1, iters)

            # Separate encode + KV-cached autoregressive decode timing
            t_enc0 = time.perf_counter()
            mem, _, rinfo_enc = model.encode_image(x, return_aux=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            encode_ms = (time.perf_counter() - t_enc0) * 1000.0

            cur_ids = torch.full((1, 1), BOS_ID, dtype=torch.long, device=device)
            past_kvs: Optional[List[Any]] = None
            t_ar0 = time.perf_counter()
            for _ in range(ar_tokens):
                logits, _, past_kvs = model.decode_tokens(
                    cur_ids,
                    memory=mem if past_kvs is None else None,
                    return_mtp=False,
                    past_kvs=past_kvs,
                    use_cache=True,
                )
                cur_ids = logits[:, -1:, :].argmax(dim=-1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ar_total_ms = (time.perf_counter() - t_ar0) * 1000.0
            ar_ms_per_tok = ar_total_ms / max(1, ar_tokens)
            ar_toks_per_sec = 1000.0 / max(1e-6, ar_ms_per_tok)

            rinfo = out["router_info"]
            n_in = int(rinfo["n_in"])
            n_out = int(rinfo["n_out"])
            ratio = n_in / max(1, n_out)
            benchmarks[name] = {
                "input_shape": list(shape),
                "grid_hw": list(rinfo["grid_hw"]),
                "tokens_in": n_in,
                "tokens_out": n_out,
                "compression_x": round(ratio, 3),
                "latency_ms": round(dt_ms, 2),
                "encode_ms": round(encode_ms, 2),
                "ar_ms_per_token": round(ar_ms_per_tok, 3),
                "ar_tokens_per_sec": round(ar_toks_per_sec, 2),
                "uv_skipped": out["uv_pred"] is None,
                "mtp_skipped": len(out["mtp_logits"]) == 0,
            }

    result = {
        "device": str(device),
        "config": asdict(cfg),
        "params": summary,
        "benchmarks": benchmarks,
    }

    print("=" * 78)
    print("UniOMR Browser Deployment Benchmark")
    print("=" * 78)
    print(f"Device              : {device}")
    print(f"Vocab size          : {cfg.vocab_size} (data.VOCAB={len(data.VOCAB)} + PAD/BOS/EOS/<sys>)")
    print(f"Total params (train): {summary['total_params']:,} ({summary['total_params'] / 1e6:.2f}M)")
    print(f"Infer params (core) : {summary['infer_params']:,} ({summary['infer_params'] / 1e6:.2f}M)")
    print(f"  UV aux head       : {summary['uv_aux_params']:,} (0% infer cost)")
    print(f"  MTP-{cfg.mtp_k} aux heads   : {summary['mtp_aux_params']:,} (0% std infer cost)")
    print("-" * 78)
    print(f"Inference Weight Sizes (core {summary['infer_params'] / 1e6:.2f}M params):")
    print(f"  FP32 : {summary['fp32_mb']:6.2f} MB")
    print(f"  FP16 : {summary['fp16_mb']:6.2f} MB")
    print(f"  INT8 : {summary['int8_mb']:6.2f} MB  ({'PASS < 55 MB' if summary['int8_mb'] < 55.0 else 'OVER 55 MB'})")
    print(f"  INT4 : {summary['int4_mb']:6.2f} MB  ({'PASS < 55 MB' if summary['int4_mb'] < 55.0 else 'OVER 55 MB'})")
    print("-" * 78)
    print(
        f"InkTokenRouter (keep_ratio={cfg.keep_ratio:.2f}, bg_summary={cfg.n_bg_summary}, "
        f"dec_len={dec_len}, ar_tokens={ar_tokens}):"
    )
    for name, b in benchmarks.items():
        print(
            f"  {name:18s} | grid {b['grid_hw'][0]:2d}x{b['grid_hw'][1]:2d} | "
            f"tokens {b['tokens_in']:4d} -> {b['tokens_out']:4d} ({b['compression_x']:.2f}x) | "
            f"fwd {b['latency_ms']:7.2f} ms | enc {b['encode_ms']:6.1f} ms | "
            f"KV-AR {b['ar_ms_per_token']:5.2f} ms/tok ({b['ar_tokens_per_sec']:5.1f} tok/s)"
        )
    print("=" * 78)
    return result


# ---------------------------------------------------------------------------
# Training Loop (Real Data + Cosine Schedule + Eval + --dry-run)
# ---------------------------------------------------------------------------

def _lr_for_step(step: int, total_steps: int, base_lr: float, warmup_steps: int, min_lr_ratio: float = 0.05) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * (float(step) / float(max(1, warmup_steps)))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def _infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


# Interleaved 70% strip (0) / 30% fullpage (1) schedule for --curriculum mixed.
_MIXED_SCHEDULE_10 = (0, 0, 1, 0, 0, 1, 0, 0, 0, 1)


class StripDomainMixer:
    """Deterministic fractional scheduler guaranteeing exact printed vs. HW strip ratio."""

    def __init__(self, hw_frac: float = 0.30) -> None:
        self.hw_frac = float(max(0.0, min(1.0, hw_frac)))
        self._acc: float = 0.0

    def next_is_hw(self) -> bool:
        if self.hw_frac <= 0.0:
            return False
        if self.hw_frac >= 1.0:
            return True
        self._acc += self.hw_frac
        if self._acc >= 1.0 - 1e-9:
            self._acc -= 1.0
            return True
        return False


class CPUEmaTeacher:
    """CPU-offloaded Exponential Moving Average teacher (0 MB VRAM between unsup rounds)."""

    def __init__(self, model: UniOMRNet, decay: float = 0.995) -> None:
        self.decay = float(decay)
        self.cfg = model.cfg
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().to(device="cpu", dtype=torch.float32 if v.is_floating_point() else v.dtype).clone()
            for k, v in model.state_dict().items()
        }
        self._teacher: Optional[UniOMRNet] = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                continue
            tgt = self.shadow[k]
            src = v.detach().to(device="cpu", dtype=tgt.dtype)
            if tgt.is_floating_point():
                tgt.mul_(d).add_(src, alpha=1.0 - d)
            else:
                tgt.copy_(src)

    def load_to_device(self, device: torch.device) -> UniOMRNet:
        if self._teacher is None:
            self._teacher = UniOMRNet(self.cfg)
        self._teacher.load_state_dict(self.shadow, strict=True)
        self._teacher.to(device)
        self._teacher.eval()
        return self._teacher

    def offload_to_cpu(self) -> None:
        if self._teacher is not None:
            self._teacher.to("cpu")


class GPUEmaShadow:
    """Lightweight device-resident EMA shadow state_dict updated every ``update_every`` steps."""

    def __init__(self, model: nn.Module, decay: float = 0.997, update_every: int = 5) -> None:
        self.decay = float(decay)
        self.update_every = max(1, int(update_every))
        self.decay_eff = self.decay ** self.update_every
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> bool:
        if step % self.update_every != 0:
            return False
        d = self.decay_eff
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                continue
            tgt = self.shadow[k]
            src = v.detach()
            if tgt.is_floating_point():
                tgt.mul_(d).add_(src.to(dtype=tgt.dtype), alpha=1.0 - d)
            else:
                tgt.copy_(src)
        return True

    @contextlib.contextmanager
    def swap_into(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap EMA weights into ``model`` for evaluation (0 extra GPU VRAM)."""
        backup = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=True)


def run_unsup_ema_round(
    student: UniOMRNet,
    ema_teacher: CPUEmaTeacher,
    opt: torch.optim.Optimizer,
    unlabeled_paths: Sequence[Path],
    device: torch.device,
    curriculum: str = "mixed",
    strip_height: int = 192,
    max_strip_width: int = 1024,
    fullpage_max_dim: int = 768,
    unsup_steps: int = 25,
    unsup_conf_thr: float = 0.85,
    unsup_weight: float = 0.5,
    max_gen_len: int = 256,
    mtp_loss_weight: float = 0.30,
    use_bf16: bool = False,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Run confidence-gated weak->strong EMA self-training on unlabeled images."""
    if not unlabeled_paths or unsup_steps <= 0:
        return {
            "accepted": 0,
            "total": 0,
            "accept_rate": 0.0,
            "unsup_loss": 0.0,
            "mean_conf": 0.0,
        }

    r = rng if rng is not None else random
    was_training = student.training
    teacher = ema_teacher.load_to_device(device)
    accepted = 0
    total = 0
    loss_sum = 0.0
    conf_sum = 0.0

    try:
        student.train()
        for _ in range(int(unsup_steps)):
            img_path = unlabeled_paths[r.randrange(len(unlabeled_paths))]
            with Image.open(img_path) as raw:
                raw_rgb = raw.convert("RGB")
                if curriculum == "strip":
                    base_img = _resize_strip_pil(raw_rgb, strip_height, max_strip_width)
                else:
                    base_img = _resize_fullpage_pil(raw_rgb, fullpage_max_dim)

            weak_img = apply_weak_augment(base_img, rng=r)
            weak_t = _pil_to_chw_tensor(weak_img).unsqueeze(0).to(device, non_blocking=True)

            with torch.no_grad():
                if use_bf16:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        pred_seq, mean_conf_t, reached_eos_t = teacher.generate(
                            weak_t, max_len=max_gen_len, return_conf=True
                        )
                else:
                    pred_seq, mean_conf_t, reached_eos_t = teacher.generate(
                        weak_t, max_len=max_gen_len, return_conf=True
                    )

            total += 1
            mean_conf = float(mean_conf_t[0].item())
            hit_eos = bool(reached_eos_t[0].item())
            conf_sum += mean_conf

            if not hit_eos or mean_conf < float(unsup_conf_thr):
                continue

            row = pred_seq[0]
            eos_pos = (row == EOS_ID).nonzero(as_tuple=False)
            if eos_pos.numel() == 0:
                continue
            end_idx = int(eos_pos[0].item()) + 1
            pseudo_seq = pred_seq[:, :end_idx]
            if pseudo_seq.shape[1] < 6:
                continue

            accepted += 1
            strong_img = apply_online_augment(base_img, rng=r)
            strong_t = _pil_to_chw_tensor(strong_img).unsqueeze(0).to(device, non_blocking=True)
            tgt_in = pseudo_seq[:, :-1]
            tgt_out = pseudo_seq[:, 1:]

            opt.zero_grad(set_to_none=True)
            if use_bf16:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    out = student(strong_t, tgt_in, return_aux=False, return_mtp=(student.cfg.mtp_k > 1))
                    losses = student.compute_loss(
                        out,
                        tgt_out,
                        uv_target=None,
                        mtp_loss_weight=mtp_loss_weight,
                        uv_loss_weight=0.0,
                    )
                    u_loss = losses["loss"] * float(unsup_weight)
            else:
                out = student(strong_t, tgt_in, return_aux=False, return_mtp=(student.cfg.mtp_k > 1))
                losses = student.compute_loss(
                    out,
                    tgt_out,
                    uv_target=None,
                    mtp_loss_weight=mtp_loss_weight,
                    uv_loss_weight=0.0,
                )
                u_loss = losses["loss"] * float(unsup_weight)

            u_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            loss_sum += float(u_loss.detach().item())
    finally:
        ema_teacher.offload_to_cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if was_training:
            student.train()
        else:
            student.eval()

    return {
        "accepted": accepted,
        "total": total,
        "accept_rate": round(accepted / max(1, total), 4),
        "unsup_loss": round(loss_sum / max(1, accepted), 6) if accepted > 0 else 0.0,
        "mean_conf": round(conf_sum / max(1, total), 4),
    }


def average_checkpoints(ckpt_paths: Sequence[Path]) -> Dict[str, torch.Tensor]:
    """Average floating-point state_dict tensors across checkpoints on CPU (SWAD / Model Soup)."""
    if not ckpt_paths:
        raise ValueError("average_checkpoints requires at least one checkpoint path")
    acc: Dict[str, torch.Tensor] = {}
    orig_dtypes: Dict[str, torch.dtype] = {}
    n = len(ckpt_paths)
    for idx, p in enumerate(ckpt_paths):
        ck = torch.load(str(p), map_location="cpu", weights_only=True)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        for k, v in sd.items():
            if idx == 0:
                orig_dtypes[k] = v.dtype
                if v.is_floating_point():
                    acc[k] = v.detach().to(dtype=torch.float32).clone()
                else:
                    acc[k] = v.detach().clone()
            else:
                if v.is_floating_point():
                    acc[k].add_(v.detach().to(dtype=torch.float32))
                else:
                    acc[k] = v.detach().clone()
    out: Dict[str, torch.Tensor] = {}
    for k, v in acc.items():
        if v.is_floating_point():
            out[k] = (v / float(n)).to(dtype=orig_dtypes[k])
        else:
            out[k] = v
    return out


def load_uni_weights_compat(
    model: nn.Module,
    sd: Dict[str, torch.Tensor],
) -> Any:
    """Load checkpoint weights with lossless tail-relocating vocab expansion.

    When ``tok_emb.weight`` or ``lm_head.weight`` has ``old_V < new_V``:
      - Copies base music token rows ``0 .. old_V - 5`` verbatim.
      - Relocates the 4 tail specials (``PAD, BOS, EOS, <sys>``) from
        ``old_V - 4 .. old_V - 1`` to ``new_V - 4 .. new_V - 1``.
      - Initializes newly appended middle rows ``old_V - 4 .. new_V - 5``
        with small normal init (``std=0.02``).
    """
    model_sd = model.state_dict()
    adapted_sd: Dict[str, torch.Tensor] = dict(sd)
    logged_expand: Optional[Tuple[int, int]] = None

    for key in ("tok_emb.weight", "lm_head.weight"):
        if key not in adapted_sd or key not in model_sd:
            continue
        old_w = adapted_sd[key]
        target_w = model_sd[key]
        if old_w.ndim == 2 and target_w.ndim == 2 and old_w.shape[1] == target_w.shape[1]:
            old_v, new_v = int(old_w.shape[0]), int(target_w.shape[0])
            if 4 <= old_v < new_v:
                expanded = torch.empty_like(target_w, device=old_w.device, dtype=old_w.dtype)
                nn.init.normal_(expanded, mean=0.0, std=0.02)
                n_base = old_v - 4
                if n_base > 0:
                    expanded[:n_base].copy_(old_w[:n_base])
                expanded[new_v - 4 :].copy_(old_w[old_v - 4 :])
                adapted_sd[key] = expanded
                logged_expand = (old_v, new_v)

    if logged_expand is not None:
        print(
            f"[UniOMR init-from] expanded vocab rows {logged_expand[0]} -> {logged_expand[1]} "
            f"(tail 4 specials relocated)",
            flush=True,
        )
    return model.load_state_dict(adapted_sd, strict=False)


def run_train(args: argparse.Namespace, cfg: UniOMRConfig) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = UniOMRNet(cfg).to(device)
    if getattr(args, "init_from", None):
        ck = torch.load(str(args.init_from), map_location="cpu", weights_only=True)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        inc = load_uni_weights_compat(model, sd)
        print(
            f"[UniOMR init-from] loaded {args.init_from} "
            f"(missing={len(inc.missing_keys)}, unexpected={len(inc.unexpected_keys)})",
            flush=True,
        )
        if getattr(args, "reset_router", False):
            model.router.reset_saliency()
            print(
                "[UniOMR reset-router] re-initialized router.saliency_net and reset ink_prior_scale=1.0",
                flush=True,
            )
    if getattr(args, "channels_last", False):
        model.backbone.to(memory_format=torch.channels_last)
    model.train()
    if device.type == "cuda":
        try:
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, fused=True)
        except (RuntimeError, TypeError):
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    steps = min(args.steps, 2) if args.dry_run else args.steps
    grad_accum = max(1, int(args.grad_accum))
    strip_batch_size_arg = getattr(args, "strip_batch_size", None)
    strip_batch_size = (
        max(1, int(strip_batch_size_arg))
        if strip_batch_size_arg is not None
        else max(1, int(args.batch_size))
    )
    strip_grad_accum = max(1, (max(1, int(args.batch_size)) * grad_accum) // strip_batch_size)
    log_every = max(1, int(getattr(args, "log_every", 10)))
    gpu_ema_decay = float(max(0.0, min(0.99999, getattr(args, "gpu_ema_decay", 0.0))))
    use_bf16 = bool(args.bf16 and (device.type == "cuda" or hasattr(torch, "bfloat16")))
    prefetch_threads_arg = getattr(args, "prefetch_threads", None)
    if prefetch_threads_arg is None:
        prefetch_threads = 6 if (device.type == "cuda" and not args.dry_run) else 0
    else:
        prefetch_threads = max(0, int(prefetch_threads_arg))
    eval_batch_size_arg = getattr(args, "eval_batch_size", None)
    if eval_batch_size_arg is None:
        eval_batch_size = 8 if (device.type == "cuda" and not args.dry_run) else max(1, int(args.batch_size))
    else:
        eval_batch_size = max(1, int(eval_batch_size_arg))
    aug_prob = float(getattr(args, "aug_prob", 0.75))
    hw_frac = float(getattr(args, "hw_frac", 0.30))
    fullpage_frac = float(max(0.0, min(1.0, getattr(args, "fullpage_frac", 0.65))))
    saliency_loss_weight = float(getattr(args, "saliency_loss_weight", 0.10))
    hw_strip_dirs = list(getattr(args, "hw_strip_dirs", []) or [])
    unlabeled_dirs = list(getattr(args, "unlabeled_dirs", []) or [])
    unsup_every = int(getattr(args, "unsup_every", 250))
    unsup_steps = int(getattr(args, "unsup_steps", 25))
    unsup_conf_thr = float(getattr(args, "unsup_conf_thr", 0.85))
    unsup_weight = float(getattr(args, "unsup_weight", 0.5))
    ema_decay = float(getattr(args, "ema_decay", 0.995))
    cass_prob = float(max(0.0, min(1.0, getattr(args, "cass_prob", 0.0))))
    soup_last_k = max(0, int(getattr(args, "soup_last_k", 0)))
    qat_int4 = bool(getattr(args, "qat_int4", False))
    qat_int4_last_frac = float(max(0.0, min(1.0, getattr(args, "qat_int4_last_frac", 0.15))))
    qat_int4_block_size = max(1, int(getattr(args, "qat_int4_block_size", 32)))
    qat_ctrl: Optional[QATInt4Controller] = (
        QATInt4Controller(model, block_size=qat_int4_block_size, enabled=False)
        if qat_int4
        else None
    )
    qat_start_step = (
        steps + 1
        if qat_int4_last_frac <= 0.0
        else max(1, int(steps * (1.0 - qat_int4_last_frac)))
    )

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None and not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Build real datasets when --strip-dirs / --hw-strip-dirs / --fullpage-dirs are supplied and not --dry-run.
    strip_train_iter = None
    hw_strip_train_iter = None
    fullpage_train_iter = None
    strip_val_ds: Optional[UniStripDataset] = None
    hw_strip_val_ds: Optional[UniStripDataset] = None
    fullpage_val_ds: Optional[UniFullPageDataset] = None
    prefetchers: List[AsyncThreadPrefetcher] = []

    use_real_data = (not args.dry_run) and (
        bool(args.strip_dirs) or bool(hw_strip_dirs) or bool(args.fullpage_dirs) or bool(args.fullpage_manifest)
    )
    if use_real_data:
        st_tr: List[Tuple[Path, Path]] = []
        st_va: List[Tuple[Path, Path]] = []
        hw_tr: List[Tuple[Path, Path]] = []
        hw_va: List[Tuple[Path, Path]] = []
        fp_tr: List[Tuple[Path, List[Path]]] = []
        fp_va: List[Tuple[Path, List[Path]]] = []

        if args.strip_dirs:
            strip_pairs = discover_strip_pairs(args.strip_dirs)
            if strip_pairs:
                st_tr, st_va = split_train_val(
                    strip_pairs,
                    key_fn=lambda p: _strip_group_key(p[0].stem),
                    val_count=args.val_count,
                    seed=args.seed,
                )

        if hw_strip_dirs:
            hw_pairs = discover_strip_pairs(hw_strip_dirs)
            if hw_pairs:
                hw_tr, hw_va = split_train_val(
                    hw_pairs,
                    key_fn=lambda p: _strip_group_key(p[0].stem),
                    val_count=args.val_count,
                    seed=args.seed,
                )

        if args.fullpage_dirs or args.fullpage_manifest:
            fp_pairs: List[Tuple[Any, ...]] = []
            if args.fullpage_dirs:
                fp_pairs.extend(discover_fullpage_pairs(args.fullpage_dirs))
            if args.fullpage_manifest:
                mp = Path(args.fullpage_manifest)
                if mp.is_file():
                    for raw_line in mp.read_text(encoding="utf-8").splitlines():
                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue
                        rec = json.loads(raw_line)
                        img_p = Path(rec["image"])
                        beks = [Path(b) for b in rec.get("bekerns", [])]
                        domain = str(rec.get("domain", "")).strip().lower()
                        if img_p.is_file() and beks:
                            fp_pairs.append((img_p, beks, domain))
            if fp_pairs:
                fp_tr, fp_va = split_train_val(
                    fp_pairs,
                    key_fn=lambda p: data.piece_key(p[0].stem),
                    val_count=args.val_count,
                    seed=args.seed,
                )

        # Shared piece-level val split: union of held-out val_keys across all modalities
        # so NO piece held out in any val set can leak into any other modality's train set.
        shared_val_keys: set[str] = set()
        for p in st_va:
            shared_val_keys.add(_strip_group_key(p[0].stem))
        for p in hw_va:
            shared_val_keys.add(_strip_group_key(p[0].stem))
        for p in fp_va:
            shared_val_keys.add(data.piece_key(p[0].stem))

        if shared_val_keys:
            leaked_st = leaked_hw = leaked_fp = 0
            if st_tr:
                filtered_st = [p for p in st_tr if _strip_group_key(p[0].stem) not in shared_val_keys]
                if filtered_st:
                    leaked_st = len(st_tr) - len(filtered_st)
                    st_tr = filtered_st
            if hw_tr:
                filtered_hw = [p for p in hw_tr if _strip_group_key(p[0].stem) not in shared_val_keys]
                if filtered_hw:
                    leaked_hw = len(hw_tr) - len(filtered_hw)
                    hw_tr = filtered_hw
            if fp_tr:
                filtered_fp = [p for p in fp_tr if data.piece_key(p[0].stem) not in shared_val_keys]
                if filtered_fp:
                    leaked_fp = len(fp_tr) - len(filtered_fp)
                    fp_tr = filtered_fp
            if leaked_st or leaked_hw or leaked_fp:
                print(
                    f"[UniOMR data] shared val_keys={len(shared_val_keys)} | "
                    f"removed cross-modality leaks: printed_strips={leaked_st}, hw_strips={leaked_hw}, fullpage={leaked_fp}"
                )

        if st_tr or st_va:
            strip_train_ds = UniStripDataset(
                st_tr,
                strip_height=args.strip_height,
                max_strip_width=args.max_strip_width,
                max_seq_len=args.max_seq_len,
                augment=args.augment,
                aug_prob=aug_prob,
                kind="strip",
                return_uint8=True,
            )
            strip_val_ds = UniStripDataset(
                st_va,
                strip_height=args.strip_height,
                max_strip_width=args.max_strip_width,
                max_seq_len=args.max_seq_len,
                augment=False,
                kind="strip",
                return_uint8=True,
            ) if st_va else None
            if prefetch_threads > 0 and args.num_workers == 0 and len(st_tr) > 0:
                pf_st = AsyncThreadPrefetcher(
                    strip_train_ds,
                    batch_size=strip_batch_size,
                    shuffle=True,
                    pin_memory=(device.type == "cuda"),
                    max_workers=prefetch_threads,
                    prefetch_batches=3,
                    collate_fn=collate_uni_batch,
                    seed=args.seed + 101,
                )
                prefetchers.append(pf_st)
                strip_train_iter = pf_st
            else:
                strip_loader = DataLoader(
                    strip_train_ds,
                    batch_size=strip_batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    collate_fn=collate_uni_batch,
                    pin_memory=(device.type == "cuda"),
                    drop_last=False,
                )
                strip_train_iter = _infinite_loader(strip_loader)
            print(
                f"[UniOMR data] printed_strips: {len(st_tr)} train / {len(st_va)} val "
                f"(H={args.strip_height}, maxW={args.max_strip_width}, bs={strip_batch_size}, aug={args.augment}, aug_prob={aug_prob:.2f})"
            )

        if hw_tr or hw_va:
            hw_train_ds = UniStripDataset(
                hw_tr,
                strip_height=args.strip_height,
                max_strip_width=args.max_strip_width,
                max_seq_len=args.max_seq_len,
                augment=args.augment,
                aug_prob=aug_prob,
                kind="hw_strip",
                return_uint8=True,
            )
            hw_strip_val_ds = UniStripDataset(
                hw_va,
                strip_height=args.strip_height,
                max_strip_width=args.max_strip_width,
                max_seq_len=args.max_seq_len,
                augment=False,
                kind="hw_strip",
                return_uint8=True,
            ) if hw_va else None
            if prefetch_threads > 0 and args.num_workers == 0 and len(hw_tr) > 0:
                pf_hw = AsyncThreadPrefetcher(
                    hw_train_ds,
                    batch_size=strip_batch_size,
                    shuffle=True,
                    pin_memory=(device.type == "cuda"),
                    max_workers=prefetch_threads,
                    prefetch_batches=3,
                    collate_fn=collate_uni_batch,
                    seed=args.seed + 202,
                )
                prefetchers.append(pf_hw)
                hw_strip_train_iter = pf_hw
            else:
                hw_loader = DataLoader(
                    hw_train_ds,
                    batch_size=strip_batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    collate_fn=collate_uni_batch,
                    pin_memory=(device.type == "cuda"),
                    drop_last=False,
                )
                hw_strip_train_iter = _infinite_loader(hw_loader)
            print(
                f"[UniOMR data] hw_strips: {len(hw_tr)} train / {len(hw_va)} val "
                f"(hw_frac={hw_frac:.2f}, H={args.strip_height}, maxW={args.max_strip_width}, "
                f"aug={args.augment}, aug_prob={aug_prob:.2f})"
            )

        if fp_tr or fp_va:
            fp_multi_scale = bool(getattr(args, "multi_scale", False))
            fp_train_ds = UniFullPageDataset(
                fp_tr,
                fullpage_max_dim=args.fullpage_max_dim,
                max_seq_len=args.max_seq_len,
                augment=args.augment,
                aug_prob=aug_prob,
                multi_scale=fp_multi_scale,
                return_uint8=True,
            )
            fullpage_val_ds = UniFullPageDataset(
                fp_va,
                fullpage_max_dim=args.fullpage_max_dim,
                max_seq_len=args.max_seq_len,
                augment=False,
                return_uint8=True,
            ) if fp_va else None
            if prefetch_threads > 0 and args.num_workers == 0 and len(fp_tr) > 0:
                pf_fp = AsyncThreadPrefetcher(
                    fp_train_ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    pin_memory=(device.type == "cuda"),
                    max_workers=prefetch_threads,
                    prefetch_batches=3,
                    collate_fn=collate_uni_batch,
                    seed=args.seed + 303,
                )
                prefetchers.append(pf_fp)
                fullpage_train_iter = pf_fp
            elif fp_multi_scale and len(fp_tr) > 0:
                fp_batch_sampler = MultiScaleBatchSampler(
                    num_samples=len(fp_train_ds),
                    batch_size=args.batch_size,
                    base_dim=args.fullpage_max_dim,
                    shuffle=True,
                    seed=args.seed + 303,
                )
                fp_loader = DataLoader(
                    fp_train_ds,
                    batch_sampler=fp_batch_sampler,
                    num_workers=args.num_workers,
                    collate_fn=collate_uni_batch,
                    pin_memory=(device.type == "cuda"),
                )
                fullpage_train_iter = _infinite_loader(fp_loader)
            else:
                fp_loader = DataLoader(
                    fp_train_ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    collate_fn=collate_uni_batch,
                    pin_memory=(device.type == "cuda"),
                    drop_last=False,
                )
                fullpage_train_iter = _infinite_loader(fp_loader)
            print(
                f"[UniOMR data] fullpage: {len(fp_tr)} train / {len(fp_va)} val "
                f"(maxDim={args.fullpage_max_dim}, maxSeq={args.max_seq_len}, "
                f"aug={args.augment}, aug_prob={aug_prob:.2f})"
            )

        if strip_train_iter is None and hw_strip_train_iter is None and fullpage_train_iter is None:
            raise FileNotFoundError(
                f"No valid training pairs found in strip_dirs={args.strip_dirs}, "
                f"hw_strip_dirs={hw_strip_dirs}, or fullpage_dirs={args.fullpage_dirs}"
            )

    # Unsupervised EMA self-training setup (CPU-offloaded teacher between unsup rounds)
    unlabeled_paths: List[Path] = []
    ema_teacher: Optional[CPUEmaTeacher] = None
    unsup_rng = random.Random(args.seed + 7919)
    if (not args.dry_run) and unlabeled_dirs and unsup_every > 0 and unsup_steps > 0:
        unlabeled_paths = discover_unlabeled_images(unlabeled_dirs)
        if unlabeled_paths:
            ema_teacher = CPUEmaTeacher(model, decay=ema_decay)
            print(
                f"[UniOMR unsup] unlabeled={len(unlabeled_paths)} images | "
                f"every={unsup_every} steps | unsup_steps={unsup_steps} | "
                f"conf_thr={unsup_conf_thr:.2f} | weight={unsup_weight:.2f} | ema_decay={ema_decay}"
            )

    strip_mixer = StripDomainMixer(hw_frac=hw_frac)
    has_any_strip = (strip_train_iter is not None) or (hw_strip_train_iter is not None)
    mix_strip_domains = (
        strip_train_iter is not None
        and hw_strip_train_iter is not None
        and strip_batch_size >= 2
        and cfg.mixstyle_prob > 0.0
    )
    pr_strip_pool: deque[Dict[str, Any]] = deque()
    hw_strip_pool: deque[Dict[str, Any]] = deque()

    def _pop_strip_sample(pool: deque[Dict[str, Any]], it: Any, fallback_kind: str) -> Dict[str, Any]:
        if not pool:
            b = next(it)
            B_loc = int(b["images"].shape[0])
            for i in range(B_loc):
                pool.append({
                    "image": b["images"][i],
                    "seq": b["seq"][i],
                    "sample_id": b["sample_ids"][i],
                    "kind": fallback_kind,
                })
        return pool.popleft()

    def _step_is_fullpage(step_idx: int) -> bool:
        if not use_real_data:
            return args.curriculum == "fullpage" or (args.curriculum == "mixed" and (step_idx % 2 == 1))
        if args.curriculum == "fullpage":
            return fullpage_train_iter is not None
        if args.curriculum == "strip":
            return not has_any_strip
        if has_any_strip and fullpage_train_iter is not None:
            return (
                int(math.floor(step_idx * fullpage_frac + 1e-9))
                > int(math.floor((step_idx - 1) * fullpage_frac + 1e-9))
            )
        return fullpage_train_iter is not None

    def _next_batch(
        step_idx: int,
        micro_idx: int,
        want_fp: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], str]:
        is_fp = _step_is_fullpage(step_idx) if want_fp is None else bool(want_fp)
        if not use_real_data:
            bs_syn = args.batch_size if is_fp else strip_batch_size
            imgs, sq, uv = _make_synthetic_batch(
                curriculum=args.curriculum,
                batch_size=bs_syn,
                seq_len=args.seq_len,
                device=device,
                step=step_idx * grad_accum + micro_idx,
                with_uv=cfg.use_uv_aux,
            )
            return imgs, sq, uv, args.curriculum

        if is_fp and fullpage_train_iter is not None:
            b = next(fullpage_train_iter)
            return _images_to_device_float(b["images"], device, non_blocking=True), b["seq"].to(device, non_blocking=True), None, "fullpage"

        # Strip step: balance Printed anchor replay (>= 1 - hw_frac) vs Labeled HW (hw_frac).
        # When MixStyle2d is enabled with B >= 2 and both strip domains exist, mix printed + HW
        # inside the same mini-batch so MixStyle crosses printed <-> HW styles directly.
        if mix_strip_domains:
            picked: List[Dict[str, Any]] = []
            n_hw_in_batch = 0
            for _ in range(strip_batch_size):
                if strip_mixer.next_is_hw():
                    picked.append(_pop_strip_sample(hw_strip_pool, hw_strip_train_iter, "hw_strip"))
                    n_hw_in_batch += 1
                else:
                    picked.append(_pop_strip_sample(pr_strip_pool, strip_train_iter, "strip"))
            b = collate_uni_batch(picked)
            batch_kind = "mixed_strip" if (0 < n_hw_in_batch < strip_batch_size) else ("hw_strip" if n_hw_in_batch > 0 else "strip")
            return _images_to_device_float(b["images"], device, non_blocking=True), b["seq"].to(device, non_blocking=True), None, batch_kind

        if strip_train_iter is not None and hw_strip_train_iter is not None:
            if strip_mixer.next_is_hw():
                b = next(hw_strip_train_iter)
                return _images_to_device_float(b["images"], device, non_blocking=True), b["seq"].to(device, non_blocking=True), None, "hw_strip"
            b = next(strip_train_iter)
            return _images_to_device_float(b["images"], device, non_blocking=True), b["seq"].to(device, non_blocking=True), None, "strip"
        if hw_strip_train_iter is not None:
            b = next(hw_strip_train_iter)
            return _images_to_device_float(b["images"], device, non_blocking=True), b["seq"].to(device, non_blocking=True), None, "hw_strip"
        assert strip_train_iter is not None
        b = next(strip_train_iter)
        return _images_to_device_float(b["images"], device, non_blocking=True), b["seq"].to(device, non_blocking=True), None, "strip"

    history: List[Dict[str, Any]] = []
    eval_history: List[Dict[str, Any]] = []
    ema_eval_history: List[Dict[str, Any]] = []
    unsup_history: List[Dict[str, Any]] = []
    soup_ckpts: List[Path] = []
    soup_eval: Optional[Dict[str, Any]] = None
    best_val_ser: float = float("inf")
    best_ema_ser: float = float("inf")
    gpu_ema: Optional[GPUEmaShadow] = (
        GPUEmaShadow(model, decay=gpu_ema_decay, update_every=5) if gpu_ema_decay > 0.0 else None
    )

    def _run_eval_suite(step_label: Any) -> Dict[str, Any]:
        eval_rec: Dict[str, Any] = {"step": step_label}
        val_sers: List[float] = []
        if strip_val_ds is not None:
            st_metrics = eval_dataset(
                model,
                strip_val_ds,
                device=device,
                max_len=args.eval_max_len,
                batch_size=eval_batch_size,
                use_bf16=use_bf16,
            )
            eval_rec["printed_strip_val"] = st_metrics
            eval_rec["strip_val"] = st_metrics
            val_sers.append(float(st_metrics["ser"]))
            val_tag = "printed_strip_val" if hw_strip_val_ds is not None else "strip_val"
            print(
                f"  [eval {val_tag} @ {step_label}] SER={st_metrics['ser']:.4f} "
                f"F1={st_metrics['token_f1']:.4f} sys_exact={st_metrics['sys_count_exact']:.3f} "
                f"speed={st_metrics['tokens_per_sec']:.1f} tok/s "
                f"comp={st_metrics['mean_compression_x']:.2f}x "
                f"ink_rec={st_metrics['ink_recall']:.3f}"
            )
        if hw_strip_val_ds is not None:
            hw_metrics = eval_dataset(
                model,
                hw_strip_val_ds,
                device=device,
                max_len=args.eval_max_len,
                batch_size=eval_batch_size,
                use_bf16=use_bf16,
            )
            eval_rec["hw_strip_val"] = hw_metrics
            val_sers.append(float(hw_metrics["ser"]))
            print(
                f"  [eval hw_strip_val @ {step_label}] SER={hw_metrics['ser']:.4f} "
                f"F1={hw_metrics['token_f1']:.4f} sys_exact={hw_metrics['sys_count_exact']:.3f} "
                f"speed={hw_metrics['tokens_per_sec']:.1f} tok/s "
                f"comp={hw_metrics['mean_compression_x']:.2f}x "
                f"ink_rec={hw_metrics['ink_recall']:.3f}"
            )
        if fullpage_val_ds is not None:
            fp_metrics = eval_dataset(
                model,
                fullpage_val_ds,
                device=device,
                max_len=args.eval_max_len,
                batch_size=eval_batch_size,
                use_bf16=use_bf16,
            )
            eval_rec["fullpage_val"] = fp_metrics
            val_sers.append(float(fp_metrics["ser"]))
            print(
                f"  [eval fullpage_val @ {step_label}] SER={fp_metrics['ser']:.4f} "
                f"F1={fp_metrics['token_f1']:.4f} sys_exact={fp_metrics['sys_count_exact']:.3f} "
                f"speed={fp_metrics['tokens_per_sec']:.1f} tok/s "
                f"comp={fp_metrics['mean_compression_x']:.2f}x "
                f"ink_rec={fp_metrics['ink_recall']:.3f}"
            )
        mean_ser = float(np.mean(val_sers)) if val_sers else float("inf")
        eval_rec["mean_val_ser"] = round(mean_ser, 6)
        return eval_rec

    log_loss_t = torch.zeros((), device=device, dtype=torch.float32)
    log_ce_t = torch.zeros((), device=device, dtype=torch.float32)
    log_mtp_t = torch.zeros((), device=device, dtype=torch.float32)
    log_uv_t = torch.zeros((), device=device, dtype=torch.float32)
    log_sal_t = torch.zeros((), device=device, dtype=torch.float32)
    log_count = 0
    pending_rows: List[Dict[str, Any]] = []

    def _flush_log_window(print_step: bool = True) -> None:
        nonlocal log_count
        if log_count <= 0 or not pending_rows:
            return
        inv_cnt = 1.0 / float(log_count)
        avg_loss, avg_ce, avg_mtp, avg_uv, avg_sal = (
            (torch.stack([log_loss_t, log_ce_t, log_mtp_t, log_uv_t, log_sal_t]) * inv_cnt).cpu().tolist()
        )
        for prow in pending_rows:
            prow["loss"] = float(avg_loss)
            prow["ce_loss"] = float(avg_ce)
            prow["mtp_loss"] = float(avg_mtp)
            prow["uv_loss"] = float(avg_uv)
            prow["saliency_loss"] = float(avg_sal)
        last_row = pending_rows[-1]
        history.extend(pending_rows)
        pending_rows.clear()
        log_loss_t.zero_()
        log_ce_t.zero_()
        log_mtp_t.zero_()
        log_uv_t.zero_()
        log_sal_t.zero_()
        log_count = 0
        if print_step:
            print(
                f"[UniOMR {'dry-run' if args.dry_run else 'train'} | {last_row['kind']}] "
                f"step {last_row['step']}/{steps} | lr={last_row['lr']:.2e} | loss={last_row['loss']:.4f} "
                f"(ce={last_row['ce_loss']:.4f}, mtp={last_row['mtp_loss']:.4f}, uv={last_row['uv_loss']:.4f}, sal={last_row['saliency_loss']:.4f}) | "
                f"tokens {last_row['n_in']}->{last_row['n_out']}"
            )

    try:
        for step in range(1, steps + 1):
            if qat_ctrl is not None and (not qat_ctrl.enabled) and step >= qat_start_step:
                qat_ctrl.enable()
                print(
                    f"[UniOMR QAT-INT4] enabled on {qat_ctrl.num_layers} Linear layers "
                    f"(block_size={qat_ctrl.block_size}) at step {step}/{steps}",
                    flush=True,
                )

            cur_lr = _lr_for_step(step, steps, args.lr, args.warmup_steps)
            for pg in opt.param_groups:
                pg["lr"] = cur_lr

            step_is_fp = _step_is_fullpage(step)
            step_ga = grad_accum if step_is_fp else strip_grad_accum
            inv_ga = 1.0 / float(step_ga)

            opt.zero_grad(set_to_none=True)
            acc_loss_t = torch.zeros((), device=device, dtype=torch.float32)
            acc_ce_t = torch.zeros((), device=device, dtype=torch.float32)
            acc_mtp_t = torch.zeros((), device=device, dtype=torch.float32)
            acc_uv_t = torch.zeros((), device=device, dtype=torch.float32)
            acc_sal_t = torch.zeros((), device=device, dtype=torch.float32)
            last_n_in = 0
            last_n_out = 0
            last_kind = args.curriculum

            oom_hit = False
            for micro in range(step_ga):
                images, seq, uv_target, kind = _next_batch(step, micro, want_fp=step_is_fp)
                tgt_in = seq[:, :-1]
                tgt_out = seq[:, 1:]

                want_aux = bool(cfg.use_uv_aux and (uv_target is not None))
                amp_ctx = (
                    torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                    if use_bf16
                    else contextlib.nullcontext()
                )
                try:
                    with amp_ctx:
                        if cass_prob > 0.0 and kind != "strip" and tgt_in.shape[1] > 1:
                            # Two-pass parallel scheduled sampling (CASS): encode image ONCE,
                            # run a no-grad draft decode (without tok_dropout to avoid double-corruption),
                            # replace cass_prob of non-BOS/non-PAD tgt_in[:, 1:] with pred_ids[:, :-1],
                            # then run the real gradient-enabled decoder pass reusing mem.
                            mem, uv_pred, router_info = model.encode_image(images, return_aux=want_aux)
                            with torch.no_grad():
                                draft_logits, _ = model.decode_tokens(
                                    tgt_in, mem.detach(), return_mtp=False, apply_tok_dropout=False
                                )
                                pred_ids = draft_logits.argmax(dim=-1)
                                tail_in = tgt_in[:, 1:]
                                cass_mask = (
                                    (torch.rand_like(tail_in, dtype=torch.float32) < cass_prob)
                                    & (tail_in != PAD_ID)
                                    & (tail_in != BOS_ID)
                                )
                                dec_in = torch.cat(
                                    [tgt_in[:, :1], torch.where(cass_mask, pred_ids[:, :-1], tail_in)],
                                    dim=1,
                                )
                            logits, mtp_logits = model.decode_tokens(dec_in, mem, return_mtp=(cfg.mtp_k > 1))
                            out = {
                                "logits": logits,
                                "mtp_logits": mtp_logits,
                                "uv_pred": uv_pred,
                                "router_info": router_info,
                            }
                        else:
                            out = model(images, tgt_in, return_aux=want_aux, return_mtp=(cfg.mtp_k > 1))
                        losses = model.compute_loss(
                            out,
                            tgt_out,
                            uv_target=uv_target,
                            mtp_loss_weight=args.mtp_loss_weight,
                            uv_loss_weight=args.uv_loss_weight,
                            saliency_loss_weight=saliency_loss_weight,
                            struct_tok_weight=args.struct_tok_weight,
                        )

                    (losses["loss"] * inv_ga).backward()
                    acc_loss_t.add_(losses["loss"].detach().float(), alpha=inv_ga)
                    acc_ce_t.add_(losses["ce_loss"].detach().float(), alpha=inv_ga)
                    acc_mtp_t.add_(losses["mtp_loss"].detach().float(), alpha=inv_ga)
                    acc_uv_t.add_(losses["uv_loss"].detach().float(), alpha=inv_ga)
                    acc_sal_t.add_(losses["saliency_loss"].detach().float(), alpha=inv_ga)
                    last_n_in = int(out["router_info"]["n_in"])
                    last_n_out = int(out["router_info"]["n_out"])
                    last_kind = kind
                except Exception as e:
                    if device.type == "cuda" and "out of memory" in str(e).lower():
                        print(
                            f"  [WARN OOM skip] step {step} micro {micro} kind={kind} "
                            f"img={tuple(images.shape)} seq={tuple(seq.shape)} "
                            f"alloc={torch.cuda.memory_allocated(device)/1e9:.2f}GB "
                            f"resv={torch.cuda.memory_reserved(device)/1e9:.2f}GB"
                        )
                        oom_hit = True
                        mem = uv_pred = router_info = logits = mtp_logits = dec_in = draft_logits = pred_ids = None
                        del images, seq, uv_target, tgt_in, tgt_out
                        opt.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        break
                    raise
                del images, seq, uv_target, tgt_in, tgt_out, out, losses
                mem = uv_pred = router_info = logits = mtp_logits = dec_in = draft_logits = pred_ids = None

            if oom_hit:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if device.type == "cuda" and torch.cuda.memory_reserved(device) > 13 * (1024 ** 3):
                torch.cuda.empty_cache()

            if ema_teacher is not None:
                ema_teacher.update(model)
            if gpu_ema is not None:
                gpu_ema.update(model, step)

            # Accumulate GPU scalars over log_every steps; sync GPU->CPU only on log steps.
            log_loss_t.add_(acc_loss_t)
            log_ce_t.add_(acc_ce_t)
            log_mtp_t.add_(acc_mtp_t)
            log_uv_t.add_(acc_uv_t)
            log_sal_t.add_(acc_sal_t)
            log_count += 1
            pending_rows.append({
                "step": step,
                "kind": last_kind,
                "lr": cur_lr,
                "n_in": last_n_in,
                "n_out": last_n_out,
            })
            if step == 1 or (step % log_every == 0) or step == steps:
                _flush_log_window(print_step=True)

            # Periodic confidence-gated EMA unsupervised self-training round.
            if (
                ema_teacher is not None
                and unlabeled_paths
                and unsup_every > 0
                and (step % unsup_every == 0)
            ):
                u_stats = run_unsup_ema_round(
                    student=model,
                    ema_teacher=ema_teacher,
                    opt=opt,
                    unlabeled_paths=unlabeled_paths,
                    device=device,
                    curriculum=args.curriculum,
                    strip_height=args.strip_height,
                    max_strip_width=args.max_strip_width,
                    fullpage_max_dim=args.fullpage_max_dim,
                    unsup_steps=unsup_steps,
                    unsup_conf_thr=unsup_conf_thr,
                    unsup_weight=unsup_weight,
                    max_gen_len=args.eval_max_len,
                    mtp_loss_weight=args.mtp_loss_weight,
                    use_bf16=use_bf16,
                    rng=unsup_rng,
                )
                u_rec = {"step": step, **u_stats}
                unsup_history.append(u_rec)
                print(
                    f"  [unsup EMA @ {step}] accepted={u_stats['accepted']}/{u_stats['total']} "
                    f"({u_stats['accept_rate'] * 100.0:.1f}%) | mean_conf={u_stats['mean_conf']:.3f} | "
                    f"unsup_loss={u_stats['unsup_loss']:.4f}"
                )

            # Periodic validation on held-out printed strips, HW strips, and/or full pages.
            should_eval = (
                use_real_data
                and args.eval_every > 0
                and (step % args.eval_every == 0 or step == steps)
                and (strip_val_ds is not None or hw_strip_val_ds is not None or fullpage_val_ds is not None)
            )
            if should_eval:
                eval_rec = _run_eval_suite(step)
                mean_ser = float(eval_rec["mean_val_ser"])

                if gpu_ema is not None:
                    with gpu_ema.swap_into(model):
                        ema_eval_rec = _run_eval_suite(f"{step} EMA")
                    ema_mean_ser = float(ema_eval_rec["mean_val_ser"])
                    eval_rec["ema_eval"] = ema_eval_rec
                    ema_eval_history.append(ema_eval_rec)
                    if out_dir is not None and ema_mean_ser < best_ema_ser:
                        best_ema_ser = ema_mean_ser
                        torch.save(
                            {
                                "step": step,
                                "val_ser": best_ema_ser,
                                "ema_decay": gpu_ema_decay,
                                "config": asdict(cfg),
                                "vocab": list(data.VOCAB),
                                "model": {k: v.detach().cpu() for k, v in gpu_ema.shadow.items()},
                            },
                            out_dir / "uni_omr_ema_best.pt",
                        )

                eval_history.append(eval_rec)

                if out_dir is not None and mean_ser < best_val_ser:
                    best_val_ser = mean_ser
                    torch.save(
                        {
                            "step": step,
                            "val_ser": best_val_ser,
                            "config": asdict(cfg),
                            "vocab": list(data.VOCAB),
                            "model": model.state_dict(),
                        },
                        out_dir / "uni_omr_best.pt",
                    )
                if out_dir is not None and soup_last_k > 0:
                    step_ckpt = out_dir / f"uni_omr_step{step}.pt"
                    torch.save(
                        {
                            "step": step,
                            "val_ser": mean_ser,
                            "config": asdict(cfg),
                            "vocab": list(data.VOCAB),
                            "model": model.state_dict(),
                        },
                        step_ckpt,
                    )
                    soup_ckpts.append(step_ckpt)
                    while len(soup_ckpts) > soup_last_k:
                        old_ckpt = soup_ckpts.pop(0)
                        try:
                            if old_ckpt.is_file():
                                old_ckpt.unlink()
                        except OSError:
                            pass
                if out_dir is not None:
                    periodic_payload: Dict[str, Any] = {
                        "best_val_ser": best_val_ser,
                        "evals": eval_history,
                        "unsup": unsup_history,
                    }
                    if ema_eval_history:
                        periodic_payload["best_ema_ser"] = best_ema_ser
                    (out_dir / "eval_summary.json").write_text(
                        json.dumps(periodic_payload, indent=2),
                        encoding="utf-8",
                    )
    finally:
        _flush_log_window(print_step=False)
        for pf in prefetchers:
            pf.close()

    if out_dir is not None and not args.dry_run:
        ckpt_path = out_dir / "uni_omr_last.pt"
        torch.save(
            {
                "step": steps,
                "config": asdict(cfg),
                "vocab": list(data.VOCAB),
                "model": model.state_dict(),
            },
            ckpt_path,
        )
        (out_dir / "train_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if soup_last_k > 0 and soup_ckpts:
            soup_sd = average_checkpoints(soup_ckpts)
            model.load_state_dict(soup_sd, strict=True)
            if strip_val_ds is not None or hw_strip_val_ds is not None or fullpage_val_ds is not None:
                soup_eval = _run_eval_suite(f"soup(k={len(soup_ckpts)})")
                soup_eval["soup_ckpts"] = [p.name for p in soup_ckpts]
                print(
                    f"  [eval soup(k={len(soup_ckpts)})] mean_val_ser={soup_eval['mean_val_ser']:.4f} "
                    f"(from {[p.name for p in soup_ckpts]})"
                )
            torch.save(
                {
                    "step": steps,
                    "soup_ckpts": [p.name for p in soup_ckpts],
                    "val_ser": soup_eval["mean_val_ser"] if soup_eval is not None else None,
                    "config": asdict(cfg),
                    "vocab": list(data.VOCAB),
                    "model": soup_sd,
                },
                out_dir / "uni_omr_soup.pt",
            )

        if eval_history or unsup_history or soup_eval is not None:
            summary_payload: Dict[str, Any] = {
                "best_val_ser": best_val_ser if eval_history else None,
                "evals": eval_history,
                "unsup": unsup_history,
            }
            if ema_eval_history:
                summary_payload["best_ema_ser"] = best_ema_ser
            if soup_eval is not None:
                summary_payload["soup_eval"] = soup_eval
            (out_dir / "eval_summary.json").write_text(
                json.dumps(summary_payload, indent=2),
                encoding="utf-8",
            )

    if qat_ctrl is not None:
        qat_ctrl.detach()

    return {
        "steps": steps,
        "history": history,
        "eval_history": eval_history,
        "ema_eval_history": ema_eval_history,
        "unsup_history": unsup_history,
        "soup_eval": soup_eval,
        "best_val_ser": best_val_ser if eval_history else None,
        "best_ema_ser": best_ema_ser if ema_eval_history else None,
        "params": model.param_summary(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="UniOMR — Unified Full-Page OMR Experiment Harness")
    p.add_argument("--mode", choices=["train", "bench_browser"], default="bench_browser")
    p.add_argument("--curriculum", choices=["strip", "mixed", "fullpage"], default="mixed")
    p.add_argument("--strip-dirs", nargs="*", default=[], help="Directories of printed strip training pairs")
    p.add_argument("--hw-strip-dirs", nargs="*", default=[], help="Directories of labeled HW strip training pairs")
    p.add_argument("--hw-frac", type=float, default=0.30, help="Fraction of strip steps sampled from --hw-strip-dirs (default: 0.30)")
    p.add_argument("--fullpage-frac", type=float, default=0.65, help="Fraction of mixed-curriculum steps sampled from fullpage data when both strips and fullpage exist (default: 0.65)")
    p.add_argument("--fullpage-dirs", nargs="*", default=[], help="Directories of full-page training pairs")
    p.add_argument("--fullpage-manifest", type=str, default=None, help="Path to full-page manifest.jsonl")
    p.add_argument("--uv-mask-dir", type=str, default=None, help="Directory of photo-degrade UV flow masks")

    # Data preprocessing & augmentation
    p.add_argument("--strip-height", type=int, default=192, help="Target strip height (multiple of 16)")
    p.add_argument("--max-strip-width", type=int, default=1024, help="Max strip width cap (multiple of 16)")
    p.add_argument("--fullpage-max-dim", type=int, default=768, help="Max full-page dimension (multiple of 16)")
    p.add_argument("--multi-scale", action="store_true", help="Weighted random full-page resolutions (15%% 512-768, 60%% 896-1152, 25%% 1280-1536, capped by --fullpage-max-dim)")
    p.add_argument("--max-seq-len", type=int, default=1024, help="Max token sequence length")
    p.add_argument("--augment", action="store_true", help="Enable online phone-photo degradation on train")
    p.add_argument("--aug-prob", type=float, default=0.75, help="Probability of applying online augmentation when --augment is set (default: 0.75)")
    p.add_argument("--val-count", type=int, default=100, help="Held-out validation sample count per modality")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    p.add_argument("--prefetch-threads", type=int, default=None, help="Background threads for async batch prefetching (default: 6 on CUDA, 0 on CPU/dry-run)")

    # Confidence-gated unsupervised EMA self-training
    p.add_argument("--unlabeled-dirs", nargs="*", default=[], help="Directories of unlabeled images (recursive .png/.jpg/.jpeg) for EMA self-training")
    p.add_argument("--unsup-every", type=int, default=250, help="Run EMA unsupervised self-training every N steps (default: 250)")
    p.add_argument("--unsup-steps", type=int, default=25, help="Unlabeled samples evaluated per EMA unsup round (default: 25)")
    p.add_argument("--unsup-conf-thr", type=float, default=0.85, help="Minimum mean token probability to accept pseudo-label (default: 0.85)")
    p.add_argument("--unsup-weight", type=float, default=0.5, help="Loss weight for accepted unsupervised consistency samples (default: 0.5)")
    p.add_argument("--ema-decay", type=float, default=0.995, help="EMA teacher decay rate (default: 0.995)")

    # Architecture
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-enc-layers", type=int, default=4)
    p.add_argument("--n-dec-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--backbone-stride", type=str, choices=["8", "16", "4x8"], default="16", help="Conv backbone spatial stride ('4x8' for 2x vertical pitch resolution, '8' for 4x finer, '16' default)")
    p.add_argument("--mixstyle-prob", type=float, default=0.0, help="Early-layer MixStyle probability after stage1/stage2 during training (e.g. 0.30; 0 extra params, 0%% infer cost)")
    p.add_argument("--tok-dropout", type=float, default=0.0, help="Decoder input token dropout probability during training (e.g. 0.15 to prevent repetition loops)")
    p.add_argument("--cass-prob", type=float, default=0.0, help="Two-pass parallel scheduled sampling (CASS) replacement prob on tgt_in[:, 1:] during training (e.g. 0.25)")
    p.add_argument("--soup-last-k", type=int, default=0, help="SWAD / checkpoint souping: average last K periodic eval checkpoints into uni_omr_soup.pt (default: 0)")
    p.add_argument("--keep-ratio", type=float, default=0.40)
    p.add_argument("--dynamic-routing", action="store_true", help="Enable saliency-thresholded adaptive token routing [min_keep_ratio..max_keep_ratio]")
    p.add_argument("--min-keep-ratio", type=float, default=0.15)
    p.add_argument("--max-keep-ratio", type=float, default=0.75)
    p.add_argument("--saliency-thr", type=float, default=0.05)
    p.add_argument("--n-bg-summary", type=int, default=4)
    p.add_argument("--mtp-k", type=int, default=4)
    p.add_argument("--mtp-loss-weight", type=float, default=0.30)
    p.add_argument("--uv-loss-weight", type=float, default=0.20)
    p.add_argument("--saliency-loss-weight", type=float, default=0.10, help="BCE auxiliary supervision weight on InkTokenRouter saliency_logits (default: 0.10)")
    p.add_argument("--struct-tok-weight", type=float, default=1.0, help="Main CE loss multiplier (>1.0) on header/structural tokens: clef-*, key*, time-*, <sys>, grace, bar* (default: 1.0)")
    p.add_argument("--no-uv-aux", action="store_true", help="Disable auxiliary UV-flow head")

    # Runtime / training
    p.add_argument("--bf16", action="store_true", help="Enable bfloat16 autocast")
    p.add_argument("--qat-int4", action="store_true", help="Enable symmetric group-wise INT4 QAT on Transformer Linear layers during the final fraction of training steps")
    p.add_argument("--qat-int4-last-frac", type=float, default=0.15, help="Final fraction of training steps to enable INT4 QAT when --qat-int4 is set (default: 0.15)")
    p.add_argument("--qat-int4-block-size", type=int, default=32, help="Group block size for symmetric INT4 QAT weight quantization (default: 32, matching MatMulNBits)")
    p.add_argument("--channels-last", action="store_true", help="Convert DynamicConvBackbone to channels_last (NHWC) memory format for faster Conv2d")
    p.add_argument("--grad-checkpoint", action="store_true", help="Enable selective gradient checkpointing on large encoder/decoder transformer tensors")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--strip-batch-size", type=int, default=None, help="Optional separate batch size for strip steps (default: --batch-size); auto-scales strip grad-accum to preserve effective batch size")
    p.add_argument("--eval-batch-size", type=int, default=None, help="Batch size for eval_dataset (default: 8 on CUDA, falls back to --batch-size on CPU/dry-run)")
    p.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation micro-steps")
    p.add_argument("--log-every", type=int, default=10, help="Sync GPU scalars and print training log every N steps (default: 10; always logs step 1 and final step)")
    p.add_argument("--gpu-ema-decay", type=float, default=0.0, help="Always-on device-resident EMA shadow decay updated every 5 steps (default: 0.0 = off; e.g. 0.997); saves uni_omr_ema_best.pt")
    p.add_argument("--warmup-steps", type=int, default=100, help="Linear LR warmup steps")
    p.add_argument("--eval-every", type=int, default=250, help="Evaluate on held-out val every N steps")
    p.add_argument("--eval-max-len", type=int, default=256, help="Max autoregressive decode tokens during eval")
    p.add_argument("--seq-len", type=int, default=32, help="Synthetic sequence length for --dry-run")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-from", type=str, default=None, help="Warm-start model weights from checkpoint")
    p.add_argument("--reset-router", action="store_true", help="When used with --init-from, re-initialize router.saliency_net and reset ink_prior_scale=1.0 after loading checkpoint")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda (default: cpu)")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--dry-run", action="store_true", help="Run 2 synthetic steps and exit")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = UniOMRConfig(
        d_model=args.d_model,
        n_enc_layers=args.n_enc_layers,
        n_dec_layers=args.n_dec_layers,
        n_heads=args.n_heads,
        keep_ratio=args.keep_ratio,
        dynamic_routing=bool(args.dynamic_routing),
        min_keep_ratio=args.min_keep_ratio,
        max_keep_ratio=args.max_keep_ratio,
        saliency_thr=args.saliency_thr,
        n_bg_summary=args.n_bg_summary,
        use_uv_aux=not args.no_uv_aux,
        backbone_stride=args.backbone_stride,
        mixstyle_prob=args.mixstyle_prob,
        tok_dropout=args.tok_dropout,
        mtp_k=args.mtp_k,
        grad_checkpoint=bool(args.grad_checkpoint),
    )

    if args.mode == "bench_browser":
        run_bench_browser(cfg, device_str=args.device)
        return 0

    run_train(args, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
