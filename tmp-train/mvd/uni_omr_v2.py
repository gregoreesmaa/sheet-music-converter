"""UniOMR v2 — Bleeding-Edge Full-Page & Handwritten OMR Architecture.

Combines Frontiers 2, 3, and 5 into a single WebGPU-ready (< 55 MB Hybrid
FP16/INT4-QAT) encoder-decoder architecture:

  1. ``AnisotropicMusicBackbone``:
     Lightweight depthwise-separable inverted-bottleneck blocks with parallel
     anisotropic strips (``1x9`` horizontal staff/beam + ``9x1`` vertical
     chord/stem + ``3x3`` local glyph) + GroupNorm/SiLU, plus a lossless
     ``P2 (/4)`` PixelUnshuffle lateral FPN skip fused into stages 3 and 4
     so tiny accidentals, augmentation dots, and ledger lines are never erased.
     (~5-6M params vs 15.4M in v1 ResNet).

  2. ``HybridAxialMergerEncoder`` (DeepSeek-OCR / Qwen3-VL inspired):
     - 2 layers of Row-then-Column Axial Attention (``O(N*(H'+W'))``) on the
       high-res ``/8`` grid with aspect-normalized 1D RoPE + QK-Norm.
     - 2x2 MLP Patch Merger (``/8 -> /16``) + ``InkTokenRouterV2`` (reusing
       v1's 2-patch eroded ``paper_gate``) capping dense memory tokens at
       ``k <= 3,072`` (25-50x fewer global attention FLOPs).
     - 4 layers of Global Interleaved ``[y, x]`` Aspect-Normalized 2D-RoPE
       Attention + QK-Norm (``(y, x)`` normalized to ``[0, 1]`` so 1280px and
       1536px share identical rotary phases).
     - Qwen3-VL DeepStack multi-level visual memory export: returns
       ``mem_levels = (mem_post_merger, mem_global_mid, mem_final)`` sharing
       the exact same routed spatial indices ``K <= 3,072``.

  3. Training-Only Zero-Inference-Cost Layout Aux Heads:
     - ``SysYHead``: predicts normalized ``y_center`` on ``<sys>`` decoder
       hidden states (eliminates ``<sys>`` vertical pointer skips).
     - ``SysCountHead``: predicts ``N_sys`` from pooled encoder memory
       (eliminates early ``<eos>``).
     - ``StaffSegHead``: predicts staff-line / ink band mask from high-res
       ``P2/P3`` features.
     - All 3 heads are bypassed at inference and stripped cleanly via
       ``strip_aux_heads()`` for ONNX/WebGPU export (0% browser cost).

  4. GQA Decoder + QK-Norm + DeepStack + MTP-4 + Hybrid FP16/INT4-QAT (``UniOMRv2Net``):
     - Grouped-Query Attention (``8 Q-heads / 2 KV-heads``) with ``RMSNorm(head_dim)``
       QK-Norm in both decoder self-attention and cross-attention (4x smaller
       WebGPU KV-cache RAM & bandwidth).
     - DeepStack early-layer visual fusion (layer 0 fuses ``mem_post_merger``,
       layer 1 fuses ``mem_global_mid``, layers >= 2 use ``mem_final`` directly)
       with 0% extra KV cache and < 0.5M params.
     - Preserves ``MTP-4`` (``mtp_k=4``) and reuses ``QATInt4Controller`` /
       ``fake_quant_int4_ste`` from ``uni_omr.py``.
     - Presets: ``UniOMRv2Config.small_45m()`` (~43M, <25 MB INT4) and
       ``UniOMRv2Config.edge85m()`` (d_model=512, ~81M, ~50 MB Hybrid
       FP16/INT4-QAT, <55 MB including group-32 FP16 scales).
"""

from __future__ import annotations

import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_ckpt

# Ensure sibling imports resolve regardless of CWD.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import data  # noqa: E402
from uni_omr import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SYS_ID,
    SYS_TOKEN,
    UNI_ID_TO_TOKEN,
    UNI_TOKEN_TO_ID,
    VOCAB_SIZE,
    InkTokenRouter,
    MixStyle2d,
    MTPHead as _BaseMTPHead,
    QATInt4Controller,
    RMSNorm as _BaseRMSNorm,
    RoPE1D,
    SwiGLU,
    _CUDA_SDPA_BACKENDS,
    _SDPA_HAS_SET_PRIORITY,
    _sdpa_kernel,
    apply_rope,
    decode_uni,
    encode_uni,
    fake_quant_int4_ste,
)

__all__ = [
    "PAD_ID",
    "BOS_ID",
    "EOS_ID",
    "SYS_ID",
    "SYS_TOKEN",
    "VOCAB_SIZE",
    "UNI_TOKEN_TO_ID",
    "UNI_ID_TO_TOKEN",
    "encode_uni",
    "decode_uni",
    "UniOMRv2Config",
    "AnisotropicMusicBlock",
    "AnisotropicMusicBackbone",
    "AspectNormalizedRoPE1D",
    "AspectNormalizedRoPE2D",
    "AxialEncoderLayer",
    "PatchMerger2x2",
    "InkTokenRouterV2",
    "GlobalEncoderLayer",
    "HybridAxialMergerEncoder",
    "SysYHead",
    "SysCountHead",
    "StaffSegHead",
    "StrokeHogHead",
    "GroupedQueryAttention",
    "DeepStackFusion",
    "GQADecoderLayer",
    "QATInt4ControllerV2",
    "UniOMRv2Net",
    "compute_sys_y_loss",
    "compute_sys_count_loss",
    "compute_staff_seg_loss",
    "compute_stroke_hog_target",
    "compute_stroke_hog_loss",
    "fake_quant_int4_ste",
    "QATInt4Controller",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HAS_F_RMS_NORM: bool = hasattr(F, "rms_norm")


def _active_compute_dtype(x: torch.Tensor) -> torch.dtype:
    """Return active autocast dtype for ``x.device.type`` when enabled, else ``x.dtype``."""
    if torch.is_autocast_enabled(x.device.type):
        return torch.get_autocast_dtype(x.device.type)
    return x.dtype


def _is_qat_linear_active(mod: nn.Linear) -> bool:
    """Return True if ``mod`` has an active QAT-INT4 monkey-patch (zero state_dict change)."""
    return bool(getattr(mod, "_qat_enabled", False))


class RMSNorm(_BaseRMSNorm):
    """ONNX-safe fused RMSNorm using native PyTorch ``F.rms_norm`` (zero state_dict change)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__(dim, eps=eps)
        self._normalized_shape: Tuple[int, ...] = (int(dim),)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_F_RMS_NORM and not torch.onnx.is_in_onnx_export():
            if x.dtype in (torch.bfloat16, torch.float16):
                with torch.amp.autocast(device_type=x.device.type, enabled=False):
                    return F.rms_norm(x, self._normalized_shape, self.weight.to(x.dtype), self.eps)
            return F.rms_norm(x, self._normalized_shape, self.weight, self.eps)
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class MTPHead(_BaseMTPHead):
    """ONNX-safe MTP projection head using v2 fused ``RMSNorm`` (zero state_dict change)."""

    def __init__(self, d_model: int) -> None:
        nn.Module.__init__(self)
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, d_model, bias=False)


def _group_count(channels: int, max_groups: int = 16) -> int:
    g = min(max_groups, channels)
    while g > 1 and (channels % g != 0):
        g -= 1
    return g


def _lru_get(cache: "OrderedDict[Any, Any]", key: Any) -> Any:
    val = cache.get(key)
    if val is not None:
        cache.move_to_end(key)
    return val


def _lru_put(cache: "OrderedDict[Any, Any]", key: Any, value: Any, max_size: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)


_SOBEL_KXY_CACHE: Dict[torch.device, torch.Tensor] = {}
_CAUSAL_MASK_CACHE: "OrderedDict[Tuple[int, int, torch.device], torch.Tensor]" = OrderedDict()
_CAUSAL_MASK_CACHE_MAX: int = 32


def _get_sobel_kxy(device: torch.device) -> torch.Tensor:
    kxy = _SOBEL_KXY_CACHE.get(device)
    if kxy is None:
        kxy = (
            torch.tensor(
                [
                    [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
                    [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
                ],
                device=device,
                dtype=torch.float32,
            )
            * 0.25
        )
        _SOBEL_KXY_CACHE[device] = kxy
    return kxy


def _get_causal_offset_mask(lq: int, past_len: int, device: torch.device) -> torch.Tensor:
    key = (lq, past_len, device)
    mask = _lru_get(_CAUSAL_MASK_CACHE, key)
    if mask is None:
        total_k = past_len + lq
        row_idx = torch.arange(lq, device=device).unsqueeze(1) + past_len
        col_idx = torch.arange(total_k, device=device).unsqueeze(0)
        mask = col_idx <= row_idx
        _lru_put(_CAUSAL_MASK_CACHE, key, mask, _CAUSAL_MASK_CACHE_MAX)
    return mask


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """SDPA wrapper enforcing CUTLASS EFFICIENT_ATTENTION on CUDA (avoids cuDNN 9.10 stalls)."""
    sdpa_kwargs: Dict[str, Any] = dict(attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal)
    if enable_gqa:
        sdpa_kwargs["enable_gqa"] = True
    if q.device.type == "cuda" and _sdpa_kernel is not None:
        if _SDPA_HAS_SET_PRIORITY:
            with _sdpa_kernel(_CUDA_SDPA_BACKENDS, set_priority=True):
                return F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
        if attn_mask is None:
            with _sdpa_kernel(_CUDA_SDPA_BACKENDS):
                return F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
    return F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class UniOMRv2Config:
    d_model: int = 384
    n_heads: int = 8
    n_kv_heads: int = 2
    n_axial_layers: int = 2
    n_global_layers: int = 4
    n_dec_layers: int = 8
    ffn_mult: float = 4.0
    axial_ffn_mult: float = 2.0
    dropout: float = 0.1
    vocab_size: int = VOCAB_SIZE

    # Backbone channels & blocks (stages 1..4: /2, /4, /8, /8)
    backbone_channels: Tuple[int, int, int, int] = (64, 128, 240, 448)
    backbone_blocks: Tuple[int, int, int, int] = (1, 2, 3, 3)
    backbone_expand: float = 2.0
    mixstyle_prob: float = 0.0

    # Router & token budget
    keep_ratio: float = 0.40
    dynamic_routing: bool = True
    min_keep_ratio: float = 0.15
    max_keep_ratio: float = 0.75
    saliency_thr: float = 0.05
    max_mem_tokens: int = 10240
    n_bg_summary: int = 4
    use_ink_prior: bool = True
    adaptive_ink_target: bool = True
    router_hidden: int = 64
    vrfm_band_alpha: float = 0.30

    # Training-only layout & stroke-gradient auxiliary heads (0% inference cost)
    use_sys_y_aux: bool = True
    use_sys_count_aux: bool = True
    use_staff_seg_aux: bool = True
    use_stroke_hog_head: bool = True

    # Frontier 2025-2026 visual upgrades (iRoPE, RoPE-exempt registers, V2PE RoPE jitter)
    irope_nope_every: int = 4
    n_register_tokens: int = 4
    rope_jitter: float = 0.08

    # Decoder / MTP / RoPE
    mtp_k: int = 4
    tok_dropout: float = 0.0
    max_dec_len: int = 4096
    rope_base: float = 10000.0
    rope_coord_scale: float = 128.0
    grad_checkpoint: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        head_dim = self.d_model // self.n_heads
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by 4 for RoPE2D")
        if self.max_mem_tokens < max(1, self.n_bg_summary + 1):
            raise ValueError(f"max_mem_tokens ({self.max_mem_tokens}) must exceed n_bg_summary ({self.n_bg_summary})")
        if self.n_register_tokens < 0:
            raise ValueError(f"n_register_tokens ({self.n_register_tokens}) must be >= 0")
        if self.irope_nope_every < 0:
            raise ValueError(f"irope_nope_every ({self.irope_nope_every}) must be >= 0")
        self.rope_jitter = float(max(0.0, self.rope_jitter))
        self.mixstyle_prob = float(max(0.0, min(1.0, self.mixstyle_prob)))
        self.vrfm_band_alpha = float(max(0.0, self.vrfm_band_alpha))

    @classmethod
    def small_45m(cls, **overrides: Any) -> "UniOMRv2Config":
        """~43M total params (~42M infer), < 25 MB pure INT4, ~32 MB Hybrid FP16/INT4-QAT."""
        params: Dict[str, Any] = dict(
            d_model=384,
            n_heads=8,
            n_kv_heads=2,
            n_axial_layers=2,
            n_global_layers=4,
            n_dec_layers=8,
            ffn_mult=4.0,
            axial_ffn_mult=2.0,
            backbone_channels=(64, 128, 240, 448),
            backbone_blocks=(1, 2, 3, 3),
            backbone_expand=2.0,
            router_hidden=64,
            max_mem_tokens=3072,
            mtp_k=4,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def edge85m(cls, **overrides: Any) -> "UniOMRv2Config":
        """d_model=512, ~81M total params (~80M infer), ~50 MB Hybrid FP16/INT4-QAT (< 55 MB w/ scales)."""
        params: Dict[str, Any] = dict(
            d_model=512,
            n_heads=8,
            n_kv_heads=2,
            n_axial_layers=2,
            n_global_layers=4,
            n_dec_layers=10,
            ffn_mult=4.0,
            axial_ffn_mult=2.0,
            backbone_channels=(64, 128, 240, 448),
            backbone_blocks=(1, 2, 3, 3),
            backbone_expand=2.0,
            router_hidden=64,
            max_mem_tokens=3072,
            mtp_k=4,
        )
        params.update(overrides)
        return cls(**params)


# ---------------------------------------------------------------------------
# 1. Anisotropic ConvNeXt / InceptionNeXt Music Backbone (~5-6M params)
# ---------------------------------------------------------------------------

class AnisotropicMusicBlock(nn.Module):
    """InceptionNeXt 1/4-channel-split inverted-bottleneck block with parallel anisotropic strips.

    Splits channels into 4 groups (``c4 = channels // 4``, remainder on identity):
      - ``1x9`` horizontal strip on ``c4`` (staff lines, beams, slurs, hairpins)
      - ``9x1`` vertical strip on ``c4`` (stems, chords, barlines, braces)
      - ``3x3`` local kernel on ``c4`` (noteheads, accidentals, dots, flags)
      - identity pass-through on remaining ``channels - 3 * c4`` channels
    Concatenated before GroupNorm + pointwise ``pw1 -> GELU -> pw2`` inverted bottleneck.
    """

    def __init__(self, channels: int, expand_ratio: float = 2.0, max_groups: int = 16) -> None:
        super().__init__()
        if channels < 4:
            raise ValueError(f"AnisotropicMusicBlock requires channels >= 4, got {channels}")
        g = _group_count(channels, max_groups)
        hidden = max(channels, int(round(channels * expand_ratio)))
        c4 = channels // 4
        self.split_sizes: Tuple[int, int, int, int] = (c4, c4, c4, channels - 3 * c4)

        self.dw_horiz = nn.Conv2d(c4, c4, kernel_size=(1, 9), padding=(0, 4), groups=c4, bias=False)
        self.dw_vert = nn.Conv2d(c4, c4, kernel_size=(9, 1), padding=(4, 0), groups=c4, bias=False)
        self.dw_local = nn.Conv2d(c4, c4, kernel_size=(3, 3), padding=(1, 1), groups=c4, bias=False)
        self.gn1 = nn.GroupNorm(g, channels)

        self.pw1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.gn2 = nn.GroupNorm(g, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_h, x_v, x_l, x_id = torch.split(x, self.split_sizes, dim=1)
        h = torch.cat([self.dw_horiz(x_h), self.dw_vert(x_v), self.dw_local(x_l), x_id], dim=1)
        h = self.gn1(h)
        h = self.pw2(F.gelu(self.pw1(h)))
        h = self.gn2(h)
        return F.silu(x + h)


class _AnisoStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_blocks: int,
        stride: int = 2,
        expand_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        g = _group_count(out_ch, 16)
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[AnisotropicMusicBlock(out_ch, expand_ratio=expand_ratio) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class _P2LateralFuse(nn.Module):
    """Lateral projection from pre-unshuffled P2 (/4 -> /8) into /8 stages."""

    def __init__(self, p2_ch: int, target_ch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(p2_ch * 4, target_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(target_ch, 16), target_ch)

    def forward(self, p2_unshuffled: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        out = F.silu(self.norm(self.proj(p2_unshuffled)))
        if out.shape[-2:] != target_hw:
            out = F.interpolate(out, size=target_hw, mode="bilinear", align_corners=False)
        return out


class AnisotropicMusicBackbone(nn.Module):
    """Anisotropic music feature extractor outputting ``/8`` features + high-res ``P2/P3``.

    Stages:
      - ``stage1``: ``/2``
      - ``stage2``: ``/4`` (``P2`` — preserved via lossless PixelUnshuffle lateral skip)
      - ``stage3``: ``/8`` (``P3`` — fused with ``P2`` lateral skip)
      - ``stage4``: ``/8`` (stride 1, fused with ``P2`` lateral skip, projected to ``d_model``)
    """

    STRIDE: int = 8

    def __init__(
        self,
        d_model: int = 384,
        channels: Tuple[int, int, int, int] = (64, 128, 240, 448),
        blocks: Tuple[int, int, int, int] = (1, 2, 3, 3),
        expand_ratio: float = 2.0,
        mixstyle_prob: float = 0.0,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        n1, n2, n3, n4 = blocks
        self.p2_ch = c2
        self.p3_ch = c3

        self.stage1 = _AnisoStage(3, c1, n_blocks=n1, stride=2, expand_ratio=expand_ratio)
        self.stage2 = _AnisoStage(c1, c2, n_blocks=n2, stride=2, expand_ratio=expand_ratio)
        self.stage3 = _AnisoStage(c2, c3, n_blocks=n3, stride=2, expand_ratio=expand_ratio)
        self.stage4 = _AnisoStage(c3, c4, n_blocks=n4, stride=1, expand_ratio=expand_ratio)

        self.p2_unshuffle = nn.PixelUnshuffle(2)
        self.p2_to_p3 = _P2LateralFuse(c2, c3)
        self.p2_to_p4 = _P2LateralFuse(c2, c4)

        self.mixstyle = MixStyle2d(p=mixstyle_prob, alpha=0.1)
        self.proj = nn.Conv2d(c4, d_model, kernel_size=1, bias=False)
        self.out_norm = nn.GroupNorm(_group_count(d_model, 16), d_model)

    def forward(
        self,
        x: torch.Tensor,
        return_pyramid: bool = False,
    ) -> Any:
        h1 = self.mixstyle(self.stage1(x))
        p2 = self.mixstyle(self.stage2(h1))  # /4
        _, _, h2, w2 = p2.shape
        pad_h = h2 % 2
        pad_w = w2 % 2
        p2_padded = F.pad(p2, (0, pad_w, 0, pad_h), mode="replicate") if (pad_h or pad_w) else p2
        p2_unshuffled = self.p2_unshuffle(p2_padded)
        p3 = self.stage3(p2)                 # /8
        p3 = p3 + self.p2_to_p3(p2_unshuffled, (p3.shape[2], p3.shape[3]))
        p4 = self.stage4(p3)                 # /8
        p4 = p4 + self.p2_to_p4(p2_unshuffled, (p4.shape[2], p4.shape[3]))
        out = self.out_norm(self.proj(p4))
        if return_pyramid:
            return out, p2, p3
        return out


# ---------------------------------------------------------------------------
# 2. Aspect-Normalized RoPE + Hybrid Axial-Merger-Global Encoder
# ---------------------------------------------------------------------------

class AspectNormalizedRoPE1D(nn.Module):
    """1D RoPE over coordinates normalized to ``[0, 1] * coord_scale``.

    Ensures identical rotary phases across different image resolutions
    (e.g. 1280px vs 1536px). Supports training-only V2PE box jitter.
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        coord_scale: float = 128.0,
        rope_jitter: float = 0.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.coord_scale = float(coord_scale)
        self.rope_jitter = float(max(0.0, rope_jitter))
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_sin_cache: "OrderedDict[Tuple[int, torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._pos_cache: "OrderedDict[Tuple[int, torch.device], torch.Tensor]" = OrderedDict()
        self._max_cache_size: int = 64

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        use_jitter = self.training and self.rope_jitter > 0.0
        if not use_jitter:
            cached = _lru_get(self._cos_sin_cache, (seq_len, device, dtype))
            if cached is not None:
                return cached

        pos_norm = _lru_get(self._pos_cache, (seq_len, device))
        if pos_norm is None:
            if seq_len <= 1:
                pos_norm = torch.zeros(seq_len, device=device, dtype=torch.float32)
            else:
                pos_norm = torch.linspace(0.0, 1.0, steps=seq_len, device=device, dtype=torch.float32)
            _lru_put(self._pos_cache, (seq_len, device), pos_norm, self._max_cache_size)

        if use_jitter:
            j = self.rope_jitter
            scale = 1.0 + (torch.rand((), device=device, dtype=torch.float32) * 2.0 - 1.0) * j
            shift = (torch.rand((), device=device, dtype=torch.float32) * 2.0 - 1.0) * (0.5 * j)
            pos_norm = pos_norm * scale + shift
        pos = pos_norm * self.coord_scale
        inv_f = self.inv_freq if self.inv_freq.device == device else self.inv_freq.to(device)
        freqs = torch.outer(pos, inv_f)  # (L, D/2)
        cos_h = freqs.cos().to(dtype)
        sin_h = freqs.sin().to(dtype)
        cos = torch.cat([cos_h, cos_h], dim=-1)[None, None, :, :]
        sin = torch.cat([sin_h, sin_h], dim=-1)[None, None, :, :]
        if not use_jitter:
            _lru_put(self._cos_sin_cache, (seq_len, device, dtype), (cos, sin), self._max_cache_size)
        return cos, sin


class AspectNormalizedRoPE2D(nn.Module):
    """Interleaved ``[y0, x0, y1, x1, ...]`` Aspect-Normalized 2D RoPE (Qwen3-VL MRoPE style).

    Maps ``(y, x)`` into ``[0, 1] * coord_scale`` and interleaves ``y`` and ``x``
    rotary frequencies across the full spectrum so both spatial axes share high
    and low frequency bands while preserving exact resolution invariance across
    e.g. 1280px and 1536px pages. Supports training-only V2PE box jitter.
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        coord_scale: float = 128.0,
        rope_jitter: float = 0.0,
    ) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"AspectNormalizedRoPE2D requires head_dim % 4 == 0, got {head_dim}")
        self.coord_scale = float(coord_scale)
        self.rope_jitter = float(max(0.0, rope_jitter))
        half_dim = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2, dtype=torch.float32) / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._grid_cache: "OrderedDict[Tuple[int, int, torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._max_cache_size: int = 32

    def _get_grid_cos_sin(
        self,
        h_p: int,
        w_p: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (h_p, w_p, device, dtype)
        cached = _lru_get(self._grid_cache, key)
        if cached is not None:
            return cached
        with torch.no_grad():
            inv_f = (
                self.inv_freq
                if (self.inv_freq.device == device and self.inv_freq.dtype == torch.float32)
                else self.inv_freq.to(device, dtype=torch.float32)
            )
            ys = torch.arange(h_p, device=device, dtype=torch.float32)
            xs = torch.arange(w_p, device=device, dtype=torch.float32)
            y_norm = ys / float(max(1, h_p - 1))
            x_norm = xs / float(max(1, w_p - 1))
            fy = (y_norm * self.coord_scale).unsqueeze(-1) * inv_f
            fx = (x_norm * self.coord_scale).unsqueeze(-1) * inv_f
            fy_2d = fy[:, None, :].expand(h_p, w_p, -1)
            fx_2d = fx[None, :, :].expand(h_p, w_p, -1)
            emb_half = torch.stack([fy_2d, fx_2d], dim=-1).flatten(-2).reshape(h_p * w_p, -1)
            cos_h = emb_half.cos().to(dtype)
            sin_h = emb_half.sin().to(dtype)
            cos_grid = torch.cat([cos_h, cos_h], dim=-1)
            sin_grid = torch.cat([sin_h, sin_h], dim=-1)

            cy = torch.tensor([(h_p - 1) * 0.5], device=device, dtype=torch.float32) / float(max(1, h_p - 1))
            cx = torch.tensor([(w_p - 1) * 0.5], device=device, dtype=torch.float32) / float(max(1, w_p - 1))
            fyc = (cy * self.coord_scale).unsqueeze(-1) * inv_f
            fxc = (cx * self.coord_scale).unsqueeze(-1) * inv_f
            emb_c_half = torch.stack([fyc, fxc], dim=-1).flatten(-2)
            cos_ch = emb_c_half.cos().to(dtype)
            sin_ch = emb_c_half.sin().to(dtype)
            cos_center = torch.cat([cos_ch, cos_ch], dim=-1)
            sin_center = torch.cat([sin_ch, sin_ch], dim=-1)

            val = (cos_grid, sin_grid, cos_center, sin_center)
            _lru_put(self._grid_cache, key, val, self._max_cache_size)
            return val

    def forward_indexed(
        self,
        kept_idx: torch.Tensor,
        grid_hw: Tuple[int, int],
        n_bg_summary: int,
        dtype: torch.dtype,
        coords: Optional[torch.Tensor] = None,
        is_full_grid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast cached 2D-RoPE lookup for router-indexed grid tokens (exact parity with ``forward``)."""
        if self.training and self.rope_jitter > 0.0:
            if coords is None:
                raise ValueError("coords must be provided when rope_jitter > 0 during training")
            return self.forward(coords, grid_hw=grid_hw, dtype=dtype)
        h_p, w_p = int(grid_hw[0]), int(grid_hw[1])
        cos_grid, sin_grid, cos_center, sin_center = self._get_grid_cos_sin(h_p, w_p, kept_idx.device, dtype)
        B = kept_idx.shape[0]
        if is_full_grid and n_bg_summary == 0:
            cos = cos_grid.unsqueeze(0).expand(B, -1, -1)
            sin = sin_grid.unsqueeze(0).expand(B, -1, -1)
        else:
            cos = cos_grid[kept_idx]
            sin = sin_grid[kept_idx]
            if n_bg_summary > 0:
                cos = torch.cat([cos, cos_center.expand(B, n_bg_summary, -1)], dim=1)
                sin = torch.cat([sin, sin_center.expand(B, n_bg_summary, -1)], dim=1)
        return cos.unsqueeze(1), sin.unsqueeze(1)

    def forward(
        self,
        coords: torch.Tensor,
        grid_hw: Tuple[int, int],
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_p, w_p = int(grid_hw[0]), int(grid_hw[1])
        inv_f = (
            self.inv_freq
            if (self.inv_freq.device == coords.device and self.inv_freq.dtype == torch.float32)
            else self.inv_freq.to(coords.device, dtype=torch.float32)
        )
        coords_f = coords if coords.dtype == torch.float32 else coords.to(torch.float32)
        y_norm = coords_f[..., 0] / float(max(1, h_p - 1))
        x_norm = coords_f[..., 1] / float(max(1, w_p - 1))
        if self.training and self.rope_jitter > 0.0:
            j = self.rope_jitter
            if y_norm.ndim >= 2:
                bcast_shape = (y_norm.shape[0],) + (1,) * (y_norm.ndim - 1)
            else:
                bcast_shape = ()
            scale_y = 1.0 + (torch.rand(bcast_shape, device=coords.device, dtype=torch.float32) * 2.0 - 1.0) * j
            scale_x = 1.0 + (torch.rand(bcast_shape, device=coords.device, dtype=torch.float32) * 2.0 - 1.0) * j
            shift_y = (torch.rand(bcast_shape, device=coords.device, dtype=torch.float32) * 2.0 - 1.0) * (0.5 * j)
            shift_x = (torch.rand(bcast_shape, device=coords.device, dtype=torch.float32) * 2.0 - 1.0) * (0.5 * j)
            y_norm = y_norm * scale_y + shift_y
            x_norm = x_norm * scale_x + shift_x
        fy = (y_norm * self.coord_scale).unsqueeze(-1) * inv_f
        fx = (x_norm * self.coord_scale).unsqueeze(-1) * inv_f
        # Interleaved [y0, x0, y1, x1, ...] across rotary frequencies
        emb_half = torch.stack([fy, fx], dim=-1).flatten(-2)
        cos_h = emb_half.cos().to(dtype)
        sin_h = emb_half.sin().to(dtype)
        cos = torch.cat([cos_h, cos_h], dim=-1).unsqueeze(1)
        sin = torch.cat([sin_h, sin_h], dim=-1).unsqueeze(1)
        return cos, sin


class _SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = self.q_norm(q.view(B, L, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(B, L, self.n_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, rope_cos, rope_sin)
        out = _sdpa(q, k, v, attn_mask=None, is_causal=False)
        out = out.transpose(1, 2).reshape(B, L, self.d_model)
        return self.o_proj(out)


class AxialEncoderLayer(nn.Module):
    """Row-then-Column Axial Attention layer with ``O(N*(H'+W'))`` complexity."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_row = RMSNorm(d_model)
        self.row_attn = _SelfAttention(d_model, n_heads)
        self.norm_col = RMSNorm(d_model)
        self.col_attn = _SelfAttention(d_model, n_heads)
        self.norm_ffn = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        row_cos: torch.Tensor,
        row_sin: torch.Tensor,
        col_cos: torch.Tensor,
        col_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Args: x of shape ``(B, H', W', D)``."""
        B, H, W, D = x.shape

        # 1. Row attention: (B * H, W, D)
        x_row = self.norm_row(x).reshape(B * H, W, D)
        x = x + self.drop(self.row_attn(x_row, row_cos, row_sin).reshape(B, H, W, D))

        # 2. Column attention: (B * W, H, D)
        x_col = self.norm_col(x).permute(0, 2, 1, 3).reshape(B * W, H, D)
        col_out = self.col_attn(x_col, col_cos, col_sin).reshape(B, W, H, D).permute(0, 2, 1, 3)
        x = x + self.drop(col_out)

        # 3. FFN
        x = x + self.drop(self.ffn(self.norm_ffn(x)))
        return x


class PatchMerger2x2(nn.Module):
    """Qwen2.5-VL style 2x2 MLP Patch Merger (4x token compression)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.fc1 = nn.Linear(d_model * 4, d_model, bias=False)
        self.fc2 = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: ``(B, H, W, D)`` -> returns ``(B, ceil(H/2), ceil(W/2), D)``."""
        x = self.norm(x)
        B, H, W, D = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            # Pad in NCHW so replicate mode works cleanly.
            x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h), mode="replicate").permute(0, 2, 3, 1)
            H += pad_h
            W += pad_w
        h_m, w_m = H // 2, W // 2
        x = x.reshape(B, h_m, 2, w_m, 2, D).permute(0, 1, 3, 2, 4, 5).reshape(B, h_m, w_m, 4 * D)
        return self.fc2(F.silu(self.fc1(x)))


class InkTokenRouterV2(nn.Module):
    """Ink-saliency router with v1's 2-patch eroded ``paper_gate`` + hard ``max_mem_tokens`` cap."""

    def __init__(
        self,
        d_model: int,
        keep_ratio: float = 0.40,
        n_bg_summary: int = 4,
        use_ink_prior: bool = True,
        dynamic_routing: bool = True,
        min_keep_ratio: float = 0.15,
        max_keep_ratio: float = 0.75,
        saliency_thr: float = 0.05,
        max_mem_tokens: int = 10240,
        router_hidden: int = 64,
        vrfm_band_alpha: float = 0.30,
        adaptive_ink_target: bool = True,
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
        self.max_mem_tokens = int(max_mem_tokens)
        self.vrfm_band_alpha = float(max(0.0, vrfm_band_alpha))
        self.adaptive_ink_target = bool(adaptive_ink_target)

        hidden = max(32, int(router_hidden))
        self.saliency_net = nn.Sequential(
            nn.Conv2d(d_model, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden, 8), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        self.ink_prior_scale = nn.Parameter(torch.tensor(1.0))

        self.bg_queries = nn.Parameter(torch.randn(1, n_bg_summary, d_model) * 0.02)
        self.bg_key_proj = nn.Linear(d_model, d_model, bias=False)
        self.bg_val_proj = nn.Linear(d_model, d_model, bias=False)
        self.bg_out_norm = RMSNorm(d_model)
        self._inv_sqrt_d: float = 1.0 / math.sqrt(d_model)
        self._grid_cache: "OrderedDict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._grid_cache_max: int = 64

    @staticmethod
    @torch.no_grad()
    def _compute_ink_and_paper_gate(
        images: torch.Tensor, h_p: int, w_p: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute edge-connected dark-border-masked high-pass ink map and soft paper_gate.

        Uses a fast vectorized 4-edge inward scan stopping at the first bright
        pixel (``luma >= 0.35``) to isolate only dark exterior desk/scanner
        borders (``luma < 0.22`` connected to the outer boundary), dilates that
        outer border mask by 6px in pixel space (separable 1x13 + 13x1 max-pool)
        to eliminate the paper-to-desk transition halo without eroding tight
        left clefs, and pools to the feature grid. Interior shadows and dense
        dark ink on aged/shadowed real HW photos are NEVER masked out.
        """
        if images.shape[1] == 3:
            luma = images[:, 0:1].to(torch.float32) * 0.299
            luma.add_(images[:, 1:2], alpha=0.587).add_(images[:, 2:3], alpha=0.114)
        else:
            luma = images.to(torch.float32).mean(dim=1, keepdim=True)

        # Edge-connected dark-border mask: 2 cumsums (0 flips) — bit-identical to 4-edge scan.
        # Since (luma < 0.22) implies bright_i == 0, cs_y == cs_y[:, :, -1:, :] iff no bright pixel at y..H-1.
        bright_i = (luma >= 0.35).to(torch.int32)
        cs_y = bright_i.cumsum(dim=2)
        cs_x = bright_i.cumsum(dim=3)
        edge_dark = (
            (cs_y == 0)
            | (cs_y == cs_y[:, :, -1:, :])
            | (cs_x == 0)
            | (cs_x == cs_x[:, :, :, -1:])
        )
        border_px = (edge_dark & (luma < 0.22)).to(luma.dtype)

        # Dilate ONLY the outer border mask by 6px in pixel space (separable 1x13 + 13x1)
        border_dilated_px = F.max_pool2d(
            F.max_pool2d(border_px, kernel_size=(1, 13), stride=1, padding=(0, 6)),
            kernel_size=(13, 1),
            stride=1,
            padding=(6, 0),
        )
        paper_px = (1.0 - border_dilated_px).clamp_(0.0, 1.0)

        local_bg = F.avg_pool2d(
            F.avg_pool2d(luma, kernel_size=(1, 31), stride=1, padding=(0, 15), count_include_pad=False),
            kernel_size=(31, 1),
            stride=1,
            padding=(15, 0),
            count_include_pad=False,
        )
        local_ink = local_bg.sub(luma).div_(local_bg.clamp(min=0.15)).clamp_(0.0, 1.0).mul_(paper_px)
        ink_max = F.adaptive_max_pool2d(local_ink, (h_p, w_p))
        ink_avg = F.adaptive_avg_pool2d(local_ink, (h_p, w_p))
        local_ink_grid = ink_max.mul_(0.5).add_(ink_avg, alpha=0.5)

        paper_grid_f = F.adaptive_avg_pool2d(paper_px, (h_p, w_p))
        effective_paper = (paper_grid_f > 0.5).detach()

        local_ink_grid = local_ink_grid.mul_(effective_paper.to(local_ink_grid.dtype))
        paper_gate = torch.where(effective_paper, 1.0, 0.05).to(paper_grid_f.dtype).detach()
        return local_ink_grid, paper_gate, effective_paper

    def forward(
        self,
        feat_2d: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        keep_ratio_override: Optional[float] = None,
        feat_nhwc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        B, D, H_p, W_p = feat_2d.shape
        N = H_p * W_p
        ratio = self.keep_ratio if keep_ratio_override is None else float(keep_ratio_override)
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {ratio}")

        saliency_logits = self.saliency_net(feat_2d)
        if self.vrfm_band_alpha > 0.0 and H_p >= 3:
            row_mean = saliency_logits.mean(dim=3, keepdim=True)
            row_smooth = F.avg_pool2d(
                row_mean,
                kernel_size=(3, 1),
                stride=1,
                padding=(1, 0),
                count_include_pad=False,
            )
            saliency_logits = saliency_logits + self.vrfm_band_alpha * row_smooth
        saliency_prob = torch.sigmoid(saliency_logits)

        ink_target: Optional[torch.Tensor] = None
        if images is not None:
            local_ink_grid, paper_gate, paper_mask_grid = self._compute_ink_and_paper_gate(
                images, H_p, W_p
            )
            local_ink_grid = local_ink_grid.to(saliency_logits.dtype)
            paper_gate = paper_gate.to(saliency_logits.dtype)
            saliency_prob = saliency_prob * paper_gate
            if getattr(self, "adaptive_ink_target", False):
                flat_ink = local_ink_grid.flatten(1).float()
                if torch.onnx.is_in_onnx_export():
                    k30 = max(1, min(N, int(round(0.30 * max(0, N - 1))) + 1))
                    bg_p30 = torch.topk(flat_ink, k=k30, dim=1, largest=False, sorted=True).values[:, -1:]
                else:
                    bg_p30 = torch.quantile(flat_ink, 0.30, dim=1, keepdim=True)
                ink_thr = (
                    (bg_p30 + 0.045)
                    .clamp(min=0.045, max=0.10)
                    .view(B, 1, 1, 1)
                    .to(local_ink_grid.dtype)
                )
                ink_target = ((local_ink_grid > ink_thr) & paper_mask_grid).to(saliency_logits.dtype)
            else:
                ink_target = ((local_ink_grid > 0.04) & paper_mask_grid).to(saliency_logits.dtype)
            if self.use_ink_prior:
                flat_f = local_ink_grid.flatten(1).float()
                denom = (
                    (flat_f.mean(dim=-1) + 2.5 * flat_f.std(dim=-1, unbiased=False))
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

        score_flat = score.flatten(1)
        tokens_flat = (
            feat_nhwc.reshape(B, N, D)
            if feat_nhwc is not None
            else feat_2d.permute(0, 2, 3, 1).reshape(B, N, D)
        )

        gkey = (H_p, W_p, feat_2d.device)
        cached_grid = _lru_get(self._grid_cache, gkey)
        if cached_grid is None:
            ys = torch.arange(H_p, device=feat_2d.device, dtype=torch.float32)
            xs = torch.arange(W_p, device=feat_2d.device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            base_coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1).unsqueeze(0)
            base_center = torch.tensor(
                [[[(H_p - 1) * 0.5, (W_p - 1) * 0.5]]],
                device=feat_2d.device,
                dtype=torch.float32,
            )
            _lru_put(self._grid_cache, gkey, (base_coords, base_center), self._grid_cache_max)
        else:
            base_coords, base_center = cached_grid
        grid_coords = base_coords.expand(B, -1, -1)

        if self.dynamic_routing and keep_ratio_override is None:
            if self.training:
                max_k = min(N, max(1, self.max_mem_tokens - self.n_bg_summary))
                k = min(max(int(math.ceil(N * self.max_keep_ratio)), 1), max_k)
            else:
                with torch.no_grad():
                    if torch.onnx.is_in_onnx_export():
                        k10 = max(1, min(N, int(round(0.10 * max(0, N - 1))) + 1))
                        p10 = torch.topk(score_flat, k=k10, dim=1, largest=False, sorted=True).values[:, -1:]
                    else:
                        p10 = torch.quantile(score_flat.float(), 0.10, dim=1, keepdim=True).to(score_flat.dtype)
                    thr = max(self.saliency_thr, 0.15)
                    above = (score_flat > (p10 + thr)).sum(dim=1)
                    k_dyn = int(above.max().item())
                    k_min = max(1, int(math.ceil(N * self.min_keep_ratio)))
                    k_max = max(k_min, min(N, int(math.ceil(N * self.max_keep_ratio))))
                    k = max(k_min, min(k_max, k_dyn))
        else:
            k = max(1, min(N, int(math.ceil(N * ratio))))

        # Hard memory cap: guarantees n_out <= max_mem_tokens (<= 3,072) on arbitrary resolutions!
        if k >= N and N <= self.max_mem_tokens:
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
                "is_full_grid": True,
            }
            return out_tokens, grid_coords, info

        max_kept = max(1, self.max_mem_tokens - self.n_bg_summary)
        k = max(1, min(k, N - 1 if N > 1 else 1, max_kept))

        _, topk_idx = torch.topk(score_flat, k=k, dim=1, largest=True, sorted=False)
        kept_idx, _ = torch.sort(topk_idx, dim=1)

        kept_scores = torch.gather(score_flat, 1, kept_idx)
        kept_gate = (0.5 + 0.5 * torch.sigmoid(kept_scores)).unsqueeze(-1)

        idx_feat = kept_idx.unsqueeze(-1).expand(-1, -1, D)
        kept_tokens = torch.gather(tokens_flat, 1, idx_feat) * kept_gate

        kept_coords = base_coords[0][kept_idx]

        kept_mask = torch.zeros(B, N, device=feat_2d.device, dtype=torch.bool)
        kept_mask.scatter_(1, kept_idx, True)

        if ink_target is not None and not self.training:
            with torch.no_grad():
                tgt_mask = ink_target.flatten(1) > 0.5
                n_ink = tgt_mask.sum(dim=1)
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

        bg_q_proj = (
            F.linear(self.bg_queries, self.bg_key_proj.weight.t()).to(tokens_flat.dtype) * self._inv_sqrt_d
        ).expand(B, -1, -1)
        bg_logits = torch.bmm(bg_q_proj, tokens_flat.transpose(1, 2))
        bg_logits = bg_logits - score_flat.unsqueeze(1)
        bg_logits = bg_logits.masked_fill(kept_mask.unsqueeze(1), -1e4)
        bg_attn = F.softmax(bg_logits, dim=-1)
        pooled = torch.bmm(bg_attn, tokens_flat)
        bg_tokens = self.bg_out_norm(self.bg_val_proj(pooled))

        center_coord = base_center.expand(B, self.n_bg_summary, 2)
        out_tokens = torch.cat([kept_tokens, bg_tokens], dim=1)
        out_coords = torch.cat([kept_coords, center_coord], dim=1)

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
            "is_full_grid": False,
        }
        return out_tokens, out_coords, info


class GlobalEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = _SelfAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.norm1(x), rope_cos=rope_cos, rope_sin=rope_sin))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class HybridAxialMergerEncoder(nn.Module):
    """DeepSeek-OCR / Qwen2.5-VL inspired Hybrid Encoder.

    Pipeline:
      1. 2 layers of Row-then-Column Axial Attention on high-res ``/8`` grid.
      2. 2x2 MLP Patch Merger (``/8 -> /16``) + ``InkTokenRouterV2`` (``k <= 3,072``).
      3. 4 layers of Global Aspect-Normalized 2D-RoPE Attention.
    """

    def __init__(self, cfg: UniOMRv2Config) -> None:
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model
        d_ff = int(round(d_model * cfg.ffn_mult))
        axial_d_ff = int(round(d_model * cfg.axial_ffn_mult))
        head_dim = d_model // cfg.n_heads

        self.axial_rope = AspectNormalizedRoPE1D(
            head_dim=head_dim,
            base=cfg.rope_base,
            coord_scale=cfg.rope_coord_scale,
            rope_jitter=cfg.rope_jitter,
        )
        self.axial_layers = nn.ModuleList([
            AxialEncoderLayer(d_model, cfg.n_heads, axial_d_ff, dropout=cfg.dropout)
            for _ in range(cfg.n_axial_layers)
        ])

        self.patch_merger = PatchMerger2x2(d_model)
        self.router = InkTokenRouterV2(
            d_model=d_model,
            keep_ratio=cfg.keep_ratio,
            n_bg_summary=cfg.n_bg_summary,
            use_ink_prior=cfg.use_ink_prior,
            dynamic_routing=cfg.dynamic_routing,
            min_keep_ratio=cfg.min_keep_ratio,
            max_keep_ratio=cfg.max_keep_ratio,
            saliency_thr=cfg.saliency_thr,
            max_mem_tokens=cfg.max_mem_tokens,
            router_hidden=cfg.router_hidden,
            vrfm_band_alpha=cfg.vrfm_band_alpha,
            adaptive_ink_target=bool(getattr(cfg, "adaptive_ink_target", False)),
        )

        self.rope2d = AspectNormalizedRoPE2D(
            head_dim=head_dim,
            base=cfg.rope_base,
            coord_scale=cfg.rope_coord_scale,
            rope_jitter=cfg.rope_jitter,
        )
        self.register_tokens: Optional[nn.Parameter] = (
            nn.Parameter(torch.randn(1, cfg.n_register_tokens, d_model) * 0.02)
            if cfg.n_register_tokens > 0
            else None
        )
        self.pos2d_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.pos2d_proj.weight)
        self.global_layers = nn.ModuleList([
            GlobalEncoderLayer(d_model, cfg.n_heads, d_ff, dropout=cfg.dropout)
            for _ in range(cfg.n_global_layers)
        ])
        self.enc_norm = RMSNorm(d_model)
        self._pos2d_grid_cache: "OrderedDict[Tuple[int, int, torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._pos2d_cache_max: int = 32
        self._reg_rope_cache: "OrderedDict[Tuple[torch.device, torch.dtype, int], Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._reg_rope_cache_max: int = 8

    def _get_pos2d_grid(
        self,
        h_p: int,
        w_p: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (h_p, w_p, device, dtype)
        cached = _lru_get(self._pos2d_grid_cache, key)
        if cached is not None:
            return cached
        with torch.no_grad():
            d = self.cfg.d_model
            q = d // 4
            freqs = torch.exp(
                torch.arange(0, q, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(1, q))
            )
            ys = torch.arange(h_p, device=device, dtype=torch.float32)
            xs = torch.arange(w_p, device=device, dtype=torch.float32)
            y_norm = ys / float(max(1, h_p - 1))
            x_norm = xs / float(max(1, w_p - 1))
            fy = (y_norm * self.cfg.rope_coord_scale).unsqueeze(-1) * freqs
            fx = (x_norm * self.cfg.rope_coord_scale).unsqueeze(-1) * freqs
            fy_2d = fy[:, None, :].expand(h_p, w_p, -1)
            fx_2d = fx[None, :, :].expand(h_p, w_p, -1)
            pe = torch.cat([fy_2d.sin(), fy_2d.cos(), fx_2d.sin(), fx_2d.cos()], dim=-1).reshape(h_p * w_p, -1)
            if pe.shape[-1] < d:
                pe = F.pad(pe, (0, d - pe.shape[-1]))
            pe_grid = pe.to(dtype)

            cy = torch.tensor([(h_p - 1) * 0.5], device=device, dtype=torch.float32) / float(max(1, h_p - 1))
            cx = torch.tensor([(w_p - 1) * 0.5], device=device, dtype=torch.float32) / float(max(1, w_p - 1))
            fyc = (cy * self.cfg.rope_coord_scale).unsqueeze(-1) * freqs
            fxc = (cx * self.cfg.rope_coord_scale).unsqueeze(-1) * freqs
            pe_c = torch.cat([fyc.sin(), fyc.cos(), fxc.sin(), fxc.cos()], dim=-1)
            if pe_c.shape[-1] < d:
                pe_c = F.pad(pe_c, (0, d - pe_c.shape[-1]))
            pe_center = pe_c.to(dtype)

            val = (pe_grid, pe_center)
            _lru_put(self._pos2d_grid_cache, key, val, self._pos2d_cache_max)
            return val

    def _sinusoidal_pos2d_indexed(
        self,
        kept_idx: torch.Tensor,
        grid_hw: Tuple[int, int],
        n_bg_summary: int,
        dtype: torch.dtype,
        is_full_grid: bool = False,
    ) -> torch.Tensor:
        h_p, w_p = int(grid_hw[0]), int(grid_hw[1])
        pe_grid, pe_center = self._get_pos2d_grid(h_p, w_p, kept_idx.device, dtype)
        B = kept_idx.shape[0]
        if is_full_grid and n_bg_summary == 0:
            return pe_grid.unsqueeze(0).expand(B, -1, -1)
        pe = pe_grid[kept_idx]
        if n_bg_summary > 0:
            pe = torch.cat([pe, pe_center.expand(B, n_bg_summary, -1)], dim=1)
        return pe

    def _sinusoidal_pos2d(
        self,
        coords: torch.Tensor,
        grid_hw: Tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        d = self.cfg.d_model
        q = d // 4
        device = coords.device
        freqs = torch.exp(
            torch.arange(0, q, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(1, q))
        )
        h_p, w_p = int(grid_hw[0]), int(grid_hw[1])
        y_norm = coords[..., 0].to(torch.float32) / float(max(1, h_p - 1))
        x_norm = coords[..., 1].to(torch.float32) / float(max(1, w_p - 1))
        fy = (y_norm * self.cfg.rope_coord_scale).unsqueeze(-1) * freqs
        fx = (x_norm * self.cfg.rope_coord_scale).unsqueeze(-1) * freqs
        pe = torch.cat([fy.sin(), fy.cos(), fx.sin(), fx.cos()], dim=-1)
        if pe.shape[-1] < d:
            pe = F.pad(pe, (0, d - pe.shape[-1]))
        return pe.to(dtype)

    def forward(
        self,
        feat_2d: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        keep_ratio_override: Optional[float] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """Encode ``(B, D, H', W')`` into DeepStack multi-level memory
        ``mem_levels = (mem_post_merger, mem_global_mid, mem_final)``, each
        ``(B, L_mem, D)`` sharing the exact same routed spatial indices
        ``L_mem <= max_mem_tokens``.
        """
        _, _, H_p, W_p = feat_2d.shape
        compute_dtype = _active_compute_dtype(feat_2d)
        x = feat_2d.permute(0, 2, 3, 1).contiguous()  # (B, H', W', D)
        if x.dtype != compute_dtype:
            x = x.to(dtype=compute_dtype)

        row_cos, row_sin = self.axial_rope(W_p, device=x.device, dtype=compute_dtype)
        col_cos, col_sin = self.axial_rope(H_p, device=x.device, dtype=compute_dtype)

        use_ckpt = self.training and self.cfg.grad_checkpoint and torch.is_grad_enabled()
        for layer in self.axial_layers:
            if use_ckpt:
                x = _grad_ckpt(
                    layer, x, row_cos, row_sin, col_cos, col_sin,
                    use_reentrant=False,
                )
            else:
                x = layer(x, row_cos, row_sin, col_cos, col_sin)

        # 2x2 MLP Patch Merger (/8 -> /16)
        merged = self.patch_merger(x)  # (B, H_m, W_m, D)
        merged_2d = merged.permute(0, 3, 1, 2)

        # InkTokenRouter with 2-patch eroded paper_gate + max_mem_tokens cap
        routed_tokens, routed_coords, router_info = self.router(
            merged_2d, images=images, keep_ratio_override=keep_ratio_override, feat_nhwc=merged
        )
        router_info["axial_grid_hw"] = (H_p, W_p)

        # DeepStack Level 0: post-merger fine-grained visual tokens (same routed indices K <= 3,072)
        mem_post_merger = routed_tokens

        # Aspect-Normalized Interleaved 2D-RoPE Global Attention
        # Prepend RoPE-exempt ViT-5 / DINOv3 register tokens (cos=1, sin=0 -> identity rotation)
        # and slice them off after global layers so DeepStack/decoder shapes remain identical.
        grid_hw = router_info["grid_hw"]
        kept_idx = router_info["kept_idx"]
        n_bg = int(router_info["n_bg_summary"])
        is_full_grid = bool(router_info.get("is_full_grid", False))
        pos2d = self.pos2d_proj(
            self._sinusoidal_pos2d_indexed(
                kept_idx, grid_hw=grid_hw, n_bg_summary=n_bg, dtype=routed_tokens.dtype, is_full_grid=is_full_grid
            )
        )
        cos2d, sin2d = self.rope2d.forward_indexed(
            kept_idx,
            grid_hw=grid_hw,
            n_bg_summary=n_bg,
            dtype=routed_tokens.dtype,
            coords=routed_coords,
            is_full_grid=is_full_grid,
        )
        mem = routed_tokens + pos2d

        n_reg = int(self.cfg.n_register_tokens) if self.register_tokens is not None else 0
        if n_reg > 0 and self.register_tokens is not None:
            B = mem.shape[0]
            reg = self.register_tokens.expand(B, -1, -1).to(mem.dtype)
            mem = torch.cat([reg, mem], dim=1)
            rkey = (cos2d.device, cos2d.dtype, cos2d.shape[-1])
            reg_cs = _lru_get(self._reg_rope_cache, rkey)
            if reg_cs is None:
                ones_c = torch.ones((1, 1, n_reg, cos2d.shape[-1]), device=cos2d.device, dtype=cos2d.dtype)
                zeros_s = torch.zeros((1, 1, n_reg, sin2d.shape[-1]), device=sin2d.device, dtype=sin2d.dtype)
                reg_cs = (ones_c, zeros_s)
                _lru_put(self._reg_rope_cache, rkey, reg_cs, self._reg_rope_cache_max)
            else:
                ones_c, zeros_s = reg_cs
            cos2d = torch.cat([ones_c.expand(B, -1, -1, -1), cos2d], dim=2)
            sin2d = torch.cat([zeros_s.expand(B, -1, -1, -1), sin2d], dim=2)

        n_global = len(self.global_layers)
        mid_idx = max(0, (n_global // 2) - 1)
        mem_global_mid = mem_post_merger
        for i, layer in enumerate(self.global_layers):
            if use_ckpt and (mem.shape[0] * mem.shape[1] > 12000):
                mem = _grad_ckpt(
                    layer, mem, cos2d, sin2d,
                    use_reentrant=False,
                )
            else:
                mem = layer(mem, rope_cos=cos2d, rope_sin=sin2d)
            if i == mid_idx:
                mem_global_mid = mem[:, n_reg:, :] if n_reg > 0 else mem

        if n_reg > 0:
            mem = mem[:, n_reg:, :]
        mem_final = self.enc_norm(mem) + pos2d
        mem_levels = (mem_post_merger, mem_global_mid, mem_final)
        return mem_levels, router_info


# ---------------------------------------------------------------------------
# 3. Training-Only Zero-Inference-Cost Layout Auxiliary Heads
# ---------------------------------------------------------------------------

class SysYHead(nn.Module):
    """Predicts normalized ``y_center in [0, 1]`` on ``<sys>`` decoder hidden states."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        mid = max(64, d_model // 2)
        self.net = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, 1, bias=True),
        )

    def forward(self, dec_hidden: torch.Tensor) -> torch.Tensor:
        """Args: ``(B, T, D)`` -> returns ``(B, T)`` in ``[0, 1]``."""
        return torch.sigmoid(self.net(dec_hidden).squeeze(-1))


class SysCountHead(nn.Module):
    """Predicts system count ``N_sys >= 0`` from pooled encoder memory."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        mid = max(64, d_model // 2)
        self.net = nn.Sequential(
            RMSNorm(d_model * 2),
            nn.Linear(d_model * 2, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, 1, bias=True),
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """Args: ``(B, L, D)`` -> returns ``(B,)`` positive predicted system count."""
        mean_pool = memory.mean(dim=1)
        max_pool = memory.amax(dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return F.softplus(self.net(pooled).squeeze(-1))


class StaffSegHead(nn.Module):
    """Predicts dense staff-line / ink band mask logits from high-res ``P2 (/4)`` + ``P3 (/8)``."""

    def __init__(self, p2_ch: int, p3_ch: int, mid_ch: int = 96) -> None:
        super().__init__()
        g = _group_count(mid_ch, 8)
        self.p2_proj = nn.Conv2d(p2_ch, mid_ch // 2, kernel_size=1, bias=False)
        self.p3_proj = nn.Conv2d(p3_ch, mid_ch // 2, kernel_size=1, bias=False)
        self.head = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, mid_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_ch, 1, kernel_size=1, bias=True),
        )

    def forward(self, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        """Returns ``(B, 1, H/4, W/4)`` raw segmentation logits."""
        p3_up = F.interpolate(self.p3_proj(p3), size=p2.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([self.p2_proj(p2), p3_up], dim=1)
        return self.head(fused)


class StrokeHogHead(nn.Module):
    """Training-only MonkeyOCRv2 / MaskFeat Sobel-HOG stroke-gradient aux head on ``P2 (/4)``.

    Zero inference cost: skipped at inference and stripped cleanly by ``strip_aux_heads()``.
    """

    def __init__(self, in_ch: int, n_bins: int = 8, mid_ch: int = 64) -> None:
        super().__init__()
        self.in_ch = int(in_ch)
        self.n_bins = int(n_bins)
        g = _group_count(mid_ch, 8)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, mid_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_ch, self.n_bins, kernel_size=1, bias=True),
        )

    def forward(self, p2: torch.Tensor) -> torch.Tensor:
        """Returns ``(B, n_bins, H/4, W/4)`` predicted Sobel-HOG orientation histogram."""
        return self.net(p2)


def compute_stroke_hog_target(
    img: torch.Tensor,
    target_hw: Tuple[int, int],
    n_bins: int = 8,
) -> torch.Tensor:
    """Fast pure-PyTorch Sobel-HOG unsigned orientation histogram pooled to ``target_hw``.

    Zero CUDA-CPU syncs; outputs L2-normalized ``(B, n_bins, H_t, W_t)`` float32 targets.
    """
    device = img.device
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=False):
        if img.shape[1] == 3:
            luma_f = img[:, 0:1].to(torch.float32) * 0.299
            luma_f.add_(img[:, 1:2], alpha=0.587).add_(img[:, 2:3], alpha=0.114)
        else:
            luma_f = img.to(torch.float32).mean(dim=1, keepdim=True)

        kxy = _get_sobel_kxy(device)
        gxy = F.conv2d(luma_f, kxy, padding=1)
        gx = gxy[:, 0:1]
        gy = gxy[:, 1:2]
        mag = gx.mul(gx).addcmul_(gy, gy).add_(1e-8).sqrt_()
        bin_f = torch.atan2(gy, gx).remainder_(math.pi).mul_(float(n_bins) / math.pi)
        bin_floor = torch.floor(bin_f)
        bin_lo = bin_floor.to(torch.long) % n_bins
        bin_hi = (bin_lo + 1) % n_bins
        w_hi = bin_f.sub_(bin_floor)
        w_lo = 1.0 - w_hi

        B, _, H, W = luma_f.shape
        hog_full = luma_f.new_zeros((B, n_bins, H, W), dtype=torch.float32)
        if n_bins >= 2:
            # bin_hi = (bin_lo + 1) % n_bins != bin_lo everywhere, so scatter_ is bit-identical
            # to scatter_add_ on zero-initialized hog_full while eliminating atomicAdd contention.
            hog_full.scatter_(1, bin_lo, mag * w_lo)
            hog_full.scatter_(1, bin_hi, mag.mul_(w_hi))
        else:
            hog_full.scatter_add_(1, bin_lo, mag * w_lo)
            hog_full.scatter_add_(1, bin_hi, mag.mul_(w_hi))

        h_t, w_t = int(target_hw[0]), int(target_hw[1])
        if H == 4 * h_t and W == 4 * w_t:
            hog_pooled = F.avg_pool2d(hog_full, kernel_size=4, stride=4)
        else:
            hog_pooled = F.adaptive_avg_pool2d(hog_full, (h_t, w_t))
        norm = hog_pooled.pow(2).sum(dim=1, keepdim=True).add_(1e-4).sqrt_()
        return hog_pooled.div_(norm)


def compute_stroke_hog_loss(
    pred_hog: torch.Tensor,
    img: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 loss between predicted ``StrokeHogHead`` output and Sobel-HOG target (0 syncs)."""
    n_bins = int(pred_hog.shape[1])
    target_hw = (int(pred_hog.shape[2]), int(pred_hog.shape[3]))
    with torch.no_grad():
        target = compute_stroke_hog_target(img, target_hw=target_hw, n_bins=n_bins)
    loss = F.smooth_l1_loss(pred_hog.float(), target.float(), beta=0.1)
    return loss.to(pred_hog.dtype)


def compute_sys_y_loss(
    sys_y_pred: torch.Tensor,
    sys_y_target: torch.Tensor,
    sys_y_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Branchless masked Smooth-L1 loss on ``<sys>`` normalized vertical center predictions.

    Positions with ``sys_y_target < 0`` (or ``sys_y_mask == False``) are ignored
    with zero CUDA-CPU synchronization.
    """
    valid = sys_y_target >= 0.0
    if sys_y_mask is not None:
        valid = valid & sys_y_mask.to(torch.bool)
    valid_f = valid.float()
    tgt = sys_y_target.float().clamp(0.0, 1.0)
    per_elem = F.smooth_l1_loss(sys_y_pred.float(), tgt, beta=0.05, reduction="none")
    loss = (per_elem * valid_f).sum() / valid_f.sum().clamp(min=1.0)
    return loss.to(sys_y_pred.dtype)


def compute_sys_count_loss(
    sys_count_pred: torch.Tensor,
    sys_count_target: torch.Tensor,
) -> torch.Tensor:
    """Branchless Smooth-L1 loss on per-page system count ``N_sys`` (ignores ``target < 0``)."""
    valid_f = (sys_count_target >= 0.0).float()
    tgt = sys_count_target.float().clamp(min=0.0)
    per_elem = F.smooth_l1_loss(sys_count_pred.float(), tgt, beta=0.5, reduction="none")
    loss = (per_elem * valid_f).sum() / valid_f.sum().clamp(min=1.0)
    return loss.to(sys_count_pred.dtype)


def compute_staff_seg_loss(
    staff_seg_logits: torch.Tensor,
    staff_seg_target: torch.Tensor,
) -> torch.Tensor:
    """BCE-with-logits + soft Dice loss on high-res staff-line / ink band mask."""
    tgt = staff_seg_target.float()
    if tgt.ndim == 3:
        tgt = tgt.unsqueeze(1)
    if tgt.shape[-2:] != staff_seg_logits.shape[-2:]:
        tgt = F.interpolate(tgt, size=staff_seg_logits.shape[-2:], mode="bilinear", align_corners=False)
    tgt = tgt.clamp(0.0, 1.0)
    logits_f = staff_seg_logits.float()
    bce = F.binary_cross_entropy_with_logits(logits_f, tgt)
    probs = torch.sigmoid(logits_f)
    inter = (probs * tgt).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3)) + 1.0
    dice = 1.0 - ((2.0 * inter + 1.0) / denom).mean()
    return (bce + 0.5 * dice).to(staff_seg_logits.dtype)


# ---------------------------------------------------------------------------
# 4. Grouped-Query Attention (GQA) Decoder (8 Q-heads / 2 KV-heads -> 4x smaller KV cache)
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention for decoder self-attention & cross-attention.

    Caches compact ``(B, n_kv_heads, L_k, head_dim)`` tensors so WebGPU
    KV-cache RAM and memory bandwidth drop by ``n_heads / n_kv_heads`` (4x).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        use_native_gqa: bool = True,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_dim = n_kv_heads * self.head_dim
        self.use_native_gqa = bool(use_native_gqa)
        self.use_rope = bool(use_rope)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        B, n_kv, L, hd = x.shape
        return x.unsqueeze(2).expand(B, n_kv, self.n_rep, L, hd).reshape(B, self.n_heads, L, hd)

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
        in_onnx = torch.onnx.is_in_onnx_export()
        can_fuse_qkv = (
            (not is_cross)
            and (kv_in is q_in)
            and (not in_onnx)
            and (self.q_proj.bias is None and self.k_proj.bias is None and self.v_proj.bias is None)
            and (not _is_qat_linear_active(self.q_proj))
            and (not _is_qat_linear_active(self.k_proj))
            and (not _is_qat_linear_active(self.v_proj))
        )
        if can_fuse_qkv:
            w_qkv = torch.cat((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=0)
            qkv = F.linear(q_in, w_qkv)
            d_q = self.n_heads * self.head_dim
            d_kv = self.n_kv_heads * self.head_dim
            q_raw, k_raw, v_raw = qkv.split((d_q, d_kv, d_kv), dim=-1)
            q = self.q_norm(q_raw.view(B, Lq, self.n_heads, self.head_dim)).transpose(1, 2)
            k = self.k_norm(k_raw.view(B, Lq, self.n_kv_heads, self.head_dim)).transpose(1, 2)
            v = v_raw.view(B, Lq, self.n_kv_heads, self.head_dim).transpose(1, 2)
            if self.use_rope and rope_cos is not None and rope_sin is not None:
                q, k = apply_rope(q, k, rope_cos, rope_sin)
            if past_kv is not None:
                k = torch.cat([past_kv[0], k], dim=2)
                v = torch.cat([past_kv[1], v], dim=2)
        else:
            q = self.q_norm(self.q_proj(q_in).view(B, Lq, self.n_heads, self.head_dim)).transpose(1, 2)

            if is_cross and past_kv is not None:
                k, v = past_kv
            else:
                if kv_in is None:
                    raise ValueError("kv_in must be provided when past_kv is None")
                Lk = kv_in.shape[1]
                can_fuse_kv = (
                    (not in_onnx)
                    and (self.k_proj.bias is None and self.v_proj.bias is None)
                    and (not _is_qat_linear_active(self.k_proj))
                    and (not _is_qat_linear_active(self.v_proj))
                )
                if can_fuse_kv:
                    w_kv = torch.cat((self.k_proj.weight, self.v_proj.weight), dim=0)
                    kv = F.linear(kv_in, w_kv)
                    d_kv = self.n_kv_heads * self.head_dim
                    k_raw, v_raw = kv.split((d_kv, d_kv), dim=-1)
                    k = self.k_norm(k_raw.view(B, Lk, self.n_kv_heads, self.head_dim)).transpose(1, 2)
                    v = v_raw.view(B, Lk, self.n_kv_heads, self.head_dim).transpose(1, 2)
                else:
                    k = self.k_norm(self.k_proj(kv_in).view(B, Lk, self.n_kv_heads, self.head_dim)).transpose(1, 2)
                    v = self.v_proj(kv_in).view(B, Lk, self.n_kv_heads, self.head_dim).transpose(1, 2)
                if self.use_rope and rope_cos is not None and rope_sin is not None:
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
                use_causal_flag = False
            else:
                eff_mask = _get_causal_offset_mask(Lq, past_len, q.device)
                use_causal_flag = False

        if self.use_native_gqa and self.n_rep > 1:
            out = _sdpa(q, k, v, attn_mask=eff_mask, is_causal=use_causal_flag, enable_gqa=True)
        else:
            k_exp = self._expand_kv(k)
            v_exp = self._expand_kv(v)
            out = _sdpa(q, k_exp, v_exp, attn_mask=eff_mask, is_causal=use_causal_flag, enable_gqa=False)
        out = out.transpose(1, 2).reshape(B, Lq, self.d_model)
        out = self.o_proj(out)
        if use_cache:
            return out, present_kv
        return out


class DeepStackFusion(nn.Module):
    """Qwen3-VL DeepStack multi-level visual memory injector.

    Fuses an earlier encoder level into ``mem_final`` via
    ``mem_final + tanh(gate) * proj(norm(mem_level))`` using a compact
    bottleneck (< 0.5M total params across early decoder layers) with 0% extra
    KV-cache storage (fused once before step-0 cross-attention K/V projection).
    """

    def __init__(self, d_model: int, bottleneck_ratio: float = 0.25, init_gate: float = 0.1) -> None:
        super().__init__()
        raw_mid = max(32, int(round(d_model * bottleneck_ratio)))
        d_mid = max(32, ((raw_mid + 15) // 32) * 32)
        self.norm = RMSNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(d_mid, d_model, bias=False),
        )
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

    def forward(self, mem_final: torch.Tensor, mem_level: torch.Tensor) -> torch.Tensor:
        delta = self.proj(self.norm(mem_level))
        return mem_final + torch.tanh(self.gate).to(mem_final.dtype) * delta


class GQADecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.use_rope = bool(use_rope)
        self.norm1 = RMSNorm(d_model)
        self.self_attn = GroupedQueryAttention(
            d_model, n_heads=n_heads, n_kv_heads=n_kv_heads, use_rope=self.use_rope
        )
        self.norm2 = RMSNorm(d_model)
        self.cross_attn = GroupedQueryAttention(
            d_model, n_heads=n_heads, n_kv_heads=n_kv_heads, use_rope=False
        )
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
        checkpoint_ffn: bool = False,
    ) -> Any:
        h = self.norm1(x)
        r_cos = rope_cos if self.use_rope else None
        r_sin = rope_sin if self.use_rope else None
        if use_cache:
            self_past = past_kv[0] if past_kv is not None else None
            cross_past = past_kv[1] if past_kv is not None else None
            sa_out, new_self_kv = self.self_attn(
                h, h,
                rope_cos=r_cos,
                rope_sin=r_sin,
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

        x = x + self.drop(self.self_attn(h, h, rope_cos=r_cos, rope_sin=r_sin, is_causal=True))
        x = x + self.drop(self.cross_attn(self.norm2(x), memory))
        if checkpoint_ffn and self.training and torch.is_grad_enabled():
            x = x + self.drop(_grad_ckpt(self.ffn, self.norm3(x), use_reentrant=False))
        else:
            x = x + self.drop(self.ffn(self.norm3(x)))
        return x


# ---------------------------------------------------------------------------
# Hybrid FP16/INT4-QAT Controller (reuses uni_omr.QATInt4Controller + fake_quant_int4_ste)
# ---------------------------------------------------------------------------

class QATInt4ControllerV2(QATInt4Controller):
    """Zero-schema-change Hybrid FP16/INT4 QAT controller for ``UniOMRv2Net``.

    Quantizes Transformer ``nn.Linear`` layers inside ``encoder.axial_layers``,
    ``encoder.patch_merger``, ``encoder.global_layers``, ``dec_layers``,
    ``mtp_heads``, and ``lm_head`` via ``fake_quant_int4_ste`` while keeping
    ``backbone``, ``encoder.router``, ``encoder.pos2d_proj``, ``tok_emb``,
    norms, and aux heads in FP16.
    """

    TARGET_ROOTS: Tuple[str, ...] = (
        "encoder.axial_layers",
        "encoder.patch_merger",
        "encoder.global_layers",
        "deepstack_fusions",
        "dec_layers",
        "mtp_heads",
        "lm_head",
    )

    def _resolve_root(self, dotted: str) -> Optional[nn.Module]:
        cur: Any = self.model
        for part in dotted.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                return None
        return cur if isinstance(cur, nn.Module) else None

    def _attach(self) -> None:
        for root_name in self.TARGET_ROOTS:
            root_mod = self._resolve_root(root_name)
            if root_mod is None:
                continue
            for sub_name, mod in root_mod.named_modules():
                if isinstance(mod, nn.Linear) and mod not in self._orig_forwards:
                    full_name = f"{root_name}.{sub_name}" if sub_name else root_name
                    self._orig_forwards[mod] = mod.forward
                    self.target_names.append(full_name)
                    self._patch_linear(mod)


# ---------------------------------------------------------------------------
# 5. UniOMRv2Net — Full Unified Model
# ---------------------------------------------------------------------------

class _CachedRoPE1D(RoPE1D):
    """RoPE1D with power-of-2 cache pre-allocation (min 512) to avoid O(T^2) rebuilds in generate()."""

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype)
        entry = self._cache.get(key)
        if entry is None or entry[0] < seq_len:
            alloc_len = max(512, 1 << (seq_len - 1).bit_length()) if seq_len > 0 else 512
            pos = torch.arange(alloc_len, device=device, dtype=torch.float32)
            freqs = torch.outer(pos, self.inv_freq.to(device))  # (L, D/2)
            cos_h = freqs.cos().to(dtype)
            sin_h = freqs.sin().to(dtype)
            cos = torch.cat([cos_h, cos_h], dim=-1)[None, None, :, :]
            sin = torch.cat([sin_h, sin_h], dim=-1)[None, None, :, :]
            self._cache[key] = (alloc_len, cos, sin)
            return cos[:, :, :seq_len, :], sin[:, :, :seq_len, :]
        _, cos, sin = entry
        return cos[:, :, :seq_len, :], sin[:, :, :seq_len, :]


class UniOMRv2Net(nn.Module):
    """UniOMR v2: Anisotropic FPN + Hybrid Axial/Merger/2D-RoPE Encoder + GQA MTP-4 Decoder."""

    def __init__(self, cfg: Optional[UniOMRv2Config] = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else UniOMRv2Config.small_45m()
        c = self.cfg
        d_ff = int(round(c.d_model * c.ffn_mult))
        head_dim = c.d_model // c.n_heads

        # 1. Anisotropic ConvNeXt/InceptionNeXt backbone with P2 (/4) FPN lateral skip
        self.backbone = AnisotropicMusicBackbone(
            d_model=c.d_model,
            channels=c.backbone_channels,
            blocks=c.backbone_blocks,
            expand_ratio=c.backbone_expand,
            mixstyle_prob=c.mixstyle_prob,
        )

        # 2. DeepSeek-OCR / Qwen2.5-VL Hybrid Encoder (Axial -> 2x2 Merger + Router -> Global 2D-RoPE)
        self.encoder = HybridAxialMergerEncoder(c)

        # 3. Training-Only Zero-Inference-Cost Layout & Stroke-Gradient Aux Heads
        self.sys_y_head: Optional[SysYHead] = SysYHead(c.d_model) if c.use_sys_y_aux else None
        self.sys_count_head: Optional[SysCountHead] = SysCountHead(c.d_model) if c.use_sys_count_aux else None
        self.staff_seg_head: Optional[StaffSegHead] = (
            StaffSegHead(self.backbone.p2_ch, self.backbone.p3_ch) if c.use_staff_seg_aux else None
        )
        self.stroke_hog_head: Optional[StrokeHogHead] = (
            StrokeHogHead(self.backbone.p2_ch, n_bins=8) if c.use_stroke_hog_head else None
        )

        # 4. GQA Causal Decoder (Llama 4 iRoPE) + DeepStack Multi-Level Visual Injection + MTP-4
        pad_idx = PAD_ID if PAD_ID < c.vocab_size else (c.vocab_size - 4 if c.vocab_size >= 4 else None)
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model, padding_idx=pad_idx)
        self.rope1d = _CachedRoPE1D(head_dim=head_dim, base=c.rope_base)
        self.deepstack_fusions = nn.ModuleList([
            DeepStackFusion(c.d_model) for _ in range(min(2, c.n_dec_layers))
        ])
        self.dec_layers = nn.ModuleList([
            GQADecoderLayer(
                d_model=c.d_model,
                n_heads=c.n_heads,
                n_kv_heads=c.n_kv_heads,
                d_ff=d_ff,
                dropout=c.dropout,
                use_rope=(c.irope_nope_every <= 0) or (((i + 1) % c.irope_nope_every) != 0),
            )
            for i in range(c.n_dec_layers)
        ])
        self.dec_norm = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

        n_extra_mtp = max(0, int(c.mtp_k) - 1)
        self.mtp_heads = nn.ModuleList([MTPHead(c.d_model) for _ in range(n_extra_mtp)])
        self._struct_weight_cache: Dict[Tuple[int, float, torch.device, torch.dtype], torch.Tensor] = {}

    def strip_aux_heads(self, strip_mtp: bool = False) -> "UniOMRv2Net":
        """Strip training-only layout & stroke-HOG aux heads (and optionally MTP heads) for 0%-cost export."""
        self.sys_y_head = None
        self.sys_count_head = None
        self.staff_seg_head = None
        self.stroke_hog_head = None
        if strip_mtp:
            self.mtp_heads = nn.ModuleList()
        return self

    def attach_qat_int4(self, block_size: int = 32, enabled: bool = False) -> QATInt4ControllerV2:
        """Attach zero-schema-change Hybrid FP16/INT4 QAT controller."""
        return QATInt4ControllerV2(self, block_size=block_size, enabled=enabled)

    def encode_image(
        self,
        images: torch.Tensor,
        keep_ratio_override: Optional[float] = None,
        return_aux: Optional[bool] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, Optional[torch.Tensor]], Dict[str, Any]]:
        """Encode arbitrary-resolution ``(B, 3, H, W)`` images into DeepStack ``mem_levels``."""
        want_aux = self.training if return_aux is None else bool(return_aux)
        need_pyramid = want_aux and (
            (self.staff_seg_head is not None) or (self.stroke_hog_head is not None)
        )

        if need_pyramid:
            feat_2d, p2, p3 = self.backbone(images, return_pyramid=True)
        else:
            feat_2d = self.backbone(images, return_pyramid=False)
            p2 = p3 = None

        mem_levels, router_info = self.encoder(
            feat_2d, images=images, keep_ratio_override=keep_ratio_override
        )
        mem_final = mem_levels[-1]

        aux_enc: Dict[str, Optional[torch.Tensor]] = {
            "sys_count_pred": None,
            "staff_seg_logits": None,
            "stroke_hog_pred": None,
        }
        if want_aux:
            if self.sys_count_head is not None:
                aux_enc["sys_count_pred"] = self.sys_count_head(mem_final)
            if self.staff_seg_head is not None and p2 is not None and p3 is not None:
                aux_enc["staff_seg_logits"] = self.staff_seg_head(p2, p3)
            if self.stroke_hog_head is not None and p2 is not None:
                aux_enc["stroke_hog_pred"] = self.stroke_hog_head(p2)

        return mem_levels, aux_enc, router_info

    def precompute_cross_kv(
        self,
        mem_levels: Any,
    ) -> List[Tuple[None, Tuple[torch.Tensor, torch.Tensor]]]:
        """Precompute DeepStackFusion + cross-attention K/V once per page for cached decoding."""
        if isinstance(mem_levels, (tuple, list)):
            mem_post_merger, mem_global_mid, mem_final = mem_levels
            level_mems: Optional[Tuple[torch.Tensor, torch.Tensor]] = (mem_post_merger, mem_global_mid)
        else:
            mem_final = mem_levels
            level_mems = None

        past_kvs: List[Tuple[None, Tuple[torch.Tensor, torch.Tensor]]] = []
        for idx, layer in enumerate(self.dec_layers):
            if level_mems is not None and idx < len(self.deepstack_fusions):
                layer_mem = self.deepstack_fusions[idx](mem_final, level_mems[idx])
            else:
                layer_mem = mem_final
            ca = layer.cross_attn
            B, Lk, _ = layer_mem.shape
            k = ca.k_norm(ca.k_proj(layer_mem).view(B, Lk, ca.n_kv_heads, ca.head_dim)).transpose(1, 2)
            v = ca.v_proj(layer_mem).view(B, Lk, ca.n_kv_heads, ca.head_dim).transpose(1, 2)
            past_kvs.append((None, (k, v)))
        return past_kvs

    def decode_tokens(
        self,
        tgt_ids: torch.Tensor,
        memory: Optional[Any],
        return_mtp: Optional[bool] = None,
        return_aux: Optional[bool] = None,
        past_kvs: Optional[List[Any]] = None,
        use_cache: bool = False,
        apply_tok_dropout: bool = True,
    ) -> Any:
        """Run causal GQA decoder over ``tgt_ids`` ``(B, T)`` conditioned on DeepStack ``mem_levels`` or tensor ``memory``."""
        _, T = tgt_ids.shape
        inp_ids = tgt_ids
        if self.training and apply_tok_dropout and self.cfg.tok_dropout > 0.0 and not use_cache:
            drop_mask = (
                (torch.rand_like(inp_ids, dtype=torch.float32) < self.cfg.tok_dropout)
                & (inp_ids != PAD_ID)
                & (inp_ids != BOS_ID)
                & (inp_ids != EOS_ID)
            )
            rand_toks = torch.randint(0, len(data.VOCAB), inp_ids.shape, device=inp_ids.device, dtype=inp_ids.dtype)
            inp_ids = torch.where(drop_mask, rand_toks, inp_ids)

        x = self.tok_emb(inp_ids)
        compute_dtype = _active_compute_dtype(x)
        if x.dtype != compute_dtype:
            x = x.to(dtype=compute_dtype)
        past_len = 0
        if past_kvs is not None and len(past_kvs) > 0 and past_kvs[0] is not None and past_kvs[0][0] is not None:
            past_len = int(past_kvs[0][0][0].shape[2])

        cos_all, sin_all = self.rope1d(past_len + T, device=x.device, dtype=compute_dtype)
        cos1d = cos_all[:, :, past_len : past_len + T, :]
        sin1d = sin_all[:, :, past_len : past_len + T, :]

        # Unpack DeepStack mem_levels = (mem_post_merger, mem_global_mid, mem_final) if provided.
        mem_final: Optional[torch.Tensor] = None
        level_mems: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if memory is not None:
            if isinstance(memory, (tuple, list)):
                mem_post_merger, mem_global_mid, mem_final = memory
                level_mems = (mem_post_merger, mem_global_mid)
            else:
                mem_final = memory

        bt = int(x.shape[0] * x.shape[1])
        can_dec_ckpt = self.training and self.cfg.grad_checkpoint and torch.is_grad_enabled()
        use_dec_full_ckpt = can_dec_ckpt and (bt > 8192)
        use_dec_ffn_ckpt = can_dec_ckpt and (4096 < bt <= 8192)
        new_past_kvs: List[Any] = []
        for idx, layer in enumerate(self.dec_layers):
            layer_past = past_kvs[idx] if past_kvs is not None else None
            need_cross_kv = (not use_cache) or (layer_past is None) or (layer_past[1] is None)
            if (not need_cross_kv) or (mem_final is None):
                layer_mem = None
            elif level_mems is not None and idx < len(self.deepstack_fusions):
                layer_mem = self.deepstack_fusions[idx](mem_final, level_mems[idx])
            else:
                layer_mem = mem_final

            if use_cache:
                x, layer_kv = layer(
                    x, memory=layer_mem, rope_cos=cos1d, rope_sin=sin1d,
                    past_kv=layer_past, use_cache=True,
                )
                new_past_kvs.append(layer_kv)
            elif use_dec_full_ckpt:
                x = _grad_ckpt(
                    layer, x, layer_mem, cos1d, sin1d,
                    use_reentrant=False,
                )
            else:
                x = layer(
                    x, memory=layer_mem, rope_cos=cos1d, rope_sin=sin1d,
                    checkpoint_ffn=use_dec_ffn_ckpt,
                )
        h = self.dec_norm(x)

        want_mtp = self.training if return_mtp is None else bool(return_mtp)
        if want_mtp and len(self.mtp_heads) > 0:
            can_fuse_mtp = (
                _HAS_F_RMS_NORM
                and (not torch.onnx.is_in_onnx_export())
                and all(
                    (head.proj.bias is None) and (not _is_qat_linear_active(head.proj))
                    for head in self.mtp_heads
                )
            )
            if can_fuse_mtp:
                eps0 = float(self.mtp_heads[0].norm.eps)
                if all(float(head.norm.eps) == eps0 for head in self.mtp_heads):
                    B_h, T_h, D_h = h.shape
                    if h.dtype in (torch.bfloat16, torch.float16):
                        with torch.amp.autocast(device_type=h.device.type, enabled=False):
                            h_unit = F.rms_norm(h, (D_h,), None, eps0)
                    else:
                        h_unit = F.rms_norm(h, (D_h,), None, eps0)
                    w_mtp = torch.cat(
                        [head.proj.weight * head.norm.weight.unsqueeze(0) for head in self.mtp_heads],
                        dim=0,
                    )
                    mtp_h = h.unsqueeze(0) + F.silu(
                        F.linear(h_unit, w_mtp)
                        .view(B_h, T_h, len(self.mtp_heads), D_h)
                        .permute(2, 0, 1, 3)
                    )
                    all_h = torch.cat((h.unsqueeze(0), mtp_h), dim=0)
                else:
                    all_h = torch.stack([h, *(head(h) for head in self.mtp_heads)], dim=0)
            else:
                all_h = torch.stack([h, *(head(h) for head in self.mtp_heads)], dim=0)
            all_logits = self.lm_head(all_h)
            logits = all_logits[0]
            mtp_logits: List[torch.Tensor] = list(all_logits[1:].unbind(0))
        else:
            logits = self.lm_head(h)
            mtp_logits = []

        want_aux = self.training if return_aux is None else bool(return_aux)
        sys_y_pred: Optional[torch.Tensor] = None
        if want_aux and self.sys_y_head is not None and not use_cache:
            sys_y_pred = self.sys_y_head(h)

        if use_cache:
            return logits, mtp_logits, new_past_kvs
        return logits, mtp_logits, sys_y_pred

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_len: int = 256,
        keep_ratio_override: Optional[float] = None,
        return_info: bool = False,
    ) -> Any:
        """Fast KV-cached GQA greedy decoding (0% aux/MTP overhead, 4x smaller KV cache)."""
        mem, _, router_info = self.encode_image(
            images, keep_ratio_override=keep_ratio_override, return_aux=False
        )
        B = images.shape[0]
        device = images.device
        v = int(self.cfg.vocab_size)
        pad_id = PAD_ID if v == VOCAB_SIZE else (v - 4)
        bos_id = BOS_ID if v == VOCAB_SIZE else (v - 3)
        eos_id = EOS_ID if v == VOCAB_SIZE else (v - 2)
        cur_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        past_kvs: Optional[List[Any]] = self.precompute_cross_kv(mem)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        out_tokens: List[torch.Tensor] = [cur_ids]

        for _ in range(max_len):
            logits, _, past_kvs = self.decode_tokens(
                cur_ids,
                memory=None,
                return_mtp=False,
                return_aux=False,
                past_kvs=past_kvs,
                use_cache=True,
            )
            next_tok = logits[:, -1, :].argmax(dim=-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, pad_id), next_tok)
            out_tokens.append(next_tok.unsqueeze(1))
            finished = finished | (next_tok == eos_id)
            if bool(finished.all()):
                break
            cur_ids = next_tok.unsqueeze(1)

        seq = torch.cat(out_tokens, dim=1)
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
        mem, aux_enc, router_info = self.encode_image(
            images, keep_ratio_override=keep_ratio_override, return_aux=return_aux
        )
        logits, mtp_logits, sys_y_pred = self.decode_tokens(
            tgt_ids, mem, return_mtp=return_mtp, return_aux=return_aux
        )
        return {
            "logits": logits,
            "mtp_logits": mtp_logits,
            "sys_y_pred": sys_y_pred,
            "sys_count_pred": aux_enc["sys_count_pred"],
            "staff_seg_logits": aux_enc["staff_seg_logits"],
            "stroke_hog_pred": aux_enc["stroke_hog_pred"],
            "router_info": router_info,
        }

    def compute_loss(
        self,
        out: Dict[str, Any],
        targets: torch.Tensor,
        sys_y_target: Optional[torch.Tensor] = None,
        sys_y_mask: Optional[torch.Tensor] = None,
        sys_count_target: Optional[torch.Tensor] = None,
        staff_seg_target: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        mtp_loss_weight: float = 0.30,
        saliency_loss_weight: float = 0.10,
        sys_y_loss_weight: float = 0.25,
        sys_count_loss_weight: float = 0.15,
        staff_seg_loss_weight: float = 0.20,
        stroke_hog_loss_weight: float = 0.15,
        label_smoothing: float = 0.0,
        ce_token_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined CE + MTP-4 + router saliency + layout & stroke-HOG aux losses."""
        logits: torch.Tensor = out["logits"]
        V = logits.shape[-1]
        ls = float(max(0.0, min(0.5, label_smoothing)))
        ce_w = (
            ce_token_weights.to(device=logits.device, dtype=logits.dtype)
            if ce_token_weights is not None
            else None
        )
        ce_loss = F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1),
            weight=ce_w,
            ignore_index=PAD_ID,
            label_smoothing=ls,
        )

        # MTP-4 auxiliary losses (unified single contiguous cross_entropy over PAD-padded shifted targets)
        mtp_logits: List[torch.Tensor] = out.get("mtp_logits", [])
        T = targets.shape[1]
        if mtp_logits and T > 1:
            n_mtp = len(mtp_logits)
            mtp_stacked = torch.stack(mtp_logits, dim=0)  # (M, B, T, V)
            shifted_tgts = torch.full(
                (n_mtp, targets.shape[0], T),
                PAD_ID,
                device=targets.device,
                dtype=targets.dtype,
            )
            for idx in range(n_mtp):
                shift = idx + 1
                if T > shift:
                    shifted_tgts[idx, :, :-shift] = targets[:, shift:]
            shifted_flat = shifted_tgts.reshape(-1)
            mtp_valid = (shifted_flat != PAD_ID).to(torch.float32)
            mtp_per_tok = F.cross_entropy(
                mtp_stacked.reshape(-1, V),
                shifted_flat,
                ignore_index=PAD_ID,
                label_smoothing=ls,
                reduction="none",
            )
            mtp_loss = (
                (mtp_per_tok.float() * mtp_valid).sum() / mtp_valid.sum().clamp(min=1.0)
            ).to(logits.dtype)
        else:
            mtp_loss = logits.new_zeros(())

        # Router saliency BCE loss
        rinfo = out.get("router_info") or {}
        sal_logits: Optional[torch.Tensor] = rinfo.get("saliency_logits")
        ink_target: Optional[torch.Tensor] = rinfo.get("ink_target")
        if sal_logits is not None and ink_target is not None and saliency_loss_weight > 0.0:
            saliency_loss = F.binary_cross_entropy_with_logits(
                sal_logits.float(), ink_target.float()
            ).to(logits.dtype)
        else:
            saliency_loss = logits.new_zeros(())

        # Training-only layout & stroke-HOG auxiliary losses
        sys_y_pred: Optional[torch.Tensor] = out.get("sys_y_pred")
        if sys_y_pred is not None and sys_y_target is not None and sys_y_loss_weight > 0.0:
            sys_y_loss = compute_sys_y_loss(sys_y_pred, sys_y_target, sys_y_mask=sys_y_mask)
        else:
            sys_y_loss = logits.new_zeros(())

        sys_count_pred: Optional[torch.Tensor] = out.get("sys_count_pred")
        if sys_count_pred is not None and sys_count_target is not None and sys_count_loss_weight > 0.0:
            sys_count_loss = compute_sys_count_loss(sys_count_pred, sys_count_target)
        else:
            sys_count_loss = logits.new_zeros(())

        staff_seg_logits: Optional[torch.Tensor] = out.get("staff_seg_logits")
        if staff_seg_logits is not None and staff_seg_target is not None and staff_seg_loss_weight > 0.0:
            staff_seg_loss = compute_staff_seg_loss(staff_seg_logits, staff_seg_target)
        else:
            staff_seg_loss = logits.new_zeros(())

        stroke_hog_pred: Optional[torch.Tensor] = out.get("stroke_hog_pred")
        if stroke_hog_pred is not None and images is not None and stroke_hog_loss_weight > 0.0:
            stroke_hog_loss = compute_stroke_hog_loss(stroke_hog_pred, images)
        else:
            stroke_hog_loss = logits.new_zeros(())

        total_loss = (
            ce_loss
            + mtp_loss_weight * mtp_loss
            + saliency_loss_weight * saliency_loss
            + sys_y_loss_weight * sys_y_loss
            + sys_count_loss_weight * sys_count_loss
            + staff_seg_loss_weight * staff_seg_loss
            + stroke_hog_loss_weight * stroke_hog_loss
        )
        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mtp_loss": mtp_loss,
            "saliency_loss": saliency_loss,
            "sys_y_loss": sys_y_loss,
            "sys_count_loss": sys_count_loss,
            "staff_seg_loss": staff_seg_loss,
            "stroke_hog_loss": stroke_hog_loss,
        }

    def param_summary(self, int4_block_size: int = 32) -> Dict[str, Any]:
        """Return exact parameter breakdown and WebGPU FP16/INT4/Hybrid byte estimates."""
        def _count(mod: Optional[nn.Module]) -> int:
            return sum(p.numel() for p in mod.parameters()) if mod is not None else 0

        total_params = _count(self)
        backbone_params = _count(self.backbone)
        encoder_params = _count(self.encoder)
        router_params = _count(self.encoder.router)
        deepstack_params = _count(self.deepstack_fusions)
        decoder_params = (
            _count(self.tok_emb)
            + deepstack_params
            + _count(self.dec_layers)
            + _count(self.dec_norm)
            + _count(self.lm_head)
        )
        mtp_params = _count(self.mtp_heads)
        aux_params = (
            _count(self.sys_y_head)
            + _count(self.sys_count_head)
            + _count(self.staff_seg_head)
            + _count(self.stroke_hog_head)
        )
        infer_params = total_params - aux_params - mtp_params

        # Exact Hybrid FP16/INT4-QAT split (matching QATInt4ControllerV2 target roots, excluding MTP/aux):
        int4_roots = (
            self.encoder.axial_layers,
            self.encoder.patch_merger,
            self.encoder.global_layers,
            self.deepstack_fusions,
            self.dec_layers,
            self.lm_head,
        )
        int4_weight_params = 0
        for root in int4_roots:
            for m in root.modules():
                if isinstance(m, nn.Linear):
                    int4_weight_params += m.weight.numel()
        fp16_keep_params = infer_params - int4_weight_params

        scale_bytes_per_param = 2.0 / float(int4_block_size)  # FP16 scale per block_size weights
        int4_raw_mb = (infer_params * 0.5) / 1_000_000.0
        int4_scaled_mb = (infer_params * (0.5 + scale_bytes_per_param)) / 1_000_000.0
        hybrid_raw_mb = (fp16_keep_params * 2.0 + int4_weight_params * 0.5) / 1_000_000.0
        hybrid_scaled_mb = (
            fp16_keep_params * 2.0 + int4_weight_params * (0.5 + scale_bytes_per_param)
        ) / 1_000_000.0

        return {
            "total_params": total_params,
            "infer_params": infer_params,
            "backbone_params": backbone_params,
            "encoder_params": encoder_params,
            "router_params": router_params,
            "deepstack_params": deepstack_params,
            "decoder_params": decoder_params,
            "mtp_params": mtp_params,
            "aux_params": aux_params,
            "int4_weight_params": int4_weight_params,
            "fp16_keep_params": fp16_keep_params,
            "fp16_mb": (infer_params * 2.0) / 1_000_000.0,
            "pure_int4_mb": int4_raw_mb,
            "pure_int4_with_scales_mb": int4_scaled_mb,
            "hybrid_fp16_int4_mb": hybrid_raw_mb,
            "hybrid_fp16_int4_with_scales_mb": hybrid_scaled_mb,
        }
