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

import bekern_full  # noqa: E402
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
    _SDPBackend,
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
    "FourierCoordMLP",
    "ChainedMTPHead",
    "BD3ReMDMRepairHead",
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
    "build_text_aux_targets",
    "sample_bd3_complementary_views",
    "compute_bd3_repair_loss",
    "warm_start_from_v265_checkpoint",
    "set_bekern_full_stage",
    "restore_bekern_full_stage1_frozen_slices",
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


def _apply_rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to a single tensor ``(B, H, L, D)`` (avoids double rotation in CLA consumers)."""
    half = x.shape[-1] // 2
    c = cos[..., :half]
    s = sin[..., :half]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)


_CUDA_SDPA_BACKENDS_CACHE: Dict[int, List[Any]] = {}
_DEVICE_VRAM_GB_CACHE: Dict[int, float] = {}


def _get_cuda_sdpa_backends(device: torch.device) -> List[Any]:
    idx = int(device.index) if device.index is not None else 0
    cached = _CUDA_SDPA_BACKENDS_CACHE.get(idx)
    if cached is not None:
        return cached
    backends = list(_CUDA_SDPA_BACKENDS)
    if (
        sys.platform.startswith("linux")
        and _SDPBackend is not None
        and hasattr(_SDPBackend, "FLASH_ATTENTION")
        and _SDPA_HAS_SET_PRIORITY
    ):
        try:
            major, _ = torch.cuda.get_device_capability(device)
            if major >= 8:
                backends = [
                    _SDPBackend.FLASH_ATTENTION,
                    _SDPBackend.EFFICIENT_ATTENTION,
                    _SDPBackend.MATH,
                ]
        except Exception:
            pass
    _CUDA_SDPA_BACKENDS_CACHE[idx] = backends
    return backends


def _get_device_vram_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    idx = int(device.index) if device.index is not None else 0
    vram_gb = _DEVICE_VRAM_GB_CACHE.get(idx)
    if vram_gb is None:
        try:
            props = torch.cuda.get_device_properties(device)
            vram_gb = float(props.total_memory) / (1024.0 ** 3)
        except Exception:
            vram_gb = 0.0
        _DEVICE_VRAM_GB_CACHE[idx] = vram_gb
    return vram_gb


def _get_grad_ckpt_thresholds(device: torch.device) -> Tuple[int, int, int]:
    """Return ``(global_thr, dec_full_thr, dec_ffn_thr)`` scaled by total GPU VRAM."""
    vram_gb = _get_device_vram_gb(device)
    if vram_gb >= 64.0:
        return (12288, 8192, 4096)
    if vram_gb >= 20.0:
        return (6144, 4096, 2048)
    return (2048, 1024, 512)


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """SDPA wrapper: FlashAttention-2 -> CUTLASS on Linux Ampere+, CUTLASS on Windows (avoids cuDNN 9.10 stalls)."""
    sdpa_kwargs: Dict[str, Any] = dict(attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal)
    if enable_gqa:
        sdpa_kwargs["enable_gqa"] = True
    if q.device.type == "cuda" and _sdpa_kernel is not None:
        backends = _get_cuda_sdpa_backends(q.device)
        if _SDPA_HAS_SET_PRIORITY:
            with _sdpa_kernel(backends, set_priority=True):
                return F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
        if attn_mask is None:
            with _sdpa_kernel(backends):
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
    pad_id: Optional[int] = None
    bos_id: Optional[int] = None
    eos_id: Optional[int] = None
    sys_id: Optional[int] = None
    music_vocab_end: Optional[int] = None

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

    # Iter 15 promoted upgrades (ALL default False/None/0 for 100% backward compatibility)
    dec_ffn_mult: Optional[float] = None
    use_yoco_cla: bool = False
    yoco_cross_banks: Tuple[int, ...] = (0, 3, 7)
    yoco_cross_kv_heads: Optional[int] = None
    yoco_self_kv_heads: Optional[int] = None
    use_resformer_v_res: bool = False
    use_diff_cross_attn: bool = False
    use_coordkey_ca: bool = False
    coord_dim: int = 16
    coord_mlp_hidden: int = 128
    coord_n_freqs: int = 16
    mode_gate_layers: int = 0
    mode_gate_rank: int = 32
    use_chained_mtp: bool = False
    text_aux_vocab_size: int = 0
    text_aux_loss_weight: float = 0.0
    use_quantvla_head_temp: bool = False
    use_bd3_repair_head: bool = False
    bd3_d_rep: int = 256
    bd3_n_heads: int = 4
    bd3_d_ff: int = 512
    bd3_n_layers: int = 2
    bd3_block_size: int = 32

    # Iter 16 bekern-full upgrades (ALL default False/0/0.75 for 100% backward compatibility)
    bekern_full: bool = False
    use_text_aux_head: bool = False
    text_vocab_size: int = 0
    mode_gate_conf_thresh: float = 0.75

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
        if self.yoco_cross_kv_heads is not None:
            if self.n_heads % int(self.yoco_cross_kv_heads) != 0:
                raise ValueError(
                    f"n_heads ({self.n_heads}) must be divisible by yoco_cross_kv_heads ({self.yoco_cross_kv_heads})"
                )
            self.yoco_cross_kv_heads = int(self.yoco_cross_kv_heads)
        if self.yoco_self_kv_heads is not None:
            if self.n_heads % int(self.yoco_self_kv_heads) != 0:
                raise ValueError(
                    f"n_heads ({self.n_heads}) must be divisible by yoco_self_kv_heads ({self.yoco_self_kv_heads})"
                )
            self.yoco_self_kv_heads = int(self.yoco_self_kv_heads)
        if self.use_resformer_v_res and not self.use_yoco_cla:
            raise ValueError("use_resformer_v_res=True requires use_yoco_cla=True")
        if self.use_yoco_cla:
            banks = tuple(sorted(set(int(b) for b in self.yoco_cross_banks if 0 <= int(b) < self.n_dec_layers)))
            if not banks or banks[0] != 0:
                banks = (0,) + tuple(b for b in banks if b != 0)
            self.yoco_cross_banks = banks
        if self.use_diff_cross_attn and (head_dim % 2 != 0):
            raise ValueError(f"head_dim ({head_dim}) must be even when use_diff_cross_attn=True")
        if self.use_diff_cross_attn and self.use_coordkey_ca and (int(self.coord_dim) % 2 != 0):
            raise ValueError(f"coord_dim ({self.coord_dim}) must be even when combining Diff + CoordKey")
        if self.use_bd3_repair_head and (int(self.bd3_d_rep) % int(self.bd3_n_heads) != 0):
            raise ValueError(
                f"bd3_d_rep ({self.bd3_d_rep}) must be divisible by bd3_n_heads ({self.bd3_n_heads})"
            )
        for _name in ("pad_id", "bos_id", "eos_id", "sys_id"):
            _val = getattr(self, _name)
            if _val is not None:
                _ival = int(_val)
                if not (0 <= _ival < self.vocab_size):
                    raise ValueError(f"{_name} ({_ival}) must be in [0, vocab_size={self.vocab_size})")
                setattr(self, _name, _ival)
        if self.music_vocab_end is not None:
            _mve = int(self.music_vocab_end)
            if not (1 <= _mve <= self.vocab_size):
                raise ValueError(
                    f"music_vocab_end ({_mve}) must be in [1, vocab_size={self.vocab_size}]"
                )
            self.music_vocab_end = _mve
        self.mode_gate_conf_thresh = float(max(0.0, min(1.0, self.mode_gate_conf_thresh)))

    def resolved_pad_id(self) -> int:
        """Return explicit ``pad_id`` if set, else bekern-full/legacy fallback."""
        if self.pad_id is not None:
            return int(self.pad_id)
        v = int(self.vocab_size)
        if self.bekern_full or v == bekern_full.FULL_VOCAB_SIZE:
            return bekern_full.PAD_ID
        return PAD_ID if v == VOCAB_SIZE else (v - 4)

    def resolved_bos_id(self) -> int:
        """Return explicit ``bos_id`` if set, else bekern-full/legacy fallback."""
        if self.bos_id is not None:
            return int(self.bos_id)
        v = int(self.vocab_size)
        if self.bekern_full or v == bekern_full.FULL_VOCAB_SIZE:
            return bekern_full.BOS_ID
        return BOS_ID if v == VOCAB_SIZE else (v - 3)

    def resolved_eos_id(self) -> int:
        """Return explicit ``eos_id`` if set, else bekern-full/legacy fallback."""
        if self.eos_id is not None:
            return int(self.eos_id)
        v = int(self.vocab_size)
        if self.bekern_full or v == bekern_full.FULL_VOCAB_SIZE:
            return bekern_full.EOS_ID
        return EOS_ID if v == VOCAB_SIZE else (v - 2)

    def resolved_sys_id(self) -> int:
        """Return explicit ``sys_id`` if set, else bekern-full/legacy fallback."""
        if self.sys_id is not None:
            return int(self.sys_id)
        v = int(self.vocab_size)
        if self.bekern_full or v == bekern_full.FULL_VOCAB_SIZE:
            return bekern_full.SYS_ID
        return SYS_ID if v == VOCAB_SIZE else (v - 1)

    def resolved_music_vocab_end(self) -> int:
        """Return exclusive upper bound for random music tokens in ``tok_dropout``."""
        if self.music_vocab_end is not None:
            return int(self.music_vocab_end)
        v = int(self.vocab_size)
        if self.bekern_full or v == bekern_full.FULL_VOCAB_SIZE:
            return bekern_full.MUSIC_VOCAB_END
        if v == VOCAB_SIZE:
            return len(data.VOCAB)
        pad_id = self.resolved_pad_id()
        if (v // 2) < pad_id <= v:
            return pad_id
        return max(1, v)

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

    @classmethod
    def pro200m(cls, **overrides: Any) -> "UniOMRv2Config":
        """d_model=768, ~195M infer params, ~120 MB Hybrid FP16/INT4-QAT (< 135 MB w/ scales)."""
        params: Dict[str, Any] = dict(
            d_model=768,
            n_heads=12,
            n_kv_heads=4,
            n_axial_layers=0,
            n_global_layers=8,
            n_dec_layers=12,
            ffn_mult=3.5,
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
    def iter15_pro200m(cls, **overrides: Any) -> "UniOMRv2Config":
        """Iter 15 flagship pro200m (d=768) with all 7+1 promoted upgrades enabled (< 135 MB INT4)."""
        params: Dict[str, Any] = dict(
            d_model=768,
            n_heads=12,
            n_kv_heads=6,
            n_axial_layers=0,
            n_global_layers=8,
            n_dec_layers=12,
            ffn_mult=3.5,
            dec_ffn_mult=3.5,
            axial_ffn_mult=2.0,
            backbone_channels=(64, 128, 240, 448),
            backbone_blocks=(1, 2, 3, 3),
            backbone_expand=2.0,
            router_hidden=64,
            max_mem_tokens=3072,
            mtp_k=4,
            # Promoted Iter 15 winners (frozen; new winners like n_register_tokens=8 go to Iter 16):
            use_yoco_cla=True,
            yoco_cross_banks=(0, 3, 7),
            yoco_cross_kv_heads=12,
            yoco_self_kv_heads=6,
            use_resformer_v_res=True,
            use_diff_cross_attn=True,
            use_coordkey_ca=True,
            coord_dim=16,
            coord_mlp_hidden=128,
            coord_n_freqs=16,
            mode_gate_layers=4,
            mode_gate_rank=32,
            use_chained_mtp=True,
            text_aux_vocab_size=129,
            text_aux_loss_weight=0.02,
            use_quantvla_head_temp=True,
            use_bd3_repair_head=True,
            bd3_d_rep=256,
            bd3_n_heads=4,
            bd3_d_ff=512,
            bd3_n_layers=2,
            bd3_block_size=32,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def iter16_pro200m(cls, **overrides: Any) -> "UniOMRv2Config":
        """Iter 16 flagship bekern-full pro200m (d=768, 8 registers, 446-vocab + 259-byte text head)."""
        params: Dict[str, Any] = dict(
            d_model=768,
            n_heads=12,
            n_kv_heads=6,
            n_axial_layers=0,
            n_global_layers=8,
            n_dec_layers=12,
            ffn_mult=3.5,
            dec_ffn_mult=3.5,
            axial_ffn_mult=2.0,
            backbone_channels=(64, 128, 240, 448),
            backbone_blocks=(1, 2, 3, 3),
            backbone_expand=2.0,
            router_hidden=64,
            max_mem_tokens=3072,
            mtp_k=4,
            # Promoted ov-17 winner:
            n_register_tokens=8,
            # Iter 15 promoted winners:
            use_yoco_cla=True,
            yoco_cross_banks=(0, 3, 7),
            yoco_cross_kv_heads=12,
            yoco_self_kv_heads=6,
            use_resformer_v_res=True,
            use_diff_cross_attn=True,
            use_coordkey_ca=True,
            coord_dim=16,
            coord_mlp_hidden=128,
            coord_n_freqs=16,
            mode_gate_layers=4,
            mode_gate_rank=32,
            use_chained_mtp=True,
            use_quantvla_head_temp=True,
            use_bd3_repair_head=True,
            bd3_d_rep=256,
            bd3_n_heads=4,
            bd3_d_ff=512,
            bd3_n_layers=2,
            bd3_block_size=32,
            # Iter 16 bekern-full 4-layer vocab + Layer 3 UTF-8 text head:
            bekern_full=True,
            vocab_size=bekern_full.FULL_VOCAB_SIZE,
            pad_id=bekern_full.PAD_ID,
            bos_id=bekern_full.BOS_ID,
            eos_id=bekern_full.EOS_ID,
            sys_id=bekern_full.SYS_ID,
            music_vocab_end=bekern_full.MUSIC_VOCAB_END,
            use_text_aux_head=True,
            text_vocab_size=bekern_full.TEXT_VOCAB_SIZE,
            text_aux_vocab_size=bekern_full.TEXT_VOCAB_SIZE,
            text_aux_loss_weight=0.02,
            mode_gate_conf_thresh=0.75,
        )
        params.update(overrides)
        return cls(**params)

    @classmethod
    def iter16_edge85m(cls, **overrides: Any) -> "UniOMRv2Config":
        """Iter 16 WebGPU bekern-full edge85m (d=512, 8 registers, 446-vocab + 259-byte text head, < 55 MB INT4)."""
        params: Dict[str, Any] = dict(
            d_model=512,
            n_heads=8,
            n_kv_heads=2,
            n_axial_layers=2,
            n_global_layers=4,
            n_dec_layers=10,
            ffn_mult=4.0,
            dec_ffn_mult=4.0,
            axial_ffn_mult=2.0,
            backbone_channels=(64, 128, 240, 448),
            backbone_blocks=(1, 2, 3, 3),
            backbone_expand=2.0,
            router_hidden=64,
            max_mem_tokens=3072,
            mtp_k=4,
            # Promoted ov-17 winner:
            n_register_tokens=8,
            # Iter 15 promoted winners (YOCO-CLA keeps KV-cache & INT4 size < 55 MB):
            use_yoco_cla=True,
            yoco_cross_banks=(0, 3, 7),
            yoco_cross_kv_heads=4,
            yoco_self_kv_heads=2,
            use_resformer_v_res=True,
            use_diff_cross_attn=True,
            use_coordkey_ca=True,
            coord_dim=16,
            coord_mlp_hidden=128,
            coord_n_freqs=16,
            mode_gate_layers=4,
            mode_gate_rank=32,
            use_chained_mtp=True,
            use_quantvla_head_temp=True,
            use_bd3_repair_head=True,
            bd3_d_rep=256,
            bd3_n_heads=4,
            bd3_d_ff=512,
            bd3_n_layers=2,
            bd3_block_size=32,
            # Iter 16 bekern-full 4-layer vocab + Layer 3 UTF-8 text head:
            bekern_full=True,
            vocab_size=bekern_full.FULL_VOCAB_SIZE,
            pad_id=bekern_full.PAD_ID,
            bos_id=bekern_full.BOS_ID,
            eos_id=bekern_full.EOS_ID,
            sys_id=bekern_full.SYS_ID,
            music_vocab_end=bekern_full.MUSIC_VOCAB_END,
            use_text_aux_head=True,
            text_vocab_size=bekern_full.TEXT_VOCAB_SIZE,
            text_aux_vocab_size=bekern_full.TEXT_VOCAB_SIZE,
            text_aux_loss_weight=0.02,
            mode_gate_conf_thresh=0.75,
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

        # Per-image adaptive dark-border thresholds from 90th-percentile luma (downsampled 4x for ~0 overhead).
        # Handles underexposed/dim aged paper (luma < 0.35) while floor >= 0.05 keeps all-black images dark.
        _, _, _lh, _lw = luma.shape
        luma_ds = F.avg_pool2d(luma, kernel_size=4, stride=4) if (_lh >= 4 and _lw >= 4) else luma
        luma_flat = luma_ds.flatten(1)
        n_ds = luma_flat.shape[1]
        if torch.onnx.is_in_onnx_export():
            k90_top = max(1, min(n_ds, n_ds - int(round(0.90 * max(0, n_ds - 1)))))
            p90_luma = torch.topk(luma_flat, k=k90_top, dim=1, largest=True, sorted=True).values[:, -1:].view(-1, 1, 1, 1)
        else:
            k90 = max(1, min(n_ds, int(round(0.90 * max(0, n_ds - 1))) + 1))
            p90_luma = torch.kthvalue(luma_flat, k=k90, dim=1).values.view(-1, 1, 1, 1)
        bright_thr = torch.clamp(0.85 * p90_luma, min=0.05, max=0.35)
        dark_thr = torch.clamp(0.65 * p90_luma, min=0.05, max=0.22)

        # Edge-connected dark-border mask: 2 cumsums (0 flips) — bit-identical to 4-edge scan.
        # Since (luma < dark_thr) implies bright_i == 0 (dark_thr <= bright_thr),
        # cs_y == cs_y[:, :, -1:, :] iff no bright pixel at y..H-1.
        bright_i = (luma >= bright_thr).to(torch.int32)
        cs_y = bright_i.cumsum(dim=2)
        cs_x = bright_i.cumsum(dim=3)
        edge_dark = (
            (cs_y == 0)
            | (cs_y == cs_y[:, :, -1:, :])
            | (cs_x == 0)
            | (cs_x == cs_x[:, :, :, -1:])
        )
        border_px = (edge_dark & (luma < dark_thr)).to(luma.dtype)

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
        _, _, H, W = local_ink.shape
        target_h, target_w = 16 * h_p, 16 * w_p
        if H == target_h and W == target_w:
            ink_max = F.max_pool2d(local_ink, kernel_size=16, stride=16)
            ink_avg = F.avg_pool2d(local_ink, kernel_size=16, stride=16)
            paper_grid_f = F.avg_pool2d(paper_px, kernel_size=16, stride=16)
        elif H <= target_h and W <= target_w:
            pad_b = target_h - H
            pad_r = target_w - W
            local_ink_pad = F.pad(local_ink, (0, pad_r, 0, pad_b), mode="constant", value=0.0)
            paper_px_pad = F.pad(paper_px, (0, pad_r, 0, pad_b), mode="constant", value=0.0)
            ink_max = F.max_pool2d(local_ink_pad, kernel_size=16, stride=16)
            ink_avg = F.avg_pool2d(local_ink_pad, kernel_size=16, stride=16)
            paper_grid_f = F.avg_pool2d(paper_px_pad, kernel_size=16, stride=16)
        else:
            ink_max = F.adaptive_max_pool2d(local_ink, (h_p, w_p))
            ink_avg = F.adaptive_avg_pool2d(local_ink, (h_p, w_p))
            paper_grid_f = F.adaptive_avg_pool2d(paper_px, (h_p, w_p))
        local_ink_grid = ink_max.mul_(0.5).add_(ink_avg, alpha=0.5)

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
        paper_mask_grid: Optional[torch.Tensor] = None
        if images is not None:
            local_ink_grid, paper_gate, paper_mask_grid = self._compute_ink_and_paper_gate(
                images, H_p, W_p
            )
            local_ink_grid = local_ink_grid.to(saliency_logits.dtype)
            paper_gate = paper_gate.to(saliency_logits.dtype)
            saliency_prob = saliency_prob * paper_gate
            flat_ink: Optional[torch.Tensor] = None
            if getattr(self, "adaptive_ink_target", False):
                flat_ink = local_ink_grid.flatten(1).float()
                flat_paper = paper_mask_grid.flatten(1)
                if torch.onnx.is_in_onnx_export():
                    k30 = max(1, min(N, int(round(0.30 * max(0, N - 1))) + 1))
                    bg_p30 = torch.topk(flat_ink, k=k30, dim=1, largest=False, sorted=True).values[:, -1:]
                else:
                    # Sync-free exact linear-interpolation 30th percentile over valid paper tokens
                    sorted_ink, _ = flat_ink.masked_fill(~flat_paper, float("inf")).sort(dim=1)
                    n_valid = flat_paper.sum(dim=1)
                    pos = (n_valid - 1).clamp(min=0).to(torch.float64) * 0.30
                    lo = pos.floor().to(torch.long)
                    hi = pos.ceil().to(torch.long)
                    frac = (pos - lo.to(torch.float64)).to(flat_ink.dtype).unsqueeze(1)
                    v_lo = sorted_ink.gather(1, lo.unsqueeze(1))
                    v_hi = sorted_ink.gather(1, hi.unsqueeze(1))
                    q30 = v_lo + frac * (v_hi - v_lo)
                    bg_p30 = torch.where(n_valid.unsqueeze(1) > 0, q30, torch.zeros_like(q30))
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
                flat_f = flat_ink if flat_ink is not None else local_ink_grid.flatten(1).float()
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
                    elif paper_mask_grid is not None:
                        # Sync-free exact 10th percentile over valid paper tokens only so dark
                        # exterior borders cannot collapse p10 and bloat k_dyn.
                        sf = score_flat.float()
                        flat_paper_p10 = paper_mask_grid.flatten(1)
                        sorted_sc, _ = sf.masked_fill(~flat_paper_p10, float("inf")).sort(dim=1)
                        n_valid_p10 = flat_paper_p10.sum(dim=1)
                        pos10 = (n_valid_p10 - 1).clamp(min=0).to(torch.float64) * 0.10
                        lo10 = pos10.floor().to(torch.long)
                        hi10 = pos10.ceil().to(torch.long)
                        frac10 = (pos10 - lo10.to(torch.float64)).to(sf.dtype).unsqueeze(1)
                        v_lo10 = sorted_sc.gather(1, lo10.unsqueeze(1))
                        v_hi10 = sorted_sc.gather(1, hi10.unsqueeze(1))
                        q10_masked = v_lo10 + frac10 * (v_hi10 - v_lo10)
                        q10_fallback = torch.quantile(sf, 0.10, dim=1, keepdim=True)
                        p10 = torch.where(n_valid_p10.unsqueeze(1) > 0, q10_masked, q10_fallback).to(score_flat.dtype)
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

        use_ckpt = self.training and self.cfg.grad_checkpoint and torch.is_grad_enabled()
        if len(self.axial_layers) > 0:
            row_cos, row_sin = self.axial_rope(W_p, device=x.device, dtype=compute_dtype)
            col_cos, col_sin = self.axial_rope(H_p, device=x.device, dtype=compute_dtype)
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
        global_ckpt_thr = _get_grad_ckpt_thresholds(mem.device)[0]
        for i, layer in enumerate(self.global_layers):
            if use_ckpt and (mem.shape[0] * mem.shape[1] > global_ckpt_thr):
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
# 3b. Iter 15 Modules: FourierCoordMLP, ChainedMTPHead, BD3ReMDMRepairHead
# ---------------------------------------------------------------------------

class FourierCoordMLP(nn.Module):
    """Shared 2D Fourier coordinate encoder: (y_norm, x_norm) -> (2 + 4*n_freqs) -> hidden -> out."""

    def __init__(self, n_freqs: int = 16, hidden_dim: int = 128, out_dim: int = 128) -> None:
        super().__init__()
        self.n_freqs = int(n_freqs)
        in_dim = 2 + 4 * self.n_freqs
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
        """Return shared spatial coordinate features ``(B, Lk, out_dim)``."""
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
        fy = y_norm.unsqueeze(-1) * f
        fx = x_norm.unsqueeze(-1) * f
        feats = torch.cat(
            [y_norm.unsqueeze(-1), x_norm.unsqueeze(-1), fy.sin(), fy.cos(), fx.sin(), fx.cos()],
            dim=-1,
        ).to(dtype)
        return self.fc2(F.silu(self.fc1(feats)))


class ChainedMTPHead(nn.Module):
    """DeepSeek-V3 sequential Chained MTP projection:
    ``h_k = h_{k-1} + SiLU(W_k([RMSNorm(h_{k-1}); RMSNorm(tok_emb(y_{t+k}))]))``.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm_h = RMSNorm(d_model)
        self.norm_e = RMSNorm(d_model)
        self.proj = nn.Linear(2 * d_model, d_model, bias=False)

    def forward(self, h_prev: torch.Tensor, next_tok_emb: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([self.norm_h(h_prev), self.norm_e(next_tok_emb)], dim=-1)
        return h_prev + F.silu(self.proj(fused))


_TEXT_AUX_PREFIXES: Tuple[str, ...] = (
    "dyn-",
    "hairpin-",
    "art-",
    "clef-",
    "time-",
    "key",
    "ottava-",
    "volta-",
)


_TEXT_AUX_LUT_CACHE: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}


def _get_text_aux_lut(vocab_size: int, text_vocab_size: int, device: torch.device) -> torch.Tensor:
    key = (int(vocab_size), int(text_vocab_size), device)
    lut = _TEXT_AUX_LUT_CACHE.get(key)
    if lut is not None:
        return lut
    other_id = max(0, int(text_vocab_size) - 1)
    lut = torch.full((int(vocab_size),), other_id, dtype=torch.long)
    if other_id > 0:
        idx = 0
        for tid in sorted(UNI_ID_TO_TOKEN.keys()):
            tok = UNI_ID_TO_TOKEN[tid]
            if 0 <= tid < int(vocab_size) and any(tok.startswith(p) for p in _TEXT_AUX_PREFIXES):
                lut[tid] = min(other_id - 1, idx)
                idx += 1
    lut = lut.to(device)
    _TEXT_AUX_LUT_CACHE[key] = lut
    return lut


def build_text_aux_targets(
    targets: torch.Tensor,
    vocab_size: int,
    pad_id: int = PAD_ID,
    text_vocab_size: int = 129,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map structural/expressive text-like tokens into ``0..text_vocab_size-2`` and others to ``text_vocab_size-1``."""
    device = targets.device
    other_id = max(0, int(text_vocab_size) - 1)
    lut = _get_text_aux_lut(int(vocab_size), int(text_vocab_size), device)
    safe_t = targets.clamp(0, vocab_size - 1)
    text_tgt = lut[safe_t]
    valid = targets != pad_id
    text_tgt = torch.where(valid, text_tgt, torch.full_like(text_tgt, -100))
    is_text = (text_tgt >= 0) & (text_tgt < other_id)
    return text_tgt, is_text


class _BD3RepairLayer(nn.Module):
    """Lightweight bidirectional intra-cell self-attn + visual cross-attn + SwiGLU."""

    def __init__(self, d_rep: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_rep = int(d_rep)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_rep // self.n_heads

        self.norm1 = RMSNorm(d_rep)
        self.qkv_proj = nn.Linear(d_rep, 3 * d_rep, bias=False)
        self.sa_o_proj = nn.Linear(d_rep, d_rep, bias=False)
        self.sa_q_norm = RMSNorm(self.head_dim)
        self.sa_k_norm = RMSNorm(self.head_dim)

        self.norm2 = RMSNorm(d_rep)
        self.ca_q_proj = nn.Linear(d_rep, d_rep, bias=False)
        self.ca_kv_proj = nn.Linear(d_rep, 2 * d_rep, bias=False)
        self.ca_o_proj = nn.Linear(d_rep, d_rep, bias=False)
        self.ca_q_norm = RMSNorm(self.head_dim)
        self.ca_k_norm = RMSNorm(self.head_dim)

        self.norm3 = RMSNorm(d_rep)
        self.ffn = SwiGLU(d_rep, d_ff, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mem_rep: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        block_size: int = 32,
        block_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # 1. Bidirectional intra-cell (block-local) self-attention (BD3-LM style)
        h = self.norm1(x)
        pad_t = (block_size - (T % block_size)) % block_size
        h_pad = F.pad(h, (0, 0, 0, pad_t)) if pad_t > 0 else h

        T_pad = h_pad.shape[1]
        n_blk = T_pad // block_size
        h_blk = h_pad.view(B * n_blk, block_size, D)
        q, k, v = self.qkv_proj(h_blk).chunk(3, dim=-1)
        q = self.sa_q_norm(q.view(B * n_blk, block_size, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.sa_k_norm(k.view(B * n_blk, block_size, self.n_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B * n_blk, block_size, self.n_heads, self.head_dim).transpose(1, 2)

        c_blk = rope_cos[:, :, :block_size, :]
        s_blk = rope_sin[:, :, :block_size, :]
        q, k = apply_rope(q, k, c_blk, s_blk)

        sa_out = _sdpa(q, k, v, attn_mask=block_attn_mask, is_causal=False, enable_gqa=False)
        sa_out = sa_out.transpose(1, 2).reshape(B, T_pad, D)[:, :T, :]
        x = x + self.drop(self.sa_o_proj(sa_out))

        # 2. Visual cross-attention to projected encoder memory
        h2 = self.norm2(x)
        Lk = mem_rep.shape[1]
        cq = self.ca_q_norm(self.ca_q_proj(h2).view(B, T, self.n_heads, self.head_dim)).transpose(1, 2)
        ck, cv = self.ca_kv_proj(mem_rep).chunk(2, dim=-1)
        ck = self.ca_k_norm(ck.view(B, Lk, self.n_heads, self.head_dim)).transpose(1, 2)
        cv = cv.view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        ca_out = _sdpa(cq, ck, cv, attn_mask=None, is_causal=False, enable_gqa=False)
        ca_out = ca_out.transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.ca_o_proj(ca_out))

        # 3. SwiGLU FFN
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x


class BD3ReMDMRepairHead(nn.Module):
    """Lightweight 2-layer bidirectional intra-cell masked-diffusion / remasking repair head.

    Shares ``model.tok_emb`` and ``model.lm_head`` (zero extra vocab table params).
    """

    def __init__(
        self,
        d_model: int,
        d_rep: int = 256,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 2,
        block_size: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_rep = int(d_rep)
        self.block_size = int(block_size)
        head_dim = self.d_rep // int(n_heads)

        self.in_proj = nn.Linear(self.d_model, self.d_rep, bias=False)
        self.mem_proj = nn.Linear(self.d_model, self.d_rep, bias=False)
        self.mask_emb = nn.Parameter(torch.randn(1, 1, self.d_rep) * 0.02)
        self.rope1d = _CachedRoPE1D(head_dim=head_dim, base=10000.0)
        self.layers = nn.ModuleList([
            _BD3RepairLayer(self.d_rep, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.out_norm = RMSNorm(self.d_rep)
        self.out_proj = nn.Linear(self.d_rep, self.d_model, bias=False)

    def forward(
        self,
        tok_embeds: torch.Tensor,
        mask_flags: torch.Tensor,
        mem_final: torch.Tensor,
        lm_head: nn.Linear,
        key_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        compute_dtype = _active_compute_dtype(tok_embeds)
        x = self.in_proj(tok_embeds.to(compute_dtype))
        m_emb = self.mask_emb.to(dtype=x.dtype).expand_as(x)
        x = torch.where(mask_flags.unsqueeze(-1), m_emb, x)

        mem_rep = self.mem_proj(mem_final.to(compute_dtype))
        B, T, _ = x.shape
        cos1d, sin1d = self.rope1d(self.block_size, device=x.device, dtype=x.dtype)
        cos1d = cos1d[:, :, : self.block_size, :]
        sin1d = sin1d[:, :, : self.block_size, :]

        pad_t = (self.block_size - (T % self.block_size)) % self.block_size
        block_attn_mask: Optional[torch.Tensor] = None
        if key_valid_mask is not None or pad_t > 0:
            if key_valid_mask is None:
                kv_mask = torch.ones((B, T), device=x.device, dtype=torch.bool)
            else:
                kv_mask = key_valid_mask.to(device=x.device, dtype=torch.bool)
            if pad_t > 0:
                kv_mask = F.pad(kv_mask, (0, pad_t), value=False)
            n_blk = (T + pad_t) // self.block_size
            kv_blk = kv_mask.view(B * n_blk, self.block_size)
            all_false = ~kv_blk.any(dim=1, keepdim=True)
            first_slot = torch.zeros((1, self.block_size), device=x.device, dtype=torch.bool)
            first_slot[:, 0] = True
            kv_blk = kv_blk | (all_false & first_slot)
            block_attn_mask = kv_blk.view(B * n_blk, 1, 1, self.block_size)

        for layer in self.layers:
            x = layer(
                x,
                mem_rep,
                cos1d,
                sin1d,
                block_size=self.block_size,
                block_attn_mask=block_attn_mask,
            )

        h_out = self.out_proj(self.out_norm(x))
        return lm_head(h_out)


def sample_bd3_complementary_views(
    tgt_y: torch.Tensor,
    vocab_size: int,
    pad_id: int = PAD_ID,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
    n_views: int = 2,
    music_vocab_end: Optional[int] = None,
    bekern_full_mode: bool = False,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build complementary masked/corrupted views (15-40% mask + 10% single-token corruption).

    When ``bekern_full_mode=True`` (Mitigation #7), ``valid`` is strictly intersected
    with ``bekern_full.compute_is_music_body_mask(tgt_y)`` so ``<bos>``, ``<schema:*>``,
    ``<sys>``, and Layer 3 text spans are never masked or corrupted by BD3-LM.
    """
    valid = (tgt_y != pad_id) & (tgt_y != bos_id) & (tgt_y != eos_id)
    if bekern_full_mode:
        body_mask = bekern_full.compute_is_music_body_mask(tgt_y).to(device=tgt_y.device, dtype=torch.bool)
        valid = valid & body_mask
    p_mask = 0.15 + 0.25 * float(torch.rand(()))
    u = torch.rand(tgt_y.shape, device=tgt_y.device)
    hi_vocab = (
        max(1, min(int(vocab_size), int(music_vocab_end)))
        if music_vocab_end is not None
        else max(1, int(vocab_size))
    )
    lo_vocab = 0

    n_v = max(1, int(n_views))
    masks: List[torch.Tensor] = [(u < p_mask) & valid]
    if n_v >= 2:
        masks.append(((1.0 - u) < p_mask) & valid)
    for _ in range(2, n_v):
        masks.append((torch.rand(tgt_y.shape, device=tgt_y.device) < p_mask) & valid)

    views: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    valid_body_pool: Optional[torch.Tensor] = (
        torch.tensor(bekern_full.VALID_MUSIC_BODY_TOKEN_IDS, device=tgt_y.device, dtype=tgt_y.dtype)
        if bekern_full_mode
        else None
    )
    for m in masks:
        corrupt_unmasked = (torch.rand(tgt_y.shape, device=tgt_y.device) < 0.10) & valid & (~m)
        if valid_body_pool is not None and valid_body_pool.numel() > 0:
            rand_idx = torch.randint(0, int(valid_body_pool.numel()), tgt_y.shape, device=tgt_y.device)
            rand_toks = valid_body_pool[rand_idx]
        else:
            rand_toks = torch.randint(lo_vocab, hi_vocab, tgt_y.shape, device=tgt_y.device, dtype=tgt_y.dtype)
        corrupt_ids = torch.where(corrupt_unmasked, rand_toks, tgt_y)
        supervise = m | corrupt_unmasked
        has_sup = supervise.any()
        supervise = torch.where(has_sup, supervise, valid)
        m = torch.where(has_sup, m, valid)
        views.append((corrupt_ids, m, supervise))
    return views


def compute_bd3_repair_loss(
    model: "UniOMRv2Net",
    mem_final: torch.Tensor,
    tgt_y: torch.Tensor,
    n_views: int = 2,
    is_v1_schema: Optional[torch.Tensor] = None,
    v2a_token_ids: Optional[Sequence[int]] = None,
    has_v1_schema: Optional[bool] = None,
    layer_masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Complementary masked-diffusion + single-token corruption repair loss."""
    if model.bd3_repair_head is None:
        return mem_final.new_zeros(())
    pad_id = model.cfg.resolved_pad_id()
    bos_id = model.cfg.resolved_bos_id()
    eos_id = model.cfg.resolved_eos_id()
    views = sample_bd3_complementary_views(
        tgt_y,
        vocab_size=int(model.cfg.vocab_size),
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        n_views=n_views,
        music_vocab_end=model.cfg.resolved_music_vocab_end(),
        bekern_full_mode=bool(getattr(model.cfg, "bekern_full", False)),
    )
    layer_suppress_mask: Optional[torch.Tensor] = None
    layer_target_mask: Optional[torch.Tensor] = None
    if layer_masks is not None:
        layer_target_mask, layer_suppress_mask = bekern_full.build_layer_ce_masks(
            tgt_y, layer_masks, vocab_size=int(model.cfg.vocab_size)
        )
    v1_v2a_mask: Optional[torch.Tensor] = None
    if (
        layer_suppress_mask is None
        and is_v1_schema is not None
        and v2a_token_ids is not None
        and (has_v1_schema is not False)
    ):
        v2a_idx = (
            v2a_token_ids.to(device=tgt_y.device, dtype=torch.long)
            if isinstance(v2a_token_ids, torch.Tensor)
            else torch.as_tensor(list(v2a_token_ids), device=tgt_y.device, dtype=torch.long)
        )
        if v2a_idx.numel() > 0:
            is_v1_bool = is_v1_schema.to(device=tgt_y.device, dtype=torch.bool)
            v2a_col_mask = torch.zeros(int(model.cfg.vocab_size), device=tgt_y.device, dtype=torch.bool)
            v2a_col_mask[v2a_idx] = True
            v1_v2a_mask = is_v1_bool[:, None, None] & v2a_col_mask[None, None, :]
    total_loss = torch.zeros((), device=tgt_y.device, dtype=torch.float32)
    safe_targets = tgt_y.clamp(0, int(model.cfg.vocab_size) - 1).reshape(-1)
    key_valid_mask = tgt_y != pad_id
    for corrupt_ids, mask_flags, supervise in views:
        tok_emb = model.tok_emb(corrupt_ids)
        logits = model.bd3_repair_head(
            tok_emb,
            mask_flags,
            mem_final,
            model.lm_head,
            key_valid_mask=key_valid_mask,
        )
        eff_sup = supervise & layer_target_mask if layer_target_mask is not None else supervise
        if not bool(eff_sup.any()):
            total_loss = total_loss + (logits.sum() * 0.0).float()
            continue
        if layer_suppress_mask is not None:
            logits = logits.masked_fill(layer_suppress_mask.unsqueeze(1), float("-inf"))
        elif v1_v2a_mask is not None:
            logits = logits.masked_fill(v1_v2a_mask, -1e4)
        eff_sup_flat = eff_sup.reshape(-1)
        masked_targets = torch.where(
            eff_sup_flat,
            safe_targets,
            torch.full_like(safe_targets, -100),
        )
        per_tok = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            masked_targets,
            ignore_index=-100,
            reduction="none",
        )
        per_tok_safe = torch.where(
            eff_sup_flat, per_tok.float(), torch.zeros_like(per_tok, dtype=torch.float32)
        )
        loss_v = per_tok_safe.sum() / eff_sup_flat.float().sum().clamp(min=1.0)
        total_loss = total_loss + loss_v
    return (total_loss / float(len(views))).to(mem_final.dtype)


# ---------------------------------------------------------------------------
# 4. Grouped-Query Attention (GQA) Decoder (8 Q-heads / 2 KV-heads -> 4x smaller KV cache)
# ---------------------------------------------------------------------------

def _diff_lambda_init(layer_idx: int) -> float:
    return 0.8 - 0.6 * math.exp(-0.3 * float(layer_idx))


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention for decoder self-attention & cross-attention.

    Supports legacy stock GQA (default, zero extra params/buffers) as well as
    Iter 15 native upgrades with zero wrapper indirection:
      - ``owns_kv=False`` (YOCO / CLA consumer reusing shared K/V bank)
      - ``use_v_res=True`` (ResFormer value residual on KV producers)
      - ``use_diff_attn=True`` (Differential Attention, Ye et al. 2024)
      - ``use_coordkey=True`` (Decoupled 2D Fourier spatial Q/K subspace;
        combines with ``use_diff_attn`` via Split-Spatial Geometry ``32+8=40``)
      - ``use_mode_gate=True`` (Modality-Gated FiLM + low-rank Q modulation)
      - ``use_head_temp=True`` (QuantVLA per-head temperature buffer ``(1, H, 1, 1)``)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        use_native_gqa: bool = True,
        use_rope: bool = True,
        *,
        owns_kv: bool = True,
        use_v_res: bool = False,
        v_res_init: float = -2.0,
        use_diff_attn: bool = False,
        layer_idx: int = 0,
        use_coordkey: bool = False,
        coord_in_dim: int = 128,
        coord_dim: int = 16,
        use_mode_gate: bool = False,
        mode_gate_rank: int = 32,
        use_head_temp: bool = False,
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
        self.owns_kv = bool(owns_kv)
        self.use_v_res = bool(use_v_res and owns_kv)
        self.use_diff_attn = bool(use_diff_attn)
        self.layer_idx = int(layer_idx)
        self.use_coordkey = bool(use_coordkey)
        self.coord_dim = int(coord_dim) if self.use_coordkey else 0
        self.use_mode_gate = bool(use_mode_gate)
        self.use_head_temp = bool(use_head_temp)

        if self.use_diff_attn:
            if self.head_dim % 2 != 0:
                raise ValueError(f"head_dim ({self.head_dim}) must be even for DiffAttention")
            self.half_dim = self.head_dim // 2
            self.lambda_init = float(_diff_lambda_init(self.layer_idx))
            if self.use_coordkey and (self.coord_dim % 2 != 0):
                raise ValueError(f"coord_dim ({self.coord_dim}) must be even for Diff+CoordKey")
            self.half_coord_dim = self.coord_dim // 2 if self.use_coordkey else 0
        else:
            self.half_dim = 0
            self.lambda_init = 0.0
            self.half_coord_dim = 0

        # Core projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        if self.owns_kv:
            self.k_proj: Optional[nn.Linear] = nn.Linear(d_model, self.kv_dim, bias=False)
            self.v_proj: Optional[nn.Linear] = nn.Linear(d_model, self.kv_dim, bias=False)
        else:
            self.k_proj = None
            self.v_proj = None
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Q/K content norms (legacy path creates exact q_norm/k_norm)
        if self.use_diff_attn:
            self.q_norm: Optional[RMSNorm] = None
            self.k_norm: Optional[RMSNorm] = None
            self.q1_norm: Optional[RMSNorm] = RMSNorm(self.half_dim)
            self.q2_norm: Optional[RMSNorm] = RMSNorm(self.half_dim)
            if self.owns_kv:
                self.k1_norm: Optional[RMSNorm] = RMSNorm(self.half_dim)
                self.k2_norm: Optional[RMSNorm] = RMSNorm(self.half_dim)
            else:
                self.k1_norm = None
                self.k2_norm = None
            self.lambda_q1 = nn.Parameter(torch.zeros(self.half_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1))
            self.lambda_k1 = nn.Parameter(torch.zeros(self.half_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1))
            self.lambda_q2 = nn.Parameter(torch.zeros(self.half_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1))
            self.lambda_k2 = nn.Parameter(torch.zeros(self.half_dim, dtype=torch.float32).normal_(mean=0.0, std=0.1))
            self.subln: Optional[RMSNorm] = RMSNorm(self.head_dim)
        else:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim) if self.owns_kv else None

        # ResFormer value residual
        if self.use_v_res:
            self.v_res_alpha: Optional[nn.Parameter] = nn.Parameter(
                torch.full((1, n_kv_heads, 1, 1), float(v_res_init))
            )
        else:
            self.v_res_alpha = None

        # CoordKey-CA spatial projections & norms
        if self.use_coordkey:
            self.q_coord_proj: Optional[nn.Linear] = nn.Linear(d_model, n_heads * self.coord_dim, bias=False)
            self.k_coord_proj: Optional[nn.Linear] = (
                nn.Linear(coord_in_dim, n_kv_heads * self.coord_dim, bias=False) if self.owns_kv else None
            )
            if self.use_diff_attn:
                self.q_s1_norm: Optional[RMSNorm] = RMSNorm(self.half_coord_dim)
                self.q_s2_norm: Optional[RMSNorm] = RMSNorm(self.half_coord_dim)
                if self.owns_kv:
                    self.k_s1_norm: Optional[RMSNorm] = RMSNorm(self.half_coord_dim)
                    self.k_s2_norm: Optional[RMSNorm] = RMSNorm(self.half_coord_dim)
                else:
                    self.k_s1_norm = None
                    self.k_s2_norm = None
            else:
                self.q_coord_norm: Optional[RMSNorm] = RMSNorm(self.coord_dim)
                self.k_coord_norm: Optional[RMSNorm] = RMSNorm(self.coord_dim) if self.owns_kv else None

        # ModeGate (FiLM + low-rank Q modulation BEFORE q_proj and q_coord_proj)
        if self.use_mode_gate:
            self.mode_gate = nn.Linear(d_model, 1, bias=True)
            nn.init.zeros_(self.mode_gate.weight)
            nn.init.constant_(self.mode_gate.bias, -2.0)
            self.film_gamma = nn.Parameter(torch.zeros(1, 1, d_model))
            self.film_beta = nn.Parameter(torch.zeros(1, 1, d_model))
            self.q_down = nn.Linear(d_model, int(mode_gate_rank), bias=False)
            self.q_up = nn.Linear(int(mode_gate_rank), d_model, bias=False)
            nn.init.zeros_(self.q_up.weight)

        # QuantVLA per-head attention temperature buffer (registered ONLY when enabled)
        if self.use_head_temp:
            self.register_buffer("head_temp", torch.ones(1, n_heads, 1, 1))
        else:
            self.head_temp: Optional[torch.Tensor] = None

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        B, n_kv, L, hd = x.shape
        return x.unsqueeze(2).expand(B, n_kv, self.n_rep, L, hd).reshape(B, self.n_heads, L, hd)

    def _compute_lambda(self, dtype: torch.dtype) -> torch.Tensor:
        lam_1 = torch.exp(torch.sum(self.lambda_q1.float() * self.lambda_k1.float()).clamp(-20.0, 20.0))
        lam_2 = torch.exp(torch.sum(self.lambda_q2.float() * self.lambda_k2.float()).clamp(-20.0, 20.0))
        return (lam_1 - lam_2 + self.lambda_init).to(dtype)

    def compute_kv(
        self,
        kv_in: torch.Tensor,
        *,
        coord_feat: Optional[torch.Tensor] = None,
        v0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project K (packed if Diff/CoordKey) and V (with optional ResFormer v_res)."""
        assert self.owns_kv and self.k_proj is not None and self.v_proj is not None
        B, Lk, _ = kv_in.shape
        v = self.v_proj(kv_in).view(B, Lk, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.use_v_res and v0 is not None and self.v_res_alpha is not None:
            g = torch.sigmoid(self.v_res_alpha).to(v.dtype)
            v = (1.0 - g) * v + g * v0

        if self.use_diff_attn:
            assert self.k1_norm is not None and self.k2_norm is not None
            k_c_raw = self.k_proj(kv_in).view(B, Lk, self.n_kv_heads, 2, self.half_dim)
            k1 = self.k1_norm(k_c_raw[:, :, :, 0, :]).transpose(1, 2)
            k2 = self.k2_norm(k_c_raw[:, :, :, 1, :]).transpose(1, 2)
            if self.use_coordkey:
                if coord_feat is None:
                    raise RuntimeError("coord_feat must be provided for CoordKey cross-attention")
                assert self.k_coord_proj is not None and self.k_s1_norm is not None and self.k_s2_norm is not None
                k_s_raw = self.k_coord_proj(coord_feat.to(kv_in.dtype)).view(
                    B, Lk, self.n_kv_heads, 2, self.half_coord_dim
                )
                k_s1 = self.k_s1_norm(k_s_raw[:, :, :, 0, :]).transpose(1, 2)
                k_s2 = self.k_s2_norm(k_s_raw[:, :, :, 1, :]).transpose(1, 2)
                k1 = torch.cat([k1, k_s1], dim=-1)  # (B, n_kv, Lk, half_dim + half_coord_dim = 40)
                k2 = torch.cat([k2, k_s2], dim=-1)  # (B, n_kv, Lk, 40)
            k_out = torch.cat([k1, k2], dim=1)      # (B, 2*n_kv, Lk, 40 or 32)
        else:
            assert self.k_norm is not None
            k_c = self.k_norm(self.k_proj(kv_in).view(B, Lk, self.n_kv_heads, self.head_dim)).transpose(1, 2)
            if self.use_coordkey:
                if coord_feat is None:
                    raise RuntimeError("coord_feat must be provided for CoordKey cross-attention")
                assert self.k_coord_proj is not None and self.k_coord_norm is not None
                k_s = self.k_coord_norm(
                    self.k_coord_proj(coord_feat.to(kv_in.dtype)).view(B, Lk, self.n_kv_heads, self.coord_dim)
                ).transpose(1, 2)
                k_out = torch.cat([k_c, k_s], dim=-1)
            else:
                k_out = k_c
        return k_out, v

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
        *,
        coord_feat: Optional[torch.Tensor] = None,
        shared_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        v0: Optional[torch.Tensor] = None,
        return_v_cur: bool = False,
    ) -> Any:
        B, Lq, _ = q_in.shape

        # ModeGate modulates q_in -> q_mod BEFORE both q_proj and q_coord_proj
        if self.use_mode_gate:
            alpha = torch.sigmoid(self.mode_gate(q_in)).to(q_in.dtype)
            gamma = self.film_gamma.to(q_in.dtype)
            beta = self.film_beta.to(q_in.dtype)
            q_delta = self.q_up(F.silu(self.q_down(q_in)))
            q_mod = q_in * (1.0 + alpha * gamma) + alpha * (beta + q_delta)
        else:
            q_mod = q_in

        v_raw_cur: Optional[torch.Tensor] = None
        in_onnx = torch.onnx.is_in_onnx_export()

        # --- Special path: Differential Attention (with or without CoordKey / YOCO) ---
        if self.use_diff_attn:
            assert not is_causal, "Differential Attention does not support causal masking"
            assert self.q1_norm is not None and self.q2_norm is not None and self.subln is not None
            if self.use_coordkey:
                assert self.q_coord_proj is not None and self.q_s1_norm is not None and self.q_s2_norm is not None
                can_fuse_qc = (
                    (not in_onnx)
                    and (self.q_proj.bias is None and self.q_coord_proj.bias is None)
                    and (not _is_qat_linear_active(self.q_proj))
                    and (not _is_qat_linear_active(self.q_coord_proj))
                )
                if can_fuse_qc:
                    d_qc = self.n_heads * self.head_dim
                    d_qs = self.n_heads * self.coord_dim
                    w_q = torch.cat((self.q_proj.weight, self.q_coord_proj.weight), dim=0)
                    q_c_flat, q_s_flat = F.linear(q_mod, w_q).split((d_qc, d_qs), dim=-1)
                    q_c_raw = q_c_flat.view(B, Lq, self.n_heads, 2, self.half_dim)
                    q_s_raw = q_s_flat.view(B, Lq, self.n_heads, 2, self.half_coord_dim)
                else:
                    q_c_raw = self.q_proj(q_mod).view(B, Lq, self.n_heads, 2, self.half_dim)
                    q_s_raw = self.q_coord_proj(q_mod).view(B, Lq, self.n_heads, 2, self.half_coord_dim)
                q1 = self.q1_norm(q_c_raw[:, :, :, 0, :]).transpose(1, 2)
                q2 = self.q2_norm(q_c_raw[:, :, :, 1, :]).transpose(1, 2)
                q_s1 = self.q_s1_norm(q_s_raw[:, :, :, 0, :]).transpose(1, 2)
                q_s2 = self.q_s2_norm(q_s_raw[:, :, :, 1, :]).transpose(1, 2)
                q1 = torch.cat([q1, q_s1], dim=-1)  # (B, H, Lq, 40)
                q2 = torch.cat([q2, q_s2], dim=-1)  # (B, H, Lq, 40)
            else:
                q_c_raw = self.q_proj(q_mod).view(B, Lq, self.n_heads, 2, self.half_dim)
                q1 = self.q1_norm(q_c_raw[:, :, :, 0, :]).transpose(1, 2)
                q2 = self.q2_norm(q_c_raw[:, :, :, 1, :]).transpose(1, 2)

            if self.head_temp is not None:
                ht = self.head_temp.to(q1.dtype)
                q1 = q1 * ht
                q2 = q2 * ht

            if shared_kv is not None:
                k_packed, v = shared_kv
            elif is_cross and past_kv is not None:
                k_packed, v = past_kv
            else:
                if kv_in is None:
                    raise ValueError("kv_in must be provided when past_kv and shared_kv are None")
                k_packed, v = self.compute_kv(kv_in, coord_feat=coord_feat, v0=v0)
                v_raw_cur = v

            present_kv = (k_packed, v)
            # Batch (q1, q2) and (k1, k2) along head dim -> 1 CUTLASS SDPA call instead of 2
            if self.n_rep == 1:
                k12_exp = k_packed
            else:
                Bk, nk2, Lk_p, dk = k_packed.shape
                k12_exp = (
                    k_packed.unsqueeze(2)
                    .expand(Bk, nk2, self.n_rep, Lk_p, dk)
                    .reshape(Bk, nk2 * self.n_rep, Lk_p, dk)
                )
            q12 = torch.cat([q1, q2], dim=1)
            v_exp = self._expand_kv(v)
            v12 = v_exp.repeat(1, 2, 1, 1)
            attn_mask_12 = (
                attn_mask.repeat(1, 2, 1, 1)
                if (attn_mask is not None and attn_mask.ndim == 4 and attn_mask.shape[1] == self.n_heads)
                else attn_mask
            )
            attn12 = _sdpa(q12, k12_exp, v12, attn_mask=attn_mask_12, is_causal=False, enable_gqa=False)
            attn1, attn2 = attn12.chunk(2, dim=1)
            lam = self._compute_lambda(q1.dtype)
            out = self.subln(attn1 - lam * attn2) * (1.0 - self.lambda_init)
            out = self.o_proj(out.transpose(1, 2).reshape(B, Lq, self.d_model))
            if return_v_cur:
                return out, present_kv, v_raw_cur
            if use_cache:
                return out, present_kv
            return out

        # --- Standard GQA path (supports legacy, CoordKey-only, YOCO/CLA, ResFormer, ModeGate, QuantVLA) ---
        can_fuse_qkv = (
            self.owns_kv
            and (shared_kv is None)
            and (not is_cross)
            and (kv_in is q_mod)
            and (not self.use_coordkey)
            and (not in_onnx)
            and self.k_proj is not None
            and self.v_proj is not None
            and (self.q_proj.bias is None and self.k_proj.bias is None and self.v_proj.bias is None)
            and (not _is_qat_linear_active(self.q_proj))
            and (not _is_qat_linear_active(self.k_proj))
            and (not _is_qat_linear_active(self.v_proj))
        )
        if can_fuse_qkv:
            assert self.q_norm is not None and self.k_norm is not None
            w_qkv = torch.cat((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=0)
            qkv = F.linear(q_mod, w_qkv)
            d_q = self.n_heads * self.head_dim
            d_kv = self.n_kv_heads * self.head_dim
            q_raw, k_raw, v_raw = qkv.split((d_q, d_kv, d_kv), dim=-1)
            q = self.q_norm(q_raw.view(B, Lq, self.n_heads, self.head_dim)).transpose(1, 2)
            k = self.k_norm(k_raw.view(B, Lq, self.n_kv_heads, self.head_dim)).transpose(1, 2)
            v = v_raw.view(B, Lq, self.n_kv_heads, self.head_dim).transpose(1, 2)
            if self.use_v_res and v0 is not None and self.v_res_alpha is not None:
                g = torch.sigmoid(self.v_res_alpha).to(v.dtype)
                v = (1.0 - g) * v + g * v0
            if return_v_cur:
                v_raw_cur = v
            if self.use_rope and rope_cos is not None and rope_sin is not None:
                q, k = apply_rope(q, k, rope_cos, rope_sin)
            if past_kv is not None:
                k = torch.cat([past_kv[0], k], dim=2)
                v = torch.cat([past_kv[1], v], dim=2)
        else:
            assert self.q_norm is not None
            if self.use_coordkey:
                assert self.q_coord_proj is not None and self.q_coord_norm is not None
                can_fuse_qc = (
                    (not in_onnx)
                    and (self.q_proj.bias is None and self.q_coord_proj.bias is None)
                    and (not _is_qat_linear_active(self.q_proj))
                    and (not _is_qat_linear_active(self.q_coord_proj))
                )
                if can_fuse_qc:
                    d_qc = self.n_heads * self.head_dim
                    d_qs = self.n_heads * self.coord_dim
                    w_q = torch.cat((self.q_proj.weight, self.q_coord_proj.weight), dim=0)
                    q_c_flat, q_s_flat = F.linear(q_mod, w_q).split((d_qc, d_qs), dim=-1)
                    q = self.q_norm(q_c_flat.view(B, Lq, self.n_heads, self.head_dim)).transpose(1, 2)
                    q_s = self.q_coord_norm(q_s_flat.view(B, Lq, self.n_heads, self.coord_dim)).transpose(1, 2)
                else:
                    q = self.q_norm(self.q_proj(q_mod).view(B, Lq, self.n_heads, self.head_dim)).transpose(1, 2)
                    q_s = self.q_coord_norm(
                        self.q_coord_proj(q_mod).view(B, Lq, self.n_heads, self.coord_dim)
                    ).transpose(1, 2)
                q = torch.cat([q, q_s], dim=-1)
            else:
                q = self.q_norm(self.q_proj(q_mod).view(B, Lq, self.n_heads, self.head_dim)).transpose(1, 2)

            if shared_kv is not None:
                if self.use_rope and rope_cos is not None and rope_sin is not None and not is_cross:
                    q = _apply_rope_single(q, rope_cos, rope_sin)
                k, v = shared_kv
            elif is_cross and past_kv is not None:
                k, v = past_kv
            else:
                if kv_in is None:
                    raise ValueError("kv_in must be provided when past_kv and shared_kv are None")
                if self.use_coordkey or self.use_v_res or return_v_cur:
                    k, v = self.compute_kv(kv_in, coord_feat=coord_feat, v0=v0)
                    v_raw_cur = v
                else:
                    assert self.k_proj is not None and self.v_proj is not None and self.k_norm is not None
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

        if self.head_temp is not None:
            q = q * self.head_temp.to(q.dtype)

        present_kv = (k, v) if (use_cache or return_v_cur) else None

        use_causal_flag = is_causal and (attn_mask is None)
        eff_mask = attn_mask
        if use_causal_flag and not is_cross and k.shape[2] > Lq:
            past_len = k.shape[2] - Lq
            if Lq == 1:
                use_causal_flag = False
            else:
                eff_mask = _get_causal_offset_mask(Lq, past_len, q.device)
                use_causal_flag = False

        if self.use_native_gqa and self.n_rep > 1 and (not self.use_coordkey):
            out = _sdpa(q, k, v, attn_mask=eff_mask, is_causal=use_causal_flag, enable_gqa=True)
        else:
            k_exp = self._expand_kv(k)
            v_exp = self._expand_kv(v)
            out = _sdpa(q, k_exp, v_exp, attn_mask=eff_mask, is_causal=use_causal_flag, enable_gqa=False)
        out = out.transpose(1, 2).reshape(B, Lq, self.d_model)
        out = self.o_proj(out)
        if return_v_cur:
            return out, present_kv, v_raw_cur
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
        *,
        n_cross_kv_heads: Optional[int] = None,
        owns_self_kv: bool = True,
        owns_cross_kv: bool = True,
        use_self_v_res: bool = False,
        use_cross_v_res: bool = False,
        use_diff_cross_attn: bool = False,
        layer_idx: int = 0,
        use_coordkey: bool = False,
        coord_in_dim: int = 128,
        coord_dim: int = 16,
        use_mode_gate: bool = False,
        mode_gate_rank: int = 32,
        use_head_temp: bool = False,
    ) -> None:
        super().__init__()
        self.use_rope = bool(use_rope)
        self.owns_self_kv = bool(owns_self_kv)
        self.owns_cross_kv = bool(owns_cross_kv)
        cross_kv = int(n_cross_kv_heads) if n_cross_kv_heads is not None else int(n_kv_heads)
        self.norm1 = RMSNorm(d_model)
        self.self_attn = GroupedQueryAttention(
            d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            use_rope=self.use_rope,
            owns_kv=self.owns_self_kv,
            use_v_res=use_self_v_res,
            use_head_temp=use_head_temp,
        )
        self.norm2 = RMSNorm(d_model)
        self.cross_attn = GroupedQueryAttention(
            d_model,
            n_heads=n_heads,
            n_kv_heads=cross_kv,
            use_rope=False,
            owns_kv=self.owns_cross_kv,
            use_v_res=use_cross_v_res,
            use_diff_attn=use_diff_cross_attn,
            layer_idx=layer_idx,
            use_coordkey=use_coordkey,
            coord_in_dim=coord_in_dim,
            coord_dim=coord_dim,
            use_mode_gate=use_mode_gate,
            mode_gate_rank=mode_gate_rank,
            use_head_temp=use_head_temp,
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
        coord_feat: Optional[torch.Tensor] = None,
        shared_self_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        shared_cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        self_v0: Optional[torch.Tensor] = None,
        cross_v0: Optional[torch.Tensor] = None,
        return_bank_state: bool = False,
    ) -> Any:
        h = self.norm1(x)
        r_cos = rope_cos if self.use_rope else None
        r_sin = rope_sin if self.use_rope else None

        if return_bank_state:
            self_past = past_kv[0] if past_kv is not None else None
            cross_past = past_kv[1] if past_kv is not None else None
            sa_out, self_kv_out, self_v_cur = self.self_attn(
                h,
                h if self.owns_self_kv else None,
                rope_cos=r_cos,
                rope_sin=r_sin,
                is_causal=True,
                past_kv=self_past if self.owns_self_kv else None,
                use_cache=use_cache,
                is_cross=False,
                shared_kv=None if self.owns_self_kv else shared_self_kv,
                v0=self_v0,
                return_v_cur=True,
            )
            x = x + self.drop(sa_out)
            ca_out, cross_kv_out, cross_v_cur = self.cross_attn(
                self.norm2(x),
                memory if self.owns_cross_kv else None,
                past_kv=cross_past if self.owns_cross_kv else None,
                use_cache=use_cache,
                is_cross=True,
                coord_feat=coord_feat,
                shared_kv=None if self.owns_cross_kv else shared_cross_kv,
                v0=cross_v0,
                return_v_cur=True,
            )
            x = x + self.drop(ca_out)
            if checkpoint_ffn and self.training and torch.is_grad_enabled():
                x = x + self.drop(_grad_ckpt(self.ffn, self.norm3(x), use_reentrant=False))
            else:
                x = x + self.drop(self.ffn(self.norm3(x)))
            return x, self_kv_out, cross_kv_out, self_v_cur, cross_v_cur

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
                coord_feat=coord_feat,
            )
            x = x + self.drop(ca_out)
            x = x + self.drop(self.ffn(self.norm3(x)))
            return x, (new_self_kv, new_cross_kv)

        x = x + self.drop(self.self_attn(h, h, rope_cos=r_cos, rope_sin=r_sin, is_causal=True))
        x = x + self.drop(self.cross_attn(self.norm2(x), memory, is_cross=True, coord_feat=coord_feat))
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
    ``encoder.patch_merger``, ``encoder.global_layers``, ``deepstack_fusions``,
    ``dec_layers``, ``mtp_heads``, ``bd3_repair_head``, and ``lm_head`` via
    ``fake_quant_int4_ste`` while keeping ``backbone``, ``encoder.router``,
    ``encoder.pos2d_proj``, ``coord_mlp``, ``tok_emb``, norms, and aux heads in FP16.
    """

    TARGET_ROOTS: Tuple[str, ...] = (
        "encoder.axial_layers",
        "encoder.patch_merger",
        "encoder.global_layers",
        "deepstack_fusions",
        "dec_layers",
        "mtp_heads",
        "bd3_repair_head",
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
                if isinstance(mod, nn.Linear) and mod.out_features > 1 and mod not in self._orig_forwards:
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
        dec_ffn_mult = c.dec_ffn_mult if c.dec_ffn_mult is not None else c.ffn_mult
        d_ff = int(round(c.d_model * dec_ffn_mult))
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

        # 2b. Optional shared 2D Fourier CoordMLP for CoordKey-CA (None by default)
        self.coord_mlp: Optional[FourierCoordMLP] = (
            FourierCoordMLP(
                n_freqs=c.coord_n_freqs,
                hidden_dim=c.coord_mlp_hidden,
                out_dim=c.coord_mlp_hidden,
            )
            if c.use_coordkey_ca
            else None
        )

        # 3. Training-Only Zero-Inference-Cost Layout & Stroke-Gradient Aux Heads
        self.sys_y_head: Optional[SysYHead] = SysYHead(c.d_model) if c.use_sys_y_aux else None
        self.sys_count_head: Optional[SysCountHead] = SysCountHead(c.d_model) if c.use_sys_count_aux else None
        self.staff_seg_head: Optional[StaffSegHead] = (
            StaffSegHead(self.backbone.p2_ch, self.backbone.p3_ch) if c.use_staff_seg_aux else None
        )
        self.stroke_hog_head: Optional[StrokeHogHead] = (
            StrokeHogHead(self.backbone.p2_ch, n_bins=8) if c.use_stroke_hog_head else None
        )

        # 4. GQA Causal Decoder (Llama 4 iRoPE / YOCO-3Bank + CLA-2:1 + ResFormer + Diff/CoordKey/ModeGate)
        _resolved_pad = c.resolved_pad_id()
        pad_idx = _resolved_pad if (0 <= _resolved_pad < c.vocab_size) else None
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model, padding_idx=pad_idx)
        self.rope1d = _CachedRoPE1D(head_dim=head_dim, base=c.rope_base)
        self.deepstack_fusions = nn.ModuleList([
            DeepStackFusion(c.d_model) for _ in range(min(2, c.n_dec_layers))
        ])

        self_kv_heads = c.yoco_self_kv_heads if c.yoco_self_kv_heads is not None else c.n_kv_heads
        cross_kv_heads = c.yoco_cross_kv_heads if c.yoco_cross_kv_heads is not None else c.n_kv_heads

        if c.use_yoco_cla:
            self.yoco_cross_banks: Tuple[int, ...] = tuple(c.yoco_cross_banks)
            bank_of_layer: List[int] = []
            for i in range(c.n_dec_layers):
                b_idx = 0
                for bi, prod in enumerate(self.yoco_cross_banks):
                    if prod <= i:
                        b_idx = bi
                bank_of_layer.append(b_idx)
            self._cross_bank_of_layer: Tuple[int, ...] = tuple(bank_of_layer)
            self._n_self_banks: int = (c.n_dec_layers + 1) // 2
        else:
            self.yoco_cross_banks = ()
            self._cross_bank_of_layer = ()
            self._n_self_banks = c.n_dec_layers

        dec_layers: List[GQADecoderLayer] = []
        for i in range(c.n_dec_layers):
            if c.use_yoco_cla:
                owns_self = (i % 2 == 0)
                owns_cross = i in self.yoco_cross_banks
                pair_idx = i // 2
                use_rope = (c.irope_nope_every <= 0) or (
                    ((pair_idx + 1) % max(1, c.irope_nope_every // 2)) != 0
                )
            else:
                owns_self = True
                owns_cross = True
                use_rope = (c.irope_nope_every <= 0) or (((i + 1) % c.irope_nope_every) != 0)
            dec_layers.append(
                GQADecoderLayer(
                    d_model=c.d_model,
                    n_heads=c.n_heads,
                    n_kv_heads=self_kv_heads,
                    d_ff=d_ff,
                    dropout=c.dropout,
                    use_rope=use_rope,
                    n_cross_kv_heads=cross_kv_heads,
                    owns_self_kv=owns_self,
                    owns_cross_kv=owns_cross,
                    use_self_v_res=bool(c.use_resformer_v_res and owns_self and i > 0),
                    use_cross_v_res=bool(c.use_resformer_v_res and owns_cross and i > 0),
                    use_diff_cross_attn=c.use_diff_cross_attn,
                    layer_idx=i,
                    use_coordkey=c.use_coordkey_ca,
                    coord_in_dim=c.coord_mlp_hidden,
                    coord_dim=c.coord_dim,
                    use_mode_gate=(i < c.mode_gate_layers),
                    mode_gate_rank=c.mode_gate_rank,
                    use_head_temp=c.use_quantvla_head_temp,
                )
            )
        self.dec_layers = nn.ModuleList(dec_layers)
        self.dec_norm = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        eff_text_vocab = (
            int(c.text_vocab_size)
            if (c.bekern_full or c.use_text_aux_head) and int(c.text_vocab_size) > 0
            else int(c.text_aux_vocab_size)
        )
        self.lm_head_text: Optional[nn.Linear] = (
            nn.Linear(c.d_model, eff_text_vocab, bias=False)
            if eff_text_vocab > 0
            else None
        )
        # Iter 16 bekern-full Layer 3 UTF-8 byte embedding + text mode gate (None when bekern_full=False -> 0 legacy params)
        if c.bekern_full and eff_text_vocab > 0:
            self.tok_emb_text: Optional[nn.Embedding] = nn.Embedding(
                eff_text_vocab,
                c.d_model,
                padding_idx=bekern_full.TEXT_PAD_ID if bekern_full.TEXT_PAD_ID < eff_text_vocab else None,
            )
        else:
            self.tok_emb_text = None
        if c.bekern_full:
            self.text_mode_gate: Optional[nn.Linear] = nn.Linear(c.d_model, 1, bias=True)
            nn.init.zeros_(self.text_mode_gate.weight)
            nn.init.constant_(self.text_mode_gate.bias, -2.0)
        else:
            self.text_mode_gate = None
        self._last_text_gate_prob: Optional[torch.Tensor] = None
        self._bekern_stage1_hooks: List[Any] = []
        self._stage1_frozen_slices: Dict[str, Tuple[int, int, torch.Tensor]] = {}
        self._warm_start_old_registers: int = 0

        n_extra_mtp = max(0, int(c.mtp_k) - 1)
        if c.use_chained_mtp:
            self.mtp_heads = nn.ModuleList([ChainedMTPHead(c.d_model) for _ in range(n_extra_mtp)])
        else:
            self.mtp_heads = nn.ModuleList([MTPHead(c.d_model) for _ in range(n_extra_mtp)])

        # 5. Optional BD3-LM + ReMDM-conf bidirectional intra-cell repair head (inference-capable)
        self.bd3_repair_head: Optional[BD3ReMDMRepairHead] = (
            BD3ReMDMRepairHead(
                d_model=c.d_model,
                d_rep=c.bd3_d_rep,
                n_heads=c.bd3_n_heads,
                d_ff=c.bd3_d_ff,
                n_layers=c.bd3_n_layers,
                block_size=c.bd3_block_size,
                dropout=c.dropout,
            )
            if c.use_bd3_repair_head
            else None
        )
        self._struct_weight_cache: Dict[Tuple[int, float, torch.device, torch.dtype], torch.Tensor] = {}

    def strip_aux_heads(
        self,
        strip_mtp: bool = False,
        strip_text_head: Optional[bool] = None,
    ) -> "UniOMRv2Net":
        """Strip training-only layout, stroke-HOG, and text aux heads (and optionally MTP heads) for 0%-cost export."""
        self.sys_y_head = None
        self.sys_count_head = None
        self.staff_seg_head = None
        self.stroke_hog_head = None
        do_strip_text = (not bool(self.cfg.bekern_full)) if strip_text_head is None else bool(strip_text_head)
        if do_strip_text:
            self.lm_head_text = None
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

        if self.coord_mlp is not None:
            coord_feat = self.coord_mlp(
                router_info["kept_idx"],
                grid_hw=router_info["grid_hw"],
                n_bg_summary=int(router_info["n_bg_summary"]),
                dtype=mem_final.dtype,
                is_full_grid=bool(router_info.get("is_full_grid", False)),
            )
            router_info["coord_feat"] = coord_feat

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
        coord_feat: Optional[torch.Tensor] = None,
    ) -> Any:
        """Precompute DeepStackFusion + cross-attention K/V once per page for cached decoding."""
        if isinstance(mem_levels, (tuple, list)):
            mem_post_merger, mem_global_mid, mem_final = mem_levels
            level_mems: Optional[Tuple[torch.Tensor, torch.Tensor]] = (mem_post_merger, mem_global_mid)
        else:
            mem_final = mem_levels
            level_mems = None

        if self.cfg.use_yoco_cla:
            cross_banks: List[Tuple[torch.Tensor, torch.Tensor]] = []
            cross_v0: Optional[torch.Tensor] = None
            for bank_idx, prod_idx in enumerate(self.yoco_cross_banks):
                if level_mems is not None and bank_idx < len(self.deepstack_fusions):
                    bank_mem = self.deepstack_fusions[bank_idx](mem_final, level_mems[bank_idx])
                else:
                    bank_mem = mem_final
                ca = self.dec_layers[prod_idx].cross_attn
                k, v = ca.compute_kv(bank_mem, coord_feat=coord_feat, v0=cross_v0)
                if bank_idx == 0:
                    cross_v0 = v
                cross_banks.append((k, v))
            return {
                "self_banks": [None] * self._n_self_banks,
                "cross_banks": cross_banks,
            }

        past_kvs: List[Tuple[None, Tuple[torch.Tensor, torch.Tensor]]] = []
        cross_v0_std: Optional[torch.Tensor] = None
        for idx, layer in enumerate(self.dec_layers):
            if level_mems is not None and idx < len(self.deepstack_fusions):
                layer_mem = self.deepstack_fusions[idx](mem_final, level_mems[idx])
            else:
                layer_mem = mem_final
            ca = layer.cross_attn
            k, v = ca.compute_kv(layer_mem, coord_feat=coord_feat, v0=cross_v0_std)
            if idx == 0:
                cross_v0_std = v
            past_kvs.append((None, (k, v)))
        return past_kvs

    def _decode_tokens_impl(
        self,
        tgt_ids: torch.Tensor,
        memory: Optional[Any],
        return_mtp: Optional[bool] = None,
        return_aux: Optional[bool] = None,
        past_kvs: Optional[Any] = None,
        use_cache: bool = False,
        apply_tok_dropout: bool = True,
        coord_feat: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        greedy_chain_mtp: bool = False,
        use_text_emb: bool = False,
        out_extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[
        torch.Tensor,
        List[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Any],
    ]:
        _, T = tgt_ids.shape
        inp_ids = tgt_ids
        if self.training and apply_tok_dropout and self.cfg.tok_dropout > 0.0 and not use_cache and not use_text_emb:
            pad_id = self.cfg.resolved_pad_id()
            bos_id = self.cfg.resolved_bos_id()
            eos_id = self.cfg.resolved_eos_id()
            sys_id = self.cfg.resolved_sys_id()
            if self.cfg.bekern_full:
                body_pool = torch.tensor(
                    bekern_full.VALID_MUSIC_BODY_TOKEN_IDS,
                    device=inp_ids.device,
                    dtype=inp_ids.dtype,
                )
                rand_toks = body_pool[
                    torch.randint(0, int(body_pool.numel()), inp_ids.shape, device=inp_ids.device)
                ]
                body_mask = bekern_full.compute_is_music_body_mask(inp_ids).to(
                    device=inp_ids.device, dtype=torch.bool
                )
                drop_mask = (torch.rand_like(inp_ids, dtype=torch.float32) < self.cfg.tok_dropout) & body_mask
            else:
                music_end = self.cfg.resolved_music_vocab_end()
                rand_toks = torch.randint(0, music_end, inp_ids.shape, device=inp_ids.device, dtype=inp_ids.dtype)
                drop_mask = (
                    (torch.rand_like(inp_ids, dtype=torch.float32) < self.cfg.tok_dropout)
                    & (inp_ids != pad_id)
                    & (inp_ids != bos_id)
                    & (inp_ids != eos_id)
                    & (inp_ids != sys_id)
                    & (rand_toks != pad_id)
                    & (rand_toks != bos_id)
                    & (rand_toks != eos_id)
                    & (rand_toks != sys_id)
                )
            inp_ids = torch.where(drop_mask, rand_toks, inp_ids)

        if use_text_emb and self.tok_emb_text is not None:
            x = self.tok_emb_text(inp_ids)
        else:
            x = self.tok_emb(inp_ids)
        compute_dtype = _active_compute_dtype(x)
        if x.dtype != compute_dtype:
            x = x.to(dtype=compute_dtype)

        past_len = 0
        if isinstance(past_kvs, dict):
            sb0 = past_kvs["self_banks"][0] if past_kvs.get("self_banks") else None
            if sb0 is not None:
                past_len = int(sb0[0].shape[2])
        elif past_kvs is not None and len(past_kvs) > 0 and past_kvs[0] is not None and past_kvs[0][0] is not None:
            past_len = int(past_kvs[0][0][0].shape[2])

        cos_all, sin_all = self.rope1d(past_len + T, device=x.device, dtype=compute_dtype)
        cos1d = cos_all[:, :, past_len : past_len + T, :]
        sin1d = sin_all[:, :, past_len : past_len + T, :]

        mem_final: Optional[torch.Tensor] = None
        level_mems: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if memory is not None:
            if isinstance(memory, (tuple, list)):
                mem_post_merger, mem_global_mid, mem_final = memory
                level_mems = (mem_post_merger, mem_global_mid)
            else:
                mem_final = memory

        bt = int(x.shape[0] * x.shape[1])
        _, dec_full_thr, dec_ffn_thr = _get_grad_ckpt_thresholds(x.device)
        can_dec_ckpt = self.training and self.cfg.grad_checkpoint and torch.is_grad_enabled()
        use_dec_full_ckpt = can_dec_ckpt and (bt > dec_full_thr)
        use_dec_ffn_ckpt = can_dec_ckpt and (dec_ffn_thr < bt <= dec_full_thr)

        new_past_kvs_out: Optional[Any] = None

        if self.cfg.use_yoco_cla:
            n_cross_banks = len(self.yoco_cross_banks)
            bank_mems: Optional[List[torch.Tensor]] = None
            if mem_final is not None:
                bank_mems = []
                for bi in range(n_cross_banks):
                    if level_mems is not None and bi < len(self.deepstack_fusions):
                        bank_mems.append(self.deepstack_fusions[bi](mem_final, level_mems[bi]))
                    else:
                        bank_mems.append(mem_final)

            self_banks_in = past_kvs["self_banks"] if isinstance(past_kvs, dict) else [None] * self._n_self_banks
            cross_banks_in = past_kvs["cross_banks"] if isinstance(past_kvs, dict) else [None] * n_cross_banks
            new_self_banks: List[Any] = [None] * self._n_self_banks
            new_cross_banks: List[Any] = list(cross_banks_in)
            cur_self_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
            cur_cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
            self_v0: Optional[torch.Tensor] = None
            cross_v0: Optional[torch.Tensor] = None

            for idx, layer in enumerate(self.dec_layers):
                self_bank_idx = idx // 2
                cross_bank_idx = self._cross_bank_of_layer[idx]
                past_c = cross_banks_in[cross_bank_idx]
                layer_mem = None
                if layer.owns_cross_kv and past_c is None and bank_mems is not None:
                    layer_mem = bank_mems[cross_bank_idx]

                if use_cache:
                    x, self_kv_out, cross_kv_out, self_v_cur, cross_v_cur = layer(
                        x,
                        memory=layer_mem,
                        rope_cos=cos1d,
                        rope_sin=sin1d,
                        past_kv=(
                            self_banks_in[self_bank_idx] if layer.owns_self_kv else None,
                            past_c if layer.owns_cross_kv else None,
                        ),
                        use_cache=True,
                        coord_feat=coord_feat,
                        shared_self_kv=None if layer.owns_self_kv else cur_self_kv,
                        shared_cross_kv=None if layer.owns_cross_kv else cur_cross_kv,
                        self_v0=self_v0,
                        cross_v0=cross_v0,
                        return_bank_state=True,
                    )
                elif use_dec_full_ckpt:
                    ss_k = None if (layer.owns_self_kv or cur_self_kv is None) else cur_self_kv[0]
                    ss_v = None if (layer.owns_self_kv or cur_self_kv is None) else cur_self_kv[1]
                    sc_k = None if (layer.owns_cross_kv or cur_cross_kv is None) else cur_cross_kv[0]
                    sc_v = None if (layer.owns_cross_kv or cur_cross_kv is None) else cur_cross_kv[1]

                    def _run_yoco_layer(
                        _x: torch.Tensor,
                        _mem: Optional[torch.Tensor],
                        _c1d: torch.Tensor,
                        _s1d: torch.Tensor,
                        _cf: Optional[torch.Tensor],
                        _ssk: Optional[torch.Tensor],
                        _ssv: Optional[torch.Tensor],
                        _sck: Optional[torch.Tensor],
                        _scv: Optional[torch.Tensor],
                        _sv0: Optional[torch.Tensor],
                        _cv0: Optional[torch.Tensor],
                        _layer: GQADecoderLayer = layer,
                    ) -> Any:
                        _ss = (_ssk, _ssv) if (_ssk is not None and _ssv is not None) else None
                        _sc = (_sck, _scv) if (_sck is not None and _scv is not None) else None
                        return _layer(
                            _x,
                            _mem,
                            _c1d,
                            _s1d,
                            coord_feat=_cf,
                            shared_self_kv=_ss,
                            shared_cross_kv=_sc,
                            self_v0=_sv0,
                            cross_v0=_cv0,
                            return_bank_state=True,
                        )

                    x, self_kv_out, cross_kv_out, self_v_cur, cross_v_cur = _grad_ckpt(
                        _run_yoco_layer,
                        x,
                        layer_mem,
                        cos1d,
                        sin1d,
                        coord_feat,
                        ss_k,
                        ss_v,
                        sc_k,
                        sc_v,
                        self_v0,
                        cross_v0,
                        use_reentrant=False,
                    )
                else:
                    x, self_kv_out, cross_kv_out, self_v_cur, cross_v_cur = layer(
                        x,
                        memory=layer_mem,
                        rope_cos=cos1d,
                        rope_sin=sin1d,
                        checkpoint_ffn=use_dec_ffn_ckpt,
                        coord_feat=coord_feat,
                        shared_self_kv=None if layer.owns_self_kv else cur_self_kv,
                        shared_cross_kv=None if layer.owns_cross_kv else cur_cross_kv,
                        self_v0=self_v0,
                        cross_v0=cross_v0,
                        return_bank_state=True,
                    )

                if layer.owns_self_kv:
                    cur_self_kv = self_kv_out
                    new_self_banks[self_bank_idx] = self_kv_out
                    if idx == 0:
                        self_v0 = self_v_cur
                if layer.owns_cross_kv:
                    cur_cross_kv = cross_kv_out
                    new_cross_banks[cross_bank_idx] = cross_kv_out
                    if idx == 0:
                        cross_v0 = cross_v_cur

            if use_cache:
                new_past_kvs_out = {"self_banks": new_self_banks, "cross_banks": new_cross_banks}
        else:
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
                        past_kv=layer_past, use_cache=True, coord_feat=coord_feat,
                    )
                    new_past_kvs.append(layer_kv)
                elif use_dec_full_ckpt:
                    if coord_feat is not None:
                        x = _grad_ckpt(
                            lambda _x, _m, _c, _s, _cf, _l=layer: _l(_x, _m, _c, _s, coord_feat=_cf),
                            x, layer_mem, cos1d, sin1d, coord_feat,
                            use_reentrant=False,
                        )
                    else:
                        x = _grad_ckpt(
                            layer, x, layer_mem, cos1d, sin1d,
                            use_reentrant=False,
                        )
                else:
                    x = layer(
                        x, memory=layer_mem, rope_cos=cos1d, rope_sin=sin1d,
                        checkpoint_ffn=use_dec_ffn_ckpt, coord_feat=coord_feat,
                    )
            if use_cache:
                new_past_kvs_out = new_past_kvs

        h = self.dec_norm(x)

        want_mtp = self.training if return_mtp is None else bool(return_mtp)
        if want_mtp and len(self.mtp_heads) > 0:
            if self.cfg.use_chained_mtp:
                B_t, T_t = tgt_ids.shape
                pad_id = self.cfg.resolved_pad_id()
                if (not greedy_chain_mtp) and (targets is not None):
                    h_states: List[torch.Tensor] = [h]
                    h_prev = h
                    for k_idx, head in enumerate(self.mtp_heads):
                        if k_idx == 0:
                            next_tok = targets
                        elif T_t > k_idx:
                            pad_tail = torch.full((B_t, k_idx), pad_id, device=targets.device, dtype=targets.dtype)
                            next_tok = torch.cat([targets[:, k_idx:], pad_tail], dim=1)
                        else:
                            next_tok = torch.full((B_t, T_t), pad_id, device=targets.device, dtype=targets.dtype)
                        next_emb = self.tok_emb(next_tok).to(h_prev.dtype)
                        h_k = head(h_prev, next_emb)
                        h_states.append(h_k)
                        h_prev = h_k
                    all_logits = self.lm_head(torch.cat(h_states, dim=0)).split(B_t, dim=0)
                    logits = all_logits[0]
                    mtp_logits = list(all_logits[1:])
                else:
                    logits = self.lm_head(h)
                    mtp_logits = []
                    h_prev = h
                    prev_logits = logits
                    for head in self.mtp_heads:
                        next_tok = prev_logits.detach().argmax(dim=-1)
                        next_emb = self.tok_emb(next_tok).to(h_prev.dtype)
                        h_k = head(h_prev, next_emb)
                        logits_k = self.lm_head(h_k)
                        mtp_logits.append(logits_k)
                        h_prev = h_k
                        prev_logits = logits_k
            else:
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
                mtp_logits = list(all_logits[1:].unbind(0))
        else:
            logits = self.lm_head(h)
            mtp_logits = []

        want_aux = self.training if return_aux is None else bool(return_aux)
        sys_y_pred: Optional[torch.Tensor] = None
        text_logits: Optional[torch.Tensor] = None
        if self.cfg.bekern_full and self.text_mode_gate is not None and not use_text_emb:
            if use_cache:
                with torch.no_grad():
                    self._last_text_gate_prob = torch.sigmoid(self.text_mode_gate(h[:, -1, :])).squeeze(-1)
            elif out_extra is not None:
                out_extra["text_gate_logits"] = self.text_mode_gate(h).squeeze(-1)
        if out_extra is not None:
            out_extra["h_last"] = h[:, -1:, :]
        if use_text_emb and self.lm_head_text is not None:
            text_logits = self.lm_head_text(h)
        elif want_aux and not use_cache:
            if self.sys_y_head is not None:
                sys_y_pred = self.sys_y_head(h)
            if self.lm_head_text is not None:
                text_logits = self.lm_head_text(h)

        return logits, mtp_logits, sys_y_pred, text_logits, new_past_kvs_out

    def decode_tokens(
        self,
        tgt_ids: torch.Tensor,
        memory: Optional[Any],
        return_mtp: Optional[bool] = None,
        return_aux: Optional[bool] = None,
        past_kvs: Optional[Any] = None,
        use_cache: bool = False,
        apply_tok_dropout: bool = True,
        coord_feat: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        greedy_chain_mtp: bool = False,
    ) -> Any:
        """Run causal GQA decoder over ``tgt_ids`` ``(B, T)`` conditioned on DeepStack ``mem_levels`` or tensor ``memory``."""
        logits, mtp_logits, sys_y_pred, _text_logits, new_past_kvs = self._decode_tokens_impl(
            tgt_ids,
            memory,
            return_mtp=return_mtp,
            return_aux=return_aux,
            past_kvs=past_kvs,
            use_cache=use_cache,
            apply_tok_dropout=apply_tok_dropout,
            coord_feat=coord_feat,
            targets=targets,
            greedy_chain_mtp=greedy_chain_mtp,
        )
        if use_cache:
            return logits, mtp_logits, new_past_kvs
        return logits, mtp_logits, sys_y_pred

    def should_emit_text(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(emit_mask, p_text)`` using ``text_mode_gate`` and ``mode_gate_conf_thresh`` (Mitigation #5)."""
        if self.text_mode_gate is not None:
            p_text = torch.sigmoid(self.text_mode_gate(h)).squeeze(-1)
        else:
            p_text = torch.ones(h.shape[:-1], device=h.device, dtype=h.dtype)
        emit_mask = p_text >= float(self.cfg.mode_gate_conf_thresh)
        return emit_mask, p_text

    @torch.no_grad()
    def draft_mtp_tokens(self, h_last: torch.Tensor, cur_tok_id: int) -> List[int]:
        """Speculatively draft up to ``len(mtp_heads)`` tokens with MTP-4 mode-switch guard (Mitigation #6).

        When ``cfg.bekern_full`` is True, speculative drafting is immediately disabled if
        ``cur_tok_id`` is a Layer 3 mode-switch token, and stops before emitting any drafted
        token that satisfies ``bekern_full.is_mode_switch_token(tok)``.
        """
        if len(self.mtp_heads) == 0:
            return []
        cur_id = int(cur_tok_id)
        if self.cfg.bekern_full and bekern_full.is_mode_switch_token(cur_id):
            return []

        if h_last.ndim == 1:
            h_cur = h_last.unsqueeze(0).unsqueeze(0)
        elif h_last.ndim == 2:
            h_cur = h_last.unsqueeze(1)
        else:
            h_cur = h_last[:, -1:, :]

        drafted: List[int] = []
        h_prev = h_cur
        prev_tok = torch.tensor([[cur_id]], device=h_cur.device, dtype=torch.long)
        for head in self.mtp_heads:
            if self.cfg.use_chained_mtp:
                next_emb = self.tok_emb(prev_tok).to(h_prev.dtype)
                h_k = head(h_prev, next_emb)
            else:
                h_k = head(h_cur)
            logits_k = self.lm_head(h_k)
            tok_k = int(logits_k[0, -1, :].argmax(dim=-1).item())
            if self.cfg.bekern_full and bekern_full.is_mode_switch_token(tok_k):
                break
            drafted.append(tok_k)
            h_prev = h_k
            prev_tok = torch.tensor([[tok_k]], device=h_cur.device, dtype=torch.long)
        return drafted

    def _slice_mem_batch(self, mem: Any, b: int) -> Any:
        if isinstance(mem, tuple):
            return tuple(m[b : b + 1] for m in mem)
        if isinstance(mem, list):
            return [m[b : b + 1] for m in mem]
        return mem[b : b + 1]

    def _slice_past_kvs_batch(self, past_kvs: Any, b: int) -> Any:
        """Clone single-sample ``b`` view of ``past_kvs`` without mutating the live music cache."""
        if past_kvs is None:
            return None
        if isinstance(past_kvs, dict):
            return {
                "self_banks": [
                    (kv[0][b : b + 1].clone(), kv[1][b : b + 1].clone()) if kv is not None else None
                    for kv in past_kvs["self_banks"]
                ],
                "cross_banks": [
                    (kv[0][b : b + 1].clone(), kv[1][b : b + 1].clone()) if kv is not None else None
                    for kv in past_kvs["cross_banks"]
                ],
            }
        sliced: List[Any] = []
        for entry in past_kvs:
            if entry is None:
                sliced.append(None)
            else:
                skv, ckv = entry
                s_out = (skv[0][b : b + 1].clone(), skv[1][b : b + 1].clone()) if skv is not None else None
                c_out = (ckv[0][b : b + 1].clone(), ckv[1][b : b + 1].clone()) if ckv is not None else None
                sliced.append((s_out, c_out))
        return sliced

    @torch.no_grad()
    def _decode_utf8_text_span(
        self,
        mem_b: Any,
        coord_b: Optional[torch.Tensor],
        max_len: int = 128,
        past_kvs_b: Optional[Any] = None,
        open_tok_id: Optional[int] = None,
    ) -> List[int]:
        """Decode one Layer 3 UTF-8 byte span conditioned on the live music-stream ``past_kvs``."""
        if self.tok_emb_text is None or self.lm_head_text is None:
            return [bekern_full.TEXT_BOS_ID, bekern_full.TEXT_EOS_ID]
        ref_t = mem_b[-1] if isinstance(mem_b, (tuple, list)) else mem_b
        device = ref_t.device
        if past_kvs_b is not None:
            text_kvs: Any = past_kvs_b
            if open_tok_id is not None:
                open_t = torch.tensor([[int(open_tok_id)]], dtype=torch.long, device=device)
                _, _, _, _, text_kvs = self._decode_tokens_impl(
                    open_t,
                    memory=None,
                    return_mtp=False,
                    return_aux=False,
                    past_kvs=text_kvs,
                    use_cache=True,
                    apply_tok_dropout=False,
                    use_text_emb=False,
                )
        else:
            text_kvs = self.precompute_cross_kv(mem_b, coord_feat=coord_b)
        cur = torch.full((1, 1), bekern_full.TEXT_BOS_ID, dtype=torch.long, device=device)
        out_bytes: List[int] = [bekern_full.TEXT_BOS_ID]
        for _ in range(max(1, int(max_len))):
            _, _, _, txt_logits, text_kvs = self._decode_tokens_impl(
                cur,
                memory=None,
                return_mtp=False,
                return_aux=False,
                past_kvs=text_kvs,
                use_cache=True,
                apply_tok_dropout=False,
                use_text_emb=True,
            )
            assert txt_logits is not None
            step_logits = txt_logits[0, -1, :].clone()
            # Never emit TEXT_BOS or TEXT_PAD inside a decoded UTF-8 byte span
            if bekern_full.TEXT_BOS_ID < step_logits.shape[0]:
                step_logits[bekern_full.TEXT_BOS_ID] = float("-inf")
            if bekern_full.TEXT_PAD_ID < step_logits.shape[0]:
                step_logits[bekern_full.TEXT_PAD_ID] = float("-inf")
            nxt = int(step_logits.argmax(dim=-1).item())
            out_bytes.append(nxt)
            if nxt == bekern_full.TEXT_EOS_ID:
                break
            cur = torch.tensor([[nxt]], dtype=torch.long, device=device)
        if out_bytes[-1] != bekern_full.TEXT_EOS_ID:
            out_bytes.append(bekern_full.TEXT_EOS_ID)
        return out_bytes

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_len: int = 256,
        keep_ratio_override: Optional[float] = None,
        return_info: bool = False,
        return_text_spans: bool = False,
        max_text_span_len: int = 128,
        use_mtp_draft: bool = False,
    ) -> Any:
        """Fast KV-cached GQA greedy decoding (with optional bekern-full UTF-8 text span decoding)."""
        mem, _, router_info = self.encode_image(
            images, keep_ratio_override=keep_ratio_override, return_aux=False
        )
        coord_feat = router_info.get("coord_feat")
        B = images.shape[0]
        device = images.device
        pad_id = self.cfg.resolved_pad_id()
        bos_id = self.cfg.resolved_bos_id()
        eos_id = self.cfg.resolved_eos_id()
        cur_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        past_kvs: Any = self.precompute_cross_kv(mem, coord_feat=coord_feat)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        out_tokens: List[torch.Tensor] = [cur_ids]
        text_open_ids: Optional[torch.Tensor] = (
            torch.tensor(sorted(bekern_full.TEXT_SPAN_OPEN_TOKEN_IDS), dtype=torch.long, device=device)
            if (self.cfg.bekern_full and self.text_mode_gate is not None)
            else None
        )
        text_spans_per_batch: List[List[List[int]]] = [[] for _ in range(B)]
        pending_close = torch.zeros(B, dtype=torch.bool, device=device)
        can_decode_text = (
            bool(self.cfg.bekern_full)
            and (self.tok_emb_text is not None)
            and (self.lm_head_text is not None)
        )

        step = 0
        while step < max_len:
            extra: Dict[str, Any] = {}
            logits, _, _, _, past_kvs = self._decode_tokens_impl(
                cur_ids,
                memory=None,
                return_mtp=False,
                return_aux=False,
                past_kvs=past_kvs,
                use_cache=True,
                apply_tok_dropout=False,
                out_extra=extra if use_mtp_draft else None,
            )
            step_logits = logits[:, -1, :]
            # Mitigation #5: confidence-gated Layer 3 text span entry (p_text >= mode_gate_conf_thresh)
            if text_open_ids is not None and self._last_text_gate_prob is not None:
                gate_block = self._last_text_gate_prob < float(self.cfg.mode_gate_conf_thresh)
                if bool(gate_block.any()):
                    step_logits = step_logits.clone()
                    mask_open = torch.zeros_like(step_logits, dtype=torch.bool).scatter_(
                        1, text_open_ids.unsqueeze(0).expand(B, -1), True
                    )
                    step_logits[gate_block.unsqueeze(1) & mask_open] = float("-inf")
            next_tok = step_logits.argmax(dim=-1)
            # If previous step opened a Layer 3 text span, close it cleanly in the music stream.
            if can_decode_text and bool(pending_close.any()):
                next_tok = torch.where(
                    pending_close,
                    torch.full_like(next_tok, bekern_full.TEXT_SPAN_CLOSE_TOKEN_ID),
                    next_tok,
                )
                pending_close.zero_()
            next_tok = torch.where(finished, torch.full_like(next_tok, pad_id), next_tok)
            out_tokens.append(next_tok.unsqueeze(1))
            step += 1

            # Dual-head music -> UTF-8 text span decoding when an open tag is emitted.
            if can_decode_text:
                for b in range(B):
                    if not bool(finished[b].item()) and int(next_tok[b].item()) in bekern_full.TEXT_SPAN_OPEN_TOKEN_IDS:
                        mem_b = self._slice_mem_batch(mem, b)
                        coord_b = coord_feat[b : b + 1] if coord_feat is not None else None
                        past_kvs_b = self._slice_past_kvs_batch(past_kvs, b)
                        span_bytes = self._decode_utf8_text_span(
                            mem_b,
                            coord_b,
                            max_len=max_text_span_len,
                            past_kvs_b=past_kvs_b,
                            open_tok_id=int(next_tok[b].item()),
                        )
                        text_spans_per_batch[b].append(span_bytes)
                        pending_close[b] = True

            # Optional speculative MTP-4 drafting (guarded by bekern_full.is_mode_switch_token).
            if (
                use_mtp_draft
                and B == 1
                and not bool(finished[0].item())
                and not bool(pending_close[0].item())
                and len(self.mtp_heads) > 0
                and "h_last" in extra
            ):
                _ = self.draft_mtp_tokens(extra["h_last"], int(next_tok[0].item()))

            finished = finished | (next_tok == eos_id)
            if bool(finished.all()):
                break
            cur_ids = next_tok.unsqueeze(1)

        seq = torch.cat(out_tokens, dim=1)
        if return_text_spans and return_info:
            return seq, text_spans_per_batch, router_info
        if return_text_spans:
            return seq, text_spans_per_batch
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
        targets: Optional[torch.Tensor] = None,
        coord_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        mem, aux_enc, router_info = self.encode_image(
            images, keep_ratio_override=keep_ratio_override, return_aux=return_aux
        )
        eff_coord_feat = coord_feat if coord_feat is not None else router_info.get("coord_feat")
        extra: Dict[str, Any] = {}
        logits, mtp_logits, sys_y_pred, text_logits, _ = self._decode_tokens_impl(
            tgt_ids,
            mem,
            return_mtp=return_mtp,
            return_aux=return_aux,
            coord_feat=eff_coord_feat,
            targets=targets,
            out_extra=extra if self.cfg.bekern_full else None,
        )
        out: Dict[str, Any] = {
            "logits": logits,
            "mtp_logits": mtp_logits,
            "sys_y_pred": sys_y_pred,
            "sys_count_pred": aux_enc["sys_count_pred"],
            "staff_seg_logits": aux_enc["staff_seg_logits"],
            "stroke_hog_pred": aux_enc["stroke_hog_pred"],
            "router_info": router_info,
        }
        if text_logits is not None:
            out["text_logits"] = text_logits
        if "text_gate_logits" in extra:
            out["text_gate_logits"] = extra["text_gate_logits"]
        if self.bd3_repair_head is not None or self.cfg.bekern_full:
            out["mem_final"] = mem[-1]
        if self.cfg.bekern_full:
            out["mem_levels"] = mem
        return out

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
        text_aux_loss_weight: Optional[float] = None,
        bd3_repair_weight: float = 0.0,
        bd3_n_views: int = 2,
        is_v1_schema: Optional[torch.Tensor] = None,
        v2a_token_ids: Optional[Any] = None,
        has_v1_schema: Optional[bool] = None,
        layer_masks: Optional[torch.Tensor] = None,
        text_byte_targets: Optional[torch.Tensor] = None,
        text_byte_inputs: Optional[torch.Tensor] = None,
        text_gate_loss_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined CE + MTP-4 + router saliency + layout & stroke-HOG + optional text/BD3 losses."""
        logits: torch.Tensor = out["logits"]
        V = logits.shape[-1]
        ls = float(max(0.0, min(0.5, label_smoothing)))
        ce_w = (
            ce_token_weights.to(device=logits.device, dtype=logits.dtype)
            if ce_token_weights is not None
            else None
        )
        pad_id = self.cfg.resolved_pad_id()
        if layer_masks is not None:
            ce_loss = bekern_full.masked_layer_cross_entropy(
                logits, targets, layer_masks, label_smoothing=ls, weight=ce_w
            )
        else:
            ce_loss = F.cross_entropy(
                logits.reshape(-1, V),
                targets.reshape(-1),
                weight=ce_w,
                ignore_index=pad_id,
                label_smoothing=ls,
            )

        # MTP-4 auxiliary losses (unified single contiguous cross_entropy over PAD-padded shifted targets)
        mtp_logits: List[torch.Tensor] = out.get("mtp_logits", [])
        T = targets.shape[1]
        if mtp_logits and T > 1:
            n_mtp = len(mtp_logits)
            shifted_tgts = torch.full(
                (n_mtp, targets.shape[0], T),
                pad_id,
                device=targets.device,
                dtype=targets.dtype,
            )
            for idx in range(n_mtp):
                shift = idx + 1
                if T > shift:
                    shifted_tgts[idx, :, :-shift] = targets[:, shift:]
            if layer_masks is not None:
                mtp_acc = logits.new_zeros((), dtype=torch.float32)
                for idx in range(n_mtp):
                    mtp_acc = mtp_acc + bekern_full.masked_layer_cross_entropy(
                        mtp_logits[idx], shifted_tgts[idx], layer_masks, label_smoothing=ls
                    ).float()
                mtp_loss = mtp_acc / float(n_mtp)
            else:
                mtp_stacked = torch.stack(mtp_logits, dim=0)  # (M, B, T, V)
                shifted_flat = shifted_tgts.reshape(-1)
                mtp_valid = (shifted_flat != pad_id).to(torch.float32)
                mtp_per_tok = F.cross_entropy(
                    mtp_stacked.reshape(-1, V),
                    shifted_flat,
                    ignore_index=pad_id,
                    label_smoothing=ls,
                    reduction="none",
                )
                mtp_loss = (
                    (mtp_per_tok.float() * mtp_valid).sum() / mtp_valid.sum().clamp(min=1.0)
                )
        else:
            mtp_loss = logits.new_zeros((), dtype=torch.float32)

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

        # Optional text-aux / Layer 3 UTF-8 byte head loss
        eff_text_aux_w = (
            float(self.cfg.text_aux_loss_weight)
            if text_aux_loss_weight is None
            else float(text_aux_loss_weight)
        )
        text_logits: Optional[torch.Tensor] = out.get("text_logits")
        if self.cfg.bekern_full:
            # bekern-full: NEVER call legacy build_text_aux_targets; supervise lm_head_text strictly on UTF-8 byte targets.
            if text_byte_targets is not None and self.lm_head_text is not None and eff_text_aux_w > 0.0:
                tbt = text_byte_targets.reshape(-1).to(device=logits.device, dtype=torch.long)
                valid_tb = tbt != bekern_full.TEXT_PAD_ID
                mem_for_text = out.get("mem_levels", out.get("mem_final"))
                if (
                    text_byte_inputs is not None
                    and self.tok_emb_text is not None
                    and mem_for_text is not None
                ):
                    _, _, _, byte_logits, _ = self._decode_tokens_impl(
                        text_byte_inputs,
                        mem_for_text,
                        return_mtp=False,
                        return_aux=False,
                        use_cache=False,
                        apply_tok_dropout=False,
                        coord_feat=rinfo.get("coord_feat"),
                        use_text_emb=True,
                    )
                    assert byte_logits is not None
                    if bool(valid_tb.any()):
                        text_aux_loss = F.cross_entropy(
                            byte_logits.reshape(-1, byte_logits.shape[-1]),
                            tbt.to(device=byte_logits.device),
                            ignore_index=bekern_full.TEXT_PAD_ID,
                        ).to(logits.dtype)
                    else:
                        text_aux_loss = (byte_logits.sum() * 0.0).to(logits.dtype)
                elif text_logits is not None and text_logits.shape[:2] == text_byte_targets.shape[:2]:
                    if bool(valid_tb.any()):
                        text_aux_loss = F.cross_entropy(
                            text_logits.reshape(-1, text_logits.shape[-1]),
                            tbt.to(device=text_logits.device),
                            ignore_index=bekern_full.TEXT_PAD_ID,
                        ).to(logits.dtype)
                    else:
                        text_aux_loss = (text_logits.sum() * 0.0).to(logits.dtype)
                else:
                    text_aux_loss = logits.new_zeros(())
            else:
                text_aux_loss = logits.new_zeros(())
        elif text_logits is not None and eff_text_aux_w > 0.0:
            text_tgt, _ = build_text_aux_targets(
                targets,
                int(self.cfg.vocab_size),
                pad_id=pad_id,
                text_vocab_size=int(text_logits.shape[-1]),
            )
            text_aux_loss = F.cross_entropy(
                text_logits.reshape(-1, text_logits.shape[-1]),
                text_tgt.reshape(-1),
                ignore_index=-100,
            ).to(logits.dtype)
        else:
            text_aux_loss = logits.new_zeros(())

        # bekern-full Layer 3 text_mode_gate BCE supervision (predicts text-span open tokens on L3-annotated rows)
        text_gate_logits: Optional[torch.Tensor] = out.get("text_gate_logits")
        eff_text_gate_w = float(text_gate_loss_weight) if self.cfg.bekern_full else 0.0
        if self.cfg.bekern_full and text_gate_logits is not None and eff_text_gate_w > 0.0:
            open_ids_t = torch.tensor(
                sorted(bekern_full.TEXT_SPAN_OPEN_TOKEN_IDS),
                device=targets.device,
                dtype=targets.dtype,
            )
            is_text_open_target = torch.isin(targets, open_ids_t).to(torch.float32)
            gate_valid = targets != pad_id
            if layer_masks is not None:
                lm_t = (
                    layer_masks.to(device=targets.device, dtype=torch.long)
                    if isinstance(layer_masks, torch.Tensor)
                    else torch.as_tensor(layer_masks, device=targets.device, dtype=torch.long)
                )
                if lm_t.ndim == 0:
                    lm_t = lm_t.expand(targets.shape[0])
                l3_row = ((lm_t & bekern_full.LAYER_BIT_L3) != 0).unsqueeze(1)
                gate_valid = gate_valid & l3_row
            if bool(gate_valid.any()):
                per_pos_bce = F.binary_cross_entropy_with_logits(
                    text_gate_logits.float(), is_text_open_target, reduction="none"
                )
                gv_f = gate_valid.to(torch.float32)
                text_gate_loss = ((per_pos_bce * gv_f).sum() / gv_f.sum().clamp(min=1.0)).to(logits.dtype)
            else:
                text_gate_loss = text_gate_logits.sum() * 0.0
        else:
            text_gate_loss = logits.new_zeros(())

        # Optional BD3-LM + ReMDM-conf complementary masked-diffusion repair loss
        eff_bd3_w = float(bd3_repair_weight)
        mem_final_bd3: Optional[torch.Tensor] = out.get("mem_final")
        if self.bd3_repair_head is not None and mem_final_bd3 is not None and eff_bd3_w > 0.0:
            bd3_repair_loss = compute_bd3_repair_loss(
                self,
                mem_final_bd3,
                targets,
                n_views=int(bd3_n_views),
                is_v1_schema=is_v1_schema,
                v2a_token_ids=v2a_token_ids,
                has_v1_schema=has_v1_schema,
                layer_masks=layer_masks,
            ).to(logits.dtype)
        else:
            bd3_repair_loss = logits.new_zeros(())

        total_loss = (
            ce_loss
            + mtp_loss_weight * mtp_loss
            + saliency_loss_weight * saliency_loss
            + sys_y_loss_weight * sys_y_loss
            + sys_count_loss_weight * sys_count_loss
            + staff_seg_loss_weight * staff_seg_loss
            + stroke_hog_loss_weight * stroke_hog_loss
            + eff_text_aux_w * text_aux_loss
            + eff_text_gate_w * text_gate_loss
            + eff_bd3_w * bd3_repair_loss
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
            "text_aux_loss": text_aux_loss,
            "text_gate_loss": text_gate_loss,
            "bd3_repair_loss": bd3_repair_loss,
        }

    def param_summary(self, int4_block_size: int = 32) -> Dict[str, Any]:
        """Return exact parameter breakdown and WebGPU FP16/INT4/Hybrid byte estimates."""
        def _count(mod: Optional[nn.Module]) -> int:
            return sum(p.numel() for p in mod.parameters()) if mod is not None else 0

        total_params = _count(self)
        backbone_params = _count(self.backbone)
        encoder_params = _count(self.encoder)
        router_params = _count(self.encoder.router)
        coord_mlp_params = _count(self.coord_mlp)
        deepstack_params = _count(self.deepstack_fusions)
        bd3_params = _count(self.bd3_repair_head)
        decoder_params = (
            _count(self.tok_emb)
            + _count(self.tok_emb_text)
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
            + (0 if self.cfg.bekern_full else _count(self.lm_head_text))
        )
        infer_params = total_params - aux_params - mtp_params

        # Exact Hybrid FP16/INT4-QAT split (matching QATInt4ControllerV2 target roots, excluding MTP/aux):
        int4_roots: Tuple[Optional[nn.Module], ...] = (
            self.encoder.axial_layers,
            self.encoder.patch_merger,
            self.encoder.global_layers,
            self.deepstack_fusions,
            self.dec_layers,
            self.bd3_repair_head,
            self.lm_head,
        )
        int4_weight_params = 0
        for root in int4_roots:
            if root is None:
                continue
            for m in root.modules():
                if isinstance(m, nn.Linear) and m.out_features > 1:
                    int4_weight_params += m.weight.numel()
        fp16_keep_params = infer_params - int4_weight_params

        mtp_int4_weight_params = 0
        for m in self.mtp_heads.modules():
            if isinstance(m, nn.Linear) and m.out_features > 1:
                mtp_int4_weight_params += m.weight.numel()
        mtp_fp16_keep_params = mtp_params - mtp_int4_weight_params

        scale_bytes_per_param = 2.0 / float(int4_block_size)  # FP16 scale per block_size weights
        int4_raw_mb = (infer_params * 0.5) / 1_000_000.0
        int4_scaled_mb = (infer_params * (0.5 + scale_bytes_per_param)) / 1_000_000.0
        hybrid_raw_mb = (fp16_keep_params * 2.0 + int4_weight_params * 0.5) / 1_000_000.0
        hybrid_scaled_mb = (
            fp16_keep_params * 2.0 + int4_weight_params * (0.5 + scale_bytes_per_param)
        ) / 1_000_000.0
        hybrid_with_mtp_mb = (
            (fp16_keep_params + mtp_fp16_keep_params) * 2.0
            + (int4_weight_params + mtp_int4_weight_params) * (0.5 + scale_bytes_per_param)
        ) / 1_000_000.0

        return {
            "total_params": total_params,
            "infer_params": infer_params,
            "backbone_params": backbone_params,
            "encoder_params": encoder_params,
            "router_params": router_params,
            "coord_mlp_params": coord_mlp_params,
            "deepstack_params": deepstack_params,
            "decoder_params": decoder_params,
            "bd3_params": bd3_params,
            "mtp_params": mtp_params,
            "aux_params": aux_params,
            "int4_weight_params": int4_weight_params,
            "fp16_keep_params": fp16_keep_params,
            "fp16_mb": (infer_params * 2.0) / 1_000_000.0,
            "pure_int4_mb": int4_raw_mb,
            "pure_int4_with_scales_mb": int4_scaled_mb,
            "hybrid_fp16_int4_mb": hybrid_raw_mb,
            "hybrid_fp16_int4_with_scales_mb": hybrid_scaled_mb,
            "hybrid_fp16_int4_with_mtp_mb": hybrid_with_mtp_mb,
        }


# ---------------------------------------------------------------------------
# 6. Iter 16 bekern-full Warm-Start & 2-Stage Freeze/Unfreeze Helpers
# ---------------------------------------------------------------------------

def warm_start_from_v265_checkpoint(
    model: UniOMRv2Net,
    ckpt_or_path: Any,
    new_tok_std: float = 0.02,
) -> Dict[str, Any]:
    """Warm-start a ``bekern-full`` model from a legacy ``vocab_size=265`` checkpoint.

    Safely copies:
      - ``tok_emb.weight[:265]``, ``lm_head.weight[:265]`` (and bias if any), and any
        ``mtp_heads`` vocab-axis slices ``[:265]``; newly appended rows ``[265:]`` are
        freshly initialized with small normal std (``new_tok_std=0.02``; biases zeroed).
      - ``encoder.register_tokens`` across ``0 -> 8`` or ``4 -> 8`` (copying ``[:, :min_r, :]``
        and leaving extra register slots freshly initialized with std ``new_tok_std``).
      - Newly added or resized Layer 3 ``tok_emb_text`` / ``lm_head_text`` / ``text_mode_gate``
        weights without failing ``strict=False``.
    """
    if isinstance(ckpt_or_path, (str, Path)):
        try:
            raw_ckpt = torch.load(str(ckpt_or_path), map_location="cpu", weights_only=True)
        except Exception:
            raw_ckpt = torch.load(str(ckpt_or_path), map_location="cpu", weights_only=False)
    else:
        raw_ckpt = ckpt_or_path

    if isinstance(raw_ckpt, dict):
        if "model" in raw_ckpt and isinstance(raw_ckpt["model"], dict):
            raw_sd_in: Dict[str, torch.Tensor] = raw_ckpt["model"]
        elif "state_dict" in raw_ckpt and isinstance(raw_ckpt["state_dict"], dict):
            raw_sd_in = raw_ckpt["state_dict"]
        elif "ema_state_dict" in raw_ckpt and isinstance(raw_ckpt["ema_state_dict"], dict):
            raw_sd_in = raw_ckpt["ema_state_dict"]
        elif "ema_model" in raw_ckpt and isinstance(raw_ckpt["ema_model"], dict):
            raw_sd_in = raw_ckpt["ema_model"]
        else:
            raw_sd_in = raw_ckpt
    else:
        raise ValueError(f"Unsupported checkpoint payload type: {type(raw_ckpt)!r}")

    # Strip torch.compile / DDP prefixes (_orig_mod., module.) cleanly.
    raw_sd: Dict[str, torch.Tensor] = {}
    for raw_k, raw_v in raw_sd_in.items():
        if not isinstance(raw_v, torch.Tensor):
            continue
        k = str(raw_k)
        while k.startswith("_orig_mod.") or k.startswith("module."):
            if k.startswith("_orig_mod."):
                k = k[len("_orig_mod."):]
            elif k.startswith("module."):
                k = k[len("module."):]
        raw_sd[k] = raw_v

    model_sd = model.state_dict()
    old_v = int(bekern_full.LEGACY_VOCAB_SIZE)  # 265
    if "tok_emb.weight" in raw_sd and raw_sd["tok_emb.weight"].ndim == 2:
        old_v = int(raw_sd["tok_emb.weight"].shape[0])
    new_v = int(model.cfg.vocab_size)

    vocab_key_markers = ("tok_emb", "lm_head", "mtp_head", "mtp_heads")
    filtered_sd: Dict[str, torch.Tensor] = {}
    vocab_expanded: List[str] = []
    register_expanded: List[str] = []
    shape_skipped: List[str] = []
    unexpected_keys: List[str] = []

    for k, v in raw_sd.items():
        if k not in model_sd:
            unexpected_keys.append(k)
            continue
        dst = model_sd[k]
        if tuple(v.shape) == tuple(dst.shape):
            filtered_sd[k] = v.to(dtype=dst.dtype)
            continue

        # 1. Append-only vocab expansion: copy [:old_v] verbatim, fresh-init [old_v:] with std=new_tok_std
        if (
            new_v > old_v
            and v.ndim >= 1
            and int(v.shape[0]) == old_v
            and int(dst.shape[0]) == new_v
            and tuple(v.shape[1:]) == tuple(dst.shape[1:])
            and any(marker in k for marker in vocab_key_markers)
            and ("lm_head_text" not in k)
            and ("tok_emb_text" not in k)
        ):
            out = torch.empty_like(dst)
            if out.ndim >= 2:
                nn.init.normal_(out, mean=0.0, std=float(new_tok_std))
            else:
                nn.init.zeros_(out)
            out[:old_v] = v.to(dtype=out.dtype)
            filtered_sd[k] = out
            vocab_expanded.append(k)
            continue

        # 2. Register tokens expansion (e.g. 4 -> 8): copy [:min_r], fresh-init [min_r:]
        if (
            k.endswith("register_tokens")
            and v.ndim == 3
            and dst.ndim == 3
            and int(v.shape[0]) == int(dst.shape[0])
            and int(v.shape[2]) == int(dst.shape[2])
        ):
            out = torch.empty_like(dst)
            nn.init.normal_(out, mean=0.0, std=float(new_tok_std))
            copy_r = min(int(v.shape[1]), int(dst.shape[1]))
            if copy_r > 0:
                out[:, :copy_r, :] = v[:, :copy_r, :].to(dtype=out.dtype)
            filtered_sd[k] = out
            register_expanded.append(f"{k}:{v.shape[1]}->{dst.shape[1]}")
            model._warm_start_old_registers = copy_r
            continue

        # 3. Any other shape mismatch (e.g. Iter 15 lm_head_text 129 -> Iter 16 259): leave freshly initialized
        shape_skipped.append(f"{k}:{tuple(v.shape)}!={tuple(dst.shape)}")

    if len(filtered_sd) == 0:
        raise RuntimeError(
            f"warm_start_from_v265_checkpoint loaded 0 matching parameters "
            f"(unexpected={len(unexpected_keys)}, shape_skipped={len(shape_skipped)})"
        )

    inc = model.load_state_dict(filtered_sd, strict=False)
    all_unexpected = list(unexpected_keys) + [
        uk for uk in inc.unexpected_keys if uk not in unexpected_keys
    ]
    return {
        "loaded_count": len(filtered_sd),
        "old_vocab_size": old_v,
        "new_vocab_size": new_v,
        "vocab_expanded": vocab_expanded,
        "register_expanded": register_expanded,
        "missing_keys": list(inc.missing_keys),
        "unexpected_keys": all_unexpected,
        "shape_skipped": shape_skipped,
    }


def _is_bekern_stage1_trainable_name(name: str) -> bool:
    """Return True for parameters kept trainable during Iter 16 Stage-1 freeze (Mitigation #10)."""
    if name in ("tok_emb.weight", "lm_head.weight", "lm_head.bias"):
        return True
    if name.endswith("register_tokens"):
        return True
    if name.startswith("lm_head_text.") or name.startswith("tok_emb_text.") or name.startswith("text_mode_gate."):
        return True
    if (
        ".mode_gate." in name
        or name.endswith(".film_gamma")
        or name.endswith(".film_beta")
        or ".q_down." in name
        or ".q_up." in name
    ):
        return True
    return False


def restore_bekern_full_stage1_frozen_slices(model: UniOMRv2Net) -> int:
    """Restore frozen Stage-1 parameter slices after ``opt.step()`` (protects against AdamW weight decay).

    Keeps legacy ``tok_emb.weight[:265]``, ``lm_head.weight[:265]``, and any copied
    ``register_tokens[:, :old_r, :]`` bit-exact invariant even under ``AdamW(weight_decay > 0)``.
    """
    frozen_slices: Dict[str, Tuple[int, int, torch.Tensor]] = getattr(model, "_stage1_frozen_slices", {})
    if not frozen_slices:
        return 0
    param_map = dict(model.named_parameters())
    restored = 0
    with torch.no_grad():
        for name, (dim, end_idx, snap) in frozen_slices.items():
            p = param_map.get(name)
            if p is None:
                continue
            snap_dev = snap.to(device=p.device, dtype=p.dtype)
            if dim == 0:
                p[:end_idx].copy_(snap_dev)
            elif dim == 1:
                p[:, :end_idx, ...].copy_(snap_dev)
            restored += 1
    return restored


def set_bekern_full_stage(
    model: UniOMRv2Net,
    stage: int,
    legacy_vocab_size: int = bekern_full.LEGACY_VOCAB_SIZE,
    legacy_register_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Configure Iter 16 2-stage freeze/unfreeze schedule (Mitigation #10).

    - ``stage == 1``: freezes encoder + core decoder layers, keeping ONLY:
        * new token rows ``[legacy_vocab_size:]`` in ``tok_emb`` & ``lm_head``
          (enforced via backward gradient hooks zeroing ``grad[:legacy_vocab_size]``
          PLUS frozen snapshots restored by ``restore_bekern_full_stage1_frozen_slices``
          after ``opt.step()`` so AdamW decoupled weight decay never drifts ``[:265]``),
        * ``encoder.register_tokens`` (preserving ``[:, :old_r, :]`` if warm-started),
        * ``ModeGate`` parameters (``mode_gate``, ``film_gamma/beta``, ``q_down/q_up``, ``text_mode_gate``),
        * Layer 3 ``tok_emb_text`` & ``lm_head_text``.
    - ``stage == 2`` (or ``stage == 0``): removes all Stage-1 gradient hooks and
      frozen snapshots, and unfreezes every parameter in ``model`` for joint fine-tuning.
    """
    st = int(stage)
    if st not in (0, 1, 2):
        raise ValueError(f"stage must be 0, 1, or 2, got {stage!r}")

    # Always clear any prior Stage-1 hooks & frozen snapshots first so repeated calls stay idempotent.
    old_hooks = getattr(model, "_bekern_stage1_hooks", None)
    if old_hooks:
        for h in old_hooks:
            try:
                h.remove()
            except Exception:
                pass
    model._bekern_stage1_hooks = []
    model._stage1_frozen_slices = {}

    if st in (0, 2):
        for p in model.parameters():
            p.requires_grad_(True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "stage": st,
            "trainable_params": trainable,
            "frozen_params": 0,
            "n_hooks": 0,
        }

    # Stage 1: freeze core; train only new vocab rows, register_tokens, ModeGate, and text head.
    cut = int(legacy_vocab_size)
    old_r = (
        int(legacy_register_tokens)
        if legacy_register_tokens is not None
        else int(getattr(model, "_warm_start_old_registers", 0))
    )
    hooks: List[Any] = []
    frozen_slices: Dict[str, Tuple[int, int, torch.Tensor]] = {}
    for name, p in model.named_parameters():
        if _is_bekern_stage1_trainable_name(name):
            p.requires_grad_(True)
            if name in ("tok_emb.weight", "lm_head.weight", "lm_head.bias") and p.shape[0] > cut:
                frozen_slices[name] = (0, cut, p[:cut].detach().clone())

                def _make_zero_prefix_hook(prefix_len: int):
                    def _hook(grad: torch.Tensor) -> torch.Tensor:
                        if grad is None:
                            return grad
                        out = grad.clone()
                        out[:prefix_len].zero_()
                        return out
                    return _hook

                hooks.append(p.register_hook(_make_zero_prefix_hook(cut)))
            elif name.endswith("register_tokens") and old_r > 0 and p.ndim == 3 and p.shape[1] > old_r:
                frozen_slices[name] = (1, old_r, p[:, :old_r, :].detach().clone())

                def _make_zero_reg_hook(prefix_r: int):
                    def _hook(grad: torch.Tensor) -> torch.Tensor:
                        if grad is None:
                            return grad
                        out = grad.clone()
                        out[:, :prefix_r, :].zero_()
                        return out
                    return _hook

                hooks.append(p.register_hook(_make_zero_reg_hook(old_r)))
        else:
            p.requires_grad_(False)

    model._bekern_stage1_hooks = hooks
    model._stage1_frozen_slices = frozen_slices
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "stage": 1,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "n_hooks": len(hooks),
        "n_frozen_slices": len(frozen_slices),
    }
