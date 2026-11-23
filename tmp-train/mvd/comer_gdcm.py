"""Exp 2B/2C: CoMER Coverage-Guided Cross-Attention + GDCM Global Dynamic Counting Head.

Modular, zero-init warm-start-safe architectural plug-ins for ``MvdNet``:

1. :class:`CoverageAttentionBias` (CoMER):
   Converts cumulative past cross-attention weights ``[B, n_heads, H_m, W_m]``
   into an additive spatial cross-attention bias ``[B, n_heads, 1, H_m * W_m]``
   via a lightweight 2D conv block with a zero-initialized 1x1 output conv
   (100% identity at step 0 for lossless checkpoint warm-starting).
   Provides both batched teacher-forced (:meth:`compute_shifted_coverage_bias`)
   and incremental KV-cached (:meth:`step_coverage_bias`) helpers that are
   bit-consistent and soft-biased (never hard-masking ``I1`` multi-voice
   backward jumps).

2. :class:`GlobalDynamicCountingHead` (GDCM) + :func:`extract_strip_counts`:
   Predicts 6 strip-level structural counts from encoder memory ``mem``
   (``n_bars``, ``n_voices_max``, ``n_notes``, ``n_rests``, ``n_dots``,
   ``n_accidentals``), trains with scale-invariant SmoothL1/Huber loss on
   ``log1p(count)``, and projects the predicted inventory via a zero-init
   conditioning head ``count_to_mem_cond(pred_counts) -> [B, 1, d_model]``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import grammar  # noqa: E402

__all__ = [
    "COUNT_NAMES",
    "N_COUNT_CLASSES",
    "CoverageAttentionBias",
    "GlobalDynamicCountingHead",
    "extract_strip_counts",
]

COUNT_NAMES: Tuple[str, ...] = (
    "n_bars",
    "n_voices_max",
    "n_notes",
    "n_rests",
    "n_dots",
    "n_accidentals",
)
N_COUNT_CLASSES: int = len(COUNT_NAMES)

# ---------------------------------------------------------------------------
# Precomputed bekern vocab lookup tables for vectorized count extraction.
# Index ``len(data.VOCAB)`` is reserved as an out-of-vocab / pad sentinel (0).
# ---------------------------------------------------------------------------

_VOCAB_SIZE: int = len(data.VOCAB)


def _build_count_tables() -> Tuple[torch.Tensor, torch.Tensor]:
    """Build static lookup tables over ``data.VOCAB`` (plus 1 sentinel row).

    Returns:
      * ``flags``: ``[V + 1, 5]`` float32 indicator matrix for the additive
        counts ``(n_bars, n_notes, n_rests, n_dots, n_accidentals)``.
      * ``voice_nums``: ``[V + 1]`` long tensor mapping ``voice-k -> k``
        and all other tokens to ``0``.
    """
    flags = torch.zeros((_VOCAB_SIZE + 1, 5), dtype=torch.float32)
    voice_nums = torch.zeros((_VOCAB_SIZE + 1,), dtype=torch.long)

    for idx, tok in enumerate(data.VOCAB):
        if tok in grammar.BAR_LIKES:
            flags[idx, 0] = 1.0  # n_bars (closing barlines)
        elif tok in grammar.STEPS:
            flags[idx, 1] = 1.0  # n_notes (every note/chord/grace has 1 step)
        elif tok == grammar.REST or tok == grammar.BAR_REST:
            flags[idx, 2] = 1.0  # n_rests
        elif tok == grammar.DOT:
            flags[idx, 3] = 1.0  # n_dots
        elif tok in grammar.ALTERS:
            flags[idx, 4] = 1.0  # n_accidentals
        elif tok.startswith("voice-"):
            voice_nums[idx] = int(tok[6:])

    return flags, voice_nums


_COUNT_FLAGS_CPU, _VOICE_NUMS_CPU = _build_count_tables()


def extract_strip_counts(
    tgt_ids: Union[torch.Tensor, Sequence[int], Sequence[Sequence[int]]],
    pad_id: int = data.PAD_ID,
) -> torch.Tensor:
    """Extract 6 strip-level structural counts from bekern token IDs.

    Column order matches :data:`COUNT_NAMES`:
      ``0: n_bars``, ``1: n_voices_max``, ``2: n_notes``,
      ``3: n_rests``, ``4: n_dots``, ``5: n_accidentals``.

    Args:
      tgt_ids: ``[B, T]`` (or ``[T]``) integer tensor or nested Python list of
        bekern token IDs.
      pad_id: Padding token ID to ignore. Defaults to ``data.PAD_ID``
        (``len(data.VOCAB)``); note that ID ``0`` is pitch step ``"A"`` in
        ``data.VOCAB``, so callers should pass ``model.pad_id`` / ``data.PAD_ID``
        on real bekern streams.

    Returns:
      Float32 tensor of shape ``[B, 6]`` (or ``[6]`` for 1-D input) on the
      same device as ``tgt_ids`` (CPU for Python sequences).
    """
    if isinstance(tgt_ids, torch.Tensor):
        ids = tgt_ids.long()
    else:
        ids = torch.as_tensor(tgt_ids, dtype=torch.long)

    squeeze_1d = ids.ndim == 1
    if squeeze_1d:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2:
        raise ValueError(f"tgt_ids must be 1D or 2D, got shape {tuple(ids.shape)}")

    device = ids.device
    flags = _COUNT_FLAGS_CPU.to(device)
    voice_nums = _VOICE_NUMS_CPU.to(device)

    valid = (ids != int(pad_id)) & (ids >= 0) & (ids < _VOCAB_SIZE)
    safe_ids = torch.where(valid, ids, torch.full_like(ids, _VOCAB_SIZE))

    # Additive counts: [B, T, 5] -> [B, 5]
    additive = flags[safe_ids].sum(dim=1)

    # Max voice number: [B, T] -> [B]; default to 1 if strip has content tokens
    # with no explicit voice-N marker, 0 if strip is empty.
    v_seq = voice_nums[safe_ids]
    max_v = v_seq.max(dim=1).values if ids.shape[1] > 0 else torch.zeros(
        ids.shape[0], dtype=torch.long, device=device
    )
    has_content = valid.any(dim=1)
    n_voices_max = torch.where(
        (max_v == 0) & has_content,
        torch.ones_like(max_v),
        max_v,
    ).to(torch.float32)

    out = torch.stack(
        [
            additive[:, 0],   # 0: n_bars
            n_voices_max,     # 1: n_voices_max
            additive[:, 1],   # 2: n_notes
            additive[:, 2],   # 3: n_rests
            additive[:, 3],   # 4: n_dots
            additive[:, 4],   # 5: n_accidentals
        ],
        dim=-1,
    )
    return out.squeeze(0) if squeeze_1d else out


# ---------------------------------------------------------------------------
# Exp 2B: CoMER Coverage-Guided Cross-Attention Bias
# ---------------------------------------------------------------------------


class CoverageAttentionBias(nn.Module):
    """CoMER-style 2D convolutional coverage bias for cross-attention.

    Maps cumulative past cross-attention weights ``[B, n_heads, H_m, W_m]`` to
    an additive spatial logit bias ``[B, n_heads, 1, H_m * W_m]``.

    The output 1x1 convolution is strictly zero-initialized so warm-starting
    from an existing ``MvdNet`` checkpoint is 100% identity at step 0.
    Because the bias is purely additive and finite (never hard-masking to
    ``-inf``), multi-voice ``I1`` backward jumps remain fully open while
    giving the decoder explicit spatial memory of already-read regions.
    """

    def __init__(
        self,
        h_mem_max: int = 64,
        n_heads: int = 8,
        hidden_dim: int = 32,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1 or kernel_size < 1:
            raise ValueError(f"kernel_size must be a positive odd int, got {kernel_size}")
        self.h_mem_max = int(h_mem_max)
        self.n_heads = int(n_heads)
        self.hidden_dim = int(hidden_dim)
        self.kernel_size = int(kernel_size)
        pad = self.kernel_size // 2

        self.conv_spatial = nn.Conv2d(
            self.n_heads, self.hidden_dim, kernel_size=self.kernel_size, padding=pad
        )
        self.act = nn.GELU()
        self.conv_out = nn.Conv2d(self.hidden_dim, self.n_heads, kernel_size=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, cov: torch.Tensor) -> torch.Tensor:
        """Compute additive spatial bias from 2D cumulative coverage.

        Args:
          cov: ``[N, n_heads, H_m, W_m]`` cumulative cross-attention map.

        Returns:
          ``[N, n_heads, 1, H_m * W_m]`` additive attention logit bias.
        """
        if cov.ndim != 4 or cov.shape[1] != self.n_heads:
            raise ValueError(
                f"Expected cov shape [N, {self.n_heads}, H_m, W_m], got {tuple(cov.shape)}"
            )
        b, h, h_mem, w_mem = cov.shape
        x = self.conv_out(self.act(self.conv_spatial(cov)))
        return x.reshape(b, h, 1, h_mem * w_mem)

    def init_coverage(
        self,
        batch_size: int,
        h_mem: int,
        w_mem: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Allocate initial zero cumulative coverage ``[B, n_heads, H_m * W_m]``."""
        return torch.zeros(
            (batch_size, self.n_heads, h_mem * w_mem),
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def update_coverage(
        cum_cov: torch.Tensor,
        step_attn: torch.Tensor,
        detach: bool = True,
    ) -> torch.Tensor:
        """Accumulate one step's cross-attention weights into ``cum_cov``.

        Args:
          cum_cov: ``[B, n_heads, S]`` or ``[B, n_heads, H_m, W_m]``.
          step_attn: ``[B, n_heads, 1, S]`` or ``[B, n_heads, S]`` or
            ``[B, n_heads, H_m, W_m]``.
          detach: Whether to detach ``step_attn`` before adding.
        """
        w = step_attn.detach() if detach else step_attn
        if w.ndim == 4 and w.shape[2] == 1 and cum_cov.ndim == 3:
            w = w.squeeze(2)
        elif w.shape != cum_cov.shape:
            w = w.reshape_as(cum_cov)
        return cum_cov + w

    def step_coverage_bias(
        self,
        cum_cov: torch.Tensor,
        h_mem: int,
        w_mem: int,
        mem_mask: Optional[torch.Tensor] = None,
        detach_cov: bool = False,
    ) -> torch.Tensor:
        """Incremental KV-cached coverage bias for a single decoding step.

        Args:
          cum_cov: Cumulative cross-attention over steps ``0..t-1``, shape
            ``[B, n_heads, H_m * W_m]`` or ``[B, n_heads, H_m, W_m]``.
          h_mem: Encoder memory height ``H_m``.
          w_mem: Encoder memory width ``W_m``.
          mem_mask: Optional bool ``[B, H_m * W_m]`` (True = ignore/padded).
          detach_cov: Detach ``cum_cov`` before applying convs.

        Returns:
          ``[B, n_heads, 1, H_m * W_m]`` additive attention bias.
        """
        cov = cum_cov.detach() if detach_cov else cum_cov
        if cov.ndim == 3:
            b, h, s = cov.shape
            if s != h_mem * w_mem:
                raise ValueError(f"cum_cov spatial len {s} != {h_mem} * {w_mem}")
            cov_2d = cov.reshape(b, h, h_mem, w_mem)
        elif cov.ndim == 4:
            b, h, hm, wm = cov.shape
            if hm != h_mem or wm != w_mem:
                raise ValueError(
                    f"cum_cov spatial {(hm, wm)} != {(h_mem, w_mem)}"
                )
            cov_2d = cov
        else:
            raise ValueError(
                f"Expected 3D or 4D cum_cov, got shape {tuple(cov.shape)}"
            )

        if mem_mask is not None:
            mask_2d = mem_mask.reshape(b, 1, h_mem, w_mem)
            cov_2d = cov_2d.masked_fill(mask_2d, 0.0)

        return self.forward(cov_2d)

    def compute_shifted_coverage_bias(
        self,
        attn_weights: torch.Tensor,
        h_mem: int,
        w_mem: int,
        mem_mask: Optional[torch.Tensor] = None,
        detach_cov: bool = True,
    ) -> torch.Tensor:
        """Batched teacher-forced causal coverage bias over all ``T`` steps.

        Step ``t`` receives the cumulative sum of attention weights over
        strictly past steps ``0..t-1`` (step ``0`` receives all zeros),
        matching :meth:`step_coverage_bias` step-for-step.

        Args:
          attn_weights: ``[B, n_heads, T, H_m * W_m]`` per-head cross-attention
            weights.
          h_mem: Encoder memory height ``H_m``.
          w_mem: Encoder memory width ``W_m``.
          mem_mask: Optional bool ``[B, H_m * W_m]`` (True = ignore/padded).
          detach_cov: Detach ``attn_weights`` before cumsum/conv (default True).

        Returns:
          ``[B, n_heads, T, H_m * W_m]`` additive attention bias.
        """
        if attn_weights.ndim != 4 or attn_weights.shape[1] != self.n_heads:
            raise ValueError(
                f"Expected attn_weights [B, {self.n_heads}, T, S], "
                f"got {tuple(attn_weights.shape)}"
            )
        b, h, t, s = attn_weights.shape
        if s != h_mem * w_mem:
            raise ValueError(f"attn_weights spatial len {s} != {h_mem} * {w_mem}")

        w = attn_weights.detach() if detach_cov else attn_weights
        if mem_mask is not None:
            w = w.masked_fill(mem_mask[:, None, None, :], 0.0)

        # Compute shifted cumulative sum in fp32 for bf16 stability, then cast back.
        if t == 1:
            cum_cov = torch.zeros_like(w)
        else:
            zeros = torch.zeros((b, h, 1, s), device=w.device, dtype=w.dtype)
            csum = torch.cumsum(w[:, :, :-1, :].float(), dim=2).to(w.dtype)
            cum_cov = torch.cat([zeros, csum], dim=2)

        cov_2d = cum_cov.transpose(1, 2).reshape(b * t, h, h_mem, w_mem)
        bias_bt = self.forward(cov_2d)  # [B * T, H, 1, S]
        return bias_bt.reshape(b, t, h, s).transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Exp 2C: GDCM Global Dynamic Counting Head
# ---------------------------------------------------------------------------


class GlobalDynamicCountingHead(nn.Module):
    """Global Dynamic Counting Module (GDCM) for strip-level structural counts.

    Predicts 6 non-negative structural counts from encoder memory
    ``mem: [B, L, d_model]`` (respecting optional ``mem_mask``):
      ``0: n_bars``, ``1: n_voices_max``, ``2: n_notes``,
      ``3: n_rests``, ``4: n_dots``, ``5: n_accidentals``.

    Uses per-class masked attention pooling + log-length conditioning so both
    extensive counts (notes, bars, dots) and intensive counts (``n_voices_max``)
    are predicted accurately across 50-token and 500-token strips.

    :meth:`count_to_mem_cond` projects predicted counts into a ``[B, 1, d_model]``
    conditioning vector via a zero-initialized output layer (100% identity at
    step 0 for lossless warm-starting).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_classes: int = N_COUNT_CLASSES,
        hidden_dim: int = 128,
        huber_beta: float = 0.5,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_classes = int(n_classes)
        self.hidden_dim = int(hidden_dim)
        self.huber_beta = float(huber_beta)

        # Per-class attention pooling over encoder memory positions L.
        self.pool_attn = nn.Linear(self.d_model, self.n_classes)
        # Scale cue from active memory length log1p(L_active).
        self.len_proj = nn.Linear(1, self.d_model)
        # Per-class count regression MLP.
        self.count_mlp = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        # Zero-initialized inventory conditioning projection -> [B, 1, d_model].
        cond_out = nn.Linear(self.hidden_dim, self.d_model)
        nn.init.zeros_(cond_out.weight)
        nn.init.zeros_(cond_out.bias)
        self.cond_proj = nn.Sequential(
            nn.Linear(self.n_classes, self.hidden_dim),
            nn.GELU(),
            cond_out,
        )

    def forward(
        self,
        mem: torch.Tensor,
        mem_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict non-negative strip-level counts ``[B, n_classes]``.

        Args:
          mem: Encoder memory ``[B, L, d_model]``.
          mem_mask: Optional bool ``[B, L]`` (True = ignore/padded).

        Returns:
          ``[B, n_classes]`` non-negative predicted counts.
        """
        if mem.ndim != 3 or mem.shape[2] != self.d_model:
            raise ValueError(
                f"Expected mem [B, L, {self.d_model}], got {tuple(mem.shape)}"
            )
        b, seq_len, _ = mem.shape
        scores = self.pool_attn(mem).float()  # [B, L, C]

        if mem_mask is not None:
            if mem_mask.shape != (b, seq_len):
                raise ValueError(
                    f"Expected mem_mask {(b, seq_len)}, got {tuple(mem_mask.shape)}"
                )
            all_masked = mem_mask.all(dim=1)  # [B]
            safe_mask = mem_mask & (~all_masked.unsqueeze(1))
            scores = scores.masked_fill(safe_mask.unsqueeze(-1), float("-inf"))
            weights = F.softmax(scores, dim=1)
            weights = weights.masked_fill(mem_mask.unsqueeze(-1), 0.0)
            active_len = (~mem_mask).sum(dim=1, keepdim=True).float()
        else:
            weights = F.softmax(scores, dim=1)
            active_len = torch.full(
                (b, 1), float(seq_len), device=mem.device, dtype=torch.float32
            )

        weights = weights.to(mem.dtype)
        # [B, C, d_model]
        pooled = torch.einsum("blc,bld->bcd", weights, mem)
        len_feat = self.len_proj(torch.log1p(active_len).to(mem.dtype)).unsqueeze(1)
        raw = self.count_mlp(pooled + len_feat).squeeze(-1)  # [B, C]
        return F.softplus(raw)

    def count_to_mem_cond(self, pred_counts: torch.Tensor) -> torch.Tensor:
        """Project predicted counts ``[B, n_classes]`` to ``[B, 1, d_model]``.

        Uses ``log1p`` compression for scale invariance and a zero-initialized
        output projection so step 0 is identically zero.
        """
        if pred_counts.ndim != 2 or pred_counts.shape[1] != self.n_classes:
            raise ValueError(
                f"Expected pred_counts [B, {self.n_classes}], "
                f"got {tuple(pred_counts.shape)}"
            )
        log_c = torch.log1p(pred_counts.clamp_min(0.0))
        return self.cond_proj(log_c).unsqueeze(1)

    def compute_loss(
        self,
        pred_counts: torch.Tensor,
        target_counts: torch.Tensor,
        beta: Optional[float] = None,
    ) -> torch.Tensor:
        """Scale-invariant SmoothL1/Huber loss on ``log1p(count)`` (fp32)."""
        b = self.huber_beta if beta is None else float(beta)
        pred_log = torch.log1p(pred_counts.float().clamp_min(0.0))
        tgt_log = torch.log1p(target_counts.float().clamp_min(0.0))
        return F.smooth_l1_loss(pred_log, tgt_log, beta=b, reduction="mean")

    def forward_with_loss(
        self,
        mem: torch.Tensor,
        tgt_ids: torch.Tensor,
        mem_mask: Optional[torch.Tensor] = None,
        pad_id: int = data.PAD_ID,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Convenience training hook: predict counts, conditioning, loss, MAE.

        Returns:
          ``(pred_counts, mem_cond, loss, metrics)``
        """
        pred_counts = self.forward(mem, mem_mask=mem_mask)
        mem_cond = self.count_to_mem_cond(pred_counts)
        target_counts = extract_strip_counts(tgt_ids, pad_id=pad_id).to(mem.device)
        loss = self.compute_loss(pred_counts, target_counts)
        with torch.no_grad():
            mae_per = (pred_counts.float() - target_counts.float()).abs().mean(dim=0)
            metrics = {"gdcm_loss": float(loss.detach())}
            for i, name in enumerate(COUNT_NAMES[: self.n_classes]):
                metrics[f"gdcm_mae_{name}"] = float(mae_per[i])
        return pred_counts, mem_cond, loss, metrics
