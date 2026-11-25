#!/usr/bin/env python3
"""SpinQuant Offline Hadamard Rotation Folding for UniOMRv2Net (0 runtime ops).

Implements offline orthogonal Walsh-Hadamard rotation folding (SpinQuant, ICLR 2025)
to suppress activation/weight outliers prior to symmetric group-wise INT4 quantization
(``MatMulNBits``, ``block_size=32``) with **zero runtime ops**:

1. ``hadamard_matrix(n)``:
   Exact normalized symmetric orthogonal Walsh-Hadamard matrix (``H @ H.T == I``).
   - For power-of-2 ``n``: full Sylvester Walsh-Hadamard matrix ``H_n / sqrt(n)``.
   - For non-power-of-2 ``n`` (e.g. ``384 = 3 * 128``, ``96 = 3 * 32``): exact
     block-diagonal Walsh-Hadamard matrix using the largest power-of-2 block divisor
     (or greedy power-of-2 decomposition if the divisor is < 16).

2. ``fold_hadamard_v2(model)``:
   Applies two exact, zero-runtime-cost orthogonal rotations to ``UniOMRv2Net``:
   - **R1 (Decoder Residual-Stream Rotation, ``d_model x d_model``)**:
     Absorbs every decoder ``RMSNorm.weight`` into its following linear projections
     (``W_in <- W_in * diag(gamma)``, ``norm.weight <- 1``) and folds orthogonal
     ``Q = hadamard_matrix(d_model)`` across:
       * ``tok_emb.weight <- tok_emb.weight @ Q``
       * Each ``GQADecoderLayer``:
         - ``self_attn.{q_proj, k_proj, v_proj}.weight <- W_in @ Q``
         - ``self_attn.o_proj.weight <- Q^T @ o_proj.weight``
         - ``cross_attn.q_proj.weight <- W_in @ Q``
         - ``cross_attn.o_proj.weight <- Q^T @ o_proj.weight``
         - ``ffn.w_gate_up.weight <- W_in @ Q``
         - ``ffn.w_down.weight <- Q^T @ ffn.w_down.weight``
       * ``dec_norm`` absorbed into ``lm_head.weight <- (lm_head.weight * gamma) @ Q``
         (and ``sys_y_head`` when ``dec_norm.weight`` is uniform).
   - **R2 (Per-Head Value-Output Rotation, ``head_dim x head_dim``)**:
     Folds ``H_hd = hadamard_matrix(head_dim)`` into every attention ``(v_proj, o_proj)``
     pair across BOTH encoder (``axial_layers.{row_attn, col_attn}``,
     ``global_layers.self_attn``) AND decoder (``dec_layers.{self_attn, cross_attn}``):
       * Each head slice of ``V`` projection: ``W_v_h <- H_hd^T @ W_v_h``
       * Each head slice of ``O`` projection: ``W_o_h <- W_o_h @ H_hd``
     Because ``attn_probs @ (V @ H_hd) @ (W_o @ H_hd)^T = (attn_probs @ V) @ W_o^T``,
     this is mathematically exact even under GQA (``n_heads != n_kv_heads``) and
     KV-cached autoregressive decoding.

3. ``measure_int4_quant_error(model_orig, model_folded, sample_img, sample_ids)``:
   Compares symmetric group-wise INT4 (``block_size=32``) logit MSE and max-abs
   error against FP32 logits before vs after Hadamard folding.
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from uni_omr import RMSNorm  # noqa: E402
from uni_omr_v2 import (  # noqa: E402
    GroupedQueryAttention,
    QATInt4ControllerV2,
    _SelfAttention,
)


def _sylvester_hadamard(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build normalized Sylvester Walsh-Hadamard matrix for power-of-2 ``n``."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"_sylvester_hadamard requires power-of-2 n >= 1, got {n}")
    h = torch.ones((1, 1), dtype=torch.float64, device=device)
    size = 1
    while size < n:
        top = torch.cat([h, h], dim=1)
        bot = torch.cat([h, -h], dim=1)
        h = torch.cat([top, bot], dim=0)
        size *= 2
    h = h / (float(n) ** 0.5)
    return h.to(dtype=dtype)


def _block_sizes_for_dim(n: int) -> List[int]:
    """Choose exact power-of-2 block sizes summing to ``n``.

    - Power-of-2 ``n`` -> ``[n]``
    - Non-power-of-2 divisible by a power of 2 ``p >= 16`` (e.g. ``384 = 3 * 128``,
      ``96 = 3 * 32``) -> ``[p] * (n // p)``
    - Otherwise -> greedy largest power-of-2 binary decomposition.
    """
    if n < 1:
        raise ValueError(f"hadamard_matrix requires n >= 1, got {n}")
    if (n & (n - 1)) == 0:
        return [n]
    lowbit = n & -n
    if lowbit >= 16:
        return [lowbit] * (n // lowbit)
    blocks: List[int] = []
    rem = n
    while rem > 0:
        p = 1 << (rem.bit_length() - 1)
        blocks.append(p)
        rem -= p
    return blocks


def hadamard_matrix(
    n: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return exact normalized symmetric orthogonal Walsh-Hadamard matrix ``(n, n)``.

    Supports any power-of-2 ``n`` (full Sylvester Hadamard) as well as any
    non-power-of-2 ``n`` via block-diagonal Hadamard (e.g. ``384 = 3 * 128``).
    Guarantees ``H @ H.T == I`` and ``H.T == H``.
    """
    dev = torch.device(device) if device is not None else torch.device("cpu")
    blocks = _block_sizes_for_dim(int(n))
    if len(blocks) == 1:
        return _sylvester_hadamard(blocks[0], device=dev, dtype=dtype)

    out = torch.zeros((n, n), dtype=torch.float64, device=dev)
    cache: Dict[int, torch.Tensor] = {}
    offset = 0
    for b in blocks:
        hb = cache.get(b)
        if hb is None:
            hb = _sylvester_hadamard(b, device=dev, dtype=torch.float64)
            cache[b] = hb
        out[offset : offset + b, offset : offset + b] = hb
        offset += b
    return out.to(dtype=dtype)


@torch.no_grad()
def _absorb_rmsnorm_into_linears(norm: RMSNorm, linears: List[nn.Linear]) -> None:
    """Absorb ``norm.weight`` into each linear's input columns and set ``norm.weight = 1``."""
    gamma = norm.weight.data.to(torch.float64)
    for lin in linears:
        w = lin.weight.data.to(torch.float64)
        lin.weight.data.copy_((w * gamma.unsqueeze(0)).to(lin.weight.dtype))
    norm.weight.data.fill_(1.0)


@torch.no_grad()
def _rotate_linear_in(lin: nn.Linear, q: torch.Tensor) -> None:
    """Right-multiply linear weight by ``Q``: ``W' = W @ Q`` (for rotated input ``x @ Q``)."""
    w = lin.weight.data.to(torch.float64)
    q64 = q.to(device=w.device, dtype=torch.float64)
    lin.weight.data.copy_((w @ q64).to(lin.weight.dtype))


@torch.no_grad()
def _rotate_linear_out(lin: nn.Linear, q: torch.Tensor) -> None:
    """Left-multiply linear weight by ``Q^T``: ``W' = Q^T @ W`` (for rotated output ``y @ Q``)."""
    w = lin.weight.data.to(torch.float64)
    q64 = q.to(device=w.device, dtype=torch.float64)
    lin.weight.data.copy_((q64.T @ w).to(lin.weight.dtype))
    if lin.bias is not None:
        b = lin.bias.data.to(torch.float64)
        lin.bias.data.copy_((b @ q64).to(lin.bias.dtype))


@torch.no_grad()
def _fold_r2_self_attn(attn: _SelfAttention, h_hd: torch.Tensor) -> None:
    """Fold per-head Hadamard ``R2`` into ``_SelfAttention`` ``(qkv_proj[V], o_proj)``."""
    d = attn.d_model
    nh = attn.n_heads
    hd = attn.head_dim
    h64 = h_hd.to(device=attn.qkv_proj.weight.device, dtype=torch.float64)

    # qkv_proj.weight has shape (3 * d, d); V occupies [2*d : 3*d, :]
    w_qkv = attn.qkv_proj.weight.data.to(torch.float64)
    w_v = w_qkv[2 * d : 3 * d, :].view(nh, hd, d)
    # Left-multiply each head's V rows by H_hd^T: (nh, hd, d)
    w_v_rot = torch.einsum("ij,hjk->hik", h64.T, w_v).reshape(d, d)
    w_qkv[2 * d : 3 * d, :] = w_v_rot
    attn.qkv_proj.weight.data.copy_(w_qkv.to(attn.qkv_proj.weight.dtype))

    # o_proj.weight has shape (d, d) -> view (d, nh, hd), right-multiply each head by H_hd
    w_o = attn.o_proj.weight.data.to(torch.float64).view(d, nh, hd)
    w_o_rot = torch.einsum("ohj,jk->ohk", w_o, h64).reshape(d, d)
    attn.o_proj.weight.data.copy_(w_o_rot.to(attn.o_proj.weight.dtype))


@torch.no_grad()
def _fold_r2_gqa(attn: GroupedQueryAttention, h_hd: torch.Tensor) -> None:
    """Fold per-head Hadamard ``R2`` into ``GroupedQueryAttention`` ``(v_proj, o_proj)``."""
    d = attn.d_model
    nh = attn.n_heads
    n_kv = attn.n_kv_heads
    hd = attn.head_dim
    h64 = h_hd.to(device=attn.v_proj.weight.device, dtype=torch.float64)

    # v_proj.weight has shape (n_kv * hd, d) -> (n_kv, hd, d)
    w_v = attn.v_proj.weight.data.to(torch.float64).view(n_kv, hd, d)
    w_v_rot = torch.einsum("ij,hjk->hik", h64.T, w_v).reshape(n_kv * hd, d)
    attn.v_proj.weight.data.copy_(w_v_rot.to(attn.v_proj.weight.dtype))

    # o_proj.weight has shape (d, n_heads * hd) -> (d, n_heads, hd)
    w_o = attn.o_proj.weight.data.to(torch.float64).view(d, nh, hd)
    w_o_rot = torch.einsum("ohj,jk->ohk", w_o, h64).reshape(d, d)
    attn.o_proj.weight.data.copy_(w_o_rot.to(attn.o_proj.weight.dtype))


@torch.no_grad()
def fold_hadamard_v2(
    model: nn.Module,
    *,
    fold_r1_decoder: bool = True,
    fold_r2_attn: bool = True,
) -> Dict[str, Any]:
    """Fold offline SpinQuant Hadamard rotations into ``UniOMRv2Net`` in-place (0 runtime ops).

    Transforms:
      - ``R1`` (``d_model x d_model``): folds across the entire decoder residual stream
        (``tok_emb`` -> all ``dec_layers`` via ``RMSNorm`` absorption -> ``dec_norm``
        -> ``lm_head``).
      - ``R2`` (``head_dim x head_dim``): folds per-head Hadamard across every
        ``(v_proj, o_proj)`` pair in encoder axial/global layers and decoder
        self/cross-attention layers.

    Returns a summary dict with folded layer counts, block structures, and dims.
    """
    cfg = getattr(model, "cfg", None)
    if cfg is None or not hasattr(cfg, "d_model") or not hasattr(cfg, "n_heads"):
        raise ValueError("fold_hadamard_v2 expects a UniOMRv2Net with cfg.d_model and cfg.n_heads")

    d_model = int(cfg.d_model)
    head_dim = int(d_model // int(cfg.n_heads))
    ref_param = next(model.parameters())
    device = ref_param.device

    q_d = hadamard_matrix(d_model, device=device, dtype=torch.float64)
    q_hd = hadamard_matrix(head_dim, device=device, dtype=torch.float64)

    n_r1_dec_layers = 0
    n_r1_linears = 0
    n_r2_attn_layers = 0

    # 1. R1: Decoder residual-stream rotation (tok_emb -> dec_layers -> dec_norm -> lm_head)
    if fold_r1_decoder and hasattr(model, "tok_emb") and hasattr(model, "dec_layers"):
        # Entry: tok_emb.weight (V, D) -> E @ Q
        emb_w = model.tok_emb.weight.data.to(torch.float64)
        model.tok_emb.weight.data.copy_((emb_w @ q_d).to(model.tok_emb.weight.dtype))

        for layer in model.dec_layers:
            # 1a. Self-attention block: norm1 -> {q_proj, k_proj, v_proj} ... o_proj -> residual
            sa: GroupedQueryAttention = layer.self_attn
            sa_in = [sa.q_proj, sa.k_proj, sa.v_proj]
            _absorb_rmsnorm_into_linears(layer.norm1, sa_in)
            for lin in sa_in:
                _rotate_linear_in(lin, q_d)
            _rotate_linear_out(sa.o_proj, q_d)

            # 1b. Cross-attention block: norm2 -> q_proj ... o_proj -> residual
            ca: GroupedQueryAttention = layer.cross_attn
            _absorb_rmsnorm_into_linears(layer.norm2, [ca.q_proj])
            _rotate_linear_in(ca.q_proj, q_d)
            _rotate_linear_out(ca.o_proj, q_d)

            # 1c. SwiGLU FFN block: norm3 -> w_gate_up ... w_down -> residual
            ffn = layer.ffn
            _absorb_rmsnorm_into_linears(layer.norm3, [ffn.w_gate_up])
            _rotate_linear_in(ffn.w_gate_up, q_d)
            _rotate_linear_out(ffn.w_down, q_d)

            n_r1_dec_layers += 1
            n_r1_linears += 8

        # Exit: dec_norm -> lm_head
        if hasattr(model, "dec_norm") and hasattr(model, "lm_head"):
            dec_gamma = model.dec_norm.weight.data.clone()
            _absorb_rmsnorm_into_linears(model.dec_norm, [model.lm_head])
            _rotate_linear_in(model.lm_head, q_d)
            n_r1_linears += 1

            # If sys_y_head is attached and dec_norm was uniform (e.g. unit init),
            # keep sys_y_head functionally consistent too.
            sys_y_head = getattr(model, "sys_y_head", None)
            if (
                sys_y_head is not None
                and torch.allclose(dec_gamma, torch.ones_like(dec_gamma), atol=1e-6)
                and isinstance(sys_y_head.net[0], RMSNorm)
                and isinstance(sys_y_head.net[1], nn.Linear)
            ):
                _absorb_rmsnorm_into_linears(sys_y_head.net[0], [sys_y_head.net[1]])
                _rotate_linear_in(sys_y_head.net[1], q_d)
                n_r1_linears += 1

    # 2. R2: Per-head V-O Hadamard rotation across ALL encoder & decoder attention layers
    if fold_r2_attn:
        for mod in model.modules():
            if isinstance(mod, _SelfAttention):
                _fold_r2_self_attn(mod, q_hd)
                n_r2_attn_layers += 1
            elif isinstance(mod, GroupedQueryAttention):
                _fold_r2_gqa(mod, q_hd)
                n_r2_attn_layers += 1

    return {
        "d_model": d_model,
        "head_dim": head_dim,
        "r1_blocks": _block_sizes_for_dim(d_model),
        "r2_blocks": _block_sizes_for_dim(head_dim),
        "n_r1_dec_layers": n_r1_dec_layers,
        "n_r1_linears": n_r1_linears,
        "n_r2_attn_layers": n_r2_attn_layers,
    }


@torch.no_grad()
def measure_int4_quant_error(
    model_orig: nn.Module,
    model_folded: nn.Module,
    sample_img: torch.Tensor,
    sample_ids: torch.Tensor,
    *,
    block_size: int = 32,
) -> Dict[str, float]:
    """Compare symmetric group-wise INT4 (``block_size=32``) logit error before vs after folding.

    Evaluates FP32 reference logits on ``model_orig``, verifies FP32 invariance on
    ``model_folded``, and measures INT4 fake-quantized logit MSE & max-abs diff on
    both models without mutating either input model.
    """
    m_orig = copy.deepcopy(model_orig).eval()
    m_fold = copy.deepcopy(model_folded).eval()

    out_orig_fp32 = m_orig(sample_img, sample_ids, return_aux=False, return_mtp=False)
    out_fold_fp32 = m_fold(sample_img, sample_ids, return_aux=False, return_mtp=False)
    logits_ref = out_orig_fp32["logits"].float()
    logits_fold_fp32 = out_fold_fp32["logits"].float()

    fp32_max_abs_diff = float((logits_fold_fp32 - logits_ref).abs().max().item())
    fp32_mse = float((logits_fold_fp32 - logits_ref).pow(2).mean().item())

    qat_orig = QATInt4ControllerV2(m_orig, block_size=block_size, enabled=True)
    qat_fold = QATInt4ControllerV2(m_fold, block_size=block_size, enabled=True)
    try:
        logits_orig_q = m_orig(sample_img, sample_ids, return_aux=False, return_mtp=False)["logits"].float()
        logits_fold_q = m_fold(sample_img, sample_ids, return_aux=False, return_mtp=False)["logits"].float()
    finally:
        qat_orig.disable()
        qat_fold.disable()

    err_orig = logits_orig_q - logits_ref
    err_fold = logits_fold_q - logits_ref

    orig_int4_mse = float(err_orig.pow(2).mean().item())
    folded_int4_mse = float(err_fold.pow(2).mean().item())
    orig_int4_max_diff = float(err_orig.abs().max().item())
    folded_int4_max_diff = float(err_fold.abs().max().item())

    return {
        "block_size": int(block_size),
        "fp32_max_abs_diff": fp32_max_abs_diff,
        "fp32_mse": fp32_mse,
        "orig_int4_mse": orig_int4_mse,
        "folded_int4_mse": folded_int4_mse,
        "orig_int4_max_diff": orig_int4_max_diff,
        "folded_int4_max_diff": folded_int4_max_diff,
        "mse_reduction_ratio": orig_int4_mse / max(folded_int4_mse, 1e-12),
        "max_diff_reduction_ratio": orig_int4_max_diff / max(folded_int4_max_diff, 1e-12),
    }


__all__ = [
    "hadamard_matrix",
    "fold_hadamard_v2",
    "measure_int4_quant_error",
]
