"""UniOMR v2 — Iter-11 Training & Hybrid v1 -> v2 Distillation Harness.

Trains ``UniOMRv2Net`` (Anisotropic FPN + Hybrid Axial/Merger/2D-RoPE Encoder +
DeepStack GQA MTP-4 Decoder) with:

  1. Model Presets & Warm-Start:
     - ``--preset small_45m|edge85m|tiny``
     - ``--init-from`` (resume full v2 checkpoint)
     - ``--init-encoder-from`` (load encoder weights from ``pretrain_ijepa_rism_v2.py``
       with ``strict=False``, logging missing/unexpected keys)
     - Automatic Axial-Only Selective Checkpointing (``--grad-checkpoint``) using
       ``uni_omr_v2.py``'s built-in thresholds (``axial`` always checkpointed,
       ``global``/``dec`` uncheckpointed at ``B=2``).

  2. Hybrid ``v1 -> v2`` Distillation (Scout 2 Blueprint):
     - Offline Top-K Teacher Logit Cache (``TopKTeacherCache``): ``fp16`` values +
       ``uint8`` token indices (``K=8``) in flat ``.npy`` memmap files
       (``--precompute-teacher-cache`` + ``--teacher-cache-dir``) for Phase-1
       Clean -> Noisy KD.
     - Online Frozen ``v1`` BF16 Teacher (``--teacher-ckpt``): Generalized JSD
       (``--kd-beta 0.5``, ``--kd-alpha 0.35``, ``--kd-temp 2.0``, cosine-annealed
       over last 25% of training) between ``v1`` teacher and ``v2`` student logits.

  3. Optimizer & WSD Schedule (Scout 5 Blueprint):
     - ``--optimizer muon|adamw``: routes 2D Transformer matrices to native
       ``torch.optim.Muon(adjust_lr_fn="match_rms_adamw", momentum=0.95,
       nesterov=True, ns_steps=5)`` and 4D convs/embeddings/norms/biases/aux heads
       to ``AdamW(fused=True on CUDA)`` (clean fallback to ``AdamW`` on CPU).
     - ``--lr-schedule wsd|cosine`` (``--warmup-steps``, ``--wsd-decay-frac``,
       ``--min-lr-ratio``).

  4. Full Loss Suite + QAT + EMA + 5-Min Atomic Autosave:
     - Supervised CE + MTP-4 + Router saliency BCE + Layout Aux (``SysY``,
       ``SysCount``, ``StaffSeg``) + Sobel-HOG (``--stroke-hog-weight 0.15``) +
       optional KD JSD + CASS (``--cass-prob``).
     - ``QATInt4ControllerV2`` (``--qat-int4``, ``--qat-int4-last-frac 0.15``,
       ``--qat-int4-block-size 32``).
     - ``GPUEmaShadow`` (``--gpu-ema-decay 0.997``) + ``--save-every-minutes 5.0``
       atomic ``.tmp -> os.replace`` saving BOTH ``uni_omr_v2_latest.pt`` and
       ``uni_omr_v2_ema_latest.pt``.
"""

from __future__ import annotations

import argparse
from collections import deque
import contextlib
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

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
    VOCAB_SIZE,
    AsyncThreadPrefetcher,
    GPUEmaShadow,
    MultiScaleBatchSampler,
    RismStylePool,
    StratifiedFullPageSampler,
    StripDomainMixer,
    UniFullPageDataset,
    UniOMRConfig,
    UniOMRNet,
    UniStripDataset,
    _fast_image_is_landscape,
    _images_to_device_float,
    _infinite_loader,
    _lr_for_step,
    _make_synthetic_batch,
    _strip_group_key,
    classify_fullpage_bucket,
    collate_uni_batch,
    compute_token_f1,
    decode_uni,
    discover_fullpage_pairs,
    filter_fullpage_pairs_by_systems,
    discover_strip_pairs,
    eval_dataset,
    levenshtein_distance,
    split_train_val,
    unpack_collated_batch,
)
from uni_omr_v2 import (  # noqa: E402
    QATInt4ControllerV2,
    UniOMRv2Config,
    UniOMRv2Net,
    compute_stroke_hog_loss,
)


# ---------------------------------------------------------------------------
# 1. Model Presets & Warm-Start Helpers
# ---------------------------------------------------------------------------

def uni_omr_v2_small_45m(**overrides: Any) -> UniOMRv2Config:
    """Return ``UniOMRv2Config`` for the ~43M ``small_45m`` preset."""
    return UniOMRv2Config.small_45m(**overrides)


def uni_omr_v2_edge85m(**overrides: Any) -> UniOMRv2Config:
    """Return ``UniOMRv2Config`` for the ~81M ``edge85m`` preset."""
    return UniOMRv2Config.edge85m(**overrides)


def uni_omr_v2_tiny(**overrides: Any) -> UniOMRv2Config:
    """Return a tiny CPU-fast ``UniOMRv2Config`` for smoke tests & unit verification."""
    params: Dict[str, Any] = dict(
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_axial_layers=1,
        n_global_layers=1,
        n_dec_layers=1,
        ffn_mult=2.0,
        axial_ffn_mult=2.0,
        backbone_channels=(16, 32, 48, 64),
        backbone_blocks=(1, 1, 1, 1),
        backbone_expand=2.0,
        router_hidden=32,
        max_mem_tokens=256,
        mtp_k=4,
    )
    params.update(overrides)
    return UniOMRv2Config(**params)


def build_v2_config(preset: str, **overrides: Any) -> UniOMRv2Config:
    """Build a ``UniOMRv2Config`` from ``small_45m``, ``edge85m``, or ``tiny``."""
    p = str(preset).strip().lower()
    if p == "small_45m":
        return uni_omr_v2_small_45m(**overrides)
    if p == "edge85m":
        return uni_omr_v2_edge85m(**overrides)
    if p == "tiny":
        return uni_omr_v2_tiny(**overrides)
    raise ValueError(f"Unknown preset={preset!r}; expected one of: small_45m, edge85m, tiny")


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unsupported checkpoint payload type: {type(ckpt)!r}")


def load_v2_checkpoint(
    model: UniOMRv2Net,
    ckpt_path: Path | str,
    encoder_only: bool = False,
) -> Dict[str, Any]:
    """Load full v2 or I-JEPA encoder-only weights with ``strict=False`` and shape safety.

    When ``encoder_only=True``, only ``backbone.*`` and ``encoder.*`` keys from the
    checkpoint are loaded into ``model`` (compatible with ``pretrain_ijepa_rism_v2.py``).
    """
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    raw_sd = _extract_state_dict(ck)
    model_sd = model.state_dict()

    filtered_sd: Dict[str, torch.Tensor] = {}
    shape_skipped: List[str] = []
    for k, v in raw_sd.items():
        if encoder_only and not (k.startswith("backbone.") or k.startswith("encoder.")):
            continue
        if k in model_sd and tuple(v.shape) != tuple(model_sd[k].shape):
            shape_skipped.append(f"{k}:{tuple(v.shape)}!={tuple(model_sd[k].shape)}")
            continue
        filtered_sd[k] = v

    inc = model.load_state_dict(filtered_sd, strict=False)
    tag = "init-encoder-from" if encoder_only else "init-from"
    print(
        f"[UniOMRv2 {tag}] loaded {ckpt_path} "
        f"(loaded={len(filtered_sd)}, missing={len(inc.missing_keys)}, "
        f"unexpected={len(inc.unexpected_keys)}, shape_skipped={len(shape_skipped)})",
        flush=True,
    )
    if inc.missing_keys:
        preview = ", ".join(inc.missing_keys[:6])
        suffix = f", ... (+{len(inc.missing_keys) - 6} more)" if len(inc.missing_keys) > 6 else ""
        print(f"  [UniOMRv2 {tag}] missing keys: {preview}{suffix}", flush=True)
    if inc.unexpected_keys:
        preview = ", ".join(inc.unexpected_keys[:6])
        suffix = f", ... (+{len(inc.unexpected_keys) - 6} more)" if len(inc.unexpected_keys) > 6 else ""
        print(f"  [UniOMRv2 {tag}] unexpected keys: {preview}{suffix}", flush=True)
    if shape_skipped:
        print(f"  [UniOMRv2 {tag}] shape_skipped: {shape_skipped[:6]}", flush=True)

    return {
        "loaded_count": len(filtered_sd),
        "missing_keys": list(inc.missing_keys),
        "unexpected_keys": list(inc.unexpected_keys),
        "shape_skipped": shape_skipped,
    }


def load_v1_teacher(
    teacher_ckpt: Path | str,
    device: torch.device,
    use_bf16: bool = False,
) -> UniOMRNet:
    """Load a frozen ``UniOMRNet`` (v1) teacher from checkpoint in eval mode.

    When ``use_bf16=True`` on CUDA, casts frozen weights to ``torch.bfloat16`` to cut
    teacher weight memory bandwidth by 2x; always stays ``float32`` on CPU.
    """
    ck = torch.load(str(teacher_ckpt), map_location="cpu", weights_only=True)
    raw_cfg = ck.get("config") or ck.get("cfg") or {}
    if isinstance(raw_cfg, UniOMRConfig):
        t_cfg = raw_cfg
    else:
        valid_fields = set(UniOMRConfig.__dataclass_fields__.keys())
        t_cfg = UniOMRConfig(**{k: v for k, v in dict(raw_cfg).items() if k in valid_fields})
    teacher = UniOMRNet(t_cfg)
    sd = _extract_state_dict(ck)
    inc = teacher.load_state_dict(sd, strict=False)
    cast_bf16 = bool(use_bf16 and device.type == "cuda")
    if cast_bf16:
        teacher.to(device=device, dtype=torch.bfloat16)
    else:
        teacher.to(device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(
        f"[UniOMRv2 teacher-v1] loaded frozen v1 teacher from {teacher_ckpt} "
        f"(d_model={t_cfg.d_model}, dtype={'bfloat16' if cast_bf16 else 'float32'}, "
        f"missing={len(inc.missing_keys)}, unexpected={len(inc.unexpected_keys)})",
        flush=True,
    )
    return teacher


# ---------------------------------------------------------------------------
# 2. Hybrid v1 -> v2 Distillation: Generalized JSD + Offline Top-K Cache
# ---------------------------------------------------------------------------

def kd_alpha_for_step(
    step: int,
    total_steps: int,
    base_alpha: float,
    anneal_last_frac: float = 0.25,
) -> float:
    """Cosine-anneal ``kd_alpha`` from ``base_alpha`` to ``0.0`` over the last ``anneal_last_frac``."""
    alpha = float(max(0.0, base_alpha))
    if alpha <= 0.0:
        return 0.0
    frac = float(max(0.0, min(1.0, anneal_last_frac)))
    if frac <= 0.0 or total_steps <= 1:
        return alpha
    anneal_steps = max(1, int(round(total_steps * frac)))
    anneal_start = max(1, total_steps - anneal_steps)
    if step <= anneal_start:
        return alpha
    prog = float(step - anneal_start) / float(max(1, anneal_steps))
    prog = min(1.0, max(0.0, prog))
    return alpha * 0.5 * (1.0 + math.cos(math.pi * prog))


class _GeneralizedJSDAutograd(torch.autograd.Function):
    """Memory- & bandwidth-efficient exact Generalized JSD ``G-JSD_beta(P_T || P_S) * tau^2``.

    Computes forward loss and exact analytic ``dL/dz_S`` in a single ``no_grad`` pass with
    in-place FP32 buffer reuse, saving only one ``(B, T, V)`` gradient tensor in
    ``student_logits.dtype`` (eliminating 10+ FP32 intermediate autograd tensors).
    Optionally gates KD to tokens where teacher CE (at ``tau=1``) ``<= teacher_ce_cap``
    using ``O(B*T)`` memory (``logsumexp - gather`` on the existing FP32 teacher buffer).
    """

    @staticmethod
    def forward(
        ctx: Any,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        beta: float,
        temp: float,
        ignore_index: int,
        teacher_ce_cap: float = 0.0,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tau = float(max(1e-4, temp))
        inv_tau = 1.0 / tau
        b = float(max(0.0, min(1.0, beta)))
        ce_cap = float(max(0.0, teacher_ce_cap))
        need_grad = bool(ctx.needs_input_grad[0])

        with torch.no_grad():
            t_buf = teacher_logits.detach().to(torch.float32, copy=True)

            base_valid = targets != ignore_index
            if ce_cap > 0.0:
                safe_tgt = torch.where(base_valid, targets, torch.zeros_like(targets))
                t_lse = torch.logsumexp(t_buf, dim=-1)
                t_tgt = t_buf.gather(dim=-1, index=safe_tgt.unsqueeze(-1)).squeeze(-1)
                teacher_ce = t_lse - t_tgt
                gate_mask = base_valid & (teacher_ce <= ce_cap)
                base_count = base_valid.to(torch.float32).sum().clamp(min=1.0)
                gate_frac = gate_mask.to(torch.float32).sum() / base_count
                valid = gate_mask.to(torch.float32)
                del safe_tgt, t_lse, t_tgt, teacher_ce, gate_mask
            else:
                gate_frac = student_logits.new_ones((), dtype=torch.float32)
                valid = base_valid.to(torch.float32)
            denom = valid.sum().clamp(min=1.0)
            if sample_weights is not None:
                sw = sample_weights.detach().to(device=valid.device, dtype=torch.float32).reshape(-1, 1)
                eff_valid = valid * sw
            else:
                eff_valid = valid

            if tau != 1.0:
                t_buf.mul_(inv_tau)
            log_p_t = F.log_softmax(t_buf, dim=-1)
            del t_buf

            s_buf = student_logits.detach().to(torch.float32, copy=True)
            if tau != 1.0:
                s_buf.mul_(inv_tau)
            log_p_s = F.log_softmax(s_buf, dim=-1)
            del s_buf

            if b <= 1e-6:
                # Pure reverse KL: KL(P_S || P_T)
                p_s = log_p_s.exp()
                diff = log_p_s.sub_(log_p_t)
                del log_p_t
                per_tok = (p_s * diff).sum(dim=-1)
                if need_grad:
                    diff.sub_(per_tok.unsqueeze(-1))
                    diff.mul_(p_s)
                    diff.mul_((eff_valid * (tau / denom)).unsqueeze(-1))
                    grad_s = diff.to(student_logits.dtype)
                del p_s, diff
            elif b >= 1.0 - 1e-6:
                # Pure forward KL: KL(P_T || P_S)
                p_t = log_p_t.exp()
                per_tok = (p_t * (log_p_t - log_p_s)).sum(dim=-1)
                del log_p_t
                if need_grad:
                    p_s = log_p_s.exp_()
                    p_s.sub_(p_t)
                    p_s.mul_((eff_valid * (tau / denom)).unsqueeze(-1))
                    grad_s = p_s.to(student_logits.dtype)
                    del p_s
                del p_t, log_p_s
            else:
                log_b = math.log(b)
                log_1mb = math.log(1.0 - b)
                log_m = torch.logaddexp(log_p_t + log_b, log_p_s + log_1mb)

                p_t = log_p_t.exp()
                log_p_t.sub_(log_m)
                kl_t = (p_t * log_p_t).sum(dim=-1)
                del p_t, log_p_t

                p_s = log_p_s.exp()
                log_p_s.sub_(log_m)
                del log_m
                kl_s = (p_s * log_p_s).sum(dim=-1)

                per_tok = b * kl_t + (1.0 - b) * kl_s

                if need_grad:
                    # Exact analytic gradient: d(G-JSD_beta * tau^2)/dz_S
                    # = (1 - beta) * tau * P_S * ((log_p_s - log_m) - KL(P_S || M))
                    log_p_s.sub_(kl_s.unsqueeze(-1))
                    log_p_s.mul_(p_s)
                    log_p_s.mul_((eff_valid * ((1.0 - b) * tau / denom)).unsqueeze(-1))
                    grad_s = log_p_s.to(student_logits.dtype)
                del p_s, log_p_s

            loss = ((per_tok * eff_valid).sum() / denom) * (tau * tau)

        if need_grad:
            ctx.save_for_backward(grad_s)
        ctx.mark_non_differentiable(gate_frac)
        return loss.to(student_logits.dtype), gate_frac

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor, _grad_gate: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], None, None, None, None, None, None, None]:
        (grad_s,) = ctx.saved_tensors
        return grad_s * grad_output.to(grad_s.dtype), None, None, None, None, None, None, None


def compute_generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 0.5,
    temp: float = 2.0,
    ignore_index: int = PAD_ID,
    teacher_ce_cap: float = 0.0,
    return_gate_frac: bool = False,
    sample_weights: Optional[torch.Tensor] = None,
) -> Any:
    """Generalized Jensen-Shannon Divergence ``G-JSD_beta(P_T || P_S) * temp^2``.

    Numerically stable in float32 using ``log_softmax`` + ``logaddexp`` with analytic
    backward (eliminates intermediate ``(B, T, V)`` autograd tensors).
    Masked over ``targets != ignore_index`` (and optionally ``teacher_ce <= teacher_ce_cap``)
    with optional per-sample ``sample_weights`` ``(B,)`` and zero CUDA-CPU syncs.
    """
    loss, gate_frac = _GeneralizedJSDAutograd.apply(
        student_logits,
        teacher_logits,
        targets,
        float(beta),
        float(temp),
        int(ignore_index),
        float(teacher_ce_cap),
        sample_weights,
    )
    if return_gate_frac:
        return loss, gate_frac
    return loss


def compute_sparse_topk_jsd_loss(
    student_logits: torch.Tensor,
    teacher_topk_vals: torch.Tensor,
    teacher_topk_idx: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 0.5,
    temp: float = 2.0,
    valid_mask: Optional[torch.Tensor] = None,
    ignore_index: int = PAD_ID,
    teacher_ce_cap: float = 0.0,
    return_gate_frac: bool = False,
    sample_weights: Optional[torch.Tensor] = None,
) -> Any:
    """Sparse Top-K Generalized JSD over cached ``(B, T, K)`` teacher logits & indices.

    Gathers student logits at ``teacher_topk_idx`` (``uint8``/``long``), normalizes both
    student and teacher over the Top-K support at temperature ``temp``, and computes
    masked ``G-JSD_beta * temp^2`` with zero CUDA-CPU syncs.
    """
    tau = float(max(1e-4, temp))
    b = float(max(0.0, min(1.0, beta)))

    nb = student_logits.device.type == "cuda"
    idx_long = teacher_topk_idx.to(device=student_logits.device, dtype=torch.long, non_blocking=nb)
    s_topk = torch.gather(student_logits.float(), dim=-1, index=idx_long)
    t_topk = teacher_topk_vals.to(device=student_logits.device, dtype=torch.float32, non_blocking=nb)

    log_p_s = F.log_softmax(s_topk / tau, dim=-1)
    log_p_t = F.log_softmax(t_topk / tau, dim=-1)
    p_s = log_p_s.exp()
    p_t = log_p_t.exp()

    if b <= 1e-6:
        per_tok = (p_s * (log_p_s - log_p_t)).sum(dim=-1)
    elif b >= 1.0 - 1e-6:
        per_tok = (p_t * (log_p_t - log_p_s)).sum(dim=-1)
    else:
        log_b = math.log(b)
        log_1mb = math.log(1.0 - b)
        log_m = torch.logaddexp(log_b + log_p_t, log_1mb + log_p_s)
        kl_t = (p_t * (log_p_t - log_m)).sum(dim=-1)
        kl_s = (p_s * (log_p_s - log_m)).sum(dim=-1)
        per_tok = b * kl_t + (1.0 - b) * kl_s

    mask = targets != ignore_index
    if valid_mask is not None:
        mask = mask & valid_mask.to(device=student_logits.device, dtype=torch.bool, non_blocking=nb)
    ce_cap = float(max(0.0, teacher_ce_cap))
    if ce_cap > 0.0:
        with torch.no_grad():
            tgt_match = idx_long == targets.unsqueeze(-1)
            in_topk = tgt_match.any(dim=-1)
            t_lse_k = torch.logsumexp(t_topk, dim=-1)
            t_tgt_k = (t_topk * tgt_match.to(torch.float32)).sum(dim=-1)
            t_ce_k = t_lse_k - t_tgt_k
            base_mask = mask
            mask = base_mask & in_topk & (t_ce_k <= ce_cap)
            gate_frac = mask.to(torch.float32).sum() / base_mask.to(torch.float32).sum().clamp(min=1.0)
    else:
        gate_frac = student_logits.new_ones((), dtype=torch.float32)
    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    if sample_weights is not None:
        sw = sample_weights.detach().to(
            device=student_logits.device, dtype=torch.float32, non_blocking=nb
        ).reshape(-1, 1)
        eff_mask_f = mask_f * sw
    else:
        eff_mask_f = mask_f
    loss = (per_tok * eff_mask_f).sum() / denom
    out_loss = (loss * (tau * tau)).to(student_logits.dtype)
    if return_gate_frac:
        return out_loss, gate_frac
    return out_loss


class TopKTeacherCache:
    """Offline flat ``.npy`` memmap cache of Top-K teacher logits (``fp16``) + indices (``uint8``).

    On-disk layout inside ``cache_dir``:
      - ``topk_vals.npy``: shape ``(N_total_tokens, K)``, dtype ``float16``
      - ``topk_idx.npy``:  shape ``(N_total_tokens, K)``, dtype ``uint8``
      - ``index.json``:    metadata + ``{sample_id: [offset, length]}``
    """

    VALS_NAME: str = "topk_vals.npy"
    IDX_NAME: str = "topk_idx.npy"
    INDEX_NAME: str = "index.json"

    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / self.INDEX_NAME
        if not meta_path.is_file():
            raise FileNotFoundError(f"TopKTeacherCache index not found: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.k: int = int(meta["k"])
        self.vocab_size: int = int(meta.get("vocab_size", VOCAB_SIZE))
        self.entries: Dict[str, Tuple[int, int]] = {
            str(sid): (int(pair[0]), int(pair[1]))
            for sid, pair in meta["entries"].items()
        }
        self._vals_mm: np.ndarray = np.load(self.cache_dir / self.VALS_NAME, mmap_mode="r")
        self._idx_mm: np.ndarray = np.load(self.cache_dir / self.IDX_NAME, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.entries)

    def has(self, sample_id: str) -> bool:
        return str(sample_id) in self.entries

    def get(self, sample_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(vals_fp16 (L, K), idx_u8 (L, K))`` memmap views for ``sample_id``."""
        off, length = self.entries[str(sample_id)]
        return (
            self._vals_mm[off : off + length],
            self._idx_mm[off : off + length],
        )

    def get_batch(
        self,
        sample_ids: Sequence[str],
        seq_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Fetch padded ``(vals, idx, valid_mask, n_hits)`` tensors for a mini-batch."""
        B = len(sample_ids)
        T = int(seq_len)
        vals = np.zeros((B, T, self.k), dtype=np.float16)
        idx = np.zeros((B, T, self.k), dtype=np.uint8)
        mask = np.zeros((B, T), dtype=np.bool_)
        n_hits = 0
        for i, sid in enumerate(sample_ids):
            entry = self.entries.get(str(sid))
            if entry is None:
                continue
            off, length = entry
            copy_len = min(T, length)
            if copy_len > 0:
                vals[i, :copy_len] = self._vals_mm[off : off + copy_len]
                idx[i, :copy_len] = self._idx_mm[off : off + copy_len]
                mask[i, :copy_len] = True
                n_hits += 1
        vals_t = torch.from_numpy(vals).to(device=device, non_blocking=(device.type == "cuda"))
        idx_t = torch.from_numpy(idx).to(device=device, non_blocking=(device.type == "cuda"))
        mask_t = torch.from_numpy(mask).to(device=device, non_blocking=(device.type == "cuda"))
        return vals_t, idx_t, mask_t, n_hits

    def close(self) -> None:
        """Release underlying memmap handles (important on Windows before tempdir cleanup)."""
        for mm in (self._vals_mm, self._idx_mm):
            mmap_obj = getattr(mm, "_mmap", None)
            if mmap_obj is not None:
                try:
                    mmap_obj.close()
                except Exception:
                    pass
        self._vals_mm = np.empty((0, self.k), dtype=np.float16)
        self._idx_mm = np.empty((0, self.k), dtype=np.uint8)

    @classmethod
    def write(
        cls,
        cache_dir: Path | str,
        records: Sequence[Tuple[str, np.ndarray, np.ndarray]],
        k: int = 8,
        vocab_size: int = VOCAB_SIZE,
        teacher_ckpt: str = "",
    ) -> "TopKTeacherCache":
        """Write ``(sample_id, vals_fp16 (L, K), idx_u8 (L, K))`` records to flat ``.npy`` files."""
        if vocab_size > 256:
            raise ValueError(f"uint8 TopKTeacherCache requires vocab_size <= 256, got {vocab_size}")
        out = Path(cache_dir)
        out.mkdir(parents=True, exist_ok=True)

        total_tokens = sum(int(v.shape[0]) for _, v, _ in records)
        all_vals = np.empty((total_tokens, int(k)), dtype=np.float16)
        all_idx = np.empty((total_tokens, int(k)), dtype=np.uint8)
        entries: Dict[str, List[int]] = {}

        offset = 0
        for sid, v_arr, i_arr in records:
            v_np = np.asarray(v_arr, dtype=np.float16)
            i_np = np.asarray(i_arr, dtype=np.uint8)
            if v_np.ndim != 2 or i_np.ndim != 2 or v_np.shape != i_np.shape or v_np.shape[1] != int(k):
                raise ValueError(
                    f"Record {sid!r} shape mismatch: vals={v_np.shape}, idx={i_np.shape}, expected (*, {k})"
                )
            length = int(v_np.shape[0])
            all_vals[offset : offset + length] = v_np
            all_idx[offset : offset + length] = i_np
            entries[str(sid)] = [offset, length]
            offset += length

        vals_path = out / cls.VALS_NAME
        idx_path = out / cls.IDX_NAME
        meta_path = out / cls.INDEX_NAME

        vals_tmp = out / (cls.VALS_NAME + ".tmp.npy")
        idx_tmp = out / (cls.IDX_NAME + ".tmp.npy")
        meta_tmp = out / (cls.INDEX_NAME + ".tmp")

        np.save(vals_tmp, all_vals)
        np.save(idx_tmp, all_idx)
        os.replace(vals_tmp, vals_path)
        os.replace(idx_tmp, idx_path)

        payload = {
            "version": 1,
            "k": int(k),
            "vocab_size": int(vocab_size),
            "total_tokens": int(total_tokens),
            "n_samples": len(entries),
            "teacher_ckpt": str(teacher_ckpt),
            "entries": entries,
        }
        meta_tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(meta_tmp, meta_path)
        return cls(out)


@torch.inference_mode()
def extract_topk_from_logits(
    logits: torch.Tensor,
    k: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract Top-K ``(vals_fp16, idx_u8)`` arrays from ``(T, V)`` or ``(B, T, V)`` logits."""
    vals, idx = torch.topk(logits.detach().float(), k=int(k), dim=-1, largest=True, sorted=True)
    return (
        vals.to(torch.float16).cpu().numpy(),
        idx.to(torch.uint8).cpu().numpy(),
    )


@torch.inference_mode()
def precompute_teacher_cache(
    teacher: UniOMRNet,
    datasets: Sequence[Tuple[str, Any]],
    cache_dir: Path | str,
    device: torch.device,
    k: int = 8,
    batch_size: int = 4,
    use_bf16: bool = False,
    teacher_ckpt: str = "",
) -> TopKTeacherCache:
    """Run frozen v1 teacher on clean datasets and write an offline ``TopKTeacherCache``."""
    teacher.eval()
    records: List[Tuple[str, np.ndarray, np.ndarray]] = []
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else contextlib.nullcontext()
    )

    for ds_name, ds in datasets:
        loader = DataLoader(
            ds,
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_uni_batch,
            pin_memory=(device.type == "cuda"),
        )
        t0 = time.perf_counter()
        n_before = len(records)
        t_first_param = next(teacher.parameters(), None)
        t_dtype = t_first_param.dtype if t_first_param is not None else torch.float32
        for batch in loader:
            images = _images_to_device_float(batch["images"], device, non_blocking=(device.type == "cuda"))
            if images.dtype != t_dtype:
                images = images.to(dtype=t_dtype)
            seq = batch["seq"].to(device, non_blocking=(device.type == "cuda"))
            tgt_in = seq[:, :-1]
            with amp_ctx:
                out = teacher(images, tgt_in, return_aux=False, return_mtp=False)
            vals_b, idx_b = extract_topk_from_logits(out["logits"], k=k)
            for i, sid in enumerate(batch["sample_ids"]):
                eff_len = max(0, int(batch["orig_seq_len"][i]) - 1)
                records.append((str(sid), vals_b[i, :eff_len], idx_b[i, :eff_len]))
        dt = max(1e-6, time.perf_counter() - t0)
        print(
            f"[TopKTeacherCache] precomputed {ds_name}: {len(records) - n_before} samples in {dt:.1f}s",
            flush=True,
        )

    cache = TopKTeacherCache.write(
        cache_dir=cache_dir,
        records=records,
        k=k,
        vocab_size=VOCAB_SIZE,
        teacher_ckpt=teacher_ckpt,
    )
    print(
        f"[TopKTeacherCache] wrote {len(cache)} samples (K={cache.k}) to {cache_dir}",
        flush=True,
    )
    return cache


# ---------------------------------------------------------------------------
# 3. Optimizer (Muon + AdamW) & WSD Schedule (Scout 5 Blueprint)
# ---------------------------------------------------------------------------

_MUON_TRANSFORMER_PREFIXES: Tuple[str, ...] = (
    "encoder.axial_layers.",
    "encoder.patch_merger.",
    "encoder.global_layers.",
    "deepstack_fusions.",
    "dec_layers.",
)


def partition_muon_adamw_params(
    model: nn.Module,
) -> Tuple[List[Tuple[str, nn.Parameter]], List[Tuple[str, nn.Parameter]]]:
    """Partition trainable parameters into ``(muon_named, adamw_named)``.

    Routes 2D Transformer block matrices (``axial_layers``, ``patch_merger``,
    ``global_layers``, ``deepstack_fusions``, ``dec_layers``) to ``Muon``, and
    all 4D convs (``backbone``, ``router.saliency_net``, ``staff_seg_head``,
    ``stroke_hog_head``), embeddings (``tok_emb``), output projection
    (``lm_head``), positional/router projections, norms, biases, and aux/MTP
    heads to ``AdamW``. Guarantees 0 dropped and 0 duplicated parameters.
    """
    muon_named: List[Tuple[str, nn.Parameter]] = []
    adamw_named: List[Tuple[str, nn.Parameter]] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_transformer_2d = (
            p.ndim == 2
            and name.startswith(_MUON_TRANSFORMER_PREFIXES)
        )
        if is_transformer_2d:
            muon_named.append((name, p))
        else:
            adamw_named.append((name, p))

    return muon_named, adamw_named


def _is_no_decay_param(name: str, p: nn.Parameter) -> bool:
    """Return True for 1D/0D params (norms, biases, scalar gates) and token/pos embeddings."""
    if p.ndim < 2:
        return True
    low = name.lower()
    return (
        "tok_emb" in low
        or "embed" in low
        or "pos_emb" in low
        or "pos_embed" in low
        or low.endswith(".bias")
        or low == "bias"
    )


class HybridOptimizerV2:
    """Unified wrapper over ``Muon + AdamW`` (or pure ``AdamW`` fallback)."""

    def __init__(
        self,
        optimizers: Sequence[Tuple[torch.optim.Optimizer, float]],
        mode: str,
        muon_param_names: Sequence[str] = (),
        adamw_param_names: Sequence[str] = (),
        no_decay_param_names: Sequence[str] = (),
    ) -> None:
        self._opts: List[Tuple[torch.optim.Optimizer, float]] = list(optimizers)
        self.mode = mode
        self.muon_param_names = list(muon_param_names)
        self.adamw_param_names = list(adamw_param_names)
        self.no_decay_param_names = list(no_decay_param_names)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt, _ in self._opts:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt, _ in self._opts:
            opt.step()

    def set_lr(self, lr_scale: float) -> None:
        """Set each optimizer's LR to ``base_lr * lr_scale``."""
        s = float(lr_scale)
        for opt, base_lr in self._opts:
            for pg in opt.param_groups:
                pg["lr"] = base_lr * s


def build_optimizer_v2(
    model: nn.Module,
    optimizer_name: str = "muon",
    lr: float = 3e-4,
    muon_lr: Optional[float] = None,
    weight_decay: float = 0.01,
    device_type: str = "cpu",
    allow_cpu_muon: bool = False,
    no_decay_1d: bool = False,
) -> HybridOptimizerV2:
    """Build native ``torch.optim.Muon`` (2D Transformer matrices) + ``AdamW`` (rest).

    On CUDA with ``optimizer_name="muon"``, uses ``torch.optim.Muon(
    adjust_lr_fn="match_rms_adamw", momentum=0.95, nesterov=True, ns_steps=5)``
    plus ``AdamW(fused=True)``. Cleanly falls back to ``AdamW`` on CPU (unless
    ``allow_cpu_muon=True``) or when ``optimizer_name="adamw"``.
    When ``no_decay_1d=True``, routes all ``ndim < 2`` params (norms, biases, scalar
    gates) and embedding params into a ``weight_decay=0.0`` AdamW param group.
    """
    muon_named, adamw_named = partition_muon_adamw_params(model)
    muon_params = [p for _, p in muon_named]
    muon_names = [n for n, _ in muon_named]
    adamw_names = [n for n, _ in adamw_named]

    def _build_adamw_groups(
        named_params: Sequence[Tuple[str, nn.Parameter]],
    ) -> Tuple[Any, List[str]]:
        if not no_decay_1d:
            return [p for _, p in named_params], []
        decay_p: List[nn.Parameter] = []
        nodecay_p: List[nn.Parameter] = []
        nodecay_names: List[str] = []
        for n, p in named_params:
            if _is_no_decay_param(n, p):
                nodecay_p.append(p)
                nodecay_names.append(n)
            else:
                decay_p.append(p)
        groups: List[Dict[str, Any]] = []
        if decay_p:
            groups.append({"params": decay_p, "weight_decay": float(weight_decay)})
        if nodecay_p:
            groups.append({"params": nodecay_p, "weight_decay": 0.0})
        return groups, nodecay_names

    def _make_adamw(param_or_groups: Any, cur_lr: float) -> torch.optim.Optimizer:
        if device_type == "cuda":
            try:
                return torch.optim.AdamW(param_or_groups, lr=cur_lr, weight_decay=weight_decay, fused=True)
            except (RuntimeError, TypeError):
                pass
        return torch.optim.AdamW(param_or_groups, lr=cur_lr, weight_decay=weight_decay)

    want_muon = (
        str(optimizer_name).strip().lower() == "muon"
        and (device_type == "cuda" or allow_cpu_muon)
        and hasattr(torch.optim, "Muon")
        and len(muon_params) > 0
    )
    if want_muon:
        eff_muon_lr = float(muon_lr) if muon_lr is not None else float(lr)
        try:
            opt_muon = torch.optim.Muon(
                muon_params,
                lr=eff_muon_lr,
                weight_decay=weight_decay,
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
                adjust_lr_fn="match_rms_adamw",
            )
            opts: List[Tuple[torch.optim.Optimizer, float]] = [(opt_muon, eff_muon_lr)]
            nodecay_names: List[str] = []
            if adamw_named:
                adamw_groups, nodecay_names = _build_adamw_groups(adamw_named)
                opts.append((_make_adamw(adamw_groups, float(lr)), float(lr)))
            return HybridOptimizerV2(
                opts,
                mode="muon+adamw",
                muon_param_names=muon_names,
                adamw_param_names=adamw_names,
                no_decay_param_names=nodecay_names,
            )
        except (RuntimeError, TypeError):
            pass

    all_named = list(muon_named) + list(adamw_named)
    all_groups, all_nodecay = _build_adamw_groups(all_named)
    return HybridOptimizerV2(
        [(_make_adamw(all_groups, float(lr)), float(lr))],
        mode="adamw",
        muon_param_names=[],
        adamw_param_names=muon_names + adamw_names,
        no_decay_param_names=all_nodecay,
    )


# ---------------------------------------------------------------------------
# 4. Atomic 5-Min Autosave & Checkpoint Helpers
# ---------------------------------------------------------------------------

def atomic_save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically save a PyTorch checkpoint via ``.tmp`` -> ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, path)


def atomic_save_v2_latest(
    out_dir: Path,
    model: UniOMRv2Net,
    cfg: UniOMRv2Config,
    step: int,
    best_val_ser: float = float("inf"),
    gpu_ema: Optional[GPUEmaShadow] = None,
    best_ema_ser: float = float("inf"),
) -> Tuple[Path, Optional[Path]]:
    """Atomically save BOTH ``uni_omr_v2_latest.pt`` and ``uni_omr_v2_ema_latest.pt``."""
    cfg_dict = asdict(cfg)
    latest_path = out_dir / "uni_omr_v2_latest.pt"
    atomic_save_checkpoint(
        latest_path,
        {
            "cfg": cfg_dict,
            "config": cfg_dict,
            "vocab": list(data.VOCAB),
            "step": int(step),
            "best_val_ser": float(best_val_ser),
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
    )
    ema_path: Optional[Path] = None
    if gpu_ema is not None:
        ema_path = out_dir / "uni_omr_v2_ema_latest.pt"
        atomic_save_checkpoint(
            ema_path,
            {
                "cfg": cfg_dict,
                "config": cfg_dict,
                "vocab": list(data.VOCAB),
                "step": int(step),
                "best_val_ser": float(best_ema_ser),
                "ema_decay": float(gpu_ema.decay),
                "model": gpu_ema.cpu_state_dict(model),
            },
        )
    return latest_path, ema_path


def ensure_v2_best_checkpoints(
    out_dir: Path,
    model: UniOMRv2Net,
    cfg: UniOMRv2Config,
    step: int,
    best_val_ser: float = float("inf"),
    gpu_ema: Optional[GPUEmaShadow] = None,
    best_ema_ser: float = float("inf"),
) -> Tuple[Path, Path]:
    """Guarantee ``uni_omr_v2_best.pt`` and ``uni_omr_v2_ema_best.pt`` exist on disk before exit.

    Never overwrites an existing best checkpoint written by an earlier improving eval;
    writes atomic fallback checkpoints from the final weights/EMA if missing so downstream
    stages (e.g. ``--init-from .../uni_omr_v2_ema_best.pt``) can never hit ``FileNotFoundError``.
    """
    cfg_dict = asdict(cfg)
    best_path = out_dir / "uni_omr_v2_best.pt"
    ema_best_path = out_dir / "uni_omr_v2_ema_best.pt"

    if not best_path.is_file():
        atomic_save_checkpoint(
            best_path,
            {
                "step": int(step),
                "val_ser": float(best_val_ser),
                "cfg": cfg_dict,
                "config": cfg_dict,
                "vocab": list(data.VOCAB),
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            },
        )
        print(f"  [handoff] saved fallback {best_path.name} at step {step}", flush=True)

    if not ema_best_path.is_file():
        ema_state = (
            gpu_ema.cpu_state_dict(model)
            if gpu_ema is not None
            else {k: v.detach().cpu() for k, v in model.state_dict().items()}
        )
        atomic_save_checkpoint(
            ema_best_path,
            {
                "step": int(step),
                "val_ser": float(best_ema_ser),
                "ema_decay": float(gpu_ema.decay) if gpu_ema is not None else 0.0,
                "cfg": cfg_dict,
                "config": cfg_dict,
                "vocab": list(data.VOCAB),
                "model": ema_state,
            },
        )
        print(f"  [handoff] saved fallback {ema_best_path.name} at step {step}", flush=True)

    return best_path, ema_best_path


# ---------------------------------------------------------------------------
# 5. Single Training Forward + Loss Helper (Used by Train Loop & Tests)
# ---------------------------------------------------------------------------

_HW_SAMPLE_ID_MARKERS: Tuple[str, ...] = (
    "real_hw",
    "debussy",
    "hw_strip",
    "hw-sim",
    "hw_sim",
    "hwsim",
    "rhw_",
    "hw-",
)

_HW_DOMAIN_VALUES: Tuple[str, ...] = (
    "hw",
    "real_hw",
    "synth_hw",
    "real_hw_archival",
    "real_hw_stitched",
    "hw_strip",
)

_STATE_TOKEN_PREFIXES: Tuple[str, ...] = (
    "time-",
    "key",
    "clef-",
    "tuplet-start-",
)

_STATE_TOKEN_WEIGHT_CACHE: Dict[Tuple[int, float, torch.device, torch.dtype], torch.Tensor] = {}


def _is_hw_sample_id(sample_id: str) -> bool:
    low = str(sample_id).replace("\\", "/").lower()
    return any(tok in low for tok in _HW_SAMPLE_ID_MARKERS)


def _is_hw_domain(domain: str) -> bool:
    dom = str(domain or "").strip().lower()
    if not dom:
        return False
    return dom in _HW_DOMAIN_VALUES or dom.startswith(("hw_", "hw-", "real_hw", "synth_hw"))


def build_state_token_weights(
    vocab_size: int,
    state_weight: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a cached 1D ``(vocab_size,)`` per-token CE weight tensor upweighting musical state tokens."""
    w_val = float(state_weight)
    key = (int(vocab_size), w_val, device, dtype)
    cached = _STATE_TOKEN_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    weights = torch.ones((int(vocab_size),), device=device, dtype=dtype)
    if w_val != 1.0:
        state_ids = [
            int(tid)
            for tid, tok in data.ID_TO_TOKEN.items()
            if int(tid) < int(vocab_size) and str(tok).startswith(_STATE_TOKEN_PREFIXES)
        ]
        if state_ids:
            weights[state_ids] = w_val
    _STATE_TOKEN_WEIGHT_CACHE[key] = weights
    return weights


@contextlib.contextmanager
def _skip_dead_aux_heads(
    model: nn.Module,
    *,
    skip_sys_y: bool = False,
    skip_sys_count: bool = False,
    skip_staff_seg: bool = False,
    skip_stroke_hog: bool = False,
) -> Iterator[None]:
    """Temporarily bypass unused aux head submodules during forward without mutating state_dict."""
    saved: List[Tuple[str, Any]] = []
    mods = getattr(model, "_modules", None)
    if isinstance(mods, dict):
        for attr, should_skip in (
            ("sys_y_head", skip_sys_y),
            ("sys_count_head", skip_sys_count),
            ("staff_seg_head", skip_staff_seg),
            ("stroke_hog_head", skip_stroke_hog),
        ):
            if should_skip and attr in mods and mods[attr] is not None:
                saved.append((attr, mods[attr]))
                mods[attr] = None
    try:
        yield
    finally:
        if isinstance(mods, dict):
            for attr, orig in saved:
                mods[attr] = orig


def forward_and_loss_v2(
    model: UniOMRv2Net,
    images: torch.Tensor,
    seq: torch.Tensor,
    *,
    teacher_v1: Optional[UniOMRNet] = None,
    teacher_cache: Optional[TopKTeacherCache] = None,
    sample_ids: Optional[Sequence[str]] = None,
    domains: Optional[Sequence[str]] = None,
    kd_alpha: float = 0.0,
    kd_beta: float = 0.5,
    kd_temp: float = 2.0,
    kd_teacher_ce_cap: float = 0.0,
    kd_hw_scale: float = 1.0,
    cass_prob: float = 0.0,
    kind: str = "fullpage",
    mtp_loss_weight: float = 0.30,
    saliency_loss_weight: float = 0.10,
    sys_y_loss_weight: float = 0.25,
    sys_count_loss_weight: float = 0.15,
    staff_seg_loss_weight: float = 0.20,
    stroke_hog_weight: float = 0.15,
    label_smoothing: float = 0.05,
    state_token_loss_weight: float = 1.0,
    sys_y_target: Optional[torch.Tensor] = None,
    sys_y_mask: Optional[torch.Tensor] = None,
    staff_seg_target: Optional[torch.Tensor] = None,
    use_bf16: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Run one forward + full loss computation (Supervised + Layout Aux + Sobel-HOG + KD + CASS)."""
    dev = device if device is not None else images.device
    tgt_in = seq[:, :-1]
    tgt_out = seq[:, 1:]

    # Derive exact system count target from <sys> separator occurrences (0 syncs).
    sys_count_target = (seq == SYS_ID).sum(dim=1).to(torch.float32) + 1.0

    # Determine which aux heads actually have active targets & weights so dead heads are skipped.
    need_sys_y = (sys_y_target is not None) and (sys_y_loss_weight > 0.0)
    need_sys_count = sys_count_loss_weight > 0.0
    need_staff_seg = (staff_seg_target is not None) and (staff_seg_loss_weight > 0.0)
    need_stroke_hog = stroke_hog_weight > 0.0
    need_any_enc_aux = need_sys_count or need_staff_seg or need_stroke_hog

    amp_ctx = (
        torch.autocast(device_type=dev.type, dtype=torch.bfloat16)
        if (use_bf16 and dev.type == "cuda")
        else contextlib.nullcontext()
    )

    with amp_ctx:
        with _skip_dead_aux_heads(
            model,
            skip_sys_y=not need_sys_y,
            skip_sys_count=not need_sys_count,
            skip_staff_seg=not need_staff_seg,
            skip_stroke_hog=not need_stroke_hog,
        ):
            if cass_prob > 0.0 and kind != "strip" and tgt_in.shape[1] > 1:
                mem_levels, aux_enc, router_info = model.encode_image(
                    images, return_aux=need_any_enc_aux
                )
                mem_det = tuple(m.detach() for m in mem_levels)
                with torch.no_grad():
                    draft_logits, _, _ = model.decode_tokens(
                        tgt_in, mem_det, return_mtp=False, return_aux=False, apply_tok_dropout=False
                    )
                    pred_ids = draft_logits.argmax(dim=-1)
                    tail_in = tgt_in[:, 1:]
                    cass_mask = (
                        (torch.rand_like(tail_in, dtype=torch.float32) < cass_prob)
                        & (tail_in != PAD_ID)
                        & (tail_in != BOS_ID)
                        & (pred_ids[:, :-1] != EOS_ID)
                    )
                    dec_in = torch.cat(
                        [tgt_in[:, :1], torch.where(cass_mask, pred_ids[:, :-1], tail_in)],
                        dim=1,
                    )
                logits, mtp_logits, sys_y_pred = model.decode_tokens(
                    dec_in,
                    mem_levels,
                    return_mtp=(model.cfg.mtp_k > 1),
                    return_aux=need_sys_y,
                    apply_tok_dropout=False,
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
            else:
                out = model(
                    images,
                    tgt_in,
                    return_aux=(need_any_enc_aux or need_sys_y),
                    return_mtp=(model.cfg.mtp_k > 1),
                )

        losses = model.compute_loss(
            out,
            tgt_out,
            sys_y_target=sys_y_target,
            sys_y_mask=sys_y_mask,
            sys_count_target=sys_count_target,
            staff_seg_target=staff_seg_target,
            images=images,
            mtp_loss_weight=mtp_loss_weight,
            saliency_loss_weight=saliency_loss_weight,
            sys_y_loss_weight=sys_y_loss_weight,
            sys_count_loss_weight=sys_count_loss_weight,
            staff_seg_loss_weight=staff_seg_loss_weight,
            stroke_hog_loss_weight=stroke_hog_weight,
            label_smoothing=label_smoothing,
        )

        # Optional per-token CE upweighting on musical state tokens (time-*, key*, clef-*, tuplet-start-*).
        # Default 1.0 leaves weight=None (zero overhead & 100% bit-identical to unweighted CE).
        st_w = float(state_token_loss_weight)
        if st_w > 1.0:
            s_logits: torch.Tensor = out["logits"]
            V_ce = int(s_logits.shape[-1])
            ls_ce = float(max(0.0, min(0.5, label_smoothing)))
            state_tok_weights = build_state_token_weights(
                V_ce, st_w, device=s_logits.device, dtype=s_logits.dtype
            )
            weighted_ce = F.cross_entropy(
                s_logits.reshape(-1, V_ce),
                tgt_out.reshape(-1),
                weight=state_tok_weights,
                ignore_index=PAD_ID,
                label_smoothing=ls_ce,
            )
            losses["ce_loss"] = weighted_ce
            losses["loss"] = (
                weighted_ce
                + mtp_loss_weight * losses["mtp_loss"]
                + saliency_loss_weight * losses["saliency_loss"]
                + sys_y_loss_weight * losses["sys_y_loss"]
                + sys_count_loss_weight * losses["sys_count_loss"]
                + staff_seg_loss_weight * losses["staff_seg_loss"]
                + stroke_hog_weight * losses["stroke_hog_loss"]
            )

        # Hybrid KD: prefer offline Top-K cache hits, otherwise fall back to online frozen v1 teacher.
        # Optionally downweight/skip KD on HW samples (--kd-hw-scale) using both batch domains and
        # sample_ids, and gate per-token KD by teacher cross-entropy (--kd-teacher-ce-cap).
        # Completely skip teacher forward when eff_kd_alpha <= 0.
        eff_kd_alpha = float(kd_alpha)
        hw_scale = float(max(0.0, kd_hw_scale))
        sample_kd_weights: Optional[torch.Tensor] = None
        if eff_kd_alpha > 0.0 and hw_scale != 1.0:
            B_kd = int(images.shape[0])
            if kind == "hw_strip":
                eff_kd_alpha *= hw_scale
            else:
                per_sample_scales: List[float] = []
                for idx in range(B_kd):
                    dom_i = domains[idx] if (domains is not None and idx < len(domains)) else ""
                    sid_i = sample_ids[idx] if (sample_ids is not None and idx < len(sample_ids)) else ""
                    is_hw_i = _is_hw_domain(dom_i) or _is_hw_sample_id(sid_i)
                    per_sample_scales.append(hw_scale if is_hw_i else 1.0)
                n_hw_batch = sum(1 for s in per_sample_scales if s != 1.0)
                if n_hw_batch == B_kd:
                    eff_kd_alpha *= hw_scale
                elif n_hw_batch > 0:
                    sample_kd_weights = torch.tensor(per_sample_scales, device=dev, dtype=torch.float32)

        kd_loss = out["logits"].new_zeros(())
        kd_gate_frac = out["logits"].new_ones((), dtype=torch.float32)
        if eff_kd_alpha > 0.0:
            used_cache = False
            if teacher_cache is not None and sample_ids is not None:
                c_vals, c_idx, c_mask, n_hits = teacher_cache.get_batch(
                    sample_ids, seq_len=tgt_out.shape[1], device=dev
                )
                if n_hits > 0:
                    kd_loss, kd_gate_frac = compute_sparse_topk_jsd_loss(
                        out["logits"],
                        c_vals,
                        c_idx,
                        tgt_out,
                        beta=kd_beta,
                        temp=kd_temp,
                        valid_mask=c_mask,
                        teacher_ce_cap=kd_teacher_ce_cap,
                        return_gate_frac=True,
                        sample_weights=sample_kd_weights,
                    )
                    used_cache = True
                del c_vals, c_idx, c_mask
            if (not used_cache) and teacher_v1 is not None:
                t_first_param = next(teacher_v1.parameters(), None)
                t_dtype = t_first_param.dtype if t_first_param is not None else images.dtype
                with torch.inference_mode():
                    t_images = images.to(dtype=t_dtype) if images.dtype != t_dtype else images
                    t_out = teacher_v1(t_images, tgt_in, return_aux=False, return_mtp=False)
                    t_logits_inf = t_out["logits"]
                    del t_out, t_images
                # _GeneralizedJSDAutograd consumes teacher_logits inside no_grad without saving it,
                # so no extra (B, T, V) clone is needed.
                kd_loss, kd_gate_frac = compute_generalized_jsd_loss(
                    out["logits"],
                    t_logits_inf,
                    tgt_out,
                    beta=kd_beta,
                    temp=kd_temp,
                    teacher_ce_cap=kd_teacher_ce_cap,
                    return_gate_frac=True,
                    sample_weights=sample_kd_weights,
                )
                del t_logits_inf

            losses["loss"] = losses["loss"] + eff_kd_alpha * kd_loss

        losses["kd_loss"] = kd_loss
        losses["kd_gate_frac"] = kd_gate_frac

    return losses, out


# ---------------------------------------------------------------------------
# 5b. Fast Exact Self-Speculative MTP-K Decoding & In-Training Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_mtp_speculative_v2(
    model: UniOMRv2Net,
    images: torch.Tensor,
    max_len: int = 256,
    keep_ratio_override: Optional[float] = None,
    return_info: bool = False,
) -> Any:
    """Exact self-speculative MTP-K greedy decoding for ``UniOMRv2Net``.

    Produces output bit-identical to ``model.generate()`` while verifying up to
    ``mtp_k`` tokens per decoder pass (1 GPU-CPU sync per block), yielding ~2.5-3x
    faster validation decoding on full pages. Falls back cleanly to ``model.generate()``
    when ``mtp_k <= 1`` or ``max_len <= 1``.
    """
    mtp_heads = getattr(model, "mtp_heads", None)
    n_extra = len(mtp_heads) if mtp_heads is not None else 0
    if n_extra <= 0 or max_len <= 1 or not hasattr(model, "precompute_cross_kv"):
        return model.generate(
            images,
            max_len=max_len,
            keep_ratio_override=keep_ratio_override,
            return_info=return_info,
        )

    mem, _, router_info = model.encode_image(
        images, keep_ratio_override=keep_ratio_override, return_aux=False
    )
    B = images.shape[0]
    device = images.device
    bos = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
    past_kvs: List[Any] = model.precompute_cross_kv(mem)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    pad_tok = torch.full((B,), PAD_ID, dtype=torch.long, device=device)
    out_tokens: List[torch.Tensor] = [bos]

    # Step 0 (prefill BOS): get exact y_1 + (K-1) draft tokens for y_2..y_K.
    logits, mtp_logits, past_kvs = model.decode_tokens(
        bos,
        memory=None,
        return_mtp=True,
        return_aux=False,
        past_kvs=past_kvs,
        use_cache=True,
    )
    tok0 = logits[:, -1, :].argmax(dim=-1)
    out_tokens.append(tok0.unsqueeze(1))
    finished = finished | (tok0 == EOS_ID)
    emitted = 1

    if emitted < max_len and not bool(finished.all()):
        drafts = [h[:, -1, :].argmax(dim=-1) for h in mtp_logits]
        verify_ids = torch.stack([tok0, *drafts], dim=1)  # (B, K)

        while emitted < max_len:
            rem = max_len - emitted
            if verify_ids.shape[1] > rem:
                verify_ids = verify_ids[:, :rem]
            m = int(verify_ids.shape[1])
            past_len = int(past_kvs[0][0][0].shape[2])

            logits, mtp_logits, new_past_kvs = model.decode_tokens(
                verify_ids,
                memory=None,
                return_mtp=True,
                return_aux=False,
                past_kvs=past_kvs,
                use_cache=True,
            )
            preds = logits.argmax(dim=-1)  # (B, m)

            # Determine consecutive accepted draft count across all unfinished items (1 sync).
            fin_cum = finished.unsqueeze(1) | ((preds == EOS_ID).cumsum(dim=1) > 0)
            all_fin = fin_cum.all(dim=0)  # (m,)
            if m > 1:
                step_ok = (preds[:, :-1] == verify_ids[:, 1:]) | fin_cum[:, :-1]
                all_ok = step_ok.all(dim=0) & (~all_fin[:-1])
                n_accept = int(all_ok.cumprod(dim=0).sum().item())
            else:
                n_accept = 0

            n_emit = min(n_accept + 1, rem)
            for j in range(n_emit):
                tok_j = torch.where(finished, pad_tok, preds[:, j])
                out_tokens.append(tok_j.unsqueeze(1))
                finished = finished | (tok_j == EOS_ID)
            emitted += n_emit

            if emitted >= max_len or bool(finished.all()):
                break

            keep_len = past_len + n_emit
            if keep_len < past_len + m:
                past_kvs = [
                    ((sk[:, :, :keep_len, :], sv[:, :, :keep_len, :]), cross_kv)
                    for (sk, sv), cross_kv in new_past_kvs
                ]
            else:
                past_kvs = new_past_kvs

            last_pos = n_emit - 1
            last_tok = out_tokens[-1].squeeze(1)
            next_drafts = [h[:, last_pos, :].argmax(dim=-1) for h in mtp_logits]
            verify_ids = torch.stack([last_tok, *next_drafts], dim=1)

    seq = torch.cat(out_tokens, dim=1)
    if return_info:
        return seq, router_info
    return seq


@torch.no_grad()
def eval_dataset_v2(
    model: UniOMRv2Net,
    val_samples: Any,
    device: torch.device,
    max_len: int = 256,
    batch_size: int = 4,
    use_bf16: bool = False,
) -> Dict[str, Any]:
    """Evaluate ``UniOMRv2Net`` using exact MTP-K speculative decoding (schema-identical to ``eval_dataset``)."""
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
        loader: Any = DataLoader(
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
                pred_ids, rinfo = generate_mtp_speculative_v2(
                    model, images, max_len=max_len, return_info=True
                )
        else:
            pred_ids, rinfo = generate_mtp_speculative_v2(
                model, images, max_len=max_len, return_info=True
            )
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
# 6. Full Training Loop + Archival HW Stratification & Val Split
# ---------------------------------------------------------------------------

STRATIFIED_FULLPAGE_WEIGHTS_V2: Dict[str, float] = {
    "legacy_printed": 0.30,
    "multi_renderer_printed": 0.26,
    "rare_printed": 0.06,
    "synth_hw": 0.22,
    "real_hw_stitched": 0.06,
    "real_hw_archival": 0.10,
}

_STITCHED_HW_MARKERS: Tuple[str, ...] = (
    "1s2s",
    "stitched",
    "concat",
    "synth_page",
    "composed",
)


def classify_fullpage_bucket_v2(
    img_path: Any,
    domain: str = "",
    split_archival_hw: bool = False,
) -> str:
    """Classify a full-page sample, optionally splitting ``real_hw`` into archival vs stitched."""
    base = classify_fullpage_bucket(img_path, domain)
    if split_archival_hw and base == "real_hw":
        p_str = str(img_path).replace("\\", "/").lower()
        dom = str(domain or "").strip().lower()
        if any(tok in p_str or tok in dom for tok in _STITCHED_HW_MARKERS):
            return "real_hw_stitched"
        return "real_hw_archival"
    return base


class ArchivalStratifiedFullPageSampler(StratifiedFullPageSampler):
    """6-bucket stratified full-page sampler splitting ``real_hw_archival`` vs ``real_hw_stitched``."""

    def __init__(
        self,
        pairs: Sequence[Tuple[Any, ...]],
        weights: Optional[Dict[str, float]] = None,
        epoch_size: Optional[int] = None,
    ) -> None:
        self.num_samples = len(pairs)
        self.epoch_size = int(epoch_size) if epoch_size is not None else self.num_samples
        raw_weights = dict(STRATIFIED_FULLPAGE_WEIGHTS_V2 if weights is None else weights)
        self.buckets: Dict[str, List[int]] = {k: [] for k in raw_weights}
        for idx, entry in enumerate(pairs):
            dom = str(entry[2]) if len(entry) > 2 else ""
            bname = classify_fullpage_bucket_v2(entry[0], dom, split_archival_hw=True)
            self.buckets.setdefault(bname, []).append(idx)
        active = {
            k: raw_weights.get(k, 0.0)
            for k, idxs in self.buckets.items()
            if idxs and raw_weights.get(k, 0.0) > 0.0
        }
        w_sum = sum(active.values())
        if w_sum <= 0.0:
            non_empty = [k for k, idxs in self.buckets.items() if idxs]
            self.active_weights: Dict[str, float] = {k: 1.0 / max(1, len(non_empty)) for k in non_empty}
        else:
            self.active_weights = {k: v / w_sum for k, v in active.items()}


def split_fullpage_train_val_with_hw(
    fp_pairs: Sequence[Tuple[Any, ...]],
    val_count: int = 16,
    val_real_hw_count: int = 0,
    seed: int = 42,
) -> Tuple[List[Tuple[Any, ...]], List[Tuple[Any, ...]], List[Tuple[Any, ...]]]:
    """Split full-page pairs into ``(train, val, real_hw_val)`` with piece-level grouping.

    When ``val_real_hw_count <= 0``, preserves exact legacy ``split_train_val`` behavior
    and returns an empty ``real_hw_val`` list.
    """
    if not fp_pairs:
        return [], [], []
    if int(val_real_hw_count) <= 0:
        fp_tr, fp_va = split_train_val(
            fp_pairs,
            key_fn=lambda p: data.piece_key(p[0].stem),
            val_count=val_count,
            seed=seed,
        )
        return fp_tr, fp_va, []

    real_hw_rows: List[Tuple[Any, ...]] = []
    other_rows: List[Tuple[Any, ...]] = []
    for entry in fp_pairs:
        dom = str(entry[2]) if len(entry) > 2 else ""
        if classify_fullpage_bucket(entry[0], dom) == "real_hw":
            real_hw_rows.append(entry)
        else:
            other_rows.append(entry)

    other_tr, other_va = split_train_val(
        other_rows,
        key_fn=lambda p: data.piece_key(p[0].stem),
        val_count=val_count,
        seed=seed,
    ) if other_rows else ([], [])

    hw_tr, hw_va = split_train_val(
        real_hw_rows,
        key_fn=lambda p: data.piece_key(p[0].stem),
        val_count=int(val_real_hw_count),
        seed=seed + 17,
    ) if real_hw_rows else ([], [])

    return other_tr + hw_tr, other_va, hw_va


def _build_datasets_and_iters(
    args: argparse.Namespace,
    cfg: UniOMRv2Config,
    device: torch.device,
    strip_batch_size: int,
    prefetch_threads: int,
) -> Dict[str, Any]:
    """Discover and construct train/val datasets & iterators reusing ``uni_omr.py``."""
    st_tr: List[Tuple[Path, Path]] = []
    st_va: List[Tuple[Path, Path]] = []
    hw_tr: List[Tuple[Path, Path]] = []
    hw_va: List[Tuple[Path, Path]] = []
    fp_tr: List[Tuple[Any, ...]] = []
    fp_va: List[Tuple[Any, ...]] = []
    fp_hw_va: List[Tuple[Any, ...]] = []

    if args.strip_dirs:
        strip_pairs = discover_strip_pairs(args.strip_dirs)
        if strip_pairs:
            st_tr, st_va = split_train_val(
                strip_pairs,
                key_fn=lambda p: _strip_group_key(p[0].stem),
                val_count=args.val_count,
                seed=args.seed,
            )

    if args.hw_strip_dirs:
        hw_pairs = discover_strip_pairs(args.hw_strip_dirs)
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
        max_sys_cap = int(getattr(args, "max_systems_per_page", 16))
        if fp_pairs and max_sys_cap > 0:
            fp_pairs, n_dropped_sys = filter_fullpage_pairs_by_systems(
                fp_pairs, max_systems_per_page=max_sys_cap
            )
            if n_dropped_sys > 0:
                print(
                    f"[v2-data] filtered {n_dropped_sys} full-page rows with > {max_sys_cap} systems "
                    f"({len(fp_pairs)} kept)",
                    flush=True,
                )
        if fp_pairs:
            fp_tr, fp_va, fp_hw_va = split_fullpage_train_val_with_hw(
                fp_pairs,
                val_count=args.val_count,
                val_real_hw_count=int(getattr(args, "val_real_hw_count", 0)),
                seed=args.seed,
            )

    # Shared piece-level val split prevents cross-modality train/val leaks.
    shared_val_keys: set[str] = set()
    for p in st_va:
        shared_val_keys.add(_strip_group_key(p[0].stem))
    for p in hw_va:
        shared_val_keys.add(_strip_group_key(p[0].stem))
    for p in fp_va:
        shared_val_keys.add(data.piece_key(p[0].stem))
    for p in fp_hw_va:
        shared_val_keys.add(data.piece_key(p[0].stem))

    if shared_val_keys:
        if st_tr:
            f_st = [p for p in st_tr if _strip_group_key(p[0].stem) not in shared_val_keys]
            if f_st:
                st_tr = f_st
        if hw_tr:
            f_hw = [p for p in hw_tr if _strip_group_key(p[0].stem) not in shared_val_keys]
            if f_hw:
                hw_tr = f_hw
        if fp_tr:
            f_fp = [p for p in fp_tr if data.piece_key(p[0].stem) not in shared_val_keys]
            if f_fp:
                fp_tr = f_fp

    prefetchers: List[AsyncThreadPrefetcher] = []
    strip_train_iter = None
    hw_strip_train_iter = None
    fullpage_train_iter = None
    strip_val_ds = None
    hw_strip_val_ds = None
    fullpage_val_ds = None
    fullpage_hw_val_ds = None
    clean_datasets_for_cache: List[Tuple[str, Any]] = []

    rism_style_dir = str(getattr(args, "rism_style_dir", "") or "").strip()
    rism_fda_prob = float(getattr(args, "rism_fda_prob", 0.0) or 0.0)
    rism_adain_prob = float(getattr(args, "rism_adain_prob", 0.0) or 0.0)
    rism_fda_beta = float(getattr(args, "rism_fda_beta", 0.02) or 0.02)
    rism_style_pool: Optional[RismStylePool] = None
    if rism_style_dir and (rism_fda_prob > 0.0 or rism_adain_prob > 0.0):
        rism_style_pool = RismStylePool(rism_style_dir, seed=args.seed)
        print(
            f"[UniOMRv2 data] RismStylePool: {len(rism_style_pool)} cached refs from {rism_style_dir} "
            f"(fda_p={rism_fda_prob:.2f}, beta={rism_fda_beta:.3f}, adain_p={rism_adain_prob:.2f})",
            flush=True,
        )

    if st_tr or st_va:
        st_train_ds = UniStripDataset(
            st_tr,
            strip_height=args.strip_height,
            max_strip_width=args.max_strip_width,
            max_seq_len=args.max_seq_len,
            augment=args.augment,
            aug_prob=args.aug_prob,
            kind="strip",
            return_uint8=True,
            rism_style_pool=rism_style_pool,
            rism_fda_prob=rism_fda_prob,
            rism_adain_prob=rism_adain_prob,
            rism_fda_beta=rism_fda_beta,
        )
        if st_tr:
            clean_datasets_for_cache.append((
                "printed_strips",
                UniStripDataset(
                    st_tr,
                    strip_height=args.strip_height,
                    max_strip_width=args.max_strip_width,
                    max_seq_len=args.max_seq_len,
                    augment=False,
                    kind="strip",
                    return_uint8=True,
                ),
            ))
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
            pf = AsyncThreadPrefetcher(
                st_train_ds,
                batch_size=strip_batch_size,
                shuffle=True,
                pin_memory=False,  # Never pin_memory inside background threads on Windows WDDM
                max_workers=prefetch_threads,
                prefetch_batches=3,
                collate_fn=collate_uni_batch,
                seed=args.seed + 101,
            )
            prefetchers.append(pf)
            strip_train_iter = pf
        elif len(st_tr) > 0:
            strip_train_iter = _infinite_loader(
                DataLoader(
                    st_train_ds,
                    batch_size=strip_batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    collate_fn=collate_uni_batch,
                    pin_memory=(device.type == "cuda"),
                )
            )
        print(f"[UniOMRv2 data] printed_strips: {len(st_tr)} train / {len(st_va)} val", flush=True)

    if hw_tr or hw_va:
        hw_train_ds = UniStripDataset(
            hw_tr,
            strip_height=args.strip_height,
            max_strip_width=args.max_strip_width,
            max_seq_len=args.max_seq_len,
            augment=args.augment,
            aug_prob=args.aug_prob,
            kind="hw_strip",
            return_uint8=True,
            rism_style_pool=rism_style_pool,
            rism_fda_prob=rism_fda_prob,
            rism_adain_prob=rism_adain_prob,
            rism_fda_beta=rism_fda_beta,
        )
        if hw_tr:
            clean_datasets_for_cache.append((
                "hw_strips",
                UniStripDataset(
                    hw_tr,
                    strip_height=args.strip_height,
                    max_strip_width=args.max_strip_width,
                    max_seq_len=args.max_seq_len,
                    augment=False,
                    kind="hw_strip",
                    return_uint8=True,
                ),
            ))
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
            pf = AsyncThreadPrefetcher(
                hw_train_ds,
                batch_size=strip_batch_size,
                shuffle=True,
                pin_memory=False,
                max_workers=prefetch_threads,
                prefetch_batches=3,
                collate_fn=collate_uni_batch,
                seed=args.seed + 202,
            )
            prefetchers.append(pf)
            hw_strip_train_iter = pf
        elif len(hw_tr) > 0:
            hw_strip_train_iter = _infinite_loader(
                DataLoader(
                    hw_train_ds,
                    batch_size=strip_batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    collate_fn=collate_uni_batch,
                    pin_memory=(device.type == "cuda"),
                )
            )
        print(f"[UniOMRv2 data] hw_strips: {len(hw_tr)} train / {len(hw_va)} val", flush=True)

    if fp_tr or fp_va or fp_hw_va:
        if args.stratified_fullpage and len(fp_tr) > 0:
            fp_strat_sampler: Optional[StratifiedFullPageSampler] = (
                ArchivalStratifiedFullPageSampler(fp_tr)
                if bool(getattr(args, "split_archival_hw", False))
                else StratifiedFullPageSampler(fp_tr)
            )
            print(
                f"[UniOMRv2 data] fullpage_strata={fp_strat_sampler.bucket_counts()} "
                f"weights={ {k: round(v, 3) for k, v in fp_strat_sampler.active_weights.items()} }",
                flush=True,
            )
        else:
            fp_strat_sampler = None
        fp_train_ds = UniFullPageDataset(
            fp_tr,
            fullpage_max_dim=args.fullpage_max_dim,
            max_seq_len=args.max_seq_len,
            augment=args.augment,
            aug_prob=args.aug_prob,
            multi_scale=args.multi_scale,
            return_uint8=True,
            rism_style_pool=rism_style_pool,
            rism_fda_prob=rism_fda_prob,
            rism_adain_prob=rism_adain_prob,
            rism_fda_beta=rism_fda_beta,
        )
        if fp_tr:
            clean_datasets_for_cache.append((
                "fullpage",
                UniFullPageDataset(
                    fp_tr,
                    fullpage_max_dim=args.fullpage_max_dim,
                    max_seq_len=args.max_seq_len,
                    augment=False,
                    return_uint8=True,
                ),
            ))
        fullpage_val_ds = UniFullPageDataset(
            fp_va,
            fullpage_max_dim=args.fullpage_max_dim,
            max_seq_len=args.max_seq_len,
            augment=False,
            return_uint8=True,
        ) if fp_va else None
        fullpage_hw_val_ds = UniFullPageDataset(
            fp_hw_va,
            fullpage_max_dim=args.fullpage_max_dim,
            max_seq_len=args.max_seq_len,
            augment=False,
            return_uint8=True,
        ) if fp_hw_va else None
        if prefetch_threads > 0 and args.num_workers == 0 and len(fp_tr) > 0:
            pf = AsyncThreadPrefetcher(
                fp_train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=False,
                max_workers=prefetch_threads,
                prefetch_batches=3,
                collate_fn=collate_uni_batch,
                seed=args.seed + 303,
                stratified_sampler=fp_strat_sampler,
            )
            prefetchers.append(pf)
            fullpage_train_iter = pf
        elif args.multi_scale and len(fp_tr) > 0:
            fp_bs = MultiScaleBatchSampler(
                num_samples=len(fp_train_ds),
                batch_size=args.batch_size,
                base_dim=args.fullpage_max_dim,
                shuffle=True,
                seed=args.seed + 303,
                stratified_sampler=fp_strat_sampler,
            )
            fullpage_train_iter = _infinite_loader(
                DataLoader(
                    fp_train_ds,
                    batch_sampler=fp_bs,
                    num_workers=args.num_workers,
                    collate_fn=collate_uni_batch,
                    pin_memory=(device.type == "cuda"),
                )
            )
        elif len(fp_tr) > 0:
            fullpage_train_iter = _infinite_loader(
                DataLoader(
                    fp_train_ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    collate_fn=collate_uni_batch,
                    pin_memory=(device.type == "cuda"),
                )
            )
        hw_val_note = f" / {len(fp_hw_va)} real_hw_val" if fp_hw_va else ""
        print(f"[UniOMRv2 data] fullpage: {len(fp_tr)} train / {len(fp_va)} val{hw_val_note}", flush=True)

    return {
        "strip_train_iter": strip_train_iter,
        "hw_strip_train_iter": hw_strip_train_iter,
        "fullpage_train_iter": fullpage_train_iter,
        "strip_val_ds": strip_val_ds,
        "hw_strip_val_ds": hw_strip_val_ds,
        "fullpage_val_ds": fullpage_val_ds,
        "fullpage_hw_val_ds": fullpage_hw_val_ds,
        "prefetchers": prefetchers,
        "clean_datasets_for_cache": clean_datasets_for_cache,
    }


def run_train_v2(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Force CUTLASS EFFICIENT_ATTENTION globally (covers frozen v1 teacher SDPA too;
        # avoids cuDNN 9.10's ~640ms/shape CPU plan recompilation on variable (mem, T) shapes).
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)

    cfg = build_v2_config(
        args.preset,
        grad_checkpoint=bool(args.grad_checkpoint),
        dropout=float(args.dropout),
        mixstyle_prob=float(args.mixstyle_prob),
        tok_dropout=float(args.tok_dropout),
        keep_ratio=float(args.keep_ratio),
        dynamic_routing=bool(args.dynamic_routing),
        max_mem_tokens=int(args.max_mem_tokens),
        mtp_k=int(args.mtp_k),
        use_sys_y_aux=bool(args.use_sys_y_aux and float(args.sys_y_loss_weight) > 0.0),
        use_sys_count_aux=bool(args.use_sys_count_aux and float(args.sys_count_loss_weight) > 0.0),
        use_staff_seg_aux=bool(args.use_staff_seg_aux and float(args.staff_seg_loss_weight) > 0.0),
        use_stroke_hog_head=bool(args.stroke_hog_weight > 0.0),
    )

    use_bf16 = bool(args.bf16 and device.type == "cuda")
    strip_batch_size = (
        max(1, int(args.strip_batch_size))
        if args.strip_batch_size is not None
        else max(1, int(args.batch_size))
    )
    prefetch_threads = (
        (6 if (device.type == "cuda" and not args.dry_run) else 0)
        if args.prefetch_threads is None
        else max(0, int(args.prefetch_threads))
    )

    use_real_data = (not args.dry_run) and (
        bool(args.strip_dirs)
        or bool(args.hw_strip_dirs)
        or bool(args.fullpage_dirs)
        or bool(args.fullpage_manifest)
    )

    # Precompute-only mode: build TopKTeacherCache on clean datasets and exit.
    if args.precompute_teacher_cache:
        if not args.teacher_ckpt:
            raise ValueError("--precompute-teacher-cache requires --teacher-ckpt")
        if not args.teacher_cache_dir:
            raise ValueError("--precompute-teacher-cache requires --teacher-cache-dir")
        teacher_v1 = load_v1_teacher(args.teacher_ckpt, device=device, use_bf16=use_bf16)
        ds_bundle = _build_datasets_and_iters(
            args, cfg, device, strip_batch_size=strip_batch_size, prefetch_threads=0
        )
        try:
            cache = precompute_teacher_cache(
                teacher=teacher_v1,
                datasets=ds_bundle["clean_datasets_for_cache"],
                cache_dir=args.teacher_cache_dir,
                device=device,
                k=int(args.teacher_cache_k),
                batch_size=max(1, int(args.batch_size)),
                use_bf16=use_bf16,
                teacher_ckpt=str(args.teacher_ckpt),
            )
            n_cached = len(cache)
            cache.close()
        finally:
            for pf in ds_bundle["prefetchers"]:
                pf.close()
        return {"precomputed_samples": n_cached, "cache_dir": str(args.teacher_cache_dir)}

    model = UniOMRv2Net(cfg).to(device)
    if args.init_encoder_from:
        load_v2_checkpoint(model, args.init_encoder_from, encoder_only=True)
    if args.init_from:
        load_v2_checkpoint(model, args.init_from, encoder_only=False)
    if args.channels_last:
        model.backbone.to(memory_format=torch.channels_last)
    model.train()

    # Optional online frozen v1 teacher & offline Top-K cache
    teacher_v1: Optional[UniOMRNet] = (
        load_v1_teacher(args.teacher_ckpt, device=device, use_bf16=use_bf16)
        if (args.teacher_ckpt and float(args.kd_alpha) > 0.0)
        else None
    )
    teacher_cache: Optional[TopKTeacherCache] = (
        TopKTeacherCache(args.teacher_cache_dir) if args.teacher_cache_dir else None
    )
    if teacher_cache is not None:
        print(
            f"[UniOMRv2 teacher-cache] opened {args.teacher_cache_dir} "
            f"(samples={len(teacher_cache)}, K={teacher_cache.k})",
            flush=True,
        )

    opt = build_optimizer_v2(
        model,
        optimizer_name=args.optimizer,
        lr=float(args.lr),
        muon_lr=args.muon_lr,
        weight_decay=float(args.weight_decay),
        device_type=device.type,
        no_decay_1d=bool(getattr(args, "no_decay_1d", False)),
    )

    steps = min(args.steps, 2) if args.dry_run else int(args.steps)
    grad_accum = max(1, int(args.grad_accum))
    req_strip_ga = int(getattr(args, "strip_grad_accum", 0))
    if req_strip_ga > 0:
        strip_grad_accum = req_strip_ga
    else:
        strip_grad_accum = max(1, (max(1, int(args.batch_size)) * grad_accum) // strip_batch_size)
    grad_clip = float(getattr(args, "grad_clip", 1.0))
    kd_teacher_ce_cap = float(getattr(args, "kd_teacher_ce_cap", 0.0))
    kd_hw_scale = float(getattr(args, "kd_hw_scale", 1.0))
    state_token_loss_weight = float(getattr(args, "state_token_loss_weight", 1.0))
    print(
        f"[UniOMRv2 opt] mode={opt.mode} | muon_params={len(opt.muon_param_names)} | "
        f"adamw_params={len(opt.adamw_param_names)} | no_decay_params={len(opt.no_decay_param_names)} | "
        f"lr={args.lr:.2e} | schedule={args.lr_schedule} | grad_clip={grad_clip:.2f} | "
        f"grad_accum={grad_accum} | strip_grad_accum={strip_grad_accum} (strip_bs={strip_batch_size})",
        flush=True,
    )
    log_every = max(1, int(args.log_every))
    eval_batch_size = (
        (8 if (device.type == "cuda" and not args.dry_run) else max(1, int(args.batch_size)))
        if args.eval_batch_size is None
        else max(1, int(args.eval_batch_size))
    )

    # QAT-INT4 controller (last frac of training)
    qat_ctrl: Optional[QATInt4ControllerV2] = (
        QATInt4ControllerV2(model, block_size=int(args.qat_int4_block_size), enabled=False)
        if args.qat_int4
        else None
    )
    qat_start_step = (
        steps + 1
        if args.qat_int4_last_frac <= 0.0
        else max(1, int(steps * (1.0 - float(args.qat_int4_last_frac))))
    )

    # Device-resident EMA shadow
    gpu_ema_decay = float(max(0.0, min(0.99999, args.gpu_ema_decay)))
    gpu_ema: Optional[GPUEmaShadow] = (
        GPUEmaShadow(model, decay=gpu_ema_decay, update_every=int(args.gpu_ema_every))
        if gpu_ema_decay > 0.0
        else None
    )

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None and not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    ds_bundle: Dict[str, Any] = {
        "strip_train_iter": None,
        "hw_strip_train_iter": None,
        "fullpage_train_iter": None,
        "strip_val_ds": None,
        "hw_strip_val_ds": None,
        "fullpage_val_ds": None,
        "fullpage_hw_val_ds": None,
        "prefetchers": [],
    }
    if use_real_data:
        ds_bundle = _build_datasets_and_iters(
            args, cfg, device, strip_batch_size=strip_batch_size, prefetch_threads=prefetch_threads
        )

    strip_train_iter = ds_bundle["strip_train_iter"]
    hw_strip_train_iter = ds_bundle["hw_strip_train_iter"]
    fullpage_train_iter = ds_bundle["fullpage_train_iter"]
    strip_val_ds = ds_bundle["strip_val_ds"]
    hw_strip_val_ds = ds_bundle["hw_strip_val_ds"]
    fullpage_val_ds = ds_bundle["fullpage_val_ds"]
    fullpage_hw_val_ds = ds_bundle.get("fullpage_hw_val_ds")
    prefetchers: List[AsyncThreadPrefetcher] = ds_bundle["prefetchers"]

    strip_mixer = StripDomainMixer(hw_frac=float(args.hw_frac))
    has_any_strip = (strip_train_iter is not None) or (hw_strip_train_iter is not None)
    mix_strip_domains = (
        strip_train_iter is not None
        and hw_strip_train_iter is not None
        and strip_batch_size >= 2
        and cfg.mixstyle_prob > 0.0
    )
    pr_strip_pool: deque[Dict[str, Any]] = deque()
    hw_strip_pool: deque[Dict[str, Any]] = deque()

    def _pop_strip(pool: deque[Dict[str, Any]], it: Any, fallback_kind: str) -> Dict[str, Any]:
        if not pool:
            b = next(it)
            for item in unpack_collated_batch(b, fallback_kind):
                pool.append(item)
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
                int(math.floor(step_idx * args.fullpage_frac + 1e-9))
                > int(math.floor((step_idx - 1) * args.fullpage_frac + 1e-9))
            )
        return fullpage_train_iter is not None

    def _next_batch(
        step_idx: int,
        micro_idx: int,
        want_fp: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], str, List[str]]:
        if not use_real_data:
            bs_syn = args.batch_size if want_fp else strip_batch_size
            imgs, sq, _ = _make_synthetic_batch(
                curriculum="fullpage" if want_fp else "strip",
                batch_size=bs_syn,
                seq_len=args.seq_len,
                device=device,
                step=step_idx * grad_accum + micro_idx,
                with_uv=False,
            )
            sids = [f"syn_{step_idx}_{micro_idx}_{i}" for i in range(bs_syn)]
            return imgs, sq, sids, ("fullpage" if want_fp else "strip"), [""] * bs_syn

        if want_fp and fullpage_train_iter is not None:
            b = next(fullpage_train_iter)
            imgs = _images_to_device_float(b["images"], device, non_blocking=(device.type == "cuda"))
            if args.channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            return (
                imgs,
                b["seq"].to(device, non_blocking=(device.type == "cuda")),
                list(b.get("sample_ids", [])),
                "fullpage",
                list(b.get("domains", [])),
            )

        if mix_strip_domains:
            picked: List[Dict[str, Any]] = []
            picked_domains: List[str] = []
            n_hw = 0
            for _ in range(strip_batch_size):
                if strip_mixer.next_is_hw():
                    picked.append(_pop_strip(hw_strip_pool, hw_strip_train_iter, "hw_strip"))
                    picked_domains.append("hw_strip")
                    n_hw += 1
                else:
                    picked.append(_pop_strip(pr_strip_pool, strip_train_iter, "strip"))
                    picked_domains.append("printed")
            b = collate_uni_batch(picked)
            kind = "mixed_strip" if (0 < n_hw < strip_batch_size) else ("hw_strip" if n_hw > 0 else "strip")
            raw_doms = list(b.get("domains", []))
            doms = [
                (raw_doms[i] if (i < len(raw_doms) and raw_doms[i]) else picked_domains[i])
                for i in range(len(picked_domains))
            ]
            imgs = _images_to_device_float(b["images"], device, non_blocking=(device.type == "cuda"))
            if args.channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            return (
                imgs,
                b["seq"].to(device, non_blocking=(device.type == "cuda")),
                list(b.get("sample_ids", [])),
                kind,
                doms,
            )

        if strip_train_iter is not None and hw_strip_train_iter is not None:
            if strip_mixer.next_is_hw():
                b = next(hw_strip_train_iter)
                kind = "hw_strip"
            else:
                b = next(strip_train_iter)
                kind = "strip"
            imgs = _images_to_device_float(b["images"], device, non_blocking=(device.type == "cuda"))
            if args.channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            return (
                imgs,
                b["seq"].to(device, non_blocking=(device.type == "cuda")),
                list(b.get("sample_ids", [])),
                kind,
                list(b.get("domains", [])),
            )
        it = hw_strip_train_iter if hw_strip_train_iter is not None else strip_train_iter
        kind = "hw_strip" if hw_strip_train_iter is not None else "strip"
        assert it is not None
        b = next(it)
        imgs = _images_to_device_float(b["images"], device, non_blocking=(device.type == "cuda"))
        if args.channels_last:
            imgs = imgs.to(memory_format=torch.channels_last)
        return (
            imgs,
            b["seq"].to(device, non_blocking=(device.type == "cuda")),
            list(b.get("sample_ids", [])),
            kind,
            list(b.get("domains", [])),
        )

    def _run_eval_suite(step_label: Any) -> Dict[str, Any]:
        eval_rec: Dict[str, Any] = {"step": step_label}
        val_sers: List[float] = []
        with torch.inference_mode():
            for tag, val_ds, bs in (
                ("strip_val", strip_val_ds, eval_batch_size),
                ("hw_strip_val", hw_strip_val_ds, eval_batch_size),
                ("fullpage_val", fullpage_val_ds, min(eval_batch_size, 2)),
                ("fullpage_hw_val", fullpage_hw_val_ds, min(eval_batch_size, 2)),
            ):
                if val_ds is None:
                    continue
                m = eval_dataset_v2(
                    model,
                    val_ds,
                    device=device,
                    max_len=args.eval_max_len,
                    batch_size=bs,
                    use_bf16=use_bf16,
                )
                eval_rec[tag] = m
                val_sers.append(float(m["ser"]))
                print(
                    f"  [eval {tag} @ {step_label}] SER={m['ser']:.4f} F1={m['token_f1']:.4f} "
                    f"sys_exact={m['sys_count_exact']:.3f} comp={m['mean_compression_x']:.2f}x",
                    flush=True,
                )
        eval_rec["mean_val_ser"] = round(float(np.mean(val_sers)) if val_sers else float("inf"), 6)
        return eval_rec

    history: List[Dict[str, Any]] = []
    eval_history: List[Dict[str, Any]] = []
    best_val_ser = float("inf")
    best_ema_ser = float("inf")
    save_every_sec = max(0.0, float(args.save_every_minutes) * 60.0)
    last_time_save = time.perf_counter()
    last_empty_cache_step = -25
    last_logged_hist_idx = 0
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    _acc_keys: Tuple[str, ...] = (
        "loss",
        "ce_loss",
        "mtp_loss",
        "saliency_loss",
        "stroke_hog_loss",
        "kd_loss",
        "kd_gate_frac",
    )
    pending_history_meta: List[Tuple[int, str, float, float, int, int]] = []
    pending_history_vals: List[torch.Tensor] = []

    def _flush_pending_history() -> None:
        if not pending_history_vals:
            return
        # Single batched D2H transfer for all steps accumulated since the last log.
        vals_cpu = torch.stack(pending_history_vals, dim=0).cpu().tolist()
        for meta, row_vals in zip(pending_history_meta, vals_cpu):
            s_idx, s_kind, s_lr, s_kd, s_nin, s_nout = meta
            row: Dict[str, Any] = {
                "step": s_idx,
                "kind": s_kind,
                "lr": s_lr,
                "kd_alpha": s_kd,
                "n_in": s_nin,
                "n_out": s_nout,
            }
            for k_name, v in zip(_acc_keys, row_vals):
                row[k_name] = float(v)
            history.append(row)
        pending_history_meta.clear()
        pending_history_vals.clear()

    try:
        for step in range(1, steps + 1):
            if qat_ctrl is not None and (not qat_ctrl.enabled) and step >= qat_start_step:
                qat_ctrl.enable()
                print(
                    f"[UniOMRv2 QAT-INT4] enabled on {qat_ctrl.num_layers} Linear layers "
                    f"(block_size={qat_ctrl.block_size}) at step {step}/{steps}",
                    flush=True,
                )

            cur_lr = _lr_for_step(
                step,
                steps,
                float(args.lr),
                int(args.warmup_steps),
                min_lr_ratio=float(args.min_lr_ratio),
                schedule=str(args.lr_schedule),
                wsd_decay_frac=float(args.wsd_decay_frac),
            )
            lr_scale = cur_lr / max(1e-12, float(args.lr))
            opt.set_lr(lr_scale)

            cur_kd_alpha = kd_alpha_for_step(
                step, steps, float(args.kd_alpha), anneal_last_frac=0.25
            ) if (teacher_v1 is not None or teacher_cache is not None) else 0.0

            step_is_fp = _step_is_fullpage(step)
            step_ga = grad_accum if step_is_fp else strip_grad_accum
            inv_ga = 1.0 / float(step_ga)

            opt.zero_grad(set_to_none=True)
            # Accumulate loss scalars on-device to avoid per-micro-batch GPU-CPU sync stalls.
            acc_t = torch.zeros(len(_acc_keys), device=device, dtype=torch.float32)
            last_n_in = 0
            last_n_out = 0
            last_kind = args.curriculum
            oom_hit = False

            for micro in range(step_ga):
                images, seq, sample_ids, kind, domains = _next_batch(step, micro, want_fp=step_is_fp)
                try:
                    losses, out = forward_and_loss_v2(
                        model,
                        images,
                        seq,
                        teacher_v1=teacher_v1,
                        teacher_cache=teacher_cache,
                        sample_ids=sample_ids,
                        domains=domains,
                        kd_alpha=cur_kd_alpha,
                        kd_beta=float(args.kd_beta),
                        kd_temp=float(args.kd_temp),
                        kd_teacher_ce_cap=kd_teacher_ce_cap,
                        kd_hw_scale=kd_hw_scale,
                        cass_prob=float(args.cass_prob),
                        kind=kind,
                        mtp_loss_weight=float(args.mtp_loss_weight),
                        saliency_loss_weight=float(args.saliency_loss_weight),
                        sys_y_loss_weight=float(args.sys_y_loss_weight),
                        sys_count_loss_weight=float(args.sys_count_loss_weight),
                        staff_seg_loss_weight=float(args.staff_seg_loss_weight),
                        stroke_hog_weight=float(args.stroke_hog_weight),
                        label_smoothing=float(args.label_smoothing),
                        state_token_loss_weight=state_token_loss_weight,
                        use_bf16=use_bf16,
                        device=device,
                    )
                    (losses["loss"] * inv_ga).backward()
                    with torch.no_grad():
                        acc_t.add_(
                            torch.stack([losses[k_name].detach().float() for k_name in _acc_keys]),
                            alpha=inv_ga,
                        )
                    rinfo = out["router_info"]
                    last_n_in = int(rinfo["n_in"])
                    last_n_out = int(rinfo["n_out"])
                    last_kind = kind
                    del losses, out, images, seq
                except Exception as e:
                    if device.type == "cuda" and "out of memory" in str(e).lower():
                        print(f"  [WARN OOM skip] step {step} micro {micro} kind={kind}", flush=True)
                        oom_hit = True
                        opt.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        break
                    raise

            if oom_hit:
                continue

            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            opt.step()
            if qat_ctrl is not None and qat_ctrl.enabled:
                qat_ctrl.refresh_cache()
            if (
                device.type == "cuda"
                and (step - last_empty_cache_step) >= 25
                and torch.cuda.memory_reserved(device) > int(13.8 * (1024 ** 3))
            ):
                torch.cuda.empty_cache()
                last_empty_cache_step = step

            if gpu_ema is not None:
                gpu_ema.update(model, step)

            pending_history_meta.append((step, last_kind, cur_lr, cur_kd_alpha, last_n_in, last_n_out))
            pending_history_vals.append(acc_t)
            if step == 1 or (step % log_every == 0) or step == steps:
                _flush_pending_history()
                recent_rows = history[last_logged_hist_idx:]
                last_logged_hist_idx = len(history)
                last_row = history[-1]
                strip_rows = [
                    r for r in recent_rows
                    if r.get("kind") in ("strip", "hw_strip", "mixed_strip")
                ]
                strip_note = ""
                if strip_rows:
                    mean_strip_loss = sum(float(r["loss"]) for r in strip_rows) / len(strip_rows)
                    strip_note = f" | strip_loss={mean_strip_loss:.4f} (n={len(strip_rows)})"
                kd_gate_note = (
                    f", kd_gate={last_row['kd_gate_frac']:.2f}"
                    if kd_teacher_ce_cap > 0.0
                    else ""
                )
                print(
                    f"[UniOMRv2 {'dry-run' if args.dry_run else 'train'} | {last_kind}] "
                    f"step {step}/{steps} | lr={cur_lr:.2e} | loss={last_row['loss']:.4f} "
                    f"(ce={last_row['ce_loss']:.4f}, mtp={last_row['mtp_loss']:.4f}, "
                    f"hog={last_row['stroke_hog_loss']:.4f}, kd={last_row['kd_loss']:.4f}{kd_gate_note})"
                    f"{strip_note} | tokens {last_n_in}->{last_n_out}",
                    flush=True,
                )

            # Time-based atomic rolling checkpoint save (default every 5.0 min)
            if (
                out_dir is not None
                and save_every_sec > 0.0
                and (not args.dry_run)
                and (time.perf_counter() - last_time_save) >= save_every_sec
            ):
                last_time_save = time.perf_counter()
                lp, ep = atomic_save_v2_latest(
                    out_dir,
                    model,
                    cfg,
                    step=step,
                    best_val_ser=best_val_ser,
                    gpu_ema=gpu_ema,
                    best_ema_ser=best_ema_ser,
                )
                ema_note = f" + {ep.name}" if ep is not None else ""
                print(f"  [autosave @ step {step}] saved {lp.name}{ema_note}", flush=True)

            # Periodic validation
            should_eval = (
                use_real_data
                and args.eval_every > 0
                and (step % args.eval_every == 0 or step == steps)
                and (
                    strip_val_ds is not None
                    or hw_strip_val_ds is not None
                    or fullpage_val_ds is not None
                    or fullpage_hw_val_ds is not None
                )
            )
            if should_eval:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                eval_rec = _run_eval_suite(step)
                mean_ser = float(eval_rec["mean_val_ser"])
                if out_dir is not None and mean_ser < best_val_ser:
                    best_val_ser = mean_ser
                    atomic_save_checkpoint(
                        out_dir / "uni_omr_v2_best.pt",
                        {
                            "step": step,
                            "val_ser": best_val_ser,
                            "cfg": asdict(cfg),
                            "config": asdict(cfg),
                            "vocab": list(data.VOCAB),
                            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        },
                    )
                if gpu_ema is not None:
                    with gpu_ema.swap_into(model, qat_ctrl=qat_ctrl):
                        ema_eval = _run_eval_suite(f"{step} EMA")
                    eval_rec["ema_eval"] = ema_eval
                    ema_ser = float(ema_eval["mean_val_ser"])
                    if out_dir is not None and ema_ser < best_ema_ser:
                        best_ema_ser = ema_ser
                        atomic_save_checkpoint(
                            out_dir / "uni_omr_v2_ema_best.pt",
                            {
                                "step": step,
                                "val_ser": best_ema_ser,
                                "ema_decay": gpu_ema_decay,
                                "cfg": asdict(cfg),
                                "config": asdict(cfg),
                                "vocab": list(data.VOCAB),
                                "model": gpu_ema.cpu_state_dict(model),
                            },
                        )
                eval_history.append(eval_rec)
    finally:
        _flush_pending_history()
        for pf in prefetchers:
            pf.close()
        if teacher_cache is not None:
            teacher_cache.close()

    if out_dir is not None and not args.dry_run:
        atomic_save_v2_latest(
            out_dir,
            model,
            cfg,
            step=steps,
            best_val_ser=best_val_ser,
            gpu_ema=gpu_ema,
            best_ema_ser=best_ema_ser,
        )
        ensure_v2_best_checkpoints(
            out_dir,
            model,
            cfg,
            step=steps,
            best_val_ser=best_val_ser,
            gpu_ema=gpu_ema,
            best_ema_ser=best_ema_ser,
        )
        (out_dir / "train_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if qat_ctrl is not None:
        qat_ctrl.detach()

    return {
        "steps": steps,
        "history": history,
        "eval_history": eval_history,
        "best_val_ser": best_val_ser if eval_history else None,
        "best_ema_ser": best_ema_ser if eval_history else None,
        "optimizer_mode": opt.mode,
        "strip_grad_accum": strip_grad_accum,
        "params": model.param_summary(),
    }


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="UniOMR v2 — Iter-11 Training & Hybrid v1->v2 Distillation")

    # Model preset & warm-start
    p.add_argument("--preset", choices=["small_45m", "edge85m", "tiny"], default="small_45m")
    p.add_argument("--init-from", type=str, default=None, help="Resume full UniOMRv2 checkpoint")
    p.add_argument("--init-encoder-from", type=str, default=None, help="Load I-JEPA v2 encoder weights (strict=False)")
    p.add_argument("--grad-checkpoint", action="store_true", help="Enable axial-only selective checkpointing")
    p.add_argument("--channels-last", action="store_true", help="Use channels_last on backbone")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mixstyle-prob", type=float, default=0.3)
    p.add_argument("--tok-dropout", type=float, default=0.05)
    p.add_argument("--keep-ratio", type=float, default=0.40)
    p.add_argument("--dynamic-routing", action="store_true", default=True)
    p.add_argument("--no-dynamic-routing", dest="dynamic_routing", action="store_false")
    p.add_argument("--max-mem-tokens", type=int, default=10240)
    p.add_argument("--mtp-k", type=int, default=4)
    p.add_argument("--use-sys-y-aux", action="store_true", default=True)
    p.add_argument("--no-use-sys-y-aux", dest="use_sys_y_aux", action="store_false")
    p.add_argument("--use-sys-count-aux", action="store_true", default=True)
    p.add_argument("--no-use-sys-count-aux", dest="use_sys_count_aux", action="store_false")
    p.add_argument("--use-staff-seg-aux", action="store_true", default=True)
    p.add_argument("--no-use-staff-seg-aux", dest="use_staff_seg_aux", action="store_false")

    # Hybrid v1 -> v2 distillation
    p.add_argument("--teacher-ckpt", type=str, default=None, help="Frozen v1 UniOMRNet teacher checkpoint")
    p.add_argument("--precompute-teacher-cache", action="store_true", help="Precompute offline Top-K teacher cache and exit")
    p.add_argument("--teacher-cache-dir", type=str, default=None, help="Offline TopKTeacherCache directory")
    p.add_argument("--teacher-cache-k", type=int, default=8, help="Top-K logits stored in cache (default 8)")
    p.add_argument("--kd-beta", type=float, default=0.5, help="Generalized JSD beta in [0, 1]")
    p.add_argument("--kd-alpha", type=float, default=0.35, help="KD loss weight (cosine-annealed over last 25%%)")
    p.add_argument("--kd-temp", type=float, default=2.0, help="KD softmax temperature")
    p.add_argument(
        "--kd-teacher-ce-cap",
        type=float,
        default=0.0,
        help="Mask per-token KD to tokens where teacher CE <= cap (0.0 = disabled)",
    )
    p.add_argument(
        "--kd-hw-scale",
        type=float,
        default=1.0,
        help="Optional multiplier for kd_alpha on handwritten batches (default 1.0)",
    )

    # Optimizer & WSD schedule
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--muon-lr", type=float, default=None, help="Optional separate LR for Muon 2D weights")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument(
        "--no-decay-1d",
        action="store_true",
        default=False,
        help="Exclude 1D/0D params (norms, biases) and embeddings from AdamW weight decay",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max gradient norm for clip_grad_norm_ (default 1.0; <= 0 disables clipping)",
    )
    p.add_argument("--lr-schedule", choices=["wsd", "cosine"], default="wsd")
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--wsd-decay-frac", type=float, default=0.25)
    p.add_argument("--min-lr-ratio", type=float, default=0.05)

    # Losses, CASS, QAT, EMA, Autosave
    p.add_argument("--mtp-loss-weight", type=float, default=0.30)
    p.add_argument("--saliency-loss-weight", type=float, default=0.10)
    p.add_argument("--sys-y-loss-weight", type=float, default=0.25)
    p.add_argument("--sys-count-loss-weight", type=float, default=0.15)
    p.add_argument("--staff-seg-loss-weight", type=float, default=0.20)
    p.add_argument("--stroke-hog-weight", type=float, default=0.15)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument(
        "--state-token-loss-weight",
        type=float,
        default=1.0,
        help="Per-token CE loss weight for musical state tokens (time-*, key*, clef-*, tuplet-start-*); 1.0 = unweighted",
    )
    p.add_argument("--cass-prob", type=float, default=0.0)
    p.add_argument("--qat-int4", action="store_true")
    p.add_argument("--qat-int4-last-frac", type=float, default=0.15)
    p.add_argument("--qat-int4-block-size", type=int, default=32)
    p.add_argument("--gpu-ema-decay", type=float, default=0.997)
    p.add_argument("--gpu-ema-every", type=int, default=5)
    p.add_argument("--save-every-minutes", type=float, default=5.0)

    # Data & training loop
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--strip-batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument(
        "--strip-grad-accum",
        type=int,
        default=0,
        help="Explicit grad accumulation steps for strip batches (0 = auto legacy formula)",
    )
    p.add_argument("--curriculum", choices=["strip", "fullpage", "mixed"], default="mixed")
    p.add_argument("--fullpage-frac", type=float, default=0.65)
    p.add_argument("--hw-frac", type=float, default=0.30)
    p.add_argument("--strip-dirs", nargs="*", default=[])
    p.add_argument("--hw-strip-dirs", nargs="*", default=[])
    p.add_argument("--fullpage-dirs", nargs="*", default=[])
    p.add_argument("--fullpage-manifest", type=str, default=None)
    p.add_argument("--max-systems-per-page", type=int, default=16, help="Drop full-page rows with more than this many systems (default: 16; 0 disables)")
    p.add_argument("--strip-height", type=int, default=192)
    p.add_argument("--max-strip-width", type=int, default=1024)
    p.add_argument("--fullpage-max-dim", type=int, default=1280)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--seq-len", type=int, default=64, help="Synthetic dry-run seq length")
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--aug-prob", type=float, default=0.75)
    p.add_argument(
        "--rism-style-dir",
        type=str,
        default="",
        help="Optional directory of unpaired archival RISM images for online FDA / Color-AdaIN transfer",
    )
    p.add_argument(
        "--rism-fda-prob",
        type=float,
        default=0.0,
        help="Probability of applying RISM 2D-FFT FDA transfer during online augmentation (default 0.0)",
    )
    p.add_argument(
        "--rism-adain-prob",
        type=float,
        default=0.0,
        help="Probability of applying RISM Color-AdaIN transfer during online augmentation (default 0.0)",
    )
    p.add_argument(
        "--rism-fda-beta",
        type=float,
        default=0.02,
        help="Low-frequency amplitude swap bandwidth beta for RISM FDA transfer (default 0.02)",
    )
    p.add_argument("--multi-scale", action="store_true", default=True)
    p.add_argument("--stratified-fullpage", action="store_true", default=True)
    p.add_argument(
        "--split-archival-hw",
        action="store_true",
        default=False,
        help="Split real_hw into real_hw_archival (10%%) vs real_hw_stitched (6%%) in stratified full-page sampler",
    )
    p.add_argument("--val-count", type=int, default=16)
    p.add_argument(
        "--val-real-hw-count",
        type=int,
        default=0,
        help="Deterministic count of real_hw full-page items reserved for fullpage_hw_val (0 = disabled)",
    )
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-max-len", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prefetch-threads", type=int, default=None)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_train_v2(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
