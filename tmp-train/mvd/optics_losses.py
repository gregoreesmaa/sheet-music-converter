"""Optics-aware training losses for MVD (Batch 4-B5).

Provides a self-contained, zero-overhead primitive ready to plug into
``model.py`` and ``train.py``:

:func:`logit_jsd_consistency` (Batch 4-B5):
   Symmetric Jensen-Shannon divergence between clean-view and degraded-view
   token logit distributions, with ``detach_clean=True`` by default so the
   degraded view aligns to the clean teacher without pulling clean logits down.
   Numerically stable in ``bf16`` / ``fp16`` / ``fp32`` and strictly invariant
   to values at ``pad_mask == True`` positions.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = [
    'logit_jsd_consistency',
]


def logit_jsd_consistency(
    logits_clean: torch.Tensor,
    logits_deg: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    detach_clean: bool = True,
) -> torch.Tensor:
    """Symmetric Jensen-Shannon divergence between clean and degraded logits.

    Computes::

        p = softmax(logits_clean / T)
        q = softmax(logits_deg   / T)
        m = 0.5 * (p + q)
        JSD(p || q) = 0.5 * (KL(p || m) + KL(q || m))

    in ``float32`` using ``log_softmax`` and ``logaddexp`` so ``log(m)`` never
    underflows to ``-inf`` even under ``bfloat16`` / extreme logit magnitudes.

    Args:
        logits_clean: ``[..., V]`` clean-view teacher logits.
        logits_deg: ``[..., V]`` degraded-view student logits (same shape).
        pad_mask: Optional bool tensor of shape ``logits_clean.shape[:-1]``
            where ``True`` marks PAD positions to exclude completely.
        temperature: Positive softmax temperature ``T`` (default ``1.0``).
        detach_clean: If ``True`` (default), stop gradients through
            ``logits_clean`` so only the degraded view moves toward clean.

    Returns:
        Scalar ``float32`` tensor in ``[0, ln(2)]``.
    """
    if temperature <= 0.0:
        raise ValueError(f'temperature must be > 0, got {temperature}')
    if logits_clean.shape != logits_deg.shape:
        raise ValueError(
            f'logits shape mismatch: {tuple(logits_clean.shape)} vs '
            f'{tuple(logits_deg.shape)}'
        )

    if pad_mask is not None:
        if pad_mask.shape != logits_clean.shape[:-1]:
            raise ValueError(
                f'pad_mask shape {tuple(pad_mask.shape)} must match '
                f'logits batch/time shape {tuple(logits_clean.shape[:-1])}'
            )
        valid = ~pad_mask.to(device=logits_deg.device, dtype=torch.bool)
        if not bool(valid.any()):
            return logits_deg.float().sum() * 0.0
        lc = logits_clean[valid]
        ld = logits_deg[valid]
    else:
        v = logits_clean.shape[-1]
        lc = logits_clean.reshape(-1, v)
        ld = logits_deg.reshape(-1, v)

    if detach_clean:
        lc = lc.detach()

    inv_t = 1.0 / float(temperature)
    lc = lc.float() * inv_t
    ld = ld.float() * inv_t

    log_p = F.log_softmax(lc, dim=-1)
    log_q = F.log_softmax(ld, dim=-1)
    # Exact log(0.5 * (p + q)) without underflowing to log(0).
    log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)

    p = log_p.exp()
    q = log_q.exp()

    kl_pm = (p * (log_p - log_m)).sum(dim=-1)
    kl_qm = (q * (log_q - log_m)).sum(dim=-1)
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd.clamp_min(0.0).mean()
