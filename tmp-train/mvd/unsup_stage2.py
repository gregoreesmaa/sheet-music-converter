"""Zero-corruption unsupervised learning primitives for Stage-2 MvdNet.

Implements:
  - :func:`stage2_encoder_ema_consistency_loss`: zero-pseudo-label encoder-only
    EMA consistency on weak vs. strong (photometric-only, ``geom=0``) strip
    pairs, combining masked cosine distance on encoder memory ``mem`` with
    scale-invariant ``SmoothL1Loss(log1p(gdcm_s), log1p(gdcm_w))`` when the
    GDCM counting head is enabled.
  - :func:`episodic_page_ln_tta`: context manager cloning/restoring ONLY
    ``LayerNorm`` affine ``(weight, bias)`` parameters and running ``steps`` of
    confidence-gated entropy minimization across one page's strips before
    cleanly restoring exact original weights on exit.
"""

from __future__ import annotations

from contextlib import contextmanager
import sys
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_MVD_DIR = str(Path(__file__).resolve().parent)
if _MVD_DIR not in sys.path:
    sys.path.insert(0, _MVD_DIR)

from unsup_stage1 import EMAModel  # noqa: E402

__all__ = [
    "EMAModel",
    "stage2_encoder_ema_consistency_loss",
    "episodic_page_ln_tta",
]


def _unwrap_model(m: nn.Module | EMAModel) -> nn.Module:
    return m.module if isinstance(m, EMAModel) else m


def _encode_mem_and_gdcm(
    net: nn.Module,
    strips: torch.Tensor,
    mem_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run ONLY the visual encoder (+ optional GDCM head), never the decoder."""
    if hasattr(net, "_encode_with_feats"):
        mem, _, _, _, pred_counts = net._encode_with_feats(
            strips, mem_mask=mem_mask, return_counts=True
        )
        return mem, pred_counts
    mem = net.encode(strips, mem_mask=mem_mask)
    pred_counts = None
    if bool(getattr(net, "use_gdcm", False)) and hasattr(net, "gdcm_head"):
        pred_counts = net.gdcm_head(mem, mem_mask=mem_mask)
    return mem, pred_counts


def stage2_encoder_ema_consistency_loss(
    student: nn.Module,
    teacher_ema: nn.Module | EMAModel,
    strip_weak: torch.Tensor,
    strip_strong: torch.Tensor,
    mem_mask: torch.Tensor | None = None,
    gdcm_weight: float = 1.0,
    gdcm_beta: float = 0.5,
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Zero-pseudo-label encoder-only EMA consistency loss for Stage-2 MvdNet.

    Args:
        student: Trainable ``MvdNet`` student.
        teacher_ema: Frozen EMA teacher (``EMAModel`` or ``MvdNet``).
        strip_weak: ``[B, C, H, W]`` weakly-augmented (or clean) strip batch.
        strip_strong: ``[B, C, H, W]`` strongly-augmented strip batch with
            **photometric-only** jitter (``geom=0`` so spatial coordinates
            align 1-to-1 with ``strip_weak``).
        mem_mask: Optional bool ``[B, S]`` memory key padding mask
            (``True`` = ignore/padded, matching ``MvdNet.encode``).
        gdcm_weight: Weight on the GDCM ``log1p`` SmoothL1 consistency term.
        gdcm_beta: Huber beta for GDCM ``SmoothL1Loss``.
        return_metrics: If True, return ``(loss, metrics_dict)``.

    Returns:
        Scalar loss tensor (or ``(loss, metrics)`` when ``return_metrics=True``).
    """
    t_net = _unwrap_model(teacher_ema)
    t_was_training = t_net.training
    t_net.eval()
    try:
        with torch.no_grad():
            mem_w, gdcm_w = _encode_mem_and_gdcm(t_net, strip_weak, mem_mask=mem_mask)
    finally:
        t_net.train(t_was_training)

    mem_s, gdcm_s = _encode_mem_and_gdcm(student, strip_strong, mem_mask=mem_mask)

    if mem_s.shape != mem_w.shape:
        raise ValueError(
            f"student/teacher memory shape mismatch: {tuple(mem_s.shape)} vs {tuple(mem_w.shape)}"
        )

    # Masked cosine distance on encoder memory [B, S, D].
    cos_sim = F.cosine_similarity(mem_s.float(), mem_w.float(), dim=-1, eps=1e-8)  # [B, S]
    cos_dist = (1.0 - cos_sim).clamp_min(0.0)

    if mem_mask is not None:
        if mem_mask.shape != cos_dist.shape:
            raise ValueError(
                f"expected mem_mask shape {tuple(cos_dist.shape)}, got {tuple(mem_mask.shape)}"
            )
        active = (~mem_mask).float()
        denom = active.sum()
        if float(denom.item()) > 0.0:
            loss_mem = (cos_dist * active).sum() / denom
        else:
            loss_mem = mem_s.sum() * 0.0
    else:
        loss_mem = cos_dist.mean()

    # Optional GDCM structural-count consistency in log1p space.
    loss_gdcm = mem_s.sum() * 0.0
    if gdcm_s is not None and gdcm_w is not None and float(gdcm_weight) > 0.0:
        log_s = torch.log1p(gdcm_s.float().clamp_min(0.0))
        log_w = torch.log1p(gdcm_w.float().clamp_min(0.0))
        loss_gdcm = F.smooth_l1_loss(log_s, log_w, beta=float(gdcm_beta), reduction="mean")

    total = loss_mem + float(gdcm_weight) * loss_gdcm

    if return_metrics:
        metrics = {
            "unsup_s2_total": float(total.detach().item()),
            "unsup_s2_mem_cos": float(loss_mem.detach().item()),
            "unsup_s2_gdcm": float(loss_gdcm.detach().item()),
        }
        return total, metrics
    return total


def _normalize_page_strips(
    page_strips: torch.Tensor | Sequence[torch.Tensor],
    mem_mask: torch.Tensor | Sequence[torch.Tensor | None] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    """Normalize ``page_strips`` into a list of ``([B, C, H, W], mem_mask)`` pairs."""
    if isinstance(page_strips, torch.Tensor):
        s = page_strips.unsqueeze(0) if page_strips.ndim == 3 else page_strips
        if s.ndim != 4:
            raise ValueError(f"expected 3D or 4D strip tensor, got {tuple(page_strips.shape)}")
        if isinstance(mem_mask, (list, tuple)):
            mm = mem_mask[0] if mem_mask else None
        else:
            mm = mem_mask
        return [(s, mm)]

    items: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    masks: Sequence[torch.Tensor | None] | None = (
        mem_mask if isinstance(mem_mask, (list, tuple)) else None
    )
    for idx, strip in enumerate(page_strips):
        s = strip.unsqueeze(0) if strip.ndim == 3 else strip
        if s.ndim != 4:
            raise ValueError(f"expected 3D or 4D strip tensor at index {idx}, got {tuple(strip.shape)}")
        mm = masks[idx] if (masks is not None and idx < len(masks)) else None
        items.append((s, mm))
    return items


@contextmanager
def episodic_page_ln_tta(
    model: nn.Module,
    page_strips: torch.Tensor | Sequence[torch.Tensor],
    steps: int = 2,
    lr: float = 1e-4,
    min_conf: float = 0.80,
    mem_mask: torch.Tensor | Sequence[torch.Tensor | None] | None = None,
    max_len: int = 32,
) -> Iterator[nn.Module]:
    """Episodic per-page LayerNorm Test-Time Adaptation context manager.

    Clones ONLY ``nn.LayerNorm`` affine ``(weight, bias)`` parameters (including
    ``torchvision.models.convnext.LayerNorm2d``, which subclasses ``nn.LayerNorm``),
    runs ``steps`` of confidence-gated entropy minimization on ``page_strips``,
    yields the adapted ``model``, and restores exact original weights,
    ``requires_grad`` flags, and ``training`` mode on exit.
    """
    was_training = model.training
    orig_req_grad = {p: p.requires_grad for p in model.parameters()}

    # Collect and snapshot ONLY LayerNorm affine (weight, bias) parameters.
    ln_params: list[nn.Parameter] = []
    saved_ln: list[tuple[nn.Parameter, torch.Tensor]] = []
    seen_ids: set[int] = set()
    for mod in model.modules():
        if isinstance(mod, nn.LayerNorm):
            for p in (mod.weight, mod.bias):
                if p is not None and id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    ln_params.append(p)
                    saved_ln.append((p, p.detach().clone()))

    try:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        for p in ln_params:
            p.requires_grad_(True)

        batches = _normalize_page_strips(page_strips, mem_mask=mem_mask)

        if int(steps) > 0 and ln_params and batches:
            opt = torch.optim.Adam(ln_params, lr=float(lr))
            pad_id = getattr(model, "pad_id", -1)
            bos_id = getattr(model, "bos_id", None)

            for _ in range(int(steps)):
                opt.zero_grad(set_to_none=True)
                total_ent = None
                total_count = 0

                for strips, mm in batches:
                    # Obtain short unlabeled prefix via no-grad greedy rollout.
                    with torch.no_grad():
                        if hasattr(model, "greedy") and bos_id is not None:
                            gen = model.greedy(
                                strips,
                                max_len=max(1, int(max_len)),
                                constrain=False,
                                mem_mask=mm,
                            )
                            bos = torch.full(
                                (strips.shape[0], 1),
                                int(bos_id),
                                dtype=torch.long,
                                device=strips.device,
                            )
                            if gen.shape[1] > 1:
                                tgt_in = torch.cat([bos, gen[:, :-1]], dim=1)
                            else:
                                tgt_in = bos
                        else:
                            tgt_in = None

                    # Differentiable forward pass (gradients flow ONLY to LN params).
                    if tgt_in is not None:
                        logits = model(strips, tgt_in, mem_mask=mm)
                        valid_pos = (tgt_in != pad_id) if pad_id >= 0 else torch.ones_like(
                            tgt_in, dtype=torch.bool
                        )
                    else:
                        logits = model(strips)
                        valid_pos = torch.ones(logits.shape[:-1], dtype=torch.bool, device=logits.device)

                    probs = F.softmax(logits.float(), dim=-1)
                    log_probs = F.log_softmax(logits.float(), dim=-1)
                    max_p = probs.max(dim=-1).values
                    ent = -(probs * log_probs).sum(dim=-1)

                    conf_mask = (max_p >= float(min_conf)) & valid_pos
                    n_sel = int(conf_mask.sum().item())
                    if n_sel > 0:
                        ent_sum = ent[conf_mask].sum()
                        total_ent = ent_sum if total_ent is None else (total_ent + ent_sum)
                        total_count += n_sel

                if total_ent is not None and total_count > 0:
                    loss = total_ent / float(total_count)
                    loss.backward()
                    opt.step()

        # Disable grads during inference inside the context block.
        for p in ln_params:
            p.requires_grad_(False)
        model.eval()
        yield model
    finally:
        with torch.no_grad():
            for p, orig_val in saved_ln:
                p.copy_(orig_val)
        for p, rg in orig_req_grad.items():
            p.requires_grad_(rg)
        model.train(was_training)


def run_unsup_round(
    net: nn.Module,
    opt: torch.optim.Optimizer,
    scaler,
    strip_paths: list,
    step: int = 0,
    steps: int = 100,
    weight: float = 0.25,
    height: int = 256,
    device: torch.device | str = "cpu",
    amp: bool = False,
    amp_dtype=torch.bfloat16,
    seed: int = 0,
    batch_size: int = 2,
    **_extra,
) -> dict:
    """Execute one Stage-2 encoder-only multi-view EMA consistency round (zero pseudo-labels)."""
    import numpy as np
    from PIL import Image

    if not strip_paths or int(steps) <= 0:
        return {"ran_steps": 0, "mean_loss": 0.0}

    dev = torch.device(device)
    ema = EMAModel(net, decay=0.995)
    ema.model.to(dev).eval()
    rng = np.random.default_rng(int(seed))
    n_imgs = len(strip_paths)
    in_ch = int(getattr(net, "in_chans", 4))
    was_train = net.training
    net.train()
    use_amp = bool(amp and dev.type == "cuda")
    total_loss = 0.0
    ran = 0
    bs = max(1, int(batch_size))

    for _ in range(int(steps)):
        idxs = rng.integers(0, n_imgs, size=bs)
        imgs = []
        max_w = 0
        for idx in idxs:
            try:
                im = Image.open(strip_paths[int(idx)]).convert("RGB")
                ow, oh = im.size
                nw = max(32, min(2048, int(round(ow * (height / max(1, oh))))))
                im = im.resize((nw, int(height)), Image.BOX)
                arr = np.asarray(im, dtype=np.float32) / 255.0
                t = (torch.from_numpy(arr).permute(2, 0, 1) - 0.85) / 0.25
                if in_ch > 3:
                    pad_c = torch.zeros((in_ch - 3, int(height), nw), dtype=torch.float32)
                    t = torch.cat([t, pad_c], dim=0)
                imgs.append(t)
                max_w = max(max_w, nw)
            except Exception:
                continue
        if not imgs:
            continue
        padded = []
        for t in imgs:
            pw = max_w - t.shape[2]
            if pw > 0:
                t = F.pad(t, (0, pw, 0, 0), value=0.6)
            padded.append(t)
        weak = torch.stack(padded, dim=0).to(dev, non_blocking=True)
        # Photometric-only strong view (geom=0 so spatial coordinates align 1-to-1)
        gain = torch.empty((weak.shape[0], 1, 1, 1), device=dev).uniform_(0.78, 1.22)
        bias = torch.empty((weak.shape[0], 1, 1, 1), device=dev).uniform_(-0.25, 0.25)
        noise = torch.randn_like(weak) * 0.08
        strong = weak.clone()
        strong[:, :3] = weak[:, :3] * gain + bias + noise

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            loss, _ = stage2_encoder_ema_consistency_loss(net, ema.model, weak, strong)
            scaled = loss * float(weight)
        if torch.isfinite(scaled):
            if scaler is not None and use_amp and amp_dtype == torch.float16:
                scaler.scale(scaled).backward()
                scaler.step(opt)
                scaler.update()
            else:
                scaled.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            ema.update(net)
            total_loss += float(loss.detach().item())
            ran += 1

    net.train(was_train)
    mean_loss = total_loss / max(1, ran)
    print(f"  [unsup-stage2-ema step {step}] ran_steps={ran}/{steps} mean_loss={mean_loss:.4f}", flush=True)
    return {"ran_steps": ran, "mean_loss": mean_loss}
