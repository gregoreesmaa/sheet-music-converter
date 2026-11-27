"""Ink-Aware I-JEPA (Masked-Feature MIM) Encoder Pretraining for UniOMR on Unlabeled RISM Pages.

Pretrains ``UniOMRNet``'s ``DynamicConvBackbone + InkTokenRouter + pos2d_proj +
RoPE2D + SparseTransformerEncoder`` on unlabeled historical manuscript pages
(e.g. 5,707 real RISM handwritten pages) while keeping all decoder, ``lm_head``,
``mtp_heads``, ``tok_emb``, and ``uv_head`` weights completely frozen and
bit-identical to ``--init-from`` so downstream supervised runs can warm-start
with ``strict=True``.

Key design choices:
  - **EMA Teacher** (cosine decay ``0.996 -> 0.999``) runs on the unmasked page
    to produce target latent representations ``mem_teacher`` and canonical
    ``kept_idx`` / ``routed_coords`` / ``saliency_prob``.
  - **Ink-Aware Strip/Block Masking** masks 45-55% of ink-bearing symbol/staff
    strips on the student image, filling masked pixels with the page's local
    ``p85`` parchment paper tone (never black/white) to avoid artificial edges.
  - **Exact 1-to-1 Spatial Alignment**: the student extracts features at the
    teacher's exact ``kept_idx`` positions, passes through the sparse encoder
    and a lightweight 2-layer MLP predictor (training-only, excluded from saved
    ``UniOMRNet`` checkpoints), and minimizes Smooth-L1 + Cosine loss only on
    masked tokens weighted ``3.0x`` on foreground ink vs ``0.2x`` on parchment.
  - **Router Saliency Consistency** (``0.2x`` BCE on unmasked interior paper)
    makes ``InkTokenRouter`` invariant to sepia/contrast/gamma/parchment shifts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import copy
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
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_ckpt

# Ensure sibling imports (``uni_omr.py``, ``data.py``) resolve regardless of CWD.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import data  # noqa: E402
from uni_omr import (  # noqa: E402
    InkTokenRouter,
    UniOMRConfig,
    UniOMRNet,
    _fast_image_is_landscape,
    _pil_to_chw_tensor,
    _resize_fullpage_pil,
    _round_mult16,
    load_uni_weights_compat,
)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_ENCODER_SUBMODULES = ("backbone", "router", "pos2d_proj", "rope2d", "enc_layers", "enc_norm")


# ---------------------------------------------------------------------------
# Image Discovery & Multi-Scale Orientation-Bucketed Loader
# ---------------------------------------------------------------------------

def discover_unlabeled_images(image_dir: str | Path) -> List[Path]:
    """Recursively discover ``.png/.jpg/.jpeg/.webp`` images under ``image_dir``."""
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    found: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in _IMAGE_EXTS:
                found.append(Path(dirpath) / fn)
    found.sort()
    return found


def load_image_resized(path: Path, target_dim: int) -> torch.Tensor:
    """Load an RGB image, resize longest side to ``target_dim`` (mult-16), return ``(3, H, W)`` float32 in ``[0, 1]``."""
    with Image.open(path) as raw:
        rgb = raw if raw.mode == "RGB" else raw.convert("RGB")
        img = _resize_fullpage_pil(rgb, fullpage_max_dim=int(target_dim))
        if img is raw:
            raw.load()
            img = raw.copy()
    return _pil_to_chw_tensor(img)


def sample_multiscale_dim(max_dim: int, rng: random.Random) -> int:
    """Sample a multi-scale target dimension in ``[0.75 * max_dim, max_dim]`` (mult-16)."""
    if max_dim <= 256:
        return _round_mult16(max_dim, min_val=32)
    scales = (0.75, 0.875, 1.0)
    weights = (0.25, 0.35, 0.40)
    frac = rng.choices(scales, weights=weights, k=1)[0]
    return _round_mult16(max_dim * frac, min_val=128, max_val=max_dim)


def collate_image_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Pad variable-aspect ``(3, H, W)`` tensors to shared mult-16 ``(B, 3, max_H, max_W)`` using each image's p85 paper tone."""
    B = len(tensors)
    max_h = _round_mult16(max(int(t.shape[1]) for t in tensors), min_val=16)
    max_w = _round_mult16(max(int(t.shape[2]) for t in tensors), min_val=16)
    batch = torch.empty((B, 3, max_h, max_w), dtype=torch.float32)
    for i, t in enumerate(tensors):
        _, h, w = t.shape
        if h == max_h and w == max_w:
            batch[i].copy_(t)
        else:
            # Pad with per-image p85 parchment tone so padding never introduces harsh white/black seams.
            flat = t.reshape(3, -1)
            if flat.shape[-1] > 32768:
                flat = flat[:, :: max(1, flat.shape[-1] // 32768)]
            fill = torch.quantile(flat, 0.85, dim=-1).view(3, 1, 1)
            batch[i].copy_(fill.expand(3, max_h, max_w))
            batch[i, :, :h, :w].copy_(t)
    return batch


class UnlabeledPageBatchPrefetcher:
    """Infinite orientation-bucketed, multi-scale prefetcher for unlabeled pages.

    Never calls ``.pin_memory()`` inside background threads (Windows WDDM safe).
    """

    def __init__(
        self,
        paths: Sequence[Path],
        batch_size: int,
        max_dim: int,
        multi_scale: bool = True,
        seed: int = 42,
        num_threads: int = 4,
        prefetch_batches: int = 3,
    ) -> None:
        if not paths:
            raise ValueError("UnlabeledPageBatchPrefetcher requires at least one image path")
        self.paths = list(paths)
        self.batch_size = max(1, int(batch_size))
        self.max_dim = int(max_dim)
        self.multi_scale = bool(multi_scale)
        self._rng = random.Random(seed)
        self.num_threads = max(0, int(num_threads))
        self.prefetch_batches = max(1, int(prefetch_batches))

        # Partition into portrait vs landscape buckets when batch_size >= 2.
        self._portrait: List[int] = []
        self._landscape: List[int] = []
        if self.batch_size >= 2 and len(self.paths) >= 2:
            for idx, p in enumerate(self.paths):
                is_land = _fast_image_is_landscape(p)
                if is_land:
                    self._landscape.append(idx)
                else:
                    self._portrait.append(idx)
        else:
            self._portrait = list(range(len(self.paths)))

        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=self.num_threads) if self.num_threads > 0 else None
        )
        self._queue: List[Future[torch.Tensor]] = []
        self._epoch_batches: List[Tuple[List[int], int]] = []
        self._refill_epoch()

    def _refill_epoch(self) -> None:
        batches: List[Tuple[List[int], int]] = []
        for bucket in (self._portrait, self._landscape):
            if not bucket:
                continue
            idxs = list(bucket)
            self._rng.shuffle(idxs)
            for start in range(0, len(idxs), self.batch_size):
                chunk = idxs[start : start + self.batch_size]
                if len(chunk) < self.batch_size and len(idxs) >= self.batch_size:
                    chunk = chunk + self._rng.choices(idxs, k=self.batch_size - len(chunk))
                tdim = sample_multiscale_dim(self.max_dim, self._rng) if self.multi_scale else self.max_dim
                batches.append((chunk, tdim))
        self._rng.shuffle(batches)
        self._epoch_batches = batches

    def _next_spec(self) -> Tuple[List[int], int]:
        if not self._epoch_batches:
            self._refill_epoch()
        return self._epoch_batches.pop()

    def _load_batch(self, idxs: List[int], target_dim: int) -> torch.Tensor:
        tensors: List[torch.Tensor] = []
        for idx in idxs:
            p = self.paths[idx]
            try:
                tensors.append(load_image_resized(p, target_dim))
            except Exception:
                # Fallback to another random image if a file is corrupted
                alt = self.paths[self._rng.randrange(len(self.paths))]
                tensors.append(load_image_resized(alt, target_dim))
        return collate_image_tensors(tensors)

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def __next__(self) -> torch.Tensor:
        if self._executor is None:
            idxs, tdim = self._next_spec()
            return self._load_batch(idxs, tdim)
        while len(self._queue) < self.prefetch_batches:
            idxs, tdim = self._next_spec()
            self._queue.append(self._executor.submit(self._load_batch, idxs, tdim))
        fut = self._queue.pop(0)
        return fut.result()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


# ---------------------------------------------------------------------------
# Ink-Aware Strip/Block Masking & Mild Photometric Augmentation
# ---------------------------------------------------------------------------

def estimate_paper_tone_p85(image: torch.Tensor) -> torch.Tensor:
    """Estimate the page's local parchment paper tone ``(3, 1, 1)`` from the 85th-percentile luma pixels.

    100% tensor-only (zero ``.item()`` / CPU syncs on GPU).
    """
    img_f = image.float()
    luma = 0.299 * img_f[0] + 0.587 * img_f[1] + 0.114 * img_f[2]
    flat_luma = luma.reshape(-1)
    flat_rgb = img_f.reshape(3, -1)
    if flat_luma.numel() > 65536:
        stride = flat_luma.numel() // 65536
        flat_luma = flat_luma[::stride]
        flat_rgb = flat_rgb[:, ::stride]
    p85 = torch.quantile(flat_luma, 0.85)
    band = (flat_luma >= (p85 - 0.03)) & (flat_luma <= (p85 + 0.05))
    count = band.sum()
    band_f = band.to(img_f.dtype)
    band_mean = (flat_rgb * band_f.unsqueeze(0)).sum(dim=1) / count.clamp(min=1).to(img_f.dtype)
    q85_rgb = torch.quantile(flat_rgb, 0.85, dim=-1)
    tone = torch.where(count >= 16, band_mean, q85_rgb)
    return tone.clamp(0.0, 1.0).view(3, 1, 1)


def apply_mild_photometric_aug(
    images: torch.Tensor,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """Apply mild contrast, gamma, brightness, sepia-tint, and noise jitter to ``(B, 3, H, W)`` in ``[0, 1]``."""
    r = rng if rng is not None else random
    out = images.float().clone()
    B = out.shape[0]
    for b in range(B):
        im = out[b]
        # 1. Mild gamma jitter
        if r.random() < 0.75:
            gamma = r.uniform(0.82, 1.22)
            im = im.clamp(min=1e-5).pow(gamma)
        # 2. Mild contrast jitter around per-image mean
        if r.random() < 0.75:
            contrast = r.uniform(0.82, 1.18)
            mean_val = im.mean()
            im = (im - mean_val) * contrast + mean_val
        # 3. Mild brightness shift
        if r.random() < 0.65:
            im = im + r.uniform(-0.06, 0.06)
        # 4. Mild sepia / archival warmth channel jitter
        if r.random() < 0.75:
            gains = torch.tensor(
                [r.uniform(0.94, 1.06), r.uniform(0.93, 1.05), r.uniform(0.88, 1.04)],
                device=im.device,
                dtype=im.dtype,
            ).view(3, 1, 1)
            im = im * gains
        # 5. Subtle sensor noise
        if r.random() < 0.50:
            sigma = r.uniform(0.005, 0.015)
            im = im + torch.randn_like(im) * sigma
        out[b] = im.clamp(0.0, 1.0)
    return out.to(images.dtype)


def apply_ink_strip_mask(
    images: torch.Tensor,
    local_ink_grid: torch.Tensor,
    mask_ratio_range: Tuple[float, float] = (0.45, 0.55),
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fully vectorized GPU-native strip/block masking (zero CPU/numpy syncs).

    Masks ~45-55% of ink-bearing symbol/staff strips by generating anisotropic
    vertical + horizontal random strip masks directly on ``images.device`` and
    filling masked pixels with each page's ``p85`` parchment paper tone.
    """
    r = rng if rng is not None else random
    B, _, H, W = images.shape
    _, _, H_p, W_p = local_ink_grid.shape
    device = images.device

    # 1. Estimate per-page p85 paper tone entirely on-device: (B, 3, 1, 1)
    tones = torch.stack([estimate_paper_tone_p85(images[b]) for b in range(B)], dim=0).to(
        device=device, dtype=images.dtype
    )

    # 2. Build anisotropic vertical (symbol/measure) + horizontal (staff-segment) strip masks on-device.
    #    Coarse seed grids dilated into strip shapes give exact ~45-55% ink coverage with zero host syncs.
    target_frac = r.uniform(float(mask_ratio_range[0]), float(mask_ratio_range[1]))
    # Mix of vertical strips (tall & narrow in feature grid) and horizontal strips (wide & short)
    ch_v = max(2, H_p // 10)
    cw_v = max(2, W_p // 4)
    ch_h = max(2, H_p // 5)
    cw_h = max(2, W_p // 10)

    # Calibrate independent seed probabilities p_v, p_h so union covers ~target_frac:
    # 1 - (1 - p_v) * (1 - p_h) = target_frac  (60% vertical share, 40% horizontal share)
    p_v = min(0.85, max(0.05, 1.0 - (1.0 - target_frac) ** 0.60))
    p_h = min(0.85, max(0.05, 1.0 - (1.0 - target_frac) ** 0.40))

    seed_v = (torch.rand((B, 1, ch_v, cw_v), device=device) < p_v).float()
    seed_h = (torch.rand((B, 1, ch_h, cw_h), device=device) < p_h).float()

    # Upsample to feature grid (H_p, W_p) with random spatial jitter via roll
    mask_v = F.interpolate(seed_v, size=(H_p, W_p), mode="nearest")
    mask_h = F.interpolate(seed_h, size=(H_p, W_p), mode="nearest")
    shift_y = r.randint(0, max(0, H_p - 1))
    shift_x = r.randint(0, max(0, W_p - 1))
    grid_mask_f = ((mask_v + mask_h) > 0.5).float()
    if H_p > 1 and W_p > 1:
        grid_mask_f = torch.roll(grid_mask_f, shifts=(shift_y, shift_x), dims=(2, 3))

    # Guarantee at least one masked ink cell per sample without host sync
    grid_mask = grid_mask_f > 0.5
    pixel_masks = F.interpolate(grid_mask_f, size=(H, W), mode="nearest") > 0.5
    masked_images = torch.where(pixel_masks, tones, images)
    return masked_images, grid_mask


# ---------------------------------------------------------------------------
# EMA Teacher & Predictor Head
# ---------------------------------------------------------------------------

class IJEPAPredictor(nn.Module):
    """Lightweight 2-layer MLP predictor (training-only; NOT saved in UniOMRNet state_dict)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model, bias=True),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EMAEncoderTeacher(nn.Module):
    """Frozen EMA copy of ``UniOMRNet``'s encoder modules."""

    def __init__(self, student: UniOMRNet) -> None:
        super().__init__()
        self.cfg = copy.deepcopy(student.cfg)
        for name in _ENCODER_SUBMODULES:
            mod = copy.deepcopy(getattr(student, name))
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)
            setattr(self, name, mod)

    @torch.no_grad()
    def update_from(self, student: UniOMRNet, decay: float) -> None:
        d = float(decay)
        one_m_d = 1.0 - d
        for name in _ENCODER_SUBMODULES:
            t_mod = getattr(self, name)
            s_mod = getattr(student, name)
            for t_p, s_p in zip(t_mod.parameters(), s_mod.parameters()):
                t_p.data.mul_(d).add_(s_p.data, alpha=one_m_d)
            for t_b, s_b in zip(t_mod.buffers(), s_mod.buffers()):
                if t_b.is_floating_point():
                    t_b.data.mul_(d).add_(s_b.data, alpha=one_m_d)
                else:
                    t_b.data.copy_(s_b.data)

    @torch.no_grad()
    def encode(
        self,
        images: torch.Tensor,
        sinusoidal_pos2d_fn: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Run teacher encoder on unmasked images and return ``(mem_teacher, routed_coords, info)``."""
        backbone_in = (
            images.to(memory_format=torch.channels_last)
            if self.backbone.stage1.down[0].weight.is_contiguous(memory_format=torch.channels_last)
            else images
        )
        feat_2d = self.backbone(backbone_in)  # (B, D, H', W')
        B, _, H_p, W_p = feat_2d.shape
        local_ink_grid, paper_gate, effective_paper = InkTokenRouter._compute_ink_and_paper_gate(
            images, H_p, W_p
        )
        routed_tokens, routed_coords, router_info = self.router(feat_2d, images=images)

        pos2d = self.pos2d_proj(
            sinusoidal_pos2d_fn(
                routed_coords,
                dtype=routed_tokens.dtype,
                grid_hw=router_info["grid_hw"],
            )
        )
        cos2d, sin2d = self.rope2d(routed_coords, dtype=routed_tokens.dtype)
        mem = routed_tokens + pos2d
        for layer in self.enc_layers:
            mem = layer(mem, rope_cos=cos2d, rope_sin=sin2d)
        mem = self.enc_norm(mem) + pos2d

        router_info["local_ink_grid"] = local_ink_grid
        router_info["paper_gate"] = paper_gate
        router_info["effective_paper"] = effective_paper
        router_info["saliency_prob"] = torch.sigmoid(router_info["saliency_logits"]) * paper_gate.to(
            router_info["saliency_logits"].dtype
        )
        return mem, routed_coords, router_info


def encode_student_aligned(
    student: UniOMRNet,
    student_images: torch.Tensor,
    teacher_kept_idx: torch.Tensor,
    teacher_routed_coords: torch.Tensor,
    teacher_info: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run student encoder extracting tokens at the exact teacher ``kept_idx`` for 1-to-1 latent alignment.

    Returns:
        mem_student: ``(B, k_kept + n_bg_summary, D)``
        sal_logits_student: ``(B, 1, H', W')``
    """
    backbone_in = (
        student_images.to(memory_format=torch.channels_last)
        if student.backbone.stage1.down[0].weight.is_contiguous(memory_format=torch.channels_last)
        else student_images
    )
    feat_2d = student.backbone(backbone_in)  # (B, D, H', W')
    B, D, H_p, W_p = feat_2d.shape
    N = H_p * W_p

    router = student.router
    sal_logits_s = router.saliency_net(feat_2d)  # (B, 1, H', W')
    paper_gate = teacher_info["paper_gate"].to(device=feat_2d.device, dtype=sal_logits_s.dtype)
    sal_prob_s = torch.sigmoid(sal_logits_s) * paper_gate

    if router.use_ink_prior:
        local_ink_s, _, _ = InkTokenRouter._compute_ink_and_paper_gate(student_images, H_p, W_p)
        local_ink_s = local_ink_s.to(sal_logits_s.dtype)
        flat = local_ink_s.flatten(1)
        denom = (
            torch.quantile(flat.float(), 0.99, dim=-1)
            .clamp(min=1e-4)
            .view(B, 1, 1, 1)
            .to(local_ink_s.dtype)
        )
        local_ink_norm = (local_ink_s / denom).clamp(0.0, 1.0)
        score = sal_prob_s + torch.sigmoid(router.ink_prior_scale).to(sal_logits_s.dtype) * local_ink_norm
    else:
        score = sal_prob_s

    score_flat = score.flatten(1)  # (B, N)
    tokens_flat = feat_2d.permute(0, 2, 3, 1).reshape(B, N, D)

    k_kept = int(teacher_info["k_kept"])
    n_bg = int(teacher_info["n_bg_summary"])

    if k_kept >= N or n_bg == 0:
        gate = (0.5 + 0.5 * torch.sigmoid(score_flat)).unsqueeze(-1)
        routed_tokens = tokens_flat * gate
    else:
        kept_scores = torch.gather(score_flat, 1, teacher_kept_idx)  # (B, k_kept)
        kept_gate = (0.5 + 0.5 * torch.sigmoid(kept_scores)).unsqueeze(-1)
        idx_feat = teacher_kept_idx.unsqueeze(-1).expand(-1, -1, D)
        kept_tokens = torch.gather(tokens_flat, 1, idx_feat) * kept_gate  # (B, k_kept, D)

        kept_mask = torch.zeros(B, N, device=feat_2d.device, dtype=torch.bool)
        kept_mask.scatter_(1, teacher_kept_idx, True)

        bg_q = router.bg_queries.expand(B, -1, -1)
        bg_q_proj = F.linear(bg_q, router.bg_key_proj.weight.t()).to(tokens_flat.dtype)
        bg_logits = torch.bmm(bg_q_proj, tokens_flat.transpose(1, 2)) / math.sqrt(D)
        bg_logits = bg_logits - score_flat.unsqueeze(1)
        bg_logits = bg_logits.masked_fill(kept_mask.unsqueeze(1), -1e4)
        bg_attn = F.softmax(bg_logits, dim=-1)
        pooled = torch.bmm(bg_attn, tokens_flat)
        bg_tokens = router.bg_out_norm(router.bg_val_proj(pooled))

        routed_tokens = torch.cat([kept_tokens, bg_tokens], dim=1)

    pos2d = student.pos2d_proj(
        student._sinusoidal_pos2d(
            teacher_routed_coords,
            dtype=routed_tokens.dtype,
            grid_hw=(H_p, W_p),
        )
    )
    cos2d, sin2d = student.rope2d(teacher_routed_coords, dtype=routed_tokens.dtype)
    mem = routed_tokens + pos2d
    if (
        student.training
        and student.cfg.grad_checkpoint
        and torch.is_grad_enabled()
        and (mem.shape[0] * mem.shape[1] > 4000)
    ):
        for layer in student.enc_layers:
            mem = _grad_ckpt(
                lambda _m, _c, _s, _l=layer: _l(_m, rope_cos=_c, rope_sin=_s),
                mem,
                cos2d,
                sin2d,
                use_reentrant=False,
            )
    else:
        for layer in student.enc_layers:
            mem = layer(mem, rope_cos=cos2d, rope_sin=sin2d)
    mem = student.enc_norm(mem) + pos2d
    return mem, sal_logits_s


# ---------------------------------------------------------------------------
# Ink-Aware I-JEPA Loss
# ---------------------------------------------------------------------------

def compute_ink_ijepa_loss(
    pred_kept: torch.Tensor,
    target_kept: torch.Tensor,
    grid_mask: torch.Tensor,
    kept_idx: torch.Tensor,
    local_ink_grid: torch.Tensor,
    sal_logits_student: torch.Tensor,
    sal_logits_teacher: torch.Tensor,
    effective_paper: torch.Tensor,
    ink_weight: float = 3.0,
    bg_weight: float = 0.2,
    router_cons_weight: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """Compute ink-weighted masked-token Smooth-L1 + Cosine loss + unmasked router consistency BCE."""
    masked_kept = torch.gather(grid_mask.flatten(1), 1, kept_idx)  # (B, k_kept) bool
    ink_kept = torch.gather(local_ink_grid.flatten(1), 1, kept_idx).to(torch.float32)  # (B, k_kept)

    is_ink = ink_kept > 0.05
    tok_w = torch.where(is_ink, float(ink_weight), float(bg_weight)) * masked_kept.to(torch.float32)
    w_sum = tok_w.sum()

    if float(w_sum.item()) > 0.0:
        p_f = pred_kept.float()
        t_f = target_kept.detach().float()
        l1_tok = F.smooth_l1_loss(p_f, t_f, reduction="none").mean(dim=-1)  # (B, k_kept)
        cos_tok = 1.0 - F.cosine_similarity(p_f, t_f, dim=-1, eps=1e-6)     # (B, k_kept)
        ijepa_loss = ((l1_tok + cos_tok) * tok_w).sum() / w_sum.clamp(min=1e-6)
    else:
        ijepa_loss = pred_kept.new_zeros((), dtype=torch.float32)

    # Router saliency consistency on unmasked interior paper regions
    unmasked_paper = (~grid_mask) & effective_paper
    up_f = unmasked_paper.to(torch.float32)
    up_sum = up_f.sum()
    if float(up_sum.item()) > 0.0 and router_cons_weight > 0.0:
        tgt_prob = torch.sigmoid(sal_logits_teacher.detach().float())
        bce_map = F.binary_cross_entropy_with_logits(
            sal_logits_student.float(), tgt_prob, reduction="none"
        )
        router_loss = (bce_map * up_f).sum() / up_sum.clamp(min=1e-6)
    else:
        router_loss = pred_kept.new_zeros((), dtype=torch.float32)

    total_loss = ijepa_loss + float(router_cons_weight) * router_loss
    n_ink_total = is_ink.sum().float().clamp(min=1.0)
    masked_ink_frac = (is_ink & masked_kept).sum().float() / n_ink_total

    return {
        "loss": total_loss,
        "ijepa_loss": ijepa_loss,
        "router_loss": router_loss,
        "masked_ink_frac": masked_ink_frac,
    }


# ---------------------------------------------------------------------------
# Checkpoint Helpers
# ---------------------------------------------------------------------------

def freeze_non_encoder_params(model: UniOMRNet) -> List[nn.Parameter]:
    """Freeze all non-encoder params (decoder, lm_head, mtp_heads, tok_emb, uv_head) and return trainable encoder params."""
    for p in model.parameters():
        p.requires_grad_(False)
    trainable: List[nn.Parameter] = []
    for name in _ENCODER_SUBMODULES:
        mod = getattr(model, name)
        for p in mod.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


def build_uniomr_checkpoint_state_dict(
    student: UniOMRNet,
    teacher: Optional[EMAEncoderTeacher] = None,
    use_teacher_encoder: bool = True,
) -> Dict[str, torch.Tensor]:
    """Build a 100% ``UniOMRNet(..., strict=True)``-compatible CPU state_dict.

    Preserves all decoder / ``lm_head`` / ``mtp_heads`` / ``tok_emb`` / ``uv_head``
    tensors verbatim from ``student`` (which stayed frozen), and optionally copies
    the smoother EMA teacher encoder weights into the encoder slots.
    """
    sd = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    if use_teacher_encoder and teacher is not None:
        for name in _ENCODER_SUBMODULES:
            t_sd = getattr(teacher, name).state_dict()
            for sub_k, v in t_sd.items():
                full_k = f"{name}.{sub_k}"
                if full_k in sd:
                    sd[full_k] = v.detach().cpu().clone()
    return sd


def save_ijepa_checkpoint(
    path: Path,
    student: UniOMRNet,
    teacher: EMAEncoderTeacher,
    cfg: UniOMRConfig,
    vocab: List[str],
    step: int,
    ijepa_loss: float,
    use_teacher_encoder: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "step": int(step),
        "ijepa_loss": float(ijepa_loss),
        "config": asdict(cfg),
        "vocab": list(vocab),
        "model": build_uniomr_checkpoint_state_dict(
            student, teacher=teacher, use_teacher_encoder=use_teacher_encoder
        ),
    }
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Single-Step Helper & Main Training Loop
# ---------------------------------------------------------------------------

def cosine_ema_decay(step: int, total_steps: int, start: float = 0.996, end: float = 0.999) -> float:
    frac = min(1.0, max(0.0, float(step) / float(max(1, total_steps))))
    return float(end - (end - start) * 0.5 * (1.0 + math.cos(math.pi * frac)))


def cosine_lr_mult(step: int, total_steps: int, warmup_steps: int = 100, min_ratio: float = 0.1) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    denom = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
    return float(min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def ijepa_forward_loss(
    student: UniOMRNet,
    teacher: EMAEncoderTeacher,
    pred_head: IJEPAPredictor,
    images: torch.Tensor,
    rng: random.Random,
    use_bf16: bool = False,
    device_type: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Run one teacher + masked-student Ink-Aware I-JEPA forward pass and return loss dict."""
    amp_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if use_bf16
        else _nullcontext()
    )
    with amp_ctx:
        mem_teacher, routed_coords, teacher_info = teacher.encode(
            images, sinusoidal_pos2d_fn=student._sinusoidal_pos2d
        )
        local_ink_grid = teacher_info["local_ink_grid"]
        kept_idx = teacher_info["kept_idx"]
        k_kept = int(teacher_info["k_kept"])

        # Mild photometric aug first, then ink-aware p85 strip masking so masked strips match student paper tone.
        aug_images = apply_mild_photometric_aug(images, rng=rng)
        student_images, grid_mask = apply_ink_strip_mask(
            aug_images, local_ink_grid, mask_ratio_range=(0.45, 0.55), rng=rng
        )

        mem_student, sal_logits_s = encode_student_aligned(
            student=student,
            student_images=student_images,
            teacher_kept_idx=kept_idx,
            teacher_routed_coords=routed_coords,
            teacher_info=teacher_info,
        )
        pred_kept = pred_head(mem_student[:, :k_kept, :])
        target_kept = mem_teacher[:, :k_kept, :].detach()

        losses = compute_ink_ijepa_loss(
            pred_kept=pred_kept,
            target_kept=target_kept,
            grid_mask=grid_mask,
            kept_idx=kept_idx,
            local_ink_grid=local_ink_grid,
            sal_logits_student=sal_logits_s,
            sal_logits_teacher=teacher_info["saliency_logits"],
            effective_paper=teacher_info["effective_paper"],
        )
    return losses


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        return None


def run_pretrain(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enforce CUTLASS EFFICIENT_ATTENTION — never cuDNN SDPA (49x slower on variable shapes)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)

    # 1. Load warm-start checkpoint & exact UniOMRConfig
    ck = torch.load(str(args.init_from), map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or "model" not in ck:
        raise ValueError(f"--init-from must be a UniOMR checkpoint dict with 'model' key: {args.init_from}")
    cfg_dict = dict(ck.get("config", {}))
    if args.grad_checkpoint is not None:
        cfg_dict["grad_checkpoint"] = bool(args.grad_checkpoint)
    cfg = UniOMRConfig(**cfg_dict)
    init_vocab = list(ck.get("vocab", list(data.VOCAB)))

    student = UniOMRNet(cfg).to(device)
    sd = ck["model"]
    try:
        student.load_state_dict(sd, strict=True)
        print(f"[I-JEPA init] loaded {args.init_from} (strict=True)", flush=True)
    except RuntimeError:
        inc = load_uni_weights_compat(student, sd)
        print(
            f"[I-JEPA init] loaded {args.init_from} via compat "
            f"(missing={len(inc.missing_keys)}, unexpected={len(inc.unexpected_keys)})",
            flush=True,
        )

    if args.channels_last:
        student.backbone.to(memory_format=torch.channels_last)

    trainable_enc_params = freeze_non_encoder_params(student)
    teacher = EMAEncoderTeacher(student).to(device)
    if args.channels_last:
        teacher.backbone.to(memory_format=torch.channels_last)

    pred_head = IJEPAPredictor(cfg.d_model).to(device)
    all_trainable = trainable_enc_params + list(pred_head.parameters())

    if device.type == "cuda":
        try:
            opt = torch.optim.AdamW(all_trainable, lr=args.lr, weight_decay=0.04, fused=True)
        except (RuntimeError, TypeError):
            opt = torch.optim.AdamW(all_trainable, lr=args.lr, weight_decay=0.04)
    else:
        opt = torch.optim.AdamW(all_trainable, lr=args.lr, weight_decay=0.04)

    # 2. Discover unlabeled RISM images
    image_paths = discover_unlabeled_images(args.image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found under {args.image_dir}")
    print(
        f"[I-JEPA data] discovered {len(image_paths)} images in {args.image_dir} "
        f"(max_dim={args.max_dim}, batch_size={args.batch_size}, grad_accum={args.grad_accum})",
        flush=True,
    )

    prefetch_threads = 4 if (device.type == "cuda" and len(image_paths) >= 4) else 0
    loader = UnlabeledPageBatchPrefetcher(
        paths=image_paths,
        batch_size=args.batch_size,
        max_dim=args.max_dim,
        multi_scale=not args.no_multi_scale,
        seed=args.seed,
        num_threads=prefetch_threads,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_bf16 = bool(args.bf16 and (device.type == "cuda" or hasattr(torch, "bfloat16")))
    warmup_steps = min(args.warmup_steps, max(1, args.steps // 10))

    student.train()
    pred_head.train()
    teacher.eval()

    best_ema_loss = float("inf")
    ema_loss_track: Optional[float] = None
    oom_skips = 0
    history: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    try:
        for step in range(1, args.steps + 1):
            lr_mult = cosine_lr_mult(step - 1, args.steps, warmup_steps=warmup_steps)
            cur_lr = args.lr * lr_mult
            for pg in opt.param_groups:
                pg["lr"] = cur_lr

            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_ijepa = 0.0
            step_router = 0.0
            step_mask_frac = 0.0
            valid_micros = 0

            for _ in range(args.grad_accum):
                batch_cpu = next(loader)
                images = batch_cpu.to(device, non_blocking=(device.type == "cuda"))
                try:
                    losses = ijepa_forward_loss(
                        student=student,
                        teacher=teacher,
                        pred_head=pred_head,
                        images=images,
                        rng=rng,
                        use_bf16=use_bf16,
                        device_type=device.type,
                    )
                    scaled_loss = losses["loss"] / float(args.grad_accum)
                    if not torch.isfinite(scaled_loss):
                        print(f"[I-JEPA warn] non-finite loss at step {step}; skipping micro-batch", flush=True)
                        continue
                    scaled_loss.backward()
                    step_loss += float(losses["loss"].detach().item())
                    step_ijepa += float(losses["ijepa_loss"].detach().item())
                    step_router += float(losses["router_loss"].detach().item())
                    step_mask_frac += float(losses["masked_ink_frac"].detach().item())
                    valid_micros += 1
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        oom_skips += 1
                        opt.zero_grad(set_to_none=True)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        print(
                            f"[I-JEPA OOM] skipped micro-batch at step {step} (total_skips={oom_skips})",
                            flush=True,
                        )
                        valid_micros = 0
                        break
                    raise

            if valid_micros == 0:
                continue

            torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
            opt.step()

            ema_decay = cosine_ema_decay(step, args.steps, start=0.996, end=0.999)
            teacher.update_from(student, decay=ema_decay)

            avg_loss = step_loss / valid_micros
            avg_ijepa = step_ijepa / valid_micros
            avg_router = step_router / valid_micros
            avg_mask = step_mask_frac / valid_micros
            ema_loss_track = avg_loss if ema_loss_track is None else (0.95 * ema_loss_track + 0.05 * avg_loss)

            if step == 1 or step % args.log_every == 0 or step == args.steps:
                elapsed = max(1e-6, time.perf_counter() - t0)
                sps = step / elapsed
                vram_gb = (
                    torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
                    if device.type == "cuda"
                    else 0.0
                )
                rec = {
                    "step": step,
                    "loss": round(avg_loss, 5),
                    "ema_loss": round(ema_loss_track, 5),
                    "ijepa_loss": round(avg_ijepa, 5),
                    "router_loss": round(avg_router, 5),
                    "masked_ink_frac": round(avg_mask, 3),
                    "lr": cur_lr,
                    "ema_decay": round(ema_decay, 5),
                    "vram_gb": round(vram_gb, 2),
                    "steps_per_sec": round(sps, 2),
                    "oom_skips": oom_skips,
                }
                history.append(rec)
                print(
                    f"[I-JEPA step {step:5d}/{args.steps}] "
                    f"loss={avg_loss:.4f} (ema={ema_loss_track:.4f}) "
                    f"ijepa={avg_ijepa:.4f} router={avg_router:.4f} "
                    f"mask_ink={avg_mask:.2f} lr={cur_lr:.2e} "
                    f"decay={ema_decay:.4f} vram={vram_gb:.2f}GB ({sps:.2f} it/s)",
                    flush=True,
                )

            if ema_loss_track < best_ema_loss and (step >= min(50, args.steps) or step == args.steps):
                best_ema_loss = ema_loss_track
                save_ijepa_checkpoint(
                    out_dir / "uni_omr_ijepa_best.pt",
                    student=student,
                    teacher=teacher,
                    cfg=cfg,
                    vocab=init_vocab,
                    step=step,
                    ijepa_loss=best_ema_loss,
                    use_teacher_encoder=not args.save_student_weights,
                )

            if args.save_every > 0 and step % args.save_every == 0:
                save_ijepa_checkpoint(
                    out_dir / f"uni_omr_ijepa_step{step}.pt",
                    student=student,
                    teacher=teacher,
                    cfg=cfg,
                    vocab=init_vocab,
                    step=step,
                    ijepa_loss=ema_loss_track,
                    use_teacher_encoder=not args.save_student_weights,
                )
    finally:
        loader.close()

    save_ijepa_checkpoint(
        out_dir / "uni_omr_ijepa_last.pt",
        student=student,
        teacher=teacher,
        cfg=cfg,
        vocab=init_vocab,
        step=args.steps,
        ijepa_loss=ema_loss_track if ema_loss_track is not None else float("nan"),
        use_teacher_encoder=not args.save_student_weights,
    )
    summary = {
        "steps": args.steps,
        "num_images": len(image_paths),
        "best_ema_loss": best_ema_loss,
        "final_ema_loss": ema_loss_track,
        "oom_skips": oom_skips,
        "history": history,
    }
    (out_dir / "ijepa_train_log.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ink-Aware I-JEPA (Masked-Feature MIM) encoder pretraining for UniOMR on unlabeled RISM pages."
    )
    p.add_argument("--init-from", type=str, required=True, help="Warm-start UniOMR checkpoint (.pt)")
    p.add_argument("--image-dir", type=str, required=True, help="Directory of unlabeled pages (recursively globbed)")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for checkpoints and logs")
    p.add_argument("--steps", type=int, default=4000, help="Total optimizer steps (default: 4000)")
    p.add_argument("--batch-size", type=int, default=2, help="Per-microbatch size (default: 2)")
    p.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    p.add_argument("--max-dim", type=int, default=1280, help="Max image dimension (default: 1280)")
    p.add_argument("--lr", type=float, default=3e-5, help="Peak learning rate (default: 3e-5)")
    p.add_argument("--warmup-steps", type=int, default=150, help="Linear LR warmup steps (default: 150)")
    p.add_argument("--bf16", action="store_true", help="Enable bfloat16 autocast")
    p.add_argument("--channels-last", action="store_true", help="Use NHWC channels_last for ConvBackbone")
    p.add_argument("--grad-checkpoint", dest="grad_checkpoint", action="store_true", default=None)
    p.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    p.add_argument("--no-multi-scale", action="store_true", help="Disable random multi-scale resizing")
    p.add_argument("--save-student-weights", action="store_true", help="Save student encoder instead of EMA teacher")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-every", type=int, default=500, help="Save periodic checkpoint every N steps (default: 500)")
    p.add_argument("--log-every", type=int, default=25, help="Log progress every N steps (default: 25)")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    run_pretrain(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
