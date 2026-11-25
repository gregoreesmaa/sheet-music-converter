"""Frontier V-JEPA 2.1 + DINOv3 Gram Anchoring + MonkeyOCRv2 Stroke-HOG SSL Pretrainer for UniOMR v2.

Pretrains ``UniOMRv2Net``'s ``AnisotropicMusicBackbone`` and
``HybridAxialMergerEncoder`` on unlabeled historical manuscript pages
(e.g. 5,707 real RISM handwritten pages in ``D:\\tmp\\unlabeled-10k-real\\rism_hw``)
while keeping all decoder, ``lm_head``, ``mtp_heads``, ``tok_emb``,
``deepstack_fusions``, and layout aux heads frozen.

Key Frontier Components:
  1. **V-JEPA 2.1 Dense Predictive Loss + Deep Self-Supervision (3 DeepStack Taps)**:
     - EMA teacher encoder (cosine decay ``0.996 -> 0.999``) runs on the clean
       archival page to produce canonical ``kept_idx``, ``routed_coords``, and
       all 3 DeepStack levels ``(mem_post_merger, mem_global_mid, mem_final)``.
     - Lightweight 2-layer MLP predictors (training-only, excluded from saved
       ``UniOMRv2Net`` checkpoints) supervise **both** masked ink tokens
       (``masked_weight=3.0``) AND visible context ink tokens
       (``visible_weight=0.5``) with Smooth-L1 + Cosine loss across all 3 levels
       (default weights ``0.25 / 0.35 / 1.0``).
  2. **DINOv3 Ink-Token Gram Anchoring** (``--gram-weight 1.0``, ``--gram-max-tokens 512``):
     - Frozen ``gram_teacher`` snapshot (from ``--init-from`` or step-0 encoder)
       anchors pairwise L2-normalized ink-token Gram matrices
       ``G_s = F_s @ F_s.T`` vs ``G_t = F_t @ F_t.T`` (deterministic stride
       subsample ``M <= 512``), preventing dense pitch/staff-line geometric
       collapse during unlabeled domain adaptation.
  3. **MonkeyOCRv2 / MaskFeat Sobel-HOG Stroke Reconstruction** (``--stroke-hog-weight 0.25``):
     - Lightweight 2-layer conv head on ``/4`` ``p2`` backbone features predicts
       8-bin unsigned Sobel gradient orientation histograms from the unmasked
       page, preserving 1-2 px iron-gall staff/ledger/accidental strokes while
       ignoring sepia parchment noise.
  4. **MiniCPM-V 4.5 Local Ink-Patch Corruption + HTR-VT Span Masking**:
     - Vectorized GPU-native anisotropic strip/span masking (``45-55%`` of ink
       patches filled with per-page ``p85`` parchment tone) combined with subtle
       local ink-patch fade/blur corruption (``10-20%`` of unmasked ink patches)
       on the student input while teacher sees the clean page.
  5. **Native ``torch.optim.Muon`` + Fused ``AdamW`` + 5-Min Atomic Rolling Autosave**:
     - Routes 2D encoder Transformer matrices to
       ``torch.optim.Muon(adjust_lr_fn="match_rms_adamw")`` and convs/norms/biases/
       predictors to fused ``AdamW`` on CUDA (with clean fallback to ``AdamW``
       on CPU or when ``--optimizer adamw``).
     - Saves ``UniOMRv2Net``-compatible checkpoints atomically via
       ``.tmp`` + ``os.replace`` every ``--save-every-minutes`` (default 5.0 min).
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

# Ensure sibling imports resolve regardless of CWD.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import data  # noqa: E402
from uni_omr import (  # noqa: E402
    _fast_image_is_landscape,
    _pil_to_chw_tensor,
    _resize_fullpage_pil,
    _round_mult16,
)
import uni_omr_v2 as _u2  # noqa: E402
from uni_omr_v2 import (  # noqa: E402
    InkTokenRouterV2,
    UniOMRv2Config,
    UniOMRv2Net,
)

# Compatibility shims in case preset factory functions exist in uni_omr_v2.
uni_omr_v2_small_45m = getattr(
    _u2,
    "uni_omr_v2_small_45m",
    lambda **kw: UniOMRv2Net(UniOMRv2Config.small_45m(**kw)),
)
uni_omr_v2_edge85m = getattr(
    _u2,
    "uni_omr_v2_edge85m",
    lambda **kw: UniOMRv2Net(UniOMRv2Config.edge85m(**kw)),
)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_ENCODER_SUBMODULES = ("backbone", "encoder")


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
    """Load RGB image, resize longest side to ``target_dim`` (mult-16), return ``(3, H, W)`` float32 in ``[0, 1]``."""
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
    """Pad variable-aspect ``(3, H, W)`` tensors to shared mult-16 ``(B, 3, max_H, max_W)`` using p85 paper tone."""
    B = len(tensors)
    max_h = _round_mult16(max(int(t.shape[1]) for t in tensors), min_val=16)
    max_w = _round_mult16(max(int(t.shape[2]) for t in tensors), min_val=16)
    batch = torch.empty((B, 3, max_h, max_w), dtype=torch.float32)
    for i, t in enumerate(tensors):
        _, h, w = t.shape
        if h == max_h and w == max_w:
            batch[i].copy_(t)
        else:
            flat = t.reshape(3, -1)
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

        self._portrait: List[int] = []
        self._landscape: List[int] = []
        if self.batch_size >= 2 and len(self.paths) >= 2:
            for idx, p in enumerate(self.paths):
                if _fast_image_is_landscape(p):
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
        return self._queue.pop(0).result()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


# ---------------------------------------------------------------------------
# Photometric Aug, HTR-VT Span Masking & MiniCPM-V 4.5 Local Ink Corruption
# ---------------------------------------------------------------------------

def estimate_paper_tone_p85(image: torch.Tensor) -> torch.Tensor:
    """Estimate page parchment tone ``(3, 1, 1)`` from 85th-percentile luma pixels (zero host syncs)."""
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
    """Apply mild gamma, contrast, brightness, sepia warmth, and sensor noise jitter to ``(B, 3, H, W)``."""
    r = rng if rng is not None else random
    out = images.float().clone()
    for b in range(out.shape[0]):
        im = out[b]
        if r.random() < 0.75:
            gamma = r.uniform(0.82, 1.22)
            im = im.clamp(min=1e-5).pow(gamma)
        if r.random() < 0.75:
            contrast = r.uniform(0.82, 1.18)
            mean_val = im.mean()
            im = (im - mean_val) * contrast + mean_val
        if r.random() < 0.65:
            im = im + r.uniform(-0.06, 0.06)
        if r.random() < 0.75:
            gains = torch.tensor(
                [r.uniform(0.94, 1.06), r.uniform(0.93, 1.05), r.uniform(0.88, 1.04)],
                device=im.device,
                dtype=im.dtype,
            ).view(3, 1, 1)
            im = im * gains
        if r.random() < 0.50:
            sigma = r.uniform(0.005, 0.015)
            im = im + torch.randn_like(im) * sigma
        out[b] = im.clamp(0.0, 1.0)
    return out.to(images.dtype)


def apply_ink_span_mask_and_corruption(
    images: torch.Tensor,
    local_ink_grid: torch.Tensor,
    mask_ratio_range: Tuple[float, float] = (0.45, 0.55),
    corrupt_ratio_range: Tuple[float, float] = (0.10, 0.20),
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized HTR-VT ink-aware span/strip masking + MiniCPM-V 4.5 local ink-patch corruption.

    1. Masks ``45-55%`` of patches via anisotropic vertical (symbol/measure) +
       horizontal (staff-segment) random spans on ``(H_m, W_m)``, filling masked
       pixels with each page's ``p85`` parchment paper tone.
    2. Applies subtle local ink-patch fade + 3x3 blur corruption to ``10-20%``
       of **unmasked** ink patches so the student learns iron-gall ink
       restoration on visible context as well.

    Returns:
        student_images: ``(B, 3, H, W)`` masked + locally corrupted student input
        grid_mask: ``(B, 1, H_m, W_m)`` bool mask (True = masked span)
        corrupt_grid: ``(B, 1, H_m, W_m)`` bool mask (True = locally corrupted visible ink)
    """
    r = rng if rng is not None else random
    B, _, H, W = images.shape
    _, _, H_p, W_p = local_ink_grid.shape
    device = images.device

    # 1. Estimate per-page p85 parchment paper tone on-device: (B, 3, 1, 1)
    tones = torch.stack([estimate_paper_tone_p85(images[b]) for b in range(B)], dim=0).to(
        device=device, dtype=images.dtype
    )

    # 2. Anisotropic vertical + horizontal span masks on-device (HTR-VT style)
    target_frac = r.uniform(float(mask_ratio_range[0]), float(mask_ratio_range[1]))
    ch_v = max(2, H_p // 8)
    cw_v = max(2, W_p // 3)
    ch_h = max(2, H_p // 4)
    cw_h = max(2, W_p // 8)

    p_v = min(0.85, max(0.05, 1.0 - (1.0 - target_frac) ** 0.60))
    p_h = min(0.85, max(0.05, 1.0 - (1.0 - target_frac) ** 0.40))

    seed_v = (torch.rand((B, 1, ch_v, cw_v), device=device) < p_v).float()
    seed_h = (torch.rand((B, 1, ch_h, cw_h), device=device) < p_h).float()

    mask_v = F.interpolate(seed_v, size=(H_p, W_p), mode="nearest")
    mask_h = F.interpolate(seed_h, size=(H_p, W_p), mode="nearest")
    grid_mask_f = ((mask_v + mask_h) > 0.5).float()
    if H_p > 1 and W_p > 1:
        shift_y = r.randint(0, H_p - 1)
        shift_x = r.randint(0, W_p - 1)
        grid_mask_f = torch.roll(grid_mask_f, shifts=(shift_y, shift_x), dims=(2, 3))

    # Guarantee at least one masked and one visible cell per sample (branchless, zero host syncs)
    grid_mask = grid_mask_f > 0.5
    if H_p * W_p >= 2:
        flat_m = grid_mask.reshape(B, -1)
        all_false = ~flat_m.any(dim=1)
        all_true = flat_m.all(dim=1)
        flat_m = flat_m.clone()
        flat_m[:, 0] = torch.where(all_false, torch.ones_like(all_false), flat_m[:, 0])
        flat_m[:, -1] = torch.where(all_true, torch.zeros_like(all_true), flat_m[:, -1])
        grid_mask = flat_m.view(B, 1, H_p, W_p)
        grid_mask_f = grid_mask.float()

    # 3. MiniCPM-V 4.5 subtle local ink-patch corruption on 10-20% of UNMASKED ink patches
    corrupt_frac = r.uniform(float(corrupt_ratio_range[0]), float(corrupt_ratio_range[1]))
    ink_cells = local_ink_grid > 0.04
    corrupt_grid = (~grid_mask) & ink_cells & (torch.rand((B, 1, H_p, W_p), device=device) < corrupt_frac)

    # Apply subtle fade (towards paper tone) + local 3x3 blur on corrupted patches
    fade_alpha = r.uniform(0.25, 0.55)
    blurred = F.avg_pool2d(images.float(), kernel_size=3, stride=1, padding=1)
    corrupted_vals = (blurred * (1.0 - fade_alpha) + tones.float() * fade_alpha).clamp(0.0, 1.0).to(images.dtype)
    corrupt_px = F.interpolate(corrupt_grid.float(), size=(H, W), mode="nearest") > 0.5
    student_images = torch.where(corrupt_px, corrupted_vals, images)

    # 4. Fill masked span pixels with p85 parchment paper tone
    pixel_masks = F.interpolate(grid_mask_f, size=(H, W), mode="nearest") > 0.5
    student_images = torch.where(pixel_masks, tones, student_images)
    return student_images, grid_mask, corrupt_grid


# ---------------------------------------------------------------------------
# MonkeyOCRv2 / MaskFeat Sobel-HOG Stroke Target & Head
# ---------------------------------------------------------------------------

def compute_sobel_hog_target(
    images: torch.Tensor,
    target_hw: Tuple[int, int],
    n_bins: int = 8,
) -> torch.Tensor:
    """Compute L2-normalized 8-bin unsigned Sobel HOG target ``(B, n_bins, H_t, W_t)`` from clean images.

    Captures 1-2 px iron-gall staff lines, ledger lines, stems, and accidentals
    while suppressing low-gradient sepia parchment background noise.
    """
    img_f = images.float()
    if img_f.shape[1] == 3:
        luma = 0.299 * img_f[:, 0:1] + 0.587 * img_f[:, 1:2] + 0.114 * img_f[:, 2:3]
    else:
        luma = img_f.mean(dim=1, keepdim=True)

    device = luma.device
    with torch.amp.autocast(device_type=device.type, enabled=False):
        luma_f = luma.float()
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        gx = F.conv2d(luma_f, kx, padding=1)
        gy = F.conv2d(luma_f, ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        # Unsigned orientation in [0, pi)
        angle = torch.remainder(torch.atan2(gy, gx), math.pi)
        bin_pos = (angle / math.pi) * float(n_bins)
        b0 = torch.floor(bin_pos).long().clamp(0, n_bins - 1)
        b1 = (b0 + 1) % n_bins
        w1 = bin_pos - torch.floor(bin_pos)
        w0 = 1.0 - w1

        B, _, H, W = luma_f.shape
        hog_full = torch.zeros((B, n_bins, H, W), device=device, dtype=torch.float32)
        hog_full.scatter_add_(1, b0, (mag * w0).float())
        hog_full.scatter_add_(1, b1, (mag * w1).float())

        hog_pool = F.adaptive_avg_pool2d(hog_full, target_hw)
        norm = torch.sqrt((hog_pool * hog_pool).sum(dim=1, keepdim=True) + 1e-6)
        return (hog_pool / norm).detach()


class StrokeHOGHead(nn.Module):
    """Lightweight 2-layer conv head on ``/4`` ``p2`` features predicting 8-bin Sobel HOG."""

    def __init__(self, in_ch: int, n_bins: int = 8) -> None:
        super().__init__()
        mid_ch = max(32, in_ch // 2)
        g = min(8, mid_ch)
        while g > 1 and (mid_ch % g != 0):
            g -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, mid_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_ch, n_bins, kernel_size=1, bias=True),
        )

    def forward(self, p2: torch.Tensor) -> torch.Tensor:
        return self.net(p2)


# ---------------------------------------------------------------------------
# DeepStack 3-Level MLP Predictors & EMA / Gram Teacher Encoders
# ---------------------------------------------------------------------------

class IJEPAPredictor(nn.Module):
    """Lightweight 2-layer MLP predictor (training-only; excluded from saved checkpoints)."""

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


class DeepStackPredictors(nn.Module):
    """Three 2-layer MLP predictors for ``(mem_post_merger, mem_global_mid, mem_final)``."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.pred_post_merger = IJEPAPredictor(d_model)
        self.pred_global_mid = IJEPAPredictor(d_model)
        self.pred_final = IJEPAPredictor(d_model)

    def forward(
        self, mem_levels: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m0, m1, m2 = mem_levels
        return self.pred_post_merger(m0), self.pred_global_mid(m1), self.pred_final(m2)


class EMAEncoderTeacherV2(nn.Module):
    """Frozen EMA (or Gram-anchor) copy of ``UniOMRv2Net``'s ``backbone`` + ``encoder``."""

    def __init__(self, student: UniOMRv2Net) -> None:
        super().__init__()
        self.cfg = copy.deepcopy(student.cfg)
        for name in _ENCODER_SUBMODULES:
            mod = copy.deepcopy(getattr(student, name))
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)
            setattr(self, name, mod)

    @torch.no_grad()
    def update_from(self, student: UniOMRv2Net, decay: float) -> None:
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
        forced_kept_idx: Optional[torch.Tensor] = None,
        forced_routed_coords: Optional[torch.Tensor] = None,
        forced_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, Dict[str, Any]]:
        """Run teacher backbone + HybridAxialMergerEncoder on clean images.

        If ``forced_kept_idx``/``forced_routed_coords``/``forced_info`` are provided
        (used by ``gram_teacher``), extracts tokens at those exact spatial indices
        for 1-to-1 alignment with the EMA teacher and student.
        """
        backbone_in = (
            images.to(memory_format=torch.channels_last)
            if self.backbone.stage1.down[0].weight.is_contiguous(memory_format=torch.channels_last)
            else images
        )
        feat_2d = self.backbone(backbone_in, return_pyramid=False)
        enc = self.encoder
        _, _, H_p, W_p = feat_2d.shape
        x = feat_2d.permute(0, 2, 3, 1).contiguous()

        row_cos, row_sin = enc.axial_rope(W_p, device=x.device, dtype=x.dtype)
        col_cos, col_sin = enc.axial_rope(H_p, device=x.device, dtype=x.dtype)
        for layer in enc.axial_layers:
            x = layer(x, row_cos, row_sin, col_cos, col_sin)

        merged = enc.patch_merger(x)
        merged_2d = merged.permute(0, 3, 1, 2).contiguous()
        _, _, H_m, W_m = merged_2d.shape

        if forced_kept_idx is not None and forced_routed_coords is not None and forced_info is not None:
            routed_tokens, sal_logits = _route_at_fixed_indices(
                router=enc.router,
                merged_2d=merged_2d,
                images=images,
                kept_idx=forced_kept_idx,
                paper_gate=forced_info["paper_gate"],
                k_kept=int(forced_info["k_kept"]),
                n_bg=int(forced_info["n_bg_summary"]),
            )
            routed_coords = forced_routed_coords
            router_info = dict(forced_info)
            router_info["saliency_logits"] = sal_logits
        else:
            routed_tokens, routed_coords, router_info = enc.router(merged_2d, images=images)
            local_ink_grid, paper_gate, effective_paper = InkTokenRouterV2._compute_ink_and_paper_gate(
                images, H_m, W_m
            )
            router_info["local_ink_grid"] = local_ink_grid
            router_info["paper_gate"] = paper_gate
            router_info["effective_paper"] = effective_paper
            router_info["axial_grid_hw"] = (H_p, W_p)

        mem_post_merger = routed_tokens
        grid_hw = router_info["grid_hw"]
        pos2d = enc.pos2d_proj(enc._sinusoidal_pos2d(routed_coords, grid_hw=grid_hw, dtype=routed_tokens.dtype))
        cos2d, sin2d = enc.rope2d(routed_coords, grid_hw=grid_hw, dtype=routed_tokens.dtype)
        mem = routed_tokens + pos2d

        n_global = len(enc.global_layers)
        mid_idx = max(0, (n_global // 2) - 1)
        mem_global_mid = mem_post_merger
        for i, layer in enumerate(enc.global_layers):
            mem = layer(mem, rope_cos=cos2d, rope_sin=sin2d)
            if i == mid_idx:
                mem_global_mid = mem

        mem_final = enc.enc_norm(mem) + pos2d
        return (mem_post_merger, mem_global_mid, mem_final), routed_coords, router_info


def _route_at_fixed_indices(
    router: InkTokenRouterV2,
    merged_2d: torch.Tensor,
    images: torch.Tensor,
    kept_idx: torch.Tensor,
    paper_gate: torch.Tensor,
    k_kept: int,
    n_bg: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather routed tokens from ``merged_2d`` at exact ``kept_idx`` with router gating + bg summary."""
    B, D, H_m, W_m = merged_2d.shape
    N = H_m * W_m

    sal_logits = router.saliency_net(merged_2d)
    p_gate = paper_gate.to(device=merged_2d.device, dtype=sal_logits.dtype)
    sal_prob = torch.sigmoid(sal_logits) * p_gate

    if router.use_ink_prior:
        local_ink, _, _ = InkTokenRouterV2._compute_ink_and_paper_gate(images, H_m, W_m)
        local_ink = local_ink.to(sal_logits.dtype)
        flat_f = local_ink.flatten(1).float()
        denom = (
            (flat_f.mean(dim=-1) + 2.5 * flat_f.std(dim=-1, unbiased=False))
            .clamp(min=1e-4)
            .view(B, 1, 1, 1)
            .to(local_ink.dtype)
        )
        local_ink_norm = (local_ink / denom).clamp(0.0, 1.0)
        score = sal_prob + torch.sigmoid(router.ink_prior_scale).to(sal_logits.dtype) * local_ink_norm
    else:
        score = sal_prob

    score_flat = score.flatten(1)
    tokens_flat = merged_2d.permute(0, 2, 3, 1).reshape(B, N, D)

    if k_kept >= N or n_bg == 0:
        gate = (0.5 + 0.5 * torch.sigmoid(score_flat)).unsqueeze(-1)
        return tokens_flat * gate, sal_logits

    kept_scores = torch.gather(score_flat, 1, kept_idx)
    kept_gate = (0.5 + 0.5 * torch.sigmoid(kept_scores)).unsqueeze(-1)
    idx_feat = kept_idx.unsqueeze(-1).expand(-1, -1, D)
    kept_tokens = torch.gather(tokens_flat, 1, idx_feat) * kept_gate

    kept_mask = torch.zeros(B, N, device=merged_2d.device, dtype=torch.bool)
    kept_mask.scatter_(1, kept_idx, True)

    bg_q = router.bg_queries.expand(B, -1, -1)
    bg_q_proj = F.linear(bg_q, router.bg_key_proj.weight.t()).to(tokens_flat.dtype)
    bg_logits = torch.bmm(bg_q_proj, tokens_flat.transpose(1, 2)) / math.sqrt(D)
    bg_logits = bg_logits - score_flat.unsqueeze(1)
    bg_logits = bg_logits.masked_fill(kept_mask.unsqueeze(1), -1e4)
    bg_attn = F.softmax(bg_logits, dim=-1)
    pooled = torch.bmm(bg_attn, tokens_flat)
    bg_tokens = router.bg_out_norm(router.bg_val_proj(pooled))

    return torch.cat([kept_tokens, bg_tokens], dim=1), sal_logits


def encode_student_aligned_v2(
    student: UniOMRv2Net,
    student_images: torch.Tensor,
    teacher_kept_idx: torch.Tensor,
    teacher_routed_coords: torch.Tensor,
    teacher_info: Dict[str, Any],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Run student backbone + HybridAxialMergerEncoder aligned 1-to-1 to teacher ``kept_idx``.

    Returns:
        mem_levels_s: ``(mem_post_merger, mem_global_mid, mem_final)``
        sal_logits_s: ``(B, 1, H_m, W_m)``
        p2: ``(B, p2_ch, H/4, W/4)`` high-res features for Stroke-HOG head
    """
    backbone_in = (
        student_images.to(memory_format=torch.channels_last)
        if student.backbone.stage1.down[0].weight.is_contiguous(memory_format=torch.channels_last)
        else student_images
    )
    feat_2d, p2, _ = student.backbone(backbone_in, return_pyramid=True)
    enc = student.encoder
    _, _, H_p, W_p = feat_2d.shape
    x = feat_2d.permute(0, 2, 3, 1).contiguous()

    row_cos, row_sin = enc.axial_rope(W_p, device=x.device, dtype=x.dtype)
    col_cos, col_sin = enc.axial_rope(H_p, device=x.device, dtype=x.dtype)

    use_ckpt = student.training and student.cfg.grad_checkpoint and torch.is_grad_enabled()
    for layer in enc.axial_layers:
        if use_ckpt:
            x = _grad_ckpt(
                lambda _x, _rc, _rs, _cc, _cs, _l=layer: _l(_x, _rc, _rs, _cc, _cs),
                x, row_cos, row_sin, col_cos, col_sin,
                use_reentrant=False,
            )
        else:
            x = layer(x, row_cos, row_sin, col_cos, col_sin)

    merged = enc.patch_merger(x)
    merged_2d = merged.permute(0, 3, 1, 2).contiguous()

    routed_tokens, sal_logits_s = _route_at_fixed_indices(
        router=enc.router,
        merged_2d=merged_2d,
        images=student_images,
        kept_idx=teacher_kept_idx,
        paper_gate=teacher_info["paper_gate"],
        k_kept=int(teacher_info["k_kept"]),
        n_bg=int(teacher_info["n_bg_summary"]),
    )

    mem_post_merger = routed_tokens
    grid_hw = teacher_info["grid_hw"]
    pos2d = enc.pos2d_proj(enc._sinusoidal_pos2d(teacher_routed_coords, grid_hw=grid_hw, dtype=routed_tokens.dtype))
    cos2d, sin2d = enc.rope2d(teacher_routed_coords, grid_hw=grid_hw, dtype=routed_tokens.dtype)
    mem = routed_tokens + pos2d

    n_global = len(enc.global_layers)
    mid_idx = max(0, (n_global // 2) - 1)
    mem_global_mid = mem_post_merger
    for i, layer in enumerate(enc.global_layers):
        if use_ckpt and (mem.shape[0] * mem.shape[1] > 12000):
            mem = _grad_ckpt(
                lambda _m, _c, _s, _l=layer: _l(_m, rope_cos=_c, rope_sin=_s),
                mem, cos2d, sin2d,
                use_reentrant=False,
            )
        else:
            mem = layer(mem, rope_cos=cos2d, rope_sin=sin2d)
        if i == mid_idx:
            mem_global_mid = mem

    mem_final = enc.enc_norm(mem) + pos2d
    return (mem_post_merger, mem_global_mid, mem_final), sal_logits_s, p2


# ---------------------------------------------------------------------------
# Losses: V-JEPA 2.1 Dense DeepStack + DINOv3 Gram Anchoring + Stroke-HOG
# ---------------------------------------------------------------------------

def compute_gram_anchoring_loss(
    feat_student: torch.Tensor,
    feat_teacher: torch.Tensor,
    ink_mask: Optional[torch.Tensor] = None,
    max_tokens: int = 512,
) -> torch.Tensor:
    """DINOv3 Ink-Token Gram Anchoring loss: MSE between pairwise Gram matrices ``F_s @ F_s.T`` and ``F_t @ F_t.T``.

    Subsamples up to ``M <= max_tokens`` ink tokens per page using a deterministic
    stride, L2-normalizes features, and computes ``F.mse_loss(G_s, G_t)``.
    Returns exact ``0.0`` when ``feat_student == feat_teacher``.
    """
    B, K, _ = feat_student.shape
    max_m = max(1, int(max_tokens))
    losses: List[torch.Tensor] = []

    for b in range(B):
        s_b = feat_student[b].float()
        t_b = feat_teacher[b].detach().float()
        if ink_mask is not None:
            idx = torch.nonzero(ink_mask[b], as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                idx = torch.arange(K, device=s_b.device)
        else:
            idx = torch.arange(K, device=s_b.device)

        n_sel = int(idx.numel())
        if n_sel > max_m:
            # Deterministic stride subsampling
            stride = max(1, n_sel // max_m)
            idx = idx[::stride][:max_m]

        f_s = F.normalize(s_b.index_select(0, idx), p=2, dim=-1, eps=1e-6)
        f_t = F.normalize(t_b.index_select(0, idx), p=2, dim=-1, eps=1e-6)
        g_s = torch.matmul(f_s, f_s.transpose(0, 1))
        g_t = torch.matmul(f_t, f_t.transpose(0, 1))
        losses.append(F.mse_loss(g_s, g_t))

    return torch.stack(losses).mean()


def compute_vjepa21_losses(
    pred_levels: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    target_levels: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mem_final_student: torch.Tensor,
    mem_final_gram: torch.Tensor,
    hog_pred: torch.Tensor,
    hog_target: torch.Tensor,
    grid_mask: torch.Tensor,
    kept_idx: torch.Tensor,
    local_ink_grid: torch.Tensor,
    sal_logits_student: torch.Tensor,
    sal_logits_teacher: torch.Tensor,
    effective_paper: torch.Tensor,
    level_weights: Tuple[float, float, float] = (0.25, 0.35, 1.0),
    masked_weight: float = 3.0,
    visible_weight: float = 0.5,
    bg_weight: float = 0.2,
    gram_weight: float = 1.0,
    gram_max_tokens: int = 512,
    stroke_hog_weight: float = 0.25,
    router_cons_weight: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """Compute combined V-JEPA 2.1 DeepStack + Gram Anchoring + Stroke-HOG + Router loss."""
    k_kept = kept_idx.shape[1]
    masked_kept = torch.gather(grid_mask.flatten(1), 1, kept_idx)           # (B, k_kept) bool
    visible_kept = ~masked_kept
    ink_kept = torch.gather(local_ink_grid.flatten(1), 1, kept_idx).float() # (B, k_kept)
    is_ink = ink_kept > 0.04

    ink_bg_w = torch.where(is_ink, 3.0, float(bg_weight))
    w_masked = ink_bg_w * masked_kept.float()
    w_visible = ink_bg_w * visible_kept.float()
    w_masked_sum = w_masked.sum().clamp(min=1e-6)
    w_visible_sum = w_visible.sum().clamp(min=1e-6)
    mw_denom = max(1e-6, float(masked_weight) + float(visible_weight))

    level_masked_losses: List[torch.Tensor] = []
    level_visible_losses: List[torch.Tensor] = []
    level_totals: List[torch.Tensor] = []

    for pred_l, tgt_l in zip(pred_levels, target_levels):
        p_f = pred_l[:, :k_kept, :].float()
        t_f = tgt_l[:, :k_kept, :].detach().float()
        l1_tok = F.smooth_l1_loss(p_f, t_f, reduction="none").mean(dim=-1)
        cos_tok = 1.0 - F.cosine_similarity(p_f, t_f, dim=-1, eps=1e-6)
        tok_err = l1_tok + cos_tok

        l_m = (tok_err * w_masked).sum() / w_masked_sum
        l_v = (tok_err * w_visible).sum() / w_visible_sum
        l_tot = (float(masked_weight) * l_m + float(visible_weight) * l_v) / mw_denom

        level_masked_losses.append(l_m)
        level_visible_losses.append(l_v)
        level_totals.append(l_tot)

    w0, w1, w2 = (float(level_weights[0]), float(level_weights[1]), float(level_weights[2]))
    jepa_masked = level_masked_losses[2]
    jepa_visible = level_visible_losses[2]
    deep_sup = w0 * level_totals[0] + w1 * level_totals[1]
    jepa_total = deep_sup + w2 * level_totals[2]

    # 2. DINOv3 Ink-Token Gram Anchoring on kept ink tokens
    if float(gram_weight) > 0.0:
        gram_loss = compute_gram_anchoring_loss(
            feat_student=mem_final_student[:, :k_kept, :],
            feat_teacher=mem_final_gram[:, :k_kept, :],
            ink_mask=is_ink,
            max_tokens=gram_max_tokens,
        )
    else:
        gram_loss = jepa_total.new_zeros((), dtype=torch.float32)

    # 3. MonkeyOCRv2 / MaskFeat Sobel-HOG Stroke Reconstruction on /4 p2
    if float(stroke_hog_weight) > 0.0:
        mask_p2 = F.interpolate(grid_mask.float(), size=hog_pred.shape[-2:], mode="nearest")
        hog_w = 1.0 + 2.0 * mask_p2  # Emphasize masked stroke reconstruction while preserving visible strokes
        hog_err = F.mse_loss(hog_pred.float(), hog_target.detach().float(), reduction="none").mean(dim=1, keepdim=True)
        stroke_hog_loss = (hog_err * hog_w).sum() / hog_w.sum().clamp(min=1e-6)
    else:
        stroke_hog_loss = jepa_total.new_zeros((), dtype=torch.float32)

    # 4. Router saliency consistency on unmasked interior paper regions
    unmasked_paper = (~grid_mask) & effective_paper
    up_f = unmasked_paper.float()
    if float(router_cons_weight) > 0.0:
        tgt_prob = torch.sigmoid(sal_logits_teacher.detach().float())
        bce_map = F.binary_cross_entropy_with_logits(sal_logits_student.float(), tgt_prob, reduction="none")
        router_loss = (bce_map * up_f).sum() / up_f.sum().clamp(min=1e-6)
    else:
        router_loss = jepa_total.new_zeros((), dtype=torch.float32)

    total_loss = (
        jepa_total
        + float(gram_weight) * gram_loss
        + float(stroke_hog_weight) * stroke_hog_loss
        + float(router_cons_weight) * router_loss
    )
    masked_ink_frac = (is_ink & masked_kept).sum().float() / is_ink.sum().float().clamp(min=1.0)

    return {
        "loss": total_loss,
        "jepa_masked": jepa_masked,
        "jepa_visible": jepa_visible,
        "deep_sup": deep_sup,
        "gram": gram_loss,
        "stroke_hog": stroke_hog_loss,
        "router_loss": router_loss,
        "masked_ink_frac": masked_ink_frac,
    }


# ---------------------------------------------------------------------------
# Optimizer Routing (Native torch.optim.Muon + Fused AdamW) & Checkpointing
# ---------------------------------------------------------------------------

def freeze_non_encoder_params_v2(model: UniOMRv2Net) -> List[nn.Parameter]:
    """Freeze all non-encoder params (decoder, lm_head, mtp_heads, aux heads) and return trainable encoder params."""
    for p in model.parameters():
        p.requires_grad_(False)
    trainable: List[nn.Parameter] = []
    for name in _ENCODER_SUBMODULES:
        mod = getattr(model, name)
        for p in mod.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


class HybridOptimizer:
    """Wraps 1 or 2 PyTorch optimizers (e.g. ``Muon`` + ``AdamW``) behind a unified step/LR interface."""

    def __init__(self, optimizers: Sequence[Tuple[torch.optim.Optimizer, float]], mode: str) -> None:
        self._opts: List[Tuple[torch.optim.Optimizer, float]] = list(optimizers)
        self.mode = mode

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt, _ in self._opts:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt, _ in self._opts:
            opt.step()

    def set_lr_mult(self, mult: float) -> None:
        m = float(mult)
        for opt, base_lr in self._opts:
            for pg in opt.param_groups:
                pg["lr"] = base_lr * m


def build_optimizer_v2(
    student: UniOMRv2Net,
    predictors: DeepStackPredictors,
    stroke_hog_head: StrokeHOGHead,
    optimizer_name: str = "muon",
    lr: float = 3e-4,
    muon_lr: Optional[float] = None,
    weight_decay: float = 0.04,
    device_type: str = "cpu",
) -> Tuple[HybridOptimizer, List[nn.Parameter]]:
    """Build native ``torch.optim.Muon`` (2D encoder Transformer matrices) + fused ``AdamW`` (rest).

    Cleanly falls back to ``AdamW`` on CPU, when ``optimizer_name == "adamw"``,
    or if ``torch.optim.Muon`` is unavailable.
    """
    muon_params: List[nn.Parameter] = []
    adamw_params: List[nn.Parameter] = []

    # Partition encoder params: 2D Transformer Linear weights in student.encoder -> Muon
    for name, p in student.named_parameters():
        if not p.requires_grad:
            continue
        is_enc_transformer_2d = (
            name.startswith("encoder.")
            and (p.ndim == 2)
            and ("saliency_net" not in name)
            and ("bg_queries" not in name)
        )
        if is_enc_transformer_2d:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    # Training-only predictors and Stroke-HOG head always use AdamW
    for mod in (predictors, stroke_hog_head):
        for p in mod.parameters():
            if p.requires_grad:
                adamw_params.append(p)

    all_trainable = muon_params + adamw_params

    def _make_adamw(params: List[nn.Parameter], cur_lr: float) -> torch.optim.Optimizer:
        if device_type == "cuda":
            try:
                return torch.optim.AdamW(params, lr=cur_lr, weight_decay=weight_decay, fused=True)
            except (RuntimeError, TypeError):
                pass
        return torch.optim.AdamW(params, lr=cur_lr, weight_decay=weight_decay)

    want_muon = (
        optimizer_name.lower() == "muon"
        and device_type == "cuda"
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
                adjust_lr_fn="match_rms_adamw",
            )
            opts: List[Tuple[torch.optim.Optimizer, float]] = [(opt_muon, eff_muon_lr)]
            if adamw_params:
                opts.append((_make_adamw(adamw_params, float(lr)), float(lr)))
            return HybridOptimizer(opts, mode="muon+adamw"), all_trainable
        except (RuntimeError, TypeError):
            pass

    return HybridOptimizer([(_make_adamw(all_trainable, float(lr)), float(lr))], mode="adamw"), all_trainable


def build_uniomr_v2_checkpoint_state_dict(
    student: UniOMRv2Net,
    teacher: Optional[EMAEncoderTeacherV2] = None,
    use_teacher_encoder: bool = True,
) -> Dict[str, torch.Tensor]:
    """Build a 100% ``UniOMRv2Net``-compatible CPU state_dict (no predictor/HOG keys)."""
    sd = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    if use_teacher_encoder and teacher is not None:
        for name in _ENCODER_SUBMODULES:
            t_sd = getattr(teacher, name).state_dict()
            for sub_k, v in t_sd.items():
                full_k = f"{name}.{sub_k}"
                if full_k in sd:
                    sd[full_k] = v.detach().cpu().clone()
    return sd


def save_ijepa_v2_checkpoint(
    path: Path,
    student: UniOMRv2Net,
    teacher: Optional[EMAEncoderTeacherV2],
    cfg: UniOMRv2Config,
    step: int,
    loss_val: float = float("nan"),
    vocab: Optional[Sequence[str]] = None,
    use_teacher_encoder: bool = True,
) -> None:
    """Atomically save a ``UniOMRv2Net``-compatible checkpoint via ``.tmp`` + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    cfg_dict = asdict(cfg)
    payload = {
        "cfg": cfg_dict,
        "config": cfg_dict,
        "step": int(step),
        "loss": float(loss_val),
        "vocab": list(vocab) if vocab is not None else list(data.VOCAB),
        "model": build_uniomr_v2_checkpoint_state_dict(
            student, teacher=teacher, use_teacher_encoder=use_teacher_encoder
        ),
    }
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Schedules & Single Forward-Loss Step
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


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        return None


def vjepa21_forward_loss(
    student: UniOMRv2Net,
    teacher: EMAEncoderTeacherV2,
    gram_teacher: EMAEncoderTeacherV2,
    predictors: DeepStackPredictors,
    stroke_hog_head: StrokeHOGHead,
    images: torch.Tensor,
    rng: random.Random,
    use_bf16: bool = False,
    device_type: str = "cpu",
    level_weights: Tuple[float, float, float] = (0.25, 0.35, 1.0),
    masked_weight: float = 3.0,
    visible_weight: float = 0.5,
    gram_weight: float = 1.0,
    gram_max_tokens: int = 512,
    stroke_hog_weight: float = 0.25,
    router_cons_weight: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """Run one full V-JEPA 2.1 + Gram Anchoring + Stroke-HOG forward pass and return loss dict."""
    amp_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if use_bf16
        else _nullcontext()
    )
    with amp_ctx:
        # 1. EMA teacher on clean archival page
        target_levels, routed_coords, teacher_info = teacher.encode(images)
        local_ink_grid = teacher_info["local_ink_grid"]
        kept_idx = teacher_info["kept_idx"]

        # 2. Frozen Gram teacher on clean archival page at exact teacher spatial indices
        if float(gram_weight) > 0.0:
            gram_levels, _, _ = gram_teacher.encode(
                images,
                forced_kept_idx=kept_idx,
                forced_routed_coords=routed_coords,
                forced_info=teacher_info,
            )
            mem_final_gram = gram_levels[2]
        else:
            mem_final_gram = target_levels[2]

        # 3. Student input: mild photometric aug + MiniCPM-V 4.5 local ink corruption + HTR-VT span masking
        aug_images = apply_mild_photometric_aug(images, rng=rng)
        student_images, grid_mask, _ = apply_ink_span_mask_and_corruption(
            aug_images,
            local_ink_grid,
            mask_ratio_range=(0.45, 0.55),
            corrupt_ratio_range=(0.10, 0.20),
            rng=rng,
        )

        # 4. Student encoder aligned 1-to-1 to teacher kept_idx
        mem_levels_s, sal_logits_s, p2 = encode_student_aligned_v2(
            student=student,
            student_images=student_images,
            teacher_kept_idx=kept_idx,
            teacher_routed_coords=routed_coords,
            teacher_info=teacher_info,
        )
        pred_levels = predictors(mem_levels_s)

        # 5. Sobel-HOG target from clean image & /4 p2 prediction
        hog_pred = stroke_hog_head(p2)
        hog_target = compute_sobel_hog_target(images, target_hw=(p2.shape[2], p2.shape[3]), n_bins=8)

        losses = compute_vjepa21_losses(
            pred_levels=pred_levels,
            target_levels=target_levels,
            mem_final_student=mem_levels_s[2],
            mem_final_gram=mem_final_gram,
            hog_pred=hog_pred,
            hog_target=hog_target,
            grid_mask=grid_mask,
            kept_idx=kept_idx,
            local_ink_grid=local_ink_grid,
            sal_logits_student=sal_logits_s,
            sal_logits_teacher=teacher_info["saliency_logits"],
            effective_paper=teacher_info["effective_paper"],
            level_weights=level_weights,
            masked_weight=masked_weight,
            visible_weight=visible_weight,
            gram_weight=gram_weight,
            gram_max_tokens=gram_max_tokens,
            stroke_hog_weight=stroke_hog_weight,
            router_cons_weight=router_cons_weight,
        )
    return losses


# ---------------------------------------------------------------------------
# Main Pretraining Entrypoint
# ---------------------------------------------------------------------------

def _parse_level_weights(spec: str) -> Tuple[float, float, float]:
    parts = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError(f"--level-weights must have 3 comma-separated floats, got: {spec!r}")
    return (parts[0], parts[1], parts[2])


def _load_or_build_model(args: argparse.Namespace) -> Tuple[UniOMRv2Net, UniOMRv2Config, List[str]]:
    init_vocab = list(data.VOCAB)
    if args.init_from:
        ck = torch.load(str(args.init_from), map_location="cpu", weights_only=True)
        if not isinstance(ck, dict) or "model" not in ck:
            raise ValueError(f"--init-from must be a checkpoint dict with 'model' key: {args.init_from}")
        raw_cfg = dict(ck.get("cfg") or ck.get("config") or {})
        if "backbone_channels" in raw_cfg:
            raw_cfg["backbone_channels"] = tuple(raw_cfg["backbone_channels"])
        if "backbone_blocks" in raw_cfg:
            raw_cfg["backbone_blocks"] = tuple(raw_cfg["backbone_blocks"])
        if args.grad_checkpoint is not None:
            raw_cfg["grad_checkpoint"] = bool(args.grad_checkpoint)
        cfg = UniOMRv2Config(**raw_cfg) if raw_cfg else UniOMRv2Config.small_45m()
        if "vocab" in ck:
            init_vocab = list(ck["vocab"])
        model = UniOMRv2Net(cfg)
        inc = model.load_state_dict(ck["model"], strict=False)
        print(
            f"[V-JEPA2.1 init] loaded {args.init_from} "
            f"(missing={len(inc.missing_keys)}, unexpected={len(inc.unexpected_keys)})",
            flush=True,
        )
        return model, cfg, init_vocab

    overrides: Dict[str, Any] = {}
    if args.grad_checkpoint is not None:
        overrides["grad_checkpoint"] = bool(args.grad_checkpoint)
    if args.preset == "edge85m":
        model = uni_omr_v2_edge85m(**overrides)
    else:
        model = uni_omr_v2_small_45m(**overrides)
    print(f"[V-JEPA2.1 init] initialized fresh UniOMRv2Net preset={args.preset}", flush=True)
    return model, model.cfg, init_vocab


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

    student, cfg, init_vocab = _load_or_build_model(args)
    student = student.to(device)
    if args.channels_last:
        student.backbone.to(memory_format=torch.channels_last)

    freeze_non_encoder_params_v2(student)

    # EMA teacher (updated every step) & frozen Gram teacher (step-0 / --init-from snapshot)
    teacher = EMAEncoderTeacherV2(student).to(device)
    gram_teacher = EMAEncoderTeacherV2(student).to(device)
    if args.channels_last:
        teacher.backbone.to(memory_format=torch.channels_last)
        gram_teacher.backbone.to(memory_format=torch.channels_last)

    predictors = DeepStackPredictors(cfg.d_model).to(device)
    stroke_hog_head = StrokeHOGHead(in_ch=student.backbone.p2_ch, n_bins=8).to(device)

    opt, all_trainable = build_optimizer_v2(
        student=student,
        predictors=predictors,
        stroke_hog_head=stroke_hog_head,
        optimizer_name=args.optimizer,
        lr=args.lr,
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
        device_type=device.type,
    )
    print(f"[V-JEPA2.1 opt] mode={opt.mode} lr={args.lr:.2e} trainable_tensors={len(all_trainable)}", flush=True)

    image_paths = discover_unlabeled_images(args.image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found under {args.image_dir}")
    print(
        f"[V-JEPA2.1 data] discovered {len(image_paths)} images in {args.image_dir} "
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
    level_weights = _parse_level_weights(args.level_weights)

    student.train()
    predictors.train()
    stroke_hog_head.train()
    teacher.eval()
    gram_teacher.eval()

    best_ema_loss = float("inf")
    ema_loss_track: Optional[float] = None
    oom_skips = 0
    history: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    last_autosave_t = time.monotonic()
    autosave_interval_s = max(0.0, float(args.save_every_minutes) * 60.0)

    try:
        for step in range(1, args.steps + 1):
            lr_mult = cosine_lr_mult(step - 1, args.steps, warmup_steps=warmup_steps)
            opt.set_lr_mult(lr_mult)
            opt.zero_grad(set_to_none=True)

            acc = {
                "loss": 0.0,
                "jepa_masked": 0.0,
                "jepa_visible": 0.0,
                "deep_sup": 0.0,
                "gram": 0.0,
                "stroke_hog": 0.0,
                "router_loss": 0.0,
                "masked_ink_frac": 0.0,
            }
            valid_micros = 0

            for _ in range(args.grad_accum):
                batch_cpu = next(loader)
                images = batch_cpu.to(device, non_blocking=(device.type == "cuda"))
                try:
                    losses = vjepa21_forward_loss(
                        student=student,
                        teacher=teacher,
                        gram_teacher=gram_teacher,
                        predictors=predictors,
                        stroke_hog_head=stroke_hog_head,
                        images=images,
                        rng=rng,
                        use_bf16=use_bf16,
                        device_type=device.type,
                        level_weights=level_weights,
                        masked_weight=args.masked_weight,
                        visible_weight=args.visible_weight,
                        gram_weight=args.gram_weight,
                        gram_max_tokens=args.gram_max_tokens,
                        stroke_hog_weight=args.stroke_hog_weight,
                        router_cons_weight=args.router_cons_weight,
                    )
                    scaled_loss = losses["loss"] / float(args.grad_accum)
                    if not torch.isfinite(scaled_loss):
                        print(f"[V-JEPA2.1 warn] non-finite loss at step {step}; skipping micro-batch", flush=True)
                        continue
                    scaled_loss.backward()
                    for k in acc:
                        acc[k] += float(losses[k].detach().item())
                    valid_micros += 1
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        oom_skips += 1
                        opt.zero_grad(set_to_none=True)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        print(
                            f"[V-JEPA2.1 OOM] skipped micro-batch at step {step} (total_skips={oom_skips})",
                            flush=True,
                        )
                        valid_micros = 0
                        break
                    raise

            if valid_micros == 0:
                continue

            torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
            opt.step()

            ema_decay = cosine_ema_decay(step, args.steps, start=args.ema_start, end=args.ema_end)
            teacher.update_from(student, decay=ema_decay)

            avg = {k: v / valid_micros for k, v in acc.items()}
            ema_loss_track = avg["loss"] if ema_loss_track is None else (0.95 * ema_loss_track + 0.05 * avg["loss"])

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
                    "loss": round(avg["loss"], 5),
                    "ema_loss": round(ema_loss_track, 5),
                    "jepa_masked": round(avg["jepa_masked"], 5),
                    "jepa_visible": round(avg["jepa_visible"], 5),
                    "deep_sup": round(avg["deep_sup"], 5),
                    "gram": round(avg["gram"], 5),
                    "stroke_hog": round(avg["stroke_hog"], 5),
                    "router_loss": round(avg["router_loss"], 5),
                    "masked_ink_frac": round(avg["masked_ink_frac"], 3),
                    "lr_mult": round(lr_mult, 5),
                    "ema_decay": round(ema_decay, 5),
                    "vram_gb": round(vram_gb, 2),
                    "steps_per_sec": round(sps, 2),
                    "oom_skips": oom_skips,
                }
                history.append(rec)
                print(
                    f"[V-JEPA2.1 step {step:5d}/{args.steps}] "
                    f"loss={avg['loss']:.4f} (ema={ema_loss_track:.4f}) "
                    f"m={avg['jepa_masked']:.4f} v={avg['jepa_visible']:.4f} "
                    f"deep={avg['deep_sup']:.4f} gram={avg['gram']:.4f} "
                    f"hog={avg['stroke_hog']:.4f} mask_ink={avg['masked_ink_frac']:.2f} "
                    f"decay={ema_decay:.4f} vram={vram_gb:.2f}GB ({sps:.2f} it/s)",
                    flush=True,
                )

            if ema_loss_track < best_ema_loss and (step >= min(50, args.steps) or step == args.steps):
                best_ema_loss = ema_loss_track
                save_ijepa_v2_checkpoint(
                    out_dir / "uni_omr_v2_ijepa_best.pt",
                    student=student,
                    teacher=teacher,
                    cfg=cfg,
                    step=step,
                    loss_val=best_ema_loss,
                    vocab=init_vocab,
                    use_teacher_encoder=not args.save_student_weights,
                )

            if args.save_every > 0 and step % args.save_every == 0:
                save_ijepa_v2_checkpoint(
                    out_dir / f"uni_omr_v2_ijepa_step{step}.pt",
                    student=student,
                    teacher=teacher,
                    cfg=cfg,
                    step=step,
                    loss_val=ema_loss_track,
                    vocab=init_vocab,
                    use_teacher_encoder=not args.save_student_weights,
                )

            # 5-minute atomic rolling autosave
            if autosave_interval_s > 0.0 and (time.monotonic() - last_autosave_t) >= autosave_interval_s:
                save_ijepa_v2_checkpoint(
                    out_dir / "uni_omr_v2_ijepa_autosave.pt",
                    student=student,
                    teacher=teacher,
                    cfg=cfg,
                    step=step,
                    loss_val=ema_loss_track,
                    vocab=init_vocab,
                    use_teacher_encoder=not args.save_student_weights,
                )
                last_autosave_t = time.monotonic()
    finally:
        loader.close()

    save_ijepa_v2_checkpoint(
        out_dir / "uni_omr_v2_ijepa_last.pt",
        student=student,
        teacher=teacher,
        cfg=cfg,
        step=args.steps,
        loss_val=ema_loss_track if ema_loss_track is not None else float("nan"),
        vocab=init_vocab,
        use_teacher_encoder=not args.save_student_weights,
    )
    summary = {
        "steps": args.steps,
        "num_images": len(image_paths),
        "optimizer": opt.mode,
        "best_ema_loss": best_ema_loss,
        "final_ema_loss": ema_loss_track,
        "oom_skips": oom_skips,
        "history": history,
    }
    (out_dir / "vjepa21_train_log.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="V-JEPA 2.1 + DINOv3 Gram Anchoring + MonkeyOCRv2 Stroke-HOG SSL pretraining for UniOMR v2."
    )
    p.add_argument("--init-from", type=str, default=None, help="Optional warm-start UniOMRv2 checkpoint (.pt)")
    p.add_argument("--preset", type=str, choices=("small_45m", "edge85m"), default="small_45m")
    p.add_argument("--image-dir", type=str, required=True, help="Directory of unlabeled pages (e.g. D:\\tmp\\unlabeled-10k-real\\rism_hw)")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for checkpoints and logs")
    p.add_argument("--steps", type=int, default=4000, help="Total optimizer steps (default: 4000)")
    p.add_argument("--batch-size", type=int, default=2, help="Per-microbatch size (default: 2)")
    p.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    p.add_argument("--max-dim", type=int, default=1280, help="Max image dimension (default: 1280)")
    p.add_argument("--optimizer", type=str, choices=("muon", "adamw"), default="muon")
    p.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate (default: 3e-4)")
    p.add_argument("--muon-lr", type=float, default=None, help="Optional separate peak LR for Muon group")
    p.add_argument("--weight-decay", type=float, default=0.04, help="Weight decay (default: 0.04)")
    p.add_argument("--warmup-steps", type=int, default=150, help="Linear LR warmup steps (default: 150)")
    p.add_argument("--ema-start", type=float, default=0.996, help="Initial EMA decay (default: 0.996)")
    p.add_argument("--ema-end", type=float, default=0.999, help="Final EMA decay (default: 0.999)")
    p.add_argument("--level-weights", type=str, default="0.25,0.35,1.0", help="DeepStack level weights (post_merger,global_mid,final)")
    p.add_argument("--masked-weight", type=float, default=3.0, help="Weight on masked ink token prediction (default: 3.0)")
    p.add_argument("--visible-weight", type=float, default=0.5, help="Weight on visible context ink token prediction (default: 0.5)")
    p.add_argument("--gram-weight", type=float, default=1.0, help="DINOv3 Gram anchoring loss weight (default: 1.0)")
    p.add_argument("--gram-max-tokens", type=int, default=512, help="Max subsampled ink tokens for Gram matrix (default: 512)")
    p.add_argument("--stroke-hog-weight", type=float, default=0.25, help="Sobel-HOG stroke reconstruction weight (default: 0.25)")
    p.add_argument("--router-cons-weight", type=float, default=0.2, help="Unmasked paper router saliency BCE weight (default: 0.2)")
    p.add_argument("--bf16", action="store_true", help="Enable bfloat16 autocast")
    p.add_argument("--channels-last", action="store_true", help="Use NHWC channels_last for ConvBackbone")
    p.add_argument("--grad-checkpoint", dest="grad_checkpoint", action="store_true", default=None)
    p.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    p.add_argument("--no-multi-scale", action="store_true", help="Disable random multi-scale resizing")
    p.add_argument("--save-student-weights", action="store_true", help="Save student encoder instead of EMA teacher")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-every", type=int, default=500, help="Save periodic checkpoint every N steps (default: 500)")
    p.add_argument("--save-every-minutes", type=float, default=5.0, help="Atomic rolling autosave interval in minutes (default: 5.0)")
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
