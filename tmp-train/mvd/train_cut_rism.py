"""Geometry-Locked CUT (Contrastive Unpaired Translation) RISM Style Transfer Pipeline.

Translates clean/wobbled ``hw-score-sim`` full pages (e.g. ``D:\\tmp\\hw-sim-v04-clean``,
``D:\\tmp\\hw-sim-v05-dense``) into real 18th-19th-century RISM archival ink,
quill-stroke, bleed-through, and parchment textures sampled from unlabeled RISM
manuscript pages (e.g. ``D:\\tmp\\unlabeled-10k-real\\rism_hw``), while preserving
exact MusicXML / ``bekern+`` ground truth via a hard spatial-skeleton & vertical
Sobel-edge geometry lock (``<= 1.5 px`` notehead / staff-line drift).

Architecture & Losses:
  1. **Lightweight Fully-Convolutional ResNet Generator** (~11.4M params at
     ``ngf=64, n_blocks=9``): ``2x`` downsampling -> ``6-9`` ResBlocks ->
     bilinear ``2x`` upsampling + residual skip in ``[0, 1]`` RGB. Arbitrary
     ``(H, W)`` inputs are reflect-padded to a multiple of 4 and cropped back
     to exact ``(H, W)``.
  2. **PatchGAN Discriminator + Multi-Layer PatchNCE (CUT, Park et al. ECCV 2020)**:
     LSGAN adversarial loss + 2-layer MLP projection heads (``PatchSampleMLP``)
     computing contrastive PatchNCE between encoder features of ``X`` and
     ``G(X)``, plus identity NCE on real RISM ``Y`` and ``G(Y)``.
  3. **Hard Spatial-Skeleton / Ink-Centroid Geometry Lock**:
     High-pass ink map ``ink(img) = clamp((local_bg - luma) / local_bg, 0, 1)``
     is invariant to parchment tint/vignetting while locking ink geometry.
     Enforces ``L1(ink(G(X)), ink(X)) + (1 - corr(SobelY(ink(G(X))), SobelY(ink(X))))``
     weighted by ``--geom-lock-weight`` (default ``4.0``).
  4. **Dual Modes**:
     - ``--mode train``: trains on ``512x512`` random crops from ``--src-dir``
       and ``--tgt-dir``, supports ``--bf16``, saves ``cut_rism_best.pt`` /
       ``cut_rism_latest.pt`` and side-by-side preview PNGs.
     - ``--mode translate``: streams a JSONL ``--manifest`` of full pages,
       translates each page via full-res conv (``<= --max-side``) or smooth
       Hann-windowed tiled overlap-blend, writes translated PNGs + updated
       ``manifest.jsonl`` with identical ``bekern``/``bekerns`` ground truth.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_DEFAULT_NCE_LAYERS = (1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CutRismConfig:
    ngf: int = 64
    ndf: int = 64
    n_blocks: int = 9
    nce_layers: Tuple[int, ...] = _DEFAULT_NCE_LAYERS
    nce_feat_dim: int = 256
    num_patches: int = 256
    nce_temp: float = 0.07
    gan_weight: float = 1.0
    nce_weight: float = 1.0
    geom_lock_weight: float = 4.0
    sobel_weight: float = 1.0
    ink_bg_kernel: int = 31
    use_spectral_norm: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["nce_layers"] = list(self.nce_layers)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CutRismConfig":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "nce_layers" in valid:
            valid["nce_layers"] = tuple(int(x) for x in valid["nce_layers"])
        return cls(**valid)


# ---------------------------------------------------------------------------
# Utilities: Seeding, Image IO, Discovery
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def discover_images(dirs: Sequence[str | Path]) -> List[Path]:
    """Recursively discover image files across one or more directories."""
    found: List[Path] = []
    for raw in dirs:
        root = Path(raw)
        if not root.is_dir():
            raise FileNotFoundError(f"Image directory not found: {root}")
        # If a hw-sim root contains a dedicated `pages/` folder alongside
        # `stage2_strips/`, prefer `pages/` so full-page style transfer trains
        # on full pages rather than thin single-system strips.
        pages_sub = root / "pages"
        strips_sub = root / "stage2_strips"
        search_root = pages_sub if (pages_sub.is_dir() and strips_sub.is_dir()) else root
        for dirpath, _dirnames, filenames in os.walk(search_root):
            for fn in filenames:
                if Path(fn).suffix.lower() in _IMAGE_EXTS:
                    found.append(Path(dirpath) / fn)
    found.sort()
    return found


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to float32 RGB tensor ``[3, H, W]`` in ``[0, 1]``."""
    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert ``[3, H, W]`` or ``[1, 3, H, W]`` tensor in ``[0, 1]`` to PIL RGB."""
    if t.ndim == 4:
        t = t[0]
    arr = (t.detach().float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def random_crop_tensor(img_t: torch.Tensor, crop_size: int, rng: random.Random) -> torch.Tensor:
    """Extract a ``[3, crop_size, crop_size]`` crop; reflect-pads if image is smaller."""
    _, h, w = img_t.shape
    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)
    if pad_h > 0 or pad_w > 0:
        # Reflect pad requires pad < dim; fall back to replicate if image is tiny.
        mode = "reflect" if (pad_h < h and pad_w < w) else "replicate"
        img_t = F.pad(img_t.unsqueeze(0), (0, pad_w, 0, pad_h), mode=mode).squeeze(0)
        _, h, w = img_t.shape
    top = rng.randint(0, h - crop_size) if h > crop_size else 0
    left = rng.randint(0, w - crop_size) if w > crop_size else 0
    return img_t[:, top : top + crop_size, left : left + crop_size]


def load_random_crop_batch(
    paths: Sequence[Path],
    batch_size: int,
    crop_size: int,
    rng: random.Random,
    hflip: bool = False,
) -> torch.Tensor:
    """Load ``batch_size`` random ``crop_size x crop_size`` RGB tensors in ``[0, 1]``."""
    crops: List[torch.Tensor] = []
    for _ in range(batch_size):
        path = paths[rng.randrange(len(paths))]
        with Image.open(path) as im:
            t = pil_to_tensor(im)
        c = random_crop_tensor(t, crop_size, rng)
        if hflip and rng.random() < 0.5:
            c = torch.flip(c, dims=[2])
        crops.append(c)
    return torch.stack(crops, dim=0)


# ---------------------------------------------------------------------------
# Generator: Lightweight Fully-Convolutional ResNet (~11.4M params)
# ---------------------------------------------------------------------------

class ResnetBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """Fully-convolutional ResNet generator with residual skip in ``[0, 1]`` RGB.

    Downsamples ``2x`` twice (stride 4 bottleneck), applies ``n_blocks``
    ResBlocks (default 9 -> ~11.4M params at ``ngf=64``), then upsamples ``2x``
    twice via bilinear interpolation + 3x3 conv (checkerboard-free) and adds a
    bounded residual delta in ``[0, 1]`` RGB space.
    """

    def __init__(self, ngf: int = 64, n_blocks: int = 9) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
        self.ngf = ngf
        self.n_blocks = n_blocks

        # Encoder layers (indexed 1..5 for PatchNCE; 0 is normalized input)
        self.conv_in = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2, affine=True),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 4, affine=True),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.ModuleList([ResnetBlock(ngf * 4) for _ in range(n_blocks)])
        self._mid_block_idx = max(0, (n_blocks // 2) - 1)

        # Decoder layers (bilinear upsample + reflect-pad conv avoids checkerboard artifacts)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(ngf * 2, affine=True),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small-scale init on final residual conv so step-0 starts near identity.
        final_conv = self.conv_out[1]
        assert isinstance(final_conv, nn.Conv2d)
        nn.init.normal_(final_conv.weight, 0.0, 0.005)
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)

    def layer_channels(self, layers: Sequence[int]) -> List[int]:
        ch_map = {
            0: 3,
            1: self.ngf,
            2: self.ngf * 2,
            3: self.ngf * 4,
            4: self.ngf * 4,
            5: self.ngf * 4,
        }
        return [ch_map[l] for l in layers]

    @staticmethod
    def _pad_to_mult4(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        pad_h = (4 - (h % 4)) % 4
        pad_w = (4 - (w % 4)) % 4
        if pad_h > 0 or pad_w > 0:
            mode = "reflect" if (pad_h < h and pad_w < w) else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        return x, h, w

    def encode_features(
        self,
        x: torch.Tensor,
        layers: Sequence[int] = _DEFAULT_NCE_LAYERS,
    ) -> List[torch.Tensor]:
        """Extract multi-scale encoder features for PatchNCE loss."""
        x_pad, _, _ = self._pad_to_mult4(x)
        x_norm = x_pad * 2.0 - 1.0
        feats_by_id: Dict[int, torch.Tensor] = {}
        layer_set = set(layers)

        if 0 in layer_set:
            feats_by_id[0] = x_norm

        h = self.conv_in(x_norm)
        if 1 in layer_set:
            feats_by_id[1] = h

        h = self.down1(h)
        if 2 in layer_set:
            feats_by_id[2] = h

        h = self.down2(h)
        if 3 in layer_set:
            feats_by_id[3] = h

        for idx, blk in enumerate(self.res_blocks):
            h = blk(h)
            if idx == self._mid_block_idx and 4 in layer_set:
                feats_by_id[4] = h
        if 5 in layer_set:
            feats_by_id[5] = h

        return [feats_by_id[l] for l in layers]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Translate ``x`` in ``[0, 1]`` RGB -> translated RGB in ``[0, 1]`` with exact ``(H, W)``."""
        x_pad, orig_h, orig_w = self._pad_to_mult4(x)
        x_norm = x_pad * 2.0 - 1.0

        h = self.conv_in(x_norm)
        h = self.down1(h)
        h = self.down2(h)
        for blk in self.res_blocks:
            h = blk(h)
        h = self.up1(h)
        h = self.up2(h)
        delta = torch.tanh(self.conv_out(h))
        out = torch.clamp(x_pad + delta, 0.0, 1.0)
        if out.shape[2] != orig_h or out.shape[3] != orig_w:
            out = out[:, :, :orig_h, :orig_w]
        return out


# ---------------------------------------------------------------------------
# PatchGAN Discriminator
# ---------------------------------------------------------------------------

class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator with LSGAN output logits."""

    def __init__(self, ndf: int = 64, use_spectral_norm: bool = True) -> None:
        super().__init__()

        def _conv(
            in_c: int,
            out_c: int,
            k: int = 4,
            s: int = 2,
            p: int = 1,
            bias: bool = True,
        ) -> nn.Module:
            layer: nn.Module = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=bias)
            if use_spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            return layer

        self.net = nn.Sequential(
            _conv(3, ndf, k=4, s=2, p=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            _conv(ndf, ndf * 2, k=4, s=2, p=1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            _conv(ndf * 2, ndf * 4, k=4, s=2, p=1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            _conv(ndf * 4, ndf * 8, k=4, s=1, p=1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            _conv(ndf * 8, 1, k=4, s=1, p=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x * 2.0 - 1.0)


def lsgan_loss(pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return F.mse_loss(pred, target)


# ---------------------------------------------------------------------------
# Multi-Layer PatchNCE (CUT, Park et al. ECCV 2020)
# ---------------------------------------------------------------------------

class PatchSampleMLP(nn.Module):
    """Per-layer 2-layer MLP projection heads with L2 normalization for PatchNCE."""

    def __init__(self, in_channels_list: Sequence[int], feat_dim: int = 256) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(c_in, feat_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(feat_dim, feat_dim),
                )
                for c_in in in_channels_list
            ]
        )

    def forward(
        self,
        feats: Sequence[torch.Tensor],
        num_patches: int = 256,
        patch_ids: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Sample spatial locations from each feature map and project to L2-normalized vectors.

        Returns:
            ``(projected_feats, used_patch_ids)`` where each projected tensor has
            shape ``[B, S, feat_dim]`` and ``S = min(num_patches, H*W)``.
        """
        out_feats: List[torch.Tensor] = []
        used_ids: List[torch.Tensor] = []

        for idx, (feat, mlp) in enumerate(zip(feats, self.mlps)):
            b, c, h, w = feat.shape
            n_loc = h * w
            feat_flat = feat.permute(0, 2, 3, 1).reshape(b, n_loc, c)  # [B, H*W, C]

            if patch_ids is not None:
                ids = patch_ids[idx]
            else:
                s = min(num_patches, n_loc)
                ids = torch.randperm(n_loc, device=feat.device)[:s]

            sampled = feat_flat[:, ids, :]  # [B, S, C]
            proj = mlp(sampled)  # [B, S, D]
            proj = F.normalize(proj, p=2, dim=-1, eps=1e-6)
            out_feats.append(proj)
            used_ids.append(ids)

        return out_feats, used_ids


class PatchNCELoss(nn.Module):
    """InfoNCE contrastive loss over spatial patches within each image."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, feat_q: torch.Tensor, feat_k: torch.Tensor) -> torch.Tensor:
        """Compute PatchNCE between query ``[B, S, D]`` and detached key ``[B, S, D]``."""
        feat_k = feat_k.detach()
        b, s, _ = feat_q.shape
        if s < 2:
            return feat_q.new_zeros(())

        # Positive logits: [B, S, 1]
        l_pos = (feat_q * feat_k).sum(dim=-1, keepdim=True)
        # All pairwise logits within each batch item: [B, S, S]
        l_neg = torch.bmm(feat_q, feat_k.transpose(1, 2))
        diag_mask = torch.eye(s, dtype=torch.bool, device=feat_q.device).unsqueeze(0)
        l_neg = l_neg.masked_fill(diag_mask, -1e4)

        logits = torch.cat([l_pos, l_neg], dim=-1) / self.temperature  # [B, S, 1 + S]
        labels = torch.zeros(b * s, dtype=torch.long, device=feat_q.device)
        return F.cross_entropy(logits.reshape(b * s, -1), labels)


def compute_multilayer_nce(
    gen: ResNetGenerator,
    net_f: PatchSampleMLP,
    nce_loss_fn: PatchNCELoss,
    src_img: torch.Tensor,
    tgt_img: torch.Tensor,
    nce_layers: Sequence[int],
    num_patches: int,
) -> torch.Tensor:
    """Compute mean PatchNCE across ``nce_layers`` between ``src_img`` (key) and ``tgt_img`` (query)."""
    feats_k = gen.encode_features(src_img, layers=nce_layers)
    feats_q = gen.encode_features(tgt_img, layers=nce_layers)

    proj_k, patch_ids = net_f(feats_k, num_patches=num_patches, patch_ids=None)
    proj_q, _ = net_f(feats_q, num_patches=num_patches, patch_ids=patch_ids)

    total = src_img.new_zeros(())
    for fq, fk in zip(proj_q, proj_k):
        total = total + nce_loss_fn(fq, fk)
    return total / max(1, len(proj_q))


# ---------------------------------------------------------------------------
# Hard Spatial-Skeleton / Ink-Centroid Geometry Lock
# ---------------------------------------------------------------------------

def compute_ink_map(img: torch.Tensor, bg_kernel: int = 31) -> torch.Tensor:
    """Compute parchment-invariant high-pass ink map in ``[0, 1]``.

    ``ink(img) = clamp((local_bg - luma) / local_bg, 0, 1)``

    Uses separable 1D average pooling for fast exact 2D local background estimation.
    """
    if img.ndim != 4 or img.shape[1] != 3:
        raise ValueError(f"Expected [B, 3, H, W], got {tuple(img.shape)}")
    luma = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

    k = int(bg_kernel)
    if k % 2 == 0:
        k += 1
    r = k // 2
    _, _, h, w = luma.shape
    # Ensure reflect padding does not exceed spatial dimensions on tiny inputs.
    r_h = min(r, max(0, h - 1))
    r_w = min(r, max(0, w - 1))
    k_h = 2 * r_h + 1
    k_w = 2 * r_w + 1

    bg = luma
    if r_w > 0:
        bg = F.avg_pool2d(F.pad(bg, (r_w, r_w, 0, 0), mode="reflect"), kernel_size=(1, k_w), stride=1)
    if r_h > 0:
        bg = F.avg_pool2d(F.pad(bg, (0, 0, r_h, r_h), mode="reflect"), kernel_size=(k_h, 1), stride=1)

    bg_safe = bg.clamp(min=0.05)
    ink = torch.clamp((bg_safe - luma) / bg_safe, 0.0, 1.0)
    return ink


class GeometryLockLoss(nn.Module):
    """Hard spatial-skeleton lock: ``L1(ink(G(X)), ink(X))`` + Sobel-Y correlation loss.

    Prevents vertical drift of staff lines and noteheads (preserving exact pitch
    and MusicXML/``bekern+`` ground truth).
    """

    def __init__(self, bg_kernel: int = 31, sobel_weight: float = 1.0) -> None:
        super().__init__()
        self.bg_kernel = bg_kernel
        self.sobel_weight = sobel_weight
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def _sobel_y(self, ink: torch.Tensor) -> torch.Tensor:
        kernel = self.sobel_y.to(dtype=ink.dtype, device=ink.device)
        return F.conv2d(F.pad(ink, (1, 1, 1, 1), mode="reflect"), kernel)

    def forward(
        self,
        fake_img: torch.Tensor,
        real_img: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ink_fake = compute_ink_map(fake_img, bg_kernel=self.bg_kernel)
        ink_real = compute_ink_map(real_img.detach(), bg_kernel=self.bg_kernel)

        loss_ink_l1 = F.l1_loss(ink_fake, ink_real)

        gy_fake = self._sobel_y(ink_fake)
        gy_real = self._sobel_y(ink_real)
        real_energy = (gy_real * gy_real).sum(dim=(1, 2, 3))
        fake_energy = (gy_fake * gy_fake).sum(dim=(1, 2, 3))
        num = (gy_fake * gy_real).sum(dim=(1, 2, 3))
        den = torch.sqrt(fake_energy + 1e-8) * torch.sqrt(real_energy + 1e-8)
        corr_y = num / den
        # Ignore Sobel correlation penalty on blank paper-margin crops with no source ink edges.
        has_ink = (real_energy > 1e-5).to(dtype=corr_y.dtype)
        per_sample_sobel = (1.0 - corr_y) * has_ink
        denom_valid = has_ink.sum().clamp(min=1.0)
        loss_sobel_y = per_sample_sobel.sum() / denom_valid
        mean_corr_y = (corr_y * has_ink).sum() / denom_valid if has_ink.sum() > 0 else torch.ones_like(loss_sobel_y)

        total = loss_ink_l1 + self.sobel_weight * loss_sobel_y
        stats = {
            "ink_l1": float(loss_ink_l1.detach().item()),
            "sobel_y_loss": float(loss_sobel_y.detach().item()),
            "sobel_y_corr": float(mean_corr_y.detach().item()),
        }
        return total, stats


@torch.no_grad()
def measure_vertical_ink_drift_px(
    src_img: torch.Tensor,
    out_img: torch.Tensor,
    max_shift: int = 8,
    bg_kernel: int = 31,
) -> float:
    """Estimate vertical ink drift (in pixels) via horizontal ink-profile cross-correlation.

    Returns mean absolute vertical shift across the batch (target: ``<= 1.5 px``).
    """
    ink_a = compute_ink_map(src_img.float(), bg_kernel=bg_kernel).mean(dim=(1, 3))  # [B, H]
    ink_b = compute_ink_map(out_img.float(), bg_kernel=bg_kernel).mean(dim=(1, 3))  # [B, H]
    b, h = ink_a.shape
    shifts = list(range(-max_shift, max_shift + 1))
    drifts: List[float] = []
    for i in range(b):
        pa = ink_a[i] - ink_a[i].mean()
        pb = ink_b[i] - ink_b[i].mean()
        if pa.norm() < 1e-6 or pb.norm() < 1e-6:
            drifts.append(0.0)
            continue
        corrs: List[float] = []
        for s in shifts:
            if s == 0:
                va, vb = pa, pb
            elif s > 0:
                if s >= h:
                    corrs.append(-1.0)
                    continue
                va, vb = pa[:-s], pb[s:]
            else:
                k = -s
                if k >= h:
                    corrs.append(-1.0)
                    continue
                va, vb = pa[k:], pb[:-k]
            c = float((va * vb).sum() / (va.norm() * vb.norm() + 1e-8))
            corrs.append(c)
        best_idx = int(np.argmax(corrs))
        best_shift = float(shifts[best_idx])
        # Parabolic sub-pixel refinement if interior peak
        if 0 < best_idx < len(corrs) - 1:
            y0, y1, y2 = corrs[best_idx - 1], corrs[best_idx], corrs[best_idx + 1]
            denom = (y0 - 2.0 * y1 + y2)
            if abs(denom) > 1e-8:
                best_shift += 0.5 * (y0 - y2) / denom
        drifts.append(abs(best_shift))
    return float(np.mean(drifts)) if drifts else 0.0


# ---------------------------------------------------------------------------
# Training Step, Checkpoints, Previews
# ---------------------------------------------------------------------------

def train_step(
    gen: ResNetGenerator,
    disc: PatchGANDiscriminator,
    net_f: PatchSampleMLP,
    nce_loss_fn: PatchNCELoss,
    geom_loss_fn: GeometryLockLoss,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    real_x: torch.Tensor,
    real_y: torch.Tensor,
    cfg: CutRismConfig,
    use_bf16: bool = False,
) -> Tuple[Dict[str, float], torch.Tensor]:
    """Run one alternating (Discriminator, Generator+PatchNCE+GeomLock) training step."""
    device_type = real_x.device.type
    amp_enabled = bool(use_bf16 and device_type == "cuda")

    # 1) Generate fake_y = G(X) and idt_y = G(Y)
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled):
        fake_y = gen(real_x)
        idt_y = gen(real_y)

    # 2) Update Discriminator D
    opt_d.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled):
        pred_real = disc(real_y)
        pred_fake = disc(fake_y.detach())
        loss_d = 0.5 * (lsgan_loss(pred_real, True) + lsgan_loss(pred_fake, False))
    loss_d.backward()
    opt_d.step()

    # 3) Update Generator G + PatchSampleMLP F
    for p in disc.parameters():
        p.requires_grad_(False)

    opt_g.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled):
        pred_fake_g = disc(fake_y)
        loss_g_gan = lsgan_loss(pred_fake_g, True)

        loss_nce_x = compute_multilayer_nce(
            gen, net_f, nce_loss_fn, real_x, fake_y, cfg.nce_layers, cfg.num_patches
        )
        loss_nce_y = compute_multilayer_nce(
            gen, net_f, nce_loss_fn, real_y, idt_y, cfg.nce_layers, cfg.num_patches
        )
        loss_nce = 0.5 * (loss_nce_x + loss_nce_y)

        loss_geom, geom_stats = geom_loss_fn(fake_y, real_x)

        loss_g_total = (
            cfg.gan_weight * loss_g_gan
            + cfg.nce_weight * loss_nce
            + cfg.geom_lock_weight * loss_geom
        )

    loss_g_total.backward()
    opt_g.step()

    for p in disc.parameters():
        p.requires_grad_(True)

    metrics = {
        "loss_d": float(loss_d.detach().item()),
        "loss_g": float(loss_g_total.detach().item()),
        "loss_g_gan": float(loss_g_gan.detach().item()),
        "loss_nce_x": float(loss_nce_x.detach().item()),
        "loss_nce_y": float(loss_nce_y.detach().item()),
        "loss_geom": float(loss_geom.detach().item()),
        "ink_l1": geom_stats["ink_l1"],
        "sobel_y_corr": geom_stats["sobel_y_corr"],
    }
    return metrics, fake_y.detach()


def save_checkpoint(
    path: str | Path,
    gen: ResNetGenerator,
    disc: PatchGANDiscriminator,
    net_f: PatchSampleMLP,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    cfg: CutRismConfig,
    step: int,
    best_score: float,
) -> Path:
    """Atomically save checkpoint (tmp + rename)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    payload = {
        "step": int(step),
        "best_score": float(best_score),
        "config": cfg.to_dict(),
        "gen": gen.state_dict(),
        "disc": disc.state_dict(),
        "net_f": net_f.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, out_path)
    return out_path


def load_generator_from_checkpoint(
    path: str | Path,
    device: torch.device,
) -> Tuple[ResNetGenerator, CutRismConfig, Dict[str, Any]]:
    """Load ``ResNetGenerator`` and ``CutRismConfig`` from a saved checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = CutRismConfig.from_dict(ckpt["config"])
    gen = ResNetGenerator(ngf=cfg.ngf, n_blocks=cfg.n_blocks).to(device)
    gen.load_state_dict(ckpt["gen"], strict=True)
    gen.eval()
    return gen, cfg, ckpt


def save_preview_grid(
    out_path: str | Path,
    real_x: torch.Tensor,
    fake_y: torch.Tensor,
    real_y: torch.Tensor,
) -> Path:
    """Save side-by-side ``[Source X | Translated G(X) | Target RISM Y]`` preview PNG."""
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    b = min(real_x.shape[0], 4)
    rows: List[Image.Image] = []
    for i in range(b):
        im_x = tensor_to_pil(real_x[i])
        im_fx = tensor_to_pil(fake_y[i])
        im_y = tensor_to_pil(real_y[i])
        w, h = im_x.size
        row = Image.new("RGB", (w * 3, h))
        row.paste(im_x, (0, 0))
        row.paste(im_fx, (w, 0))
        row.paste(im_y, (w * 2, 0))
        rows.append(row)
    grid_w = rows[0].size[0]
    grid_h = sum(r.size[1] for r in rows)
    grid = Image.new("RGB", (grid_w, grid_h))
    y_off = 0
    for r in rows:
        grid.paste(r, (0, y_off))
        y_off += r.size[1]
    tmp_p = out_p.with_suffix(".tmp.png")
    grid.save(tmp_p, format="PNG")
    os.replace(tmp_p, out_p)
    return out_p


def run_train(
    src_dirs: Sequence[str | Path],
    tgt_dirs: Sequence[str | Path],
    out_dir: str | Path,
    cfg: CutRismConfig,
    steps: int = 20000,
    batch_size: int = 2,
    crop_size: int = 512,
    lr: float = 2e-4,
    save_every: int = 1000,
    preview_every: int = 500,
    log_every: int = 50,
    use_bf16: bool = False,
    device_str: str = "auto",
    seed: int = 42,
    resume_path: Optional[str | Path] = None,
) -> Path:
    """Train geometry-locked CUT RISM style transfer model."""
    seed_everything(seed)
    rng = random.Random(seed)
    device = resolve_device(device_str)

    src_paths = discover_images(src_dirs)
    tgt_paths = discover_images(tgt_dirs)
    if not src_paths:
        raise RuntimeError(f"No source images found in {src_dirs}")
    if not tgt_paths:
        raise RuntimeError(f"No target RISM images found in {tgt_dirs}")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    preview_dir = out_root / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "train_log.jsonl"

    gen = ResNetGenerator(ngf=cfg.ngf, n_blocks=cfg.n_blocks).to(device)
    disc = PatchGANDiscriminator(ndf=cfg.ndf, use_spectral_norm=cfg.use_spectral_norm).to(device)
    net_f = PatchSampleMLP(gen.layer_channels(cfg.nce_layers), feat_dim=cfg.nce_feat_dim).to(device)
    nce_loss_fn = PatchNCELoss(temperature=cfg.nce_temp).to(device)
    geom_loss_fn = GeometryLockLoss(bg_kernel=cfg.ink_bg_kernel, sobel_weight=cfg.sobel_weight).to(device)

    opt_g = torch.optim.Adam(
        list(gen.parameters()) + list(net_f.parameters()),
        lr=lr,
        betas=(0.5, 0.999),
    )
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    start_step = 0
    best_score = float("inf")
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        gen.load_state_dict(ckpt["gen"], strict=True)
        disc.load_state_dict(ckpt["disc"], strict=True)
        net_f.load_state_dict(ckpt["net_f"], strict=True)
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_step = int(ckpt.get("step", 0))
        best_score = float(ckpt.get("best_score", float("inf")))

    n_gen_params = sum(p.numel() for p in gen.parameters())
    print(
        f"[cut-rism] device={device} bf16={use_bf16} gen_params={n_gen_params:,} "
        f"src_imgs={len(src_paths)} tgt_imgs={len(tgt_paths)} steps={steps} crop={crop_size}",
        flush=True,
    )

    best_ckpt_path = out_root / "cut_rism_best.pt"
    latest_ckpt_path = out_root / "cut_rism_latest.pt"
    decay_start = max(1, steps // 2)
    t0 = time.time()

    for step in range(start_step + 1, steps + 1):
        # Linear LR decay in second half of training
        if step > decay_start:
            frac = max(0.0, 1.0 - (step - decay_start) / max(1, steps - decay_start))
            cur_lr = lr * frac
            for pg in opt_g.param_groups:
                pg["lr"] = cur_lr
            for pg in opt_d.param_groups:
                pg["lr"] = cur_lr

        real_x = load_random_crop_batch(src_paths, batch_size, crop_size, rng, hflip=False).to(device)
        real_y = load_random_crop_batch(tgt_paths, batch_size, crop_size, rng, hflip=True).to(device)

        metrics, fake_y = train_step(
            gen=gen,
            disc=disc,
            net_f=net_f,
            nce_loss_fn=nce_loss_fn,
            geom_loss_fn=geom_loss_fn,
            opt_g=opt_g,
            opt_d=opt_d,
            real_x=real_x,
            real_y=real_y,
            cfg=cfg,
            use_bf16=use_bf16,
        )

        # Selection score balances geometry preservation + PatchNCE fidelity
        score = metrics["loss_geom"] + 0.5 * metrics["loss_nce_x"]
        improved = score < best_score
        if improved:
            best_score = score

        if step == 1 or step % log_every == 0 or step == steps:
            drift_px = measure_vertical_ink_drift_px(real_x, fake_y, bg_kernel=cfg.ink_bg_kernel)
            rec = {"step": step, "elapsed_s": round(time.time() - t0, 2), "drift_px": round(drift_px, 4), **metrics}
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(rec) + "\n")
            print(
                f"[step {step:6d}/{steps}] d={metrics['loss_d']:.3f} g_gan={metrics['loss_g_gan']:.3f} "
                f"nce_x={metrics['loss_nce_x']:.3f} nce_y={metrics['loss_nce_y']:.3f} "
                f"geom={metrics['loss_geom']:.4f} (ink={metrics['ink_l1']:.4f}, corr_y={metrics['sobel_y_corr']:.4f}) "
                f"drift={drift_px:.2f}px",
                flush=True,
            )

        if preview_every > 0 and (step % preview_every == 0 or step == steps):
            save_preview_grid(preview_dir / f"step_{step:06d}.png", real_x, fake_y, real_y)

        if (save_every > 0 and step % save_every == 0) or step == steps:
            save_checkpoint(latest_ckpt_path, gen, disc, net_f, opt_g, opt_d, cfg, step, best_score)
            if improved or not best_ckpt_path.exists():
                save_checkpoint(best_ckpt_path, gen, disc, net_f, opt_g, opt_d, cfg, step, best_score)
        elif improved and step >= min(100, steps):
            save_checkpoint(best_ckpt_path, gen, disc, net_f, opt_g, opt_d, cfg, step, best_score)

    return best_ckpt_path


# ---------------------------------------------------------------------------
# Full-Page Inference / Translation (Full-Res Conv + Tiled Overlap-Blend)
# ---------------------------------------------------------------------------

def _hann_weight_2d(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    wy = torch.hann_window(h + 2, periodic=False, device=device, dtype=dtype)[1:-1].clamp(min=1e-3)
    wx = torch.hann_window(w + 2, periodic=False, device=device, dtype=dtype)[1:-1].clamp(min=1e-3)
    return (wy[:, None] * wx[None, :]).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


@torch.no_grad()
def translate_tensor_fullpage(
    gen: ResNetGenerator,
    img_chw: torch.Tensor,
    max_side: int = 1536,
    tile_size: int = 768,
    tile_overlap: int = 96,
    use_bf16: bool = False,
) -> torch.Tensor:
    """Translate an arbitrary-size ``[3, H, W]`` page tensor, preserving exact ``(H, W)``.

    Uses direct full-resolution conv when ``max(H, W) <= max_side``; otherwise
    runs overlapping tiles blended with a 2D Hann window so tile boundaries are
    invisible and VRAM stays bounded.
    """
    gen.eval()
    device = next(gen.parameters()).device
    device_type = device.type
    amp_enabled = bool(use_bf16 and device_type == "cuda")

    _, h, w = img_chw.shape
    x = img_chw.unsqueeze(0).to(device)  # [1, 3, H, W]

    if max(h, w) <= max_side:
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled):
            out = gen(x)
        return out[0].float().clamp(0.0, 1.0).cpu()

    # Tiled overlap-blend inference
    step = max(32, tile_size - tile_overlap)
    ys = list(range(0, max(1, h - tile_size + 1), step))
    if not ys or ys[-1] + tile_size < h:
        ys.append(max(0, h - tile_size))
    xs = list(range(0, max(1, w - tile_size + 1), step))
    if not xs or xs[-1] + tile_size < w:
        xs.append(max(0, w - tile_size))

    acc = torch.zeros((1, 3, h, w), device=device, dtype=torch.float32)
    w_acc = torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)

    for y0 in ys:
        y1 = min(h, y0 + tile_size)
        for x0 in xs:
            x1 = min(w, x0 + tile_size)
            tile = x[:, :, y0:y1, x0:x1]
            with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp_enabled):
                pred = gen(tile).float()
            tw = _hann_weight_2d(y1 - y0, x1 - x0, device=device, dtype=torch.float32)
            acc[:, :, y0:y1, x0:x1] += pred * tw
            w_acc[:, :, y0:y1, x0:x1] += tw

    out = (acc / w_acc.clamp(min=1e-6)).clamp(0.0, 1.0)
    return out[0].cpu()


def translate_manifest(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    out_dir: str | Path,
    max_side: int = 1536,
    tile_size: int = 768,
    tile_overlap: int = 96,
    use_bf16: bool = False,
    device_str: str = "auto",
    limit: Optional[int] = None,
    resume: bool = False,
) -> Path:
    """Stream a JSONL manifest, translate every full page, and write updated ``manifest.jsonl``.

    Preserves all ground-truth keys (``bekerns``, ``bekern``, ``domain``, etc.)
    identically while updating ``"image"`` to the translated PNG path.
    """
    device = resolve_device(device_str)
    gen, cfg, _ = load_generator_from_checkpoint(checkpoint_path, device=device)

    out_root = Path(out_dir)
    pages_dir = out_root / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = out_root / "manifest.jsonl"

    records: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as mf:
        for line in mf:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break

    out_records: List[Dict[str, Any]] = []
    drifts: List[float] = []
    t0 = time.time()

    for idx, rec in enumerate(records):
        src_img_path = Path(rec["image"])
        out_name = f"{src_img_path.stem}.png"
        out_img_path = pages_dir / out_name
        # Avoid collisions if two records share a stem
        if out_img_path.exists() and not resume and any(r["image"] == str(out_img_path) for r in out_records):
            out_img_path = pages_dir / f"{src_img_path.stem}_{idx:05d}.png"

        if not (resume and out_img_path.exists()):
            with Image.open(src_img_path) as im:
                src_t = pil_to_tensor(im)
            out_t = translate_tensor_fullpage(
                gen=gen,
                img_chw=src_t,
                max_side=max_side,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                use_bf16=use_bf16,
            )
            drifts.append(
                measure_vertical_ink_drift_px(
                    src_t.unsqueeze(0), out_t.unsqueeze(0), bg_kernel=cfg.ink_bg_kernel
                )
            )
            tmp_img = out_img_path.with_suffix(".tmp.png")
            tensor_to_pil(out_t).save(tmp_img, format="PNG")
            os.replace(tmp_img, out_img_path)

        new_rec = dict(rec)
        new_rec["source_image"] = str(src_img_path)
        new_rec["image"] = str(out_img_path.resolve())
        new_rec["style_transfer"] = "cut_rism"
        out_records.append(new_rec)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(records):
            mean_drift = float(np.mean(drifts)) if drifts else 0.0
            print(
                f"[translate {idx + 1}/{len(records)}] elapsed={time.time() - t0:.1f}s "
                f"mean_vertical_drift={mean_drift:.3f}px",
                flush=True,
            )

    tmp_manifest = out_manifest.with_suffix(".jsonl.tmp")
    with open(tmp_manifest, "w", encoding="utf-8") as wf:
        for r in out_records:
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_manifest, out_manifest)
    return out_manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Geometry-Locked CUT RISM Style Transfer Pipeline (train & translate)"
    )
    p.add_argument("--mode", choices=("train", "translate"), required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    p.add_argument("--bf16", action="store_true", help="Enable CUDA bfloat16 autocast")
    p.add_argument("--seed", type=int, default=42)

    # Train arguments
    p.add_argument(
        "--src-dir",
        type=str,
        action="append",
        default=None,
        help="Source hw-score-sim image directory (repeatable or comma-separated)",
    )
    p.add_argument(
        "--tgt-dir",
        type=str,
        action="append",
        default=None,
        help="Target RISM archival image directory (repeatable or comma-separated)",
    )
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--preview-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--n-blocks", type=int, default=9)
    p.add_argument("--nce-feat-dim", type=int, default=256)
    p.add_argument("--num-patches", type=int, default=256)
    p.add_argument("--gan-weight", type=float, default=1.0)
    p.add_argument("--nce-weight", type=float, default=1.0)
    p.add_argument("--geom-lock-weight", type=float, default=4.0)
    p.add_argument("--sobel-weight", type=float, default=1.0)
    p.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume training")

    # Translate arguments
    p.add_argument("--checkpoint", type=str, default=None, help="Checkpoint for --mode translate")
    p.add_argument("--manifest", type=str, default=None, help="Input JSONL manifest for --mode translate")
    p.add_argument("--max-side", type=int, default=1536, help="Direct full-res conv threshold; above uses tiled blend")
    p.add_argument("--tile-size", type=int, default=768)
    p.add_argument("--tile-overlap", type=int, default=96)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume-translate", action="store_true", help="Skip already-translated PNGs")
    return p


def _expand_dirs(raw_list: Optional[List[str]]) -> List[str]:
    if not raw_list:
        return []
    out: List[str] = []
    for item in raw_list:
        for part in item.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.mode == "train":
        src_dirs = _expand_dirs(args.src_dir)
        tgt_dirs = _expand_dirs(args.tgt_dir)
        if not src_dirs or not tgt_dirs:
            raise SystemExit("--mode train requires both --src-dir and --tgt-dir")
        cfg = CutRismConfig(
            ngf=args.ngf,
            ndf=args.ndf,
            n_blocks=args.n_blocks,
            nce_feat_dim=args.nce_feat_dim,
            num_patches=args.num_patches,
            gan_weight=args.gan_weight,
            nce_weight=args.nce_weight,
            geom_lock_weight=args.geom_lock_weight,
            sobel_weight=args.sobel_weight,
        )
        run_train(
            src_dirs=src_dirs,
            tgt_dirs=tgt_dirs,
            out_dir=args.out_dir,
            cfg=cfg,
            steps=args.steps,
            batch_size=args.batch_size,
            crop_size=args.crop_size,
            lr=args.lr,
            save_every=args.save_every,
            preview_every=args.preview_every,
            log_every=args.log_every,
            use_bf16=args.bf16,
            device_str=args.device,
            seed=args.seed,
            resume_path=args.resume,
        )
        return 0

    if not args.checkpoint or not args.manifest:
        raise SystemExit("--mode translate requires both --checkpoint and --manifest")
    translate_manifest(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        max_side=args.max_side,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        use_bf16=args.bf16,
        device_str=args.device,
        limit=args.limit,
        resume=args.resume_translate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
