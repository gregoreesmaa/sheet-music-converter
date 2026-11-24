"""Vetted academic losses and domain-generalization modules for Stage-1 StaffUNet2D.

Includes:
  1. Skeleton Recall Loss (SRL, ECCV 2024 — Kirchhoff et al.) for thin 1-2px
     staff lines (ch0) and staff centerlines (ch1). Zero inference/ONNX cost.
  2. Scale-Invariant Log Loss (SILog, CVPR 2024 UniDepth) for ch3 (log_hm)
     to prevent intra-page relative scale drift across dynamic resolutions.
  3. Boundary Difference-over-Union Loss (B-DoU, MICCAI 2023/2024) for ch2
     (system_body) to sharpen inter-system valleys.
  4. Unlabeled-HW Fourier Amplitude Mixup (FACT / RASS / FADA NeurIPS 2024)
     transferring low-frequency manuscript texture/sepia/contrast from real
     unlabeled HW pages while preserving exact phase-aligned GT staff geometry.
  5. Early-Encoder EFDMix (CVPR 2022) exact empirical CDF feature matching
     preserving spatial rank order (identity at eval() -> zero inference cost).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def skeleton_recall_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """ECCV 2024 Skeleton Recall Loss on analytical thin staff targets.

    Args:
        logits: [B, H, W] raw logits for ch0 (staff lines) or ch1 (centerline).
        target: [B, H, W] float target heatmap in [0, 1].
    """
    probs = torch.sigmoid(logits.float())
    tgt = target.float()
    skel = (tgt > 0.5).float()
    tubed = (
        F.max_pool2d(skel.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        * (tgt > 0.1).float()
    )
    inter = (probs * tubed).sum(dim=(1, 2))
    denom = tubed.sum(dim=(1, 2))
    valid = denom > 0.0
    if not valid.any():
        return logits.new_zeros(())
    return (1.0 - (inter[valid] + eps) / (denom[valid] + eps)).mean()


def silog_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    lam: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """UniDepth (CVPR 2024) Scale-Invariant Log Loss for ch3 (log_hm).

    Args:
        pred:   [B, H, W] predicted log staff-space map.
        target: [B, H, W] ground-truth log staff-space map.
        mask:   [B, H, W] positive staff-support mask in {0, 1}.
        lam:    Scale-invariance weight in [0, 1); <= 0.5 anchors global scale.
    """
    m = (mask > 0.5).float()
    n = m.sum(dim=(1, 2))
    valid = n > 0.0
    if not valid.any():
        return pred.new_zeros(())
    d = (pred.float() - target.float()) * m
    nv = n[valid].clamp_min(1.0)
    mean_sq = (d[valid] * d[valid]).sum(dim=(1, 2)) / nv
    sq_mean = (d[valid].sum(dim=(1, 2)) / nv) ** 2
    return torch.sqrt((mean_sq - float(lam) * sq_mean).clamp_min(eps)).mean()


def boundary_dou_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    band_kernel: int = 5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Boundary Difference-over-Union Loss (MICCAI 2023/2024) on ch2 system boundaries.

    Args:
        logits: [B, H, W] raw logits for ch2 (system_body).
        target: [B, H, W] float target mask in [0, 1].
    """
    p = torch.sigmoid(logits.float())
    tgt = target.float()
    g = (tgt > 0.5).float().unsqueeze(1)
    pad = band_kernel // 2
    dil = F.max_pool2d(g, kernel_size=band_kernel, stride=1, padding=pad)
    ero = -F.max_pool2d(-g, kernel_size=band_kernel, stride=1, padding=pad)
    band = (dil - ero).squeeze(1)
    pb = p * band
    gb = tgt * band
    inter = (pb * gb).sum(dim=(1, 2))
    union = pb.sum(dim=(1, 2)) + gb.sum(dim=(1, 2)) - inter
    valid = union > 0.0
    if not valid.any():
        return logits.new_zeros(())
    return (1.0 - (inter[valid] + eps) / (union[valid] + eps)).mean()


@torch.no_grad()
def fourier_hw_amplitude_mix(
    x_lbl: torch.Tensor,
    x_hw: torch.Tensor,
    p: float = 0.4,
    beta: float = 0.15,
    alpha: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Transfer low-frequency amplitude spectrum from unlabeled HW pages to labeled pages.

    Preserves 100% of labeled phase spectrum so staff-line GT masks stay pixel-aligned.
    """
    if p <= 0.0 or alpha <= 0.0 or beta <= 0.0:
        return x_lbl
    if float(torch.rand(1, generator=generator).item()) > float(p):
        return x_lbl
    if x_hw.device != x_lbl.device:
        x_hw = x_hw.to(device=x_lbl.device)
    if x_hw.shape[0] != x_lbl.shape[0]:
        idx = torch.randint(0, x_hw.shape[0], (x_lbl.shape[0],), generator=generator).to(x_lbl.device)
        x_hw = x_hw.index_select(0, idx)
    if x_hw.shape[-2:] != x_lbl.shape[-2:]:
        x_hw = F.interpolate(x_hw.float(), size=x_lbl.shape[-2:], mode="bilinear", align_corners=False)
    f_l = torch.fft.rfft2(x_lbl.float(), dim=(-2, -1))
    f_h = torch.fft.rfft2(x_hw.float(), dim=(-2, -1))
    amp_l = torch.abs(f_l)
    ph_l = torch.angle(f_l)
    amp_h = torch.abs(f_h)
    H, W = f_l.shape[-2], f_l.shape[-1]
    bh = max(1, int(round(H * float(beta))))
    bw = max(1, int(round(W * float(beta))))
    lam = torch.empty((x_lbl.size(0), 1, 1, 1), dtype=torch.float32).uniform_(
        0.0, float(alpha), generator=generator
    ).to(x_lbl.device)
    amp_mix = amp_l.clone()
    amp_mix[:, :, :bh, :bw] = (1.0 - lam) * amp_l[:, :, :bh, :bw] + lam * amp_h[:, :, :bh, :bw]
    amp_mix[:, :, -bh:, :bw] = (1.0 - lam) * amp_l[:, :, -bh:, :bw] + lam * amp_h[:, :, -bh:, :bw]
    out = torch.fft.irfft2(amp_mix * torch.exp(1j * ph_l), s=x_lbl.shape[-2:], dim=(-2, -1))
    return out.clamp(0.0, 1.0).to(dtype=x_lbl.dtype)


class EFDMix(nn.Module):
    """Exact Feature Distribution Matching (CVPR 2022) for early-encoder DG.

    Strict identity during eval() -> zero inference/ONNX overhead.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1) -> None:
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or x.size(0) < 2 or self.p <= 0.0:
            return x
        if float(torch.rand(1).item()) > self.p:
            return x
        B, C, H, W = x.shape
        xf = x.float().reshape(B, C, -1)
        perm = torch.randperm(B, device=x.device)
        _, idx_x = torch.sort(xf, dim=-1)
        val_y, _ = torch.sort(xf[perm], dim=-1)
        inv_idx = idx_x.argsort(dim=-1)
        matched = val_y.gather(-1, inv_idx)
        beta_dist = torch.distributions.Beta(self.alpha, self.alpha)
        lam = beta_dist.sample((B, 1, 1)).to(device=x.device, dtype=torch.float32)
        out = xf + lam * (matched - xf.detach())
        return out.reshape(B, C, H, W).to(dtype=x.dtype)
