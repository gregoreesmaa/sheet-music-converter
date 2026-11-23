"""Lightweight 2D Staff+System UNet and curved-system extractor.

Replaces 1D RowNet (which fails on tilted/curved/bordered phone photos) with a
compact 2D UNet (~1.35M params, <15 ms GPU / <120 ms CPU at 1024x768 RGB,
ONNX-exportable) predicting 4 dense channels at 1/2 page resolution (512x384
for 1024x768 input, preventing 5-line fusion on dense 6-staff pages):

  ch0: 5-line staff-line mask (BCE + Dice)
  ch1: staff-centerline mask (BCE + Dice; for tracing curved staves)
  ch2: system-body mask (BCE + Dice; groups staves into systems under tilt/curve)
  ch3: local normal staff-space scale map ``sp_normal_px / H`` (SmoothL1 on staff pixels)

Also provides:
  - ``StaffUNetLoss``: combined multi-task loss.
  - ``CurvedStaff`` / ``CurvedSystem``: cubic-Bezier staff/system geometry.
  - ``extract_curved_systems``: connected-component centerline tracing,
    same-row horizontal fragment merging, relative-width coda survival,
    least-squares cubic Bezier fitting, and valley-guarded ch2 system grouping.
  - ``rasterize_staff_targets``: target generator from symbox /
    ``hybrid_staff.jsonl`` / explicit staff tables with normal-sp consistency.
  - ``resize_page_image`` / ``preprocess_page_image``: anti-aliased BOX/LANCZOS
    resizing helpers so thin 1-2 px staff lines do not alias/wash out.
  - ``StaffUNetDataset`` & ``train_staff_unet2d``: training pipeline with full
    seeded on-the-fly geometric + photometric augmentations and CLI entrypoint.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "StaffUNet2D",
    "StaffUNetLoss",
    "CurvedStaff",
    "CurvedSystem",
    "fit_cubic_bezier_1d",
    "eval_cubic_bezier_1d",
    "extract_curved_systems",
    "rasterize_staff_targets",
    "resize_page_image",
    "preprocess_page_image",
    "StaffUNetDataset",
    "train_staff_unet2d",
]


# ---------------------------------------------------------------------------
# Anti-aliased page image resizing & inference preprocessing
# ---------------------------------------------------------------------------


def resize_page_image(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize a page image using BOX (downscaling) or LANCZOS (upscaling).

    Avoids bilinear aliasing/washout of thin 1-2 px staff lines.
    """
    orig_w, orig_h = img.size
    if orig_w == target_w and orig_h == target_h:
        return img
    if orig_w > target_w or orig_h > target_h:
        resample = Image.Resampling.BOX
    else:
        resample = Image.Resampling.LANCZOS
    return img.resize((int(target_w), int(target_h)), resample)


def preprocess_page_image(
    img: Image.Image,
    target_h: int = 1024,
    target_w: int = 768,
    in_channels: int = 3,
) -> torch.Tensor:
    """Convert and anti-alias-resize a PIL page image to ``[C, target_h, target_w]`` in ``[0, 1]``."""
    mode = "L" if int(in_channels) == 1 else "RGB"
    work = img if img.mode == mode else img.convert(mode)
    work = resize_page_image(work, target_w=target_w, target_h=target_h)
    arr = np.asarray(work, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        return torch.from_numpy(arr[None, :, :])
    return torch.from_numpy(arr.transpose(2, 0, 1))


# ---------------------------------------------------------------------------
# Model: Lightweight 2D Staff+System UNet (~1.35M params, 1/2 res output)
# ---------------------------------------------------------------------------


class _ConvBlock(nn.Module):
    """Two 3x3 Conv-BN-ReLU layers (ONNX-clean, no exotic ops)."""

    def __init__(self, in_ch: int, out_ch: int, dilation: tuple[int, int] = (1, 1)):
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=pad,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StaffUNet2D(nn.Module):
    """Compact 2D UNet predicting 4 dense staff/system maps at 1/2 (or 1/4) resolution.

    Args:
        in_channels: Input image channels (default 3 for RGB; 1 for grayscale).
        base_channels: Channel width at 1/4 resolution (default 40 -> ~1.35M params).
        out_stride: Output spatial stride (2 for 1/2-res skip-up head [default,
            512x384 out for 1024x768 in], or 4 for 1/4-res output).
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 40, out_stride: int = 2):
        super().__init__()
        if out_stride not in (2, 4):
            raise ValueError(f"out_stride must be 2 or 4, got {out_stride}")
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.out_stride = int(out_stride)

        c1 = base_channels             # 40  @ 1/4
        c2 = int(base_channels * 1.8)  # 72  @ 1/8
        c3 = int(base_channels * 2.8)  # 112 @ 1/16
        c4 = int(base_channels * 4.2)  # 168 @ 1/32

        # Split stride-4 stem into stride-2 stages so s2 (1/2 res) feeds the skip-up head.
        stem_mid = max(16, c1 // 2)
        self.stem1 = nn.Sequential(
            nn.Conv2d(in_channels, stem_mid, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_mid),
            nn.ReLU(inplace=True),
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(stem_mid, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.enc1 = _ConvBlock(c1, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _ConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(2)

        # Horizontal dilation in bottleneck widens receptive field along staves.
        self.bottleneck = _ConvBlock(c3, c4, dilation=(1, 2))

        self.dec3 = _ConvBlock(c4 + c3, c3)
        self.dec2 = _ConvBlock(c3 + c2, c2)
        self.dec1 = _ConvBlock(c2 + c1, c1)

        if self.out_stride == 2:
            # Lightweight 1/2-res skip-up fusion: preserves fine 5-line separation on 6-staff pages.
            self.dec0 = nn.Sequential(
                nn.Conv2d(c1 + stem_mid, stem_mid, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(stem_mid),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Conv2d(stem_mid, 4, kernel_size=1)
        else:
            self.dec0 = None
            self.head = nn.Conv2d(c1, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``[B, C, H, W]`` float tensor.

        Returns:
            ``[B, 4, ceil(H/out_stride), ceil(W/out_stride)]`` raw outputs:
              - ``[:, 0:3]``: logits for (staff_lines, staff_center, system_body)
              - ``[:, 3:4]``: raw scale pre-activation (use ``F.softplus`` for ``sp_normal/H``)
        """
        s2 = self.stem1(x)
        s4 = self.enc1(self.stem2(s2))
        s8 = self.enc2(self.pool1(s4))
        s16 = self.enc3(self.pool2(s8))
        b32 = self.bottleneck(self.pool3(s16))

        u16 = F.interpolate(b32, size=s16.shape[-2:], mode="bilinear", align_corners=False)
        d16 = self.dec3(torch.cat([u16, s16], dim=1))

        u8 = F.interpolate(d16, size=s8.shape[-2:], mode="bilinear", align_corners=False)
        d8 = self.dec2(torch.cat([u8, s8], dim=1))

        u4 = F.interpolate(d8, size=s4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.dec1(torch.cat([u4, s4], dim=1))

        if self.dec0 is not None:
            u2 = F.interpolate(d4, size=s2.shape[-2:], mode="bilinear", align_corners=False)
            d2 = self.dec0(torch.cat([u2, s2], dim=1))
            return self.head(d2)
        return self.head(d4)

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Inference helper returning ``[B, 4, H/out_stride, W/out_stride]`` probs + ``sp/H``."""
        raw = self.forward(x)
        probs = torch.sigmoid(raw[:, :3])
        sp_map = F.softplus(raw[:, 3:4])
        return torch.cat([probs, sp_map], dim=1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Loss: BCE + Dice on ch0..2, masked SmoothL1 on ch3
# ---------------------------------------------------------------------------


def _soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    dice = (2.0 * inter + eps) / (denom + eps)
    return (1.0 - dice).mean()


class StaffUNetLoss(nn.Module):
    """Multi-task loss for ``StaffUNet2D``.

    - ``ch0`` (staff lines): BCE + Dice
    - ``ch1`` (staff center): BCE + Dice
    - ``ch2`` (system body): BCE + Dice
    - ``ch3`` (``sp_px / H``): SmoothL1 masked on staff pixels
    """

    def __init__(
        self,
        w_lines: float = 1.0,
        w_center: float = 1.0,
        w_system: float = 1.0,
        w_sp: float = 10.0,
        smooth_l1_beta: float = 0.005,
    ):
        super().__init__()
        self.w_lines = float(w_lines)
        self.w_center = float(w_center)
        self.w_system = float(w_system)
        self.w_sp = float(w_sp)
        self.smooth_l1_beta = float(smooth_l1_beta)

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total loss and per-head scalar metrics.

        Args:
            preds: ``[B, 4, Hp, Wp]`` raw model output.
            targets: ``[B, 4, Hp, Wp]`` float targets in ``[0, 1]`` for ch0..2
                and ``sp_px / H`` for ch3.
        """
        if preds.shape != targets.shape:
            targets = F.interpolate(
                targets, size=preds.shape[-2:], mode="bilinear", align_corners=False
            )

        l_lines = F.binary_cross_entropy_with_logits(preds[:, 0], targets[:, 0]) + _soft_dice_loss(
            preds[:, 0], targets[:, 0]
        )
        l_center = F.binary_cross_entropy_with_logits(preds[:, 1], targets[:, 1]) + _soft_dice_loss(
            preds[:, 1], targets[:, 1]
        )
        l_system = F.binary_cross_entropy_with_logits(preds[:, 2], targets[:, 2]) + _soft_dice_loss(
            preds[:, 2], targets[:, 2]
        )

        # Mask sp loss to pixels on/near staves (ch0 or ch1 active, or target sp > 0).
        sp_pred = F.softplus(preds[:, 3])
        sp_tgt = targets[:, 3]
        mask = ((targets[:, 0] > 0.25) | (targets[:, 1] > 0.25) | (sp_tgt > 1e-6)).float()
        denom = mask.sum()
        if float(denom.item()) > 0.0:
            elem = F.smooth_l1_loss(sp_pred, sp_tgt, beta=self.smooth_l1_beta, reduction="none")
            l_sp = (elem * mask).sum() / denom
        else:
            l_sp = sp_pred.sum() * 0.0

        total = (
            self.w_lines * l_lines
            + self.w_center * l_center
            + self.w_system * l_system
            + self.w_sp * l_sp
        )
        metrics = {
            "loss_total": float(total.detach().item()),
            "loss_lines": float(l_lines.detach().item()),
            "loss_center": float(l_center.detach().item()),
            "loss_system": float(l_system.detach().item()),
            "loss_sp": float(l_sp.detach().item()),
        }
        return total, metrics


# ---------------------------------------------------------------------------
# Cubic Bezier helpers & CurvedStaff / CurvedSystem data structures
# ---------------------------------------------------------------------------


def _bernstein3(t: np.ndarray) -> np.ndarray:
    """Cubic Bernstein basis matrix of shape ``(len(t), 4)``."""
    t = np.asarray(t, dtype=np.float64)
    u = 1.0 - t
    return np.stack([u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t], axis=-1)


def fit_cubic_bezier_1d(t: np.ndarray, values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted least-squares fit of 4 cubic Bezier control values ``(4,)`` for ``t in [0, 1]``."""
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if t.size == 0:
        raise ValueError("cannot fit cubic Bezier to empty samples")
    if t.size < 4:
        med = float(np.median(v))
        return np.full(4, med, dtype=np.float64)
    B = _bernstein3(t)
    if weights is not None:
        w = np.sqrt(np.maximum(np.asarray(weights, dtype=np.float64), 1e-6))
        Bw = B * w[:, None]
        vw = v * w
    else:
        Bw, vw = B, v
    # Tiny ridge prevents singular matrix if t spans a degenerate interval.
    gram = Bw.T @ Bw + 1e-6 * np.eye(4, dtype=np.float64)
    rhs = Bw.T @ vw
    return np.linalg.solve(gram, rhs)


def eval_cubic_bezier_1d(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate 1D cubic Bezier control values ``(4,)`` at ``t in [0, 1]``."""
    return _bernstein3(t) @ np.asarray(ctrl, dtype=np.float64)


@dataclass
class CurvedStaff:
    """One curved staff parameterized by a cubic Bezier centerline + local ``sp(t)``."""

    ctrl_pts: np.ndarray      # (4, 2) float64 [(x0,y0), ..., (x3,y3)] in page px
    sp_ctrl: np.ndarray       # (4,) float64 local staff-space Bezier control values in px
    x_start: float
    x_end: float
    sp_median: float
    confidence: float = 1.0

    def eval_center(self, t: np.ndarray) -> np.ndarray:
        """Evaluate centerline ``(N, 2)`` ``(x, y)`` at ``t in [0, 1]``."""
        return _bernstein3(t) @ self.ctrl_pts

    def eval_y_at_x(self, x: np.ndarray) -> np.ndarray:
        """Evaluate ``y_center(x)`` (exact when ``ctrl_pts[:, 0]`` is uniform in ``x``)."""
        x = np.asarray(x, dtype=np.float64)
        span = max(self.x_end - self.x_start, 1e-6)
        t = np.clip((x - self.x_start) / span, 0.0, 1.0)
        return eval_cubic_bezier_1d(self.ctrl_pts[:, 1], t)

    def eval_sp_at_x(self, x: np.ndarray) -> np.ndarray:
        """Evaluate local staff space ``sp(x)`` in page pixels."""
        x = np.asarray(x, dtype=np.float64)
        span = max(self.x_end - self.x_start, 1e-6)
        t = np.clip((x - self.x_start) / span, 0.0, 1.0)
        return np.maximum(eval_cubic_bezier_1d(self.sp_ctrl, t), 1.0)

    def eval_lines_at_x(self, x: np.ndarray) -> np.ndarray:
        """Evaluate all 5 staff lines ``(5, len(x))`` at page columns ``x``."""
        yc = self.eval_y_at_x(x)
        sp = self.eval_sp_at_x(x)
        offsets = np.arange(-2, 3, dtype=np.float64)[:, None]
        return yc[None, :] + offsets * sp[None, :]

    def mean_y(self) -> float:
        return float(np.mean(self.eval_center(np.linspace(0.0, 1.0, 17))[:, 1]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ctrl_pts": np.round(self.ctrl_pts, 3).tolist(),
            "sp_ctrl": np.round(self.sp_ctrl, 3).tolist(),
            "x_start": round(float(self.x_start), 2),
            "x_end": round(float(self.x_end), 2),
            "sp_median": round(float(self.sp_median), 3),
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class CurvedSystem:
    """One musical system composed of 1..N curved staves + bounding polygon."""

    staves: list[CurvedStaff]
    x_start: float
    x_end: float
    polygon: np.ndarray       # (M, 2) closed curved bounding polygon in page px
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "staves": [s.to_dict() for s in self.staves],
            "x_start": round(float(self.x_start), 2),
            "x_end": round(float(self.x_end), 2),
            "polygon": np.round(self.polygon, 2).tolist(),
            "confidence": round(float(self.confidence), 4),
        }


def _build_system_polygon(staves: list[CurvedStaff], n_samples: int = 24, pad_sp: float = 2.5) -> np.ndarray:
    """Build a closed curved polygon enclosing top..bottom staves of a system."""
    x0 = min(s.x_start for s in staves)
    x1 = max(s.x_end for s in staves)
    xs = np.linspace(x0, x1, num=n_samples)
    top_s = staves[0]
    bot_s = staves[-1]
    y_top = top_s.eval_y_at_x(xs) - (2.0 + pad_sp) * top_s.eval_sp_at_x(xs)
    y_bot = bot_s.eval_y_at_x(xs) + (2.0 + pad_sp) * bot_s.eval_sp_at_x(xs)
    top_pts = np.stack([xs, y_top], axis=-1)
    bot_pts = np.stack([xs[::-1], y_bot[::-1]], axis=-1)
    return np.concatenate([top_pts, bot_pts], axis=0)


# ---------------------------------------------------------------------------
# Post-processor: extract_curved_systems
# ---------------------------------------------------------------------------


def _extrapolate_fragment_y(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    x_target: float,
    side: str,
    max_slope: float = 0.5,
) -> float:
    """Linearly extrapolate a fragment's left or right edge to ``x_target``."""
    n = x_arr.size
    if n < 2:
        return float(y_arr[0])
    k = min(n, max(4, n // 3))
    if side == "right":
        xs = x_arr[-k:]
        ys = y_arr[-k:]
        x_ref = float(xs[-1])
        y_ref = float(ys[-1])
    else:
        xs = x_arr[:k]
        ys = y_arr[:k]
        x_ref = float(xs[0])
        y_ref = float(ys[0])
    dx = xs - xs.mean()
    denom = float((dx * dx).sum())
    if denom < 1e-6:
        slope = 0.0
    else:
        slope = float((dx * (ys - ys.mean())).sum() / denom)
        slope = float(np.clip(slope, -max_slope, max_slope))
    return y_ref + slope * (x_target - x_ref)


def _merge_horizontal_fragments(
    frags: list[dict[str, Any]],
    sp_tol_factor: float = 0.5,
) -> list[dict[str, Any]]:
    """Merge disjoint horizontal staff fragments belonging to the same staff row.

    Prevents stains, creases, or vertical folds from duplicating a single staff
    into ``S+1`` slots.
    """
    if len(frags) <= 1:
        return frags

    work = [dict(f) for f in frags]

    while True:
        best_pair: tuple[int, int] | None = None
        best_key: tuple[float, float] | None = None

        for i in range(len(work)):
            for j in range(i + 1, len(work)):
                a, b = work[i], work[j]
                if a["x_start"] > b["x_start"]:
                    a, b = b, a

                sp_a = a["sp_med"]
                sp_b = b["sp_med"]
                sp_ref = 0.5 * (sp_a + sp_b)
                if sp_a <= 0.0 or sp_b <= 0.0:
                    continue
                ratio = sp_a / sp_b
                if ratio < 0.5 or ratio > 2.0:
                    continue

                span_min = min(a["x_end"] - a["x_start"], b["x_end"] - b["x_start"])
                overlap = min(a["x_end"], b["x_end"]) - max(a["x_start"], b["x_start"])
                # Must be horizontally disjoint (or at most tiny 1-2 sp edge overlap).
                if overlap > max(2.0 * sp_ref, 0.15 * max(span_min, 1.0)):
                    continue

                x_mid = 0.5 * (a["x_end"] + b["x_start"])
                ya_mid = _extrapolate_fragment_y(a["x"], a["y"], x_mid, side="right")
                yb_mid = _extrapolate_fragment_y(b["x"], b["y"], x_mid, side="left")
                y_diff = abs(ya_mid - yb_mid)
                if y_diff > sp_tol_factor * sp_ref:
                    continue

                gap = max(0.0, b["x_start"] - a["x_end"])
                key = (gap, y_diff)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair
        a, b = work[i], work[j]
        x_cat = np.concatenate([a["x"], b["x"]])
        y_cat = np.concatenate([a["y"], b["y"]])
        w_cat = np.concatenate([a["w"], b["w"]])
        sp_cat = np.concatenate([a["sp"], b["sp"]])

        order = np.argsort(x_cat, kind="mergesort")
        x_cat = x_cat[order]
        y_cat = y_cat[order]
        w_cat = w_cat[order]
        sp_cat = sp_cat[order]

        # Collapse any duplicate x columns by weight-averaging.
        ux, idx = np.unique(x_cat, return_index=True)
        if ux.size < x_cat.size:
            y_groups = np.split(y_cat, idx[1:])
            w_groups = np.split(w_cat, idx[1:])
            sp_groups = np.split(sp_cat, idx[1:])
            x_cat = ux
            y_cat = np.array([float((yg * wg).sum() / max(wg.sum(), 1e-6)) for yg, wg in zip(y_groups, w_groups)], dtype=np.float64)
            sp_cat = np.array([float((sg * wg).sum() / max(wg.sum(), 1e-6)) for sg, wg in zip(sp_groups, w_groups)], dtype=np.float64)
            w_cat = np.array([float(wg.sum()) for wg in w_groups], dtype=np.float64)

        valid_sp = sp_cat[sp_cat > 1.0]
        sp_med = float(np.median(valid_sp)) if valid_sp.size > 0 else 0.5 * (a["sp_med"] + b["sp_med"])
        total_px = a["n_px"] + b["n_px"]
        merged = {
            "x": x_cat,
            "y": y_cat,
            "w": w_cat,
            "sp": sp_cat,
            "x_start": float(x_cat[0]),
            "x_end": float(x_cat[-1]),
            "sp_med": sp_med,
            "conf_sum": a["conf_sum"] + b["conf_sum"],
            "n_px": total_px,
        }
        # Replace i with merged and remove j (j > i).
        work[i] = merged
        work.pop(j)

    return work


def extract_curved_systems(
    probs: torch.Tensor | np.ndarray,
    orig_w: int,
    orig_h: int,
    center_thresh: float = 0.4,
    sys_thresh: float = 0.55,
    min_width_frac: float = 0.10,
    min_width_floor_frac: float = 0.06,
    sys_link_thresh: float = 0.45,
) -> list[CurvedSystem]:
    """Extract curved staves & grouped systems from 4-channel UNet predictions.

    Args:
        probs: ``[4, Hp, Wp]`` or ``[1, 4, Hp, Wp]`` tensor/array containing
            ``(p_lines, p_center, p_system, sp_normal_over_H)``.
        orig_w: Original page width in pixels.
        orig_h: Original page height in pixels.
        center_thresh: Threshold on ``ch1`` for staff-centerline components.
        sys_thresh: Threshold on ``ch2`` for system-body components (default
            ``0.55`` to resist cross-system bleed on dense 5-6 staff pages).
        min_width_frac: Minimum horizontal span relative to the median detected
            staff width (default ``0.10 * median_w``), allowing short trailing
            codas to survive dark desk borders.
        min_width_floor_frac: Hard minimum width floor as a fraction of ``Wp``
            (default ``0.06 * wp``).
        sys_link_thresh: Minimum ``ch2`` bridge / inter-staff valley probability
            required to group adjacent staves into one system.

    Returns:
        List of :class:`CurvedSystem` sorted top-to-bottom on the page.
    """
    if isinstance(probs, torch.Tensor):
        arr = probs.detach().cpu().float().numpy()
    else:
        arr = np.asarray(probs, dtype=np.float32)

    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got {arr.shape[0]}")
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[0] != 4:
        raise ValueError(f"expected [4, Hp, Wp] probs, got {arr.shape}")

    p_lines, p_center, p_sys, sp_rel = arr[0], arr[1], arr[2], arr[3]
    hp, wp = p_center.shape
    scale_x = float(orig_w) / float(wp)
    scale_y = float(orig_h) / float(hp)

    # 1. Connected components on staff centerline mask (8-connectivity handles steep tilts).
    center_mask = p_center >= center_thresh
    # Bridge tiny 1-pixel horizontal gaps from faint/occluded staff sections.
    center_closed = ndimage.binary_closing(center_mask, structure=np.ones((1, 3), dtype=bool))
    labeled, n_comp = ndimage.label(center_closed, structure=np.ones((3, 3), dtype=int))

    # Extract raw fragments with a permissive 3-column floor so stain/fold fragments
    # can be merged before applying the relative staff-width filter.
    raw_frags: list[dict[str, Any]] = []
    for cid in range(1, n_comp + 1):
        ys_idx, xs_idx = np.nonzero(labeled == cid)
        if xs_idx.size < 3:
            continue
        col_min, col_max = int(xs_idx.min()), int(xs_idx.max())
        if (col_max - col_min + 1) < 3:
            continue

        cols_x: list[float] = []
        cols_y: list[float] = []
        cols_w: list[float] = []
        cols_sp: list[float] = []

        order = np.argsort(xs_idx, kind="mergesort")
        xs_sorted = xs_idx[order]
        ys_sorted = ys_idx[order]
        unique_x, split_idx = np.unique(xs_sorted, return_index=True)
        y_groups = np.split(ys_sorted, split_idx[1:])

        for col, y_rows in zip(unique_x, y_groups):
            weights = np.maximum(p_center[y_rows, col].astype(np.float64), 1e-4)
            w_sum = float(weights.sum())
            yc = float((y_rows.astype(np.float64) * weights).sum() / w_sum)

            # Sample sp_rel around the centerline in a +-2 px window.
            r0 = max(0, int(round(yc)) - 2)
            r1 = min(hp, int(round(yc)) + 3)
            sp_patch = sp_rel[r0:r1, col]
            sp_w = np.maximum(p_center[r0:r1, col] + p_lines[r0:r1, col], 1e-4)
            sp_val = float((sp_patch * sp_w).sum() / sp_w.sum()) * float(orig_h)

            cols_x.append((float(col) + 0.5) * scale_x)
            cols_y.append((yc + 0.5) * scale_y)
            cols_w.append(w_sum)
            cols_sp.append(sp_val)

        if len(cols_x) < 3:
            continue

        x_arr = np.asarray(cols_x, dtype=np.float64)
        y_arr = np.asarray(cols_y, dtype=np.float64)
        w_arr = np.asarray(cols_w, dtype=np.float64)
        sp_arr = np.asarray(cols_sp, dtype=np.float64)

        valid_sp = sp_arr[sp_arr > 1.0]
        sp_med = float(np.median(valid_sp)) if valid_sp.size > 0 else max(4.0, 0.01 * orig_h)
        mask_c = labeled == cid
        raw_frags.append(
            {
                "x": x_arr,
                "y": y_arr,
                "w": w_arr,
                "sp": sp_arr,
                "x_start": float(x_arr[0]),
                "x_end": float(x_arr[-1]),
                "sp_med": sp_med,
                "conf_sum": float(p_center[mask_c].sum()),
                "n_px": int(mask_c.sum()),
            }
        )

    if not raw_frags:
        return []

    # Merge same-row horizontal fragments split by stains, creases, or folds.
    merged_frags = _merge_horizontal_fragments(raw_frags, sp_tol_factor=0.5)

    # Relative width threshold: max(min_width_floor_frac * wp, min_width_frac * median_w).
    floor_cols = max(4, int(round(min_width_floor_frac * wp)))
    frag_col_spans = [
        max(1.0, (f["x_end"] - f["x_start"]) / max(scale_x, 1e-6) + 1.0)
        for f in merged_frags
    ]
    sig_spans = [s for s in frag_col_spans if s >= floor_cols]
    median_w_cols = float(np.median(sig_spans)) if sig_spans else float(np.median(frag_col_spans))
    min_cols = max(floor_cols, int(round(min_width_frac * median_w_cols)))

    staves: list[CurvedStaff] = []
    for f, col_span in zip(merged_frags, frag_col_spans):
        if col_span < min_cols or f["x"].size < 4:
            continue

        x_arr = f["x"]
        y_arr = f["y"]
        w_arr = f["w"]
        sp_arr = f["sp"]
        x_start = f["x_start"]
        x_end = f["x_end"]
        span = max(x_end - x_start, 1e-6)
        t = (x_arr - x_start) / span

        y_ctrl = fit_cubic_bezier_1d(t, y_arr, weights=w_arr)
        x_ctrl = np.linspace(x_start, x_end, 4, dtype=np.float64)
        ctrl_pts = np.stack([x_ctrl, y_ctrl], axis=-1)

        sp_med = f["sp_med"]
        sp_clipped = np.clip(sp_arr, 0.5 * sp_med, 2.0 * sp_med)
        sp_ctrl = fit_cubic_bezier_1d(t, sp_clipped, weights=w_arr)
        sp_ctrl = np.clip(sp_ctrl, 0.5 * sp_med, 2.0 * sp_med)

        conf = f["conf_sum"] / max(f["n_px"], 1)
        staves.append(
            CurvedStaff(
                ctrl_pts=ctrl_pts,
                sp_ctrl=sp_ctrl,
                x_start=x_start,
                x_end=x_end,
                sp_median=sp_med,
                confidence=conf,
            )
        )

    if not staves:
        return []

    # Sort staves top-to-bottom by mean y.
    staves.sort(key=lambda s: s.mean_y())

    # 2. Group staves into systems using ch2 (system-body mask) with 1-px vertical
    #    erosion + inter-staff vertical valley guard against cross-system bleed.
    sys_mask = p_sys >= sys_thresh
    sys_eroded = ndimage.binary_erosion(sys_mask, structure=np.ones((3, 1), dtype=bool))
    sys_for_cc = sys_eroded if np.any(sys_eroded) else sys_mask
    sys_labeled, _ = ndimage.label(sys_for_cc, structure=np.ones((3, 3), dtype=int))

    def _staff_sys_label(st: CurvedStaff) -> int:
        ts = np.linspace(0.1, 0.9, 17)
        pts = st.eval_center(ts)
        gx = np.clip((pts[:, 0] / scale_x - 0.5).round().astype(int), 0, wp - 1)
        gy = np.clip((pts[:, 1] / scale_y - 0.5).round().astype(int), 0, hp - 1)
        lbls = sys_labeled[gy, gx]
        pos = lbls[lbls > 0]
        if pos.size == 0:
            return 0
        vals, counts = np.unique(pos, return_counts=True)
        return int(vals[np.argmax(counts)])

    def _bridge_stats(a: CurvedStaff, b: CurvedStaff) -> tuple[float, float]:
        """Return ``(mean_bridge_prob, median_vertical_valley_min)`` across x-overlap."""
        ov_x0 = max(a.x_start, b.x_start)
        ov_x1 = min(a.x_end, b.x_end)
        if ov_x1 <= ov_x0:
            return 0.0, 0.0
        xs = np.linspace(ov_x0, ov_x1, 15)
        ya = a.eval_y_at_x(xs)
        yb = b.eval_y_at_x(xs)
        means: list[float] = []
        mins: list[float] = []
        for x, y0, y1 in zip(xs, ya, yb):
            gx = int(np.clip(round(x / scale_x - 0.5), 0, wp - 1))
            r0 = int(np.clip(round(min(y0, y1) / scale_y - 0.5), 0, hp - 1))
            r1 = int(np.clip(round(max(y0, y1) / scale_y - 0.5), 0, hp - 1))
            if r1 >= r0:
                col_slice = p_sys[r0 : r1 + 1, gx]
                means.append(float(col_slice.mean()))
                mins.append(float(col_slice.min()))
        if not means:
            return 0.0, 0.0
        return float(np.mean(means)), float(np.median(mins))

    # Union-Find over staves.
    n = len(staves)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # Precompute adjacent-staff valley guards so a 1-px bleed cannot merge two systems.
    adj_valley_ok: list[bool] = []
    adj_bridge_mean: list[float] = []
    for k in range(n - 1):
        b_mean, b_valley = _bridge_stats(staves[k], staves[k + 1])
        adj_bridge_mean.append(b_mean)
        adj_valley_ok.append(b_valley >= sys_link_thresh)

    sys_ids = [_staff_sys_label(s) for s in staves]
    for i in range(n):
        for j in range(i + 1, n):
            no_valley_between = all(adj_valley_ok[k] for k in range(i, j))
            if not no_valley_between:
                continue
            same_cc = sys_ids[i] > 0 and sys_ids[i] == sys_ids[j]
            if same_cc:
                union(i, j)
            elif j == i + 1 and adj_bridge_mean[i] >= sys_link_thresh:
                union(i, j)

    groups_map: dict[int, list[CurvedStaff]] = {}
    for i, st in enumerate(staves):
        groups_map.setdefault(find(i), []).append(st)

    systems: list[CurvedSystem] = []
    for grp in groups_map.values():
        grp.sort(key=lambda s: s.mean_y())
        x0 = min(s.x_start for s in grp)
        x1 = max(s.x_end for s in grp)
        poly = _build_system_polygon(grp)
        conf = float(np.mean([s.confidence for s in grp]))
        systems.append(CurvedSystem(staves=grp, x_start=x0, x_end=x1, polygon=poly, confidence=conf))

    systems.sort(key=lambda sys_obj: sys_obj.staves[0].mean_y())
    return systems


def _suppress_overlapping_staff_duplicates(
    staves: list[CurvedStaff],
    min_center_gap_sp: float = 4.5,
) -> list[CurvedStaff]:
    """Suppress vertically overlapping duplicate/split staff fragments (NMS).

    A real 5-line staff spans ``4.0 * sp`` vertically, so two distinct staves
    sharing horizontal overlap cannot have centerlines closer than ``~4.5 * sp``.
    When a fold/stain or strong beam triggers a duplicate ``ch1`` ridge inside
    or adjacent to a staff, keep the higher-confidence/wider one.
    """
    if len(staves) <= 1:
        return list(staves)
    # Rank by confidence * width so real full-width staves beat short spurious fragments.
    ranked = sorted(
        staves,
        key=lambda s: float(s.confidence) * max(1.0, float(s.x_end - s.x_start)),
        reverse=True,
    )
    kept: list[CurvedStaff] = []
    for cand in ranked:
        dup = False
        for prev in kept:
            ov0 = max(cand.x_start, prev.x_start)
            ov1 = min(cand.x_end, prev.x_end)
            min_w = max(1.0, min(cand.x_end - cand.x_start, prev.x_end - prev.x_start))
            if (ov1 - ov0) / min_w < 0.25:
                continue
            xs = np.linspace(ov0, ov1, 9)
            dy = float(np.median(np.abs(cand.eval_y_at_x(xs) - prev.eval_y_at_x(xs))))
            sp = max(4.0, 0.5 * (cand.sp_median + prev.sp_median))
            if dy < min_center_gap_sp * sp:
                dup = True
                break
        if not dup:
            kept.append(cand)
    kept.sort(key=lambda s: s.mean_y())
    return kept


def _has_left_system_connector(
    gray: np.ndarray,
    a: CurvedStaff,
    b: CurvedStaff,
    min_coverage: float = 0.70,
) -> bool:
    """True iff full-res page image has a continuous left-edge system barline between ``a`` and ``b``.

    In Western notation, staves in the same system are linked at their left
    margin (``x_start``) by a continuous vertical system barline (and often a
    brace/bracket), whereas separate systems have white paper between them.
    ``StaffUNet2D`` at ``512x384`` cannot resolve a 1-px vertical barline, so
    this full-res check cleanly separates single-staff folk melodies and dense
    5-6 staff choral systems.
    """
    h, w = gray.shape
    sp = max(4.0, 0.5 * (a.sp_median + b.sp_median))
    if abs(a.x_start - b.x_start) > 6.0 * sp:
        return False
    ya = float(a.eval_y_at_x(np.array([a.x_start]))[0]) + 2.3 * sp
    yb = float(b.eval_y_at_x(np.array([b.x_start]))[0]) - 2.3 * sp
    if yb - ya < 1.5 * sp:
        return True
    x0 = min(a.x_start, b.x_start)
    dx_min = int(round(-2.5 * sp))
    dx_max = int(round(1.5 * sp))
    c_lo = max(0, int(round(x0 + dx_min - 2.0 * sp)))
    c_hi = min(w, int(round(x0 + dx_max + 4.0 * sp)))
    r_lo = max(0, int(round(ya)))
    r_hi = min(h, int(round(yb)) + 1)
    if c_hi <= c_lo or r_hi <= r_lo:
        return False

    patch = gray[r_lo:r_hi, c_lo:c_hi]
    bg = float(np.percentile(patch, 80))
    # If the gap region is completely washed out / uniform, return False safely.
    if bg < 0.15:
        return False
    thr = max(0.08, 0.22 * bg)

    n_rows = max(12, int(round(yb - ya)))
    fracs = np.linspace(0.0, 1.0, n_rows)
    ys = np.clip(np.round(ya + fracs * (yb - ya)).astype(int), 0, h - 1)
    xs_base = a.x_start + fracs * (b.x_start - a.x_start)

    best_cov = 0.0
    for dx in range(dx_min, dx_max + 1):
        xc = np.clip(np.round(xs_base + dx).astype(int), 1, w - 2)
        dark_band = np.minimum(np.minimum(gray[ys, xc - 1], gray[ys, xc]), gray[ys, xc + 1])
        cov = float(((bg - dark_band) >= thr).mean())
        if cov > best_cov:
            best_cov = cov
            if best_cov >= min_coverage:
                return True
    return False


def refine_systems_with_page(
    page_img: Image.Image | np.ndarray,
    systems: list[CurvedSystem],
    min_center_gap_sp: float = 4.5,
    min_connector_coverage: float = 0.70,
) -> list[CurvedSystem]:
    """Refine ``StaffUNet2D`` system grouping using full-res page cues.

    1. Suppresses vertically overlapping duplicate staff fragments (``< 4.5 sp``).
    2. Regroups staves top-to-bottom using full-res left-edge system barlines
       (:func:`_has_left_system_connector`), eliminating cross-system merges on
       single-staff folk pages and 5-6 staff choral pages.
    """
    if not systems:
        return []
    if isinstance(page_img, Image.Image):
        gray = np.asarray(page_img.convert("L"), dtype=np.float32) / 255.0
    else:
        arr = np.asarray(page_img, dtype=np.float32)
        if arr.ndim == 3:
            gray = arr.mean(axis=-1)
        else:
            gray = arr
        if gray.max() > 1.5:
            gray = gray / 255.0

    all_staves = [st for sys_obj in systems for st in sys_obj.staves]
    staves = _suppress_overlapping_staff_duplicates(all_staves, min_center_gap_sp=min_center_gap_sp)
    if not staves:
        return []

    groups: list[list[CurvedStaff]] = []
    for st in staves:
        if not groups:
            groups.append([st])
        elif _has_left_system_connector(
            gray, groups[-1][-1], st, min_coverage=min_connector_coverage
        ):
            groups[-1].append(st)
        else:
            groups.append([st])

    refined: list[CurvedSystem] = []
    for grp in groups:
        grp.sort(key=lambda s: s.mean_y())
        x0 = min(s.x_start for s in grp)
        x1 = max(s.x_end for s in grp)
        poly = _build_system_polygon(grp)
        conf = float(np.mean([s.confidence for s in grp]))
        refined.append(CurvedSystem(staves=grp, x_start=x0, x_end=x1, polygon=poly, confidence=conf))
    refined.sort(key=lambda s: s.staves[0].mean_y())
    return refined


# ---------------------------------------------------------------------------
# Target rasterization: symbox / hybrid_staff.jsonl / explicit curves -> [4, out_h, out_w]
# ---------------------------------------------------------------------------


def _parse_staff_entry(entry: Any, page_w: int) -> tuple[np.ndarray, float, float, int | None]:
    """Normalize a staff entry into ``(lines_5xN, x0, x1, system_id)``.

    Supports:
      - ``CurvedStaff``
      - dict with ``'lines'`` (5 scalars or 5 curves), optional ``x0/x1`` or
        ``x_start/x_end``, optional ``system_id``
      - dict with ``'y_center'`` + ``'sp'``, optional ``x0/x1``, ``system_id``
      - raw 5-line array/list ``(5,)`` or ``(5, N)``
    """
    if isinstance(entry, CurvedStaff):
        xs = np.linspace(entry.x_start, entry.x_end, 64)
        return entry.eval_lines_at_x(xs), entry.x_start, entry.x_end, None

    if isinstance(entry, dict):
        x0 = float(entry.get("x0", entry.get("x_start", 0.0)))
        x1 = float(entry.get("x1", entry.get("x_end", float(page_w - 1))))
        sys_id = entry.get("system_id", None)
        if "lines" in entry:
            raw = entry["lines"]
        elif "y_center" in entry and "sp" in entry:
            yc = np.atleast_1d(np.asarray(entry["y_center"], dtype=np.float64))
            sp = np.atleast_1d(np.asarray(entry["sp"], dtype=np.float64))
            n = max(yc.size, sp.size)
            if yc.size == 1:
                yc = np.full(n, yc[0], dtype=np.float64)
            if sp.size == 1:
                sp = np.full(n, sp[0], dtype=np.float64)
            offsets = np.arange(-2, 3, dtype=np.float64)[:, None]
            return yc[None, :] + offsets * sp[None, :], x0, x1, sys_id
        else:
            raise ValueError(f"unrecognized staff dict keys: {list(entry.keys())}")
    else:
        raw = entry
        x0, x1, sys_id = 0.0, float(page_w - 1), None

    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 1 and arr.shape[0] == 5:
        lines = np.repeat(arr[:, None], 2, axis=1)
    elif arr.ndim == 2 and arr.shape[0] == 5:
        lines = arr
    else:
        raise ValueError(f"staff lines must have shape (5,) or (5, N), got {arr.shape}")
    return lines, x0, x1, sys_id


def rasterize_staff_targets(
    staves: Sequence[Any] | dict[str, Any],
    w: int,
    h: int,
    out_h: int,
    out_w: int,
    systems: Sequence[Sequence[int]] | None = None,
) -> torch.Tensor:
    """Rasterize 4-channel training targets ``[4, out_h, out_w]`` at 1/4 resolution.

    Args:
        staves: Either a list of staff entries (dicts, ``CurvedStaff``, or
            ``(5,)`` / ``(5, N)`` line arrays), or a ``hybrid_staff.jsonl``
            record dict containing ``{'staves': [...]}``.
        w: Original page width in pixels.
        h: Original page height in pixels.
        out_h: Target map height (typically ``h // 4``).
        out_w: Target map width (typically ``w // 4``).
        systems: Optional explicit list of staff-index groups belonging to
            each system (e.g. ``[[0, 1], [2, 3]]``). If omitted, groups by
            ``system_id`` when present, else all staves in a ``hybrid_staff``
            dict form one system, or each staff forms its own system.

    Returns:
        Float32 tensor ``[4, out_h, out_w]``:
          - ``[0]``: 5-line staff-line mask in ``[0, 1]``
          - ``[1]``: staff-centerline mask in ``[0, 1]``
          - ``[2]``: system-body mask in ``[0, 1]``
          - ``[3]``: ``sp_px / h`` on staff regions, ``0.0`` elsewhere
    """
    from_hybrid_dict = False
    if isinstance(staves, dict):
        from_hybrid_dict = True
        staff_list = list(staves.get("staves", []))
    else:
        staff_list = list(staves)

    tgt = np.zeros((4, out_h, out_w), dtype=np.float32)
    if not staff_list:
        return torch.from_numpy(tgt)

    sx = float(out_w) / float(w)
    sy = float(out_h) / float(h)
    col_x = (np.arange(out_w, dtype=np.float64) + 0.5) / sx  # page x for each out col
    row_y = np.arange(out_h, dtype=np.float64) + 0.5         # out-grid y centers

    parsed: list[tuple[np.ndarray, np.ndarray, np.ndarray, int | None]] = []
    # Each item: (lines_5xOutW in out_y, sp_px_OutW in orig px, active_cols bool[OutW], sys_id)

    for entry in staff_list:
        lines_orig, x0, x1, sys_id = _parse_staff_entry(entry, w)
        n_pts = lines_orig.shape[1]
        src_x = np.linspace(x0, x1, num=n_pts)
        active = (col_x >= x0) & (col_x <= x1)
        if not np.any(active):
            continue

        lines_out = np.empty((5, out_w), dtype=np.float64)
        for k in range(5):
            lines_out[k] = np.interp(col_x, src_x, lines_orig[k]) * sy

        # Vertical line spacing along the staff, plus normal spacing sp_vert * cos(atan(dy/dx))
        # so ch3 supervises true normal staff spacing matching BezierRibbonSampler(offset_mode="normal").
        sp_vert_orig = np.maximum(np.mean(np.diff(lines_orig, axis=0), axis=0), 1.0)
        if n_pts >= 2 and (x1 - x0) > 1e-6:
            dy_dx = np.gradient(lines_orig[2], src_x)
        else:
            dy_dx = np.zeros_like(lines_orig[2])
        cos_theta = np.cos(np.arctan(dy_dx))
        sp_norm_orig = np.maximum(sp_vert_orig * cos_theta, 1.0)

        sp_vert_col = np.interp(col_x, src_x, sp_vert_orig)
        sp_norm_col = np.interp(col_x, src_x, sp_norm_orig)
        sp_out_col = np.maximum(sp_vert_col * sy, 0.75)

        act_cols = np.nonzero(active)[0]
        for c in act_cols:
            sp_o = sp_out_col[c]
            # ch0: 5 staff lines (Gaussian band, sigma ~ 0.35 px or 0.22*sp_o)
            line_sigma = max(0.55, 0.22 * sp_o)
            for k in range(5):
                dy = row_y - lines_out[k, c]
                band = np.exp(-0.5 * (dy / line_sigma) ** 2)
                tgt[0, :, c] = np.maximum(tgt[0, :, c], (band >= 0.35).astype(np.float32))

            # ch1: staff centerline (k=2) — smooth Gaussian for sub-pixel centroiding
            yc = lines_out[2, c]
            c_sigma = max(0.75, 0.42 * sp_o)
            c_band = np.exp(-0.5 * ((row_y - yc) / c_sigma) ** 2).astype(np.float32)
            c_band[c_band < 0.05] = 0.0
            tgt[1, :, c] = np.maximum(tgt[1, :, c], c_band)

            # ch3: local normal sp_px / h over staff span [line0 - sp, line4 + sp]
            y_lo = lines_out[0, c] - sp_o
            y_hi = lines_out[4, c] + sp_o
            smask = (row_y >= y_lo) & (row_y <= y_hi)
            tgt[3, smask, c] = float(sp_norm_col[c] / float(h))

        parsed.append((lines_out, sp_vert_col, active, sys_id))

    # Build system index groups for ch2.
    if systems is not None:
        groups = [list(g) for g in systems if g]
    elif any(p[3] is not None for p in parsed):
        by_id: dict[int, list[int]] = {}
        for idx, p in enumerate(parsed):
            key = p[3] if p[3] is not None else -(idx + 1)
            by_id.setdefault(int(key), []).append(idx)
        groups = list(by_id.values())
    elif from_hybrid_dict:
        groups = [list(range(len(parsed)))]
    else:
        groups = [[i] for i in range(len(parsed))]

    for grp in groups:
        valid = [i for i in grp if 0 <= i < len(parsed)]
        if not valid:
            continue
        act_union = np.zeros(out_w, dtype=bool)
        for i in valid:
            act_union |= parsed[i][2]
        for c in np.nonzero(act_union)[0]:
            y_tops = []
            y_bots = []
            for i in valid:
                lines_out, sp_vert_col, active, _ = parsed[i]
                if active[c]:
                    sp_o = max(sp_vert_col[c] * sy, 0.75)
                    y_tops.append(lines_out[0, c] - 0.5 * sp_o)
                    y_bots.append(lines_out[4, c] + 0.5 * sp_o)
            if y_tops:
                y0 = min(y_tops)
                y1 = max(y_bots)
                mask = (row_y >= y0) & (row_y <= y1)
                tgt[2, mask, c] = 1.0

    return torch.from_numpy(tgt)


# ---------------------------------------------------------------------------
# Joint image + (5, N) staff-curve on-the-fly augmentations
# ---------------------------------------------------------------------------


def _solve_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Solve 3x3 projective homography mapping 4 2D points ``src -> dst``."""
    A = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    for i in range(4):
        x, y = float(src[i, 0]), float(src[i, 1])
        u, v = float(dst[i, 0]), float(dst[i, 1])
        A[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y]
        b[2 * i] = u
        A[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y]
        b[2 * i + 1] = v
    h = np.linalg.solve(A, b)
    return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]], dtype=np.float64)


def _apply_homography(M: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply 3x3 homography ``M`` to coordinate arrays ``xs, ys``."""
    denom = M[2, 0] * xs + M[2, 1] * ys + M[2, 2]
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
    ox = (M[0, 0] * xs + M[0, 1] * ys + M[0, 2]) / denom
    oy = (M[1, 0] * xs + M[1, 1] * ys + M[1, 2]) / denom
    return ox, oy


def _make_desk_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Synthesize a dark/wood/clutter desk background ``[H, W, 3]`` float32 in ``[0, 1]``."""
    style = int(rng.integers(0, 3))
    if style == 0:
        # Solid dark desk with mild sensor noise.
        base = rng.uniform(0.05, 0.22, size=(1, 1, 3))
        bg = np.broadcast_to(base, (h, w, 3)).astype(np.float32).copy()
        bg += rng.normal(0.0, 0.015, size=(h, w, 3)).astype(np.float32)
    elif style == 1:
        # Warm wood-grain desk texture.
        r_base = rng.uniform(0.22, 0.45)
        g_base = r_base * rng.uniform(0.55, 0.78)
        b_base = r_base * rng.uniform(0.30, 0.52)
        bg = np.empty((h, w, 3), dtype=np.float32)
        bg[:, :, 0] = r_base
        bg[:, :, 1] = g_base
        bg[:, :, 2] = b_base
        # Low-frequency directional wood grain streaks.
        horiz = bool(rng.random() < 0.5)
        if horiz:
            band = rng.normal(0.0, 0.04, size=(max(4, h // 8), 1)).astype(np.float32)
            streak = np.repeat(band, (h + band.shape[0] - 1) // band.shape[0], axis=0)[:h, :]
            bg += streak[:, :, None]
        else:
            band = rng.normal(0.0, 0.04, size=(1, max(4, w // 8))).astype(np.float32)
            streak = np.repeat(band, (w + band.shape[1] - 1) // band.shape[1], axis=1)[:, :w]
            bg += streak[:, :, None]
        bg += rng.normal(0.0, 0.015, size=(h, w, 3)).astype(np.float32)
    else:
        # Cluttered noise/fabric desk with subtle random rectangular patches.
        base = rng.uniform(0.08, 0.30, size=(1, 1, 3)).astype(np.float32)
        bg = np.broadcast_to(base, (h, w, 3)).copy()
        coarse_h, coarse_w = max(4, h // 16), max(4, w // 16)
        coarse = rng.normal(0.0, 0.05, size=(coarse_h, coarse_w, 3)).astype(np.float32)
        coarse_img = Image.fromarray(
            np.clip((coarse + 0.5) * 255.0, 0, 255).astype(np.uint8), mode="RGB"
        ).resize((w, h), Image.Resampling.BILINEAR)
        bg += np.asarray(coarse_img, dtype=np.float32) / 255.0 - 0.5
        for _ in range(int(rng.integers(2, 6))):
            rx0 = int(rng.integers(0, max(1, w - 10)))
            ry0 = int(rng.integers(0, max(1, h - 10)))
            rx1 = min(w, rx0 + int(rng.integers(10, max(12, w // 3))))
            ry1 = min(h, ry0 + int(rng.integers(10, max(12, h // 3))))
            patch_col = rng.uniform(0.05, 0.45, size=(1, 1, 3)).astype(np.float32)
            bg[ry0:ry1, rx0:rx1] = 0.6 * bg[ry0:ry1, rx0:rx1] + 0.4 * patch_col
    return np.clip(bg, 0.0, 1.0)


def _augment_page_and_staves(
    img_rgb: np.ndarray,
    staff_entries: list[dict[str, Any]],
    systems: Sequence[Sequence[int]] | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, Any]], list[list[int]] | None]:
    """Jointly augment an RGB page ``[H, W, 3]`` in ``[0, 1]`` and its ``(5, N)`` staff curves.

    Implements:
      - 10% short-coda masking (blank right 65-85% of bottom system + truncate GT ``x1``)
      - 30% sepia/parchment tint + faded brown/gray ink + subtle bleed-through ghost staves
        + stain / blur / JPEG
      - Synthetic S-curve / sine vertical warp (``+-4% H``)
      - Perspective keystone
      - ``+-12 deg`` rotation
      - ``0-25%`` random outer desk border (solid dark / wood / noise clutter)
    """
    h, w, _ = img_rgb.shape
    work = img_rgb.astype(np.float32, copy=True)

    # Normalize all staves into dense (5, N, 2) point arrays + sys_id.
    n_pts = 64
    curves: list[dict[str, Any]] = []
    for idx, entry in enumerate(staff_entries):
        lines_5, x0, x1, sys_id = _parse_staff_entry(entry, w)
        src_x = np.linspace(x0, x1, num=lines_5.shape[1])
        xs = np.linspace(x0, x1, num=n_pts)
        ys = np.empty((5, n_pts), dtype=np.float64)
        for k in range(5):
            ys[k] = np.interp(xs, src_x, lines_5[k])
        pts = np.stack([np.broadcast_to(xs[None, :], (5, n_pts)), ys], axis=-1).copy()
        curves.append({"pts": pts, "sys_id": sys_id, "orig_idx": idx})

    # Resolve system index groups so short-coda masking targets the bottom system cleanly.
    if systems is not None:
        sys_groups = [list(g) for g in systems if g]
    elif any(c["sys_id"] is not None for c in curves):
        by_id: dict[int, list[int]] = {}
        for i, c in enumerate(curves):
            key = c["sys_id"] if c["sys_id"] is not None else -(i + 1)
            by_id.setdefault(int(key), []).append(i)
        sys_groups = list(by_id.values())
    else:
        sys_groups = [[i] for i in range(len(curves))]

    # 1. Short-coda masking (10%): blank right 65-85% of bottom system + truncate GT x1.
    if curves and sys_groups and rng.random() < 0.10:
        bot_grp = max(
            sys_groups,
            key=lambda g: float(np.mean([curves[i]["pts"][2, :, 1].mean() for i in g if i < len(curves)])),
        )
        valid_bot = [i for i in bot_grp if 0 <= i < len(curves)]
        if valid_bot:
            blank_frac = float(rng.uniform(0.65, 0.85))
            keep_frac = 1.0 - blank_frac
            paper_col = np.median(work.reshape(-1, 3), axis=0).astype(np.float32)
            for i in valid_bot:
                pts = curves[i]["pts"]
                x0_s = float(pts[2, 0, 0])
                x1_s = float(pts[2, -1, 0])
                x_cut = x0_s + keep_frac * (x1_s - x0_s)
                sp_est = float(np.maximum(np.mean(np.diff(pts[:, :, 1], axis=0)), 2.0))
                y_lo = max(0, int(math.floor(float(pts[0, :, 1].min()) - 2.0 * sp_est)))
                y_hi = min(h, int(math.ceil(float(pts[4, :, 1].max()) + 2.0 * sp_est)))
                cx0 = max(0, int(math.floor(x_cut)))
                cx1 = min(w, int(math.ceil(x1_s + 2.0 * sp_est)))
                if cx1 > cx0 and y_hi > y_lo:
                    work[y_lo:y_hi, cx0:cx1] = paper_col[None, None, :]
                # Truncate GT curve to [x0_s, x_cut] and resample n_pts.
                old_x = pts[2, :, 0]
                new_x = np.linspace(x0_s, x_cut, num=n_pts)
                new_pts = np.empty_like(pts)
                new_pts[:, :, 0] = new_x[None, :]
                for k in range(5):
                    new_pts[k, :, 1] = np.interp(new_x, old_x, pts[k, :, 1])
                curves[i]["pts"] = new_pts

    # 2. Photometric / aging / bleed-through (30%): sepia/parchment tint + faded ink
    #    + subtle bleed-through ghost staves + stain / blur / JPEG.
    do_jpeg = False
    if rng.random() < 0.30:
        # Sepia / parchment paper + faded brown/gray ink.
        lum = 0.299 * work[:, :, 0] + 0.587 * work[:, :, 1] + 0.114 * work[:, :, 2]
        paper_tint = np.array(
            [rng.uniform(0.92, 1.0), rng.uniform(0.84, 0.95), rng.uniform(0.68, 0.86)],
            dtype=np.float32,
        )
        ink_tint = np.array(
            [rng.uniform(0.14, 0.42), rng.uniform(0.11, 0.35), rng.uniform(0.08, 0.28)],
            dtype=np.float32,
        )
        aged = ink_tint[None, None, :] * (1.0 - lum[:, :, None]) + paper_tint[None, None, :] * lum[:, :, None]
        blend = float(rng.uniform(0.55, 1.0))
        work = (1.0 - blend) * work + blend * aged

        # Subtle bleed-through ghost staves (flipped + shifted copy).
        if rng.random() < 0.5:
            ghost = work[:, ::-1, :]
            shift_y = int(rng.integers(-max(4, h // 25), max(5, h // 25)))
            ghost = np.roll(ghost, shift_y, axis=0)
            alpha = float(rng.uniform(0.05, 0.14))
            work = work * (1.0 - alpha) + np.minimum(work, ghost) * alpha

        # Elliptical translucent stain.
        if rng.random() < 0.5:
            yy, xx = np.ogrid[:h, :w]
            for _ in range(int(rng.integers(1, 3))):
                cx_s = float(rng.uniform(0.1 * w, 0.9 * w))
                cy_s = float(rng.uniform(0.1 * h, 0.9 * h))
                rx_s = float(rng.uniform(0.04 * w, 0.16 * w))
                ry_s = float(rng.uniform(0.03 * h, 0.12 * h))
                dist2 = ((xx - cx_s) / max(rx_s, 1.0)) ** 2 + ((yy - cy_s) / max(ry_s, 1.0)) ** 2
                blob = np.exp(-0.5 * dist2).astype(np.float32)
                strength = float(rng.uniform(0.12, 0.30))
                stain_rgb = np.array([0.78, 0.62, 0.40], dtype=np.float32)
                work = work * (1.0 - strength * blob[:, :, None] * (1.0 - stain_rgb[None, None, :]))

        # Mild defocus blur.
        if rng.random() < 0.5:
            pil_tmp = Image.fromarray(np.clip(work * 255.0, 0, 255).astype(np.uint8), mode="RGB")
            pil_tmp = pil_tmp.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 1.1))))
            work = np.asarray(pil_tmp, dtype=np.float32) / 255.0

        if rng.random() < 0.5:
            do_jpeg = True

    # 3. Composed single-pass geometric warp:
    #    S-curve (+-4% H) -> Perspective keystone -> +-12 deg rotation -> 0-25% desk border.
    use_sine = bool(rng.random() < 0.65)
    amp = float(rng.uniform(-0.04, 0.04) * h) if use_sine else 0.0
    cycles = float(rng.uniform(0.5, 1.25))
    phase = float(rng.uniform(0.0, 2.0 * math.pi))

    def _dy(x_val: np.ndarray) -> np.ndarray:
        if amp == 0.0:
            return np.zeros_like(x_val)
        return amp * np.sin(2.0 * math.pi * cycles * (x_val / max(float(w), 1.0)) + phase)

    # Perspective keystone homography H_persp.
    src_corners = np.array([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]], dtype=np.float64)
    if rng.random() < 0.65:
        kt_x = float(rng.uniform(-0.05, 0.05) * w)
        kb_x = float(rng.uniform(-0.05, 0.05) * w)
        kl_y = float(rng.uniform(-0.04, 0.04) * h)
        kr_y = float(rng.uniform(-0.04, 0.04) * h)
        dst_corners = np.array(
            [
                [max(0.0, kt_x), max(0.0, kl_y)],
                [w - 1.0 - max(0.0, kt_x), max(0.0, kr_y)],
                [w - 1.0 - max(0.0, kb_x), h - 1.0 - max(0.0, kr_y)],
                [max(0.0, kb_x), h - 1.0 - max(0.0, kl_y)],
            ],
            dtype=np.float64,
        )
        H_persp = _solve_homography(src_corners, dst_corners)
    else:
        H_persp = np.eye(3, dtype=np.float64)

    # Rotation (+-12 deg) around page center.
    if rng.random() < 0.75:
        angle_rad = math.radians(float(rng.uniform(-12.0, 12.0)))
    else:
        angle_rad = 0.0
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    cx, cy = 0.5 * (w - 1.0), 0.5 * (h - 1.0)
    R_mat = np.array(
        [
            [ca, -sa, cx - ca * cx + sa * cy],
            [sa, ca, cy - sa * cx - ca * cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # 0-25% random outer desk border.
    if rng.random() < 0.70:
        b_left = float(rng.uniform(0.0, 0.25))
        b_right = float(rng.uniform(0.0, 0.25))
        b_top = float(rng.uniform(0.0, 0.25))
        b_bot = float(rng.uniform(0.0, 0.25))
    else:
        b_left = b_right = b_top = b_bot = 0.0

    sx_in = max(0.5, 1.0 - b_left - b_right)
    sy_in = max(0.5, 1.0 - b_top - b_bot)
    x_off = b_left * w
    y_off = b_top * h
    A_border = np.array(
        [
            [sx_in, 0.0, x_off],
            [0.0, sy_in, y_off],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    M_total = A_border @ R_mat @ H_persp
    M_inv = np.linalg.inv(M_total)

    # Single-pass backward sampling via F.grid_sample + desk compositing.
    gy_grid, gx_grid = np.meshgrid(
        np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij"
    )
    x1_g, y1_g = _apply_homography(M_inv, gx_grid, gy_grid)
    x0_g = x1_g
    y0_g = y1_g - _dy(x1_g)

    valid_mask = (x0_g >= 0.0) & (x0_g <= (w - 1.0)) & (y0_g >= 0.0) & (y0_g <= (h - 1.0))
    grid_x = (2.0 * x0_g / max(float(w - 1), 1.0) - 1.0).astype(np.float32)
    grid_y = (2.0 * y0_g / max(float(h - 1), 1.0) - 1.0).astype(np.float32)
    grid_t = torch.from_numpy(np.stack([grid_x, grid_y], axis=-1))[None, ...]  # [1, H, W, 2]
    src_t = torch.from_numpy(work.transpose(2, 0, 1))[None, ...]  # [1, 3, H, W]
    warped_t = F.grid_sample(src_t, grid_t, mode="bilinear", padding_mode="border", align_corners=True)
    warped = warped_t[0].numpy().transpose(1, 2, 0)

    if not np.all(valid_mask):
        desk = _make_desk_background(h, w, rng)
        out_img = np.where(valid_mask[:, :, None], warped, desk)
    else:
        out_img = warped

    # Optional JPEG round-trip (part of the 30% photometric block).
    if do_jpeg:
        buf = io.BytesIO()
        pil_j = Image.fromarray(np.clip(out_img * 255.0, 0, 255).astype(np.uint8), mode="RGB")
        pil_j.save(buf, format="JPEG", quality=int(rng.integers(38, 82)))
        buf.seek(0)
        out_img = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0

    # Transform all GT (5, N, 2) staff curves through the exact analytical forward map.
    new_entries: list[dict[str, Any]] = []
    old_to_new: dict[int, int] = {}
    for i, c in enumerate(curves):
        pts = c["pts"]  # (5, N, 2)
        px = pts[:, :, 0]
        py = pts[:, :, 1] + _dy(px)
        tx, ty = _apply_homography(M_total, px, py)

        # Keep columns where the staff centerline is inside the output frame.
        cx_line = tx[2]
        cy_line = ty[2]
        in_bounds = (cx_line >= 0.0) & (cx_line <= (w - 1.0)) & (cy_line >= 0.0) & (cy_line <= (h - 1.0))
        if int(in_bounds.sum()) < 2:
            continue

        order = np.argsort(cx_line[in_bounds])
        x_in = cx_line[in_bounds][order]
        x_start_n = float(x_in[0])
        x_end_n = float(x_in[-1])
        if (x_end_n - x_start_n) < 4.0:
            continue

        xs_uni = np.linspace(x_start_n, x_end_n, num=n_pts)
        lines_uni = np.empty((5, n_pts), dtype=np.float64)
        for k in range(5):
            # Re-interpolate each of the 5 warped lines at uniform page x.
            xk = tx[k][in_bounds]
            yk = ty[k][in_bounds]
            ok = np.argsort(xk)
            lines_uni[k] = np.interp(xs_uni, xk[ok], yk[ok])

        new_idx = len(new_entries)
        old_to_new[i] = new_idx
        entry_dict: dict[str, Any] = {
            "lines": lines_uni,
            "x0": x_start_n,
            "x1": x_end_n,
        }
        if c["sys_id"] is not None:
            entry_dict["system_id"] = c["sys_id"]
        new_entries.append(entry_dict)

    new_systems: list[list[int]] | None = None
    if systems is not None:
        new_systems = []
        for grp in systems:
            remapped = [old_to_new[i] for i in grp if i in old_to_new]
            if remapped:
                new_systems.append(remapped)

    return np.clip(out_img, 0.0, 1.0).astype(np.float32), new_entries, new_systems


# ---------------------------------------------------------------------------
# Dataset & minimal CLI training entrypoint
# ---------------------------------------------------------------------------


class StaffUNetDataset(Dataset):
    """Dataset reading JSONL rows of full-page images + staff/system annotations.

    Each JSONL line should be a dict with:
      - ``image`` (or ``path``): path to page PNG/JPEG
      - ``staves``: list of staff dicts / 5-line arrays
      - optional ``systems``: list of index lists grouping staves into systems

    Args:
        manifest_path: Path to JSONL manifest.
        target_h: Resized input height (default 1024).
        target_w: Resized input width (default 768).
        in_channels: 3 for RGB (default, preserving sepia/parchment/desk cues)
            or 1 for grayscale.
        out_stride: Target map spatial stride (default 2 -> 512x384 for 1024x768;
            4 -> 256x192).
        augment: Enable seeded joint image + ``(5, N)`` staff-curve augmentations.
        seed: Base RNG seed for deterministic augmentations.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        target_h: int = 1024,
        target_w: int = 768,
        in_channels: int = 3,
        out_stride: int = 2,
        augment: bool = False,
        seed: int = 42,
    ):
        self.manifest_path = Path(manifest_path)
        self.target_h = int(target_h)
        self.target_w = int(target_w)
        self.in_channels = int(in_channels)
        self.out_stride = int(out_stride)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        self.rows: list[dict[str, Any]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def set_epoch(self, epoch: int) -> None:
        """Advance the deterministic augmentation stream for a new training epoch."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        img_path = Path(row.get("image", row.get("path", "")))
        if not img_path.is_absolute():
            img_path = self.manifest_path.parent / img_path
        img = Image.open(img_path)
        orig_w, orig_h = img.size

        # Always resize with anti-aliased BOX (downscaling) / LANCZOS (upscaling).
        img_rgb = resize_page_image(img.convert("RGB"), target_w=self.target_w, target_h=self.target_h)
        arr_rgb = np.asarray(img_rgb, dtype=np.float32) / 255.0

        staves_raw = row.get("staves", [])
        systems_raw = row.get("systems", None)
        out_h = math.ceil(self.target_h / self.out_stride)
        out_w = math.ceil(self.target_w / self.out_stride)

        if self.augment:
            # Scale raw GT staves into resized (target_w, target_h) coordinates before joint warp.
            sx = float(self.target_w) / float(orig_w)
            sy = float(self.target_h) / float(orig_h)
            scaled_staves: list[dict[str, Any]] = []
            for entry in staves_raw:
                lines_5, x0, x1, sys_id = _parse_staff_entry(entry, orig_w)
                d: dict[str, Any] = {
                    "lines": lines_5 * sy,
                    "x0": x0 * sx,
                    "x1": x1 * sx,
                }
                if sys_id is not None:
                    d["system_id"] = sys_id
                scaled_staves.append(d)

            rng = np.random.default_rng((self.seed, self.epoch, int(idx)))
            arr_rgb, aug_staves, aug_systems = _augment_page_and_staves(
                arr_rgb, scaled_staves, systems_raw, rng
            )
            target = rasterize_staff_targets(
                aug_staves,
                w=self.target_w,
                h=self.target_h,
                out_h=out_h,
                out_w=out_w,
                systems=aug_systems,
            )
        else:
            target = rasterize_staff_targets(
                staves_raw,
                w=orig_w,
                h=orig_h,
                out_h=out_h,
                out_w=out_w,
                systems=systems_raw,
            )

        if self.in_channels == 1:
            gray = 0.299 * arr_rgb[:, :, 0] + 0.587 * arr_rgb[:, :, 1] + 0.114 * arr_rgb[:, :, 2]
            x = torch.from_numpy(gray[None, :, :].astype(np.float32))
        else:
            x = torch.from_numpy(arr_rgb.transpose(2, 0, 1).astype(np.float32))

        return x, target


def train_staff_unet2d(
    manifest: str | Path,
    out_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-3,
    target_h: int = 1024,
    target_w: int = 768,
    in_channels: int = 3,
    base_channels: int = 40,
    out_stride: int = 2,
    augment: bool = True,
    device: str | None = None,
    seed: int = 42,
    num_workers: int = 0,
) -> Path:
    """Minimal self-contained training loop for ``StaffUNet2D``."""
    torch.manual_seed(seed)
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ds = StaffUNetDataset(
        manifest,
        target_h=target_h,
        target_w=target_w,
        in_channels=in_channels,
        out_stride=out_stride,
        augment=augment,
        seed=seed,
    )
    use_pin = dev.type == "cuda"
    loader_kw: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": max(0, int(num_workers)),
        "drop_last": False,
        "pin_memory": use_pin,
    }
    if num_workers > 0:
        loader_kw["prefetch_factor"] = 2
    loader = DataLoader(ds, **loader_kw)

    model = StaffUNet2D(
        in_channels=in_channels, base_channels=base_channels, out_stride=out_stride
    ).to(dev)
    criterion = StaffUNetLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_path = out_path / "staff_unet2d.pt"
    n_batches = len(loader)
    for ep in range(1, epochs + 1):
        ds.set_epoch(ep - 1)
        model.train()
        t0 = time.perf_counter()
        running = 0.0
        steps = 0
        for x, y in loader:
            x = x.to(dev, non_blocking=use_pin)
            y = y.to(dev, non_blocking=use_pin)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(dev.type, enabled=use_amp):
                preds = model(x)
                loss, _ = criterion(preds, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach().item())
            steps += 1
            if steps % 100 == 0 or steps == 1:
                print(
                    f"  [ep {ep:02d}/{epochs:02d} step {steps:04d}/{n_batches:04d}] "
                    f"loss={running / steps:.4f} ({time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )
        sched.step()
        dt = time.perf_counter() - t0
        avg = running / max(1, steps)
        print(
            f"[epoch {ep:03d}/{epochs:03d}] loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e} ({dt:.1f}s)",
            flush=True,
        )
        payload = {
            "state_dict": model.state_dict(),
            "in_channels": in_channels,
            "base_channels": base_channels,
            "out_stride": out_stride,
            "target_h": target_h,
            "target_w": target_w,
            "epoch": ep,
            "loss": avg,
        }
        tmp_ckpt = out_path / "staff_unet2d.pt.tmp"
        torch.save(payload, tmp_ckpt)
        tmp_ckpt.replace(ckpt_path)

    return ckpt_path


def _main() -> None:
    p = argparse.ArgumentParser(description="Train lightweight 2D Staff+System UNet.")
    p.add_argument("--manifest", required=True, help="JSONL manifest of full-page images + staves")
    p.add_argument("--out-dir", required=True, help="Checkpoint output directory")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target-h", type=int, default=1024)
    p.add_argument("--target-w", type=int, default=768)
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--base-channels", type=int, default=40)
    p.add_argument("--out-stride", type=int, default=2, choices=[2, 4])
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()
    ckpt = train_staff_unet2d(
        manifest=args.manifest,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        target_h=args.target_h,
        target_w=args.target_w,
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        out_stride=args.out_stride,
        augment=not args.no_augment,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    print(f"Saved checkpoint: {ckpt}", flush=True)


if __name__ == "__main__":
    _main()
