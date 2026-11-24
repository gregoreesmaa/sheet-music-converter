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
import copy
import importlib
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
    "parse_unlabeled_dirs",
    "list_unlabeled_page_images",
    "unsup_epoch_round_due",
    "run_stage1_unsup_round",
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

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 40,
        out_stride: int = 2,
        row_ctx: bool = False,
    ):
        super().__init__()
        if out_stride not in (2, 4):
            raise ValueError(f"out_stride must be 2 or 4, got {out_stride}")
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.out_stride = int(out_stride)
        self.use_row_ctx = bool(row_ctx)

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
        self.bottleneck = _ConvBlock(c3, c4, dilation=(2, 4) if self.use_row_ctx else (1, 2))
        if self.use_row_ctx:
            # Global horizontal-mean row context (ONNX-clean ReduceMean + 3x1 vertical conv):
            # broadcasts left-margin barlines/brackets and inter-system gaps across the full page width.
            rc_out = nn.Conv2d(c4, c4, kernel_size=1)
            nn.init.zeros_(rc_out.weight)
            nn.init.zeros_(rc_out.bias)
            self.row_ctx = nn.Sequential(
                nn.Conv2d(c4, c4, kernel_size=(3, 1), padding=(1, 0), bias=False),
                nn.BatchNorm2d(c4),
                nn.ReLU(inplace=True),
                nn.Conv2d(c4, c4, kernel_size=(3, 1), padding=(2, 0), dilation=(2, 1), bias=False),
                nn.BatchNorm2d(c4),
                nn.ReLU(inplace=True),
                rc_out,
            )
        else:
            self.row_ctx = None

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
        if self.row_ctx is not None:
            b32 = b32 + self.row_ctx(b32.mean(dim=3, keepdim=True))

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

    def predict_probs(self, x: torch.Tensor, sp_mode: str = "rel_h") -> torch.Tensor:
        """Inference helper returning ``[B, 4, H/out_stride, W/out_stride]`` probs + scale map."""
        raw = self.forward(x)
        probs = torch.sigmoid(raw[:, :3])
        if sp_mode == "rel_h":
            sp_map = F.softplus(raw[:, 3:4])
        elif sp_mode == "log_hm":
            sp_map = raw[:, 3:4]
        else:
            raise ValueError(f"unsupported sp_mode: {sp_mode!r}")
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
        sp_mode: str = "rel_h",
    ):
        super().__init__()
        if sp_mode not in ("rel_h", "log_hm"):
            raise ValueError(f"unsupported sp_mode: {sp_mode!r}")
        self.w_lines = float(w_lines)
        self.w_center = float(w_center)
        self.w_system = float(w_system)
        self.w_sp = float(w_sp)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.sp_mode = str(sp_mode)

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        return_metrics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total loss and optional per-head scalar metrics.

        Args:
            preds: ``[B, 4, Hp, Wp]`` raw model output.
            targets: ``[B, 4, Hp, Wp]`` float targets in ``[0, 1]`` for ch0..2
                and ``sp_px / H`` for ch3.
            return_metrics: When ``False``, skip ``.item()`` host-syncs and
                return an empty metrics dict (fast path for training loop).
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

        # Mask sp loss to pixels on/near staves (clamp_min avoids forced .item() GPU sync).
        sp_tgt = targets[:, 3]
        if self.sp_mode == "log_hm":
            sp_pred = preds[:, 3]
            mask = ((targets[:, 0] > 0.3) | (targets[:, 1] > 0.3)).float()
            elem = F.smooth_l1_loss(sp_pred, sp_tgt, beta=0.15, reduction="none")
            l_sp = (elem * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            sp_pred = F.softplus(preds[:, 3])
            mask = ((targets[:, 0] > 0.25) | (targets[:, 1] > 0.25) | (sp_tgt > 1e-6)).float()
            elem = F.smooth_l1_loss(sp_pred, sp_tgt, beta=self.smooth_l1_beta, reduction="none")
            l_sp = (elem * mask).sum() / mask.sum().clamp_min(1.0)

        total = (
            self.w_lines * l_lines
            + self.w_center * l_center
            + self.w_system * l_system
            + self.w_sp * l_sp
        )
        if not return_metrics:
            return total, {}
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
    max_gap_px: float | None = None,
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
                if max_gap_px is not None and gap > max_gap_px:
                    continue
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
    sp_mode: str = "rel_h",
    iso_scale: float | None = None,
    out_stride: int = 2,
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

    if sp_mode not in ("rel_h", "log_hm"):
        raise ValueError(f"unsupported sp_mode: {sp_mode!r}")

    p_lines, p_center, p_sys, sp_rel = arr[0], arr[1], arr[2], arr[3]
    hp, wp = p_center.shape
    scale_x = float(orig_w) / float(wp)
    scale_y = float(orig_h) / float(hp)
    hm_scale = float(iso_scale) if iso_scale is not None else (float(hp * out_stride) / float(orig_h))

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

            # Sample sp map around the centerline in a +-2 px window.
            r0 = max(0, int(round(yc)) - 2)
            r1 = min(hp, int(round(yc)) + 3)
            sp_patch = sp_rel[r0:r1, col]
            sp_w = np.maximum(p_center[r0:r1, col] + p_lines[r0:r1, col], 1e-4)
            sp_raw = float((sp_patch * sp_w).sum() / sp_w.sum())
            if sp_mode == "log_hm":
                sp_val = math.exp(float(np.clip(sp_raw, -2.0, 6.0))) * float(out_stride) / hm_scale
            else:
                sp_val = sp_raw * float(orig_h)

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

    # Merge same-row horizontal fragments split by stains, creases, or folds,
    # capped at 0.18 * orig_w so 2:1 double-page gutters cannot bridge.
    merged_frags = _merge_horizontal_fragments(
        raw_frags, sp_tol_factor=0.5, max_gap_px=0.18 * float(orig_w)
    )

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
    sp_mode: str = "rel_h",
    iso_scale: float | None = None,
    out_stride: int = 2,
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
    if sp_mode not in ("rel_h", "log_hm"):
        raise ValueError(f"unsupported sp_mode: {sp_mode!r}")

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
    ch0_thresh_mult = math.sqrt(-2.0 * math.log(0.35))
    hm_scale = (
        float(iso_scale)
        if iso_scale is not None
        else (float(out_h * out_stride) / float(h))
    )

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
        c0 = int(act_cols[0])
        c1 = int(act_cols[-1]) + 1
        sp_o = sp_out_col[c0:c1]  # (M,)
        line_sigma = np.maximum(0.55, 0.22 * sp_o)
        line_thresh = line_sigma * ch0_thresh_mult
        c_sigma = np.maximum(0.75, 0.42 * sp_o)

        max_pad = max(
            float(np.max(sp_o)),
            float(np.max(line_thresh)),
            float(np.max(c_sigma)) * 2.5,
        ) + 2.0
        r_lo = max(0, int(math.floor(float(np.min(lines_out[:, c0:c1])) - max_pad)))
        r_hi = min(out_h, int(math.ceil(float(np.max(lines_out[:, c0:c1])) + max_pad)) + 1)
        if r_hi > r_lo:
            row_y_sub = row_y[r_lo:r_hi, None]  # (R, 1)

            # ch0: 5 staff lines (exact threshold equivalent to exp(-0.5*(dy/line_sigma)^2) >= 0.35)
            ch0_hit = np.zeros((r_hi - r_lo, c1 - c0), dtype=bool)
            for k in range(5):
                ch0_hit |= np.abs(row_y_sub - lines_out[k, None, c0:c1]) <= line_thresh[None, :]
            np.maximum(tgt[0, r_lo:r_hi, c0:c1], ch0_hit.astype(np.float32), out=tgt[0, r_lo:r_hi, c0:c1])

            # ch1: staff centerline (k=2) — smooth Gaussian for sub-pixel centroiding
            c_band = np.exp(
                -0.5 * ((row_y_sub - lines_out[2, None, c0:c1]) / c_sigma[None, :]) ** 2
            ).astype(np.float32)
            c_band[c_band < 0.05] = 0.0
            np.maximum(tgt[1, r_lo:r_hi, c0:c1], c_band, out=tgt[1, r_lo:r_hi, c0:c1])

            # ch3: local normal staff-space scale over staff span [line0 - sp, line4 + sp]
            y_lo = lines_out[0, c0:c1] - sp_o
            y_hi = lines_out[4, c0:c1] + sp_o
            smask = (row_y_sub >= y_lo[None, :]) & (row_y_sub <= y_hi[None, :])
            if sp_mode == "log_hm":
                sp_vals = np.log(
                    np.clip(sp_norm_col[c0:c1] * hm_scale / float(out_stride), 0.5, 256.0)
                ).astype(np.float32)
            else:
                sp_vals = (sp_norm_col[c0:c1] / float(h)).astype(np.float32)
            np.copyto(
                tgt[3, r_lo:r_hi, c0:c1],
                np.broadcast_to(sp_vals[None, :], smask.shape),
                where=smask,
            )

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
        union_cols = np.nonzero(act_union)[0]
        if union_cols.size == 0:
            continue
        c0 = int(union_cols[0])
        c1 = int(union_cols[-1]) + 1
        y0_col = np.full(c1 - c0, np.inf, dtype=np.float64)
        y1_col = np.full(c1 - c0, -np.inf, dtype=np.float64)
        for i in valid:
            lines_out, sp_vert_col, active, _ = parsed[i]
            act_sub = active[c0:c1]
            if not np.any(act_sub):
                continue
            sp_o_sub = np.maximum(sp_vert_col[c0:c1] * sy, 0.75)
            top_i = lines_out[0, c0:c1] - 0.5 * sp_o_sub
            bot_i = lines_out[4, c0:c1] + 0.5 * sp_o_sub
            y0_col = np.where(act_sub, np.minimum(y0_col, top_i), y0_col)
            y1_col = np.where(act_sub, np.maximum(y1_col, bot_i), y1_col)
        has_staff = np.isfinite(y0_col)
        if not np.any(has_staff):
            continue
        r_lo = max(0, int(math.floor(float(np.min(y0_col[has_staff])))) - 1)
        r_hi = min(out_h, int(math.ceil(float(np.max(y1_col[has_staff])))) + 2)
        if r_hi > r_lo:
            row_y_sub = row_y[r_lo:r_hi, None]
            mask = has_staff[None, :] & (row_y_sub >= y0_col[None, :]) & (row_y_sub <= y1_col[None, :])
            tgt[2, r_lo:r_hi, c0:c1] = np.where(mask, 1.0, tgt[2, r_lo:r_hi, c0:c1])

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
      - 20% left-margin barline masking on multi-staff systems (paint a narrow
        ``1-2*sp`` paper-colored vertical band between staves so ``ch2`` learns
        system grouping from layout/brackets/content rather than printed barlines)
      - 30% sepia/parchment tint + faded brown/gray ink + subtle bleed-through ghost staves
        + stain / blur / JPEG
      - Synthetic S-curve / sine vertical warp (``+-4% H``) + 35% multi-harmonic
        hand-drawn wavy staff wobble (``2-6`` cycles, ``+-0.15%-0.65% H``)
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
            paper_col = np.median(work[::8, ::8].reshape(-1, 3), axis=0).astype(np.float32)
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

    # 1b. Left-margin barline masking (20% per multi-staff system): paint a narrow
    #     1-2*sp paper-colored vertical band between staves over the left margin so
    #     ch2 learns system grouping from staff layout/brackets/content rather than
    #     overfitting to a crisp printed left barline (handwritten-safe).
    if curves and sys_groups:
        paper_col_bm: np.ndarray | None = None
        for grp in sys_groups:
            valid_grp = [i for i in grp if 0 <= i < len(curves)]
            if len(valid_grp) < 2 or rng.random() >= 0.05:
                continue
            if paper_col_bm is None:
                paper_col_bm = np.median(work[::8, ::8].reshape(-1, 3), axis=0).astype(np.float32)
            valid_grp.sort(key=lambda idx_s: float(curves[idx_s]["pts"][2, :, 1].mean()))
            band_mult = float(rng.uniform(1.0, 2.0))
            for top_i, bot_i in zip(valid_grp, valid_grp[1:]):
                pts_top = curves[top_i]["pts"]
                pts_bot = curves[bot_i]["pts"]
                sp_top = float(np.mean(np.diff(pts_top[:, :, 1], axis=0)))
                sp_bot = float(np.mean(np.diff(pts_bot[:, :, 1], axis=0)))
                sp_est = max(2.0, 0.5 * (sp_top + sp_bot))
                x_left = min(float(pts_top[2, 0, 0]), float(pts_bot[2, 0, 0]))
                half_w = 0.5 * band_mult * sp_est
                cx0 = max(0, int(math.floor(x_left - half_w)))
                cx1 = min(w, int(math.ceil(x_left + half_w)))
                y_lo = max(0, int(math.ceil(float(pts_top[4, 0, 1]) + 0.25 * sp_est)))
                y_hi = min(h, int(math.floor(float(pts_bot[0, 0, 1]) - 0.25 * sp_est)))
                if cx1 > cx0 and y_hi > y_lo:
                    work[y_lo:y_hi, cx0:cx1] = paper_col_bm[None, None, :]

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
    #    S-curve (+-4% H) + hand-drawn wavy wobble -> Perspective keystone -> +-12 deg rotation -> 0-25% desk border.
    use_sine = bool(rng.random() < 0.65)
    amp = float(rng.uniform(-0.04, 0.04) * h) if use_sine else 0.0
    cycles = float(rng.uniform(0.5, 1.25))
    phase = float(rng.uniform(0.0, 2.0 * math.pi))

    # Multi-harmonic wavy hand-drawn staff wobble (35%): 2-3 smooth local waves
    # across the page width so StaffUNet2D tracks hand-drawn wavy staves cleanly.
    wobble_terms: list[tuple[float, float, float]] = []
    if rng.random() < 0.35:
        n_harm = int(rng.integers(2, 4))
        for _ in range(n_harm):
            w_amp = float(rng.uniform(-0.0045, 0.0045) * h)
            w_cyc = float(rng.uniform(2.0, 6.0))
            w_phi = float(rng.uniform(0.0, 2.0 * math.pi))
            wobble_terms.append((w_amp, w_cyc, w_phi))

    def _dy(x_val: np.ndarray) -> np.ndarray:
        x_norm = x_val / max(float(w), 1.0)
        dy_val = (
            amp * np.sin(2.0 * math.pi * cycles * x_norm + phase)
            if amp != 0.0
            else np.zeros_like(x_val)
        )
        for w_amp, w_cyc, w_phi in wobble_terms:
            dy_val = dy_val + w_amp * np.sin(2.0 * math.pi * w_cyc * x_norm + w_phi)
        return dy_val

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


def _get_dynamic_res():
    """Lazy-import sibling ``dynamic_res`` module (safe for spawned workers)."""
    import sys as _sys

    mvd_dir = str(Path(__file__).resolve().parent)
    if mvd_dir not in _sys.path:
        _sys.path.insert(0, mvd_dir)
    import dynamic_res as _dyn_res

    return _dyn_res


_UNSUP_IMAGE_EXTS = frozenset((".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"))


def parse_unlabeled_dirs(raw: str | Sequence[str] | None) -> list[str]:
    """Normalize comma-separated or sequence ``--unlabeled-dirs`` values."""
    if not raw:
        return []
    items = [raw] if isinstance(raw, (str, Path)) else list(raw)
    out: list[str] = []
    for item in items:
        if not item:
            continue
        for part in str(item).split(","):
            p = part.strip()
            if p:
                out.append(p)
    return out


def list_unlabeled_page_images(dirs_raw: str | Sequence[str] | None) -> list[Path]:
    """Collect sorted unlabeled full-page image paths across comma-separated dirs."""
    seen: set[str] = set()
    out: list[Path] = []
    for d in parse_unlabeled_dirs(dirs_raw):
        root = Path(d)
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.name.startswith("."):
                continue
            if p.suffix.lower() not in _UNSUP_IMAGE_EXTS:
                continue
            sp = str(p)
            if sp not in seen:
                seen.add(sp)
                out.append(p)
    return out


def unsup_epoch_round_due(epoch: int, every_epochs: int) -> bool:
    """True when an unsupervised EMA round should run after supervised ``epoch``."""
    return int(every_epochs) > 0 and int(epoch) > 0 and (int(epoch) % int(every_epochs) == 0)


def _lazy_import_unsup_stage1():
    """Lazy-import optional sibling ``unsup_stage1`` module."""
    import sys as _sys

    mvd_dir = str(Path(__file__).resolve().parent)
    if mvd_dir not in _sys.path:
        _sys.path.insert(0, mvd_dir)
    try:
        return importlib.import_module("unsup_stage1")
    except ModuleNotFoundError:
        return None


def run_stage1_unsup_round(
    model: nn.Module,
    ema_model: nn.Module | None,
    opt: torch.optim.Optimizer,
    image_paths: Sequence[Path | str],
    steps: int = 200,
    ema_decay: float = 0.999,
    device: torch.device | str = "cpu",
    epoch: int = 1,
    seed: int = 42,
    batch_size: int = 4,
    target_h: int = 1024,
    target_w: int = 768,
    dynamic_res: bool = False,
    sp_mode: str = "rel_h",
    out_stride: int = 2,
    bf16: bool = False,
    unsup_stage1_mod: Any = None,
    unsup_round_fn: Any = None,
) -> dict[str, Any]:
    """Run one scheduled Stage-1 unsupervised EMA equivariance round.

    Uses ``unsup_round_fn`` if supplied (for tests/custom hooks), otherwise
    lazy-imports ``unsup_stage1`` and invokes its round/step entrypoint.
    Gracefully skips if ``unsup_stage1`` is not installed/present.
    """
    summary: dict[str, Any] = {
        "epoch": int(epoch),
        "steps": int(steps),
        "n_images": len(image_paths),
        "ema_decay": float(ema_decay),
        "ran": False,
        "metrics": {},
    }
    if int(steps) <= 0:
        return summary
    if not image_paths and unsup_round_fn is None:
        return summary

    kw = dict(
        model=model,
        ema_model=ema_model,
        opt=opt,
        image_paths=[Path(p) for p in image_paths],
        steps=int(steps),
        ema_decay=float(ema_decay),
        device=device,
        epoch=int(epoch),
        seed=int(seed) + int(epoch) * 1009,
        batch_size=int(batch_size),
        target_h=int(target_h),
        target_w=int(target_w),
        dynamic_res=bool(dynamic_res),
        sp_mode=str(sp_mode),
        out_stride=int(out_stride),
        bf16=bool(bf16),
    )

    if callable(unsup_round_fn):
        res = unsup_round_fn(**kw)
        summary["ran"] = True
        summary["metrics"] = res if isinstance(res, dict) else {"result": res}
        return summary

    mod = unsup_stage1_mod if unsup_stage1_mod is not None else _lazy_import_unsup_stage1()
    if mod is None:
        print(
            f"[unsup-stage1] module unsup_stage1 not found; skipping round at epoch {epoch}",
            flush=True,
        )
        return summary

    for fn_name in ("run_unsup_round", "run_stage1_unsup_round", "unsup_ema_round"):
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            res = fn(**kw)
            summary["ran"] = True
            summary["metrics"] = res if isinstance(res, dict) else {"result": res}
            return summary

    print(
        f"[unsup-stage1] module unsup_stage1 has no round entrypoint; skipping epoch {epoch}",
        flush=True,
    )
    return summary


def _worker_init_fn(_worker_id: int) -> None:
    """Keep each Windows-spawned DataLoader worker single-threaded to avoid CPU thrashing."""
    torch.set_num_threads(1)


class StaffUNetDataset(Dataset):
    """Dataset reading JSONL rows of full-page images + staff/system annotations.

    Each JSONL line should be a dict with:
      - ``image`` (or ``path``): path to page PNG/JPEG
      - ``staves``: list of staff dicts / 5-line arrays
      - optional ``systems``: list of index lists grouping staves into systems

    Args:
        manifest_path: Path to JSONL manifest.
        target_h: Resized input height (default 1024; ignored when ``dynamic_res=True``).
        target_w: Resized input width (default 768; ignored when ``dynamic_res=True``).
        in_channels: 3 for RGB (default, preserving sepia/parchment/desk cues)
            or 1 for grayscale.
        out_stride: Target map spatial stride (default 2 -> 512x384 for 1024x768;
            4 -> 256x192).
        augment: Enable seeded joint image + ``(5, N)`` staff-curve augmentations.
        seed: Base RNG seed for deterministic augmentations.
        dynamic_res: If True, isotropically resize + right/bottom-pad each page into
            its nearest discrete stride-32 aspect-ratio bucket.
        sp_mode: ``"rel_h"`` (default, legacy ``sp_normal / H``) or ``"log_hm"``
            (scale-invariant ``log(sp_hm)``).
        buckets: Optional sequence of ``(H, W)`` buckets for ``dynamic_res=True``
            (defaults to ``dynamic_res.DISCRETE_AR_BUCKETS``).
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
        dynamic_res: bool = False,
        sp_mode: str = "rel_h",
        buckets: Sequence[tuple[int, int]] | None = None,
        hw_hard_neg_prob: float = 0.0,
        clean_frac: float = 0.30,
    ):
        if sp_mode not in ("rel_h", "log_hm"):
            raise ValueError(f"unsupported sp_mode: {sp_mode!r}")
        self.manifest_path = Path(manifest_path)
        self.target_h = int(target_h)
        self.target_w = int(target_w)
        self.in_channels = int(in_channels)
        self.out_stride = int(out_stride)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.dynamic_res = bool(dynamic_res)
        self.sp_mode = str(sp_mode)
        self.hw_hard_neg_prob = max(0.0, min(1.0, float(hw_hard_neg_prob)))
        self.clean_frac = max(0.0, min(1.0, float(clean_frac)))
        self.epoch = 0
        self.rows: list[dict[str, Any]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.buckets: list[tuple[int, int]] | None = None
        self.sizes: list[tuple[int, int]] | None = None
        if self.dynamic_res:
            dyn_res = _get_dynamic_res()
            self.buckets = (
                [(int(bh), int(bw)) for bh, bw in buckets]
                if buckets is not None
                else list(dyn_res.DISCRETE_AR_BUCKETS)
            )
            sizes: list[tuple[int, int]] = []
            for row in self.rows:
                rw = row.get("width", row.get("w"))
                rh = row.get("height", row.get("h"))
                if rw is not None and rh is not None:
                    sizes.append((int(rw), int(rh)))
                else:
                    p = Path(row.get("image", row.get("path", "")))
                    if not p.is_absolute():
                        p = self.manifest_path.parent / p
                    with Image.open(p) as im:
                        sizes.append((int(im.size[0]), int(im.size[1])))
            self.sizes = sizes

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

        staves_raw = row.get("staves", [])
        systems_raw = row.get("systems", None)
        use_dynamic_res = bool(getattr(self, "dynamic_res", False))
        sp_mode = str(getattr(self, "sp_mode", "rel_h"))

        if use_dynamic_res:
            dyn_res = _get_dynamic_res()
            buckets = getattr(self, "buckets", None) or dyn_res.DISCRETE_AR_BUCKETS
            bh, bw = dyn_res.choose_ar_bucket(orig_w, orig_h, buckets=buckets)
            scale = min(float(bw) / float(orig_w), float(bh) / float(orig_h))
            scaled_w = max(1, min(bw, int(round(orig_w * scale))))
            scaled_h = max(1, min(bh, int(round(orig_h * scale))))
            img_small = resize_page_image(img.convert("RGB"), target_w=scaled_w, target_h=scaled_h)
            arr_small = np.asarray(img_small, dtype=np.float32) / 255.0
            if bh > scaled_h or bw > scaled_w:
                fill_rgb = np.asarray(dyn_res.estimate_border_color(arr_small), dtype=np.float32)
                arr_rgb = np.empty((bh, bw, 3), dtype=np.float32)
                arr_rgb[:, :, :] = fill_rgb[None, None, :]
                arr_rgb[:scaled_h, :scaled_w, :] = arr_small
            else:
                arr_rgb = arr_small

            sx = float(scaled_w) / float(orig_w)
            sy = float(scaled_h) / float(orig_h)
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

            out_h = math.ceil(bh / self.out_stride)
            out_w = math.ceil(bw / self.out_stride)
            clean_frac = float(getattr(self, "clean_frac", 0.30))
            if self.augment:
                rng = np.random.default_rng((self.seed, self.epoch, int(idx)))
                is_clean = (clean_frac > 0.0) and (float(rng.random()) < clean_frac)
                if is_clean:
                    target = rasterize_staff_targets(
                        scaled_staves,
                        w=bw,
                        h=bh,
                        out_h=out_h,
                        out_w=out_w,
                        systems=systems_raw,
                        sp_mode=sp_mode,
                        iso_scale=1.0,
                        out_stride=self.out_stride,
                    )
                else:
                    arr_rgb, aug_staves, aug_systems = _augment_page_and_staves(
                        arr_rgb, scaled_staves, systems_raw, rng
                    )
                    target = rasterize_staff_targets(
                        aug_staves,
                        w=bw,
                        h=bh,
                        out_h=out_h,
                        out_w=out_w,
                        systems=aug_systems,
                        sp_mode=sp_mode,
                        iso_scale=1.0,
                        out_stride=self.out_stride,
                    )
            else:
                target = rasterize_staff_targets(
                    scaled_staves,
                    w=bw,
                    h=bh,
                    out_h=out_h,
                    out_w=out_w,
                    systems=systems_raw,
                    sp_mode=sp_mode,
                    iso_scale=1.0,
                    out_stride=self.out_stride,
                )
        else:
            # Always resize with anti-aliased BOX (downscaling) / LANCZOS (upscaling).
            img_rgb = resize_page_image(img.convert("RGB"), target_w=self.target_w, target_h=self.target_h)
            arr_rgb = np.asarray(img_rgb, dtype=np.float32) / 255.0

            out_h = math.ceil(self.target_h / self.out_stride)
            out_w = math.ceil(self.target_w / self.out_stride)
            clean_frac = float(getattr(self, "clean_frac", 0.30))

            if self.augment:
                rng = np.random.default_rng((self.seed, self.epoch, int(idx)))
                is_clean = (clean_frac > 0.0) and (float(rng.random()) < clean_frac)
                if is_clean:
                    target = rasterize_staff_targets(
                        staves_raw,
                        w=orig_w,
                        h=orig_h,
                        out_h=out_h,
                        out_w=out_w,
                        systems=systems_raw,
                        sp_mode=sp_mode,
                        out_stride=self.out_stride,
                    )
                else:
                    # Scale raw GT staves into resized (target_w, target_h) coordinates before joint warp.
                    sx = float(self.target_w) / float(orig_w)
                    sy = float(self.target_h) / float(orig_h)
                    scaled_staves = []
                    for entry in staves_raw:
                        lines_5, x0, x1, sys_id = _parse_staff_entry(entry, orig_w)
                        d = {
                            "lines": lines_5 * sy,
                            "x0": x0 * sx,
                            "x1": x1 * sx,
                        }
                        if sys_id is not None:
                            d["system_id"] = sys_id
                        scaled_staves.append(d)

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
                        sp_mode=sp_mode,
                        iso_scale=(1.0 if sp_mode == "log_hm" else None),
                        out_stride=self.out_stride,
                    )
            else:
                target = rasterize_staff_targets(
                    staves_raw,
                    w=orig_w,
                    h=orig_h,
                    out_h=out_h,
                    out_w=out_w,
                    systems=systems_raw,
                    sp_mode=sp_mode,
                    out_stride=self.out_stride,
                )

        hw_hn_p = float(getattr(self, "hw_hard_neg_prob", 0.0))
        if self.augment and hw_hn_p > 0.0:
            u1 = _lazy_import_unsup_stage1()
            hn_fn = getattr(u1, "augment_hw_hard_negatives", None) if u1 is not None else None
            if callable(hn_fn):
                rng_hn = np.random.default_rng((self.seed, self.epoch, int(idx), 9173))
                if float(rng_hn.random()) < hw_hn_p:
                    res_hn = hn_fn(arr_rgb, target, rng=rng_hn)
                    if isinstance(res_hn, tuple) and len(res_hn) == 2:
                        arr_rgb, target = res_hn
                    else:
                        arr_rgb = res_hn

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
    bf16: bool = False,
    dynamic_res: bool = False,
    pixel_budget: int = 1536 * 1152,
    max_side: int = 2048,
    sp_mode: str = "rel_h",
    init_from: str | Path | None = None,
    buckets: Sequence[tuple[int, int]] | None = None,
    row_ctx: bool = False,
    unlabeled_dirs: str | Sequence[str] | None = None,
    unsup_round_every: int = 0,
    unsup_round_steps: int = 200,
    unsup_ema_decay: float = 0.999,
    hw_hard_neg_prob: float = 0.0,
    unsup_round_fn: Any = None,
    clean_frac: float = 0.30,
) -> Path:
    """Minimal self-contained training loop for ``StaffUNet2D``."""
    if sp_mode not in ("rel_h", "log_hm"):
        raise ValueError(f"unsupported sp_mode: {sp_mode!r}")
    torch.manual_seed(seed)
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_cuda = dev.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
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
        dynamic_res=dynamic_res,
        sp_mode=sp_mode,
        buckets=buckets,
        hw_hard_neg_prob=hw_hard_neg_prob,
        clean_frac=clean_frac,
    )
    unlabeled_images = list_unlabeled_page_images(unlabeled_dirs) if unlabeled_dirs else []
    ema_model: nn.Module | None = None
    use_pin = use_cuda
    batch_sampler = None
    if dynamic_res:
        dyn_res = _get_dynamic_res()
        batch_sampler = dyn_res.AspectBucketBatchSampler(
            ds.sizes,
            batch_size=batch_size,
            buckets=ds.buckets,
            drop_last=False,
            seed=seed,
            shuffle=True,
        )
        loader_kw: dict[str, Any] = {
            "batch_sampler": batch_sampler,
            "num_workers": max(0, int(num_workers)),
            "pin_memory": use_pin,
        }
    else:
        loader_kw = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": max(0, int(num_workers)),
            "drop_last": False,
            "pin_memory": use_pin,
        }
    if num_workers > 0:
        loader_kw["worker_init_fn"] = _worker_init_fn
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 4
    loader = DataLoader(ds, **loader_kw)

    model = StaffUNet2D(
        in_channels=in_channels, base_channels=base_channels,
        out_stride=out_stride, row_ctx=row_ctx,
    ).to(dev)
    if init_from is not None:
        init_state = torch.load(str(init_from), map_location="cpu", weights_only=True)
        if isinstance(init_state, dict) and ("state_dict" in init_state or "model" in init_state):
            init_sd = init_state.get("state_dict", init_state.get("model"))
            init_sp_mode = str(init_state.get("sp_mode", "rel_h"))
        else:
            init_sd = init_state
            init_sp_mode = "rel_h"
        model.load_state_dict(init_sd, strict=True)
        if init_sp_mode != sp_mode:
            with torch.no_grad():
                nn.init.normal_(model.head.weight[3], mean=0.0, std=0.01)
                if model.head.bias is not None:
                    model.head.bias[3].fill_(1.5 if sp_mode == "log_hm" else 0.0)
    if use_cuda:
        model = model.to(memory_format=torch.channels_last)

    criterion = StaffUNetLoss(
        w_sp=(1.0 if sp_mode == "log_hm" else 10.0),
        sp_mode=sp_mode,
    )
    if use_cuda:
        try:
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, fused=True)
        except (TypeError, RuntimeError):
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    use_amp = use_cuda
    amp_dtype = torch.bfloat16 if (use_amp and bf16) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and not bf16))

    ckpt_path = out_path / "staff_unet2d.pt"
    n_batches = len(loader)
    for ep in range(1, epochs + 1):
        ds.set_epoch(ep - 1)
        if batch_sampler is not None:
            batch_sampler.set_epoch(ep - 1)
        model.train()
        t0 = time.perf_counter()
        running_loss_t = torch.zeros((), device=dev, dtype=torch.float32)
        steps = 0
        for x, y in loader:
            if use_cuda:
                x = x.to(dev, non_blocking=use_pin, memory_format=torch.channels_last)
            else:
                x = x.to(dev, non_blocking=use_pin)
            y = y.to(dev, non_blocking=use_pin)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(dev.type, dtype=amp_dtype, enabled=use_amp):
                preds = model(x)
                loss, _ = criterion(preds, y, return_metrics=False)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running_loss_t += loss.detach().float()
            steps += 1
            if steps % 100 == 0 or steps == 1:
                print(
                    f"  [ep {ep:02d}/{epochs:02d} step {steps:04d}/{n_batches:04d}] "
                    f"loss={float(running_loss_t.item()) / steps:.4f} ({time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )
        sched.step()
        dt = time.perf_counter() - t0
        avg = float(running_loss_t.item()) / max(1, steps)
        print(
            f"[epoch {ep:03d}/{epochs:03d}] loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e} ({dt:.1f}s)",
            flush=True,
        )
        if unsup_epoch_round_due(ep, unsup_round_every) and (
            unlabeled_images or unsup_round_fn is not None
        ):
            if ema_model is None:
                ema_model = copy.deepcopy(model).to(dev)
                ema_model.eval()
                for p_ema in ema_model.parameters():
                    p_ema.requires_grad_(False)
            u_res = run_stage1_unsup_round(
                model=model,
                ema_model=ema_model,
                opt=opt,
                image_paths=unlabeled_images,
                steps=unsup_round_steps,
                ema_decay=unsup_ema_decay,
                device=dev,
                epoch=ep,
                seed=seed,
                batch_size=batch_size,
                target_h=target_h,
                target_w=target_w,
                dynamic_res=dynamic_res,
                sp_mode=sp_mode,
                out_stride=out_stride,
                bf16=bf16,
                unsup_round_fn=unsup_round_fn,
            )
            print(
                f"  [unsup-stage1 ep {ep:03d}] ran={u_res['ran']} "
                f"steps={u_res['steps']} images={u_res['n_images']}",
                flush=True,
            )
        payload = {
            "state_dict": model.state_dict(),
            "in_channels": in_channels,
            "base_channels": base_channels,
            "out_stride": out_stride,
            "target_h": target_h,
            "target_w": target_w,
            "dynamic_res": bool(dynamic_res),
            "pixel_budget": int(pixel_budget),
            "max_side": int(max_side),
            "sp_mode": str(sp_mode),
            "row_ctx": bool(row_ctx),
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
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 AMP on CUDA (no GradScaler)")
    p.add_argument("--dynamic-res", action="store_true", help="Enable aspect-bucketed dynamic-resolution training")
    p.add_argument("--pixel-budget", type=int, default=1536 * 1152, help="Dynamic-res inference pixel budget")
    p.add_argument("--max-side", type=int, default=2048, help="Dynamic-res max side length")
    p.add_argument("--sp-mode", default="rel_h", choices=["rel_h", "log_hm"], help="ch3 staff-space parameterization")
    p.add_argument("--init-from", default=None, help="Optional checkpoint path to warm-start from")
    p.add_argument("--row-ctx", action="store_true", help="Enable global horizontal row-context + (2,4) bottleneck dilation")
    p.add_argument(
        "--unlabeled-dirs",
        default=None,
        help="Comma-separated dirs of unlabeled full pages (e.g. printed, RISM HW, intl HW)",
    )
    p.add_argument(
        "--unsup-round-every",
        type=int,
        default=0,
        help="Run an unsupervised EMA equivariance round every N supervised epochs (0 = off)",
    )
    p.add_argument(
        "--unsup-round-steps",
        type=int,
        default=200,
        help="Number of steps per unsupervised round (default 200)",
    )
    p.add_argument(
        "--unsup-ema-decay",
        type=float,
        default=0.999,
        help="EMA teacher decay for unsupervised rounds (default 0.999)",
    )
    p.add_argument(
        "--hw-hard-neg-prob",
        type=float,
        default=0.0,
        help="Probability of synthetic blank pre-ruled staves & cursive lyric hard negatives via unsup_stage1 (0.0 = off)",
    )
    p.add_argument(
        "--clean-frac",
        type=float,
        default=0.30,
        help="Fraction of augmented samples kept clean (skipping heavy photo degradation to preserve crisp 1px braces)",
    )
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
        bf16=args.bf16,
        dynamic_res=args.dynamic_res,
        pixel_budget=args.pixel_budget,
        max_side=args.max_side,
        sp_mode=args.sp_mode,
        init_from=args.init_from,
        row_ctx=args.row_ctx,
        unlabeled_dirs=args.unlabeled_dirs,
        unsup_round_every=args.unsup_round_every,
        unsup_round_steps=args.unsup_round_steps,
        unsup_ema_decay=args.unsup_ema_decay,
        hw_hard_neg_prob=args.hw_hard_neg_prob,
        clean_frac=args.clean_frac,
    )
    print(f"Saved checkpoint: {ckpt}", flush=True)


if __name__ == "__main__":
    _main()
