"""Zero-corruption unsupervised learning primitives for Stage-1 StaffUNet2D.

Implements:
  - :class:`EMAModel`: exponential moving average frozen teacher wrapper.
  - :func:`stage1_equivariance_loss`: confidence-gated geometric + photometric
    consistency on ``ch0..2`` and exact scale equivariance
    ``log_hm(Scale_s(I)) == log_hm(I) + log(s)`` on ``ch3``, with exact
    out-of-bounds masking and optional horizontal-flip consistency on ``ch2``.
  - :func:`augment_hw_hard_negatives`: targeted handwritten hard-negative
    augmentation injecting (1) faint blank pre-ruled 5-line manuscript staves
    and (2) dense multi-line cursive/lyric text blocks into empty margins and
    inter-system gaps while keeping GT ``ch0=0, ch1=0, ch2=0``.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "EMAModel",
    "build_equivariant_affine_theta",
    "stage1_equivariance_loss",
    "augment_hw_hard_negatives",
]


class EMAModel:
    """Frozen Exponential Moving Average (EMA) teacher wrapper.

    Maintains a deep-copied, eval-mode, gradient-disabled copy of ``student``.
    """

    def __init__(self, student: nn.Module, decay: float = 0.999) -> None:
        if not (0.0 <= float(decay) <= 1.0):
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        self.decay = float(decay)
        self.module = copy.deepcopy(student)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @property
    def teacher(self) -> nn.Module:
        return self.module

    @torch.no_grad()
    def update(self, student: nn.Module, decay: float | None = None) -> None:
        """Update EMA parameters: ``ema = d * ema + (1 - d) * student``."""
        d = self.decay if decay is None else float(decay)
        if not (0.0 <= d <= 1.0):
            raise ValueError(f"decay must be in [0, 1], got {d}")
        for p_ema, p_stu in zip(self.module.parameters(), student.parameters()):
            src = p_stu.detach().to(device=p_ema.device, dtype=p_ema.dtype)
            if p_ema.is_floating_point():
                p_ema.mul_(d).add_(src, alpha=1.0 - d)
            else:
                p_ema.copy_(src)
        for b_ema, b_stu in zip(self.module.buffers(), student.buffers()):
            b_ema.copy_(b_stu.detach().to(device=b_ema.device, dtype=b_ema.dtype))
        self.module.eval()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def to(self, *args: Any, **kwargs: Any) -> "EMAModel":
        self.module.to(*args, **kwargs)
        return self

    def eval(self) -> "EMAModel":
        self.module.eval()
        return self

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> Any:
        res = self.module.load_state_dict(state_dict, strict=strict)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        return res


def build_equivariant_affine_theta(
    scales: torch.Tensor,
    angles_deg: torch.Tensor,
    tx_norm: torch.Tensor,
    ty_norm: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Build aspect-correct inverse affine matrix ``[B, 2, 3]`` for ``F.affine_grid``.

    Models the exact pixel-space forward similarity transform centered at the
    image midpoint:
      ``dst_px = s * R(angle) * src_px + t_px``
    where ``s > 1`` zooms in (increasing staff spacing ``sp`` by factor ``s``).
    Converting to normalized ``[-1, 1]`` coordinates (``align_corners=False``)
    yields the exact backward sampling matrix for ``F.affine_grid``.
    """
    device = scales.device
    dtype = scales.dtype
    inv_s = 1.0 / scales.clamp_min(1e-4)
    rad = angles_deg * (math.pi / 180.0)
    cos_a = torch.cos(rad)
    sin_a = torch.sin(rad)

    aspect = float(height) / max(float(width), 1.0)
    inv_aspect = float(width) / max(float(height), 1.0)

    a00 = cos_a * inv_s
    a01 = aspect * sin_a * inv_s
    a02 = -(a00 * tx_norm + a01 * ty_norm)

    a10 = -inv_aspect * sin_a * inv_s
    a11 = cos_a * inv_s
    a12 = -(a10 * tx_norm + a11 * ty_norm)

    row0 = torch.stack([a00, a01, a02], dim=-1)
    row1 = torch.stack([a10, a11, a12], dim=-1)
    return torch.stack([row0, row1], dim=1).to(device=device, dtype=dtype)


def _photometric_jitter(
    img: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Lightweight differentiable-input photometric jitter on ``[B, C, H, W]`` in ``[0, 1]``."""
    b, c, _, _ = img.shape
    dev = img.device
    dtype = img.dtype
    out = img.float()

    # Brightness + contrast + gamma + mild sensor noise.
    bright = torch.empty((b, 1, 1, 1), device=dev).uniform_(-0.08, 0.08, generator=generator)
    contr = torch.empty((b, 1, 1, 1), device=dev).uniform_(0.85, 1.15, generator=generator)
    gamma = torch.empty((b, 1, 1, 1), device=dev).uniform_(0.85, 1.18, generator=generator)
    noise_std = torch.empty((b, 1, 1, 1), device=dev).uniform_(0.0, 0.025, generator=generator)

    mean = out.mean(dim=(2, 3), keepdim=True)
    out = ((out - mean) * contr + mean + bright).clamp(1e-4, 1.0)
    out = out.pow(gamma)
    if c == 3:
        wb = torch.empty((b, 3, 1, 1), device=dev).uniform_(0.94, 1.06, generator=generator)
        out = out * wb
    noise = torch.empty_like(out).normal_(0.0, 1.0, generator=generator)
    out = (out + noise * noise_std).clamp(0.0, 1.0)
    return out.to(dtype=dtype)


def _unwrap_teacher(teacher: nn.Module | EMAModel) -> nn.Module:
    return teacher.module if isinstance(teacher, EMAModel) else teacher


def stage1_equivariance_loss(
    student: nn.Module,
    teacher: nn.Module | EMAModel,
    img_batch: torch.Tensor,
    sp_mode: str = "log_hm",
    conf_thresh: float = 0.85,
    scale_range: tuple[float, float] = (0.80, 1.25),
    rot_deg: float = 4.0,
    trans_frac: float = 0.05,
    photo_jitter: bool = True,
    hflip_weight: float = 0.0,
    w_ch02: float = 1.0,
    w_sp: float = 1.0,
    warp_params: dict[str, torch.Tensor | float] | None = None,
    generator: torch.Generator | None = None,
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Compute Stage-1 geometric + scale-equivariance unsupervised loss.

    1. Runs frozen EMA ``teacher`` under ``torch.no_grad()`` on clean unlabeled
       ``img_batch`` to obtain ``T_probs`` (``ch0..2`` sigmoid, ``ch3`` scale).
    2. Samples (or uses ``warp_params``) a known similarity warp ``W``
       (isotropic scale ``s in scale_range``, rotation ``+-rot_deg``,
       translation ``+-trans_frac``) plus optional photometric jitter, and runs
       ``student`` on ``W(I)``.
    3. Warps ``T_probs`` through the exact same geometric transform ``W`` with
       strict out-of-bounds mask ``valid_mask``.
    4. Enforces:
       - Confidence-gated (``max(p, 1-p) > conf_thresh``) MSE consistency on
         ``ch0..2`` within ``valid_mask``.
       - Exact scale equivariance on ``ch3``:
         ``log_hm(Scale_s(I)) == log_hm(I) + log(s)`` (for ``sp_mode="log_hm"``)
         or ``rel_h(Scale_s(I)) == rel_h(I) * s`` (for ``sp_mode="rel_h"``),
         masked to ``(W(T_ch0) > 0.5) & valid_mask``.
       - Optional horizontal-flip consistency on ``ch2`` when ``hflip_weight > 0``.
    """
    if sp_mode not in ("log_hm", "rel_h"):
        raise ValueError(f"unsupported sp_mode: {sp_mode!r}")
    if img_batch.ndim != 4:
        raise ValueError(f"expected [B, C, H, W] img_batch, got {tuple(img_batch.shape)}")

    b, _, h, w = img_batch.shape
    dev = img_batch.device

    # 1. Frozen teacher prediction on clean unlabeled page I.
    t_net = _unwrap_teacher(teacher)
    t_was_training = t_net.training
    t_net.eval()
    try:
        with torch.no_grad():
            t_raw = t_net(img_batch)
            t_probs_02 = torch.sigmoid(t_raw[:, :3]).float()
            t_sp = t_raw[:, 3:4].float() if sp_mode == "log_hm" else F.softplus(t_raw[:, 3:4]).float()
            t_all = torch.cat([t_probs_02, t_sp], dim=1)
    finally:
        t_net.train(t_was_training)

    _, _, hp, wp = t_all.shape

    # 2. Sample or unpack similarity warp parameters W.
    def _param(name: str, lo: float, hi: float) -> torch.Tensor:
        if warp_params is not None and name in warp_params:
            val = warp_params[name]
            if isinstance(val, torch.Tensor):
                return val.to(device=dev, dtype=torch.float32).reshape(b)
            return torch.full((b,), float(val), device=dev, dtype=torch.float32)
        return torch.empty((b,), device=dev, dtype=torch.float32).uniform_(lo, hi, generator=generator)

    scales = _param("scales", float(scale_range[0]), float(scale_range[1]))
    angles = _param("angles_deg", -float(rot_deg), float(rot_deg))
    tx = _param("tx", -float(trans_frac), float(trans_frac))
    ty = _param("ty", -float(trans_frac), float(trans_frac))

    theta_img = build_equivariant_affine_theta(scales, angles, tx, ty, height=h, width=w)
    grid_img = F.affine_grid(theta_img, size=(b, img_batch.shape[1], h, w), align_corners=False)

    warped_img = F.grid_sample(
        img_batch.float(),
        grid_img,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).to(dtype=img_batch.dtype)
    if photo_jitter:
        warped_img = _photometric_jitter(warped_img, generator=generator)

    # 3. Student forward on W(I).
    s_raw = student(warped_img)
    s_probs_02 = torch.sigmoid(s_raw[:, :3]).float()
    s_sp = s_raw[:, 3:4].float() if sp_mode == "log_hm" else F.softplus(s_raw[:, 3:4]).float()

    # 4. Warp teacher predictions W(T_probs) + exact OOB valid_mask.
    theta_hm = build_equivariant_affine_theta(scales, angles, tx, ty, height=hp, width=wp)
    grid_hm = F.affine_grid(theta_hm, size=(b, 4, hp, wp), align_corners=False)
    w_t_all = F.grid_sample(
        t_all,
        grid_hm,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    w_t_02 = w_t_all[:, :3].clamp(0.0, 1.0)
    w_t_sp = w_t_all[:, 3:4]

    # Strict interior-bounds check avoids any border-padding leakage.
    eps_x = 1.0 / max(float(wp), 1.0)
    eps_y = 1.0 / max(float(hp), 1.0)
    valid_mask = (
        (grid_hm[..., 0].abs() <= (1.0 - eps_x)) & (grid_hm[..., 1].abs() <= (1.0 - eps_y))
    ).float().unsqueeze(1)  # [B, 1, Hp, Wp]

    # Confidence-gated consistency on ch0..2.
    conf_02 = torch.maximum(w_t_02, 1.0 - w_t_02)
    mask_02 = (conf_02 > float(conf_thresh)).float() * valid_mask
    denom_02 = mask_02.sum()
    if float(denom_02.item()) > 0.0:
        loss_02 = ((s_probs_02 - w_t_02).pow(2) * mask_02).sum() / denom_02
    else:
        loss_02 = s_raw.sum() * 0.0

    # Exact scale equivariance on ch3: log_hm(Scale_s(I)) == log_hm(I) + log(s).
    if sp_mode == "log_hm":
        target_sp = w_t_sp + torch.log(scales.clamp_min(1e-4)).view(b, 1, 1, 1)
        sp_beta = 0.15
    else:
        target_sp = w_t_sp * scales.view(b, 1, 1, 1)
        sp_beta = 0.005

    sp_mask = (w_t_02[:, 0:1] > 0.5).float() * valid_mask
    denom_sp = sp_mask.sum()
    if float(denom_sp.item()) > 0.0:
        sp_elem = F.smooth_l1_loss(s_sp, target_sp, beta=sp_beta, reduction="none")
        loss_sp = (sp_elem * sp_mask).sum() / denom_sp
    else:
        loss_sp = s_raw.sum() * 0.0

    # Optional horizontal-flip consistency on ch2 (system span).
    loss_hflip = s_raw.sum() * 0.0
    if float(hflip_weight) > 0.0:
        img_flip = torch.flip(img_batch, dims=[3])
        if photo_jitter:
            img_flip = _photometric_jitter(img_flip, generator=generator)
        s_flip_raw = student(img_flip)
        s_flip_ch2 = torch.flip(torch.sigmoid(s_flip_raw[:, 2:3]).float(), dims=[3])
        t_ch2 = t_probs_02[:, 2:3]
        conf_ch2 = (torch.maximum(t_ch2, 1.0 - t_ch2) > float(conf_thresh)).float()
        denom_hf = conf_ch2.sum()
        if float(denom_hf.item()) > 0.0:
            loss_hflip = ((s_flip_ch2 - t_ch2).pow(2) * conf_ch2).sum() / denom_hf
        else:
            loss_hflip = s_flip_raw.sum() * 0.0

    total = float(w_ch02) * loss_02 + float(w_sp) * loss_sp + float(hflip_weight) * loss_hflip

    if return_metrics:
        metrics = {
            "unsup_s1_total": float(total.detach().item()),
            "unsup_s1_ch02": float(loss_02.detach().item()),
            "unsup_s1_sp": float(loss_sp.detach().item()),
            "unsup_s1_hflip": float(loss_hflip.detach().item()),
            "unsup_s1_valid_frac": float(valid_mask.mean().item()),
            "unsup_s1_conf_frac": float(mask_02.mean().item()),
            "unsup_s1_sp_frac": float(sp_mask.mean().item()),
        }
        return total, metrics
    return total


# ---------------------------------------------------------------------------
# Targeted HW Stage-1 Hard-Negative Augmentations
# ---------------------------------------------------------------------------


def _find_blank_bands(row_occ: np.ndarray, min_height: int) -> list[tuple[int, int]]:
    """Return contiguous ``(y0, y1)`` vertical intervals where ``row_occ`` is False."""
    h = int(row_occ.shape[0])
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for y in range(h):
        if not row_occ[y]:
            if start is None:
                start = y
        elif start is not None:
            if y - start >= min_height:
                bands.append((start, y))
            start = None
    if start is not None and h - start >= min_height:
        bands.append((start, h))
    return bands


def _draw_blank_manuscript_staves(
    work: np.ndarray,
    occ_img: np.ndarray,
    bands: list[tuple[int, int]],
    sp_est: float,
    rng: np.random.Generator,
) -> None:
    """Draw faint empty 5-line pre-ruled manuscript staves into blank bands."""
    h, w, _ = work.shape
    min_band_h = max(16, int(math.ceil(4.5 * sp_est)))
    eligible = [b for b in bands if (b[1] - b[0]) >= min_band_h]
    if not eligible:
        return

    n_draw = int(rng.integers(1, min(3, len(eligible)) + 1))
    chosen_idx = rng.choice(len(eligible), size=n_draw, replace=False)
    xs = np.arange(w, dtype=np.float32)

    for idx in chosen_idx:
        y0, y1 = eligible[int(idx)]
        band_h = y1 - y0
        sp_max = max(3.5, min(20.0, (band_h - 6.0) / 4.4))
        sp_lo = min(sp_max, max(3.5, 0.70 * sp_est))
        sp_hi = max(sp_lo, min(sp_max, 1.25 * sp_est))
        sp = float(rng.uniform(sp_lo, sp_hi))

        staff_h = 4.0 * sp
        margin = 0.5 * (band_h - staff_h)
        yc_lo = y0 + 2.0 * sp + min(3.0, margin)
        yc_hi = y1 - 2.0 * sp - min(3.0, margin)
        yc = float(rng.uniform(yc_lo, max(yc_lo, yc_hi)))

        x_start = int(rng.uniform(0.03, 0.14) * w)
        x_end = int(rng.uniform(0.86, 0.97) * w)
        if x_end - x_start < max(16, w // 6):
            continue

        # Slight tilt + subtle hand-ruled wobble.
        slope = float(rng.uniform(-0.015, 0.015))
        wob_amp = float(rng.uniform(0.25, 1.25))
        wob_cyc = float(rng.uniform(1.0, 3.0))
        wob_phi = float(rng.uniform(0.0, 2.0 * math.pi))
        x_span = max(float(x_end - x_start), 1.0)
        x_mid = 0.5 * (x_start + x_end)

        # Faint sepia/gray manuscript ink.
        ink = np.array(
            [rng.uniform(0.32, 0.62), rng.uniform(0.28, 0.56), rng.uniform(0.22, 0.48)],
            dtype=np.float32,
        )
        alpha = float(rng.uniform(0.20, 0.50))
        thickness = 1 if rng.random() < 0.7 else 2

        cols = xs[x_start:x_end]
        dy_curve = slope * (cols - x_mid) + wob_amp * np.sin(
            2.0 * math.pi * wob_cyc * ((cols - x_start) / x_span) + wob_phi
        )
        cols_i = cols.astype(np.int32)

        for k in range(-2, 3):
            yk = np.rint(yc + k * sp + dy_curve).astype(np.int32)
            for dt in range(thickness):
                yy = yk + dt
                ok = (yy >= y0) & (yy < y1) & (yy >= 0) & (yy < h)
                if not np.any(ok):
                    continue
                yy_v = yy[ok]
                xx_v = cols_i[ok]
                free = ~occ_img[yy_v, xx_v]
                if not np.any(free):
                    continue
                yy_v = yy_v[free]
                xx_v = xx_v[free]
                work[yy_v, xx_v] = work[yy_v, xx_v] * (1.0 - alpha) + ink[None, :] * alpha


def _draw_cursive_lyric_blocks(
    work: np.ndarray,
    occ_img: np.ndarray,
    bands: list[tuple[int, int]],
    sp_est: float,
    rng: np.random.Generator,
) -> None:
    """Draw dense multi-line horizontal cursive/lyric text hard negatives."""
    h, w, _ = work.shape
    min_band_h = max(12, int(math.ceil(1.6 * sp_est)))
    eligible = [b for b in bands if (b[1] - b[0]) >= min_band_h]
    if not eligible:
        return

    n_blocks = int(rng.integers(1, min(3, len(eligible)) + 1))
    chosen_idx = rng.choice(len(eligible), size=n_blocks, replace=False)

    for idx in chosen_idx:
        y0, y1 = eligible[int(idx)]
        band_h = y1 - y0
        line_h = max(6.0, min(18.0, float(rng.uniform(0.85, 1.45) * sp_est)))
        max_lines = max(1, min(4, int(band_h // max(line_h * 1.25, 8.0))))
        n_lines = int(rng.integers(1, max_lines + 1))

        block_h = n_lines * line_h * 1.25
        y_top = float(rng.uniform(y0 + 2.0, max(y0 + 2.0, y1 - block_h - 2.0)))

        x_left = int(rng.uniform(0.06, 0.20) * w)
        x_right = int(rng.uniform(0.75, 0.94) * w)
        if x_right - x_left < max(20, w // 6):
            continue

        ink = np.array(
            [rng.uniform(0.10, 0.38), rng.uniform(0.08, 0.33), rng.uniform(0.06, 0.28)],
            dtype=np.float32,
        )
        alpha = float(rng.uniform(0.35, 0.78))

        for li in range(n_lines):
            y_base = y_top + (li + 0.5) * line_h * 1.2
            if y_base - 0.5 * line_h < y0 or y_base + 0.5 * line_h >= y1:
                continue

            # Walk across the line drawing dense cursive word segments.
            x_cur = x_left + int(rng.integers(0, max(1, int(0.05 * w))))
            while x_cur < x_right - 8:
                word_len = int(rng.integers(max(10, int(0.025 * w)), max(16, int(0.11 * w))))
                x_end_w = min(x_right, x_cur + word_len)
                if x_end_w - x_cur < 6:
                    break

                xs_w = np.arange(x_cur, x_end_w, dtype=np.float32)
                t = xs_w - float(x_cur)
                f1 = float(rng.uniform(0.45, 0.95))
                f2 = float(rng.uniform(1.10, 2.20))
                p1 = float(rng.uniform(0.0, 2.0 * math.pi))
                p2 = float(rng.uniform(0.0, 2.0 * math.pi))
                amp = 0.28 * line_h

                # Primary cursive baseline wave + secondary loop/ascender stroke.
                y_wave1 = y_base + amp * (0.65 * np.sin(f1 * t + p1) + 0.35 * np.sin(f2 * t + p2))
                y_wave2 = y_base - 0.22 * line_h + amp * 0.55 * np.cos(0.8 * f1 * t + p2)

                for y_wave in (y_wave1, y_wave2):
                    yi = np.rint(y_wave).astype(np.int32)
                    xi = xs_w.astype(np.int32)
                    for dy_t in (0, 1):
                        yy = yi + dy_t
                        ok = (yy >= y0) & (yy < y1) & (yy >= 0) & (yy < h)
                        if not np.any(ok):
                            continue
                        yy_v = yy[ok]
                        xx_v = xi[ok]
                        free = ~occ_img[yy_v, xx_v]
                        if not np.any(free):
                            continue
                        yy_v = yy_v[free]
                        xx_v = xx_v[free]
                        work[yy_v, xx_v] = work[yy_v, xx_v] * (1.0 - alpha) + ink[None, :] * alpha

                gap = int(rng.integers(max(4, int(0.008 * w)), max(8, int(0.03 * w))))
                x_cur = x_end_w + gap


def augment_hw_hard_negatives(
    img: np.ndarray | torch.Tensor,
    heatmaps: np.ndarray | torch.Tensor,
    rng: np.random.Generator | int | None = None,
    p_blank_staves: float = 0.35,
    p_cursive_text: float = 0.35,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """Inject handwritten Stage-1 hard negatives into empty page regions.

    Directly targets the top Stage-1 handwritten false-positive modes while
    keeping GT ``ch0=0, ch1=0, ch2=0`` strictly untouched:
      1. Synthetic blank pre-ruled 5-line manuscript staves (``p=0.35``) in
         empty inter-system gaps & top/bottom margins.
      2. Synthetic dense multi-line cursive/lyric text blocks (``p=0.35``) in
         empty margins & inter-system gaps.

    Supports both ``np.ndarray`` (``[H, W, 3]``, ``[3, H, W]``, ``[H, W]``,
    float ``[0, 1]`` or ``uint8``) and ``torch.Tensor`` (``[C, H, W]`` or
    ``[H, W, C]``), returning the same type, layout, and dtype without mutating
    inputs in-place.
    """
    if rng is None:
        gen = np.random.default_rng()
    elif isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng(int(rng))

    # Normalize image to float32 [H, W, 3] in [0, 1] while recording original format.
    is_tensor = isinstance(img, torch.Tensor)
    orig_dtype = img.dtype
    orig_device = img.device if is_tensor else None
    arr = img.detach().cpu().numpy() if is_tensor else np.asarray(img)

    is_uint8 = (not is_tensor) and (arr.dtype == np.uint8)
    was_2d = arr.ndim == 2
    was_chw = (arr.ndim == 3) and (arr.shape[0] in (1, 3)) and (arr.shape[2] not in (1, 3))
    was_1ch = False

    if was_2d:
        rgb = np.repeat(arr[:, :, None], 3, axis=2)
    elif was_chw:
        hwc = np.transpose(arr, (1, 2, 0))
        if hwc.shape[2] == 1:
            was_1ch = True
            rgb = np.repeat(hwc, 3, axis=2)
        else:
            rgb = hwc
    elif arr.ndim == 3 and arr.shape[2] == 1:
        was_1ch = True
        rgb = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        rgb = arr
    else:
        raise ValueError(f"unsupported img shape: {arr.shape}")

    work = (rgb.astype(np.float32) / 255.0) if is_uint8 else rgb.astype(np.float32, copy=True)
    h, w, _ = work.shape

    # Extract GT positive occupancy from heatmaps ch0..2 (never mutated).
    hm_arr = heatmaps.detach().cpu().numpy() if isinstance(heatmaps, torch.Tensor) else np.asarray(heatmaps)
    if hm_arr.ndim != 3 or hm_arr.shape[0] < 3:
        raise ValueError(f"expected [>=3, Hp, Wp] heatmaps, got {hm_arr.shape}")
    hp, wp = int(hm_arr.shape[1]), int(hm_arr.shape[2])
    occ_hm = (hm_arr[0] > 0.05) | (hm_arr[1] > 0.05) | (hm_arr[2] > 0.05)

    # Upsample occupancy to image resolution (H, W) via nearest-neighbor indexing.
    y_idx = np.clip((np.arange(h) * hp) // max(h, 1), 0, hp - 1)
    x_idx = np.clip((np.arange(w) * wp) // max(w, 1), 0, wp - 1)
    occ_img = occ_hm[np.ix_(y_idx, x_idx)]

    # Estimate typical staff spacing in image pixels for realistic scale.
    sp_est = float(np.clip(0.012 * h, 4.0, 20.0))
    pad_rows = max(4, int(round(1.2 * sp_est)))

    # Dilate row occupancy by pad_rows so hard negatives stay clear of real staves.
    raw_row_occ = occ_img.any(axis=1)
    if np.any(raw_row_occ):
        kernel = np.ones(2 * pad_rows + 1, dtype=np.int32)
        row_occ = np.convolve(raw_row_occ.astype(np.int32), kernel, mode="same") > 0
    else:
        row_occ = raw_row_occ

    bands = _find_blank_bands(row_occ, min_height=max(12, int(round(1.5 * sp_est))))

    if bands:
        if float(gen.random()) < float(p_blank_staves):
            _draw_blank_manuscript_staves(work, occ_img, bands, sp_est, gen)
        if float(gen.random()) < float(p_cursive_text):
            _draw_cursive_lyric_blocks(work, occ_img, bands, sp_est, gen)

    work = np.clip(work, 0.0, 1.0)

    # Convert back to original layout and type.
    if was_2d:
        out_np = work.mean(axis=2)
    elif was_1ch:
        gray = work.mean(axis=2, keepdims=True)
        out_np = np.transpose(gray, (2, 0, 1)) if was_chw else gray
    else:
        out_np = np.transpose(work, (2, 0, 1)) if was_chw else work

    if is_uint8:
        out_img: np.ndarray | torch.Tensor = np.clip(np.rint(out_np * 255.0), 0, 255).astype(np.uint8)
    elif is_tensor:
        out_img = torch.from_numpy(out_np.copy()).to(device=orig_device, dtype=orig_dtype)
    else:
        out_img = out_np.astype(orig_dtype, copy=False)

    return out_img, heatmaps


def run_unsup_round(
    model: nn.Module,
    ema_model: nn.Module | None,
    opt: torch.optim.Optimizer,
    image_paths: list,
    steps: int = 200,
    ema_decay: float = 0.999,
    device: torch.device | str = "cpu",
    epoch: int = 1,
    seed: int = 42,
    batch_size: int = 4,
    target_h: int = 1024,
    target_w: int = 768,
    dynamic_res: bool = False,
    sp_mode: str = "log_hm",
    out_stride: int = 2,
    bf16: bool = False,
    **_extra,
) -> dict:
    """Execute one Stage-1 unsupervised EMA geometric + log_hm scale equivariance round."""
    from PIL import Image

    if not image_paths or int(steps) <= 0:
        return {"ran_steps": 0, "mean_loss": 0.0}

    dev = torch.device(device)
    teacher = ema_model if isinstance(ema_model, EMAModel) else EMAModel(model, decay=ema_decay)
    if isinstance(ema_model, nn.Module) and not isinstance(ema_model, EMAModel):
        teacher = ema_model
    teacher_is_wrapper = isinstance(teacher, EMAModel)
    teacher_net = getattr(teacher, "module", getattr(teacher, "model", teacher)) if teacher_is_wrapper else teacher
    teacher_net.to(dev).eval()

    rng = np.random.default_rng(int(seed))
    n_imgs = len(image_paths)
    was_train = model.training
    model.train()
    use_bf16 = bool(bf16 and dev.type == "cuda")
    total_loss = 0.0
    ran = 0
    bs = max(1, min(int(batch_size), 4))
    in_h, in_w = (1024, 768) if dynamic_res else (int(target_h), int(target_w))

    for s in range(int(steps)):
        idxs = rng.integers(0, n_imgs, size=bs)
        batch_tensors = []
        for idx in idxs:
            try:
                im = Image.open(image_paths[int(idx)]).convert("RGB")
                im = im.resize((in_w, in_h), Image.BOX)
                arr = np.asarray(im, dtype=np.float32) / 255.0
                batch_tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
            except Exception:
                continue
        if not batch_tensors:
            continue
        x = torch.stack(batch_tensors, dim=0).to(dev, non_blocking=True)
        if dev.type == "cuda":
            x = x.to(memory_format=torch.channels_last)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            eq_out = stage1_equivariance_loss(
                model,
                teacher_net,
                x,
                sp_mode=sp_mode,
            )
            loss = eq_out[0] if isinstance(eq_out, tuple) else eq_out
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            if teacher_is_wrapper:
                teacher.update(model, decay=ema_decay)
            else:
                with torch.no_grad():
                    for p_t, p_s in zip(teacher_net.parameters(), model.parameters()):
                        p_t.mul_(ema_decay).add_(p_s, alpha=1.0 - ema_decay)
            total_loss += float(loss.detach().item())
            ran += 1

    model.train(was_train)
    mean_loss = total_loss / max(1, ran)
    print(f"  [unsup-stage1 ep {epoch:03d}] ran_steps={ran}/{steps} mean_loss={mean_loss:.4f}", flush=True)
    return {"ran_steps": ran, "mean_loss": mean_loss}
