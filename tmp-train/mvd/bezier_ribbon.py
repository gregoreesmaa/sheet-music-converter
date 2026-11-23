"""Batch 5-B2: Differentiable Bezier Ribbon Sampler (crop-free curved system unwarper).

Given a full-page image tensor ``[C, H, W]`` (or ``[B, C, H, W]`` or PIL Image)
and a curved system's staff-centerline cubic Bezier curves + local staff space
``sp(t)``, samples an unwarped, canonical-height (default ``256 px``) 4-slot
strip directly via ``F.grid_sample`` — zero axis-aligned rectangular cropping!

Canonical 4-slot layout matches ``rectify.py`` exactly:
    ``y_canon(s, k) = canon_top + slot_pitch * s + k * target_sp``
    (at default ``target_sp=10.0, height=256``: ``12.0 + 64.0 * s + 10.0 * k``)

Even on a ``15 deg`` S-curved page (``ds0324-7``) or ``10 deg`` tilted page
(``sq7070319-1``), all 5 staff lines of every staff in the system land flat and
unclipped at their canonical vertical positions, while remaining GPU-differentiable
and CPU-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

CANON_TOP = 12.0
SLOT_PITCH = 64.0
MAX_STAVES = 4
LINES_PER_STAFF = 5
REF_TARGET_SP = 10.0

__all__ = [
    "BezierStaff",
    "canonical_layout",
    "canon_y",
    "eval_cubic_bezier",
    "eval_cubic_bezier_tangent",
    "staves_from_top_bottom",
    "chunk_staves_for_ribbon",
    "build_ribbon_grid",
    "refine_staves_comb",
    "sample_bezier_ribbon",
    "sample_curved_system_chunks",
    "BezierRibbonSampler",
]


def canonical_layout(
    target_sp: float = REF_TARGET_SP,
    height: int | None = 256,
    n_slots: int = MAX_STAVES,
) -> tuple[float, float, int]:
    """Return ``(canon_top, slot_pitch, out_h)`` matching ``rectify.py``."""
    sp = float(target_sp)
    if not np.isfinite(sp) or sp <= 0:
        raise ValueError(f"target_sp must be positive, got {target_sp}")
    if height is None:
        if sp == REF_TARGET_SP and n_slots == MAX_STAVES:
            return CANON_TOP, SLOT_PITCH, int(SLOT_PITCH * MAX_STAVES)
        scale = sp / REF_TARGET_SP
        min_pitch = 4.0 * sp + max(8.0, 2.0 * sp)
        raw_pitch = max(SLOT_PITCH * scale, min_pitch)
        slot_pitch = float(int(round(raw_pitch / 8.0)) * 8)
        if slot_pitch <= 4.0 * sp:
            slot_pitch = float(int(np.ceil((4.0 * sp + 8.0) / 8.0)) * 8)
        out_h = int(round(slot_pitch * n_slots))
    else:
        out_h = int(height)
        if out_h <= 0:
            raise ValueError(f"height must be positive, got {height}")
        slot_pitch = float(out_h) / float(n_slots)
    canon_top = (slot_pitch - 4.0 * sp) / 2.0
    if slot_pitch <= 4.0 * sp:
        raise ValueError("canonical control rows overlap; slots require pitch > 4*target_sp")
    return canon_top, slot_pitch, out_h


def canon_y(
    slot: int,
    line: float,
    target_sp: float = REF_TARGET_SP,
    height: int | None = 256,
    n_slots: int = MAX_STAVES,
) -> float:
    """Canonical row of staff ``slot`` at staff-step ``line`` (0=top, 2=center, 4=bottom)."""
    top, pitch, _ = canonical_layout(target_sp=target_sp, height=height, n_slots=n_slots)
    return top + pitch * slot + line * float(target_sp)


@dataclass
class BezierStaff:
    """Differentiable or numpy cubic-Bezier staff centerline + local ``sp(t)``."""

    ctrl_pts: torch.Tensor | np.ndarray  # (4, 2) [(x0, y0), ..., (x3, y3)] in page px
    sp_ctrl: torch.Tensor | np.ndarray | float  # scalar or (4,) Bezier control values in px


def eval_cubic_bezier(ctrl_pts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate cubic Bezier curve ``ctrl_pts`` ``(..., 4, D)`` at ``t`` ``(N,)`` -> ``(..., N, D)``."""
    u = 1.0 - t
    b0 = u * u * u
    b1 = 3.0 * u * u * t
    b2 = 3.0 * u * t * t
    b3 = t * t * t
    B = torch.stack([b0, b1, b2, b3], dim=-1)  # (N, 4)
    return torch.einsum("nc,...cd->...nd", B, ctrl_pts)


def eval_cubic_bezier_tangent(ctrl_pts: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate cubic Bezier derivative ``dB/dt`` ``(..., N, D)`` at ``t`` ``(N,)``."""
    u = 1.0 - t
    d0 = 3.0 * u * u
    d1 = 6.0 * u * t
    d2 = 3.0 * t * t
    D = torch.stack([d0, d1, d2], dim=-1)  # (N, 3)
    diffs = ctrl_pts[..., 1:, :] - ctrl_pts[..., :-1, :]  # (..., 3, D)
    return torch.einsum("nc,...cd->...nd", D, diffs)


def staves_from_top_bottom(
    top_ctrl: torch.Tensor | np.ndarray,
    bottom_ctrl: torch.Tensor | np.ndarray,
    sp_ctrl: torch.Tensor | np.ndarray | float,
    n_staves: int = 2,
) -> list[BezierStaff]:
    """Interpolate ``n_staves`` evenly spaced ``BezierStaff`` curves from top & bottom Beziers."""
    if n_staves < 1:
        raise ValueError(f"n_staves must be >= 1, got {n_staves}")
    top_t = torch.as_tensor(top_ctrl, dtype=torch.float32)
    bot_t = torch.as_tensor(bottom_ctrl, dtype=torch.float32)
    if n_staves == 1:
        return [BezierStaff(ctrl_pts=0.5 * (top_t + bot_t), sp_ctrl=sp_ctrl)]
    out: list[BezierStaff] = []
    for s in range(n_staves):
        alpha = float(s) / float(n_staves - 1)
        out.append(BezierStaff(ctrl_pts=(1.0 - alpha) * top_t + alpha * bot_t, sp_ctrl=sp_ctrl))
    return out


def _coerce_staves(
    staves: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize input staves into ``(ctrl_pts [S, 4, 2], sp_ctrl [S, 4])`` tensors."""
    if hasattr(staves, "staves"):
        staff_list = list(staves.staves)
    elif isinstance(staves, Sequence):
        staff_list = list(staves)
    else:
        staff_list = [staves]

    if not staff_list:
        raise ValueError("staves must be non-empty")

    ctrls: list[torch.Tensor] = []
    sps: list[torch.Tensor] = []
    for st in staff_list:
        if isinstance(st, dict):
            cp = st["ctrl_pts"]
            sp = st.get("sp_ctrl", st.get("sp"))
        else:
            cp = st.ctrl_pts
            sp = getattr(st, "sp_ctrl", getattr(st, "sp", None))
        if sp is None:
            raise ValueError("each staff must provide sp_ctrl or sp")

        cp_t = cp.to(device=device, dtype=dtype) if isinstance(cp, torch.Tensor) else torch.as_tensor(cp, device=device, dtype=dtype)
        if cp_t.shape != (4, 2):
            raise ValueError(f"ctrl_pts must have shape (4, 2), got {tuple(cp_t.shape)}")

        if isinstance(sp, torch.Tensor):
            sp_t = sp.to(device=device, dtype=dtype)
        else:
            sp_t = torch.as_tensor(sp, device=device, dtype=dtype)
        if sp_t.ndim == 0:
            sp_t = sp_t.expand(4)
        elif sp_t.shape == (1,):
            sp_t = sp_t.expand(4)
        elif sp_t.shape != (4,):
            raise ValueError(f"sp_ctrl must be scalar or shape (4,), got {tuple(sp_t.shape)}")

        ctrls.append(cp_t)
        sps.append(sp_t)

    return torch.stack(ctrls, dim=0), torch.stack(sps, dim=0)


def _arclength_t(ctrl_pts: torch.Tensor, n_out: int, n_dense: int = 256) -> torch.Tensor:
    """Compute ``n_out`` parameter values ``t in [0, 1]`` uniformly spaced by arc length."""
    device, dtype = ctrl_pts.device, ctrl_pts.dtype
    t_dense = torch.linspace(0.0, 1.0, n_dense, device=device, dtype=dtype)
    pts = eval_cubic_bezier(ctrl_pts, t_dense)  # (S, n_dense, 2)
    # Use the mean centerline across staves for shared system arc-length parameterization.
    mean_pts = pts.mean(dim=0)  # (n_dense, 2)
    seg = torch.linalg.norm(mean_pts[1:] - mean_pts[:-1], dim=-1)
    cum = torch.cat([torch.zeros(1, device=device, dtype=dtype), torch.cumsum(seg, dim=0)])
    total = cum[-1].clamp(min=1e-6)
    s_norm = cum / total  # (n_dense,)

    target_s = torch.linspace(0.0, 1.0, n_out, device=device, dtype=dtype)
    # Differentiable piecewise-linear inversion of monotonic s_norm -> t_dense.
    idx = torch.searchsorted(s_norm.detach(), target_s.detach()).clamp(1, n_dense - 1)
    s0 = s_norm[idx - 1]
    s1 = s_norm[idx]
    t0 = t_dense[idx - 1]
    t1 = t_dense[idx]
    alpha = (target_s - s0) / (s1 - s0).clamp(min=1e-6)
    return t0 + alpha * (t1 - t0)


def build_ribbon_grid(
    staves: Any,
    page_h: int,
    page_w: int,
    target_sp: float = REF_TARGET_SP,
    height: int | None = 256,
    width: int | None = None,
    n_slots: int = MAX_STAVES,
    offset_mode: str = "normal",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a differentiable ``F.grid_sample`` sampling grid for a curved system.

    Args:
        staves: ``CurvedSystem`` or sequence of ``BezierStaff`` / ``CurvedStaff``
            (``1 <= S <= n_slots``).
        page_h, page_w: Full page image dimensions in pixels.
        target_sp: Canonical staff space in px (default 10.0).
        height: Canonical strip height in px (default 256).
        width: Output strip width in px. ``None`` computes
            ``round(arc_length * target_sp / median_sp)`` automatically.
        n_slots: Number of canonical staff slots (default 4).
        offset_mode: ``"normal"`` (offsets along local staff normal; exact under
            rotation/bend and <=0.7 px on 15 deg S-curves) or ``"vertical"``
            (pure vertical y-offsets; exact under vertical-shear S-curves).
        device, dtype: Target tensor device/dtype.

    Returns:
        ``(grid, active_mask, ref_channel)``:
          - ``grid``: ``[1, out_h, out_w, 2]`` normalized ``[-1, 1]`` grid for
            ``F.grid_sample(..., align_corners=True)``.
          - ``active_mask``: ``[1, 1, out_h, out_w]`` float mask (``1.0`` inside
            active staff slots ``0..S-1``, ``0.0`` in inactive slots).
          - ``ref_channel``: ``[1, out_h, out_w]`` float tensor with ``1.0`` on
            the 5 canonical rows of each active staff.
    """
    if offset_mode not in ("normal", "vertical"):
        raise ValueError(f"offset_mode must be 'normal' or 'vertical', got {offset_mode!r}")

    dev = torch.device(device) if device is not None else torch.device("cpu")
    ctrl_pts, sp_ctrl = _coerce_staves(staves, device=dev, dtype=dtype)
    n_staff = ctrl_pts.shape[0]
    if not 1 <= n_staff <= n_slots:
        raise ValueError(f"need 1..{n_slots} staves, got {n_staff}")

    canon_top, slot_pitch, out_h = canonical_layout(
        target_sp=target_sp, height=height, n_slots=n_slots
    )

    # Determine output width from arc length * (target_sp / median_sp) when omitted.
    if width is None:
        t_probe = torch.linspace(0.0, 1.0, 65, device=dev, dtype=dtype)
        pts_probe = eval_cubic_bezier(ctrl_pts, t_probe).mean(dim=0)
        arc_len = float(torch.linalg.norm(pts_probe[1:] - pts_probe[:-1], dim=-1).sum().item())
        med_sp = float(sp_ctrl.median().item())
        if not np.isfinite(med_sp) or med_sp <= 0:
            raise ValueError(f"non-positive staff space: {med_sp}")
        out_w = max(4, int(round(arc_len * float(target_sp) / med_sp)))
    else:
        out_w = int(width)
        if out_w < 1:
            raise ValueError(f"width must be positive, got {width}")

    # Arc-length-uniform parameter t along the system: (out_w,)
    t = _arclength_t(ctrl_pts, n_out=out_w)

    # Evaluate centerlines (S, W, 2), tangents (S, W, 2), and local sp (S, W).
    centers = eval_cubic_bezier(ctrl_pts, t)  # (S, W, 2)
    sp_loc = eval_cubic_bezier(sp_ctrl.unsqueeze(-1), t).squeeze(-1).clamp(min=1e-3)  # (S, W)

    if offset_mode == "normal":
        tang = eval_cubic_bezier_tangent(ctrl_pts, t)  # (S, W, 2)
        t_norm = torch.linalg.norm(tang, dim=-1, keepdim=True).clamp(min=1e-6)
        t_hat = tang / t_norm
        # Downward unit normal (since image +y is down and +x is right).
        n_hat = torch.stack([-t_hat[..., 1], t_hat[..., 0]], dim=-1)  # (S, W, 2)
        # Ensure normal points downward (+y) even if a curve was reversed.
        sign = torch.where(n_hat[..., 1:2] >= 0, 1.0, -1.0)
        n_hat = n_hat * sign
    else:
        n_hat = torch.zeros_like(centers)
        n_hat[..., 1] = 1.0

    # Build control points in canonical y and corresponding page (x, y) per column:
    #   [top_margin, staff0 line0..4, ..., staff(S-1) line0..4, bottom_margin]
    line_steps = torch.arange(LINES_PER_STAFF, device=dev, dtype=dtype)  # (5,)
    staff_ids = torch.arange(n_staff, device=dev, dtype=dtype)           # (S,)

    yc = (
        canon_top
        + slot_pitch * staff_ids[:, None]
        + float(target_sp) * line_steps[None, :]
    )  # (S, 5)
    p_out = torch.cat(
        [
            (yc[0, 0] - 2.0 * float(target_sp)).view(1),
            yc.reshape(-1),
            (yc[-1, -1] + 2.0 * float(target_sp)).view(1),
        ],
        dim=0,
    )  # (K,) where K = 5*S + 2

    # Page (x, y) for each staff line: (S, 5, W, 2)
    line_offsets = (line_steps - 2.0)[None, :, None, None]  # (1, 5, 1, 1)
    step_vec = sp_loc[:, None, :, None] * n_hat[:, None, :, :]  # (S, 1, W, 2)
    line_pts = centers[:, None, :, :] + line_offsets * step_vec  # (S, 5, W, 2)

    top_margin_pt = centers[0] + (-4.0 * sp_loc[0, :, None]) * n_hat[0]  # (W, 2)
    bot_margin_pt = centers[-1] + (4.0 * sp_loc[-1, :, None]) * n_hat[-1]  # (W, 2)

    p_in = torch.cat(
        [
            top_margin_pt.unsqueeze(0),
            line_pts.reshape(n_staff * LINES_PER_STAFF, out_w, 2),
            bot_margin_pt.unsqueeze(0),
        ],
        dim=0,
    )  # (K, W, 2)

    # Piecewise-linear interpolation along output rows [0..out_h-1] (fully differentiable!).
    rows = torch.arange(out_h, device=dev, dtype=dtype)  # (H_out,)
    K = p_out.shape[0]
    rows_clamped = rows.clamp(min=p_out[0], max=p_out[-1])
    seg_idx = torch.searchsorted(p_out.detach(), rows_clamped.detach()).clamp(1, K - 1)
    y0 = p_out[seg_idx - 1]  # (H_out,)
    y1 = p_out[seg_idx]      # (H_out,)
    alpha = ((rows_clamped - y0) / (y1 - y0).clamp(min=1e-6))[:, None, None]  # (H_out, 1, 1)

    pt0 = p_in[seg_idx - 1]  # (H_out, W, 2)
    pt1 = p_in[seg_idx]      # (H_out, W, 2)
    page_xy = pt0 + alpha * (pt1 - pt0)  # (H_out, W, 2)

    # Normalize page pixel coords to F.grid_sample [-1, 1] (align_corners=True).
    gx = 2.0 * page_xy[..., 0] / max(float(page_w - 1), 1.0) - 1.0
    gy = 2.0 * page_xy[..., 1] / max(float(page_h - 1), 1.0) - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1, H_out, W, 2)

    active_end = int(round(n_staff * slot_pitch))
    active_mask = torch.zeros((1, 1, out_h, out_w), device=dev, dtype=dtype)
    active_mask[:, :, :active_end, :] = 1.0

    ref = torch.zeros((1, out_h, out_w), device=dev, dtype=dtype)
    for s in range(n_staff):
        for k in range(LINES_PER_STAFF):
            r = int(round(canon_top + slot_pitch * s + k * float(target_sp)))
            if 0 <= r < out_h:
                ref[0, r, :] = 1.0

    return grid, active_mask, ref


def _fit_cubic_bezier_nd(t_vals: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Least-squares fit of 4 cubic Bezier control points ``(4, D)`` to ``pts`` ``(N, D)``."""
    u = 1.0 - t_vals
    B = np.stack([u ** 3, 3.0 * u * u * t_vals, 3.0 * u * t_vals * t_vals, t_vals ** 3], axis=-1)
    ctrl, *_ = np.linalg.lstsq(B, pts, rcond=None)
    return ctrl.astype(np.float32)


def refine_staves_comb(
    page: torch.Tensor | Image.Image,
    staves: Any,
    target_sp: float = REF_TARGET_SP,
    height: int | None = 256,
    width: int | None = None,
    n_slots: int = MAX_STAVES,
    offset_mode: str = "normal",
    paper_value: float | None = None,
    n_chunks: int = 8,
    max_dy_sp: float = 0.45,
    max_sp_frac: float = 0.12,
    min_contrast: float = 0.04,
) -> list[BezierStaff]:
    """Full-resolution 5-line vertical comb snap for ``StaffUNet2D`` Bezier curves.

    ``StaffUNet2D`` predicts staff centerlines and local ``sp`` at ``512x384``
    (~5-7x downsampled from a full page), leaving ~1-3 px vertical ribbon
    jitter at canonical scale. This fast (~2-4 ms CPU) pass samples the initial
    ribbon in grayscale, takes horizontal median profiles across ``n_chunks``
    bins (ignoring noteheads/stems/beams), and maximizes a 5-line-minus-4-gap
    comb score over ``(dy, sp_scale)`` to lock each staff's 5 lines onto the
    canonical ``ref_channel`` rows to ``< 0.25 px``.

    Safely falls back to the un-snapped Bezier for any staff with low contrast
    (blank, extreme washout, or occluded). Always runs under ``torch.no_grad``.
    """
    with torch.no_grad():
        if isinstance(page, Image.Image):
            arr = np.asarray(page.convert("L"), dtype=np.float32) / 255.0
            gray_ten = torch.from_numpy(arr[None, None, :, :])
        elif isinstance(page, torch.Tensor):
            t_in = page.detach().to(device="cpu", dtype=torch.float32)
            if t_in.ndim == 3:
                t_in = t_in.unsqueeze(0)
            elif t_in.ndim != 4:
                raise ValueError(f"page tensor must be [C,H,W] or [B,C,H,W], got {tuple(page.shape)}")
            gray_ten = t_in[:1].mean(dim=1, keepdim=True)
            if paper_value is not None and float(paper_value) > 1.5:
                gray_ten = gray_ten / float(paper_value)
        else:
            raise TypeError(f"page must be torch.Tensor or PIL.Image, got {type(page).__name__}")

        ctrl_pts, sp_ctrl = _coerce_staves(staves, device=torch.device("cpu"), dtype=torch.float32)
        n_staff = int(ctrl_pts.shape[0])
        canon_top, slot_pitch, out_h = canonical_layout(
            target_sp=target_sp, height=height, n_slots=n_slots
        )

        init_ribbon, _ = sample_bezier_ribbon(
            page=gray_ten,
            staves=staves,
            target_sp=target_sp,
            height=height,
            width=width,
            n_slots=n_slots,
            offset_mode=offset_mode,
            paper_value=1.0,
        )
        assert isinstance(init_ribbon, torch.Tensor)
        rib = init_ribbon[0, 0].cpu().numpy()  # (H_out, W_out) in [0, 1]
        out_w = rib.shape[1]

        t_cols = _arclength_t(ctrl_pts, n_out=out_w).cpu().numpy()
        t_dense = torch.linspace(0.0, 1.0, 33, dtype=torch.float32)
        centers_dense = eval_cubic_bezier(ctrl_pts, t_dense).cpu().numpy()  # (S, 33, 2)
        sp_dense = eval_cubic_bezier(sp_ctrl.unsqueeze(-1), t_dense).squeeze(-1).clamp(min=1e-3).cpu().numpy()
        if offset_mode == "normal":
            tang = eval_cubic_bezier_tangent(ctrl_pts, t_dense)
            t_hat = tang / torch.linalg.norm(tang, dim=-1, keepdim=True).clamp(min=1e-6)
            n_hat = torch.stack([-t_hat[..., 1], t_hat[..., 0]], dim=-1)
            n_hat = (n_hat * torch.where(n_hat[..., 1:2] >= 0, 1.0, -1.0)).cpu().numpy()
        else:
            n_hat = np.zeros_like(centers_dense)
            n_hat[..., 1] = 1.0

        t_dense_np = t_dense.numpy()

        # Candidate grid in canonical ribbon pixels.
        dy_grid = np.linspace(-max_dy_sp * target_sp, max_dy_sp * target_sp, 37, dtype=np.float32)
        sc_grid = np.linspace(1.0 - max_sp_frac, 1.0 + max_sp_frac, 13, dtype=np.float32)
        line_k = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        gap_k = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)

        # Horizontal chunk bounds (skip outermost 4% margin when strip is wide enough).
        margin = int(round(0.04 * out_w)) if out_w >= 80 else 0
        c_edges = np.linspace(margin, max(margin + 1, out_w - margin), max(2, int(n_chunks)) + 1)
        c_edges = np.round(c_edges).astype(int)

        refined: list[BezierStaff] = []
        for s in range(n_staff):
            yc_center = canon_top + slot_pitch * s + 2.0 * float(target_sp)
            r0 = max(0, int(np.floor(yc_center - 3.2 * target_sp)))
            r1 = min(out_h, int(np.ceil(yc_center + 3.2 * target_sp)) + 1)
            if r1 - r0 < int(4 * target_sp):
                refined.append(BezierStaff(ctrl_pts=ctrl_pts[s].clone(), sp_ctrl=sp_ctrl[s].clone()))
                continue

            ys_local = np.arange(r0, r1, dtype=np.float32)
            # Precompute comb sampling coordinates relative to r0: (D, S_sc, 5) and (D, S_sc, 4).
            y_lines = (
                yc_center
                + dy_grid[:, None, None]
                + float(target_sp) * sc_grid[None, :, None] * line_k[None, None, :]
                - float(r0)
            )
            y_gaps = (
                yc_center
                + dy_grid[:, None, None]
                + float(target_sp) * sc_grid[None, :, None] * gap_k[None, None, :]
                - float(r0)
            )
            p_len = r1 - r0

            obs_t: list[float] = []
            obs_dy: list[float] = []
            obs_sc: list[float] = []
            obs_w: list[float] = []

            for m in range(len(c_edges) - 1):
                c0, c1 = int(c_edges[m]), int(c_edges[m + 1])
                if c1 - c0 < 4:
                    continue
                patch = rib[r0:r1, c0:c1]
                prof_gray = np.median(patch, axis=1)
                bg = float(np.percentile(prof_gray, 80))
                dark = np.clip(bg - prof_gray, 0.0, 1.0)
                if float(dark.max()) < min_contrast:
                    continue

                # Linear interpolation of 1D darkness profile at candidate comb rows.
                def _interp1(coords: np.ndarray) -> np.ndarray:
                    c_clamped = np.clip(coords, 0.0, float(p_len - 1) - 1e-5)
                    i0 = c_clamped.astype(np.int32)
                    frac = c_clamped - i0
                    return dark[i0] * (1.0 - frac) + dark[i0 + 1] * frac

                line_val = _interp1(y_lines).mean(axis=-1)  # (D, S_sc)
                gap_val = _interp1(y_gaps).mean(axis=-1)    # (D, S_sc)
                scores = line_val - 0.85 * gap_val
                flat_idx = int(np.argmax(scores))
                best_score = float(scores.reshape(-1)[flat_idx])
                if best_score < min_contrast:
                    continue
                di, si = divmod(flat_idx, len(sc_grid))
                c_mid = min(out_w - 1, (c0 + c1) // 2)
                obs_t.append(float(t_cols[c_mid]))
                obs_dy.append(float(dy_grid[di]))
                obs_sc.append(float(sc_grid[si]))
                obs_w.append(best_score)

            if len(obs_t) < 2:
                refined.append(BezierStaff(ctrl_pts=ctrl_pts[s].clone(), sp_ctrl=sp_ctrl[s].clone()))
                continue

            ot = np.asarray(obs_t, dtype=np.float64)
            ody = np.asarray(obs_dy, dtype=np.float64)
            osc = np.asarray(obs_sc, dtype=np.float64)
            ow = np.asarray(obs_w, dtype=np.float64)

            # Robust inlier filter against local stains / dense beams.
            if len(ot) >= 4:
                med_dy = float(np.median(ody))
                med_sc = float(np.median(osc))
                keep = (np.abs(ody - med_dy) <= 0.25 * target_sp) & (np.abs(osc - med_sc) <= 0.06)
                if int(keep.sum()) >= 2:
                    ot, ody, osc, ow = ot[keep], ody[keep], osc[keep], ow[keep]

            # Ridge-regularized quadratic fit for dy(t) and linear fit for sp_scale(t).
            u_obs = ot - 0.5
            u_den = t_dense_np.astype(np.float64) - 0.5
            W = np.diag(ow)

            if len(ot) >= 4:
                X_dy = np.stack([np.ones_like(u_obs), u_obs, u_obs ** 2], axis=-1)
                reg_dy = np.diag([0.0, 1e-2, 5e-2])
                coef_dy = np.linalg.solve(X_dy.T @ W @ X_dy + reg_dy, X_dy.T @ W @ ody)
                dy_den = coef_dy[0] + coef_dy[1] * u_den + coef_dy[2] * (u_den ** 2)
            else:
                X_dy = np.stack([np.ones_like(u_obs), u_obs], axis=-1)
                reg_dy = np.diag([0.0, 2e-2])
                coef_dy = np.linalg.solve(X_dy.T @ W @ X_dy + reg_dy, X_dy.T @ W @ ody)
                dy_den = coef_dy[0] + coef_dy[1] * u_den

            X_sc = np.stack([np.ones_like(u_obs), u_obs], axis=-1)
            reg_sc = np.diag([0.0, 5e-2])
            coef_sc = np.linalg.solve(X_sc.T @ W @ X_sc + reg_sc, X_sc.T @ W @ (osc - 1.0))
            sc_den = np.clip(1.0 + coef_sc[0] + coef_sc[1] * u_den, 1.0 - max_sp_frac, 1.0 + max_sp_frac)

            # Map canonical ribbon dy back to page-space normal shift & refit cubic Beziers.
            delta_page = dy_den * (sp_dense[s] / float(target_sp))  # (33,)
            new_centers = centers_dense[s] + delta_page[:, None] * n_hat[s]
            new_sp = np.clip(sp_dense[s] * sc_den, 2.0, None)

            new_cp = _fit_cubic_bezier_nd(t_dense_np, new_centers)
            new_sp_ctrl = _fit_cubic_bezier_nd(t_dense_np, new_sp[:, None]).squeeze(-1)
            refined.append(BezierStaff(ctrl_pts=new_cp, sp_ctrl=new_sp_ctrl))

        return refined


def sample_bezier_ribbon(
    page: torch.Tensor | Image.Image,
    staves: Any,
    target_sp: float = REF_TARGET_SP,
    height: int | None = 256,
    width: int | None = None,
    n_slots: int = MAX_STAVES,
    offset_mode: str = "normal",
    paper_value: float | None = None,
    comb_snap: bool = False,
) -> tuple[torch.Tensor | Image.Image, torch.Tensor]:
    """Sample a canonical rectified strip directly from a full page via ``F.grid_sample``.

    Args:
        page: Full-page ``torch.Tensor`` (``[C, H, W]`` or ``[B, C, H, W]``) or
            ``PIL.Image.Image`` (``L`` or ``RGB``).
        staves: ``CurvedSystem`` or sequence of ``BezierStaff`` / ``CurvedStaff``.
        target_sp: Canonical staff space in px (default 10.0).
        height: Canonical strip height in px (default 256).
        width: Output strip width in px (``None`` auto-scales from arc length).
        n_slots: Number of canonical staff slots (default 4).
        offset_mode: ``"normal"`` (default) or ``"vertical"``.
        paper_value: Background fill value for out-of-bounds / inactive slots.
            Defaults to ``255.0`` for PIL images and ``1.0`` for float tensors.

    Returns:
        ``(ribbon, ref_channel)`` with the same container type as ``page``
        (PIL Image -> PIL Image; ``[C, H, W]`` -> ``[C, out_h, out_w]``;
        ``[B, C, H, W]`` -> ``[B, C, out_h, out_w]``).
    """
    if comb_snap:
        staves = refine_staves_comb(
            page=page,
            staves=staves,
            target_sp=target_sp,
            height=height,
            width=width,
            n_slots=n_slots,
            offset_mode=offset_mode,
            paper_value=paper_value,
        )

    is_pil = isinstance(page, Image.Image)
    if is_pil:
        work = page if page.mode in ("L", "RGB") else page.convert("RGB")
        arr = np.asarray(work, dtype=np.float32)
        is_rgb = arr.ndim == 3
        ten = torch.from_numpy(arr.transpose(2, 0, 1) if is_rgb else arr[None, :, :]).unsqueeze(0)
        pv = 255.0 if paper_value is None else float(paper_value)
        squeeze_batch = True
    else:
        if not isinstance(page, torch.Tensor):
            raise TypeError(f"page must be torch.Tensor or PIL.Image, got {type(page).__name__}")
        if page.ndim == 3:
            ten = page.unsqueeze(0)
            squeeze_batch = True
        elif page.ndim == 4:
            ten = page
            squeeze_batch = False
        else:
            raise ValueError(f"page tensor must be [C,H,W] or [B,C,H,W], got {tuple(page.shape)}")
        pv = 1.0 if paper_value is None else float(paper_value)

    B, _, page_h, page_w = ten.shape
    grid, active_mask, ref = build_ribbon_grid(
        staves=staves,
        page_h=page_h,
        page_w=page_w,
        target_sp=target_sp,
        height=height,
        width=width,
        n_slots=n_slots,
        offset_mode=offset_mode,
        device=ten.device,
        dtype=ten.dtype,
    )
    if B > 1:
        grid = grid.expand(B, -1, -1, -1)

    # Invert around paper_value so zeros padding seamlessly becomes paper white.
    inv = pv - ten
    sampled_inv = F.grid_sample(
        inv,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    sampled = pv - sampled_inv
    # Fill inactive slots with clean paper value.
    out = sampled * active_mask + pv * (1.0 - active_mask)

    if is_pil:
        arr_out = out[0].clamp(0.0, 255.0).round().to(torch.uint8).cpu().numpy()
        if arr_out.shape[0] == 1:
            img_out = Image.fromarray(arr_out[0], mode="L")
        else:
            img_out = Image.fromarray(arr_out.transpose(1, 2, 0), mode="RGB")
        return img_out, ref

    if squeeze_batch:
        return out[0], ref
    return out, ref


def chunk_staves_for_ribbon(
    staves: Any,
    max_staves: int = MAX_STAVES,
    overlap: int = 0,
) -> list[list[Any]]:
    """Slice a system's staves into ``<= max_staves`` chunks for 4-slot ribbon sampling.

    Enables 5-6+ staff systems (e.g. ``palestrina-agnus``, ``beach-prayer``,
    ``aloha-oe``) to be processed cleanly without raising ``ValueError``.

    Args:
        staves: ``CurvedSystem`` (with ``.staves``), sequence of staves, or single staff.
        max_staves: Maximum staves per chunk (default ``MAX_STAVES = 4``).
        overlap: Number of trailing staves to repeat at the start of the next
            chunk (``0 <= overlap < max_staves``).

    Returns:
        List of staff sub-lists, each of length ``1..max_staves``.
    """
    max_s = int(max_staves)
    ov = int(overlap)
    if max_s < 1:
        raise ValueError(f"max_staves must be >= 1, got {max_staves}")
    if ov < 0 or ov >= max_s:
        raise ValueError(f"overlap must satisfy 0 <= overlap < max_staves ({max_s}), got {overlap}")

    if hasattr(staves, "staves"):
        staff_list = list(staves.staves)
    elif isinstance(staves, Sequence):
        staff_list = list(staves)
    else:
        staff_list = [staves]

    n = len(staff_list)
    if n == 0:
        raise ValueError("staves must be non-empty")
    if n <= max_s:
        return [staff_list]

    step = max_s - ov
    chunks: list[list[Any]] = []
    start = 0
    while start < n:
        end = min(start + max_s, n)
        chunks.append(staff_list[start:end])
        if end >= n:
            break
        start += step
    return chunks


def sample_curved_system_chunks(
    page: torch.Tensor | Image.Image,
    staves: Any,
    max_staves: int = MAX_STAVES,
    overlap: int = 0,
    target_sp: float = REF_TARGET_SP,
    height: int | None = 256,
    width: int | None = None,
    offset_mode: str = "normal",
    paper_value: float | None = None,
    comb_snap: bool = False,
) -> list[tuple[torch.Tensor | Image.Image, torch.Tensor, list[int]]]:
    """Sample a curved system of any staff count (including >4) as ``<= max_staves`` ribbons.

    Returns:
        List of ``(ribbon, ref_channel, staff_indices)`` tuples where
        ``staff_indices`` are the 0-based indices of the staves in ``staves``
        placed into that ribbon's active slots ``0..len(staff_indices)-1``.
    """
    chunks = chunk_staves_for_ribbon(staves, max_staves=max_staves, overlap=overlap)
    step = int(max_staves) - int(overlap)
    results: list[tuple[torch.Tensor | Image.Image, torch.Tensor, list[int]]] = []
    start = 0
    for chunk in chunks:
        indices = list(range(start, start + len(chunk)))
        ribbon, ref = sample_bezier_ribbon(
            page=page,
            staves=chunk,
            target_sp=target_sp,
            height=height,
            width=width,
            n_slots=int(max_staves),
            offset_mode=offset_mode,
            paper_value=paper_value,
            comb_snap=comb_snap,
        )
        results.append((ribbon, ref, indices))
        start += step
    return results


class BezierRibbonSampler(nn.Module):
    """Differentiable nn.Module wrapper for Batch 5-B2 Bezier ribbon sampling."""

    def __init__(
        self,
        target_sp: float = REF_TARGET_SP,
        height: int | None = 256,
        n_slots: int = MAX_STAVES,
        offset_mode: str = "normal",
        paper_value: float = 1.0,
    ):
        super().__init__()
        self.target_sp = float(target_sp)
        self.height = height
        self.n_slots = int(n_slots)
        self.offset_mode = str(offset_mode)
        self.paper_value = float(paper_value)

    def forward(
        self,
        page: torch.Tensor,
        staves: Any,
        width: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, ref = sample_bezier_ribbon(
            page=page,
            staves=staves,
            target_sp=self.target_sp,
            height=self.height,
            width=width,
            n_slots=self.n_slots,
            offset_mode=self.offset_mode,
            paper_value=self.paper_value,
        )
        assert isinstance(out, torch.Tensor)
        return out, ref
