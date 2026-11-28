"""Gauge-RoPE: Diffeomorphism-Equivariant Intrinsic Rotary Embeddings for UniOMR v2.

Implements:
  1. ``GaugeFieldHead`` (~0.14M params, < 0.3 MB INT8):
     Lightweight anisotropic conv head predicting smooth intrinsic staff
     coordinate residuals ``(Δv̂, Δû)`` (perpendicular staff offset + arc-length
     along staff) directly from page images, paired with a zero-initialized
     residual gate ``alpha`` (exact Euclidean ``RoPE2D`` identity at step 0).
  2. ``GaugeRoPE2D`` & ``attach_gauge_rope(model)``:
     Clean, non-invasive wrapper around ``UniOMRv2Net.encoder.rope2d`` that
     computes continuous rotary phases at ``(y + α·Δv̂, x + α·Δû)`` with zero
     image resampling and strict FP32 trig safety under BF16 autocast.
  3. ``diffeomorphic_equivariance_loss(gauge_head, img_real)``:
     Self-supervised diffeomorphic equivariance objective for unlabeled RISM
     historical manuscripts: applies a random smooth composite warp ``φ``
     (sine staff droop + thin-plate spline control grid + mild affine) via
     ``F.grid_sample`` and minimizes masked Smooth-L1 discrepancy between
     ``g_θ(φ(I))`` and ``φ*(g_θ(I))``.
  4. ``apply_synthetic_droop(images, droop_frac=0.025)``:
     Deterministic synthetic ~3-staff-space vertical droop warp for OOD
     geometric robustness evaluation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GaugeFieldHead",
    "GaugeRoPE2D",
    "attach_gauge_rope",
    "get_gauge_head",
    "get_gauge_alpha",
    "sample_random_diffeomorphism",
    "diffeomorphic_equivariance_loss",
    "apply_synthetic_droop",
]


def _group_count(channels: int, max_groups: int = 8) -> int:
    g = min(max_groups, channels)
    while g > 1 and (channels % g != 0):
        g -= 1
    return g


class _AnisoGaugeBlock(nn.Module):
    """Lightweight anisotropic depthwise-separable block for tracking curved staves."""

    def __init__(self, channels: int, expand_ratio: float = 2.0) -> None:
        super().__init__()
        hidden = max(channels, int(round(channels * expand_ratio)))
        g = _group_count(channels, 8)
        self.dw_horiz = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels, bias=False)
        self.dw_vert = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels, bias=False)
        self.dw_local = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1), groups=channels, bias=False)
        self.gn1 = nn.GroupNorm(g, channels)
        self.pw1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.gn2 = nn.GroupNorm(g, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw_horiz(x) + self.dw_vert(x) + self.dw_local(x)
        h = self.gn1(h)
        h = self.pw2(F.silu(self.pw1(h)))
        h = self.gn2(h)
        return x + h


class GaugeFieldHead(nn.Module):
    """Lightweight (~0.14M params) intrinsic staff coordinate field estimator.

    Predicts dense normalized coordinate residuals ``(Δv̂, Δû)`` at ``/16`` page
    resolution bounded in ``[-max_disp, +max_disp]``, with a zero-initialized
    scalar residual gate ``alpha`` so ``y + α·Δv̂ == y`` at step 0.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Tuple[int, int, int] = (24, 48, 80),
        head_channels: int = 48,
        n_blocks: int = 2,
        max_disp: float = 0.15,
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.max_disp = float(max_disp)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(_group_count(c1), c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(c2), c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(c3), c3),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_AnisoGaugeBlock(c3, expand_ratio=2.0) for _ in range(n_blocks)])
        self.down4 = nn.Sequential(
            nn.Conv2d(c3, head_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(head_channels), head_channels),
            nn.SiLU(inplace=True),
        )
        # Non-zero small init on out_proj so d(loss)/d(alpha) is non-zero at step 0
        # while alpha=0.0 guarantees exact Euclidean RoPE2D identity!
        self.out_proj = nn.Conv2d(head_channels, 2, kernel_size=1, bias=True)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

        # Zero-initialized residual gate alpha (0.0 -> exact RoPE2D identity at init)
        self.alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Predict dense residual displacement field ``(B, 2, H/16, W/16)`` in ``[-max_disp, max_disp]``.

        Channel 0 is ``Δv̂`` (vertical / perpendicular to staff).
        Channel 1 is ``Δû`` (horizontal / arc-length along staff).
        """
        h = self.stem(images)
        h = self.blocks(h)
        h = self.down4(h)
        raw = self.out_proj(h)
        with torch.amp.autocast(device_type=raw.device.type, enabled=False):
            delta = torch.tanh(raw.float()) * self.max_disp
        return delta

    @staticmethod
    def make_base_grid(
        batch_size: int,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalized Euclidean coordinate grid ``(B, 2, H, W)`` in ``[0, 1]``: ch0=y, ch1=x."""
        ys = torch.linspace(0.0, 1.0, steps=max(1, h), device=device, dtype=torch.float32)
        xs = torch.linspace(0.0, 1.0, steps=max(1, w), device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gy, gx], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        return grid

    def predict_intrinsic_coords(
        self,
        images: torch.Tensor,
        return_delta: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return dense intrinsic coordinates ``g_θ(I) = (y, x) + (Δv̂, Δû)`` of shape ``(B, 2, H_g, W_g)``."""
        delta = self.forward(images)
        B, _, H_g, W_g = delta.shape
        base = self.make_base_grid(B, H_g, W_g, device=delta.device)
        coords = base + delta
        if return_delta:
            return coords, delta
        return coords


class GaugeRoPE2D(nn.Module):
    """Diffeomorphism-equivariant continuous 2D-RoPE wrapper around ``AspectNormalizedRoPE2D``.

    Computes rotary phases at continuous intrinsic coordinates:
        ``v̂ = y_norm + α · Δv̂(y_norm, x_norm)``
        ``û = x_norm + α · Δû(y_norm, x_norm)``
    with zero image resampling and strict FP32 trig under BF16/FP16 autocast.
    At ``α = 0.0``, reduces bit-identically to Euclidean ``AspectNormalizedRoPE2D``.
    """

    def __init__(
        self,
        orig_rope2d: nn.Module,
        gauge_head: GaugeFieldHead,
    ) -> None:
        super().__init__()
        self.coord_scale = float(getattr(orig_rope2d, "coord_scale", 128.0))
        self.rope_jitter = float(getattr(orig_rope2d, "rope_jitter", 0.0))
        inv_freq = orig_rope2d.inv_freq.detach().clone().to(torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.gauge_head = gauge_head
        self._current_images: Optional[torch.Tensor] = None
        self._last_delta_stats: Dict[str, float] = {}

    @property
    def alpha(self) -> nn.Parameter:
        return self.gauge_head.alpha

    def forward(
        self,
        coords: torch.Tensor,
        grid_hw: Tuple[int, int],
        dtype: torch.dtype,
        images: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute continuous Gauge-RoPE2D ``(cos, sin)`` for routed tokens ``coords: (B, L, 2)``."""
        h_p, w_p = int(grid_hw[0]), int(grid_hw[1])
        img = images if images is not None else self._current_images

        with torch.amp.autocast(device_type=coords.device.type, enabled=False):
            inv_f = self.inv_freq.to(coords.device, dtype=torch.float32)
            y_norm = coords[..., 0].to(torch.float32) / float(max(1, h_p - 1))
            x_norm = coords[..., 1].to(torch.float32) / float(max(1, w_p - 1))

            if img is not None:
                # Predict dense gauge field (B, 2, H_g, W_g) and bilinearly sample at routed coords
                delta_field = self.gauge_head(img)  # float32 (B, 2, H_g, W_g)
                # F.grid_sample expects (x, y) in [-1, 1] with shape (B, H_out, W_out, 2)
                sample_x = (x_norm * 2.0 - 1.0).clamp(-1.0, 1.0)
                sample_y = (y_norm * 2.0 - 1.0).clamp(-1.0, 1.0)
                sample_grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(2)  # (B, L, 1, 2)
                sampled = F.grid_sample(
                    delta_field,
                    sample_grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                ).squeeze(-1)  # (B, 2, L)
                delta_v = sampled[:, 0, :]
                delta_u = sampled[:, 1, :]
                alpha = self.gauge_head.alpha.to(torch.float32)
                v_hat = y_norm + alpha * delta_v
                u_hat = x_norm + alpha * delta_u
            else:
                v_hat = y_norm
                u_hat = x_norm

            if self.training and self.rope_jitter > 0.0:
                j = self.rope_jitter
                if v_hat.ndim >= 2:
                    bcast_shape = (v_hat.shape[0],) + (1,) * (v_hat.ndim - 1)
                else:
                    bcast_shape = ()
                scale_y = 1.0 + (torch.rand(bcast_shape, device=coords.device, dtype=torch.float32) * 2.0 - 1.0) * j
                scale_x = 1.0 + (torch.rand(bcast_shape, device=coords.device, dtype=torch.float32) * 2.0 - 1.0) * j
                shift_y = (torch.rand(bcast_shape, device=coords.device, dtype=torch.float32) * 2.0 - 1.0) * (0.5 * j)
                shift_x = (torch.rand(bcast_shape, device=coords.device, dtype=torch.float32) * 2.0 - 1.0) * (0.5 * j)
                v_hat = v_hat * scale_y + shift_y
                u_hat = u_hat * scale_x + shift_x

            fy = (v_hat * self.coord_scale).unsqueeze(-1) * inv_f
            fx = (u_hat * self.coord_scale).unsqueeze(-1) * inv_f
            emb_half = torch.stack([fy, fx], dim=-1).flatten(-2)
            emb = torch.cat([emb_half, emb_half], dim=-1)
            cos = emb.cos().to(dtype).unsqueeze(1)
            sin = emb.sin().to(dtype).unsqueeze(1)
            return cos, sin


def attach_gauge_rope(
    model: nn.Module,
    *,
    channels: Tuple[int, int, int] = (24, 48, 80),
    head_channels: int = 48,
    n_blocks: int = 2,
    max_disp: float = 0.15,
) -> GaugeFieldHead:
    """Attach ``GaugeFieldHead`` + ``GaugeRoPE2D`` cleanly onto a ``UniOMRv2Net`` without editing ``uni_omr_v2.py``.

    Idempotent: if already attached, returns the existing ``GaugeFieldHead``.
    """
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "rope2d"):
        raise ValueError("attach_gauge_rope expects a UniOMRv2Net with model.encoder.rope2d")

    if isinstance(encoder.rope2d, GaugeRoPE2D):
        return encoder.rope2d.gauge_head

    orig_rope2d = encoder.rope2d
    device = orig_rope2d.inv_freq.device
    gauge_head = GaugeFieldHead(
        in_channels=3,
        channels=channels,
        head_channels=head_channels,
        n_blocks=n_blocks,
        max_disp=max_disp,
    ).to(device)
    gauge_rope = GaugeRoPE2D(orig_rope2d=orig_rope2d, gauge_head=gauge_head).to(device)
    encoder.rope2d = gauge_rope

    # Wire encoder forward hooks so GaugeRoPE2D receives the current batch `images`
    # without modifying uni_omr_v2.py.
    def _pre_hook(mod: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        img = kwargs.get("images", None)
        if img is None and len(args) >= 2:
            img = args[1]
        mod.rope2d._current_images = img

    def _post_hook(mod: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any) -> None:
        mod.rope2d._current_images = None

    encoder.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    encoder.register_forward_hook(_post_hook, with_kwargs=True)
    return gauge_head


def get_gauge_head(model: nn.Module) -> Optional[GaugeFieldHead]:
    """Return attached ``GaugeFieldHead`` from a ``UniOMRv2Net``, or ``None`` if not attached."""
    encoder = getattr(model, "encoder", None)
    rope2d = getattr(encoder, "rope2d", None) if encoder is not None else None
    if isinstance(rope2d, GaugeRoPE2D):
        return rope2d.gauge_head
    return None


def get_gauge_alpha(model: nn.Module) -> float:
    """Return scalar float value of ``gauge_head.alpha`` (or 0.0 if not attached)."""
    gh = get_gauge_head(model)
    if gh is None:
        return 0.0
    return float(gh.alpha.detach().item())


# ---------------------------------------------------------------------------
# Self-Supervised Diffeomorphic Equivariance Loss (Unlabeled RISM Manuscripts)
# ---------------------------------------------------------------------------

def sample_random_diffeomorphism(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    *,
    max_droop: float = 0.045,
    max_tps: float = 0.035,
    max_rot_deg: float = 3.0,
    max_shear: float = 0.03,
    tps_grid_size: int = 4,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a smooth invertible 2D warp sampling grid ``(B, H, W, 2)`` in ``[-1, 1]``.

    Combines:
      1. Multi-frequency sinusoidal staff droop (vertical wave along x).
      2. Smooth bicubic thin-plate-like control grid displacement.
      3. Mild affine rotation + shear.
    """
    ys = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=torch.float32)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
    gx_b = gx.unsqueeze(0).expand(batch_size, -1, -1)
    gy_b = gy.unsqueeze(0).expand(batch_size, -1, -1)

    def _rand(shape: Tuple[int, ...]) -> torch.Tensor:
        return torch.rand(shape, device=device, dtype=torch.float32, generator=generator) * 2.0 - 1.0

    # 1. Mild affine rotation & shear
    ang = _rand((batch_size, 1, 1)) * (max_rot_deg * math.pi / 180.0)
    sh_x = _rand((batch_size, 1, 1)) * max_shear
    sh_y = _rand((batch_size, 1, 1)) * max_shear
    ca, sa = torch.cos(ang), torch.sin(ang)
    x_aff = ca * gx_b - sa * gy_b + sh_x * gy_b
    y_aff = sa * gx_b + ca * gy_b + sh_y * gx_b

    # 2. Sinusoidal staff droop (1st & 2nd harmonic in x, smooth y modulation)
    # Note: [-1, 1] coordinates are 2x [0, 1] coordinates
    a1 = _rand((batch_size, 1, 1)) * (2.0 * max_droop)
    a2 = _rand((batch_size, 1, 1)) * (1.0 * max_droop)
    p1 = _rand((batch_size, 1, 1)) * math.pi
    p2 = _rand((batch_size, 1, 1)) * math.pi
    droop_y = a1 * torch.sin(0.5 * math.pi * gx_b + p1) + a2 * torch.sin(math.pi * gx_b + p2)

    # 3. Smooth low-frequency bicubic control-point field (thin-plate surrogate)
    ctrl = _rand((batch_size, 2, tps_grid_size, tps_grid_size)) * (2.0 * max_tps)
    tps = F.interpolate(ctrl, size=(height, width), mode="bicubic", align_corners=True)
    dx_tps = tps[:, 0, :, :]
    dy_tps = tps[:, 1, :, :]

    grid_x = x_aff + dx_tps
    grid_y = y_aff + droop_y + dy_tps
    return torch.stack([grid_x, grid_y], dim=-1)


def diffeomorphic_equivariance_loss(
    gauge_head: GaugeFieldHead,
    img_real: torch.Tensor,
    *,
    max_droop: float = 0.045,
    max_tps: float = 0.035,
    max_rot_deg: float = 3.0,
    max_shear: float = 0.03,
    anchor_weight: float = 0.05,
    smooth_weight: float = 0.02,
    beta: float = 0.02,
    generator: Optional[torch.Generator] = None,
    return_stats: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """Self-supervised diffeomorphic equivariance loss on unlabeled real manuscripts.

    Applies a random smooth diffeomorphism ``φ`` via ``F.grid_sample``:
        ``I_φ(p) = I(ψ(p))``
    and minimizes masked Smooth-L1 error between ``g_θ(I_φ)(p)`` and
    ``stopgrad(g_θ(I))(ψ(p))``, where ``g_θ(I)(p) = p + Δ_θ(I)(p)``.
    Because ``g_θ`` includes the base Euclidean grid ``p``, trivial ``Δ_θ = 0``
    has non-zero error equal to the warp magnitude ``‖ψ(p) - p‖``.
    """
    B, _, H, W = img_real.shape
    device = img_real.device

    with torch.amp.autocast(device_type=device.type, enabled=False):
        img_f = img_real.float()
        grid = sample_random_diffeomorphism(
            B,
            H,
            W,
            device=device,
            max_droop=max_droop,
            max_tps=max_tps,
            max_rot_deg=max_rot_deg,
            max_shear=max_shear,
            generator=generator,
        )
        img_warped = F.grid_sample(
            img_f,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    # Forward gauge head on both original and warped manuscript images
    g_orig, delta_orig = gauge_head.predict_intrinsic_coords(img_real, return_delta=True)
    g_warped_pred, delta_warped = gauge_head.predict_intrinsic_coords(img_warped.to(img_real.dtype), return_delta=True)

    with torch.amp.autocast(device_type=device.type, enabled=False):
        _, _, H_g, W_g = g_orig.shape
        # Downsample sampling grid to gauge field resolution (H_g, W_g)
        grid_g = F.interpolate(
            grid.permute(0, 3, 1, 2),
            size=(H_g, W_g),
            mode="bilinear",
            align_corners=True,
        ).permute(0, 2, 3, 1)  # (B, H_g, W_g, 2)

        # Warp g_orig(I) by the exact same diffeomorphism (stop-grad on target prevents collapse/drift)
        g_target = F.grid_sample(
            g_orig.detach().float(),
            grid_g,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        # Mask out pixels sampled from outside the original image boundary
        valid = (
            (grid_g[..., 0].abs() < 0.96) & (grid_g[..., 1].abs() < 0.96)
        ).float().unsqueeze(1)  # (B, 1, H_g, W_g)

        per_elem = F.smooth_l1_loss(g_warped_pred.float(), g_target, beta=beta, reduction="none")
        eq_loss = (per_elem * valid).sum() / (valid.sum() * 2.0).clamp(min=1.0)

        # Gentle zero-mean anchor (prevents global translation drift) + spatial smoothness (TV)
        anchor_loss = (delta_orig.float().mean(dim=(2, 3)) ** 2).mean()
        tv_h = (delta_warped.float()[:, :, 1:, :] - delta_warped.float()[:, :, :-1, :]).abs().mean()
        tv_w = (delta_warped.float()[:, :, :, 1:] - delta_warped.float()[:, :, :, :-1]).abs().mean()
        smooth_loss = tv_h + tv_w

        total = eq_loss + anchor_weight * anchor_loss + smooth_weight * smooth_loss

    if return_stats:
        stats = {
            "eq_loss": float(eq_loss.detach().item()),
            "anchor_loss": float(anchor_loss.detach().item()),
            "smooth_loss": float(smooth_loss.detach().item()),
            "delta_abs_mean": float(delta_warped.detach().abs().mean().item()),
        }
        return total, stats
    return total


# ---------------------------------------------------------------------------
# Deterministic Synthetic Staff-Droop Warp (OOD Robustness Evaluation)
# ---------------------------------------------------------------------------

def apply_synthetic_droop(
    images: torch.Tensor,
    droop_frac: float = 0.025,
    cycles: float = 1.0,
    phase: float = 0.0,
) -> torch.Tensor:
    """Apply deterministic smooth vertical staff droop to ``(B, C, H, W)`` page images.

    ``droop_frac=0.025`` corresponds to ~32 px vertical droop on a 1280px page
    (~3 staff spaces on typical 6–8 system full-page scores).
    """
    B, _, H, W = images.shape
    device = images.device
    orig_dtype = images.dtype
    with torch.amp.autocast(device_type=device.type, enabled=False):
        ys = torch.linspace(-1.0, 1.0, steps=H, device=device, dtype=torch.float32)
        xs = torch.linspace(-1.0, 1.0, steps=W, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        # Smooth arch/sine droop along x: 0 at edges or full half-wave across page
        # (2.0 * droop_frac converts [0, 1] page fraction into [-1, 1] grid_sample units)
        dy = (2.0 * float(droop_frac)) * torch.sin(math.pi * float(cycles) * 0.5 * (gx + 1.0) + float(phase))
        grid_x = gx.unsqueeze(0).expand(B, -1, -1)
        grid_y = (gy + dy).unsqueeze(0).expand(B, -1, -1)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        warped = F.grid_sample(
            images.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
    return warped.to(orig_dtype)
