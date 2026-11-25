"""InternVL 3.5 ViR+ViCO 2-Tier Resolution Router + VisionZip Contextual Merging + Static Token Buckets.

Standalone drop-in replacement for ``InkTokenRouterV2`` in ``HybridAxialMergerEncoder``
(Iter-11/12 ablation). Strictly pure float32 <= 4D tensor ops:
  - Zero ``bool`` ``Where`` / ``~`` / ``&`` ops (WebGPU/NPU/ONNX opset-17 native)
  - Zero ``SpaceToDepth`` / 5D+ permutes
  - Zero CUDA-CPU syncs during training

Components:
  1. ``ViRTwoTierRouter``:
     - Pure-float32 3x3 eroded ``paper_gate`` (matching ``InkTokenRouterV2``).
     - Tier 1 (Fine ``2x2`` /16 ink tokens): selects top ``k_fine`` high-saliency
       ink patches snapped to static token buckets (default ``(256, 512, 768,
       1024, 1536, 2048)``) for WebGPU Graph Capture & NPU static compilation.
     - Tier 2 (Coarse ``4x4`` /32 context + VisionZip contextual merging):
       downsamples the full grid ``2x2 -> 4x4`` via ``AvgPool2d(2) + Linear``
       weighted by ``(1 - kept_mask)`` and merges non-anchor coarse cells into
       ``k_coarse`` (default ``96``, typical ``64..128``) spatial context tokens
       carrying accurate 2D ``(y, x)`` coordinates in the ``/16`` grid frame.
     - Concatenates ``[fine_tokens, coarse_tokens]`` with aligned 2D coordinates
       and a learned 2-entry ``tier_embed`` (``0=fine, 1=coarse``).
  2. ``compute_vico_consistency_loss``:
     - Training-only (0% inference cost) InternVL 3.5 ViCO KL consistency loss
       aligning decoder logits under aggressive 2-tier routing to dense logits.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DEFAULT_STATIC_BUCKETS",
    "snap_to_static_bucket",
    "erode_paper_gate_f32",
    "compute_ink_and_paper_gate_f32",
    "ViRTwoTierRouter",
    "ViRTwoTierRouterONNX",
    "compute_vico_consistency_loss",
]

DEFAULT_STATIC_BUCKETS: Tuple[int, ...] = (256, 512, 768, 1024, 1536, 2048)


def _group_count(channels: int, max_groups: int = 8) -> int:
    g = min(max_groups, channels)
    while g > 1 and (channels % g != 0):
        g -= 1
    return g


def _expand_onnx_safe(t: torch.Tensor, *sizes: int) -> torch.Tensor:
    """Zero-copy ``expand`` in eager mode; ``Tile``-based ``repeat`` during ONNX export.

    PyTorch's legacy ONNX exporter lowers ``Tensor.expand`` into ``Equal + Where``
    on shape tensors; using ``repeat`` during export keeps the ONNX graph 100% free
    of ``Where`` / ``Equal`` ops while preserving exact zero-copy ``expand`` in training.
    """
    if torch.onnx.is_in_onnx_export():
        reps = tuple(
            1 if (int(s) == -1 or int(s) == int(cur)) else int(s) // max(1, int(cur))
            for cur, s in zip(t.shape, sizes)
        )
        if all(r == 1 for r in reps):
            return t
        return t.repeat(*reps)
    return t.expand(*sizes)


class _RMSNormF32(nn.Module):
    """Explicit opset-17 RMSNorm (no custom contrib op)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def snap_to_static_bucket(
    k_raw: int,
    buckets: Sequence[int] = DEFAULT_STATIC_BUCKETS,
    max_k: Optional[int] = None,
) -> int:
    """Snap a requested token count ``k_raw`` up to the smallest static bucket ``>= k_raw``.

    If ``k_raw`` exceeds the largest bucket, returns the largest bucket.
    Finally clamps to ``[1, max_k]`` when ``max_k`` is specified.
    """
    if not buckets:
        k = max(1, int(k_raw))
        return min(k, max_k) if max_k is not None else k

    sorted_buckets = sorted(int(b) for b in buckets if int(b) > 0)
    if not sorted_buckets:
        k = max(1, int(k_raw))
        return min(k, max_k) if max_k is not None else k

    target = max(1, int(k_raw))
    chosen = sorted_buckets[-1]
    for b in sorted_buckets:
        if b >= target:
            chosen = b
            break
    if max_k is not None:
        chosen = max(1, min(chosen, int(max_k)))
    return chosen


def erode_paper_gate_f32(
    paper_f: torch.Tensor,
    bg_floor: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure float32 3x3 morphological erosion of a ``(B, 1, H, W)`` paper mask.

    Zero ``bool`` ``~`` / ``&`` / ``Where`` ops (WebGPU & ONNX opset-17 native).
    Falls back to the uneroded ``paper_f`` per sample if erosion wipes out the
    entire interior.

    Returns:
        ``(paper_gate, effective_paper_f)`` both float32 ``(B, 1, H, W)``.
    """
    paper_f = paper_f.to(torch.float32)
    non_paper_f = 1.0 - paper_f
    border_dilated_f = F.max_pool2d(non_paper_f, kernel_size=3, stride=1, padding=1)
    interior_f = paper_f * (1.0 - border_dilated_f)
    has_interior_f = interior_f.amax(dim=(1, 2, 3), keepdim=True)
    effective_paper_f = has_interior_f * interior_f + (1.0 - has_interior_f) * paper_f
    paper_gate = float(bg_floor) + (1.0 - float(bg_floor)) * effective_paper_f
    return paper_gate, effective_paper_f


def compute_ink_and_paper_gate_f32(
    images: torch.Tensor,
    h_p: int,
    w_p: int,
    bg_floor: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure float32 relative high-pass ink map + 3x3 eroded ``paper_gate``.

    Matches ``InkTokenRouterV2._compute_ink_and_paper_gate`` and
    ``export_uni_v2_onnx._compute_ink_and_paper_gate_f32`` with zero ``bool``
    ``Where`` ops. Uses exact fixed-stride pooling when ``(H, W) == (16*h_p, 16*w_p)``
    (ONNX static/WebGPU path) and adaptive pooling otherwise.

    Returns:
        ``(local_ink_grid, paper_gate, effective_paper_f)`` all float32 ``(B, 1, h_p, w_p)``.
    """
    img_f = images.to(torch.float32)
    if img_f.shape[1] == 3:
        luma = (
            0.299 * img_f[:, 0:1]
            + 0.587 * img_f[:, 1:2]
            + 0.114 * img_f[:, 2:3]
        )
    else:
        luma = img_f.mean(dim=1, keepdim=True)

    local_bg = F.avg_pool2d(
        F.avg_pool2d(
            luma,
            kernel_size=(1, 31),
            stride=1,
            padding=(0, 15),
            count_include_pad=False,
        ),
        kernel_size=(31, 1),
        stride=1,
        padding=(15, 0),
        count_include_pad=False,
    )
    paper_mask = (local_bg > 0.35).to(torch.float32)
    local_ink = ((local_bg - luma) / local_bg.clamp(min=0.15)).clamp(0.0, 1.0) * paper_mask

    in_h, in_w = int(img_f.shape[2]), int(img_f.shape[3])
    if in_h == h_p * 16 and in_w == w_p * 16:
        ks = (16, 16)
        ink_max = F.max_pool2d(local_ink, kernel_size=ks, stride=ks)
        ink_avg = F.avg_pool2d(local_ink, kernel_size=ks, stride=ks)
        local_bg_grid = F.avg_pool2d(local_bg, kernel_size=ks, stride=ks)
    else:
        ink_max = F.adaptive_max_pool2d(local_ink, (h_p, w_p))
        ink_avg = F.adaptive_avg_pool2d(local_ink, (h_p, w_p))
        local_bg_grid = F.adaptive_avg_pool2d(local_bg, (h_p, w_p))

    local_ink_grid = 0.5 * ink_max + 0.5 * ink_avg
    paper_f = (local_bg_grid > 0.35).to(torch.float32)
    paper_gate, effective_paper_f = erode_paper_gate_f32(paper_f, bg_floor=bg_floor)
    local_ink_grid = local_ink_grid * effective_paper_f
    return local_ink_grid.detach(), paper_gate.detach(), effective_paper_f.detach()


class ViRTwoTierRouter(nn.Module):
    """InternVL 3.5 ViR 2-Tier Resolution Router + VisionZip Contextual Merging + Static Buckets.

    Replaces binary keep-or-drop routing with:
      - **Tier 1 (Fine ``/16`` ink tokens):** Top ``k_fine`` high-saliency patches
        (snapped to static token buckets when ``static_buckets=True``).
      - **Tier 2 (Coarse ``/32`` context + VisionZip merging):** Full-grid ``2x2 -> 4x4``
        masked average pooling weighted by ``(1 - kept_mask)`` followed by a learned
        ``coarse_proj`` and VisionZip cosine+spatial contextual merging into ``k_coarse``
        spatial context tokens with accurate 2D ``(y, x)`` coordinates.
      - Learned 2-entry ``tier_embed`` (``0=fine``, ``1=coarse``) added before concatenation.

    Drop-in compatible with ``HybridAxialMergerEncoder`` (accepts ``(x_2d, images=...,
    keep_ratio_override=...)``) and also supports direct ``(x_2d, saliency_map=...)``
    invocation for standalone/ONNX usage.
    """

    def __init__(
        self,
        d_model: int,
        keep_ratio: float = 0.40,
        k_coarse: int = 96,
        static_buckets: bool = True,
        fine_buckets: Sequence[int] = DEFAULT_STATIC_BUCKETS,
        use_ink_prior: bool = True,
        dynamic_routing: bool = True,
        min_keep_ratio: float = 0.15,
        max_keep_ratio: float = 0.75,
        saliency_thr: float = 0.05,
        max_mem_tokens: int = 2176,
        router_hidden: int = 64,
        merge_temp: float = 0.25,
        spatial_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        if k_coarse < 1:
            raise ValueError(f"k_coarse must be >= 1, got {k_coarse}")
        if max_mem_tokens <= k_coarse:
            raise ValueError(
                f"max_mem_tokens ({max_mem_tokens}) must exceed k_coarse ({k_coarse})"
            )

        self.d_model = int(d_model)
        self.keep_ratio = float(keep_ratio)
        self.k_coarse = int(k_coarse)
        self.static_buckets = bool(static_buckets)
        self.fine_buckets: Tuple[int, ...] = tuple(sorted(int(b) for b in fine_buckets if int(b) > 0))
        self.use_ink_prior = bool(use_ink_prior)
        self.dynamic_routing = bool(dynamic_routing)
        self.min_keep_ratio = float(min_keep_ratio)
        self.max_keep_ratio = float(max_keep_ratio)
        self.saliency_thr = float(saliency_thr)
        self.max_mem_tokens = int(max_mem_tokens)
        self.n_bg_summary = int(k_coarse)  # Drop-in alias for HybridAxialMergerEncoder telemetry
        self.merge_temp = float(max(1e-3, merge_temp))
        self.spatial_gamma = float(max(0.0, spatial_gamma))

        hidden = max(32, int(router_hidden))
        self.saliency_net = nn.Sequential(
            nn.Conv2d(self.d_model, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden, 8), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        self.ink_prior_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # Tier-2 coarse projection: AvgPool2d(2) (applied in forward) + Linear + RMSNorm
        self.coarse_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.coarse_norm = _RMSNormF32(self.d_model)

        # Learned 2-entry tier embedding: index 0 = fine (Tier 1), index 1 = coarse (Tier 2)
        self.tier_embed = nn.Parameter(torch.randn(2, self.d_model) * 0.02)

        self._grid_cache: Dict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_coord_grid(self, h_m: int, w_m: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cached ``(coord_flat (1, N, 2), coord_2d (1, 2, H_m, W_m))`` in float32."""
        if torch.onnx.is_in_onnx_export():
            ys = torch.arange(h_m, device=device, dtype=torch.float32)
            xs = torch.arange(w_m, device=device, dtype=torch.float32)
            grid_y = _expand_onnx_safe(ys.unsqueeze(1), h_m, w_m)
            grid_x = _expand_onnx_safe(xs.unsqueeze(0), h_m, w_m)
            coord_2d = torch.stack([grid_y, grid_x], dim=0).unsqueeze(0)
            coord_flat = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1).unsqueeze(0)
            return coord_flat, coord_2d

        gkey = (h_m, w_m, device)
        cached = self._grid_cache.get(gkey)
        if cached is not None:
            return cached
        ys = torch.arange(h_m, device=device, dtype=torch.float32)
        xs = torch.arange(w_m, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coord_2d = torch.stack([grid_y, grid_x], dim=0).unsqueeze(0)
        coord_flat = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1).unsqueeze(0)
        self._grid_cache[gkey] = (coord_flat, coord_2d)
        return coord_flat, coord_2d

    def resolve_k_fine(
        self,
        n_total: int,
        score_flat: Optional[torch.Tensor] = None,
        keep_ratio_override: Optional[float] = None,
        k_fine_override: Optional[int] = None,
        k_coarse_actual: Optional[int] = None,
    ) -> int:
        """Determine Tier-1 ``k_fine`` with optional static-bucket snapping.

        Strictly avoids device-to-host syncs during training and ONNX export.
        """
        kc = self.k_coarse if k_coarse_actual is None else int(k_coarse_actual)
        max_fine = max(1, min(n_total, self.max_mem_tokens - kc))

        if k_fine_override is not None:
            k_raw = int(k_fine_override)
        elif keep_ratio_override is not None:
            ratio = float(keep_ratio_override)
            if not (0.0 < ratio <= 1.0):
                raise ValueError(f"keep_ratio must be in (0, 1], got {ratio}")
            k_raw = max(1, int(math.ceil(n_total * ratio)))
        elif self.dynamic_routing:
            if self.training or torch.onnx.is_in_onnx_export() or score_flat is None:
                # Zero CUDA-CPU sync during training & ONNX export
                k_raw = max(1, int(math.ceil(n_total * self.keep_ratio)))
            else:
                with torch.no_grad():
                    med = score_flat.median(dim=1, keepdim=True).values
                    above = (score_flat > (med + self.saliency_thr)).to(torch.float32).sum(dim=1)
                    k_dyn = int(above.max().item())
                    k_min = max(1, int(math.ceil(n_total * self.min_keep_ratio)))
                    k_max = max(k_min, min(n_total, int(math.ceil(n_total * self.max_keep_ratio))))
                    k_raw = max(k_min, min(k_max, k_dyn))
        else:
            k_raw = max(1, int(math.ceil(n_total * self.keep_ratio)))

        if self.static_buckets:
            return snap_to_static_bucket(k_raw, self.fine_buckets, max_k=max_fine)
        return max(1, min(k_raw, max_fine))

    def _compute_saliency(
        self,
        x_2d: torch.Tensor,
        saliency_map: Optional[torch.Tensor],
        images: Optional[torch.Tensor],
        paper_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute pure-float32 saliency score ``(B, 1, H_m, W_m)`` with 3x3 paper_gate erosion."""
        B, _, H_m, W_m = x_2d.shape
        saliency_logits: Optional[torch.Tensor] = None
        ink_target: Optional[torch.Tensor] = None

        if saliency_map is not None:
            score = saliency_map.to(torch.float32)
            if score.ndim == 3:
                score = score.unsqueeze(1)
        else:
            saliency_logits = self.saliency_net(x_2d).to(torch.float32)
            score = torch.sigmoid(saliency_logits)

        if images is not None:
            local_ink_grid, paper_gate, effective_paper_f = compute_ink_and_paper_gate_f32(
                images, H_m, W_m
            )
            score = score * paper_gate
            ink_target = ((local_ink_grid > 0.04).to(torch.float32) * effective_paper_f)
            if self.use_ink_prior and saliency_map is None:
                flat_f = local_ink_grid.flatten(1)
                mean_f = flat_f.mean(dim=-1, keepdim=True)
                var_f = ((flat_f - mean_f) * (flat_f - mean_f)).mean(dim=-1, keepdim=True)
                std_f = torch.sqrt(var_f.clamp(min=0.0))
                denom = (mean_f + 2.5 * std_f).clamp(min=1e-4).view(B, 1, 1, 1)
                local_ink_norm = (local_ink_grid / denom).clamp(0.0, 1.0)
                score = score + torch.sigmoid(self.ink_prior_scale) * local_ink_norm
        elif paper_mask is not None:
            pm = paper_mask.to(torch.float32)
            if pm.ndim == 3:
                pm = pm.unsqueeze(1)
            paper_gate, _ = erode_paper_gate_f32(pm)
            score = score * paper_gate

        return score, saliency_logits, ink_target

    def _build_coarse_tier(
        self,
        x_2d: torch.Tensor,
        score_2d: torch.Tensor,
        kept_mask: torch.Tensor,
        coord_2d: torch.Tensor,
        k_coarse_target: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Tier 2: 2x2 -> 4x4 masked AvgPool2d + Linear + VisionZip spatial contextual merging.

        All operations are pure float32 <= 4D (zero ``bool`` ``Where``, zero 5D+ permutes).
        """
        B, D, H_m, W_m = x_2d.shape
        unkept_w = (1.0 - kept_mask).view(B, 1, H_m, W_m)  # (B, 1, H_m, W_m), float32

        pad_h = H_m % 2
        pad_w = W_m % 2
        if pad_h or pad_w:
            pad_arg = (0, pad_w, 0, pad_h)
            x_pad = F.pad(x_2d, pad_arg, mode="constant", value=0.0)
            w_pad = F.pad(unkept_w, pad_arg, mode="constant", value=0.0)
            s_pad = F.pad(score_2d, pad_arg, mode="constant", value=0.0)
            c_pad = F.pad(_expand_onnx_safe(coord_2d, B, 2, H_m, W_m), pad_arg, mode="replicate")
        else:
            x_pad = x_2d
            w_pad = unkept_w
            s_pad = score_2d
            c_pad = _expand_onnx_safe(coord_2d, B, 2, H_m, W_m)

        H_c = (H_m + pad_h) // 2
        W_c = (W_m + pad_w) // 2
        N_c = H_c * W_c

        # Masked 2x2 -> 4x4 pooling weighted by (1 - kept_mask)
        coarse_den = F.avg_pool2d(w_pad, kernel_size=2, stride=2)  # (B, 1, H_c, W_c) in [0, 1]
        has_unkept = (coarse_den * 4.0).clamp(0.0, 1.0)            # 0.0 if all 4 kept, 1.0 if >=1 unkept

        coarse_num = F.avg_pool2d(x_pad * w_pad, kernel_size=2, stride=2)
        coarse_avg_unkept = coarse_num / coarse_den.clamp(min=1e-4)
        coarse_avg_all = F.avg_pool2d(x_pad, kernel_size=2, stride=2)
        coarse_feat_2d = has_unkept * coarse_avg_unkept + (1.0 - has_unkept) * coarse_avg_all

        # Accurate 2D (y, x) coordinates in the /16 frame (unkept centroid when unkept > 0)
        coord_num = F.avg_pool2d(c_pad * w_pad, kernel_size=2, stride=2)
        coord_unkept = coord_num / coarse_den.clamp(min=1e-4)
        coord_center = F.avg_pool2d(c_pad, kernel_size=2, stride=2)
        coarse_coord_2d = has_unkept * coord_unkept + (1.0 - has_unkept) * coord_center

        # Coarse saliency score prioritizes unkept faint ink (staff lines, slurs) + unkept mass
        coarse_sal_2d = F.avg_pool2d(s_pad * w_pad, kernel_size=2, stride=2)
        coarse_score_flat = (coarse_sal_2d + 0.05 * coarse_den).flatten(1)  # (B, N_c)
        coarse_den_flat = coarse_den.flatten(1)                             # (B, N_c)

        # Lightweight learned projection: AvgPool2d(2) -> Linear -> RMSNorm
        coarse_flat_in = coarse_feat_2d.permute(0, 2, 3, 1).reshape(B, N_c, D)
        coarse_all = self.coarse_norm(self.coarse_proj(coarse_flat_in))     # (B, N_c, D)
        coarse_coords_all = coarse_coord_2d.permute(0, 2, 3, 1).reshape(B, N_c, 2)

        k_c = max(1, min(int(k_coarse_target), N_c))
        if N_c <= k_c:
            anc_idx = _expand_onnx_safe(
                torch.arange(N_c, device=x_2d.device, dtype=torch.int64).unsqueeze(0), B, N_c
            )
            return coarse_all, coarse_coords_all, anc_idx, (H_c, W_c)

        # VisionZip contextual merging: select k_c spatial anchors, merge remaining N_c - k_c cells
        _, topk_c = torch.topk(coarse_score_flat, k=k_c, dim=1, largest=True, sorted=False)
        anc_idx, _ = torch.sort(topk_c, dim=1)  # raster order

        anc_feat = torch.gather(coarse_all, 1, _expand_onnx_safe(anc_idx.unsqueeze(-1), B, k_c, D))
        anc_coords = torch.gather(coarse_coords_all, 1, _expand_onnx_safe(anc_idx.unsqueeze(-1), B, k_c, 2))

        # Pure-float32 non-anchor mask (0.0 at anchors, 1.0 at merged cells)
        anc_mask = torch.zeros(B, N_c, device=x_2d.device, dtype=torch.float32)
        anc_mask = anc_mask.scatter(1, anc_idx, torch.ones((B, k_c), device=x_2d.device, dtype=torch.float32))
        non_anc_w = 1.0 - anc_mask  # (B, N_c)

        # Feature cosine similarity + normalized spatial proximity bias
        # Explicit pure-float32 L2 norm (avoids ONNX Equal/Where from F.normalize)
        u_all = coarse_all * torch.rsqrt(coarse_all.pow(2).sum(dim=-1, keepdim=True) + 1e-12)
        u_anc = anc_feat * torch.rsqrt(anc_feat.pow(2).sum(dim=-1, keepdim=True) + 1e-12)
        sim = torch.bmm(u_all, u_anc.transpose(1, 2))             # (B, N_c, k_c)

        if self.spatial_gamma > 0.0:
            inv_h = 1.0 / float(max(1, H_m - 1))
            inv_w = 1.0 / float(max(1, W_m - 1))
            all_y = coarse_coords_all[:, :, 0:1] * inv_h  # (B, N_c, 1)
            all_x = coarse_coords_all[:, :, 1:2] * inv_w
            anc_y = anc_coords[:, :, 0:1].transpose(1, 2) * inv_h  # (B, 1, k_c)
            anc_x = anc_coords[:, :, 1:2].transpose(1, 2) * inv_w
            dist2 = (all_y - anc_y).pow(2) + (all_x - anc_x).pow(2)  # (B, N_c, k_c)
            sim = sim - self.spatial_gamma * dist2

        assign = F.softmax(sim / self.merge_temp, dim=-1)  # (B, N_c, k_c)
        w_src = (non_anc_w * (coarse_den_flat + 1e-3)).unsqueeze(-1)  # (B, N_c, 1)

        assign_t = assign.transpose(1, 2)  # (B, k_c, N_c)
        merged_mass = torch.bmm(assign_t, w_src)                       # (B, k_c, 1)
        merged_feat = torch.bmm(assign_t, coarse_all * w_src)          # (B, k_c, D)
        merged_coord = torch.bmm(assign_t, coarse_coords_all * w_src)  # (B, k_c, 2)

        denom = 1.0 + merged_mass
        coarse_tokens = (anc_feat + merged_feat) / denom
        coarse_coords = (anc_coords + merged_coord) / denom
        return coarse_tokens, coarse_coords, anc_idx, (H_c, W_c)

    def forward_tensors(
        self,
        x_2d: torch.Tensor,
        saliency_map: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        paper_mask: Optional[torch.Tensor] = None,
        keep_ratio_override: Optional[float] = None,
        k_fine_override: Optional[int] = None,
        k_coarse_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Core pure-float32 forward returning ``(out_tokens, out_coords, coords_y, coords_x, info)``."""
        x_f32 = x_2d.to(torch.float32)
        B, D, H_m, W_m = x_f32.shape
        N = H_m * W_m

        score_2d, saliency_logits, ink_target = self._compute_saliency(
            x_f32, saliency_map=saliency_map, images=images, paper_mask=paper_mask
        )
        score_flat = score_2d.flatten(1)  # (B, N)
        tokens_flat = x_f32.permute(0, 2, 3, 1).reshape(B, N, D)

        coord_flat_base, coord_2d = self._get_coord_grid(H_m, W_m, x_f32.device)
        grid_coords = _expand_onnx_safe(coord_flat_base, B, N, 2)

        H_c_est = (H_m + (H_m % 2)) // 2
        W_c_est = (W_m + (W_m % 2)) // 2
        N_c_est = H_c_est * W_c_est
        k_coarse_req = self.k_coarse if k_coarse_override is None else int(k_coarse_override)
        k_coarse_actual = max(1, min(k_coarse_req, N_c_est))

        k_fine = self.resolve_k_fine(
            n_total=N,
            score_flat=score_flat,
            keep_ratio_override=keep_ratio_override,
            k_fine_override=k_fine_override,
            k_coarse_actual=k_coarse_actual,
        )

        # Tier 1: top k_fine high-saliency fine /16 tokens in raster order
        _, topk_idx = torch.topk(score_flat, k=k_fine, dim=1, largest=True, sorted=False)
        kept_idx, _ = torch.sort(topk_idx, dim=1)

        kept_scores = torch.gather(score_flat, 1, kept_idx)
        kept_gate = (0.5 + 0.5 * torch.sigmoid(kept_scores)).unsqueeze(-1)
        fine_tokens = torch.gather(tokens_flat, 1, _expand_onnx_safe(kept_idx.unsqueeze(-1), B, k_fine, D)) * kept_gate
        fine_coords = torch.gather(grid_coords, 1, _expand_onnx_safe(kept_idx.unsqueeze(-1), B, k_fine, 2))

        # Pure-float32 kept mask (1.0 at kept Tier-1 tokens, 0.0 elsewhere)
        kept_mask = torch.zeros(B, N, device=x_f32.device, dtype=torch.float32)
        kept_mask = kept_mask.scatter(1, kept_idx, torch.ones((B, k_fine), device=x_f32.device, dtype=torch.float32))

        # Tier 2: coarse 4x4 (/32) context + VisionZip contextual merging
        coarse_tokens, coarse_coords, coarse_idx, coarse_hw = self._build_coarse_tier(
            x_2d=x_f32,
            score_2d=score_2d,
            kept_mask=kept_mask,
            coord_2d=coord_2d,
            k_coarse_target=k_coarse_actual,
        )

        # Add learned 2-entry tier_embed (0=fine, 1=coarse)
        fine_embed = self.tier_embed[0].view(1, 1, D)
        coarse_embed = self.tier_embed[1].view(1, 1, D)
        fine_tokens = fine_tokens + fine_embed
        coarse_tokens = coarse_tokens + coarse_embed

        out_tokens = torch.cat([fine_tokens, coarse_tokens], dim=1)
        out_coords = torch.cat([fine_coords, coarse_coords], dim=1)
        coords_y = out_coords[:, :, 0]
        coords_x = out_coords[:, :, 1]

        if x_2d.dtype != torch.float32:
            out_tokens = out_tokens.to(x_2d.dtype)

        # Optional eval-only ink recall telemetry (never syncs during training or ONNX export)
        ink_recall = 1.0
        if (
            ink_target is not None
            and not self.training
            and not torch.onnx.is_in_onnx_export()
        ):
            with torch.no_grad():
                tgt = (ink_target.flatten(1) > 0.5).to(torch.float32)
                n_ink = tgt.sum(dim=1)
                kept_ink = (tgt * kept_mask).sum(dim=1)
                has_ink = (n_ink > 0.0).to(torch.float32)
                per_sample = has_ink * (kept_ink / n_ink.clamp(min=1.0)) + (1.0 - has_ink)
                ink_recall = float(per_sample.mean().item())

        info: Dict[str, Any] = {
            "n_in": N,
            "n_out": k_fine + k_coarse_actual,
            "k_kept": k_fine,
            "k_fine": k_fine,
            "k_coarse": k_coarse_actual,
            "n_bg_summary": k_coarse_actual,
            "kept_idx": kept_idx,
            "coarse_idx": coarse_idx,
            "kept_mask": kept_mask,
            "saliency_map": score_2d,
            "saliency_logits": saliency_logits,
            "ink_target": ink_target,
            "ink_recall": ink_recall,
            "grid_hw": (H_m, W_m),
            "coarse_grid_hw": coarse_hw,
            "coords_y": coords_y,
            "coords_x": coords_x,
        }
        return out_tokens, out_coords, coords_y, coords_x, info

    def forward(
        self,
        feat_2d: torch.Tensor,
        saliency_map: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        paper_mask: Optional[torch.Tensor] = None,
        keep_ratio_override: Optional[float] = None,
        k_fine_override: Optional[int] = None,
        k_coarse_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Drop-in ``InkTokenRouterV2`` signature returning ``(out_tokens, out_coords, info)``."""
        out_tokens, out_coords, _, _, info = self.forward_tensors(
            x_2d=feat_2d,
            saliency_map=saliency_map,
            images=images,
            paper_mask=paper_mask,
            keep_ratio_override=keep_ratio_override,
            k_fine_override=k_fine_override,
            k_coarse_override=k_coarse_override,
        )
        return out_tokens, out_coords, info


class ViRTwoTierRouterONNX(nn.Module):
    """Pure-tensor ONNX opset-17 export wrapper around ``ViRTwoTierRouter``.

    Takes ``(x_2d, saliency_map)`` (plus optional ``paper_mask``) and returns
    ``(out_tokens, coords_y, coords_x)`` with 100% pure float32 ops.
    """

    def __init__(
        self,
        router: ViRTwoTierRouter,
        static_k_fine: Optional[int] = None,
        static_k_coarse: Optional[int] = None,
        use_paper_mask: bool = False,
    ) -> None:
        super().__init__()
        self.router = router
        self.static_k_fine = static_k_fine
        self.static_k_coarse = static_k_coarse
        self.use_paper_mask = bool(use_paper_mask)

    def forward(
        self,
        x_2d: torch.Tensor,
        saliency_map: torch.Tensor,
        paper_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pm = paper_mask if self.use_paper_mask else None
        out_tokens, _, coords_y, coords_x, _ = self.router.forward_tensors(
            x_2d=x_2d,
            saliency_map=saliency_map,
            paper_mask=pm,
            k_fine_override=self.static_k_fine,
            k_coarse_override=self.static_k_coarse,
        )
        return out_tokens, coords_y, coords_x


def compute_vico_consistency_loss(
    logits_dense: torch.Tensor,
    logits_routed: torch.Tensor,
    temp: float = 1.5,
    valid_mask: Optional[torch.Tensor] = None,
    symmetric: bool = True,
    detach_dense: bool = True,
) -> torch.Tensor:
    """InternVL 3.5 ViCO training-only KL consistency loss (0% inference cost).

    Aligns decoder logits produced under aggressive 2-tier routing (``logits_routed``)
    to reference dense/high-keep logits (``logits_dense``) using temperature-scaled
    KL divergence (scaled by ``temp ** 2``).

    Args:
        logits_dense: ``(..., V)`` reference dense/high-keep decoder logits.
        logits_routed: ``(..., V)`` aggressively routed decoder logits (same shape).
        temp: Softmax temperature ``> 0`` (default ``1.5``).
        valid_mask: Optional ``(...)`` float/bool mask selecting non-PAD positions.
        symmetric: If ``True`` (default), uses symmetric KL
            ``0.5 * (KL(P_dense || Q_routed) + KL(Q_routed || P_dense))``;
            if ``False``, uses forward KL ``KL(P_dense || Q_routed)``.
        detach_dense: If ``True`` (default), stops gradients through ``logits_dense``
            so the dense teacher is not degraded by the routed student.

    Returns:
        Scalar float32 loss tensor (``0.0`` when ``logits_dense == logits_routed``).
    """
    if logits_dense.shape != logits_routed.shape:
        raise ValueError(
            f"Shape mismatch: logits_dense {tuple(logits_dense.shape)} vs "
            f"logits_routed {tuple(logits_routed.shape)}"
        )
    t = float(temp)
    if t <= 0.0:
        raise ValueError(f"temp must be > 0, got {temp}")

    dense_src = logits_dense.detach() if detach_dense else logits_dense
    p_log = F.log_softmax(dense_src.to(torch.float32) / t, dim=-1)
    q_log = F.log_softmax(logits_routed.to(torch.float32) / t, dim=-1)

    p_prob = p_log.exp()
    kl_fwd = (p_prob * (p_log - q_log)).sum(dim=-1).clamp(min=0.0)
    if symmetric:
        q_prob = q_log.exp()
        kl_rev = (q_prob * (q_log - p_log)).sum(dim=-1).clamp(min=0.0)
        kl = 0.5 * (kl_fwd + kl_rev)
    else:
        kl = kl_fwd

    loss_per_pos = kl * (t * t)
    if valid_mask is not None:
        m = valid_mask.to(torch.float32)
        return (loss_per_pos * m).sum() / m.sum().clamp(min=1.0)
    return loss_per_pos.mean()
