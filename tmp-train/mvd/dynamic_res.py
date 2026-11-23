"""Native dynamic-resolution preprocessing, aspect bucketing, and ONNX export for Stage-1 StaffUNet2D.

Pure NumPy + Pillow + PyTorch (CPU/GPU compatible, zero heuristics):
  - ``DISCRETE_AR_BUCKETS``: 5 stride-32 aligned ``(H, W)`` training buckets.
  - ``compute_dynamic_hw``: isotropic pixel-budget + max-side sizing with stride-32 padding.
  - ``preprocess_page_dynamic``: anti-aliased BOX/LANCZOS isotropic resize + paper-border pad.
  - ``unpad_probs_dynamic``: crop padded heatmap predictions back to the valid scaled region.
  - ``choose_ar_bucket``: select the discrete ``(H, W)`` bucket closest in log aspect ratio.
  - ``AspectBucketBatchSampler``: deterministic per-epoch aspect-bucketed batch sampler
    (uniform stride-32 ``(H, W)`` per batch, zero Windows WDDM allocator fragmentation).
  - ``sp_orig_to_log_hm`` / ``log_hm_to_sp_orig``: scale-invariant log heatmap staff-space transforms.
  - ``export_staff_unet_dynamic_onnx``: dynamic-axis ONNX export for arbitrary stride-32 pages.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Sampler

__all__ = [
    "DISCRETE_AR_BUCKETS",
    "compute_dynamic_hw",
    "estimate_border_color",
    "preprocess_page_dynamic",
    "unpad_probs_dynamic",
    "choose_ar_bucket",
    "AspectBucketBatchSampler",
    "sp_orig_to_log_hm",
    "log_hm_to_sp_orig",
    "export_staff_unet_dynamic_onnx",
]

# Five stride-32 aligned (H, W) buckets covering portrait, square, landscape,
# tall portrait, and wide landscape/spread aspect ratios (all <= ~1.81 MP).
DISCRETE_AR_BUCKETS: list[tuple[int, int]] = [
    (1536, 1152),
    (1344, 1344),
    (1152, 1536),
    (1728, 1024),
    (1024, 1728),
]


def compute_dynamic_hw(
    orig_w: int,
    orig_h: int,
    pixel_budget: int = 1536 * 1152,
    max_side: int = 2048,
    stride: int = 32,
) -> dict[str, int | float]:
    """Compute isotropic resized dimensions and stride-aligned padded dimensions.

    Args:
        orig_w: Original page width in pixels.
        orig_h: Original page height in pixels.
        pixel_budget: Maximum pixel area ``scaled_w * scaled_h`` before padding.
        max_side: Maximum allowed side length ``max(scaled_w, scaled_h)``.
        stride: Spatial divisibility requirement for network input (default 32).

    Returns:
        Dict with ``orig_w, orig_h, scaled_w, scaled_h, in_w, in_h, scale, stride``.
    """
    ow = int(orig_w)
    oh = int(orig_h)
    st = int(stride)
    if ow <= 0 or oh <= 0:
        raise ValueError(f"orig_w and orig_h must be > 0, got ({ow}, {oh})")
    if st <= 0:
        raise ValueError(f"stride must be > 0, got {st}")

    scale = min(
        math.sqrt(float(pixel_budget) / float(ow * oh)),
        float(max_side) / float(max(ow, oh)),
        2.0,
    )

    scaled_w = max(st, int(round(ow * scale)))
    scaled_h = max(st, int(round(oh * scale)))
    # Ensure rounding never nudges pre-pad area over pixel_budget or side over max_side.
    if scaled_w * scaled_h > pixel_budget or max(scaled_w, scaled_h) > max_side:
        scaled_w = max(st, int(math.floor(ow * scale)))
        scaled_h = max(st, int(math.floor(oh * scale)))

    in_w = int(math.ceil(scaled_w / float(st)) * st)
    in_h = int(math.ceil(scaled_h / float(st)) * st)

    return {
        "orig_w": ow,
        "orig_h": oh,
        "scaled_w": scaled_w,
        "scaled_h": scaled_h,
        "in_w": in_w,
        "in_h": in_h,
        "scale": float(scale),
        "stride": st,
    }


def estimate_border_color(arr: np.ndarray) -> np.ndarray | float:
    """Estimate paper/border color as the median of the 1-pixel perimeter of ``arr``.

    Works for both ``[H, W, C]`` and ``[H, W]`` arrays.
    """
    if arr.ndim == 2:
        border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]], axis=0)
        return float(np.median(border))
    border = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0)
    return np.median(border, axis=0)


def preprocess_page_dynamic(
    img: Image.Image | str | Path,
    pixel_budget: int = 1536 * 1152,
    max_side: int = 2048,
    stride: int = 32,
    in_channels: int = 3,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Isotropically resize and right/bottom-pad a page image to stride-aligned dimensions.

    Uses ``Image.Resampling.BOX`` when downscaling (preserving thin 1-2 px staff lines)
    and ``Image.Resampling.LANCZOS`` when upscaling. Pads right/bottom with the median
    border color so no artificial high-contrast edge is introduced.

    Returns:
        ``(tensor, meta)`` where ``tensor`` is ``[C, in_h, in_w]`` float32 in ``[0, 1]``.
    """
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    mode = "L" if int(in_channels) == 1 else "RGB"
    work = img if img.mode == mode else img.convert(mode)
    orig_w, orig_h = work.size
    meta = compute_dynamic_hw(
        orig_w, orig_h, pixel_budget=pixel_budget, max_side=max_side, stride=stride
    )
    sw, sh = int(meta["scaled_w"]), int(meta["scaled_h"])
    in_w, in_h = int(meta["in_w"]), int(meta["in_h"])

    if orig_w != sw or orig_h != sh:
        resample = Image.Resampling.BOX if (orig_w > sw or orig_h > sh) else Image.Resampling.LANCZOS
        work = work.resize((sw, sh), resample)

    arr = np.asarray(work, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        if in_h > sh or in_w > sw:
            fill = float(estimate_border_color(arr))
            canvas = np.full((in_h, in_w), fill, dtype=np.float32)
            canvas[:sh, :sw] = arr
            arr = canvas
        ten = torch.from_numpy(arr[None, :, :])
    else:
        if in_h > sh or in_w > sw:
            fill_rgb = np.asarray(estimate_border_color(arr), dtype=np.float32)
            canvas = np.empty((in_h, in_w, arr.shape[2]), dtype=np.float32)
            canvas[:, :, :] = fill_rgb[None, None, :]
            canvas[:sh, :sw, :] = arr
            arr = canvas
        ten = torch.from_numpy(arr.transpose(2, 0, 1))

    return ten, meta


def unpad_probs_dynamic(
    probs: torch.Tensor | np.ndarray,
    meta: dict[str, int | float],
    out_stride: int = 2,
) -> torch.Tensor | np.ndarray:
    """Crop padded heatmap predictions back to the valid scaled content region."""
    hp_valid = max(1, int(round(float(meta["scaled_h"]) / float(out_stride))))
    wp_valid = max(1, int(round(float(meta["scaled_w"]) / float(out_stride))))
    hp_valid = min(hp_valid, int(probs.shape[-2]))
    wp_valid = min(wp_valid, int(probs.shape[-1]))
    return probs[..., :hp_valid, :wp_valid]


def choose_ar_bucket(
    orig_w: int,
    orig_h: int,
    buckets: Sequence[tuple[int, int]] = DISCRETE_AR_BUCKETS,
) -> tuple[int, int]:
    """Pick the discrete ``(H, W)`` bucket minimizing ``|log((orig_h/orig_w) / (bh/bw))|``."""
    if not buckets:
        raise ValueError("buckets must be non-empty")
    orig_ar = float(orig_h) / max(float(orig_w), 1e-6)
    best = min(
        buckets,
        key=lambda b: abs(math.log(orig_ar / (float(b[0]) / float(b[1])))),
    )
    return (int(best[0]), int(best[1]))


class AspectBucketBatchSampler(Sampler[list[int]]):
    """Deterministic seeded per-epoch batch sampler grouping items by aspect-ratio bucket.

    Every yielded batch contains indices assigned to the same discrete stride-32
    ``(H, W)`` bucket, eliminating variable-shape padding waste and Windows WDDM
    allocator fragmentation.
    """

    def __init__(
        self,
        sizes: Sequence[tuple[int, int]],
        batch_size: int,
        buckets: Sequence[tuple[int, int]] = DISCRETE_AR_BUCKETS,
        drop_last: bool = False,
        seed: int = 42,
        shuffle: bool = True,
    ):
        self.sizes = [(int(w), int(h)) for (w, h) in sizes]
        self.batch_size = max(1, int(batch_size))
        self.buckets = [(int(bh), int(bw)) for (bh, bw) in buckets]
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

        self.bucket_groups: dict[tuple[int, int], list[int]] = {}
        self.index_buckets: list[tuple[int, int]] = []
        for idx, (w, h) in enumerate(self.sizes):
            b = choose_ar_bucket(w, h, self.buckets)
            self.index_buckets.append(b)
            self.bucket_groups.setdefault(b, []).append(idx)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def bucket_for_index(self, idx: int) -> tuple[int, int]:
        return self.index_buckets[int(idx)]

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng((self.seed, self.epoch))
        batches: list[list[int]] = []
        for b in sorted(self.bucket_groups.keys()):
            idxs = list(self.bucket_groups[b])
            if self.shuffle and len(idxs) > 1:
                perm = rng.permutation(len(idxs))
                idxs = [idxs[int(i)] for i in perm]
            for start in range(0, len(idxs), self.batch_size):
                chunk = idxs[start : start + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                batches.append(chunk)
        if self.shuffle and len(batches) > 1:
            order = rng.permutation(len(batches))
            batches = [batches[int(i)] for i in order]
        yield from batches

    def __len__(self) -> int:
        total = 0
        for idxs in self.bucket_groups.values():
            n = len(idxs)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


def sp_orig_to_log_hm(
    sp_orig: float | np.ndarray,
    scale: float,
    out_stride: int = 2,
) -> float | np.ndarray:
    """Convert original-page staff spacing ``sp_orig`` to ``log(sp_hm)`` heatmap pixels."""
    is_scalar = np.ndim(sp_orig) == 0
    sp_hm = np.maximum(np.asarray(sp_orig, dtype=np.float64) * float(scale) / float(out_stride), 1e-3)
    out = np.log(sp_hm)
    return float(out) if is_scalar else out


def log_hm_to_sp_orig(
    log_sp_hm: float | np.ndarray,
    scale: float,
    out_stride: int = 2,
) -> float | np.ndarray:
    """Invert ``log(sp_hm)`` heatmap pixels back to original-page staff spacing ``sp_orig``."""
    is_scalar = np.ndim(log_sp_hm) == 0
    out = np.exp(np.asarray(log_sp_hm, dtype=np.float64)) * float(out_stride) / float(scale)
    return float(out) if is_scalar else out


def export_staff_unet_dynamic_onnx(
    net: torch.nn.Module,
    out_path: str | Path,
    opset: int = 17,
) -> Path:
    """Export ``StaffUNet2D`` to ONNX with dynamic batch, height, and width axes."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    was_training = net.training
    net.eval()
    try:
        try:
            dev = next(net.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        in_ch = int(getattr(net, "in_channels", 3))
        dummy = torch.randn(1, in_ch, 256, 192, device=dev)
        torch.onnx.export(
            net,
            dummy,
            str(out),
            input_names=["x"],
            output_names=["logits"],
            dynamic_axes={
                "x": {0: "B", 2: "H", 3: "W"},
                "logits": {0: "B", 2: "Hp", 3: "Wp"},
            },
            opset_version=int(opset),
            dynamo=False,
        )
    finally:
        net.train(was_training)
    return out
