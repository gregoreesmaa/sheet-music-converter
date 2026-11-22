"""Compact hybrid sidecar format and on-the-fly fallback (storage fix).

Legacy ``.hybrid.pt`` sidecars store five uncompressed ``uint8`` arrays
via ``torch.save`` (``ZIP_STORED``): ``rgb [3, 256, W]`` (60%),
``gray [256, W]`` (20%), and ``staff [1, 256, W]`` (20%). At typical
rectified widths (~1,188 px), each file is ~1.52 MB, so 206,496 unified
pairs consume ~322 GB on disk.

Three redundancies drive that blowup:
1. ``staff`` is 100% deterministic from ``(S, height, width, target_sp)``
   (five horizontal 1-px rows per active staff via ``rectify.canon_y``).
2. ``gray`` is 100% deterministic from ``rgb`` via PIL's ``.convert('L')``.
3. ``rgb`` is stored uncompressed even though ~90% of pixels are white
   paper (including pure-white inactive slots) and ~42% of strips are
   pure grayscale (``R == G == B``).

This module provides:
* **Compact sidecars (``<stem>.hybrid.npz``):** lossless PNG (default,
  ~13x smaller, ~2.6x faster to load than ``torch.load``) or lossless
  WebP (~19.5x smaller) encoding of the rectified strip (auto-selecting
  1-channel ``L`` mode when ``R == G == B`` else ``RGB``), tiny JSON
  metadata, deterministic ``staff`` and ``gray`` reconstruction (0.0
  max abs error), and optional ``uint8``-quantized + zlib-compressed
  7-channel HeatNet heatmaps (max abs error <= 0.5/255 < 1/255).
* **On-the-fly mode:** when no sidecar file exists on disk, rectifies
  directly from the source strip image + ``hybrid_staff.jsonl`` geometry
  row in memory (0 GB of sidecar storage on disk).
* **Full backward compatibility:** transparently reads legacy
  ``.hybrid.pt`` files and provides a legacy-to-compact converter.
"""

from __future__ import annotations

import io
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rectify as R  # noqa: E402

COMPACT_SUFFIX = '.hybrid.npz'
LEGACY_SUFFIX = '.hybrid.pt'
FORMAT_VERSION = 1

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _default_standardize_rgb(chw: torch.Tensor) -> torch.Tensor:
    """ImageNet standardization matching ``train.standardize_rgb``."""
    return chw.sub_(_IMAGENET_MEAN).div_(_IMAGENET_STD)


def compact_sidecar_path_for(img_path) -> Path:
    """``<stem>.hybrid.npz`` alongside ``img_path``."""
    p = Path(img_path)
    return p.parent / (p.stem + COMPACT_SUFFIX)


def legacy_sidecar_path_for(img_path) -> Path:
    """``<stem>.hybrid.pt`` alongside ``img_path``."""
    p = Path(img_path)
    return p.parent / (p.stem + LEGACY_SUFFIX)


def resolve_sidecar_path(img_path) -> Path | None:
    """Return compact sidecar path if present, else legacy, else None."""
    c = compact_sidecar_path_for(img_path)
    if c.exists():
        return c
    leg = legacy_sidecar_path_for(img_path)
    if leg.exists():
        return leg
    return None


def reconstruct_staff_grid(n_staff: int, height: int, width: int,
                           target_sp: float = 10.0) -> torch.Tensor:
    """Deterministically rebuild ``[1, height, width]`` float32 staff map.

    Bit-identical to the ``ref`` tensor returned by
    :func:`rectify.rectify_system_strip`.
    """
    ref = torch.zeros((1, int(height), int(width)), dtype=torch.float32)
    s_clamped = max(0, min(int(n_staff), R.MAX_STAVES))
    for s in range(s_clamped):
        for k in range(R.LINES_PER_STAFF):
            r = int(round(R.canon_y(s, k, float(target_sp))))
            if 0 <= r < height:
                ref[0, r, :] = 1.0
    return ref


def quantize_heat(probs: torch.Tensor) -> tuple[bytes, tuple[int, int, int]]:
    """Quantize ``[C, H, W]`` probabilities in ``[0, 1]`` to zlib-packed uint8.

    Max absolute reconstruction error is ``<= 0.5 / 255`` (~0.00196).
    """
    if probs.dim() != 3:
        raise ValueError(f'heat probs must be [C, H, W], got {tuple(probs.shape)}')
    q = torch.clamp(torch.round(probs.detach().cpu().float() * 255.0),
                    0.0, 255.0).to(torch.uint8).contiguous()
    raw = q.numpy().tobytes()
    return zlib.compress(raw, level=6), tuple(int(d) for d in q.shape)


def dequantize_heat(compressed: bytes, shape: tuple[int, int, int],
                    target_hw: tuple[int, int] | None = None) -> torch.Tensor:
    """Restore ``[C, H, W]`` float32 probabilities from zlib-packed uint8."""
    raw = zlib.decompress(compressed)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(tuple(int(d) for d in shape))
    probs = torch.from_numpy(arr.copy()).float().div_(255.0)
    if target_hw is not None and tuple(probs.shape[1:]) != tuple(target_hw):
        probs = torch.nn.functional.interpolate(
            probs.unsqueeze(0), size=tuple(target_hw),
            mode='bilinear', align_corners=False)[0]
    return probs


def _encode_image_lossless(rect_img: Image.Image, codec: str = 'png',
                           compress_level: int = 6) -> tuple[bytes, str]:
    """Encode rectified PIL image losslessly, collapsing exact grayscale to L."""
    rgb_img = rect_img.convert('RGB')
    arr = np.asarray(rgb_img, dtype=np.uint8)
    is_gray = bool(
        np.array_equal(arr[:, :, 0], arr[:, :, 1])
        and np.array_equal(arr[:, :, 0], arr[:, :, 2])
    )
    img_to_save = Image.fromarray(arr[:, :, 0], mode='L') if is_gray else rgb_img
    buf = io.BytesIO()
    codec_l = codec.lower()
    if codec_l == 'png':
        img_to_save.save(buf, format='PNG', compress_level=int(compress_level))
    elif codec_l == 'webp':
        img_to_save.save(buf, format='WEBP', lossless=True, quality=50)
    else:
        raise ValueError(f'unsupported compact sidecar codec: {codec!r}')
    return buf.getvalue(), ('L' if is_gray else 'RGB')


def _atomic_savez(dest: Path, **arrays) -> None:
    """Write an uncompressed ``.npz`` atomically via tmp + fsync + replace."""
    dest = Path(dest)
    tmp = dest.with_suffix(dest.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dest)


def save_compact_sidecar(
    dest,
    rect_img: Image.Image,
    *,
    img_name: str,
    S: int,
    sp: float = 0.0,
    kind: str = '',
    height: int = 256,
    target_sp: float = 10.0,
    codec: str = 'png',
    compress_level: int = 6,
    heat_probs: torch.Tensor | None = None,
    heat_half_res: bool = False,
) -> Path:
    """Write a compact ``.hybrid.npz`` sidecar atomically.

    Stores lossless compressed rectified image bytes (PNG or WebP, ``L``
    mode when grayscale else ``RGB``), JSON metadata, and optionally
    ``uint8``-quantized zlib-compressed 7-channel HeatNet probabilities.
    ``gray`` and ``staff`` are omitted and reconstructed bit-exactly on load.
    """
    dest = Path(dest)
    img_bytes, mode = _encode_image_lossless(
        rect_img, codec=codec, compress_level=compress_level)
    w_out, h_out = rect_img.size
    if int(h_out) != int(height):
        raise ValueError(
            f'rect_img height {h_out} != declared height {height}')
    meta = {
        'v': FORMAT_VERSION,
        'img': str(img_name),
        'S': int(S),
        'sp': float(sp),
        'kind': str(kind),
        'height': int(height),
        'width': int(w_out),
        'target_sp': float(target_sp),
        'codec': codec.lower(),
        'mode': mode,
    }
    arrays = {
        'img': np.frombuffer(img_bytes, dtype=np.uint8),
    }
    if heat_probs is not None:
        hp = heat_probs
        if heat_half_res and hp.shape[1] >= 2 and hp.shape[2] >= 2:
            hp = torch.nn.functional.interpolate(
                hp.unsqueeze(0),
                size=(max(1, hp.shape[1] // 2), max(1, hp.shape[2] // 2)),
                mode='bilinear', align_corners=False)[0]
        hz, hshape = quantize_heat(hp)
        meta['heat_shape'] = list(hshape)
        arrays['heat_z'] = np.frombuffer(hz, dtype=np.uint8)
    meta_bytes = json.dumps(meta, separators=(',', ':')).encode('utf-8')
    arrays['meta'] = np.frombuffer(meta_bytes, dtype=np.uint8)
    _atomic_savez(dest, **arrays)
    return dest


def _parse_nslot(raw_s) -> int:
    """Match ``train.load_sidecar`` slot fallback: ``1..4`` else ``4``."""
    try:
        nslot = int(raw_s)
    except (TypeError, ValueError):
        return 4
    return nslot if 1 <= nslot <= 4 else 4


def _decode_pil_to_rgb_gray(pil_img: Image.Image, standardize_fn=None):
    """PIL image -> ``(rgb [3,H,W] standardized, gray [1,H,W] /255)``."""
    std_fn = standardize_fn or _default_standardize_rgb
    rgb_img = pil_img.convert('RGB')
    gray_img = rgb_img.convert('L')
    rgb_u8 = np.ascontiguousarray(np.array(rgb_img, dtype=np.uint8, copy=True))
    gray_u8 = np.ascontiguousarray(np.array(gray_img, dtype=np.uint8, copy=True))
    rgb = std_fn(
        torch.from_numpy(rgb_u8).permute(2, 0, 1).contiguous().float().div_(255.0))
    gray = torch.from_numpy(gray_u8).float().div_(255.0).unsqueeze(0)
    return rgb, gray


def load_compact_sidecar(
    path,
    height: int,
    standardize_fn=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Load ``(rgb, gray, staff, S)`` from a compact ``.hybrid.npz`` file.

    Output shapes, dtypes, and numerical values match ``train.load_sidecar``
    on legacy ``.hybrid.pt`` files bit-for-bit.
    """
    path = Path(path)
    with np.load(str(path), allow_pickle=False) as z:
        meta = json.loads(bytes(z['meta'].tolist()).decode('utf-8'))
        img_bytes = z['img'].tobytes()
    if int(meta['height']) != int(height):
        raise ValueError(
            f'sidecar {path} height {meta["height"]} != train height {height}')
    pil_img = Image.open(io.BytesIO(img_bytes))
    rgb, gray = _decode_pil_to_rgb_gray(pil_img, standardize_fn=standardize_fn)
    h, w = int(gray.shape[1]), int(gray.shape[2])
    raw_s = meta.get('S', 4)
    nslot = _parse_nslot(raw_s)
    staff_s = int(raw_s) if isinstance(raw_s, (int, float)) else nslot
    target_sp = float(meta.get('target_sp', 10.0))
    staff = reconstruct_staff_grid(staff_s, h, w, target_sp=target_sp)
    return rgb, gray, staff, nslot


def load_compact_heat(
    path,
    target_hw: tuple[int, int] | None = None,
) -> torch.Tensor | None:
    """Load cached quantized ``[7, H, W]`` heatmaps from ``.hybrid.npz`` if present."""
    path = Path(path)
    if not path.exists():
        return None
    with np.load(str(path), allow_pickle=False) as z:
        if 'heat_z' not in z:
            return None
        meta = json.loads(bytes(z['meta'].tolist()).decode('utf-8'))
        shape = tuple(int(d) for d in meta['heat_shape'])
        raw = z['heat_z'].tobytes()
    return dequantize_heat(raw, shape, target_hw=target_hw)


def rectify_on_the_fly(
    img_path,
    row: dict,
    height: int = 256,
    target_sp: float = 10.0,
    max_width: int = 8000,
    standardize_fn=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Compute ``(rgb, gray, staff, S)`` in memory from source image + table row.

    Zero sidecar disk footprint. Produces outputs bit-identical to
    materialized ``.hybrid.pt`` and ``.hybrid.npz`` sidecars.
    """
    import strip_staff as sst  # local import avoids circular import at module load

    img = Image.open(img_path).convert('RGB')
    table = row.get('table') or {}
    if int(table.get('w_in', -1)) != img.size[0]:
        raise ValueError(
            f'w_in {table.get("w_in")} != img width {img.size[0]} for {img_path}')
    curves = sst.table_to_curves(table)
    rect_img, ref = R.rectify_system_strip(
        img, curves, target_sp=target_sp, height=height)
    if rect_img.size[0] > int(max_width):
        raise ValueError(
            f'rectified width {rect_img.size[0]} > cap {max_width} for {img_path}')
    rgb, gray = _decode_pil_to_rgb_gray(rect_img, standardize_fn=standardize_fn)
    staff = (ref > 0.5).float()
    if staff.dim() == 2:
        staff = staff.unsqueeze(0)
    nslot = _parse_nslot(row.get('S', len(curves)))
    return rgb, gray, staff, nslot


def load_legacy_sidecar(
    path,
    height: int,
    standardize_fn=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Load ``(rgb, gray, staff, S)`` from a legacy ``.hybrid.pt`` file."""
    std_fn = standardize_fn or _default_standardize_rgb
    blob = torch.load(str(path), map_location='cpu', weights_only=True)
    if int(blob['height']) != int(height):
        raise ValueError(
            f'sidecar {path} height {blob["height"]} != train height {height}')
    rgb = std_fn(blob['rgb'].float().div_(255.0))
    gray = blob['gray'].float().div_(255.0).unsqueeze(0)
    staff = blob['staff'].float()
    if staff.dim() == 2:
        staff = staff.unsqueeze(0)
    nslot = _parse_nslot(blob.get('S', 4))
    return rgb, gray, staff, nslot


def load_or_compute_sidecar(
    img_path,
    height: int,
    *,
    hybrid_index: dict | None = None,
    target_sp: float = 10.0,
    max_width: int = 8000,
    standardize_fn=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Unified loader: compact ``.hybrid.npz`` -> legacy ``.hybrid.pt`` -> on-the-fly."""
    cpath = compact_sidecar_path_for(img_path)
    if cpath.exists():
        return load_compact_sidecar(cpath, height, standardize_fn=standardize_fn)
    lpath = legacy_sidecar_path_for(img_path)
    if lpath.exists():
        return load_legacy_sidecar(lpath, height, standardize_fn=standardize_fn)
    if hybrid_index is not None:
        name = Path(img_path).name
        row = hybrid_index.get(name)
        if row is not None:
            return rectify_on_the_fly(
                img_path, row, height=height, target_sp=target_sp,
                max_width=max_width, standardize_fn=standardize_fn)
    raise FileNotFoundError(
        f'no compact ({cpath}), legacy ({lpath}), or index entry for {img_path}')


def convert_legacy_sidecar(
    pt_path,
    dest_path=None,
    *,
    codec: str = 'png',
    compress_level: int = 6,
    remove_legacy: bool = False,
) -> Path:
    """Convert an existing ``.hybrid.pt`` file to a compact ``.hybrid.npz`` file."""
    pt_path = Path(pt_path)
    if dest_path is None:
        name = pt_path.name
        if name.endswith(LEGACY_SUFFIX):
            dest_path = pt_path.with_name(name[:-len(LEGACY_SUFFIX)] + COMPACT_SUFFIX)
        else:
            dest_path = pt_path.with_suffix(COMPACT_SUFFIX)
    dest_path = Path(dest_path)
    blob = torch.load(str(pt_path), map_location='cpu', weights_only=True)
    rgb_np = blob['rgb'].permute(1, 2, 0).contiguous().numpy()
    rect_img = Image.fromarray(rgb_np, mode='RGB')
    save_compact_sidecar(
        dest_path,
        rect_img,
        img_name=str(blob.get('img', pt_path.stem)),
        S=int(blob.get('S', 4)),
        sp=float(blob.get('sp', 0.0)),
        kind=str(blob.get('kind', '')),
        height=int(blob['height']),
        codec=codec,
        compress_level=compress_level,
    )
    if remove_legacy and dest_path.exists():
        pt_path.unlink()
    return dest_path
