"""T4.1 Render-and-Diff Measure Verifier (CPU-only, Verovio + resvg-py).

Re-renders any predicted ``bekern`` strip back to a crisp grayscale PNG via
``assembly-serialize bekern2musicxml`` + in-process ``verovio`` + ``resvg_py``
and compares its ink structure against the observed rectified strip.

Capabilities:
1. ``bekern_to_strip_png``: fast, deterministic, fail-soft bekern -> grayscale
   strip rasterizer (~40-120 ms/strip on CPU, zero GPL/MuseScore).
2. ``visual_similarity``: flat-field adaptive ink binarization (works on clean
   and faded/shadowed phone-photo strips), ink-bbox alignment, 2D FFT shift
   search (``±max_shift`` px), soft dilated ink IoU/F1, and per-bar/per-slice
   residual flagging.
3. ``rerank_beams_visual``: combines beam log-probabilities with visual IoU
   (``combined_score = logprob + alpha * visual_iou``) to rerank beam
   candidates and report suspect measures/slices.

Strictly CPU-only (numpy + scipy + Pillow + verovio + resvg_py). No CUDA.
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
from PIL import Image
from scipy import ndimage

__all__ = [
    'default_assembly_serialize_bin',
    'bekern_to_strip_png',
    'visual_similarity',
    'rerank_beams_visual',
]

_VEROVIO_LOCAL = threading.local()


def default_assembly_serialize_bin() -> Path:
    """Locate the release ``assembly-serialize`` binary relative to repo root."""
    repo = Path(__file__).resolve().parents[2]
    exe = 'assembly-serialize.exe' if os.name == 'nt' else 'assembly-serialize'
    return repo / 'converter' / 'assembly-serialize' / 'target' / 'release' / exe


def _get_verovio_toolkit(font: str = 'Leipzig'):
    """Return a thread-local Verovio toolkit configured for single-strip rendering."""
    import verovio

    tk = getattr(_VEROVIO_LOCAL, 'tk', None)
    cur_font = getattr(_VEROVIO_LOCAL, 'font', None)
    if tk is None or cur_font != font:
        tk = verovio.toolkit()
        tk.setOptions({
            'font': font,
            'breaks': 'none',
            'adjustPageHeight': True,
            'adjustPageWidth': True,
            'footer': 'none',
            'header': 'none',
            'pageMarginTop': 0,
            'pageMarginBottom': 0,
            'pageMarginLeft': 0,
            'pageMarginRight': 0,
            'scale': 40,
        })
        _VEROVIO_LOCAL.tk = tk
        _VEROVIO_LOCAL.font = font
    return tk


def _normalize_tokens(toks: Union[str, Sequence[str]]) -> list[str]:
    if isinstance(toks, str):
        return toks.split()
    return [str(t).strip() for t in toks if str(t).strip()]


def _count_bekern_bars(toks: Sequence[str]) -> int:
    """Estimate number of measures in a bekern token list."""
    # Ignore leading repeat-left barline right after header before first note.
    bar_indices = [
        i for i, t in enumerate(toks)
        if t == 'bar' or t.startswith('bar-')
    ]
    if not bar_indices:
        return 1
    # Count internal/closing barlines that follow at least one non-header token.
    non_header_seen = False
    n_bars = 0
    for t in toks:
        if t == 'bar' or t.startswith('bar-'):
            if non_header_seen:
                n_bars += 1
        elif not (
            t.startswith('clef-')
            or t.startswith('key')
            or t.startswith('time-')
            or t.startswith('voice-')
            or t.startswith('staff-')
        ):
            non_header_seen = True
    # If tokens continue after the last barline, count the trailing measure.
    last_bar = bar_indices[-1]
    if last_bar < len(toks) - 1:
        n_bars += 1
    return max(1, n_bars)


def bekern_to_strip_png(
    toks: Union[str, Sequence[str]],
    aser_bin: Optional[Union[str, Path]] = None,
    target_h: int = 256,
    font: str = 'Leipzig',
) -> Optional[np.ndarray]:
    """Render bekern tokens to a tightly-cropped grayscale ``uint8`` strip.

    Pipeline: ``bekern`` -> ``assembly-serialize bekern2musicxml`` ->
    in-process ``verovio`` SVG -> ``resvg_py`` PNG -> tight crop to notation
    band -> resize to ``target_h``.

    Fail-soft: returns ``None`` on empty/invalid bekern, conversion failure,
    or rendering error without raising.
    """
    tok_list = _normalize_tokens(toks)
    if not tok_list or target_h <= 0:
        return None

    exe = Path(aser_bin) if aser_bin is not None else default_assembly_serialize_bin()
    if not exe.exists():
        return None

    try:
        with tempfile.TemporaryDirectory(prefix='rdv_') as td:
            td_path = Path(td)
            bk_path = td_path / 'strip-bekern.txt'
            bk_path.write_text(' '.join(tok_list) + '\n', encoding='utf-8')

            proc = subprocess.run(
                [
                    str(exe),
                    'bekern2musicxml',
                    '--input', str(bk_path),
                    '--out', str(td_path),
                    '--stem', 'strip',
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            xml_path = td_path / 'strip.musicxml'
            if proc.returncode != 0 or not xml_path.exists():
                return None
            musicxml = xml_path.read_text(encoding='utf-8')
            if not musicxml.strip():
                return None
    except Exception:
        return None

    try:
        import resvg_py

        tk = _get_verovio_toolkit(font=font)
        if not tk.loadData(musicxml):
            return None
        if tk.getPageCount() < 1:
            return None
        svg = tk.renderToSVG(1)
        if not svg or '<svg' not in svg:
            return None

        png_bytes = resvg_py.svg_to_bytes(svg_string=svg, height=int(target_h))
        rgba = Image.open(io.BytesIO(png_bytes)).convert('RGBA')
        bg = Image.new('RGB', rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        gray = np.asarray(bg.convert('L'), dtype=np.uint8)
    except Exception:
        return None

    # Tightly crop to dark ink band (with tiny padding) and resize to target_h.
    ink = gray < 245
    if not ink.any():
        return None
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    pad_y = max(2, int(round(0.04 * (y1 - y0))))
    pad_x = max(2, int(round(0.01 * (x1 - x0))))
    h, w = gray.shape
    y0 = max(0, y0 - pad_y)
    y1 = min(h, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(w, x1 + pad_x)
    cropped = gray[y0:y1, x0:x1]

    ch, cw = cropped.shape
    if ch <= 0 or cw <= 0:
        return None
    if ch != int(target_h):
        new_w = max(1, int(round(cw * (float(target_h) / float(ch)))))
        pil = Image.fromarray(cropped, mode='L').resize(
            (new_w, int(target_h)), Image.Resampling.BILINEAR
        )
        cropped = np.asarray(pil, dtype=np.uint8)

    return cropped


def _to_gray_float(img: Union[np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(img, Image.Image):
        arr = np.asarray(img.convert('L'), dtype=np.float32)
    else:
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=2)
    if arr.ndim != 2:
        raise ValueError(f'Expected 2D grayscale image, got shape {arr.shape}')
    if arr.size > 0 and arr.max() <= 1.0 and arr.min() >= 0.0:
        arr = arr * 255.0
    return arr


def _otsu_threshold(flat_u8: np.ndarray) -> float:
    """Compute Otsu binarization threshold on a uint8 array."""
    hist = np.bincount(flat_u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 140.0
    bins = np.arange(256, dtype=np.float64)
    sum_all = float((hist * bins).sum())
    w_bg = np.cumsum(hist)
    w_fg = total - w_bg
    sum_bg = np.cumsum(hist * bins)
    valid = (w_bg > 0) & (w_fg > 0)
    if not valid.any():
        return 140.0
    m_bg = np.zeros(256, dtype=np.float64)
    m_fg = np.zeros(256, dtype=np.float64)
    m_bg[valid] = sum_bg[valid] / w_bg[valid]
    m_fg[valid] = (sum_all - sum_bg[valid]) / w_fg[valid]
    between = w_bg * w_fg * (m_bg - m_fg) ** 2
    return float(np.argmax(between))


def _adaptive_ink_mask(gray: np.ndarray) -> np.ndarray:
    """Flat-field illumination normalization + adaptive Otsu/quantile ink mask.

    Works on both crisp rendered strips and shadowed/faded phone-photo strips.
    Returns a boolean ``(H, W)`` array where ``True`` = dark ink.
    """
    if gray.size == 0:
        return np.zeros_like(gray, dtype=bool)
    p05, p95 = float(np.percentile(gray, 5)), float(np.percentile(gray, 95))
    if p95 - p05 < 12.0:
        return np.zeros(gray.shape, dtype=bool)

    # Local background estimation via Gaussian blur (removes illumination slope/vignette).
    sigma = max(12.0, min(32.0, gray.shape[0] * 0.14))
    bg = ndimage.gaussian_filter(gray, sigma=sigma, mode='nearest')
    flat = np.clip(gray / np.maximum(bg, 1.0) * 235.0, 0.0, 255.0)
    flat_u8 = np.rint(flat).astype(np.uint8)

    paper = float(np.percentile(flat, 85))
    otsu = _otsu_threshold(flat_u8)
    # Clamp Otsu between a dark floor and a paper-relative ceiling so clean
    # white noise on faded paper never triggers false ink.
    q_ceil = max(110.0, paper - 28.0)
    thresh = min(q_ceil, max(100.0, otsu))
    return flat < thresh


def _staff_anchored_crop(
    mask: np.ndarray,
    pad_v: float = 1.15,
) -> Optional[np.ndarray]:
    """Crop a boolean ink mask to a staff-anchored coordinate frame.

    Locates the topmost and bottommost long horizontal staff-line rows, then
    extracts ``[yc - pad_v * staff_h, yc + pad_v * staff_h]`` vertically (with
    False zero-padding if needed) and the staff's horizontal span ``[x0, x1]``.
    This locks the 5 staff lines to identical vertical coordinates across
    engravers so pitch mutations cannot cheat by shifting the ink bbox.
    Returns ``None`` if no horizontal staff lines are detected.
    """
    h, w = mask.shape
    if h < 12 or w < 16 or not mask.any():
        return None
    row_mean = mask.mean(axis=1)
    max_rm = float(row_mean.max())
    if max_rm < 0.20:
        return None
    staff_rows = np.where(row_mean >= max(0.25, 0.55 * max_rm))[0]
    if staff_rows.size < 2:
        return None
    sy0, sy1 = int(staff_rows[0]), int(staff_rows[-1])
    staff_h = max(8, sy1 - sy0)
    yc = 0.5 * (sy0 + sy1)

    band = mask[sy0:sy1 + 1, :]
    col_mean = band.mean(axis=0)
    scols = np.where(col_mean > 0.05)[0]
    x0, x1 = (int(scols[0]), int(scols[-1]) + 1) if scols.size else (0, w)
    if x1 - x0 < 8:
        return None

    y0 = int(round(yc - pad_v * staff_h))
    y1 = int(round(yc + pad_v * staff_h))
    out_h = max(1, y1 - y0)
    out = np.zeros((out_h, x1 - x0), dtype=bool)
    src_y0 = max(0, y0)
    src_y1 = min(h, y1)
    dst_y0 = src_y0 - y0
    if src_y1 > src_y0:
        out[dst_y0:dst_y0 + (src_y1 - src_y0), :] = mask[src_y0:src_y1, x0:x1]
    return out


def _extract_symbol_mask(mask: np.ndarray, line_len: int = 25) -> np.ndarray:
    """Suppress long horizontal staff lines to isolate notation symbols."""
    k = max(5, min(int(line_len), max(5, mask.shape[1] // 8)))
    staff_only = ndimage.binary_opening(mask, structure=np.ones((1, k), dtype=bool))
    if not staff_only.any():
        return mask
    staff_band = ndimage.binary_dilation(
        staff_only, structure=np.ones((3, 1), dtype=bool)
    )
    sym = mask & (~staff_band)
    return sym if sym.any() else mask


def _crop_to_ink_bbox(mask: np.ndarray, min_count: int = 2) -> np.ndarray:
    """Tightly crop a boolean ink mask to its notation bounding box."""
    if not mask.any():
        return mask
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    rows = np.where(row_counts >= min_count)[0]
    cols = np.where(col_counts >= min_count)[0]
    if rows.size == 0 or cols.size == 0:
        rows = np.where(row_counts > 0)[0]
        cols = np.where(col_counts > 0)[0]
    if rows.size == 0 or cols.size == 0:
        return mask
    return mask[int(rows[0]):int(rows[-1]) + 1, int(cols[0]):int(cols[-1]) + 1]


def _resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a boolean mask to ``(target_h, target_w)`` preserving thin strokes."""
    if mask.shape == (target_h, target_w):
        return mask
    im = Image.fromarray((mask.astype(np.uint8) * 255), mode='L').resize(
        (int(target_w), int(target_h)), Image.Resampling.BILINEAR
    )
    return np.asarray(im, dtype=np.uint8) >= 64


def _best_shift_fft(
    obs_f: np.ndarray,
    ren_f: np.ndarray,
    max_shift: int = 8,
) -> tuple[int, int]:
    """Find ``(dx, dy)`` in ``[-max_shift, +max_shift]`` maximizing correlation."""
    h, w = obs_f.shape
    ms = int(max(0, min(max_shift, h // 4, w // 4)))
    if ms == 0:
        return (0, 0)

    # Zero-mean normalized FFT cross-correlation over exact circular window.
    a = obs_f - float(obs_f.mean())
    b = ren_f - float(ren_f.mean())
    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    cc = np.fft.irfft2(fa * np.conj(fb), s=(h, w))

    # Extract shifts dy in [-ms, ms], dx in [-ms, ms].
    ys = np.arange(-ms, ms + 1)
    xs = np.arange(-ms, ms + 1)
    patch = cc[np.ix_(ys % h, xs % w)]
    idx = int(np.argmax(patch))
    iy, ix = divmod(idx, patch.shape[1])
    dy = int(ys[iy])
    dx = int(xs[ix])
    return (dx, dy)


def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift a boolean mask by ``(dx, dy)`` with False zero-padding."""
    if dx == 0 and dy == 0:
        return mask
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    if dst_y1 > dst_y0 and dst_x1 > dst_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return out


def _disk_structure(radius: int) -> np.ndarray:
    r = max(0, int(radius))
    if r == 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= (r * r)


def _soft_prf(
    obs_ink: np.ndarray,
    ren_ink: np.ndarray,
    obs_dil: np.ndarray,
    ren_dil: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute soft dilated ``(precision, recall, f1, iou)``."""
    n_ren = int(ren_ink.sum())
    n_obs = int(obs_ink.sum())
    if n_ren == 0 and n_obs == 0:
        return (1.0, 1.0, 1.0, 1.0)
    if n_ren == 0 or n_obs == 0:
        return (0.0, 0.0, 0.0, 0.0)
    prec = float((ren_ink & obs_dil).sum()) / float(n_ren)
    rec = float((obs_ink & ren_dil).sum()) / float(n_obs)
    if prec + rec <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)
    f1 = 2.0 * prec * rec / (prec + rec)
    denom = prec + rec - prec * rec
    iou = (prec * rec / denom) if denom > 0.0 else 0.0
    return (prec, rec, f1, iou)


def visual_similarity(
    obs_gray: Union[np.ndarray, Image.Image],
    ren_gray: Union[np.ndarray, Image.Image],
    n_bars: Optional[int] = None,
    dilation_px: int = 3,
    max_shift: int = 8,
    flag_thresh: float = 0.35,
    work_h: int = 160,
) -> dict:
    """Compare an observed rectified strip against a rendered hypothesis strip.

    Steps:
    1. Flat-field + adaptive Otsu/quantile ink binarization on both strips.
    2. Staff-anchored vertical & horizontal crop (with ink-bbox fallback),
       width-normalized to a common ``(H, W)`` at ``work_h`` so ``dilation_px``
       is scale-consistent across input resolutions.
    3. Global 2D FFT cross-correlation shift search within ``±max_shift`` px,
       followed by per-slice horizontal micro-alignment (``±max_shift`` px,
       vertical ``dy`` locked to the staff-aligned global shift so pitch errors
       cannot cheat vertically).
    4. Soft dilated ink IoU / F1 (``dilation_px=3``) computed on both full ink
       (``iou_all``) and staff-suppressed notation symbols (``iou_sym``),
       combined as ``iou = 0.5 * iou_all + 0.5 * iou_sym``, globally and per
       horizontal slice/bar (``n_bars`` slices, defaulting to 4).

    Returns a dict with ``iou``, ``iou_all``, ``iou_sym``, ``f1``,
    ``precision``, ``recall``, ``shift_xy``, ``slice_ious``,
    ``slice_sym_ious``, ``slice_residuals``, and ``flagged_slices``.
    """
    obs_f = _to_gray_float(obs_gray)
    ren_f = _to_gray_float(ren_gray)

    obs_mask_raw = _adaptive_ink_mask(obs_f)
    ren_mask_raw = _adaptive_ink_mask(ren_f)

    n_slices = int(n_bars) if (n_bars is not None and int(n_bars) >= 1) else 4

    if not obs_mask_raw.any() or not ren_mask_raw.any():
        return {
            'iou': 0.0,
            'iou_all': 0.0,
            'iou_sym': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'shift_xy': (0, 0),
            'slice_ious': [0.0] * n_slices,
            'slice_sym_ious': [0.0] * n_slices,
            'slice_residuals': [1.0] * n_slices,
            'flagged_slices': list(range(n_slices)),
        }

    obs_crop = _staff_anchored_crop(obs_mask_raw)
    if obs_crop is None:
        obs_crop = _crop_to_ink_bbox(obs_mask_raw, min_count=2)
    ren_crop = _staff_anchored_crop(ren_mask_raw)
    if ren_crop is None:
        ren_crop = _crop_to_ink_bbox(ren_mask_raw, min_count=1)

    oh, ow = obs_crop.shape
    target_h = int(work_h) if work_h > 0 else oh
    target_w = max(16, int(round(ow * (float(target_h) / float(max(1, oh))))))
    if target_w > 2048:
        target_w = 2048

    obs_norm = _resize_mask(obs_crop, target_h, target_w)
    ren_norm = _resize_mask(ren_crop, target_h, target_w)

    obs_sym = _extract_symbol_mask(obs_norm)
    ren_sym = _extract_symbol_mask(ren_norm)

    # Global 2D FFT shift search on slightly dilated masks.
    pre_struct = _disk_structure(1)
    dx, dy = _best_shift_fft(
        ndimage.binary_dilation(obs_norm, structure=pre_struct).astype(np.float32),
        ndimage.binary_dilation(ren_norm, structure=pre_struct).astype(np.float32),
        max_shift=max_shift,
    )
    ren_aligned = _shift_mask(ren_norm, dx=dx, dy=dy)
    ren_sym_aligned = _shift_mask(ren_sym, dx=dx, dy=dy)

    struct = _disk_structure(dilation_px)
    obs_dil = ndimage.binary_dilation(obs_norm, structure=struct)
    obs_sym_dil = ndimage.binary_dilation(obs_sym, structure=struct)
    ren_dil = ndimage.binary_dilation(ren_aligned, structure=struct)
    ren_sym_dil = ndimage.binary_dilation(ren_sym_aligned, structure=struct)

    # Per-bar / per-slice scoring with local horizontal micro-shift (dy locked).
    edges = np.linspace(0, target_w, n_slices + 1, dtype=int)
    local_shifts = range(-int(max_shift), int(max_shift) + 1, 2) if max_shift > 0 else (0,)

    slice_ious: list[float] = []
    slice_all_ious: list[float] = []
    slice_sym_ious: list[float] = []
    slice_precs: list[float] = []
    slice_recs: list[float] = []
    slice_f1s: list[float] = []
    slice_residuals: list[float] = []
    flagged_slices: list[int] = []

    for i in range(n_slices):
        x0, x1 = int(edges[i]), int(edges[i + 1])
        sw = x1 - x0
        if sw <= 0:
            slice_ious.append(0.0)
            slice_all_ious.append(0.0)
            slice_sym_ious.append(0.0)
            slice_precs.append(0.0)
            slice_recs.append(0.0)
            slice_f1s.append(0.0)
            slice_residuals.append(1.0)
            flagged_slices.append(i)
            continue

        o_sl = obs_norm[:, x0:x1]
        od_sl = obs_dil[:, x0:x1]
        os_sl = obs_sym[:, x0:x1]
        osd_sl = obs_sym_dil[:, x0:x1]

        best_comb = -1.0
        best_ia = 0.0
        best_is = 0.0
        best_p = 0.0
        best_r = 0.0
        best_f1 = 0.0

        for ldx in local_shifts:
            rx0 = max(0, min(target_w - sw, x0 - ldx))
            rx1 = rx0 + sw
            rs_sl = ren_sym_aligned[:, rx0:rx1]
            _, _, f1_s, iou_s = _soft_prf(
                os_sl, rs_sl, osd_sl, ren_sym_dil[:, rx0:rx1]
            )
            r_sl = ren_aligned[:, rx0:rx1]
            p_a, r_a, f1_a, iou_a = _soft_prf(
                o_sl, r_sl, od_sl, ren_dil[:, rx0:rx1]
            )
            comb = 0.5 * iou_a + 0.5 * iou_s
            if comb > best_comb:
                best_comb = comb
                best_ia = iou_a
                best_is = iou_s
                best_p = p_a
                best_r = r_a
                best_f1 = 0.5 * f1_a + 0.5 * f1_s

        s_iou = round(float(best_comb), 4)
        slice_ious.append(s_iou)
        slice_all_ious.append(round(float(best_ia), 4))
        slice_sym_ious.append(round(float(best_is), 4))
        slice_precs.append(float(best_p))
        slice_recs.append(float(best_r))
        slice_f1s.append(float(best_f1))
        slice_residuals.append(round(1.0 - s_iou, 4))
        if s_iou < float(flag_thresh):
            flagged_slices.append(i)

    iou_all = float(np.mean(slice_all_ious))
    iou_sym = float(np.mean(slice_sym_ious))
    iou = 0.5 * iou_all + 0.5 * iou_sym
    prec = float(np.mean(slice_precs))
    rec = float(np.mean(slice_recs))
    f1 = float(np.mean(slice_f1s))

    return {
        'iou': round(float(iou), 4),
        'iou_all': round(float(iou_all), 4),
        'iou_sym': round(float(iou_sym), 4),
        'f1': round(float(f1), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'shift_xy': (int(dx), int(dy)),
        'slice_ious': slice_ious,
        'slice_sym_ious': slice_sym_ious,
        'slice_residuals': slice_residuals,
        'flagged_slices': flagged_slices,
    }


def rerank_beams_visual(
    obs_gray: Union[np.ndarray, Image.Image],
    beam_toks_list: Sequence[Union[str, Sequence[str]]],
    beam_logprobs: Sequence[float],
    alpha: float = 2.0,
    aser_bin: Optional[Union[str, Path]] = None,
    n_bars: Optional[int] = None,
    target_h: int = 256,
    dilation_px: int = 3,
    max_shift: int = 8,
    flag_thresh: float = 0.35,
) -> dict:
    """Rerank beam hypotheses via ``combined_score = logprob + alpha * visual_iou``.

    Invalid bekern candidates fail soft (``visual_iou = 0.0``) instead of
    crashing. Ties break deterministically toward the lower beam index.
    """
    if len(beam_toks_list) != len(beam_logprobs):
        raise ValueError(
            f'beam_toks_list ({len(beam_toks_list)}) and '
            f'beam_logprobs ({len(beam_logprobs)}) must have equal length'
        )
    if len(beam_toks_list) == 0:
        raise ValueError('beam_toks_list must be non-empty')

    scores: list[dict] = []
    best_idx = 0
    best_combined: Optional[float] = None

    for idx, (toks, lp) in enumerate(zip(beam_toks_list, beam_logprobs)):
        tok_list = _normalize_tokens(toks)
        ren = bekern_to_strip_png(tok_list, aser_bin=aser_bin, target_h=target_h)
        if ren is None:
            vis_iou = 0.0
            vis_f1 = 0.0
            slice_ious: list[float] = []
            flagged: list[int] = []
            render_ok = False
        else:
            bars_i = n_bars if n_bars is not None else _count_bekern_bars(tok_list)
            vis = visual_similarity(
                obs_gray,
                ren,
                n_bars=bars_i,
                dilation_px=dilation_px,
                max_shift=max_shift,
                flag_thresh=flag_thresh,
            )
            vis_iou = float(vis['iou'])
            vis_f1 = float(vis['f1'])
            slice_ious = list(vis['slice_ious'])
            flagged = list(vis['flagged_slices'])
            render_ok = True

        combined = float(lp) + float(alpha) * vis_iou
        scores.append({
            'idx': idx,
            'logprob': float(lp),
            'visual_iou': vis_iou,
            'visual_f1': vis_f1,
            'combined_score': round(combined, 6),
            'render_ok': render_ok,
            'slice_ious': slice_ious,
            'flagged_slices': flagged,
        })
        if best_combined is None or combined > best_combined:
            best_combined = combined
            best_idx = idx

    return {
        'best_idx': best_idx,
        'best_toks': beam_toks_list[best_idx],
        'scores': scores,
        'flagged_slices': scores[best_idx]['flagged_slices'],
        'best_visual_iou': scores[best_idx]['visual_iou'],
        'best_combined_score': scores[best_idx]['combined_score'],
    }
