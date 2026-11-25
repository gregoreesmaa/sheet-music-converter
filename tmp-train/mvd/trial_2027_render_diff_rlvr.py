#!/usr/bin/env python3
"""2027 Trial: Closed-Loop Render-and-Diff Self-Verification RLVR.

100% label-free verifiable reward for UniOMR (v1 & v2) GRPO post-training on
unlabeled sheet music pages (Finale, Sibelius, IMSLP, RISM PDFs, phone photos):

1. ``render_bekern_to_ink_image(bekern_str)``:
   Converts ``bekern_str`` -> MusicXML via ``assembly-serialize`` (or ``mscz``
   CLI fallback) -> SVG via ``verovio.toolkit()`` -> grayscale ink image via
   ``resvg_py.svg_to_bytes``.

2. ``compute_visual_render_diff_reward(input_img_gray, rendered_img_gray)``:
   Tightly crops both images to their foreground ink bounding boxes
   (``ink > 0.15``), resizes to a canonical ``(256, 256)`` grid, and computes
   a font/margin-invariant visual similarity score in ``[0, 1]``:
     - ``0.40 * horizontal_ink_profile_corr`` (catches skipped/extra systems)
     - ``0.30 * vertical_ink_profile_corr``   (catches missing/extra bars & rhythm)
     - ``0.30 * multi_scale_pooled_ink_cosine`` (catches notehead density & pitch contour)

3. ``compute_label_free_render_diff_reward(input_img, pred_bekern)``:
   Combines ``0.50 * visual_render_diff_reward`` + ``0.20 * r_parse`` +
   ``0.15 * r_meter`` + ``0.15 * r_noloop`` (reusing helpers from
   ``rlvr_grpo.py`` — zero ground-truth labels required).

4. ``--mode trial`` CLI:
   Runs ``G=4`` GRPO steps via ``rlvr_grpo.grpo_step`` on unlabeled page images.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import resvg_py
import torch
import verovio

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parent.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import eval_uni_suite31 as eval_s31  # noqa: E402
import grammar  # noqa: E402
import rlvr_grpo  # noqa: E402
import uni_omr  # noqa: E402

# ---------------------------------------------------------------------------
# Reward Weights & Constants
# ---------------------------------------------------------------------------

INK_THRESHOLD: float = 0.15
CANONICAL_SIZE: int = 256
MULTISCALE_GRIDS: Tuple[int, ...] = (8, 16, 32, 64)

W_VIS_HORIZ: float = 0.40
W_VIS_VERT: float = 0.30
W_VIS_MSCOS: float = 0.30

W_RF_VISUAL: float = 0.50
W_RF_PARSE: float = 0.20
W_RF_METER: float = 0.15
W_RF_NOLOOP: float = 0.15

IMG_EXTS = frozenset((".png", ".jpg", ".jpeg", ".webp", ".bmp"))


# ---------------------------------------------------------------------------
# Binary Discovery & Bekern -> MusicXML -> Verovio SVG -> Grayscale Image
# ---------------------------------------------------------------------------

def find_assembly_serialize_bin() -> Path:
    """Locate ``assembly-serialize`` (release preferred, then debug, then mscz-rs)."""
    candidates = [
        REPO_ROOT / "converter" / "assembly-serialize" / "target" / "release" / "assembly-serialize.exe",
        REPO_ROOT / "converter" / "assembly-serialize" / "target" / "release" / "assembly-serialize",
        REPO_ROOT / "converter" / "assembly-serialize" / "target" / "debug" / "assembly-serialize.exe",
        REPO_ROOT / "converter" / "assembly-serialize" / "target" / "debug" / "assembly-serialize",
        REPO_ROOT.parent / "mscz-rs" / "target" / "release" / "mscz.exe",
        REPO_ROOT.parent / "mscz-rs" / "target" / "release" / "mscz",
        REPO_ROOT.parent / "mscz-rs" / "target" / "debug" / "mscz.exe",
        REPO_ROOT.parent / "mscz-rs" / "target" / "debug" / "mscz",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find assembly-serialize (or mscz) binary. "
        "Build converter/assembly-serialize with `cargo build --release`."
    )


def _count_chunk_measures(chunk: Sequence[str]) -> int:
    """Count measures in a single-system bekern token chunk."""
    n_bars = sum(1 for t in chunk if t in grammar.BAR_LIKES)
    if n_bars > 0:
        # If there are non-bar tokens after the last barline, that's a trailing measure.
        last_bar = max(i for i, t in enumerate(chunk) if t in grammar.BAR_LIKES)
        has_tail = any(
            t not in grammar.BAR_LIKES and not t.startswith(("clef-", "key", "time-"))
            for t in chunk[last_bar + 1 :]
        )
        return n_bars + (1 if has_tail else 0)
    return 1 if chunk else 0


def _prepare_bekern_for_musicxml(
    bekern: Union[str, Sequence[str]],
) -> Tuple[List[str], List[int]]:
    """Stitch ``<sys>``-separated bekern tokens and record 1-based new-system measure numbers."""
    raw_toks = rlvr_grpo._as_tokens(bekern)
    if not raw_toks:
        return [], []

    chunks: List[List[str]] = []
    cur: List[str] = []
    for tok in raw_toks:
        if tok == uni_omr.SYS_TOKEN:
            if cur:
                chunks.append(cur)
                cur = []
        else:
            cur.append(tok)
    if cur:
        chunks.append(cur)

    if not chunks:
        return [], []

    new_sys_measures: List[int] = []
    if len(chunks) > 1:
        cum_m = 0
        for idx, ch in enumerate(chunks):
            if idx > 0:
                new_sys_measures.append(cum_m + 1)
            cum_m += _count_chunk_measures(ch)

    stitched, _ = eval_s31.stitch_uni_tokens(raw_toks)
    return stitched, new_sys_measures


def _inject_system_breaks(musicxml: str, new_sys_measures: Sequence[int]) -> str:
    """Inject ``<print new-system="yes"/>`` into specified 1-based ``<measure>`` elements."""
    if not new_sys_measures:
        return musicxml
    targets = set(int(m) for m in new_sys_measures if int(m) > 1)
    if not targets:
        return musicxml

    def _repl(match: re.Match[str]) -> str:
        full = match.group(0)
        num_str = match.group(1)
        try:
            m_num = int(num_str)
        except ValueError:
            return full
        if m_num in targets:
            return full + '<print new-system="yes"/>'
        return full

    return re.sub(r'<measure\b[^>]*\bnumber="(\d+)"[^>]*>', _repl, musicxml)


def bekern_to_musicxml(
    bekern_str: Union[str, Sequence[str]],
    *,
    aser_bin: Optional[Path] = None,
    timeout_s: float = 15.0,
) -> Optional[str]:
    """Convert ``bekern_str`` to a MusicXML string via ``assembly-serialize bekern2musicxml``."""
    stitched, new_sys_measures = _prepare_bekern_for_musicxml(bekern_str)
    if not stitched:
        return None

    exe = aser_bin if aser_bin is not None else find_assembly_serialize_bin()
    with tempfile.TemporaryDirectory(prefix="rd_rlvr_") as td:
        td_path = Path(td)
        in_txt = td_path / "pred-bekern.txt"
        out_dir = td_path / "out"
        in_txt.write_text(" ".join(stitched) + "\n", encoding="utf-8")

        try:
            proc = subprocess.run(
                [
                    str(exe),
                    "bekern2musicxml",
                    "--input",
                    str(in_txt),
                    "--out",
                    str(out_dir),
                    "--stem",
                    "pred",
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None

        xml_path = out_dir / "pred.musicxml"
        if proc.returncode != 0 or not xml_path.exists():
            return None

        xml_text = xml_path.read_text(encoding="utf-8", errors="replace")
        if not xml_text.strip():
            return None
        if new_sys_measures:
            xml_text = _inject_system_breaks(xml_text, new_sys_measures)
        return xml_text


def render_musicxml_to_ink_image(
    musicxml: str,
    *,
    page_width: int = 2100,
    scale: int = 40,
    use_encoded_breaks: bool = False,
) -> Optional[np.ndarray]:
    """Render MusicXML via Verovio -> SVG -> ``resvg_py`` -> float32 grayscale image in ``[0, 1]``.

    Returns an ``(H, W)`` float32 array where ``0.0`` is black ink and ``1.0``
    is white paper, or ``None`` if rendering fails.
    """
    if not musicxml or not musicxml.strip():
        return None

    try:
        tk = verovio.toolkit()
        verovio.enableLog(verovio.LOG_OFF)
    except Exception:
        tk = verovio.toolkit()

    opts: Dict[str, Any] = {
        "pageWidth": int(page_width),
        "pageHeight": 60000,
        "scale": int(scale),
        "adjustPageHeight": True,
        "header": "none",
        "footer": "none",
        "breaks": "encoded" if use_encoded_breaks else "auto",
    }
    try:
        tk.setOptions(opts)
        if not tk.loadData(musicxml):
            return None
        n_pages = int(tk.getPageCount())
        if n_pages < 1:
            return None

        rendered_pages: List[np.ndarray] = []
        for p_idx in range(1, n_pages + 1):
            svg = tk.renderToSVG(p_idx)
            if not svg:
                continue
            png_bytes = resvg_py.svg_to_bytes(svg_string=svg, background="#ffffff")
            im = Image.open(io.BytesIO(png_bytes)).convert("L")
            arr = np.asarray(im, dtype=np.float32) / 255.0
            rendered_pages.append(arr)

        if not rendered_pages:
            return None
        if len(rendered_pages) == 1:
            return rendered_pages[0]

        max_w = max(p.shape[1] for p in rendered_pages)
        padded: List[np.ndarray] = []
        for p in rendered_pages:
            if p.shape[1] < max_w:
                pad = np.ones((p.shape[0], max_w - p.shape[1]), dtype=np.float32)
                p = np.concatenate([p, pad], axis=1)
            padded.append(p)
        return np.concatenate(padded, axis=0)
    except Exception:
        return None


def render_bekern_to_ink_image(
    bekern_str: Union[str, Sequence[str]],
    *,
    aser_bin: Optional[Path] = None,
    page_width: int = 2100,
    scale: int = 40,
) -> Optional[np.ndarray]:
    """Convert ``bekern_str`` -> MusicXML -> Verovio SVG -> ``resvg_py`` grayscale image.

    Returns an ``(H, W)`` float32 grayscale array in ``[0, 1]`` (``0.0`` = ink,
    ``1.0`` = white paper), or ``None`` if conversion or rendering fails.
    """
    raw_toks = rlvr_grpo._as_tokens(bekern_str)
    if not raw_toks:
        return None
    has_sys = uni_omr.SYS_TOKEN in raw_toks
    xml_text = bekern_to_musicxml(raw_toks, aser_bin=aser_bin)
    if xml_text is None:
        return None
    use_encoded = has_sys and ('new-system="yes"' in xml_text)
    return render_musicxml_to_ink_image(
        xml_text,
        page_width=page_width,
        scale=scale,
        use_encoded_breaks=use_encoded,
    )


# ---------------------------------------------------------------------------
# Visual Render-and-Diff Similarity Score
# ---------------------------------------------------------------------------

def to_gray_float01(img: Any) -> np.ndarray:
    """Normalize a PIL Image, numpy array, or torch Tensor to ``(H, W)`` float32 in ``[0, 1]``."""
    if isinstance(img, (str, Path)):
        im = Image.open(img).convert("L")
        return np.asarray(im, dtype=np.float32) / 255.0

    if isinstance(img, Image.Image):
        return np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    if isinstance(img, torch.Tensor):
        t = img.detach().cpu().float()
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3:
            if t.shape[0] in (1, 3, 4):
                t = t[:3].mean(dim=0)
            else:
                t = t[..., :3].mean(dim=-1)
        arr = t.numpy()
        # Handle UniOMR-normalized tensors ((x - 0.85) / 0.25) where values can be negative
        if arr.min() < -0.05:
            arr = arr * 0.25 + 0.85
        elif arr.max() > 1.5:
            arr = arr / 255.0
        return np.clip(arr.astype(np.float32), 0.0, 1.0)

    arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.shape[2] in (1, 3, 4):
            arr = arr[..., :3].mean(axis=2)
        elif arr.shape[0] in (1, 3, 4):
            arr = arr[:3].mean(axis=0)
    arr = arr.astype(np.float32)
    if arr.size == 0:
        return np.ones((1, 1), dtype=np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    elif arr.min() < -0.05:
        arr = arr * 0.25 + 0.85
    return np.clip(arr, 0.0, 1.0)


def _crop_and_resize_ink(
    img_gray: Any,
    *,
    ink_thresh: float = INK_THRESHOLD,
    size: int = CANONICAL_SIZE,
) -> Optional[np.ndarray]:
    """Convert grayscale image to ink map ``1 - gray``, crop to ``ink > ink_thresh`` bbox, resize to ``(size, size)``."""
    gray = to_gray_float01(img_gray)
    ink = np.clip(1.0 - gray, 0.0, 1.0)
    ys, xs = np.where(ink > float(ink_thresh))
    if ys.size == 0 or xs.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = ink[y0:y1, x0:x1]
    pil_crop = Image.fromarray(np.clip(crop * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    pil_resized = pil_crop.resize((int(size), int(size)), Image.BILINEAR)
    return np.asarray(pil_resized, dtype=np.float32) / 255.0


def _pearson_corr_clamped01(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation clamped to ``[0.0, 1.0]``."""
    a_c = a.astype(np.float64) - float(a.mean())
    b_c = b.astype(np.float64) - float(b.mean())
    na = float(np.linalg.norm(a_c))
    nb = float(np.linalg.norm(b_c))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    corr = float(np.dot(a_c, b_c) / (na * nb))
    return float(max(0.0, min(1.0, corr)))


def _multiscale_pooled_cosine(
    ink_a: np.ndarray,
    ink_b: np.ndarray,
    grids: Sequence[int] = MULTISCALE_GRIDS,
) -> float:
    """Average cosine similarity over multi-scale spatial average-pooled ink grids."""
    h, w = ink_a.shape
    sims: List[float] = []
    for g in grids:
        g_int = int(g)
        if h % g_int == 0 and w % g_int == 0:
            pa = ink_a.reshape(g_int, h // g_int, g_int, w // g_int).mean(axis=(1, 3)).ravel()
            pb = ink_b.reshape(g_int, h // g_int, g_int, w // g_int).mean(axis=(1, 3)).ravel()
        else:
            im_a = Image.fromarray(np.clip(ink_a * 255.0, 0.0, 255.0).astype(np.uint8))
            im_b = Image.fromarray(np.clip(ink_b * 255.0, 0.0, 255.0).astype(np.uint8))
            pa = (np.asarray(im_a.resize((g_int, g_int), Image.BOX), dtype=np.float32) / 255.0).ravel()
            pb = (np.asarray(im_b.resize((g_int, g_int), Image.BOX), dtype=np.float32) / 255.0).ravel()
        na = float(np.linalg.norm(pa))
        nb = float(np.linalg.norm(pb))
        if na < 1e-12 or nb < 1e-12:
            sims.append(0.0)
        else:
            sims.append(float(max(0.0, min(1.0, float(np.dot(pa, pb)) / (na * nb)))))
    return float(sum(sims) / max(1, len(sims)))


def compute_visual_render_diff_details(
    input_img_gray: Any,
    rendered_img_gray: Any,
    *,
    ink_thresh: float = INK_THRESHOLD,
    size: int = CANONICAL_SIZE,
) -> Dict[str, float]:
    """Compute visual render-and-diff similarity score and its 3 sub-components."""
    if input_img_gray is None or rendered_img_gray is None:
        return {
            "visual_reward": 0.0,
            "horizontal_ink_profile_corr": 0.0,
            "vertical_ink_profile_corr": 0.0,
            "multi_scale_pooled_ink_cosine": 0.0,
        }

    ink_a = _crop_and_resize_ink(input_img_gray, ink_thresh=ink_thresh, size=size)
    ink_b = _crop_and_resize_ink(rendered_img_gray, ink_thresh=ink_thresh, size=size)
    if ink_a is None or ink_b is None:
        return {
            "visual_reward": 0.0,
            "horizontal_ink_profile_corr": 0.0,
            "vertical_ink_profile_corr": 0.0,
            "multi_scale_pooled_ink_cosine": 0.0,
        }

    h_corr = _pearson_corr_clamped01(ink_a.mean(axis=1), ink_b.mean(axis=1))
    v_corr = _pearson_corr_clamped01(ink_a.mean(axis=0), ink_b.mean(axis=0))
    ms_cos = _multiscale_pooled_cosine(ink_a, ink_b, grids=MULTISCALE_GRIDS)

    score = W_VIS_HORIZ * h_corr + W_VIS_VERT * v_corr + W_VIS_MSCOS * ms_cos
    score = float(max(0.0, min(1.0, score)))
    return {
        "visual_reward": score,
        "horizontal_ink_profile_corr": float(h_corr),
        "vertical_ink_profile_corr": float(v_corr),
        "multi_scale_pooled_ink_cosine": float(ms_cos),
    }


def compute_visual_render_diff_reward(
    input_img_gray: Any,
    rendered_img_gray: Any,
    *,
    ink_thresh: float = INK_THRESHOLD,
    size: int = CANONICAL_SIZE,
) -> float:
    """Compute font/margin-invariant visual similarity score in ``[0, 1]``."""
    return compute_visual_render_diff_details(
        input_img_gray,
        rendered_img_gray,
        ink_thresh=ink_thresh,
        size=size,
    )["visual_reward"]


# ---------------------------------------------------------------------------
# Label-Free Closed-Loop Render-and-Diff Reward
# ---------------------------------------------------------------------------

def compute_label_free_render_diff_reward(
    input_img: Any,
    pred_bekern: Union[str, Sequence[str]],
    *,
    aser_bin: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compute 100% label-free render-and-diff RLVR reward in ``[0, 1]``.

    Combines:
      - ``0.50 * visual_render_diff_reward``
      - ``0.20 * r_parse``
      - ``0.15 * r_meter``
      - ``0.15 * r_noloop``
    """
    pred_toks = rlvr_grpo._as_tokens(pred_bekern)

    # 1. Parse + meter conservation (reusing rlvr_grpo helpers)
    parse_ok, n_notes, n_valid_bars, n_checked_bars = rlvr_grpo._walk_parse_and_meter(
        pred_toks, tol=rlvr_grpo.METER_TOL
    )
    r_parse = 1.0 if (parse_ok and n_notes >= 1) else 0.0

    if parse_ok and n_checked_bars > 0:
        r_meter = float(n_valid_bars) / float(n_checked_bars)
    else:
        r_meter = 0.0

    # 2. Anti-loop reward (reusing rlvr_grpo helper)
    loop_detected = rlvr_grpo.has_repetition_loop(
        pred_toks,
        min_period=rlvr_grpo.LOOP_MIN_PERIOD,
        max_period=rlvr_grpo.LOOP_MAX_PERIOD,
        min_repeats=rlvr_grpo.LOOP_MIN_REPEATS,
    )
    r_noloop = 0.0 if (not pred_toks or loop_detected) else 1.0

    # 3. Closed-loop render-and-diff visual similarity
    render_ok = False
    vis_details = {
        "visual_reward": 0.0,
        "horizontal_ink_profile_corr": 0.0,
        "vertical_ink_profile_corr": 0.0,
        "multi_scale_pooled_ink_cosine": 0.0,
    }
    if pred_toks and n_notes >= 1:
        rendered = render_bekern_to_ink_image(pred_toks, aser_bin=aser_bin)
        if rendered is not None:
            render_ok = True
            vis_details = compute_visual_render_diff_details(input_img, rendered)

    r_visual = float(vis_details["visual_reward"])
    reward = (
        W_RF_VISUAL * r_visual
        + W_RF_PARSE * r_parse
        + W_RF_METER * r_meter
        + W_RF_NOLOOP * r_noloop
    )
    reward = float(max(0.0, min(1.0, reward)))

    return {
        "reward": reward,
        "r_visual": r_visual,
        "r_parse": float(r_parse),
        "r_meter": float(r_meter),
        "r_noloop": float(r_noloop),
        "horizontal_ink_profile_corr": float(vis_details["horizontal_ink_profile_corr"]),
        "vertical_ink_profile_corr": float(vis_details["vertical_ink_profile_corr"]),
        "multi_scale_pooled_ink_cosine": float(vis_details["multi_scale_pooled_ink_cosine"]),
        "render_ok": bool(render_ok),
        "parse_ok": bool(parse_ok),
        "n_notes": int(n_notes),
        "n_valid_interior_bars": int(n_valid_bars),
        "n_interior_bars": int(n_checked_bars),
        "has_loop": bool(loop_detected),
    }


# ---------------------------------------------------------------------------
# Unlabeled GRPO Trial Runner (--mode trial)
# ---------------------------------------------------------------------------

def _load_page_tensor_and_gray(
    img_path: Path,
    *,
    img_h: int = 256,
    img_w: int = 256,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Load an unlabeled page image as both a model input tensor ``(3, H, W)`` and grayscale ``(H0, W0)``."""
    im_rgb = Image.open(img_path).convert("RGB")
    raw_gray = np.asarray(im_rgb.convert("L"), dtype=np.float32) / 255.0
    im_resized = im_rgb.resize((int(img_w), int(img_h)), Image.BILINEAR)
    arr = np.asarray(im_resized, dtype=np.float32) / 255.0
    t = (torch.from_numpy(arr).permute(2, 0, 1) - 0.85) / 0.25
    return t, raw_gray


def _build_tiny_trial_model() -> uni_omr.UniOMRNet:
    cfg = uni_omr.UniOMRConfig(
        d_model=32,
        n_enc_layers=1,
        n_dec_layers=1,
        n_heads=4,
        ffn_mult=2.0,
        dropout=0.0,
        keep_ratio=0.5,
        use_uv_aux=False,
        mtp_k=1,
        backbone_stride=16,
    )
    return uni_omr.UniOMRNet(cfg)


def discover_unlabeled_images(image_dir: Path, limit: int = 16) -> List[Path]:
    """Find unlabeled sheet-music page images in ``image_dir``."""
    if image_dir.is_file() and image_dir.suffix.lower() in IMG_EXTS:
        return [image_dir]
    if not image_dir.is_dir():
        return []
    found = [
        p
        for p in sorted(image_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    return found[: max(1, int(limit))]


def run_trial_grpo(
    *,
    image_dir: Path,
    ckpt_path: Optional[Path] = None,
    steps: int = 2,
    batch_size: int = 1,
    group_size: int = 4,
    max_len: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.95,
    kl_beta: float = 0.02,
    lr: float = 1e-4,
    img_h: int = 256,
    img_w: int = 256,
    device_str: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """Run ``steps`` of label-free render-and-diff GRPO (``G=4``) on unlabeled images."""
    torch.manual_seed(int(seed))
    device = torch.device(device_str)

    img_paths = discover_unlabeled_images(image_dir, limit=max(steps * batch_size, 8))
    if not img_paths:
        raise FileNotFoundError(f"No unlabeled images found in {image_dir}")

    if ckpt_path is not None and ckpt_path.exists():
        model, _, _ = eval_s31.load_uni_checkpoint(ckpt_path, device=device)
    else:
        model = _build_tiny_trial_model().to(device)

    ref_model = copy.deepcopy(model).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
    grpo_cfg = rlvr_grpo.GRPOConfig(
        group_size=int(group_size),
        max_len=int(max_len),
        temperature=float(temperature),
        top_p=float(top_p),
        kl_beta=float(kl_beta),
        use_grammar=True,
        strict_grammar=False,
    )
    aser_bin = find_assembly_serialize_bin()

    def _label_free_reward_fn(pred_toks: Sequence[str], ref_img_gray: Any) -> Dict[str, Any]:
        return compute_label_free_render_diff_reward(
            ref_img_gray, pred_toks, aser_bin=aser_bin
        )

    history: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for step_idx in range(int(steps)):
        batch_paths = [
            img_paths[(step_idx * batch_size + b) % len(img_paths)]
            for b in range(int(batch_size))
        ]
        tensors: List[torch.Tensor] = []
        grays: List[np.ndarray] = []
        for p in batch_paths:
            t_img, g_img = _load_page_tensor_and_gray(p, img_h=img_h, img_w=img_w)
            tensors.append(t_img)
            grays.append(g_img)
        images = torch.stack(tensors, dim=0).to(device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, stats = rlvr_grpo.grpo_step(
            model,
            ref_model,
            images,
            grays,  # Passed straight through to _label_free_reward_fn as unlabeled images
            cfg=grpo_cfg,
            reward_fn=_label_free_reward_fn,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        row = {
            "step": step_idx + 1,
            "images": [p.name for p in batch_paths],
            "loss": float(stats["loss"]),
            "pg_loss": float(stats["pg_loss"]),
            "kl_loss": float(stats["kl_loss"]),
            "mean_reward": float(stats["mean_reward"]),
            "std_reward": float(stats["std_reward"]),
            "mean_r_parse": float(stats["mean_r_parse"]),
            "mean_r_meter": float(stats["mean_r_meter"]),
            "mean_r_noloop": float(stats["mean_r_noloop"]),
            "seq_len": int(stats["seq_len"]),
        }
        history.append(row)
        print(
            f"[trial step {row['step']}/{steps}] loss={row['loss']:.4f} "
            f"reward={row['mean_reward']:.4f}±{row['std_reward']:.4f} "
            f"parse={row['mean_r_parse']:.2f} meter={row['mean_r_meter']:.2f} "
            f"noloop={row['mean_r_noloop']:.2f}",
            flush=True,
        )

    return {
        "mode": "trial",
        "steps": int(steps),
        "group_size": int(group_size),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "history": history,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="2027 Trial: Closed-Loop Render-and-Diff Self-Verification RLVR."
    )
    ap.add_argument(
        "--mode",
        choices=("trial", "score"),
        default="trial",
        help="Run unlabeled GRPO trial (`trial`) or score a single image+bekern pair (`score`).",
    )
    ap.add_argument(
        "--image-dir",
        type=Path,
        default=REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "clean",
        help="Directory (or single image) of unlabeled sheet-music pages for `--mode trial`.",
    )
    ap.add_argument("--image", type=Path, default=None, help="Input image path for `--mode score`.")
    ap.add_argument("--bekern", type=Path, default=None, help="Bekern token file path for `--mode score`.")
    ap.add_argument("--ckpt", type=Path, default=None, help="Optional UniOMR .pt checkpoint (else tiny CPU model).")
    ap.add_argument("--steps", type=int, default=2, help="Number of GRPO steps for `--mode trial`.")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--group-size", type=int, default=4, help="GRPO rollout group size G (default: 4).")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--kl-beta", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-h", type=int, default=256)
    ap.add_argument("--img-w", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.mode == "score":
        if args.image is None or args.bekern is None:
            raise SystemExit("--mode score requires both --image and --bekern")
        bk_str = args.bekern.read_text(encoding="utf-8").strip()
        res = compute_label_free_render_diff_reward(args.image, bk_str)
        print(json.dumps(res, indent=2))
        return 0

    summary = run_trial_grpo(
        image_dir=args.image_dir,
        ckpt_path=args.ckpt,
        steps=args.steps,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_len=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        kl_beta=args.kl_beta,
        lr=args.lr,
        img_h=args.img_h,
        img_w=args.img_w,
        device_str=args.device,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


__all__ = [
    "INK_THRESHOLD",
    "CANONICAL_SIZE",
    "W_VIS_HORIZ",
    "W_VIS_VERT",
    "W_VIS_MSCOS",
    "W_RF_VISUAL",
    "W_RF_PARSE",
    "W_RF_METER",
    "W_RF_NOLOOP",
    "find_assembly_serialize_bin",
    "bekern_to_musicxml",
    "render_musicxml_to_ink_image",
    "render_bekern_to_ink_image",
    "to_gray_float01",
    "compute_visual_render_diff_details",
    "compute_visual_render_diff_reward",
    "compute_label_free_render_diff_reward",
    "run_trial_grpo",
    "build_arg_parser",
]


if __name__ == "__main__":
    raise SystemExit(main())
