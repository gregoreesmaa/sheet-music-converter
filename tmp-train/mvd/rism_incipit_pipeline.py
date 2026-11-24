#!/usr/bin/env python3
"""RISM Opening-Incipit & Unlabeled Full-Page Training Pipeline.

Converts RISM public-domain handwritten manuscript corpora (e.g.
``D:\\tmp\\rism-pdm-hw`` and ``D:\\tmp\\rism-pdm-hw-intl``) into THREE clean
artifacts under ``--out-dir``:

1. ``renders/<id>.png`` + ``targets/<id>-bekern.txt`` + ``manifest.jsonl``:
   High-precision, barline-cropped & margin-gated opening-staff incipit strips
   for Stage-2 semi-supervised training (``train.py``-compatible layout).
2. ``unlabeled_strips/<id>_s<k>[_c<j>].png``:
   ALL unwrapped system strips (every system ``s0..sK`` on every RISM page,
   chunked to ``<= 4`` staves per canonical 256 px ribbon) for Stage-2
   unsupervised consistency / bar-math rounds.
3. ``stage1_pages.txt``:
   Sorted list of all full RISM page image paths for Stage-1 unsupervised EMA
   geometric + ``log_hm`` equivariance.

Usage (CPU-safe, atomic tmp+rename writes, ``--resume`` supported):
    .\\.venv\\Scripts\\python.exe tmp-train/mvd/rism_incipit_pipeline.py \\
        --rism-dirs D:\\tmp\\rism-pdm-hw,D:\\tmp\\rism-pdm-hw-intl \\
        --staff-unet-ckpt D:\\tmp\\mvd-runs\\staff-unet2d-v3-dynres\\staff_unet2d.pt \\
        --out-dir D:\\tmp\\rism-incipit-strips \\
        --limit 50 --min-bars 2 --min-tokens 25 --device cpu --resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MVD_DIR = Path(__file__).resolve().parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import bezier_ribbon  # noqa: E402
import data as mvd_data  # noqa: E402
import dynamic_res  # noqa: E402
import eval_sys_gate  # noqa: E402
import staff_unet2d  # noqa: E402

DEFAULT_EVAL_MANIFESTS = (
    REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json",
    REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_manifest.json",
)

_RISM_URL_RE = re.compile(r"/sources/(\d+)")
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")
_KEY_RE = re.compile(r"^key([+-]\d+)$")


# ---------------------------------------------------------------------------
# Atomic file I/O helpers
# ---------------------------------------------------------------------------


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text via sibling ``.tmp`` + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_save_png(img: Image.Image, path: Path) -> None:
    """Atomically save a PIL image as PNG via sibling ``.tmp`` + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    img.save(tmp, format="PNG")
    os.replace(tmp, path)


def atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Atomically write a list of JSON objects as JSONL."""
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    atomic_write_text(path, ("\n".join(lines) + "\n") if lines else "")


# ---------------------------------------------------------------------------
# Step 1: Bekern parsing, RISM manifest loading, short-tail & eval-overlap gate
# ---------------------------------------------------------------------------


def is_barline_token(tok: str) -> bool:
    """True iff ``tok`` is a measure-closing barline token (excludes ``bar-rest``)."""
    return tok == "bar" or (tok.startswith("bar-") and tok != "bar-rest")


def parse_bekern_stats(bekern: str) -> dict[str, Any]:
    """Compute token count, barline count, key-signature fifths, and vocab validity."""
    toks = str(bekern or "").strip().split()
    n_tokens = len(toks)
    n_bars = sum(1 for t in toks if is_barline_token(t))
    key_fifths = 0
    for t in toks[:8]:
        m = _KEY_RE.match(t)
        if m:
            key_fifths = int(m.group(1))
            break
    unknown_tokens = [t for t in toks if t not in mvd_data.TOKEN_TO_ID]
    return {
        "tokens": toks,
        "n_tokens": n_tokens,
        "n_bars": n_bars,
        "key_fifths": key_fifths,
        "vocab_valid": len(unknown_tokens) == 0,
        "unknown_tokens": unknown_tokens,
    }


def normalize_rism_id(raw: Any) -> str | None:
    """Extract numeric RISM source ID from either a raw ID or a RISM URL."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    m = _RISM_URL_RE.search(s)
    if m:
        return m.group(1)
    if s.isdigit():
        return s
    return s


def make_sample_id(record: dict[str, Any]) -> str:
    """Build a deterministic, filesystem-safe sample ID for a RISM manifest row."""
    img_path = str(record.get("image_path") or "").strip()
    if img_path:
        stem = Path(img_path).stem
        if stem:
            return _SAFE_ID_RE.sub("_", stem)
    siglum = str(record.get("siglum") or "rism").strip()
    rid = str(record.get("rism_id") or normalize_rism_id(record.get("source_url")) or "unknown")
    cidx = int(record.get("canvas_index", 0) or 0)
    return _SAFE_ID_RE.sub("_", f"{siglum}_{rid}_c{cidx:02d}")


def load_eval_signatures(
    manifest_paths: Sequence[str | Path] = DEFAULT_EVAL_MANIFESTS,
) -> dict[str, set[str]]:
    """Collect RISM IDs, IIIF manifests, image SHA-256s, and slugs from eval manifests."""
    sigs: dict[str, set[str]] = {
        "rism_ids": set(),
        "iiif_manifests": set(),
        "sha256s": set(),
        "slugs": set(),
    }
    for mp in manifest_paths:
        p = Path(mp)
        if not p.exists():
            raise FileNotFoundError(f"eval manifest not found: {p}")
        doc = json.loads(p.read_text(encoding="utf-8"))
        for pc in doc.get("pieces", []):
            slug = str(pc.get("slug") or "").strip()
            if slug:
                sigs["slugs"].add(slug)
            rid = normalize_rism_id(pc.get("rism_url") or pc.get("rism_id") or pc.get("source_url"))
            if rid:
                sigs["rism_ids"].add(rid)
            iiif = str(pc.get("iiif_manifest") or pc.get("iiif_manifest_url") or "").strip().rstrip("/")
            if iiif:
                sigs["iiif_manifests"].add(iiif)
            for key in ("clean", "photo"):
                entry = pc.get(key)
                if isinstance(entry, dict) and entry.get("sha256"):
                    sigs["sha256s"].add(str(entry["sha256"]).strip().lower())
    return sigs


def find_eval_overlaps(
    records: Sequence[dict[str, Any]],
    eval_sigs: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Return every RISM record overlapping with any eval signature."""
    overlaps: list[dict[str, Any]] = []
    for rec in records:
        sid = rec.get("sample_id") or make_sample_id(rec)
        reasons: list[str] = []
        rid = normalize_rism_id(rec.get("rism_id") or rec.get("source_url"))
        if rid and rid in eval_sigs["rism_ids"]:
            reasons.append(f"rism_id={rid}")
        iiif = str(rec.get("iiif_manifest_url") or rec.get("iiif_manifest") or "").strip().rstrip("/")
        if iiif and iiif in eval_sigs["iiif_manifests"]:
            reasons.append(f"iiif_manifest={iiif}")
        sha = str(rec.get("image_sha256") or "").strip().lower()
        if sha and sha in eval_sigs["sha256s"]:
            reasons.append(f"image_sha256={sha[:12]}")
        if sid in eval_sigs["slugs"]:
            reasons.append(f"slug={sid}")
        if reasons:
            overlaps.append({"sample_id": sid, "reasons": reasons})
    return overlaps


def assert_no_eval_overlap(
    records: Sequence[dict[str, Any]],
    eval_manifests: Sequence[str | Path] = DEFAULT_EVAL_MANIFESTS,
) -> dict[str, set[str]]:
    """Assert 0 overlap between ``records`` and all eval manifests."""
    sigs = load_eval_signatures(eval_manifests)
    overlaps = find_eval_overlaps(records, sigs)
    if overlaps:
        preview = "; ".join(
            f"{o['sample_id']} ({', '.join(o['reasons'])})" for o in overlaps[:5]
        )
        raise AssertionError(
            f"eval leakage detected: {len(overlaps)} RISM record(s) overlap eval manifests: {preview}"
        )
    return sigs


def load_rism_manifests(
    rism_dirs: Sequence[str | Path],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load and deduplicate RISM ``manifest.jsonl`` rows across one or more directories."""
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for d in rism_dirs:
        root = Path(d)
        mpath = root / "manifest.jsonl"
        if not mpath.exists():
            raise FileNotFoundError(f"RISM manifest not found: {mpath}")
        with open(mpath, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{mpath}:{lineno}: invalid JSON: {exc}") from None
                sid = make_sample_id(rec)
                if sid in seen_ids:
                    continue
                seen_ids.add(sid)
                stats = parse_bekern_stats(rec.get("bekern", ""))
                enriched = dict(rec)
                enriched["sample_id"] = sid
                enriched["source_dir"] = str(root)
                enriched["n_bars"] = stats["n_bars"]
                enriched["n_tokens"] = stats["n_tokens"]
                enriched["key_fifths"] = stats["key_fifths"]
                enriched["vocab_valid"] = stats["vocab_valid"]
                enriched["unknown_tokens"] = stats["unknown_tokens"]
                records.append(enriched)
                if limit is not None and len(records) >= int(limit):
                    return records
    return records


def filter_short_tail(
    record: dict[str, Any],
    min_bars: int = 2,
    min_tokens: int = 25,
) -> tuple[bool, str | None]:
    """Check whether a RISM record's opening incipit meets length & vocab gates."""
    bekern = str(record.get("bekern") or "").strip()
    if not bekern:
        return False, "empty_bekern"
    stats = (
        {
            "n_bars": int(record["n_bars"]),
            "n_tokens": int(record["n_tokens"]),
            "vocab_valid": bool(record.get("vocab_valid", True)),
        }
        if "n_bars" in record and "n_tokens" in record and "vocab_valid" in record
        else parse_bekern_stats(bekern)
    )
    if stats["n_bars"] < int(min_bars):
        return False, f"short_tail_bars:{stats['n_bars']}<{min_bars}"
    if stats["n_tokens"] < int(min_tokens):
        return False, f"short_tail_tokens:{stats['n_tokens']}<{min_tokens}"
    if not stats["vocab_valid"]:
        return False, "unknown_bekern_tokens"
    return True, None


# ---------------------------------------------------------------------------
# Step 3: Barline-Count Horizontal Crop & Margin Gate (pure image functions)
# ---------------------------------------------------------------------------


def compute_local_ink_map(
    gray: np.ndarray,
    bg_kernel_h: int = 15,
    bg_kernel_w: int = 31,
) -> np.ndarray:
    """Compute normalized local darkness ``ink in [0, 1]`` robust to aged paper."""
    arr = np.asarray(gray, dtype=np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    kh = min(int(bg_kernel_h), max(3, arr.shape[0]))
    kw = min(int(bg_kernel_w), max(3, arr.shape[1]))
    bg = ndimage.percentile_filter(arr, percentile=88, size=(kh, kw))
    denom = np.maximum(bg, 0.25)
    return np.clip((bg - arr) / denom, 0.0, 1.0)


def _interline_space_slices(
    canon_top: float,
    target_sp: float,
    band_h: int,
    line_mask_half_px: float = 1.5,
) -> list[slice]:
    """Return the 4 vertical row slices between the 5 canonical staff lines."""
    slices: list[slice] = []
    for k in range(4):
        y0 = int(math.ceil(canon_top + k * target_sp + line_mask_half_px))
        y1 = int(math.floor(canon_top + (k + 1) * target_sp - line_mask_half_px)) + 1
        y0 = max(0, min(y0, band_h - 1))
        y1 = max(y0 + 1, min(y1, band_h))
        slices.append(slice(y0, y1))
    return slices


def detect_barline_columns(
    strip_img: Image.Image | np.ndarray,
    target_sp: float = 10.0,
    key_fifths: int = 0,
    canon_top: float = 12.0,
    slot_pitch: float = 64.0,
    peak_thresh: float = 0.55,
    min_bar_dist_sp: float = 3.0,
) -> dict[str, Any]:
    """Detect vertical/slightly-slanted barline columns on a canonical single-staff strip.

    Operates on slot 0 (rows ``0..int(slot_pitch)``) by:
      1. Local background subtraction (robust to yellowed/sepia manuscripts).
      2. Sampling only the 4 inter-line spaces (staff lines masked out).
      3. Small-slant shear bank (``+-5 px`` over ``4*sp``) to handle slanted
         handwritten barlines.
      4. Width/contrast suppression to reject solid black margins/stains.
      5. Left-header exclusion (clef + key signature + time signature) and NMS.
    """
    if isinstance(strip_img, Image.Image):
        gray_full = np.asarray(strip_img.convert("L"), dtype=np.float32) / 255.0
    else:
        arr = np.asarray(strip_img, dtype=np.float32)
        if arr.ndim == 3:
            gray_full = arr.mean(axis=2)
        else:
            gray_full = arr
        if gray_full.max() > 1.5:
            gray_full = gray_full / 255.0

    band_h = min(int(round(slot_pitch)), gray_full.shape[0])
    w = gray_full.shape[1]
    band = gray_full[:band_h, :]
    ink = compute_local_ink_map(band)

    spaces = _interline_space_slices(canon_top, target_sp, band_h, line_mask_half_px=1.5)
    space_rows = np.concatenate([np.arange(s.start, s.stop) for s in spaces])
    space_ink = ink[space_rows, :]

    p92 = float(np.percentile(space_ink, 92)) if space_ink.size else 0.0
    ink_thr = max(0.14, min(0.32, p92 * 0.45))
    ink_bin = ink >= ink_thr
    ink_coverage = float(ink_bin[space_rows, :].mean()) if space_rows.size else 0.0

    # Find left music onset (first column with meaningful inter-line ink).
    col_activity = ink_bin[space_rows, :].mean(axis=0)
    active_cols = np.flatnonzero(col_activity >= 0.08)
    x_onset = int(active_cols[0]) if active_cols.size > 0 else 0
    header_sp = 3.2 + 0.45 * min(abs(int(key_fifths)), 7)
    x_header_end = min(w, int(round(x_onset + header_sp * target_sp)))

    # Dilate horizontally by +-1 px so 1-px pen jitter does not break vertical continuity.
    ink_dil = ndimage.maximum_filter1d(ink_bin.astype(np.float32), size=3, axis=1)

    y_center = canon_top + 2.0 * target_sp
    staff_h = max(4.0 * target_sp, 1.0)
    slants = (-5, -3, -1, 0, 1, 3, 5)
    cols_idx = np.arange(w, dtype=np.int32)

    best_score = np.zeros(w, dtype=np.float32)
    best_mean_cov = np.zeros(w, dtype=np.float32)
    best_second_min = np.zeros(w, dtype=np.float32)
    best_min_cov = np.zeros(w, dtype=np.float32)

    for dx_tot in slants:
        covs = []
        for sl in spaces:
            rows = np.arange(sl.start, sl.stop, dtype=np.float32)
            shifts = np.round(dx_tot * (rows - y_center) / staff_h).astype(np.int32)
            # Sample shifted columns with edge clamping.
            acc = np.zeros(w, dtype=np.float32)
            for r, sh in zip(range(sl.start, sl.stop), shifts):
                src_x = np.clip(cols_idx + sh, 0, w - 1)
                acc += ink_dil[r, src_x]
            covs.append(acc / float(max(sl.stop - sl.start, 1)))

        cov_stack = np.stack(covs, axis=0)  # (4, W)
        mean_cov = cov_stack.mean(axis=0)
        sorted_cov = np.sort(cov_stack, axis=0)
        min_cov = sorted_cov[0]
        second_min = sorted_cov[1]

        # Local neighborhood ink density to reject solid black borders/stains.
        wide_bg = ndimage.uniform_filter1d(mean_cov, size=15, mode="nearest")
        raw = 0.40 * mean_cov + 0.25 * second_min + 0.35 * min_cov
        # Suppress thick dark blobs (borders/smudges) where wide neighborhood is mostly ink.
        blob_penalty = np.clip((wide_bg - 0.55) / 0.35, 0.0, 1.0)
        score = raw * (1.0 - 0.65 * blob_penalty)

        better = score > best_score
        best_score = np.where(better, score, best_score)
        best_mean_cov = np.where(better, mean_cov, best_mean_cov)
        best_second_min = np.where(better, second_min, best_second_min)
        best_min_cov = np.where(better, min_cov, best_min_cov)

    # Slight 3-px smoothing for clean local-maxima peak picking.
    smooth_score = ndimage.uniform_filter1d(best_score, size=3, mode="nearest")

    # Candidate peaks after left header zone: a true barline must span ALL 4
    # inter-line spaces (best_min_cov >= 0.35), rejecting 2-3 space note stems.
    min_dist = max(4, int(round(min_bar_dist_sp * target_sp)))
    cand_xs = [
        int(x)
        for x in range(max(1, x_header_end), w - 1)
        if smooth_score[x] >= peak_thresh
        and best_mean_cov[x] >= 0.65
        and best_second_min[x] >= 0.55
        and best_min_cov[x] >= 0.35
        and smooth_score[x] >= smooth_score[x - 1]
        and smooth_score[x] >= smooth_score[x + 1]
    ]

    # Greedy score-descending NMS, then sort left-to-right.
    cand_xs.sort(key=lambda x: float(best_score[x]), reverse=True)
    kept: list[int] = []
    for x in cand_xs:
        if all(abs(x - kx) >= min_dist for kx in kept):
            kept.append(x)
    kept.sort()

    barlines = [
        {
            "x": int(x),
            "score": round(float(best_score[x]), 4),
            "mean_cov": round(float(best_mean_cov[x]), 4),
            "second_min_cov": round(float(best_second_min[x]), 4),
        }
        for x in kept
    ]

    return {
        "barlines": barlines,
        "x_onset": int(x_onset),
        "x_header_end": int(x_header_end),
        "ink_coverage": round(float(ink_coverage), 5),
        "ink_threshold": round(float(ink_thr), 4),
        "strip_width": int(w),
    }


def crop_and_gate_opening_strip(
    strip_img: Image.Image,
    n_bars: int,
    n_tokens: int,
    key_fifths: int = 0,
    target_sp: float = 10.0,
    pad_right_sp: float = 1.5,
    min_confidence: float = 0.45,
    staff_conf: float = 1.0,
    canon_top: float = 12.0,
    slot_pitch: float = 64.0,
) -> dict[str, Any]:
    """Horizontally crop an unwrapped opening-staff strip to ``n_bars`` and gate quality.

    Returns a dict with:
      - ``status``: ``"accepted"`` or ``"rejected"``
      - ``reason``: ``None`` when accepted, else machine-readable rejection reason
      - ``cropped_img``: ``PIL.Image`` when accepted, else ``None``
      - ``confidence``: float in ``[0, 1]``
      - ``crop_x_end``, ``detected_barlines``, ``ink_coverage``, ``px_per_token``, etc.
    """
    strip_w, strip_h = strip_img.size
    det = detect_barline_columns(
        strip_img,
        target_sp=target_sp,
        key_fifths=key_fifths,
        canon_top=canon_top,
        slot_pitch=slot_pitch,
    )
    barlines = det["barlines"]
    ink_cov = det["ink_coverage"]
    x_header_end = det["x_header_end"]

    base_info: dict[str, Any] = {
        "status": "rejected",
        "reason": None,
        "cropped_img": None,
        "confidence": 0.0,
        "crop_x_end": None,
        "strip_width": strip_w,
        "strip_height": strip_h,
        "n_bars_expected": int(n_bars),
        "n_barlines_detected": len(barlines),
        "detected_barlines": [b["x"] for b in barlines],
        "x_onset": det["x_onset"],
        "x_header_end": x_header_end,
        "ink_coverage": ink_cov,
        "px_per_token": None,
        "exact_barline_match": False,
    }

    if ink_cov < 0.012:
        base_info["reason"] = "blank_staff"
        return base_info
    if ink_cov > 0.55:
        base_info["reason"] = "excessive_ink_or_border"
        return base_info

    if n_bars < 1:
        base_info["reason"] = "invalid_n_bars"
        return base_info

    exact_match = False
    if len(barlines) >= n_bars:
        used_bars = barlines[:n_bars]
        x_bar = used_bars[-1]["x"]
        crop_x_end = min(strip_w, int(round(x_bar + pad_right_sp * target_sp)))
        exact_match = True
    elif len(barlines) == n_bars - 1 and n_bars >= 2:
        # Allow right-edge fallback ONLY if the trailing span is a plausible final measure.
        used_bars = barlines
        prev_edges = [x_header_end] + [b["x"] for b in used_bars]
        bar_widths = [b - a for a, b in zip(prev_edges, prev_edges[1:])]
        med_w = float(np.median(bar_widths)) if bar_widths else 10.0 * target_sp
        rem_w = strip_w - used_bars[-1]["x"]
        if rem_w < max(3.0 * target_sp, 0.35 * med_w) or rem_w > 1.85 * max(med_w, 6.0 * target_sp):
            base_info["reason"] = "insufficient_barlines"
            return base_info
        crop_x_end = strip_w
        exact_match = False
    else:
        base_info["reason"] = "insufficient_barlines"
        return base_info

    # Aspect ratio & cropped width sanity.
    min_crop_w = max(int(round(6.0 * target_sp)), int(round(n_bars * 3.5 * target_sp)))
    max_crop_w = int(round(n_bars * 55.0 * target_sp))
    if crop_x_end < min_crop_w or crop_x_end > max_crop_w:
        base_info["crop_x_end"] = int(crop_x_end)
        base_info["reason"] = "bad_aspect_ratio"
        return base_info

    # Token density sanity.
    px_per_tok = float(crop_x_end) / float(max(n_tokens, 1))
    base_info["crop_x_end"] = int(crop_x_end)
    base_info["px_per_token"] = round(px_per_tok, 3)
    base_info["exact_barline_match"] = exact_match
    if px_per_tok < 2.0 or px_per_tok > 40.0:
        base_info["reason"] = "bad_token_density"
        return base_info

    # Re-check ink coverage inside the cropped region so an empty prefix never sneaks through.
    cropped_img = strip_img.crop((0, 0, int(crop_x_end), strip_h))
    crop_gray = np.asarray(cropped_img.convert("L"), dtype=np.float32)[: int(round(slot_pitch)), :] / 255.0
    crop_ink = compute_local_ink_map(crop_gray)
    spaces = _interline_space_slices(canon_top, target_sp, crop_gray.shape[0], line_mask_half_px=1.5)
    space_rows = np.concatenate([np.arange(s.start, s.stop) for s in spaces])
    crop_ink_cov = float((crop_ink[space_rows, :] >= det["ink_threshold"]).mean())
    base_info["crop_ink_coverage"] = round(crop_ink_cov, 5)
    if crop_ink_cov < 0.015:
        base_info["reason"] = "blank_cropped_strip"
        return base_info

    # Decomposed quality / alignment confidence score in [0, 1].
    bar_score = float(np.mean([b["score"] for b in used_bars]))
    if not exact_match:
        bar_score = max(0.0, bar_score - 0.12)

    edges = [x_header_end] + [b["x"] for b in used_bars]
    if not exact_match:
        edges.append(crop_x_end)
    widths_sp = [max(0.1, (b - a) / target_sp) for a, b in zip(edges, edges[1:])]
    med_w_sp = float(np.median(widths_sp)) if widths_sp else 0.0
    if 5.0 <= med_w_sp <= 35.0:
        w_range_score = 1.0
    elif med_w_sp < 5.0:
        w_range_score = max(0.0, med_w_sp / 5.0)
    else:
        w_range_score = max(0.0, 1.0 - (med_w_sp - 35.0) / 30.0)
    if len(widths_sp) >= 2:
        cv = float(np.std(widths_sp) / max(float(np.mean(widths_sp)), 1e-6))
        consistency_score = max(0.0, 1.0 - max(0.0, cv - 0.35) / 1.0)
    else:
        consistency_score = 0.85
    spacing_score = 0.6 * w_range_score + 0.4 * consistency_score

    if 4.5 <= px_per_tok <= 24.0:
        density_score = 1.0
    elif px_per_tok < 4.5:
        density_score = max(0.0, (px_per_tok - 2.0) / 2.5)
    else:
        density_score = max(0.0, 1.0 - (px_per_tok - 24.0) / 16.0)

    if 0.03 <= crop_ink_cov <= 0.30:
        ink_score = 1.0
    elif crop_ink_cov < 0.03:
        ink_score = max(0.0, (crop_ink_cov - 0.012) / 0.018)
    else:
        ink_score = max(0.0, 1.0 - (crop_ink_cov - 0.30) / 0.25)

    s_conf = float(np.clip(staff_conf, 0.0, 1.0))
    confidence = (
        0.40 * bar_score
        + 0.25 * spacing_score
        + 0.15 * density_score
        + 0.10 * ink_score
        + 0.10 * s_conf
    )
    confidence = float(np.clip(confidence, 0.0, 1.0))
    base_info["confidence"] = round(confidence, 4)
    base_info["score_parts"] = {
        "bar": round(bar_score, 4),
        "spacing": round(spacing_score, 4),
        "density": round(density_score, 4),
        "ink": round(ink_score, 4),
        "staff": round(s_conf, 4),
    }

    if confidence < float(min_confidence):
        base_info["reason"] = f"low_confidence:{confidence:.3f}<{min_confidence:.3f}"
        return base_info

    base_info["status"] = "accepted"
    base_info["reason"] = None
    base_info["cropped_img"] = cropped_img
    return base_info


# ---------------------------------------------------------------------------
# Step 2 + 4: End-to-end page processing & pipeline driver
# ---------------------------------------------------------------------------


def _sample_single_staff_ribbon(
    page_img: Image.Image,
    staff: Any,
    sampler: bezier_ribbon.BezierRibbonSampler,
) -> Image.Image:
    """Unwrap one ``CurvedStaff`` into a canonical 4-slot PIL RGB strip via ``BezierRibbonSampler``."""
    import torch

    rgb = page_img if page_img.mode == "RGB" else page_img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    page_ten = torch.from_numpy(arr.transpose(2, 0, 1))
    with torch.no_grad():
        ribbon_ten, _ref = sampler(page_ten, [staff])
    out_u8 = (
        ribbon_ten.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )
    return Image.fromarray(out_u8, mode="RGB")


def process_rism_page(
    record: dict[str, Any],
    staff_unet: Any,
    staff_unet_meta: dict[str, Any],
    out_dir: Path,
    device: str = "cpu",
    min_bars: int = 2,
    min_tokens: int = 25,
    min_confidence: float = 0.45,
    target_sp: float = 10.0,
    height: int = 256,
    pad_right_sp: float = 1.5,
    max_candidate_staves: int = 4,
) -> dict[str, Any]:
    """Process one RISM page: export all unlabeled strips + gated opening-incipit strip."""
    sid = record["sample_id"]
    img_path = Path(str(record.get("image_path") or ""))
    renders_dir = out_dir / "renders"
    targets_dir = out_dir / "targets"
    unlabeled_dir = out_dir / "unlabeled_strips"

    row: dict[str, Any] = {
        "id": sid,
        "split": "train",
        "rism_id": record.get("rism_id"),
        "siglum": record.get("siglum"),
        "image_path": str(img_path),
        "n_bars": int(record.get("n_bars", 0)),
        "n_tokens": int(record.get("n_tokens", 0)),
        "status": "rejected",
        "reason": None,
        "confidence": 0.0,
        "n_systems": 0,
        "n_unlabeled_strips": 0,
        "unlabeled_strips": [],
        "render_path": None,
        "target_path": None,
    }

    if not img_path.is_file():
        row["reason"] = "missing_image"
        return row

    try:
        with Image.open(img_path) as im:
            page_img = im.convert("RGB")
    except OSError as exc:
        row["reason"] = f"unreadable_image:{exc}"
        return row

    # Always run Stage-1 StaffUNet2D v3 (dynamic-res + log_hm + row-ctx) so ALL
    # unwrapped system strips go to unlabeled_strips/ even if the incipit is rejected.
    systems = eval_sys_gate.extract_page_curved_systems(
        staff_unet,
        page_img,
        device=device,
        meta=staff_unet_meta,
        force_dynamic=True,
    )
    row["n_systems"] = len(systems)

    unlabeled_paths: list[str] = []
    for sys_idx, csys in enumerate(systems):
        chunks = bezier_ribbon.sample_curved_system_chunks(
            page_img,
            csys,
            max_staves=bezier_ribbon.MAX_STAVES,
            target_sp=target_sp,
            height=height,
        )
        for chunk_idx, (strip_pil, _ref, _sidxs) in enumerate(chunks):
            assert isinstance(strip_pil, Image.Image)
            if len(chunks) == 1:
                fname = f"{sid}_s{sys_idx}.png"
            else:
                fname = f"{sid}_s{sys_idx}_c{chunk_idx}.png"
            dst = unlabeled_dir / fname
            atomic_save_png(strip_pil, dst)
            unlabeled_paths.append(str(Path("unlabeled_strips") / fname))

    row["n_unlabeled_strips"] = len(unlabeled_paths)
    row["unlabeled_strips"] = unlabeled_paths

    # Short-tail incipit gate (checked after exporting unlabeled strips so Stage-2
    # unsupervised rounds still get all systems from short-tail RISM pages).
    ok_tail, tail_reason = filter_short_tail(record, min_bars=min_bars, min_tokens=min_tokens)
    if not ok_tail:
        row["reason"] = tail_reason
        return row

    if not systems or not systems[0].staves:
        row["reason"] = "no_systems_detected"
        return row

    # Candidate opening staves: top system's staves, prioritizing staff 0.
    canon_top, slot_pitch, _out_h = bezier_ribbon.canonical_layout(
        target_sp=target_sp, height=height, n_slots=bezier_ribbon.MAX_STAVES
    )
    sampler = bezier_ribbon.BezierRibbonSampler(
        target_sp=target_sp,
        height=height,
        n_slots=bezier_ribbon.MAX_STAVES,
        offset_mode="normal",
        paper_value=1.0,
    )

    cand_staves = list(systems[0].staves[: max(1, int(max_candidate_staves))])
    chosen_result: dict[str, Any] | None = None
    chosen_staff_idx = 0
    first_result: dict[str, Any] | None = None

    for s_idx, staff in enumerate(cand_staves):
        strip_pil = _sample_single_staff_ribbon(page_img, staff, sampler)
        res = crop_and_gate_opening_strip(
            strip_pil,
            n_bars=int(record["n_bars"]),
            n_tokens=int(record["n_tokens"]),
            key_fifths=int(record.get("key_fifths", 0)),
            target_sp=target_sp,
            pad_right_sp=pad_right_sp,
            min_confidence=min_confidence,
            staff_conf=float(getattr(staff, "confidence", 1.0)),
            canon_top=canon_top,
            slot_pitch=slot_pitch,
        )
        if first_result is None:
            first_result = res
        if res["status"] == "accepted":
            chosen_result = res
            chosen_staff_idx = s_idx
            break
        # If staff 0 had ink and failed barline/aspect gate (not blank), still allow
        # checking staff 1..k in case top staff is a non-incipit instrument in a score.

    final_res = chosen_result if chosen_result is not None else first_result
    assert final_res is not None
    row["opening_staff_idx"] = chosen_staff_idx
    row["confidence"] = final_res["confidence"]
    row["crop_x_end"] = final_res["crop_x_end"]
    row["strip_width"] = final_res["strip_width"]
    row["n_barlines_detected"] = final_res["n_barlines_detected"]
    row["detected_barlines"] = final_res["detected_barlines"]
    row["ink_coverage"] = final_res["ink_coverage"]
    row["px_per_token"] = final_res["px_per_token"]
    row["exact_barline_match"] = final_res["exact_barline_match"]

    if final_res["status"] != "accepted":
        row["status"] = "rejected"
        row["reason"] = final_res["reason"]
        return row

    cropped_img = final_res["cropped_img"]
    assert isinstance(cropped_img, Image.Image)
    render_dst = renders_dir / f"{sid}.png"
    target_dst = targets_dir / f"{sid}-bekern.txt"
    atomic_save_png(cropped_img, render_dst)
    bekern_line = " ".join(str(record["bekern"]).strip().split()) + "\n"
    atomic_write_text(target_dst, bekern_line)

    row["status"] = "accepted"
    row["reason"] = None
    row["render_path"] = str(Path("renders") / f"{sid}.png")
    row["target_path"] = str(Path("targets") / f"{sid}-bekern.txt")
    return row


def run_pipeline(
    rism_dirs: Sequence[str | Path],
    staff_unet_ckpt: str | Path,
    out_dir: str | Path,
    limit: int | None = None,
    min_bars: int = 2,
    min_tokens: int = 25,
    min_confidence: float = 0.45,
    device: str = "cpu",
    resume: bool = False,
    eval_manifests: Sequence[str | Path] = DEFAULT_EVAL_MANIFESTS,
    target_sp: float = 10.0,
    height: int = 256,
    pad_right_sp: float = 1.5,
) -> dict[str, Any]:
    """Run the full RISM opening-incipit & unlabeled-strip pipeline."""
    t0 = time.time()
    out_root = Path(out_dir)
    (out_root / "renders").mkdir(parents=True, exist_ok=True)
    (out_root / "targets").mkdir(parents=True, exist_ok=True)
    (out_root / "unlabeled_strips").mkdir(parents=True, exist_ok=True)

    records = load_rism_manifests(rism_dirs, limit=limit)
    assert_no_eval_overlap(records, eval_manifests=eval_manifests)

    manifest_path = out_root / "manifest.jsonl"
    existing_rows: dict[str, dict[str, Any]] = {}
    if resume and manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                if "id" in r:
                    existing_rows[str(r["id"])] = r

    # Export stage1_pages.txt (all valid existing full page paths, sorted & deduplicated).
    stage1_pages = sorted(
        {
            str(Path(str(rec["image_path"])))
            for rec in records
            if rec.get("image_path") and Path(str(rec["image_path"])).is_file()
        }
    )
    stage1_path = out_root / "stage1_pages.txt"
    atomic_write_text(
        stage1_path,
        ("\n".join(stage1_pages) + "\n") if stage1_pages else "",
    )

    to_process = [rec for rec in records if rec["sample_id"] not in existing_rows]
    staff_unet = None
    staff_unet_meta: dict[str, Any] = {}
    if to_process:
        staff_unet, staff_unet_meta = eval_sys_gate.load_staff_unet(
            str(staff_unet_ckpt), device=device
        )

    rows_by_id = dict(existing_rows)
    for rec in to_process:
        row = process_rism_page(
            record=rec,
            staff_unet=staff_unet,
            staff_unet_meta=staff_unet_meta,
            out_dir=out_root,
            device=device,
            min_bars=min_bars,
            min_tokens=min_tokens,
            min_confidence=min_confidence,
            target_sp=target_sp,
            height=height,
            pad_right_sp=pad_right_sp,
        )
        rows_by_id[row["id"]] = row

    # Preserve manifest order matching loaded records (+ any resumed rows).
    ordered_rows: list[dict[str, Any]] = []
    emitted: set[str] = set()
    for rec in records:
        sid = rec["sample_id"]
        if sid in rows_by_id:
            ordered_rows.append(rows_by_id[sid])
            emitted.add(sid)
    for sid, row in rows_by_id.items():
        if sid not in emitted:
            ordered_rows.append(row)

    atomic_write_jsonl(manifest_path, ordered_rows)

    accepted = sum(1 for r in ordered_rows if r.get("status") == "accepted")
    rejected = sum(1 for r in ordered_rows if r.get("status") != "accepted")
    n_unlabeled = sum(int(r.get("n_unlabeled_strips", 0)) for r in ordered_rows)
    reasons: dict[str, int] = {}
    for r in ordered_rows:
        if r.get("status") != "accepted":
            key = str(r.get("reason") or "unknown").split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1

    return {
        "total_records": len(ordered_rows),
        "processed_this_run": len(to_process),
        "resumed_skipped": len(records) - len(to_process),
        "accepted": accepted,
        "rejected": rejected,
        "accept_rate": round(accepted / max(len(ordered_rows), 1), 4),
        "total_unlabeled_strips": n_unlabeled,
        "stage1_pages": len(stage1_pages),
        "reject_reasons": reasons,
        "elapsed_secs": round(time.time() - t0, 2),
        "out_dir": str(out_root),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build RISM opening-incipit strips, unlabeled system strips, and Stage-1 page list."
    )
    ap.add_argument(
        "--rism-dirs",
        required=True,
        help="Comma-separated RISM dataset directories containing manifest.jsonl",
    )
    ap.add_argument(
        "--staff-unet-ckpt",
        default=r"D:\tmp\mvd-runs\staff-unet2d-v3-dynres\staff_unet2d.pt",
        help="Stage-1 StaffUNet2D v3 checkpoint path",
    )
    ap.add_argument(
        "--out-dir",
        default=r"D:\tmp\rism-incipit-strips",
        help="Output directory for renders/, targets/, unlabeled_strips/, manifest.jsonl, stage1_pages.txt",
    )
    ap.add_argument("--limit", type=int, default=None, help="Cap total RISM records processed")
    ap.add_argument("--min-bars", type=int, default=2, help="Minimum barlines in opening incipit")
    ap.add_argument("--min-tokens", type=int, default=25, help="Minimum bekern tokens in opening incipit")
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.45,
        help="Minimum alignment/quality confidence to accept opening strip",
    )
    ap.add_argument("--pad-right-sp", type=float, default=1.5, help="Right padding past closing barline (in sp)")
    ap.add_argument("--target-sp", type=float, default=10.0, help="Canonical ribbon staff spacing (px)")
    ap.add_argument("--height", type=int, default=256, help="Canonical ribbon strip height (px)")
    ap.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    ap.add_argument("--resume", action="store_true", help="Skip sample IDs already recorded in manifest.jsonl")
    ap.add_argument(
        "--eval-manifests",
        default=None,
        help="Optional comma-separated eval manifest JSONs for leakage check",
    )
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rism_dirs = [p.strip() for p in args.rism_dirs.split(",") if p.strip()]
    eval_manifests = (
        [p.strip() for p in args.eval_manifests.split(",") if p.strip()]
        if args.eval_manifests
        else list(DEFAULT_EVAL_MANIFESTS)
    )
    summary = run_pipeline(
        rism_dirs=rism_dirs,
        staff_unet_ckpt=args.staff_unet_ckpt,
        out_dir=args.out_dir,
        limit=args.limit,
        min_bars=args.min_bars,
        min_tokens=args.min_tokens,
        min_confidence=args.min_confidence,
        device=args.device,
        resume=args.resume,
        eval_manifests=eval_manifests,
        target_sp=args.target_sp,
        height=args.height,
        pad_right_sp=args.pad_right_sp,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
