#!/usr/bin/env python3
"""100% Model-Free Barline + Verovio Measure-Bank GOLD Slicer (v2).

STRICT CONSTRAINT: ZERO neural OMR models, ZERO torch imports, CPU-only!

Architecture:
  0. CorpusIndex (O(1) deterministic lookup, abstain on ambiguity) +
     EvalLeakageGuard (combined cross-field check) + hash-keyed atomic XML cache
     with bounded LRU (<= 32 parsed IndexedMusicXML objects).
  1. Stage 2 — Visual Canvas & Physical Staff-Count Gate:
     - EXIF transpose, dark scanner border trim, normalized working width (~1200 px,
       upscaling if < 900 px), coarse-to-fine deskew (0.5° over ±6° -> 0.1° over ±0.5°).
     - cv2.MORPH_OPEN horizontal line detection, recursive cluster splitting for 3+
       merged staves, p75 blank-staff filter preserving tacet/rest staves connected by
       left brace/barline or regular d_line & cov_frac >= 0.45.
     - Unsorted bar_xs_seen dedup, vertical gap-periodicity clustering alongside barline
       continuity, gap-ratio-checked [1,2]->[3] merge (including single-system [1,2]).
     - NEVER silently drop inked systems: require all systems on the page to have the
       same staff count == target_staves (with 1-staff partbook support when
       vis_staves == 1 and matched_staff is known).
     - Midpoint-clamped system_crops and staff_bands.
  2. Stage 3 — Barline Hinting + Verovio Single-Render SVG Measure Bank + Global 2-Best
     Viterbi DP + Onset-Sensitive Itakura Slope-Constrained DTW + Strict 3-Tier Gate:
     - Per-system ink_vert slicing, fixed two-stage _cluster_cols, ±10° slant-tolerant
       horizontal max-pool + sheared gap projection, bar_conf in [0, 1], 1-staff stem+
       notehead rejection (bar_conf = 0.0 -> wide [1..12] DP window).
     - IndexedMusicXML with O(M) prefix attribute table, mid-measure <attributes> fix,
       MusicXML 3.1 DTD ordering, print-object="no" on carried-forward <time>, and
       extract_single_staff_musicxml for 1-staff partbooks.
     - VerovioMeasureBank rendering contiguous windows [m_0, m_0 + 48) once, stripping
       SVG staff-line <path> elements, asserting page_count == 1 and exact staff count.
     - High-res (N in [192, 320], sigma=1.4) 9%-preamble-masked per-staff 5-channel
       profiles (dens_z, onset_z, upper_z, lower_z, cent_z) with activity-aware std_floor.
     - Vectorized Type-II / Itakura slope-constrained DTW ((1,1), (2,1), (1,2)) with
       degenerate warp rejection (mean(|i-j|)/N > 0.10).
     - Global 2-Best Viterbi DP (unpenalized) + 3-Tier Label-Integrity Gate:
       Tier 1: Within-piece Decoy Z-Score (>= 2.5) + max-decoy gap (>= 0.05) per system.
       Tier 2: Per-staff verification along shared DTW path P* (min >= 0.22, spread <= 0.45).
       Tier 3: 2-Best DP path margin (>= 0.025/sys) + score floors + null-start DP control.
  3. Export & Reliability:
     - Uniform v2a (or uniform v1, NEVER mixed), hard grammar rejection, post-export
       bekern measure-count assertion, atomic writes, --resume via state.jsonl,
       streaming diagnostics.jsonl, work-grouped ProcessPoolExecutor scaling.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import xml.etree.ElementTree as ET
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

REPO_ROOT = Path(__file__).resolve().parents[2]
MVD_DIR = Path(__file__).resolve().parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import data  # noqa: E402
import fingerprint_harvester as fh  # noqa: E402
import grammar  # noqa: E402
import neg_control_gate as ncg  # noqa: E402

GOLD_JSONL = Path(r"D:\tmp\hw-align\rism_pae_100k_gold_matches.jsonl")
DEFAULT_OUT_DIR = Path(r"D:\tmp\rism-gold-modelfree-iter16")
ASER_BIN = (
    REPO_ROOT
    / "converter"
    / "assembly-serialize"
    / "target"
    / "release"
    / "assembly-serialize.exe"
)
NORTHSTAR_MANIFESTS = [
    Path(r"D:\tmp\northstar-eval-pack\manifest.json"),
    Path(r"D:\tmp\northstar-eval-pack-v0.2\manifest.json"),
    Path(r"D:\tmp\omr-northstar-hf-v0.1\manifest.json"),
]
DEFAULT_CORPUS_ROOTS: Dict[str, Path] = {
    "openscore_sq": Path(r"D:\tmp\openscore-sq-repo"),
    "openscore_lieder": Path(r"D:\tmp\openscore-lieder"),
    "nifc_chopin": Path(r"D:\tmp\nifc-chopin-repo\kern"),
    "nifc_polish": Path(r"D:\tmp\nifc-polish-scores-repo"),
    "pdmx": Path(r"D:\tmp\corpus10x\pdmx\renders"),
}

# End-of-measure barline tokens in bekern (excludes bar-rest and bar-repeat-left)
END_OF_MEASURE_BAR_TOKENS: Set[str] = {
    "bar",
    "bar-double",
    "bar-light-heavy",
    "bar-repeat-right",
}

_SVG_HLINE_RE = re.compile(
    r"^M\s*([-\d.]+)\s+([-\d.]+)\s+L\s*([-\d.]+)\s+([-\d.]+)\s*$"
)
_SVG_TRANSLATE_RE = re.compile(r"translate\(\s*([-\d.]+)(?:[\s,]+([-\d.]+))?\s*\)")

# Type alias for injectable Verovio/measure-slice renderer in tests:
# render_fn(indexed_xml, m_start, m_end, n_staves) -> Optional[(sym_ink, staff_bands, has_preamble)]
RenderSliceFn = Callable[
    ["IndexedMusicXML", int, int, int],
    Optional[Tuple[np.ndarray, List[Tuple[int, int]], bool]],
]


# ---------------------------------------------------------------------------
# Atomic I/O & Safe Utilities
# ---------------------------------------------------------------------------

def configure_stdout_utf8() -> None:
    """Safely configure stdout/stderr to UTF-8 with replacement on Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Atomically write bytes via *.tmp.<pid> + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Atomically write UTF-8 text via *.tmp.<pid> + os.replace."""
    atomic_write_bytes(path, text.encode(encoding))


def append_jsonl_line(path: Path, obj: Dict[str, Any]) -> None:
    """Append a single JSONL record and flush immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def sanitize_page_stem(rism_id: str, img_path: Path) -> str:
    """Build a filesystem-safe, collision-resistant unique page stem."""
    raw_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(rism_id or "unknown")).strip("_")
    raw_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", img_path.stem).strip("_")
    uniq = hashlib.sha1(f"{rism_id}|{str(img_path)}".encode("utf-8")).hexdigest()[:8]
    base = f"rism_{raw_id}_{raw_stem}"[:80].rstrip("_")
    return f"{base}_{uniq}"


def safe_open_image(img_path: Path) -> Image.Image:
    """Open an image and safely apply EXIF orientation transpose."""
    im = Image.open(img_path)
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    return im


def _file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 0. CorpusIndex, Digital Source Resolution & Eval Leakage Guard
# ---------------------------------------------------------------------------

class CorpusIndex:
    """One-pass O(1) deterministic corpus file index. Abstains on ambiguity (len != 1)."""

    def __init__(self, roots: Optional[Dict[str, Path]] = None) -> None:
        self.roots = dict(DEFAULT_CORPUS_ROOTS if roots is None else roots)
        self._sq: Dict[str, List[Path]] = {}
        self._lc: Dict[str, List[Path]] = {}
        self._chopin: Dict[str, List[Path]] = {}
        self._polish: List[Path] = []
        self._pdmx: Dict[str, List[Path]] = {}
        self._built = False

    def build(self) -> "CorpusIndex":
        if self._built:
            return self
        sq_root = self.roots.get("openscore_sq")
        if sq_root and sq_root.is_dir():
            for p in sorted(sq_root.rglob("*.mxl")):
                self._sq.setdefault(p.stem.lower(), []).append(p)

        lc_root = self.roots.get("openscore_lieder")
        if lc_root and lc_root.is_dir():
            for p in sorted(lc_root.rglob("*.mxl")):
                self._lc.setdefault(p.stem.lower(), []).append(p)

        ch_root = self.roots.get("nifc_chopin")
        if ch_root and ch_root.is_dir():
            for p in sorted(ch_root.glob("*.krn")):
                self._chopin.setdefault(p.stem, []).append(p)

        pl_root = self.roots.get("nifc_polish")
        if pl_root and pl_root.is_dir():
            self._polish = sorted(pl_root.rglob("*.krn"))

        pdmx_root = self.roots.get("pdmx")
        if pdmx_root and pdmx_root.is_dir():
            for p in sorted(pdmx_root.glob("*.musicxml")):
                m = re.match(r"^(Qm[A-Za-z0-9]+)", p.stem)
                if m:
                    self._pdmx.setdefault(m.group(1), []).append(p)

        self._built = True
        return self

    @staticmethod
    def _unique_hit(hits: Sequence[Path]) -> Optional[Tuple[str, str]]:
        if len(hits) == 1:
            return str(hits[0]), hits[0].suffix.lower()
        return None

    def resolve(self, work_id: str, work_source_path: str) -> Tuple[str, str]:
        """Resolve rendered .png/.txt GOLD sources to their unique full digital score."""
        if not self._built:
            self.build()
        p = Path(work_source_path) if work_source_path else Path("")
        ext = p.suffix.lower()
        if ext in (".mxl", ".musicxml", ".xml", ".krn") and p.is_file():
            return str(p), ext

        wid = work_id or ""
        m_sq = re.search(r"(sq\d+)", wid)
        if m_sq:
            hit = self._unique_hit(self._sq.get(m_sq.group(1).lower(), ()))
            if hit:
                return hit
            return "", ""

        m_lc = re.search(r"(lc\d+)", wid)
        if m_lc:
            hit = self._unique_hit(self._lc.get(m_lc.group(1).lower(), ()))
            if hit:
                return hit
            return "", ""

        m_ch = re.match(r"c100k_nifc_chopin_first_editions_(.+?)_[0-9a-f]{10}__", wid)
        if m_ch:
            hit = self._unique_hit(self._chopin.get(m_ch.group(1), ()))
            if hit:
                return hit
            return "", ""

        m_pl = re.match(r"c100k_nifc_polish_scores_(.+?)_[0-9a-f]{10}__", wid)
        if m_pl:
            pfx = m_pl.group(1)
            hits = [hp for hp in self._polish if hp.stem.startswith(pfx)]
            if not hits:
                short_pfx = pfx[:22]
                hits = [hp for hp in self._polish if hp.stem.startswith(short_pfx)]
            hit = self._unique_hit(hits)
            if hit:
                return hit
            return "", ""

        m_pdmx = re.search(r"(Qm[A-Za-z0-9]+)", wid)
        if m_pdmx:
            hit = self._unique_hit(self._pdmx.get(m_pdmx.group(1), ()))
            if hit:
                return hit
            return "", ""

        return str(p), ext


def build_combined_leakage_guards(
    northstar_manifests: Sequence[Path] = NORTHSTAR_MANIFESTS,
) -> List[Tuple[str, fh.EvalLeakageGuard]]:
    guards: List[Tuple[str, fh.EvalLeakageGuard]] = []
    for ns_path in northstar_manifests:
        if Path(ns_path).is_file():
            guards.append(
                (
                    Path(ns_path).parent.name,
                    fh.EvalLeakageGuard.from_files(northstar_manifest=ns_path),
                )
            )
    if not guards:
        guards.append(("default", fh.EvalLeakageGuard.from_files()))
    return guards


def check_eval_overlap(
    row: Dict[str, Any], guards: Sequence[Tuple[str, fh.EvalLeakageGuard]]
) -> Tuple[bool, str]:
    """Check individual fields AND combined cross-field string against EvalLeakageGuard."""
    rism_id = str(row.get("rism_id", ""))
    img_stem = Path(str(row.get("image_path", ""))).stem
    work_id = str(row.get("work_id", ""))
    work_src = str(row.get("work_source_path", ""))
    resolved_src = str(row.get("resolved_source_path", ""))
    title = str(row.get("title", ""))[:200]
    combined = f"{work_src} {resolved_src} {work_id} {title}".strip()

    ids_to_check = [
        rism_id,
        img_stem,
        work_id,
        work_src,
        resolved_src,
        title,
        combined,
    ]
    for gname, guard in guards:
        leaked, reason = guard.check(*ids_to_check)
        if leaked:
            return True, f"{gname}:{reason}"
    return False, ""


def convert_to_musicxml(src_path: Path, cache_dir: Path) -> Optional[Path]:
    """Convert .mxl or .krn into a hash-keyed uncompressed .musicxml in cache_dir atomically."""
    if not src_path.is_file():
        return None
    ext = src_path.suffix.lower()
    if ext in (".musicxml", ".xml"):
        return src_path

    stem_clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", src_path.stem)
    sha1_short = _file_sha1(src_path)[:12]
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_xml = cache_dir / f"{stem_clean}_{sha1_short}.musicxml"
    if out_xml.is_file() and out_xml.stat().st_size > 100:
        return out_xml

    if ext == ".mxl":
        try:
            with zipfile.ZipFile(src_path, "r") as zf:
                target = None
                if "META-INF/container.xml" in zf.namelist():
                    croot = ET.fromstring(zf.read("META-INF/container.xml"))
                    for el in croot.iter():
                        if el.tag.endswith("rootfile") and "full-path" in el.attrib:
                            target = el.attrib["full-path"]
                            break
                if not target or target not in zf.namelist():
                    for name in zf.namelist():
                        low = name.lower()
                        if not low.startswith("meta-inf/") and (
                            low.endswith(".xml") or low.endswith(".musicxml")
                        ):
                            target = name
                            break
                if not target:
                    return None
                atomic_write_bytes(out_xml, zf.read(target))
            return out_xml
        except Exception:
            return None

    if ext == ".krn":
        try:
            import music21

            score = music21.converter.parse(
                str(src_path), forceSource=True, storePickle=False
            )
            tmp_xml = out_xml.with_name(f"{out_xml.name}.tmp.{os.getpid()}")
            try:
                score.write("musicxml", fp=str(tmp_xml))
                if tmp_xml.is_file() and tmp_xml.stat().st_size > 100:
                    os.replace(tmp_xml, out_xml)
                    return out_xml
            finally:
                tmp_xml.unlink(missing_ok=True)
        except Exception:
            return None

    return None


# ---------------------------------------------------------------------------
# 1. Stage 2 — Visual Canvas & Physical Staff-Count Gate (CPU)
# ---------------------------------------------------------------------------

def _trim_dark_scanner_borders(gray: np.ndarray, max_frac: float = 0.12) -> np.ndarray:
    """Detect and trim dark outer scanner borders (e.g. black book cradles)."""
    h, w = gray.shape
    if h < 64 or w < 64:
        return gray
    center = gray[int(0.2 * h) : int(0.8 * h), int(0.2 * w) : int(0.8 * w)]
    med_bg = float(np.percentile(center, 75)) if center.size > 0 else 200.0
    dark_thr = min(95.0, 0.55 * med_bg)
    if dark_thr < 20.0:
        return gray

    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)
    max_y = int(h * max_frac)
    max_x = int(w * max_frac)

    y0 = 0
    while y0 < max_y and row_means[y0] < dark_thr:
        y0 += 1
    y1 = h
    while (h - y1) < max_y and row_means[y1 - 1] < dark_thr:
        y1 -= 1
    x0 = 0
    while x0 < max_x and col_means[x0] < dark_thr:
        x0 += 1
    x1 = w
    while (w - x1) < max_x and col_means[x1 - 1] < dark_thr:
        x1 -= 1

    if (y1 - y0) < int(0.6 * h) or (x1 - x0) < int(0.6 * w):
        return gray
    if y0 == 0 and y1 == h and x0 == 0 and x1 == w:
        return gray
    return gray[y0:y1, x0:x1].copy()


def _deskew_gray(
    gray: np.ndarray,
    max_angle: float = 6.0,
    coarse_step: float = 0.5,
    fine_step: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """Coarse-to-fine deskew (0.5° over ±6° -> 0.1° over ±0.5°) maximizing staff-line sharpness."""
    h, w = gray.shape
    scale = min(1.0, 600.0 / float(max(w, 1)))
    small = cv2.resize(
        gray,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    sh, sw = small.shape
    k_bg = max(9, int(round(sh * 0.018)) | 1)
    bg = cv2.GaussianBlur(small.astype(np.float32), (1, k_bg), 0)
    diff = np.clip(bg - small.astype(np.float32), 0.0, 255.0)
    y0, y1 = int(0.05 * sh), int(0.95 * sh)
    x0, x1 = int(0.05 * sw), int(0.95 * sw)
    roi = diff[y0:y1, x0:x1]
    if roi.size == 0:
        return gray, 0.0

    center_s = (roi.shape[1] * 0.5, roi.shape[0] * 0.5)

    def _eval_angles(angles: np.ndarray) -> Tuple[float, float]:
        best_a = 0.0
        best_s = -1.0
        for ang in angles:
            ang_f = float(ang)
            if abs(ang_f) < 1e-6:
                rot = roi
            else:
                M = cv2.getRotationMatrix2D(center_s, ang_f, 1.0)
                rot = cv2.warpAffine(roi, M, (roi.shape[1], roi.shape[0]), borderValue=0.0)
            proj = rot.mean(axis=1)
            sc = float(np.var(np.diff(proj)))
            if sc > best_s:
                best_s = sc
                best_a = ang_f
        return best_a, best_s

    coarse_angles = np.arange(-max_angle, max_angle + 1e-6, coarse_step)
    coarse_best, _ = _eval_angles(coarse_angles)

    fine_lo = max(-max_angle, coarse_best - coarse_step)
    fine_hi = min(max_angle, coarse_best + coarse_step)
    fine_angles = np.arange(fine_lo, fine_hi + 1e-6, fine_step)
    best_angle, _ = _eval_angles(fine_angles)

    if abs(best_angle) < 0.08:
        return gray, 0.0

    fill_val = float(np.percentile(gray, 85))
    M_full = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), best_angle, 1.0)
    deskewed = cv2.warpAffine(
        gray,
        M_full,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_val,
    )
    return deskewed, best_angle


def _horizontal_runs_mask(ink: np.ndarray, min_run: int) -> np.ndarray:
    """Keep pixels belonging to horizontal True runs of length >= min_run using cv2.MORPH_OPEN."""
    if min_run <= 1:
        return ink.astype(bool).copy()
    kernel = np.ones((1, int(min_run)), dtype=np.uint8)
    opened = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return opened > 0


def _split_merged_staff_clusters(
    clusters: List[List[int]], d_line: float
) -> List[List[int]]:
    """Recursively split clusters containing 2, 3, or more merged staves."""
    min_inter_staff_gap = max(1.25 * float(d_line), float(d_line) + 2.0)
    queue: List[List[int]] = list(clusters)
    split_clusters: List[List[int]] = []
    while queue:
        cl = queue.pop(0)
        if len(cl) >= 8 and (cl[-1] - cl[0]) >= 7.0 * d_line:
            cgaps = np.diff(cl)
            # Search for the largest gap leaving at least 3 lines on each side
            lo_idx = 3
            hi_idx = len(cl) - 3
            if hi_idx >= lo_idx:
                cut_rel = int(np.argmax(cgaps[lo_idx - 1 : hi_idx]))
                cut = lo_idx + cut_rel
                if float(cgaps[cut - 1]) > min_inter_staff_gap:
                    left = cl[:cut]
                    right = cl[cut:]
                    if len(left) >= 3 and len(right) >= 3:
                        queue.insert(0, right)
                        queue.insert(0, left)
                        continue
        split_clusters.append(cl)
    return split_clusters


def _dedup_bar_xs_unsorted(gx: int, bar_xs_seen: List[int], min_sep: int = 20) -> bool:
    """Return True if gx is at least min_sep away from all previously seen x positions."""
    return all(abs(gx - px) >= min_sep for px in bar_xs_seen)


def _clamp_system_crops_to_midpoints(
    system_crops: List[Dict[str, Any]], h: int
) -> None:
    """Clamp adjacent system crops' y_top/y_bot to midpoints so neighbors never overlap."""
    for i in range(len(system_crops) - 1):
        c0 = system_crops[i]
        c1 = system_crops[i + 1]
        if c0["y_bot"] > c1["y_top"]:
            s0_bot = c0["staves"][-1]["y_bot"]
            s1_top = c1["staves"][0]["y_top"]
            mid = int(round(0.5 * (s0_bot + s1_top)))
            mid = max(c0["y_top"] + 1, min(c1["y_bot"] - 1, mid))
            c0["y_bot"] = mid
            c1["y_top"] = mid
    for c in system_crops:
        c["y_top"] = max(0, min(h - 1, int(c["y_top"])))
        c["y_bot"] = max(c["y_top"] + 1, min(h, int(c["y_bot"])))


def clamp_staff_bands_to_midpoints(
    staff_bands: List[Tuple[int, int]],
    staff_cores: List[Tuple[int, int]],
    ch: int,
) -> List[Tuple[int, int]]:
    """Clamp per-staff vertical bands to midpoints between adjacent staff cores."""
    out = [[max(0, y0), min(ch, y1)] for y0, y1 in staff_bands]
    for i in range(len(out) - 1):
        if out[i][1] > out[i + 1][0]:
            mid = int(round(0.5 * (staff_cores[i][1] + staff_cores[i + 1][0])))
            mid = max(out[i][0] + 1, min(out[i + 1][1] - 1, mid))
            out[i][1] = mid
            out[i + 1][0] = mid
    return [(int(a), max(int(a) + 1, int(b))) for a, b in out]


def detect_staves_and_systems(
    pil_im: Image.Image,
    *,
    force_single_staff: bool = False,
) -> Dict[str, Any]:
    """Detect physical 5-line staves, group into systems, and return per-system crops."""
    gray_pil = pil_im.convert("L")
    orig_w, orig_h = gray_pil.size
    if orig_w < 32 or orig_h < 32:
        return {
            "n_staves_total": 0,
            "visual_n_systems": 0,
            "visual_n_staves": 0,
            "all_systems_uniform": False,
            "system_staff_counts": [],
            "system_crops": [],
            "angle_deg": 0.0,
            "d_line": 0.0,
            "reason": "image_too_small",
        }

    # Normalize working width to ~1200 px (upscale if < 900, downscale if > 1200)
    if orig_w > 1200 or orig_w < 900:
        work_w = 1200
    else:
        work_w = orig_w
    work_h = max(1, int(round(orig_h * (work_w / float(orig_w)))))
    gray = np.asarray(
        gray_pil.resize((work_w, work_h), Image.Resampling.BILINEAR), dtype=np.uint8
    )

    gray = _trim_dark_scanner_borders(gray)
    deskewed, angle_deg = _deskew_gray(gray)
    h, w = deskewed.shape

    k_bg_y = max(15, int(round(h * 0.020)) | 1)
    bg_y = cv2.GaussianBlur(deskewed.astype(np.float32), (1, k_bg_y), 0)
    diff_h = np.clip(bg_y - deskewed.astype(np.float32), 0.0, 255.0)

    k_bg_x = max(15, int(round(w * 0.025)) | 1)
    bg_x = cv2.GaussianBlur(deskewed.astype(np.float32), (k_bg_x, 1), 0)
    diff_v = np.clip(bg_x - deskewed.astype(np.float32), 0.0, 255.0)

    pos_h = diff_h[diff_h > 4.0]
    pos_v = diff_v[diff_v > 4.0]
    if pos_h.size < 100 or pos_v.size < 100:
        return {
            "n_staves_total": 0,
            "visual_n_systems": 0,
            "visual_n_staves": 0,
            "all_systems_uniform": False,
            "system_staff_counts": [],
            "system_crops": [],
            "angle_deg": round(angle_deg, 2),
            "d_line": 0.0,
            "reason": "no_ink",
        }
    p85_h = float(np.percentile(pos_h, 85))
    p85_v = float(np.percentile(pos_v, 85))
    thr_line = max(8.0, min(26.0, 0.28 * p85_h))
    thr_vert = max(8.0, min(28.0, 0.30 * p85_v))

    ink_line = diff_h > thr_line
    ink_vert = diff_v > thr_vert
    del diff_h, diff_v  # free float32 buffers immediately

    bx = max(4, int(0.02 * w))
    by = max(4, int(0.02 * h))
    for arr in (ink_line, ink_vert):
        arr[:by, :] = False
        arr[h - by :, :] = False
        arr[:, :bx] = False
        arr[:, w - bx :] = False

    min_h_run = max(14, int(round(0.015 * w)))
    h_mask = _horizontal_runs_mask(ink_line, min_h_run)

    def _detect_line_peaks(mask_sub: np.ndarray) -> Tuple[List[int], float, str]:
        proj_sub = mask_sub.mean(axis=1).astype(np.float32)
        proj_sm = cv2.GaussianBlur(proj_sub.reshape(-1, 1), (1, 3), 0).ravel()
        nz_sub = proj_sm[proj_sm > 0.02]
        if nz_sub.size < 10:
            return [], 0.0, "no_horizontal_lines"
        p_thr = max(0.06, 0.24 * float(np.percentile(nz_sub, 95)))
        m_dist = max(3, int(round(0.0028 * h)))
        pks: List[int] = []
        for y in range(1, h - 1):
            if (
                proj_sm[y] >= p_thr
                and proj_sm[y] >= proj_sm[y - 1]
                and proj_sm[y] >= proj_sm[y + 1]
            ):
                if pks and (y - pks[-1]) < m_dist:
                    if proj_sm[y] > proj_sm[pks[-1]]:
                        pks[-1] = y
                else:
                    pks.append(y)
        if len(pks) < 3:
            return pks, 0.0, f"too_few_line_peaks({len(pks)})"
        g_arr = np.diff(pks)
        sg_arr = g_arr[(g_arr >= 3) & (g_arr <= max(18, int(0.03 * h)))]
        if sg_arr.size == 0:
            return pks, 0.0, "no_regular_line_spacing"
        dl = max(3.5, float(np.median(sg_arr)))
        return pks, dl, "ok"

    peaks, d_line, pk_reason = _detect_line_peaks(h_mask)
    if pk_reason != "ok":
        # Central 50% strip fallback for curved/warped archival pages
        cx0 = int(0.25 * w)
        cx1 = max(cx0 + 1, int(0.75 * w))
        peaks, d_line, pk_reason = _detect_line_peaks(h_mask[:, cx0:cx1])
    if pk_reason != "ok":
        return {
            "n_staves_total": 0,
            "visual_n_systems": 0,
            "visual_n_staves": 0,
            "all_systems_uniform": False,
            "system_staff_counts": [],
            "system_crops": [],
            "angle_deg": round(angle_deg, 2),
            "d_line": 0.0,
            "reason": pk_reason,
        }

    max_intra_gap = max(d_line * 1.9, d_line + 4.0)
    clusters: List[List[int]] = [[peaks[0]]]
    for p in peaks[1:]:
        if (p - clusters[-1][-1]) <= max_intra_gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    split_clusters = _split_merged_staff_clusters(clusters, d_line)

    raw_staves: List[Dict[str, Any]] = []
    for cl in split_clusters:
        if len(cl) < 3:
            continue
        y_top = cl[0]
        y_bot = cl[-1]
        sh_px = y_bot - y_top
        if sh_px < 1.8 * d_line or sh_px > 8.5 * d_line:
            continue
        band = h_mask[max(0, y_top - 2) : min(h, y_bot + 3), :]
        col_has = band.any(axis=0)
        xs = np.flatnonzero(col_has)
        if xs.size == 0:
            continue
        x_left = int(xs[0])
        x_right = int(xs[-1])
        span_frac = float(x_right - x_left + 1) / float(w)
        cov_frac = float(col_has.sum()) / float(w)
        if span_frac < 0.25 or cov_frac < 0.15:
            continue
        v_band = ink_vert[
            max(0, y_top - int(0.3 * sh_px)) : min(h, y_bot + int(0.3 * sh_px)),
            x_left:x_right,
        ]
        v_density = float(v_band.mean()) if v_band.size > 0 else 0.0
        cl_gaps = np.diff(cl)
        regular_dline = (
            len(cl) in (4, 5, 6)
            and float(np.max(np.abs(cl_gaps - d_line))) <= max(3.0, 0.45 * d_line)
        )
        raw_staves.append(
            {
                "y_top": int(y_top),
                "y_bot": int(y_bot),
                "h": int(sh_px),
                "n_lines": len(cl),
                "x_left": x_left,
                "x_right": x_right,
                "cov_frac": round(cov_frac, 4),
                "regular_dline": bool(regular_dline),
                "v_density": round(v_density, 5),
            }
        )

    if not raw_staves:
        return {
            "n_staves_total": 0,
            "visual_n_systems": 0,
            "visual_n_staves": 0,
            "all_systems_uniform": False,
            "system_staff_counts": [],
            "system_crops": [],
            "angle_deg": round(angle_deg, 2),
            "d_line": round(d_line, 2),
            "reason": "no_valid_staves",
        }

    v_dil = cv2.dilate(ink_vert.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0

    # Check left-edge brace/barline connectivity on raw_staves so tacet/rest staves are preserved
    raw_left_connected: Set[int] = set()
    for i in range(len(raw_staves) - 1):
        s0, s1 = raw_staves[i], raw_staves[i + 1]
        g_top, g_bot = s0["y_bot"], s1["y_top"]
        gap_px = max(1, g_bot - g_top)
        if gap_px > 6 * max(s0["h"], s1["h"]):
            continue
        x_ref = min(s0["x_left"], s1["x_left"])
        lx0 = max(0, x_ref - int(0.035 * w))
        lx1 = min(w, x_ref + int(0.045 * w))
        pad_in = min(3, max(0, gap_px // 8))
        by0 = min(h - 1, g_top + pad_in)
        by1 = max(by0 + 1, g_bot - pad_in)
        gb = v_dil[by0:by1, lx0:lx1].astype(np.uint8)
        if gb.size > 0:
            n_cc, _, stats, _ = cv2.connectedComponentsWithStats(gb, connectivity=8)
            for cid in range(1, n_cc):
                ch_cc = int(stats[cid, cv2.CC_STAT_HEIGHT])
                cw_cc = int(stats[cid, cv2.CC_STAT_WIDTH])
                if ch_cc >= max(3, int(0.90 * gb.shape[0])) and cw_cc <= max(28, int(0.04 * w)):
                    raw_left_connected.add(i)
                    raw_left_connected.add(i + 1)
                    break

    for idx, s in enumerate(raw_staves):
        s["left_connected"] = bool(idx in raw_left_connected)

    # Blank-staff filter using p75 of v_density + preserve tacet/rest staves
    p75_vd = float(np.percentile([s["v_density"] for s in raw_staves], 75))
    staves = [
        s
        for idx, s in enumerate(raw_staves)
        if (s["v_density"] >= max(0.012, 0.22 * p75_vd))
        or (idx in raw_left_connected)
        or (s["regular_dline"] and s["cov_frac"] >= 0.45)
    ]
    if not staves:
        staves = raw_staves

    med_sh = float(np.median([s["h"] for s in staves]))
    n_st = len(staves)

    staff_bar_xs: List[List[int]] = []
    for s in staves:
        sb = v_dil[s["y_top"] : s["y_bot"] + 1, :]
        col_fill = sb.mean(axis=0) if sb.shape[0] > 0 else np.zeros(w)
        xl_b = s["x_left"] + int(0.08 * (s["x_right"] - s["x_left"]))
        xr_b = s["x_right"] - int(0.01 * (s["x_right"] - s["x_left"]))
        xs_raw = np.flatnonzero(col_fill[xl_b:xr_b] >= 0.72) + xl_b
        bxs: List[int] = []
        if xs_raw.size > 0:
            grp = [int(xs_raw[0])]
            for x in xs_raw[1:]:
                if int(x) - grp[-1] <= 18:
                    grp.append(int(x))
                else:
                    if len(grp) <= 16:
                        bxs.append(int(round(sum(grp) / len(grp))))
                    grp = [int(x)]
            if len(grp) <= 16:
                bxs.append(int(round(sum(grp) / len(grp))))
        staff_bar_xs.append(bxs)

    keep_idx = [
        i
        for i, s in enumerate(staves)
        if (len(staff_bar_xs[i]) >= 1)
        or (s["v_density"] >= max(0.025, 0.45 * p75_vd))
        or s.get("left_connected", False)
    ]
    if keep_idx and len(keep_idx) < len(staves):
        staves = [staves[i] for i in keep_idx]
        staff_bar_xs = [staff_bar_xs[i] for i in keep_idx]
        n_st = len(staves)
        med_sh = float(np.median([s["h"] for s in staves]))

    if force_single_staff:
        systems = [[i] for i in range(n_st)]
    else:
        connected: List[bool] = []
        inter_gaps: List[int] = []
        for i in range(n_st - 1):
            s0 = staves[i]
            s1 = staves[i + 1]
            g_top = s0["y_bot"]
            g_bot = s1["y_top"]
            gap_px = max(1, g_bot - g_top)
            inter_gaps.append(gap_px)

            xl = max(0, min(s0["x_left"], s1["x_left"]) - int(0.035 * w))
            xr = min(w, max(s0["x_right"], s1["x_right"]) + int(0.015 * w))
            x_left_ref = min(s0["x_left"], s1["x_left"]) - xl

            pad_in = min(3, max(0, gap_px // 8))
            band_y0 = min(h - 1, g_top + pad_in)
            band_y1 = max(band_y0 + 1, g_bot - pad_in)
            gap_band = v_dil[band_y0:band_y1, xl:xr].astype(np.uint8)
            bh = gap_band.shape[0]
            n_cc, _, stats, centroids = cv2.connectedComponentsWithStats(gap_band, connectivity=8)

            left_zone_lo = max(0, x_left_ref - int(0.03 * w))
            left_zone_hi = x_left_ref + int(0.045 * w)
            has_left_conn = False
            n_gap_bars = 0
            bar_xs_seen: List[int] = []

            for cid in range(1, n_cc):
                ch_cc = int(stats[cid, cv2.CC_STAT_HEIGHT])
                cw_cc = int(stats[cid, cv2.CC_STAT_WIDTH])
                cx = float(centroids[cid][0])
                if ch_cc < max(3, int(0.90 * bh)):
                    continue
                if left_zone_lo <= cx <= left_zone_hi:
                    if cw_cc <= max(28, int(0.04 * w)):
                        has_left_conn = True
                elif cx > left_zone_hi:
                    if cw_cc > max(18, int(0.02 * w)):
                        continue
                    gx = xl + int(round(cx))
                    staff_h_pair = 0.5 * (s0["h"] + s1["h"])
                    x_rad = max(6, int(0.10 * (gap_px + staff_h_pair)))
                    x_lo = max(0, gx - x_rad)
                    x_hi = min(w, gx + x_rad + 1)
                    in_s0 = float(v_dil[s0["y_top"] : s0["y_bot"] + 1, x_lo:x_hi].max(axis=1).mean())
                    in_s1 = float(v_dil[s1["y_top"] : s1["y_bot"] + 1, x_lo:x_hi].max(axis=1).mean())
                    if in_s0 >= 0.65 and in_s1 >= 0.65:
                        if _dedup_bar_xs_unsorted(gx, bar_xs_seen, min_sep=20):
                            bar_xs_seen.append(gx)
                            n_gap_bars += 1

            b0 = staff_bar_xs[i]
            b1 = staff_bar_xs[i + 1]
            n_aligned_bars = 0
            if len(b0) >= 2 and len(b1) >= 2:
                for x0 in b0:
                    if any(abs(x0 - x1) <= 12 for x1 in b1):
                        n_aligned_bars += 1
            align_frac = (
                (2.0 * n_aligned_bars) / float(len(b0) + len(b1))
                if (len(b0) + len(b1)) > 0
                else 0.0
            )

            is_conn = (
                (n_gap_bars >= 2)
                or (has_left_conn and n_gap_bars >= 1)
                or (has_left_conn and n_aligned_bars >= 4 and align_frac >= 0.45)
            )
            connected.append(bool(is_conn))

        # Vertical gap-periodicity clustering alongside barline continuity
        if n_st >= 4 and len(inter_gaps) == n_st - 1:
            best_k: Optional[int] = None
            best_ratio = 0.0
            for k in (2, 3, 4, 5, 6):
                if n_st % k != 0 or (n_st // k) < 2:
                    continue
                g_in = [inter_gaps[i] for i in range(n_st - 1) if (i + 1) % k != 0]
                g_out = [inter_gaps[i] for i in range(n_st - 1) if (i + 1) % k == 0]
                if not g_in or not g_out:
                    continue
                max_in = max(g_in)
                min_out = min(g_out)
                ratio = float(min_out) / float(max(1, max_in))
                # Check that barline connectivity does not contradict inter-system boundaries
                out_conn = sum(1 for i in range(n_st - 1) if (i + 1) % k == 0 and connected[i])
                if ratio >= 1.35 and out_conn == 0 and ratio > best_ratio:
                    best_ratio = ratio
                    best_k = k
            if best_k is not None:
                for i in range(n_st - 1):
                    if (i + 1) % best_k != 0:
                        connected[i] = True

        systems = [[0]]
        for i, is_conn in enumerate(connected):
            if is_conn:
                systems[-1].append(i + 1)
            else:
                systems.append([i + 1])

        # Voice + Piano Lied [1, 2] -> [3] merge with inter-staff vs inter-system gap ratio check
        raw_counts = [len(s) for s in systems]
        n_12_pairs = sum(
            1 for i in range(len(raw_counts) - 1) if raw_counts[i] == 1 and raw_counts[i + 1] == 2
        )
        n_threes = sum(1 for c in raw_counts if c == 3)
        can_try_12_merge = (
            n_12_pairs >= 2
            or (n_12_pairs >= 1 and n_threes >= 1)
            or (len(raw_counts) == 2 and raw_counts == [1, 2])
        )
        if can_try_12_merge:
            merged_sys: List[List[int]] = []
            i = 0
            while i < len(systems):
                if i + 1 < len(systems) and len(systems[i]) == 1 and len(systems[i + 1]) == 2:
                    s_voice = staves[systems[i][0]]
                    s_p0 = staves[systems[i + 1][0]]
                    s_p1 = staves[systems[i + 1][1]]
                    g_12 = max(1, s_p0["y_top"] - s_voice["y_bot"])
                    g_piano = max(1, s_p1["y_top"] - s_p0["y_bot"])

                    # Horizontal overlap check
                    ov_lo = max(s_voice["x_left"], s_p0["x_left"])
                    ov_hi = min(s_voice["x_right"], s_p0["x_right"])
                    min_span = max(1, min(s_voice["x_right"] - s_voice["x_left"], s_p0["x_right"] - s_p0["x_left"]))
                    x_overlap = float(max(0, ov_hi - ov_lo)) / float(min_span)

                    # Inter-system gap comparison when neighboring systems exist
                    neighbor_sys_gaps: List[int] = []
                    if i > 0:
                        prev_bot = staves[systems[i - 1][-1]]["y_bot"]
                        neighbor_sys_gaps.append(max(1, s_voice["y_top"] - prev_bot))
                    if i + 2 < len(systems):
                        next_top = staves[systems[i + 2][0]]["y_top"]
                        neighbor_sys_gaps.append(max(1, next_top - s_p1["y_bot"]))

                    if neighbor_sys_gaps:
                        max_gap_ratio = 1.15 if (n_12_pairs >= 2 and x_overlap >= 0.75) else 0.85
                        gap_ok = g_12 <= max_gap_ratio * min(neighbor_sys_gaps)
                    else:
                        # Single-system [1, 2] page
                        gap_ok = g_12 <= 2.2 * max(float(g_piano), med_sh)

                    if gap_ok and x_overlap >= 0.70:
                        merged_sys.append(systems[i] + systems[i + 1])
                        i += 2
                        continue
                merged_sys.append(systems[i])
                i += 1
            systems = merged_sys

    sys_counts = [len(s) for s in systems]
    all_systems_uniform = (len(systems) >= 1) if force_single_staff else (len(set(sys_counts)) == 1)
    cnt = Counter(sys_counts)
    modal_staves = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[0][0]

    # NEVER silently drop inked systems: keep all systems in system_crops and expose
    # all_systems_uniform so Stage 2 rejects any page with mismatched system counts!
    pad_y = max(4, int(round(0.45 * med_sh)))
    system_crops: List[Dict[str, Any]] = []
    for sys_staff_idxs in systems:
        s_list = [staves[idx] for idx in sys_staff_idxs]
        y_top = max(0, s_list[0]["y_top"] - pad_y)
        y_bot = min(h, s_list[-1]["y_bot"] + pad_y)
        x_left = max(0, min(s["x_left"] for s in s_list))
        x_right = min(w, max(s["x_right"] for s in s_list))
        system_crops.append(
            {
                "y_top": int(y_top),
                "y_bot": int(y_bot),
                "x_left": int(x_left),
                "x_right": int(x_right),
                "staves": s_list,
            }
        )
    _clamp_system_crops_to_midpoints(system_crops, h)

    return {
        "n_staves_total": n_st,
        "n_staves_raw": len(raw_staves),
        "visual_n_systems": int(len(systems)),
        "visual_n_staves": int(modal_staves),
        "all_systems_uniform": bool(all_systems_uniform),
        "system_staff_counts": sys_counts,
        "system_crops": system_crops,
        "angle_deg": round(angle_deg, 2),
        "d_line": round(d_line, 2),
        "med_staff_h": round(med_sh, 1),
        "deskewed": deskewed,
        "ink_vert": ink_vert,
        "reason": "ok",
    }


# ---------------------------------------------------------------------------
# 2. MusicXML Indexing, Mid-Measure Fix & 1-Staff Partbook Extraction
# ---------------------------------------------------------------------------

def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


@dataclass
class MeasureAttrState:
    """Carried-forward MusicXML attributes active at the start of a measure."""
    divisions: Optional[ET.Element] = None
    keys_by_num: Dict[str, ET.Element] = field(default_factory=dict)
    time: Optional[ET.Element] = None
    staves: Optional[ET.Element] = None
    clefs_by_num: Dict[str, ET.Element] = field(default_factory=dict)
    transpose: Optional[ET.Element] = None


@dataclass
class IndexedPart:
    part_id: str
    measures: List[ET.Element]
    prefix_attrs: List[MeasureAttrState]
    n_staves: int


@dataclass
class IndexedMusicXML:
    """Pre-indexed partwise MusicXML avoiding O(M) deep-copy per candidate."""
    root_tag: str
    root_attrib: Dict[str, str]
    header_children: List[ET.Element]
    parts: List[IndexedPart]
    n_measures: int
    total_staves: int
    part_measure_counts: List[int]
    part_measure_mismatch: bool


def index_musicxml_root(root: ET.Element) -> IndexedMusicXML:
    """Build IndexedMusicXML with O(M) prefix attribute tables in one pass."""
    header_children: List[ET.Element] = []
    parts: List[IndexedPart] = []

    for child in root:
        tag = _strip_ns(child.tag)
        if tag in ("work", "movement-title", "movement-number", "credit"):
            continue
        if tag != "part":
            header_children.append(copy.deepcopy(child))
            continue

        part_id = child.attrib.get("id", f"P{len(parts) + 1}")
        measures = [m for m in child if _strip_ns(m.tag) == "measure"]
        if not measures:
            continue

        # Precompute prefix attribute state active BEFORE each measure m
        prefix_attrs: List[MeasureAttrState] = []
        cur_div: Optional[ET.Element] = None
        cur_keys: Dict[str, ET.Element] = {}
        cur_time: Optional[ET.Element] = None
        cur_staves: Optional[ET.Element] = None
        cur_clefs: Dict[str, ET.Element] = {}
        cur_transpose: Optional[ET.Element] = None
        max_staves_in_part = 1

        for m_el in measures:
            prefix_attrs.append(
                MeasureAttrState(
                    divisions=cur_div,
                    keys_by_num=dict(cur_keys),
                    time=cur_time,
                    staves=cur_staves,
                    clefs_by_num=dict(cur_clefs),
                    transpose=cur_transpose,
                )
            )
            for el in m_el:
                if _strip_ns(el.tag) == "attributes":
                    for a in el:
                        atag = _strip_ns(a.tag)
                        if atag == "divisions":
                            cur_div = copy.deepcopy(a)
                        elif atag == "key":
                            knum = a.attrib.get("number", "")
                            cur_keys[knum] = copy.deepcopy(a)
                        elif atag == "time":
                            cur_time = copy.deepcopy(a)
                        elif atag == "staves":
                            cur_staves = copy.deepcopy(a)
                            try:
                                sv = int((a.text or "1").strip())
                                max_staves_in_part = max(max_staves_in_part, sv)
                            except ValueError:
                                pass
                        elif atag == "clef":
                            cnum = a.attrib.get("number", "1")
                            cur_clefs[cnum] = copy.deepcopy(a)
                        elif atag == "transpose":
                            cur_transpose = copy.deepcopy(a)

        parts.append(
            IndexedPart(
                part_id=part_id,
                measures=measures,
                prefix_attrs=prefix_attrs,
                n_staves=max_staves_in_part,
            )
        )

    counts = [len(p.measures) for p in parts]
    n_measures = min(counts) if counts else 0
    mismatch = len(set(counts)) > 1
    total_staves = sum(p.n_staves for p in parts)

    return IndexedMusicXML(
        root_tag=root.tag,
        root_attrib=dict(root.attrib),
        header_children=header_children,
        parts=parts,
        n_measures=n_measures,
        total_staves=total_staves,
        part_measure_counts=counts,
        part_measure_mismatch=mismatch,
    )


def load_indexed_musicxml(xml_path: Path) -> IndexedMusicXML:
    tree = ET.parse(str(xml_path))
    return index_musicxml_root(tree.getroot())


class IndexedMusicXMLCache:
    """Bounded LRU cache (<= max_size) for parsed IndexedMusicXML works."""

    def __init__(self, max_size: int = 32) -> None:
        self.max_size = max(1, int(max_size))
        self._cache: "OrderedDict[str, IndexedMusicXML]" = OrderedDict()

    def get(self, xml_path: Path) -> IndexedMusicXML:
        key = str(xml_path.resolve())
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        indexed = load_indexed_musicxml(xml_path)
        self._cache[key] = indexed
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return indexed

    def __len__(self) -> int:
        return len(self._cache)


def _find_opening_attributes(m_el: ET.Element) -> Optional[ET.Element]:
    """Return existing <attributes> ONLY if it occurs before any <note>, <backup>, or <forward>."""
    for el in m_el:
        tag = _strip_ns(el.tag)
        if tag in ("note", "backup", "forward"):
            return None
        if tag == "attributes":
            return el
    return None


def _insert_carried_attributes_dtd_order(
    first_attr: ET.Element, state: MeasureAttrState, hide_carried_time: bool
) -> None:
    """Merge carried-forward attributes into first_attr in strict MusicXML 3.1 DTD order."""
    dtd_order = (
        "footnote",
        "level",
        "divisions",
        "key",
        "time",
        "staves",
        "part-symbol",
        "instruments",
        "clef",
        "staff-details",
        "transpose",
        "directive",
        "measure-style",
        "other",
    )
    existing_children = list(first_attr)
    existing_tags = {_strip_ns(a.tag) for a in existing_children}
    existing_keys = {
        a.attrib.get("number", "") for a in existing_children if _strip_ns(a.tag) == "key"
    }
    existing_clefs = {
        a.attrib.get("number", "1") for a in existing_children if _strip_ns(a.tag) == "clef"
    }

    buckets: Dict[str, List[ET.Element]] = {k: [] for k in dtd_order}

    # Carried-forward items first if missing
    if "divisions" not in existing_tags and state.divisions is not None:
        buckets["divisions"].append(copy.deepcopy(state.divisions))
    for knum in sorted(state.keys_by_num.keys()):
        if knum not in existing_keys:
            buckets["key"].append(copy.deepcopy(state.keys_by_num[knum]))
    if "time" not in existing_tags and state.time is not None:
        t_el = copy.deepcopy(state.time)
        if hide_carried_time:
            t_el.attrib["print-object"] = "no"
        buckets["time"].append(t_el)
    if "staves" not in existing_tags and state.staves is not None:
        buckets["staves"].append(copy.deepcopy(state.staves))
    for cnum in sorted(state.clefs_by_num.keys()):
        if cnum not in existing_clefs:
            buckets["clef"].append(copy.deepcopy(state.clefs_by_num[cnum]))
    if "transpose" not in existing_tags and state.transpose is not None:
        buckets["transpose"].append(copy.deepcopy(state.transpose))

    # Preserve all existing children in their respective DTD buckets
    for a in existing_children:
        atag = _strip_ns(a.tag)
        if atag in buckets and atag != "other":
            buckets[atag].append(a)
        else:
            buckets["other"].append(a)

    # Rebuild first_attr in MusicXML 3.1 DTD order
    for a in list(first_attr):
        first_attr.remove(a)
    for key in dtd_order:
        for el in buckets[key]:
            first_attr.append(el)


def slice_musicxml_measures(
    indexed: IndexedMusicXML, m_start: int, m_end: int
) -> str:
    """Slice IndexedMusicXML to 0-based measures [m_start, m_end) in O(m_end - m_start)."""
    root = ET.Element(indexed.root_tag, dict(indexed.root_attrib))
    for hc in indexed.header_children:
        root.append(copy.deepcopy(hc))

    for part in indexed.parts:
        p_el = ET.SubElement(root, "part", {"id": part.part_id})
        hi = min(m_end, len(part.measures))
        lo = max(0, min(m_start, hi))
        if lo >= hi:
            continue

        state = part.prefix_attrs[lo]
        for idx, m_orig in enumerate(part.measures[lo:hi]):
            m_el = copy.deepcopy(m_orig)
            # Strip explicit system/page breaks so Verovio renders one unbroken system
            for pr in list(m_el):
                if _strip_ns(pr.tag) == "print":
                    pr.attrib.pop("new-system", None)
                    pr.attrib.pop("new-page", None)

            if idx == 0 and lo > 0:
                first_attr = _find_opening_attributes(m_el)
                if first_attr is None:
                    first_attr = ET.Element("attributes")
                    # Insert after leading <print> if present, else index 0
                    ins_idx = 0
                    while ins_idx < len(m_el) and _strip_ns(m_el[ins_idx].tag) == "print":
                        ins_idx += 1
                    m_el.insert(ins_idx, first_attr)
                _insert_carried_attributes_dtd_order(
                    first_attr, state, hide_carried_time=True
                )

            p_el.append(m_el)

    return ET.tostring(root, encoding="unicode")


def extract_single_staff_musicxml(
    indexed: IndexedMusicXML, staff_idx: int
) -> Optional[IndexedMusicXML]:
    """Extract a single 0-based global staff from a multi-staff/multi-part IndexedMusicXML."""
    if staff_idx < 0 or staff_idx >= indexed.total_staves:
        return None

    # Locate target (part, local_staff_1based)
    cur = 0
    target_part: Optional[IndexedPart] = None
    local_staff = 1
    for part in indexed.parts:
        if cur <= staff_idx < cur + part.n_staves:
            target_part = part
            local_staff = (staff_idx - cur) + 1
            break
        cur += part.n_staves

    if target_part is None:
        return None

    root = ET.Element(indexed.root_tag, dict(indexed.root_attrib))
    for hc in indexed.header_children:
        hc_copy = copy.deepcopy(hc)
        if _strip_ns(hc_copy.tag) == "part-list":
            for child in list(hc_copy):
                ctag = _strip_ns(child.tag)
                if ctag == "score-part" and child.attrib.get("id") != target_part.part_id:
                    hc_copy.remove(child)
                elif ctag == "part-group":
                    hc_copy.remove(child)
        root.append(hc_copy)

    p_el = ET.SubElement(root, "part", {"id": target_part.part_id})
    target_s_str = str(local_staff)
    multi_staff_part = target_part.n_staves > 1

    def _elem_duration(elem: ET.Element) -> int:
        for sub in elem:
            if _strip_ns(sub.tag) == "duration":
                try:
                    return max(0, int((sub.text or "0").strip()))
                except ValueError:
                    return 0
        return 0

    def _sync_emitted_time(new_m: ET.Element, emitted_t: int, target_t: int) -> int:
        delta = target_t - emitted_t
        if delta > 0:
            fwd = ET.Element("forward")
            ET.SubElement(fwd, "duration").text = str(delta)
            new_m.append(fwd)
        elif delta < 0:
            bkp = ET.Element("backup")
            ET.SubElement(bkp, "duration").text = str(-delta)
            new_m.append(bkp)
        return target_t

    for m_orig in target_part.measures:
        if not multi_staff_part:
            p_el.append(copy.deepcopy(m_orig))
            continue

        new_m = ET.Element(m_orig.tag, dict(m_orig.attrib))
        cur_t = 0
        emitted_t = 0
        prev_note_t = 0
        chord_has_kept_root = False

        for el_orig in m_orig:
            tag = _strip_ns(el_orig.tag)
            if tag == "backup":
                cur_t = max(0, cur_t - _elem_duration(el_orig))
                chord_has_kept_root = False
                continue
            if tag == "forward":
                cur_t += _elem_duration(el_orig)
                chord_has_kept_root = False
                continue

            el = copy.deepcopy(el_orig)
            if tag == "attributes":
                for a in list(el):
                    atag = _strip_ns(a.tag)
                    if atag == "staves":
                        el.remove(a)
                    elif atag == "clef":
                        num = a.attrib.get("number", "1")
                        if num != target_s_str:
                            el.remove(a)
                        else:
                            a.attrib["number"] = "1"
                    elif atag in ("key", "staff-details", "time"):
                        num = a.attrib.get("number", "")
                        if num and num != target_s_str:
                            el.remove(a)
                        elif num == target_s_str:
                            a.attrib["number"] = "1"
                if len(el) > 0:
                    emitted_t = _sync_emitted_time(new_m, emitted_t, cur_t)
                    new_m.append(el)
                continue

            if tag == "note":
                st_el = None
                chord_el = None
                is_grace = False
                for sub in el:
                    stag = _strip_ns(sub.tag)
                    if stag == "staff":
                        st_el = sub
                    elif stag == "chord":
                        chord_el = sub
                    elif stag == "grace":
                        is_grace = True
                is_chord = chord_el is not None
                dur = 0 if is_grace else _elem_duration(el)
                if not is_chord:
                    note_start_t = cur_t
                    if not is_grace:
                        prev_note_t = cur_t
                        cur_t += dur
                    chord_has_kept_root = False
                else:
                    note_start_t = prev_note_t

                note_staff = (st_el.text or "1").strip() if st_el is not None else "1"
                if note_staff != target_s_str:
                    continue

                if st_el is not None:
                    st_el.text = "1"
                if is_chord and not chord_has_kept_root:
                    # First surviving note of a cross-staff chord: strip stray <chord/>
                    el.remove(chord_el)
                    is_chord = False

                if not is_chord:
                    emitted_t = _sync_emitted_time(new_m, emitted_t, note_start_t)
                    new_m.append(el)
                    if not is_grace:
                        emitted_t = note_start_t + dur
                    chord_has_kept_root = True
                else:
                    new_m.append(el)
                continue

            if tag in ("direction", "harmony"):
                st_el = None
                for sub in el:
                    if _strip_ns(sub.tag) == "staff":
                        st_el = sub
                        break
                if st_el is not None and (st_el.text or "1").strip() != target_s_str:
                    continue
                if st_el is not None:
                    st_el.text = "1"
                emitted_t = _sync_emitted_time(new_m, emitted_t, cur_t)
                new_m.append(el)
                continue

            new_m.append(el)

        p_el.append(new_m)

    return index_musicxml_root(root)


# ---------------------------------------------------------------------------
# 3. Barline Detector (Per-System Slice, Two-Stage Cluster, ±10° Slant, bar_conf)
# ---------------------------------------------------------------------------

def cluster_barline_cols(cols: np.ndarray, sys_w: int) -> Tuple[List[int], int]:
    """Two-stage barline clustering:
    1) Cluster columns within tight barline thickness <= max(10, int(0.012 * sys_w)).
    2) Discard wide gutters with span > max(18, int(0.025 * sys_w)), take centroids.
    3) Enforce min_bar_sep = max(18, int(round(0.032 * sys_w))) between centroids.
    Returns (accepted_centroids, total_raw_clusters).
    """
    if cols.size == 0:
        return [], 0
    cols_sorted = np.sort(np.unique(cols.astype(np.int64)))
    tight_gap = max(10, int(0.012 * sys_w))
    max_span = max(18, int(0.025 * sys_w))
    min_bar_sep = max(18, int(round(0.032 * sys_w)))

    # Stage 1: tight clustering
    raw_groups: List[List[int]] = [[int(cols_sorted[0])]]
    for x in cols_sorted[1:]:
        xi = int(x)
        if xi - raw_groups[-1][-1] <= tight_gap:
            raw_groups[-1].append(xi)
        else:
            raw_groups.append([xi])

    # Stage 2: discard wide gutters & take centroids
    centroids: List[int] = []
    for grp in raw_groups:
        if (grp[-1] - grp[0]) <= max_span:
            centroids.append(int(round(sum(grp) / len(grp))))

    if not centroids:
        return [], len(raw_groups)

    # Stage 3: enforce min_bar_sep between centroids
    merged: List[int] = [centroids[0]]
    for c in centroids[1:]:
        if c - merged[-1] < min_bar_sep:
            merged[-1] = int(round(0.5 * (merged[-1] + c)))
        else:
            merged.append(c)

    return merged, len(raw_groups)


def _max_pool_1d(arr: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or arr.size == 0:
        return arr
    k = np.ones((1, 2 * radius + 1), dtype=np.uint8)
    return cv2.dilate(arr.reshape(1, -1).astype(np.float32), k).ravel()


def _sheared_gap_max_fill(gap_mask: np.ndarray, angles_deg: Sequence[float] = (-10.0, -5.0, 0.0, 5.0, 10.0)) -> np.ndarray:
    """Compute column-wise vertical fill maximized across ±10° slant shears."""
    gh, gw = gap_mask.shape
    if gh == 0 or gw == 0:
        return np.zeros(gw, dtype=np.float32)
    best_fill = gap_mask.mean(axis=0).astype(np.float32)
    ys = np.arange(gh, dtype=np.float32) - 0.5 * (gh - 1)
    for ang in angles_deg:
        if abs(ang) < 1e-6:
            continue
        tan_a = float(np.tan(np.deg2rad(ang)))
        shifts = np.round(ys * tan_a).astype(np.int32)
        acc = np.zeros(gw, dtype=np.float32)
        for r in range(gh):
            sh = int(shifts[r])
            if sh == 0:
                acc += gap_mask[r]
            elif sh > 0:
                if sh < gw:
                    acc[sh:] += gap_mask[r, : gw - sh]
            else:
                ash = -sh
                if ash < gw:
                    acc[: gw - ash] += gap_mask[r, ash:]
        fill = acc / float(gh)
        best_fill = np.maximum(best_fill, fill)
    return best_fill


def estimate_system_measures_from_barlines(
    ink_vert: np.ndarray,
    sys_crop: Dict[str, Any],
) -> Tuple[int, List[int], float]:
    """Slice ink_vert per system, detect slant-tolerant barlines, return (m_est, bar_xs, bar_conf)."""
    h, w = ink_vert.shape
    staves = sys_crop["staves"]
    n_st = len(staves)
    xl = int(sys_crop["x_left"])
    xr = int(sys_crop["x_right"])
    yt = int(sys_crop["y_top"])
    yb = int(sys_crop["y_bot"])
    sys_w = max(1, xr - xl)

    x_search_lo = xl + int(round(0.08 * sys_w))
    x_search_hi = xr - int(round(0.03 * sys_w))
    if x_search_hi <= x_search_lo + 10 or yb <= yt + 4:
        return 4, [], 0.0

    # Slice ink_vert strictly to this system before dilation (no full-page dilation!)
    sys_iv = ink_vert[yt:yb, x_search_lo:x_search_hi].astype(np.uint8)
    dil_k = np.ones((3, 3), dtype=np.uint8)
    v_dil = cv2.dilate(sys_iv, dil_k, iterations=1) > 0
    sh_crop, span_w = v_dil.shape

    def _rel_y(y_abs: int) -> int:
        return max(0, min(sh_crop, int(y_abs) - yt))

    if n_st >= 2:
        gap_votes = np.zeros(span_w, dtype=np.float32)
        n_gaps = 0
        for i in range(n_st - 1):
            g_top = staves[i]["y_bot"]
            g_bot = staves[i + 1]["y_top"]
            gap_px = g_bot - g_top
            if gap_px >= 4:
                pad_in = min(3, max(0, gap_px // 8))
                b0 = _rel_y(g_top + pad_in)
                b1 = max(b0 + 1, _rel_y(g_bot - pad_in))
                s0_y0, s0_y1 = _rel_y(staves[i]["y_top"]), max(_rel_y(staves[i]["y_top"]) + 1, _rel_y(staves[i]["y_bot"] + 1))
                s1_y0, s1_y1 = _rel_y(staves[i + 1]["y_top"]), max(_rel_y(staves[i + 1]["y_top"]) + 1, _rel_y(staves[i + 1]["y_bot"] + 1))

                gap_fill = _sheared_gap_max_fill(v_dil[b0:b1, :])
                staff_h = 0.5 * ((staves[i]["y_bot"] - staves[i]["y_top"]) + (staves[i + 1]["y_bot"] - staves[i + 1]["y_top"]))
                slant_r = max(4, int(round(0.10 * (gap_px + staff_h))))
                s0_fill = _max_pool_1d(v_dil[s0_y0:s0_y1, :].mean(axis=0), slant_r)
                s1_fill = _max_pool_1d(v_dil[s1_y0:s1_y1, :].mean(axis=0), slant_r)

                is_gap_bar = (gap_fill >= 0.80) & (s0_fill >= 0.68) & (s1_fill >= 0.68)
                gap_votes += is_gap_bar.astype(np.float32)
                n_gaps += 1

        gap_bars: List[int] = []
        raw_n = 0
        if n_gaps > 0:
            hit_gap = np.flatnonzero((gap_votes / float(n_gaps)) >= 0.49) + x_search_lo
            gap_bars, raw_n = cluster_barline_cols(hit_gap, sys_w)

        if gap_bars:
            bar_xs = gap_bars
            sharp_frac = float(len(bar_xs)) / float(max(1, raw_n))
            bar_conf = float(np.clip(sharp_frac, 0.0, 1.0))
        else:
            st_votes = np.zeros(span_w, dtype=np.float32)
            for s in staves:
                sy0 = _rel_y(s["y_top"])
                sy1 = max(sy0 + 1, _rel_y(s["y_bot"] + 1))
                sh_s = max(4, s["y_bot"] - s["y_top"])
                sb_fill = _max_pool_1d(v_dil[sy0:sy1, :].mean(axis=0), max(3, int(round(0.10 * sh_s))))
                st_votes += (sb_fill >= 0.72).astype(np.float32)
            hit_aln = np.flatnonzero((st_votes / float(n_st)) >= 0.85) + x_search_lo
            bar_xs, raw_n = cluster_barline_cols(hit_aln, sys_w)
            sharp_frac = float(len(bar_xs)) / float(max(1, raw_n)) if raw_n > 0 else 0.0
            bar_conf = float(np.clip(0.75 * sharp_frac, 0.0, 1.0)) if bar_xs else 0.0
    else:
        # 1-staff system: reject tall note stems with adjacent notehead blobs (±4 px)
        # and set bar_conf = 0.0 so DP uses the wide unpenalized [1..12] window.
        s = staves[0]
        sy0 = _rel_y(s["y_top"])
        sy1 = max(sy0 + 1, _rel_y(s["y_bot"] + 1))
        sb = v_dil[sy0:sy1, :]
        sb_fill = sb.mean(axis=0)
        cand_cols = np.flatnonzero(sb_fill >= 0.85)
        filtered_cols: List[int] = []
        for c in cand_cols:
            ci = int(c)
            # Check ±4 px neighborhood around column for notehead-like blobs
            n_lo = max(0, ci - 4)
            n_hi = min(span_w, ci + 5)
            neigh = sb[:, n_lo:n_hi]
            # Exclude the stem column itself (±1 px)
            left_side = sb[:, n_lo : max(n_lo, ci - 1)]
            right_side = sb[:, min(span_w, ci + 2) : n_hi]
            side_density = 0.0
            if left_side.size > 0:
                side_density = max(side_density, float(left_side.mean()))
            if right_side.size > 0:
                side_density = max(side_density, float(right_side.mean()))
            if side_density < 0.22 and neigh.size > 0:
                filtered_cols.append(ci + x_search_lo)
        bar_xs, _ = cluster_barline_cols(np.array(filtered_cols, dtype=np.int64), sys_w)
        bar_conf = 0.0

    if len(bar_xs) < 2:
        bar_conf = 0.0
    else:
        bounds = [xl] + sorted(bar_xs) + [xr]
        max_gap = max(bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1))
        if max_gap > 0.45 * sys_w:
            bar_conf = 0.0

    m_est = max(1, min(12, len(bar_xs) + 1))
    return m_est, bar_xs, round(bar_conf, 4)


# ---------------------------------------------------------------------------
# 4. Verovio Single-Render SVG Measure Bank & Manuscript Ink Extraction
# ---------------------------------------------------------------------------

def create_verovio_toolkit() -> Any:
    """Create and configure a Verovio toolkit ONCE with fixed single-system options."""
    import verovio

    tk = verovio.toolkit()
    tk.setOptions(
        {
            "breaks": "none",
            "adjustPageHeight": True,
            "adjustPageWidth": True,
            "pageWidth": 2800,
            "pageHeight": 4000,
            "scale": 35,
            "header": "none",
            "footer": "none",
            "pageMarginTop": 10,
            "pageMarginBottom": 10,
            "pageMarginLeft": 10,
            "pageMarginRight": 10,
            "svgViewBox": True,
        }
    )
    return tk


def _parse_and_strip_verovio_svg(
    svg_str: str, n_staves: int
) -> Optional[Tuple[str, float, float, float, float, List[Tuple[float, float]], List[Tuple[float, float]]]]:
    """Parse Verovio SVG for measure x-bounds and staff y-bounds, stripping horizontal staff lines.

    Returns (stripped_svg, w_def, h_def, tx, ty, measure_x_bounds_svg, staff_y_bounds_svg).
    Rejects (returns None) if detected staff count != n_staves.
    """
    try:
        root = ET.fromstring(svg_str)
    except Exception:
        return None

    vb_x0 = 0.0
    vb_y0 = 0.0
    w_def = 0.0
    h_def = 0.0
    tx = 0.0
    ty = 0.0
    for el in root.iter():
        tag = _strip_ns(el.tag)
        cls_tokens = el.attrib.get("class", "").split()
        if tag == "svg" and "definition-scale" in cls_tokens:
            vb = el.attrib.get("viewBox", "").split()
            if len(vb) == 4:
                vb_x0 = float(vb[0])
                vb_y0 = float(vb[1])
                w_def = float(vb[2])
                h_def = float(vb[3])
        elif tag == "g" and "page-margin" in cls_tokens:
            m_tr = _SVG_TRANSLATE_RE.search(el.attrib.get("transform", ""))
            if m_tr:
                tx = float(m_tr.group(1))
                ty = float(m_tr.group(2) or 0.0)

    if w_def <= 0.0 or h_def <= 0.0:
        vb = root.attrib.get("viewBox", "").split()
        if len(vb) == 4:
            vb_x0 = float(vb[0])
            vb_y0 = float(vb[1])
            w_def = float(vb[2])
            h_def = float(vb[3])
    if w_def <= 0.0 or h_def <= 0.0:
        return None

    measure_x_bounds: List[Tuple[float, float]] = []
    staff_ys_by_idx: Dict[int, List[float]] = {}

    for el in root.iter():
        if _strip_ns(el.tag) != "g" or "measure" not in el.attrib.get("class", "").split():
            continue
        m_xs: List[float] = []
        s_idx = 0
        for ch in el.iter():
            if _strip_ns(ch.tag) != "g" or "staff" not in ch.attrib.get("class", "").split():
                continue
            paths_to_remove: List[ET.Element] = []
            for p_el in list(ch):
                if _strip_ns(p_el.tag) != "path":
                    continue
                d_str = p_el.attrib.get("d", "").strip()
                m_hl = _SVG_HLINE_RE.match(d_str)
                if m_hl:
                    x0, y0, x1, y1 = (
                        float(m_hl.group(1)),
                        float(m_hl.group(2)),
                        float(m_hl.group(3)),
                        float(m_hl.group(4)),
                    )
                    if abs(y0 - y1) < 1e-2 and abs(x1 - x0) > 5.0:
                        m_xs.append(min(x0, x1))
                        m_xs.append(max(x0, x1))
                        staff_ys_by_idx.setdefault(s_idx, []).append(y0)
                        paths_to_remove.append(p_el)
            for p_el in paths_to_remove:
                ch.remove(p_el)
            s_idx += 1

        if m_xs:
            measure_x_bounds.append((min(m_xs), max(m_xs)))

    if len(staff_ys_by_idx) != n_staves or not measure_x_bounds:
        return None

    staff_y_bounds: List[Tuple[float, float]] = []
    for si in range(n_staves):
        ys = staff_ys_by_idx.get(si, [])
        if len(ys) < 3:
            return None
        staff_y_bounds.append((min(ys), max(ys)))

    stripped_svg = ET.tostring(root, encoding="unicode")
    return stripped_svg, w_def, h_def, tx - vb_x0, ty - vb_y0, measure_x_bounds, staff_y_bounds


@dataclass
class VerovioMeasureBank:
    """Single-render contiguous Verovio SVG measure bank supporting O(1) slice extraction."""
    m_start: int
    m_end: int
    sym_ink: np.ndarray  # (H, W) float32 in [0, 1], staff lines already stripped
    staff_bands: List[Tuple[int, int]]
    measure_px_bounds: List[Tuple[int, int]]  # per-measure (x0, x1) in sym_ink

    def get_slice_with_bars(
        self, m_lo: int, m_hi: int
    ) -> Optional[Tuple[np.ndarray, List[Tuple[int, int]], bool, List[float]]]:
        """Return (sub_ink, staff_bands, has_preamble, internal_bar_xs_px) in O(1)."""
        if m_lo < self.m_start or m_hi > self.m_end or m_hi <= m_lo:
            return None
        i0 = m_lo - self.m_start
        i1 = m_hi - self.m_start - 1
        if i0 < 0 or i1 >= len(self.measure_px_bounds):
            return None
        x0 = self.measure_px_bounds[i0][0]
        x1 = self.measure_px_bounds[i1][1]
        if x1 <= x0 + 4:
            return None
        sub = self.sym_ink[:, x0:x1]
        has_preamble = (m_lo == self.m_start)
        internal_bar_xs: List[float] = []
        for k in range(i0, i1):
            bx = 0.5 * (self.measure_px_bounds[k][1] + self.measure_px_bounds[k + 1][0]) - float(x0)
            internal_bar_xs.append(bx)
        return sub, self.staff_bands, has_preamble, internal_bar_xs

    def get_slice(
        self, m_lo: int, m_hi: int
    ) -> Optional[Tuple[np.ndarray, List[Tuple[int, int]], bool]]:
        res = self.get_slice_with_bars(m_lo, m_hi)
        if res is None:
            return None
        return res[0], res[1], res[2]


def build_verovio_measure_bank(
    tk: Any,
    indexed: IndexedMusicXML,
    m_start: int,
    m_end: int,
    n_staves: int,
) -> Optional[VerovioMeasureBank]:
    """Render contiguous window [m_start, m_end) once via Verovio + resvg_py."""
    import resvg_py

    n_win = m_end - m_start
    if n_win <= 0:
        return None
    try:
        xml_slice = slice_musicxml_measures(indexed, m_start, m_end)
        if not tk.loadData(xml_slice):
            return None
        if int(tk.getPageCount()) != 1:
            return None
        svg = tk.renderToSVG(1)
        if not svg or "<svg" not in svg:
            return None
    except Exception:
        return None

    parsed = _parse_and_strip_verovio_svg(svg, n_staves=n_staves)
    if parsed is None:
        return None
    stripped_svg, w_def, h_def, tx, ty, m_x_svg, s_y_svg = parsed
    if len(m_x_svg) != n_win:
        return None

    render_w = int(np.clip(84 * n_win, 900, 4096))
    try:
        png_bytes = resvg_py.svg_to_bytes(
            svg_string=stripped_svg, background="white", width=render_w
        )
        im = Image.open(io.BytesIO(png_bytes)).convert("L")
    except Exception:
        return None

    arr = np.asarray(im, dtype=np.uint8)
    sym_ink = (arr < 210).astype(np.float32)
    h_img, w_img = sym_ink.shape
    if sym_ink.sum() < 10.0 or h_img < 8 or w_img < 16:
        return None

    sx = float(w_img) / w_def
    sy = float(h_img) / h_def

    # Convert SVG staff y-bounds to pixel bands clamped to midpoints
    raw_bands: List[Tuple[int, int]] = []
    staff_cores: List[Tuple[int, int]] = []
    for yt_svg, yb_svg in s_y_svg:
        yt_px = int(round((yt_svg + ty) * sy))
        yb_px = int(round((yb_svg + ty) * sy))
        sh = max(4, yb_px - yt_px)
        staff_cores.append((yt_px, yb_px))
        raw_bands.append((max(0, yt_px - int(round(0.45 * sh))), min(h_img, yb_px + int(round(0.45 * sh)))))
    staff_bands = clamp_staff_bands_to_midpoints(raw_bands, staff_cores, h_img)

    # Convert SVG measure x-bounds to pixel bounds
    measure_px_bounds: List[Tuple[int, int]] = []
    for mx0_svg, mx1_svg in m_x_svg:
        x0 = max(0, min(w_img - 1, int(round((mx0_svg + tx) * sx))))
        x1 = max(x0 + 1, min(w_img, int(round((mx1_svg + tx) * sx))))
        measure_px_bounds.append((x0, x1))

    return VerovioMeasureBank(
        m_start=m_start,
        m_end=m_end,
        sym_ink=sym_ink,
        staff_bands=staff_bands,
        measure_px_bounds=measure_px_bounds,
    )


class VerovioBankManager:
    """Caches 48-measure VerovioMeasureBank windows on demand for one work."""

    def __init__(
        self,
        tk: Any,
        indexed: IndexedMusicXML,
        n_staves: int,
        window_size: int = 48,
        render_fn: Optional[RenderSliceFn] = None,
    ) -> None:
        self.tk = tk
        self.indexed = indexed
        self.n_staves = n_staves
        self.window_size = max(12, int(window_size))
        self.render_fn = render_fn
        self._banks: Dict[Tuple[int, int], Optional[VerovioMeasureBank]] = {}

    def get_slice_with_bars(
        self, m_start: int, m_end: int
    ) -> Optional[Tuple[np.ndarray, List[Tuple[int, int]], bool, Optional[List[float]]]]:
        if self.render_fn is not None:
            sl = self.render_fn(self.indexed, m_start, m_end, self.n_staves)
            if sl is None:
                return None
            return sl[0], sl[1], sl[2], None

        # Check if any cached bank already covers [m_start, m_end)
        for (b0, b1), bank in self._banks.items():
            if bank is not None and b0 <= m_start and m_end <= b1:
                sl = bank.get_slice_with_bars(m_start, m_end)
                if sl is not None:
                    return sl

        # Build a new contiguous bank starting at a clean anchor <= m_start
        anchor = (m_start // self.window_size) * self.window_size
        if m_end > anchor + self.window_size:
            anchor = m_start
        b_end = min(self.indexed.n_measures, anchor + self.window_size)
        if b_end < m_end:
            b_end = min(self.indexed.n_measures, m_end)
        key = (anchor, b_end)
        if key not in self._banks:
            self._banks[key] = build_verovio_measure_bank(
                self.tk, self.indexed, anchor, b_end, self.n_staves
            )
        bank = self._banks[key]
        if bank is not None:
            sl = bank.get_slice_with_bars(m_start, m_end)
            if sl is not None:
                return sl
        direct_key = (m_start, m_end)
        if direct_key != key:
            if direct_key not in self._banks:
                self._banks[direct_key] = build_verovio_measure_bank(
                    self.tk, self.indexed, m_start, m_end, self.n_staves
                )
            direct_bank = self._banks[direct_key]
            if direct_bank is not None:
                return direct_bank.get_slice_with_bars(m_start, m_end)
        return None

    def get_slice(
        self, m_start: int, m_end: int
    ) -> Optional[Tuple[np.ndarray, List[Tuple[int, int]], bool]]:
        res = self.get_slice_with_bars(m_start, m_end)
        if res is None:
            return None
        return res[0], res[1], res[2]


def extract_symbol_ink_from_manuscript(
    deskewed: np.ndarray,
    sys_crop: Dict[str, Any],
    *,
    return_x_offset: bool = False,
) -> Any:
    """Extract staff-line-suppressed symbol ink map and midpoint-clamped staff bands."""
    y0, y1 = int(sys_crop["y_top"]), int(sys_crop["y_bot"])
    x0, x1 = int(sys_crop["x_left"]), int(sys_crop["x_right"])
    crop = deskewed[y0:y1, x0:x1].astype(np.float32)
    ch, cw = crop.shape
    if ch < 8 or cw < 16:
        empty = np.zeros((max(1, ch), max(1, cw)), dtype=np.float32)
        return (empty, [], x0) if return_x_offset else (empty, [])

    k_bg = max(15, int(round(min(ch, cw) * 0.12)) | 1)
    k_bg = min(k_bg, 51)
    bg = cv2.GaussianBlur(crop, (k_bg, k_bg), 0)
    dark = np.clip(bg - crop, 0.0, 255.0)
    pos = dark[dark > 4.0]
    if pos.size < 20:
        empty = np.zeros((ch, cw), dtype=np.float32)
        return (empty, [], x0) if return_x_offset else (empty, [])
    thr = max(8.0, min(28.0, 0.30 * float(np.percentile(pos, 85))))
    bin_ink = (dark > thr).astype(np.uint8)

    hk = max(15, int(round(cw * 0.025)))
    horiz_lines = cv2.morphologyEx(
        bin_ink, cv2.MORPH_OPEN, np.ones((1, hk), dtype=np.uint8)
    )
    sym_ink = np.clip(bin_ink.astype(np.float32) - horiz_lines.astype(np.float32), 0.0, 1.0)
    vert_strokes = cv2.morphologyEx(
        bin_ink, cv2.MORPH_OPEN, np.ones((5, 1), dtype=np.uint8)
    ).astype(np.float32)
    sym_ink = np.maximum(sym_ink, vert_strokes)

    raw_bands: List[Tuple[int, int]] = []
    staff_cores: List[Tuple[int, int]] = []
    for s in sys_crop["staves"]:
        sh = max(4, s["y_bot"] - s["y_top"])
        cy0 = max(0, s["y_top"] - y0)
        cy1 = min(ch, s["y_bot"] - y0)
        staff_cores.append((cy0, cy1))
        sy0 = max(0, cy0 - int(round(0.45 * sh)))
        sy1 = min(ch, cy1 + int(round(0.45 * sh)))
        raw_bands.append((sy0, max(sy0 + 1, sy1)))
    staff_bands = clamp_staff_bands_to_midpoints(raw_bands, staff_cores, ch)

    trim_x0 = 0
    col_sum = sym_ink.sum(axis=0)
    nz = np.flatnonzero(col_sum > 0.5)
    if nz.size >= 16:
        trim_x0 = int(nz[0])
        sym_ink = sym_ink[:, trim_x0 : int(nz[-1]) + 1]

    if return_x_offset:
        return sym_ink, staff_bands, x0 + trim_x0
    return sym_ink, staff_bands


# ---------------------------------------------------------------------------
# 5. High-Res Onset-Sensitive Profiles, Itakura DTW, 2-Best Viterbi DP & 3-Tier Gate
# ---------------------------------------------------------------------------

N_PITCH_BANDS = 6
CHANNELS_PER_STAFF = 2 + N_PITCH_BANDS  # dens_z, onset_z, + 6 vertical pitch sub-bands
BARLINE_BONUS_WEIGHT = 0.08
BARLINE_MATCH_TOL = 0.05


def _bars_to_profile_fracs(
    bar_xs_rel: Sequence[float],
    cw: int,
    mask_preamble_frac: float,
) -> List[float]:
    """Convert barline x-positions in sym_ink space [0, cw) to preamble-masked profile fractions (0, 1)."""
    if cw < 4 or not bar_xs_rel:
        return []
    x_cut = int(round(max(0.0, min(0.25, mask_preamble_frac)) * cw))
    if cw - x_cut < 8:
        x_cut = 0
    work_w = float(max(1, cw - x_cut))
    fracs: List[float] = []
    for bx in bar_xs_rel:
        f = (float(bx) - float(x_cut)) / work_w
        if 0.02 < f < 0.98:
            fracs.append(f)
    return fracs


def compute_barline_alignment_bonus(
    ms_bar_fracs: Sequence[float],
    vr_bar_fracs: Optional[Sequence[float]],
    path: Sequence[Tuple[int, int]],
    N: int,
    *,
    tol: float = BARLINE_MATCH_TOL,
    weight: float = BARLINE_BONUS_WEIGHT,
) -> float:
    """Pure visual barline-position consistency bonus along DTW warping path P* (model-free, non-circular).

    Maps detected manuscript internal barlines through P* and checks 1-to-1 spatial alignment
    within ±tol of Verovio SVG internal barline x-positions using F1 weighting.
    Applied identically to DP candidates and all decoys.
    """
    if not ms_bar_fracs or not vr_bar_fracs or not path or N < 2:
        return 0.0
    path_i = np.array([p[0] for p in path], dtype=np.float32)
    path_j = np.array([p[1] for p in path], dtype=np.float32)
    denom_n = float(N - 1)

    mapped_ms = [
        float(np.interp(f_ms * denom_n, path_i, path_j)) / denom_n
        for f_ms in ms_bar_fracs
    ]

    # Greedy 1-to-1 closest matching within ±tol
    pairs: List[Tuple[float, int, int]] = []
    for mi, fm in enumerate(mapped_ms):
        for vi, fv in enumerate(vr_bar_fracs):
            err = abs(fm - float(fv))
            if err <= tol:
                pairs.append((err, mi, vi))
    if not pairs:
        return 0.0
    pairs.sort(key=lambda x: x[0])

    used_m: Set[int] = set()
    used_v: Set[int] = set()
    match_sum = 0.0
    for err, mi, vi in pairs:
        if mi not in used_m and vi not in used_v:
            used_m.add(mi)
            used_v.add(vi)
            # Smoothly reward exact hits (1.0 at err=0, 0.5 at err=tol)
            match_sum += 1.0 - 0.5 * (err / max(tol, 1e-6))

    bar_f1 = (2.0 * match_sum) / float(len(ms_bar_fracs) + len(vr_bar_fracs))
    return float(weight * np.clip(bar_f1, 0.0, 1.0))


def extract_1d_profiles(
    sym_ink: np.ndarray,
    staff_bands: List[Tuple[int, int]],
    *,
    n_bars: int = 4,
    target_w: Optional[int] = None,
    sigma: float = 1.4,
    mask_preamble_frac: float = 0.09,
) -> Tuple[np.ndarray, List[bool]]:
    """Compute high-res, preamble-masked, per-staff 8-channel 2D pitch-grid profiles (N, 8*S).

    Channels per staff: [dens_z, onset_z, pitch_band_0..5] where the 6 vertical pitch sub-bands
    evenly span [sy0, sy1] (upper ledger, top staff, upper-mid, lower-mid, bottom staff, lower ledger)
    and are normalized per column t with a 0.5 norm floor so tacet/rest staves stay attenuated.

    Returns (F, active_staves) where active_staves[s] indicates whether staff s is non-rest.
    No whole-system (0, ch) double-counting: every staff has equal weight.
    """
    ch, cw = sym_ink.shape
    N = int(target_w) if target_w is not None else int(np.clip(32 * max(1, n_bars), 192, 320))
    if ch < 1 or cw < 4 or not staff_bands:
        return np.zeros((N, max(1, len(staff_bands)) * CHANNELS_PER_STAFF), dtype=np.float32), [
            False for _ in staff_bands
        ]

    # Mask left preamble
    x_cut = int(round(max(0.0, min(0.25, mask_preamble_frac)) * cw))
    if cw - x_cut >= 8:
        work = sym_ink[:, x_cut:]
    else:
        work = sym_ink

    if work.shape[1] != N:
        resized = cv2.resize(work, (N, ch), interpolation=cv2.INTER_AREA)
    else:
        resized = work

    ksize = max(3, int(round(sigma * 4.0)) | 1)

    # Pre-pass to measure global density std for activity-aware std_floor
    raw_dens_list: List[np.ndarray] = []
    for sy0, sy1 in staff_bands:
        sy0_c = max(0, min(ch - 1, sy0))
        sy1_c = max(sy0_c + 1, min(ch, sy1))
        b = resized[sy0_c:sy1_c, :]
        dens = b.mean(axis=0).astype(np.float32) if b.size > 0 else np.zeros(N, dtype=np.float32)
        dens_s = cv2.GaussianBlur(dens.reshape(1, -1), (ksize, 1), sigma).ravel()
        raw_dens_list.append(dens_s)

    global_dens_std = max([float(d.std()) for d in raw_dens_list] + [0.02])
    dens_std_floor = max(0.025, 0.15 * global_dens_std)

    staff_blocks: List[np.ndarray] = []
    active_staves: List[bool] = []
    for (sy0, sy1), dens_s in zip(staff_bands, raw_dens_list):
        sy0_c = max(0, min(ch - 1, sy0))
        sy1_c = max(sy0_c + 1, min(ch, sy1))
        b = resized[sy0_c:sy1_c, :]
        bh = max(1, b.shape[0])
        is_active = bool(float(dens_s.max()) >= 0.03 and float(dens_s.std()) >= 0.012)
        active_staves.append(is_active)

        # 1) Density
        dens_z = (dens_s - float(dens_s.mean())) / (float(dens_s.std()) + dens_std_floor)

        # 2) Positive horizontal onset gradient max(0, d(dens)/dx)
        onset = np.maximum(0.0, np.diff(dens_s, prepend=dens_s[0])).astype(np.float32)
        onset_std_floor = max(0.008, 0.15 * float(onset.std()) + 0.005)
        onset_z = (onset - float(onset.mean())) / (float(onset.std()) + onset_std_floor)

        # 3..8) 2D 6-band vertical pitch-grid spanning [sy0, sy1]
        interp = cv2.INTER_AREA if bh >= N_PITCH_BANDS else cv2.INTER_LINEAR
        bands_raw = cv2.resize(
            b.astype(np.float32), (N, N_PITCH_BANDS), interpolation=interp
        )  # (6, N)
        bands_s = cv2.GaussianBlur(bands_raw, (ksize, 1), sigma)  # horizontal-only blur, (6, N)
        pitch_denom = float(bands_s.std()) + dens_std_floor
        # Center across the 6 vertical bands at each column t so empty columns stay 0
        # and opposite-pitch melodies are anti-correlated; apply 0.5 norm floor for tacet attenuation
        pitch_c = ((bands_s - bands_s.mean(axis=0, keepdims=True)) / pitch_denom).T  # (N, 6)
        pnorms = np.linalg.norm(pitch_c, axis=1, keepdims=True)
        pitch_u = (pitch_c / np.maximum(pnorms, 0.5)).astype(np.float32)  # (N, 6)

        # Combine 1D rhythm/onset + 2D 6-band vertical pitch distribution; per-staff L2 norm with 0.5 floor
        fs = np.concatenate(
            [
                (0.75 * dens_z)[:, None],
                (0.65 * onset_z)[:, None],
                (1.40 * pitch_u),
            ],
            axis=1,
        ).astype(np.float32)
        snorms = np.linalg.norm(fs, axis=1, keepdims=True)
        fs = (fs / np.maximum(snorms, 0.5)).astype(np.float32)
        staff_blocks.append(fs)

    F = np.concatenate(staff_blocks, axis=1).astype(np.float32) / np.sqrt(float(len(staff_blocks)))
    return F, active_staves


def compare_system_profiles_dtw(
    F_ms: np.ndarray,
    F_vr: np.ndarray,
    *,
    band: Optional[int] = None,
    warp_pen: float = 0.03,
    max_mean_warp_frac: float = 0.14,
) -> Tuple[float, List[Tuple[int, int]], float]:
    """Vectorized Type-II / Itakura slope-constrained DTW in [-1.0, 1.0].

    Strictly allows transitions (1,1), (2,1), (1,2) -> bounds local stretch to [0.5x, 2.0x].
    Rejects degenerate warps where mean(|i - j|) / N > max_mean_warp_frac.
    Returns (score, path, mean_abs_warp_frac).
    """
    if F_ms.shape != F_vr.shape or F_ms.ndim != 2 or F_ms.shape[0] == 0:
        return -1.0, [], 1.0
    N = F_ms.shape[0]
    if N == 1:
        sc = float(np.dot(F_ms[0], F_vr[0]))
        return sc, [(0, 0)], 0.0

    b = int(band) if band is not None else max(12, int(0.20 * N))
    sim = (F_ms @ F_vr.T).astype(np.float32)  # (N, N) in [-1, 1]

    neg_inf = np.float32(-1e9)
    dp = np.full((N, N), neg_inf, dtype=np.float32)
    ptr = np.zeros((N, N), dtype=np.uint8)  # 1=(1,1), 2=(2,1), 3=(1,2)
    dp[0, 0] = sim[0, 0]

    for i in range(1, N):
        j_lo = max(1, i - b)
        j_hi = min(N - 1, i + b)
        if j_lo > j_hi:
            continue
        js = np.arange(j_lo, j_hi + 1)

        # (1, 1): from (i-1, j-1), weight 2
        v1 = dp[i - 1, js - 1] + 2.0 * sim[i, js]
        best = v1
        best_p = np.ones_like(js, dtype=np.uint8)

        # (2, 1): from (i-2, j-1), weight 3
        if i >= 2:
            v2 = dp[i - 2, js - 1] + 1.5 * sim[i - 1, js] + 1.5 * sim[i, js] - warp_pen
            m2 = v2 > best
            best = np.where(m2, v2, best)
            best_p = np.where(m2, np.uint8(2), best_p)

        # (1, 2): from (i-1, j-2), weight 3 (only where j >= 2)
        if j_hi >= 2:
            valid3 = js >= 2
            j3 = js[valid3]
            v3_sub = dp[i - 1, j3 - 2] + 1.5 * sim[i, j3 - 1] + 1.5 * sim[i, j3] - warp_pen
            cur_sub = best[valid3]
            m3_sub = v3_sub > cur_sub
            if np.any(m3_sub):
                full_m3 = np.zeros_like(valid3)
                full_m3[valid3] = m3_sub
                v3_full = np.full_like(best, neg_inf)
                v3_full[valid3] = v3_sub
                best = np.where(full_m3, v3_full, best)
                best_p = np.where(full_m3, np.uint8(3), best_p)

        dp[i, j_lo : j_hi + 1] = best
        ptr[i, j_lo : j_hi + 1] = best_p

    final_val = float(dp[N - 1, N - 1])
    if final_val <= -1e8:
        return -1.0, [], 1.0

    # Backtrack optimal path P*
    path: List[Tuple[int, int]] = []
    ci, cj = N - 1, N - 1
    while ci > 0 or cj > 0:
        path.append((ci, cj))
        p = int(ptr[ci, cj])
        if p == 1:
            ci -= 1
            cj -= 1
        elif p == 2:
            path.append((ci - 1, cj))
            ci -= 2
            cj -= 1
        elif p == 3:
            path.append((ci, cj - 1))
            ci -= 1
            cj -= 2
        else:
            return -1.0, [], 1.0
    path.append((0, 0))
    path.reverse()

    mean_abs_warp_frac = float(np.mean([abs(i - j) for i, j in path])) / float(N)
    if mean_abs_warp_frac > max_mean_warp_frac:
        return -1.0, path, mean_abs_warp_frac

    score = final_val / float(2 * N - 1)
    return float(np.clip(score, -1.0, 1.0)), path, mean_abs_warp_frac


def compute_per_staff_path_correlations(
    F_ms: np.ndarray,
    F_vr: np.ndarray,
    path: Sequence[Tuple[int, int]],
    n_staves: int,
) -> List[float]:
    """Compute per-staff channel-centered Pearson correlation along shared DTW path P*."""
    if not path or n_staves <= 0:
        return []
    is_arr = np.array([p[0] for p in path], dtype=np.int64)
    js_arr = np.array([p[1] for p in path], dtype=np.int64)
    corrs: List[float] = []
    for s in range(n_staves):
        c0 = s * CHANNELS_PER_STAFF
        c1 = c0 + CHANNELS_PER_STAFF
        a = F_ms[is_arr, c0:c1]
        b = F_vr[js_arr, c0:c1]
        # Center each channel over the aligned path P* so sparse empty frames don't inflate r
        a_c = a - a.mean(axis=0, keepdims=True)
        b_c = b - b.mean(axis=0, keepdims=True)
        denom = float(np.linalg.norm(a_c) * np.linalg.norm(b_c))
        if denom < 1e-6:
            corrs.append(0.0)
        else:
            corrs.append(float(np.sum(a_c * b_c) / denom))
    return corrs


@dataclass
class CandidateEval:
    score: float
    path: List[Tuple[int, int]]
    warp_frac: float
    F_ms: np.ndarray
    F_vr: np.ndarray
    active_ms: List[bool]
    active_vr: List[bool]
    valid: bool = True
    dtw_score: float = 0.0
    bar_bonus: float = 0.0
    vr_bar_fracs: Optional[List[float]] = None


def solve_2best_viterbi_dp(
    n_systems: int,
    n_measures: int,
    m_start_candidates: Sequence[int],
    bar_ests: Sequence[int],
    bar_confs: Sequence[float],
    score_fn: Callable[[int, int, int], Optional[CandidateEval]],
) -> Tuple[Optional[Tuple[float, Tuple[Tuple[int, int], ...]]], Optional[Tuple[float, Tuple[Tuple[int, int], ...]]]]:
    """Global exact 2-Best Viterbi DP over (system_idx, measure_idx) with ZERO barline penalty."""
    prev: Dict[int, List[Tuple[float, Tuple[Tuple[int, int], ...]]]] = {}
    for m0 in m_start_candidates:
        if 0 <= m0 < n_measures:
            prev[int(m0)] = [(0.0, ())]

    for s in range(n_systems):
        if not prev:
            return None, None
        m_est = int(bar_ests[s])
        b_conf = float(bar_confs[s])
        if b_conf >= 0.4:
            lo_len = max(1, m_est - 4)
            hi_len = min(14, m_est + 5)
        else:
            lo_len = 1
            hi_len = 12

        curr: Dict[int, List[Tuple[float, Tuple[Tuple[int, int], ...]]]] = {}
        for m_s, paths in prev.items():
            for length in range(lo_len, hi_len + 1):
                m_e = m_s + length
                if m_e > n_measures:
                    break
                ev = score_fn(s, m_s, m_e)
                if ev is None or not ev.valid or ev.score <= -0.99:
                    continue
                bucket = curr.setdefault(m_e, [])
                for cum_sc, path in paths:
                    bucket.append((cum_sc + ev.score, path + ((m_s, m_e),)))

        # Keep exact top-2 distinct paths per state m_e
        next_prev: Dict[int, List[Tuple[float, Tuple[Tuple[int, int], ...]]]] = {}
        for m_e, items in curr.items():
            items.sort(key=lambda x: x[0], reverse=True)
            dedup: List[Tuple[float, Tuple[Tuple[int, int], ...]]] = []
            seen_paths: Set[Tuple[Tuple[int, int], ...]] = set()
            for sc, p in items:
                if p not in seen_paths:
                    seen_paths.add(p)
                    dedup.append((sc, p))
                    if len(dedup) == 2:
                        break
            if dedup:
                next_prev[m_e] = dedup
        prev = next_prev

    all_finals: List[Tuple[float, Tuple[Tuple[int, int], ...]]] = []
    for items in prev.values():
        all_finals.extend(items)
    all_finals.sort(key=lambda x: x[0], reverse=True)

    best1: Optional[Tuple[float, Tuple[Tuple[int, int], ...]]] = None
    best2: Optional[Tuple[float, Tuple[Tuple[int, int], ...]]] = None
    for sc, p in all_finals:
        if best1 is None:
            best1 = (sc, p)
        elif p != best1[1]:
            best2 = (sc, p)
            break
    return best1, best2


def _sample_decoy_ranges(
    m_s: int, m_e: int, n_measures: int, target_count: int = 10
) -> List[Tuple[int, int]]:
    """Select up to target_count strictly non-overlapping decoy slices of length L = m_e - m_s outside [m_s, m_e)."""
    L = m_e - m_s
    if L <= 0 or n_measures <= L:
        return []
    non_overlap: List[Tuple[int, int]] = []
    for d0 in range(0, n_measures - L + 1):
        d1 = d0 + L
        if d1 <= m_s or d0 >= m_e:
            non_overlap.append((d0, d1))

    if len(non_overlap) >= target_count:
        idxs = np.linspace(0, len(non_overlap) - 1, target_count, dtype=int)
        return [non_overlap[int(i)] for i in sorted(set(idxs.tolist()))]
    return non_overlap


def slice_page_modelfree(
    vis: Dict[str, Any],
    indexed: IndexedMusicXML,
    n_staves: int,
    tk: Any = None,
    *,
    m_start_candidates: Sequence[int] = (0, 1),
    min_dp_margin: float = 0.025,
    min_sys_score: float = 0.25,
    min_decoy_z: float = 2.5,
    min_decoy_gap: float = 0.05,
    min_staff_r: float = 0.22,
    max_staff_spread: float = 0.55,
    min_null_start_gap: float = 0.05,
    render_fn: Optional[RenderSliceFn] = None,
) -> Dict[str, Any]:
    """Run Stage 3 Global 2-Best Viterbi DP + Onset-DTW + Strict 3-Tier Label-Integrity Gate."""
    deskewed = vis["deskewed"]
    ink_vert = vis["ink_vert"]
    sys_crops = vis["system_crops"]
    n_systems = len(sys_crops)
    n_measures = indexed.n_measures

    if indexed.part_measure_mismatch:
        return {
            "accepted": False,
            "reason": f"rej_part_measure_mismatch({indexed.part_measure_counts})",
            "system_breaks": [],
            "sys_details": [],
        }

    if n_systems < 1 or n_measures < 1:
        return {"accepted": False, "reason": "empty_page_or_work", "system_breaks": [], "sys_details": []}

    # Verify every inked system has exact n_staves
    for s_idx, sc in enumerate(sys_crops):
        if len(sc["staves"]) != n_staves:
            return {
                "accepted": False,
                "reason": f"sys{s_idx}_staff_count_mismatch({len(sc['staves'])}!={n_staves})",
                "system_breaks": [],
                "sys_details": [],
            }

    # Pre-extract manuscript ink and barline estimates per system
    bar_ests: List[int] = []
    bar_confs: List[float] = []
    bar_hints_n: List[int] = []
    ms_bar_xs_rel: List[List[float]] = []
    ms_inks: List[np.ndarray] = []
    ms_bands_list: List[List[Tuple[int, int]]] = []

    for s_idx, sc in enumerate(sys_crops):
        m_est, bar_xs, b_conf = estimate_system_measures_from_barlines(ink_vert, sc)
        ms_ink, ms_bands, ink_x_origin = extract_symbol_ink_from_manuscript(
            deskewed, sc, return_x_offset=True
        )
        if ms_ink.sum() < 10.0 or len(ms_bands) != n_staves:
            return {
                "accepted": False,
                "reason": f"empty_or_bad_manuscript_sys{s_idx}",
                "system_breaks": [],
                "sys_details": [],
            }
        bar_ests.append(m_est)
        bar_confs.append(b_conf)
        bar_hints_n.append(len(bar_xs))
        if n_staves >= 2 and bar_xs:
            ms_bar_xs_rel.append([float(bx) - float(ink_x_origin) for bx in bar_xs])
        else:
            ms_bar_xs_rel.append([])
        ms_inks.append(ms_ink)
        ms_bands_list.append(ms_bands)

    bank_mgr = VerovioBankManager(tk, indexed, n_staves=n_staves, render_fn=render_fn)

    # Memoize manuscript profiles per (sys_idx, n_bars) and candidate evals per (sys_idx, m_s, m_e)
    ms_prof_cache: Dict[Tuple[int, int], Tuple[np.ndarray, List[bool]]] = {}
    cand_cache: Dict[Tuple[int, int, int], Optional[CandidateEval]] = {}

    def _preamble_frac(n_bars: int) -> float:
        return float(np.clip(0.32 / max(2, n_bars), 0.04, 0.11))

    def _get_ms_prof(s_idx: int, n_bars: int) -> Tuple[np.ndarray, List[bool]]:
        N = int(np.clip(32 * max(1, n_bars), 192, 320))
        key = (s_idx, n_bars)
        if key not in ms_prof_cache:
            ms_prof_cache[key] = extract_1d_profiles(
                ms_inks[s_idx],
                ms_bands_list[s_idx],
                n_bars=n_bars,
                target_w=N,
                sigma=1.4,
                mask_preamble_frac=_preamble_frac(n_bars),
            )
        return ms_prof_cache[key]

    def _get_ms_bar_fracs(s_idx: int, n_bars: int) -> List[float]:
        if not ms_bar_xs_rel[s_idx]:
            return []
        return _bars_to_profile_fracs(
            ms_bar_xs_rel[s_idx],
            ms_inks[s_idx].shape[1],
            _preamble_frac(n_bars),
        )

    def _eval_cand(s_idx: int, m_s: int, m_e: int) -> Optional[CandidateEval]:
        key = (s_idx, m_s, m_e)
        if key in cand_cache:
            return cand_cache[key]
        sl = bank_mgr.get_slice_with_bars(m_s, m_e)
        if sl is None:
            cand_cache[key] = None
            return None
        vr_ink, vr_bands, has_preamble, vr_bar_xs_px = sl
        if len(vr_bands) != n_staves:
            cand_cache[key] = None
            return None
        n_bars = m_e - m_s
        N = int(np.clip(32 * max(1, n_bars), 192, 320))
        vr_pf = _preamble_frac(n_bars) if has_preamble else 0.0
        F_ms, active_ms = _get_ms_prof(s_idx, n_bars)
        F_vr, active_vr = extract_1d_profiles(
            vr_ink,
            vr_bands,
            n_bars=n_bars,
            target_w=N,
            sigma=1.4,
            mask_preamble_frac=vr_pf,
        )
        dtw_sc, path, warp_frac = compare_system_profiles_dtw(F_ms, F_vr)
        vr_bar_fracs: Optional[List[float]] = None
        if vr_bar_xs_px is not None:
            vr_bar_fracs = _bars_to_profile_fracs(vr_bar_xs_px, vr_ink.shape[1], vr_pf)

        bar_bonus = 0.0
        if dtw_sc > -0.99 and len(path) > 0 and n_staves >= 2:
            ms_fracs = _get_ms_bar_fracs(s_idx, n_bars)
            if ms_fracs and vr_bar_fracs:
                bar_bonus = compute_barline_alignment_bonus(ms_fracs, vr_bar_fracs, path, N)

        total_sc = float(np.clip(dtw_sc + bar_bonus, -1.0, 1.0)) if dtw_sc > -0.99 else -1.0
        ev = CandidateEval(
            score=total_sc,
            path=path,
            warp_frac=warp_frac,
            F_ms=F_ms,
            F_vr=F_vr,
            active_ms=active_ms,
            active_vr=active_vr,
            valid=(dtw_sc > -0.99 and len(path) > 0),
            dtw_score=dtw_sc,
            bar_bonus=bar_bonus,
            vr_bar_fracs=vr_bar_fracs,
        )
        cand_cache[key] = ev
        return ev

    # Global 2-Best Viterbi DP (unpenalized)
    best1, best2 = solve_2best_viterbi_dp(
        n_systems=n_systems,
        n_measures=n_measures,
        m_start_candidates=m_start_candidates,
        bar_ests=bar_ests,
        bar_confs=bar_confs,
        score_fn=_eval_cand,
    )
    if best1 is None:
        return {
            "accepted": False,
            "reason": "no_valid_dp_path",
            "system_breaks": [],
            "sys_details": [],
        }

    dp1_sum, win_path = best1
    dp2_sum = best2[0] if best2 is not None else -float(n_systems)
    dp_margin = (dp1_sum - dp2_sum) / float(n_systems)
    mean_score = dp1_sum / float(n_systems)
    if n_staves == 1:
        min_mean_floor = 0.32
    else:
        min_mean_floor = max(0.28, 0.36 - 0.015 * float(n_staves - 2))

    # Build per-system diagnostics and run 3-Tier Label-Integrity Gate
    sys_details: List[Dict[str, Any]] = []
    sys_scores: List[float] = []
    tier_fail_reason: Optional[str] = None

    for s_idx, (m_s, m_e) in enumerate(win_path):
        ev = _eval_cand(s_idx, m_s, m_e)
        assert ev is not None
        sys_scores.append(ev.score)

        # Tier 2: Per-staff verification along shared DTW path P* (never skip one-sided tacet mismatches)
        staff_rs = compute_per_staff_path_correlations(ev.F_ms, ev.F_vr, ev.path, n_staves)
        active_rs = [
            staff_rs[st]
            for st in range(n_staves)
            if ev.active_ms[st] or ev.active_vr[st]
        ]
        if not active_rs:
            active_rs = staff_rs
        min_r_st = float(min(active_rs)) if active_rs else -1.0
        max_r_st = float(max(active_rs)) if active_rs else -1.0
        spread_st = max_r_st - min_r_st if len(active_rs) >= 2 else 0.0

        # Tier 1: Within-piece strictly non-overlapping Decoy Z-Score Gate (+ reversed template)
        # Use identical DTW + barline-bonus parameters across ALL decoys (never append -1.0 sentinels).
        n_bars_win = m_e - m_s
        N_win = ev.F_ms.shape[0]
        ms_fracs_win = _get_ms_bar_fracs(s_idx, n_bars_win) if n_staves >= 2 else []
        decoy_ranges = _sample_decoy_ranges(m_s, m_e, n_measures, target_count=10)
        decoy_scores: List[float] = []
        rev_sc, rev_path, _ = compare_system_profiles_dtw(
            ev.F_ms, ev.F_vr[::-1].copy(), max_mean_warp_frac=1.0
        )
        if rev_sc > -0.99:
            rev_bonus = 0.0
            if ms_fracs_win and ev.vr_bar_fracs:
                rev_vr_fracs = [1.0 - f for f in reversed(ev.vr_bar_fracs)]
                rev_bonus = compute_barline_alignment_bonus(
                    ms_fracs_win, rev_vr_fracs, rev_path, N_win
                )
            decoy_scores.append(float(np.clip(rev_sc + rev_bonus, -1.0, 1.0)))
        for d0, d1 in decoy_ranges:
            dev = _eval_cand(s_idx, d0, d1)
            if dev is not None:
                if dev.score > -0.99:
                    decoy_scores.append(dev.score)
                else:
                    d_sc, d_path, _ = compare_system_profiles_dtw(
                        ev.F_ms, dev.F_vr, max_mean_warp_frac=1.0
                    )
                    if d_sc > -0.99:
                        d_bonus = 0.0
                        if ms_fracs_win and dev.vr_bar_fracs:
                            d_bonus = compute_barline_alignment_bonus(
                                ms_fracs_win, dev.vr_bar_fracs, d_path, N_win
                            )
                        decoy_scores.append(float(np.clip(d_sc + d_bonus, -1.0, 1.0)))

        if len(decoy_scores) >= 6:
            sorted_decoys = sorted(decoy_scores)
            trimmed = sorted_decoys[:-1]
            decoy_mean = float(np.mean(trimmed))
            decoy_std = max(float(np.std(trimmed)), 1e-4)
            decoy_ref = float(sorted_decoys[-2])
            decoy_max = float(sorted_decoys[-1])
        elif len(decoy_scores) == 5:
            decoy_mean = float(np.mean(decoy_scores))
            decoy_std = max(float(np.std(decoy_scores)), 1e-4)
            decoy_ref = float(np.max(decoy_scores))
            decoy_max = decoy_ref
        else:
            decoy_mean = float(np.mean(decoy_scores)) if decoy_scores else 0.0
            decoy_std = max(float(np.std(decoy_scores)), 1e-4) if len(decoy_scores) >= 2 else 0.1
            decoy_ref = float(np.max(decoy_scores)) if decoy_scores else -1.0
            decoy_max = decoy_ref
        z_decoy = (ev.score - decoy_mean) / decoy_std
        decoy_gap = ev.score - decoy_ref

        detail = {
            "sys_idx": s_idx,
            "m_start": int(m_s),
            "m_end": int(m_e),
            "m_est": bar_ests[s_idx],
            "bar_conf": bar_confs[s_idx],
            "n_bar_hints": bar_hints_n[s_idx],
            "score": round(ev.score, 4),
            "dtw_score": round(ev.dtw_score, 4),
            "bar_bonus": round(ev.bar_bonus, 4),
            "warp_frac": round(ev.warp_frac, 4),
            "n_decoys": len(decoy_scores),
            "decoy_mean": round(decoy_mean, 4),
            "decoy_max": round(decoy_max, 4),
            "z_decoy": round(z_decoy, 3),
            "decoy_gap": round(decoy_gap, 4),
            "staff_rs": [round(r, 4) for r in staff_rs],
            "min_staff_r": round(min_r_st, 4),
            "staff_spread": round(spread_st, 4),
        }
        sys_details.append(detail)

        if tier_fail_reason is None:
            if ev.score < min_sys_score:
                tier_fail_reason = f"tier3_low_sys_score_s{s_idx}({ev.score:.3f}<{min_sys_score:.2f})"
            elif len(decoy_scores) < 5:
                tier_fail_reason = f"tier1_insufficient_decoys_s{s_idx}(n={len(decoy_scores)}<5)"
            elif z_decoy < min_decoy_z or decoy_gap < min_decoy_gap:
                tier_fail_reason = (
                    f"tier1_decoy_fail_s{s_idx}(z={z_decoy:.2f}<{min_decoy_z:.1f},"
                    f"gap={decoy_gap:.3f}<{min_decoy_gap:.2f})"
                )
            elif min_r_st < min_staff_r:
                tier_fail_reason = f"tier2_low_staff_r_s{s_idx}({min_r_st:.3f}<{min_staff_r:.2f})"
            elif len(active_rs) >= 2 and spread_st > max_staff_spread:
                tier_fail_reason = f"tier2_staff_spread_s{s_idx}({spread_st:.3f}>{max_staff_spread:.2f})"

    # Tier 3: Global score floor + 2-Best DP path margin + Null-start DP control
    if tier_fail_reason is None and mean_score < min_mean_floor:
        tier_fail_reason = f"tier3_low_mean_score({mean_score:.3f}<{min_mean_floor:.3f})"
    if tier_fail_reason is None and best2 is not None and dp_margin < min_dp_margin:
        tier_fail_reason = f"tier3_low_dp_margin({dp_margin:.4f}<{min_dp_margin:.3f})"

    win_m_start = int(win_path[0][0])
    win_m_end = int(win_path[-1][1])
    win_span = win_m_end - win_m_start
    null_start_gap: Optional[float] = None

    m_null: Optional[int] = None
    if n_measures >= win_m_end + max(8, win_span):
        m_null = min(n_measures - max(1, win_span), win_m_end + 4)
    elif win_m_start >= max(8, win_span):
        m_null = max(0, win_m_start - win_span - 4)

    if tier_fail_reason is None and m_null is not None:
        null_best1, _ = solve_2best_viterbi_dp(
            n_systems=n_systems,
            n_measures=n_measures,
            m_start_candidates=(m_null,),
            bar_ests=bar_ests,
            bar_confs=bar_confs,
            score_fn=_eval_cand,
        )
        if null_best1 is not None:
            null_start_gap = (dp1_sum - null_best1[0]) / float(n_systems)
            if null_start_gap < min_null_start_gap:
                tier_fail_reason = (
                    f"tier3_null_start_fail(gap={null_start_gap:.3f}<{min_null_start_gap:.2f},m_null={m_null})"
                )

    system_breaks = [[int(a), int(b)] for a, b in win_path]
    accepted = tier_fail_reason is None
    return {
        "accepted": accepted,
        "reason": "ok" if accepted else tier_fail_reason,
        "m_start": win_m_start,
        "m_end": win_m_end,
        "system_breaks": system_breaks,
        "mean_score": round(mean_score, 4),
        "min_sys_score": round(float(min(sys_scores)), 4) if sys_scores else -1.0,
        "dp_margin": round(dp_margin, 4),
        "null_start_gap": round(null_start_gap, 4) if null_start_gap is not None else None,
        "sys_details": sys_details,
    }


# ---------------------------------------------------------------------------
# 6. Export, Measure-Count Assertion, Uniform v2a/v1 & Grammar Validation
# ---------------------------------------------------------------------------

def count_bekern_measures(toks: Sequence[str]) -> int:
    """Count end-of-measure barline tokens in a single-system bekern sequence."""
    return sum(1 for t in toks if t in END_OF_MEASURE_BAR_TOKENS)


def export_system_bekern(
    xml_path: Path,
    out_bekerns_dir: Path,
    page_stem: str,
    sys_idx: int,
    m_start_0: int,
    m_end_0: int,
    *,
    version: str = "v2a",
    aser_bin: Path = ASER_BIN,
    timeout_s: int = 30,
) -> Tuple[Optional[Path], str]:
    """Export one system [m_start_0, m_end_0) in a specific bekern version (v2a or v1).

    Unlinks stale output before and on error, enforces 0 OOV and exact measure-count match.
    """
    out_bekerns_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{page_stem}_s{sys_idx}"
    out_txt = out_bekerns_dir / f"{stem}-bekern.txt"
    out_txt.unlink(missing_ok=True)

    measures_arg = f"{m_start_0 + 1}..{m_end_0}"
    expected_bars = m_end_0 - m_start_0
    extra_flags = ["--v2"] if version == "v2a" else []
    cmd = [
        str(aser_bin),
        "bekern",
        "--input",
        str(xml_path),
        "--out",
        str(out_bekerns_dir),
        "--stem",
        stem,
        "--measures",
        measures_arg,
        *extra_flags,
    ]
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        out_txt.unlink(missing_ok=True)
        return None, f"timeout({measures_arg})"
    except Exception as exc:
        out_txt.unlink(missing_ok=True)
        return None, f"exec_err({exc})"

    if res.returncode != 0 or not out_txt.is_file():
        out_txt.unlink(missing_ok=True)
        return None, f"rc={res.returncode}({measures_arg})"

    toks = out_txt.read_text(encoding="utf-8", errors="replace").split()
    if not toks:
        out_txt.unlink(missing_ok=True)
        return None, f"empty_bekern({measures_arg})"

    oov = [t for t in toks if t not in data.TOKEN_TO_ID]
    if oov:
        out_txt.unlink(missing_ok=True)
        return None, f"oov({oov[:3]})"

    bar_count = count_bekern_measures(toks)
    if bar_count != expected_bars:
        out_txt.unlink(missing_ok=True)
        return None, f"measure_count_mismatch(got={bar_count}!=exp={expected_bars})"

    return out_txt, "ok"


def validate_page_grammar(sys_paths: Sequence[str]) -> Tuple[bool, str]:
    """Check 0 OOV and PrefixState(strict=False) sanity across the exported page."""
    ps = grammar.PrefixState(strict=False)
    for s_idx, p_str in enumerate(sys_paths):
        if s_idx > 0:
            ps.reset_system()
        toks = Path(p_str).read_text(encoding="utf-8", errors="replace").split()
        oov = [t for t in toks if t not in data.TOKEN_TO_ID]
        if oov:
            return False, f"oov({oov[:3]})"
        for tidx, tok in enumerate(toks):
            try:
                ps.step(tok)
            except Exception as exc:
                return False, f"grammar_err_s{s_idx}_{tidx}({tok}:{exc})"
            if ps.broken:
                return False, f"grammar_broken_s{s_idx}_{tidx}({tok})"
    return True, "ok"


def export_page_uniform_bekern(
    xml_path: Path,
    out_bekerns_dir: Path,
    page_stem: str,
    system_breaks: Sequence[Sequence[int]],
    *,
    aser_bin: Path = ASER_BIN,
    export_fn: Optional[Callable[..., Tuple[Optional[Path], str]]] = None,
    grammar_fn: Optional[Callable[[Sequence[str]], Tuple[bool, str]]] = None,
) -> Tuple[Optional[List[str]], str, str]:
    """Export all systems uniformly in v2a (or uniformly in v1, NEVER mixed) with hard grammar gate."""
    _exp = export_fn or export_system_bekern
    _gram = grammar_fn or validate_page_grammar

    last_err = "export_failed"
    for ver in ("v2a", "v1"):
        sys_paths: List[str] = []
        ok = True
        for s_idx, (ms, me) in enumerate(system_breaks):
            out_txt, err = _exp(
                xml_path,
                out_bekerns_dir,
                page_stem,
                s_idx,
                int(ms),
                int(me),
                version=ver,
                aser_bin=aser_bin,
            )
            if out_txt is None:
                ok = False
                last_err = f"{ver}_s{s_idx}:{err}"
                break
            sys_paths.append(str(out_txt))

        if not ok:
            for p_str in sys_paths:
                Path(p_str).unlink(missing_ok=True)
            continue

        gram_ok, gram_msg = _gram(sys_paths)
        if not gram_ok:
            for p_str in sys_paths:
                Path(p_str).unlink(missing_ok=True)
            last_err = f"{ver}_grammar_reject:{gram_msg}"
            continue

        return sys_paths, ver, "ok"

    return None, "", last_err


def save_page_jpeg(
    src_img_path: Path, dst_jpg_path: Path, max_dim: int = 1536, quality: int = 95
) -> Tuple[int, int]:
    """Save EXIF-transposed RGB JPEG atomically."""
    dst_jpg_path.parent.mkdir(parents=True, exist_ok=True)
    with safe_open_image(src_img_path) as raw:
        im = raw.convert("RGB")
        w, h = im.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            im = im.resize(
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, subsampling=0)
        atomic_write_bytes(dst_jpg_path, buf.getvalue())
        return im.size


# ---------------------------------------------------------------------------
# 7. Per-Work Worker & Main Orchestrator (with --resume & ProcessPoolExecutor)
# ---------------------------------------------------------------------------

_CANVAS_NUM_RE = re.compile(r"^(.*_c)(\d+)(.*)$")
_WORKER_TK: Any = None
_WORKER_XML_CACHE: Optional[IndexedMusicXMLCache] = None
_WORKER_GUARDS: Optional[List[Tuple[str, fh.EvalLeakageGuard]]] = None


def next_canvas_image_path(img_path: Path) -> Optional[Path]:
    """Synthesize sibling canvas c+1 path matching original _c{idx:02d}/_c{idx} zero-padding."""
    m = _CANVAS_NUM_RE.match(img_path.stem)
    if not m:
        return None
    prefix, num_str, suffix = m.group(1), m.group(2), m.group(3)
    next_num = int(num_str) + 1
    width = len(num_str)
    next_num_str = f"{next_num:0{width}d}" if width >= 2 else str(next_num)
    return img_path.with_name(f"{prefix}{next_num_str}{suffix}{img_path.suffix}")


def _get_worker_resources() -> Tuple[Any, IndexedMusicXMLCache, List[Tuple[str, fh.EvalLeakageGuard]]]:
    """Lazily initialize one Verovio toolkit, 32-entry XML LRU cache, and leakage guards per worker."""
    global _WORKER_TK, _WORKER_XML_CACHE, _WORKER_GUARDS
    if _WORKER_TK is None:
        _WORKER_TK = create_verovio_toolkit()
    if _WORKER_XML_CACHE is None:
        _WORKER_XML_CACHE = IndexedMusicXMLCache(max_size=32)
    if _WORKER_GUARDS is None:
        _WORKER_GUARDS = build_combined_leakage_guards()
    return _WORKER_TK, _WORKER_XML_CACHE, _WORKER_GUARDS


def _process_work_group(
    rows: List[Dict[str, Any]],
    *,
    xml_cache_dir: Path,
    bekerns_dir: Path,
    pages_dir: Path,
    aser_bin: Path,
    min_dp_margin: float,
    min_sys_score: float,
    done_stems: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Process all pages belonging to one work (with multi-page c->c+1..N sequential chaining)."""
    tk, xml_cache, guards = _get_worker_resources()
    local_seen_stems: Set[str] = set(done_stems)
    results: List[Dict[str, Any]] = []

    for seed_row in rows:
        cur_row: Optional[Dict[str, Any]] = seed_row
        while cur_row is not None:
            row = cur_row
            cur_row = None
            t_p = time.perf_counter()
            rism_id = str(row.get("rism_id", ""))
            img_path = Path(str(row.get("image_path", "")))
            page_stem = sanitize_page_stem(rism_id, img_path)
            local_seen_stems.add(page_stem)

            try:
                work_n_staves = int(row.get("work_n_staves") or 0)
                if not img_path.is_file():
                    results.append({"status": "rej_missing_img", "page_stem": page_stem, "rism_id": rism_id, "reason": "missing_image"})
                    continue

                # Stage 2: Visual Canvas & Physical Staff-Count Gate
                matched_staff = row.get("matched_staff")
                with safe_open_image(img_path) as im:
                    is_nm, nm_info = ncg.is_non_music_page(im)
                    if is_nm:
                        results.append({
                            "status": "rej_non_music",
                            "page_stem": page_stem,
                            "rism_id": rism_id,
                            "reason": f"non_music:{nm_info['reason']}",
                        })
                        continue
                    vis = detect_staves_and_systems(
                        im, force_single_staff=(work_n_staves == 1)
                    )
                    vis_staves = int(vis["visual_n_staves"])
                    vis_systems = int(vis["visual_n_systems"])
                    all_uniform = bool(vis.get("all_systems_uniform", False))

                    # Support 1-staff historical partbooks when matched_staff is known
                    is_partbook_1staff = (
                        vis_staves == 1
                        and work_n_staves > 1
                        and matched_staff is not None
                        and 0 <= int(matched_staff) < work_n_staves
                    )
                    if is_partbook_1staff and not all_uniform:
                        vis = detect_staves_and_systems(im, force_single_staff=True)
                        vis_staves = int(vis["visual_n_staves"])
                        vis_systems = int(vis["visual_n_systems"])
                        all_uniform = bool(vis.get("all_systems_uniform", False))

                target_staves = 1 if is_partbook_1staff else work_n_staves

                if vis_systems < 1 or not all_uniform or vis_staves != target_staves:
                    sys_counts_diag = vis.get("system_staff_counts")
                    vis.clear()
                    results.append({
                        "status": "rej_staff_mismatch",
                        "page_stem": page_stem,
                        "rism_id": rism_id,
                        "reason": (
                            f"staff_mismatch(vis_st={vis_staves},vis_sys={vis_systems},"
                            f"uniform={all_uniform},counts={sys_counts_diag},target={target_staves})"
                        ),
                    })
                    continue

                # Stage 3: XML resolution & slicing
                src_path = Path(str(row.get("resolved_source_path", "")))
                xml_path = convert_to_musicxml(src_path, xml_cache_dir)
                if xml_path is None or not xml_path.is_file():
                    vis.clear()
                    results.append({
                        "status": "rej_no_xml",
                        "page_stem": page_stem,
                        "rism_id": rism_id,
                        "reason": f"no_xml({src_path.name})",
                    })
                    continue

                indexed = xml_cache.get(xml_path)
                if indexed.part_measure_mismatch:
                    vis.clear()
                    results.append({
                        "status": "rej_part_measure_mismatch",
                        "page_stem": page_stem,
                        "rism_id": rism_id,
                        "reason": f"part_measure_mismatch({indexed.part_measure_counts})",
                    })
                    continue

                export_xml_path = xml_path
                if is_partbook_1staff:
                    single_indexed = extract_single_staff_musicxml(indexed, int(matched_staff))
                    if single_indexed is None:
                        vis.clear()
                        results.append({
                            "status": "rej_no_xml",
                            "page_stem": page_stem,
                            "rism_id": rism_id,
                            "reason": f"partbook_extract_failed(staff={matched_staff})",
                        })
                        continue
                    indexed = single_indexed
                    # Materialize hash-keyed single-staff XML for assembly-serialize export
                    stem_clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", xml_path.stem)
                    export_xml_path = (
                        xml_cache_dir
                        / f"{stem_clean}_{_file_sha1(xml_path)[:12]}_st{int(matched_staff)}.musicxml"
                    )
                    if not export_xml_path.is_file():
                        atomic_write_text(
                            export_xml_path,
                            slice_musicxml_measures(indexed, 0, indexed.n_measures),
                        )

                m_start_cands = row.get("m_start_candidates", (0, 1))
                res = slice_page_modelfree(
                    vis,
                    indexed,
                    n_staves=target_staves,
                    tk=tk,
                    m_start_candidates=m_start_cands,
                    min_dp_margin=min_dp_margin,
                    min_sys_score=min_sys_score,
                )
                # Immediately free large numpy arrays from _vis
                vis.pop("deskewed", None)
                vis.pop("ink_vert", None)
                dt_p = round(time.perf_counter() - t_p, 2)

                diag = {
                    "page_stem": page_stem,
                    "rism_id": rism_id,
                    "image": img_path.name,
                    "n_staves": target_staves,
                    "is_partbook_1staff": is_partbook_1staff,
                    "n_systems": vis_systems,
                    "n_measures_work": indexed.n_measures,
                    "part_measure_mismatch": indexed.part_measure_mismatch,
                    "elapsed_s": dt_p,
                    **res,
                }

                if not res["accepted"]:
                    results.append({
                        "status": "rej_verovio",
                        "page_stem": page_stem,
                        "rism_id": rism_id,
                        "reason": res["reason"],
                        "diag": diag,
                    })
                    continue

                sbs = res["system_breaks"]
                sys_paths, bk_ver, exp_err = export_page_uniform_bekern(
                    export_xml_path, bekerns_dir, page_stem, sbs, aser_bin=aser_bin
                )
                if sys_paths is None:
                    results.append({
                        "status": "rej_export",
                        "page_stem": page_stem,
                        "rism_id": rism_id,
                        "reason": exp_err,
                        "diag": diag,
                    })
                    continue

                dst_jpg = pages_dir / f"{page_stem}.jpg"
                out_w, out_h = save_page_jpeg(img_path, dst_jpg, max_dim=1536, quality=95)
                short_title = str(row.get("title", "")).split(";")[0].strip()[:80]
                entry = {
                    "image": str(dst_jpg),
                    "bekerns": sys_paths,
                    "domain": "real_hw",
                    "source": "rism_gold_modelfree_v2",
                    "page_stem": page_stem,
                    "rism_id": rism_id,
                    "orig_image_path": str(img_path),
                    "work_id": str(row.get("work_id", "")),
                    "work_source_path": str(row.get("resolved_source_path", "")),
                    "title": short_title,
                    "n_staves": target_staves,
                    "is_partbook_1staff": is_partbook_1staff,
                    "n_systems": len(sbs),
                    "m_start": int(res["m_start"]),
                    "m_end": int(res["m_end"]),
                    "system_breaks": sbs,
                    "mean_score": float(res["mean_score"]),
                    "dp_margin": float(res["dp_margin"]),
                    "bekern_version": bk_ver,
                    "grammar_ok": True,
                    "width": out_w,
                    "height": out_h,
                }
                results.append({
                    "status": "accepted",
                    "page_stem": page_stem,
                    "rism_id": rism_id,
                    "entry": entry,
                    "diag": diag,
                })

                # Multi-page sequential chaining (c -> c+1..N)
                next_m_start = int(res["m_end"])
                if next_m_start < indexed.n_measures:
                    nxt_path = next_canvas_image_path(img_path)
                    if nxt_path is not None and nxt_path.is_file():
                        nxt_stem = sanitize_page_stem(rism_id, nxt_path)
                        if nxt_stem not in local_seen_stems:
                            nxt_row = dict(row)
                            nxt_row["image_path"] = str(nxt_path)
                            nxt_row["m_start_candidates"] = (next_m_start,)
                            leaked, _ = check_eval_overlap(nxt_row, guards)
                            if not leaked:
                                cur_row = nxt_row
            except Exception as exc:
                results.append({
                    "status": "rej_error",
                    "page_stem": page_stem,
                    "rism_id": rism_id,
                    "reason": f"exception:{type(exc).__name__}:{exc}",
                })

    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    configure_stdout_utf8()
    ap = argparse.ArgumentParser(
        description="100% Model-Free Barline + Verovio Measure-Bank GOLD Slicer (v2)"
    )
    ap.add_argument("--gold-jsonl", type=Path, default=GOLD_JSONL)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--xml-cache-dir", type=Path, default=None)
    ap.add_argument("--aser-bin", type=Path, default=ASER_BIN)
    ap.add_argument("--min-dp-margin", type=float, default=0.025)
    ap.add_argument("--min-sys-score", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    if not args.aser_bin.is_file():
        raise FileNotFoundError(f"assembly-serialize binary not found: {args.aser_bin}")

    t_start = time.perf_counter()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_cache_dir = args.xml_cache_dir or (out_dir / "_xml_cache")
    bekerns_dir = out_dir / "bekerns"
    pages_dir = out_dir / "pages"
    state_path = out_dir / "state.jsonl"
    diag_path = out_dir / "diagnostics.jsonl"
    manifest_path = out_dir / "manifest.jsonl"

    done_stems: Set[str] = set()
    accepted_rows: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    if args.resume and state_path.is_file():
        for line in state_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done_stems.add(str(rec["page_stem"]))
                st = rec.get("status")
                if st:
                    counts[str(st)] += 1
                if st == "accepted" and "entry" in rec:
                    accepted_rows.append(rec["entry"])
            except Exception:
                continue
    elif not args.resume:
        state_path.unlink(missing_ok=True)
        diag_path.unlink(missing_ok=True)

    raw_rows = [
        json.loads(line)
        for line in args.gold_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    corpus_idx = CorpusIndex().build()
    for row in raw_rows:
        rpath, rext = corpus_idx.resolve(
            str(row.get("work_id", "")), str(row.get("work_source_path", ""))
        )
        row["resolved_source_path"] = rpath
        row["resolved_source_ext"] = rext

    guards = build_combined_leakage_guards()
    leaked_rows: List[Tuple[str, str]] = []
    clean_golds: List[Dict[str, Any]] = []
    for row in raw_rows:
        is_leak, leak_reason = check_eval_overlap(row, guards)
        if is_leak:
            leaked_rows.append((str(row.get("rism_id", "")), leak_reason))
        else:
            clean_golds.append(row)

    print(
        f"[Step 0] Loaded {len(raw_rows)} GOLD matches | Eval overlap: {len(leaked_rows)} | "
        f"Resumed done: {len(done_stems)}",
        flush=True,
    )
    assert len(leaked_rows) == 0, f"Eval overlap detected: {leaked_rows}"

    # Group pending rows by normalized resolved_source_path for work-level XML/Verovio reuse
    work_groups: Dict[str, List[Dict[str, Any]]] = {}
    seen_stems: Set[str] = set(done_stems)
    for row in clean_golds:
        stem = sanitize_page_stem(str(row.get("rism_id", "")), Path(str(row.get("image_path", ""))))
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        rpath = str(row.get("resolved_source_path", "")).strip()
        wkey = str(Path(rpath).resolve()).lower() if rpath else str(row.get("work_id", ""))
        work_groups.setdefault(wkey, []).append(row)

    def _handle_batch(batch_res: List[Dict[str, Any]]) -> None:
        for r in batch_res:
            st = str(r["status"])
            counts[st] += 1
            diag = r.pop("diag", None)
            if diag is not None:
                append_jsonl_line(diag_path, diag)
            append_jsonl_line(state_path, r)
            if st == "accepted" and "entry" in r:
                accepted_rows.append(r["entry"])
                e = r["entry"]
                print(
                    f"  ACCEPTED  {e['rism_id']:10s} | st={e['n_staves']} sys={e['n_systems']} | "
                    f"sbs={e['system_breaks']} | score={e['mean_score']:.3f} "
                    f"dp_margin={e['dp_margin']:.3f} | {e['bekern_version']}",
                    flush=True,
                )
            else:
                print(
                    f"  {st.upper():18s} {r.get('rism_id', ''):10s} -> {r.get('reason', '')}",
                    flush=True,
                )

    worker_kwargs = dict(
        xml_cache_dir=xml_cache_dir,
        bekerns_dir=bekerns_dir,
        pages_dir=pages_dir,
        aser_bin=args.aser_bin,
        min_dp_margin=args.min_dp_margin,
        min_sys_score=args.min_sys_score,
        done_stems=tuple(seen_stems),
    )

    if args.workers <= 1:
        for grp in work_groups.values():
            _handle_batch(_process_work_group(grp, **worker_kwargs))
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = [
                ex.submit(_process_work_group, grp, **worker_kwargs)
                for grp in work_groups.values()
            ]
            for fut in as_completed(futs):
                _handle_batch(fut.result())

    manifest_lines = "\n".join(json.dumps(e, ensure_ascii=False) for e in accepted_rows)
    if manifest_lines:
        manifest_lines += "\n"
    atomic_write_text(manifest_path, manifest_lines)

    summary = {
        "input_gold_matches": len(raw_rows),
        "eval_overlap_count": len(leaked_rows),
        "resumed_skipped": len(done_stems),
        "status_counts": dict(counts),
        "final_accepted": len(accepted_rows),
        "oov_count": 0,
        "elapsed_secs": round(time.perf_counter() - t_start, 2),
        "accepted": accepted_rows,
    }
    atomic_write_text(out_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"\n[DONE] Accepted={len(accepted_rows)} / {len(raw_rows)} in {summary['elapsed_secs']:.1f}s "
        f"-> {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
