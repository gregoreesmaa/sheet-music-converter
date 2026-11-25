r"""Resumable parallel generator for simulated handwritten Stage-1 & Stage-2 training data.

Generates diverse handwritten full pages using all 159 named styles (50%) and
continuous `sample_random_scribe(rng, jitter=0.25)` unseen scribes (50%), with:
  - Hard eval-overlap blocklist from `suite31_manifest.json` and `suite31_hw_manifest.json`
    (`eval_overlap == 0` strictly enforced).
  - Exact Stage-1 `manifest.jsonl` schema matching `D:\tmp\staff-unet-data\manifest.jsonl`
    (with 32-point curved Bezier 5-line arrays `(5, 32)` per staff).
  - Merged Stage-1 manifest `D:\tmp\staff-unet-data\manifest_with_hwsim.jsonl`
    (non-destructive; original `manifest.jsonl` untouched).
  - Exact Stage-2 `stage2_strips/<stem>_s<k>.png` + `<stem>_s<k>-bekern.txt` flat strip
    layout validated against `tmp-train/mvd/data.py`'s closed `VOCAB`.

CPU-only by design (`CUDA_VISIBLE_DEVICES=""`) so it never interferes with GPU training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Enforce CPU-only execution before importing anything else
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_MVD_DIR = _REPO_ROOT / "tmp-train" / "mvd"

for _p in (str(_THIS_DIR), str(_MVD_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Eval-overlap blocklist (hard zero-leakage gate)
# ---------------------------------------------------------------------------

SUITE31_MANIFESTS: Tuple[Path, ...] = (
    _REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_manifest.json",
    _REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json",
    _REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_only_manifest.json",
)

GATE20_EXCLUSION_JSON: Path = _REPO_ROOT / "data-gen" / "score-corpus" / "gate20" / "exclusion.json"
GATE20_MANIFEST_JSON: Path = _REPO_ROOT / "data-gen" / "score-corpus" / "gate20" / "gate_manifest.json"

# Separator-preserving raw patterns for opus+number pairs that would collide with unrelated
# opus numbers if stripped of punctuation (e.g. "op6-2" normalized -> "op62" would falsely
# block Opus 62 works!). Matched on raw lowercased path/stem with digit-boundary guards.
_EXTRA_RAW_EVAL_REGEXES: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(pat, re.IGNORECASE)
    for pat in (
        r"(?:^|[^a-z0-9])op\.?\s*0?6[\s_\-\.]+(?:no\.?\s*)?2(?:[^0-9]|$)",
        r"(?:^|[^a-z0-9])op\.?\s*17[\s_\-\.]+(?:no\.?\s*)?3(?:[^0-9]|$)",
        r"(?:^|[^a-z0-9])op\.?\s*18[\s_\-\.]+(?:no\.?\s*|n)?1(?:[^0-9]|$)",
        r"(?:^|[^a-z0-9])op\.?\s*41[\s_\-\.]+(?:no\.?\s*|n)?1(?:[^0-9]|$)",
        r"(?:^|[^a-z0-9])op\.?\s*74[\s_\-\.]+(?:no\.?\s*|n)?1(?:[^0-9]|$)",
        r"(?:^|[^a-z0-9])op\.?\s*0?1[\s_\-\.]+(?:no\.?\s*|n)?1(?:[^0-9]|$)",
        r"(?:^|[^a-z0-9])op\.?\s*19(?:[^0-9]|$)",
    )
)

# Explicit normalized substring tokens derived from all 45 suite31 + hw eval pieces
# to catch differently-named copies of the exact same compositions across repos.
# NOTE: Never put bare "op6-2" here (normalizes to "op62" -> false-positive on Opus 62).
_EXTRA_EVAL_TOKENS: Tuple[str, ...] = (
    "bwv66",
    "bwv_66",
    "bwv-66",
    "maple_leaf",
    "maple-leaf",
    "mapleleaf",
    "grosse_fuge",
    "grosse-fuge",
    "grossefuge",
    "k155",
    "k_155",
    "k-155",
    "k458",
    "k_458",
    "k-458",
    "op18n1",
    "op18_no1",
    "op74n1",
    "op74_no1",
    "mazurka06",
    "op06_no2",
    "op6_no2",
    "lindenbaum",
    "dichterliebe",
    "op41n1",
    "op1n1",
    "concertino",
    "la_donna",
    "la-donna",
    "h186",
    "beach-prayer",
    "lascia",
    "luca-gloria",
    "aloha",
    "foster-leadsheet",
    "palestrina-agnus",
    "op135",
    "op_135",
    "op-135",
    "k464",
    "k_464",
    "k-464",
    "k387",
    "k_387",
    "k-387",
    "k421",
    "k_421",
    "k-421",
    "d810",
    "d_810",
    "d-810",
    "vetement",
    "berceuse_heroique",
    "berceuse-heroique",
    "la_plus_que_lente",
    "la-plus-que-lente",
    "lindaraja",
    "morgengruss",
    "herbstlied",
    "blumchen",
    "bluemchen",
    "botschaft",
    "im_herbste",
    "im-herbste",
    "repeats-melody",
    "fiddle-strathspey",
    "ryans-jig",
    "ryans-hornpipe",
    "oneills-tune",
    "airds-air",
    "essen-altdeu",
    "essen-ballad",
)


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def load_eval_blocklist() -> Tuple[Set[str], Set[str], Set[str]]:
    """Return (raw_slugs, norm_tokens, gt_sha256s) from all suite31_hw + gate20 manifests."""
    raw_slugs: Set[str] = set()
    norm_tokens: Set[str] = set()
    gt_sha256s: Set[str] = set()

    for mp in SUITE31_MANIFESTS:
        if not mp.exists():
            continue
        doc = json.loads(mp.read_text(encoding="utf-8"))
        for piece in doc.get("pieces", []):
            slug = str(piece.get("slug", "")).strip().lower()
            if slug:
                raw_slugs.add(slug)
                norm_tokens.add(_norm_key(slug))
                if slug.startswith("hw-"):
                    stripped = slug[3:]
                    raw_slugs.add(stripped)
                    norm_tokens.add(_norm_key(stripped))
            gt = piece.get("gt")
            if isinstance(gt, dict) and gt.get("sha256"):
                gt_sha256s.add(str(gt["sha256"]).lower())

    # Strict gate20 work-level exclusion (page stems -> work stems + upstream lc* IDs + GT sha256)
    if GATE20_EXCLUSION_JSON.exists():
        g_ex = json.loads(GATE20_EXCLUSION_JSON.read_text(encoding="utf-8"))
        for key in ("do_not_train", "held_out_4page_gate"):
            for page_id in g_ex.get(key, []):
                work_stem = re.sub(r"-\d+$", "", str(page_id).strip().lower())
                if work_stem:
                    raw_slugs.add(work_stem)
                    norm_tokens.add(_norm_key(work_stem))

    if GATE20_MANIFEST_JSON.exists():
        g_man = json.loads(GATE20_MANIFEST_JSON.read_text(encoding="utf-8"))
        for page in g_man.get("pages", []):
            score_id = str(page.get("score", "")).strip().lower()
            if score_id:
                raw_slugs.add(score_id)
                norm_tokens.add(_norm_key(score_id))
            src = str(page.get("source", ""))
            for lc_m in re.findall(r"lc\d{5,}", src, flags=re.IGNORECASE):
                raw_slugs.add(lc_m.lower())
                norm_tokens.add(lc_m.lower())
            gt_sha = page.get("ground_truth_musicxml_sha256")
            if gt_sha:
                gt_sha256s.add(str(gt_sha).lower())

    for tok in _EXTRA_EVAL_TOKENS:
        norm_tokens.add(_norm_key(tok))

    # Keep only meaningful length tokens (>= 4 chars) to avoid false positives
    norm_tokens = {t for t in norm_tokens if len(t) >= 4}
    return raw_slugs, norm_tokens, gt_sha256s


def is_eval_blocked(
    path: Path,
    raw_slugs: Set[str],
    norm_tokens: Set[str],
    gt_sha256s: Set[str],
    check_sha: bool = True,
) -> bool:
    """Return True if `path` matches any eval slug, regex, normalized token, or GT SHA-256."""
    full_str = str(path).lower()
    stem_l = path.stem.lower()
    parent_and_stem = f"{path.parent.name}/{path.stem}".lower()
    norm_full = _norm_key(path.parent.name + "_" + path.stem)
    norm_stem = _norm_key(path.stem)

    for slug in raw_slugs:
        if slug in full_str or slug in stem_l:
            return True
    for pat in _EXTRA_RAW_EVAL_REGEXES:
        if pat.search(parent_and_stem) or pat.search(stem_l):
            return True
    for tok in norm_tokens:
        if tok in norm_stem or tok in norm_full:
            return True
    if check_sha and gt_sha256s:
        try:
            h = hashlib.sha256(path.read_bytes()).hexdigest().lower()
            if h in gt_sha256s:
                return True
        except OSError:
            return True
    return False


# ---------------------------------------------------------------------------
# Permissive / CC0 / CC-BY Score Pool Discovery
# ---------------------------------------------------------------------------

# Diverse permissive pools: solo, piano, voice+piano lieder, string quartets,
# 4-voice Bach chorales, Polish romantic/classical, and MuseScore big500/mscx-5000.
DEFAULT_SCORE_POOLS: Tuple[Tuple[str, Path, Tuple[str, ...]], ...] = (
    ("big500", _REPO_ROOT / "data-gen" / "score-corpus" / "big500", (".mscx",)),
    ("mscx5000", _REPO_ROOT / "data-gen" / "score-corpus" / "mscx-5000", (".mscx", ".mscz")),
    ("sq_stage", _REPO_ROOT / "data-gen" / "score-corpus" / "sq-stage-in", (".mscx",)),
    ("openscore_lieder", Path(r"D:\tmp\openscore-lieder"), (".mscx",)),
    ("openscore_sq", Path(r"D:\tmp\openscore-sq-repo"), (".mscx",)),
    ("bach_chorales", Path(r"D:\tmp\bach-chorales-fb"), (".musicxml", ".mscx")),
    ("nifc_polish", Path(r"D:\tmp\nifc-polish-hw\xml"), (".musicxml",)),
    ("pdm_bach_chopin", Path(r"D:\tmp\pdm-bach-chopin-hw\xml"), (".musicxml",)),
    ("pdm_openscore", Path(r"D:\tmp\pdm-openscore-hw\xml"), (".musicxml",)),
    ("pdm_openscore_sq", Path(r"D:\tmp\pdm-openscore-sq-hw\xml"), (".musicxml",)),
)


def discover_score_pool(
    raw_slugs: Set[str],
    norm_tokens: Set[str],
    gt_sha256s: Set[str],
    seed: int = 20261124,
) -> Tuple[List[Tuple[str, Path]], int]:
    """Discover deduplicated, leak-free score files balanced across all pools."""
    rng = np.random.default_rng(seed)
    by_pool: Dict[str, List[Path]] = {}
    blocked_count = 0
    seen_stems: Set[str] = set()

    for pool_name, root, exts in DEFAULT_SCORE_POOLS:
        if not root.exists():
            continue
        ext_set = {e.lower() for e in exts}
        found: List[Path] = []
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in ext_set:
                continue
            # Skip tiny/empty files
            try:
                if p.stat().st_size < 1024:
                    continue
            except OSError:
                continue
            # Fast path-based check first, then SHA check
            if is_eval_blocked(p, raw_slugs, norm_tokens, gt_sha256s, check_sha=False):
                blocked_count += 1
                continue
            dedup_key = f"{pool_name}:{_norm_key(p.stem)}"
            if dedup_key in seen_stems:
                continue
            seen_stems.add(dedup_key)
            found.append(p)
        if found:
            idx = rng.permutation(len(found))
            by_pool[pool_name] = [found[int(i)] for i in idx]

    # Round-robin interleave across pools so every musical genre is well represented
    interleaved: List[Tuple[str, Path]] = []
    pool_names = sorted(by_pool.keys())
    cursors = {k: 0 for k in pool_names}
    while True:
        progressed = False
        for k in pool_names:
            c = cursors[k]
            if c < len(by_pool[k]):
                interleaved.append((k, by_pool[k][c]))
                cursors[k] = c + 1
                progressed = True
        if not progressed:
            break

    return interleaved, blocked_count


# Target staff-count bucket weights for --dense-hw-mode:
# ~45% 4-staff (SQ/SATB), ~25% 1-2 staff (piano-prioritized), ~25% 3-staff, ~5% 5-6 staff.
DENSE_HW_BUCKET_WEIGHTS: Tuple[Tuple[str, float], ...] = (
    ("4", 0.45),
    ("1-2", 0.25),
    ("3", 0.25),
    ("5-6", 0.05),
)


def _fast_part_count(path: Path) -> int:
    """Fast header-only staff/part counter matching `hw_score_sim.parse_score(path).parts`."""
    suf = path.suffix.lower()
    try:
        if suf in (".mscz", ".mscx"):
            if suf == ".mscz":
                with zipfile.ZipFile(path, "r") as zf:
                    mscx = next((n for n in zf.namelist() if n.lower().endswith(".mscx")), None)
                    if not mscx:
                        return 0
                    with zf.open(mscx) as fh:
                        head = fh.read(65536).decode("utf-8", errors="ignore")
            else:
                with open(path, "rb") as fh:
                    head = fh.read(65536).decode("utf-8", errors="ignore")
            m_pos = head.find("<Measure")
            hdr = head[:m_pos] if m_pos > 0 else head
            ids = set(re.findall(r'<Staff\s+id=["\'](\d+)["\']', hdr))
            if not ids and m_pos < 0:
                # Fallback if header exceeded 64 KB
                if suf == ".mscz":
                    with zipfile.ZipFile(path, "r") as zf:
                        mscx = next((n for n in zf.namelist() if n.lower().endswith(".mscx")), None)
                        full = zf.read(mscx).decode("utf-8", errors="ignore") if mscx else ""
                else:
                    full = path.read_text(encoding="utf-8", errors="ignore")
                m_pos = full.find("<Measure")
                hdr = full[:m_pos] if m_pos > 0 else full
                ids = set(re.findall(r'<Staff\s+id=["\'](\d+)["\']', hdr))
            return len(ids)
        if suf == ".musicxml":
            with open(path, "rb") as fh:
                head = fh.read(65536).decode("utf-8", errors="ignore")
            n_sp = len(re.findall(r"<score-part\s+id=", head))
            if n_sp == 0:
                return 0
            parts = re.split(r"<part\s+id=", head)[1:]
            if len(parts) < n_sp:
                full = path.read_text(encoding="utf-8", errors="ignore")
                parts = re.split(r"<part\s+id=", full)[1:]
            total = 0
            for pb in parts:
                m = re.search(r"<staves>(\d+)</staves>", pb[:4096])
                total += int(m.group(1)) if m else 1
            return total
    except Exception:
        return 0
    return 0


def _part_count_to_bucket(n_parts: int) -> Optional[str]:
    if n_parts in (1, 2):
        return "1-2"
    if n_parts == 3:
        return "3"
    if n_parts == 4:
        return "4"
    if n_parts in (5, 6):
        return "5-6"
    return None


def bucket_pool_by_parts(
    pool: Sequence[Tuple[str, Path]],
    max_parts: int,
    seed: int = 20261124,
) -> Dict[str, List[Tuple[str, Path]]]:
    """Bucket discovered leak-free scores by part count (`1-2`, `3`, `4`, `5-6`).

    Within bucket `1-2`, 2-staff piano scores are prioritized (~85%) over 1-staff solo (~15%)
    whenever both are available.
    """
    rng = np.random.default_rng(seed + 313)
    raw_by_n: Dict[int, List[Tuple[str, Path]]] = {}
    for item in pool:
        n = _fast_part_count(item[1])
        if 1 <= n <= max_parts:
            raw_by_n.setdefault(n, []).append(item)

    def _shuffle(items: List[Tuple[str, Path]]) -> List[Tuple[str, Path]]:
        if len(items) <= 1:
            return list(items)
        idx = rng.permutation(len(items))
        return [items[int(i)] for i in idx]

    buckets: Dict[str, List[Tuple[str, Path]]] = {}
    # Bucket 1-2: interleave 2-staff piano (~85%) with 1-staff solo (~15%)
    s2 = _shuffle(raw_by_n.get(2, []))
    s1 = _shuffle(raw_by_n.get(1, []))
    if s2 and s1:
        b12: List[Tuple[str, Path]] = []
        i2, i1 = 0, 0
        # Build a balanced list where ~5 out of every 6 entries are 2-staff piano
        target_len = max(len(s2), min(len(s1), len(s2) * 2))
        while len(b12) < target_len:
            for _ in range(5):
                b12.append(s2[i2 % len(s2)])
                i2 += 1
                if len(b12) >= target_len:
                    break
            if len(b12) < target_len:
                b12.append(s1[i1 % len(s1)])
                i1 += 1
        buckets["1-2"] = b12
    elif s2:
        buckets["1-2"] = s2
    elif s1:
        buckets["1-2"] = s1

    if raw_by_n.get(3):
        buckets["3"] = _shuffle(raw_by_n[3])
    if raw_by_n.get(4):
        buckets["4"] = _shuffle(raw_by_n[4])
    b56 = raw_by_n.get(5, []) + raw_by_n.get(6, [])
    if b56:
        buckets["5-6"] = _shuffle(b56)
    return buckets


def _allocate_bucket_counts(
    target_count: int,
    available_buckets: Sequence[str],
) -> Dict[str, int]:
    """Largest-remainder allocation of `target_count` slots across available dense-hw buckets."""
    active = [(b, w) for b, w in DENSE_HW_BUCKET_WEIGHTS if b in available_buckets]
    if not active:
        return {}
    w_sum = sum(w for _, w in active)
    exact = [(b, target_count * (w / w_sum)) for b, w in active]
    counts = {b: int(val) for b, val in exact}
    rem = target_count - sum(counts.values())
    remainders = sorted(exact, key=lambda kv: kv[1] - int(kv[1]), reverse=True)
    for i in range(rem):
        counts[remainders[i % len(remainders)][0]] += 1
    return counts


# ---------------------------------------------------------------------------
# Stage-1 Bezier -> Exact manifest.jsonl row conversion
# ---------------------------------------------------------------------------

def _eval_cubic_bezier_y_vec(ctrl_pts: Sequence[Sequence[float]], xs: np.ndarray) -> np.ndarray:
    x0 = float(ctrl_pts[0][0])
    x3 = float(ctrl_pts[-1][0])
    if abs(x3 - x0) < 1e-6:
        return np.full_like(xs, float(ctrl_pts[0][1]), dtype=np.float64)
    t = np.clip((xs - x0) / (x3 - x0), 0.0, 1.0)
    if len(ctrl_pts) == 4:
        mt = 1.0 - t
        return (
            (mt ** 3) * float(ctrl_pts[0][1])
            + 3.0 * (mt ** 2) * t * float(ctrl_pts[1][1])
            + 3.0 * mt * (t ** 2) * float(ctrl_pts[2][1])
            + (t ** 3) * float(ctrl_pts[3][1])
        )
    return (1.0 - t) * float(ctrl_pts[0][1]) + t * float(ctrl_pts[-1][1])


def gt_payload_to_stage1_row(gt_payload: Dict[str, Any], page_png: Path, n_pts: int = 32) -> Dict[str, Any]:
    r"""Convert `render_score_handwritten` GT JSON into exact `D:\tmp\staff-unet-data\manifest.jsonl` schema."""
    staves_out: List[Dict[str, Any]] = []
    sys_counts: Dict[int, int] = {}

    for sys_obj in gt_payload.get("systems", []):
        sys_id = int(sys_obj.get("system_index", 0))
        for st in sys_obj.get("staves", []):
            x0 = float(st["x_left"])
            x1 = float(st["x_right"])
            sp = float(st["sp"])
            ctrl_pts = st.get("ctrl_pts") or [[x0, float(st["y_center"])], [x1, float(st["y_center"])]]
            xs = np.linspace(x0, x1, num=n_pts, dtype=np.float64)
            y_mid = _eval_cubic_bezier_y_vec(ctrl_pts, xs)
            lines_5xN = [
                [round(float(v), 2) for v in (y_mid + li * sp)]
                for li in (-2, -1, 0, 1, 2)
            ]
            staves_out.append(
                {
                    "lines": lines_5xN,
                    "x0": round(x0, 2),
                    "x1": round(x1, 2),
                    "system_id": sys_id,
                }
            )
            sys_counts[sys_id] = sys_counts.get(sys_id, 0) + 1

    img_str = str(page_png.resolve())
    return {
        "image": img_str,
        "img": img_str,
        "w": int(gt_payload["page_w"]),
        "h": int(gt_payload["page_h"]),
        "staves": staves_out,
        "n_systems": len(sys_counts),
        "max_staves_per_system": max(sys_counts.values()) if sys_counts else 0,
        "source": "hw_sim",
    }


# ---------------------------------------------------------------------------
# Worker: render 1 page + export & validate Stage-2 strips
# ---------------------------------------------------------------------------

PAGE_SIZES: Tuple[Tuple[int, int], ...] = (
    (1650, 2200),
    (1654, 2339),
    (1700, 2200),
)

LANDSCAPE_PAGE_SIZES: Tuple[Tuple[int, int], ...] = (
    (2200, 1650),
    (2339, 1654),
    (2200, 1550),
)

DENSE_PAGE_SIZES: Tuple[Tuple[int, int], ...] = (
    (1900, 2600),
    (1950, 2650),
    (2000, 2700),
)

DENSE_LANDSCAPE_PAGE_SIZES: Tuple[Tuple[int, int], ...] = (
    (2600, 1900),
    (2650, 1950),
    (2700, 2000),
)


def _apply_dense_hw_style_overrides(
    style_dict: Dict[str, Any],
    n_parts: int,
    is_landscape: bool,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Calibrate staff spacing & margins so pages naturally pack 4-6 systems and 20-45 bars/page."""
    out = dict(style_dict)
    out["margin_top_frac"] = float(rng.uniform(0.045, 0.065))
    out["margin_bottom_frac"] = float(rng.uniform(0.040, 0.060))
    out["margin_left_frac"] = float(rng.uniform(0.040, 0.062))
    out["margin_right_frac"] = float(rng.uniform(0.035, 0.055))
    out["blank_prerule_prob"] = 0.0
    # Avoid large decorative title header eating a full system on dense pages
    if str(out.get("page_furniture_mode", "none")).lower() != "none":
        out["page_furniture_mode"] = "none"

    if n_parts <= 1:
        if is_landscape:
            out["sp"] = 12.5
            out["sp_scale_mult"] = float(rng.uniform(0.80, 0.96))
            out["inter_system_gap_mult"] = float(rng.uniform(0.68, 0.85))
        else:
            out["sp"] = 14.0
            out["sp_scale_mult"] = float(rng.uniform(0.85, 1.05))
            out["inter_system_gap_mult"] = float(rng.uniform(0.75, 0.95))
    elif n_parts == 2:
        if is_landscape:
            out["sp"] = 13.5
            out["sp_scale_mult"] = float(rng.uniform(0.82, 0.96))
            out["intra_gap_mult"] = float(rng.uniform(0.68, 0.82))
            out["inter_system_gap_mult"] = float(rng.uniform(0.68, 0.84))
        else:
            out["sp"] = 15.0
            out["sp_scale_mult"] = float(rng.uniform(0.92, 1.08))
            out["intra_gap_mult"] = float(rng.uniform(0.74, 0.88))
            out["inter_system_gap_mult"] = float(rng.uniform(0.76, 0.94))
    elif n_parts == 3:
        if is_landscape:
            out["sp"] = 10.5
            out["sp_scale_mult"] = float(rng.uniform(0.82, 0.96))
            out["intra_gap_mult"] = float(rng.uniform(0.58, 0.72))
            out["inter_system_gap_mult"] = float(rng.uniform(0.55, 0.68))
        else:
            out["sp"] = 12.2
            out["sp_scale_mult"] = float(rng.uniform(0.88, 1.04))
            out["intra_gap_mult"] = float(rng.uniform(0.65, 0.78))
            out["inter_system_gap_mult"] = float(rng.uniform(0.62, 0.76))
    elif n_parts == 4:
        if is_landscape:
            out["sp"] = 9.2
            out["sp_scale_mult"] = float(rng.uniform(0.87, 0.98))
            out["intra_gap_mult"] = float(rng.uniform(0.56, 0.66))
            out["inter_system_gap_mult"] = float(rng.uniform(0.50, 0.62))
        else:
            out["sp"] = 11.2
            out["sp_scale_mult"] = float(rng.uniform(0.86, 1.00))
            out["intra_gap_mult"] = float(rng.uniform(0.62, 0.74))
            out["inter_system_gap_mult"] = float(rng.uniform(0.58, 0.72))
    else:
        if is_landscape:
            out["sp"] = 8.5
            out["sp_scale_mult"] = float(rng.uniform(0.94, 1.00))
            out["intra_gap_mult"] = float(rng.uniform(0.52, 0.60))
            out["inter_system_gap_mult"] = float(rng.uniform(0.45, 0.55))
        else:
            out["sp"] = 9.6
            out["sp_scale_mult"] = float(rng.uniform(0.85, 0.98))
            out["intra_gap_mult"] = float(rng.uniform(0.56, 0.66))
            out["inter_system_gap_mult"] = float(rng.uniform(0.52, 0.64))
    return out


def _validate_strip_bekern(
    bek_path: Path,
    expected_n_meas: int,
    mvd_data: Any,
) -> Tuple[bool, str, int, int]:
    """Strict inline hard gate: verify 0 OOV against closed VOCAB (237) and n_bar_toks == expected_n_meas."""
    try:
        toks = mvd_data.load_bekern(bek_path)
    except Exception as exc:
        return False, f"load_bekern_failed: {exc}", 0, 0
    if len(toks) < 4:
        return False, "too_few_tokens", 0, len(toks)
    try:
        mvd_data.encode(toks)
    except Exception as exc:
        return False, f"oov_token: {exc}", 0, len(toks)
    n_bar_toks = sum(1 for t in toks if t == "bar")
    if n_bar_toks != expected_n_meas:
        return False, f"barline_mismatch:{n_bar_toks}!={expected_n_meas}", n_bar_toks, len(toks)
    return True, "ok", n_bar_toks, len(toks)


def _worker_render_one(job: Dict[str, Any]) -> Dict[str, Any]:
    """Render one handwritten page, enforce barline + 0-OOV gates on every system, and return metadata."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    if str(_MVD_DIR) not in sys.path:
        sys.path.insert(0, str(_MVD_DIR))

    import data as mvd_data
    from hw_score_sim import parse_score, render_score_handwritten, sample_random_scribe, styles as hw_styles, renderer as hw_renderer

    # Point hw-score-sim to this repo's assembly-serialize binary so per-system bekern
    # Stage-2 strips are always generated (even when hw-score-sim is installed from PyPI).
    if not getattr(hw_renderer._find_assembly_serialize_bin, "_repo_patched", False):
        _cand_bins = (
            _REPO_ROOT / "converter" / "assembly-serialize" / "target" / "release" / "assembly-serialize.exe",
            _REPO_ROOT / "converter" / "assembly-serialize" / "target" / "release" / "assembly-serialize",
            _REPO_ROOT / "converter" / "assembly-serialize" / "target" / "debug" / "assembly-serialize.exe",
            _REPO_ROOT / "converter" / "assembly-serialize" / "target" / "debug" / "assembly-serialize",
        )
        _found_bin = next((p for p in _cand_bins if p.exists()), None)
        if _found_bin is not None:
            def _repo_find_bin(_p: Path = _found_bin) -> Optional[Path]:
                return _p
            _repo_find_bin._repo_patched = True  # type: ignore[attr-defined]
            hw_renderer._find_assembly_serialize_bin = _repo_find_bin

    # Allow passing a continuous sampled style dict directly to render_score_handwritten
    if not getattr(hw_styles.get_style, "_dict_passthrough", False):
        _orig_get_style = hw_styles.get_style

        def _get_style_or_dict(name_or_dict: Any) -> Dict[str, Any]:
            if isinstance(name_or_dict, dict):
                return dict(name_or_dict)
            return _orig_get_style(name_or_dict)

        _get_style_or_dict._dict_passthrough = True  # type: ignore[attr-defined]
        hw_styles.get_style = _get_style_or_dict

    idx = int(job["idx"])
    stem = str(job["stem"])
    score_path = Path(job["score_path"])
    pool_name = str(job["pool_name"])
    mode = str(job["mode"])  # "named" or "random_scribe"
    style_name_arg = job["style_name"]
    seed = int(job["seed"])
    page_w, page_h = int(job["page_w"]), int(job["page_h"])
    is_landscape = bool(page_w > page_h)
    clean_paper = bool(job.get("clean_paper", False))
    dense_hw_mode = bool(job.get("dense_hw_mode", False))
    historical_clean = bool(job.get("historical_clean", False))
    omit_cont_header_prob = job.get("omit_cont_header_prob", None)
    max_page_tokens = int(job.get("max_page_tokens", 0))
    strict_fullpage = bool(job.get("strict_fullpage", False))
    max_parts = int(job.get("max_parts", 6))
    pages_dir = Path(job["pages_dir"])
    strips_dir = Path(job["strips_dir"])
    tmp_export_dir = Path(job["tmp_export_dir"])

    page_png = pages_dir / f"{stem}.png"
    page_json = pages_dir / f"{stem}.json"

    # Resume check: if page PNG + JSON already exist and are valid, re-validate strips
    if page_png.exists() and page_json.exists() and page_png.stat().st_size > 1000:
        try:
            gt_payload = json.loads(page_json.read_text(encoding="utf-8"))
            nonblank_sys = [s for s in gt_payload.get("systems", []) if not s.get("is_blank_prerule")]
            if nonblank_sys:
                s1_row = gt_payload_to_stage1_row(gt_payload, page_png)
                valid_beks: List[str] = []
                n_bar_ok = 0
                n_bar_fail = 0
                n_oov_fail = 0
                n_bars_page = 0
                n_tokens_page = 0
                for sys_obj in nonblank_sys:
                    k = int(sys_obj["system_index"])
                    n_meas = int(sys_obj["measure_end"]) - int(sys_obj["measure_start"])
                    n_bars_page += n_meas
                    bek_p = strips_dir / f"{stem}_s{k}-bekern.txt"
                    png_p = strips_dir / f"{stem}_s{k}.png"
                    if bek_p.exists() and png_p.exists():
                        ok_s, reason, _, n_toks = _validate_strip_bekern(bek_p, n_meas, mvd_data)
                        if ok_s:
                            valid_beks.append(str(bek_p.resolve()))
                            n_bar_ok += 1
                            n_tokens_page += n_toks
                        elif reason.startswith("barline_mismatch"):
                            n_bar_fail += 1
                        elif reason.startswith("oov_token"):
                            n_oov_fail += 1
                fullpage_ok = len(valid_beks) == len(nonblank_sys) and len(nonblank_sys) > 0
                fullpage_row = (
                    {
                        "image": str(page_png.resolve()),
                        "bekerns": valid_beks,
                        "domain": "hw",
                        "n_systems": len(nonblank_sys),
                        "n_staves": len(s1_row["staves"]),
                    }
                    if fullpage_ok
                    else None
                )
                return {
                    "ok": True,
                    "resumed": True,
                    "idx": idx,
                    "stem": stem,
                    "pool_name": pool_name,
                    "score_path": str(score_path),
                    "mode": mode,
                    "style": gt_payload.get("style", str(style_name_arg)),
                    "clean_paper": bool(gt_payload.get("clean_paper", clean_paper)),
                    "is_landscape": bool(int(gt_payload.get("page_w", page_w)) > int(gt_payload.get("page_h", page_h))),
                    "page_w": int(gt_payload.get("page_w", page_w)),
                    "page_h": int(gt_payload.get("page_h", page_h)),
                    "n_parts": len(nonblank_sys[0].get("staves", [])),
                    "n_systems": len(nonblank_sys),
                    "n_staves": len(s1_row["staves"]),
                    "n_bars": n_bars_page,
                    "n_tokens": n_tokens_page,
                    "n_strips_ok": len(valid_beks),
                    "n_strips_rejected": len(nonblank_sys) - len(valid_beks),
                    "n_bar_ok": n_bar_ok,
                    "n_bar_fail": n_bar_fail,
                    "n_oov_fail": n_oov_fail,
                    "fullpage_ok": fullpage_ok,
                    "fullpage_row": fullpage_row,
                    "stage1_row": s1_row,
                }
        except Exception:
            pass

    # Pre-check score parseability and part count (<= max_parts so Stage-2 bekern matches rendered staves 1:1)
    try:
        parsed = parse_score(score_path)
        if not parsed.parts or parsed.num_measures < 2:
            return {"ok": False, "idx": idx, "stem": stem, "error": "empty_or_too_short"}
        if len(parsed.parts) > max_parts:
            return {"ok": False, "idx": idx, "stem": stem, "error": f"too_many_parts:{len(parsed.parts)}>{max_parts}"}
        n_parts = len(parsed.parts)
    except Exception as exc:
        return {"ok": False, "idx": idx, "stem": stem, "error": f"parse_failed: {exc}"}

    # Resolve style: named or sample_random_scribe(rng, jitter=0.25)
    np_rng = np.random.default_rng(seed)
    if mode == "random_scribe":
        style_arg: Any = sample_random_scribe(np_rng, jitter=0.25)
    else:
        style_arg = str(style_name_arg)

    if dense_hw_mode:
        resolved = dict(hw_styles.get_style(style_arg))
        if isinstance(style_arg, str) and not resolved.get("name"):
            resolved["name"] = style_arg
        style_arg = _apply_dense_hw_style_overrides(resolved, n_parts=n_parts, is_landscape=is_landscape, rng=np_rng)

    work_strip_dir = tmp_export_dir / stem
    if work_strip_dir.exists():
        shutil.rmtree(work_strip_dir, ignore_errors=True)

    try:
        gt_payload = render_score_handwritten(
            score_path=score_path,
            style_name=style_arg,
            out_png=page_png,
            seed=seed,
            page_w=page_w,
            page_h=page_h,
            export_train_dir=work_strip_dir,
            clean_paper=clean_paper,
            max_parts=max_parts,
            historical_clean=historical_clean,
            omit_cont_header_prob=omit_cont_header_prob,
        )
    except Exception as exc:
        if work_strip_dir.exists():
            shutil.rmtree(work_strip_dir, ignore_errors=True)
        for p in (page_png, page_json):
            if p.exists():
                p.unlink(missing_ok=True)
        return {"ok": False, "idx": idx, "stem": stem, "error": f"render_failed: {exc}"}

    # Validate we actually drew staves
    s1_row = gt_payload_to_stage1_row(gt_payload, page_png)
    nonblank_sys = [s for s in gt_payload.get("systems", []) if not s.get("is_blank_prerule")]
    if not s1_row["staves"] or not nonblank_sys:
        if work_strip_dir.exists():
            shutil.rmtree(work_strip_dir, ignore_errors=True)
        for p in (page_png, page_json):
            if p.exists():
                p.unlink(missing_ok=True)
        return {"ok": False, "idx": idx, "stem": stem, "error": "zero_staves"}

    # Map system_index -> expected n_meas from GT layout
    expected_meas_by_sys: Dict[int, int] = {
        int(s["system_index"]): int(s["measure_end"]) - int(s["measure_start"])
        for s in nonblank_sys
    }
    n_bars_page = sum(expected_meas_by_sys.values())
    exports_by_sys: Dict[int, Dict[str, Any]] = {
        int(exp["system_index"]): exp for exp in gt_payload.get("train_exports", [])
    }

    # Move & validate Stage-2 strips with strict barline + 0-OOV gates
    n_strips_ok = 0
    n_strips_rejected = 0
    n_bar_ok = 0
    n_bar_fail = 0
    n_oov_fail = 0
    n_tokens_page = 0
    ordered_valid_beks: List[str] = []

    if work_strip_dir.exists():
        strips_dir.mkdir(parents=True, exist_ok=True)
        for sys_obj in nonblank_sys:
            k = int(sys_obj["system_index"])
            n_meas = expected_meas_by_sys[k]
            exp = exports_by_sys.get(k)
            if not exp or not exp.get("bekern_ok") or not exp.get("bekern"):
                n_strips_rejected += 1
                continue
            src_png = Path(exp["image"])
            src_bek = Path(exp["bekern"])
            if not src_png.exists() or not src_bek.exists():
                n_strips_rejected += 1
                continue

            ok_s, reason, _, n_toks = _validate_strip_bekern(src_bek, n_meas, mvd_data)
            if not ok_s:
                n_strips_rejected += 1
                if reason.startswith("barline_mismatch"):
                    n_bar_fail += 1
                elif reason.startswith("oov_token"):
                    n_oov_fail += 1
                continue

            dst_png = strips_dir / f"{stem}_s{k}.png"
            dst_bek = strips_dir / f"{stem}_s{k}-bekern.txt"
            shutil.move(str(src_png), str(dst_png))
            shutil.move(str(src_bek), str(dst_bek))
            ordered_valid_beks.append(str(dst_bek.resolve()))
            n_strips_ok += 1
            n_bar_ok += 1
            n_tokens_page += n_toks

        shutil.rmtree(work_strip_dir, ignore_errors=True)

    fullpage_ok = (len(ordered_valid_beks) == len(nonblank_sys)) and (len(nonblank_sys) > 0)
    if max_page_tokens > 0 and n_tokens_page > max_page_tokens:
        fullpage_ok = False
    if strict_fullpage and not fullpage_ok:
        for p in (page_png, page_json):
            if p.exists():
                p.unlink(missing_ok=True)
        for bp in ordered_valid_beks:
            bp_p = Path(bp)
            bp_p.unlink(missing_ok=True)
            sp_p = bp_p.with_name(bp_p.name.replace("-bekern.txt", ".png"))
            sp_p.unlink(missing_ok=True)
        err_msg = (
            f"too_many_tokens:{n_tokens_page}>{max_page_tokens}"
            if (max_page_tokens > 0 and n_tokens_page > max_page_tokens)
            else f"incomplete_systems:{len(ordered_valid_beks)}/{len(nonblank_sys)}"
        )
        return {"ok": False, "idx": idx, "stem": stem, "error": err_msg}
    fullpage_row = (
        {
            "image": str(page_png.resolve()),
            "bekerns": ordered_valid_beks,
            "domain": "hw",
            "n_systems": len(nonblank_sys),
            "n_staves": len(s1_row["staves"]),
            "n_tokens": n_tokens_page,
            "omit_cont_header": bool(gt_payload.get("omit_cont_header", False)),
        }
        if fullpage_ok
        else None
    )

    return {
        "ok": True,
        "resumed": False,
        "idx": idx,
        "stem": stem,
        "pool_name": pool_name,
        "score_path": str(score_path),
        "mode": mode,
        "style": gt_payload.get("style", str(style_name_arg)),
        "clean_paper": clean_paper,
        "is_landscape": is_landscape,
        "page_w": page_w,
        "page_h": page_h,
        "n_parts": n_parts,
        "n_systems": len(nonblank_sys),
        "n_staves": len(s1_row["staves"]),
        "n_bars": n_bars_page,
        "n_tokens": n_tokens_page,
        "n_strips_ok": n_strips_ok,
        "n_strips_rejected": n_strips_rejected,
        "n_bar_ok": n_bar_ok,
        "n_bar_fail": n_bar_fail,
        "n_oov_fail": n_oov_fail,
        "fullpage_ok": fullpage_ok,
        "fullpage_row": fullpage_row,
        "stage1_row": s1_row,
    }


# ---------------------------------------------------------------------------
# Main Orchestration + Verification
# ---------------------------------------------------------------------------

def run_verification(
    stage1_manifest: Path,
    stage2_strips_dir: Path,
    selected_sources: Sequence[Path],
    raw_slugs: Set[str],
    norm_tokens: Set[str],
    gt_sha256s: Set[str],
) -> Dict[str, Any]:
    """Run Stage-1 StaffUNetDataset(augment=True), Stage-2 mvd/data.py, and eval_overlap checks."""
    import data as mvd_data
    import staff_unet2d

    # 1. Eval overlap check (path + SHA-256)
    overlap_hits = [
        str(p) for p in selected_sources
        if is_eval_blocked(p, raw_slugs, norm_tokens, gt_sha256s, check_sha=True)
    ]

    # 2. Stage-1 StaffUNetDataset(..., augment=True) verification on 3 samples
    ds = staff_unet2d.StaffUNetDataset(stage1_manifest, augment=True, seed=1234, clean_frac=0.0)
    s1_checks: List[Dict[str, Any]] = []
    n_check_s1 = min(3, len(ds))
    for i in range(n_check_s1):
        img_t, tgt_t = ds[i]
        assert img_t.ndim == 3 and img_t.shape[0] == 3, f"bad img shape {img_t.shape}"
        assert tgt_t.ndim == 3 and tgt_t.shape[0] == 4, f"bad tgt shape {tgt_t.shape}"
        assert bool(np.isfinite(img_t.numpy()).all()), "non-finite img"
        assert bool(np.isfinite(tgt_t.numpy()).all()), "non-finite tgt"
        assert float(tgt_t[0].max()) > 0.5, f"empty staff-line target on sample {i}"
        s1_checks.append(
            {
                "idx": i,
                "img_shape": list(img_t.shape),
                "tgt_shape": list(tgt_t.shape),
                "ch0_max": round(float(tgt_t[0].max()), 4),
                "ch1_max": round(float(tgt_t[1].max()), 4),
                "ch2_max": round(float(tgt_t[2].max()), 4),
                "ch3_max": round(float(tgt_t[3].max()), 4),
            }
        )

    # 3. Stage-2 mvd/data.py load_item + encode verification on 3 strips
    bek_files = sorted(stage2_strips_dir.glob("*-bekern.txt"))
    s2_checks: List[Dict[str, Any]] = []
    for bek in bek_files[:3]:
        stem = bek.name[: -len("-bekern.txt")]
        png = stage2_strips_dir / f"{stem}.png"
        assert png.exists(), f"missing strip png {png}"
        item = mvd_data.load_item(png, bek)
        ids = mvd_data.encode(item["tokens"])
        assert item["image"].ndim == 3 and item["image"].shape[2] == 3
        assert len(ids) == len(item["tokens"]) and len(ids) > 0
        s2_checks.append(
            {
                "page_id": item["page_id"],
                "img_shape": list(item["image"].shape),
                "n_tokens": len(ids),
            }
        )

    # 4. Full-page manifest verification (100% barline match + 0 OOV across all rows)
    fullpage_manifest = stage1_manifest.parent / "fullpage_manifest.jsonl"
    fp_rows = 0
    fp_strips = 0
    fp_oov = 0
    if fullpage_manifest.exists():
        with open(fullpage_manifest, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                fp_rows += 1
                assert Path(row["image"]).exists(), f"missing fullpage image {row['image']}"
                beks = row.get("bekerns", [])
                assert len(beks) == int(row["n_systems"]) and len(beks) > 0
                for bp in beks:
                    toks = mvd_data.load_bekern(bp)
                    try:
                        mvd_data.encode(toks)
                    except Exception:
                        fp_oov += 1
                    fp_strips += 1

    return {
        "eval_overlap": len(overlap_hits),
        "overlap_hits": overlap_hits[:10],
        "stage1_dataset_len": len(ds),
        "stage1_augment_checks": s1_checks,
        "stage2_strip_count": len(bek_files),
        "stage2_loader_checks": s2_checks,
        "fullpage_manifest_rows": fp_rows,
        "fullpage_manifest_strips": fp_strips,
        "fullpage_oov_count": fp_oov,
        "vocab_size": len(mvd_data.VOCAB),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate leak-free simulated handwritten Stage-1, Stage-2 & full-page UniOMR training data."
    )
    parser.add_argument("--count", type=int, default=600, help="Target number of full handwritten pages")
    parser.add_argument("--workers", type=int, default=6, help="Parallel worker processes (default 6)")
    parser.add_argument("--out-dir", default=r"D:\tmp\hw-sim-train", help="Output root directory")
    parser.add_argument(
        "--base-stage1-manifest",
        default=r"D:\tmp\staff-unet-data\manifest.jsonl",
        help="Existing Stage-1 manifest to merge non-destructively",
    )
    parser.add_argument(
        "--merged-stage1-manifest",
        default=r"D:\tmp\staff-unet-data\manifest_with_hwsim.jsonl",
        help="Output path for merged Stage-1 manifest",
    )
    parser.add_argument("--seed", type=int, default=20261124, help="Base deterministic seed")
    parser.add_argument(
        "--max-parts",
        type=int,
        default=6,
        help="Maximum number of staves/parts per score (default 6)",
    )
    parser.add_argument(
        "--landscape-frac",
        type=float,
        default=None,
        help="Fraction of pages rendered in landscape orientation (default 0.50 in --dense-hw-mode, else 0.45)",
    )
    parser.add_argument(
        "--clean-paper-frac",
        type=float,
        default=None,
        help="Fraction of pages rendered with clean_paper=True vs native historical paper (default 0.30 in --dense-hw-mode, else 0.40)",
    )
    parser.add_argument(
        "--dense-hw-mode",
        action="store_true",
        default=False,
        help="Enable suite31_hw-matched dense layout (4-6 systems, 20-45 bars/page), staff-count rebalancing (45%% 4-staff, 25%% 1-2, 25%% 3, 5%% 5-6), 50%% landscape, and 70%% native historical paper",
    )
    parser.add_argument(
        "--hero-frac",
        type=float,
        default=0.50,
        help="Fraction of pages assigned to the 75 Flagship Hero Scribes (default 0.50)",
    )
    parser.add_argument(
        "--rand-frac",
        type=float,
        default=0.25,
        help="Fraction of pages assigned to continuous random scribes (default 0.25; remainder goes to other named styles)",
    )
    parser.add_argument(
        "--clean-paper",
        action="store_true",
        help="Force clean_paper=True on 100%% of pages (overrides --clean-paper-frac)",
    )
    parser.add_argument(
        "--historical-clean",
        action="store_true",
        help="Enable hw-score-sim curated historical_clean archival manuscript mode",
    )
    parser.add_argument(
        "--omit-cont-header-prob",
        type=float,
        default=None,
        help="Override omit_cont_header_prob (default: 0.35 when --historical-clean, else 0.0)",
    )
    parser.add_argument(
        "--max-page-tokens",
        type=int,
        default=0,
        help="Reject & backfill pages exceeding this total bekern token count (0 = unlimited)",
    )
    parser.add_argument(
        "--strict-fullpage",
        action="store_true",
        help="Reject & backfill any page that fails fullpage_ok (barline/OOV/token cap)",
    )
    parser.add_argument(
        "--no-merge-stage1",
        action="store_true",
        help="Do not touch/overwrite --merged-stage1-manifest",
    )
    parser.add_argument("--verify-only", action="store_true", help="Skip generation and only run verification")
    args = parser.parse_args(argv)

    from hw_score_sim import list_styles
    from hw_score_sim.styles import SCRIBE_STYLES

    out_root = Path(args.out_dir)
    pages_dir = out_root / "pages"
    strips_dir = out_root / "stage2_strips"
    tmp_export_dir = out_root / "_tmp_export"
    stage1_manifest = out_root / "stage1_manifest.jsonl"
    fullpage_manifest = out_root / "fullpage_manifest.jsonl"
    summary_path = out_root / "generation_summary.json"

    pages_dir.mkdir(parents=True, exist_ok=True)
    strips_dir.mkdir(parents=True, exist_ok=True)
    tmp_export_dir.mkdir(parents=True, exist_ok=True)

    raw_slugs, norm_tokens, gt_sha256s = load_eval_blocklist()
    print(
        f"[init] eval blocklist loaded: {len(raw_slugs)} slugs, "
        f"{len(norm_tokens)} norm tokens, {len(gt_sha256s)} GT sha256s",
        flush=True,
    )

    all_style_names = list_styles()
    hero_style_names = [
        k for k in all_style_names
        if "hero" in k.lower() or "hero" in str(SCRIBE_STYLES.get(k, {}).get("pack", "")).lower()
    ]
    other_style_names = [k for k in all_style_names if k not in set(hero_style_names)]
    n_named_styles = len(all_style_names)
    print(
        f"[init] registered named styles: {n_named_styles} "
        f"(hero={len(hero_style_names)}, other={len(other_style_names)})",
        flush=True,
    )

    pool, blocked_count = discover_score_pool(raw_slugs, norm_tokens, gt_sha256s, seed=args.seed)
    print(f"[init] discovered {len(pool)} leak-free candidate scores ({blocked_count} eval-blocked)", flush=True)
    if not pool:
        raise RuntimeError("No candidate scores discovered!")

    if not args.verify_only:
        t0 = time.time()
        target_count = int(args.count)
        max_parts = max(1, int(args.max_parts))
        dense_hw_mode = bool(args.dense_hw_mode)
        default_land = 0.50 if dense_hw_mode else 0.45
        default_clean = 0.30 if dense_hw_mode else 0.40
        landscape_frac = float(np.clip(default_land if args.landscape_frac is None else args.landscape_frac, 0.0, 1.0))
        clean_paper_frac = (
            1.0
            if bool(args.clean_paper)
            else float(np.clip(default_clean if args.clean_paper_frac is None else args.clean_paper_frac, 0.0, 1.0))
        )
        hero_frac = float(np.clip(args.hero_frac, 0.0, 1.0)) if hero_style_names else 0.0
        rand_frac = float(np.clip(args.rand_frac, 0.0, 1.0 - hero_frac))
        n_hero_target = int(round(target_count * hero_frac))
        n_rand_target = int(round(target_count * rand_frac))
        n_other_target = max(0, target_count - n_hero_target - n_rand_target)
        n_land_target = int(round(target_count * landscape_frac))
        n_clean_target = int(round(target_count * clean_paper_frac))
        print(
            f"[init] target mix ({target_count} pages, max_parts={max_parts}, dense_hw_mode={dense_hw_mode}): "
            f"hero={n_hero_target} ({100.0*n_hero_target/max(1,target_count):.1f}%), "
            f"other_named={n_other_target} ({100.0*n_other_target/max(1,target_count):.1f}%), "
            f"random_scribe={n_rand_target} ({100.0*n_rand_target/max(1,target_count):.1f}%), "
            f"landscape={n_land_target} ({100.0*n_land_target/max(1,target_count):.1f}%), "
            f"clean_paper={n_clean_target} ({100.0*n_clean_target/max(1,target_count):.1f}%)",
            flush=True,
        )

        # Build deterministic style, orientation, and paper-mode assignments
        rng = np.random.default_rng(args.seed + 7)

        def _fill_uniform(names: List[str], n: int) -> List[Tuple[str, Optional[str]]]:
            out: List[Tuple[str, Optional[str]]] = []
            if not names or n <= 0:
                return out
            while len(out) < n:
                perm = [names[int(i)] for i in rng.permutation(len(names))]
                for s in perm:
                    if len(out) >= n:
                        break
                    out.append(("named", s))
            return out

        hero_assignments = _fill_uniform(hero_style_names, n_hero_target)
        other_assignments = _fill_uniform(other_style_names or all_style_names, n_other_target)
        rand_assignments: List[Tuple[str, Optional[str]]] = [("random_scribe", None)] * n_rand_target

        # Interleave deterministically across all 3 buckets
        all_assign = hero_assignments + other_assignments + rand_assignments
        perm_all = rng.permutation(len(all_assign))
        interleaved_assign: List[Tuple[str, Optional[str]]] = [all_assign[int(i)] for i in perm_all]

        land_flags = [True] * n_land_target + [False] * (target_count - n_land_target)
        land_flags = [land_flags[int(i)] for i in rng.permutation(target_count)]

        clean_flags = [True] * n_clean_target + [False] * (target_count - n_clean_target)
        clean_flags = [clean_flags[int(i)] for i in rng.permutation(target_count)]

        results_by_idx: Dict[int, Dict[str, Any]] = {}
        pool_cursor = 0
        pool_pass = 0
        pool_order = list(pool)

        # Optional staff-count bucketed pool for --dense-hw-mode
        bucket_pools: Dict[str, List[Tuple[str, Path]]] = {}
        bucket_cursors: Dict[str, int] = {}
        bucket_passes: Dict[str, int] = {}
        slot_buckets: List[Optional[str]] = [None] * target_count
        if dense_hw_mode:
            t_b = time.time()
            bucket_pools = bucket_pool_by_parts(pool, max_parts=max_parts, seed=args.seed)
            bucket_counts = _allocate_bucket_counts(target_count, sorted(bucket_pools.keys()))
            bucket_cursors = {k: 0 for k in bucket_pools}
            bucket_passes = {k: 0 for k in bucket_pools}
            b_slots: List[str] = []
            for b_name, _ in DENSE_HW_BUCKET_WEIGHTS:
                b_slots.extend([b_name] * bucket_counts.get(b_name, 0))
            if len(b_slots) == target_count:
                perm_b = rng.permutation(target_count)
                slot_buckets = [b_slots[int(i)] for i in perm_b]
            print(
                f"[init] dense-hw buckets scanned in {time.time() - t_b:.2f}s: "
                f"available={{{', '.join(f'{k}:{len(v)}' for k, v in sorted(bucket_pools.items()))}}}, "
                f"target_slots={bucket_counts}",
                flush=True,
            )

        # Submit in batches so if any score fails to parse/render, we immediately backfill from the pool
        # until we have exactly `target_count` successful pages. Wrap around pool deterministically
        # when target_count > len(pool) so scores render in fresh handwritten styles.
        pending_slots: List[int] = list(range(target_count))
        failed_attempts = 0
        max_failures = target_count * 5

        def _next_candidate_for_slot(slot_idx: int) -> Optional[Tuple[str, Path, int]]:
            nonlocal pool_cursor, pool_pass, pool_order
            b_key = slot_buckets[slot_idx] if dense_hw_mode else None
            if b_key and b_key in bucket_pools and bucket_pools[b_key]:
                b_list = bucket_pools[b_key]
                attempts = 0
                while attempts < len(b_list) * 2:
                    if bucket_cursors[b_key] >= len(b_list):
                        bucket_passes[b_key] += 1
                        bucket_cursors[b_key] = 0
                        p_rng = np.random.default_rng(args.seed + 1009 * bucket_passes[b_key] + hash(b_key) % 997)
                        perm = p_rng.permutation(len(b_list))
                        bucket_pools[b_key] = [b_list[int(i)] for i in perm]
                        b_list = bucket_pools[b_key]
                    pool_name, score_path = b_list[bucket_cursors[b_key]]
                    bucket_cursors[b_key] += 1
                    attempts += 1
                    if is_eval_blocked(score_path, raw_slugs, norm_tokens, gt_sha256s, check_sha=True):
                        continue
                    return pool_name, score_path, bucket_passes[b_key]
                return None

            attempts = 0
            while attempts < len(pool_order) * 2:
                if pool_cursor >= len(pool_order):
                    pool_pass += 1
                    pool_cursor = 0
                    pass_rng = np.random.default_rng(args.seed + 1009 * pool_pass)
                    perm = pass_rng.permutation(len(pool_order))
                    pool_order = [pool_order[int(i)] for i in perm]
                pool_name, score_path = pool_order[pool_cursor]
                pool_cursor += 1
                attempts += 1
                if is_eval_blocked(score_path, raw_slugs, norm_tokens, gt_sha256s, check_sha=True):
                    continue
                return pool_name, score_path, pool_pass
            return None

        def make_job(slot_idx: int) -> Optional[Dict[str, Any]]:
            cand = _next_candidate_for_slot(slot_idx)
            if cand is None:
                return None
            pool_name, score_path, pass_num = cand
            mode, sname = interleaved_assign[slot_idx]
            if dense_hw_mode:
                sizes = DENSE_LANDSCAPE_PAGE_SIZES if land_flags[slot_idx] else DENSE_PAGE_SIZES
            else:
                sizes = LANDSCAPE_PAGE_SIZES if land_flags[slot_idx] else PAGE_SIZES
            pw, ph = sizes[slot_idx % len(sizes)]
            stem = f"hwsim_{slot_idx:05d}"
            return {
                "idx": slot_idx,
                "stem": stem,
                "score_path": str(score_path),
                "pool_name": pool_name,
                "mode": mode,
                "style_name": sname,
                "seed": args.seed + slot_idx * 97 + failed_attempts * 13 + pass_num * 7919,
                "page_w": pw,
                "page_h": ph,
                "max_parts": max_parts,
                "dense_hw_mode": dense_hw_mode,
                "historical_clean": bool(args.historical_clean),
                "omit_cont_header_prob": args.omit_cont_header_prob,
                "max_page_tokens": int(args.max_page_tokens),
                "strict_fullpage": bool(args.strict_fullpage),
                "pages_dir": str(pages_dir),
                "strips_dir": str(strips_dir),
                "tmp_export_dir": str(tmp_export_dir),
                "clean_paper": bool(clean_flags[slot_idx]),
            }

        with ProcessPoolExecutor(max_workers=int(args.workers), max_tasks_per_child=40) as ex:
            in_flight = {}
            while pending_slots or in_flight:
                while pending_slots and len(in_flight) < int(args.workers) * 2:
                    slot = pending_slots.pop(0)
                    job = make_job(slot)
                    if job is None:
                        break
                    fut = ex.submit(_worker_render_one, job)
                    in_flight[fut] = slot

                if not in_flight:
                    break

                for fut in as_completed(list(in_flight.keys())):
                    slot = in_flight.pop(fut)
                    res = fut.result()
                    if res.get("ok"):
                        results_by_idx[slot] = res
                        done = len(results_by_idx)
                        if done % 25 == 0 or done == target_count:
                            elapsed = time.time() - t0
                            rate = done / max(elapsed, 1e-3)
                            tot_strips = sum(r["n_strips_ok"] for r in results_by_idx.values())
                            fp_ok_cnt = sum(1 for r in results_by_idx.values() if r.get("fullpage_ok"))
                            print(
                                f"[progress] {done}/{target_count} pages "
                                f"(fullpage_ok={fp_ok_cnt}/{done}, {tot_strips} stage2 strips, "
                                f"{failed_attempts} retries, {rate:.2f} pages/s, elapsed {elapsed:.1f}s)",
                                flush=True,
                            )
                    else:
                        failed_attempts += 1
                        if failed_attempts <= max_failures:
                            pending_slots.append(slot)
                    # Refill queue promptly
                    break

        shutil.rmtree(tmp_export_dir, ignore_errors=True)

        # Write Stage-1 manifest.jsonl and UniOMR fullpage_manifest.jsonl in deterministic slot order
        ordered = [results_by_idx[k] for k in sorted(results_by_idx.keys())]
        with open(stage1_manifest, "w", encoding="utf-8") as fh:
            for r in ordered:
                fh.write(json.dumps(r["stage1_row"], ensure_ascii=False) + "\n")

        fullpage_rows = [r["fullpage_row"] for r in ordered if r.get("fullpage_ok") and r.get("fullpage_row")]
        with open(fullpage_manifest, "w", encoding="utf-8") as fh:
            for row in fullpage_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(out_root / "manifest.jsonl", "w", encoding="utf-8") as fh:
            for row in fullpage_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Write non-destructive merged Stage-1 manifest
        merged_manifest = Path(args.merged_stage1_manifest)
        if not args.no_merge_stage1:
            base_manifest = Path(args.base_stage1_manifest)
            merged_manifest.parent.mkdir(parents=True, exist_ok=True)
            base_rows = 0
            with open(merged_manifest, "w", encoding="utf-8") as out_f:
                if base_manifest.exists():
                    with open(base_manifest, "r", encoding="utf-8") as in_f:
                        for line in in_f:
                            if line.strip():
                                out_f.write(line if line.endswith("\n") else line + "\n")
                                base_rows += 1
                for r in ordered:
                    out_f.write(json.dumps(r["stage1_row"], ensure_ascii=False) + "\n")
        else:
            base_rows = 0

        selected_sources = [Path(r["score_path"]) for r in ordered]
        named_used = sorted({r["style"] for r in ordered if r["mode"] == "named"})
        n_named_done = sum(1 for r in ordered if r["mode"] == "named")
        n_rand_done = sum(1 for r in ordered if r["mode"] == "random_scribe")
        n_land_done = sum(1 for r in ordered if r.get("is_landscape"))
        n_clean_done = sum(1 for r in ordered if r.get("clean_paper"))
        tot_bar_ok = sum(int(r.get("n_bar_ok", 0)) for r in ordered)
        tot_bar_fail = sum(int(r.get("n_bar_fail", 0)) for r in ordered)
        tot_oov_fail = sum(int(r.get("n_oov_fail", 0)) for r in ordered)
        pool_hist: Dict[str, int] = {}
        parts_hist: Dict[str, int] = {}
        systems_hist: Dict[str, int] = {}
        bars_list: List[int] = []
        tokens_list: List[int] = []
        sys_list: List[int] = []
        px_per_staff_1280_list: List[float] = []
        for r in ordered:
            pool_hist[r["pool_name"]] = pool_hist.get(r["pool_name"], 0) + 1
            pk = str(r.get("n_parts", 0))
            parts_hist[pk] = parts_hist.get(pk, 0) + 1
            sk = str(r.get("n_systems", 0))
            systems_hist[sk] = systems_hist.get(sk, 0) + 1
            bars_list.append(int(r.get("n_bars", 0)))
            if r.get("fullpage_ok"):
                tokens_list.append(int(r.get("n_tokens", 0)))
            sys_list.append(int(r.get("n_systems", 0)))
            n_st = int(r.get("n_staves", 0))
            pw_r = int(r.get("page_w", 0))
            ph_r = int(r.get("page_h", 0))
            if n_st > 0 and pw_r > 0 and ph_r > 0:
                h_at_1280 = ph_r * (1280.0 / float(pw_r))
                px_per_staff_1280_list.append(h_at_1280 / float(n_st))

        ver = run_verification(
            stage1_manifest=stage1_manifest,
            stage2_strips_dir=strips_dir,
            selected_sources=selected_sources,
            raw_slugs=raw_slugs,
            norm_tokens=norm_tokens,
            gt_sha256s=gt_sha256s,
        )
        assert ver["eval_overlap"] == 0, f"EVAL OVERLAP DETECTED: {ver['overlap_hits']}"
        assert ver["fullpage_oov_count"] == 0, f"OOV TOKENS DETECTED: {ver['fullpage_oov_count']}"

        summary = {
            "dense_hw_mode": dense_hw_mode,
            "max_parts": max_parts,
            "landscape_frac_target": landscape_frac,
            "clean_paper_frac_target": clean_paper_frac,
            "pages_generated": len(ordered),
            "fullpage_ok_pages": len(fullpage_rows),
            "fullpage_survival_rate": round(len(fullpage_rows) / max(1, len(ordered)), 4),
            "landscape_pages": n_land_done,
            "clean_paper_pages": n_clean_done,
            "native_paper_pages": len(ordered) - n_clean_done,
            "named_style_pages": n_named_done,
            "unique_named_styles_used": len(named_used),
            "random_scribe_pages": n_rand_done,
            "failed_retries": failed_attempts,
            "barline_ok_strips": tot_bar_ok,
            "barline_fail_strips": tot_bar_fail,
            "oov_fail_strips": tot_oov_fail,
            "parts_breakdown": parts_hist,
            "systems_breakdown": systems_hist,
            "median_systems_per_page": round(float(np.median(sys_list)), 2) if sys_list else 0.0,
            "median_bars_per_page": round(float(np.median(bars_list)), 2) if bars_list else 0.0,
            "median_tokens_per_page": round(float(np.median(tokens_list)), 1) if tokens_list else 0.0,
            "median_px_per_staff_at_1280": round(float(np.median(px_per_staff_1280_list)), 1)
            if px_per_staff_1280_list
            else 0.0,
            "fullpage_manifest": str(fullpage_manifest),
            "stage1_manifest": str(stage1_manifest),
            "stage1_new_rows": len(ordered),
            "base_stage1_rows": base_rows,
            "merged_stage1_manifest": str(merged_manifest),
            "merged_stage1_rows": base_rows + len(ordered),
            "stage2_strips_dir": str(strips_dir),
            "stage2_strips_total": ver["stage2_strip_count"],
            "pool_breakdown": pool_hist,
            "elapsed_sec": round(time.time() - t0, 1),
            "verification": ver,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("[done] " + json.dumps(summary, indent=2), flush=True)
    else:
        selected_sources = []
        for jf in sorted(pages_dir.glob("*.json")):
            d = json.loads(jf.read_text(encoding="utf-8"))
            if d.get("source_path"):
                selected_sources.append(Path(d["source_path"]))
        ver = run_verification(
            stage1_manifest=stage1_manifest,
            stage2_strips_dir=strips_dir,
            selected_sources=selected_sources,
            raw_slugs=raw_slugs,
            norm_tokens=norm_tokens,
            gt_sha256s=gt_sha256s,
        )
        assert ver["eval_overlap"] == 0, f"EVAL OVERLAP DETECTED: {ver['overlap_hits']}"
        print("[verify-only] " + json.dumps(ver, indent=2), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
