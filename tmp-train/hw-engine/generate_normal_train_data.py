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

# Explicit normalized substring tokens derived from all 45 suite31 + hw eval pieces
# to catch differently-named copies of the exact same compositions across repos.
_EXTRA_EVAL_TOKENS: Tuple[str, ...] = (
    "bwv66",
    "bwv_66",
    "bwv-66",
    "maple_leaf",
    "maple-leaf",
    "mapleleaf",
    "op19",
    "op_19",
    "op-19",
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
    "op18_1",
    "op18-1",
    "op18_no1",
    "op74n1",
    "op74_1",
    "op74-1",
    "op74_no1",
    "mazurka06",
    "op06_no2",
    "op6_no2",
    "op6-2",
    "lindenbaum",
    "dichterliebe",
    "op41n1",
    "op41_1",
    "op41-1",
    "op1n1",
    "op17-3",
    "op17_3",
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
    """Return (raw_slugs, norm_tokens, gt_sha256s) from all suite31 manifests."""
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
    """Return True if `path` matches any eval slug, normalized token, or GT SHA-256."""
    full_str = str(path).lower()
    stem_l = path.stem.lower()
    norm_full = _norm_key(path.parent.name + "_" + path.stem)
    norm_stem = _norm_key(path.stem)

    for slug in raw_slugs:
        if slug in full_str or slug in stem_l:
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


_CLEAN_PAPER_PATCHED = False


def _enable_clean_paper_mode() -> None:
    """Strip all paper texture, tint, vignette, stains, bleed, folds, foxing, blots, smears & wear.

    Keeps 100% of pure handwritten ink, nib physics, 234+random scribe morphologies,
    86 expressive glyphs, rastrum staves, cursive ligatures, ductus/spacing quirks,
    and pure-ink annotations/chords/measure numbers on crisp [1.0, 1.0, 1.0] white paper,
    so downstream online augmentation (epoch_aug / doc-photo-degrade) owns all paper/photo
    effects without double-degrading.
    """
    global _CLEAN_PAPER_PATCHED
    if _CLEAN_PAPER_PATCHED:
        return
    from hw_score_sim import layout_quirks, manuscript_marks, profile_resolver

    def _pure_white_paper(
        width: int,
        height: int,
        style: Dict[str, Any],
        rng: np.random.Generator,
    ) -> np.ndarray:
        return np.ones((int(height), int(width), 3), dtype=np.float32)

    def _noop_pre_ink(
        canvas: Any,
        style: Dict[str, Any],
        rng: np.random.Generator,
        marks_out: List[Dict[str, Any]],
    ) -> None:
        return None

    _orig_resolve = profile_resolver.resolve_page_and_expressive_profile

    def _clean_resolve(raw_style: Dict[str, Any]) -> Dict[str, Any]:
        resolved = dict(_orig_resolve(raw_style))
        resolved.update(
            {
                "paper_rgb": (1.0, 1.0, 1.0),
                "paper_type": "bright_white",
                "paper_grain": 0.0,
                "paper_mottle": 0.0,
                "bleed_through": 0.0,
                "stain_prob": 0.0,
                "eraser_smudge_prob": 0.0,
                "verso_bleed_strength": 0.0,
                "fold_crease_prob": 0.0,
                "foxing_cluster_prob": 0.0,
                "ink_blot_prob": 0.0,
                "palm_smear_prob": 0.0,
                "dog_ear_prob": 0.0,
                "dog_ear_wear_prob": 0.0,
                "paper_border_mode": "none",
                "paper_border_style": "clean",
                "paper_binding_mode": "none",
                "binding_edge": "none",
                "paper_background_grid": "none",
                "paper_crease_fold": "none",
            }
        )
        return resolved

    layout_quirks.make_manuscript_paper = _pure_white_paper
    manuscript_marks.apply_pre_ink_manuscript_layers = _noop_pre_ink
    profile_resolver.resolve_page_and_expressive_profile = _clean_resolve
    _CLEAN_PAPER_PATCHED = True


def _worker_render_one(job: Dict[str, Any]) -> Dict[str, Any]:
    """Render one handwritten page, validate Stage-2 bekern strips, and return metadata."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    if str(_MVD_DIR) not in sys.path:
        sys.path.insert(0, str(_MVD_DIR))

    if job.get("clean_paper"):
        _enable_clean_paper_mode()

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
    # (hw-score-sim 0.3.0's styles.get_style expects a str key).
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
    pages_dir = Path(job["pages_dir"])
    strips_dir = Path(job["strips_dir"])
    tmp_export_dir = Path(job["tmp_export_dir"])

    page_png = pages_dir / f"{stem}.png"
    page_json = pages_dir / f"{stem}.json"

    # Resume check: if page PNG + JSON already exist and are valid, collect existing strips
    if page_png.exists() and page_json.exists() and page_png.stat().st_size > 1000:
        try:
            gt_payload = json.loads(page_json.read_text(encoding="utf-8"))
            if gt_payload.get("systems"):
                existing_strips = sorted(strips_dir.glob(f"{stem}_s*-bekern.txt"))
                s1_row = gt_payload_to_stage1_row(gt_payload, page_png)
                return {
                    "ok": True,
                    "resumed": True,
                    "idx": idx,
                    "stem": stem,
                    "pool_name": pool_name,
                    "score_path": str(score_path),
                    "mode": mode,
                    "style": gt_payload.get("style", str(style_name_arg)),
                    "n_staves": len(s1_row["staves"]),
                    "n_strips_ok": len(existing_strips),
                    "stage1_row": s1_row,
                }
        except Exception:
            pass

    # Pre-check score parseability and part count (<= 4 parts so Stage-2 bekern matches rendered staves 1:1)
    try:
        parsed = parse_score(score_path)
        if not parsed.parts or parsed.num_measures < 2:
            return {"ok": False, "idx": idx, "stem": stem, "error": "empty_or_too_short"}
        export_stage2 = len(parsed.parts) <= 4
    except Exception as exc:
        return {"ok": False, "idx": idx, "stem": stem, "error": f"parse_failed: {exc}"}

    # Resolve style: named or sample_random_scribe(rng, jitter=0.25)
    np_rng = np.random.default_rng(seed)
    if mode == "random_scribe":
        style_arg: Any = sample_random_scribe(np_rng, jitter=0.25)
    else:
        style_arg = str(style_name_arg)

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
            export_train_dir=work_strip_dir if export_stage2 else None,
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
    if not s1_row["staves"]:
        if work_strip_dir.exists():
            shutil.rmtree(work_strip_dir, ignore_errors=True)
        for p in (page_png, page_json):
            if p.exists():
                p.unlink(missing_ok=True)
        return {"ok": False, "idx": idx, "stem": stem, "error": "zero_staves"}

    # Move & validate Stage-2 strips into exact `<stem>_s<k>.png` + `<stem>_s<k>-bekern.txt`
    n_strips_ok = 0
    n_strips_rejected = 0
    if export_stage2 and work_strip_dir.exists():
        strips_dir.mkdir(parents=True, exist_ok=True)
        for exp in gt_payload.get("train_exports", []):
            if not exp.get("bekern_ok") or not exp.get("bekern"):
                n_strips_rejected += 1
                continue
            src_png = Path(exp["image"])
            src_bek = Path(exp["bekern"])
            if not src_png.exists() or not src_bek.exists():
                n_strips_rejected += 1
                continue
            # Validate tokens strictly against tmp-train/mvd/data.py closed VOCAB
            try:
                toks = mvd_data.load_bekern(src_bek)
                if len(toks) < 4:
                    n_strips_rejected += 1
                    continue
            except Exception:
                n_strips_rejected += 1
                continue

            k = int(exp["system_index"])
            dst_png = strips_dir / f"{stem}_s{k}.png"
            dst_bek = strips_dir / f"{stem}_s{k}-bekern.txt"
            shutil.move(str(src_png), str(dst_png))
            shutil.move(str(src_bek), str(dst_bek))
            n_strips_ok += 1

        shutil.rmtree(work_strip_dir, ignore_errors=True)

    return {
        "ok": True,
        "resumed": False,
        "idx": idx,
        "stem": stem,
        "pool_name": pool_name,
        "score_path": str(score_path),
        "mode": mode,
        "style": gt_payload.get("style", str(style_name_arg)),
        "n_staves": len(s1_row["staves"]),
        "n_strips_ok": n_strips_ok,
        "n_strips_rejected": n_strips_rejected,
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

    return {
        "eval_overlap": len(overlap_hits),
        "overlap_hits": overlap_hits[:10],
        "stage1_dataset_len": len(ds),
        "stage1_augment_checks": s1_checks,
        "stage2_strip_count": len(bek_files),
        "stage2_loader_checks": s2_checks,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate leak-free simulated handwritten Stage-1 & Stage-2 training data."
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
        help="Strip all paper texture, tint, stains, bleed, folds, foxing, blots & smears (pure white paper; augmentation owns paper/photo effects)",
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
        hero_frac = float(np.clip(args.hero_frac, 0.0, 1.0)) if hero_style_names else 0.0
        rand_frac = float(np.clip(args.rand_frac, 0.0, 1.0 - hero_frac))
        n_hero_target = int(round(target_count * hero_frac))
        n_rand_target = int(round(target_count * rand_frac))
        n_other_target = max(0, target_count - n_hero_target - n_rand_target)
        print(
            f"[init] target mix ({target_count} pages): "
            f"hero={n_hero_target} ({100.0*n_hero_target/max(1,target_count):.1f}%), "
            f"other_named={n_other_target} ({100.0*n_other_target/max(1,target_count):.1f}%), "
            f"random_scribe={n_rand_target} ({100.0*n_rand_target/max(1,target_count):.1f}%)",
            flush=True,
        )

        # Build deterministic style assignment preferring Flagship Hero Scribes:
        #   - n_hero_target: uniform across all 75 Hero Scribes
        #   - n_other_target: uniform across the remaining 159 named styles
        #   - n_rand_target: continuous sample_random_scribe(rng, jitter=0.25)
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

        results_by_idx: Dict[int, Dict[str, Any]] = {}
        pool_cursor = 0
        pool_pass = 0
        pool_order = list(pool)

        # Submit in batches so if any score fails to parse/render, we immediately backfill from the pool
        # until we have exactly `target_count` successful pages. Wrap around pool deterministically
        # when target_count > len(pool) so scores render in fresh handwritten styles.
        pending_slots: List[int] = list(range(target_count))
        failed_attempts = 0
        max_failures = target_count * 3

        def make_job(slot_idx: int) -> Optional[Dict[str, Any]]:
            nonlocal pool_cursor, pool_pass, pool_order
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
                # Full SHA check on actually selected score before rendering
                if is_eval_blocked(score_path, raw_slugs, norm_tokens, gt_sha256s, check_sha=True):
                    continue
                mode, sname = interleaved_assign[slot_idx]
                pw, ph = PAGE_SIZES[slot_idx % len(PAGE_SIZES)]
                stem = f"hwsim_{slot_idx:05d}"
                return {
                    "idx": slot_idx,
                    "stem": stem,
                    "score_path": str(score_path),
                    "pool_name": pool_name,
                    "mode": mode,
                    "style_name": sname,
                    "seed": args.seed + slot_idx * 97 + failed_attempts * 13 + pool_pass * 7919,
                    "page_w": pw,
                    "page_h": ph,
                    "pages_dir": str(pages_dir),
                    "strips_dir": str(strips_dir),
                    "tmp_export_dir": str(tmp_export_dir),
                    "clean_paper": bool(args.clean_paper),
                }
            return None

        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
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
                            print(
                                f"[progress] {done}/{target_count} pages "
                                f"({tot_strips} stage2 strips, {failed_attempts} retries, "
                                f"{rate:.2f} pages/s, elapsed {elapsed:.1f}s)",
                                flush=True,
                            )
                    else:
                        failed_attempts += 1
                        if failed_attempts <= max_failures and pool_cursor < len(pool):
                            pending_slots.append(slot)
                    # Refill queue promptly
                    break

        shutil.rmtree(tmp_export_dir, ignore_errors=True)

        # Write Stage-1 manifest.jsonl in deterministic slot order
        ordered = [results_by_idx[k] for k in sorted(results_by_idx.keys())]
        with open(stage1_manifest, "w", encoding="utf-8") as fh:
            for r in ordered:
                fh.write(json.dumps(r["stage1_row"], ensure_ascii=False) + "\n")

        # Write non-destructive merged Stage-1 manifest
        base_manifest = Path(args.base_stage1_manifest)
        merged_manifest = Path(args.merged_stage1_manifest)
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

        selected_sources = [Path(r["score_path"]) for r in ordered]
        named_used = sorted({r["style"] for r in ordered if r["mode"] == "named"})
        n_named_done = sum(1 for r in ordered if r["mode"] == "named")
        n_rand_done = sum(1 for r in ordered if r["mode"] == "random_scribe")
        pool_hist: Dict[str, int] = {}
        for r in ordered:
            pool_hist[r["pool_name"]] = pool_hist.get(r["pool_name"], 0) + 1

        ver = run_verification(
            stage1_manifest=stage1_manifest,
            stage2_strips_dir=strips_dir,
            selected_sources=selected_sources,
            raw_slugs=raw_slugs,
            norm_tokens=norm_tokens,
            gt_sha256s=gt_sha256s,
        )
        assert ver["eval_overlap"] == 0, f"EVAL OVERLAP DETECTED: {ver['overlap_hits']}"

        summary = {
            "clean_paper": bool(args.clean_paper),
            "pages_generated": len(ordered),
            "named_style_pages": n_named_done,
            "unique_named_styles_used": len(named_used),
            "random_scribe_pages": n_rand_done,
            "failed_retries": failed_attempts,
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
