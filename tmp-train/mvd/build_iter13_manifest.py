#!/usr/bin/env python3
"""Build the expanded, zero-leak Iter 13 Full-Page Manifest (100% CPU-only).

Never imports torch or touches CUDA.

Starts from `D:\\tmp\\uni_fullpage_iter11_clean.jsonl` (44,734 rows) and adds:
  1. `D:\\tmp\\hw-sim-v06-historical-clean\\fullpage_manifest.jsonl`
     (500 historical HW-sim pages, `domain="hw"`, 47% landscape, 37.4% clef-C-3,
     `omit_cont_header_prob=0.35`).
  2. `D:\\tmp\\real-hw-aligned-pages\\targets\\*.txt` mapped back to exact archival
     page images via `align_hw_work_to_pages.load_corpus_works(...)` with strict
     quality & zero-leakage gating:
       - `1 <= n_staves <= 6`
       - `1 <= len(bekerns) <= 16`
       - `full_tok_len <= 3840`
       - `0` OOV tokens against `data.VOCAB`
       - `0` overlap against ALL 45 `suite31_hw` slugs (+ `mozart-k458` + `gate20`)
       - `domain="real_hw"` (feeds `real_hw_archival` under `--split-archival-hw`).

Writes:
  - `D:\\tmp\\uni_fullpage_iter13_clean.jsonl` (atomic `.tmp` -> `os.replace`)
  - `D:\\tmp\\uni_fullpage_iter13_clean_stats.json`
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import collections
import concurrent.futures
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

MVD_DIR = Path(__file__).resolve().parent
REPO_ROOT = MVD_DIR.parent.parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import align_hw_work_to_pages as ahw  # noqa: E402
import data as mvd_data  # noqa: E402
import render_verovio_dataset as rvd  # noqa: E402

VOCAB_SET = frozenset(mvd_data.VOCAB)
SUITE31_HW_MANIFEST = REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"

_STITCHED_HW_MARKERS: Tuple[str, ...] = (
    "1s2s",
    "stitched",
    "concat",
    "synth_page",
    "composed",
)

_TARGET_RE = re.compile(r"^(.+)_p(\d+)-s(\d+)-bekern\.txt$")


def _norm_path_key(p: str) -> str:
    return os.path.normcase(os.path.normpath(p))


def classify_bucket_v2(img_path: str, domain: str = "") -> str:
    """Match `train_uni_v2.classify_fullpage_bucket_v2(..., split_archival_hw=True)`."""
    dom = str(domain or "").strip().lower()
    p_str = str(img_path).replace("\\", "/").lower()
    if dom == "real_hw" or "real_hw" in p_str or "debussy" in p_str:
        if any(tok in p_str or tok in dom for tok in _STITCHED_HW_MARKERS):
            return "real_hw_stitched"
        return "real_hw_archival"
    if (
        dom in ("hw", "synth_hw")
        or "hw-sim-v04" in p_str
        or "hw-sim-v05" in p_str
        or "hw-sim-v06" in p_str
        or "hw-sim" in p_str
        or "hw_sim" in p_str
    ):
        return "synth_hw"
    if "c-8vb" in p_str or "c_8vb" in p_str or "rare" in p_str or dom == "rare_printed":
        return "rare_printed"
    if (
        "verovio" in p_str
        or "osmd" in p_str
        or "spacing" in p_str
        or "corpus100k" in p_str
        or dom == "multi_renderer_printed"
    ):
        return "multi_renderer_printed"
    return "legacy_printed"


def classify_source_tag(img_path: str, domain: str = "") -> str:
    p = str(img_path).replace("\\", "/").lower()
    if "hw-sim-v06-historical-clean" in p:
        return "hw_sim_v06_historical"
    if "hw-sim-v05-dense" in p:
        return "hw_sim_v05_dense"
    if "hw-sim-v04-clean" in p:
        return "hw_sim_v04_clean"
    if "debussy-omr/train_pages" in p:
        return "real_hw_debussy_archival"
    if "pdm-openscore-sq-hw" in p:
        return "real_hw_aligned_openscore_sq"
    if "pdm-openscore-hw" in p:
        return "real_hw_aligned_openscore"
    if "pdm-bach-chopin-hw" in p:
        return "real_hw_aligned_bach_chopin"
    if "nifc-polish-hw" in p:
        return "real_hw_aligned_nifc_polish"
    if "uni-synth-1s2s-pages/real_hw" in p:
        return "real_hw_stitched_1s2s"
    if "uni-synth-1s2s-pages" in p:
        return "printed_stitched_1s2s"
    if "corpus10x" in p:
        return "printed_corpus10x"
    if "t32farm" in p:
        return "printed_t32farm"
    if "corpus100k" in p:
        return "printed_corpus100k"
    if "uni-verovio-spacing-iter10" in p:
        return "printed_verovio_spacing"
    if "uni-verovio-iter10" in p:
        return "printed_verovio"
    if "uni-osmd-spacing-iter10" in p:
        return "printed_osmd_spacing"
    if "uni-osmd-iter10" in p:
        return "printed_osmd"
    if "uni-printed-c-8vb-iter8" in p:
        return "printed_c_8vb"
    if "uni-printed-rare-iter8" in p:
        return "printed_rare"
    return f"other_{domain or 'unknown'}"


def _check_bekern_file(bp: str) -> Tuple[str, bool, bool, int]:
    """Return (path, exists, vocab_ok, tok_len)."""
    p = Path(bp)
    if not p.is_file():
        return bp, False, False, 0
    toks = mvd_data.load_bekern(p)
    if not toks:
        return bp, True, False, 0
    for t in toks:
        if t not in VOCAB_SET:
            return bp, True, False, len(toks)
    return bp, True, True, len(toks)


def _check_file_exists(path_str: str) -> Tuple[str, bool]:
    return path_str, Path(path_str).is_file()


def load_suite31_hw_slugs(manifest_path: Path = SUITE31_HW_MANIFEST) -> List[str]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing suite31_hw_manifest.json: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    slugs = [str(p["slug"]).strip() for p in data.get("pieces", []) if str(p.get("slug", "")).strip()]
    if len(slugs) != 45:
        raise ValueError(f"Expected 45 suite31_hw slugs, got {len(slugs)}")
    return slugs


def _path_matches_suite31_slug(text: str, suite31_slugs: Sequence[str]) -> Optional[str]:
    """Check if `text` matches any of the 45 suite31_hw slugs or mozart-k458."""
    norm_text = ahw._norm_slug(text)
    padded = f"-{norm_text}-"
    if "mozart-k458" in norm_text or "k-458" in norm_text or "kv-458" in norm_text or "kv458" in norm_text:
        return "mozart-k458"
    for slug in suite31_slugs:
        ns = ahw._norm_slug(slug)
        bare = ns[3:] if ns.startswith("hw-") else ns
        for cand in (ns, bare):
            if len(cand) >= 5 and f"-{cand}-" in padded:
                return slug
    return None


def discover_real_hw_aligned_rows(
    aligned_dir: Path,
    corpus_dirs: Sequence[str | Path],
    exclusion: ahw.ExclusionSet,
    policy: Any,
    blocked: Any,
    suite31_slugs: Sequence[str],
    *,
    max_staves: int = 6,
    max_systems: int = 16,
    max_tokens: int = 3840,
    min_bar_ratio: float = 0.40,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Discover and strictly gate clean archival pages from `D:\\tmp\\real-hw-aligned-pages`."""
    targets_dir = aligned_dir / "targets"
    if not targets_dir.is_dir():
        raise FileNotFoundError(f"Missing aligned targets dir: {targets_dir}")

    preds_path = aligned_dir / "iter7plus_12k_page_preds.json"
    preds_map: Dict[str, List[str]] = {}
    if preds_path.is_file():
        preds_map = json.loads(preds_path.read_text(encoding="utf-8"))

    works = ahw.load_corpus_works([str(d) for d in corpus_dirs])
    work_by_safe_wid = {ahw._norm_slug(w.work_id): w for w in works}

    pages: Dict[Tuple[str, int], List[Tuple[int, Path]]] = collections.defaultdict(list)
    unmatched_target_files = 0
    for t in sorted(targets_dir.glob("*.txt")):
        m = _TARGET_RE.match(t.name)
        if not m:
            unmatched_target_files += 1
            continue
        pages[(m.group(1), int(m.group(2)))].append((int(m.group(3)), t))

    kept_rows: List[Dict[str, Any]] = []
    drops: collections.Counter = collections.Counter()
    kept_by_corpus: collections.Counter = collections.Counter()
    kept_by_staves: collections.Counter = collections.Counter()
    kept_works: Set[str] = set()
    bar_ratios: List[float] = []

    for (safe_wid, pnum), sys_list in sorted(pages.items()):
        sys_list.sort(key=lambda x: x[0])
        sys_idxs = [idx for idx, _ in sys_list]
        if not sys_idxs or sys_idxs != list(range(len(sys_idxs))):
            drops["non_contiguous_or_empty_systems"] += 1
            continue
        if len(sys_list) > max_systems:
            drops["gt_max_systems"] += 1
            continue

        w = work_by_safe_wid.get(safe_wid)
        if w is None or pnum < 1 or pnum > len(w.page_paths):
            drops["unmapped_work_or_page"] += 1
            continue

        img_str = str(w.page_paths[pnum - 1])
        if not Path(img_str).is_file():
            drops["missing_source_image"] += 1
            continue

        # Strict zero-leakage checks (ExclusionSet + rvd strict eval/NC + direct 45-slug check)
        is_ex, _reason = exclusion.is_excluded(
            w.work_id,
            composer=w.composer,
            title=w.title,
            rism_label=w.rism_label,
            catalog_keys=w.catalog_keys,
            openscore_mxl=w.openscore_mxl,
        )
        if (
            is_ex
            or rvd.is_path_blocked(img_str, policy, blocked)
            or _path_matches_suite31_slug(img_str, suite31_slugs) is not None
            or _path_matches_suite31_slug(w.work_id, suite31_slugs) is not None
        ):
            drops["eval_or_nc_blocked"] += 1
            continue

        bks = [str(p) for _, p in sys_list]
        first_toks = mvd_data.load_bekern(Path(bks[0]))
        headers, _ = ahw._parse_opening_headers(first_toks)
        n_staves = len(headers)
        if n_staves < 1 or n_staves > max_staves:
            drops[f"staves_out_of_range_{n_staves}"] += 1
            continue

        raw_tok_len = 0
        has_oov = False
        gt_all_toks: List[str] = []
        for s_idx, b in enumerate(bks):
            toks = mvd_data.load_bekern(Path(b))
            if not toks or any(t not in VOCAB_SET for t in toks):
                has_oov = True
                break
            raw_tok_len += len(toks)
            if s_idx > 0:
                gt_all_toks.append("<sys>")
            gt_all_toks.extend(toks)

        if has_oov:
            drops["oov_or_empty_bekern"] += 1
            continue

        full_tok_len = raw_tok_len + max(0, len(bks) - 1) + 2
        if full_tok_len > max_tokens:
            drops["gt_max_tokens"] += 1
            continue

        if preds_map and img_str in preds_map:
            pred_toks = preds_map[img_str]
            gt_bars = sum(1 for t in gt_all_toks if t in ahw.BAR_CLOSING_TOKENS)
            pr_bars = sum(1 for t in pred_toks if t in ahw.BAR_CLOSING_TOKENS)
            br = min(gt_bars, pr_bars) / max(1, max(gt_bars, pr_bars))
            if br < min_bar_ratio:
                drops["low_bar_ratio"] += 1
                continue
            bar_ratios.append(br)

        kept_rows.append({"image": img_str, "bekerns": bks, "domain": "real_hw"})
        kept_by_corpus[w.source_corpus] += 1
        kept_by_staves[n_staves] += 1
        kept_works.add(w.work_id)

    bar_ratios.sort()
    meta = {
        "total_candidate_pages": len(pages),
        "unmatched_target_files": unmatched_target_files,
        "kept_pages": len(kept_rows),
        "kept_unique_works": len(kept_works),
        "kept_by_corpus": dict(sorted(kept_by_corpus.items())),
        "kept_by_staves": {str(k): int(v) for k, v in sorted(kept_by_staves.items())},
        "drops": dict(sorted(drops.items())),
        "bar_ratio_summary": {
            "count": len(bar_ratios),
            "min": round(bar_ratios[0], 4) if bar_ratios else None,
            "median": round(bar_ratios[len(bar_ratios) // 2], 4) if bar_ratios else None,
            "max": round(bar_ratios[-1], 4) if bar_ratios else None,
        },
    }
    return kept_rows, meta


def build_iter13_manifest(
    base_path: Path,
    hw_v06_manifest: Path,
    real_hw_aligned_dir: Path,
    out_path: Path,
    stats_path: Path,
    *,
    max_tokens: int = 3840,
    max_systems: int = 16,
    max_real_hw_staves: int = 6,
    min_bar_ratio: float = 0.40,
    seed: int = 20261126,
    workers: int = 16,
) -> Dict[str, Any]:
    t0 = time.time()
    if not base_path.is_file():
        raise FileNotFoundError(f"Base Iter-11 manifest not found: {base_path}")
    if not hw_v06_manifest.is_file():
        raise FileNotFoundError(f"HW-sim v06 manifest not found: {hw_v06_manifest}")

    suite31_slugs = load_suite31_hw_slugs()
    exclusion = ahw.ExclusionSet.from_files()
    policy, blocked = rvd.build_strict_eval_filter(REPO_ROOT)

    print(f"[build-iter13] Loading base manifest: {base_path} ...", flush=True)
    base_rows: List[Dict[str, Any]] = []
    with base_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                base_rows.append(json.loads(s))
    print(f"[build-iter13] Loaded {len(base_rows):,} base rows ({time.time() - t0:.1f}s)", flush=True)

    print(f"[build-iter13] Loading hw-sim-v06 manifest: {hw_v06_manifest} ...", flush=True)
    v06_rows: List[Dict[str, Any]] = []
    with hw_v06_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            r = json.loads(s)
            v06_rows.append(
                {
                    "image": str(r["image"]),
                    "bekerns": [str(b) for b in r["bekerns"]],
                    "domain": "hw",
                }
            )
    print(f"[build-iter13] Loaded {len(v06_rows):,} hw-sim-v06 rows", flush=True)

    print(f"[build-iter13] Discovering aligned real-hw pages: {real_hw_aligned_dir} ...", flush=True)
    aligned_rows, aligned_meta = discover_real_hw_aligned_rows(
        real_hw_aligned_dir,
        ahw.DEFAULT_CORPUS_DIRS,
        exclusion,
        policy,
        blocked,
        suite31_slugs,
        max_staves=max_real_hw_staves,
        max_systems=max_systems,
        max_tokens=max_tokens,
        min_bar_ratio=min_bar_ratio,
    )
    print(
        f"[build-iter13] Aligned real-hw: {aligned_meta['kept_pages']} clean pages "
        f"across {aligned_meta['kept_unique_works']} works (drops={aligned_meta['drops']})",
        flush=True,
    )

    # Combine & deduplicate by normalized image path
    combined_raw = base_rows + v06_rows + aligned_rows
    dedup_rows: List[Dict[str, Any]] = []
    seen_imgs: Set[str] = set()
    dup_dropped = 0
    for row in combined_raw:
        img = str(row.get("image", ""))
        key = _norm_path_key(img)
        if key in seen_imgs:
            dup_dropped += 1
            continue
        seen_imgs.add(key)
        dedup_rows.append(
            {
                "image": img,
                "bekerns": [str(b) for b in row.get("bekerns", [])],
                "domain": str(row.get("domain", "printed")),
            }
        )

    # Full parallel disk + vocab + token-length verification across EVERY row
    unique_imgs = {r["image"] for r in dedup_rows}
    unique_beks = {b for r in dedup_rows for b in r["bekerns"]}
    print(
        f"[build-iter13] Verifying {len(unique_imgs):,} unique images and "
        f"{len(unique_beks):,} unique bekerns (workers={workers}) ...",
        flush=True,
    )
    img_exists: Dict[str, bool] = {}
    bek_info: Dict[str, Tuple[bool, bool, int]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for p, ok in ex.map(_check_file_exists, unique_imgs):
            img_exists[p] = ok
        for bp, exists, vocab_ok, tlen in ex.map(_check_bekern_file, unique_beks):
            bek_info[bp] = (exists, vocab_ok, tlen)
    print(f"[build-iter13] Parallel verification complete ({time.time() - t0:.1f}s)", flush=True)

    # Filter & strict QA check
    out_rows: List[Dict[str, Any]] = []
    dropped: collections.Counter = collections.Counter()
    suite31_hits: List[str] = []

    for row in dedup_rows:
        img = row["image"]
        bks = row["bekerns"]
        if not img or not bks or not img_exists.get(img, False):
            dropped["missing_image_or_empty_bekerns"] += 1
            continue
        if len(bks) > max_systems:
            dropped["gt_max_systems"] += 1
            continue

        # Zero-leakage check against all 45 suite31_hw slugs + mozart-k458 + rvd strict filter
        hit = _path_matches_suite31_slug(img, suite31_slugs)
        if hit is not None or rvd.is_path_blocked(img, policy, blocked):
            dropped["eval_or_nc_blocked"] += 1
            if hit is not None:
                suite31_hits.append(f"{hit}:{img}")
            continue

        bek_blocked = False
        missing_bek = False
        oov_bek = False
        raw_tok_len = 0
        for b in bks:
            bhit = _path_matches_suite31_slug(b, suite31_slugs)
            if bhit is not None or rvd.is_path_blocked(b, policy, blocked):
                bek_blocked = True
                if bhit is not None:
                    suite31_hits.append(f"{bhit}:{b}")
                break
            exists, vocab_ok, tlen = bek_info[b]
            if not exists:
                missing_bek = True
                break
            if not vocab_ok:
                oov_bek = True
                break
            raw_tok_len += tlen

        if bek_blocked:
            dropped["eval_or_nc_blocked"] += 1
            continue
        if missing_bek:
            dropped["missing_bekern"] += 1
            continue
        if oov_bek:
            dropped["oov_or_empty_bekern"] += 1
            continue

        full_tok_len = raw_tok_len + max(0, len(bks) - 1) + 2
        if full_tok_len > max_tokens:
            dropped["gt_max_tokens"] += 1
            continue

        out_rows.append(row)

    # Deterministic shuffle
    rng = random.Random(seed)
    rng.shuffle(out_rows)

    # Hard final QA pass over out_rows
    domain_counts: collections.Counter = collections.Counter()
    bucket_counts: collections.Counter = collections.Counter()
    source_counts: collections.Counter = collections.Counter()
    verify_imgs: Set[str] = set()
    final_suite31_overlap = 0

    for row in out_rows:
        img = row["image"]
        bks = row["bekerns"]
        dom = row["domain"]
        k = _norm_path_key(img)
        assert k not in verify_imgs, f"QA FAIL: duplicate image {img}"
        verify_imgs.add(k)
        if _path_matches_suite31_slug(img, suite31_slugs) is not None:
            final_suite31_overlap += 1
        for b in bks:
            if _path_matches_suite31_slug(b, suite31_slugs) is not None:
                final_suite31_overlap += 1
        domain_counts[dom] += 1
        bucket_counts[classify_bucket_v2(img, dom)] += 1
        source_counts[classify_source_tag(img, dom)] += 1

    assert final_suite31_overlap == 0, f"QA FAIL: {final_suite31_overlap} suite31_hw overlaps!"

    # Atomic write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_out, out_path)

    total = len(out_rows)
    stats: Dict[str, Any] = {
        "inputs": {
            "base_manifest": str(base_path),
            "base_rows": len(base_rows),
            "hw_sim_v06_manifest": str(hw_v06_manifest),
            "hw_sim_v06_rows": len(v06_rows),
            "real_hw_aligned_dir": str(real_hw_aligned_dir),
            "real_hw_aligned_kept": len(aligned_rows),
            "real_hw_aligned_detail": aligned_meta,
        },
        "output_manifest": str(out_path),
        "seed": seed,
        "total_rows": total,
        "duplicates_removed": dup_dropped,
        "filter_drops": dict(sorted(dropped.items())),
        "counts_by_domain": {
            k: {"rows": int(v), "pct": round(100.0 * v / max(1, total), 2)}
            for k, v in sorted(domain_counts.items())
        },
        "counts_by_bucket_v2": {
            k: {"rows": int(v), "pct": round(100.0 * v / max(1, total), 2)}
            for k, v in sorted(bucket_counts.items())
        },
        "counts_by_source": {
            k: int(v) for k, v in sorted(source_counts.items(), key=lambda x: -x[1])
        },
        "qa_verification": {
            "suite31_hw_slugs_checked": len(suite31_slugs),
            "suite31_hw_overlap": final_suite31_overlap,
            "duplicate_images": 0,
            "missing_files": 0,
            "oov_tokens": 0,
            "rows_over_max_tokens": 0,
            "rows_over_max_systems": 0,
            "passed": True,
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_stats = stats_path.with_suffix(stats_path.suffix + ".tmp")
    tmp_stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_stats, stats_path)

    print("\n" + "=" * 78, flush=True)
    print(f"[build-iter13] DONE -> {out_path} ({stats['elapsed_sec']:.1f}s)", flush=True)
    print(f"  Total rows: {total:,} (base={len(base_rows):,} + v06={len(v06_rows):,} + aligned_real_hw={len(aligned_rows):,}, dup={dup_dropped})", flush=True)
    print("  By domain:", {k: v["rows"] for k, v in stats["counts_by_domain"].items()}, flush=True)
    print("  By bucket (v2 --split-archival-hw):", {k: v["rows"] for k, v in stats["counts_by_bucket_v2"].items()}, flush=True)
    print("  By source:", flush=True)
    for src, cnt in stats["counts_by_source"].items():
        print(f"    {src:<32s}: {cnt:6,d}", flush=True)
    print(
        f"  QA: 0 / {len(suite31_slugs)} suite31_hw overlap | 0 dup | 0 missing | 0 OOV | PASS",
        flush=True,
    )
    print("=" * 78, flush=True)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Build expanded zero-leak Iter 13 full-page manifest (CPU-only).")
    ap.add_argument("--base", type=Path, default=Path(r"D:\tmp\uni_fullpage_iter11_clean.jsonl"))
    ap.add_argument("--hw-v06", type=Path, default=Path(r"D:\tmp\hw-sim-v06-historical-clean\fullpage_manifest.jsonl"))
    ap.add_argument("--real-hw-aligned", type=Path, default=Path(r"D:\tmp\real-hw-aligned-pages"))
    ap.add_argument("--out", type=Path, default=Path(r"D:\tmp\uni_fullpage_iter13_clean.jsonl"))
    ap.add_argument("--stats", type=Path, default=Path(r"D:\tmp\uni_fullpage_iter13_clean_stats.json"))
    ap.add_argument("--max-tokens", type=int, default=3840)
    ap.add_argument("--max-systems", type=int, default=16)
    ap.add_argument("--max-real-hw-staves", type=int, default=6)
    ap.add_argument("--min-bar-ratio", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=20261126)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    build_iter13_manifest(
        base_path=args.base,
        hw_v06_manifest=args.hw_v06,
        real_hw_aligned_dir=args.real_hw_aligned,
        out_path=args.out,
        stats_path=args.stats,
        max_tokens=args.max_tokens,
        max_systems=args.max_systems,
        max_real_hw_staves=args.max_real_hw_staves,
        min_bar_ratio=args.min_bar_ratio,
        seed=args.seed,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
