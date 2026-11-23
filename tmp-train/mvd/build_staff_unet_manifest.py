#!/usr/bin/env python3
"""Build the Exp 1 StaffUNet2D full-page training manifest (JSONL).

Zero-copy JSONL referencing existing full-page PNG renders paired with exact
SVG-derived symbox staff-line ground truth (``*-boxes.json``).

Features:
- Surveys ``big500-symbox`` + ``big500-renders``, ``sq-symbox`` + ``sq-renders``,
  and ``D:\\tmp\\corpus10x\\pdmx\\leland-symbox`` + ``quota-renders-leland``.
- Strictly excludes all ``gate20`` (``exclusion.json``, ``gate_manifest.json``,
  ``quarantine.txt``), ``suite31``, and ``suite31_hw`` eval IDs and all NC
  sources (verified programmatically: ``leaked_eval == 0`` and ``nc_count == 0``).
- Groups staves into exact ``system_id`` values using structural symbox glyphs
  (``BarLine``, ``Bracket``, ``Brace``) with gap fallback when barlines are
  absent.
- Oversamples 1-staff and 5-6+ staff pages (default 3x) so ``StaffUNet2D``
  learns both monophonic and dense choral/orchestral layouts.
- Emits JSONL rows compatible with ``staff_unet2d.StaffUNetDataset``
  (``image``, ``img``, ``w``, ``h``, ``staves`` with ``lines``, ``x0``, ``x1``,
  ``system_id``).

Usage (CPU-only):
  .\\.venv\\Scripts\\python.exe tmp-train/mvd/build_staff_unet_manifest.py
  .\\.venv\\Scripts\\python.exe tmp-train/mvd/build_staff_unet_manifest.py \\
      --out D:\\tmp\\staff-unet-data\\manifest.jsonl --target-pages 7500
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Sequence

MVD_DIR = Path(__file__).resolve().parent
REPO_ROOT = MVD_DIR.parent.parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import build_unlabeled_10k as bu  # noqa: E402

DEFAULT_OUT = Path(r"D:\tmp\staff-unet-data\manifest.jsonl")
DEFAULT_SEED = 20261123
DEFAULT_TARGET_PAGES = 7500

DEFAULT_SOURCES: tuple[tuple[str, Path, Path], ...] = (
    (
        "big500",
        REPO_ROOT / "data-gen" / "score-corpus" / "big500-symbox",
        REPO_ROOT / "data-gen" / "score-corpus" / "big500-renders",
    ),
    (
        "sq",
        REPO_ROOT / "data-gen" / "score-corpus" / "sq-symbox",
        REPO_ROOT / "data-gen" / "score-corpus" / "sq-renders",
    ),
    (
        "leland",
        Path(r"D:\tmp\corpus10x\pdmx\leland-symbox"),
        Path(r"D:\tmp\corpus10x\pdmx\quota-renders-leland"),
    ),
)

_STRUCTURAL_GLYPHS = frozenset(("BarLine", "Bracket", "Brace"))


def load_exclusion_policy(repo_root: Path = REPO_ROOT) -> bu.ExclusionPolicy:
    """Load eval-seal + NC exclusion policy covering gate20, suite31, suite31_hw."""
    policy = bu.ExclusionPolicy.from_repo(repo_root)
    s31_hw_full = (
        repo_root
        / "data-gen"
        / "score-corpus"
        / "suite31"
        / "suite31_hw_manifest.json"
    )
    if s31_hw_full.is_file():
        data = json.loads(s31_hw_full.read_text(encoding="utf-8"))
        for piece in data.get("pieces", []):
            slug = str(piece.get("slug", "")).strip().lower()
            if slug:
                policy.suite31_slugs.add(slug)
                if slug.startswith("hw-"):
                    policy.held_out_hw_works.add(slug[3:])
        policy._compile()
    return policy


def is_excluded_eval(policy: bu.ExclusionPolicy, path: Path | str) -> bool:
    """Return True if ``path`` matches any gate20, suite31, or suite31_hw eval ID."""
    if policy.is_leaked_eval(path):
        return True
    p = Path(path)
    stem = p.stem.lower()
    stem_nopage = re.sub(r"-\d+$", "", stem)
    if stem_nopage in policy.suite31_slugs or stem_nopage in policy.gate20_ids or stem_nopage in policy.gate20_scores:
        return True
    return False


def _is_valid_staff(staff: dict[str, Any], w: int, h: int) -> bool:
    lines = staff.get("lines")
    bbox = staff.get("bbox")
    if not isinstance(lines, list) or len(lines) != 5:
        return False
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    ys = [float(y) for y in lines]
    if any(b <= a for a, b in zip(ys, ys[1:])):
        return False
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if x1 <= x0 + 4.0 or y1 <= y0:
        return False
    if ys[0] < -50.0 or ys[-1] > h + 50.0 or x0 < -50.0 or x1 > w + 50.0:
        return False
    return True


def group_staves_into_systems(
    staves: Sequence[dict[str, Any]],
    glyphs: Sequence[dict[str, Any]],
) -> list[int]:
    """Assign top-to-bottom ``system_id`` (0..M-1) to each staff in ``staves``.

    Uses exact structural symbox glyphs (``BarLine``, ``Bracket``, ``Brace``)
    whose vertical bounding box spans >= 2 staves. When no ``BarLine`` glyphs
    exist on the page (e.g. older big500 symbox exports where only piano
    ``Bracket`` glyphs were recorded), also merges adjacent singleton staves
    whose gap is clearly intra-system relative to the page's inter-system gap.
    """
    n = len(staves)
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Sort indices top-to-bottom by staff centerline y.
    order = sorted(range(n), key=lambda i: float(staves[i]["lines"][2]))
    bboxes = [staves[i]["bbox"] for i in order]
    sps = [
        max(1.0, (float(staves[i]["lines"][-1]) - float(staves[i]["lines"][0])) / 4.0)
        for i in order
    ]
    med_sp = sorted(sps)[len(sps) // 2]
    tol_y = max(4.0, 0.5 * med_sp)

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    has_barlines = False
    for g in glyphs:
        cls = g.get("cls", "")
        if cls not in _STRUCTURAL_GLYPHS:
            continue
        if cls == "BarLine":
            has_barlines = True
        gb = g.get("bbox")
        if not isinstance(gb, list) or len(gb) != 4:
            continue
        gx0, gy0, gx1, gy1 = (float(v) for v in gb)
        if gy1 <= gy0:
            continue
        hit: list[int] = []
        for idx, b in enumerate(bboxes):
            sx0, sy0, sx1, sy1 = (float(v) for v in b)
            # Must vertically intersect the staff and sit horizontally near it.
            if (
                gy0 <= sy1 + tol_y
                and gy1 >= sy0 - tol_y
                and gx1 >= sx0 - 8.0 * med_sp
                and gx0 <= sx1 + 4.0 * med_sp
            ):
                hit.append(idx)
        if len(hit) >= 2:
            for j in hit[1:]:
                union(hit[0], j)

    # Fallback for older symbox exports without BarLine glyphs (e.g. big500
    # voice+piano where Bracket only spans the 2 piano staves): if inter-staff
    # gaps are clearly bimodal, merge adjacent staves across small gaps.
    if not has_barlines and n >= 3:
        gaps = [
            float(bboxes[i + 1][1]) - float(bboxes[i][3])
            for i in range(n - 1)
        ]
        max_gap = max(gaps)
        min_gap = min(gaps)
        if max_gap > 1.75 * max(min_gap, 1.0):
            split_thresh = 0.72 * max_gap
            for i, gap in enumerate(gaps):
                if gap < split_thresh:
                    union(i, i + 1)

    groups_by_root: dict[int, list[int]] = {}
    for pos in range(n):
        groups_by_root.setdefault(find(pos), []).append(pos)

    ordered_groups = sorted(
        groups_by_root.values(), key=lambda grp: float(bboxes[grp[0]][1])
    )
    sys_id_by_pos = [0] * n
    for sid, grp in enumerate(ordered_groups):
        for pos in grp:
            sys_id_by_pos[pos] = sid

    out = [0] * n
    for pos, orig_idx in enumerate(order):
        out[orig_idx] = sys_id_by_pos[pos]
    return out


def parse_symbox_page(
    box_path: Path,
    png_path: Path,
    source_name: str,
) -> dict[str, Any] | None:
    """Parse one ``*-boxes.json`` + matching PNG into a StaffUNetDataset row."""
    try:
        data = json.loads(box_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    png_size = data.get("png_size")
    if not isinstance(png_size, list) or len(png_size) != 2:
        return None
    w, h = int(png_size[0]), int(png_size[1])
    if w <= 0 or h <= 0:
        return None

    raw_staves = data.get("staves")
    if not isinstance(raw_staves, list) or not raw_staves:
        return None

    valid_staves = [s for s in raw_staves if _is_valid_staff(s, w, h)]
    if not valid_staves:
        return None

    # Sort staves top-to-bottom for canonical ordering.
    valid_staves.sort(key=lambda s: float(s["lines"][2]))
    sys_ids = group_staves_into_systems(valid_staves, data.get("glyphs", []))

    staves_out: list[dict[str, Any]] = []
    for s, sid in zip(valid_staves, sys_ids):
        x0 = max(0.0, min(float(w - 1), float(s["bbox"][0])))
        x1 = max(x0 + 1.0, min(float(w - 1), float(s["bbox"][2])))
        lines = [round(float(y), 2) for y in s["lines"]]
        staves_out.append(
            {
                "lines": lines,
                "x0": round(x0, 2),
                "x1": round(x1, 2),
                "system_id": int(sid),
            }
        )

    per_sys_counts = Counter(sys_ids)
    n_systems = len(per_sys_counts)
    max_staves_per_sys = max(per_sys_counts.values())
    img_str = str(png_path.resolve())
    return {
        "image": img_str,
        "img": img_str,
        "w": w,
        "h": h,
        "staves": staves_out,
        "n_systems": n_systems,
        "max_staves_per_system": max_staves_per_sys,
        "source": source_name,
    }


def collect_candidate_pairs(
    sources: Sequence[tuple[str, Path, Path]],
    policy: bu.ExclusionPolicy,
) -> tuple[list[tuple[str, Path, Path]], dict[str, int]]:
    """Discover all symbox<->PNG pairs, strictly filtering eval & NC leaks."""
    pairs: list[tuple[str, Path, Path]] = []
    stats = {
        "scanned_boxes": 0,
        "missing_png": 0,
        "excluded_eval": 0,
        "excluded_nc": 0,
    }

    for src_name, symbox_dir, render_dir in sources:
        if not symbox_dir.is_dir() or not render_dir.is_dir():
            continue
        # Pre-index PNGs in render_dir for O(1) lookup.
        png_map: dict[str, Path] = {}
        with os.scandir(render_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.lower().endswith(".png"):
                    png_map[entry.name[:-4]] = Path(entry.path)

        for box_path in sorted(symbox_dir.rglob("*-boxes.json")):
            stats["scanned_boxes"] += 1
            stem = box_path.name[: -len("-boxes.json")]
            png_path = png_map.get(stem)
            if png_path is None:
                stats["missing_png"] += 1
                continue
            if policy.is_forbidden_nc(png_path) or policy.is_forbidden_nc(box_path):
                stats["excluded_nc"] += 1
                continue
            if is_excluded_eval(policy, png_path) or is_excluded_eval(policy, box_path):
                stats["excluded_eval"] += 1
                continue
            pairs.append((src_name, box_path, png_path))

    return pairs, stats


def balance_and_oversample(
    rows: Sequence[dict[str, Any]],
    *,
    target_pages: int = DEFAULT_TARGET_PAGES,
    oversample_1staff: int = 3,
    oversample_dense: int = 2,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Oversample 1-staff (3x) and 5-6 staff (2x) pages into a balanced manifest.

    When the raw pool is small (e.g. unit tests with ``len(rows) * 3 <= target_pages``),
    every 1-staff page is repeated ``oversample_1staff`` times, every >=5-staff page
    is repeated ``oversample_dense`` times, and all 2-4 staff pages are kept.
    When the raw pool exceeds ``target_pages`` (full corpus ~16k pages), allocates
    stratified quotas so 1-staff (3x) and 5-6 staff (2x) are strongly represented
    without letting 10-42 staff orchestral scores drown out chamber/piano layouts.
    """
    rng = random.Random(seed)
    b1: list[dict[str, Any]] = []
    b2: list[dict[str, Any]] = []
    b3: list[dict[str, Any]] = []
    b4: list[dict[str, Any]] = []
    b56: list[dict[str, Any]] = []
    b7p: list[dict[str, Any]] = []

    for r in rows:
        ms = int(r["max_staves_per_system"])
        if ms == 1:
            b1.append(r)
        elif ms == 2:
            b2.append(r)
        elif ms == 3:
            b3.append(r)
        elif ms == 4:
            b4.append(r)
        elif ms in (5, 6):
            b56.append(r)
        else:
            b7p.append(r)

    for b in (b1, b2, b3, b4, b56, b7p):
        b.sort(key=lambda r: r["image"])
        rng.shuffle(b)

    os1 = max(1, int(oversample_1staff))
    osd = max(1, int(oversample_dense))

    raw_expanded = (
        len(b1) * os1
        + len(b2)
        + len(b3)
        + len(b4)
        + len(b56) * osd
        + len(b7p) * osd
    )
    if target_pages <= 0 or raw_expanded <= target_pages:
        out: list[dict[str, Any]] = []
        for _ in range(os1):
            out.extend(b1)
        out.extend(b2)
        out.extend(b3)
        out.extend(b4)
        for _ in range(osd):
            out.extend(b56)
            out.extend(b7p)
        rng.shuffle(out)
        return out

    # Full-corpus stratified allocation summing to target_pages (~7,500):
    # - 1-staff: all unique * os1 (up to 15% of budget)
    # - 5-6 staff: oversampled osd (target ~26% of budget)
    # - 2, 3, 4 staff: balanced chamber/piano/quartet share (~47% of budget)
    # - 7+ staff orchestral: capped (~12% of budget)
    out = []
    cap_1 = max(os1, int(round(0.15 * target_pages)) // os1)
    take_1 = b1[:cap_1]
    for _ in range(os1):
        out.extend(take_1)

    cap_56 = max(osd, int(round(0.26 * target_pages)) // osd)
    take_56 = b56[:cap_56]
    for _ in range(osd):
        out.extend(take_56)

    cap_7p = int(round(0.12 * target_pages))
    take_7p = b7p[:cap_7p]
    out.extend(take_7p)

    rem = max(0, int(target_pages) - len(out))
    mid_buckets = [b2, b3, b4]
    non_empty = [b for b in mid_buckets if b]
    if non_empty and rem > 0:
        per = rem // len(non_empty)
        taken_mid: list[dict[str, Any]] = []
        leftovers: list[dict[str, Any]] = []
        for b in non_empty:
            taken_mid.extend(b[:per])
            leftovers.extend(b[per:])
        if len(taken_mid) < rem and leftovers:
            rng.shuffle(leftovers)
            taken_mid.extend(leftovers[: rem - len(taken_mid)])
        out.extend(taken_mid[:rem])

    rng.shuffle(out)
    return out


def verify_manifest_rows(
    rows: Sequence[dict[str, Any]],
    policy: bu.ExclusionPolicy,
) -> dict[str, int]:
    """Programmatically verify zero eval leakage and zero NC leakage."""
    leaked_eval = 0
    nc_count = 0
    missing_files = 0
    invalid_staves = 0
    for r in rows:
        img = r.get("image") or r.get("img") or ""
        if is_excluded_eval(policy, img):
            leaked_eval += 1
        if policy.is_forbidden_nc(img):
            nc_count += 1
        if not Path(img).is_file():
            missing_files += 1
        staves = r.get("staves", [])
        if not staves or any(
            len(s.get("lines", [])) != 5 or "system_id" not in s
            for s in staves
        ):
            invalid_staves += 1
    return {
        "leaked_eval": leaked_eval,
        "nc_count": nc_count,
        "missing_files": missing_files,
        "invalid_staves": invalid_staves,
    }


def build_manifest(
    *,
    out_path: Path = DEFAULT_OUT,
    sources: Sequence[tuple[str, Path, Path]] = DEFAULT_SOURCES,
    target_pages: int = DEFAULT_TARGET_PAGES,
    oversample_1staff: int = 3,
    oversample_dense: int = 2,
    seed: int = DEFAULT_SEED,
    workers: int = 8,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build and atomically write the StaffUNet2D JSONL manifest + summary."""
    policy = load_exclusion_policy(repo_root)
    pairs, scan_stats = collect_candidate_pairs(sources, policy)

    parsed_rows: list[dict[str, Any]] = []
    if pairs:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            for row in ex.map(
                lambda item: parse_symbox_page(item[1], item[2], item[0]),
                pairs,
            ):
                if row is not None:
                    parsed_rows.append(row)

    unique_hist = Counter(int(r["max_staves_per_system"]) for r in parsed_rows)
    unique_by_source = Counter(str(r["source"]) for r in parsed_rows)

    final_rows = balance_and_oversample(
        parsed_rows,
        target_pages=target_pages,
        oversample_1staff=oversample_1staff,
        oversample_dense=oversample_dense,
        seed=seed,
    )

    audit = verify_manifest_rows(final_rows, policy)
    if audit["leaked_eval"] != 0:
        raise RuntimeError(f"eval leakage detected: {audit['leaked_eval']}")
    if audit["nc_count"] != 0:
        raise RuntimeError(f"NC leakage detected: {audit['nc_count']}")
    if audit["missing_files"] != 0:
        raise RuntimeError(f"missing PNG files: {audit['missing_files']}")
    if audit["invalid_staves"] != 0:
        raise RuntimeError(f"invalid staff records: {audit['invalid_staves']}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    os.replace(tmp_path, out_path)

    final_hist = Counter(int(r["max_staves_per_system"]) for r in final_rows)
    final_by_source = Counter(str(r["source"]) for r in final_rows)
    summary = {
        "manifest": str(out_path),
        "seed": seed,
        "scan_stats": scan_stats,
        "unique_valid_pages": len(parsed_rows),
        "unique_by_source": dict(sorted(unique_by_source.items())),
        "unique_max_staves_per_sys": {str(k): v for k, v in sorted(unique_hist.items())},
        "total_manifest_rows": len(final_rows),
        "final_by_source": dict(sorted(final_by_source.items())),
        "final_max_staves_per_sys": {str(k): v for k, v in sorted(final_hist.items())},
        "oversample_1staff": oversample_1staff,
        "oversample_dense": oversample_dense,
        "audit": audit,
    }
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build StaffUNet2D full-page JSONL manifest")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--target-pages", type=int, default=DEFAULT_TARGET_PAGES)
    ap.add_argument("--oversample-1staff", type=int, default=3)
    ap.add_argument("--oversample-dense", type=int, default=2)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    summary = build_manifest(
        out_path=args.out,
        target_pages=args.target_pages,
        oversample_1staff=args.oversample_1staff,
        oversample_dense=args.oversample_dense,
        seed=args.seed,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
