#!/usr/bin/env python3
"""CPU-only offline phone-photo degradation shard builder for Iter 13.

Selects a deterministic, stratified sample of clean full-page renders from
``D:\\tmp\\uni_fullpage_iter11_clean.jsonl`` (balanced across ``domain`` and
source corpora), runs the compiled Rust ``photo-degrade.exe`` binary in
``--cpu --format jpeg`` mode across realistic phone-photo profiles, and writes
a ready-to-merge JSONL manifest atomically.

Usage examples:
  # Inspect stratified plan without running photo-degrade.exe
  .\\.venv\\Scripts\\python.exe tmp-train\\mvd\\gen_photo_degrade_iter13.py --dry-run --count 6000

  # Quick end-to-end smoke test on 2 pages
  .\\.venv\\Scripts\\python.exe tmp-train\\mvd\\gen_photo_degrade_iter13.py --limit 2 --out-dir D:\\tmp\\photo-degrade-iter13-smoke --out-manifest D:\\tmp\\photo-degrade-iter13-smoke.jsonl

  # Full resumable CPU shard build (e.g. 6,000 pages across 8 workers)
  .\\.venv\\Scripts\\python.exe tmp-train\\mvd\\gen_photo_degrade_iter13.py --count 6000 --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_INPUT_MANIFEST = Path(r"D:\tmp\uni_fullpage_iter11_clean.jsonl")
DEFAULT_OUT_DIR = Path(r"D:\tmp\photo-degrade-iter13")
DEFAULT_OUT_MANIFEST = Path(r"D:\tmp\uni_fullpage_iter13_photo_degraded.jsonl")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHOTO_DEGRADE_BINS: Tuple[Path, ...] = (
    REPO_ROOT / "data-gen" / "photo-degrade" / "target" / "release" / "photo-degrade.exe",
    Path(r"C:\Users\grego\projects\doc-photo-degrade\target\release\doc-photo-degrade.exe"),
)

# Realistic phone-photo profile weights per domain.
DOMAIN_PROFILE_WEIGHTS: Dict[str, Tuple[Tuple[str, float], ...]] = {
    "printed": (
        ("phone-photo", 0.45),
        ("sheet-music", 0.30),
        ("book-gutter-bend", 0.15),
        ("office-scan", 0.10),
    ),
    "hw": (
        ("phone-photo", 0.40),
        ("archival-manuscript", 0.30),
        ("book-gutter-bend", 0.20),
        ("sheet-music", 0.10),
    ),
    "real_hw": (
        ("phone-photo", 0.40),
        ("archival-manuscript", 0.30),
        ("book-gutter-bend", 0.20),
        ("sheet-music", 0.10),
    ),
}

DEFAULT_DOMAIN_FRACS: Dict[str, float] = {
    "printed": 0.55,
    "hw": 0.35,
    "real_hw": 0.10,
}


@dataclass(frozen=True)
class DegradeJob:
    index: int
    row: Dict[str, Any]
    domain: str
    source: str
    short_hash: str
    profile: str
    strength: float
    seed: int
    out_subdir: Path
    expected_out_jpg: Path


def _blake64(master_seed: int, tag: str, key: str) -> int:
    payload = f"iter13_pd:{master_seed}:{tag}:{key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _u01(master_seed: int, tag: str, key: str) -> float:
    return _blake64(master_seed, tag, key) / float(1 << 64)


def _extract_source(image_path: str) -> str:
    norm = image_path.replace("/", "\\")
    parts = [p for p in norm.split("\\") if p]
    if len(parts) >= 3 and parts[1].lower() == "tmp":
        return parts[2]
    if len(parts) >= 2:
        return parts[-2]
    return "other"


def _pick_weighted(u: float, choices: Sequence[Tuple[str, float]]) -> str:
    total = sum(w for _, w in choices)
    target = u * total
    acc = 0.0
    for name, w in choices:
        acc += w
        if target <= acc:
            return name
    return choices[-1][0]


def _resolve_binary(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"photo-degrade binary not found: {p}")
        return p
    for cand in DEFAULT_PHOTO_DEGRADE_BINS:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Could not locate release photo-degrade.exe. Checked: "
        + ", ".join(str(c) for c in DEFAULT_PHOTO_DEGRADE_BINS)
    )


def _allocate_counts(capacities: Dict[str, int], weights: Dict[str, float], total: int) -> Dict[str, int]:
    """Allocate up to `total` items across buckets respecting capacities and target weights."""
    active = {k: cap for k, cap in capacities.items() if cap > 0}
    if not active or total <= 0:
        return {k: 0 for k in capacities}

    target_total = min(total, sum(active.values()))
    alloc: Dict[str, int] = {k: 0 for k in capacities}
    remaining = target_total

    while remaining > 0:
        avail = [k for k in active if alloc[k] < active[k]]
        if not avail:
            break
        w_sum = sum(weights.get(k, 1.0) for k in avail)
        if w_sum <= 0.0:
            w_sum = float(len(avail))
        progressed = False
        # Fractional proportional step (snapshot remaining so later buckets use the same pool)
        rem_snapshot = remaining
        remainders: List[Tuple[float, str]] = []
        for k in avail:
            exact = rem_snapshot * (weights.get(k, 1.0) / w_sum)
            give = min(int(exact), active[k] - alloc[k])
            if give > 0:
                alloc[k] += give
                remaining -= give
                progressed = True
            remainders.append((exact - int(exact), k))
        if remaining <= 0:
            break
        # Distribute leftover 1-by-1 by largest remainder
        remainders.sort(key=lambda x: (-x[0], x[1]))
        for _, k in remainders:
            if remaining <= 0:
                break
            if alloc[k] < active[k]:
                alloc[k] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return alloc


def build_stratified_jobs(
    rows: Sequence[Dict[str, Any]],
    *,
    count: int,
    master_seed: int,
    out_dir: Path,
    suffix: str,
    domain_fracs: Dict[str, float],
    strength_min: float,
    strength_max: float,
) -> List[DegradeJob]:
    """Deterministically select a stratified subset and assign phone-photo profiles."""
    by_dom_src: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        img = str(r.get("image", ""))
        if not img:
            continue
        dom = str(r.get("domain", "printed"))
        src = _extract_source(img)
        by_dom_src[dom][src].append(r)

    # Sort each (domain, source) deterministically by hash(master_seed, image)
    for dom, src_map in by_dom_src.items():
        for src, items in src_map.items():
            items.sort(key=lambda r: (_blake64(master_seed, "order", str(r["image"])), str(r["image"])))

    dom_caps = {dom: sum(len(v) for v in src_map.values()) for dom, src_map in by_dom_src.items()}
    dom_alloc = _allocate_counts(dom_caps, domain_fracs, count)

    selected: List[Tuple[str, str, Dict[str, Any]]] = []
    for dom in sorted(dom_alloc.keys()):
        n_dom = dom_alloc[dom]
        if n_dom <= 0:
            continue
        src_map = by_dom_src[dom]
        src_caps = {s: len(items) for s, items in src_map.items()}
        # Proportional weights across sources within domain (sqrt-damped slightly so rare sources get representation)
        src_weights = {s: float(cap) ** 0.85 for s, cap in src_caps.items()}
        src_alloc = _allocate_counts(src_caps, src_weights, n_dom)
        for src in sorted(src_alloc.keys()):
            take = src_alloc[src]
            for r in src_map[src][:take]:
                selected.append((dom, src, r))

    # Interleave deterministically so --limit N sees all domains/sources
    selected.sort(key=lambda item: (_blake64(master_seed, "interleave", str(item[2]["image"])), str(item[2]["image"])))

    jobs: List[DegradeJob] = []
    for idx, (dom, src, row) in enumerate(selected):
        img_str = str(row["image"])
        img_path = Path(img_str)
        short_hash = hashlib.blake2b(img_str.encode("utf-8"), digest_size=6).hexdigest()
        prof_choices = DOMAIN_PROFILE_WEIGHTS.get(dom, DOMAIN_PROFILE_WEIGHTS["printed"])
        profile = _pick_weighted(_u01(master_seed, "profile", img_str), prof_choices)
        u_str = _u01(master_seed, "strength", img_str)
        strength = round(strength_min + u_str * (strength_max - strength_min), 3)
        photo_seed = _blake64(master_seed, "pd_seed", img_str)
        # Put each degraded file in <out_dir>/<domain>/<hash[:2]>/<short_hash>/ to avoid stem collisions
        out_subdir = out_dir / dom / short_hash[:2] / short_hash
        expected_jpg = out_subdir / f"{img_path.stem}{suffix}.jpg"
        jobs.append(
            DegradeJob(
                index=idx,
                row=row,
                domain=dom,
                source=src,
                short_hash=short_hash,
                profile=profile,
                strength=strength,
                seed=photo_seed,
                out_subdir=out_subdir,
                expected_out_jpg=expected_jpg,
            )
        )
    return jobs


def _build_cmd(
    bin_path: Path,
    job: DegradeJob,
    *,
    suffix: str,
    max_width: int,
    scale_min: float,
    reduced: bool,
) -> List[str]:
    cmd = [
        str(bin_path),
        "--input",
        str(job.row["image"]),
        "--out-dir",
        str(job.out_subdir),
        "--profile",
        job.profile,
        "--strength",
        f"{job.strength:.3f}",
        "--seed",
        str(job.seed),
        "--max-width",
        str(max_width),
        "--scale-min",
        f"{scale_min:.2f}",
        "--suffix",
        suffix,
        "--format",
        "jpeg",
        "--cpu",
    ]
    if reduced:
        cmd.append("--reduced")
    return cmd


def _run_one_job(
    job: DegradeJob,
    *,
    bin_path: Path,
    suffix: str,
    max_width: int,
    scale_min: float,
    reduced: bool,
    resume: bool,
) -> Tuple[DegradeJob, str, Optional[str]]:
    """Run one photo-degrade job on CPU. Returns (job, status, error_msg)."""
    if resume and job.expected_out_jpg.exists() and job.expected_out_jpg.stat().st_size > 1024:
        return job, "skipped", None

    src_path = Path(str(job.row["image"]))
    if not src_path.exists():
        return job, "missing", f"source image missing: {src_path}"

    job.out_subdir.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd(
        bin_path,
        job,
        suffix=suffix,
        max_width=max_width,
        scale_min=scale_min,
        reduced=reduced,
    )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()[-400:]
        return job, "failed", f"rc={proc.returncode}: {err}"
    if not job.expected_out_jpg.exists() or job.expected_out_jpg.stat().st_size <= 1024:
        return job, "failed", f"expected output missing or empty: {job.expected_out_jpg}"
    return job, "ok", None


def _atomic_write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="CPU-only stratified offline phone-photo degradation builder for Iter 13.",
    )
    ap.add_argument("--manifest", type=Path, default=DEFAULT_INPUT_MANIFEST, help="Input clean JSONL manifest")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output root directory for degraded JPEGs")
    ap.add_argument("--out-manifest", type=Path, default=DEFAULT_OUT_MANIFEST, help="Output JSONL manifest path")
    ap.add_argument("--photo-degrade-bin", type=str, default=None, help="Path to photo-degrade.exe (auto-detected if omitted)")
    ap.add_argument("--count", type=int, default=6000, help="Target stratified sample size before --limit")
    ap.add_argument("--limit", type=int, default=None, help="Hard cap on number of jobs to process (for smoke tests)")
    ap.add_argument("--seed", type=int, default=20261126, help="Master deterministic seed")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4), help="CPU worker threads")
    ap.add_argument("--max-width", type=int, default=2048, help="Max width passed to photo-degrade.exe")
    ap.add_argument("--scale-min", type=float, default=0.72, help="Min random downscale passed to photo-degrade.exe")
    ap.add_argument("--strength-min", type=float, default=0.35, help="Minimum global degrade strength")
    ap.add_argument("--strength-max", type=float, default=0.78, help="Maximum global degrade strength")
    ap.add_argument("--suffix", type=str, default="_pd", help="Output filename suffix")
    ap.add_argument("--reduced", action="store_true", help="Pass --reduced preset to photo-degrade.exe for faster CPU runs")
    ap.add_argument("--printed-frac", type=float, default=DEFAULT_DOMAIN_FRACS["printed"])
    ap.add_argument("--hw-frac", type=float, default=DEFAULT_DOMAIN_FRACS["hw"])
    ap.add_argument("--real-hw-frac", type=float, default=DEFAULT_DOMAIN_FRACS["real_hw"])
    ap.add_argument("--no-resume", action="store_true", help="Re-render outputs even if already present on disk")
    ap.add_argument("--dry-run", action="store_true", help="Print stratified plan and sample commands without executing")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parse_args(argv)
    t0 = time.time()

    bin_path = _resolve_binary(args.photo_degrade_bin)
    if not args.manifest.exists():
        print(f"ERROR: input manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    rows: List[Dict[str, Any]] = []
    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    domain_fracs = {
        "printed": float(args.printed_frac),
        "hw": float(args.hw_frac),
        "real_hw": float(args.real_hw_frac),
    }

    target_n = args.limit if (args.limit is not None and args.limit < args.count) else args.count
    jobs = build_stratified_jobs(
        rows,
        count=target_n,
        master_seed=args.seed,
        out_dir=args.out_dir,
        suffix=args.suffix,
        domain_fracs=domain_fracs,
        strength_min=args.strength_min,
        strength_max=args.strength_max,
    )
    if args.limit is not None:
        jobs = jobs[: args.limit]

    dom_counts = Counter(j.domain for j in jobs)
    prof_counts = Counter(j.profile for j in jobs)
    src_counts = Counter((j.domain, j.source) for j in jobs)

    print(f"[photo-degrade-iter13] binary={bin_path}")
    print(f"[photo-degrade-iter13] input_rows={len(rows)} selected={len(jobs)} seed={args.seed} workers={args.workers}")
    print(f"[photo-degrade-iter13] domains={dict(dom_counts)}")
    print(f"[photo-degrade-iter13] profiles={dict(prof_counts)}")
    print(f"[photo-degrade-iter13] top_strata={src_counts.most_common(10)}")

    if args.dry_run:
        print("[photo-degrade-iter13] --dry-run active; sample commands:")
        for j in jobs[:5]:
            cmd = _build_cmd(
                bin_path,
                j,
                suffix=args.suffix,
                max_width=args.max_width,
                scale_min=args.scale_min,
                reduced=args.reduced,
            )
            print("  " + " ".join(cmd))
        return 0

    resume = not args.no_resume
    status_counts: Counter[str] = Counter()
    manifest_rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [
            ex.submit(
                _run_one_job,
                j,
                bin_path=bin_path,
                suffix=args.suffix,
                max_width=args.max_width,
                scale_min=args.scale_min,
                reduced=args.reduced,
                resume=resume,
            )
            for j in jobs
        ]
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            job, status, err = fut.result()
            status_counts[status] += 1
            done += 1
            if status in ("ok", "skipped"):
                manifest_rows.append(
                    {
                        "image": str(job.expected_out_jpg),
                        "bekerns": list(job.row.get("bekerns", [])),
                        "domain": job.domain,
                        "source_image": str(job.row["image"]),
                        "source_stratum": job.source,
                        "photo_profile": job.profile,
                        "photo_seed": job.seed,
                        "photo_strength": job.strength,
                    }
                )
            elif err:
                errors.append(f"[{job.index}] {err}")
            if done % 100 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-3)
                print(
                    f"[photo-degrade-iter13] {done}/{len(jobs)} ({rate:.2f} img/s) "
                    f"status={dict(status_counts)}",
                    flush=True,
                )

    manifest_rows.sort(key=lambda r: r["image"])
    _atomic_write_jsonl(args.out_manifest, manifest_rows)
    elapsed = time.time() - t0
    print(
        f"[photo-degrade-iter13] DONE in {elapsed:.2f}s | written={len(manifest_rows)} "
        f"| manifest={args.out_manifest} | status={dict(status_counts)}"
    )
    if errors:
        print(f"[photo-degrade-iter13] errors ({len(errors)}), first 5:", file=sys.stderr)
        for e in errors[:5]:
            print("  " + e, file=sys.stderr)
        return 1 if len(manifest_rows) == 0 else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
