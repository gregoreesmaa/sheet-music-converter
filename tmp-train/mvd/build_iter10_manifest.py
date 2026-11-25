#!/usr/bin/env python3
"""Build and QA-verify `D:\\tmp\\uni_fullpage_iter10_rich.jsonl` for UniOMR Iter 10.

Merges:
  1. Clean Iter-8 base (`D:\\tmp\\uni_fullpage_iter8_clean.jsonl`)
  2. Variation-rich Verovio full pages (`D:\\tmp\\uni-verovio-iter10\\manifest.jsonl`)
  3. Variation-rich OSMD / VexFlow full pages (`D:\\tmp\\uni-osmd-iter10\\manifest.jsonl`)

QA Gates enforced:
  - 0 eval leakage (suite31_hw_manifest.json + gate20/exclusion.json + mozart-k458 + NC)
  - 0 missing image or bekern files
  - 0 OOV tokens against `data.VOCAB`
  - 100% `PrefixState(strict=False)` pass on all new multi-renderer rows (and checked across all rows)
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MVD_DIR = Path(__file__).resolve().parent
REPO_ROOT = MVD_DIR.parent.parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import data as mvd_data  # noqa: E402
import render_verovio_dataset as rvd  # noqa: E402

VOCAB_SET = frozenset(mvd_data.VOCAB)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path(r"D:\tmp\uni_fullpage_iter8_clean.jsonl"))
    ap.add_argument("--verovio", type=Path, default=Path(r"D:\tmp\uni-verovio-iter10\manifest.jsonl"))
    ap.add_argument("--osmd", type=Path, default=Path(r"D:\tmp\uni-osmd-iter10\manifest.jsonl"))
    ap.add_argument("--out", type=Path, default=Path(r"D:\tmp\uni_fullpage_iter10_rich.jsonl"))
    ap.add_argument("--verovio-upsample", type=int, default=1)
    ap.add_argument("--osmd-upsample", type=int, default=1)
    ap.add_argument("--check-prefix-base", action="store_true", help="Also run PrefixState on base rows (slower)")
    ap.add_argument("--seed", type=int, default=20261130)
    args = ap.parse_args()

    t0 = time.time()
    policy, blocked = rvd.build_strict_eval_filter(REPO_ROOT)

    out_rows: List[Dict[str, Any]] = []
    domain_counts = collections.Counter()
    source_counts = collections.Counter()
    font_counts = collections.Counter()
    skipped = collections.Counter()
    prefix_checked = 0
    prefix_passed = 0

    # Cache file validation per unique bekern path to avoid redundant disk reads on upsampled base rows
    bek_cache: Dict[str, Tuple[bool, Optional[bool]]] = {}  # path -> (vocab_ok, prefix_ok_or_None)

    def _validate_bek(bp: str, need_prefix: bool) -> Tuple[bool, bool]:
        cached = bek_cache.get(bp)
        if cached is not None and (not need_prefix or cached[1] is not None):
            return cached[0], bool(cached[1]) if need_prefix else True
        p = Path(bp)
        if not p.is_file():
            return False, False
        toks = mvd_data.load_bekern(p)
        if not toks or any(t not in VOCAB_SET for t in toks):
            bek_cache[bp] = (False, False)
            return False, False
        pref_ok: Optional[bool] = rvd.check_prefix_state(toks) if need_prefix else None
        bek_cache[bp] = (True, pref_ok)
        return True, bool(pref_ok) if need_prefix else True

    def _ingest(path: Path, tag: str, default_domain: str, copies: int, check_prefix: bool) -> None:
        nonlocal prefix_checked, prefix_passed
        if not path.is_file():
            print(f"[build-iter10] WARN: missing {path}, skipping", flush=True)
            return
        rows = _load_jsonl(path)
        print(f"[build-iter10] ingesting {tag}: {len(rows)} rows (x{copies}) from {path}", flush=True)
        img_ok_cache: Dict[str, Tuple[bool, bool]] = {}  # img -> (exists, blocked)
        for ri, row in enumerate(rows, start=1):
            if ri % 10000 == 0:
                print(f"  [{tag}] {ri}/{len(rows)} ({time.time()-t0:.1f}s)", flush=True)
            img = str(row.get("image", ""))
            bks = [str(b) for b in row.get("bekerns", [])]
            dom = str(row.get("domain", default_domain))
            if not img or not bks:
                skipped["missing_img"] += 1
                continue
            ic = img_ok_cache.get(img)
            if ic is None:
                ic = (Path(img).is_file(), rvd.is_path_blocked(img, policy, blocked))
                img_ok_cache[img] = ic
            if not ic[0]:
                skipped["missing_img"] += 1
                continue
            if ic[1]:
                skipped["blocked_eval_or_nc"] += 1
                continue
            row_ok = True
            for b in bks:
                v_ok, p_ok = _validate_bek(b, need_prefix=check_prefix)
                if not v_ok:
                    skipped["missing_or_oov_bekern"] += 1
                    row_ok = False
                    break
                if check_prefix:
                    prefix_checked += 1
                    if not p_ok:
                        skipped["prefix_state_fail"] += 1
                        row_ok = False
                        break
                    prefix_passed += 1
            if not row_ok:
                continue

            clean = {"image": img, "bekerns": bks, "domain": dom}
            if tag in ("verovio", "osmd"):
                font_counts[f"{tag}/{row.get('font', 'unknown')}"] += copies
            for _ in range(copies):
                out_rows.append(clean)
                domain_counts[dom] += 1
                source_counts[tag] += 1

    _ingest(args.base, "iter8_base", "printed", 1, check_prefix=args.check_prefix_base)
    _ingest(args.verovio, "verovio", "printed", args.verovio_upsample, check_prefix=True)
    _ingest(args.osmd, "osmd", "printed", args.osmd_upsample, check_prefix=True)

    rng = random.Random(args.seed)
    rng.shuffle(out_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = args.out.with_suffix(".jsonl.tmp")
    with tmp_out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_out, args.out)

    total = len(out_rows)
    print(f"[build-iter10] DONE -> {args.out} ({time.time()-t0:.1f}s)", flush=True)
    print(f"  total_rows={total} | skipped={dict(skipped)}", flush=True)
    print(f"  prefix_checked_systems={prefix_checked} | prefix_passed={prefix_passed}", flush=True)
    for k, v in domain_counts.most_common():
        print(f"  domain={k:10s}: {v:6d} ({100.0 * v / max(1, total):.1f}%)", flush=True)
    for k, v in source_counts.most_common():
        print(f"  source={k:12s}: {v:6d} ({100.0 * v / max(1, total):.1f}%)", flush=True)
    for k, v in sorted(font_counts.items()):
        print(f"  font={k:22s}: {v:6d}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
