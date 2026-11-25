#!/usr/bin/env python3
"""Build the unified, leak-free Iter 8 full-page manifest for UniOMR.

Combines:
1. Iter 7+ base (`D:\\tmp\\uni_fullpage_iter7plus.jsonl`)
2. Printed C / C| / clef-G-2-8vb / bar-repeat-left / grace supplement
   (`D:\\tmp\\uni-printed-c-8vb-iter8\\manifest.jsonl`, upsampled 3x)
3. Dense HW-sim v0.5 (`D:\\tmp\\hw-sim-v05-dense\\fullpage_manifest.jsonl`, upsampled 2x)
4. Newly force-aligned real archival HW pages across 254 leak-free works
   (`D:\\tmp\\real-hw-aligned-pages\\manifest.jsonl`, upsampled 4x)

Strictly verifies 0 eval overlap (suite31_hw + gate20 + mozart-k458) and 0 OOV (VOCAB 237).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tmp-train" / "mvd"))
import data as mvd_data  # noqa: E402
from build_iter7plus_manifest import _is_blocked, _load_eval_blocklist, _load_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path(r"D:\tmp\uni_fullpage_iter7plus.jsonl"))
    ap.add_argument("--printed-c8vb", type=Path, default=Path(r"D:\tmp\uni-printed-c-8vb-iter8\manifest.jsonl"))
    ap.add_argument("--hw-dense", type=Path, default=Path(r"D:\tmp\hw-sim-v05-dense\fullpage_manifest.jsonl"))
    ap.add_argument("--real-hw-aligned", type=Path, default=Path(r"D:\tmp\real-hw-aligned-pages\manifest.jsonl"))
    ap.add_argument("--out", type=Path, default=Path(r"D:\tmp\uni_fullpage_iter8.jsonl"))
    ap.add_argument("--c8vb-upsample", type=int, default=3)
    ap.add_argument("--hw-dense-upsample", type=int, default=2)
    ap.add_argument("--real-hw-aligned-upsample", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261126)
    args = ap.parse_args()

    blocked = _load_eval_blocklist()
    vocab_set = set(mvd_data.VOCAB)

    out_rows: List[Dict] = []
    counts = collections.Counter()
    source_counts = collections.Counter()
    skipped_blocked = 0
    skipped_missing = 0
    skipped_oov = 0

    def _add_rows(path: Path, tag: str, default_domain: str, copies: int) -> None:
        nonlocal skipped_blocked, skipped_missing, skipped_oov
        if not path.is_file():
            print(f"[build-iter8] WARN: missing {path}, skipping")
            return
        rows = _load_jsonl(path)
        print(f"[build-iter8] loading {tag}: {len(rows)} rows (x{copies}) from {path}")
        for row in rows:
            img = str(row.get("image", ""))
            bks = [str(b) for b in row.get("bekerns", [])]
            dom = str(row.get("domain", default_domain))
            if not img or not bks or not Path(img).is_file() or any(not Path(b).is_file() for b in bks):
                skipped_missing += 1
                continue
            if _is_blocked(img, blocked):
                skipped_blocked += 1
                continue
            ok = True
            for b in bks:
                toks = mvd_data.load_bekern(Path(b))
                if not toks or any(t not in vocab_set for t in toks):
                    ok = False
                    break
            if not ok:
                skipped_oov += 1
                continue
            clean = {"image": img, "bekerns": bks, "domain": dom}
            for _ in range(copies):
                out_rows.append(clean)
                counts[dom] += 1
                source_counts[tag] += 1

    _add_rows(args.base, "iter7plus_base", "printed", 1)
    _add_rows(args.printed_c8vb, "printed_c_8vb", "printed", args.c8vb_upsample)
    _add_rows(args.hw_dense, "hw_sim_v05_dense", "hw", args.hw_dense_upsample)
    _add_rows(args.real_hw_aligned, "real_hw_aligned", "real_hw", args.real_hw_aligned_upsample)

    rng = random.Random(args.seed)
    rng.shuffle(out_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(out_rows)
    print(f"[build-iter8] DONE -> {args.out}")
    print(f"  total={total} | skipped: blocked={skipped_blocked}, missing={skipped_missing}, oov={skipped_oov}")
    for k, v in counts.most_common():
        print(f"  domain={k:10s}: {v:6d} ({100.0 * v / max(1, total):.1f}%)")
    for k, v in source_counts.most_common():
        print(f"  source={k:18s}: {v:6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
