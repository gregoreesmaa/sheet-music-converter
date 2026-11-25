#!/usr/bin/env python3
"""Build the unified, leak-free Iter 7+ full-page manifest for UniOMR.

Combines:
1. Base Iter-7 printed + real_hw rows (`D:\\tmp\\uni_fullpage_iter7.jsonl`, dropping old v03 `hw`).
2. Rare-meter / pickup (`1/16`, `1/32`, `4/2`, `3/2`, `5/8`, `7/8`) + 5-7 staff + dense piano
   printed supplement (`D:\\tmp\\uni-printed-rare-iter8\\manifest.jsonl`), upsampling `1/16` & `1/32` 3x.
3. Barline-clean, 1-6 staff, 45% landscape, 60% historical-paper synthetic HW full pages
   (`D:\\tmp\\hw-sim-v04-clean\\fullpage_manifest.jsonl`).

Enforces strict 0-overlap against `suite31_hw_manifest.json` and `gate20/exclusion.json`,
verifies all image/bekern files exist, and checks 0 OOV against `data.VOCAB` (237).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tmp-train" / "mvd"))
import data as mvd_data  # noqa: E402


def _load_eval_blocklist() -> Set[str]:
    blocked: Set[str] = set()
    s31 = REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"
    if s31.is_file():
        obj = json.loads(s31.read_text(encoding="utf-8"))
        for p in obj.get("pieces", []):
            for key in ("id", "stem", "slug"):
                v = str(p.get(key, "")).strip().lower()
                if v:
                    blocked.add(v)
                    if v.startswith("hw-"):
                        blocked.add(v[3:])
    g20 = REPO_ROOT / "data-gen" / "score-corpus" / "gate20" / "exclusion.json"
    if g20.is_file():
        obj = json.loads(g20.read_text(encoding="utf-8"))
        for v in obj.get("do_not_train", []):
            s = str(v).strip().lower()
            if s:
                blocked.add(s)
    blocked.add("mozart-k458")
    return blocked


def _is_blocked(img_path: str, blocked: Set[str]) -> bool:
    low = img_path.lower().replace("\\", "/")
    stem = Path(low).stem
    for b in blocked:
        if not b or len(b) < 5:
            continue
        if b == stem or f"/{b}" in low or f"_{b}" in stem or f"-{b}" in stem:
            return True
    return False


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_has_rare_pickup(row: Dict) -> bool:
    for bp in row.get("bekerns", []):
        p = Path(bp)
        if not p.is_file():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if "time-1-16" in txt or "time-1-32" in txt:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path(r"D:\tmp\uni_fullpage_iter7.jsonl"))
    ap.add_argument("--printed-rare", type=Path, default=Path(r"D:\tmp\uni-printed-rare-iter8\manifest.jsonl"))
    ap.add_argument("--hw-sim", type=Path, default=Path(r"D:\tmp\hw-sim-v04-clean\fullpage_manifest.jsonl"))
    ap.add_argument("--out", type=Path, default=Path(r"D:\tmp\uni_fullpage_iter7plus.jsonl"))
    ap.add_argument("--pickup-upsample", type=int, default=3)
    ap.add_argument("--real-hw-extra-copies", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261125)
    args = ap.parse_args()

    blocked = _load_eval_blocklist()
    vocab_set = set(mvd_data.VOCAB)

    out_rows: List[Dict] = []
    counts = collections.Counter()
    skipped_blocked = 0
    skipped_missing = 0
    skipped_oov = 0

    def _validate_and_add(row: Dict, copies: int = 1) -> None:
        nonlocal skipped_blocked, skipped_missing, skipped_oov
        img = str(row.get("image", ""))
        bks = [str(b) for b in row.get("bekerns", [])]
        dom = str(row.get("domain", "printed"))
        if not img or not bks or not Path(img).is_file() or any(not Path(b).is_file() for b in bks):
            skipped_missing += 1
            return
        if _is_blocked(img, blocked):
            skipped_blocked += 1
            return
        for b in bks:
            toks = mvd_data.load_bekern(Path(b))
            if any(t not in vocab_set for t in toks):
                skipped_oov += 1
                return
        clean_row = {"image": img, "bekerns": bks, "domain": dom}
        for _ in range(copies):
            out_rows.append(clean_row)
            counts[dom] += 1

    print(f"[build] loading base: {args.base}")
    for row in _load_jsonl(args.base):
        dom = row.get("domain", "printed")
        if dom == "hw":
            # Replace old v03 hw pages with barline-clean v04 hw pages
            continue
        copies = 1 + (args.real_hw_extra_copies if dom == "real_hw" else 0)
        _validate_and_add(row, copies=copies)

    print(f"[build] loading printed rare supplement: {args.printed_rare}")
    rare_pickup_hits = 0
    for row in _load_jsonl(args.printed_rare):
        is_pickup = _row_has_rare_pickup(row)
        if is_pickup:
            rare_pickup_hits += 1
        _validate_and_add(row, copies=args.pickup_upsample if is_pickup else 1)

    print(f"[build] loading barline-clean hw-sim v04: {args.hw_sim}")
    for row in _load_jsonl(args.hw_sim):
        row["domain"] = "hw"
        _validate_and_add(row, copies=1)

    rng = random.Random(args.seed)
    rng.shuffle(out_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(out_rows)
    print(f"[build] DONE -> {args.out}")
    print(f"  total={total} | rare_pickup_rows={rare_pickup_hits} (x{args.pickup_upsample})")
    print(f"  skipped: blocked={skipped_blocked}, missing={skipped_missing}, oov={skipped_oov}")
    for k, v in counts.most_common():
        print(f"  domain={k:10s}: {v:6d} ({100.0 * v / max(1, total):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
