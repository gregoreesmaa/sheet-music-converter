#!/usr/bin/env python3
"""W2/W3 Rare Strata & Long-System (512-768 Token) Dataset Builder (CPU-only).

Materializes ``D:\\tmp\\w2w3-strata-data\\`` containing:
  - Clean system strip PNGs (``<stem>.png``)
  - Verified post-I1 ``<stem>-bekern.txt`` targets (0 OOV, 0 non-monotonic bars)
  - ``hybrid_staff.jsonl`` (valid ``1 <= S <= 4``, ``skip == False`` rows)
  - ``summary.json`` (exact counts per stratum + safety invariants)

Two complementary permissive sources on disk:
  (a) **W3 Rare-Feature Mining** from ``unified-sysdata`` / ``quota-sysdata``
      (already post-I1 strips with verified ``hybrid_staff.jsonl`` rows):
      selects up to ``--target-w3`` (default 1,500) hard positives containing:
        - ``c_clef``: ``clef-C-1/2/3/4/5``
        - ``mid_bar_clef``: ``clef-*`` token in body after leading headers
        - ``irregular_tuplet``: non-triplet ``tuplet-start-*`` (not ``3-2``)
        - ``extreme_ledger``: >= ``--extreme-ledger-min`` (default 4) notes
          with pitch ``<= E2`` (MIDI 40) or ``>= A5`` (MIDI 81)
  (b) **W2 Long-System Recovery (512 < tokens <= 768)** from
      ``D:\\tmp\\corpus10x\\pdmx\\leland-symbox`` + ``quota-renders-leland``:
      recovers up to ``--target-w2`` (default 1,500) dense 513..768-token
      systems using post-I1 ``assembly-serialize.exe bekern --measures`` +
      ``bekern-decode`` round-trip gate + exact symbox staff tables
      (``1 <= S <= 4``, ``0`` OOV tokens, ``0`` bad I1 bars).

Strictly CPU-only (zero CUDA imports/usage). Strictly excludes all ``gate20``
(``exclusion.json`` + ``gate_manifest.json`` + ``quarantine.txt``),
``suite31``, ``suite31_hw``, and non-commercial (NC) material via
``build_unlabeled_10k.ExclusionPolicy.from_repo()``.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe tmp-train/mvd/build_w2w3_strata.py
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PIL import Image

MVD_DIR = Path(__file__).resolve().parent
REPO_ROOT = MVD_DIR.parent.parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import build_hybrid_cache  # noqa: E402
import build_sys_data  # noqa: E402
import build_unlabeled_10k  # noqa: E402
import data  # noqa: E402
import recanon_i1  # noqa: E402
import strip_staff  # noqa: E402

DEFAULT_OUT_DIR = Path(r"D:\tmp\w2w3-strata-data")
DEFAULT_UNIFIED_SYSDATA = Path(r"D:\tmp\corpus10x\pdmx\unified-sysdata")
DEFAULT_UNIFIED_HYBRID = Path(r"D:\tmp\unified-hybrid\hybrid_staff.jsonl")
DEFAULT_QUOTA_SYSDATA = Path(r"D:\tmp\corpus10x\pdmx\quota-sysdata")
DEFAULT_QUOTA_HYBRID = Path(r"D:\tmp\quota-hybrid\hybrid_staff.jsonl")
DEFAULT_LELAND_RENDERS = Path(r"D:\tmp\corpus10x\pdmx\quota-renders-leland")
DEFAULT_LELAND_SYMBOX = Path(r"D:\tmp\corpus10x\pdmx\leland-symbox")
DEFAULT_LELAND_SYSDATA = Path(r"D:\tmp\corpus10x\pdmx\leland-sysdata")
DEFAULT_SQ_RENDERS = REPO_ROOT / "data-gen" / "score-corpus" / "sq-renders"
DEFAULT_SQ_SYMBOX = REPO_ROOT / "data-gen" / "score-corpus" / "sq-symbox"
DEFAULT_QUOTA_RENDERS = Path(r"D:\tmp\corpus10x\pdmx\quota-renders")
DEFAULT_ASER = (
    REPO_ROOT
    / "converter"
    / "assembly-serialize"
    / "target"
    / "release"
    / "assembly-serialize.exe"
)

VOCAB_SET: frozenset[str] = frozenset(data.VOCAB)
STEPS_SET: frozenset[str] = frozenset("ABCDEFG")
OCTAVES_SET: frozenset[str] = frozenset("012345678")
STEP_SEMITONE: Dict[str, int] = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}

# MIDI thresholds for extreme ledger-line notes (diatonic/natural pitch):
# E2 = (2 + 1) * 12 + 4 = 40; A5 = (5 + 1) * 12 + 9 = 81.
EXTREME_LOW_MIDI = 40   # <= E2
EXTREME_HIGH_MIDI = 81  # >= A5

LONG_MIN_TOKENS = 513   # strictly > 512
LONG_MAX_TOKENS = 768   # <= 768

STRATA_KEYS: Tuple[str, ...] = (
    "long_512_768",
    "c_clef",
    "mid_bar_clef",
    "irregular_tuplet",
    "extreme_ledger",
)
W3_STRATA_KEYS: Tuple[str, ...] = (
    "mid_bar_clef",
    "irregular_tuplet",
    "c_clef",
    "extreme_ledger",
)

_SYS_STEM_RE = re.compile(r"^(.+)-(\d+)-s(\d+)(?:c\d+)?$")
_FONT_SUFFIX_RE = re.compile(r"_(?:Leland|Emmentaler|Bravura|Petaluma|Gonville|MuseJazz|FinaleMaestro)$")


# ---------------------------------------------------------------------------
# Pure feature detectors & quality gates
# ---------------------------------------------------------------------------


def find_oov_tokens(tokens: Sequence[str]) -> List[str]:
    """Return list of tokens not in closed ``data.VOCAB``."""
    return [t for t in tokens if t not in VOCAB_SET]


def count_i1_bars(tokens: Sequence[str]) -> Tuple[int, int]:
    """Return ``(total_bars, bad_i1_bars)`` for a bekern token sequence."""
    _, body = recanon_i1.split_leading_headers(list(tokens))
    total = 0
    bad = 0
    for m_toks, _ in recanon_i1.split_measures(body):
        total += 1
        if not recanon_i1.is_measure_monotonic(m_toks):
            bad += 1
    return total, bad


def is_long_512_768(tokens: Sequence[str]) -> bool:
    """True iff token count satisfies ``512 < len(tokens) <= 768``."""
    n = len(tokens)
    return LONG_MIN_TOKENS <= n <= LONG_MAX_TOKENS


def has_c_clef(tokens: Sequence[str]) -> bool:
    """True iff any C-clef token (``clef-C-1``..``clef-C-5``) is present."""
    return any(t.startswith("clef-C-") for t in tokens)


def has_mid_bar_clef(tokens: Sequence[str]) -> bool:
    """True iff any ``clef-*`` token occurs after the leading staff headers."""
    _, body = recanon_i1.split_leading_headers(list(tokens))
    return any(t.startswith("clef-") for t in body)


def is_irregular_tuplet_token(tok: str) -> bool:
    """True for any non-triplet tuplet start token (e.g. 5/6/7/9/10-tuplets)."""
    return tok.startswith("tuplet-start-") and tok != "tuplet-start-3-2"


def has_irregular_tuplet(tokens: Sequence[str]) -> bool:
    """True iff any non-triplet ``tuplet-start-*`` token is present."""
    return any(is_irregular_tuplet_token(t) for t in tokens)


def count_extreme_ledger_notes(tokens: Sequence[str]) -> int:
    """Count pitched note events with natural pitch ``<= E2`` or ``>= A5``.

    In bekern, a pitched note emits ``<step> <octave>`` (optionally followed
    by an accidental ``a-2/a-1/a1/a2``). Ledger-line position is determined by
    diatonic step and octave, so natural pitch MIDI (``<= 40`` for ``<= E2``,
    ``>= 81`` for ``>= A5``) is exact.
    """
    n = len(tokens)
    count = 0
    i = 0
    while i < n - 1:
        step = tokens[i]
        if step in STEPS_SET:
            oct_s = tokens[i + 1]
            if oct_s in OCTAVES_SET:
                midi = (int(oct_s) + 1) * 12 + STEP_SEMITONE[step]
                if midi <= EXTREME_LOW_MIDI or midi >= EXTREME_HIGH_MIDI:
                    count += 1
                i += 2
                continue
        i += 1
    return count


def has_extreme_ledger(tokens: Sequence[str], min_notes: int = 4) -> bool:
    """True iff the strip has at least ``min_notes`` extreme ledger-line notes."""
    return count_extreme_ledger_notes(tokens) >= min_notes


def classify_strata(
    tokens: Sequence[str], *, extreme_ledger_min: int = 4
) -> Dict[str, bool]:
    """Return boolean stratum membership dict for all W2/W3 strata."""
    return {
        "long_512_768": is_long_512_768(tokens),
        "c_clef": has_c_clef(tokens),
        "mid_bar_clef": has_mid_bar_clef(tokens),
        "irregular_tuplet": has_irregular_tuplet(tokens),
        "extreme_ledger": has_extreme_ledger(tokens, min_notes=extreme_ledger_min),
    }


def base_score_id(stem: str) -> str:
    """Extract base piece/score ID (stripping ``-<page>-s<sys>`` and font suffix)."""
    m = _SYS_STEM_RE.match(stem)
    score = m.group(1) if m else stem
    return _FONT_SUFFIX_RE.sub("", score)


def is_excluded(
    identifier: str | Path,
    policy: build_unlabeled_10k.ExclusionPolicy,
) -> bool:
    """True iff identifier/path hits eval-leakage (gate20/suite31/hw) or NC policy."""
    if policy.is_leaked_eval(identifier) or policy.is_forbidden_nc(identifier):
        return True
    stem = Path(identifier).stem
    base = base_score_id(stem)
    if base and base != stem and policy.is_leaked_eval(base):
        return True
    return False


# ---------------------------------------------------------------------------
# Atomic I/O helpers
# ---------------------------------------------------------------------------


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink ``src`` -> ``dst`` if possible, else atomic copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


# ---------------------------------------------------------------------------
# Part A: W3 Rare-Feature Mining from existing post-I1 sysdata
# ---------------------------------------------------------------------------


def _scan_w3_chunk(
    args: Tuple[str, List[str], int],
) -> List[Dict[str, Any]]:
    """Worker: scan a batch of ``*-bekern.txt`` files for W3 rare features."""
    sysdata_dir_s, bek_names, extreme_ledger_min = args
    sysdata_dir = Path(sysdata_dir_s)
    hits: List[Dict[str, Any]] = []
    for fn in bek_names:
        stem = fn[: -len("-bekern.txt")]
        png_path = sysdata_dir / f"{stem}.png"
        if not png_path.is_file():
            continue
        bek_path = sysdata_dir / fn
        try:
            text = bek_path.read_text(encoding="utf-8")
        except OSError:
            continue
        toks = text.split()
        if not toks:
            continue
        if find_oov_tokens(toks):
            continue
        _, bad_bars = count_i1_bars(toks)
        if bad_bars > 0:
            continue
        flags = classify_strata(toks, extreme_ledger_min=extreme_ledger_min)
        if not any(flags[k] for k in W3_STRATA_KEYS):
            continue
        n_ext = count_extreme_ledger_notes(toks) if flags["extreme_ledger"] else 0
        hits.append(
            {
                "stem": stem,
                "img": f"{stem}.png",
                "base_score": base_score_id(stem),
                "sysdata_dir": sysdata_dir_s,
                "n_tokens": len(toks),
                "n_ext_ledger": n_ext,
                "flags": flags,
            }
        )
    return hits


def _select_balanced_w3(
    candidates: List[Dict[str, Any]],
    *,
    target: int,
    max_per_score: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Select up to ``target`` W3 candidates balanced across the 4 W3 strata.

    Prioritizes rarer strata (``mid_bar_clef``, ``irregular_tuplet``,
    ``c_clef``, ``extreme_ledger``) and multi-feature hard positives while
    enforcing a per-piece diversity cap (``max_per_score``).
    """
    rng = random.Random(seed)
    per_stratum_quota = max(1, (target + len(W3_STRATA_KEYS) - 1) // len(W3_STRATA_KEYS))

    # Pre-shuffle deterministically, then sort by multi-feature richness desc.
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    shuffled.sort(
        key=lambda c: (
            sum(1 for k in W3_STRATA_KEYS if c["flags"][k]),
            c["n_ext_ledger"],
            c["n_tokens"],
        ),
        reverse=True,
    )

    selected: List[Dict[str, Any]] = []
    seen_stems: Set[str] = set()
    score_counts: Dict[str, int] = collections.Counter()
    stratum_counts: Dict[str, int] = collections.Counter()

    # Pass 1: fill each W3 stratum up to per_stratum_quota (rarest first).
    for key in W3_STRATA_KEYS:
        for cand in shuffled:
            if len(selected) >= target:
                break
            if stratum_counts[key] >= per_stratum_quota:
                break
            if not cand["flags"][key]:
                continue
            stem = cand["stem"]
            if stem in seen_stems:
                continue
            if score_counts[cand["base_score"]] >= max_per_score:
                continue
            seen_stems.add(stem)
            score_counts[cand["base_score"]] += 1
            for k in W3_STRATA_KEYS:
                if cand["flags"][k]:
                    stratum_counts[k] += 1
            selected.append(cand)

    # Pass 2: top up remaining slots up to target (preferring least-filled strata).
    if len(selected) < target:
        for cand in shuffled:
            if len(selected) >= target:
                break
            stem = cand["stem"]
            if stem in seen_stems:
                continue
            if score_counts[cand["base_score"]] >= max_per_score:
                continue
            seen_stems.add(stem)
            score_counts[cand["base_score"]] += 1
            for k in W3_STRATA_KEYS:
                if cand["flags"][k]:
                    stratum_counts[k] += 1
            selected.append(cand)

    return selected


def load_matching_hybrid_rows(
    hybrid_jsonl: Path,
    wanted_imgs: Set[str],
) -> Dict[str, Dict[str, Any]]:
    """Stream ``hybrid_staff.jsonl`` and return valid rows for ``wanted_imgs``.

    Uses fast pre-filtering on raw line text before JSON decoding so scanning
    multi-GB JSONL files finishes in ~1-2 seconds.
    """
    found: Dict[str, Dict[str, Any]] = {}
    if not wanted_imgs or not hybrid_jsonl.is_file():
        return found
    remaining = set(wanted_imgs)
    with hybrid_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not remaining:
                break
            # Fast skip: clean strips are .png and skip: false
            if ".png" not in line or '"skip": true' in line:
                continue
            # Extract img name cheaply: `"img": "<name>"`
            idx = line.find('"img": "')
            if idx < 0:
                idx = line.find('"img":"')
                if idx < 0:
                    continue
                start = idx + 7
            else:
                start = idx + 8
            end = line.find('"', start)
            if end < 0:
                continue
            img_name = line[start:end]
            if img_name not in remaining:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("skip"):
                continue
            s_val = int(row.get("S", 0) or 0)
            if not (1 <= s_val <= 4):
                continue
            table = row.get("table")
            if not isinstance(table, dict) or len(table.get("staves", [])) != s_val:
                continue
            found[img_name] = row
            remaining.discard(img_name)
    return found


def validate_staff_row_against_png(row: Dict[str, Any], png_path: Path) -> bool:
    """Verify staff table ``w_in`` matches PNG width and curves reconstruct."""
    try:
        with Image.open(png_path) as im:
            w, h = im.size
        table = row.get("table") or {}
        if int(table.get("w_in", -1)) != w:
            return False
        s_val = int(row.get("S", 0))
        if not (1 <= s_val <= 4):
            return False
        staves = table.get("staves") or []
        if len(staves) != s_val:
            return False
        # Sanity check all y coords inside [0, h]
        for staff in staves:
            if len(staff) != 5:
                return False
            for line_pts in staff:
                if not line_pts:
                    return False
                if min(line_pts) < -2.0 or max(line_pts) > h + 2.0:
                    return False
        curves = strip_staff.table_to_curves(table)
        return len(curves) == s_val
    except Exception:
        return False


def mine_w3_rare_strata(
    sources: Sequence[Tuple[Path, Path]],
    out_dir: Path,
    policy: build_unlabeled_10k.ExclusionPolicy,
    *,
    target_w3: int = 1500,
    extreme_ledger_min: int = 4,
    max_per_score: int = 4,
    workers: int = 8,
    seed: int = 20261123,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Mine up to ``target_w3`` post-I1 rare-feature strips from disk."""
    t0 = time.time()
    all_candidates: List[Dict[str, Any]] = []
    seen_stems: Set[str] = set()
    excluded_count = 0

    for sysdata_dir, hybrid_jsonl in sources:
        if not sysdata_dir.is_dir() or not hybrid_jsonl.is_file():
            print(f"[W3] skipping missing source: {sysdata_dir} / {hybrid_jsonl}", flush=True)
            continue
        bek_names: List[str] = []
        with os.scandir(sysdata_dir) as it:
            for e in it:
                name = e.name
                if not name.endswith("-bekern.txt"):
                    continue
                stem = name[: -len("-bekern.txt")]
                if stem in seen_stems:
                    continue
                if is_excluded(stem, policy) or is_excluded(sysdata_dir / name, policy):
                    excluded_count += 1
                    continue
                seen_stems.add(stem)
                bek_names.append(name)

        bek_names.sort()
        chunk_size = max(128, (len(bek_names) + workers * 4 - 1) // (workers * 4))
        chunks = [
            (str(sysdata_dir), bek_names[i : i + chunk_size], extreme_ledger_min)
            for i in range(0, len(bek_names), chunk_size)
        ]
        src_hits: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for batch in ex.map(_scan_w3_chunk, chunks):
                src_hits.extend(batch)
        for h in src_hits:
            h["hybrid_jsonl"] = str(hybrid_jsonl)
        all_candidates.extend(src_hits)
        print(
            f"[W3] scanned {len(bek_names)} strips in {sysdata_dir.name}: "
            f"{len(src_hits)} rare-feature hits ({time.time() - t0:.1f}s)",
            flush=True,
        )

    # Pre-select an oversampled pool (1.5x) so any hybrid-row skips still hit target_w3.
    pool_limit = int(target_w3 * 1.5) + 200
    pool = _select_balanced_w3(
        all_candidates,
        target=pool_limit,
        max_per_score=max_per_score,
        seed=seed,
    )

    # Fetch hybrid_staff.jsonl rows grouped by source jsonl.
    by_jsonl: Dict[str, Set[str]] = collections.defaultdict(set)
    for c in pool:
        by_jsonl[c["hybrid_jsonl"]].add(c["img"])
    staff_map: Dict[str, Dict[str, Any]] = {}
    for jpath_s, wanted in by_jsonl.items():
        matched = load_matching_hybrid_rows(Path(jpath_s), wanted)
        staff_map.update(matched)
        print(
            f"[W3] matched {len(matched)}/{len(wanted)} hybrid rows from {Path(jpath_s).parent.name}",
            flush=True,
        )

    # Filter pool to validated staff rows, then run final balanced selection.
    valid_pool: List[Dict[str, Any]] = []
    for c in pool:
        row = staff_map.get(c["img"])
        if row is None:
            continue
        png_path = Path(c["sysdata_dir"]) / c["img"]
        if not validate_staff_row_against_png(row, png_path):
            continue
        c["staff_row"] = row
        valid_pool.append(c)

    chosen = _select_balanced_w3(
        valid_pool,
        target=target_w3,
        max_per_score=max_per_score,
        seed=seed + 1,
    )

    emitted_rows: List[Dict[str, Any]] = []
    for c in chosen:
        src_dir = Path(c["sysdata_dir"])
        stem = c["stem"]
        link_or_copy(src_dir / f"{stem}.png", out_dir / f"{stem}.png")
        link_or_copy(src_dir / f"{stem}-bekern.txt", out_dir / f"{stem}-bekern.txt")
        emitted_rows.append(c["staff_row"])

    stats = {
        "candidates_scanned_hits": len(all_candidates),
        "excluded_skipped": excluded_count,
        "valid_with_hybrid": len(valid_pool),
        "emitted": len(emitted_rows),
        "elapsed_s": round(time.time() - t0, 2),
    }
    print(f"[W3] emitted {len(emitted_rows)} strips in {stats['elapsed_s']}s", flush=True)
    return emitted_rows, stats


# ---------------------------------------------------------------------------
# Part B: W2 Long-System Recovery (512 < tokens <= 768)
# ---------------------------------------------------------------------------


def load_symbox_staves_with_lines(
    box_path: Path, img_w: int, img_h: int
) -> Tuple[List[Dict[str, Any]], float]:
    """Load symbox staves preserving scaled 5-line y-coordinates."""
    d = json.loads(box_path.read_text(encoding="utf-8"))
    sx, sy = 1.0, 1.0
    pw, ph = d.get("png_size", [img_w, img_h])
    if abs(pw - img_w) > 1 or abs(ph - img_h) > 1:
        sx, sy = img_w / pw, img_h / ph
    staves: List[Dict[str, Any]] = []
    gaps: List[float] = []
    for s in d["staves"]:
        x1, y1, x2, y2 = s["bbox"]
        ls = [float(y) * sy for y in s["lines"]]
        if len(ls) != 5:
            raise ValueError(f"non-5-line symbox staff: {len(ls)}")
        for i in range(len(ls) - 1):
            gaps.append(ls[i + 1] - ls[i])
        staves.append(
            {
                "x1": float(x1) * sx,
                "y1": float(y1) * sy,
                "x2": float(x2) * sx,
                "y2": float(y2) * sy,
                "lines": ls,
            }
        )
    staves.sort(key=lambda s: s["y1"])
    return staves, float(statistics.median(gaps))


def discover_leland_long_candidates(
    leland_sysdata: Path,
    leland_renders: Path,
    leland_symbox: Path,
    policy: build_unlabeled_10k.ExclusionPolicy,
) -> Tuple[Dict[str, List[Tuple[int, int]]], int]:
    """Discover candidate ``(page, sys_idx)`` systems that previously exceeded 512 tokens.

    Reads ``*c0.png`` (previously split systems) and ``emit_failures.jsonl``
    (``split-*`` failures) in ``leland_sysdata``, grouped by ``score_stem``.
    """
    raw_stems: Set[str] = set()
    if leland_sysdata.is_dir():
        with os.scandir(leland_sysdata) as it:
            for e in it:
                if e.name.endswith("c0.png"):
                    raw_stems.add(e.name[: -len("c0.png")])
        ef = leland_sysdata / "emit_failures.jsonl"
        if ef.is_file():
            with ef.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if str(r.get("reason", "")).startswith("split-"):
                        s = re.sub(r"c\d+$", "", str(r.get("system", "")))
                        if s:
                            raw_stems.add(s)

    by_score: Dict[str, Set[Tuple[int, int]]] = collections.defaultdict(set)
    excluded_count = 0
    for s in raw_stems:
        m = _SYS_STEM_RE.match(s)
        if not m:
            continue
        score_stem, page_s, sys_s = m.group(1), m.group(2), m.group(3)
        if is_excluded(score_stem, policy) or is_excluded(s, policy):
            excluded_count += 1
            continue
        xml_path = leland_renders / f"{score_stem}.musicxml"
        sym_dir = leland_symbox / score_stem
        if not xml_path.is_file() or not sym_dir.is_dir():
            continue
        by_score[score_stem].add((int(page_s), int(sys_s)))

    return {k: sorted(v) for k, v in sorted(by_score.items())}, excluded_count


def _build_symbox_staff_row(
    img_name: str,
    chunk: Sequence[Dict[str, Any]],
    crop_x1: int,
    crop_y1: int,
    crop_w: int,
    crop_h: int,
) -> Optional[Dict[str, Any]]:
    """Construct and validate a ``hybrid_staff.jsonl`` row from symbox lines."""
    s_val = len(chunk)
    if not (1 <= s_val <= 4) or crop_w < 32 or crop_h < 16:
        return None
    lines5: List[List[float]] = []
    gaps: List[float] = []
    for st in chunk:
        ls = [float(y) - float(crop_y1) for y in st["lines"]]
        if len(ls) != 5:
            return None
        if any(ls[i] >= ls[i + 1] for i in range(4)):
            return None
        if ls[0] < 0.0 or ls[-1] >= float(crop_h):
            return None
        for i in range(4):
            gaps.append(ls[i + 1] - ls[i])
        lines5.append(ls)
    sp = float(statistics.median(gaps))
    if sp <= 1.0:
        return None
    table = build_hybrid_cache.rows_to_table(lines5, crop_w)
    # Verify reconstruction with strip_staff.table_to_curves
    curves = strip_staff.table_to_curves(table)
    if len(curves) != s_val:
        return None
    return {
        "img": img_name,
        "S": s_val,
        "kind": "rows",
        "table": table,
        "sp": round(sp, 2),
        "source": "symbox",
        "skip": False,
    }


def _process_w2_score(
    args: Tuple[str, List[Tuple[int, int]], str, str, str, str, int],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Worker: recover valid 513..768-token systems for one score."""
    (
        score_stem,
        page_sys_list,
        renders_s,
        symbox_s,
        out_dir_s,
        aser_s,
        max_per_score,
    ) = args
    renders_dir = Path(renders_s)
    symbox_dir = Path(symbox_s)
    out_dir = Path(out_dir_s)
    xml_path = renders_dir / f"{score_stem}.musicxml"

    counts: Dict[str, int] = collections.Counter()
    emitted_rows: List[Dict[str, Any]] = []

    try:
        r_sb = subprocess.run(
            [aser_s, "system-breaks", "--input", str(xml_path)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=30,
        )
        if r_sb.returncode != 0:
            counts["system_breaks_fail"] += 1
            return emitted_rows, dict(counts)
        spans_all = json.loads(r_sb.stdout)
        staves_per_part, active = build_sys_data.parse_parts(str(xml_path))
    except Exception:
        counts["score_parse_fail"] += 1
        return emitted_rows, dict(counts)

    by_page: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for sp in spans_all:
        by_page[int(sp["page"])].append(sp)
    for pg in by_page.values():
        pg.sort(key=lambda s: int(s["system_idx"]))

    # Group requested systems by page so each page PNG/symbox is loaded once.
    req_by_page: Dict[int, List[int]] = collections.defaultdict(list)
    for page, sys_idx in page_sys_list:
        req_by_page[page].append(sys_idx)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        for page, sys_indices in sorted(req_by_page.items()):
            if len(emitted_rows) >= max_per_score:
                break
            if page not in by_page:
                counts["page_no_spans"] += 1
                continue
            png_path = renders_dir / f"{score_stem}-{page}.png"
            box_path = symbox_dir / score_stem / f"{score_stem}-{page}-boxes.json"
            if not png_path.is_file() or not box_path.is_file():
                counts["missing_page_or_box"] += 1
                continue
            try:
                with Image.open(png_path) as im:
                    img_w, img_h = im.size
                    staves, space = load_symbox_staves_with_lines(box_path, img_w, img_h)
                    page_glyphs = build_sys_data.load_symbox_glyphs(str(box_path), img_w, img_h)
                    page_spans = by_page[page]
                    groups = build_sys_data.group_page(
                        staves, space, page_spans, staves_per_part, active
                    )
                    if groups is None:
                        counts["group_page_fail"] += 1
                        continue
                    n_sys = len(groups)
                    for sys_idx in sys_indices:
                        if len(emitted_rows) >= max_per_score:
                            break
                        if not (0 <= sys_idx < n_sys):
                            continue
                        chunk = groups[sys_idx]
                        if not (1 <= len(chunk) <= 4):
                            counts["bad_staff_count"] += 1
                            continue
                        span = page_spans[sys_idx]
                        lo = int(span["start_measure"])
                        hi = int(span["end_measure"])
                        sys_stem = f"{score_stem}-{page}-s{sys_idx}"

                        # 1. Run post-I1 bekern --measures
                        r_bk = subprocess.run(
                            [
                                aser_s,
                                "bekern",
                                "--input",
                                str(xml_path),
                                "--measures",
                                f"{lo}..{hi}",
                                "--out",
                                tmpdir,
                                "--stem",
                                sys_stem,
                            ],
                            capture_output=True,
                            text=True,
                            cwd=str(REPO_ROOT),
                            timeout=30,
                        )
                        bek_tmp = tmp_p / f"{sys_stem}-bekern.txt"
                        if r_bk.returncode != 0 or not bek_tmp.is_file():
                            counts["bekern_fail"] += 1
                            continue
                        bek_text = bek_tmp.read_text(encoding="utf-8")
                        toks = bek_text.split()
                        if not is_long_512_768(toks):
                            if len(toks) <= 512:
                                counts["tokens_le_512"] += 1
                            else:
                                counts["tokens_gt_768"] += 1
                            continue
                        if find_oov_tokens(toks):
                            counts["oov_reject"] += 1
                            continue
                        _, bad_bars = count_i1_bars(toks)
                        if bad_bars > 0:
                            counts["bad_i1_reject"] += 1
                            continue

                        # 2. Run bekern-decode round-trip gate
                        dec_out = tmp_p / f"{sys_stem}-dec.json"
                        r_dec = subprocess.run(
                            [
                                aser_s,
                                "bekern-decode",
                                "--input",
                                str(bek_tmp),
                                "--out",
                                str(dec_out),
                            ],
                            capture_output=True,
                            text=True,
                            cwd=str(REPO_ROOT),
                            timeout=30,
                        )
                        if r_dec.returncode != 0:
                            counts["decode_fail"] += 1
                            continue

                        # 3. Compute glyph-aware crop box & symbox staff table
                        ty1 = (
                            float("-inf")
                            if sys_idx == 0
                            else (groups[sys_idx - 1][-1]["y2"] + chunk[0]["y1"]) / 2.0
                        )
                        ty2 = (
                            float("inf")
                            if sys_idx == n_sys - 1
                            else (chunk[-1]["y2"] + groups[sys_idx + 1][0]["y1"]) / 2.0
                        )
                        if page_glyphs:
                            x1b = min(s["x1"] for s in chunk) - build_sys_data.PAD_H_SPACES * space
                            x2b = max(s["x2"] for s in chunk) + build_sys_data.PAD_H_SPACES * space
                            prev = (
                                [
                                    g
                                    for g in page_glyphs
                                    if g["bbox"][2] >= x1b
                                    and g["bbox"][0] <= x2b
                                    and (g["bbox"][1] + g["bbox"][3]) / 2.0 < ty1
                                ]
                                if sys_idx > 0
                                else []
                            )
                            nxt = (
                                [
                                    g
                                    for g in page_glyphs
                                    if g["bbox"][2] >= x1b
                                    and g["bbox"][0] <= x2b
                                    and (g["bbox"][1] + g["bbox"][3]) / 2.0 >= ty2
                                ]
                                if sys_idx < n_sys - 1
                                else []
                            )
                            cap_up = (
                                max([groups[sys_idx - 1][-1]["y2"]] + [g["bbox"][3] for g in prev])
                                if sys_idx > 0
                                else None
                            )
                            cap_dn = (
                                min([groups[sys_idx + 1][0]["y1"]] + [g["bbox"][1] for g in nxt])
                                if sys_idx < n_sys - 1
                                else None
                            )
                        else:
                            cap_up = cap_dn = None

                        x1, y1, x2, y2 = build_sys_data.crop_box_for_system(
                            chunk, space, page_glyphs, ty1, ty2, cap_up, cap_dn, img_w, img_h
                        )
                        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                        crop_w, crop_h = ix2 - ix1, iy2 - iy1
                        img_name = f"{sys_stem}.png"
                        staff_row = _build_symbox_staff_row(
                            img_name, chunk, ix1, iy1, crop_w, crop_h
                        )
                        if staff_row is None:
                            counts["staff_table_fail"] += 1
                            continue

                        # 4. Atomic crop + bekern write
                        dst_png = out_dir / img_name
                        tmp_png = out_dir / f"{sys_stem}.png.tmp"
                        im.crop((ix1, iy1, ix2, iy2)).save(tmp_png, format="PNG")
                        os.replace(tmp_png, dst_png)
                        atomic_write_text(out_dir / f"{sys_stem}-bekern.txt", bek_text)

                        emitted_rows.append(staff_row)
                        counts["emitted"] += 1
            except Exception:
                counts["page_error"] += 1
                continue

    return emitted_rows, dict(counts)


def _process_w2_general_score(
    args: Tuple[str, str, Optional[str], str, str, int],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Worker: scan all systems of a 1..4-staff score for 513..768-token strips.

    Uses exact symbox when ``symbox_s`` has ``<score>/<score>-<page>-boxes.json``,
    otherwise falls back to CPU RowNet (``build_sys_data.neural_detect_staves`` +
    ``build_hybrid_cache.clean_lines``) ONLY on pages that already have a
    verified 513..768-token bekern+decode system.
    """
    score_stem, renders_s, symbox_s, out_dir_s, aser_s, max_per_score = args
    renders_dir = Path(renders_s)
    symbox_dir = Path(symbox_s) if symbox_s else None
    out_dir = Path(out_dir_s)
    xml_path = renders_dir / f"{score_stem}.musicxml"

    counts: Dict[str, int] = collections.Counter()
    emitted_rows: List[Dict[str, Any]] = []

    try:
        staves_per_part, active = build_sys_data.parse_parts(str(xml_path))
        total_staves = sum(staves_per_part)
        if not (1 <= total_staves <= 4):
            counts["skip_gt4_staves"] += 1
            return emitted_rows, dict(counts)
        r_sb = subprocess.run(
            [aser_s, "system-breaks", "--input", str(xml_path)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=30,
        )
        if r_sb.returncode != 0:
            counts["system_breaks_fail"] += 1
            return emitted_rows, dict(counts)
        spans_all = json.loads(r_sb.stdout)
    except Exception:
        counts["score_parse_fail"] += 1
        return emitted_rows, dict(counts)

    by_page: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for sp in spans_all:
        by_page[int(sp["page"])].append(sp)
    for pg in by_page.values():
        pg.sort(key=lambda s: int(s["system_idx"]))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        for page, page_spans in sorted(by_page.items()):
            if len(emitted_rows) >= max_per_score:
                break
            png_path = renders_dir / f"{score_stem}-{page}.png"
            if not png_path.is_file():
                continue

            # Pre-filter spans via bekern + decode BEFORE touching image/RowNet.
            passing_spans: List[Tuple[int, str]] = []
            for sys_idx, span in enumerate(page_spans):
                if len(emitted_rows) + len(passing_spans) >= max_per_score:
                    break
                lo = int(span["start_measure"])
                hi = int(span["end_measure"])
                sys_stem = f"{score_stem}-{page}-s{sys_idx}"
                r_bk = subprocess.run(
                    [
                        aser_s,
                        "bekern",
                        "--input",
                        str(xml_path),
                        "--measures",
                        f"{lo}..{hi}",
                        "--out",
                        tmpdir,
                        "--stem",
                        sys_stem,
                    ],
                    capture_output=True,
                    text=True,
                    cwd=str(REPO_ROOT),
                    timeout=30,
                )
                bek_tmp = tmp_p / f"{sys_stem}-bekern.txt"
                if r_bk.returncode != 0 or not bek_tmp.is_file():
                    continue
                bek_text = bek_tmp.read_text(encoding="utf-8")
                toks = bek_text.split()
                if not is_long_512_768(toks):
                    continue
                if find_oov_tokens(toks):
                    counts["oov_reject"] += 1
                    continue
                _, bad_bars = count_i1_bars(toks)
                if bad_bars > 0:
                    counts["bad_i1_reject"] += 1
                    continue
                dec_out = tmp_p / f"{sys_stem}-dec.json"
                r_dec = subprocess.run(
                    [aser_s, "bekern-decode", "--input", str(bek_tmp), "--out", str(dec_out)],
                    capture_output=True,
                    text=True,
                    cwd=str(REPO_ROOT),
                    timeout=30,
                )
                if r_dec.returncode != 0:
                    counts["decode_fail"] += 1
                    continue
                passing_spans.append((sys_idx, bek_text))

            if not passing_spans:
                continue

            # Now load page image & detect staves (symbox preferred, else CPU RowNet).
            box_path = (
                (symbox_dir / score_stem / f"{score_stem}-{page}-boxes.json")
                if symbox_dir is not None
                else None
            )
            use_symbox = box_path is not None and box_path.is_file()
            try:
                with Image.open(png_path) as im:
                    img_w, img_h = im.size
                    page_glyphs: List[Dict[str, Any]] = []
                    if use_symbox:
                        staves, space = load_symbox_staves_with_lines(box_path, img_w, img_h)
                        page_glyphs = build_sys_data.load_symbox_glyphs(str(box_path), img_w, img_h)
                    else:
                        staves, space = build_sys_data.neural_detect_staves(str(png_path))
                    groups = build_sys_data.group_page(
                        staves, space, page_spans, staves_per_part, active
                    )
                    if groups is None:
                        counts["group_page_fail"] += 1
                        continue
                    n_sys = len(groups)
                    for sys_idx, bek_text in passing_spans:
                        if len(emitted_rows) >= max_per_score:
                            break
                        if not (0 <= sys_idx < n_sys):
                            continue
                        chunk = groups[sys_idx]
                        if not (1 <= len(chunk) <= 4):
                            counts["bad_staff_count"] += 1
                            continue
                        sys_stem = f"{score_stem}-{page}-s{sys_idx}"
                        img_name = f"{sys_stem}.png"
                        if use_symbox:
                            ty1 = (
                                float("-inf")
                                if sys_idx == 0
                                else (groups[sys_idx - 1][-1]["y2"] + chunk[0]["y1"]) / 2.0
                            )
                            ty2 = (
                                float("inf")
                                if sys_idx == n_sys - 1
                                else (chunk[-1]["y2"] + groups[sys_idx + 1][0]["y1"]) / 2.0
                            )
                            x1, y1, x2, y2 = build_sys_data.crop_box_for_system(
                                chunk, space, page_glyphs, ty1, ty2, None, None, img_w, img_h
                            )
                            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                            staff_row = _build_symbox_staff_row(
                                img_name, chunk, ix1, iy1, ix2 - ix1, iy2 - iy1
                            )
                            if staff_row is None:
                                counts["staff_table_fail"] += 1
                                continue
                            tmp_png = out_dir / f"{sys_stem}.png.tmp"
                            im.crop((ix1, iy1, ix2, iy2)).save(tmp_png, format="PNG")
                            os.replace(tmp_png, out_dir / img_name)
                        else:
                            x1 = max(0.0, min(s["x1"] for s in chunk) - build_sys_data.PAD_H_SPACES * space)
                            y1 = max(0.0, min(s["y1"] for s in chunk) - build_sys_data.PAD_V_SPACES * space)
                            x2 = min(float(img_w), max(s["x2"] for s in chunk) + build_sys_data.PAD_H_SPACES * space)
                            y2 = min(float(img_h), max(s["y2"] for s in chunk) + build_sys_data.PAD_V_SPACES * space)
                            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                            if ix2 - ix1 < 32 or iy2 - iy1 < 16:
                                counts["staff_table_fail"] += 1
                                continue
                            tmp_png = out_dir / f"{sys_stem}.png.tmp"
                            im.crop((ix1, iy1, ix2, iy2)).save(tmp_png, format="PNG")
                            try:
                                s_det, lines5, sp_det, src_det = build_hybrid_cache.clean_lines(str(tmp_png))
                            except Exception:
                                tmp_png.unlink(missing_ok=True)
                                counts["clean_lines_fail"] += 1
                                continue
                            if s_det != len(chunk) or not (1 <= s_det <= 4):
                                tmp_png.unlink(missing_ok=True)
                                counts["staff_count_mismatch"] += 1
                                continue
                            table = build_hybrid_cache.rows_to_table(lines5, ix2 - ix1)
                            staff_row = {
                                "img": img_name,
                                "S": int(s_det),
                                "kind": "rows",
                                "table": table,
                                "sp": round(float(sp_det), 2),
                                "source": src_det,
                                "skip": False,
                            }
                            if not validate_staff_row_against_png(staff_row, tmp_png):
                                tmp_png.unlink(missing_ok=True)
                                counts["staff_table_fail"] += 1
                                continue
                            os.replace(tmp_png, out_dir / img_name)

                        atomic_write_text(out_dir / f"{sys_stem}-bekern.txt", bek_text)
                        emitted_rows.append(staff_row)
                        counts["emitted"] += 1
            except Exception:
                counts["page_error"] += 1
                continue

    return emitted_rows, dict(counts)


def mine_w2_general_pool(
    pool_name: str,
    renders_dir: Path,
    symbox_dir: Optional[Path],
    out_dir: Path,
    aser_bin: Path,
    policy: build_unlabeled_10k.ExclusionPolicy,
    *,
    need: int,
    min_xml_bytes: int = 0,
    max_per_score: int = 10,
    workers: int = 8,
    seed: int = 20261123,
    existing_imgs: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Recover up to ``need`` additional 513..768-token systems from a render pool."""
    t0 = time.time()
    if need <= 0 or not renders_dir.is_dir():
        return [], {"pool": pool_name, "emitted": 0, "elapsed_s": 0.0}

    score_stems: List[str] = []
    excluded_count = 0
    with os.scandir(renders_dir) as it:
        for e in it:
            if not e.name.endswith(".musicxml"):
                continue
            if min_xml_bytes > 0 and e.stat().st_size < min_xml_bytes:
                continue
            stem = e.name[: -len(".musicxml")]
            if is_excluded(stem, policy) or is_excluded(renders_dir / e.name, policy):
                excluded_count += 1
                continue
            score_stems.append(stem)

    score_stems.sort()
    rng = random.Random(seed)
    rng.shuffle(score_stems)

    # Warm RowNet ONNX session on main thread before spawning workers if no symbox.
    if symbox_dir is None:
        try:
            import build_gate20_sys_dev as bgate
            bgate.ensure_weights_readable()
        except Exception:
            pass

    print(
        f"[W2:{pool_name}] scanning up to {len(score_stems)} scores for need={need} long strips",
        flush=True,
    )
    emitted_rows: List[Dict[str, Any]] = []
    agg_counts: Dict[str, int] = collections.Counter()
    seen = set(existing_imgs or ())

    wave_size = max(workers * 3, 24)
    idx = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        while idx < len(score_stems) and len(emitted_rows) < need:
            wave = score_stems[idx : idx + wave_size]
            idx += len(wave)
            job_args = [
                (
                    sc,
                    str(renders_dir),
                    str(symbox_dir) if symbox_dir is not None else None,
                    str(out_dir),
                    str(aser_bin),
                    max_per_score,
                )
                for sc in wave
            ]
            for rows, cts in ex.map(_process_w2_general_score, job_args):
                for k, v in cts.items():
                    agg_counts[k] += v
                for row in rows:
                    if row["img"] in seen:
                        continue
                    if len(emitted_rows) < need:
                        seen.add(row["img"])
                        emitted_rows.append(row)
                    else:
                        img_p = out_dir / row["img"]
                        bek_p = out_dir / (Path(row["img"]).stem + "-bekern.txt")
                        img_p.unlink(missing_ok=True)
                        bek_p.unlink(missing_ok=True)
            print(
                f"[W2:{pool_name}] progress: scores={idx}/{len(score_stems)} "
                f"emitted={len(emitted_rows)}/{need} ({time.time() - t0:.1f}s)",
                flush=True,
            )

    stats = {
        "pool": pool_name,
        "candidate_scores": len(score_stems),
        "scores_scanned": idx,
        "excluded_skipped": excluded_count,
        "emitted": len(emitted_rows),
        "counters": dict(sorted(agg_counts.items())),
        "elapsed_s": round(time.time() - t0, 2),
    }
    print(f"[W2:{pool_name}] emitted {len(emitted_rows)} long strips in {stats['elapsed_s']}s", flush=True)
    return emitted_rows, stats


def mine_w2_long_systems(
    leland_sysdata: Path,
    leland_renders: Path,
    leland_symbox: Path,
    out_dir: Path,
    aser_bin: Path,
    policy: build_unlabeled_10k.ExclusionPolicy,
    *,
    target_w2: int = 1500,
    max_per_score: int = 10,
    workers: int = 8,
    seed: int = 20261123,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Recover up to ``target_w2`` dense 513..768-token systems from leland."""
    t0 = time.time()
    by_score, excluded_count = discover_leland_long_candidates(
        leland_sysdata, leland_renders, leland_symbox, policy
    )
    score_items = list(by_score.items())
    rng = random.Random(seed)
    rng.shuffle(score_items)
    # Deduplicate/spread base pieces so both fonts of a piece aren't clustered first.
    seen_base: Set[str] = set()
    first_pass: List[Tuple[str, List[Tuple[int, int]]]] = []
    second_pass: List[Tuple[str, List[Tuple[int, int]]]] = []
    for score_stem, ps_list in score_items:
        b = base_score_id(score_stem)
        if b not in seen_base:
            seen_base.add(b)
            first_pass.append((score_stem, ps_list))
        else:
            second_pass.append((score_stem, ps_list))
    ordered_scores = first_pass + second_pass

    print(
        f"[W2] discovered {sum(len(v) for v in by_score.values())} >512 candidates "
        f"across {len(ordered_scores)} scores ({len(first_pass)} unique pieces)",
        flush=True,
    )

    emitted_rows: List[Dict[str, Any]] = []
    agg_counts: Dict[str, int] = collections.Counter()

    # Submit in waves so we stop cleanly once target_w2 is reached.
    wave_size = max(workers * 3, 24)
    idx = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        while idx < len(ordered_scores) and len(emitted_rows) < target_w2:
            wave = ordered_scores[idx : idx + wave_size]
            idx += len(wave)
            job_args = [
                (
                    score_stem,
                    ps_list,
                    str(leland_renders),
                    str(leland_symbox),
                    str(out_dir),
                    str(aser_bin),
                    max_per_score,
                )
                for score_stem, ps_list in wave
            ]
            for rows, cts in ex.map(_process_w2_score, job_args):
                for k, v in cts.items():
                    agg_counts[k] += v
                for row in rows:
                    if len(emitted_rows) < target_w2:
                        emitted_rows.append(row)
                    else:
                        # Remove excess emitted files cleanly
                        img_p = out_dir / row["img"]
                        bek_p = out_dir / (Path(row["img"]).stem + "-bekern.txt")
                        if img_p.exists():
                            img_p.unlink()
                        if bek_p.exists():
                            bek_p.unlink()
            print(
                f"[W2] progress: scores={idx}/{len(ordered_scores)} "
                f"emitted={len(emitted_rows)}/{target_w2} ({time.time() - t0:.1f}s)",
                flush=True,
            )

    stats = {
        "candidate_scores": len(ordered_scores),
        "excluded_skipped": excluded_count,
        "emitted": len(emitted_rows),
        "counters": dict(sorted(agg_counts.items())),
        "elapsed_s": round(time.time() - t0, 2),
    }
    print(f"[W2] emitted {len(emitted_rows)} long strips in {stats['elapsed_s']}s", flush=True)
    return emitted_rows, stats


# ---------------------------------------------------------------------------
# Full Independent Verification & Summary
# ---------------------------------------------------------------------------


def verify_output_dir(
    out_dir: Path,
    policy: build_unlabeled_10k.ExclusionPolicy,
    *,
    extreme_ledger_min: int = 4,
    w2_imgs: Optional[Set[str]] = None,
    w3_imgs: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Independently verify every strip PNG, bekern file, and hybrid_staff row."""
    staff_path = out_dir / "hybrid_staff.jsonl"
    staff_rows: Dict[str, Dict[str, Any]] = {}
    if staff_path.is_file():
        for ln in staff_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                staff_rows[r["img"]] = r

    pngs = sorted(out_dir.glob("*.png"))
    beks = sorted(out_dir.glob("*-bekern.txt"))
    png_names = {p.name for p in pngs}
    bek_stems = {p.name[: -len("-bekern.txt")] for p in beks}

    leaked_eval = 0
    nc_count = 0
    oov_tokens = 0
    oov_files = 0
    total_bars = 0
    bad_i1_bars = 0
    bad_i1_files = 0
    bad_s_rows = 0
    missing_pair = 0

    strata_counts: Dict[str, int] = {k: 0 for k in STRATA_KEYS}
    by_staves: Dict[int, int] = collections.Counter()
    by_source: Dict[str, int] = collections.Counter()
    token_lengths: List[int] = []
    long_token_lengths: List[int] = []
    unique_pieces: Set[str] = set()

    for p in pngs:
        if policy.is_leaked_eval(p) or policy.is_leaked_eval(p.stem) or policy.is_leaked_eval(base_score_id(p.stem)):
            leaked_eval += 1
        if policy.is_forbidden_nc(p):
            nc_count += 1
        if p.stem not in bek_stems:
            missing_pair += 1
        srow = staff_rows.get(p.name)
        if not srow or srow.get("skip") or not validate_staff_row_against_png(srow, p):
            bad_s_rows += 1
        else:
            by_staves[int(srow["S"])] += 1
            by_source[str(srow.get("source", "unknown"))] += 1
        unique_pieces.add(base_score_id(p.stem))

    for bp in beks:
        stem = bp.name[: -len("-bekern.txt")]
        if f"{stem}.png" not in png_names:
            missing_pair += 1
        text = bp.read_text(encoding="utf-8")
        toks = text.split()
        token_lengths.append(len(toks))
        oov = find_oov_tokens(toks)
        if oov:
            oov_tokens += len(oov)
            oov_files += 1
        tb, bb = count_i1_bars(toks)
        total_bars += tb
        bad_i1_bars += bb
        if bb:
            bad_i1_files += 1
        flags = classify_strata(toks, extreme_ledger_min=extreme_ledger_min)
        for k, v in flags.items():
            if v:
                strata_counts[k] += 1
        if flags["long_512_768"]:
            long_token_lengths.append(len(toks))

    token_lengths.sort()
    long_token_lengths.sort()

    def _pct(vals: List[int], q: float) -> int:
        if not vals:
            return 0
        idx = min(len(vals) - 1, int(round((q / 100.0) * (len(vals) - 1))))
        return int(vals[idx])

    return {
        "total_strip_pngs": len(pngs),
        "total_bekern_files": len(beks),
        "total_staff_jsonl_rows": len(staff_rows),
        "unique_pieces": len(unique_pieces),
        "w2_emitted": len(w2_imgs) if w2_imgs is not None else strata_counts["long_512_768"],
        "w3_emitted": len(w3_imgs) if w3_imgs is not None else (len(pngs) - strata_counts["long_512_768"]),
        "strata_counts": strata_counts,
        "by_staves": {str(k): v for k, v in sorted(by_staves.items())},
        "by_staff_source": dict(sorted(by_source.items())),
        "tokens_all": {
            "min": token_lengths[0] if token_lengths else 0,
            "p50": _pct(token_lengths, 50),
            "p95": _pct(token_lengths, 95),
            "max": token_lengths[-1] if token_lengths else 0,
        },
        "tokens_long_512_768": {
            "count": len(long_token_lengths),
            "min": long_token_lengths[0] if long_token_lengths else 0,
            "p50": _pct(long_token_lengths, 50),
            "p95": _pct(long_token_lengths, 95),
            "max": long_token_lengths[-1] if long_token_lengths else 0,
        },
        "leaked_eval": leaked_eval,
        "nc_count": nc_count,
        "oov_tokens": oov_tokens,
        "oov_files": oov_files,
        "total_bars": total_bars,
        "bad_i1_bars": bad_i1_bars,
        "bad_i1_files": bad_i1_files,
        "bad_s_rows": bad_s_rows,
        "missing_pair": missing_pair,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build W2/W3 rare strata & long-system (512-768 tok) dataset (CPU-only)."
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--unified-sysdata", type=Path, default=DEFAULT_UNIFIED_SYSDATA)
    ap.add_argument("--unified-hybrid", type=Path, default=DEFAULT_UNIFIED_HYBRID)
    ap.add_argument("--quota-sysdata", type=Path, default=DEFAULT_QUOTA_SYSDATA)
    ap.add_argument("--quota-hybrid", type=Path, default=DEFAULT_QUOTA_HYBRID)
    ap.add_argument("--leland-renders", type=Path, default=DEFAULT_LELAND_RENDERS)
    ap.add_argument("--leland-symbox", type=Path, default=DEFAULT_LELAND_SYMBOX)
    ap.add_argument("--leland-sysdata", type=Path, default=DEFAULT_LELAND_SYSDATA)
    ap.add_argument("--sq-renders", type=Path, default=DEFAULT_SQ_RENDERS)
    ap.add_argument("--sq-symbox", type=Path, default=DEFAULT_SQ_SYMBOX)
    ap.add_argument("--quota-renders", type=Path, default=DEFAULT_QUOTA_RENDERS)
    ap.add_argument("--aser-bin", type=Path, default=DEFAULT_ASER)
    ap.add_argument("--target-w3", type=int, default=1500)
    ap.add_argument("--target-w2", type=int, default=1500)
    ap.add_argument("--extreme-ledger-min", type=int, default=4)
    ap.add_argument("--max-per-score-w3", type=int, default=4)
    ap.add_argument("--max-per-score-w2", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--clean-out", action="store_true", help="Remove existing files in --out-dir first.")
    args = ap.parse_args(argv)

    t_start = time.time()
    out_dir: Path = args.out_dir
    if args.clean_out and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.aser_bin.is_file():
        print(f"ERROR: assembly-serialize binary missing: {args.aser_bin}", file=sys.stderr)
        return 1

    policy = build_unlabeled_10k.ExclusionPolicy.from_repo(REPO_ROOT)
    print(
        f"[init] ExclusionPolicy: gate20_ids={len(policy.gate20_ids)}, "
        f"gate20_scores={len(policy.gate20_scores)}, "
        f"suite31_slugs={len(policy.suite31_slugs)}, "
        f"hw_works={len(policy.held_out_hw_works)}",
        flush=True,
    )

    # Part A: W3 Rare-Feature Mining
    w3_sources = [
        (args.unified_sysdata, args.unified_hybrid),
        (args.quota_sysdata, args.quota_hybrid),
    ]
    w3_rows, w3_stats = mine_w3_rare_strata(
        w3_sources,
        out_dir,
        policy,
        target_w3=args.target_w3,
        extreme_ledger_min=args.extreme_ledger_min,
        max_per_score=args.max_per_score_w3,
        workers=args.workers,
        seed=args.seed,
    )

    # Part B: W2 Long-System Recovery (512 < tokens <= 768)
    w2_rows, w2_leland_stats = mine_w2_long_systems(
        args.leland_sysdata,
        args.leland_renders,
        args.leland_symbox,
        out_dir,
        args.aser_bin,
        policy,
        target_w2=args.target_w2,
        max_per_score=args.max_per_score_w2,
        workers=args.workers,
        seed=args.seed + 17,
    )
    w2_seen = {r["img"] for r in w2_rows}
    w2_sq_stats: Dict[str, Any] = {}
    w2_quota_stats: Dict[str, Any] = {}

    # Top up from sq-renders + sq-symbox (dense 4-staff string quartets, exact symbox)
    if len(w2_rows) < args.target_w2:
        sq_rows, w2_sq_stats = mine_w2_general_pool(
            "sq-symbox",
            args.sq_renders,
            args.sq_symbox,
            out_dir,
            args.aser_bin,
            policy,
            need=args.target_w2 - len(w2_rows),
            min_xml_bytes=0,
            max_per_score=args.max_per_score_w2,
            workers=args.workers,
            seed=args.seed + 29,
            existing_imgs=w2_seen,
        )
        w2_rows.extend(sq_rows)
        w2_seen.update(r["img"] for r in sq_rows)

    # Top up from quota-renders (dense >150KB 1..4-staff scores, CPU RowNet staff tables)
    if len(w2_rows) < args.target_w2:
        q_rows, w2_quota_stats = mine_w2_general_pool(
            "quota-renders",
            args.quota_renders,
            None,
            out_dir,
            args.aser_bin,
            policy,
            need=args.target_w2 - len(w2_rows),
            min_xml_bytes=150_000,
            max_per_score=args.max_per_score_w2,
            workers=args.workers,
            seed=args.seed + 41,
            existing_imgs=w2_seen,
        )
        w2_rows.extend(q_rows)
        w2_seen.update(r["img"] for r in q_rows)

    w2_stats = {
        "leland": w2_leland_stats,
        "sq_symbox": w2_sq_stats,
        "quota_renders": w2_quota_stats,
        "total_emitted": len(w2_rows),
    }

    # Deduplicate hybrid_staff.jsonl rows by img (W2 takes precedence if any overlap).
    merged_rows: Dict[str, Dict[str, Any]] = {}
    for r in w3_rows:
        merged_rows[r["img"]] = r
    for r in w2_rows:
        merged_rows[r["img"]] = r

    jsonl_lines = [
        json.dumps(merged_rows[k], separators=(",", ":"))
        for k in sorted(merged_rows.keys())
    ]
    atomic_write_text(out_dir / "hybrid_staff.jsonl", "\n".join(jsonl_lines) + "\n")

    summary = verify_output_dir(
        out_dir,
        policy,
        extreme_ledger_min=args.extreme_ledger_min,
        w2_imgs={r["img"] for r in w2_rows},
        w3_imgs={r["img"] for r in w3_rows},
    )
    summary["w3_mining_stats"] = w3_stats
    summary["w2_recovery_stats"] = w2_stats
    summary["total_elapsed_s"] = round(time.time() - t_start, 2)
    atomic_write_text(out_dir / "summary.json", json.dumps(summary, indent=2) + "\n")

    print("=== W2/W3 SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    # Fail loud on any safety/quality gate violation.
    failures = []
    for key in ("leaked_eval", "nc_count", "oov_tokens", "bad_i1_bars", "bad_s_rows", "missing_pair"):
        if summary[key] != 0:
            failures.append(f"{key}={summary[key]}")
    if failures:
        print("FATAL: quality/safety gate violations: " + ", ".join(failures), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
