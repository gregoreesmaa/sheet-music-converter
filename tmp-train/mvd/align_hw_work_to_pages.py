#!/usr/bin/env python3
"""Auto-align real archival handwritten manuscript pages to work-level GT bekern/MusicXML.

Given work-level ground-truth MusicXML or full-work ``-bekern.txt`` files and ordered
page-level UniOMR predictions (``<sys>``-separated systems, ``bar``-separated measures),
this tool:

  1. Enforces **strict zero-leakage exclusion** against ``suite31_hw_manifest.json``
     (all 45 pieces, including the 5 SBB Berlin Lieder and ``mozart-k458``) and
     ``gate20/exclusion.json``.
  2. Runs **monotone dynamic programming** over ``(page k, measure index m)`` to find
     the globally optimal contiguous measure partition ``[m_0..m_1), ..., [m_{K-1}..m_K)``
     across ordered manuscript pages, allowing skipped blank/cover/title pages at the
     start, interior, or end.
  3. Runs **intra-page system-break DP** over predicted ``<sys>`` chunks vs the page's
     assigned GT measures to recover exact per-system measure spans.
  4. Re-emits clean, lossless per-system GT ``bekern`` files with carried clef/key/time
     headers at the start of every ``<sys>`` (either via a fast header-carrying Python
     slicer or ``assembly-serialize.exe bekern --measures A..B``).
  5. Applies **strict confidence gating** (``--min-align-score``, ``--min-bar-ratio``,
     ``0`` OOV against ``data.VOCAB``, and valid ``grammar.PrefixState`` structure) and
     writes a ``manifest.jsonl`` matching ``D:\\tmp\\uni_fullpage_iter7.jsonl``
     (``{"image": ..., "bekerns": [...], "domain": "real_hw", ...}``).

Supports both:
  - ``--preds-json <path>``: pure CPU alignment on cached predictions (never imports torch).
  - ``--ckpt <path>``: runs UniOMR inference (GPU or CPU) and optionally caches predictions
    with ``--save-preds-json``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import data  # noqa: E402
import grammar  # noqa: E402

SYS_TOKEN = "<sys>"
DEFAULT_SUITE31_MANIFEST = (
    REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"
)
DEFAULT_GATE20_EXCLUSION = (
    REPO_ROOT / "data-gen" / "score-corpus" / "gate20" / "exclusion.json"
)
DEFAULT_ASER_BIN = (
    REPO_ROOT
    / "converter"
    / "assembly-serialize"
    / "target"
    / "release"
    / "assembly-serialize.exe"
)
DEFAULT_CORPUS_DIRS = [
    r"D:\tmp\pdm-openscore-sq-hw",
    r"D:\tmp\pdm-openscore-hw",
    r"D:\tmp\pdm-bach-chopin-hw",
    r"D:\tmp\nifc-polish-hw",
]

BAR_CLOSING_TOKENS: Set[str] = {
    "bar",
    "bar-double",
    "bar-light-heavy",
    "bar-repeat-right",
}
BAR_ALL_TOKENS: Set[str] = BAR_CLOSING_TOKENS | {"bar-repeat-left"}

DURATION_TOKENS: Set[str] = {
    "breve",
    "whole",
    "half",
    "quarter",
    "eighth",
    "16th",
    "32nd",
    "64th",
    "128th",
}
STEP_TOKENS: Set[str] = {"A", "B", "C", "D", "E", "F", "G"}
OCTAVE_TOKENS: Set[str] = {str(i) for i in range(-1, 9)}
ALTER_TOKENS: Set[str] = {"a-2", "a-1", "a1", "a2"}


# ---------------------------------------------------------------------------
# 1. Strict Zero-Leakage Exclusion
# ---------------------------------------------------------------------------

def _norm_slug(text: str) -> str:
    """Normalize a work slug/identifier to lowercase alphanumeric-hyphen form."""
    s = str(text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def _strip_movement_suffix(slug: str) -> str:
    """Strip trailing movement/part index like '-1', '-2', '-3' when preceded by a work key."""
    m = re.match(r"^(.+?[a-z]\d+[a-z0-9]*)[-_](\d+)$", slug)
    if m:
        return m.group(1)
    return slug


@dataclass
class ExclusionSet:
    """Strict zero-leakage guard blocking all suite31_hw (45) and gate20 works."""

    exact_slugs: Set[str] = field(default_factory=set)
    base_work_keys: Set[str] = field(default_factory=set)
    gate20_ids: Set[str] = field(default_factory=set)
    gate20_prefixes: Set[str] = field(default_factory=set)
    catalog_patterns: List[Tuple[str, str, re.Pattern[str]]] = field(default_factory=list)

    @classmethod
    def from_files(
        cls,
        suite31_manifest: Path | str = DEFAULT_SUITE31_MANIFEST,
        gate20_exclusion: Path | str = DEFAULT_GATE20_EXCLUSION,
    ) -> "ExclusionSet":
        ex = cls()

        s31_path = Path(suite31_manifest)
        if not s31_path.is_file():
            raise FileNotFoundError(f"Required suite31_hw manifest not found: {s31_path}")
        s31 = json.loads(s31_path.read_text(encoding="utf-8"))
        for piece in s31.get("pieces", []):
            raw_slug = str(piece.get("slug", "")).strip()
            if not raw_slug:
                continue
            norm = _norm_slug(raw_slug)
            ex.exact_slugs.add(norm)
            if norm.startswith("hw-"):
                stripped = norm[3:]
                ex.exact_slugs.add(stripped)
                ex.base_work_keys.add(stripped)
            else:
                ex.base_work_keys.add(norm)
                base_mvt = _strip_movement_suffix(norm)
                if base_mvt != norm:
                    ex.exact_slugs.add(base_mvt)
                    ex.base_work_keys.add(base_mvt)

            # Extract composer + catalog number patterns (e.g. mozart + k458, beethoven + op135)
            bare = norm[3:] if norm.startswith("hw-") else norm
            m_cat = re.match(
                r"^([a-z]+)-(k|kv|bwv|op|d|hob|h)(\d+[a-z0-9]*)",
                bare,
            )
            if m_cat:
                comp, cat_pfx, cat_num = m_cat.group(1), m_cat.group(2), m_cat.group(3)
                # Match e.g. mozart-...-k-458, mozart-...-kv-458, mozart-k458
                if cat_pfx in ("k", "kv"):
                    cat_re = rf"(?:kv?|k\.?v?\.?)[- ]*0*{re.escape(cat_num)}(?:\b|[^0-9])"
                elif cat_pfx == "op":
                    cat_re = rf"op\.?[- ]*0*{re.escape(cat_num)}(?:\b|[^0-9])"
                else:
                    cat_re = rf"{re.escape(cat_pfx)}\.?[- ]*0*{re.escape(cat_num)}(?:\b|[^0-9])"
                pat = re.compile(rf"(?:^|[^a-z]){re.escape(comp)}(?:$|[^a-z]).*{cat_re}", re.I)
                ex.catalog_patterns.append((raw_slug, f"{comp}-{cat_pfx}{cat_num}", pat))

        g20_path = Path(gate20_exclusion)
        if not g20_path.is_file():
            raise FileNotFoundError(f"Required gate20 exclusion file not found: {g20_path}")
        g20 = json.loads(g20_path.read_text(encoding="utf-8"))
        for key in ("do_not_train", "held_out_4page_gate"):
            for item in g20.get(key, []):
                norm_item = _norm_slug(item)
                if not norm_item:
                    continue
                ex.gate20_ids.add(norm_item)
                # Also block work-level stem e.g. ds0072-3 -> ds0072, sq7070319-1 -> sq7070319
                stem = re.sub(r"-\d+$", "", norm_item)
                if stem:
                    ex.gate20_prefixes.add(stem)

        return ex

    def is_excluded(
        self,
        work_id: str,
        *,
        composer: str = "",
        title: str = "",
        rism_label: str = "",
        catalog_keys: Sequence[str] = (),
        openscore_mxl: str = "",
    ) -> Tuple[bool, str]:
        """Return ``(True, reason)`` if the work matches ANY suite31_hw or gate20 entry."""
        norm_id = _norm_slug(work_id)
        bare_id = norm_id[3:] if norm_id.startswith("hw-") else norm_id

        # 1. Exact suite31_hw slug or stripped hw- slug
        if norm_id in self.exact_slugs or bare_id in self.exact_slugs:
            return True, f"suite31_hw_exact:{norm_id}"

        # 2. Substring / prefix match against suite31_hw base work keys
        for bk in self.base_work_keys:
            if len(bk) >= 6 and (
                norm_id == bk
                or norm_id.startswith(bk + "-")
                or bare_id == bk
                or bare_id.startswith(bk + "-")
                or f"-{bk}-" in f"-{norm_id}-"
            ):
                return True, f"suite31_hw_key:{bk}"

        # 3. gate20 exact page ID or work stem
        if norm_id in self.gate20_ids:
            return True, f"gate20_exact:{norm_id}"
        for gp in self.gate20_prefixes:
            if (
                norm_id == gp
                or norm_id.startswith(gp + "-")
                or (openscore_mxl and gp in _norm_slug(openscore_mxl))
            ):
                return True, f"gate20_prefix:{gp}"

        # 4. Composer + catalog number match across work_id and metadata fields
        haystack = " ".join(
            [
                work_id,
                composer,
                title,
                rism_label,
                " ".join(str(c) for c in catalog_keys),
                openscore_mxl,
            ]
        )
        for src_slug, tag, pat in self.catalog_patterns:
            if pat.search(haystack):
                return True, f"suite31_hw_catalog:{tag}({src_slug})"

        return False, ""


# ---------------------------------------------------------------------------
# 2. Bekern Parsing, Measure Splitting & Header-Carrying Slicing
# ---------------------------------------------------------------------------

def is_clef(tok: str) -> bool:
    return tok.startswith("clef-")


def is_key(tok: str) -> bool:
    return tok.startswith("key") and len(tok) >= 5 and tok[3] in ("+", "-")


def is_time(tok: str) -> bool:
    return tok.startswith("time-")


def is_header_token(tok: str) -> bool:
    return is_clef(tok) or is_key(tok) or is_time(tok)


@dataclass
class StaffHeader:
    clef: str
    key: str
    time: str

    def to_tokens(self) -> List[str]:
        out: List[str] = []
        if self.clef:
            out.append(self.clef)
        if self.key:
            out.append(self.key)
        if self.time:
            out.append(self.time)
        return out

    def copy(self) -> "StaffHeader":
        return StaffHeader(self.clef, self.key, self.time)


@dataclass
class ParsedWorkBekern:
    """Work-level bekern split into contiguous measures with exact carried headers."""

    measures: List[List[str]]  # body tokens per measure (excluding opening work header)
    headers_before: List[List[StaffHeader]]  # carried staff headers before each measure m
    initial_headers: List[StaffHeader]

    @property
    def n_measures(self) -> int:
        return len(self.measures)

    def slice_system_tokens(self, start_m: int, end_m: int) -> List[str]:
        """Re-emit clean bekern for contiguous measures ``[start_m, end_m)`` with carried headers.

        Deduplicates leading header tokens if measure ``start_m`` itself starts with a
        redundant header change, while preserving mid-system clef/key/time changes.
        """
        if not (0 <= start_m < end_m <= self.n_measures):
            raise ValueError(
                f"Invalid measure span [{start_m}, {end_m}) for work with {self.n_measures} measures"
            )
        hdr_toks: List[str] = []
        for sh in self.headers_before[start_m]:
            hdr_toks.extend(sh.to_tokens())

        body_toks: List[str] = []
        for m_idx in range(start_m, end_m):
            m_toks = self.measures[m_idx]
            if m_idx == start_m and hdr_toks:
                # If measure start_m begins with header tokens that duplicate or override
                # the carried header at the very start of the system, keep carried headers
                # cleanly without doubling identical tokens.
                k = 0
                while k < len(m_toks) and is_header_token(m_toks[k]):
                    k += 1
                if k > 0:
                    # Apply those leading changes onto a copy of headers_before[start_m]
                    merged = [sh.copy() for sh in self.headers_before[start_m]]
                    _apply_header_tokens_to_staves(merged, m_toks[:k], staff_hint=0)
                    hdr_toks = []
                    for sh in merged:
                        hdr_toks.extend(sh.to_tokens())
                    body_toks.extend(m_toks[k:])
                    continue
            body_toks.extend(m_toks)

        return hdr_toks + body_toks


def _parse_opening_headers(tokens: Sequence[str]) -> Tuple[List[StaffHeader], int]:
    """Parse leading ``(clef, key, time)`` triples from the start of a bekern stream."""
    i = 0
    n = len(tokens)
    staves: List[StaffHeader] = []
    cur: Optional[StaffHeader] = None

    while i < n and is_header_token(tokens[i]):
        tok = tokens[i]
        if is_clef(tok):
            if cur is not None:
                staves.append(cur)
            cur = StaffHeader(clef=tok, key="", time="")
        elif is_key(tok):
            if cur is None:
                cur = StaffHeader(clef="", key=tok, time="")
            else:
                cur.key = tok
        elif is_time(tok):
            if cur is None:
                cur = StaffHeader(clef="", key="", time=tok)
            else:
                cur.time = tok
        i += 1
    if cur is not None:
        staves.append(cur)

    # Propagate shared key/time if only specified once across staves
    if staves:
        first_key = next((s.key for s in staves if s.key), "")
        first_time = next((s.time for s in staves if s.time), "")
        for s in staves:
            if not s.key and first_key:
                s.key = first_key
            if not s.time and first_time:
                s.time = first_time

    return staves, i


def _apply_header_tokens_to_staves(
    staves: List[StaffHeader],
    hdr_toks: Sequence[str],
    staff_hint: int = 0,
) -> None:
    """Update carried staff headers with mid-work clef/key/time changes."""
    if not staves:
        return
    idx = min(max(0, staff_hint), len(staves) - 1)
    seen_clef = False
    for tok in hdr_toks:
        if is_clef(tok):
            if seen_clef and idx + 1 < len(staves):
                idx += 1
            staves[idx].clef = tok
            seen_clef = True
        elif is_key(tok):
            # Key changes usually apply to the current staff (or all staves if single change)
            staves[idx].key = tok
        elif is_time(tok):
            staves[idx].time = tok


def _update_staves_from_measure(staves: List[StaffHeader], measure_toks: Sequence[str]) -> None:
    """Track mid-measure clef/key/time changes into carried staff headers.

    In bekern, mid-work clef/key/time changes for staff ``k`` appear immediately
    *before* ``voice-k`` (e.g. ``time-3-4 voice-1 ... time-3-4 voice-2 ...``).
    We resolve the target staff by looking ahead to the next ``voice-k`` token.
    """
    if not staves:
        return
    n_toks = len(measure_toks)
    n_staves = len(staves)

    def _next_staff_idx(from_i: int, fallback: int) -> int:
        for j in range(from_i + 1, n_toks):
            tj = measure_toks[j]
            if tj.startswith("voice-"):
                try:
                    v = int(tj.split("-")[1])
                    if 1 <= v <= n_staves:
                        return v - 1
                except (IndexError, ValueError):
                    pass
                break
        return min(fallback, n_staves - 1)

    cur_staff = 0
    seen_time_count = 0
    seen_key_count = 0
    last_time = ""
    last_key = ""
    for i, tok in enumerate(measure_toks):
        if tok.startswith("voice-"):
            try:
                v = int(tok.split("-")[1])
                if 1 <= v <= n_staves:
                    cur_staff = v - 1
            except (IndexError, ValueError):
                pass
        elif is_clef(tok):
            staves[_next_staff_idx(i, cur_staff)].clef = tok
        elif is_key(tok):
            staves[_next_staff_idx(i, cur_staff)].key = tok
            seen_key_count += 1
            last_key = tok
        elif is_time(tok):
            staves[_next_staff_idx(i, cur_staff)].time = tok
            seen_time_count += 1
            last_time = tok

    # If a single global time/key change appeared once in a multi-staff measure, propagate it
    if seen_time_count == 1 and last_time and n_staves > 1:
        for s in staves:
            s.time = last_time
    if seen_key_count == 1 and last_key and n_staves > 1:
        for s in staves:
            s.key = last_key


def parse_work_bekern(tokens: Sequence[str]) -> ParsedWorkBekern:
    """Split a full-work bekern token sequence into measures with carried headers."""
    toks = [t for t in tokens if t and t != SYS_TOKEN]
    initial_headers, pos = _parse_opening_headers(toks)
    cur_staves = [sh.copy() for sh in initial_headers]

    measures: List[List[str]] = []
    headers_before: List[List[StaffHeader]] = []
    cur_m: List[str] = []

    i = pos
    n = len(toks)
    while i < n:
        tok = toks[i]
        if tok == "bar-repeat-left":
            # Opening repeat barline: if cur_m is empty, keep it at the start of the
            # upcoming measure rather than creating an empty measure.
            if not cur_m:
                cur_m.append(tok)
            else:
                cur_m.append(tok)
                headers_before.append([sh.copy() for sh in cur_staves])
                measures.append(cur_m)
                _update_staves_from_measure(cur_staves, cur_m)
                cur_m = []
        elif tok in BAR_CLOSING_TOKENS:
            cur_m.append(tok)
            headers_before.append([sh.copy() for sh in cur_staves])
            measures.append(cur_m)
            _update_staves_from_measure(cur_staves, cur_m)
            cur_m = []
        else:
            cur_m.append(tok)
        i += 1

    if cur_m:
        # Trailing tokens without a final barline
        headers_before.append([sh.copy() for sh in cur_staves])
        measures.append(cur_m)

    return ParsedWorkBekern(
        measures=measures,
        headers_before=headers_before,
        initial_headers=initial_headers,
    )


def count_xml_measures(xml_path: Path) -> int:
    """Count the number of ``<measure>`` elements in the first ``<part>`` of a MusicXML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Handle optional namespace
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "part":
            cnt = sum(1 for ch in elem if ch.tag.split("}")[-1] == "measure")
            if cnt > 0:
                return cnt
    return 0


def xml_to_bekern_tokens(
    xml_path: Path,
    *,
    measures_span: Optional[Tuple[int, int]] = None,
    aser_bin: Path = DEFAULT_ASER_BIN,
) -> List[str]:
    """Invoke ``assembly-serialize.exe bekern --measures A..B`` to emit exact GT bekern."""
    if not aser_bin.is_file():
        raise FileNotFoundError(f"assembly-serialize binary not found: {aser_bin}")
    if measures_span is None:
        n_m = count_xml_measures(xml_path)
        if n_m <= 0:
            raise ValueError(f"No measures found in MusicXML: {xml_path}")
        a, b = 1, n_m
    else:
        a, b = measures_span

    with tempfile.TemporaryDirectory(prefix="align_hw_bk_") as td:
        out_dir = Path(td)
        stem = "slice"
        cmd = [
            str(aser_bin),
            "bekern",
            "--input",
            str(xml_path),
            "--out",
            str(out_dir),
            "--measures",
            f"{a}..{b}",
            "--stem",
            stem,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out_file = out_dir / f"{stem}-bekern.txt"
        if res.returncode != 0 or not out_file.is_file():
            raise RuntimeError(
                f"assembly-serialize bekern failed (rc={res.returncode}): "
                f"{(res.stderr or res.stdout).strip()[:300]}"
            )
        return out_file.read_text(encoding="utf-8").split()


# ---------------------------------------------------------------------------
# 3. Event Extraction & Similarity Scoring (Pitch+Duration Bag + Token LCS)
# ---------------------------------------------------------------------------

@dataclass
class EventSignature:
    """Compact musical event representation for fast DP similarity scoring."""

    pd_seq: List[str]  # ordered pitch+duration / rest events
    pd_bag: Counter[str]  # multiset of pitch+duration / rest events
    pitch_bag: Counter[str]  # multiset of pitch-only events (robust to duration noise)
    n_bars: int


def extract_events(tokens: Sequence[str]) -> EventSignature:
    """Extract ordered pitch+duration events, bags, and barline count from bekern tokens."""
    pd_seq: List[str] = []
    pitch_bag: Counter[str] = Counter()
    n_bars = 0

    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok in BAR_CLOSING_TOKENS:
            n_bars += 1
            i += 1
            continue
        if tok == "bar-rest":
            pd_seq.append("BR")
            i += 1
            continue
        if tok == "rest":
            dur = tokens[i + 1] if (i + 1 < n and tokens[i + 1] in DURATION_TOKENS) else ""
            pd_seq.append(f"R:{dur}")
            i += 2 if dur else 1
            continue
        if tok in DURATION_TOKENS:
            # Note event: <dur> <step> <octave> [alter]
            if i + 2 < n and tokens[i + 1] in STEP_TOKENS and tokens[i + 2] in OCTAVE_TOKENS:
                step = tokens[i + 1]
                octv = tokens[i + 2]
                alt = ""
                step_adv = 3
                if i + 3 < n and tokens[i + 3] in ALTER_TOKENS:
                    alt = tokens[i + 3]
                    step_adv = 4
                pitch_key = f"{step}{octv}{alt}"
                pd_seq.append(f"{tok}:{pitch_key}")
                pitch_bag[pitch_key] += 1
                i += step_adv
                continue
        i += 1

    return EventSignature(
        pd_seq=pd_seq,
        pd_bag=Counter(pd_seq),
        pitch_bag=pitch_bag,
        n_bars=n_bars,
    )


def _bag_dice(a: Counter[str], b: Counter[str]) -> float:
    na = sum(a.values())
    nb = sum(b.values())
    if na == 0 and nb == 0:
        return 0.0
    if na == 0 or nb == 0:
        return 0.0
    # Iterate over the smaller Counter for speed
    if len(a) > len(b):
        a, b = b, a
    inter = 0
    for k, va in a.items():
        vb = b.get(k, 0)
        if vb:
            inter += va if va < vb else vb
    return (2.0 * inter) / float(na + nb)


def _subsampled_lcs_f1(seq_a: Sequence[str], seq_b: Sequence[str], max_len: int = 96) -> float:
    """Compute ordered LCS F1 between two event sequences (strided if longer than ``max_len``)."""
    na, nb = len(seq_a), len(seq_b)
    if na == 0 or nb == 0:
        return 0.0
    if na > max_len:
        step_a = na / float(max_len)
        a = [seq_a[int(i * step_a)] for i in range(max_len)]
    else:
        a = list(seq_a)
    if nb > max_len:
        step_b = nb / float(max_len)
        b = [seq_b[int(i * step_b)] for i in range(max_len)]
    else:
        b = list(seq_b)

    la, lb = len(a), len(b)
    prev = [0] * (lb + 1)
    for x in a:
        curr = [0] * (lb + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                curr[j] = prev[j - 1] + 1
            else:
                u = prev[j]
                l = curr[j - 1]
                curr[j] = u if u >= l else l
        prev = curr
    lcs = prev[lb]
    return (2.0 * lcs) / float(la + lb)


def _bar_ratio(pred_bars: int, gt_bars: int) -> float:
    if pred_bars <= 0 or gt_bars <= 0:
        return 0.0
    return float(min(pred_bars, gt_bars)) / float(max(pred_bars, gt_bars))


def compute_span_similarity(
    pred_sig: EventSignature,
    gt_pd_bag: Counter[str],
    gt_pitch_bag: Counter[str],
    gt_pd_seq: Sequence[str],
    gt_bars: int,
    *,
    use_lcs: bool = True,
) -> Tuple[float, float]:
    """Return ``(combined_score, bar_count_ratio)`` in ``[0, 1]``."""
    pd_f1 = _bag_dice(pred_sig.pd_bag, gt_pd_bag)
    pitch_f1 = _bag_dice(pred_sig.pitch_bag, gt_pitch_bag)
    if not pred_sig.pitch_bag and not gt_pitch_bag:
        # Pure rest spans (e.g. bar-rest measures)
        pitch_f1 = pd_f1

    if use_lcs and (pd_f1 > 0.05 or pitch_f1 > 0.05):
        lcs_f1 = _subsampled_lcs_f1(pred_sig.pd_seq, gt_pd_seq)
        content = 0.55 * pd_f1 + 0.25 * pitch_f1 + 0.20 * lcs_f1
    else:
        content = 0.70 * pd_f1 + 0.30 * pitch_f1

    br = _bar_ratio(pred_sig.n_bars, gt_bars)
    if pred_sig.n_bars > 0 and gt_bars > 0:
        score = 0.85 * content + 0.15 * br
    else:
        score = content
    return score, br


# ---------------------------------------------------------------------------
# 4. Monotone DP: Work-to-Page Partition & Intra-Page System-Break Recovery
# ---------------------------------------------------------------------------

@dataclass
class AlignedPage:
    page_index: int  # 0-based index in the work's ordered page list
    image_path: str
    skipped: bool
    measure_span: Optional[Tuple[int, int]]  # 0-based [start, end)
    system_spans: List[Tuple[int, int]]  # 0-based [s_start, s_end) per <sys>
    align_score: float
    bar_ratio: float
    pred_bars: int
    gt_bars: int


def split_pred_systems(pred_tokens: Sequence[str]) -> List[List[str]]:
    """Split a predicted full-page token sequence on ``<sys>`` into non-empty chunks."""
    chunks: List[List[str]] = []
    cur: List[str] = []
    for tok in pred_tokens:
        if tok == SYS_TOKEN:
            if cur:
                chunks.append(cur)
                cur = []
        else:
            cur.append(tok)
    if cur:
        chunks.append(cur)
    return chunks


def _precompute_measure_sigs(work: ParsedWorkBekern) -> List[EventSignature]:
    return [extract_events(m) for m in work.measures]


def align_systems_within_page(
    pred_sys_chunks: Sequence[Sequence[str]],
    measure_sigs: Sequence[EventSignature],
    m_start: int,
    m_end: int,
) -> List[Tuple[int, int]]:
    """Use monotone DP to partition ``[m_start, m_end)`` across predicted ``<sys>`` chunks."""
    L = m_end - m_start
    if L <= 0:
        return []

    # Filter out completely empty / event-free predicted chunks
    non_empty = [c for c in pred_sys_chunks if len(extract_events(c).pd_seq) > 0]
    if len(non_empty) <= 1 or L == 1:
        return [(m_start, m_end)]

    # If more predicted systems than GT measures, iteratively merge the shortest adjacent pair
    chunks: List[List[str]] = [list(c) for c in non_empty]
    while len(chunks) > L:
        # Merge the shortest chunk with its shorter neighbor
        min_idx = min(range(len(chunks)), key=lambda i: len(chunks[i]))
        if min_idx == 0:
            chunks[0] = chunks[0] + chunks[1]
            del chunks[1]
        elif min_idx == len(chunks) - 1:
            chunks[-2] = chunks[-2] + chunks[-1]
            del chunks[-1]
        else:
            if len(chunks[min_idx - 1]) <= len(chunks[min_idx + 1]):
                chunks[min_idx - 1] = chunks[min_idx - 1] + chunks[min_idx]
                del chunks[min_idx]
            else:
                chunks[min_idx] = chunks[min_idx] + chunks[min_idx + 1]
                del chunks[min_idx + 1]

    R = len(chunks)
    if R == 1:
        return [(m_start, m_end)]
    if R == L:
        return [(m_start + i, m_start + i + 1) for i in range(L)]

    pred_sigs = [extract_events(c) for c in chunks]
    sub_sigs = measure_sigs[m_start:m_end]

    # Precompute cumulative bags/seqs for fast O(R * L^2) DP (L is small: typically 4..50)
    NEG_INF = -1e18
    dp: List[List[float]] = [[NEG_INF] * (L + 1) for _ in range(R + 1)]
    back: List[List[int]] = [[-1] * (L + 1) for _ in range(R + 1)]
    dp[0][0] = 0.0

    for r in range(R):
        p_sig = pred_sigs[r]
        # Each remaining system needs at least 1 measure
        min_pos = r
        max_pos = L - (R - r)
        for pos in range(min_pos, max_pos + 1):
            base = dp[r][pos]
            if base <= NEG_INF / 2:
                continue
            pd_bag: Counter[str] = Counter()
            pitch_bag: Counter[str] = Counter()
            pd_seq: List[str] = []
            max_end = L - (R - r - 1)
            for end in range(pos + 1, max_end + 1):
                ms = sub_sigs[end - 1]
                pd_bag.update(ms.pd_bag)
                pitch_bag.update(ms.pitch_bag)
                pd_seq.extend(ms.pd_seq)
                sc, _ = compute_span_similarity(
                    p_sig, pd_bag, pitch_bag, pd_seq, end - pos, use_lcs=True
                )
                cand = base + sc
                if cand > dp[r + 1][end]:
                    dp[r + 1][end] = cand
                    back[r + 1][end] = pos

    if back[R][L] == -1:
        return [(m_start, m_end)]

    spans: List[Tuple[int, int]] = []
    cur = L
    for r in range(R, 0, -1):
        prev = back[r][cur]
        spans.append((m_start + prev, m_start + cur))
        cur = prev
    spans.reverse()
    return spans


def align_work_to_pages(
    work: ParsedWorkBekern,
    page_paths: Sequence[str],
    page_pred_tokens: Sequence[Sequence[str]],
    *,
    max_measures_per_page: int = 80,
    min_page_events: int = 4,
    skip_nonblank_penalty: float = 0.15,
) -> List[AlignedPage]:
    """Globally optimal monotone DP partitioning of GT work measures across ordered pages.

    Allows blank/cover/title pages at the start, interior, or end to be skipped without
    consuming GT measures.
    """
    K = len(page_paths)
    N = work.n_measures
    if K != len(page_pred_tokens):
        raise ValueError(f"len(page_paths)={K} != len(page_pred_tokens)={len(page_pred_tokens)}")
    if K == 0 or N == 0:
        return []

    measure_sigs = _precompute_measure_sigs(work)
    page_sys_chunks = [split_pred_systems(toks) for toks in page_pred_tokens]
    page_sigs = [extract_events([t for t in toks if t != SYS_TOKEN]) for toks in page_pred_tokens]

    NEG_INF = -1e18
    # dp[k][m] = best cumulative score after processing pages 0..k-1 and consuming measures 0..m-1
    dp: List[List[float]] = [[NEG_INF] * (N + 1) for _ in range(K + 1)]
    # back[k][m] = (prev_m, align_score, bar_ratio, skipped)
    back: List[List[Optional[Tuple[int, float, float, bool]]]] = [
        [None] * (N + 1) for _ in range(K + 1)
    ]
    dp[0][0] = 0.0

    for k in range(K):
        p_sig = page_sigs[k]
        is_blankish = len(p_sig.pd_seq) < min_page_events

        for m in range(N + 1):
            base = dp[k][m]
            if base <= NEG_INF / 2:
                continue

            # Transition 1: skip page k (blank cover, title page, or unaligned page)
            skip_gain = 0.0 if is_blankish else skip_nonblank_penalty
            cand_skip = base + skip_gain
            if cand_skip > dp[k + 1][m]:
                dp[k + 1][m] = cand_skip
                back[k + 1][m] = (m, 0.0, 0.0, True)

            # Transition 2: align page k to contiguous GT measures [m, m_end)
            if is_blankish or m >= N:
                continue

            pd_bag: Counter[str] = Counter()
            pitch_bag: Counter[str] = Counter()
            pd_seq: List[str] = []
            hi = min(N, m + max_measures_per_page)
            pred_ev_n = len(p_sig.pd_seq)

            for m_end in range(m + 1, hi + 1):
                ms = measure_sigs[m_end - 1]
                pd_bag.update(ms.pd_bag)
                pitch_bag.update(ms.pitch_bag)
                pd_seq.extend(ms.pd_seq)

                # Fast length-ratio prune once GT span is > 4x larger than predicted events
                if len(pd_seq) > max(40, pred_ev_n * 4) and (m_end - m) > max(4, p_sig.n_bars * 3):
                    break

                # Use LCS only when bag similarity is non-trivial (big speedup on long works)
                fast_pd = _bag_dice(p_sig.pd_bag, pd_bag)
                use_lcs = fast_pd >= 0.12
                sc, br = compute_span_similarity(
                    p_sig, pd_bag, pitch_bag, pd_seq, m_end - m, use_lcs=use_lcs
                )
                cand = base + sc
                if cand > dp[k + 1][m_end]:
                    dp[k + 1][m_end] = cand
                    back[k + 1][m_end] = (m, sc, br, False)

    # Pick best ending measure m_final in [0..N]
    best_m = max(range(N + 1), key=lambda m: dp[K][m])

    # Backtrack
    raw_steps: List[Tuple[int, int, float, float, bool]] = []
    cur_m = best_m
    for k in range(K, 0, -1):
        entry = back[k][cur_m]
        if entry is None:
            raw_steps.append((cur_m, cur_m, 0.0, 0.0, True))
        else:
            prev_m, sc, br, skipped = entry
            raw_steps.append((prev_m, cur_m, sc, br, skipped))
            cur_m = prev_m
    raw_steps.reverse()

    aligned: List[AlignedPage] = []
    for k, (m_start, m_end, sc, br, skipped) in enumerate(raw_steps):
        if skipped or m_start >= m_end:
            aligned.append(
                AlignedPage(
                    page_index=k,
                    image_path=str(page_paths[k]),
                    skipped=True,
                    measure_span=None,
                    system_spans=[],
                    align_score=0.0,
                    bar_ratio=0.0,
                    pred_bars=page_sigs[k].n_bars,
                    gt_bars=0,
                )
            )
        else:
            sys_spans = align_systems_within_page(
                page_sys_chunks[k], measure_sigs, m_start, m_end
            )
            aligned.append(
                AlignedPage(
                    page_index=k,
                    image_path=str(page_paths[k]),
                    skipped=False,
                    measure_span=(m_start, m_end),
                    system_spans=sys_spans,
                    align_score=round(float(sc), 6),
                    bar_ratio=round(float(br), 6),
                    pred_bars=page_sigs[k].n_bars,
                    gt_bars=m_end - m_start,
                )
            )

    return aligned


# ---------------------------------------------------------------------------
# 5. Strict Validation & Confidence Gating
# ---------------------------------------------------------------------------

def validate_system_bekern(tokens: Sequence[str]) -> Tuple[bool, str]:
    """Verify 0 OOV against ``data.VOCAB`` (233 base / 237 UniOMR) and valid bekern grammar."""
    if not tokens:
        return False, "empty_system"

    oov = [t for t in tokens if t not in data.TOKEN_TO_ID]
    if oov:
        return False, f"oov_tokens:{oov[:5]}"

    st = grammar.PrefixState(strict=False)
    for tok in tokens:
        try:
            st.step(tok)
        except ValueError as exc:
            return False, f"grammar_error:{exc}"
        if st.broken:
            return False, "grammar_broken"

    ev = extract_events(tokens)
    if ev.n_bars == 0 and len(ev.pd_seq) == 0:
        return False, "no_musical_content"

    return True, "ok"


def gate_aligned_page(
    ap: AlignedPage,
    system_token_lists: Sequence[Sequence[str]],
    *,
    min_align_score: float = 0.45,
    min_bar_ratio: float = 0.50,
) -> Tuple[bool, str]:
    """Apply confidence, bar-ratio, OOV, and grammar structure gates to an aligned page."""
    if ap.skipped or ap.measure_span is None:
        return False, "skipped_page"
    if ap.align_score < min_align_score:
        return False, f"low_align_score:{ap.align_score:.4f}<{min_align_score:.4f}"
    if ap.pred_bars > 0 and ap.bar_ratio < min_bar_ratio:
        return False, f"low_bar_ratio:{ap.bar_ratio:.4f}<{min_bar_ratio:.4f}"
    if not system_token_lists:
        return False, "no_systems"

    for s_idx, sys_toks in enumerate(system_token_lists):
        ok, reason = validate_system_bekern(sys_toks)
        if not ok:
            return False, f"sys{s_idx}_{reason}"

    return True, "ok"


# ---------------------------------------------------------------------------
# 6. Unified Corpus Loader & Prediction I/O
# ---------------------------------------------------------------------------

@dataclass
class WorkRecord:
    work_id: str
    source_corpus: str
    page_paths: List[str]
    bekern_path: Optional[Path]
    xml_path: Optional[Path]
    composer: str = ""
    title: str = ""
    rism_label: str = ""
    catalog_keys: List[str] = field(default_factory=list)
    openscore_mxl: str = ""


def load_corpus_works(corpus_dirs: Sequence[str | Path]) -> List[WorkRecord]:
    """Load work records from all 4 archival HW corpus manifest layouts."""
    works: List[WorkRecord] = []
    for cdir_raw in corpus_dirs:
        cdir = Path(cdir_raw)
        mpath = cdir / "manifest.jsonl"
        if not mpath.is_file():
            continue
        corpus_name = cdir.name
        for line in mpath.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") and row.get("status") != "ok":
                continue

            wid = str(row.get("slug") or row.get("work_id") or "").strip()
            if not wid:
                continue

            # Resolve ordered page paths
            pages: List[str] = []
            if "page_paths" in row and isinstance(row["page_paths"], list):
                pages = [str(p) for p in row["page_paths"]]
            elif "pages" in row and isinstance(row["pages"], list):
                for pinfo in row["pages"]:
                    if isinstance(pinfo, dict) and pinfo.get("file"):
                        p = Path(pinfo["file"])
                        pages.append(str(p if p.is_absolute() else (cdir / p)))
                    elif isinstance(pinfo, str):
                        p = Path(pinfo)
                        pages.append(str(p if p.is_absolute() else (cdir / p)))

            # Resolve bekern_path (absolute or relative)
            bk_raw = row.get("bekern_path") or row.get("bekern_file")
            bk_path: Optional[Path] = None
            if bk_raw:
                p = Path(bk_raw)
                bk_path = p if p.is_absolute() else (cdir / p)

            # Resolve xml_path (absolute or relative)
            xml_raw = row.get("xml_path") or row.get("xml_file")
            xml_path: Optional[Path] = None
            if xml_raw:
                p = Path(xml_raw)
                xml_path = p if p.is_absolute() else (cdir / p)

            cat_keys = row.get("catalog_keys") or (
                [row["catalog_key"]] if row.get("catalog_key") else []
            )
            works.append(
                WorkRecord(
                    work_id=wid,
                    source_corpus=corpus_name,
                    page_paths=pages,
                    bekern_path=bk_path,
                    xml_path=xml_path,
                    composer=str(row.get("composer", "")),
                    title=str(row.get("title") or row.get("work") or ""),
                    rism_label=str(row.get("rism_label", "")),
                    catalog_keys=[str(c) for c in cat_keys],
                    openscore_mxl=str(row.get("openscore_mxl", "")),
                )
            )
    return works


def _normalize_pred_tokens(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return val.split()
    if isinstance(val, list):
        return [str(t) for t in val]
    if isinstance(val, dict):
        if "tokens" in val:
            return _normalize_pred_tokens(val["tokens"])
        if "pred" in val:
            return _normalize_pred_tokens(val["pred"])
    return []


def load_preds_json(preds_path: Path) -> Dict[str, List[str]]:
    """Load cached page predictions from JSON into a normalized lookup map."""
    raw = json.loads(preds_path.read_text(encoding="utf-8"))
    out: Dict[str, List[str]] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            toks = _normalize_pred_tokens(v)
            out[str(k)] = toks
            out[str(Path(k)).lower()] = toks
            out[Path(k).name.lower()] = toks
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = str(item.get("image") or item.get("page_path") or item.get("path") or "")
            if not key:
                continue
            toks = _normalize_pred_tokens(item)
            out[key] = toks
            out[str(Path(key)).lower()] = toks
            out[Path(key).name.lower()] = toks
    return out


def lookup_page_pred(preds_map: Dict[str, List[str]], page_path: str) -> List[str]:
    if page_path in preds_map:
        return preds_map[page_path]
    low = str(Path(page_path)).lower()
    if low in preds_map:
        return preds_map[low]
    name = Path(page_path).name.lower()
    return preds_map.get(name, [])


def run_uni_inference(
    ckpt_path: Path,
    page_paths: Sequence[str],
    *,
    device_str: str = "cuda",
    max_dim: int = 1280,
    max_len: int = 1024,
    use_grammar: bool = True,
    use_mtp_spec: bool = True,
    bf16: bool = True,
) -> Dict[str, List[str]]:
    """Run UniOMR checkpoint inference over a list of page images (lazy torch import)."""
    from contextlib import nullcontext
    from PIL import Image
    import torch
    import eval_uni_suite31 as eus
    import uni_omr

    device = torch.device(device_str)
    model, cfg, meta = eus.load_uni_checkpoint(ckpt_path, device=device)
    use_bf16 = bool(bf16 and device.type == "cuda")
    _, _, eos_id, _, _, _ = eus._vocab_spec(int(cfg.vocab_size))

    print(
        f"[align_hw] Loaded ckpt={ckpt_path} (step={meta.get('step')}) "
        f"on {device} | pages={len(page_paths)} | max_dim={max_dim} | max_len={max_len}",
        flush=True,
    )

    preds: Dict[str, List[str]] = {}
    for idx, pstr in enumerate(page_paths, 1):
        ppath = Path(pstr)
        if not ppath.is_file():
            preds[pstr] = []
            continue
        with Image.open(ppath) as raw_im:
            pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), max_dim)
        img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
        with torch.no_grad():
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            )
            with ctx:
                if use_mtp_spec:
                    seq = eus.generate_mtp_spec(
                        model,
                        img_t,
                        max_len=max_len,
                        use_grammar=use_grammar,
                        strict_grammar=False,
                        return_stats=False,
                    )
                elif use_grammar:
                    seq = eus.generate_constrained(
                        model, img_t, max_len=max_len, strict=False
                    )
                else:
                    seq = model.generate(img_t, max_len=max_len)
        seq_ids = seq[0].detach().cpu().tolist()
        raw_toks = eus.decode_uni_compat(
            seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True
        )
        if eos_id not in seq_ids[1:]:
            raw_toks = eus.trim_truncated_tokens(raw_toks)
        preds[pstr] = raw_toks
        if idx % 50 == 0 or idx == len(page_paths):
            print(f"[align_hw] Decoded {idx}/{len(page_paths)} pages", flush=True)

    return preds


# ---------------------------------------------------------------------------
# 7. End-to-End Pipeline & CLI
# ---------------------------------------------------------------------------

def align_corpus_pipeline(
    works: Sequence[WorkRecord],
    preds_map: Dict[str, List[str]],
    out_dir: Path,
    exclusion: ExclusionSet,
    *,
    min_align_score: float = 0.45,
    min_bar_ratio: float = 0.50,
    max_measures_per_page: int = 80,
    slicer: str = "python",
    aser_bin: Path = DEFAULT_ASER_BIN,
) -> Dict[str, Any]:
    """Run zero-leakage filtering, DP alignment, gating, and manifest emission."""
    out_dir.mkdir(parents=True, exist_ok=True)
    targets_dir = out_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    summary_path = out_dir / "summary.json"

    stats: Dict[str, Any] = {
        "works_total": len(works),
        "works_excluded_leakage": 0,
        "works_missing_gt": 0,
        "works_aligned": 0,
        "pages_total": 0,
        "pages_skipped_blank": 0,
        "pages_rejected_gate": 0,
        "pages_emitted": 0,
        "systems_emitted": 0,
        "excluded_works": [],
        "rejection_reasons": Counter(),
    }

    manifest_rows: List[Dict[str, Any]] = []

    for work in works:
        is_excl, excl_reason = exclusion.is_excluded(
            work.work_id,
            composer=work.composer,
            title=work.title,
            rism_label=work.rism_label,
            catalog_keys=work.catalog_keys,
            openscore_mxl=work.openscore_mxl,
        )
        if is_excl:
            stats["works_excluded_leakage"] += 1
            stats["excluded_works"].append({"work_id": work.work_id, "reason": excl_reason})
            continue

        # Load full-work GT bekern
        gt_tokens: Optional[List[str]] = None
        if work.bekern_path and work.bekern_path.is_file():
            gt_tokens = work.bekern_path.read_text(encoding="utf-8").split()
        elif work.xml_path and work.xml_path.is_file():
            try:
                gt_tokens = xml_to_bekern_tokens(work.xml_path, aser_bin=aser_bin)
            except Exception as exc:  # noqa: BLE001
                stats["works_missing_gt"] += 1
                stats["rejection_reasons"][f"xml_convert_err:{type(exc).__name__}"] += 1
                continue
        else:
            stats["works_missing_gt"] += 1
            stats["rejection_reasons"]["missing_gt_file"] += 1
            continue

        parsed = parse_work_bekern(gt_tokens)
        if parsed.n_measures == 0:
            stats["works_missing_gt"] += 1
            stats["rejection_reasons"]["zero_measures"] += 1
            continue

        page_preds = [lookup_page_pred(preds_map, p) for p in work.page_paths]
        aligned_pages = align_work_to_pages(
            parsed,
            work.page_paths,
            page_preds,
            max_measures_per_page=max_measures_per_page,
        )
        stats["works_aligned"] += 1
        stats["pages_total"] += len(aligned_pages)

        safe_wid = _norm_slug(work.work_id)
        for ap in aligned_pages:
            if ap.skipped or ap.measure_span is None:
                stats["pages_skipped_blank"] += 1
                continue

            sys_token_lists: List[List[str]] = []
            slice_failed = False
            for s_start, s_end in ap.system_spans:
                if slicer == "assembly-serialize" and work.xml_path and work.xml_path.is_file():
                    try:
                        s_toks = xml_to_bekern_tokens(
                            work.xml_path,
                            measures_span=(s_start + 1, s_end),
                            aser_bin=aser_bin,
                        )
                    except Exception:  # noqa: BLE001
                        slice_failed = True
                        break
                else:
                    s_toks = parsed.slice_system_tokens(s_start, s_end)
                sys_token_lists.append(s_toks)

            if slice_failed:
                stats["pages_rejected_gate"] += 1
                stats["rejection_reasons"]["slicer_error"] += 1
                continue

            ok, reason = gate_aligned_page(
                ap,
                sys_token_lists,
                min_align_score=min_align_score,
                min_bar_ratio=min_bar_ratio,
            )
            if not ok:
                stats["pages_rejected_gate"] += 1
                stats["rejection_reasons"][reason.split(":")[0]] += 1
                continue

            # Emit per-system bekern files atomically (tmp + rename)
            emitted_bekerns: List[str] = []
            for s_idx, s_toks in enumerate(sys_token_lists):
                fname = f"{safe_wid}_p{ap.page_index + 1:03d}-s{s_idx}-bekern.txt"
                fpath = targets_dir / fname
                tmp_path = fpath.with_suffix(".txt.tmp")
                tmp_path.write_text(" ".join(s_toks) + "\n", encoding="utf-8")
                tmp_path.replace(fpath)
                emitted_bekerns.append(str(fpath))

            manifest_rows.append(
                {
                    "image": ap.image_path,
                    "bekerns": emitted_bekerns,
                    "domain": "real_hw",
                    "work_id": work.work_id,
                    "source_corpus": work.source_corpus,
                    "page_index": ap.page_index + 1,
                    "measure_span": [ap.measure_span[0] + 1, ap.measure_span[1]],
                    "system_spans": [[a + 1, b] for a, b in ap.system_spans],
                    "align_score": ap.align_score,
                    "bar_ratio": ap.bar_ratio,
                }
            )
            stats["pages_emitted"] += 1
            stats["systems_emitted"] += len(emitted_bekerns)

    # Atomic write of manifest.jsonl & summary.json
    tmp_m = manifest_path.with_suffix(".jsonl.tmp")
    with open(tmp_m, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_m.replace(manifest_path)

    stats["rejection_reasons"] = dict(stats["rejection_reasons"])
    tmp_s = summary_path.with_suffix(".json.tmp")
    tmp_s.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_s.replace(summary_path)

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Auto-align real archival handwritten manuscript pages to work-level GT bekern/MusicXML."
    )
    p.add_argument(
        "--corpus-dirs",
        nargs="+",
        default=DEFAULT_CORPUS_DIRS,
        help="One or more HW corpus directories containing manifest.jsonl.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for targets/*-bekern.txt, manifest.jsonl, and summary.json.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--preds-json",
        type=Path,
        help="Cached page predictions JSON (pure CPU mode; never imports torch).",
    )
    src.add_argument(
        "--ckpt",
        type=Path,
        help="UniOMR checkpoint .pt to decode all non-excluded pages before alignment.",
    )
    p.add_argument(
        "--save-preds-json",
        type=Path,
        default=None,
        help="When using --ckpt, save decoded page predictions to this JSON for fast CPU re-alignment.",
    )
    p.add_argument("--device", default="cuda", help="Torch device when using --ckpt (default: cuda).")
    p.add_argument("--max-dim", type=int, default=1280, help="Full-page resize max_dim for --ckpt.")
    p.add_argument("--max-len", type=int, default=1024, help="Max decode tokens per page for --ckpt.")
    p.add_argument("--no-grammar", action="store_true", help="Disable grammar masking for --ckpt.")
    p.add_argument("--no-mtp-spec", action="store_true", help="Disable MTP-spec decoding for --ckpt.")
    p.add_argument("--no-bf16", action="store_true", help="Disable bfloat16 autocast for --ckpt.")
    p.add_argument(
        "--min-align-score",
        type=float,
        default=0.45,
        help="Minimum normalized alignment similarity in [0, 1] to emit a page (default: 0.45).",
    )
    p.add_argument(
        "--min-bar-ratio",
        type=float,
        default=0.50,
        help="Minimum min/max bar-count ratio to emit a page (default: 0.50).",
    )
    p.add_argument(
        "--max-measures-per-page",
        type=int,
        default=80,
        help="Maximum GT measures allowed per single page in DP (default: 80).",
    )
    p.add_argument(
        "--slicer",
        choices=["python", "assembly-serialize"],
        default="python",
        help="System bekern slicer: fast header-carrying Python slicer or assembly-serialize.exe.",
    )
    p.add_argument(
        "--suite31-manifest",
        type=Path,
        default=DEFAULT_SUITE31_MANIFEST,
        help="Path to suite31_hw_manifest.json for zero-leakage exclusion.",
    )
    p.add_argument(
        "--gate20-exclusion",
        type=Path,
        default=DEFAULT_GATE20_EXCLUSION,
        help="Path to gate20/exclusion.json for zero-leakage exclusion.",
    )
    p.add_argument(
        "--aser-bin",
        type=Path,
        default=DEFAULT_ASER_BIN,
        help="Path to assembly-serialize.exe.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    exclusion = ExclusionSet.from_files(args.suite31_manifest, args.gate20_exclusion)
    works = load_corpus_works(args.corpus_dirs)

    if args.preds_json is not None:
        preds_map = load_preds_json(args.preds_json)
    else:
        # Filter excluded works BEFORE running GPU inference so we never even decode eval pages
        eligible_pages: List[str] = []
        for w in works:
            is_excl, _ = exclusion.is_excluded(
                w.work_id,
                composer=w.composer,
                title=w.title,
                rism_label=w.rism_label,
                catalog_keys=w.catalog_keys,
                openscore_mxl=w.openscore_mxl,
            )
            if not is_excl:
                eligible_pages.extend(w.page_paths)

        raw_preds = run_uni_inference(
            args.ckpt,
            eligible_pages,
            device_str=args.device,
            max_dim=args.max_dim,
            max_len=args.max_len,
            use_grammar=not args.no_grammar,
            use_mtp_spec=not args.no_mtp_spec,
            bf16=not args.no_bf16,
        )
        if args.save_preds_json is not None:
            args.save_preds_json.parent.mkdir(parents=True, exist_ok=True)
            tmp_p = args.save_preds_json.with_suffix(".json.tmp")
            tmp_p.write_text(
                json.dumps(raw_preds, ensure_ascii=False), encoding="utf-8"
            )
            tmp_p.replace(args.save_preds_json)
        preds_map = {}
        for k, v in raw_preds.items():
            preds_map[k] = v
            preds_map[str(Path(k)).lower()] = v
            preds_map[Path(k).name.lower()] = v

    stats = align_corpus_pipeline(
        works,
        preds_map,
        args.out_dir,
        exclusion,
        min_align_score=args.min_align_score,
        min_bar_ratio=args.min_bar_ratio,
        max_measures_per_page=args.max_measures_per_page,
        slicer=args.slicer,
        aser_bin=args.aser_bin,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
