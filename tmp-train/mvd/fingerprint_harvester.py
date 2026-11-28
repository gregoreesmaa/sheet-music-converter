#!/usr/bin/env python3
"""CPU-only Fingerprint-First Forced Alignment Harvester.

Strictly fingerprint-first symbolic retrieval + measure-level banded-DTW alignment
for harvesting aligned ground-truth scores from noisy UniOMR ``bekern`` page decodes
with **zero title or metadata reliance**.

Pipeline:
  1. **Symbolic Inverted Index (``build_index``):**
     Extracts per-staff, per-measure note sequences from MusicXML (``.mxl`` /
     ``.musicxml`` / ``.xml``), ``bekern`` files, and digital ``bekern`` JSONL
     training manifests (e.g. ``D:\\tmp\\uni_fullpage_iter14_v2a.jsonl``) into a
     fast inverted index of **5-note chromatic/diatonic pitch-interval + quantized
     IOI-ratio n-grams** -> ``(work_idx, measure_idx)`` plus
     ``(n_staves, key_sig, time_sig, n_measures)``.
  2. **Page Fingerprinting & Shazam-Style Retrieval (``retrieve_best_fit``):**
     Extracts the same n-grams from any noisy ``UniOMR`` decoded ``bekern`` page
     stream and runs Hough-style ``(work_idx, measure_offset)`` diagonal voting
     to find the best-fit work & measure window, returning ``top1_score``,
     ``top2_score``, and ``margin = (top1 - top2) / max(1, top1)``.
  3. **Full Banded-DTW Alignment (``align_page_to_work``):**
     Runs measure-level banded-DTW between the noisy decoded page measures and
     the retrieved work's GT measures around the voted offset to lock exact
     ``(m_start, m_end, system_breaks)`` and compute ``align_score``
     (pitch-interval + rhythm F1).
  4. **Strict Rejection Gate (``verify_harvest_candidate``):**
     Discards any match failing:
       (a) exact ``n_staves`` parity,
       (b) ``align_score >= tau_align``,
       (c) ``margin >= tau_margin``,
       (d) zero eval-set leakage against ``suite31_hw``, ``gate20``, and
           ``OMR-NorthStar`` stems.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import xml.etree.ElementTree as ET
import zipfile


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SUITE31_MANIFEST = (
    REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"
)
DEFAULT_GATE20_EXCLUSION = (
    REPO_ROOT / "data-gen" / "score-corpus" / "gate20" / "exclusion.json"
)
DEFAULT_NORTHSTAR_MANIFEST = Path(r"D:\tmp\northstar-eval-pack-v0.2\manifest.json")
DEFAULT_SYNTHETIC_PREFIXES: Tuple[str, ...] = ("hwsim_", "hw_sim_", "synth_")


def _matches_exclude_prefix(
    work_id: str,
    source_path: str = "",
    exclude_prefixes: Sequence[str] = (),
) -> bool:
    """Return True if work_id or source_path stem starts with any excluded prefix (case-insensitive)."""
    if not exclude_prefixes:
        return False
    pfxs = tuple(p.lower() for p in exclude_prefixes if p)
    if not pfxs:
        return False
    wid = (work_id or "").strip().lower()
    if wid and wid.startswith(pfxs):
        return True
    if source_path:
        stem = Path(source_path).stem.strip().lower()
        if stem and stem.startswith(pfxs):
            return True
    return False

SYS_TOKEN = "<sys>"

BAR_CLOSING_TOKENS: Set[str] = {
    "bar",
    "bar-double",
    "bar-light-heavy",
    "bar-repeat-right",
}
BAR_ALL_TOKENS: Set[str] = BAR_CLOSING_TOKENS | {"bar-repeat-left"}

DURATION_TO_QUARTER: Dict[str, float] = {
    "breve": 8.0,
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "16th": 0.25,
    "32nd": 0.125,
    "64th": 0.0625,
    "128th": 0.03125,
}
DURATION_TOKENS: Set[str] = set(DURATION_TO_QUARTER.keys())

STEP_TO_SEMITONE: Dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}
STEP_TO_DIATONIC: Dict[str, int] = {
    "C": 0,
    "D": 1,
    "E": 2,
    "F": 3,
    "G": 4,
    "A": 5,
    "B": 6,
}
STEP_TOKENS: Set[str] = set(STEP_TO_SEMITONE.keys())
OCTAVE_TOKENS: Set[str] = {str(i) for i in range(-1, 10)}
ALTER_TO_SEMITONE: Dict[str, int] = {
    "a-2": -2,
    "a-1": -1,
    "a0": 0,
    "a1": 1,
    "a2": 2,
}

# Quantized IOI ratio thresholds on log2(ioi_{k+1} / ioi_k)
_IOI_LOG2_EDGES: Tuple[float, ...] = (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)


def quantize_ioi_ratio(ratio: float) -> int:
    """Quantize an inter-onset interval ratio into 7 discrete log2 buckets [-3..+3]."""
    if ratio <= 0.0 or not math.isfinite(ratio):
        return 0
    lr = math.log2(max(1e-4, min(1e4, ratio)))
    if lr < -2.5:
        return -3
    if lr < -1.5:
        return -2
    if lr < -0.5:
        return -1
    if lr <= 0.5:
        return 0
    if lr <= 1.5:
        return 1
    if lr <= 2.5:
        return 2
    return 3


def quantize_dur(dur_q: float) -> int:
    """Quantize a note duration (in quarter notes) to a compact integer bucket."""
    if dur_q <= 0.0 or not math.isfinite(dur_q):
        return 0
    # Quarter note = 0, eighth = -2, dotted quarter = +1, half = +2, etc. (half-log2 steps)
    return max(-10, min(10, int(round(math.log2(max(1e-3, dur_q)) * 2.0))))


def _clamp(val: int, lo: int, hi: int) -> int:
    return lo if val < lo else (hi if val > hi else val)


# ---------------------------------------------------------------------------
# 1. Core Musical Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoteEvent:
    """Single skyline note event on one staff within a measure."""

    midi: int
    diatonic: int
    dur_q: float
    onset_q: float
    staff: int


@dataclass
class MeasureSummary:
    """Compact per-measure representation used for fingerprinting & DTW alignment."""

    measure_idx: int
    # staff_idx -> ordered skyline notes (sorted by onset_q)
    staff_notes: Dict[int, List[NoteEvent]] = field(default_factory=dict)
    # Precomputed interval & rhythm bags for fast banded-DTW
    interval_bag: Counter[int] = field(default_factory=Counter)
    rhythm_bag: Counter[int] = field(default_factory=Counter)
    n_notes: int = 0

    def finalize(self, prev_staff_last: Optional[Dict[int, NoteEvent]] = None) -> Dict[int, NoteEvent]:
        """Populate interval & rhythm bags and return last note per staff for cross-bar context."""
        ibag: Counter[int] = Counter()
        rbag: Counter[int] = Counter()
        total = 0
        out_last: Dict[int, NoteEvent] = dict(prev_staff_last) if prev_staff_last else {}

        for st, notes in self.staff_notes.items():
            if not notes:
                continue
            total += len(notes)
            # Cross-bar incoming interval from previous measure's last note on same staff
            if prev_staff_last and st in prev_staff_last:
                prev_n = prev_staff_last[st]
                d_dia = _clamp(notes[0].diatonic - prev_n.diatonic, -12, 12)
                ibag[d_dia] += 1
            for k, n in enumerate(notes):
                rbag[quantize_dur(n.dur_q)] += 1
                if k + 1 < len(notes):
                    d_dia = _clamp(notes[k + 1].diatonic - n.diatonic, -12, 12)
                    ibag[d_dia] += 1
                if k + 2 < len(notes):
                    # Lag-2 interval adds robustness to single-note deletion/corruption
                    d2 = _clamp(notes[k + 2].diatonic - n.diatonic, -12, 12)
                    ibag[100 + d2] += 1
            out_last[st] = notes[-1]

        self.interval_bag = ibag
        self.rhythm_bag = rbag
        self.n_notes = total
        return out_last


@dataclass
class ParsedScore:
    """Canonical representation of either a GT work or a noisy decoded page."""

    work_id: str
    source_path: str
    n_staves: int
    key_sig: Optional[int]  # fifths, e.g. -1 = 1 flat, +2 = 2 sharps
    time_sig: str  # e.g. "4/4", "6/8", ""
    measures: List[MeasureSummary]
    system_breaks: List[Tuple[int, int]] = field(default_factory=list)  # [(m_start, m_end), ...]
    content_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_measures(self) -> int:
        return len(self.measures)

    @property
    def n_notes(self) -> int:
        return sum(m.n_notes for m in self.measures)


def _collapse_skyline(notes: List[NoteEvent]) -> List[NoteEvent]:
    """Keep the highest-pitched note at each distinct onset on a staff (chord-invariant)."""
    if not notes:
        return []
    by_onset: Dict[int, NoteEvent] = {}
    for n in notes:
        key = int(round(n.onset_q * 96.0))
        prev = by_onset.get(key)
        if prev is None or n.midi > prev.midi:
            by_onset[key] = n
    return [by_onset[k] for k in sorted(by_onset.keys())]


def _compute_content_hash(n_staves: int, measures: Sequence[MeasureSummary]) -> str:
    """Deterministic musical content hash to deduplicate multi-font renders of identical scores."""
    h = hashlib.sha1()
    h.update(f"s={n_staves}|m={len(measures)}|".encode("ascii"))
    for m in measures:
        h.update(b"|")
        for st in sorted(m.staff_notes.keys()):
            h.update(f"{st}:".encode("ascii"))
            for n in m.staff_notes[st]:
                h.update(f"{n.midi},{quantize_dur(n.dur_q)};".encode("ascii"))
    return h.hexdigest()[:16]


def compute_decode_hash(query: ParsedScore | Sequence[str] | str) -> str:
    """Compute a deterministic normalized pitch-interval + rhythm hash of a decoded query.

    Used by the batch-level Hallucination Attractor Guard to detect identical
    hallucinated decodes emitted across multiple distinct manuscript images.
    Returns empty string if the decoded sequence has no notes.
    """
    if isinstance(query, ParsedScore):
        score = query
    elif isinstance(query, str):
        score = parse_bekern_tokens(query.split(), work_id="query")
    else:
        score = parse_bekern_tokens(query, work_id="query")

    if score.n_measures == 0 or score.n_notes == 0:
        return ""

    h = hashlib.sha1()
    h.update(f"s={score.n_staves}|m={score.n_measures}|".encode("ascii"))
    prev_last_midi: Dict[int, int] = {}
    for m in score.measures:
        h.update(b"|")
        for st in sorted(m.staff_notes.keys()):
            notes = m.staff_notes[st]
            if not notes:
                continue
            h.update(f"{st}:".encode("ascii"))
            for k, n in enumerate(notes):
                if k == 0:
                    iv = (n.midi - prev_last_midi[st]) if st in prev_last_midi else 0
                else:
                    iv = n.midi - notes[k - 1].midi
                h.update(f"{_clamp(iv, -24, 24)},{quantize_dur(n.dur_q)};".encode("ascii"))
            prev_last_midi[st] = notes[-1].midi
    return h.hexdigest()[:16]


def _finalize_measures(raw_measures: List[Dict[int, List[NoteEvent]]]) -> List[MeasureSummary]:
    measures: List[MeasureSummary] = []
    prev_last: Dict[int, NoteEvent] = {}
    for m_idx, st_map in enumerate(raw_measures):
        skyline_map = {st: _collapse_skyline(ns) for st, ns in st_map.items() if ns}
        ms = MeasureSummary(measure_idx=m_idx, staff_notes=skyline_map)
        prev_last = ms.finalize(prev_last)
        measures.append(ms)
    return measures


# ---------------------------------------------------------------------------
# 2. Fast Parsers: Bekern & MusicXML (.mxl / .musicxml / .xml)
# ---------------------------------------------------------------------------


def _parse_key_token(tok: str) -> Optional[int]:
    if tok.startswith("key") and len(tok) >= 5 and tok[3] in ("+", "-"):
        try:
            return int(tok[3:])
        except ValueError:
            return None
    return None


def _parse_time_token(tok: str) -> str:
    if tok.startswith("time-"):
        parts = tok[5:].split("-")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return f"{parts[0]}/{parts[1]}"
    return ""


def parse_bekern_tokens(
    tokens: Sequence[str],
    *,
    work_id: str = "query",
    source_path: str = "",
    system_streams: Optional[Sequence[Sequence[str]]] = None,
) -> ParsedScore:
    """Parse a UniOMR/GT bekern token sequence (or list of per-system streams) into ParsedScore."""
    if system_streams is not None:
        systems = [list(s) for s in system_streams if s]
    else:
        systems = []
        cur_sys: List[str] = []
        for t in tokens:
            if t == SYS_TOKEN:
                if cur_sys:
                    systems.append(cur_sys)
                    cur_sys = []
            elif t:
                cur_sys.append(t)
        if cur_sys:
            systems.append(cur_sys)

    n_staves_hdr = 0
    max_voice_seen = 0
    key_sig: Optional[int] = None
    time_sig = ""

    raw_measures: List[Dict[int, List[NoteEvent]]] = []
    system_breaks: List[Tuple[int, int]] = []

    for sys_idx, sys_toks in enumerate(systems):
        sys_start_m = len(raw_measures)
        # Parse opening header of this system
        i = 0
        n = len(sys_toks)
        sys_clefs = 0
        while i < n:
            tok = sys_toks[i]
            if tok.startswith("clef-"):
                sys_clefs += 1
                i += 1
            elif tok.startswith("key") and len(tok) >= 5 and tok[3] in ("+", "-"):
                if key_sig is None:
                    key_sig = _parse_key_token(tok)
                i += 1
            elif tok.startswith("time-"):
                if not time_sig:
                    time_sig = _parse_time_token(tok)
                i += 1
            else:
                break
        if sys_idx == 0 and sys_clefs > 0:
            n_staves_hdr = sys_clefs
        elif n_staves_hdr == 0 and sys_clefs > 0:
            n_staves_hdr = sys_clefs

        cur_measure: Dict[int, List[NoteEvent]] = defaultdict(list)
        cur_staff = 0
        staff_onset: Dict[int, float] = defaultdict(float)
        staff_last_onset: Dict[int, float] = defaultdict(float)
        pending_chord = False
        has_any_content = False

        while i < n:
            tok = sys_toks[i]
            if tok in BAR_CLOSING_TOKENS:
                raw_measures.append(dict(cur_measure))
                cur_measure = defaultdict(list)
                cur_staff = 0
                staff_onset = defaultdict(float)
                staff_last_onset = defaultdict(float)
                pending_chord = False
                has_any_content = False
                i += 1
                continue
            if tok == "bar-repeat-left":
                if has_any_content:
                    raw_measures.append(dict(cur_measure))
                    cur_measure = defaultdict(list)
                    cur_staff = 0
                    staff_onset = defaultdict(float)
                    staff_last_onset = defaultdict(float)
                    pending_chord = False
                    has_any_content = False
                i += 1
                continue
            if tok.startswith("voice-"):
                try:
                    v = int(tok.split("-")[1])
                    if v >= 1:
                        cur_staff = v - 1
                        if v > max_voice_seen:
                            max_voice_seen = v
                except (IndexError, ValueError):
                    pass
                pending_chord = False
                i += 1
                continue
            if tok == "chord":
                pending_chord = True
                i += 1
                continue
            if tok == "bar-rest":
                has_any_content = True
                pending_chord = False
                i += 1
                continue
            if tok == "rest":
                has_any_content = True
                pending_chord = False
                dur_q = 1.0
                adv = 1
                if i + 1 < n and sys_toks[i + 1] in DURATION_TOKENS:
                    dur_q = DURATION_TO_QUARTER[sys_toks[i + 1]]
                    adv = 2
                    if i + 2 < n and sys_toks[i + 2] == "dot":
                        dur_q *= 1.5
                        adv = 3
                staff_onset[cur_staff] += dur_q
                i += adv
                continue
            if tok in DURATION_TOKENS:
                base_dur = DURATION_TO_QUARTER[tok]
                j = i + 1
                dots = 0
                # Support optional dur-dot-first ordering
                while j < n and sys_toks[j] == "dot":
                    dots += 1
                    j += 1
                if j + 1 < n and sys_toks[j] in STEP_TOKENS and sys_toks[j + 1] in OCTAVE_TOKENS:
                    step = sys_toks[j]
                    octv = int(sys_toks[j + 1])
                    j += 2
                    alt = 0
                    if j < n and sys_toks[j] in ALTER_TO_SEMITONE:
                        alt = ALTER_TO_SEMITONE[sys_toks[j]]
                        j += 1
                    while j < n and sys_toks[j] == "dot":
                        dots += 1
                        j += 1
                    dur_q = base_dur * (1.75 if dots >= 2 else (1.5 if dots == 1 else 1.0))
                    midi = (octv + 1) * 12 + STEP_TO_SEMITONE[step] + alt
                    diat = octv * 7 + STEP_TO_DIATONIC[step]

                    if pending_chord and cur_measure[cur_staff]:
                        onset = staff_last_onset[cur_staff]
                    else:
                        onset = staff_onset[cur_staff]
                        staff_last_onset[cur_staff] = onset
                        staff_onset[cur_staff] = onset + dur_q
                    pending_chord = False
                    has_any_content = True
                    cur_measure[cur_staff].append(
                        NoteEvent(
                            midi=midi,
                            diatonic=diat,
                            dur_q=dur_q,
                            onset_q=onset,
                            staff=cur_staff,
                        )
                    )
                    i = j
                    continue
            i += 1

        if has_any_content:
            raw_measures.append(dict(cur_measure))
        sys_end_m = len(raw_measures)
        if sys_end_m > sys_start_m:
            system_breaks.append((sys_start_m, sys_end_m))

    n_staves = n_staves_hdr if n_staves_hdr > 0 else max(1, max_voice_seen)
    measures = _finalize_measures(raw_measures)
    chash = _compute_content_hash(n_staves, measures)
    return ParsedScore(
        work_id=work_id,
        source_path=source_path,
        n_staves=n_staves,
        key_sig=key_sig,
        time_sig=time_sig,
        measures=measures,
        system_breaks=system_breaks,
        content_hash=chash,
    )


def parse_bekern_file(path: Path | str, *, work_id: Optional[str] = None) -> ParsedScore:
    p = Path(path)
    toks = p.read_text(encoding="utf-8", errors="ignore").split()
    return parse_bekern_tokens(
        toks,
        work_id=work_id or p.stem,
        source_path=str(p),
    )


def _strip_ns(tag: str) -> str:
    idx = tag.rfind("}")
    return tag[idx + 1 :] if idx != -1 else tag


def _read_musicxml_root(path: Path) -> ET.Element:
    if path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(path, "r") as zf:
            target_member: Optional[str] = None
            if "META-INF/container.xml" in zf.namelist():
                try:
                    croot = ET.fromstring(zf.read("META-INF/container.xml"))
                    for el in croot.iter():
                        if _strip_ns(el.tag) == "rootfile" and "full-path" in el.attrib:
                            target_member = el.attrib["full-path"]
                            break
                except ET.ParseError:
                    target_member = None
            if not target_member or target_member not in zf.namelist():
                for name in zf.namelist():
                    low = name.lower()
                    if not low.startswith("meta-inf/") and (
                        low.endswith(".xml") or low.endswith(".musicxml")
                    ):
                        target_member = name
                        break
            if not target_member:
                raise ValueError(f"No XML score found inside MXL archive: {path}")
            return ET.fromstring(zf.read(target_member))
    return ET.parse(path).getroot()


def parse_musicxml_file(path: Path | str, *, work_id: Optional[str] = None) -> ParsedScore:
    """Fast pure-Python parser for MusicXML (``.mxl``, ``.musicxml``, ``.xml``)."""
    p = Path(path)
    root = _read_musicxml_root(p)

    parts = [ch for ch in root if _strip_ns(ch.tag) == "part"]
    if not parts:
        # Handle score-timewise or nested parts defensively
        parts = [el for el in root.iter() if _strip_ns(el.tag) == "part"]

    key_sig: Optional[int] = None
    time_sig = ""
    part_staff_counts: List[int] = []
    part_measures_raw: List[List[Dict[int, List[NoteEvent]]]] = []

    for part in parts:
        divisions = 1.0
        n_staves_part = 1
        max_local_staff = 1
        part_has_notes = False
        part_has_clef = False
        p_measures: List[Dict[int, List[NoteEvent]]] = []

        for m_el in part:
            if _strip_ns(m_el.tag) != "measure":
                continue
            cur_pos = 0.0
            last_note_pos = 0.0
            m_notes: Dict[int, List[NoteEvent]] = defaultdict(list)

            for ch in m_el:
                tag = _strip_ns(ch.tag)
                if tag == "attributes":
                    for sub in ch:
                        stag = _strip_ns(sub.tag)
                        if stag == "clef":
                            part_has_clef = True
                        elif stag == "divisions" and sub.text:
                            try:
                                dval = float(sub.text.strip())
                                if dval > 0:
                                    divisions = dval
                            except ValueError:
                                pass
                        elif stag == "staves" and sub.text:
                            try:
                                sval = int(sub.text.strip())
                                if sval > n_staves_part:
                                    n_staves_part = sval
                            except ValueError:
                                pass
                        elif stag == "key" and key_sig is None:
                            for ksub in sub:
                                if _strip_ns(ksub.tag) == "fifths" and ksub.text:
                                    try:
                                        key_sig = int(ksub.text.strip())
                                    except ValueError:
                                        pass
                        elif stag == "time" and not time_sig:
                            beats = ""
                            btype = ""
                            for tsub in sub:
                                ttag = _strip_ns(tsub.tag)
                                if ttag == "beats" and tsub.text:
                                    beats = tsub.text.strip()
                                elif ttag == "beat-type" and tsub.text:
                                    btype = tsub.text.strip()
                            if beats and btype:
                                time_sig = f"{beats}/{btype}"
                elif tag == "backup":
                    for sub in ch:
                        if _strip_ns(sub.tag) == "duration" and sub.text:
                            try:
                                cur_pos = max(0.0, cur_pos - float(sub.text.strip()))
                            except ValueError:
                                pass
                elif tag == "forward":
                    for sub in ch:
                        if _strip_ns(sub.tag) == "duration" and sub.text:
                            try:
                                cur_pos += float(sub.text.strip())
                            except ValueError:
                                pass
                elif tag == "note":
                    is_grace = False
                    is_chord = False
                    is_rest = False
                    dur_div = 0.0
                    local_staff = 1
                    step = ""
                    octv: Optional[int] = None
                    alt = 0

                    for sub in ch:
                        stag = _strip_ns(sub.tag)
                        if stag == "grace":
                            is_grace = True
                        elif stag == "chord":
                            is_chord = True
                        elif stag == "rest":
                            is_rest = True
                        elif stag == "duration" and sub.text:
                            try:
                                dur_div = float(sub.text.strip())
                            except ValueError:
                                pass
                        elif stag == "staff" and sub.text:
                            try:
                                local_staff = int(sub.text.strip())
                                if local_staff > max_local_staff:
                                    max_local_staff = local_staff
                            except ValueError:
                                pass
                        elif stag == "pitch":
                            for psub in sub:
                                ptag = _strip_ns(psub.tag)
                                if ptag == "step" and psub.text:
                                    step = psub.text.strip().upper()
                                elif ptag == "octave" and psub.text:
                                    try:
                                        octv = int(psub.text.strip())
                                    except ValueError:
                                        pass
                                elif ptag == "alter" and psub.text:
                                    try:
                                        alt = int(round(float(psub.text.strip())))
                                    except ValueError:
                                        pass

                    if is_grace:
                        continue
                    onset_div = last_note_pos if is_chord else cur_pos
                    if not is_chord:
                        last_note_pos = cur_pos
                        cur_pos += dur_div

                    if not is_rest and step in STEP_TO_SEMITONE and octv is not None:
                        part_has_notes = True
                        dur_q = dur_div / divisions if divisions > 0 else 1.0
                        onset_q = onset_div / divisions if divisions > 0 else 0.0
                        midi = (octv + 1) * 12 + STEP_TO_SEMITONE[step] + alt
                        diat = octv * 7 + STEP_TO_DIATONIC[step]
                        m_notes[local_staff - 1].append(
                            NoteEvent(
                                midi=midi,
                                diatonic=diat,
                                dur_q=dur_q,
                                onset_q=onset_q,
                                staff=local_staff - 1,
                            )
                        )

            p_measures.append(dict(m_notes))

        # Ignore empty dummy/hidden parts with zero notes
        if not part_has_notes:
            continue
        part_staff_counts.append(max(n_staves_part, max_local_staff))
        part_measures_raw.append(p_measures)

    n_staves = sum(part_staff_counts) if part_staff_counts else 1
    n_measures = max((len(pm) for pm in part_measures_raw), default=0)

    # Merge parts into unified global staff indices
    merged_raw: List[Dict[int, List[NoteEvent]]] = [defaultdict(list) for _ in range(n_measures)]
    staff_offset = 0
    for p_staves, p_meas in zip(part_staff_counts, part_measures_raw):
        for m_idx, m_dict in enumerate(p_meas):
            dst = merged_raw[m_idx]
            for loc_st, notes in m_dict.items():
                gst = staff_offset + loc_st
                dst[gst].extend(
                    NoteEvent(
                        midi=n.midi,
                        diatonic=n.diatonic,
                        dur_q=n.dur_q,
                        onset_q=n.onset_q,
                        staff=gst,
                    )
                    for n in notes
                )
        staff_offset += p_staves

    measures = _finalize_measures([dict(m) for m in merged_raw])
    chash = _compute_content_hash(n_staves, measures)
    return ParsedScore(
        work_id=work_id or p.stem,
        source_path=str(p),
        n_staves=n_staves,
        key_sig=key_sig,
        time_sig=time_sig,
        measures=measures,
        system_breaks=[],
        content_hash=chash,
    )


# ---------------------------------------------------------------------------
# 3. 5-Note Chromatic/Diatonic Pitch-Interval + Quantized IOI-Ratio N-Grams
# ---------------------------------------------------------------------------


def _pack_ngram(kind: int, i0: int, i1: int, i2: int, i3: int, r0: int, r1: int, r2: int) -> int:
    """Pack a 5-note (4-interval, 3-IOI-ratio) fingerprint into a single compact Python int."""
    # Shift signed values into non-negative ranges:
    # intervals in [-12, 12] -> [0, 24] (5 bits each)
    # ioi buckets in [-3, 3] -> [0, 6] (3 bits each)
    # kind in {0 (chromatic), 1 (diatonic)} (1 bit)
    v = kind & 1
    v = (v << 5) | (_clamp(i0, -12, 12) + 12)
    v = (v << 5) | (_clamp(i1, -12, 12) + 12)
    v = (v << 5) | (_clamp(i2, -12, 12) + 12)
    v = (v << 5) | (_clamp(i3, -12, 12) + 12)
    v = (v << 3) | (_clamp(r0, -3, 3) + 3)
    v = (v << 3) | (_clamp(r1, -3, 3) + 3)
    v = (v << 3) | (_clamp(r2, -3, 3) + 3)
    return v


def extract_ngrams(score: ParsedScore, *, n: int = 5) -> List[Tuple[int, int]]:
    """Extract ``(ngram_hash, measure_idx)`` pairs across consecutive measures per staff.

    Emits both:
      - kind 0: chromatic 4-interval + 3 quantized IOI ratios (100% transposition-invariant)
      - kind 1: diatonic 4-interval + 3 quantized IOI ratios (resilient to accidental errors)
    """
    if n < 2 or score.n_measures == 0:
        return []

    # Build continuous per-staff note streams with running cumulative onset
    staff_streams: Dict[int, List[Tuple[int, int, int, float, float]]] = defaultdict(list)
    # Estimate measure length per measure from max(onset + dur) across staves
    cum_q = 0.0
    for m in score.measures:
        m_len = 0.0
        for st, notes in m.staff_notes.items():
            for ev in notes:
                end_q = ev.onset_q + max(0.0625, ev.dur_q)
                if end_q > m_len:
                    m_len = end_q
                staff_streams[st].append(
                    (m.measure_idx, ev.midi, ev.diatonic, cum_q + ev.onset_q, max(0.0625, ev.dur_q))
                )
        cum_q += max(m_len, 1.0)

    out: List[Tuple[int, int]] = []
    for st, stream in staff_streams.items():
        L = len(stream)
        if L < n:
            continue
        for k in range(L - n + 1):
            m0 = stream[k][0]
            # 4 pitch intervals (chromatic & diatonic)
            c0 = stream[k + 1][1] - stream[k][1]
            c1 = stream[k + 2][1] - stream[k + 1][1]
            c2 = stream[k + 3][1] - stream[k + 2][1]
            c3 = stream[k + 4][1] - stream[k + 3][1]

            # Skip degenerate repeated-note ostinatos (5 identical pitches)
            if c0 == 0 and c1 == 0 and c2 == 0 and c3 == 0:
                continue

            d0 = stream[k + 1][2] - stream[k][2]
            d1 = stream[k + 2][2] - stream[k + 1][2]
            d2 = stream[k + 3][2] - stream[k + 2][2]
            d3 = stream[k + 4][2] - stream[k + 3][2]

            # 4 inter-onset intervals -> 3 IOI ratios
            ioi0 = max(0.0625, stream[k + 1][3] - stream[k][3])
            ioi1 = max(0.0625, stream[k + 2][3] - stream[k + 1][3])
            ioi2 = max(0.0625, stream[k + 3][3] - stream[k + 2][3])
            ioi3 = max(0.0625, stream[k + 4][3] - stream[k + 3][3])

            r0 = quantize_ioi_ratio(ioi1 / ioi0)
            r1 = quantize_ioi_ratio(ioi2 / ioi1)
            r2 = quantize_ioi_ratio(ioi3 / ioi2)

            out.append((_pack_ngram(0, c0, c1, c2, c3, r0, r1, r2), m0))
            out.append((_pack_ngram(1, d0, d1, d2, d3, r0, r1, r2), m0))

    return out


# ---------------------------------------------------------------------------
# 4. Strict Zero-Leakage Exclusion Guard (suite31_hw + gate20 + OMR-NorthStar)
# ---------------------------------------------------------------------------


def _norm_slug(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def _strip_movement_suffix(slug: str) -> str:
    m = re.match(r"^(.+?[a-z]\d+[a-z0-9]*)[-_](\d+)$", slug)
    return m.group(1) if m else slug


@dataclass
class EvalLeakageGuard:
    """Strict zero-leakage guard covering suite31_hw, gate20, and OMR-NorthStar stems."""

    exact_slugs: Set[str] = field(default_factory=set)
    base_work_keys: Set[str] = field(default_factory=set)
    gate20_ids: Set[str] = field(default_factory=set)
    gate20_prefixes: Set[str] = field(default_factory=set)
    northstar_ids: Set[str] = field(default_factory=set)
    catalog_patterns: List[Tuple[str, str, re.Pattern[str]]] = field(default_factory=list)

    def _register_slug(self, raw_slug: str) -> None:
        norm = _norm_slug(raw_slug)
        if not norm:
            return
        self.exact_slugs.add(norm)
        # Strip common benchmark prefixes
        bare = norm
        for pfx in ("hwm-clean-", "hwm-photo-", "cdr-", "rwp-", "als-", "work-", "gt-p1-", "hw-"):
            if bare.startswith(pfx):
                bare = bare[len(pfx) :]
                if bare:
                    self.exact_slugs.add(bare)
        if bare.startswith("hw-"):
            bare = bare[3:]
            if bare:
                self.exact_slugs.add(bare)
        if len(bare) >= 6:
            self.base_work_keys.add(bare)
            base_mvt = _strip_movement_suffix(bare)
            if base_mvt != bare and len(base_mvt) >= 6:
                self.exact_slugs.add(base_mvt)
                self.base_work_keys.add(base_mvt)

        m_cat = re.match(r"^([a-z]+)-(k|kv|bwv|op|d|hob|h)(\d+[a-z0-9]*)", bare)
        if m_cat:
            comp, cat_pfx, cat_num = m_cat.group(1), m_cat.group(2), m_cat.group(3)
            if cat_pfx in ("k", "kv"):
                cat_re = rf"(?:kv?|k\.?v?\.?)[- ]*0*{re.escape(cat_num)}(?:\b|[^0-9])"
            elif cat_pfx == "op":
                cat_re = rf"op\.?[- ]*0*{re.escape(cat_num)}(?:\b|[^0-9])"
            else:
                cat_re = rf"{re.escape(cat_pfx)}\.?[- ]*0*{re.escape(cat_num)}(?:\b|[^0-9])"
            pat = re.compile(rf"(?:^|[^a-z]){re.escape(comp)}(?:$|[^a-z]).*{cat_re}", re.I)
            self.catalog_patterns.append((raw_slug, f"{comp}-{cat_pfx}{cat_num}", pat))

    @classmethod
    def from_files(
        cls,
        *,
        suite31_manifest: Optional[Path | str] = DEFAULT_SUITE31_MANIFEST,
        gate20_exclusion: Optional[Path | str] = DEFAULT_GATE20_EXCLUSION,
        northstar_manifest: Optional[Path | str] = DEFAULT_NORTHSTAR_MANIFEST,
        extra_excluded_stems: Sequence[str] = (),
    ) -> "EvalLeakageGuard":
        guard = cls()

        if suite31_manifest and Path(suite31_manifest).is_file():
            s31 = json.loads(Path(suite31_manifest).read_text(encoding="utf-8"))
            for piece in s31.get("pieces", []):
                slug = str(piece.get("slug", "")).strip()
                if slug:
                    guard._register_slug(slug)

        if gate20_exclusion and Path(gate20_exclusion).is_file():
            g20 = json.loads(Path(gate20_exclusion).read_text(encoding="utf-8"))
            for key in ("do_not_train", "held_out_4page_gate"):
                for item in g20.get(key, []):
                    norm_item = _norm_slug(item)
                    if not norm_item:
                        continue
                    guard.gate20_ids.add(norm_item)
                    stem = re.sub(r"-\d+$", "", norm_item)
                    if stem:
                        guard.gate20_prefixes.add(stem)

        if northstar_manifest and Path(northstar_manifest).is_file():
            ns = json.loads(Path(northstar_manifest).read_text(encoding="utf-8"))
            for page in ns.get("pages", []):
                for field_name in ("page_id", "work_id", "paired_anchor_id"):
                    val = str(page.get(field_name, "")).strip()
                    if val:
                        guard.northstar_ids.add(_norm_slug(val))
                        guard._register_slug(val)
            for ex in ns.get("audit_exclusions", []):
                for field_name in ("page_id", "work_id"):
                    val = str(ex.get(field_name, "")).strip()
                    if val:
                        guard.northstar_ids.add(_norm_slug(val))
                        guard._register_slug(val)

        for stem in extra_excluded_stems:
            guard._register_slug(stem)

        return guard

    def check(self, *identifiers: str) -> Tuple[bool, str]:
        """Return ``(True, reason)`` if any identifier leaks into eval sets."""
        for raw in identifiers:
            if not raw:
                continue
            norm = _norm_slug(raw)
            if not norm:
                continue
            bare = norm
            for pfx in ("hwm-clean-", "hwm-photo-", "gt-p1-", "work-", "hw-"):
                if bare.startswith(pfx):
                    bare = bare[len(pfx) :]
            if norm in self.exact_slugs or bare in self.exact_slugs:
                return True, f"eval_exact:{norm}"
            if norm in self.northstar_ids or bare in self.northstar_ids:
                return True, f"northstar_exact:{norm}"
            if norm in self.gate20_ids:
                return True, f"gate20_exact:{norm}"
            for gp in self.gate20_prefixes:
                if norm == gp or norm.startswith(gp + "-") or f"-{gp}-" in f"-{norm}-":
                    return True, f"gate20_prefix:{gp}"
            for bk in self.base_work_keys:
                if (
                    norm == bk
                    or norm.startswith(bk + "-")
                    or bare == bk
                    or bare.startswith(bk + "-")
                    or f"-{bk}-" in f"-{norm}-"
                ):
                    return True, f"eval_work_key:{bk}"

        haystack = " ".join(s for s in identifiers if s)
        if haystack:
            for src_slug, tag, pat in self.catalog_patterns:
                if pat.search(haystack):
                    return True, f"eval_catalog:{tag}({src_slug})"
        return False, ""


# ---------------------------------------------------------------------------
# 5. Symbolic Inverted Index (build_index)
# ---------------------------------------------------------------------------


@dataclass
class WorkMeta:
    """Indexed work metadata + compact measure summaries for instant DTW."""

    work_idx: int
    work_id: str
    source_path: str
    n_staves: int
    key_sig: Optional[int]
    time_sig: str
    n_measures: int
    content_hash: str
    measures: List[MeasureSummary]
    n_ngrams: int = 0


class SymbolicIndex:
    """Fast CPU inverted index of 5-note pitch-interval + IOI-ratio n-grams."""

    def __init__(self) -> None:
        self.works: List[WorkMeta] = []
        self.work_id_to_idx: Dict[str, int] = {}
        self.content_hash_to_idx: Dict[str, int] = {}
        # ngram_hash -> list of (work_idx, measure_idx)
        self.postings: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.skipped_duplicates: int = 0
        self.skipped_empty: int = 0
        self.skipped_eval_excluded: int = 0
        self.skipped_synthetic: int = 0

    @property
    def n_works(self) -> int:
        return len(self.works)

    @property
    def n_postings(self) -> int:
        return sum(len(v) for v in self.postings.values())

    def add_score(
        self,
        score: ParsedScore,
        *,
        deduplicate: bool = True,
        leakage_guard: Optional[EvalLeakageGuard] = None,
        exclude_prefixes: Sequence[str] = (),
    ) -> Optional[int]:
        """Add a ParsedScore to the inverted index. Returns work_idx or None if skipped."""
        if _matches_exclude_prefix(score.work_id, score.source_path, exclude_prefixes):
            self.skipped_synthetic = getattr(self, "skipped_synthetic", 0) + 1
            return None

        if score.n_measures == 0 or score.n_notes < 5:
            self.skipped_empty += 1
            return None

        if leakage_guard is not None:
            leaked, _ = leakage_guard.check(score.work_id, score.source_path)
            if leaked:
                self.skipped_eval_excluded += 1
                return None

        if deduplicate and score.content_hash in self.content_hash_to_idx:
            self.skipped_duplicates += 1
            return None

        ngrams = extract_ngrams(score)
        if not ngrams:
            self.skipped_empty += 1
            return None

        w_idx = len(self.works)
        # Ensure unique work_id
        wid = score.work_id
        if wid in self.work_id_to_idx:
            wid = f"{wid}__{w_idx}"

        # Deduplicate (ngram_hash, measure_idx) within this work
        seen_pairs: Set[Tuple[int, int]] = set()
        for h, m_idx in ngrams:
            pair = (h, m_idx)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                self.postings[h].append((w_idx, m_idx))

        meta = WorkMeta(
            work_idx=w_idx,
            work_id=wid,
            source_path=score.source_path,
            n_staves=score.n_staves,
            key_sig=score.key_sig,
            time_sig=score.time_sig,
            n_measures=score.n_measures,
            content_hash=score.content_hash,
            measures=score.measures,
            n_ngrams=len(seen_pairs),
        )
        self.works.append(meta)
        self.work_id_to_idx[wid] = w_idx
        self.content_hash_to_idx[score.content_hash] = w_idx
        return w_idx

    def add_musicxml_paths(
        self,
        paths: Iterable[Path | str],
        *,
        deduplicate: bool = True,
        leakage_guard: Optional[EvalLeakageGuard] = None,
        work_id_prefix: str = "",
        exclude_prefixes: Sequence[str] = DEFAULT_SYNTHETIC_PREFIXES,
    ) -> int:
        added = 0
        for raw_p in paths:
            p = Path(raw_p)
            if not p.is_file():
                continue
            wid = f"{work_id_prefix}{p.stem}" if work_id_prefix else p.stem
            if _matches_exclude_prefix(wid, str(p), exclude_prefixes):
                self.skipped_synthetic = getattr(self, "skipped_synthetic", 0) + 1
                continue
            if leakage_guard is not None:
                leaked, _ = leakage_guard.check(wid, str(p))
                if leaked:
                    self.skipped_eval_excluded += 1
                    continue
            try:
                score = parse_musicxml_file(p, work_id=wid)
            except Exception:
                continue
            if self.add_score(score, deduplicate=deduplicate, leakage_guard=None) is not None:
                added += 1
        return added

    def add_bekern_jsonl(
        self,
        jsonl_path: Path | str,
        *,
        allowed_domains: Optional[Set[str]] = None,
        deduplicate: bool = True,
        leakage_guard: Optional[EvalLeakageGuard] = None,
        max_rows: Optional[int] = None,
        exclude_prefixes: Sequence[str] = DEFAULT_SYNTHETIC_PREFIXES,
    ) -> int:
        """Index digital bekern pages directly from a UniOMR JSONL training manifest.

        Zero XML parsing needed: reads each row's ``bekerns`` list of per-system
        ``-bekern.txt`` files directly. Deduplicates identical musical content
        (e.g. multi-font renders) via ``content_hash``.
        """
        jp = Path(jsonl_path)
        if not jp.is_file():
            raise FileNotFoundError(f"JSONL manifest not found: {jp}")
        if allowed_domains is None:
            allowed_domains = {"printed", "hw"}

        added = 0
        rows_seen = 0
        with jp.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if max_rows is not None and rows_seen >= max_rows:
                    break
                rows_seen += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                dom = str(row.get("domain", "")).strip()
                if allowed_domains and dom not in allowed_domains:
                    continue
                bk_paths = row.get("bekerns") or []
                if isinstance(bk_paths, str):
                    bk_paths = [bk_paths]
                if not bk_paths:
                    continue

                img_path = str(row.get("image", ""))
                wid = Path(img_path).stem if img_path else Path(bk_paths[0]).stem
                if _matches_exclude_prefix(wid, img_path or str(bk_paths[0]), exclude_prefixes):
                    self.skipped_synthetic = getattr(self, "skipped_synthetic", 0) + 1
                    continue
                if leakage_guard is not None:
                    leaked, _ = leakage_guard.check(wid, img_path, str(bk_paths[0]))
                    if leaked:
                        self.skipped_eval_excluded += 1
                        continue

                sys_streams: List[List[str]] = []
                ok = True
                for bp in bk_paths:
                    p = Path(bp)
                    if not p.is_file():
                        ok = False
                        break
                    toks = p.read_text(encoding="utf-8", errors="ignore").split()
                    if toks:
                        sys_streams.append(toks)
                if not ok or not sys_streams:
                    continue

                score = parse_bekern_tokens(
                    [],
                    work_id=wid,
                    source_path=img_path or str(bk_paths[0]),
                    system_streams=sys_streams,
                )
                if self.add_score(score, deduplicate=deduplicate, leakage_guard=None) is not None:
                    added += 1
        return added

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "wb", compresslevel=3) as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path | str) -> "SymbolicIndex":
        with gzip.open(Path(path), "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls) and type(obj).__name__ != cls.__name__:
            raise TypeError(f"Expected SymbolicIndex, got {type(obj)}")
        return obj


def build_index(
    *,
    musicxml_dirs: Sequence[Path | str] = (),
    musicxml_files: Sequence[Path | str] = (),
    bekern_files: Sequence[Path | str] = (),
    bekern_jsonls: Sequence[Path | str] = (),
    allowed_jsonl_domains: Optional[Set[str]] = None,
    deduplicate: bool = True,
    leakage_guard: Optional[EvalLeakageGuard] = None,
    max_jsonl_rows: Optional[int] = None,
    exclude_prefixes: Sequence[str] = DEFAULT_SYNTHETIC_PREFIXES,
) -> SymbolicIndex:
    """Build a SymbolicIndex from any combination of MusicXML dirs/files, bekern files, and JSONL manifests."""
    index = SymbolicIndex()

    for d in musicxml_dirs:
        dp = Path(d)
        if not dp.is_dir():
            continue
        found = sorted(
            p
            for p in dp.rglob("*")
            if p.is_file() and p.suffix.lower() in (".mxl", ".musicxml", ".xml")
        )
        index.add_musicxml_paths(
            found,
            deduplicate=deduplicate,
            leakage_guard=leakage_guard,
            exclude_prefixes=exclude_prefixes,
        )

    if musicxml_files:
        index.add_musicxml_paths(
            musicxml_files,
            deduplicate=deduplicate,
            leakage_guard=leakage_guard,
            exclude_prefixes=exclude_prefixes,
        )

    for bf in bekern_files:
        bp = Path(bf)
        if not bp.is_file():
            continue
        if _matches_exclude_prefix(bp.stem, str(bp), exclude_prefixes):
            index.skipped_synthetic += 1
            continue
        score = parse_bekern_file(bp)
        index.add_score(
            score,
            deduplicate=deduplicate,
            leakage_guard=leakage_guard,
            exclude_prefixes=exclude_prefixes,
        )

    for jp in bekern_jsonls:
        index.add_bekern_jsonl(
            jp,
            allowed_domains=allowed_jsonl_domains,
            deduplicate=deduplicate,
            leakage_guard=leakage_guard,
            max_rows=max_jsonl_rows,
            exclude_prefixes=exclude_prefixes,
        )

    return index


# ---------------------------------------------------------------------------
# 6. Page Fingerprinting & Shazam-Style Hough Retrieval (retrieve_best_fit)
# ---------------------------------------------------------------------------


@dataclass
class RetrievalCandidate:
    work_idx: int
    work_id: str
    source_path: str
    measure_offset: int
    score: float
    n_staves: int
    n_measures: int


@dataclass
class RetrievalResult:
    top1: Optional[RetrievalCandidate]
    top2: Optional[RetrievalCandidate]
    top1_score: float
    top2_score: float
    margin: float  # (top1 - top2) / max(1, top1)
    n_query_ngrams: int
    candidates: List[RetrievalCandidate] = field(default_factory=list)


def _works_window_similarity(
    w1: WorkMeta, off1: int, w2: WorkMeta, off2: int, n_m: int
) -> float:
    """Fast bag F1 between two candidate works on their matched measure windows (dedups same-piece slices)."""
    span = max(4, n_m)
    s1, e1 = max(0, off1), min(w1.n_measures, max(0, off1) + span)
    s2, e2 = max(0, off2), min(w2.n_measures, max(0, off2) + span)
    if e1 <= s1 or e2 <= s2:
        return 0.0
    iv1, rh1 = Counter(), Counter()
    for m in w1.measures[s1:e1]:
        iv1.update(m.interval_bag)
        rh1.update(m.rhythm_bag)
    iv2, rh2 = Counter(), Counter()
    for m in w2.measures[s2:e2]:
        iv2.update(m.interval_bag)
        rh2.update(m.rhythm_bag)
    return 0.6 * _bag_f1(iv1, iv2) + 0.4 * _bag_f1(rh1, rh2)


def _get_excluded_work_idxs(
    index: SymbolicIndex, exclude_prefixes: Sequence[str]
) -> Set[int]:
    """Return cached set of work_idx values matching exclude_prefixes for fast retrieval filtering."""
    if not exclude_prefixes:
        return set()
    pfxs = tuple(p.lower() for p in exclude_prefixes if p)
    if not pfxs:
        return set()
    cache = getattr(index, "_excluded_prefix_cache", None)
    if cache is None:
        cache = {}
        setattr(index, "_excluded_prefix_cache", cache)
    key = (pfxs, len(index.works))
    if key not in cache:
        cache[key] = {
            w.work_idx
            for w in index.works
            if _matches_exclude_prefix(w.work_id, w.source_path, pfxs)
        }
    return cache[key]


def retrieve_best_fit(
    query: ParsedScore | Sequence[str],
    index: SymbolicIndex,
    *,
    top_k: int = 5,
    max_df_frac: float = 0.08,
    min_max_df: int = 250,
    offset_jitter: int = 2,
    same_work_dedup_sim: float = 0.85,
    exclude_prefixes: Sequence[str] = (),
) -> RetrievalResult:
    """Shazam-style Hough diagonal voting over 5-note pitch-interval + IOI-ratio n-grams.

    Returns ``top1_score``, ``top2_score``, and
    ``margin = (top1_score - top2_score) / max(1.0, top1_score)``.
    """
    q_score = (
        query
        if isinstance(query, ParsedScore)
        else parse_bekern_tokens(query, work_id="query")
    )
    q_ngrams = extract_ngrams(q_score)
    if not q_ngrams or index.n_works == 0:
        return RetrievalResult(
            top1=None,
            top2=None,
            top1_score=0.0,
            top2_score=0.0,
            margin=0.0,
            n_query_ngrams=len(q_ngrams),
            candidates=[],
        )

    max_df = max(min_max_df, int(max_df_frac * index.n_works))
    postings = index.postings
    excluded_idxs = _get_excluded_work_idxs(index, exclude_prefixes)

    # Deduplicate query (ngram_hash, q_m) pairs so repeated bars don't over-vote
    unique_q = set(q_ngrams)
    # Hough accumulator: (work_idx, offset) -> weighted vote score
    votes: Dict[Tuple[int, int], float] = defaultdict(float)

    for h, q_m in unique_q:
        plist = postings.get(h)
        if not plist:
            continue
        df = len(plist)
        if df > max_df:
            continue
        w = 1.0 / math.log2(2.0 + df)
        for w_idx, ref_m in plist:
            if excluded_idxs and w_idx in excluded_idxs:
                continue
            off = ref_m - q_m
            votes[(w_idx, off)] += w
            if offset_jitter >= 1:
                w1 = 0.7 * w
                votes[(w_idx, off - 1)] += w1
                votes[(w_idx, off + 1)] += w1
            if offset_jitter >= 2:
                w2 = 0.35 * w
                votes[(w_idx, off - 2)] += w2
                votes[(w_idx, off + 2)] += w2

    if not votes:
        return RetrievalResult(
            top1=None,
            top2=None,
            top1_score=0.0,
            top2_score=0.0,
            margin=0.0,
            n_query_ngrams=len(unique_q),
            candidates=[],
        )

    # Best (offset, score) per work
    best_per_work: Dict[int, Tuple[float, int]] = {}
    for (w_idx, off), sc in votes.items():
        prev = best_per_work.get(w_idx)
        if prev is None or sc > prev[0]:
            best_per_work[w_idx] = (sc, off)

    ranked = sorted(best_per_work.items(), key=lambda kv: kv[1][0], reverse=True)
    candidates: List[RetrievalCandidate] = []
    for w_idx, (sc, off) in ranked[: max(top_k, 8)]:
        wm = index.works[w_idx]
        candidates.append(
            RetrievalCandidate(
                work_idx=w_idx,
                work_id=wm.work_id,
                source_path=wm.source_path,
                measure_offset=off,
                score=round(sc, 6),
                n_staves=wm.n_staves,
                n_measures=wm.n_measures,
            )
        )

    top1 = candidates[0] if candidates else None
    top2: Optional[RetrievalCandidate] = None
    if top1 is not None and len(candidates) >= 2:
        w1_meta = index.works[top1.work_idx]
        for cand in candidates[1:]:
            w2_meta = index.works[cand.work_idx]
            if (
                same_work_dedup_sim < 1.0
                and _works_window_similarity(
                    w1_meta, top1.measure_offset, w2_meta, cand.measure_offset, q_score.n_measures
                )
                >= same_work_dedup_sim
            ):
                # Same musical passage (e.g. page slice vs full work edition) — skip for margin
                continue
            top2 = cand
            break

    s1 = top1.score if top1 else 0.0
    s2 = top2.score if top2 else 0.0
    margin = (s1 - s2) / max(1.0, s1)

    return RetrievalResult(
        top1=top1,
        top2=top2,
        top1_score=s1,
        top2_score=s2,
        margin=round(margin, 6),
        n_query_ngrams=len(unique_q),
        candidates=candidates[:top_k],
    )


# ---------------------------------------------------------------------------
# 7. Full Banded-DTW Measure Alignment (align_page_to_work)
# ---------------------------------------------------------------------------


def _bag_f1(a: Counter[int], b: Counter[int]) -> float:
    na = sum(a.values())
    nb = sum(b.values())
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    inter = 0
    for k, va in a.items():
        vb = b.get(k, 0)
        if vb:
            inter += va if va < vb else vb
    return (2.0 * inter) / float(na + nb)


def _measure_similarity(q: MeasureSummary, r: MeasureSummary) -> Tuple[float, float, float]:
    """Return ``(combined_sim, interval_f1, rhythm_f1)`` in [0, 1] (transposition-invariant)."""
    if q.n_notes == 0 and r.n_notes == 0:
        return 0.5, 0.5, 0.5
    if q.n_notes == 0 or r.n_notes == 0:
        return 0.0, 0.0, 0.0
    iv_f1 = _bag_f1(q.interval_bag, r.interval_bag)
    rh_f1 = _bag_f1(q.rhythm_bag, r.rhythm_bag)
    return 0.6 * iv_f1 + 0.4 * rh_f1, iv_f1, rh_f1


@dataclass
class AlignmentResult:
    work_id: str
    m_start: int  # 0-based inclusive GT measure index
    m_end: int  # 0-based exclusive GT measure index
    system_breaks: List[Tuple[int, int]]  # per-system GT measure spans [(gt_s, gt_e), ...]
    align_score: float  # pitch-interval + rhythm F1
    interval_f1: float
    rhythm_f1: float
    measure_pairs: List[Tuple[int, int]] = field(default_factory=list)  # [(q_m, gt_m), ...]


def align_page_to_work(
    query: ParsedScore,
    work: WorkMeta | ParsedScore,
    voted_offset: int,
    *,
    window_pad: Optional[int] = None,
    band_width: Optional[int] = None,
    gap_penalty: float = 0.25,
) -> AlignmentResult:
    """Measure-level banded-DTW between noisy decoded page measures and GT work measures.

    Locks exact ``(m_start, m_end, system_breaks)`` around ``voted_offset`` and computes
    ``align_score`` (pitch-interval + rhythm F1).
    """
    q_meas = query.measures
    r_meas = work.measures
    mq = len(q_meas)
    mr = len(r_meas)

    if mq == 0 or mr == 0:
        return AlignmentResult(
            work_id=work.work_id,
            m_start=0,
            m_end=0,
            system_breaks=[],
            align_score=0.0,
            interval_f1=0.0,
            rhythm_f1=0.0,
            measure_pairs=[],
        )

    pad = window_pad if window_pad is not None else max(6, mq // 2)
    win_start = max(0, voted_offset - pad)
    win_end = min(mr, max(voted_offset + mq + pad, win_start + mq + pad))
    if win_end <= win_start:
        win_start = 0
        win_end = mr

    w_meas = r_meas[win_start:win_end]
    mw = len(w_meas)
    rel_off = voted_offset - win_start
    band = band_width if band_width is not None else max(5, mq // 2 + 2)

    # Precompute similarity matrix inside the band
    NEG_INF = -1e9
    dp = [[NEG_INF] * (mw + 1) for _ in range(mq + 1)]
    bt = [[0] * (mw + 1) for _ in range(mq + 1)]  # 1=diag, 2=up(q-skip), 3=left(r-skip)

    # Free start anywhere within band around rel_off
    for j in range(mw + 1):
        if abs(j - rel_off) <= band:
            dp[0][j] = 0.0

    for i in range(1, mq + 1):
        center = rel_off + i
        j_lo = max(1, center - band)
        j_hi = min(mw, center + band)
        # Allow free-ish edge if window is narrower than band
        if j_lo > j_hi:
            j_lo, j_hi = 1, mw
        qm = q_meas[i - 1]
        for j in range(j_lo, j_hi + 1):
            sim, _, _ = _measure_similarity(qm, w_meas[j - 1])
            best_val = NEG_INF
            best_dir = 0

            # Diagonal match (i-1, j-1)
            if dp[i - 1][j - 1] > NEG_INF * 0.5:
                v = dp[i - 1][j - 1] + sim
                if v > best_val:
                    best_val = v
                    best_dir = 1
            # Query extra barline (i-1, j)
            if dp[i - 1][j] > NEG_INF * 0.5:
                v = dp[i - 1][j] - gap_penalty
                if v > best_val:
                    best_val = v
                    best_dir = 2
            # GT skipped measure (i, j-1)
            if dp[i][j - 1] > NEG_INF * 0.5:
                v = dp[i][j - 1] - gap_penalty
                if v > best_val:
                    best_val = v
                    best_dir = 3

            dp[i][j] = best_val
            bt[i][j] = best_dir

    # Best ending column at row mq
    best_j = max(range(1, mw + 1), key=lambda j: dp[mq][j])
    if dp[mq][best_j] <= NEG_INF * 0.5:
        # Fallback to unbanded clamping at voted_offset
        ms = max(0, min(voted_offset, mr - 1))
        me = min(mr, ms + mq)
        return AlignmentResult(
            work_id=work.work_id,
            m_start=ms,
            m_end=me,
            system_breaks=[(ms, me)],
            align_score=0.0,
            interval_f1=0.0,
            rhythm_f1=0.0,
            measure_pairs=[],
        )

    # Backtrack aligned measure pairs
    pairs: List[Tuple[int, int]] = []
    i, j = mq, best_j
    while i > 0 and j > 0:
        d = bt[i][j]
        if d == 1:
            pairs.append((i - 1, win_start + j - 1))
            i -= 1
            j -= 1
        elif d == 2:
            i -= 1
        elif d == 3:
            j -= 1
        else:
            break
    pairs.reverse()

    if pairs:
        m_start = pairs[0][1]
        m_end = pairs[-1][1] + 1
    else:
        m_start = max(0, min(voted_offset, mr - 1))
        m_end = min(mr, m_start + mq)

    # Map query system_breaks to GT measure spans
    q_to_gt: Dict[int, int] = {qm_i: gm_i for qm_i, gm_i in pairs}
    mapped_sys_breaks: List[Tuple[int, int]] = []
    if query.system_breaks:
        for qs, qe in query.system_breaks:
            matched_gt = [q_to_gt[m] for m in range(qs, qe) if m in q_to_gt]
            if matched_gt:
                mapped_sys_breaks.append((min(matched_gt), max(matched_gt) + 1))
    if not mapped_sys_breaks and m_end > m_start:
        mapped_sys_breaks = [(m_start, m_end)]

    # Compute global pitch-interval F1 and rhythm F1 over the aligned measure span
    q_iv_total: Counter[int] = Counter()
    q_rh_total: Counter[int] = Counter()
    r_iv_total: Counter[int] = Counter()
    r_rh_total: Counter[int] = Counter()

    for qm in q_meas:
        q_iv_total.update(qm.interval_bag)
        q_rh_total.update(qm.rhythm_bag)
    for rm in r_meas[m_start:m_end]:
        r_iv_total.update(rm.interval_bag)
        r_rh_total.update(rm.rhythm_bag)

    span_iv_f1 = _bag_f1(q_iv_total, r_iv_total)
    span_rh_f1 = _bag_f1(q_rh_total, r_rh_total)

    # Also compute mean local measure-by-measure F1 along the DTW path (rewards tight bar alignment)
    if pairs:
        loc_iv = sum(_measure_similarity(q_meas[qi], r_meas[ri])[1] for qi, ri in pairs) / float(
            max(mq, len(pairs))
        )
        loc_rh = sum(_measure_similarity(q_meas[qi], r_meas[ri])[2] for qi, ri in pairs) / float(
            max(mq, len(pairs))
        )
    else:
        loc_iv, loc_rh = 0.0, 0.0

    # Blend local measure-aligned F1 and global span F1 (0.5 / 0.5)
    iv_f1 = 0.5 * loc_iv + 0.5 * span_iv_f1
    rh_f1 = 0.5 * loc_rh + 0.5 * span_rh_f1
    align_score = 0.6 * iv_f1 + 0.4 * rh_f1

    return AlignmentResult(
        work_id=work.work_id,
        m_start=m_start,
        m_end=m_end,
        system_breaks=mapped_sys_breaks,
        align_score=round(align_score, 6),
        interval_f1=round(iv_f1, 6),
        rhythm_f1=round(rh_f1, 6),
        measure_pairs=pairs,
    )


# ---------------------------------------------------------------------------
# 8. Strict Rejection Gate (verify_harvest_candidate)
# ---------------------------------------------------------------------------


@dataclass
class HarvestDecision:
    accepted: bool
    reasons: List[str]
    query_id: str
    work_id: str
    query_n_staves: int
    work_n_staves: int
    top1_score: float
    top2_score: float
    margin: float
    align_score: float
    interval_f1: float
    rhythm_f1: float
    m_start: int
    m_end: int
    system_breaks: List[Tuple[int, int]]
    query_source_path: str = ""
    decode_hash: str = ""


def verify_harvest_candidate(
    query: ParsedScore,
    retrieval: RetrievalResult,
    alignment: Optional[AlignmentResult],
    *,
    tau_align: float = 0.38,
    tau_margin: float = 0.15,
    tau_score: float = 3.5,
    leakage_guard: Optional[EvalLeakageGuard] = None,
    check_eval_leakage: bool = True,
) -> HarvestDecision:
    """Strict rejection filter for harvested page->work alignments.

    Rejects any candidate failing:
      (a) exact ``n_staves`` parity,
      (b) ``align_score >= tau_align``,
      (c) ``margin >= tau_margin`` and ``top1_score >= tau_score``,
      (d) zero eval-set leakage against ``suite31_hw``, ``gate20``, and
          ``OMR-NorthStar`` stems.
    """
    reasons: List[str] = []
    top1 = retrieval.top1
    q_src = query.source_path or query.work_id
    dhash = compute_decode_hash(query)

    if top1 is None or alignment is None:
        return HarvestDecision(
            accepted=False,
            reasons=["NO_CANDIDATE"],
            query_id=query.work_id,
            work_id="",
            query_n_staves=query.n_staves,
            work_n_staves=0,
            top1_score=retrieval.top1_score,
            top2_score=retrieval.top2_score,
            margin=retrieval.margin,
            align_score=0.0,
            interval_f1=0.0,
            rhythm_f1=0.0,
            m_start=0,
            m_end=0,
            system_breaks=[],
            query_source_path=q_src,
            decode_hash=dhash,
        )

    # (a) Exact staff-count parity
    if query.n_staves <= 0 or query.n_staves != top1.n_staves:
        reasons.append(f"STAFF_COUNT_MISMATCH({query.n_staves}!={top1.n_staves})")

    # (b) Banded-DTW alignment score threshold
    if alignment.align_score < tau_align:
        reasons.append(f"LOW_ALIGN_SCORE({alignment.align_score:.3f}<{tau_align:.3f})")

    # (c) Shazam retrieval margin & minimum Hough score threshold
    if retrieval.top1_score < tau_score:
        reasons.append(f"LOW_SCORE({retrieval.top1_score:.2f}<{tau_score:.2f})")
    if retrieval.margin < tau_margin:
        reasons.append(f"LOW_MARGIN({retrieval.margin:.3f}<{tau_margin:.3f})")

    # (d) Zero eval-set leakage guard
    if check_eval_leakage:
        guard = leakage_guard or EvalLeakageGuard.from_files()
        leaked, leak_reason = guard.check(
            top1.work_id, top1.source_path, query.work_id, query.source_path
        )
        if leaked:
            reasons.append(f"EVAL_LEAKAGE({leak_reason})")

    return HarvestDecision(
        accepted=(len(reasons) == 0),
        reasons=reasons,
        query_id=query.work_id,
        work_id=top1.work_id,
        query_n_staves=query.n_staves,
        work_n_staves=top1.n_staves,
        top1_score=retrieval.top1_score,
        top2_score=retrieval.top2_score,
        margin=retrieval.margin,
        align_score=alignment.align_score,
        interval_f1=alignment.interval_f1,
        rhythm_f1=alignment.rhythm_f1,
        m_start=alignment.m_start,
        m_end=alignment.m_end,
        system_breaks=alignment.system_breaks,
        query_source_path=q_src,
        decode_hash=dhash,
    )


def harvest_page(
    query: ParsedScore | Sequence[str],
    index: SymbolicIndex,
    *,
    tau_align: float = 0.38,
    tau_margin: float = 0.15,
    tau_score: float = 3.5,
    leakage_guard: Optional[EvalLeakageGuard] = None,
    check_eval_leakage: bool = True,
    exclude_prefixes: Sequence[str] = (),
) -> Tuple[RetrievalResult, Optional[AlignmentResult], HarvestDecision]:
    """Convenience end-to-end call: retrieve -> banded-DTW align -> verify gate."""
    q_score = (
        query
        if isinstance(query, ParsedScore)
        else parse_bekern_tokens(query, work_id="query")
    )
    ret = retrieve_best_fit(q_score, index, exclude_prefixes=exclude_prefixes)
    if ret.top1 is None:
        dec = verify_harvest_candidate(
            q_score,
            ret,
            None,
            tau_align=tau_align,
            tau_margin=tau_margin,
            tau_score=tau_score,
            leakage_guard=leakage_guard,
            check_eval_leakage=check_eval_leakage,
        )
        return ret, None, dec

    work_meta = index.works[ret.top1.work_idx]
    aln = align_page_to_work(q_score, work_meta, ret.top1.measure_offset)
    dec = verify_harvest_candidate(
        q_score,
        ret,
        aln,
        tau_align=tau_align,
        tau_margin=tau_margin,
        tau_score=tau_score,
        leakage_guard=leakage_guard,
        check_eval_leakage=check_eval_leakage,
    )
    return ret, aln, dec


# ---------------------------------------------------------------------------
# 8b. Batch-Level Hallucination Attractor Guard (filter_attractors)
# ---------------------------------------------------------------------------


def _item_get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _item_reject(item: Any, reason: str) -> None:
    if isinstance(item, dict):
        item["accepted"] = False
        reasons = item.setdefault("reasons", [])
        if reason not in reasons:
            reasons.append(reason)
    else:
        item.accepted = False
        if reason not in item.reasons:
            item.reasons.append(reason)


def _spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """Return True if two half-open measure spans [s, e) overlap."""
    return a[0] < b[1] and b[0] < a[1]


def filter_attractors(
    results: Sequence[Any],
    max_queries_per_work: int = 2,
    max_duplicate_decodes: int = 1,
) -> List[Any]:
    """Batch-level Hallucination Attractor Guard.

    Mutates each result (``HarvestDecision`` or dict) in-place and returns the list:
      1. **Duplicate-decode rejection (``HALLUCINATION_DUPLICATE_DECODE``):**
         Computes/reads the normalized pitch-interval/token ``decode_hash`` of each
         query's decoded sequence. If the exact same ``decode_hash`` appears across
         ``> max_duplicate_decodes`` (default ``1``, i.e. ``>= 2``) distinct image
         paths, marks ALL matching items rejected with
         ``"HALLUCINATION_DUPLICATE_DECODE"``.
      2. **Attractor-work rejection (``HALLUCINATION_ATTRACTOR_WORK``):**
         If the same ``work_id`` is matched (passing per-page gates) by
         ``> max_queries_per_work`` (default ``2``) distinct query images whose
         matched measure spans overlap (or total ``> max_queries_per_work`` across
         the batch), marks all of them rejected with
         ``"HALLUCINATION_ATTRACTOR_WORK"``.
    """
    items = list(results)
    if not items:
        return items

    # Snapshot initial acceptance before batch-level rejections so attractor works
    # remain detectable even when some of their queries also collide on decode_hash.
    initially_accepted: List[bool] = [bool(_item_get(it, "accepted", False)) for it in items]

    img_keys: List[str] = []
    hash_to_indices: Dict[str, List[int]] = defaultdict(list)
    hash_to_images: Dict[str, Set[str]] = defaultdict(set)

    for idx, it in enumerate(items):
        raw_img = (
            _item_get(it, "image", "")
            or _item_get(it, "query_source_path", "")
            or _item_get(it, "source_path", "")
            or _item_get(it, "query_id", "")
        )
        img_key = str(raw_img).strip().lower() or f"__row_{idx}"
        img_keys.append(img_key)

        dhash = str(_item_get(it, "decode_hash", "") or "").strip()
        if not dhash:
            raw_seq = (
                _item_get(it, "query", None)
                or _item_get(it, "tokens", None)
                or _item_get(it, "bekern", None)
            )
            if raw_seq is not None:
                dhash = compute_decode_hash(raw_seq)
                if isinstance(it, dict):
                    it["decode_hash"] = dhash
                elif hasattr(it, "decode_hash"):
                    it.decode_hash = dhash
        if dhash:
            hash_to_indices[dhash].append(idx)
            hash_to_images[dhash].add(img_key)

    # 1. Duplicate-decode rejection
    for dhash, idxs in hash_to_indices.items():
        if len(hash_to_images[dhash]) > max_duplicate_decodes:
            for idx in idxs:
                _item_reject(items[idx], "HALLUCINATION_DUPLICATE_DECODE")

    # 2. Attractor-work rejection
    work_to_indices: Dict[str, List[int]] = defaultdict(list)
    work_to_images: Dict[str, Set[str]] = defaultdict(set)
    for idx, it in enumerate(items):
        if not initially_accepted[idx]:
            continue
        wid = str(_item_get(it, "work_id", "") or "").strip()
        if not wid:
            continue
        work_to_indices[wid].append(idx)
        work_to_images[wid].add(img_keys[idx])

    for wid, idxs in work_to_indices.items():
        distinct_imgs = work_to_images[wid]
        if len(distinct_imgs) > max_queries_per_work:
            for idx in idxs:
                _item_reject(items[idx], "HALLUCINATION_ATTRACTOR_WORK")
            continue

        # Also explicitly check overlapping measure spans per distinct image
        spans = [
            (
                img_keys[idx],
                (int(_item_get(items[idx], "m_start", 0)), int(_item_get(items[idx], "m_end", 0))),
            )
            for idx in idxs
        ]
        for idx_pos, (im_a, sp_a) in enumerate(spans):
            if sp_a[1] <= sp_a[0]:
                continue
            overlap_imgs = {im_a}
            for im_b, sp_b in spans:
                if sp_b[1] > sp_b[0] and _spans_overlap(sp_a, sp_b):
                    overlap_imgs.add(im_b)
            if len(overlap_imgs) > max_queries_per_work:
                _item_reject(items[idxs[idx_pos]], "HALLUCINATION_ATTRACTOR_WORK")

    return items


# ---------------------------------------------------------------------------
# 9. CLI Entrypoint (build-index, retrieve, sweep-rism)
# ---------------------------------------------------------------------------


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="CPU-only Fingerprint-First Forced Alignment Harvester."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("build-index", help="Build and save a SymbolicIndex.")
    p_idx.add_argument("--musicxml-dir", action="append", default=[], help="Dir of .mxl/.musicxml")
    p_idx.add_argument("--bekern-jsonl", action="append", default=[], help="Digital bekern JSONL")
    p_idx.add_argument("--out", required=True, help="Output .pkl.gz index path")
    p_idx.add_argument("--no-leakage-guard", action="store_true", help="Disable eval exclusion")
    p_idx.add_argument("--max-jsonl-rows", type=int, default=None)
    p_idx.add_argument(
        "--exclude-synthetic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude synthetic prefixes (hwsim_, hw_sim_, synth_) from indexing (default: True)",
    )

    p_ret = sub.add_parser("retrieve", help="Retrieve & align a single bekern page.")
    p_ret.add_argument("--index", required=True, help="Path to .pkl.gz index")
    p_ret.add_argument("--bekern", required=True, help="Path to decoded bekern .txt")
    p_ret.add_argument("--tau-align", type=float, default=0.38)
    p_ret.add_argument("--tau-margin", type=float, default=0.15)
    p_ret.add_argument("--no-leakage-guard", action="store_true")
    p_ret.add_argument(
        "--exclude-synthetic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude synthetic prefixes (hwsim_, hw_sim_, synth_) at retrieval time (default: True)",
    )

    p_swp = sub.add_parser(
        "sweep-jsonl",
        help="Batch-harvest a decoded bekern JSONL with synthetic exclusion & Hallucination Attractor Guard.",
    )
    p_swp.add_argument("--index", required=True, help="Path to .pkl.gz index")
    p_swp.add_argument("--jsonl", required=True, help="Input decoded bekern JSONL")
    p_swp.add_argument("--out", required=True, help="Output JSONL of clean accepted matches")
    p_swp.add_argument("--tau-align", type=float, default=0.38)
    p_swp.add_argument("--tau-margin", type=float, default=0.15)
    p_swp.add_argument("--tau-score", type=float, default=3.5)
    p_swp.add_argument("--max-queries-per-work", type=int, default=2)
    p_swp.add_argument("--max-duplicate-decodes", type=int, default=1)
    p_swp.add_argument("--no-leakage-guard", action="store_true")
    p_swp.add_argument(
        "--exclude-synthetic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude synthetic prefixes (hwsim_, hw_sim_, synth_) at retrieval time (default: True)",
    )

    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = _build_cli()
    args = ap.parse_args(argv)

    if args.cmd == "build-index":
        t0 = time.perf_counter()
        guard = None if args.no_leakage_guard else EvalLeakageGuard.from_files()
        excl = DEFAULT_SYNTHETIC_PREFIXES if args.exclude_synthetic else ()
        idx = build_index(
            musicxml_dirs=args.musicxml_dir,
            bekern_jsonls=args.bekern_jsonl,
            leakage_guard=guard,
            max_jsonl_rows=args.max_jsonl_rows,
            exclude_prefixes=excl,
        )
        idx.save(args.out)
        dt = time.perf_counter() - t0
        print(
            json.dumps(
                {
                    "out": args.out,
                    "n_works": idx.n_works,
                    "n_unique_ngrams": len(idx.postings),
                    "n_postings": idx.n_postings,
                    "skipped_duplicates": idx.skipped_duplicates,
                    "skipped_empty": idx.skipped_empty,
                    "skipped_eval_excluded": idx.skipped_eval_excluded,
                    "skipped_synthetic": getattr(idx, "skipped_synthetic", 0),
                    "elapsed_s": round(dt, 2),
                },
                indent=2,
            )
        )
        return 0

    if args.cmd == "retrieve":
        idx = SymbolicIndex.load(args.index)
        q = parse_bekern_file(args.bekern)
        excl = DEFAULT_SYNTHETIC_PREFIXES if args.exclude_synthetic else ()
        _, _, dec = harvest_page(
            q,
            idx,
            tau_align=args.tau_align,
            tau_margin=args.tau_margin,
            check_eval_leakage=not args.no_leakage_guard,
            exclude_prefixes=excl,
        )
        print(json.dumps(dec.__dict__, indent=2))
        return 0

    if args.cmd == "sweep-jsonl":
        t0 = time.perf_counter()
        idx = SymbolicIndex.load(args.index)
        guard = None if args.no_leakage_guard else EvalLeakageGuard.from_files()
        excl = DEFAULT_SYNTHETIC_PREFIXES if args.exclude_synthetic else ()

        rows_in: List[Dict[str, Any]] = []
        with Path(args.jsonl).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows_in.append(json.loads(line))

        batch_records: List[Dict[str, Any]] = []
        per_page_accepted = 0
        for row in rows_in:
            img_path = str(row.get("image", ""))
            bk_paths = row.get("bekerns") or []
            if isinstance(bk_paths, str):
                bk_paths = [bk_paths]
            sys_streams: List[List[str]] = []
            for bp in bk_paths:
                p = Path(bp)
                if p.is_file():
                    toks = p.read_text(encoding="utf-8", errors="ignore").split()
                    if toks:
                        sys_streams.append(toks)
            qid = Path(img_path).stem if img_path else (Path(bk_paths[0]).stem if bk_paths else "query")
            q_score = parse_bekern_tokens(
                [],
                work_id=qid,
                source_path=img_path,
                system_streams=sys_streams,
            )
            ret, _, dec = harvest_page(
                q_score,
                idx,
                tau_align=args.tau_align,
                tau_margin=args.tau_margin,
                tau_score=args.tau_score,
                leakage_guard=guard,
                check_eval_leakage=not args.no_leakage_guard,
                exclude_prefixes=excl,
            )
            if dec.accepted:
                per_page_accepted += 1
            rec = dict(dec.__dict__)
            rec["image"] = img_path
            rec["work_source_path"] = ret.top1.source_path if ret.top1 else ""
            rec["bekerns"] = bk_paths
            batch_records.append(rec)

        filter_attractors(
            batch_records,
            max_queries_per_work=args.max_queries_per_work,
            max_duplicate_decodes=args.max_duplicate_decodes,
        )

        dup_rejected = sum(
            1 for r in batch_records if "HALLUCINATION_DUPLICATE_DECODE" in r["reasons"]
        )
        attr_rejected = sum(
            1 for r in batch_records if "HALLUCINATION_ATTRACTOR_WORK" in r["reasons"]
        )
        clean = [r for r in batch_records if r["accepted"]]
        clean.sort(key=lambda r: (r["align_score"], r["top1_score"]), reverse=True)

        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as fh:
            for r in clean:
                fh.write(json.dumps(r) + "\n")

        dt = time.perf_counter() - t0
        summary = {
            "total_queries": len(batch_records),
            "per_page_accepted_before_attractor_guard": per_page_accepted,
            "rejected_duplicate_decode": dup_rejected,
            "rejected_attractor_work": attr_rejected,
            "clean_accepted": len(clean),
            "out": str(out_p),
            "elapsed_s": round(dt, 2),
            "top5": [
                {
                    "query_id": r["query_id"],
                    "work_id": r["work_id"],
                    "align_score": r["align_score"],
                    "margin": r["margin"],
                    "top1_score": r["top1_score"],
                    "m_start": r["m_start"],
                    "m_end": r["m_end"],
                }
                for r in clean[:5]
            ],
        }
        print(json.dumps(summary, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
