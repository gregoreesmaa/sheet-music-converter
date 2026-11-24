"""Complete MuseScore (.mscx/.mscz) & MusicXML (.musicxml/.mxl) parser and exporter.

Supports MuseScore 3 & 4 XML and MusicXML Partwise 3.1/4.0 with full notation:
- Multi-voice polyphony (voices 1..4 per staff), opposing stems & collision nudges
- Ties & Slurs (intra-measure and cross-measure spans)
- Tuplets (triplets, quintuplets, etc.)
- Articulations & Ornaments (staccato, accent, tenuto, fermata, trill, grace notes)
- Dynamics, Hairpins (crescendo/diminuendo), Tempo text, Under-staff Lyrics
- Mid-measure & mid-piece clef/key/time changes (including 8vb/8va clefs)
- Double sharps/flats, courtesy accidentals, multi-bar rests, volta repeat brackets
- MusicXML export helper for generating exact `assembly-serialize bekern` targets
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Musical Constants
# ---------------------------------------------------------------------------

DURATION_BEATS: Dict[str, float] = {
    "breve": 8.0,
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "16th": 0.25,
    "32nd": 0.125,
    "64th": 0.0625,
}

# Diatonic step of bottom staff line (step 0) relative to C0 = 0
# (C0=0, D0=1, ..., C4=28, E4=30, G4=32, B4=34, F5=38)
CLEF_BOTTOM_DIATONIC: Dict[str, int] = {
    "G-2": 30,      # E4 on line 1 (G4 on line 2)
    "G-2-8vb": 23,  # E3 on line 1 (treble 8vb)
    "G-2-8va": 37,  # E5 on line 1 (treble 8va)
    "F-4": 18,      # G2 on line 1 (F3 on line 4)
    "F-4-8vb": 11,  # G1 on line 1 (bass 8vb)
    "F-4-8va": 25,  # G3 on line 1 (bass 8va)
    "C-3": 24,      # F3 on line 1 (C4 on line 3)
    "C-4": 22,      # D3 on line 1 (C4 on line 4)
}

LETTER_TO_DIATONIC: Dict[str, int] = {
    "C": 0,
    "D": 1,
    "E": 2,
    "F": 3,
    "G": 4,
    "A": 5,
    "B": 6,
}

DIATONIC_TO_LETTER: Tuple[str, ...] = ("C", "D", "E", "F", "G", "A", "B")
NATURAL_SEMITONES: Tuple[int, ...] = (0, 2, 4, 5, 7, 9, 11)

# Key signature staff steps (relative to bottom line = 0)
SHARP_STEPS_BY_CLEF: Dict[str, Tuple[int, ...]] = {
    "G-2": (8, 5, 9, 6, 3, 7, 4),  # F5 C5 G5 D5 A4 E5 B4
    "F-4": (6, 3, 7, 4, 1, 5, 2),  # F3 C3 G3 D3 A2 E3 B2
    "C-3": (7, 4, 8, 5, 2, 6, 3),
    "C-4": (2, 6, 3, 7, 4, 8, 5),
}

FLAT_STEPS_BY_CLEF: Dict[str, Tuple[int, ...]] = {
    "G-2": (4, 7, 3, 6, 2, 5, 1),  # B4 E5 A4 D5 G4 C5 F4
    "F-4": (2, 5, 1, 4, 0, 3, -1),  # B2 E3 A2 D3 G2 C3 F2
    "C-3": (3, 6, 2, 5, 1, 4, 0),
    "C-4": (5, 8, 4, 7, 3, 6, 2),
}


# ---------------------------------------------------------------------------
# Score Data Model
# ---------------------------------------------------------------------------


@dataclass
class NoteEvent:
    """Single note or chord at a rhythmic position."""

    staff_steps: List[int]  # sorted low-to-high, 0 = bottom staff line, 8 = top line
    accidentals: List[Optional[str]]  # 'sharp', 'flat', 'natural', 'double_sharp', 'double_flat', None
    duration_kind: str  # 'whole', 'half', 'quarter', 'eighth', '16th', '32nd'
    dots: int = 0
    beat_offset: float = 0.0
    beam_group: Optional[int] = None
    stem_dir: Optional[int] = None  # +1 up, -1 down, None = auto
    voice: int = 1
    is_grace: bool = False
    grace_slash: bool = False
    courtesy_accidentals: List[bool] = field(default_factory=list)
    articulations: List[str] = field(default_factory=list)
    tie_starts: List[bool] = field(default_factory=list)
    tie_stops: List[bool] = field(default_factory=list)
    slur_starts: List[int] = field(default_factory=list)
    slur_stops: List[int] = field(default_factory=list)
    tuplet_id: Optional[int] = None
    tuplet_actual: Optional[int] = None
    tuplet_normal: Optional[int] = None
    tuplet_pos: Optional[str] = None  # 'start', 'middle', 'stop'
    lyrics: List[Tuple[int, str]] = field(default_factory=list)  # (verse_index, text)
    x_nudge_sp: float = 0.0
    head_x_nudges_sp: List[float] = field(default_factory=list)
    pitches_info: List[Tuple[str, int, int]] = field(default_factory=list)

    @property
    def beats(self) -> float:
        if self.is_grace:
            return 0.0
        base = DURATION_BEATS.get(self.duration_kind, 1.0)
        mult = 1.0
        add = 0.5
        for _ in range(self.dots):
            mult += add
            add *= 0.5
        dur = base * mult
        if self.tuplet_actual and self.tuplet_normal and self.tuplet_actual > 0:
            dur = dur * (float(self.tuplet_normal) / float(self.tuplet_actual))
        return dur


@dataclass
class RestEvent:
    """Rest at a rhythmic position."""

    duration_kind: str  # 'whole', 'half', 'quarter', 'eighth', '16th', '32nd'
    dots: int = 0
    beat_offset: float = 0.0
    is_measure_rest: bool = False
    voice: int = 1
    multi_measure_count: int = 0
    articulations: List[str] = field(default_factory=list)
    tuplet_id: Optional[int] = None
    tuplet_actual: Optional[int] = None
    tuplet_normal: Optional[int] = None
    tuplet_pos: Optional[str] = None

    @property
    def beats(self) -> float:
        base = DURATION_BEATS.get(self.duration_kind, 1.0)
        mult = 1.0
        add = 0.5
        for _ in range(self.dots):
            mult += add
            add *= 0.5
        dur = base * mult
        if self.tuplet_actual and self.tuplet_normal and self.tuplet_actual > 0:
            dur = dur * (float(self.tuplet_normal) / float(self.tuplet_actual))
        return dur


ScoreEvent = Union[NoteEvent, RestEvent]


@dataclass
class DynamicEvent:
    """Dynamic marking (p, pp, mp, mf, f, ff, sfz, fp, etc.) at a beat offset."""

    beat_offset: float
    text: str
    placement: str = "below"


@dataclass
class HairpinSpan:
    """Crescendo (<) or diminuendo (>) wedge spanning across beats/measures."""

    kind: str  # 'crescendo' or 'diminuendo'
    start_measure: int
    start_beat: float
    end_measure: int
    end_beat: float
    placement: str = "below"


@dataclass
class TempoEvent:
    """Tempo or expression text above the staff."""

    beat_offset: float
    text: str


@dataclass
class MidMeasureChange:
    """Mid-measure clef, key signature, or time signature change."""

    beat_offset: float
    kind: str  # 'clef', 'key', 'time'
    value: Any


@dataclass
class VoltaBracket:
    """First/second ending (volta) repeat bracket."""

    text: str  # e.g. '1.', '2.'
    start_measure: int
    end_measure: int
    closed_right: bool = True


@dataclass
class TieSlurSpan:
    """Resolved tie or slur connecting two NoteEvents on a staff."""

    kind: str  # 'tie' or 'slur'
    span_id: int
    voice: int
    start_measure: int
    start_event_idx: int
    start_step: int
    end_measure: int
    end_event_idx: int
    end_step: int


@dataclass
class MeasureData:
    """One measure within a single staff."""

    number: int
    clef: str = "G-2"
    key_fifths: int = 0
    time_sig: Tuple[int, int] = (4, 4)
    clef_change: bool = False
    key_change: bool = False
    time_change: bool = False
    barline: str = "single"  # 'single', 'double', 'final', 'repeat_right', 'repeat_left'
    events: List[ScoreEvent] = field(default_factory=list)
    left_barline: str = "none"  # 'none', 'repeat_left'
    multi_bar_rest: int = 0
    mid_changes: List[MidMeasureChange] = field(default_factory=list)
    dynamics: List[DynamicEvent] = field(default_factory=list)
    tempos: List[TempoEvent] = field(default_factory=list)
    volta: Optional[VoltaBracket] = None
    voices_present: List[int] = field(default_factory=lambda: [1])


@dataclass
class PartData:
    """One notated staff (e.g. Soprano, Piano RH, Piano LH)."""

    name: str
    staff_index: int
    measures: List[MeasureData] = field(default_factory=list)
    brace_group: Optional[int] = None
    hairpins: List[HairpinSpan] = field(default_factory=list)
    slurs: List[TieSlurSpan] = field(default_factory=list)
    ties: List[TieSlurSpan] = field(default_factory=list)


@dataclass
class ParsedScore:
    """Normalized multi-staff score extracted from MusicXML or MuseScore XML."""

    title: str
    source_path: str
    parts: List[PartData] = field(default_factory=list)

    @property
    def num_measures(self) -> int:
        if not self.parts:
            return 0
        return max((len(p.measures) for p in self.parts), default=0)


# ---------------------------------------------------------------------------
# Normalization & Musical Helpers
# ---------------------------------------------------------------------------


def base_clef_code(clef: str) -> str:
    """Return base clef ('G-2', 'F-4', 'C-3', 'C-4') stripping '-8vb'/'-8va'."""
    if clef.endswith("-8vb") or clef.endswith("-8va"):
        return clef[:-4]
    return clef


def clef_octave_shift(clef: str) -> int:
    """Return -1 for 8vb, +1 for 8va, else 0."""
    if clef.endswith("-8vb"):
        return -1
    if clef.endswith("-8va"):
        return 1
    return 0


def _normalize_duration_kind(raw: Optional[str]) -> str:
    if not raw:
        return "quarter"
    k = raw.strip().lower()
    mapping = {
        "breve": "whole",
        "long": "whole",
        "whole": "whole",
        "half": "half",
        "quarter": "quarter",
        "eighth": "eighth",
        "8th": "eighth",
        "16th": "16th",
        "sixteenth": "16th",
        "32nd": "32nd",
        "64th": "32nd",
        "128th": "32nd",
        "measure": "whole",
    }
    return mapping.get(k, "quarter")


def _normalize_accidental(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    k = raw.strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "sharp": "sharp",
        "flat": "flat",
        "natural": "natural",
        "double_sharp": "double_sharp",
        "double_flat": "double_flat",
        "sharp_sharp": "double_sharp",
        "flat_flat": "double_flat",
        "sharp2": "double_sharp",
        "flat2": "double_flat",
        "accidentalsharp": "sharp",
        "accidentalflat": "flat",
        "accidentalnatural": "natural",
        "accidentaldoublesharp": "double_sharp",
        "accidentaldoubleflat": "double_flat",
        "accidentalsharp2": "double_sharp",
        "accidentalflat2": "double_flat",
    }
    return mapping.get(k)


def _diatonic_to_staff_step(letter: str, octave: int, clef: str) -> int:
    diatonic = octave * 7 + LETTER_TO_DIATONIC.get(letter.upper(), 0)
    bottom = CLEF_BOTTOM_DIATONIC.get(clef, CLEF_BOTTOM_DIATONIC.get(base_clef_code(clef), 30))
    return diatonic - bottom


def _staff_step_to_letter_octave(step: int, clef: str) -> Tuple[str, int]:
    bottom = CLEF_BOTTOM_DIATONIC.get(clef, CLEF_BOTTOM_DIATONIC.get(base_clef_code(clef), 30))
    diatonic = bottom + step
    octave = diatonic // 7
    letter = DIATONIC_TO_LETTER[diatonic % 7]
    return letter, octave


def _mscx_pitch_tpc_to_letter_octave_alter(pitch: int, tpc: Optional[int]) -> Tuple[str, int, int]:
    """Convert MuseScore MIDI pitch + tonal pitch class (tpc) to (letter, octave, alter)."""
    letters = ("C", "D", "E", "F", "G", "A", "B")
    if tpc is not None:
        fifths = tpc - 14
        letter_idx = (fifths * 4) % 7
        alter = (fifths + 1) // 7
        nat_semi = NATURAL_SEMITONES[letter_idx]
        octave = int(round((pitch - alter - nat_semi) / 12.0)) - 1
        return letters[letter_idx], octave, alter
    pc_map = [
        ("C", 0), ("C", 1), ("D", 0), ("E", -1), ("E", 0), ("F", 0),
        ("F", 1), ("G", 0), ("A", -1), ("A", 0), ("B", -1), ("B", 0),
    ]
    letter, alter = pc_map[pitch % 12]
    octave = (pitch // 12) - 1
    return letter, octave, alter


def _strip_xml_tags(text: str) -> str:
    """Clean MuseScore rich-text markup (<sym>...</sym>, <b>, etc.) into readable text."""
    if not text:
        return ""
    s = re.sub(r"<sym>metNoteQuarterUp</sym>", "q", text)
    s = re.sub(r"<sym>metNoteHalfUp</sym>", "h", s)
    s = re.sub(r"<sym>metNote8thUp</sym>", "e", s)
    s = re.sub(r"<sym>metAugmentationDot</sym>", ".", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_fraction_beats(frac_str: Optional[str]) -> float:
    """Parse MuseScore fraction like '1/4' or '-3/8' into quarter-note beats."""
    if not frac_str:
        return 0.0
    s = frac_str.strip()
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            return 4.0 * float(a) / float(b)
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return 4.0 * float(s)
    except ValueError:
        return 0.0


def _assign_auto_beams(events: List[ScoreEvent], time_sig: Tuple[int, int]) -> None:
    """Assign beat-grouped beam_group IDs per voice to beamable NoteEvents lacking explicit beams."""
    voices = sorted({getattr(e, "voice", 1) for e in events})
    num, den = time_sig
    if den == 8 and num in (6, 9, 12):
        group_beats = 1.5
    elif den == 2:
        group_beats = 2.0
    else:
        group_beats = 1.0

    beamable = {"eighth", "16th", "32nd"}
    existing_ids = [
        e.beam_group
        for e in events
        if isinstance(e, NoteEvent) and e.beam_group is not None
    ]
    next_group_id = (max(existing_ids) + 1) if existing_ids else 0

    for v in voices:
        v_events = [e for e in events if getattr(e, "voice", 1) == v]
        if any(isinstance(e, NoteEvent) and e.beam_group is not None for e in v_events):
            continue

        run: List[NoteEvent] = []
        run_bucket: Optional[int] = None

        def flush() -> None:
            nonlocal next_group_id, run
            if len(run) >= 2:
                for ev in run:
                    ev.beam_group = next_group_id
                next_group_id += 1
            run = []

        for ev in v_events:
            if isinstance(ev, NoteEvent) and not ev.is_grace and ev.duration_kind in beamable:
                bucket = int(math.floor((ev.beat_offset + 1e-4) / group_beats))
                if run_bucket is not None and bucket != run_bucket:
                    flush()
                run_bucket = bucket
                run.append(ev)
            else:
                if isinstance(ev, NoteEvent) and ev.is_grace:
                    continue
                flush()
                run_bucket = None
        flush()


def resolve_polyphony_and_collisions(meas: MeasureData) -> None:
    """Resolve multi-voice stem directions and horizontal collision nudges for a measure.

    - When >= 2 voices have notes in a measure, odd voices (1, 3) get stems up (+1)
      and even voices (2, 4) get stems down (-1) unless explicitly set.
    - Within each chord, adjacent seconds (step diff <= 1) get alternating notehead
      x-nudges on the opposite side of the stem.
    - Across simultaneous voices at the same beat_offset, close/colliding noteheads
      (within <= 1 staff step or crossing) receive an x_nudge_sp so they never overlap.
    """
    note_voices = sorted(
        {ev.voice for ev in meas.events if isinstance(ev, NoteEvent) and not ev.is_grace}
    )
    all_voices = sorted({getattr(ev, "voice", 1) for ev in meas.events}) or [1]
    meas.voices_present = all_voices
    is_poly = len(note_voices) >= 2

    # 1. Assign opposing stem directions in polyphonic measures
    if is_poly:
        for ev in meas.events:
            if isinstance(ev, NoteEvent) and not ev.is_grace and ev.stem_dir is None:
                ev.stem_dir = 1 if (ev.voice % 2 == 1) else -1

    # 2. Within-chord seconds collision nudge
    for ev in meas.events:
        if not isinstance(ev, NoteEvent):
            continue
        n_heads = len(ev.staff_steps)
        nudges = [0.0] * n_heads
        if n_heads >= 2:
            sdir = ev.stem_dir
            if sdir is None:
                avg_step = sum(ev.staff_steps) / float(n_heads)
                sdir = -1 if avg_step >= 4.0 else 1
            if sdir == 1:
                # Stem up (right side): scan bottom-to-top, push upper note of a 2nd to the right
                for i in range(1, n_heads):
                    if ev.staff_steps[i] - ev.staff_steps[i - 1] <= 1 and nudges[i - 1] == 0.0:
                        nudges[i] = 1.05
            else:
                # Stem down (left side): scan top-to-bottom, push lower note of a 2nd to the left
                for i in range(n_heads - 2, -1, -1):
                    if ev.staff_steps[i + 1] - ev.staff_steps[i] <= 1 and nudges[i + 1] == 0.0:
                        nudges[i] = -1.05
        ev.head_x_nudges_sp = nudges

    # 3. Cross-voice simultaneous beat collision nudges
    if is_poly:
        by_beat: Dict[int, List[NoteEvent]] = {}
        for ev in meas.events:
            if isinstance(ev, NoteEvent) and not ev.is_grace:
                key = int(round(ev.beat_offset * 96.0))
                by_beat.setdefault(key, []).append(ev)

        for beat_notes in by_beat.values():
            if len(beat_notes) < 2:
                continue
            beat_notes.sort(key=lambda e: e.voice)
            v1 = beat_notes[0]
            for v_other in beat_notes[1:]:
                if not v1.staff_steps or not v_other.staff_steps:
                    continue
                min_dist = min(
                    abs(s1 - s2) for s1 in v1.staff_steps for s2 in v_other.staff_steps
                )
                # Unison (0) with different durations, or 2nd (1), or voice crossing
                crossed = max(v_other.staff_steps) > min(v1.staff_steps)
                diff_head_shape = (v1.duration_kind in ("whole", "half")) != (
                    v_other.duration_kind in ("whole", "half")
                )
                if min_dist == 1 or crossed or (min_dist == 0 and diff_head_shape):
                    v_other.x_nudge_sp = 1.15


def _resolve_part_ties_and_slurs(part: PartData) -> None:
    """Resolve tie_starts/tie_stops and slur_starts/slur_stops into TieSlurSpan lists."""
    open_ties: Dict[Tuple[int, int], Tuple[int, int]] = {}  # (voice, step) -> (m_idx, ev_idx)
    open_slurs: Dict[Tuple[int, int], Tuple[int, int, int]] = {}  # (voice, slur_id) -> (m_idx, ev_idx, step)
    ties: List[TieSlurSpan] = []
    slurs: List[TieSlurSpan] = []
    tie_counter = 1

    for m_idx, meas in enumerate(part.measures):
        for ev_idx, ev in enumerate(meas.events):
            if not isinstance(ev, NoteEvent):
                continue
            # Resolve tie stops first (so chained ties close then reopen cleanly)
            for h_i, step in enumerate(ev.staff_steps):
                is_stop = h_i < len(ev.tie_stops) and ev.tie_stops[h_i]
                if is_stop and (ev.voice, step) in open_ties:
                    sm, se = open_ties.pop((ev.voice, step))
                    ties.append(
                        TieSlurSpan(
                            kind="tie",
                            span_id=tie_counter,
                            voice=ev.voice,
                            start_measure=sm,
                            start_event_idx=se,
                            start_step=step,
                            end_measure=m_idx,
                            end_event_idx=ev_idx,
                            end_step=step,
                        )
                    )
                    tie_counter += 1
            for h_i, step in enumerate(ev.staff_steps):
                is_start = h_i < len(ev.tie_starts) and ev.tie_starts[h_i]
                if is_start:
                    open_ties[(ev.voice, step)] = (m_idx, ev_idx)

            # Resolve slurs
            rep_step = ev.staff_steps[-1] if (ev.stem_dir == -1) else ev.staff_steps[0]
            for sid in ev.slur_stops:
                key = (ev.voice, sid)
                if key not in open_slurs:
                    # Fallback match any voice with same slur_id
                    match = next((k for k in open_slurs if k[1] == sid), None)
                    key = match if match is not None else key
                if key in open_slurs:
                    sm, se, s_step = open_slurs.pop(key)
                    slurs.append(
                        TieSlurSpan(
                            kind="slur",
                            span_id=sid,
                            voice=ev.voice,
                            start_measure=sm,
                            start_event_idx=se,
                            start_step=s_step,
                            end_measure=m_idx,
                            end_event_idx=ev_idx,
                            end_step=rep_step,
                        )
                    )
            for sid in ev.slur_starts:
                open_slurs[(ev.voice, sid)] = (m_idx, ev_idx, rep_step)

    part.ties = ties
    part.slurs = slurs


# ---------------------------------------------------------------------------
# Direct MusicXML (.musicxml / .xml / .mxl) Parser (Full Feature Support)
# ---------------------------------------------------------------------------


def _mxml_clef_code(clef_el: ET.Element) -> str:
    sign = (clef_el.findtext("sign") or "G").strip().upper()
    line_txt = (clef_el.findtext("line") or "").strip()
    if not line_txt:
        line = 2 if sign == "G" else (4 if sign == "F" else 3)
    else:
        try:
            line = int(line_txt)
        except ValueError:
            line = 2
    base = f"{sign}-{line}"
    if base not in ("G-2", "F-4", "C-3", "C-4"):
        if sign == "F":
            base = "F-4"
        elif sign == "C":
            base = "C-3" if line == 3 else "C-4"
        else:
            base = "G-2"
    oct_txt = clef_el.findtext("clef-octave-change")
    if oct_txt:
        try:
            oct_ch = int(oct_txt.strip())
            if oct_ch < 0 and f"{base}-8vb" in CLEF_BOTTOM_DIATONIC:
                return f"{base}-8vb"
            if oct_ch > 0 and f"{base}-8va" in CLEF_BOTTOM_DIATONIC:
                return f"{base}-8va"
        except ValueError:
            pass
    return base


def _parse_musicxml_xml(xml_text: str, source_path: str) -> ParsedScore:
    root = ET.fromstring(xml_text)

    # Strip XML namespace if present
    if "}" in root.tag:
        for el in root.iter():
            if "}" in el.tag:
                el.tag = el.tag.split("}", 1)[1]

    title = Path(source_path).stem
    work_title = root.findtext("work/work-title") or root.findtext("movement-title")
    if work_title and work_title.strip():
        title = work_title.strip()

    # Parse part-list for names and brace groups
    part_names: Dict[str, str] = {}
    part_brace_group: Dict[str, int] = {}
    part_list = root.find("part-list")
    if part_list is not None:
        active_brace: Optional[int] = None
        brace_counter = 0
        for child in part_list:
            if child.tag == "part-group":
                ptype = child.attrib.get("type", "")
                sym = (child.findtext("group-symbol") or "").strip().lower()
                if ptype == "start" and sym == "brace":
                    active_brace = brace_counter
                    brace_counter += 1
                elif ptype == "stop":
                    active_brace = None
            elif child.tag == "score-part":
                pid = child.attrib.get("id", "")
                pname = (child.findtext("part-name") or pid or "Part").strip()
                part_names[pid] = pname
                if active_brace is not None:
                    part_brace_group[pid] = active_brace

    xml_parts = root.findall("part")
    # Heuristic brace if 2 single-staff piano parts lack explicit part-group
    if not part_brace_group and len(xml_parts) == 2:
        names = [part_names.get(p.attrib.get("id", ""), "").lower() for p in xml_parts]
        if any("piano" in n for n in names):
            for p in xml_parts:
                part_brace_group[p.attrib.get("id", "")] = 0

    parsed_parts: List[PartData] = []
    next_brace_id = (max(part_brace_group.values()) + 1) if part_brace_group else 10

    for p_el in xml_parts:
        pid = p_el.attrib.get("id", f"P{len(parsed_parts) + 1}")
        pname = part_names.get(pid, f"Part {len(parsed_parts) + 1}")

        # Determine number of staves in this <part> (e.g. grand staff = 2)
        n_staves = 1
        for st_el in p_el.findall(".//attributes/staves"):
            if st_el.text:
                try:
                    n_staves = max(n_staves, int(st_el.text.strip()))
                except ValueError:
                    pass

        p_brace = part_brace_group.get(pid)
        if n_staves >= 2 and p_brace is None:
            p_brace = next_brace_id
            next_brace_id += 1

        # Per-staff running state
        active_clef: Dict[int, str] = {
            s: ("F-4" if (n_staves >= 2 and s == 2) else "G-2") for s in range(1, n_staves + 1)
        }
        active_key: Dict[int, int] = {s: 0 for s in range(1, n_staves + 1)}
        active_ts: Dict[int, Tuple[int, int]] = {s: (4, 4) for s in range(1, n_staves + 1)}
        staff_measures: Dict[int, List[MeasureData]] = {s: [] for s in range(1, n_staves + 1)}
        staff_hairpins: Dict[int, List[HairpinSpan]] = {s: [] for s in range(1, n_staves + 1)}
        open_hairpins: Dict[Tuple[int, int], Tuple[str, int, float, str]] = {}  # (staff, num) -> (kind, m_idx, beat, placement)
        open_volta: Dict[int, VoltaBracket] = {}

        divisions = 1
        xml_measures = p_el.findall("measure")

        for m_idx, m_el in enumerate(xml_measures):
            m_num_raw = m_el.attrib.get("number", str(m_idx + 1))
            try:
                m_num = int(re.sub(r"[^0-9]", "", m_num_raw) or (m_idx + 1))
            except ValueError:
                m_num = m_idx + 1

            clef_change = {s: False for s in range(1, n_staves + 1)}
            key_change = {s: False for s in range(1, n_staves + 1)}
            time_change = {s: False for s in range(1, n_staves + 1)}
            mid_changes: Dict[int, List[MidMeasureChange]] = {s: [] for s in range(1, n_staves + 1)}
            dynamics: Dict[int, List[DynamicEvent]] = {s: [] for s in range(1, n_staves + 1)}
            tempos: Dict[int, List[TempoEvent]] = {s: [] for s in range(1, n_staves + 1)}
            events_by_staff: Dict[int, List[ScoreEvent]] = {s: [] for s in range(1, n_staves + 1)}
            multi_bar_rest = {s: 0 for s in range(1, n_staves + 1)}
            right_barline = "single"
            left_barline = "none"
            volta_for_measure: Dict[int, Optional[VoltaBracket]] = {s: None for s in range(1, n_staves + 1)}

            # Track last NoteEvent per (staff, voice) for <chord/>
            last_note: Dict[Tuple[int, int], NoteEvent] = {}
            # Track explicit beam groups per (staff, voice)
            beam_open: Dict[Tuple[int, int], Optional[int]] = {}
            beam_counter = 0
            # Track active tuplets per (staff, voice)
            tuplet_active: Dict[Tuple[int, int], Optional[Tuple[int, int, int]]] = {}
            tuplet_counter = 1

            beat_cursor = 0.0
            seen_any_note_or_rest = False

            for child in m_el:
                tag = child.tag

                if tag == "attributes":
                    div_txt = child.findtext("divisions")
                    if div_txt:
                        try:
                            divisions = max(1, int(float(div_txt.strip())))
                        except ValueError:
                            pass

                    # Key signature
                    key_el = child.find("key")
                    if key_el is not None:
                        f_txt = key_el.findtext("fifths")
                        if f_txt is not None:
                            try:
                                new_key = max(-7, min(7, int(f_txt.strip())))
                                k_staff_attr = key_el.attrib.get("number")
                                target_staves = (
                                    [int(k_staff_attr)] if (k_staff_attr and k_staff_attr.isdigit()) else list(range(1, n_staves + 1))
                                )
                                for s in target_staves:
                                    if s not in active_key:
                                        continue
                                    if seen_any_note_or_rest and beat_cursor > 1e-4:
                                        if new_key != active_key[s]:
                                            mid_changes[s].append(MidMeasureChange(beat_cursor, "key", new_key))
                                            active_key[s] = new_key
                                    else:
                                        if new_key != active_key[s]:
                                            key_change[s] = m_idx > 0
                                            active_key[s] = new_key
                            except ValueError:
                                pass

                    # Time signature
                    time_el = child.find("time")
                    if time_el is not None:
                        b_txt = time_el.findtext("beats")
                        bt_txt = time_el.findtext("beat-type")
                        if b_txt and bt_txt:
                            try:
                                new_ts = (int(b_txt.strip()), int(bt_txt.strip()))
                                for s in range(1, n_staves + 1):
                                    if seen_any_note_or_rest and beat_cursor > 1e-4:
                                        if new_ts != active_ts[s]:
                                            mid_changes[s].append(MidMeasureChange(beat_cursor, "time", new_ts))
                                            active_ts[s] = new_ts
                                    else:
                                        if new_ts != active_ts[s]:
                                            time_change[s] = m_idx > 0
                                            active_ts[s] = new_ts
                            except ValueError:
                                pass

                    # Clef(s)
                    for clef_el in child.findall("clef"):
                        c_num = clef_el.attrib.get("number")
                        s = int(c_num) if (c_num and c_num.isdigit()) else 1
                        if s not in active_clef:
                            continue
                        new_clef = _mxml_clef_code(clef_el)
                        if seen_any_note_or_rest and beat_cursor > 1e-4:
                            if new_clef != active_clef[s]:
                                mid_changes[s].append(MidMeasureChange(beat_cursor, "clef", new_clef))
                                active_clef[s] = new_clef
                        else:
                            if new_clef != active_clef[s] or m_idx == 0:
                                clef_change[s] = m_idx > 0 and new_clef != active_clef[s]
                                active_clef[s] = new_clef

                    # Multi-bar rest
                    mr_el = child.find("measure-style/multiple-rest")
                    if mr_el is not None and mr_el.text:
                        try:
                            mr_n = int(mr_el.text.strip())
                            if mr_n > 1:
                                for s in range(1, n_staves + 1):
                                    multi_bar_rest[s] = mr_n
                        except ValueError:
                            pass

                elif tag == "backup":
                    dur_txt = child.findtext("duration")
                    if dur_txt:
                        try:
                            beat_cursor = max(0.0, beat_cursor - float(dur_txt) / float(divisions))
                        except ValueError:
                            pass

                elif tag == "forward":
                    dur_txt = child.findtext("duration")
                    if dur_txt:
                        try:
                            beat_cursor += float(dur_txt) / float(divisions)
                        except ValueError:
                            pass

                elif tag == "direction":
                    placement = child.attrib.get("placement", "below")
                    d_staff_txt = child.findtext("staff")
                    d_staff = int(d_staff_txt) if (d_staff_txt and d_staff_txt.isdigit()) else 1
                    d_staff = max(1, min(n_staves, d_staff))

                    for dt in child.findall("direction-type"):
                        # Dynamics
                        dyn_el = dt.find("dynamics")
                        if dyn_el is not None:
                            for d_child in dyn_el:
                                d_txt = d_child.tag if d_child.tag != "other-dynamics" else (d_child.text or "")
                                if d_txt:
                                    dynamics[d_staff].append(
                                        DynamicEvent(beat_offset=beat_cursor, text=d_txt.strip(), placement=placement)
                                    )
                        # Hairpin wedge
                        wedge_el = dt.find("wedge")
                        if wedge_el is not None:
                            wtype = (wedge_el.attrib.get("type") or "").lower()
                            wnum = int(wedge_el.attrib.get("number", "1") or "1")
                            if wtype in ("crescendo", "diminuendo"):
                                open_hairpins[(d_staff, wnum)] = (wtype, m_idx, beat_cursor, placement)
                            elif wtype == "stop" and (d_staff, wnum) in open_hairpins:
                                wkind, sm, sb, wpl = open_hairpins.pop((d_staff, wnum))
                                staff_hairpins[d_staff].append(
                                    HairpinSpan(
                                        kind=wkind,
                                        start_measure=sm,
                                        start_beat=sb,
                                        end_measure=m_idx,
                                        end_beat=beat_cursor,
                                        placement=wpl,
                                    )
                                )
                        # Tempo / Words
                        words_el = dt.find("words")
                        if words_el is not None and words_el.text and words_el.text.strip():
                            tempos[d_staff].append(
                                TempoEvent(beat_offset=beat_cursor, text=words_el.text.strip())
                            )
                        met_el = dt.find("metronome")
                        if met_el is not None:
                            bu = (met_el.findtext("beat-unit") or "quarter").strip()
                            pm = (met_el.findtext("per-minute") or "").strip()
                            if pm:
                                sym = {"quarter": "q", "half": "h", "eighth": "e"}.get(bu, "q")
                                tempos[d_staff].append(
                                    TempoEvent(beat_offset=beat_cursor, text=f"{sym} = {pm}")
                                )

                elif tag == "barline":
                    loc = (child.attrib.get("location") or "right").lower()
                    bstyle = (child.findtext("bar-style") or "").lower()
                    rep_el = child.find("repeat")
                    end_el = child.find("ending")

                    if loc == "left":
                        if rep_el is not None and rep_el.attrib.get("direction") == "forward":
                            left_barline = "repeat_left"
                    else:
                        if rep_el is not None and rep_el.attrib.get("direction") == "backward":
                            right_barline = "repeat_right"
                        elif bstyle in ("light-heavy", "final"):
                            right_barline = "final"
                        elif bstyle in ("light-light", "double"):
                            right_barline = "double"

                    if end_el is not None:
                        etype = (end_el.attrib.get("type") or "").lower()
                        etxt = (end_el.text or "").strip() or (end_el.attrib.get("number", "1") + ".")
                        if not etxt.endswith("."):
                            etxt += "."
                        if etype == "start":
                            vb = VoltaBracket(text=etxt, start_measure=m_idx, end_measure=m_idx, closed_right=False)
                            for s in range(1, n_staves + 1):
                                open_volta[s] = vb
                                volta_for_measure[s] = vb
                        elif etype in ("stop", "discontinue"):
                            for s in range(1, n_staves + 1):
                                if s in open_volta:
                                    open_volta[s].end_measure = m_idx
                                    open_volta[s].closed_right = (etype == "stop")
                                    volta_for_measure[s] = open_volta[s]
                                    del open_volta[s]

                elif tag == "note":
                    seen_any_note_or_rest = True
                    is_chord = child.find("chord") is not None
                    grace_el = child.find("grace")
                    is_grace = grace_el is not None
                    grace_slash = bool(grace_el is not None and grace_el.attrib.get("slash") == "yes")

                    st_txt = child.findtext("staff")
                    s = int(st_txt) if (st_txt and st_txt.isdigit()) else 1
                    s = max(1, min(n_staves, s))

                    v_txt = child.findtext("voice")
                    try:
                        voice = max(1, min(4, int(v_txt))) if v_txt else 1
                    except ValueError:
                        voice = 1

                    dur_txt = child.findtext("duration")
                    dur_beats = (float(dur_txt) / float(divisions)) if (dur_txt and not is_grace) else 0.0
                    raw_type = child.findtext("type")
                    dots = len(child.findall("dot"))

                    # Tuplet / time-modification
                    tm_el = child.find("time-modification")
                    t_actual: Optional[int] = None
                    t_normal: Optional[int] = None
                    if tm_el is not None:
                        try:
                            t_actual = int(tm_el.findtext("actual-notes") or "0") or None
                            t_normal = int(tm_el.findtext("normal-notes") or "0") or None
                        except ValueError:
                            pass

                    notations = child.find("notations")
                    t_pos: Optional[str] = None
                    t_id: Optional[int] = None
                    if notations is not None:
                        tup_el = notations.find("tuplet")
                        if tup_el is not None:
                            ttype = (tup_el.attrib.get("type") or "").lower()
                            if ttype == "start":
                                t_id = tuplet_counter
                                tuplet_counter += 1
                                tuplet_active[(s, voice)] = (t_id, t_actual or 3, t_normal or 2)
                                t_pos = "start"
                            elif ttype == "stop":
                                cur = tuplet_active.pop((s, voice), None)
                                if cur is not None:
                                    t_id, t_actual, t_normal = cur
                                t_pos = "stop"
                    if t_id is None and (s, voice) in tuplet_active and tuplet_active[(s, voice)] is not None:
                        t_id, ta, tn = tuplet_active[(s, voice)]  # type: ignore[misc]
                        t_actual = t_actual or ta
                        t_normal = t_normal or tn
                        t_pos = "middle"
                    elif t_id is None and t_actual and t_normal and t_actual != t_normal:
                        # Implicit tuplet via time-modification only
                        t_id = tuplet_counter
                        tuplet_active[(s, voice)] = (t_id, t_actual, t_normal)
                        t_pos = "start"

                    # Rest
                    rest_el = child.find("rest")
                    if rest_el is not None:
                        is_mr = rest_el.attrib.get("measure") == "yes"
                        dur_kind = "whole" if is_mr else _normalize_duration_kind(raw_type)
                        if raw_type is None and not is_mr and dur_beats > 0:
                            # Infer duration kind from beats
                            if dur_beats >= 3.5:
                                dur_kind = "whole"
                            elif dur_beats >= 1.75:
                                dur_kind = "half"
                            elif dur_beats >= 0.85:
                                dur_kind = "quarter"
                            elif dur_beats >= 0.4:
                                dur_kind = "eighth"
                            else:
                                dur_kind = "16th"
                        arts: List[str] = []
                        if notations is not None and notations.find("fermata") is not None:
                            arts.append("fermata")
                        ev_r = RestEvent(
                            duration_kind=dur_kind,
                            dots=dots,
                            beat_offset=beat_cursor,
                            is_measure_rest=is_mr,
                            voice=voice,
                            multi_measure_count=multi_bar_rest[s] if is_mr else 0,
                            articulations=arts,
                            tuplet_id=t_id,
                            tuplet_actual=t_actual,
                            tuplet_normal=t_normal,
                            tuplet_pos=t_pos,
                        )
                        events_by_staff[s].append(ev_r)
                        beam_open[(s, voice)] = None
                        beat_cursor += dur_beats
                        continue

                    # Pitch
                    pitch_el = child.find("pitch")
                    if pitch_el is None:
                        if not is_chord and not is_grace:
                            beat_cursor += dur_beats
                        continue

                    letter = (pitch_el.findtext("step") or "C").strip().upper()
                    octave = int(pitch_el.findtext("octave") or "4")
                    alter_txt = pitch_el.findtext("alter")
                    alter = int(round(float(alter_txt))) if alter_txt else 0
                    st_step = _diatonic_to_staff_step(letter, octave, active_clef[s])

                    # Accidental & courtesy
                    acc_el = child.find("accidental")
                    acc_str: Optional[str] = None
                    is_courtesy = False
                    if acc_el is not None:
                        acc_str = _normalize_accidental(acc_el.text)
                        is_courtesy = (
                            acc_el.attrib.get("parentheses") == "yes"
                            or acc_el.attrib.get("cautionary") == "yes"
                        )

                    # Ties
                    t_start = False
                    t_stop = False
                    for tel in child.findall("tie"):
                        if tel.attrib.get("type") == "start":
                            t_start = True
                        elif tel.attrib.get("type") == "stop":
                            t_stop = True
                    if notations is not None:
                        for tel in notations.findall("tied"):
                            if tel.attrib.get("type") == "start":
                                t_start = True
                            elif tel.attrib.get("type") == "stop":
                                t_stop = True

                    # Slurs, Articulations, Ornaments, Fermata
                    s_starts: List[int] = []
                    s_stops: List[int] = []
                    arts_n: List[str] = []
                    if notations is not None:
                        for sl_el in notations.findall("slur"):
                            snum = int(sl_el.attrib.get("number", "1") or "1")
                            stype = (sl_el.attrib.get("type") or "").lower()
                            if stype == "start":
                                s_starts.append(snum)
                            elif stype == "stop":
                                s_stops.append(snum)
                        art_el = notations.find("articulations")
                        if art_el is not None:
                            for ac in art_el:
                                if ac.tag == "staccato":
                                    arts_n.append("staccato")
                                elif ac.tag == "accent":
                                    arts_n.append("accent")
                                elif ac.tag == "tenuto":
                                    arts_n.append("tenuto")
                                elif ac.tag == "staccatissimo":
                                    arts_n.append("staccatissimo")
                                elif ac.tag == "strong-accent":
                                    arts_n.append("marcato")
                        orn_el = notations.find("ornaments")
                        if orn_el is not None:
                            if orn_el.find("trill-mark") is not None:
                                arts_n.append("trill")
                        if notations.find("fermata") is not None:
                            arts_n.append("fermata")

                    # Lyrics
                    lyrics_list: List[Tuple[int, str]] = []
                    for lyr_el in child.findall("lyric"):
                        l_num_raw = lyr_el.attrib.get("number", "1")
                        try:
                            l_num = int(re.sub(r"[^0-9]", "", l_num_raw) or "1")
                        except ValueError:
                            l_num = 1
                        l_txt = (lyr_el.findtext("text") or "").strip()
                        syll = (lyr_el.findtext("syllabic") or "single").strip().lower()
                        if l_txt:
                            if syll in ("begin", "middle"):
                                l_txt = l_txt + "-"
                            lyrics_list.append((l_num, l_txt))

                    # Stem direction override
                    stem_txt = (child.findtext("stem") or "").strip().lower()
                    stem_dir: Optional[int] = None
                    if stem_txt == "up":
                        stem_dir = 1
                    elif stem_txt == "down":
                        stem_dir = -1

                    # If <chord/>, merge into previous NoteEvent of same (staff, voice)
                    if is_chord and (s, voice) in last_note:
                        prev = last_note[(s, voice)]
                        combined = list(
                            zip(
                                prev.staff_steps,
                                prev.accidentals,
                                prev.courtesy_accidentals,
                                prev.tie_starts,
                                prev.tie_stops,
                                prev.pitches_info,
                            )
                        )
                        combined.append((st_step, acc_str, is_courtesy, t_start, t_stop, (letter, octave, alter)))
                        combined.sort(key=lambda t: t[0])
                        prev.staff_steps = [c[0] for c in combined]
                        prev.accidentals = [c[1] for c in combined]
                        prev.courtesy_accidentals = [c[2] for c in combined]
                        prev.tie_starts = [c[3] for c in combined]
                        prev.tie_stops = [c[4] for c in combined]
                        prev.pitches_info = [c[5] for c in combined]
                        for a in arts_n:
                            if a not in prev.articulations:
                                prev.articulations.append(a)
                        prev.slur_starts.extend(s_starts)
                        prev.slur_stops.extend(s_stops)
                        continue

                    dur_kind = _normalize_duration_kind(raw_type)

                    # Explicit beams (beam number="1")
                    beam_grp: Optional[int] = None
                    if not is_grace:
                        b1 = None
                        for bel in child.findall("beam"):
                            if bel.attrib.get("number", "1") == "1":
                                b1 = (bel.text or "").strip().lower()
                                break
                        if b1 == "begin":
                            beam_grp = beam_counter
                            beam_open[(s, voice)] = beam_counter
                            beam_counter += 1
                        elif b1 == "continue" and beam_open.get((s, voice)) is not None:
                            beam_grp = beam_open[(s, voice)]
                        elif b1 == "end" and beam_open.get((s, voice)) is not None:
                            beam_grp = beam_open[(s, voice)]
                            beam_open[(s, voice)] = None
                        else:
                            beam_open[(s, voice)] = None

                    ev_n = NoteEvent(
                        staff_steps=[st_step],
                        accidentals=[acc_str],
                        duration_kind=dur_kind,
                        dots=dots,
                        beat_offset=beat_cursor,
                        beam_group=beam_grp,
                        stem_dir=stem_dir,
                        voice=voice,
                        is_grace=is_grace,
                        grace_slash=grace_slash,
                        courtesy_accidentals=[is_courtesy],
                        articulations=arts_n,
                        tie_starts=[t_start],
                        tie_stops=[t_stop],
                        slur_starts=s_starts,
                        slur_stops=s_stops,
                        tuplet_id=t_id,
                        tuplet_actual=t_actual,
                        tuplet_normal=t_normal,
                        tuplet_pos=t_pos,
                        lyrics=lyrics_list,
                        pitches_info=[(letter, octave, alter)],
                    )
                    events_by_staff[s].append(ev_n)
                    if not is_grace:
                        last_note[(s, voice)] = ev_n
                        beat_cursor += dur_beats

            # Finalize measure per staff
            for s in range(1, n_staves + 1):
                evs = events_by_staff[s]
                _assign_auto_beams(evs, active_ts[s])
                md = MeasureData(
                    number=m_num,
                    clef=active_clef[s],
                    key_fifths=active_key[s],
                    time_sig=active_ts[s],
                    clef_change=clef_change[s],
                    key_change=key_change[s],
                    time_change=time_change[s],
                    barline=right_barline,
                    events=evs,
                    left_barline=left_barline,
                    multi_bar_rest=multi_bar_rest[s],
                    mid_changes=mid_changes[s],
                    dynamics=dynamics[s],
                    tempos=tempos[s],
                    volta=volta_for_measure[s] or open_volta.get(s),
                )
                resolve_polyphony_and_collisions(md)
                staff_measures[s].append(md)

        for s in range(1, n_staves + 1):
            label = pname if n_staves == 1 else f"{pname} (Staff {s})"
            pd = PartData(
                name=label,
                staff_index=len(parsed_parts),
                measures=staff_measures[s],
                brace_group=p_brace,
                hairpins=staff_hairpins[s],
            )
            _resolve_part_ties_and_slurs(pd)
            parsed_parts.append(pd)

    return ParsedScore(title=title, source_path=source_path, parts=parsed_parts)


def _parse_musicxml(path: Path) -> ParsedScore:
    if path.suffix.lower() == ".mxl":
        xml_text: Optional[str] = None
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith((".musicxml", ".xml")) and not name.startswith("META-INF"):
                    xml_text = zf.read(name).decode("utf-8", errors="replace")
                    break
        if xml_text is None:
            raise ValueError(f"No MusicXML document found inside {path}")
    else:
        xml_text = path.read_text(encoding="utf-8", errors="replace")
    return _parse_musicxml_xml(xml_text, str(path))


# ---------------------------------------------------------------------------
# MuseScore (.mscx / .mscz) XML Parser (MuseScore 3 & 4, Zero Binary)
# ---------------------------------------------------------------------------


def _mscx_clef_code(raw: Optional[str]) -> str:
    if not raw:
        return "G-2"
    r = raw.strip().upper()
    if "8VB" in r:
        return "F-4-8vb" if r.startswith("F") else "G-2-8vb"
    if "8VA" in r:
        return "F-4-8va" if r.startswith("F") else "G-2-8va"
    if r.startswith("F"):
        return "F-4"
    if r.startswith("C4"):
        return "C-4"
    if r.startswith("C"):
        return "C-3"
    return "G-2"


def _mscx_articulation_name(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().lower()
    if "staccatissimo" in s:
        return "staccatissimo"
    if "staccato" in s:
        return "staccato"
    if "marcato" in s:
        return "marcato"
    if "accent" in s or "sforzato" in s:
        return "accent"
    if "tenuto" in s:
        return "tenuto"
    if "fermata" in s:
        return "fermata"
    if "trill" in s:
        return "trill"
    return None


def _parse_mscx_xml(xml_bytes: bytes, source_path: str) -> ParsedScore:
    root = ET.fromstring(xml_bytes)
    score_el = root.find("Score")
    if score_el is None:
        score_el = root

    title = Path(source_path).stem
    for meta in score_el.findall("metaTag"):
        if meta.attrib.get("name") == "workTitle" and meta.text:
            title = meta.text.strip()
            break

    # Map staff id -> (part_name, default_clef, brace_group)
    staff_meta: Dict[str, Tuple[str, str, Optional[int]]] = {}
    brace_counter = 0
    for p_idx, part_el in enumerate(score_el.findall("Part")):
        track_name = (part_el.findtext("trackName") or f"Part {p_idx + 1}").strip()
        p_staves = part_el.findall("Staff")
        bg: Optional[int] = None
        if len(p_staves) >= 2:
            bg = brace_counter
            brace_counter += 1
        for s_pos, st_el in enumerate(p_staves):
            sid = st_el.attrib.get("id", str(len(staff_meta) + 1))
            def_clef_el = st_el.find("defaultClef")
            if def_clef_el is not None and def_clef_el.text:
                def_clef = _mscx_clef_code(def_clef_el.text)
            elif len(p_staves) >= 2 and s_pos == 1:
                def_clef = "F-4"
            else:
                def_clef = "G-2"
            staff_meta[sid] = (track_name, def_clef, bg)

    parsed_parts: List[PartData] = []
    for s_idx, staff_el in enumerate(score_el.findall("Staff")):
        sid = staff_el.attrib.get("id", str(s_idx + 1))
        part_name, active_clef, brace_group = staff_meta.get(
            sid, (f"Staff {s_idx + 1}", "G-2", None)
        )
        active_key = 0
        active_ts = (4, 4)
        measures_out: List[MeasureData] = []
        part_hairpins: List[HairpinSpan] = []

        # Open spanners across measures in this staff
        open_hairpin: Optional[Tuple[str, int, float, str]] = None
        open_volta: Optional[VoltaBracket] = None
        slur_id_counter = 1
        pending_slur_starts: Dict[int, List[int]] = {}  # voice -> list of slur ids
        pending_slur_stops: Dict[int, List[int]] = {}   # voice -> list of slur ids
        active_slur_stack: List[int] = []
        tuplet_counter = 1

        xml_measures = staff_el.findall("Measure")
        for m_idx, m_el in enumerate(xml_measures):
            clef_change = False
            key_change = False
            time_change = False
            barline = "single"
            left_barline = "none"
            multi_bar_rest = 0
            events: List[ScoreEvent] = []
            mid_changes: List[MidMeasureChange] = []
            dynamics: List[DynamicEvent] = []
            tempos: List[TempoEvent] = []
            volta_for_m: Optional[VoltaBracket] = open_volta

            # MuseScore 3/4 repeat barlines on <Measure>
            if m_el.find("startRepeat") is not None:
                left_barline = "repeat_left"
            if m_el.find("endRepeat") is not None:
                barline = "repeat_right"

            # Multi-measure rest count
            mmr_txt = m_el.findtext("multiMeasureRest")
            if mmr_txt:
                try:
                    mmr_val = int(mmr_txt.strip())
                    if mmr_val > 1:
                        multi_bar_rest = mmr_val
                except ValueError:
                    pass

            voices = m_el.findall("voice")
            voice_containers = voices if voices else [m_el]

            for v_idx, container in enumerate(voice_containers):
                voice_num = min(4, v_idx + 1)
                beat_cursor = 0.0
                seen_in_voice = False
                pending_fermata = False
                active_tuplet: Optional[Tuple[int, int, int, bool]] = None  # (tid, actual, normal, is_first)

                for child in container:
                    tag = child.tag

                    if tag == "location":
                        beat_cursor = max(0.0, beat_cursor + _parse_fraction_beats(child.findtext("fractions")))

                    elif tag == "Clef":
                        ctype = (
                            child.findtext("concertClefType")
                            or child.findtext("transposingClefType")
                            or child.findtext("subtype")
                        )
                        new_clef = _mscx_clef_code(ctype)
                        if seen_in_voice and beat_cursor > 1e-4:
                            if new_clef != active_clef:
                                mid_changes.append(MidMeasureChange(beat_cursor, "clef", new_clef))
                                active_clef = new_clef
                        else:
                            if new_clef != active_clef:
                                clef_change = m_idx > 0
                                active_clef = new_clef

                    elif tag == "KeySig":
                        acc_txt = child.findtext("accidental") or child.findtext("concertKey")
                        if acc_txt is not None:
                            try:
                                new_key = max(-7, min(7, int(acc_txt)))
                                if seen_in_voice and beat_cursor > 1e-4:
                                    if new_key != active_key:
                                        mid_changes.append(MidMeasureChange(beat_cursor, "key", new_key))
                                        active_key = new_key
                                else:
                                    if new_key != active_key:
                                        key_change = m_idx > 0
                                        active_key = new_key
                            except ValueError:
                                pass

                    elif tag == "TimeSig":
                        sn = child.findtext("sigN")
                        sd = child.findtext("sigD")
                        if sn and sd:
                            try:
                                new_ts = (int(sn), int(sd))
                                if seen_in_voice and beat_cursor > 1e-4:
                                    if new_ts != active_ts:
                                        mid_changes.append(MidMeasureChange(beat_cursor, "time", new_ts))
                                        active_ts = new_ts
                                else:
                                    if new_ts != active_ts:
                                        time_change = m_idx > 0
                                        active_ts = new_ts
                            except ValueError:
                                pass

                    elif tag == "BarLine":
                        sub = (child.findtext("subtype") or "").lower()
                        if "end" in sub or "final" in sub:
                            barline = "final"
                        elif "double" in sub:
                            barline = "double"
                        elif "start-repeat" in sub:
                            left_barline = "repeat_left"
                        elif "repeat" in sub:
                            barline = "repeat_right"

                    elif tag == "Dynamic":
                        sub = (child.findtext("subtype") or "").strip()
                        if sub:
                            pl = (child.findtext("placement") or "below").strip().lower()
                            dynamics.append(DynamicEvent(beat_offset=beat_cursor, text=sub, placement=pl))

                    elif tag in ("Tempo", "StaffText"):
                        txt = _strip_xml_tags(child.findtext("text") or "")
                        if txt:
                            tempos.append(TempoEvent(beat_offset=beat_cursor, text=txt))

                    elif tag == "Fermata":
                        pending_fermata = True

                    elif tag == "Tuplet":
                        try:
                            actual = int(child.findtext("actualNotes") or "3")
                            normal = int(child.findtext("normalNotes") or "2")
                        except ValueError:
                            actual, normal = 3, 2
                        active_tuplet = (tuplet_counter, actual, normal, True)
                        tuplet_counter += 1

                    elif tag == "endTuplet":
                        if active_tuplet is not None:
                            # Mark last event in this tuplet as 'stop'
                            tid = active_tuplet[0]
                            for prev_ev in reversed(events):
                                if getattr(prev_ev, "tuplet_id", None) == tid:
                                    prev_ev.tuplet_pos = "stop"
                                    break
                            active_tuplet = None

                    elif tag == "Spanner":
                        stype = child.attrib.get("type", "")
                        has_next = child.find("next") is not None
                        has_prev = child.find("prev") is not None

                        if stype == "Slur":
                            if has_next and not has_prev:
                                new_sid = slur_id_counter
                                slur_id_counter += 1
                                active_slur_stack.append(new_sid)
                                pending_slur_starts.setdefault(voice_num, []).append(new_sid)
                            elif has_prev and not has_next:
                                end_sid = active_slur_stack.pop() if active_slur_stack else slur_id_counter
                                pending_slur_stops.setdefault(voice_num, []).append(end_sid)

                        elif stype == "HairPin":
                            if has_next and not has_prev:
                                hp_el = child.find("HairPin")
                                hp_sub = (hp_el.findtext("subtype") if hp_el is not None else "0") or "0"
                                hp_kind = "diminuendo" if hp_sub.strip() in ("1", "3", "diminuendo") else "crescendo"
                                hp_pl = (hp_el.findtext("placement") if hp_el is not None else "below") or "below"
                                open_hairpin = (hp_kind, m_idx, beat_cursor, hp_pl)
                            elif has_prev and not has_next and open_hairpin is not None:
                                wkind, sm, sb, wpl = open_hairpin
                                part_hairpins.append(
                                    HairpinSpan(
                                        kind=wkind,
                                        start_measure=sm,
                                        start_beat=sb,
                                        end_measure=m_idx,
                                        end_beat=beat_cursor,
                                        placement=wpl,
                                    )
                                )
                                open_hairpin = None

                        elif stype == "Volta":
                            if has_next and not has_prev:
                                v_el = child.find("Volta")
                                v_txt = _strip_xml_tags(
                                    (v_el.findtext("beginText") if v_el is not None else None) or "1."
                                ) or "1."
                                open_volta = VoltaBracket(
                                    text=v_txt,
                                    start_measure=m_idx,
                                    end_measure=m_idx,
                                    closed_right=True,
                                )
                                volta_for_m = open_volta
                            elif has_prev and not has_next and open_volta is not None:
                                open_volta.end_measure = max(open_volta.start_measure, m_idx - 1)
                                open_volta = None

                    elif tag == "Rest":
                        seen_in_voice = True
                        raw_dur = child.findtext("durationType") or "quarter"
                        dots = int(child.findtext("dots") or "0")
                        is_mr = raw_dur.strip().lower() == "measure"
                        dur_kind = "whole" if is_mr else _normalize_duration_kind(raw_dur)

                        t_id: Optional[int] = None
                        t_act: Optional[int] = None
                        t_norm: Optional[int] = None
                        t_pos: Optional[str] = None
                        if active_tuplet is not None:
                            t_id, t_act, t_norm, is_first = active_tuplet
                            t_pos = "start" if is_first else "middle"
                            active_tuplet = (t_id, t_act, t_norm, False)

                        r_arts: List[str] = []
                        if pending_fermata:
                            r_arts.append("fermata")
                            pending_fermata = False

                        ev_r = RestEvent(
                            duration_kind=dur_kind,
                            dots=dots,
                            beat_offset=beat_cursor,
                            is_measure_rest=is_mr,
                            voice=voice_num,
                            multi_measure_count=multi_bar_rest if is_mr else 0,
                            articulations=r_arts,
                            tuplet_id=t_id,
                            tuplet_actual=t_act,
                            tuplet_normal=t_norm,
                            tuplet_pos=t_pos,
                        )
                        events.append(ev_r)
                        beat_cursor += (
                            active_ts[0] * (4.0 / active_ts[1]) if is_mr else ev_r.beats
                        )

                    elif tag == "Chord":
                        seen_in_voice = True
                        raw_dur = child.findtext("durationType") or "quarter"
                        dots = int(child.findtext("dots") or "0")
                        dur_kind = _normalize_duration_kind(raw_dur)

                        is_acciaccatura = child.find("acciaccatura") is not None
                        is_appoggiatura = child.find("appoggiatura") is not None
                        is_grace = is_acciaccatura or is_appoggiatura or any(
                            child.find(g) is not None for g in ("grace4", "grace8", "grace16", "grace32")
                        )

                        # Articulations & ornaments on Chord
                        arts_c: List[str] = []
                        if pending_fermata:
                            arts_c.append("fermata")
                            pending_fermata = False
                        for art_el in child.findall("Articulation"):
                            aname = _mscx_articulation_name(art_el.findtext("subtype"))
                            if aname and aname not in arts_c:
                                arts_c.append(aname)
                        for orn_el in child.findall("Ornament"):
                            oname = _mscx_articulation_name(orn_el.findtext("subtype"))
                            if oname and oname not in arts_c:
                                arts_c.append(oname)
                        if child.find("Fermata") is not None and "fermata" not in arts_c:
                            arts_c.append("fermata")

                        # Lyrics on Chord
                        lyrics_c: List[Tuple[int, str]] = []
                        for lyr_el in child.findall("Lyrics"):
                            no_txt = lyr_el.findtext("no") or "0"
                            try:
                                vno = int(no_txt) + 1
                            except ValueError:
                                vno = 1
                            l_txt = _strip_xml_tags(lyr_el.findtext("text") or "")
                            syll = (lyr_el.findtext("syllabic") or "single").strip().lower()
                            if l_txt:
                                if syll in ("begin", "middle"):
                                    l_txt += "-"
                                lyrics_c.append((vno, l_txt))

                        # Stem direction override
                        sdir_txt = (child.findtext("StemDirection") or "").strip().lower()
                        stem_dir: Optional[int] = None
                        if sdir_txt == "up":
                            stem_dir = 1
                        elif sdir_txt == "down":
                            stem_dir = -1

                        # Tuplet assignment
                        t_id = None
                        t_act = None
                        t_norm = None
                        t_pos = None
                        if active_tuplet is not None and not is_grace:
                            t_id, t_act, t_norm, is_first = active_tuplet
                            t_pos = "start" if is_first else "middle"
                            active_tuplet = (t_id, t_act, t_norm, False)

                        # Notes in Chord
                        note_tuples: List[Tuple[int, Optional[str], bool, bool, bool, Tuple[str, int, int]]] = []
                        for note_el in child.findall("Note"):
                            p_txt = note_el.findtext("pitch")
                            t_txt = note_el.findtext("tpc")
                            if p_txt is None:
                                continue
                            midi_p = int(p_txt)
                            tpc = int(t_txt) if t_txt is not None else None
                            letter, octave, alter = _mscx_pitch_tpc_to_letter_octave_alter(midi_p, tpc)
                            st = _diatonic_to_staff_step(letter, octave, active_clef)

                            acc_el = note_el.find("Accidental")
                            acc_str: Optional[str] = None
                            is_courtesy = False
                            if acc_el is not None:
                                acc_str = _normalize_accidental(acc_el.findtext("subtype"))
                                br_txt = (acc_el.findtext("bracket") or "0").strip()
                                is_courtesy = br_txt in ("1", "2", "parentheses")

                            t_start = False
                            t_stop = False
                            for sp_el in note_el.findall("Spanner"):
                                if sp_el.attrib.get("type") == "Tie":
                                    if sp_el.find("next") is not None:
                                        t_start = True
                                    if sp_el.find("prev") is not None:
                                        t_stop = True

                            note_tuples.append((st, acc_str, is_courtesy, t_start, t_stop, (letter, octave, alter)))

                        if note_tuples:
                            note_tuples.sort(key=lambda t: t[0])
                            s_starts = pending_slur_starts.pop(voice_num, [])
                            s_stops = pending_slur_stops.pop(voice_num, [])
                            ev_n = NoteEvent(
                                staff_steps=[nt[0] for nt in note_tuples],
                                accidentals=[nt[1] for nt in note_tuples],
                                duration_kind=dur_kind,
                                dots=dots,
                                beat_offset=beat_cursor,
                                stem_dir=stem_dir,
                                voice=voice_num,
                                is_grace=is_grace,
                                grace_slash=is_acciaccatura,
                                courtesy_accidentals=[nt[2] for nt in note_tuples],
                                articulations=arts_c,
                                tie_starts=[nt[3] for nt in note_tuples],
                                tie_stops=[nt[4] for nt in note_tuples],
                                slur_starts=s_starts,
                                slur_stops=s_stops,
                                tuplet_id=t_id,
                                tuplet_actual=t_act,
                                tuplet_normal=t_norm,
                                tuplet_pos=t_pos,
                                lyrics=lyrics_c,
                                pitches_info=[nt[5] for nt in note_tuples],
                            )
                            events.append(ev_n)
                            if not is_grace:
                                beat_cursor += ev_n.beats

            _assign_auto_beams(events, active_ts)
            if m_idx == len(xml_measures) - 1 and barline == "single":
                barline = "final"

            md = MeasureData(
                number=m_idx + 1,
                clef=active_clef,
                key_fifths=active_key,
                time_sig=active_ts,
                clef_change=clef_change,
                key_change=key_change,
                time_change=time_change,
                barline=barline,
                events=events,
                left_barline=left_barline,
                multi_bar_rest=multi_bar_rest,
                mid_changes=mid_changes,
                dynamics=dynamics,
                tempos=tempos,
                volta=volta_for_m,
            )
            resolve_polyphony_and_collisions(md)
            measures_out.append(md)

        pd = PartData(
            name=part_name,
            staff_index=s_idx,
            measures=measures_out,
            brace_group=brace_group,
            hairpins=part_hairpins,
        )
        _resolve_part_ties_and_slurs(pd)
        parsed_parts.append(pd)

    return ParsedScore(title=title, source_path=source_path, parts=parsed_parts)


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------


def parse_score(path: Union[str, Path]) -> ParsedScore:
    """Parse a `.musicxml`, `.xml`, `.mxl`, `.mscx`, or `.mscz` score into a `ParsedScore`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Score file not found: {p}")
    ext = p.suffix.lower()
    if ext == ".mscx":
        return _parse_mscx_xml(p.read_bytes(), str(p))
    if ext == ".mscz":
        with zipfile.ZipFile(p, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith(".mscx"):
                    return _parse_mscx_xml(zf.read(name), str(p))
        raise ValueError(f"No .mscx file found inside {p}")
    if ext in (".musicxml", ".xml", ".mxl"):
        return _parse_musicxml(p)
    raise ValueError(f"Unsupported score extension: {ext}")


# ---------------------------------------------------------------------------
# MusicXML Export Helper (for `assembly-serialize bekern` on .mscx/.mscz)
# ---------------------------------------------------------------------------


def export_score_to_musicxml(
    score: ParsedScore,
    out_path: Union[str, Path],
    measure_range: Optional[Tuple[int, int]] = None,
) -> Path:
    """Export a `ParsedScore` (or 0-based `[start, end)` measure slice) to MusicXML 3.1 Partwise."""
    divisions = 480  # MuseScore native division: cleanly divides 2,3,4,5,6,8,12,16,32
    root = ET.Element("score-partwise", version="3.1")
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = score.title

    part_list = ET.SubElement(root, "part-list")
    for p_idx, part in enumerate(score.parts):
        sp = ET.SubElement(part_list, "score-part", id=f"P{p_idx + 1}")
        ET.SubElement(sp, "part-name").text = part.name

    acc_to_mxml = {
        "sharp": "sharp",
        "flat": "flat",
        "natural": "natural",
        "double_sharp": "double-sharp",
        "double_flat": "flat-flat",
    }

    for p_idx, part in enumerate(score.parts):
        p_el = ET.SubElement(root, "part", id=f"P{p_idx + 1}")
        meas_slice = (
            part.measures[measure_range[0] : measure_range[1]]
            if measure_range is not None
            else part.measures
        )
        for m_idx, meas in enumerate(meas_slice):
            m_num_out = (m_idx + 1) if measure_range is not None else meas.number
            m_el = ET.SubElement(p_el, "measure", number=str(m_num_out))
            if m_idx == 0 or meas.clef_change or meas.key_change or meas.time_change:
                attr = ET.SubElement(m_el, "attributes")
                if m_idx == 0:
                    ET.SubElement(attr, "divisions").text = str(divisions)
                if m_idx == 0 or meas.key_change:
                    k_el = ET.SubElement(attr, "key")
                    ET.SubElement(k_el, "fifths").text = str(meas.key_fifths)
                if m_idx == 0 or meas.time_change:
                    t_el = ET.SubElement(attr, "time")
                    ET.SubElement(t_el, "beats").text = str(meas.time_sig[0])
                    ET.SubElement(t_el, "beat-type").text = str(meas.time_sig[1])
                if m_idx == 0 or meas.clef_change:
                    c_el = ET.SubElement(attr, "clef")
                    base = base_clef_code(meas.clef)
                    sign, line = base.split("-", 1)
                    ET.SubElement(c_el, "sign").text = sign
                    ET.SubElement(c_el, "line").text = line
                    shift = clef_octave_shift(meas.clef)
                    if shift != 0:
                        ET.SubElement(c_el, "clef-octave-change").text = str(shift)

            # Group events by voice and track exact divisions cursor for clean <backup>/<forward>
            voices = sorted({getattr(e, "voice", 1) for e in meas.events}) or [1]
            prev_cursor_div = 0
            for v_pos, v in enumerate(voices):
                v_evs = [e for e in meas.events if getattr(e, "voice", 1) == v]
                if v_pos > 0 and prev_cursor_div > 0:
                    bk = ET.SubElement(m_el, "backup")
                    ET.SubElement(bk, "duration").text = str(prev_cursor_div)
                cursor_div = 0
                for ev in v_evs:
                    if isinstance(ev, RestEvent):
                        target_div = int(round(ev.beat_offset * divisions))
                        if target_div > cursor_div:
                            fw = ET.SubElement(m_el, "forward")
                            ET.SubElement(fw, "duration").text = str(target_div - cursor_div)
                            cursor_div = target_div
                        n_el = ET.SubElement(m_el, "note")
                        r_el = ET.SubElement(n_el, "rest")
                        if ev.is_measure_rest:
                            r_el.set("measure", "yes")
                        dur_div = max(
                            1,
                            int(
                                round(
                                    (
                                        meas.time_sig[0] * (4.0 / meas.time_sig[1])
                                        if ev.is_measure_rest
                                        else ev.beats
                                    )
                                    * divisions
                                )
                            ),
                        )
                        ET.SubElement(n_el, "duration").text = str(dur_div)
                        ET.SubElement(n_el, "voice").text = str(v)
                        if not ev.is_measure_rest:
                            ET.SubElement(n_el, "type").text = ev.duration_kind
                            for _ in range(ev.dots):
                                ET.SubElement(n_el, "dot")
                            if ev.tuplet_actual and ev.tuplet_normal:
                                tm = ET.SubElement(n_el, "time-modification")
                                ET.SubElement(tm, "actual-notes").text = str(ev.tuplet_actual)
                                ET.SubElement(tm, "normal-notes").text = str(ev.tuplet_normal)
                        cursor_div += dur_div
                    elif isinstance(ev, NoteEvent):
                        if not ev.is_grace:
                            target_div = int(round(ev.beat_offset * divisions))
                            if target_div > cursor_div:
                                fw = ET.SubElement(m_el, "forward")
                                ET.SubElement(fw, "duration").text = str(target_div - cursor_div)
                                cursor_div = target_div
                        dur_div = max(1, int(round(ev.beats * divisions))) if not ev.is_grace else 0
                        for h_i, step in enumerate(ev.staff_steps):
                            n_el = ET.SubElement(m_el, "note")
                            if ev.is_grace:
                                g_el = ET.SubElement(n_el, "grace")
                                if ev.grace_slash:
                                    g_el.set("slash", "yes")
                            if h_i > 0:
                                ET.SubElement(n_el, "chord")
                            if h_i < len(ev.pitches_info):
                                letter, octave, alter = ev.pitches_info[h_i]
                            else:
                                letter, octave = _staff_step_to_letter_octave(step, meas.clef)
                                alter = 0
                            p_el = ET.SubElement(n_el, "pitch")
                            ET.SubElement(p_el, "step").text = letter
                            if alter != 0:
                                ET.SubElement(p_el, "alter").text = str(alter)
                            ET.SubElement(p_el, "octave").text = str(octave)
                            if not ev.is_grace:
                                ET.SubElement(n_el, "duration").text = str(dur_div)
                            ET.SubElement(n_el, "voice").text = str(v)
                            ET.SubElement(n_el, "type").text = ev.duration_kind
                            for _ in range(ev.dots):
                                ET.SubElement(n_el, "dot")
                            acc = ev.accidentals[h_i] if h_i < len(ev.accidentals) else None
                            if acc and acc in acc_to_mxml:
                                ET.SubElement(n_el, "accidental").text = acc_to_mxml[acc]
                            if ev.tuplet_actual and ev.tuplet_normal and not ev.is_grace:
                                tm = ET.SubElement(n_el, "time-modification")
                                ET.SubElement(tm, "actual-notes").text = str(ev.tuplet_actual)
                                ET.SubElement(tm, "normal-notes").text = str(ev.tuplet_normal)
                        if not ev.is_grace:
                            cursor_div += dur_div
                prev_cursor_div = cursor_div

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out
